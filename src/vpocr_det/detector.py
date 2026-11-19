"""`PlateDetNet`: Ultra-compact multi-scale (P2 stride-4 / P3 / P4) anchor-free plate & vehicle detector.

100% first-party permissive (MIT/Apache-2.0 compatible) PyTorch/ONNX architecture engineered for:
- Distant / tiny license plates (down to ~10x5 px via dedicated Stride-4 P2 high-acuity pyramid head
  + optional native-resolution Far-Vehicle Smart-Zoom).
- Low-resolution camera streams (240p/360p/480p CCTV & RTSP substreams).
- Joint `plate` + `vehicle` detection with coarse 4-corner offsets and Task-Aligned Learning (TAL).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vpocr_det.dfine_core import (
    DFINE_PRETRAINED_SPECS,
    DFineCore,
    DFineCoreConfig,
    center_to_corners_format_torch,
)
from vpocr_det.types import (
    CLS_PLATE,
    CLS_SIGNBOARD_HARDNEG,
    CLS_VEHICLE,
    NUM_DET_CLASSES,
    NUM_GEN4_DET_CLASSES,
    FrameDetectionResult,
    PlateDetection,
    VehicleDetection,
    ObservationProvenance,
)


@dataclass(frozen=True)
class DetModelConfig:
    """Configuration for `PlateDetNet` presets (`nano` for weak CPU, `base` for high accuracy)."""

    preset: str = "base"
    c1: int = 32
    c2: int = 64
    c3: int = 128
    c4: int = 224
    neck_ch: int = 96
    head_ch: int = 96
    depth_mult: int = 2
    strides: tuple[int, ...] = (4, 8, 16)
    num_classes: int = NUM_DET_CLASSES

    @classmethod
    def from_preset(cls, preset: str, *, num_classes: int = NUM_DET_CLASSES) -> DetModelConfig:
        p = preset.lower().strip()
        if p == "nano":
            return cls(
                preset="nano",
                c1=24,
                c2=48,
                c3=96,
                c4=144,
                neck_ch=64,
                head_ch=64,
                depth_mult=1,
                num_classes=int(num_classes),
            )
        if p == "base":
            return cls(
                preset="base",
                c1=32,
                c2=64,
                c3=128,
                c4=224,
                neck_ch=96,
                head_ch=96,
                depth_mult=2,
                num_classes=int(num_classes),
            )
        raise ValueError(f"Unknown PlateDetNet preset: {preset!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "c1": self.c1,
            "c2": self.c2,
            "c3": self.c3,
            "c4": self.c4,
            "neck_ch": self.neck_ch,
            "head_ch": self.head_ch,
            "depth_mult": self.depth_mult,
            "strides": list(self.strides),
            "num_classes": int(self.num_classes),
        }


class ConvBNAct(nn.Module):
    """Standard 2D convolution + BatchNorm + SiLU activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        g: int = 1,
        d: int = 1,
    ) -> None:
        super().__init__()
        pad = (k // 2) * d
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=pad, dilation=d, groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConvBlock(nn.Module):
    """Depthwise-separable inverted residual / CSP-lite block with channel squeeze-excitation."""

    def __init__(self, ch: int, k: int = 5, use_residual: bool = True) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.dw = ConvBNAct(ch, ch, k=k, s=1, g=ch)
        self.pw1 = ConvBNAct(ch, ch, k=1, s=1)
        self.dw2 = ConvBNAct(ch, ch, k=3, s=1, g=ch)
        self.pw2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(self.dw2(self.pw1(self.dw(x))))
        if self.use_residual:
            return self.act(x + h)
        return self.act(h)


class StageBlock(nn.Module):
    """Cross-stage feature block combining fast 1x1 split and multi-receptive-field DW blocks."""

    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.down = ConvBNAct(in_ch, out_ch, k=3, s=stride)
        half = out_ch // 2
        self.split_a = ConvBNAct(out_ch, half, k=1, s=1)
        self.split_b = ConvBNAct(out_ch, half, k=1, s=1)
        self.blocks = nn.Sequential(*[DWConvBlock(half, k=5) for _ in range(num_blocks)])
        self.fuse = ConvBNAct(half * 2, out_ch, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        a = self.split_a(x)
        b = self.blocks(self.split_b(x))
        return self.fuse(torch.cat([a, b], dim=1))


class SPPFLite(nn.Module):
    """Fast Spatial Pyramid Pooling capturing full-vehicle context around small plates."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = in_ch // 2
        self.cv1 = ConvBNAct(in_ch, mid, k=1, s=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv2 = ConvBNAct(mid * 4, out_ch, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        y4 = self.pool(y3)
        return self.cv2(torch.cat([y1, y2, y3, y4], dim=1))


class DecoupledScaleHead(nn.Module):
    """Decoupled classification + LTRB box + coarse 4-corner prediction head for one FPN level."""

    def __init__(self, in_ch: int, head_ch: int, num_classes: int = NUM_DET_CLASSES) -> None:
        super().__init__()
        self.cls_stem = nn.Sequential(
            ConvBNAct(in_ch, head_ch, k=3, s=1, g=math.gcd(in_ch, head_ch)),
            ConvBNAct(head_ch, head_ch, k=1, s=1),
            ConvBNAct(head_ch, head_ch, k=3, s=1, g=head_ch),
            ConvBNAct(head_ch, head_ch, k=1, s=1),
        )
        self.reg_stem = nn.Sequential(
            ConvBNAct(in_ch, head_ch, k=3, s=1, g=math.gcd(in_ch, head_ch)),
            ConvBNAct(head_ch, head_ch, k=1, s=1),
            ConvBNAct(head_ch, head_ch, k=3, s=1, g=head_ch),
            ConvBNAct(head_ch, head_ch, k=1, s=1),
        )
        self.cls_pred = nn.Conv2d(head_ch, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_ch, 4, kernel_size=1)
        self.quad_pred = nn.Conv2d(head_ch, 8, kernel_size=1)
        self.reg_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

        # Focal prior bias initialization (p = 0.01)
        nn.init.constant_(self.cls_pred.bias, -math.log((1.0 - 0.01) / 0.01))
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 1.5)
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.zeros_(self.quad_pred.bias)
        nn.init.normal_(self.quad_pred.weight, std=0.01)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_f = self.cls_stem(feat)
        r_f = self.reg_stem(feat)
        cls_logits = self.cls_pred(c_f)  # (B, C, H, W)
        ltrb_norm = F.softplus(self.reg_pred(r_f) * self.reg_scale.clamp(0.2, 5.0))  # (B, 4, H, W) >= 0
        quad_norm = self.quad_pred(r_f)  # (B, 8, H, W)
        return cls_logits, ltrb_norm, quad_norm


class PlateDetNet(nn.Module):
    """Multi-scale (`P2` stride-4, `P3` stride-8, `P4` stride-16) Plate & Vehicle Detector."""

    def __init__(self, config: DetModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or DetModelConfig.from_preset("base")
        cfg = self.config
        nc = cfg.neck_ch

        # Local contrast-robust stem (helps low-contrast fog & cheap CCTV!)
        self.stem = nn.Sequential(
            ConvBNAct(3, cfg.c1, k=3, s=2),
            ConvBNAct(cfg.c1, cfg.c1, k=3, s=1),
        )
        self.stage2 = StageBlock(cfg.c1, cfg.c2, num_blocks=cfg.depth_mult, stride=2)  # stride 4 (C2)
        self.stage3 = StageBlock(cfg.c2, cfg.c3, num_blocks=cfg.depth_mult + 1, stride=2)  # stride 8 (C3)
        self.stage4 = nn.Sequential(
            StageBlock(cfg.c3, cfg.c4, num_blocks=cfg.depth_mult + 1, stride=2),  # stride 16 (C4)
            SPPFLite(cfg.c4, cfg.c4),
        )

        # Top-down FPN (C4 -> P3 -> P2)
        self.lat4 = ConvBNAct(cfg.c4, nc, k=1, s=1)
        self.lat3 = ConvBNAct(cfg.c3, nc, k=1, s=1)
        self.lat2 = ConvBNAct(cfg.c2, nc, k=1, s=1)

        self.td3 = nn.Sequential(ConvBNAct(nc * 2, nc, k=1, s=1), DWConvBlock(nc, k=5))
        self.td2 = nn.Sequential(ConvBNAct(nc * 2, nc, k=1, s=1), DWConvBlock(nc, k=5))

        # Bottom-up PAN (P2 -> N3 -> N4)
        self.down2_3 = ConvBNAct(nc, nc, k=3, s=2, g=nc)
        self.bu3 = nn.Sequential(ConvBNAct(nc * 2, nc, k=1, s=1), DWConvBlock(nc, k=5))
        self.down3_4 = ConvBNAct(nc, nc, k=3, s=2, g=nc)
        self.bu4 = nn.Sequential(ConvBNAct(nc * 2, nc, k=1, s=1), DWConvBlock(nc, k=5))

        self.heads = nn.ModuleList(
            [DecoupledScaleHead(nc, cfg.head_ch, num_classes=cfg.num_classes) for _ in cfg.strides]
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_raw_levels(
        self, images: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward returning per-level `(cls_logits, ltrb_norm, quad_norm)` tensors."""
        # Normalize [0, 1] RGB with zero-mean unit-scale
        x = (images - 0.45) / 0.25
        c1 = self.stem(x)
        c2 = self.stage2(c1)  # H/4, W/4
        c3 = self.stage3(c2)  # H/8, W/8
        c4 = self.stage4(c3)  # H/16, W/16

        p4 = self.lat4(c4)
        p3 = self.td3(torch.cat([self.lat3(c3), F.interpolate(p4, size=c3.shape[-2:], mode="nearest")], dim=1))
        p2 = self.td2(torch.cat([self.lat2(c2), F.interpolate(p3, size=c2.shape[-2:], mode="nearest")], dim=1))

        n2 = p2
        n3 = self.bu3(torch.cat([p3, self.down2_3(n2)], dim=1))
        n4 = self.bu4(torch.cat([p4, self.down3_4(n3)], dim=1))

        feats = [n2, n3, n4]
        return [head(f) for head, f in zip(self.heads, feats)]

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward returning flattened anchor predictions across all 3 pyramid levels:
        - `cls_logits`: `(B, A, 2)`
        - `pred_boxes`: `(B, A, 4)` in input-image pixel coords `[x0, y0, x1, y1]`
        - `pred_corners`: `(B, A, 4, 2)` in input-image pixel coords `[TL, TR, BR, BL]`
        - `anchor_points`: `(A, 2)` `(cx, cy)` in pixels
        - `anchor_strides`: `(A,)` stride per anchor
        """
        level_outs = self.forward_raw_levels(images)
        bsz = images.shape[0]
        device = images.device
        dtype = images.dtype

        cls_all: list[torch.Tensor] = []
        box_all: list[torch.Tensor] = []
        quad_all: list[torch.Tensor] = []
        pts_all: list[torch.Tensor] = []
        str_all: list[torch.Tensor] = []

        for stride, (cls_l, ltrb_l, quad_l) in zip(self.config.strides, level_outs):
            _, _, hl, wl = cls_l.shape
            gy, gx = torch.meshgrid(
                torch.arange(hl, device=device, dtype=torch.float32) + 0.5,
                torch.arange(wl, device=device, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            pts = torch.stack([gx * stride, gy * stride], dim=-1).reshape(-1, 2)  # (N_l, 2)
            st = torch.full((pts.shape[0],), float(stride), device=device, dtype=torch.float32)

            cls_flat = cls_l.float().permute(0, 2, 3, 1).reshape(bsz, -1, cls_l.shape[1])
            ltrb_px = ltrb_l.float().permute(0, 2, 3, 1).reshape(bsz, -1, 4) * float(stride)
            quad_px = quad_l.float().permute(0, 2, 3, 1).reshape(bsz, -1, 4, 2) * float(stride)

            cx = pts[:, 0].unsqueeze(0)
            cy = pts[:, 1].unsqueeze(0)
            boxes = torch.stack(
                [
                    cx - ltrb_px[:, :, 0],
                    cy - ltrb_px[:, :, 1],
                    cx + ltrb_px[:, :, 2],
                    cy + ltrb_px[:, :, 3],
                ],
                dim=-1,
            )
            corners = pts.view(1, -1, 1, 2) + quad_px

            cls_all.append(cls_flat)
            box_all.append(boxes)
            quad_all.append(corners)
            pts_all.append(pts)
            str_all.append(st)

        return (
            torch.cat(cls_all, dim=1),
            torch.cat(box_all, dim=1),
            torch.cat(quad_all, dim=1),
            torch.cat(pts_all, dim=0),
            torch.cat(str_all, dim=0),
        )


def compute_box_iou_ciou(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise `(IoU, CIoU)` between aligned boxes `(..., 4)` `[x0, y0, x1, y1]`."""
    b1_x0, b1_y0, b1_x1, b1_y1 = boxes1.unbind(-1)
    b2_x0, b2_y0, b2_x1, b2_y1 = boxes2.unbind(-1)

    inter_w = (torch.minimum(b1_x1, b2_x1) - torch.maximum(b1_x0, b2_x0)).clamp_min(0.0)
    inter_h = (torch.minimum(b1_y1, b2_y1) - torch.maximum(b1_y0, b2_y0)).clamp_min(0.0)
    inter = inter_w * inter_h

    w1 = (b1_x1 - b1_x0).clamp_min(eps)
    h1 = (b1_y1 - b1_y0).clamp_min(eps)
    w2 = (b2_x1 - b2_x0).clamp_min(eps)
    h2 = (b2_y1 - b2_y0).clamp_min(eps)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = (torch.maximum(b1_x1, b2_x1) - torch.minimum(b1_x0, b2_x0)).clamp_min(eps)
    ch = (torch.maximum(b1_y1, b2_y1) - torch.minimum(b1_y0, b2_y0)).clamp_min(eps)
    c2 = cw * cw + ch * ch + eps

    cx1 = 0.5 * (b1_x0 + b1_x1)
    cy1 = 0.5 * (b1_y0 + b1_y1)
    cx2 = 0.5 * (b2_x0 + b2_x1)
    cy2 = 0.5 * (b2_y0 + b2_y1)
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    v = (4.0 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    ciou = iou - (rho2 / c2 + alpha * v)
    return iou, ciou


def compute_box_giou(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Elementwise `GIoU` between aligned boxes `(..., 4)` `[x0, y0, x1, y1]`."""
    b1_x0, b1_y0, b1_x1, b1_y1 = boxes1.float().unbind(-1)
    b2_x0, b2_y0, b2_x1, b2_y1 = boxes2.float().unbind(-1)

    inter_w = (torch.minimum(b1_x1, b2_x1) - torch.maximum(b1_x0, b2_x0)).clamp_min(0.0)
    inter_h = (torch.minimum(b1_y1, b2_y1) - torch.maximum(b1_y0, b2_y0)).clamp_min(0.0)
    inter = inter_w * inter_h

    w1 = (b1_x1 - b1_x0).clamp_min(eps)
    h1 = (b1_y1 - b1_y0).clamp_min(eps)
    w2 = (b2_x1 - b2_x0).clamp_min(eps)
    h2 = (b2_y1 - b2_y0).clamp_min(eps)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = (torch.maximum(b1_x1, b2_x1) - torch.minimum(b1_x0, b2_x0)).clamp_min(eps)
    ch = (torch.maximum(b1_y1, b2_y1) - torch.minimum(b1_y0, b2_y0)).clamp_min(eps)
    enclose = cw * ch + eps
    return iou - (enclose - union) / enclose


def compute_box_nwd(
    boxes1: torch.Tensor, boxes2: torch.Tensor, const_c: float = 11.0, eps: float = 1e-6
) -> torch.Tensor:
    """Elementwise Normalized Wasserstein Distance (`NWD`) in `(0, 1]` for tiny distant plates."""
    b1_x0, b1_y0, b1_x1, b1_y1 = boxes1.float().unbind(-1)
    b2_x0, b2_y0, b2_x1, b2_y1 = boxes2.float().unbind(-1)
    cx1, cy1 = 0.5 * (b1_x0 + b1_x1), 0.5 * (b1_y0 + b1_y1)
    cx2, cy2 = 0.5 * (b2_x0 + b2_x1), 0.5 * (b2_y0 + b2_y1)
    w1, h1 = (b1_x1 - b1_x0).clamp_min(eps), (b1_y1 - b1_y0).clamp_min(eps)
    w2, h2 = (b2_x1 - b2_x0).clamp_min(eps), (b2_y1 - b2_y0).clamp_min(eps)
    w2_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + 0.25 * ((w1 - w2) ** 2 + (h1 - h2) ** 2)
    return torch.exp(-torch.sqrt(w2_sq + eps) / const_c)


def pairwise_box_nwd(boxes1: torch.Tensor, boxes2: torch.Tensor, const_c: float = 11.0, eps: float = 1e-6) -> torch.Tensor:
    """Pairwise NWD matrix `(M, N)` between `boxes1 (M, 4)` and `boxes2 (N, 4)` in FP32."""
    b1 = boxes1.float()
    b2 = boxes2.float()
    cx1 = 0.5 * (b1[:, 0:1] + b1[:, 2:3])
    cy1 = 0.5 * (b1[:, 1:2] + b1[:, 3:4])
    w1 = (b1[:, 2:3] - b1[:, 0:1]).clamp_min(eps)
    h1 = (b1[:, 3:4] - b1[:, 1:2]).clamp_min(eps)

    cx2 = 0.5 * (b2[:, 0] + b2[:, 2]).unsqueeze(0)
    cy2 = 0.5 * (b2[:, 1] + b2[:, 3]).unsqueeze(0)
    w2 = (b2[:, 2] - b2[:, 0]).clamp_min(eps).unsqueeze(0)
    h2 = (b2[:, 3] - b2[:, 1]).clamp_min(eps).unsqueeze(0)

    w2_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + 0.25 * ((w1 - w2) ** 2 + (h1 - h2) ** 2)
    return torch.exp(-torch.sqrt(w2_sq + eps) / const_c)


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU matrix `(M, N)` between `boxes1 (M, 4)` and `boxes2 (N, 4)`."""
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp_min(0.0)) * ((boxes1[:, 3] - boxes1[:, 1]).clamp_min(0.0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp_min(0.0)) * ((boxes2[:, 3] - boxes2[:, 1]).clamp_min(0.0))
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def pairwise_box_iomin(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise `IoMin = |A n B| / min(|A|, |B|)` matrix `(M, N)` between `boxes1 (M, 4)` and `boxes2 (N, 4)`."""
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp_min(0.0)) * ((boxes1[:, 3] - boxes1[:, 1]).clamp_min(0.0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp_min(0.0)) * ((boxes2[:, 3] - boxes2[:, 1]).clamp_min(0.0))
    return inter / torch.minimum(area1[:, None], area2[None, :]).clamp_min(eps)


class TaskAlignedDetLoss(nn.Module):
    """Dynamic Task-Aligned Assigner (`TAL`) + Varifocal Quality BCE + CIoU + 4-Corner SmoothL1 Loss."""

    def __init__(
        self,
        topk: int = 9,
        alpha: float = 1.0,
        beta: float = 6.0,
        cls_weight: float = 1.2,
        box_weight: float = 2.5,
        quad_weight: float = 1.2,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.quad_weight = quad_weight

    def forward(
        self,
        cls_logits: torch.Tensor,  # (B, A, C)
        pred_boxes: torch.Tensor,  # (B, A, 4)
        pred_corners: torch.Tensor,  # (B, A, 4, 2)
        anchor_points: torch.Tensor,  # (A, 2)
        anchor_strides: torch.Tensor,  # (A,)
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        bsz, num_anchors, num_cls = cls_logits.shape
        device = cls_logits.device

        target_scores = torch.zeros((bsz, num_anchors, num_cls), device=device, dtype=torch.float32)
        ignore_anchor_mask = torch.zeros((bsz, num_anchors), device=device, dtype=torch.bool)
        fg_mask = torch.zeros((bsz, num_anchors), device=device, dtype=torch.bool)
        matched_gt_boxes = torch.zeros((bsz, num_anchors, 4), device=device, dtype=torch.float32)
        matched_gt_corners = torch.zeros((bsz, num_anchors, 4, 2), device=device, dtype=torch.float32)
        matched_gt_is_plate = torch.zeros((bsz, num_anchors), device=device, dtype=torch.bool)
        matched_weights = torch.zeros((bsz, num_anchors), device=device, dtype=torch.float32)

        with torch.no_grad():
            probs = torch.sigmoid(cls_logits.detach())  # (B, A, C)
            ax = anchor_points[:, 0]  # (A,)
            ay = anchor_points[:, 1]  # (A,)

            for b in range(bsz):
                ig_boxes = targets[b].get("ignore_boxes")
                if ig_boxes is not None and ig_boxes.numel() > 0:
                    ig_b = ig_boxes.to(device, dtype=torch.float32).reshape(-1, 4)
                    ix0, iy0, ix1, iy1 = ig_b.unbind(-1)
                    in_ig = (
                        (ax[None, :] >= ix0[:, None])
                        & (ax[None, :] <= ix1[:, None])
                        & (ay[None, :] >= iy0[:, None])
                        & (ay[None, :] <= iy1[:, None])
                    ).any(dim=0)
                    ignore_anchor_mask[b] = in_ig

                gt_labels = targets[b]["labels"].to(device, dtype=torch.long)
                gt_boxes = targets[b]["boxes"].to(device, dtype=torch.float32)
                gt_corners = targets[b]["corners"].to(device, dtype=torch.float32)
                valid_cls = (gt_labels >= 0) & (gt_labels < num_cls)
                if not valid_cls.all():
                    gt_labels = gt_labels[valid_cls]
                    gt_boxes = gt_boxes[valid_cls]
                    gt_corners = gt_corners[valid_cls]
                num_gt = gt_labels.shape[0]
                if num_gt == 0:
                    continue

                # Candidate mask: anchor inside GT box OR within center radius (2.5 * stride) so tiny 8px plates always match P2!
                gx0, gy0, gx1, gy1 = gt_boxes.unbind(-1)  # (G,)
                gcx = 0.5 * (gx0 + gx1)
                gcy = 0.5 * (gy0 + gy1)
                in_box = (
                    (ax[None, :] >= gx0[:, None])
                    & (ax[None, :] <= gx1[:, None])
                    & (ay[None, :] >= gy0[:, None])
                    & (ay[None, :] <= gy1[:, None])
                )  # (G, A)
                rad = 2.5 * anchor_strides[None, :]
                in_center = (torch.abs(ax[None, :] - gcx[:, None]) <= rad) & (
                    torch.abs(ay[None, :] - gcy[:, None]) <= rad
                )
                # Decouple stride assignment between CLS_PLATE (P2/P3 small-medium) and CLS_VEHICLE (P3/P4 medium-large)
                gt_max_dim = torch.maximum(gx1 - gx0, gy1 - gy0)  # (G,)
                is_plate_gt = (gt_labels == CLS_PLATE)[:, None]
                plate_stride_ok = (gt_max_dim[:, None] >= 56.0) | (anchor_strides[None, :] <= 8.0)
                veh_stride_ok = anchor_strides[None, :] >= 8.0
                stride_ok = torch.where(is_plate_gt, plate_stride_ok, veh_stride_ok)
                cand_mask = (in_box | in_center) & stride_ok
                # Guarantee at least center-nearest anchors if stride filter excluded anything
                empty_gt = cand_mask.sum(dim=1) == 0
                if empty_gt.any():
                    cand_mask[empty_gt] = in_box[empty_gt] | in_center[empty_gt]

                ious = pairwise_box_iou(gt_boxes, pred_boxes[b].detach())  # (G, A)
                nwds = pairwise_box_nwd(gt_boxes, pred_boxes[b].detach())  # (G, A)
                sim_quality = torch.where(is_plate_gt, 0.65 * ious + 0.35 * nwds, ious)
                gt_cls_probs = probs[b, :, gt_labels].transpose(0, 1)  # (G, A)
                align_metric = (gt_cls_probs.clamp_min(1e-6) ** self.alpha) * (sim_quality.clamp_min(1e-6) ** self.beta)
                align_metric = torch.where(cand_mask, align_metric, torch.zeros_like(align_metric))

                k_sel = min(self.topk, num_anchors)
                topk_vals, topk_idx = torch.topk(align_metric, k=k_sel, dim=1)
                pos_mask = torch.zeros_like(cand_mask)
                pos_mask.scatter_(1, topk_idx, topk_vals > 0.0)
                pos_mask = pos_mask & cand_mask

                # Resolve anchors matched to multiple GTs via max IoU
                anchor_pos_count = pos_mask.sum(dim=0)
                if (anchor_pos_count > 1).any():
                    masked_ious = torch.where(pos_mask, ious, torch.full_like(ious, -1.0))
                    best_gt = masked_ious.argmax(dim=0)  # (A,)
                    gt_idx_grid = torch.arange(num_gt, device=device)[:, None]
                    pos_mask = pos_mask & (gt_idx_grid == best_gt[None, :])

                fg_b = pos_mask.any(dim=0)  # (A,)
                if not fg_b.any():
                    continue

                assigned_gt = pos_mask.float().argmax(dim=0)  # (A,)
                pos_align = align_metric * pos_mask.float()
                max_align_per_gt = pos_align.amax(dim=1, keepdim=True).clamp_min(1e-6)
                max_sim_per_gt = (sim_quality * pos_mask.float()).amax(dim=1, keepdim=True)
                norm_align = ((pos_align / max_align_per_gt) * max_sim_per_gt).amax(dim=0)  # (A,)

                fg_idx = torch.nonzero(fg_b, as_tuple=False).squeeze(1)
                gt_sel = assigned_gt[fg_idx]
                cls_sel = gt_labels[gt_sel]
                qual_sel = (pos_align / max_align_per_gt).amax(dim=0)[fg_idx].clamp(0.25, 1.0)

                target_scores[b, fg_idx, cls_sel] = qual_sel
                fg_mask[b] = fg_b
                matched_gt_boxes[b, fg_idx] = gt_boxes[gt_sel]
                matched_gt_corners[b, fg_idx] = gt_corners[gt_sel]
                matched_gt_is_plate[b, fg_idx] = cls_sel == CLS_PLATE
                # Boost small plate regression weight slightly so distant 12px plates get crisp boxes
                gt_w = (gt_boxes[gt_sel, 2] - gt_boxes[gt_sel, 0]).clamp_min(4.0)
                small_boost = torch.where(
                    (cls_sel == CLS_PLATE) & (gt_w < 32.0),
                    torch.full_like(qual_sel, 1.65),
                    torch.ones_like(qual_sel),
                )
                matched_weights[b, fg_idx] = qual_sel * small_boost

        # 1. Varifocal Quality Focal BCE Loss across all anchors (stronger negative background focal suppression)
        pred_prob = torch.sigmoid(cls_logits)
        focal_weight = torch.where(
            target_scores > 0.0,
            target_scores,
            1.35 * (pred_prob.detach() ** 2.0),
        )
        if ignore_anchor_mask.any():
            neg_ignore = (~fg_mask) & ignore_anchor_mask
            focal_weight = torch.where(neg_ignore.unsqueeze(-1), torch.zeros_like(focal_weight), focal_weight)
        bce = F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="none")
        norm_denom = target_scores.sum().clamp_min(3.0)
        loss_cls = (bce * focal_weight).sum() / norm_denom

        # 2. Hybrid CIoU + NWD Box Regression Loss on positive anchors (FP32 for stability on tiny plates)
        if fg_mask.any():
            p_boxes_fg = pred_boxes[fg_mask].float()
            g_boxes_fg = matched_gt_boxes[fg_mask].float()
            w_fg = matched_weights[fg_mask].float()
            is_pl_fg = matched_gt_is_plate[fg_mask]
            _, ciou = compute_box_iou_ciou(p_boxes_fg, g_boxes_fg)
            nwd_fg = compute_box_nwd(p_boxes_fg, g_boxes_fg)
            per_box = torch.where(is_pl_fg, 0.70 * (1.0 - ciou) + 0.30 * (1.0 - nwd_fg), 1.0 - ciou)
            loss_box = (per_box * w_fg).sum() / w_fg.sum().clamp_min(1e-6)

            # 3. Coarse 4-Corner Loss on positive plate anchors (normalized by plate diag)
            plate_fg = fg_mask & matched_gt_is_plate
            if plate_fg.any():
                p_quad = pred_corners[plate_fg]  # (P, 4, 2)
                g_quad = matched_gt_corners[plate_fg]  # (P, 4, 2)
                g_box_p = matched_gt_boxes[plate_fg]
                diag = torch.hypot(
                    (g_box_p[:, 2] - g_box_p[:, 0]).clamp_min(6.0),
                    (g_box_p[:, 3] - g_box_p[:, 1]).clamp_min(4.0),
                ).view(-1, 1, 1)
                loss_quad = F.smooth_l1_loss(p_quad / diag, g_quad / diag, beta=0.08, reduction="mean")
            else:
                loss_quad = pred_corners.sum() * 0.0
        else:
            loss_box = pred_boxes.sum() * 0.0
            loss_quad = pred_corners.sum() * 0.0

        total = self.cls_weight * loss_cls + self.box_weight * loss_box + self.quad_weight * loss_quad
        return {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "loss_quad": loss_quad.detach(),
            "num_fg": fg_mask.sum().float().detach(),
        }


def letterbox_rgb(
    image_rgb: np.ndarray,
    target_hw: tuple[int, int] = (448, 768),
    pad_val: int = 114,
) -> tuple[np.ndarray, float, int, int]:
    """Aspect-preserving resize + pad to `target_hw` `(H_t, W_t)`.
    Returns `(padded_rgb, scale, pad_left, pad_top)`.
    """
    h0, w0 = image_rgb.shape[:2]
    ht, wt = target_hw
    scale = min(wt / float(max(1, w0)), ht / float(max(1, h0)))
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=interp)
    if nh == ht and nw == wt:
        return resized, scale, 0, 0
    canvas = np.full((ht, wt, 3), pad_val, dtype=np.uint8)
    pad_x = (wt - nw) // 2
    pad_y = (ht - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> list[int]:
    """Fast deterministic greedy NMS on `(N, 4)` `[x0, y0, x1, y1]` boxes."""
    if boxes.shape[0] == 0:
        return []
    order = scores.argsort()[::-1]
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    areas = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(x0[i], x0[rest])
        yy0 = np.maximum(y0[i], y0[rest])
        xx1 = np.minimum(x1[i], x1[rest])
        yy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
        iou = inter / np.maximum(1e-6, areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thresh]
    return keep


@torch.no_grad()
def decode_plate_det_predictions(
    model: PlateDetNet | TemporalPlateDetNet,
    frame_rgb: np.ndarray,
    device: torch.device | str = "cpu",
    input_hw: tuple[int, int] = (448, 768),
    plate_thresh: float = 0.28,
    vehicle_thresh: float = 0.30,
    nms_iou: float = 0.45,
    enable_far_zoom: bool = True,
    max_zoom_rois: int = 6,
) -> FrameDetectionResult:
    """Run full-frame multi-scale detection + optional native-resolution Far-Vehicle Smart-Zoom."""
    if getattr(model, "is_temporal_det_net", False):
        res_t, _ = decode_temporal_det_predictions(
            model,  # type: ignore[arg-type]
            frame_rgb,
            device=device,
            input_hw=input_hw,
            plate_thresh=plate_thresh,
            vehicle_thresh=vehicle_thresh,
            nms_iou=nms_iou,
            enable_far_zoom=enable_far_zoom,
            max_zoom_rois=max_zoom_rois,
            hidden_state=None,
        )
        return res_t
    model.eval()
    dev = torch.device(device)
    h_orig, w_orig = frame_rgb.shape[:2]

    lb_img, scale, pad_x, pad_y = letterbox_rgb(frame_rgb, target_hw=input_hw)
    inp = torch.from_numpy(lb_img).to(dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    cls_logits, pred_boxes, pred_corners, _, _ = model(inp)

    probs = torch.sigmoid(cls_logits[0]).cpu().numpy()  # (A, 2)
    boxes = pred_boxes[0].cpu().numpy()  # (A, 4)
    corners = pred_corners[0].cpu().numpy()  # (A, 4, 2)

    # Map back from letterbox coordinates to original frame pixels
    inv_s = 1.0 / max(1e-6, scale)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) * inv_s, 0.0, float(w_orig))
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) * inv_s, 0.0, float(h_orig))
    corners[:, :, 0] = np.clip((corners[:, :, 0] - pad_x) * inv_s, 0.0, float(w_orig))
    corners[:, :, 1] = np.clip((corners[:, :, 1] - pad_y) * inv_s, 0.0, float(h_orig))

    # Vehicle detections (exclude low-score full-frame background glare boxes while keeping close-up gate trucks!)
    v_scores = probs[:, CLS_VEHICLE]
    s_scores = probs[:, CLS_SIGNBOARD_HARDNEG] if probs.shape[-1] > CLS_SIGNBOARD_HARDNEG else np.zeros_like(v_scores)
    vw_all = boxes[:, 2] - boxes[:, 0]
    vh_all = boxes[:, 3] - boxes[:, 1]
    v_area_frac = (vw_all * vh_all) / max(1.0, float(w_orig * h_orig))
    v_mask = (
        (v_scores >= vehicle_thresh)
        & (v_scores > s_scores)
        & (vw_all >= 12.0)
        & (vh_all >= 10.0)
        & ((vw_all <= 0.92 * w_orig) | ((vw_all <= 0.98 * w_orig) & (v_scores >= 0.45)))
        & ((vh_all <= 0.90 * h_orig) | ((vh_all <= 0.97 * h_orig) & (v_scores >= 0.45)))
        & (~((v_scores < 0.42) & (v_area_frac > 0.48)))
    )
    v_boxes = boxes[v_mask]
    v_sc = v_scores[v_mask]
    v_keep = nms_numpy(v_boxes, v_sc, iou_thresh=0.50)
    vehicles_out: list[tuple[float, float, float, float, float]] = [
        (float(v_boxes[i, 0]), float(v_boxes[i, 1]), float(v_boxes[i, 2]), float(v_boxes[i, 3]), float(v_sc[i]))
        for i in v_keep
    ]

    # Plate detections (Pass 1): enforce class-exclusivity (p_score > v_score & p_score > sign_score),
    # physical plate scale/aspect, and coarse box-vs-quad agreement (since `pred_corners` is ONLY supervised on CLS_PLATE!)
    p_scores = probs[:, CLS_PLATE]
    qw_all = corners[:, :, 0].max(axis=-1) - corners[:, :, 0].min(axis=-1)
    qh_all = corners[:, :, 1].max(axis=-1) - corners[:, :, 1].min(axis=-1)
    aspect_all = vw_all / np.maximum(1e-3, vh_all)
    p_mask = (
        (p_scores >= plate_thresh)
        & (p_scores > v_scores)
        & (p_scores > s_scores)
        & (vw_all >= 5.0)
        & (vh_all >= 3.0)
        & (vw_all <= 0.92 * w_orig)
        & (vh_all <= 0.85 * h_orig)
        & (aspect_all >= 0.55)
        & (aspect_all <= 4.60)
        & (qw_all >= 0.36 * vw_all)
        & (qw_all <= 1.85 * vw_all)
        & (qh_all >= 0.36 * vh_all)
        & (qh_all <= 1.85 * vh_all)
    )
    all_p_boxes = [boxes[p_mask]]
    all_p_scores = [p_scores[p_mask]]
    all_p_corners = [corners[p_mask]]

    # Optional Pass 2: Native-Resolution Smart-Zoom with True Magnification Gain `g = z_sc / scale >= 1.25`
    # Uses a single fixed `(256, 384)` tile shape for clean batching, overlapping sub-tiles on large work vehicles,
    # band-scoped `has_strong` (so windshield wipers never block Pass-2!), and interior tile-edge partial-box rejection!
    zoom_tile_hw = (256, 384)
    if enable_far_zoom and scale < 0.88 and len(vehicles_out) > 0:
        cur_p_boxes = all_p_boxes[0]
        cur_p_scores = all_p_scores[0]
        zoom_crops: list[np.ndarray] = []
        # (z_scale, z_pad_x, z_pad_y, rx0, ry0, rx1, ry1)
        zoom_meta: list[tuple[float, int, int, int, int, int, int]] = []

        g_min = 1.25
        tw_max_src = float(zoom_tile_hw[1]) / max(1e-6, g_min * scale)
        th_max_src = float(zoom_tile_hw[0]) / max(1e-6, g_min * scale)

        # Small/distant vehicles first, then larger trucks/buses up to `vw_lb <= 580` via overlapping sub-tiles
        veh_sorted = sorted(
            vehicles_out,
            key=lambda v: (0 if (v[2] - v[0]) * scale <= 240.0 else 1, -v[4]),
        )
        for vx0, vy0, vx1, vy1, vsc in veh_sorted:
            if len(zoom_crops) >= max_zoom_rois:
                break
            vw = vx1 - vx0
            vh = vy1 - vy0
            vw_lb = vw * scale
            vh_lb = vh * scale
            if not (12.0 <= vw_lb <= 580.0 and vh_lb >= 10.0):
                continue

            # Band-scoped `has_strong`: only skip if a confident Pass-1 plate already lies in the grille/bumper zone
            # (`pcy >= vy0 + 0.28 * vh`), so upper-windshield wipers never block Pass-2 plate recovery!
            has_strong = False
            for pb, ps in zip(cur_p_boxes, cur_p_scores):
                pcx = 0.5 * (pb[0] + pb[2])
                pcy = 0.5 * (pb[1] + pb[3])
                if vx0 <= pcx <= vx1 and (vy0 + 0.28 * vh) <= pcy <= (vy1 + 0.16 * vh) and ps >= 0.68:
                    has_strong = True
                    break
            if has_strong:
                continue

            cand_windows: list[tuple[int, int, int, int]] = []
            tw_lim = max(24.0, float(math.floor(tw_max_src)))
            th_lim = max(24.0, float(math.floor(th_max_src)))
            if math.ceil(1.24 * vw) <= tw_lim and math.ceil(1.24 * vh) <= th_lim:
                mx = 0.12 * vw
                my = 0.12 * vh
                rx0 = max(0, int(math.floor(vx0 - mx)))
                ry0 = max(0, int(math.floor(vy0 - my)))
                rx1 = min(w_orig, rx0 + min(int(tw_lim), max(24, int(math.ceil(vx1 + mx)) - rx0)))
                ry1 = min(h_orig, ry0 + min(int(th_lim), max(24, int(math.ceil(vy1 + my)) - ry0)))
                cand_windows.append((rx0, ry0, rx1, ry1))
            else:
                # Tile across high-grille + lower bumper/underrun plate band with >= 28% overlap and g >= 1.25
                bx0 = max(0.0, vx0 - 0.06 * vw)
                bx1 = min(float(w_orig), vx1 + 0.06 * vw)
                by0 = max(0.0, vy0 + 0.28 * vh)
                by1 = min(float(h_orig), vy1 + 0.14 * vh + 12.0)
                band_w = bx1 - bx0
                band_h = by1 - by0
                if band_w < 24.0 or band_h < 20.0:
                    continue
                t_h = int(min(th_lim, max(32.0, math.ceil(band_h))))
                t_w = int(min(tw_lim, max(48.0, math.ceil(band_w))))
                # Center vertically on lower 65% of plate band (covers both 0.35..0.55 grille & underrun bumper!)
                ty_c = by0 + 0.56 * band_h
                ry0_t = max(0, int(math.floor(min(max(0.0, ty_c - 0.5 * t_h), max(0.0, float(h_orig - t_h))))))
                ry1_t = min(h_orig, ry0_t + t_h)
                if band_w <= t_w * 1.05:
                    rx0_t = max(0, int(math.floor(0.5 * (bx0 + bx1 - t_w))))
                    rx1_t = min(w_orig, rx0_t + t_w)
                    cand_windows.append((rx0_t, ry0_t, rx1_t, ry1_t))
                else:
                    # 2 overlapping horizontal sub-tiles with >= 28% overlap covering left/center & center/right mounts
                    rx0_l = max(0, int(math.floor(bx0)))
                    rx1_l = min(w_orig, rx0_l + t_w)
                    rx1_r = min(w_orig, int(math.ceil(bx1)))
                    rx0_r = max(0, rx1_r - t_w)
                    cand_windows.append((rx0_l, ry0_t, rx1_l, ry1_t))
                    if abs(rx0_r - rx0_l) >= 0.15 * t_w:
                        cand_windows.append((rx0_r, ry0_t, rx1_r, ry1_t))

            for rx0, ry0, rx1, ry1 in cand_windows:
                if len(zoom_crops) >= max_zoom_rois:
                    break
                if rx1 - rx0 < 24 or ry1 - ry0 < 24:
                    continue
                sub_rgb = frame_rgb[ry0:ry1, rx0:rx1]
                z_img, z_sc, z_px, z_py = letterbox_rgb(sub_rgb, target_hw=zoom_tile_hw)
                if (z_sc / max(1e-6, scale)) < (g_min - 1e-3):
                    continue
                zoom_crops.append(z_img)
                zoom_meta.append((z_sc, z_px, z_py, rx0, ry0, rx1, ry1))

        if zoom_crops:
            z_batch = (
                torch.from_numpy(np.stack(zoom_crops, axis=0))
                .to(dev, dtype=torch.float32)
                .permute(0, 3, 1, 2)
                / 255.0
            )
            z_cls, z_boxes, z_corners, _, _ = model(z_batch)
            z_all_probs = torch.sigmoid(z_cls).cpu().numpy()
            z_probs = z_all_probs[:, :, CLS_PLATE]
            z_vprobs = z_all_probs[:, :, CLS_VEHICLE]
            z_sprobs = (
                z_all_probs[:, :, CLS_SIGNBOARD_HARDNEG]
                if z_all_probs.shape[-1] > CLS_SIGNBOARD_HARDNEG
                else np.zeros_like(z_probs)
            )
            z_boxes_np = z_boxes.cpu().numpy()
            z_corners_np = z_corners.cpu().numpy()

            for k_i, (z_sc, z_px, z_py, rx0, ry0, rx1, ry1) in enumerate(zoom_meta):
                zm = (
                    (z_probs[k_i] >= max(plate_thresh, 0.32))
                    & (z_probs[k_i] > z_vprobs[k_i])
                    & (z_probs[k_i] > z_sprobs[k_i])
                )
                if not np.any(zm):
                    continue
                zb = z_boxes_np[k_i, zm].copy()
                zc = z_corners_np[k_i, zm].copy()
                zs = z_probs[k_i, zm]
                inv_z = 1.0 / max(1e-6, z_sc)
                zb[:, [0, 2]] = np.clip((zb[:, [0, 2]] - z_px) * inv_z + rx0, 0.0, float(w_orig))
                zb[:, [1, 3]] = np.clip((zb[:, [1, 3]] - z_py) * inv_z + ry0, 0.0, float(h_orig))
                zc[:, :, 0] = np.clip((zc[:, :, 0] - z_px) * inv_z + rx0, 0.0, float(w_orig))
                zc[:, :, 1] = np.clip((zc[:, :, 1] - z_py) * inv_z + ry0, 0.0, float(h_orig))
                zbw = zb[:, 2] - zb[:, 0]
                zbh = zb[:, 3] - zb[:, 1]
                zqw = zc[:, :, 0].max(axis=-1) - zc[:, :, 0].min(axis=-1)
                zqh = zc[:, :, 1].max(axis=-1) - zc[:, :, 1].min(axis=-1)
                sub_w = float(max(24, rx1 - rx0))
                edge_pad = 3.0 * inv_z
                not_cut_left = (zb[:, 0] > float(rx0) + edge_pad) if rx0 > 2 else np.ones(len(zs), dtype=bool)
                not_cut_right = (zb[:, 2] < float(rx1) - edge_pad) if rx1 < (w_orig - 2) else np.ones(len(zs), dtype=bool)
                not_cut_top = (zb[:, 1] > float(ry0) + edge_pad) if ry0 > 2 else np.ones(len(zs), dtype=bool)
                not_cut_bot = (zb[:, 3] < float(ry1) - edge_pad) if ry1 < (h_orig - 2) else np.ones(len(zs), dtype=bool)
                valid_wh = (
                    (zbw >= 5.0)
                    & (zbh >= 3.0)
                    & (zbw <= min(0.35 * w_orig, 0.65 * sub_w))
                    & ((zbw / np.maximum(1e-3, zbh)) >= 0.58)
                    & ((zbw / np.maximum(1e-3, zbh)) <= 4.50)
                    & (zqw >= 0.38 * zbw)
                    & (zqw <= 1.85 * zbw)
                    & (zqh >= 0.38 * zbh)
                    & (zqh <= 1.85 * zbh)
                    & not_cut_left
                    & not_cut_right
                    & not_cut_top
                    & not_cut_bot
                )
                if np.any(valid_wh):
                    all_p_boxes.append(zb[valid_wh])
                    all_p_scores.append(zs[valid_wh])
                    all_p_corners.append(zc[valid_wh])

    cat_boxes = np.concatenate(all_p_boxes, axis=0)
    cat_scores = np.concatenate(all_p_scores, axis=0)
    cat_corners = np.concatenate(all_p_corners, axis=0)
    p_keep = nms_numpy(cat_boxes, cat_scores, iou_thresh=nms_iou)

    # Suppress oversized bumper+grille boxes that >=85% enclose a tighter high-confidence plate candidate
    if len(p_keep) > 1:
        filtered_keep: list[int] = []
        for idx_a in p_keep:
            ax0, ay0, ax1, ay1 = [float(v) for v in cat_boxes[idx_a]]
            area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
            drop_a = False
            for idx_b in p_keep:
                if idx_a == idx_b:
                    continue
                bx0_c, by0_c, bx1_c, by1_c = [float(v) for v in cat_boxes[idx_b]]
                area_b = max(1.0, (bx1_c - bx0_c) * (by1_c - by0_c))
                if area_a >= 2.6 * area_b and float(cat_scores[idx_b]) >= max(0.32, 0.72 * float(cat_scores[idx_a])):
                    inter = max(0.0, min(ax1, bx1_c) - max(ax0, bx0_c)) * max(0.0, min(ay1, by1_c) - max(ay0, by0_c))
                    if inter / area_b >= 0.85:
                        drop_a = True
                        break
            if not drop_a:
                filtered_keep.append(idx_a)
        p_keep = filtered_keep

    plates_out: list[PlateDetection] = []
    for idx in p_keep:
        bx0, by0, bx1, by1 = [float(v) for v in cat_boxes[idx]]
        pcx = 0.5 * (bx0 + bx1)
        pcy = 0.5 * (by0 + by1)
        parent_veh: tuple[float, float, float, float] | None = None
        best_v_area = float("inf")
        for vx0, vy0, vx1, vy1, _ in vehicles_out:
            vh_cand = max(1.0, vy1 - vy0)
            vw_cand = max(1.0, vx1 - vx0)
            # Tolerate top-truncated cab or low-hanging dump-truck underrun plate at close gate range
            if (vx0 - 0.06 * vw_cand - 6.0) <= pcx <= (vx1 + 0.06 * vw_cand + 6.0) and (vy0 + 0.04 * vh_cand) <= pcy <= (vy1 + 0.18 * vh_cand + 14.0):
                area = vw_cand * vh_cand
                if area < best_v_area:
                    best_v_area = area
                    parent_veh = (vx0, vy0, vx1, vy1)
        plates_out.append(
            PlateDetection(
                bbox_xyxy=(bx0, by0, bx1, by1),
                score=float(cat_scores[idx]),
                corners_xy=cat_corners[idx].astype(np.float32),
                vehicle_bbox_xyxy=parent_veh,
                raw_det_score=float(cat_scores[idx]),
            )
        )

    return FrameDetectionResult(plates=plates_out, vehicle_bboxes_xyxy=vehicles_out)


# ==============================================================================
# Gen-4 (`v0.4.0`) Phase 4: `TemporalPlateDetNet` + `EdgeGuidedVehicleCutoutHead`
# ==============================================================================

NUM_VEHICLE_SUBTYPES: int = 9


@dataclass(frozen=True)
class TemporalDetModelConfig:
    """Configuration for Gen-4/Gen-5 `TemporalPlateDetNet` (Apache-2.0 `DFineCore` + Spatio-Temporal Plate/Cutout Heads)."""

    arch: str = "TemporalPlateDetNet"
    preset: str = "base"
    dfine_variant: str = "small"
    c1: int = 24
    c2: int = 48
    c3: int = 80
    c4: int = 128
    neck_ch: int = 64
    head_ch: int = 64
    veh_roi_ch: int = 64
    cutout_ch: int = 32
    depth_mult: int = 2
    strides: tuple[int, ...] = (4, 8, 16)
    num_classes: int = NUM_GEN4_DET_CLASSES
    num_subtypes: int = NUM_VEHICLE_SUBTYPES
    num_vehicle_queries: int = 300
    max_disp_cells: float = 2.5
    use_v5_heads: bool = True

    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        num_classes: int = NUM_GEN4_DET_CLASSES,
        num_subtypes: int = NUM_VEHICLE_SUBTYPES,
        use_v5_heads: bool = True,
    ) -> TemporalDetModelConfig:
        p = preset.lower().strip()
        if p == "nano":
            return cls(
                preset="nano",
                dfine_variant="nano",
                c1=24,
                c2=48,
                c3=80,
                c4=112,
                neck_ch=36,
                head_ch=56,
                veh_roi_ch=40,
                cutout_ch=28,
                depth_mult=2,
                strides=(4, 8, 16),
                num_classes=int(num_classes),
                num_subtypes=int(num_subtypes),
                num_vehicle_queries=300,
                use_v5_heads=bool(use_v5_heads),
            )
        if p == "base":
            return cls(
                preset="base",
                dfine_variant="small",
                c1=24,
                c2=48,
                c3=80,
                c4=128,
                neck_ch=64,
                head_ch=64,
                veh_roi_ch=64,
                cutout_ch=32,
                depth_mult=2,
                strides=(4, 8, 16),
                num_classes=int(num_classes),
                num_subtypes=int(num_subtypes),
                num_vehicle_queries=300,
                use_v5_heads=bool(use_v5_heads),
            )
        raise ValueError(f"Unknown TemporalPlateDetNet preset: {preset!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TemporalDetModelConfig:
        preset_s = str(d.get("preset", "base"))
        default_variant = "nano" if preset_s == "nano" else "small"
        default_roi_ch = 40 if preset_s == "nano" else 64
        default_cut_ch = 28 if preset_s == "nano" else 32
        return cls(
            arch=str(d.get("arch", "TemporalPlateDetNet")),
            preset=preset_s,
            dfine_variant=str(d.get("dfine_variant", default_variant)),
            c1=int(d["c1"]),
            c2=int(d["c2"]),
            c3=int(d["c3"]),
            c4=int(d["c4"]),
            neck_ch=int(d["neck_ch"]),
            head_ch=int(d["head_ch"]),
            veh_roi_ch=int(d.get("veh_roi_ch", default_roi_ch)),
            cutout_ch=int(d.get("cutout_ch", default_cut_ch)),
            depth_mult=int(d["depth_mult"]),
            strides=tuple(int(s) for s in d.get("strides", (4, 8, 16))),
            num_classes=int(d.get("num_classes", NUM_GEN4_DET_CLASSES)),
            num_subtypes=int(d.get("num_subtypes", NUM_VEHICLE_SUBTYPES)),
            num_vehicle_queries=int(d.get("num_vehicle_queries", 300)),
            max_disp_cells=float(d.get("max_disp_cells", 2.5)),
            use_v5_heads=bool(d.get("use_v5_heads", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "arch": self.arch,
            "preset": self.preset,
            "dfine_variant": self.dfine_variant,
            "c1": int(self.c1),
            "c2": int(self.c2),
            "c3": int(self.c3),
            "c4": int(self.c4),
            "neck_ch": int(self.neck_ch),
            "head_ch": int(self.head_ch),
            "veh_roi_ch": int(self.veh_roi_ch),
            "cutout_ch": int(self.cutout_ch),
            "depth_mult": int(self.depth_mult),
            "strides": [int(s) for s in self.strides],
            "num_classes": int(self.num_classes),
            "num_subtypes": int(self.num_subtypes),
            "num_vehicle_queries": int(self.num_vehicle_queries),
            "max_disp_cells": float(self.max_disp_cells),
        }
        if self.use_v5_heads:
            out["use_v5_heads"] = True
        return out


def _replicate_pad_1px(x: torch.Tensor) -> torch.Tensor:
    """ONNX constant-folding-clean (`steps=1`) 1-pixel edge-replication pad `(B, C, H+2, W+2)`."""
    xh = torch.cat([x[:, :, :1, :], x, x[:, :, -1:, :]], dim=2)
    return torch.cat([xh[:, :, :, :1], xh, xh[:, :, :, -1:]], dim=3)


class LocalPhotometricDetStem(nn.Module):
    """Chroma-enhanced 7-channel (or legacy 5-ch) photometric + multi-scale local Z-score stem (`k=9` + `k=25`).

    Adds green-plate chromaticity `(g - 0.5*(r + b)) / (y + 0.06)` and yellow-plate chromaticity
    `(0.5*(r + g) - b) / (y + 0.06)` internally so dark commercial green plates (`事業用緑ナンバー`) mounted
    inside matte-black bumpers (`w3/03`) pop strongly even when luminance contrast approaches zero.
    Also emits `aux_edge_3ch = [z9, z25, sobel_mag]` at native `(H, W)` resolution for razor-sharp
    `64x64` vehicle silhouette boundary refinement in `EdgeGuidedVehicleCutoutHead`.
    """

    def __init__(self, out_ch: int, noise_floor: float = 0.025, use_v5_chroma: bool = True) -> None:
        super().__init__()
        self.noise_floor_sq = float(noise_floor * noise_floor)
        self.use_v5_chroma = bool(use_v5_chroma)
        rgb_w = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
        sobel_x = torch.tensor(
            [[-0.25, 0.0, 0.25], [-0.50, 0.0, 0.50], [-0.25, 0.0, 0.25]],
            dtype=torch.float32,
        )
        sobel_y = sobel_x.t().contiguous()
        sobel_k = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)  # (2, 1, 3, 3)
        self.register_buffer("rgb_to_y", rgb_w, persistent=False)
        self.register_buffer("sobel_kernels", sobel_k, persistent=False)
        stem_in_ch = 7 if self.use_v5_chroma else 5
        self.conv_stem = nn.Sequential(
            ConvBNAct(stem_in_ch, out_ch, k=3, s=2),
            ConvBNAct(out_ch, out_ch, k=3, s=1),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # `images`: (B, 3, H, W) in [0, 1]
        y = (images * self.rgb_to_y.to(dtype=images.dtype)).sum(dim=1, keepdim=True)
        y2 = y * y

        mu9 = F.avg_pool2d(y, kernel_size=9, stride=1, padding=4, count_include_pad=False)
        var9 = (F.avg_pool2d(y2, kernel_size=9, stride=1, padding=4, count_include_pad=False) - mu9 * mu9).clamp_min(0.0)
        z9 = torch.clamp((y - mu9) / torch.sqrt(var9 + self.noise_floor_sq), -3.5, 3.5) * 0.5

        mu25 = F.avg_pool2d(y, kernel_size=25, stride=1, padding=12, count_include_pad=False)
        var25 = (F.avg_pool2d(y2, kernel_size=25, stride=1, padding=12, count_include_pad=False) - mu25 * mu25).clamp_min(0.0)
        z25 = torch.clamp((y - mu25) / torch.sqrt(var25 + self.noise_floor_sq), -3.5, 3.5) * 0.5

        y_pad = _replicate_pad_1px(y)
        grad = F.conv2d(y_pad, self.sobel_kernels.to(dtype=images.dtype))
        sobel_mag = torch.clamp(torch.sqrt((grad * grad).sum(dim=1, keepdim=True) + 1e-6), 0.0, 2.0)

        rgb_norm = (images - 0.45) / 0.25
        if self.use_v5_chroma:
            r_ch = images[:, 0:1]
            g_ch = images[:, 1:2]
            b_ch = images[:, 2:3]
            denom_c = y + 0.06
            chroma_green = torch.clamp((g_ch - 0.5 * (r_ch + b_ch)) / denom_c, -2.5, 2.5)
            chroma_yellow = torch.clamp((0.5 * (r_ch + g_ch) - b_ch) / denom_c, -2.5, 2.5)
            stem_in = torch.cat([rgb_norm, z9, z25, chroma_green, chroma_yellow], dim=1)
        else:
            stem_in = torch.cat([rgb_norm, z9, z25], dim=1)
        c1 = self.conv_stem(stem_in)
        aux_edge = torch.cat([z9, z25, sobel_mag], dim=1)
        return c1, aux_edge


class DWConvGRUCell(nn.Module):
    """Motion-compensated depthwise-separable ConvGRU pyramid memory cell with `d=1` + `d=2` context.

    Aligns `h_prev` to `x_t` via bounded sub-cell flow warping (`F.grid_sample(..., padding_mode='zeros')`),
    captures larger inter-frame motion via parallel `k=3, d=1` + `k=3, d=2` depthwise spatial context,
    updates gated spatial state `h_new`, and injects residual memory `x_t + mem_proj(h_new)`.
    """

    def __init__(self, ch: int, max_disp_cells: float = 2.5) -> None:
        super().__init__()
        self.ch = int(ch)
        self.max_disp_cells = float(max_disp_cells)
        mid = max(16, ch // 2)
        self.flow_pw = ConvBNAct(ch * 2, mid, k=1, s=1)
        self.flow_dw1 = ConvBNAct(mid, mid, k=3, s=1, g=mid, d=1)
        self.flow_dw2 = ConvBNAct(mid, mid, k=3, s=1, g=mid, d=2)
        self.flow_pred = nn.Conv2d(mid, 2, kernel_size=3, padding=1)
        nn.init.normal_(self.flow_pred.weight, std=1e-4)
        nn.init.zeros_(self.flow_pred.bias)

        self.gate_dw = ConvBNAct(ch * 2, ch * 2, k=3, s=1, g=ch * 2, d=1)
        self.gate_dw_d2 = ConvBNAct(ch * 2, ch * 2, k=3, s=1, g=ch * 2, d=2)
        self.gate_pw = nn.Conv2d(ch * 2, ch * 2, kernel_size=1)
        nn.init.normal_(self.gate_pw.weight, std=0.01)
        nn.init.zeros_(self.gate_pw.bias)
        with torch.no_grad():
            self.gate_pw.bias[ch:].fill_(-0.45)  # initial update gate ~0.39

        self.cand_pw = ConvBNAct(ch * 2, ch, k=1, s=1)
        self.cand_dw1 = ConvBNAct(ch, ch, k=3, s=1, g=ch, d=1)
        self.cand_dw2 = ConvBNAct(ch, ch, k=3, s=1, g=ch, d=2)
        self.mem_proj = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        nn.init.constant_(self.mem_proj[1].weight, 0.25)
        nn.init.zeros_(self.mem_proj[1].bias)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        hl = int(x_t.shape[-2])
        wl = int(x_t.shape[-1])
        if h_prev is None:
            h_prev = torch.zeros_like(x_t)
        elif not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()) and h_prev.shape[-2:] != (hl, wl):
            h_prev = F.interpolate(h_prev, size=(hl, wl), mode="bilinear", align_corners=False)

        flow_in = self.flow_pw(torch.cat([x_t, h_prev], dim=1))
        delta_cells = torch.tanh(self.flow_pred(self.flow_dw1(flow_in) + self.flow_dw2(flow_in))) * self.max_disp_cells
        inv_w = 2.0 / float(max(1, wl))
        inv_h = 2.0 / float(max(1, hl))
        xs = ((torch.arange(wl, device=x_t.device, dtype=x_t.dtype) + 0.5) * inv_w - 1.0).view(1, 1, wl)
        ys = ((torch.arange(hl, device=x_t.device, dtype=x_t.dtype) + 0.5) * inv_h - 1.0).view(1, hl, 1)
        wx = xs + delta_cells[:, 0] * inv_w
        wy = ys + delta_cells[:, 1] * inv_h
        warp_grid = torch.stack([wx, wy], dim=-1)
        h_aligned = F.grid_sample(
            h_prev,
            warp_grid.to(dtype=h_prev.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        gx = torch.cat([x_t, h_aligned], dim=1)
        gates = torch.sigmoid(self.gate_pw(self.gate_dw(gx) + self.gate_dw_d2(gx)))
        r_gate, u_gate = gates.chunk(2, dim=1)
        cx = self.cand_pw(torch.cat([x_t, r_gate * h_aligned], dim=1))
        cand = self.cand_dw1(cx) + self.cand_dw2(cx)
        h_new = (1.0 - u_gate) * h_aligned + u_gate * cand
        out_feat = x_t + self.mem_proj(h_new)
        return out_feat, h_new


ConvGRUCell = DWConvGRUCell


class PlateTemporalScaleHead(nn.Module):
    """Decoupled high-acuity `[plate, signboard_hardneg]` + LTRB + 4-corner head for `P2/P3/P4`."""

    def __init__(self, in_ch: int, head_ch: int) -> None:
        super().__init__()
        if head_ch <= 48:
            self.cls_stem = nn.Sequential(
                ConvBNAct(in_ch, in_ch, k=3, s=1, g=in_ch),
                ConvBNAct(in_ch, head_ch, k=1, s=1),
            )
            self.reg_stem = nn.Sequential(
                ConvBNAct(in_ch, in_ch, k=3, s=1, g=in_ch),
                ConvBNAct(in_ch, head_ch, k=1, s=1),
            )
        else:
            self.cls_stem = nn.Sequential(
                ConvBNAct(in_ch, in_ch, k=3, s=1, g=in_ch),
                ConvBNAct(in_ch, head_ch, k=1, s=1),
                ConvBNAct(head_ch, head_ch, k=3, s=1, g=head_ch),
                ConvBNAct(head_ch, head_ch, k=1, s=1),
            )
            self.reg_stem = nn.Sequential(
                ConvBNAct(in_ch, in_ch, k=3, s=1, g=in_ch),
                ConvBNAct(in_ch, head_ch, k=1, s=1),
                ConvBNAct(head_ch, head_ch, k=3, s=1, g=head_ch),
                ConvBNAct(head_ch, head_ch, k=1, s=1),
            )
        # 2 foreground/hardneg channels on dense pyramid: [CLS_PLATE, CLS_SIGNBOARD_HARDNEG]
        self.cls_pred = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_ch, 4, kernel_size=1)
        self.quad_pred = nn.Conv2d(head_ch, 8, kernel_size=1)
        self.reg_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

        nn.init.constant_(self.cls_pred.bias, -math.log((1.0 - 0.01) / 0.01))
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 1.5)
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.zeros_(self.quad_pred.bias)
        nn.init.normal_(self.quad_pred.weight, std=0.01)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_f = self.cls_stem(feat)
        r_f = self.reg_stem(feat)
        cls_2ch = self.cls_pred(c_f)
        ltrb_norm = F.softplus(self.reg_pred(r_f) * self.reg_scale.clamp(0.2, 5.0))
        quad_norm = self.quad_pred(r_f)
        return cls_2ch, ltrb_norm, quad_norm


class _QueryContextMLP(nn.Module):
    """Compact O(Q) per-query residual MLP with pooled scene context (replaces quadratic 300x300 self-attention)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(dim)
        self.ctx_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )
        nn.init.zeros_(self.ctx_proj.bias)
        nn.init.normal_(self.ctx_proj.weight, std=0.01)
        nn.init.zeros_(self.ffn[2].bias)
        nn.init.normal_(self.ffn[2].weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scene_ctx = h.mean(dim=1, keepdim=True)
        return x + self.ffn(h + self.ctx_proj(scene_ctx))


class VehicleQueryHead(nn.Module):
    """Frozen COCO vehicle probability & boxes + synthetic-trained Re-ID/attribute/mount/cutout query head.

    `veh_logit_q` and `pred_boxes_px` are derived strictly from the frozen Apache-2.0 `DFineCore`
    COCO logits & boxes with zero synthetic-trained residual overrides or box warping.
    """

    def __init__(
        self,
        roi_feat_ch: int,
        dfine_d_model: int,
        hidden_dim: int = 96,
        spatial_ch: int = 16,
        num_subtypes: int = NUM_VEHICLE_SUBTYPES,
        use_v5_heads: bool = True,
    ) -> None:
        super().__init__()
        self.use_v5_heads = bool(use_v5_heads)
        self.grid_s = 5
        self.roi_spatial_conv = ConvBNAct(roi_feat_ch, spatial_ch, k=3, s=1)
        roi_flat_dim = spatial_ch * self.grid_s * self.grid_s
        self.roi_proj = nn.Sequential(
            nn.Linear(roi_flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        # Fuse query embeddings, multiscale ROI features, geometry and all class logits for attribute/Re-ID heads.
        q_in_dim = int(dfine_d_model) + hidden_dim + 5 + 80
        attn_dim = hidden_dim if hidden_dim % 4 == 0 else ((hidden_dim + 3) // 4) * 4
        self.query_in_proj = nn.Sequential(
            nn.Linear(q_in_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.SiLU(inplace=True),
        )
        self.query_ctx_mlp = _QueryContextMLP(attn_dim)

        # All observation attributes share one learned query representation.
        self.attr_mlp = nn.Sequential(nn.Linear(attn_dim, hidden_dim), nn.SiLU())
        self.subtype_pred = nn.Linear(hidden_dim, num_subtypes + 1)
        coco_veh = torch.tensor([2, 3, 5, 7], dtype=torch.long)
        coco_non_veh = torch.tensor([i for i in range(80) if i not in (2, 3, 5, 7)], dtype=torch.long)
        self.register_buffer("coco_veh_idx", coco_veh, persistent=False)
        self.register_buffer("coco_non_veh_idx", coco_non_veh, persistent=False)
        self.register_buffer("cal_temp", torch.tensor(1.0))
        self.register_buffer("cal_bias", torch.tensor(0.0))
        self.cov_pred = nn.Linear(attn_dim, 10)
        self.appearance_pred = nn.Linear(attn_dim, 48)
        rows, cols = torch.tril_indices(4, 4)
        self.register_buffer("chol_basis", F.one_hot(rows * 4 + cols, 16).float(), persistent=False)
        self.register_buffer("chol_diag", (rows == cols).float(), persistent=False)
        nn.init.zeros_(self.cov_pred.weight)
        nn.init.zeros_(self.cov_pred.bias)
        with torch.no_grad():
            # Initial normalized localization standard deviation: 2% of frame size.
            self.cov_pred.bias[rows == cols] = math.log(math.expm1(0.02))
        self.cabin_pred = nn.Linear(hidden_dim, 5)      # [valid_logit, uc, vc, wu, hv]
        self.footprint_pred = nn.Linear(hidden_dim, 9)  # [valid_logit, 4*(dx, dy)]
        self.plate_mount_pred = nn.Linear(hidden_dim, 5)  # [mcx, mcy, mw, mh, mount_vis_logit]

        nn.init.zeros_(self.subtype_pred.bias)
        nn.init.normal_(self.subtype_pred.weight, std=0.02)
        nn.init.zeros_(self.cabin_pred.bias)
        nn.init.normal_(self.cabin_pred.weight, std=0.01)
        nn.init.zeros_(self.footprint_pred.bias)
        nn.init.normal_(self.footprint_pred.weight, std=0.01)
        with torch.no_grad():
            self.footprint_pred.bias[1:].copy_(
                torch.tensor([-0.45, -0.05, 0.45, -0.05, 0.48, 0.02, -0.48, 0.02], dtype=torch.float32)
            )
            nn.init.normal_(self.plate_mount_pred.weight, std=0.01)
            nn.init.zeros_(self.plate_mount_pred.bias)

    @torch.no_grad()
    def set_calibration(self, temperature: float, bias: float) -> None:
        """Load calibration fitted on a disjoint calibration partition after export."""
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(bias):
            raise ValueError("Calibration requires finite temperature > 0 and finite bias")
        self.cal_temp.fill_(temperature)
        self.cal_bias.fill_(bias)

    def _pool_rois(self, veh_feat_s8: torch.Tensor, boxes_xyxy_norm: torch.Tensor) -> torch.Tensor:
        bsz, num_q, _ = boxes_xyxy_norm.shape
        ch = int(veh_feat_s8.shape[1])
        s = self.grid_s
        feat_src = veh_feat_s8
        u = (torch.arange(s, device=boxes_xyxy_norm.device, dtype=torch.float32) + 0.5) / float(s)
        x0, y0, x1, y1 = boxes_xyxy_norm.float().unbind(-1)
        bw = (x1 - x0).clamp_min(1e-3)
        bh = (y1 - y0).clamp_min(1e-3)
        gx = 2.0 * (x0.unsqueeze(-1) + bw.unsqueeze(-1) * u.view(1, 1, s)) - 1.0  # (B, Q, S)
        gy = 2.0 * (y0.unsqueeze(-1) + bh.unsqueeze(-1) * u.view(1, 1, s)) - 1.0  # (B, Q, S)
        grid_x = gx.unsqueeze(2).expand(bsz, num_q, s, s)
        grid_y = gy.unsqueeze(3).expand(bsz, num_q, s, s)
        packed_grid = torch.stack([grid_x, grid_y], dim=-1).reshape(bsz, num_q * s, s, 2)
        sampled = F.grid_sample(
            feat_src,
            packed_grid.to(dtype=feat_src.dtype),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # (B, C, Q*S, S)
        roi_sxs = sampled.reshape(bsz, ch, num_q, s, s).permute(0, 2, 1, 3, 4).reshape(bsz * num_q, ch, s, s)
        roi_sp = self.roi_spatial_conv(roi_sxs).float().reshape(bsz, num_q, -1)
        return self.roi_proj(roi_sp)

    def forward(
        self,
        veh_feat_s8: torch.Tensor,
        dfine_logits_80: torch.Tensor,
        dfine_boxes_cxcywh: torch.Tensor,
        dfine_query_embeds: torch.Tensor,
        img_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        logits_80 = dfine_logits_80.float()
        boxes_cxcywh = dfine_boxes_cxcywh.float().clamp(0.0, 1.0)
        q_embeds = dfine_query_embeds.float()
        bsz, num_q, _ = boxes_cxcywh.shape
        img_h, img_w = float(img_hw[0]), float(img_hw[1])

        raw_xyxy_norm = center_to_corners_format_torch(boxes_cxcywh).clamp(0.0, 1.0)
        scale_xyxy = raw_xyxy_norm.new_tensor([img_w, img_h, img_w, img_h])
        raw_boxes_px = raw_xyxy_norm * scale_xyxy

        # Pure, uncorrupted frozen COCO vehicle vs non-vehicle probability directly from `dfine_core`.
        # Non-vehicle suppression ensures persons, mascots, signs, bicycles, etc. suppress weak vehicle logits.
        probs_80 = logits_80.sigmoid()
        coco_veh_p = probs_80.index_select(-1, self.coco_veh_idx).amax(dim=-1)
        coco_non_p = probs_80.index_select(-1, self.coco_non_veh_idx).amax(dim=-1)
        eff_p = (coco_veh_p * torch.sigmoid((coco_veh_p - 0.85 * coco_non_p) * 12.0)).clamp(1e-5, 1.0 - 1e-5)
        dfine_veh_logit = torch.log(eff_p / (1.0 - eff_p))
        veh_logit_q = dfine_veh_logit / self.cal_temp + self.cal_bias
        # Zero synthetic box warping: preserve exact frozen D-FINE COCO vehicle boxes.
        pred_boxes_px = raw_boxes_px

        roi_emb = self._pool_rois(veh_feat_s8, raw_xyxy_norm)  # (B, Q, D_h)
        roi_emb = F.dropout(roi_emb, p=0.20, training=self.training)
        cx_n, cy_n, bw_n, bh_n = boxes_cxcywh.clamp(1e-3, 1.0).unbind(-1)
        log_ar = torch.log(bw_n / bh_n).clamp(-3.0, 3.0)
        geom_5 = torch.stack([cx_n, cy_n, bw_n, bh_n, log_ar], dim=-1)

        q_in = torch.cat([q_embeds, roi_emb, geom_5, logits_80], dim=-1)
        q_ctx = self.query_ctx_mlp(self.query_in_proj(q_in))
        chol_raw = self.cov_pred(q_ctx)
        # Preserve the positive diagonal and Gram matrix precision under AMP.
        with torch.amp.autocast(device_type=q_ctx.device.type, enabled=False):
            chol_raw = chol_raw.float()
            chol_entries = chol_raw.clamp(-0.12, 0.12) * (1.0 - self.chol_diag) + (F.softplus(chol_raw.clamp(-8.0, -1.8)) + 5e-4) * self.chol_diag
            chol_norm = (chol_entries @ self.chol_basis).reshape(bsz, num_q, 4, 4)
            chol_px = chol_norm * scale_xyxy.view(1, 1, 4, 1)
            bbox_cov = chol_px @ chol_px.transpose(-1, -2) + torch.eye(4, device=q_ctx.device) * 1e-4
        appearance = F.normalize(self.appearance_pred(q_ctx), dim=-1, eps=1e-6)
        attr_h = self.attr_mlp(q_ctx)
        subtype_logits = self.subtype_pred(attr_h)
        cab_raw = self.cabin_pred(attr_h)
        fp_raw = self.footprint_pred(attr_h)

        vx0 = pred_boxes_px[:, :, 0]
        vy0 = pred_boxes_px[:, :, 1]
        vw = (pred_boxes_px[:, :, 2] - vx0).clamp_min(2.0)
        vh = (pred_boxes_px[:, :, 3] - vy0).clamp_min(2.0)
        uc = 1.16 * torch.sigmoid(cab_raw[:, :, 1]) - 0.08
        vc = 1.16 * torch.sigmoid(cab_raw[:, :, 2]) - 0.08
        wu = 1.08 * torch.sigmoid(cab_raw[:, :, 3]).clamp_min(0.04)
        hv = 1.08 * torch.sigmoid(cab_raw[:, :, 4]).clamp_min(0.04)
        cabin_boxes = torch.stack(
            [
                vx0 + (uc - 0.5 * wu) * vw,
                vy0 + (vc - 0.5 * hv) * vh,
                vx0 + (uc + 0.5 * wu) * vw,
                vy0 + (vc + 0.5 * hv) * vh,
            ],
            dim=-1,
        )

        bot_cx = 0.5 * (pred_boxes_px[:, :, 0] + pred_boxes_px[:, :, 2])
        bot_cy = pred_boxes_px[:, :, 3]
        fp_off = fp_raw[:, :, 1:].reshape(bsz, num_q, 4, 2)
        fp_scale = torch.stack([vw, vh], dim=-1).unsqueeze(2)
        fp_center = torch.stack([bot_cx, bot_cy], dim=-1).unsqueeze(2)
        footprints = fp_center + fp_off * fp_scale

        # Axis-aligned 4-corner dummy representation for query boxes (corners only used for CLS_PLATE)
        q_corners = torch.stack(
            [
                torch.stack([pred_boxes_px[:, :, 0], pred_boxes_px[:, :, 1]], dim=-1),
                torch.stack([pred_boxes_px[:, :, 2], pred_boxes_px[:, :, 1]], dim=-1),
                torch.stack([pred_boxes_px[:, :, 2], pred_boxes_px[:, :, 3]], dim=-1),
                torch.stack([pred_boxes_px[:, :, 0], pred_boxes_px[:, :, 3]], dim=-1),
            ],
            dim=2,
        )

        out_q: dict[str, torch.Tensor] = {
            "veh_logit_q": veh_logit_q,
            "synth_adapter_logit": veh_logit_q,
            "veh_log_odds": veh_logit_q,
            "bbox_cov_4x4": bbox_cov,
            "appearance_emb": appearance,
            "query_condition": q_ctx,
            "subtype_probs": torch.softmax(subtype_logits, dim=-1)[..., :-1],
            "subtype_unknown_prob": torch.softmax(subtype_logits, dim=-1)[..., -1],
            "dfine_veh_logit": dfine_veh_logit,
            "raw_q_boxes_px": raw_boxes_px,
            "pred_q_boxes": pred_boxes_px,
            "pred_q_corners": q_corners,
            "pred_q_subtypes": subtype_logits,
            "pred_q_cabin_valid_logit": cab_raw[:, :, 0:1],
            "pred_q_cabin_boxes": cabin_boxes,
            "pred_q_footprint_valid_logit": fp_raw[:, :, 0:1],
            "pred_q_footprints": footprints,
        }
        m_raw = self.plate_mount_pred(attr_h)
        mcx_r = 1.16 * torch.sigmoid(m_raw[:, :, 0]) - 0.08
        mcy_r = 1.24 * torch.sigmoid(m_raw[:, :, 1]) - 0.06
        mw_r = (0.65 * torch.sigmoid(m_raw[:, :, 2])).clamp_min(0.04)
        mh_r = (0.45 * torch.sigmoid(m_raw[:, :, 3])).clamp_min(0.03)
        mount_boxes = torch.stack(
            [
                vx0 + (mcx_r - 0.5 * mw_r) * vw,
                vy0 + (mcy_r - 0.5 * mh_r) * vh,
                vx0 + (mcx_r + 0.5 * mw_r) * vw,
                vy0 + (mcy_r + 0.5 * mh_r) * vh,
            ],
            dim=-1,
        )
        out_q["pred_q_mount_boxes"] = mount_boxes
        out_q["pred_q_mount_vis_logit"] = m_raw[:, :, 4:5]
        return out_q


class EdgeGuidedVehicleCutoutHead(nn.Module):
    """Crisp `64x64` instance-conditioned fractional optical vehicle silhouette segmentation head.

    Samples `P2` (`stride=4`, fused with `DFineCore` `C2/C3/PAN`) onto `32x32` with canonical ROI-relative
    coordinates `[u_rel, v_rel, r_box]` + global ROI center-vs-border context, then upsamples `2x` to `64x64`
    and fuses with native-resolution `(z9, z25, sobel_mag)` boundary cues at exact `align_corners=False`
    pixel-center coordinates (`0.000` sub-pixel drift).
    """

    def __init__(self, feat_ch: int, aux_ch: int = 3, hidden_ch: int = 24, query_dim: int = 64) -> None:
        super().__init__()
        self.register_buffer("coord_grid_32", self._make_canonical_coord_grid(32), persistent=False)
        self.register_buffer("coord_grid_64", self._make_canonical_coord_grid(64), persistent=False)
        self.coarse_in = ConvBNAct(feat_ch + 3, hidden_ch, k=3, s=1)
        self.coarse_global = nn.Sequential(
            nn.AvgPool2d(kernel_size=32, stride=1),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.coarse_net = nn.Sequential(
            DWConvBlock(hidden_ch, k=5),
            DWConvBlock(hidden_ch, k=5),
        )
        self.fine_net = nn.Sequential(
            ConvBNAct(hidden_ch + aux_ch + 3, hidden_ch, k=3, s=1),
            DWConvBlock(hidden_ch, k=3),
            ConvBNAct(hidden_ch, hidden_ch, k=3, s=1, g=hidden_ch),
        )
        self.query_dim = query_dim
        self.dynamic_filter = nn.Linear(query_dim + 4, hidden_ch + 1)
        nn.init.normal_(self.dynamic_filter.weight, std=0.02)
        nn.init.zeros_(self.dynamic_filter.bias)

    @staticmethod
    def _make_canonical_coord_grid(size: int) -> torch.Tensor:
        u = ((torch.arange(size, dtype=torch.float32) + 0.5) / float(size)) * 2.0 - 1.0
        yy, xx = torch.meshgrid(u, u, indexing="ij")
        r_box = torch.maximum(xx.abs(), yy.abs())
        return torch.stack([xx, yy, r_box], dim=0).unsqueeze(0)  # (1, 3, S, S)

    @staticmethod
    def _build_roi_grid(
        boxes_xyxy: torch.Tensor,
        img_h: int,
        img_w: int,
        out_s: int,
    ) -> torch.Tensor:
        """Exact `align_corners=False` normalized grid `(N, out_s, out_s, 2)` over `[x0, y0, x1, y1]`."""
        device = boxes_xyxy.device
        dtype = boxes_xyxy.dtype
        u = (torch.arange(out_s, device=device, dtype=dtype) + 0.5) / float(out_s)  # (S,)
        x0 = boxes_xyxy[:, 0:1]
        y0 = boxes_xyxy[:, 1:2]
        bw = (boxes_xyxy[:, 2:3] - x0).clamp_min(1.0)
        bh = (boxes_xyxy[:, 3:4] - y0).clamp_min(1.0)
        xs = 2.0 * (x0 + bw * u.unsqueeze(0)) / float(max(1, img_w)) - 1.0  # (N, S)
        ys = 2.0 * (y0 + bh * u.unsqueeze(0)) / float(max(1, img_h)) - 1.0  # (N, S)
        gx = xs.unsqueeze(1).expand(-1, out_s, out_s)
        gy = ys.unsqueeze(2).expand(-1, out_s, out_s)
        return torch.stack([gx, gy], dim=-1)

    def forward(
        self,
        p2_feat: torch.Tensor,
        aux_edge_3ch: torch.Tensor,
        veh_rois: torch.Tensor,
        query_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict `(N_veh, 1, 64, 64)` cutout logits for `veh_rois` `(N_veh, 5)` `[batch_idx, x0, y0, x1, y1]`."""
        img_h, img_w = int(aux_edge_3ch.shape[-2]), int(aux_edge_3ch.shape[-1])

        if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()) and veh_rois.shape[0] == 0:
            return p2_feat.new_zeros((0, 1, 64, 64))

        # Gather by ROI batch index: no traced Python branch or fixed batch loop.
        b_idx = veh_rois[:, 0].long()
        p2_rois = F.grid_sample(
            p2_feat.index_select(0, b_idx),
            self._build_roi_grid(veh_rois[:, 1:5], img_h, img_w, 32).to(p2_feat.dtype),
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        edge_rois = F.grid_sample(
            aux_edge_3ch.index_select(0, b_idx),
            self._build_roi_grid(veh_rois[:, 1:5], img_h, img_w, 64).to(aux_edge_3ch.dtype),
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        # Broadcast canonical ROI coordinates dynamically for any N_veh without baking static batch size in ONNX
        coord32 = self.coord_grid_32.to(dtype=p2_rois.dtype, device=p2_rois.device) + (p2_rois[:, :1] * 0.0)
        h32_in = self.coarse_in(torch.cat([p2_rois, coord32], dim=1))
        h32 = self.coarse_net(h32_in + self.coarse_global(h32_in))
        h64 = F.interpolate(h32, size=(64, 64), mode="bilinear", align_corners=False)
        coord64 = self.coord_grid_64.to(dtype=h64.dtype, device=h64.device) + (h64[:, :1] * 0.0)
        fused = self.fine_net(torch.cat([h64, edge_rois.to(dtype=h64.dtype), coord64], dim=1))
        box = veh_rois[:, 1:5]
        geometry = box / box.new_tensor([img_w, img_h, img_w, img_h])
        theta = self.dynamic_filter(torch.cat([query_condition.to(fused.dtype), geometry], dim=-1))
        weights = theta[:, :-1, None, None]
        bias = theta[:, -1:, None, None]
        return (weights * fused).sum(dim=1, keepdim=True) + bias


class TemporalPlateDetNet(nn.Module):
    """Gen-6 calibrated query observations, temporal plate pyramid and dynamic instance masks."""

    is_temporal_det_net: bool = True

    def __init__(
        self,
        config: TemporalDetModelConfig | None = None,
        *,
        pretrained_dir: str | Path | None = None,
        load_pretrained_dfine: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or TemporalDetModelConfig.from_preset("base")
        cfg = self.config
        nc = cfg.neck_ch

        if load_pretrained_dfine:
            self.dfine_core = DFineCore.from_pretrained_apache20(
                cfg.dfine_variant, pretrained_dir=pretrained_dir, freeze=True
            )
        else:
            self.dfine_core = DFineCore(DFineCoreConfig.from_variant(cfg.dfine_variant))
            self.dfine_core.freeze_all()

        df_cfg = self.dfine_core.cfg
        df_c2_ch = int(df_cfg.stage_out_channels[0])  # 64
        df_c3_ch = int(df_cfg.stage_out_channels[1])  # 256
        df_pan_ch = int(df_cfg.encoder_hidden_dim)    # 128 (nano) or 256 (small)

        # High-acuity photometric & local Z-score branch for tiny/dark/glare Japanese license plates & edges
        self.photo_stem = LocalPhotometricDetStem(cfg.c1, use_v5_chroma=cfg.use_v5_heads)
        self.stage2 = StageBlock(cfg.c1, cfg.c2, num_blocks=cfg.depth_mult, stride=2)      # stride 4  (C2)
        self.stage3 = StageBlock(cfg.c2, cfg.c3, num_blocks=cfg.depth_mult + 1, stride=2)  # stride 8  (C3)
        self.stage4 = StageBlock(cfg.c3, cfg.c4, num_blocks=cfg.depth_mult + 1, stride=2)  # stride 16 (C4)

        # Fuse photometric multi-scale features with detached D-FINE semantic features
        self.lat4 = ConvBNAct(cfg.c4 + df_pan_ch, nc, k=1, s=1)
        self.p4_ctx = DWConvBlock(nc, k=5)
        self.lat3 = ConvBNAct(cfg.c3 + df_c3_ch, nc, k=1, s=1)
        self.lat2 = ConvBNAct(cfg.c2 + df_c2_ch, nc, k=1, s=1)

        self.td3 = nn.Sequential(ConvBNAct(nc * 2, nc, k=1, s=1), DWConvBlock(nc, k=5))
        self.p2_temp_gate = nn.Sequential(
            ConvBNAct(nc, nc, k=3, s=1, g=nc),
            nn.Conv2d(nc, nc, kernel_size=1),
        )
        nn.init.normal_(self.p2_temp_gate[1].weight, std=0.01)
        nn.init.zeros_(self.p2_temp_gate[1].bias)
        self.p2_out = nn.Sequential(DWConvBlock(nc, k=5), DWConvBlock(nc, k=3))

        # Motion-compensated recurrent temporal memory on P3 (stride 8) & P4 (stride 16)
        self.gru_p3 = DWConvGRUCell(nc, max_disp_cells=cfg.max_disp_cells)
        self.gru_p4 = DWConvGRUCell(nc, max_disp_cells=cfg.max_disp_cells)

        # Bottom-up PAN (P2 -> N3 -> N4)
        self.down2_3 = ConvBNAct(nc, nc, k=3, s=2, g=nc)
        self.bu3 = nn.Sequential(ConvBNAct(nc * 2, nc, k=1, s=1), DWConvBlock(nc, k=5))
        self.down3_4 = ConvBNAct(nc, nc, k=3, s=2, g=nc)
        self.bu4 = nn.Sequential(ConvBNAct(nc * 2, nc, k=1, s=1), DWConvBlock(nc, k=5))

        self.heads = nn.ModuleList(
            [PlateTemporalScaleHead(nc, cfg.head_ch) for _ in cfg.strides]
        )

        # Dedicated stride-8 real+temporal ROI feature neck for whole-vehicle heavy-vehicle attributes
        self.veh_roi_neck = nn.Sequential(
            ConvBNAct(df_c3_ch + df_pan_ch + nc, cfg.veh_roi_ch, k=1, s=1),
            DWConvBlock(cfg.veh_roi_ch, k=3),
        )
        q_hidden = 64 if cfg.preset == "nano" else 96
        q_sp_ch = 16 if cfg.preset == "nano" else 24
        self.veh_query_head = VehicleQueryHead(
            roi_feat_ch=cfg.veh_roi_ch + nc,
            dfine_d_model=int(df_cfg.d_model),
            hidden_dim=q_hidden,
            spatial_ch=q_sp_ch,
            num_subtypes=cfg.num_subtypes,
            use_v5_heads=cfg.use_v5_heads,
        )
        self.cutout_head = EdgeGuidedVehicleCutoutHead(
            feat_ch=nc,
            aux_ch=3,
            hidden_ch=cfg.cutout_ch,
            query_dim=q_hidden,
        )

    def train(self, mode: bool = True) -> TemporalPlateDetNet:
        super().train(mode)
        # Keep Apache-2.0 D-FINE core permanently frozen & in deterministic eval mode!
        self.dfine_core.eval()
        for p in self.dfine_core.parameters():
            p.requires_grad_(False)
        return self

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_hidden_state(
        self,
        batch_size: int,
        input_hw: tuple[int, int],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nc = self.config.neck_ch
        h, w = int(input_hw[0]), int(input_hw[1])
        h_p3 = torch.zeros((batch_size, nc, h // 8, w // 8), device=device, dtype=dtype)
        h_p4 = torch.zeros((batch_size, nc, h // 16, w // 16), device=device, dtype=dtype)
        return h_p3, h_p4

    def forward_backbone_neck(
        self,
        images: torch.Tensor,
        h_prev: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.training:
            with torch.no_grad():
                dfine_out = self.dfine_core(images)
        else:
            dfine_out = self.dfine_core(images)

        df_c2: torch.Tensor = dfine_out["c2"]  # type: ignore[assignment]
        df_c3: torch.Tensor = dfine_out["c3"]  # type: ignore[assignment]
        pan_feats: list[torch.Tensor] = dfine_out["pan_feats"]  # type: ignore[assignment]
        if self.config.dfine_variant == "nano":
            df_s16 = pan_feats[0]
            df_pan_s8 = F.interpolate(df_s16, size=df_c3.shape[-2:], mode="bilinear", align_corners=False)
        else:
            df_pan_s8 = pan_feats[0]
            df_s16 = pan_feats[1]

        c1, aux_edge = self.photo_stem(images)
        c2 = self.stage2(c1)  # H/4
        c3 = self.stage3(c2)  # H/8
        c4 = self.stage4(c3)  # H/16

        p4_td = self.p4_ctx(self.lat4(torch.cat([c4, df_s16.to(dtype=c4.dtype)], dim=1)))
        l3 = self.lat3(torch.cat([c3, df_c3.to(dtype=c3.dtype)], dim=1))
        p3_td = self.td3(torch.cat([l3, F.interpolate(p4_td, size=c3.shape[-2:], mode="nearest")], dim=1))

        hp3_in = h_prev[0] if h_prev is not None else None
        hp4_in = h_prev[1] if h_prev is not None else None
        p3_temporal, hp3_next = self.gru_p3(p3_td, hp3_in)
        p4_mem, hp4_next = self.gru_p4(p4_td, hp4_in)

        c1_lat = self.lat2(torch.cat([c2, df_c2.to(dtype=c2.dtype)], dim=1))
        if (
            torch.jit.is_tracing()
            or torch.onnx.is_in_onnx_export()
            or c1_lat.shape[-2:] == (p3_temporal.shape[-2] * 2, p3_temporal.shape[-1] * 2)
        ):
            p3_up = F.interpolate(p3_temporal, scale_factor=2.0, mode="nearest")
        else:
            p3_up = F.interpolate(p3_temporal, size=c1_lat.shape[-2:], mode="nearest")
        p2 = self.p2_out(c1_lat + p3_up * torch.sigmoid(self.p2_temp_gate(p3_up)))

        n2 = p2
        n3 = self.bu3(torch.cat([p3_temporal, self.down2_3(n2)], dim=1))
        n4 = self.bu4(torch.cat([p4_mem, self.down3_4(n3)], dim=1))

        veh_feat_s8 = self.veh_roi_neck(
            torch.cat([df_c3.to(dtype=n3.dtype), df_pan_s8.to(dtype=n3.dtype), n3], dim=1)
        )
        return [n2, n3, n4], (hp3_next, hp4_next), aux_edge, veh_feat_s8, dfine_out

    def forward_step(
        self,
        images: torch.Tensor,
        h_prev: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Single-step forward returning unified `[pyramid_anchors + 300_vehicle_queries]` predictions."""
        feats, h_next, aux_edge, veh_feat_s8, dfine_out = self.forward_backbone_neck(images, h_prev=h_prev)
        bsz = images.shape[0]
        img_h, img_w = int(images.shape[-2]), int(images.shape[-1])
        device = images.device

        cls_all: list[torch.Tensor] = []
        box_all: list[torch.Tensor] = []
        quad_all: list[torch.Tensor] = []
        sub_all: list[torch.Tensor] = []
        cab_val_all: list[torch.Tensor] = []
        cab_box_all: list[torch.Tensor] = []
        fp_val_all: list[torch.Tensor] = []
        fp_xy_all: list[torch.Tensor] = []
        pts_all: list[torch.Tensor] = []
        str_all: list[torch.Tensor] = []

        num_cls = int(self.config.num_classes)
        num_sub = int(self.config.num_subtypes) + 1

        for stride, head, feat in zip(self.config.strides, self.heads, feats):
            cls_2ch, ltrb_l, quad_l = head(feat)
            hl = int(cls_2ch.shape[2])
            wl = int(cls_2ch.shape[3])
            n_l = hl * wl
            st_f = float(stride)

            cx_1d = ((torch.arange(wl, device=device, dtype=torch.float32) + 0.5) * st_f).view(1, 1, wl)
            cy_1d = ((torch.arange(hl, device=device, dtype=torch.float32) + 0.5) * st_f).view(1, hl, 1)

            c2_flat = cls_2ch.float().permute(0, 2, 3, 1).reshape(bsz, n_l, 2)
            neg_veh = torch.full((bsz, n_l, 1), -20.0, device=device, dtype=torch.float32)
            if num_cls == 3:
                cls_flat = torch.cat([c2_flat[:, :, 0:1], neg_veh, c2_flat[:, :, 1:2]], dim=-1)
            else:
                extra_neg = torch.full((bsz, n_l, max(0, num_cls - 2)), -20.0, device=device, dtype=torch.float32)
                cls_flat = torch.cat([c2_flat[:, :, 0:1], neg_veh, extra_neg], dim=-1)[:, :, :num_cls]

            ltrb_2d = ltrb_l.float().permute(0, 2, 3, 1) * st_f
            quad_2d = quad_l.float().permute(0, 2, 3, 1).reshape(bsz, hl, wl, 4, 2) * st_f

            boxes = torch.stack(
                [
                    cx_1d - ltrb_2d[:, :, :, 0],
                    cy_1d - ltrb_2d[:, :, :, 1],
                    cx_1d + ltrb_2d[:, :, :, 2],
                    cy_1d + ltrb_2d[:, :, :, 3],
                ],
                dim=-1,
            ).reshape(bsz, n_l, 4)
            corners = torch.stack(
                [
                    cx_1d.unsqueeze(-1) + quad_2d[:, :, :, :, 0],
                    cy_1d.unsqueeze(-1) + quad_2d[:, :, :, :, 1],
                ],
                dim=-1,
            ).reshape(bsz, n_l, 4, 2)

            # Runtime-dependent zero grid avoids constant-folding ~332 KiB of FP32 anchors into ONNX.
            zero_hw = cls_2ch[0, 0].float().detach() * 0.0
            gx = zero_hw + cx_1d[0]
            gy = zero_hw + cy_1d[0]
            pts = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
            st = zero_hw.reshape(-1) + st_f
            pts_all.append(pts)
            str_all.append(st)

            cls_all.append(cls_flat)
            box_all.append(boxes)
            quad_all.append(corners)
            sub_all.append(torch.zeros((bsz, n_l, num_sub), device=device, dtype=torch.float32))
            cab_val_all.append(torch.full((bsz, n_l, 1), -20.0, device=device, dtype=torch.float32))
            cab_box_all.append(boxes)
            fp_val_all.append(torch.full((bsz, n_l, 1), -20.0, device=device, dtype=torch.float32))
            fp_xy_all.append(torch.zeros((bsz, n_l, 4, 2), device=device, dtype=torch.float32))

        # Whole-vehicle query predictions from D-FINE 300 queries + 5x5 ROI head
        q_out = self.veh_query_head(
            torch.cat([veh_feat_s8, F.interpolate(feats[0], size=veh_feat_s8.shape[-2:], mode="bilinear", align_corners=False)], dim=1),
            dfine_out["logits"],        # type: ignore[arg-type]
            dfine_out["pred_boxes"],    # type: ignore[arg-type]
            dfine_out["query_embeds"],  # type: ignore[arg-type]
            (img_h, img_w),
        )
        num_q = int(q_out["veh_logit_q"].shape[1])
        neg_q1 = torch.full((bsz, num_q, 1), -20.0, device=device, dtype=torch.float32)
        if num_cls == 3:
            q_cls_3 = torch.cat([neg_q1, q_out["veh_logit_q"].unsqueeze(-1), neg_q1], dim=-1)
        else:
            extra_q = torch.full((bsz, num_q, max(0, num_cls - 2)), -20.0, device=device, dtype=torch.float32)
            q_cls_3 = torch.cat([neg_q1, q_out["veh_logit_q"].unsqueeze(-1), extra_q], dim=-1)[:, :, :num_cls]

        cls_all.append(q_cls_3)
        box_all.append(q_out["pred_q_boxes"])
        quad_all.append(q_out["pred_q_corners"])
        sub_all.append(q_out["pred_q_subtypes"])
        cab_val_all.append(q_out["pred_q_cabin_valid_logit"])
        cab_box_all.append(q_out["pred_q_cabin_boxes"])
        fp_val_all.append(q_out["pred_q_footprint_valid_logit"])
        fp_xy_all.append(q_out["pred_q_footprints"])

        q_cx = 0.5 * (q_out["raw_q_boxes_px"][0, :, 0] + q_out["raw_q_boxes_px"][0, :, 2])
        q_cy = 0.5 * (q_out["raw_q_boxes_px"][0, :, 1] + q_out["raw_q_boxes_px"][0, :, 3])
        pts_all.append(torch.stack([q_cx, q_cy], dim=-1).detach())
        str_all.append(q_cx.detach() * 0.0 - 1.0)

        out_d: dict[str, Any] = {
            "cls_logits": torch.cat(cls_all, dim=1),
            "pred_boxes": torch.cat(box_all, dim=1),
            "pred_corners": torch.cat(quad_all, dim=1),
            "pred_subtype_logits": torch.cat(sub_all, dim=1),
            "pred_cabin_valid_logit": torch.cat(cab_val_all, dim=1),
            "pred_cabin_boxes": torch.cat(cab_box_all, dim=1),
            "pred_footprint_valid_logit": torch.cat(fp_val_all, dim=1),
            "pred_footprints": torch.cat(fp_xy_all, dim=1),
            "h_next": h_next,
            "p2_feat": feats[0],
            "aux_edge_3ch": aux_edge,
        }
        out_d["anchor_points"] = torch.cat(pts_all, dim=0)
        out_d["anchor_strides"] = torch.cat(str_all, dim=0)
        out_d["num_vehicle_queries"] = num_q
        out_d["synth_veh_logits"] = q_out["synth_adapter_logit"]
        out_d["raw_q_boxes_px"] = q_out["raw_q_boxes_px"]
        for key in ("veh_log_odds", "bbox_cov_4x4", "appearance_emb", "query_condition",
                    "subtype_probs", "subtype_unknown_prob"):
            out_d[key] = q_out[key]
        if "pred_q_mount_boxes" in q_out:
            out_d["pred_q_mount_boxes"] = q_out["pred_q_mount_boxes"]
            out_d["pred_q_mount_vis_logit"] = q_out["pred_q_mount_vis_logit"]
        if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
            out_d["p3_feat"] = feats[1]
            out_d["p4_feat"] = feats[2]
        return out_d

    def predict_vehicle_cutouts(
        self,
        p2_feat: torch.Tensor,
        aux_edge_3ch: torch.Tensor,
        veh_rois: torch.Tensor,
        query_condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.cutout_head(p2_feat, aux_edge_3ch, veh_rois, query_condition)

    def forward_clip(
        self,
        clips_bkhw: torch.Tensor,
        *,
        reset_prob: float = 0.25,
        force_k1: bool = False,
        generator: torch.Generator | None = None,
    ) -> list[dict[str, Any]]:
        """Unroll `(B, K, 3, H, W)` video clip sequentially with `reset_prob` (`25%`) hidden-state dropout."""
        bsz, num_frames = clips_bkhw.shape[:2]
        h_state: tuple[torch.Tensor, torch.Tensor] | None = None
        outputs: list[dict[str, Any]] = []
        for k in range(num_frames):
            if force_k1 or k == 0:
                h_in = None
            elif h_state is not None and reset_prob > 0.0:
                if generator is not None:
                    keep = (torch.rand((bsz, 1, 1, 1), generator=generator) >= reset_prob).to(
                        device=clips_bkhw.device, dtype=h_state[0].dtype
                    )
                else:
                    keep = (torch.rand((bsz, 1, 1, 1), device=clips_bkhw.device) >= reset_prob).to(
                        dtype=h_state[0].dtype
                    )
                h_in = (h_state[0] * keep, h_state[1] * keep)
            else:
                h_in = h_state

            step_out = self.forward_step(clips_bkhw[:, k], h_prev=h_in)
            h_state = step_out["h_next"]
            outputs.append(step_out)
        return outputs

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stateless `K=1` 5-tuple forward compatible with `PlateDetNet` signature."""
        out = self.forward_step(images, h_prev=None)
        return (
            out["cls_logits"],
            out["pred_boxes"],
            out["pred_corners"],
            out["anchor_points"],
            out["anchor_strides"],
        )


def _match_query_nll(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular Hungarian assignment without an optional SciPy dependency."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("Query matching requires a finite cost matrix")
    transpose = cost.shape[0] > cost.shape[1]
    if transpose:
        cost = cost.T
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=np.int64), np.zeros(m + 1, dtype=np.int64)
    for row in range(1, n + 1):
        p[0] = row
        col = 0
        minimum = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[col] = True
            current = p[col]
            candidates = np.flatnonzero(~used[1:]) + 1
            reduced = cost[current - 1, candidates - 1] - u[current] - v[candidates]
            improved = reduced < minimum[candidates]
            minimum[candidates[improved]] = reduced[improved]
            way[candidates[improved]] = col
            next_col = candidates[np.argmin(minimum[candidates])]
            delta = minimum[next_col]
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            col = next_col
            if p[col] == 0:
                break
        while col:
            previous = way[col]
            p[col] = p[previous]
            col = previous
    columns = np.flatnonzero(p[1:])
    rows = p[columns + 1] - 1
    return (columns, rows) if transpose else (rows, columns)


class TemporalTaskAlignedDetLoss(nn.Module):
    """Gen-4 Hybrid Loss: Pyramid TAL (`plate` + `hardneg`) + 1-to-1 `D-FINE` Query Vehicle/Attribute + `64x64` Cutout Loss."""

    def __init__(
        self,
        topk: int = 9,
        alpha: float = 1.0,
        beta: float = 6.0,
        cls_weight: float = 1.30,
        box_weight: float = 2.50,
        quad_weight: float = 1.30,
        subtype_weight: float = 0.70,
        cabin_weight: float = 0.50,
        footprint_weight: float = 0.50,
        cutout_weight: float = 1.55,
        max_cutout_rois_per_step: int = 32,
        teacher_kl_weight: float = 0.0,
        appearance_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.teacher_kl_weight = teacher_kl_weight
        self.appearance_weight = appearance_weight
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.quad_weight = quad_weight
        self.subtype_weight = subtype_weight
        self.cabin_weight = cabin_weight
        self.footprint_weight = footprint_weight
        self.cutout_weight = cutout_weight
        self.max_cutout_rois_per_step = max_cutout_rois_per_step

        sobel_x = torch.tensor(
            [[-0.25, 0.0, 0.25], [-0.50, 0.0, 0.50], [-0.25, 0.0, 0.25]],
            dtype=torch.float32,
        )
        sobel_k = torch.stack([sobel_x, sobel_x.t().contiguous()], dim=0).unsqueeze(1)
        self.register_buffer("sobel_k", sobel_k, persistent=False)
        self.register_buffer(
            "subtype_ema_freq",
            torch.full((NUM_VEHICLE_SUBTYPES,), 1.0 / float(NUM_VEHICLE_SUBTYPES), dtype=torch.float32),
            persistent=False,
        )

    def _mask_boundary_l1(self, pred_prob: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        sk = self.sobel_k.to(dtype=torch.float32, device=pred_prob.device)
        gp = F.conv2d(_replicate_pad_1px(pred_prob.float()), sk)
        gt = F.conv2d(_replicate_pad_1px(target_mask.float()), sk)
        ep = torch.sqrt((gp * gp).sum(dim=1, keepdim=True) + 1e-6)
        et = torch.sqrt((gt * gt).sum(dim=1, keepdim=True) + 1e-6)
        return F.l1_loss(ep, et, reduction="mean")

    def forward(
        self,
        step_out: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        *,
        cutout_head: EdgeGuidedVehicleCutoutHead | None = None,
    ) -> dict[str, torch.Tensor]:
        cls_logits: torch.Tensor = step_out["cls_logits"]
        pred_boxes: torch.Tensor = step_out["pred_boxes"]
        pred_corners: torch.Tensor = step_out["pred_corners"]
        pred_subtypes: torch.Tensor = step_out["pred_subtype_logits"]
        pred_cab_val: torch.Tensor = step_out["pred_cabin_valid_logit"].squeeze(-1)
        pred_cab_box: torch.Tensor = step_out["pred_cabin_boxes"]
        pred_fp_val: torch.Tensor = step_out["pred_footprint_valid_logit"].squeeze(-1)
        pred_fp_xy: torch.Tensor = step_out["pred_footprints"]
        anchor_points: torch.Tensor = step_out["anchor_points"]
        anchor_strides: torch.Tensor = step_out["anchor_strides"]

        bsz, num_total, num_cls = cls_logits.shape
        num_q = int(step_out.get("num_vehicle_queries", 0))
        a_pyr = num_total - num_q
        device = cls_logits.device

        # Pyramid buffers (CLS_PLATE=0, CLS_SIGNBOARD_HARDNEG=2)
        pyr_pos_2ch = torch.zeros((bsz, a_pyr, 2), device=device, dtype=torch.float32)
        pyr_qual_2ch = torch.zeros_like(pyr_pos_2ch)
        pyr_ignore_2ch = torch.zeros((bsz, a_pyr, 2), device=device, dtype=torch.bool)
        pyr_hardneg_boost = torch.ones((bsz, a_pyr), device=device, dtype=torch.float32)
        pyr_pos_boost = torch.ones((bsz, a_pyr), device=device, dtype=torch.float32)
        pyr_fg_mask = torch.zeros((bsz, a_pyr), device=device, dtype=torch.bool)
        pyr_gt_cls = torch.full((bsz, a_pyr), -1, device=device, dtype=torch.long)
        pyr_gt_boxes = torch.zeros((bsz, a_pyr, 4), device=device, dtype=torch.float32)
        pyr_gt_corners = torch.zeros((bsz, a_pyr, 4, 2), device=device, dtype=torch.float32)
        pyr_weights = torch.zeros((bsz, a_pyr), device=device, dtype=torch.float32)

        # Query buffers (1-to-1 whole-vehicle matching on 300 D-FINE queries)
        q_target_score = torch.zeros((bsz, max(1, num_q)), device=device, dtype=torch.float32)
        q_fg_mask = torch.zeros((bsz, max(1, num_q)), device=device, dtype=torch.bool)
        q_cls_weight = torch.ones((bsz, max(1, num_q)), device=device, dtype=torch.float32)
        q_gt_boxes = torch.zeros((bsz, max(1, num_q), 4), device=device, dtype=torch.float32)
        q_gt_subtypes = torch.zeros((bsz, max(1, num_q)), device=device, dtype=torch.long)
        q_gt_cab_val = torch.zeros((bsz, max(1, num_q)), device=device, dtype=torch.float32)
        q_gt_cab_box = torch.zeros((bsz, max(1, num_q), 4), device=device, dtype=torch.float32)
        q_gt_fp_val = torch.zeros((bsz, max(1, num_q)), device=device, dtype=torch.float32)
        q_gt_fp_xy = torch.zeros((bsz, max(1, num_q), 4, 2), device=device, dtype=torch.float32)
        q_gt_mount_val = torch.zeros((bsz, max(1, num_q)), device=device, dtype=torch.float32)
        q_gt_mount_box = torch.zeros((bsz, max(1, num_q), 4), device=device, dtype=torch.float32)
        q_weights = torch.zeros((bsz, max(1, num_q)), device=device, dtype=torch.float32)

        cutout_roi_list: list[torch.Tensor] = []
        cutout_gt_list: list[torch.Tensor] = []
        cutout_query_indices: list[tuple[int, int]] = []
        q_identity = torch.full((bsz, max(1, num_q)), -1, device=device, dtype=torch.long)

        synth_veh_logits: torch.Tensor | None = step_out.get("veh_log_odds")
        raw_q_boxes_px: torch.Tensor | None = step_out.get("raw_q_boxes_px")
        pred_q_mount_boxes: torch.Tensor | None = step_out.get("pred_q_mount_boxes")
        pred_q_mount_vis_logit: torch.Tensor | None = step_out.get("pred_q_mount_vis_logit")

        with torch.no_grad():
            probs_pyr = torch.sigmoid(cls_logits[:, :a_pyr].detach())
            ax = anchor_points[:a_pyr, 0]
            ay = anchor_points[:a_pyr, 1]
            st_pyr = anchor_strides[:a_pyr]
            synth_q_probs = (
                torch.sigmoid(synth_veh_logits.detach())
                if synth_veh_logits is not None
                else torch.zeros((bsz, max(1, num_q)), device=device, dtype=torch.float32)
            )

            for b in range(bsz):
                ig_boxes = targets[b].get("ignore_boxes")
                if ig_boxes is not None and ig_boxes.numel() > 0 and a_pyr > 0:
                    ig_b = ig_boxes.to(device, dtype=torch.float32).reshape(-1, 4)
                    ix0, iy0, ix1, iy1 = ig_b.unbind(-1)
                    in_ig = (
                        (ax[None, :] >= ix0[:, None])
                        & (ax[None, :] <= ix1[:, None])
                        & (ay[None, :] >= iy0[:, None])
                        & (ay[None, :] <= iy1[:, None])
                    )
                    ig_cls = targets[b].get("ignore_classes")
                    if ig_cls is not None and ig_cls.numel() == ig_b.shape[0]:
                        ig_c = ig_cls.to(device, dtype=torch.long).reshape(-1)
                        any_all = ig_c < 0
                        if any_all.any():
                            pyr_ignore_2ch[b] |= in_ig[any_all].any(dim=0).unsqueeze(-1).expand(a_pyr, 2)
                        if (ig_c == CLS_PLATE).any():
                            pyr_ignore_2ch[b, :, 0] |= in_ig[ig_c == CLS_PLATE].any(dim=0)
                        if (ig_c == CLS_SIGNBOARD_HARDNEG).any():
                            pyr_ignore_2ch[b, :, 1] |= in_ig[ig_c == CLS_SIGNBOARD_HARDNEG].any(dim=0)
                    else:
                        pyr_ignore_2ch[b] = in_ig.any(dim=0).unsqueeze(-1).expand(a_pyr, 2)

                gt_labels = targets[b]["labels"].to(device, dtype=torch.long)
                gt_boxes = targets[b]["boxes"].to(device, dtype=torch.float32)
                gt_corners = targets[b]["corners"].to(device, dtype=torch.float32)
                num_raw = gt_labels.shape[0]
                gt_sub = (
                    targets[b]["vehicle_subtypes"].to(device, dtype=torch.long)
                    if "vehicle_subtypes" in targets[b]
                    else torch.full((num_raw,), pred_subtypes.shape[-1] - 1, device=device, dtype=torch.long)
                )
                gt_cab_v = (
                    targets[b]["cabin_valid"].to(device, dtype=torch.float32)
                    if "cabin_valid" in targets[b]
                    else torch.zeros((num_raw,), device=device, dtype=torch.float32)
                )
                gt_cab_b = (
                    targets[b]["cabin_boxes"].to(device, dtype=torch.float32)
                    if "cabin_boxes" in targets[b]
                    else gt_boxes.clone()
                )
                gt_fp_v = (
                    targets[b]["footprint_valid"].to(device, dtype=torch.float32)
                    if "footprint_valid" in targets[b]
                    else torch.zeros((num_raw,), device=device, dtype=torch.float32)
                )
                gt_fp_q = (
                    targets[b]["footprints"].to(device, dtype=torch.float32)
                    if "footprints" in targets[b]
                    else gt_corners.clone()
                )
                gt_masks = targets[b].get("visible_masks", targets[b].get("cutout_masks"))
                gt_amodal = targets[b].get("amodal_boxes", gt_boxes).to(device, dtype=torch.float32)
                gt_ids = targets[b].get("identity_ids", torch.full((num_raw,), -1)).to(device, dtype=torch.long)
                gt_box_vis = (
                    targets[b]["box_visibility"].to(device, dtype=torch.float32).reshape(-1)
                    if "box_visibility" in targets[b] and targets[b]["box_visibility"].numel() == num_raw
                    else torch.ones((num_raw,), device=device, dtype=torch.float32)
                )
                gt_shadow = (
                    targets[b]["overhang_shadow"].to(device, dtype=torch.float32).reshape(-1)
                    if "overhang_shadow" in targets[b] and targets[b]["overhang_shadow"].numel() == num_raw
                    else torch.zeros((num_raw,), device=device, dtype=torch.float32)
                )

                valid_cls = (gt_labels >= 0) & (gt_labels < num_cls)
                if not valid_cls.all():
                    gt_labels = gt_labels[valid_cls]
                    gt_boxes = gt_boxes[valid_cls]
                    gt_corners = gt_corners[valid_cls]
                    gt_sub = gt_sub[valid_cls]
                    gt_amodal = gt_amodal[valid_cls]
                    gt_ids = gt_ids[valid_cls]
                    gt_cab_v = gt_cab_v[valid_cls]
                    gt_cab_b = gt_cab_b[valid_cls]
                    gt_fp_v = gt_fp_v[valid_cls]
                    gt_fp_q = gt_fp_q[valid_cls]
                    gt_box_vis = gt_box_vis[valid_cls]
                    gt_shadow = gt_shadow[valid_cls]
                    if gt_masks is not None and gt_masks.shape[0] == num_raw:
                        gt_masks = gt_masks[valid_cls.to(gt_masks.device)]

                # A. Vehicle GT -> Cutout ROIs + Strict 1-to-1 D-FINE Query Matching
                veh_all_idx = torch.nonzero(gt_labels == CLS_VEHICLE, as_tuple=False).squeeze(1)
                veh_ig_idx = (
                    veh_all_idx[gt_box_vis.index_select(0, veh_all_idx) < 0.15]
                    if veh_all_idx.numel() > 0
                    else veh_all_idx
                )
                veh_idx = (
                    veh_all_idx[gt_box_vis.index_select(0, veh_all_idx) >= 0.15]
                    if veh_all_idx.numel() > 0
                    else veh_all_idx
                )
                if veh_idx.numel() > 0:
                    vb = gt_boxes.index_select(0, veh_idx)
                    vw_m = vb[:, 2] - vb[:, 0]
                    vh_m = vb[:, 3] - vb[:, 1]
                    keep_m = (vw_m >= 8.0) & (vh_m >= 8.0)
                    if num_q > 0 and raw_q_boxes_px is not None:
                        q_pol_b = pred_boxes[b, a_pyr:].detach()
                        # Matched joint likelihood: class + full amodal box Gaussian.
                        cov = step_out["bbox_cov_4x4"][b].detach().double()
                        cov_match = cov + torch.eye(4, device=cov.device, dtype=torch.float64) * (24.0 ** 2)
                        chol = torch.linalg.cholesky(cov_match)
                        amodal_vb = gt_amodal.index_select(0, veh_idx)
                        error = (amodal_vb[:, None, :] - q_pol_b[None, :, :]).double()
                        whitened = torch.linalg.solve_triangular(chol[None], error[..., None], upper=False).squeeze(-1)
                        box_nll = (0.5 * whitened.square().sum(-1) + chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)[None]).float()
                        cost = box_nll - synth_veh_logits[b].detach()[None] + 4.0 * (1.0 - pairwise_box_iou(amodal_vb, q_pol_b))
                        n_vg = int(vb.shape[0])
                        pl_only_idx = torch.nonzero(gt_labels == CLS_PLATE, as_tuple=False).squeeze(1)
                        pl_boxes_b = gt_boxes.index_select(0, pl_only_idx) if pl_only_idx.numel() > 0 else None
                        veh_mount_map: dict[int, torch.Tensor] = {}
                        if pl_boxes_b is not None and pl_boxes_b.shape[0] > 0:
                            for p_k in range(int(pl_boxes_b.shape[0])):
                                pb_k = pl_boxes_b[p_k]
                                pcx_k = float(0.5 * (pb_k[0] + pb_k[2]).item())
                                pcy_k = float(0.5 * (pb_k[1] + pb_k[3]).item())
                                p_area_k = float(((pb_k[2] - pb_k[0]) * (pb_k[3] - pb_k[1])).item())
                                best_gi = -1
                                best_va = float("inf")
                                for gi_c in range(n_vg):
                                    vx0_c, vy0_c, vx1_c, vy1_c = [float(x) for x in vb[gi_c]]
                                    vw_c = max(1.0, vx1_c - vx0_c)
                                    vh_c = max(1.0, vy1_c - vy0_c)
                                    if (
                                        (vx0_c - 0.08 * vw_c - 4.0) <= pcx_k <= (vx1_c + 0.08 * vw_c + 4.0)
                                        and (vy0_c - 0.02 * vh_c) <= pcy_k <= (vy1_c + 0.18 * vh_c + 10.0)
                                    ):
                                        va_c = vw_c * vh_c
                                        if va_c < best_va:
                                            best_va = va_c
                                            best_gi = gi_c
                                if best_gi >= 0:
                                    prev_pb = veh_mount_map.get(best_gi)
                                    if prev_pb is None or p_area_k > float(((prev_pb[2] - prev_pb[0]) * (prev_pb[3] - prev_pb[1])).item()):
                                        veh_mount_map[best_gi] = pb_k

                        match_gt, match_q = _match_query_nll(cost.cpu().numpy())
                        for gi, qj in zip(match_gt.tolist(), match_q.tolist()):
                            orig_g = veh_idx[gi]
                            q_fg_mask[b, qj] = True
                            q_target_score[b, qj] = 1.0
                            q_gt_boxes[b, qj] = amodal_vb[gi]
                            q_identity[b, qj] = gt_ids[orig_g]
                            q_gt_subtypes[b, qj] = torch.where(
                                (gt_sub[orig_g] >= 0) & (gt_sub[orig_g] < pred_subtypes.shape[-1] - 1),
                                gt_sub[orig_g], gt_sub.new_tensor(pred_subtypes.shape[-1] - 1),
                            )
                            q_gt_cab_val[b, qj] = gt_cab_v[orig_g]
                            q_gt_cab_box[b, qj] = gt_cab_b[orig_g]
                            q_gt_fp_val[b, qj] = gt_fp_v[orig_g]
                            q_gt_fp_xy[b, qj] = gt_fp_q[orig_g]
                            if gi in veh_mount_map:
                                q_gt_mount_val[b, qj] = 1.0
                                q_gt_mount_box[b, qj] = veh_mount_map[gi]
                            q_weights[b, qj] = 1.0
                            if cutout_head is not None and gt_masks is not None and bool(keep_m[gi]):
                                # Visible mask labels stay in their annotated visible ROI,
                                # even though localization targets are amodal.
                                b_col = vb.new_tensor([[float(b)]])
                                cutout_roi_list.append(torch.cat([b_col, vb[gi:gi+1]], dim=1))
                                mask_index = orig_g if gt_masks.shape[0] == gt_labels.shape[0] else gi
                                cutout_gt_list.append(gt_masks[mask_index].to(device).reshape(1, 1, 64, 64))
                                cutout_query_indices.append((b, qj))

                        # 2.5x negative weight on unmatched interior sub-box queries (e.g. bus rear windows inside bus)
                        q_area = (q_pol_b[:, 2] - q_pol_b[:, 0]).clamp_min(0.0) * (q_pol_b[:, 3] - q_pol_b[:, 1]).clamp_min(0.0)
                        for ref_v in (amodal_vb, vb):
                            v_area = (ref_v[:, 2] - ref_v[:, 0]).clamp_min(0.0) * (ref_v[:, 3] - ref_v[:, 1]).clamp_min(0.0)
                            iou_qv = pairwise_box_iou(q_pol_b, ref_v)
                            iomin_qv = pairwise_box_iomin(q_pol_b, ref_v)
                            is_sub = (iomin_qv >= 0.80) & (iou_qv < 0.45) & (q_area[:, None] <= v_area[None, :])
                            sub_q_mask = (~q_fg_mask[b]) & is_sub.any(dim=1)
                            if sub_q_mask.any():
                                q_cls_weight[b] = torch.where(
                                    sub_q_mask, torch.full_like(q_cls_weight[b], 2.5), q_cls_weight[b]
                                )

                # Mask severely occluded sliver vehicles (box_visibility < 0.15) as ignore on queries with IoU >= 0.5
                if num_q > 0 and veh_ig_idx.numel() > 0:
                    q_pol_ig = pred_boxes[b, a_pyr:].detach()
                    vb_ig = gt_boxes.index_select(0, veh_ig_idx)
                    amodal_ig = gt_amodal.index_select(0, veh_ig_idx)
                    ig_hit = (pairwise_box_iou(q_pol_ig, vb_ig).amax(dim=1) >= 0.50) | (
                        pairwise_box_iou(q_pol_ig, amodal_ig).amax(dim=1) >= 0.50
                    )
                    ig_q_mask = (~q_fg_mask[b]) & ig_hit
                    if ig_q_mask.any():
                        q_cls_weight[b] = torch.where(
                            ig_q_mask, torch.zeros_like(q_cls_weight[b]), q_cls_weight[b]
                        )

                # B. Plate & Signboard Hard-Negative GT -> Dense Pyramid TAL (strides 4, 8, 16)
                ps_idx = torch.nonzero(
                    (gt_labels == CLS_PLATE) | (gt_labels == CLS_SIGNBOARD_HARDNEG), as_tuple=False
                ).squeeze(1)
                if ps_idx.numel() == 0 or a_pyr == 0:
                    continue

                ps_labels = gt_labels.index_select(0, ps_idx)
                ps_boxes = gt_boxes.index_select(0, ps_idx)
                ps_corners = gt_corners.index_select(0, ps_idx)
                ps_shadow = gt_shadow.index_select(0, ps_idx)
                num_ps = int(ps_labels.shape[0])

                gx0, gy0, gx1, gy1 = ps_boxes.unbind(-1)
                gcx = 0.5 * (gx0 + gx1)
                gcy = 0.5 * (gy0 + gy1)
                in_box = (
                    (ax[None, :] >= gx0[:, None])
                    & (ax[None, :] <= gx1[:, None])
                    & (ay[None, :] >= gy0[:, None])
                    & (ay[None, :] <= gy1[:, None])
                )
                gt_max_dim = torch.maximum(gx1 - gx0, gy1 - gy0)[:, None]
                st_row = st_pyr[None, :]
                rad = torch.where(
                    (gt_max_dim < 44.0) | ((gt_max_dim >= 60.0) & (st_row >= 16.0)),
                    3.0 * st_row,
                    2.5 * st_row,
                )
                in_center = (torch.abs(ax[None, :] - gcx[:, None]) <= rad) & (
                    torch.abs(ay[None, :] - gcy[:, None]) <= rad
                )
                # Balanced negative focal suppression strictly inside hard-negative boxes (high-recall Stage-1; Stage-2 verifies).
                hn_rows = ps_labels == CLS_SIGNBOARD_HARDNEG
                in_any_hn = torch.zeros((a_pyr,), device=device, dtype=torch.bool)
                if hn_rows.any():
                    in_any_hn = in_box[hn_rows].any(dim=0)
                    pyr_hardneg_boost[b] = torch.where(
                        in_any_hn,
                        torch.full_like(pyr_hardneg_boost[b], 2.0),
                        pyr_hardneg_boost[b],
                    )

                is_pl = (ps_labels == CLS_PLATE)[:, None]
                # Cover tiny 8-26px plates on P2/P3, medium 26-76px across P2/P3/P4, and close-up 76-220px on P3/P4.
                pl_ok = torch.where(
                    gt_max_dim < 26.0,
                    st_row <= 8.0,
                    torch.where(
                        gt_max_dim < 76.0,
                        st_row <= 16.0,
                        torch.where(gt_max_dim < 128.0, st_row >= 8.0, st_row >= 16.0),
                    ),
                )
                sign_ok = torch.where(gt_max_dim < 56.0, st_row <= 8.0, st_row >= 8.0)
                stride_ok = torch.where(is_pl, pl_ok, sign_ok)

                cand_mask = (in_box | in_center) & stride_ok
                empty_gt = cand_mask.sum(dim=1) == 0
                if empty_gt.any():
                    cand_mask[empty_gt] = in_box[empty_gt] | in_center[empty_gt]

                ious = pairwise_box_iou(ps_boxes, pred_boxes[b, :a_pyr].detach())
                nwds = pairwise_box_nwd(ps_boxes, pred_boxes[b, :a_pyr].detach())
                sim_quality = torch.where(is_pl, 0.60 * ious + 0.40 * nwds, ious)
                gt_cls_probs = probs_pyr[b, :, ps_labels].transpose(0, 1)
                align_metric = (gt_cls_probs.clamp_min(1e-6) ** self.alpha) * (sim_quality.clamp_min(1e-6) ** self.beta)
                align_metric = torch.where(cand_mask, align_metric, torch.zeros_like(align_metric))

                k_sel = min(min(self.topk, 9), a_pyr)
                topk_vals, topk_idx = torch.topk(align_metric, k=k_sel, dim=1)
                pos_mask = torch.zeros_like(cand_mask)
                pos_mask.scatter_(1, topk_idx, topk_vals > 0.0)
                # Ensure close-up large plates (>= 56px) always supervise P4 (stride=16) so P3 density never starves P4.
                large_pl_p4 = (is_pl & (gt_max_dim >= 56.0)) & (st_row >= 16.0) & cand_mask
                if large_pl_p4.any():
                    p4_metric = torch.where(large_pl_p4, align_metric, torch.zeros_like(align_metric))
                    k_p4 = min(4, a_pyr)
                    p4_vals, p4_idx = torch.topk(p4_metric, k=k_p4, dim=1)
                    p4_pos = torch.zeros_like(cand_mask)
                    p4_pos.scatter_(1, p4_idx, p4_vals > 0.0)
                    pos_mask = pos_mask | (p4_pos & large_pl_p4)
                pos_mask = pos_mask & cand_mask

                anchor_pos_count = pos_mask.sum(dim=0)
                if (anchor_pos_count > 1).any():
                    masked_ious = torch.where(pos_mask, ious, torch.full_like(ious, -1.0))
                    best_gt = masked_ious.argmax(dim=0)
                    gt_idx_grid = torch.arange(num_ps, device=device)[:, None]
                    pos_mask = pos_mask & (gt_idx_grid == best_gt[None, :])

                fg_b = pos_mask.any(dim=0)
                if not fg_b.any():
                    continue

                assigned_gt = pos_mask.float().argmax(dim=0)
                pos_align = align_metric * pos_mask.float()
                max_align_per_gt = pos_align.amax(dim=1, keepdim=True).clamp_min(1e-6)

                fg_idx = torch.nonzero(fg_b, as_tuple=False).squeeze(1)
                gt_sel = assigned_gt[fg_idx]
                cls_sel = ps_labels[gt_sel]
                qual_sel = (pos_align / max_align_per_gt).amax(dim=0)[fg_idx].clamp(0.25, 1.0)
                col_2ch = torch.where(cls_sel == CLS_PLATE, torch.zeros_like(cls_sel), torch.ones_like(cls_sel))

                pyr_pos_2ch[b, fg_idx, col_2ch] = 1.0
                pyr_qual_2ch[b, fg_idx, col_2ch] = qual_sel
                ambig = cand_mask & (~pos_mask) & (ious >= 0.45)
                for channel, cls_id in ((0, CLS_PLATE), (1, CLS_SIGNBOARD_HARDNEG)):
                    rows = ps_labels == cls_id
                    if rows.any():
                        pyr_ignore_2ch[b, :, channel] |= ambig[rows].any(dim=0)
                if hn_rows.any():
                    # Never ignore plate negatives inside hard-negative boxes unless matched to an actual plate.
                    pyr_ignore_2ch[b, :, 0] &= ~in_any_hn
                pyr_fg_mask[b] = fg_b
                pyr_gt_cls[b, fg_idx] = cls_sel
                pyr_gt_boxes[b, fg_idx] = ps_boxes[gt_sel]
                pyr_gt_corners[b, fg_idx] = ps_corners[gt_sel]

                gt_w = (ps_boxes[gt_sel, 2] - ps_boxes[gt_sel, 0]).clamp_min(4.0)
                gt_h = (ps_boxes[gt_sel, 3] - ps_boxes[gt_sel, 1]).clamp_min(2.0)
                gt_ar = gt_w / gt_h
                is_small_or_shadow_pl = (cls_sel == CLS_PLATE) & (
                    (gt_w < 20.0) | (torch.sqrt(gt_w * gt_h) < 18.0) | (ps_shadow[gt_sel] > 0.60)
                )
                is_any_pl = cls_sel == CLS_PLATE
                pyr_pos_boost[b, fg_idx] = torch.where(
                    is_small_or_shadow_pl,
                    torch.full_like(qual_sel, 2.20),
                    torch.where(
                        is_any_pl,
                        torch.full_like(qual_sel, 2.00),
                        torch.ones_like(qual_sel),
                    ),
                )
                scale_boost = torch.where(
                    is_small_or_shadow_pl,
                    torch.full_like(qual_sel, 2.20),
                    torch.where(
                        is_any_pl,
                        torch.full_like(qual_sel, 2.00),
                        torch.ones_like(qual_sel),
                    ),
                )
                pyr_weights[b, fg_idx] = qual_sel * scale_boost

        # 1. Pyramid Classification Loss (`CLS_PLATE` high-recall + balanced hardneg margin)
        pyr_logits_2ch = torch.stack(
            [cls_logits[:, :a_pyr, CLS_PLATE], cls_logits[:, :a_pyr, CLS_SIGNBOARD_HARDNEG]],
            dim=-1,
        )
        pyr_prob_2ch = torch.sigmoid(pyr_logits_2ch)
        neg_w_2ch = 1.20 * (pyr_prob_2ch.detach() ** 2.0)
        neg_w_2ch[:, :, 0] = 0.90 * (pyr_prob_2ch[:, :, 0].detach() ** 2.0) * pyr_hardneg_boost
        pos_w_2ch = pyr_qual_2ch * pyr_pos_boost.unsqueeze(-1)
        focal_pyr = torch.where(pyr_pos_2ch > 0.0, pos_w_2ch, neg_w_2ch)
        if pyr_ignore_2ch.any():
            neg_ig = (pyr_pos_2ch <= 0.0) & pyr_ignore_2ch
            focal_pyr = torch.where(neg_ig, torch.zeros_like(focal_pyr), focal_pyr)
        bce_pyr = F.binary_cross_entropy_with_logits(pyr_logits_2ch, pyr_pos_2ch, reduction="none")
        loss_cls_pyr = (bce_pyr * focal_pyr).sum() / pyr_qual_2ch.sum().clamp_min(3.0)
        hn_pos_mask = pyr_pos_2ch[:, :, 1] > 0.0
        if hn_pos_mask.any():
            pyr_cls_f = cls_logits[:, :a_pyr].float()
            loss_hn_margin = 0.18 * F.softplus(
                pyr_cls_f[:, :, CLS_PLATE][hn_pos_mask]
                - pyr_cls_f[:, :, CLS_SIGNBOARD_HARDNEG][hn_pos_mask]
                + 0.75
            ).mean()
            loss_cls_pyr = loss_cls_pyr + loss_hn_margin

        # Frozen COCO vehicle logits are untouched by synthetic training; no vehicle cls/box gradient needed.
        loss_cls_q = cls_logits.sum() * 0.0
        loss_teacher_kl = cls_logits.sum() * 0.0
        loss_cls = loss_cls_pyr

        zero = pred_boxes.sum() * 0.0
        loss_box_pyr = zero
        loss_quad = pred_corners.sum() * 0.0
        if pyr_fg_mask.any():
            p_boxes_fg = pred_boxes[:, :a_pyr][pyr_fg_mask].float()
            g_boxes_fg = pyr_gt_boxes[pyr_fg_mask].float()
            w_fg = pyr_weights[pyr_fg_mask].float()
            cls_fg = pyr_gt_cls[pyr_fg_mask]
            is_pl_fg = cls_fg == CLS_PLATE
            _, ciou = compute_box_iou_ciou(p_boxes_fg, g_boxes_fg)
            nwd_fg = compute_box_nwd(p_boxes_fg, g_boxes_fg)
            per_box = torch.where(is_pl_fg, 0.70 * (1.0 - ciou) + 0.30 * (1.0 - nwd_fg), 1.0 - ciou)
            loss_box_pyr = (per_box * w_fg).sum() / w_fg.sum().clamp_min(1e-6)

            p_quad = pred_corners[:, :a_pyr][pyr_fg_mask]
            g_quad = pyr_gt_corners[pyr_fg_mask]
            diag = torch.hypot(
                (g_boxes_fg[:, 2] - g_boxes_fg[:, 0]).clamp_min(6.0),
                (g_boxes_fg[:, 3] - g_boxes_fg[:, 1]).clamp_min(4.0),
            ).view(-1, 1, 1)
            per_q = F.smooth_l1_loss(p_quad / diag, g_quad / diag, beta=0.08, reduction="none").mean(dim=(1, 2))
            q_w = torch.where(is_pl_fg, pyr_pos_boost[pyr_fg_mask], torch.full_like(per_q, 0.25))
            loss_quad = (per_q * q_w).sum() / q_w.sum().clamp_min(1e-6)

        loss_appearance = zero
        loss_box_q_reg = zero
        loss_box_nll = zero
        loss_subtype = pred_subtypes.sum() * 0.0
        loss_cabin = pred_cab_val.sum() * 0.0 + pred_cab_box.sum() * 0.0
        loss_footprint = pred_fp_val.sum() * 0.0 + pred_fp_xy.sum() * 0.0
        loss_mount = (
            (pred_q_mount_vis_logit.sum() * 0.0 + pred_q_mount_boxes.sum() * 0.0)
            if (pred_q_mount_vis_logit is not None and pred_q_mount_boxes is not None)
            else zero
        )

        if num_q > 0 and q_fg_mask.any():
            q_pred_boxes_fg = pred_boxes[:, a_pyr:][q_fg_mask].float()
            q_gt_boxes_fg = q_gt_boxes[q_fg_mask].float()
            vw_fg = q_weights[q_fg_mask].float()
            vw_sum = vw_fg.sum().clamp_min(1e-6)
            cov_fg = step_out["bbox_cov_4x4"][q_fg_mask].double()
            chol = torch.linalg.cholesky(cov_fg)
            error_det = (q_pred_boxes_fg.detach() - q_gt_boxes_fg).double().unsqueeze(-1)
            white = torch.linalg.solve_triangular(chol, error_det, upper=False).squeeze(-1)
            loss_box_nll = (0.5 * white.square().sum(-1)
                          + chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                          + 2.0 * math.log(2.0 * math.pi)).mean().float()
            emb = step_out["appearance_emb"][q_fg_mask].float()
            ids = q_identity[q_fg_mask]
            valid_ids = ids >= 0
            if valid_ids.sum() > 1:
                emb, ids = emb[valid_ids], ids[valid_ids]
                logits = (emb @ emb.T) / 0.1
                off_diag = ~torch.eye(len(ids), device=device, dtype=torch.bool)
                positives = (ids[:, None] == ids[None]) & off_diag
                has_positive = positives.any(dim=1)
                if has_positive.any():
                    log_probs = F.log_softmax(logits.masked_fill(~off_diag, -1e4), dim=-1)
                    per_anchor = -(log_probs * positives).sum(-1) / positives.sum(-1).clamp_min(1)
                    loss_appearance = per_anchor[has_positive].mean()

            sub_logits_fg = pred_subtypes[:, a_pyr:][q_fg_mask].float()
            sub_gt_fg = q_gt_subtypes[q_fg_mask]
            num_sub_c = int(sub_logits_fg.shape[-1])
            with torch.no_grad():
                if self.subtype_ema_freq.numel() != num_sub_c:
                    self.subtype_ema_freq = torch.full(
                        (num_sub_c,), 1.0 / float(max(1, num_sub_c)), device=device, dtype=torch.float32
                    )
                counts = torch.bincount(sub_gt_fg, minlength=num_sub_c).float()
                batch_freq = counts / counts.sum().clamp_min(1.0)
                self.subtype_ema_freq.mul_(0.98).add_(batch_freq.to(self.subtype_ema_freq.device), alpha=0.02)
                cls_w = (1.0 / (self.subtype_ema_freq.to(device) + 0.03)).sqrt()
                cls_w = cls_w / cls_w.mean().clamp_min(1e-6)

            ce_sub = F.cross_entropy(
                sub_logits_fg,
                sub_gt_fg,
                weight=cls_w,
                reduction="none",
                label_smoothing=0.05,
            )
            loss_subtype = (ce_sub * vw_fg).sum() / vw_sum

            cab_pred_v = pred_cab_val[:, a_pyr:][q_fg_mask].float()
            cab_pred_b = pred_cab_box[:, a_pyr:][q_fg_mask].float()
            cab_v_gt = q_gt_cab_val[q_fg_mask]
            loss_cab_v = F.binary_cross_entropy_with_logits(cab_pred_v, cab_v_gt, reduction="mean")
            has_cab = cab_v_gt > 0.5
            if has_cab.any():
                _, cab_ciou = compute_box_iou_ciou(cab_pred_b[has_cab], q_gt_cab_box[q_fg_mask][has_cab].float())
                loss_cabin = loss_cab_v + (1.0 - cab_ciou).mean()
            else:
                loss_cabin = loss_cab_v + pred_cab_box.sum() * 0.0

            fp_pred_v = pred_fp_val[:, a_pyr:][q_fg_mask].float()
            fp_pred_q = pred_fp_xy[:, a_pyr:][q_fg_mask].float()
            fp_v_gt = q_gt_fp_val[q_fg_mask]
            loss_fp_v = F.binary_cross_entropy_with_logits(fp_pred_v, fp_v_gt, reduction="mean")
            has_fp = fp_v_gt > 0.5
            if has_fp.any():
                vb_fp = q_gt_boxes_fg[has_fp]
                v_diag = torch.hypot(
                    (vb_fp[:, 2] - vb_fp[:, 0]).clamp_min(12.0),
                    (vb_fp[:, 3] - vb_fp[:, 1]).clamp_min(12.0),
                ).view(-1, 1, 1)
                loss_fp_reg = F.smooth_l1_loss(
                    fp_pred_q[has_fp] / v_diag,
                    q_gt_fp_xy[q_fg_mask][has_fp].float() / v_diag,
                    beta=0.08,
                    reduction="mean",
                )
                loss_footprint = loss_fp_v + loss_fp_reg
            else:
                loss_footprint = loss_fp_v + pred_fp_xy.sum() * 0.0

            if pred_q_mount_vis_logit is not None and pred_q_mount_boxes is not None:
                m_pred_v = pred_q_mount_vis_logit.squeeze(-1)[q_fg_mask].float()
                m_pred_b = pred_q_mount_boxes[q_fg_mask].float()
                m_v_gt = q_gt_mount_val[q_fg_mask]
                loss_m_v = F.binary_cross_entropy_with_logits(m_pred_v, m_v_gt, reduction="mean")
                has_m = m_v_gt > 0.5
                if has_m.any():
                    gt_mb = q_gt_mount_box[q_fg_mask][has_m].float()
                    _, m_ciou = compute_box_iou_ciou(m_pred_b[has_m], gt_mb)
                    m_nwd = compute_box_nwd(m_pred_b[has_m], gt_mb)
                    loss_mount = loss_m_v + (0.65 * (1.0 - m_ciou) + 0.35 * (1.0 - m_nwd)).mean()
                else:
                    loss_mount = loss_m_v + pred_q_mount_boxes.sum() * 0.0

        loss_box = loss_box_pyr + 1.25 * loss_box_q_reg

        loss_cutout = step_out["p2_feat"].sum() * 0.0
        if cutout_head is not None and cutout_roi_list:
            rois_cat = torch.cat(cutout_roi_list, dim=0)
            masks_cat = torch.cat(cutout_gt_list, dim=0).clamp(0.0, 1.0)
            if rois_cat.shape[0] > self.max_cutout_rois_per_step:
                rois_cat = rois_cat[: self.max_cutout_rois_per_step]
                masks_cat = masks_cat[: self.max_cutout_rois_per_step]
            indices = cutout_query_indices[:rois_cat.shape[0]]
            query_vectors = torch.stack([step_out["query_condition"][b, q] for b, q in indices])
            cut_logits = cutout_head(step_out["p2_feat"], step_out["aux_edge_3ch"], rois_cat, query_vectors)
            cut_prob = torch.sigmoid(cut_logits.float())
            masks_f = masks_cat.float()
            bce_cut = F.binary_cross_entropy_with_logits(cut_logits.float(), masks_f, reduction="mean")
            inter = (cut_prob * masks_f).sum(dim=(1, 2, 3))
            denom = cut_prob.sum(dim=(1, 2, 3)) + masks_f.sum(dim=(1, 2, 3)) + 1.0
            dice_cut = (1.0 - (2.0 * inter + 1.0) / denom).mean()
            edge_cut = self._mask_boundary_l1(cut_prob, masks_f)
            loss_cutout = bce_cut + 0.80 * dice_cut + 0.60 * edge_cut
        elif cutout_head is not None:
            loss_cutout = sum(p.sum() for p in cutout_head.parameters()) * 0.0

        total = (
            self.cls_weight * loss_cls
            + self.box_weight * loss_box
            + 0.02 * loss_box_nll
            + self.quad_weight * loss_quad
            + self.subtype_weight * loss_subtype
            + self.cabin_weight * loss_cabin
            + self.footprint_weight * loss_footprint
            + self.cutout_weight * loss_cutout
            + 0.45 * loss_mount
            + self.appearance_weight * loss_appearance
            + self.teacher_kl_weight * loss_teacher_kl
        )
        num_fg_tot = pyr_fg_mask.sum().float() + (q_fg_mask.sum().float() if num_q > 0 else 0.0)
        return {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "loss_box_nll": loss_box_nll.detach(),
            "loss_appearance": loss_appearance.detach(),
            "loss_teacher_kl": loss_teacher_kl.detach(),
            "loss_quad": loss_quad.detach(),
            "loss_subtype": loss_subtype.detach(),
            "loss_cabin": loss_cabin.detach(),
            "loss_footprint": loss_footprint.detach(),
            "loss_cutout": loss_cutout.detach(),
            "loss_mount": loss_mount.detach(),
            "num_fg": num_fg_tot.detach(),
        }


TemporalPlateDetLoss = TemporalTaskAlignedDetLoss


def extract_rgba_cutout(
    frame_rgb: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | list[float] | np.ndarray | VehicleDetection,
    cutout_mask_64: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Extract an `(H_crop, W_crop, 4)` `uint8` RGBA vehicle cutout from `frame_rgb` and a `64x64` soft mask."""
    if isinstance(bbox_xyxy, VehicleDetection):
        box_raw: Any = bbox_xyxy.bbox_xyxy
        if cutout_mask_64 is None:
            cutout_mask_64 = bbox_xyxy.cutout_mask_64
    else:
        box_raw = bbox_xyxy
    if cutout_mask_64 is None:
        raise ValueError("cutout_mask_64 is required to extract RGBA vehicle cutout")

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB frame, got shape {frame_rgb.shape}")
    h_orig, w_orig = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    if h_orig < 1 or w_orig < 1:
        raise ValueError(f"Empty frame shape {frame_rgb.shape}")

    b_arr = np.asarray(box_raw, dtype=np.float64).reshape(-1)
    if b_arr.size < 4 or not np.all(np.isfinite(b_arr[:4])):
        raise ValueError(f"Invalid bbox_xyxy: {box_raw!r}")
    vx0, vy0, vx1, vy1 = float(b_arr[0]), float(b_arr[1]), float(b_arr[2]), float(b_arr[3])

    ix0 = max(0, min(w_orig - 1, int(round(vx0))))
    iy0 = max(0, min(h_orig - 1, int(round(vy0))))
    ix1 = max(ix0 + 1, min(w_orig, int(round(vx1))))
    iy1 = max(iy0 + 1, min(h_orig, int(round(vy1))))

    if isinstance(cutout_mask_64, torch.Tensor):
        m_np = cutout_mask_64.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        m_np = np.asarray(cutout_mask_64)
    if m_np.dtype == np.uint8:
        m_f = m_np.astype(np.float32) / 255.0
    else:
        m_f = m_np.astype(np.float32)
    m_2d = np.squeeze(m_f)
    if m_2d.ndim != 2 or m_2d.shape[0] < 1 or m_2d.shape[1] < 1:
        raise ValueError(f"Expected 2D cutout mask after squeeze, got shape {m_np.shape}")
    m_2d = np.clip(np.nan_to_num(m_2d, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    rgb_patch = np.ascontiguousarray(frame_rgb[iy0:iy1, ix0:ix1, :3], dtype=np.uint8)
    alpha_u8 = np.clip(
        np.rint(cv2.resize(m_2d, (ix1 - ix0, iy1 - iy0), interpolation=cv2.INTER_LINEAR) * 255.0),
        0.0,
        255.0,
    ).astype(np.uint8)
    return np.dstack([rgb_patch, alpha_u8])


class TemporalDetKDAdapter(nn.Module):
    """Training-only `1x1` FPN feature adapters (`Student neck_ch -> Teacher neck_ch`) stored outside Student."""

    def __init__(self, student_ch: int, teacher_ch: int) -> None:
        super().__init__()
        self.projs = nn.ModuleDict(
            {
                key: nn.Conv2d(int(student_ch), int(teacher_ch), kernel_size=1, bias=True)
                for key in ("p2_feat", "p3_feat", "p4_feat")
            }
        )
        for m in self.projs.values():
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            nn.init.zeros_(m.bias)


def compute_temporal_det_kd_loss(
    student_out: dict[str, Any],
    teacher_out: dict[str, Any],
    *,
    temperature: float = 2.0,
    kd_adapter: TemporalDetKDAdapter | nn.Module | None = None,
) -> torch.Tensor:
    """Synthetic-only `Base -> Nano` distillation loss (pyramid + FPN `p2/p3/p4` + Teacher vehicle/mount queries)."""
    s_cls_all = student_out["cls_logits"].float()
    t_cls_all = teacher_out["cls_logits"].detach().float()
    num_q_s = int(student_out.get("num_vehicle_queries", 300))
    num_q_t = int(teacher_out.get("num_vehicle_queries", 300))
    a_pyr_s = max(0, s_cls_all.shape[1] - num_q_s)
    a_pyr_t = max(0, t_cls_all.shape[1] - num_q_t)
    a_pyr = min(a_pyr_s, a_pyr_t)

    # 1. Spatially aligned P2/P3/P4 pyramid anchors: distill `[CLS_PLATE, CLS_SIGNBOARD_HARDNEG]` + plate boxes/corners
    pyr_idx = [CLS_PLATE, CLS_SIGNBOARD_HARDNEG]
    s_pyr = s_cls_all[:, :a_pyr, pyr_idx]
    t_pyr = t_cls_all[:, :a_pyr, pyr_idx]
    t_prob_pyr = torch.sigmoid(t_pyr)
    s_prob_pyr = torch.sigmoid(s_pyr.detach())
    focus = (torch.maximum(t_prob_pyr, 0.65 * s_prob_pyr).amax(dim=-1, keepdim=True).clamp_min(0.02)) ** 0.75
    t_soft = torch.sigmoid(t_pyr / temperature)
    kd_cls = F.binary_cross_entropy_with_logits(s_pyr / temperature, t_soft, reduction="none")
    loss_kd = (kd_cls * focus).sum() / focus.sum().clamp_min(1.0) * (temperature * temperature)

    t_plate_mask = t_prob_pyr[:, :, 0] > 0.32
    if t_plate_mask.any():
        sp_b = student_out["pred_boxes"][:, :a_pyr][t_plate_mask].float()
        tp_b = teacher_out["pred_boxes"][:, :a_pyr][t_plate_mask].detach().float()
        _, p_ciou = compute_box_iou_ciou(sp_b, tp_b)
        p_nwd = compute_box_nwd(sp_b, tp_b)
        sp_c = student_out["pred_corners"][:, :a_pyr][t_plate_mask].float()
        tp_c = teacher_out["pred_corners"][:, :a_pyr][t_plate_mask].detach().float()
        diag_t = torch.hypot((tp_b[:, 2] - tp_b[:, 0]).clamp_min(6.0), (tp_b[:, 3] - tp_b[:, 1]).clamp_min(4.0)).view(-1, 1, 1)
        loss_kd = loss_kd + 0.30 * (0.65 * (1.0 - p_ciou) + 0.35 * (1.0 - p_nwd)).mean() + 0.25 * F.smooth_l1_loss(
            sp_c / diag_t, tp_c / diag_t, beta=0.06
        )

    # 2. Multi-scale FPN feature distillation (`p2, p3, p4`) via external 1x1 adapters (or spatial energy fallback)
    fpn_terms: list[torch.Tensor] = []
    projs = getattr(kd_adapter, "projs", None) if kd_adapter is not None else None
    for f_key in ("p2_feat", "p3_feat", "p4_feat"):
        sf = student_out.get(f_key)
        tf = teacher_out.get(f_key)
        if not isinstance(sf, torch.Tensor) or not isinstance(tf, torch.Tensor):
            continue
        tf_det = tf.detach().float()
        if tf_det.shape[-2:] != sf.shape[-2:]:
            tf_det = F.interpolate(tf_det, size=sf.shape[-2:], mode="bilinear", align_corners=False)
        if projs is not None and f_key in projs:
            sf_proj = projs[f_key](sf).float()
        elif sf.shape[1] == tf_det.shape[1]:
            sf_proj = sf.float()
        else:
            sf_proj = None
        if sf_proj is not None:
            t_rms = tf_det.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()
            s_rms = sf_proj.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()
            sp_weight = (t_rms / t_rms.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)).clamp(0.4, 3.0)
            cos_term = 1.0 - F.cosine_similarity(sf_proj, tf_det, dim=1, eps=1e-6).unsqueeze(1)
            l1_term = F.smooth_l1_loss(sf_proj / s_rms, tf_det / t_rms, beta=0.10, reduction="none").mean(dim=1, keepdim=True)
            fpn_terms.append(((0.60 * cos_term + 0.40 * l1_term) * sp_weight).mean())
        else:
            s_att = F.normalize(sf.float().square().mean(dim=1).flatten(1), dim=-1, eps=1e-6)
            t_att = F.normalize(tf_det.square().mean(dim=1).flatten(1), dim=-1, eps=1e-6)
            fpn_terms.append(F.smooth_l1_loss(s_att, t_att, beta=0.05) * float(s_att.shape[-1]) * 0.05)
    if fpn_terms:
        loss_kd = loss_kd + 0.35 * (sum(fpn_terms) / float(len(fpn_terms)))

    # 3. Teacher (`dfine-small` stride-8/16/32) confident vehicle queries (`prob >= 0.35`, matched by IoU + center dist)
    if num_q_s > 0 and num_q_t > 0 and s_cls_all.shape[1] > a_pyr_s and t_cls_all.shape[1] > a_pyr_t:
        t_v_raw = teacher_out.get("veh_log_odds")
        s_v_raw = student_out.get("veh_log_odds")
        t_veh_logits = (
            t_v_raw.detach().float().reshape(t_cls_all.shape[0], -1)
            if isinstance(t_v_raw, torch.Tensor)
            else t_cls_all[:, a_pyr_t:, CLS_VEHICLE]
        )
        s_veh_logits = (
            s_v_raw.float().reshape(s_cls_all.shape[0], -1)
            if isinstance(s_v_raw, torch.Tensor)
            else s_cls_all[:, a_pyr_s:, CLS_VEHICLE]
        )
        t_q_prob = torch.sigmoid(t_veh_logits)
        s_q_boxes_live = student_out["pred_boxes"][:, a_pyr_s:].float()
        s_q_boxes = s_q_boxes_live.detach()
        t_q_boxes = teacher_out["pred_boxes"][:, a_pyr_t:].detach().float()
        s_q_sub = student_out["pred_subtype_logits"][:, a_pyr_s:].float()
        t_q_sub = teacher_out["pred_subtype_logits"][:, a_pyr_t:].detach().float()
        s_m_boxes = student_out.get("pred_q_mount_boxes")
        t_m_boxes = teacher_out.get("pred_q_mount_boxes")
        s_m_vis = student_out.get("pred_q_mount_vis_logit")
        t_m_vis = teacher_out.get("pred_q_mount_vis_logit")

        p2_ref = student_out.get("p2_feat")
        if isinstance(p2_ref, torch.Tensor):
            img_h_f = float(p2_ref.shape[-2] * 4)
            img_w_f = float(p2_ref.shape[-1] * 4)
        else:
            img_h_f = float(max(64.0, float(t_q_boxes[..., 3].max().item())))
            img_w_f = float(max(64.0, float(t_q_boxes[..., 2].max().item())))
        img_whwh = s_q_boxes_live.new_tensor([img_w_f, img_h_f, img_w_f, img_h_f]).clamp_min(32.0)

        matched_s_sub: list[torch.Tensor] = []
        matched_t_sub: list[torch.Tensor] = []
        matched_s_vlogit: list[torch.Tensor] = []
        matched_t_vlogit: list[torch.Tensor] = []
        matched_s_box: list[torch.Tensor] = []
        matched_t_box: list[torch.Tensor] = []
        matched_s_mvis: list[torch.Tensor] = []
        matched_t_mvis: list[torch.Tensor] = []
        matched_s_mbox: list[torch.Tensor] = []
        matched_t_mbox: list[torch.Tensor] = []
        neg_s_vlogit: list[torch.Tensor] = []

        for b in range(s_cls_all.shape[0]):
            t_valid = t_q_prob[b] >= 0.35
            s_vb = s_veh_logits[b]
            if not t_valid.any():
                fp_q = torch.sigmoid(s_vb.detach()) > 0.16
                if fp_q.any():
                    neg_s_vlogit.append(s_vb[fp_q])
                continue

            tb = t_q_boxes[b, t_valid]
            tsub = t_q_sub[b, t_valid]
            tvl = t_veh_logits[b, t_valid]
            sb = s_q_boxes[b]

            ious_ts = pairwise_box_iou(tb, sb)  # (M_t, Q_s)
            tcx = 0.5 * (tb[:, 0:1] + tb[:, 2:3])
            tcy = 0.5 * (tb[:, 1:2] + tb[:, 3:4])
            scx = 0.5 * (sb[:, 0] + sb[:, 2]).unsqueeze(0)
            scy = 0.5 * (sb[:, 1] + sb[:, 3]).unsqueeze(0)
            t_diag = torch.hypot((tb[:, 2:3] - tb[:, 0:1]).clamp_min(24.0), (tb[:, 3:4] - tb[:, 1:2]).clamp_min(24.0))
            center_dist = torch.hypot(tcx - scx, tcy - scy) / (2.0 * t_diag)
            match_cost = 2.0 * (1.0 - ious_ts) + center_dist - 0.15 * torch.sigmoid(s_vb.detach()).unsqueeze(0)

            ti_np, si_np = _match_query_nll(match_cost.cpu().numpy())
            ti_t = torch.as_tensor(ti_np, device=sb.device, dtype=torch.long)
            si_t = torch.as_tensor(si_np, device=sb.device, dtype=torch.long)
            valid_pair = (ious_ts[ti_t, si_t] >= 0.05) | (center_dist[ti_t, si_t] <= 1.25)
            ti_t = ti_t[valid_pair]
            si_t = si_t[valid_pair]

            s_matched_mask = torch.zeros((sb.shape[0],), device=sb.device, dtype=torch.bool)
            if si_t.numel() > 0:
                s_matched_mask[si_t] = True
                matched_s_vlogit.append(s_vb[si_t])
                matched_t_vlogit.append(tvl[ti_t])
                matched_s_box.append(s_q_boxes_live[b, si_t])
                matched_t_box.append(tb[ti_t])
                matched_s_sub.append(s_q_sub[b, si_t])
                matched_t_sub.append(tsub[ti_t])
                if (
                    isinstance(s_m_vis, torch.Tensor)
                    and isinstance(t_m_vis, torch.Tensor)
                    and isinstance(s_m_boxes, torch.Tensor)
                    and isinstance(t_m_boxes, torch.Tensor)
                ):
                    smv_b = s_m_vis[b, :, 0].float()[si_t]
                    tmv_b = t_m_vis[b, t_valid, 0].detach().float()[ti_t]
                    matched_s_mvis.append(smv_b)
                    matched_t_mvis.append(tmv_b)
                    m_active = torch.sigmoid(tmv_b) >= 0.35
                    if m_active.any():
                        matched_s_mbox.append(s_m_boxes[b].float()[si_t[m_active]])
                        matched_t_mbox.append(t_m_boxes[b, t_valid].detach().float()[ti_t[m_active]])

            max_iou_s = ious_ts.amax(dim=0)
            fp_q = (~s_matched_mask) & (max_iou_s < 0.22) & (torch.sigmoid(s_vb.detach()) > 0.16)
            if fp_q.any():
                neg_s_vlogit.append(s_vb[fp_q])

        # Distill trainable attribute/mount heads (frozen COCO vehicle logits/boxes need no synthetic KD).
        if matched_s_mvis:
            smv_cat = torch.cat(matched_s_mvis, dim=0)
            tmv_cat = torch.cat(matched_t_mvis, dim=0)
            loss_kd = loss_kd + 0.30 * F.binary_cross_entropy_with_logits(
                smv_cat / temperature, torch.sigmoid(tmv_cat / temperature), reduction="mean"
            ) * (temperature * temperature)
        if matched_s_mbox:
            smb_cat = torch.cat(matched_s_mbox, dim=0)
            tmb_cat = torch.cat(matched_t_mbox, dim=0)
            m_giou = compute_box_giou(smb_cat, tmb_cat)
            m_nwd = compute_box_nwd(smb_cat, tmb_cat)
            m_l1 = F.l1_loss(smb_cat / img_whwh, tmb_cat / img_whwh, reduction="mean")
            loss_kd = loss_kd + 0.35 * (0.65 * (1.0 - m_giou) + 0.35 * (1.0 - m_nwd)).mean() + 1.5 * m_l1
        if matched_s_sub:
            s_sub_cat = F.log_softmax(torch.cat(matched_s_sub, dim=0) / temperature, dim=-1)
            t_sub_cat = F.softmax(torch.cat(matched_t_sub, dim=0) / temperature, dim=-1)
            loss_kd = loss_kd + 0.25 * F.kl_div(s_sub_cat, t_sub_cat, reduction="batchmean") * (
                temperature * temperature
            )

    return loss_kd


@torch.no_grad()
def decode_temporal_det_predictions(
    model: TemporalPlateDetNet | UnifiedTemporalDetORT,
    frame_rgb: np.ndarray,
    device: torch.device | str = "cpu",
    input_hw: tuple[int, int] = (448, 768),
    plate_thresh: float = 0.26,
    vehicle_thresh: float = 0.28,
    nms_iou: float = 0.45,
    enable_far_zoom: bool = True,
    max_zoom_rois: int = 6,
    weak_plate_zoom_thresh: float = 0.08,
    hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    scout_rois_xyxy: list[tuple[int, int, int, int]] | None = None,
    compute_vehicle_cutouts: bool = True,
    emit_rgba_cutouts: bool = False,
    frame_key: str | None = None,
    provenance: ObservationProvenance = ObservationProvenance.DETECTOR_FULL,
) -> tuple[FrameDetectionResult, tuple[torch.Tensor, torch.Tensor]]:
    """Stateful/Stateless Gen-4 full-frame decode with `D-FINE` whole-vehicle queries, Smart-Zoom & `64x64` cutouts.

    Invariants:
    1. Pass-1 updates `h_next` on the full-frame pyramid ONLY; Pass-2 Smart-Zoom tiles run with `h_prev=None`
       so cropped sub-tiles NEVER pollute or shift full-frame spatial recurrent memory.
    2. Query observations carry calibrated log-odds, full-frame covariance, class-free
       appearance and subtype/unknown attributes without subtype identity vetoes.
    3. Smart-Zoom triggers from (a) weak standalone `CLS_PLATE` peaks (`>= weak_plate_zoom_thresh`) even when
       `len(vehicles) == 0`, (b) external motion-scout ROIs, AND (c) distant/medium vehicles (`vw * scale <= 340`)
       without chopping giant close-up trucks into context-free container-wall tiles.
    """
    model.eval()
    dev = torch.device(device)
    h_orig, w_orig = frame_rgb.shape[:2]

    lb_img, scale, pad_x, pad_y = letterbox_rgb(frame_rgb, target_hw=input_hw)
    inp = torch.from_numpy(lb_img).to(dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    step_out = model.forward_step(inp, h_prev=hidden_state)
    h_next = (step_out["h_next"][0].detach(), step_out["h_next"][1].detach())

    probs = torch.sigmoid(step_out["cls_logits"][0]).cpu().numpy()  # (A, 3)
    boxes_lb = step_out["pred_boxes"][0].cpu().numpy()              # (A, 4) in letterbox coords
    boxes = boxes_lb.copy()
    corners = step_out["pred_corners"][0].cpu().numpy()             # (A, 4, 2)
    cab_val_all = torch.sigmoid(step_out["pred_cabin_valid_logit"][0, :, 0]).cpu().numpy()
    cab_boxes = step_out["pred_cabin_boxes"][0].cpu().numpy()
    fp_val_all = torch.sigmoid(step_out["pred_footprint_valid_logit"][0, :, 0]).cpu().numpy()
    footprints = step_out["pred_footprints"][0].cpu().numpy()

    inv_s = 1.0 / max(1e-6, scale)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) * inv_s, 0.0, float(w_orig))
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) * inv_s, 0.0, float(h_orig))
    corners[:, :, 0] = np.clip((corners[:, :, 0] - pad_x) * inv_s, 0.0, float(w_orig))
    corners[:, :, 1] = np.clip((corners[:, :, 1] - pad_y) * inv_s, 0.0, float(h_orig))
    cab_boxes[:, [0, 2]] = np.clip((cab_boxes[:, [0, 2]] - pad_x) * inv_s, 0.0, float(w_orig))
    cab_boxes[:, [1, 3]] = np.clip((cab_boxes[:, [1, 3]] - pad_y) * inv_s, 0.0, float(h_orig))
    footprints[:, :, 0] = (footprints[:, :, 0] - pad_x) * inv_s
    footprints[:, :, 1] = (footprints[:, :, 1] - pad_y) * inv_s

    num_q_dec = int(step_out.get("num_vehicle_queries", 0))
    a_pyr_dec = max(0, int(probs.shape[0]) - num_q_dec)
    q_mount_boxes_np: np.ndarray | None = None
    q_mount_probs_np: np.ndarray | None = None
    if "pred_q_mount_boxes" in step_out and "pred_q_mount_vis_logit" in step_out:
        mb_raw = step_out["pred_q_mount_boxes"][0].cpu().numpy().copy()
        mb_raw[:, [0, 2]] = np.clip((mb_raw[:, [0, 2]] - pad_x) * inv_s, 0.0, float(w_orig))
        mb_raw[:, [1, 3]] = np.clip((mb_raw[:, [1, 3]] - pad_y) * inv_s, 0.0, float(h_orig))
        q_mount_boxes_np = mb_raw
        q_mount_probs_np = torch.sigmoid(step_out["pred_q_mount_vis_logit"][0, :, 0]).cpu().numpy()

    v_scores = probs[:, CLS_VEHICLE]
    s_scores = probs[:, CLS_SIGNBOARD_HARDNEG] if probs.shape[-1] > CLS_SIGNBOARD_HARDNEG else np.zeros_like(v_scores)
    vw_all = boxes[:, 2] - boxes[:, 0]
    vh_all = boxes[:, 3] - boxes[:, 1]
    v_mask = (
        (v_scores >= vehicle_thresh)
        & (v_scores > s_scores)
        & (vw_all >= 10.0)
        & (vh_all >= 10.0)
    )
    v_indices_cand = np.nonzero(v_mask)[0]
    v_keep_local = nms_numpy(boxes[v_indices_cand], v_scores[v_indices_cand], iou_thresh=0.52)
    v_keep_anchors = v_indices_cand[v_keep_local]

    if len(v_keep_anchors) > 1:
        # Suppress low-confidence multi-vehicle wrapper boxes & interior cab/bed sub-boxes
        v_filtered: list[int] = []
        for ai in v_keep_anchors:
            ax0, ay0, ax1, ay1 = [float(x) for x in boxes[ai]]
            a_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
            asc = float(v_scores[ai])
            wrapped_cnt = 0
            drop_sub = False
            for bi in v_keep_anchors:
                if int(ai) == int(bi):
                    continue
                bx0, by0, bx1, by1 = [float(x) for x in boxes[bi]]
                b_area = max(1.0, (bx1 - bx0) * (by1 - by0))
                bsc = float(v_scores[bi])
                inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
                if a_area > 1.85 * b_area and (inter / b_area) >= 0.78 and bsc >= 0.85 * asc:
                    wrapped_cnt += 1
                if b_area > 1.75 * a_area and (inter / a_area) >= 0.86 and bsc >= asc + 0.14:
                    drop_sub = True
                    break
                if (
                    b_area > 1.65 * a_area
                    and (inter / a_area) >= 0.82
                    and ay1 < by0 + 0.76 * (by1 - by0)
                    and bsc >= 0.55 * asc
                ):
                    drop_sub = True
                    break
            if not drop_sub and wrapped_cnt < 2:
                v_filtered.append(int(ai))
        v_keep_anchors = np.asarray(v_filtered, dtype=v_indices_cand.dtype)

    # Compute 64x64 cutouts early when needed so silhouette-aware sub-part vs foreground-car check has exact masks
    need_early_cutouts = compute_vehicle_cutouts
    cutouts_cand_np: np.ndarray | None = None
    if need_early_cutouts and len(v_keep_anchors) > 0:
        lb_v_boxes = np.zeros((len(v_keep_anchors), 5), dtype=np.float32)
        lb_v_boxes[:, 1] = np.clip(boxes[v_keep_anchors, 0] * scale + pad_x, 0.0, float(input_hw[1]))
        lb_v_boxes[:, 2] = np.clip(boxes[v_keep_anchors, 1] * scale + pad_y, 0.0, float(input_hw[0]))
        lb_v_boxes[:, 3] = np.clip(boxes[v_keep_anchors, 2] * scale + pad_x, 0.0, float(input_hw[1]))
        lb_v_boxes[:, 4] = np.clip(boxes[v_keep_anchors, 3] * scale + pad_y, 0.0, float(input_hw[0]))
        rois_t = torch.from_numpy(lb_v_boxes).to(dev)
        cut_logits = model.predict_vehicle_cutouts(
            step_out["p2_feat"], step_out["aux_edge_3ch"], rois_t,
            step_out["query_condition"][0, torch.as_tensor(v_keep_anchors - a_pyr_dec, device=dev)],
        )
        cutouts_cand_np = torch.sigmoid(cut_logits[:, 0]).cpu().numpy()

    kept_cand_pos = list(range(len(v_keep_anchors)))

    cutouts_64_np: np.ndarray | None = (
        cutouts_cand_np[kept_cand_pos] if (compute_vehicle_cutouts and cutouts_cand_np is not None and len(kept_cand_pos) > 0) else None
    )

    max_v_area = 1.0
    if len(v_keep_anchors) > 0:
        max_v_area = float(
            max(
                1.0,
                max((float(boxes[ai, 2] - boxes[ai, 0]) * float(boxes[ai, 3] - boxes[ai, 1])) for ai in v_keep_anchors),
            )
        )

    vehicles_legacy: list[tuple[float, float, float, float, float]] = []
    vehicles_detailed: list[VehicleDetection] = []
    for k_v, a_idx in enumerate(v_keep_anchors):
        vx0, vy0, vx1, vy1 = [float(x) for x in boxes[a_idx]]
        vsc = float(v_scores[a_idx])
        vehicles_legacy.append((vx0, vy0, vx1, vy1, vsc))
        sub_p = step_out["subtype_probs"][0, int(a_idx) - a_pyr_dec].cpu().numpy().astype(np.float32)
        sub_id = int(np.argmax(sub_p))
        c_valid = bool(cab_val_all[a_idx] >= 0.50)
        c_box = (
            (
                float(cab_boxes[a_idx, 0]),
                float(cab_boxes[a_idx, 1]),
                float(cab_boxes[a_idx, 2]),
                float(cab_boxes[a_idx, 3]),
            )
            if c_valid
            else None
        )
        f_valid = bool(fp_val_all[a_idx] >= 0.50)
        f_xy = footprints[a_idx].astype(np.float32) if f_valid else None
        mask64 = cutouts_64_np[k_v].astype(np.float32) if cutouts_64_np is not None else None
        rgba_crop: np.ndarray | None = None
        if emit_rgba_cutouts and mask64 is not None:
            rgba_crop = extract_rgba_cutout(frame_rgb, (vx0, vy0, vx1, vy1), mask64)

        qj_v = int(a_idx) - a_pyr_dec
        m_box_v: tuple[float, float, float, float] | None = None
        m_prob_v = 0.0
        if q_mount_boxes_np is not None and q_mount_probs_np is not None and 0 <= qj_v < num_q_dec:
            m_prob_v = float(q_mount_probs_np[qj_v])
            mx0_v, my0_v, mx1_v, my1_v = [float(x) for x in q_mount_boxes_np[qj_v]]
            if (mx1_v - mx0_v) >= 4.0 and (my1_v - my0_v) >= 2.5:
                m_box_v = (mx0_v, my0_v, mx1_v, my1_v)

        v_area = max(1.0, (vx1 - vx0) * (vy1 - vy0))
        is_primary = bool(len(v_keep_anchors) == 1 or v_area >= 0.25 * max_v_area or vy1 >= 0.66 * float(h_orig))

        vehicles_detailed.append(
            VehicleDetection(
                bbox_xyxy=(vx0, vy0, vx1, vy1),
                score=vsc,
                vehicle_subtype=sub_id,
                subtype_probs=sub_p,
                cutout_mask_64=mask64,
                cutout_rgba=rgba_crop,
                cabin_bbox_xyxy=c_box,
                cabin_valid=c_valid,
                ground_footprint_xy=f_xy,
                footprint_valid=f_valid,
                primary_gate_subject=is_primary,
                plate_mount_box_xyxy=m_box_v,
                plate_mount_prob=m_prob_v,
                veh_log_odds=float(step_out["veh_log_odds"][0, qj_v]),
                bbox_cov_4x4=step_out["bbox_cov_4x4"][0, qj_v].cpu().numpy().astype(np.float32) * inv_s**2,
                appearance_emb=step_out["appearance_emb"][0, qj_v].cpu().numpy().astype(np.float32),
                subtype_unknown_prob=float(step_out["subtype_unknown_prob"][0, qj_v]),
                amodal_bbox_xyxy=tuple(float(x) for x in (boxes_lb[a_idx] - np.array([pad_x, pad_y, pad_x, pad_y])) * inv_s),
                provenance=provenance,
            )
        )

    # Pass-1 Plate Detections (`p_score > v_score` and `p_score > s_score`)
    p_scores = probs[:, CLS_PLATE]
    qw_all = corners[:, :, 0].max(axis=-1) - corners[:, :, 0].min(axis=-1)
    qh_all = corners[:, :, 1].max(axis=-1) - corners[:, :, 1].min(axis=-1)
    aspect_all = vw_all / np.maximum(1e-3, vh_all)
    p_mask = (
        (p_scores >= plate_thresh)
        & (p_scores > v_scores)
        & (p_scores > s_scores)
        & (vw_all >= 5.0)
        & (vh_all >= 2.8)
        & (vw_all <= 0.96 * w_orig)
        & (vh_all <= 0.90 * h_orig)
        & (aspect_all >= 0.45)
        & (aspect_all <= 5.50)
        & (qw_all >= 0.34 * vw_all)
        & (qw_all <= 1.90 * vw_all)
        & (qh_all >= 0.34 * vh_all)
        & (qh_all <= 1.90 * vh_all)
    )
    all_p_boxes = [boxes[p_mask]]
    all_p_scores = [p_scores[p_mask]]
    all_p_corners = [corners[p_mask]]
    all_p_provenance = [np.full(int(p_mask.sum()), provenance.value, dtype=object)]

    # Pass-2: Gen-5 Vehicle-First + 2x2 Mosaic Multi-Vehicle + Standalone-Peak Smart-Zoom (active on low-res too!)
    zoom_tile_hw = (256, 384)
    max_zoom_mag = 3.2
    if isinstance(model, UnifiedTemporalDetORT):
        max_zoom_rois = min(max_zoom_rois, 2)
    if enable_far_zoom and max_zoom_rois > 0:
        cur_p_boxes = all_p_boxes[0]
        cur_p_scores = all_p_scores[0]
        zoom_crops: list[np.ndarray] = []
        # Each tile holds 1..4 sub-regions: (qx0, qy0, qx1, qy1, z_sc, z_px, z_py, rx0, ry0, rx1, ry1)
        zoom_tile_regions: list[list[tuple[int, int, int, int, float, int, int, int, int, int, int]]] = []
        enqueued_src_windows: list[tuple[int, int, int, int]] = []

        g_min = 1.22
        tw_lim = max(32, int(math.floor(float(zoom_tile_hw[1]) / max(1e-6, g_min * scale))))
        th_lim = max(24, int(math.floor(float(zoom_tile_hw[0]) / max(1e-6, g_min * scale))))
        qw_half = zoom_tile_hw[1] // 2  # 192
        qh_half = zoom_tile_hw[0] // 2  # 128
        tw_quad_lim = max(24, int(math.floor(float(qw_half - 2) / max(1e-6, g_min * scale))))
        th_quad_lim = max(24, int(math.floor(float(qh_half - 2) / max(1e-6, g_min * scale))))

        def _letterbox_zoom_roi(
            sub_rgb: np.ndarray,
            target_hw: tuple[int, int],
            pad_val: int = 114,
        ) -> tuple[np.ndarray, float, int, int]:
            sh, sw = sub_rgb.shape[:2]
            ht, wt = target_hw
            z_sc = min(wt / float(max(1, sw)), ht / float(max(1, sh)), max_zoom_mag)
            nw = max(1, min(wt, int(round(sw * z_sc))))
            nh = max(1, min(ht, int(round(sh * z_sc))))
            interp = cv2.INTER_AREA if z_sc < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(sub_rgb, (nw, nh), interpolation=interp)
            if nh == ht and nw == wt:
                return resized, z_sc, 0, 0
            canvas_z = np.full((ht, wt, 3), pad_val, dtype=np.uint8)
            z_px = (wt - nw) // 2
            z_py = (ht - nh) // 2
            canvas_z[z_py : z_py + nh, z_px : z_px + nw] = resized
            return canvas_z, z_sc, z_px, z_py

        def _expand_for_mag_cap(
            rx0: int, ry0: int, rx1: int, ry1: int, target_hw: tuple[int, int]
        ) -> tuple[int, int, int, int]:
            min_w = min(w_orig, max(24, int(math.ceil(float(target_hw[1]) / max_zoom_mag))))
            min_h = min(h_orig, max(24, int(math.ceil(float(target_hw[0]) / max_zoom_mag))))
            cur_w = max(1, rx1 - rx0)
            cur_h = max(1, ry1 - ry0)
            if cur_w < min_w:
                cx_w = 0.5 * (rx0 + rx1)
                rx0 = int(round(cx_w - 0.5 * min_w))
                rx1 = rx0 + min_w
            if cur_h < min_h:
                cy_h = 0.5 * (ry0 + ry1)
                ry0 = int(round(cy_h - 0.5 * min_h))
                ry1 = ry0 + min_h
            return rx0, ry0, rx1, ry1

        def _clip_and_dedup_window(
            rx0: int, ry0: int, rx1: int, ry1: int, target_hw: tuple[int, int] = zoom_tile_hw
        ) -> tuple[int, int, int, int] | None:
            if w_orig < 24 or h_orig < 24:
                return None
            rx0, ry0, rx1, ry1 = _expand_for_mag_cap(rx0, ry0, rx1, ry1, target_hw)
            win_w_req = min(w_orig, max(24, rx1 - rx0))
            win_h_req = min(h_orig, max(24, ry1 - ry0))
            rx0_c = max(0, min(w_orig - win_w_req, int(rx0)))
            ry0_c = max(0, min(h_orig - win_h_req, int(ry0)))
            rx1_c = min(w_orig, rx0_c + win_w_req)
            ry1_c = min(h_orig, ry0_c + win_h_req)
            if rx1_c - rx0_c < 24 or ry1_c - ry0_c < 24:
                return None
            area_new = float((rx1_c - rx0_c) * (ry1_c - ry0_c))
            for ex0, ey0, ex1, ey1 in enqueued_src_windows:
                ix = max(0, min(rx1_c, ex1) - max(rx0_c, ex0))
                iy = max(0, min(ry1_c, ey1) - max(ry0_c, ey0))
                if float(ix * iy) / max(1.0, area_new) >= 0.65:
                    return None
            return rx0_c, ry0_c, rx1_c, ry1_c

        def _try_enqueue_window(rx0: int, ry0: int, rx1: int, ry1: int) -> None:
            if len(zoom_crops) >= max_zoom_rois:
                return
            win = _clip_and_dedup_window(rx0, ry0, rx1, ry1, zoom_tile_hw)
            if win is None:
                return
            rx0_c, ry0_c, rx1_c, ry1_c = win
            sub_rgb = frame_rgb[ry0_c:ry1_c, rx0_c:rx1_c]
            z_img, z_sc, z_px, z_py = _letterbox_zoom_roi(sub_rgb, target_hw=zoom_tile_hw)
            if (z_sc / max(1e-6, scale)) < (g_min - 1e-3):
                return
            zoom_crops.append(z_img)
            zoom_tile_regions.append([(0, 0, zoom_tile_hw[1], zoom_tile_hw[0], z_sc, z_px, z_py, rx0_c, ry0_c, rx1_c, ry1_c)])
            enqueued_src_windows.append((rx0_c, ry0_c, rx1_c, ry1_c))

        # Priority 1: Detected vehicles lacking a high-confidence plate (`ps >= 0.58`)
        veh_need_zoom: list[VehicleDetection] = []
        for vd in vehicles_detailed:
            vx0, vy0, vx1, vy1 = vd.bbox_xyxy
            vw = vx1 - vx0
            vh = vy1 - vy0
            if vw * scale < 10.0 or vh * scale < 10.0:
                continue
            if scale >= 0.92 and vw * scale >= 260.0:
                continue
            has_strong = False
            for pb, ps in zip(cur_p_boxes, cur_p_scores):
                pcx = 0.5 * (pb[0] + pb[2])
                pcy = 0.5 * (pb[1] + pb[3])
                if (
                    (vx0 - 0.05 * vw) <= pcx <= (vx1 + 0.05 * vw)
                    and (vy0 + 0.02 * vh) <= pcy <= (vy1 + 0.16 * vh)
                    and ps >= 0.58
                ):
                    has_strong = True
                    break
            if not has_strong:
                veh_need_zoom.append(vd)

        veh_need_zoom.sort(key=lambda v: (0 if (v.bbox_xyxy[2] - v.bbox_xyxy[0]) * scale <= 240.0 else 1, -v.score))

        # Separate quadrant-packable small/medium vehicles when `>= 4` vehicles exist in scene
        mosaic_cands: list[tuple[int, int, int, int]] = []
        regular_vehs: list[VehicleDetection] = []
        use_mosaic_pack = len(vehicles_detailed) >= 4
        quad_target_hw = (qh_half - 1, qw_half - 1)
        for vd in veh_need_zoom:
            vx0, vy0, vx1, vy1 = vd.bbox_xyxy
            vw = vx1 - vx0
            vh = vy1 - vy0
            if use_mosaic_pack and math.ceil(1.20 * vw) <= tw_quad_lim and math.ceil(1.20 * vh) <= th_quad_lim:
                mx = 0.10 * vw
                my = 0.10 * vh
                win_m = _clip_and_dedup_window(
                    int(math.floor(vx0 - mx)),
                    int(math.floor(vy0 - my)),
                    int(math.ceil(vx1 + mx)),
                    int(math.ceil(vy1 + my)),
                    quad_target_hw,
                )
                if win_m is not None:
                    mosaic_cands.append(win_m)
                    enqueued_src_windows.append(win_m)
                    continue
            regular_vehs.append(vd)

        if len(mosaic_cands) >= 2:
            q_slots = (
                (0, 0, qw_half - 1, qh_half - 1),
                (qw_half + 1, 0, zoom_tile_hw[1], qh_half - 1),
                (0, qh_half + 1, qw_half - 1, zoom_tile_hw[0]),
                (qw_half + 1, qh_half + 1, zoom_tile_hw[1], zoom_tile_hw[0]),
            )
            for m_start in range(0, len(mosaic_cands), 4):
                if len(zoom_crops) >= max_zoom_rois:
                    break
                chunk = mosaic_cands[m_start : m_start + 4]
                if len(chunk) == 1:
                    rx0_1, ry0_1, rx1_1, ry1_1 = chunk[0]
                    enqueued_src_windows.remove(chunk[0])
                    _try_enqueue_window(rx0_1, ry0_1, rx1_1, ry1_1)
                    continue
                canvas = np.full((zoom_tile_hw[0], zoom_tile_hw[1], 3), 114, dtype=np.uint8)
                regs_m: list[tuple[int, int, int, int, float, int, int, int, int, int, int]] = []
                for qi, (rx0_q, ry0_q, rx1_q, ry1_q) in enumerate(chunk):
                    qx0, qy0, qx1, qy1 = q_slots[qi]
                    qw_s = qx1 - qx0
                    qh_s = qy1 - qy0
                    sub_q = frame_rgb[ry0_q:ry1_q, rx0_q:rx1_q]
                    q_img, q_sc, q_px, q_py = _letterbox_zoom_roi(sub_q, target_hw=(qh_s, qw_s))
                    if (q_sc / max(1e-6, scale)) < (g_min - 1e-3):
                        continue
                    canvas[qy0:qy1, qx0:qx1] = q_img
                    regs_m.append((qx0, qy0, qx1, qy1, q_sc, qx0 + q_px, qy0 + q_py, rx0_q, ry0_q, rx1_q, ry1_q))
                if regs_m:
                    zoom_crops.append(canvas)
                    zoom_tile_regions.append(regs_m)
        elif len(mosaic_cands) == 1:
            rx0_1, ry0_1, rx1_1, ry1_1 = mosaic_cands[0]
            enqueued_src_windows.remove(mosaic_cands[0])
            _try_enqueue_window(rx0_1, ry0_1, rx1_1, ry1_1)

        for vd in regular_vehs:
            if len(zoom_crops) >= max_zoom_rois:
                break
            vx0, vy0, vx1, vy1 = vd.bbox_xyxy
            vw = vx1 - vx0
            vh = vy1 - vy0
            vw_lb = vw * scale
            if math.ceil(1.22 * vw) <= tw_lim and math.ceil(1.22 * vh) <= th_lim:
                mx = 0.11 * vw
                my = 0.11 * vh
                _try_enqueue_window(
                    int(math.floor(vx0 - mx)),
                    int(math.floor(vy0 - my)),
                    int(math.ceil(vx1 + mx)),
                    int(math.ceil(vy1 + my)),
                )
            elif vw_lb > 260.0:
                # Targeted lower-bumper zoom band `y in [vy0 + 0.48*vh, min(h_orig, vy1 + 0.08*vh)]`
                # (never drops wide trucks `vw_lb > 340`!) + upper hopper band on `concrete_mixer` (`subtype == 5`)
                by0 = int(math.floor(vy0 + 0.48 * vh))
                by1 = int(math.ceil(min(float(h_orig), vy1 + 0.08 * vh)))
                band_h = min(th_lim, max(48, by1 - by0))
                band_w = min(tw_lim, max(64, int(math.ceil(0.92 * vw))))
                cy_low = min(float(h_orig) - 0.5 * band_h, max(float(by0) + 0.5 * band_h, vy0 + 0.76 * vh))
                if vw > 1.32 * vh:
                    # Wide side-oblique chassis (`vw > 1.32 * vh`): focus high-magnification tiles on both longitudinal bumper ends
                    end_w = min(tw_lim, max(96, int(round(min(0.42 * vw, 1.50 * float(band_h))))))
                    cx_l = vx0 + 0.16 * vw
                    cx_r = vx1 - 0.16 * vw
                    end_centers = [cx_l, cx_r]
                    if vd.plate_mount_box_xyxy is not None and vd.plate_mount_prob >= 0.35:
                        cx_m = 0.5 * (vd.plate_mount_box_xyxy[0] + vd.plate_mount_box_xyxy[2])
                        if abs(cx_m - cx_r) < abs(cx_m - cx_l):
                            end_centers = [cx_r, cx_l]
                    for cx_e in end_centers:
                        _try_enqueue_window(
                            int(round(cx_e - 0.5 * end_w)),
                            int(round(cy_low - 0.5 * band_h)),
                            int(round(cx_e + 0.5 * end_w)),
                            int(round(cy_low + 0.5 * band_h)),
                        )
                else:
                    if vd.plate_mount_box_xyxy is not None and vd.plate_mount_prob >= 0.35:
                        cx_pref = 0.5 * (vd.plate_mount_box_xyxy[0] + vd.plate_mount_box_xyxy[2])
                    elif vd.cabin_valid and vd.cabin_bbox_xyxy is not None:
                        cx_pref = 0.5 * (vd.cabin_bbox_xyxy[0] + vd.cabin_bbox_xyxy[2])
                    else:
                        cx_pref = 0.5 * (vx0 + vx1)
                    _try_enqueue_window(
                        int(round(cx_pref - 0.5 * band_w)),
                        int(round(cy_low - 0.5 * band_h)),
                        int(round(cx_pref + 0.5 * band_w)),
                        int(round(cy_low + 0.5 * band_h)),
                    )
                if vd.vehicle_subtype == 5:  # SUBTYPE_CONCRETE_MIXER upper rear hopper mount
                    cy_up = vy0 + 0.34 * vh
                    _try_enqueue_window(
                        int(round(0.5 * (vx0 + vx1) - 0.5 * band_w)),
                        int(round(cy_up - 0.5 * band_h)),
                        int(round(0.5 * (vx0 + vx1) + 0.5 * band_w)),
                        int(round(cy_up + 0.5 * band_h)),
                    )
            else:
                t_w = min(tw_lim, max(64, int(math.ceil(1.08 * vw))))
                t_h = min(th_lim, max(48, int(math.ceil(0.62 * vh))))
                cx_v = 0.5 * (vx0 + vx1)
                for yc_band in (vy0 + 0.72 * vh, vy0 + 0.30 * vh):
                    _try_enqueue_window(
                        int(round(cx_v - 0.5 * t_w)),
                        int(round(yc_band - 0.5 * t_h)),
                        int(round(cx_v + 0.5 * t_w)),
                        int(round(yc_band + 0.5 * t_h)),
                    )

        # Priority 2 (Source A): Weak standalone `CLS_PLATE` peaks (active especially when `len(vehicles) == 0`!)
        weak_mask = (
            (p_scores >= weak_plate_zoom_thresh)
            & (p_scores < 0.65)
            & (p_scores > 0.80 * s_scores)
            & (vw_all >= 4.0)
            & (vh_all >= 2.5)
            & (vw_all <= 0.45 * w_orig)
        )
        if np.any(weak_mask) and len(zoom_crops) < max_zoom_rois:
            w_idx = np.nonzero(weak_mask)[0]
            w_keep = nms_numpy(boxes[w_idx], p_scores[w_idx], iou_thresh=0.35)[:3]
            for wk in w_keep:
                wb = boxes[w_idx[wk]]
                wcx = 0.5 * (wb[0] + wb[2])
                wcy = 0.5 * (wb[1] + wb[3])
                already_strong = False
                for pb, ps in zip(cur_p_boxes, cur_p_scores):
                    if ps >= 0.65 and pb[0] - 8.0 <= wcx <= pb[2] + 8.0 and pb[1] - 8.0 <= wcy <= pb[3] + 8.0:
                        already_strong = True
                        break
                if already_strong:
                    continue
                win_w = min(tw_lim, max(96, int(round(4.5 * (wb[2] - wb[0])))))
                win_h = min(th_lim, max(64, int(round(5.0 * (wb[3] - wb[1])))))
                _try_enqueue_window(
                    int(round(wcx - 0.5 * win_w)),
                    int(round(wcy - 0.5 * win_h)),
                    int(round(wcx + 0.5 * win_w)),
                    int(round(wcy + 0.5 * win_h)),
                )

        # Priority 3 (Source B): External motion-scout / border-entry ROIs
        if scout_rois_xyxy and len(zoom_crops) < max_zoom_rois:
            for sx0, sy0, sx1, sy1 in scout_rois_xyxy:
                scx = 0.5 * (sx0 + sx1)
                scy = 0.5 * (sy0 + sy1)
                win_w = min(tw_lim, max(96, int(sx1 - sx0)))
                win_h = min(th_lim, max(64, int(sy1 - sy0)))
                _try_enqueue_window(
                    int(round(scx - 0.5 * win_w)),
                    int(round(scy - 0.5 * win_h)),
                    int(round(scx + 0.5 * win_w)),
                    int(round(scy + 0.5 * win_h)),
                )

        if zoom_crops:
            z_batch = (
                torch.from_numpy(np.stack(zoom_crops, axis=0))
                .to(dev, dtype=torch.float32)
                .permute(0, 3, 1, 2)
                / 255.0
            )
            # Stateless `h_prev=None` on sub-tiles so full-frame `h_next` is never polluted!
            # ONNX graphs have fixed spatial resolution and dynamic batch. Resize
            # zoom tiles for that graph, then restore tile coordinates below.
            graph_hw = getattr(model, "input_hw", zoom_tile_hw)
            if tuple(graph_hw) != zoom_tile_hw:
                z_batch = F.interpolate(z_batch, size=graph_hw, mode="bilinear", align_corners=False)
            z_out = model.forward_step(z_batch, h_prev=None)
            z_all_probs = torch.sigmoid(z_out["cls_logits"]).cpu().numpy()
            z_probs = z_all_probs[:, :, CLS_PLATE]
            z_vprobs = z_all_probs[:, :, CLS_VEHICLE]
            z_sprobs = z_all_probs[:, :, CLS_SIGNBOARD_HARDNEG]
            z_boxes_np = z_out["pred_boxes"].cpu().numpy()
            z_corners_np = z_out["pred_corners"].cpu().numpy().copy()
            if tuple(graph_hw) != zoom_tile_hw:
                tile_scale = np.array([zoom_tile_hw[1] / graph_hw[1], zoom_tile_hw[0] / graph_hw[0]], dtype=np.float32)
                z_boxes_np = z_boxes_np * np.tile(tile_scale, 2)
                z_corners_np *= tile_scale

            for k_i, regs_k in enumerate(zoom_tile_regions):
                zm_tile = (
                    (z_probs[k_i] >= max(plate_thresh, 0.30))
                    & (z_probs[k_i] > z_vprobs[k_i])
                    & (z_probs[k_i] > z_sprobs[k_i])
                )
                if not np.any(zm_tile):
                    continue
                zb_tile = z_boxes_np[k_i, zm_tile]
                zc_tile = z_corners_np[k_i, zm_tile]
                zs_tile = z_probs[k_i, zm_tile]
                tcx = 0.5 * (zb_tile[:, 0] + zb_tile[:, 2])
                tcy = 0.5 * (zb_tile[:, 1] + zb_tile[:, 3])

                for qx0, qy0, qx1, qy1, z_sc, z_px, z_py, rx0, ry0, rx1, ry1 in regs_k:
                    in_reg = (tcx >= float(qx0)) & (tcx <= float(qx1)) & (tcy >= float(qy0)) & (tcy <= float(qy1))
                    if not np.any(in_reg):
                        continue
                    zb = zb_tile[in_reg].copy()
                    zc = zc_tile[in_reg].copy()
                    zs = zs_tile[in_reg]
                    inv_z = 1.0 / max(1e-6, z_sc)
                    zb[:, [0, 2]] = np.clip((zb[:, [0, 2]] - z_px) * inv_z + rx0, 0.0, float(w_orig))
                    zb[:, [1, 3]] = np.clip((zb[:, [1, 3]] - z_py) * inv_z + ry0, 0.0, float(h_orig))
                    zc[:, :, 0] = np.clip((zc[:, :, 0] - z_px) * inv_z + rx0, 0.0, float(w_orig))
                    zc[:, :, 1] = np.clip((zc[:, :, 1] - z_py) * inv_z + ry0, 0.0, float(h_orig))
                    zbw = zb[:, 2] - zb[:, 0]
                    zbh = zb[:, 3] - zb[:, 1]
                    zqw = zc[:, :, 0].max(axis=-1) - zc[:, :, 0].min(axis=-1)
                    zqh = zc[:, :, 1].max(axis=-1) - zc[:, :, 1].min(axis=-1)
                    sub_w = float(max(24, rx1 - rx0))
                    edge_pad = 3.0 * inv_z
                    not_cut_left = (zb[:, 0] > float(rx0) + edge_pad) if rx0 > 2 else np.ones(len(zs), dtype=bool)
                    not_cut_right = (zb[:, 2] < float(rx1) - edge_pad) if rx1 < (w_orig - 2) else np.ones(len(zs), dtype=bool)
                    not_cut_top = (zb[:, 1] > float(ry0) + edge_pad) if ry0 > 2 else np.ones(len(zs), dtype=bool)
                    not_cut_bot = (zb[:, 3] < float(ry1) - edge_pad) if ry1 < (h_orig - 2) else np.ones(len(zs), dtype=bool)
                    valid_wh = (
                        (zbw >= 5.0)
                        & (zbh >= 2.8)
                        & (zbw <= min(0.40 * w_orig, 0.70 * sub_w))
                        & ((zbw / np.maximum(1e-3, zbh)) >= 0.45)
                        & ((zbw / np.maximum(1e-3, zbh)) <= 5.50)
                        & (zqw >= 0.35 * zbw)
                        & (zqw <= 1.90 * zbw)
                        & (zqh >= 0.35 * zbh)
                        & (zqh <= 1.90 * zbh)
                        & not_cut_left
                        & not_cut_right
                        & not_cut_top
                        & not_cut_bot
                    )
                    if np.any(valid_wh):
                        all_p_boxes.append(zb[valid_wh])
                        all_p_scores.append(zs[valid_wh])
                        all_p_corners.append(zc[valid_wh])
                        all_p_provenance.append(np.full(int(valid_wh.sum()), ObservationProvenance.DETECTOR_FAR_ZOOM.value, dtype=object))

    cat_boxes = np.concatenate(all_p_boxes, axis=0)
    cat_scores = np.concatenate(all_p_scores, axis=0)
    cat_corners = np.concatenate(all_p_corners, axis=0)
    cat_provenance = np.concatenate(all_p_provenance, axis=0)
    p_keep = nms_numpy(cat_boxes, cat_scores, iou_thresh=nms_iou)

    if len(p_keep) > 1:
        filtered_keep: list[int] = []
        for idx_a in p_keep:
            ax0, ay0, ax1, ay1 = [float(v) for v in cat_boxes[idx_a]]
            area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
            drop_a = False
            for idx_b in p_keep:
                if idx_a == idx_b:
                    continue
                bx0_c, by0_c, bx1_c, by1_c = [float(v) for v in cat_boxes[idx_b]]
                area_b = max(1.0, (bx1_c - bx0_c) * (by1_c - by0_c))
                if area_a >= 2.6 * area_b and float(cat_scores[idx_b]) >= max(0.32, 0.72 * float(cat_scores[idx_a])):
                    inter = max(0.0, min(ax1, bx1_c) - max(ax0, bx0_c)) * max(0.0, min(ay1, by1_c) - max(ay0, by0_c))
                    if inter / area_b >= 0.85:
                        drop_a = True
                        break
            if not drop_a:
                filtered_keep.append(idx_a)
        p_keep = filtered_keep

    plates_out: list[PlateDetection] = []
    for idx in p_keep:
        bx0, by0, bx1, by1 = [float(v) for v in cat_boxes[idx]]
        pcx = 0.5 * (bx0 + bx1)
        pcy = 0.5 * (by0 + by1)
        parent_veh: tuple[float, float, float, float] | None = None
        best_v_area = float("inf")
        for vi_p, (vx0, vy0, vx1, vy1, _) in enumerate(vehicles_legacy):
            vh_cand = max(1.0, vy1 - vy0)
            vw_cand = max(1.0, vx1 - vx0)
            if (vx0 - 0.06 * vw_cand - 6.0) <= pcx <= (vx1 + 0.06 * vw_cand + 6.0) and (vy0 - 0.02 * vh_cand) <= pcy <= (vy1 + 0.18 * vh_cand + 14.0):
                area = vw_cand * vh_cand
                if area < best_v_area:
                    best_v_area = area
                    parent_veh = (vx0, vy0, vx1, vy1)
        plates_out.append(
            PlateDetection(
                bbox_xyxy=(bx0, by0, bx1, by1),
                score=float(cat_scores[idx]),
                corners_xy=cat_corners[idx].astype(np.float32),
                vehicle_bbox_xyxy=parent_veh,
                raw_det_score=float(cat_scores[idx]),
                frame_key=frame_key,
                provenance=ObservationProvenance(cat_provenance[idx]),
            )
        )

    res = FrameDetectionResult(
        plates=plates_out,
        vehicle_bboxes_xyxy=vehicles_legacy,
        vehicles=vehicles_detailed,
    )
    return res, h_next


# Full tensor contract shared by eager training, inference and deployed ONNX.
UNIFIED_TEMPORAL_OUTPUT_NAMES = (
    "cls_logits", "pred_boxes", "pred_corners", "pred_subtype_logits",
    "pred_cabin_valid_logit", "pred_cabin_boxes", "pred_footprint_valid_logit",
    "pred_footprints", "p2_feat", "aux_edge_3ch", "anchor_points", "anchor_strides",
    "synth_veh_logits", "raw_q_boxes_px", "veh_log_odds", "bbox_cov_4x4",
    "appearance_emb", "query_condition", "subtype_probs", "subtype_unknown_prob",
    "pred_q_mount_boxes", "pred_q_mount_vis_logit",
)


class UnifiedTemporalDetExport(nn.Module):
    """Lossless flattening of forward_step's tensor contract for ONNX."""

    def __init__(self, model: TemporalPlateDetNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, h_p3: torch.Tensor, h_p4: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model.forward_step(images, (h_p3, h_p4))
        return tuple(out[name] for name in UNIFIED_TEMPORAL_OUTPUT_NAMES) + out["h_next"]


class UnifiedVehicleCutoutExport(nn.Module):
    def __init__(self, head: EdgeGuidedVehicleCutoutHead) -> None:
        super().__init__()
        self.head = head

    def forward(self, p2_feat: torch.Tensor, aux_edge_3ch: torch.Tensor,
                veh_rois: torch.Tensor, query_condition: torch.Tensor) -> torch.Tensor:
        return self.head(p2_feat, aux_edge_3ch, veh_rois, query_condition)


def export_unified_temporal_det_onnx(
    model: TemporalPlateDetNet, out_dir: str | Path, input_hw: tuple[int, int] = (448, 768),
) -> dict[str, Any]:
    """Export complete Stage-1 graphs, with dynamic image and ROI batch axes.

    The older Gen-4 trainer wrapper has a reduced contract. Gen-6 callers must
    use this exporter and UnifiedTemporalDetORT to retain observation fields.
    INT8 uses the repository's dense-weight quantizer; depthwise weights stay FP32.
    """
    import copy
    from vpocr_det.trainer import _fuse_conv_bn_inplace, _quantize_onnx_dense_convs_and_strip

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cpu = _fuse_conv_bn_inplace(copy.deepcopy(model).cpu().eval())
    image = torch.zeros((1, 3, *input_hw))
    hidden = cpu.init_hidden_state(1, input_hw)
    names = UNIFIED_TEMPORAL_OUTPUT_NAMES + ("h_p3_out", "h_p4_out")
    detector = destination / f"temporal_det_{model.config.preset}.onnx"
    cutout = destination / f"vehicle_cutout_{model.config.preset}.onnx"
    axes = {name: {0: "batch"} for name in ("images", "h_p3_in", "h_p4_in", *names)
            if name not in ("anchor_points", "anchor_strides")}
    torch.onnx.export(
        UnifiedTemporalDetExport(cpu), (image, *hidden), str(detector),
        input_names=["images", "h_p3_in", "h_p4_in"], output_names=list(names),
        dynamic_axes=axes, opset_version=17, dynamo=False,
    )
    with torch.no_grad():
        out = cpu.forward_step(image, hidden)
    rois = torch.tensor([[0., 0., 0., float(input_hw[1]), float(input_hw[0])]])
    query = out["query_condition"][:, 0]
    torch.onnx.export(
        UnifiedVehicleCutoutExport(cpu.cutout_head), (out["p2_feat"], out["aux_edge_3ch"], rois, query),
        str(cutout), input_names=["p2_feat", "aux_edge_3ch", "veh_rois", "query_condition"],
        output_names=["cutout_logits"], opset_version=17, dynamo=False,
        dynamic_axes={"p2_feat": {0: "batch"}, "aux_edge_3ch": {0: "batch"},
                      "veh_rois": {0: "num_vehicles"}, "query_condition": {0: "num_vehicles"},
                      "cutout_logits": {0: "num_vehicles"}},
    )
    paths: dict[str, Any] = {}
    total_bytes = 0
    for name, fp32 in (("det", detector), ("cutout", cutout)):
        int8 = fp32.with_suffix(".int8.onnx")
        _quantize_onnx_dense_convs_and_strip(fp32, int8)
        paths[f"{name}_fp32_path"] = str(fp32)
        paths[f"{name}_int8_path"] = str(int8)
        total_bytes += int8.stat().st_size
    paths["combined_int8_bytes"] = total_bytes
    if model.config.preset == "nano" and total_bytes > 5_662_310:
        raise ValueError(f"Stage-1 Nano INT8 {total_bytes} bytes exceeds 5,662,310 byte (5.4 MiB) cap")
    return paths


class UnifiedTemporalDetORT:
    """CPU ORT implementation of the same step/cutout interface used by decode.

    Returned tensors are CPU tensors, including covariance in letterbox pixels;
    decode applies the full-frame covariance Jacobian just as for PyTorch.
    """

    def __init__(self, detector_path: str | Path, cutout_path: str | Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.detector = ort.InferenceSession(str(detector_path), options, providers=["CPUExecutionProvider"])
        self.cutout = ort.InferenceSession(str(cutout_path), options, providers=["CPUExecutionProvider"])
        self.input_hw = tuple(int(value) for value in self.detector.get_inputs()[0].shape[-2:])
        actual = {out.name for out in self.detector.get_outputs()}
        expected = set(UNIFIED_TEMPORAL_OUTPUT_NAMES) | {"h_p3_out", "h_p4_out"}
        if actual != expected:
            raise ValueError("ONNX detector lacks the complete unified observation contract")

    def eval(self) -> UnifiedTemporalDetORT:
        return self

    def forward_step(self, images: torch.Tensor,
                     h_prev: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict[str, Any]:
        if h_prev is None:
            shapes = [item.shape for item in self.detector.get_inputs()[1:3]]
            h_prev = tuple(torch.zeros((images.shape[0], *shape[1:])) for shape in shapes)
        feed = {name: value.detach().cpu().numpy() for name, value in
                zip(("images", "h_p3_in", "h_p4_in"), (images, *h_prev))}
        values = self.detector.run(None, feed)
        out = {meta.name: torch.from_numpy(value) for meta, value in zip(self.detector.get_outputs(), values)}
        out["h_next"] = (out.pop("h_p3_out"), out.pop("h_p4_out"))
        out["num_vehicle_queries"] = out["veh_log_odds"].shape[1]
        return out

    def predict_vehicle_cutouts(self, p2_feat: torch.Tensor, aux_edge_3ch: torch.Tensor,
                                veh_rois: torch.Tensor, query_condition: torch.Tensor) -> torch.Tensor:
        if veh_rois.shape[0] == 0:
            return torch.zeros((0, 1, 64, 64))
        feed = {name: value.detach().cpu().numpy() for name, value in
                zip(("p2_feat", "aux_edge_3ch", "veh_rois", "query_condition"),
                    (p2_feat, aux_edge_3ch, veh_rois, query_condition))}
        return torch.from_numpy(self.cutout.run(None, feed)[0])


def set_unified_onnx_calibration(path: str | Path, temperature: float, bias: float) -> None:
    """Install disjoint-data calibration in an exported FP32 or INT8 graph.

    Calibration changes the final log-odds scale/offset only and requires no
    quantization rerun. It does not assert calibration on an unseen camera.
    """
    import onnx
    from onnx import numpy_helper

    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(bias):
        raise ValueError("Calibration requires finite temperature > 0 and finite bias")
    graph = onnx.load(str(path))
    values = {"cal_temp": temperature, "cal_bias": bias}
    found = set()
    for initializer in graph.graph.initializer:
        key = initializer.name.rsplit(".", 1)[-1]
        if key in values:
            replacement = np.full(numpy_helper.to_array(initializer).shape, values[key], dtype=np.float32)
            initializer.CopyFrom(numpy_helper.from_array(replacement, name=initializer.name))
            found.add(key)
    if found != set(values):
        raise ValueError("Exported graph does not preserve both calibration buffers")
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))
