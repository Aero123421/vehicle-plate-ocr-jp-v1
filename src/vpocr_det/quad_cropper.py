"""Equivariant projective plate inference, posterior containment and exact cropping.

``PlateQuadTrackNet`` is the Gen-6 two-mode IC-STN. ``PlateQuadNet`` keeps
its single-input training interface for legacy callers, without position channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
import copy
import math
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vpocr_det.detector import ConvBNAct, DWConvBlock
from vpocr_det.types import ObservationProvenance, PlateDetection, VehicleDetection

ROI_HEIGHT: int = 96
ROI_WIDTH: int = 160
SURROUND_HEIGHT: int = 64
SURROUND_WIDTH: int = 96
FOVEA_BASE_SCALE: float = 1.70
SURROUND_BASE_SCALE: float = 4.50


@dataclass(frozen=True)
class QuadModelConfig:
    """Configuration for `PlateQuadNet`."""

    base_ch: int = 32
    temperature: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"base_ch": self.base_ch, "temperature": self.temperature}


class PlateQuadNet(nn.Module):
    """Photometric UNet-DW (`stride 4`, `24x40`) + 2D DSNT Sub-Pixel 4-Corner Cropper & Verifier."""

    def __init__(self, config: QuadModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or QuadModelConfig()
        c = self.config.base_ch  # 32 -> ~0.37M params

        # Raw RGB/luminance plus locally normalized photometry, without position channels.
        self.stem = nn.Sequential(
            ConvBNAct(8, c, k=3, s=2),  # 48 x 80
            DWConvBlock(c, k=5),
        )
        self.enc2 = nn.Sequential(
            ConvBNAct(c, c * 2, k=3, s=2),  # 24 x 40 (stride 4)
            DWConvBlock(c * 2, k=5),
            DWConvBlock(c * 2, k=5),
        )
        self.enc3 = nn.Sequential(
            ConvBNAct(c * 2, c * 3, k=3, s=2),  # 12 x 20 (stride 8)
            DWConvBlock(c * 3, k=5),
            DWConvBlock(c * 3, k=5),
        )
        self.enc4 = nn.Sequential(
            ConvBNAct(c * 3, c * 4, k=3, s=2),  # 6 x 10 (stride 16)
            DWConvBlock(c * 4, k=5),
            DWConvBlock(c * 4, k=5),
        )

        # UNet decoder back to stride 4 (24 x 40)
        self.up3 = ConvBNAct(c * 4 + c * 3, c * 3, k=1, s=1)
        self.dec3 = DWConvBlock(c * 3, k=5)
        self.up2 = ConvBNAct(c * 3 + c * 2, c * 2, k=1, s=1)
        self.dec2 = nn.Sequential(
            DWConvBlock(c * 2, k=5),
            DWConvBlock(c * 2, k=3),
        )

        # DSNT 4-corner heatmap + sub-pixel offset heads on 24 x 40 grid
        self.heatmap_head = nn.Sequential(
            ConvBNAct(c * 2, c * 2, k=3, s=1),
            nn.Conv2d(c * 2, 4, kernel_size=1),
        )
        self.offset_head = nn.Sequential(
            ConvBNAct(c * 2, c, k=3, s=1),
            nn.Conv2d(c, 8, kernel_size=1),
        )

        # Global + corner-peak pooled verification & attribute head
        # Combines global pooled enc4 (4c) + heatmap-weighted dec2 features (4 * 2c = 8c) -> 12c? Keep compact: mean over 4 corners (2c) + global (4c) = 6c
        self.cls_head = nn.Sequential(
            nn.Linear(c * 6, c * 3),
            nn.SiLU(inplace=True),
            nn.Linear(c * 3, 5),  # [is_plate_logit, side_front, side_rear, side_pure, quality_logit]
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, rois: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass on `rois` `(B, 3, 96, 160)` float32 in `[0, 1]`.

        Returns dict with:
        - `corners_norm`: `(B, 4, 2)` normalized `[x, y]` in `[0, 1]` for `[TL, TR, BR, BL]`
        - `corner_log_probs`: `(B, 4, H4*W4)` spatial log-probabilities for KL supervision
        - `is_plate_logit`: `(B,)`
        - `side_logits`: `(B, 3)`
        - `quality_logit`: `(B,)`
        """
        bsz, _, h, w = rois.shape
        device = rois.device
        dtype = rois.dtype

        x_in = photometric_channels(rois)

        f1 = self.stem(x_in)  # 48x80
        f2 = self.enc2(f1)  # 24x40
        f3 = self.enc3(f2)  # 12x20
        f4 = self.enc4(f3)  # 6x10

        u3 = self.dec3(self.up3(torch.cat([F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False), f3], dim=1)))
        u2 = self.dec2(self.up2(torch.cat([F.interpolate(u3, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2], dim=1)))

        hm_logits = self.heatmap_head(u2).float() / max(0.1, self.config.temperature)  # (B, 4, 24, 40)
        _, _, h4, w4 = hm_logits.shape
        h4_f = h4.to(dtype=torch.float32) if isinstance(h4, torch.Tensor) else float(h4)
        w4_f = w4.to(dtype=torch.float32) if isinstance(w4, torch.Tensor) else float(w4)
        flat_logits = hm_logits.reshape(bsz, 4, h4 * w4)
        log_probs = F.log_softmax(flat_logits, dim=-1)
        probs = torch.exp(log_probs)  # (B, 4, 960)

        off_raw = self.offset_head(u2).float().reshape(bsz, 4, 2, h4 * w4)  # [-1, 1]
        off_x = off_raw[:, :, 0, :]
        off_y = off_raw[:, :, 1, :]

        cy, cx = torch.meshgrid(
            (torch.arange(h4, device=device, dtype=torch.float32) + 0.5) / h4_f,
            (torch.arange(w4, device=device, dtype=torch.float32) + 0.5) / w4_f,
            indexing="ij",
        )
        grid_x = cx.reshape(1, 1, h4 * w4) + off_x
        grid_y = cy.reshape(1, 1, h4 * w4) + off_y

        exp_x = (probs * grid_x).sum(dim=-1)  # (B, 4)
        exp_y = (probs * grid_y).sum(dim=-1)  # (B, 4)
        corners_norm = torch.stack([exp_x, exp_y], dim=-1)  # (B, 4, 2)

        # Pool global bottleneck + corner-attended features for `is_plate` verification
        glob_feat = f4.float().mean(dim=(2, 3))  # (B, 4c)
        u2_flat = u2.float().reshape(bsz, u2.shape[1], h4 * w4)  # (B, 2c, 960)
        avg_prob = probs.mean(dim=1, keepdim=True)  # (B, 1, 960)
        peak_feat = (u2_flat * avg_prob).sum(dim=-1)  # (B, 2c)
        cls_out = self.cls_head(torch.cat([glob_feat, peak_feat], dim=1))  # (B, 5)

        return {
            "corners_norm": corners_norm,
            "corner_log_probs": log_probs,
            "grid_hw": (h4, w4),
            "is_plate_logit": cls_out[:, 0],
            "side_logits": cls_out[:, 1:4],
            "quality_logit": cls_out[:, 4],
        }


def _wing_loss(pred: torch.Tensor, target: torch.Tensor, w: float = 0.04, eps: float = 0.008,
               reduction: str = "mean") -> torch.Tensor:
    """Sub-pixel Wing Loss (Feng et al., CVPR 2018) amplifying gradient near tiny corner offsets."""
    diff = (pred.float() - target.float()).abs()
    c = w - w * math.log(1.0 + w / eps)
    loss = torch.where(diff < w, w * torch.log1p(diff / eps), diff - c)
    return loss if reduction == "none" else loss.mean()


class PlateQuadLoss(nn.Module):
    """Multi-task loss for sub-pixel 4-corner DSNT localization + secondary false-positive rejection."""

    def __init__(self, sigma_cells: float = 1.15) -> None:
        super().__init__()
        self.sigma_cells = sigma_cells

    def forward(
        self,
        out: dict[str, torch.Tensor],
        is_plate_gt: torch.Tensor,  # (B,) float32 0/1
        corners_norm_gt: torch.Tensor,  # (B, 4, 2) float32
        plate_side_gt: torch.Tensor,  # (B,) long
        quality_gt: torch.Tensor,  # (B,) float32
    ) -> dict[str, torch.Tensor]:
        pred_is_plate = out["is_plate_logit"]
        loss_cls = F.binary_cross_entropy_with_logits(pred_is_plate, is_plate_gt)

        pos_mask = is_plate_gt > 0.5
        if pos_mask.any():
            p_c = out["corners_norm"][pos_mask].float()  # (P, 4, 2)
            g_c = corners_norm_gt[pos_mask].float()  # (P, 4, 2)

            # 1. Sub-pixel coordinate Wing Loss + Euclidean corner distance
            loss_coord = _wing_loss(p_c, g_c, w=0.04, eps=0.008) * 14.0
            euclid = torch.linalg.norm(p_c - g_c, dim=-1).mean()
            loss_coord = loss_coord + 2.5 * euclid

            # 2. Edge vector geometric consistency + clockwise convexity penalty (prevents quad twist/bowtie!)
            p_edges = torch.roll(p_c, shifts=-1, dims=1) - p_c
            g_edges = torch.roll(g_c, shifts=-1, dims=1) - g_c
            next_p_edges = torch.roll(p_edges, shifts=-1, dims=1)
            cross_p = p_edges[:, :, 0] * next_p_edges[:, :, 1] - p_edges[:, :, 1] * next_p_edges[:, :, 0]
            loss_convex = F.relu(1e-4 - cross_p).mean() * 8.0
            loss_edge = F.smooth_l1_loss(p_edges, g_edges, beta=0.015, reduction="mean") * 6.0 + loss_convex

            # 3. Spatial Gaussian KL divergence on 24x40 heatmaps
            h4, w4 = out["grid_hw"]
            log_p = out["corner_log_probs"][pos_mask]  # (P, 4, 960)
            device = p_c.device
            gy, gx = torch.meshgrid(
                torch.arange(h4, device=device, dtype=torch.float32) + 0.5,
                torch.arange(w4, device=device, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            gx_f = gx.reshape(1, 1, -1)  # (1, 1, 960)
            gy_f = gy.reshape(1, 1, -1)
            tx = g_c[:, :, 0:1] * float(w4)
            ty = g_c[:, :, 1:2] * float(h4)
            d2 = (gx_f - tx) ** 2 + (gy_f - ty) ** 2
            target_hm = torch.softmax(-d2 / (2.0 * (self.sigma_cells ** 2)), dim=-1)
            loss_hm = F.kl_div(log_p, target_hm, reduction="batchmean") * 0.35

            # 4. Plate side & quality
            loss_side = F.cross_entropy(out["side_logits"][pos_mask], plate_side_gt[pos_mask]) * 0.25
            loss_qual = F.smooth_l1_loss(torch.sigmoid(out["quality_logit"][pos_mask]), quality_gt[pos_mask], beta=0.05) * 0.35
        else:
            zero = pred_is_plate.sum() * 0.0
            loss_coord = zero
            loss_edge = zero
            loss_hm = zero
            loss_side = zero
            loss_qual = zero
            euclid = zero

        total = 1.2 * loss_cls + loss_coord + loss_edge + loss_hm + loss_side + loss_qual
        return {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_coord": loss_coord.detach(),
            "loss_hm": loss_hm.detach(),
            "mean_corner_err_norm": euclid.detach(),
        }


@dataclass(frozen=True)
class QuadTrackModelConfig:
    """Channel/resource configuration for the Gen-6 dual-input projective model.

    ``use_v5_heads`` is a serialized legacy flag; Gen-6 always uses IC-STN.
    """

    base_ch: int = 24
    surround_ch: int = 16
    reid_dim: int = 48
    temperature: float = 1.0
    fovea_scale: float = FOVEA_BASE_SCALE
    surround_scale: float = SURROUND_BASE_SCALE
    use_v5_heads: bool = True

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "arch": "PlateQuadTrackNet",
            "geometry_version": "gen6_ic_stn",
            "base_ch": self.base_ch,
            "surround_ch": self.surround_ch,
            "reid_dim": self.reid_dim,
            "temperature": self.temperature,
            "fovea_scale": self.fovea_scale,
            "surround_scale": self.surround_scale,
        }
        if self.use_v5_heads:
            out["use_v5_heads"] = True
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuadTrackModelConfig:
        return cls(
            base_ch=int(d.get("base_ch", 24)),
            surround_ch=int(d.get("surround_ch", 16)),
            reid_dim=int(d.get("reid_dim", 48)),
            temperature=float(d.get("temperature", 1.0)),
            fovea_scale=float(d.get("fovea_scale", FOVEA_BASE_SCALE)),
            surround_scale=float(d.get("surround_scale", SURROUND_BASE_SCALE)),
            use_v5_heads=bool(d.get("use_v5_heads", False)),
        )


def photometric_channels(x: torch.Tensor) -> torch.Tensor:
    """Local equivariant contrast alongside raw RGB/luminance; no position channels.

    Equivariance holds where translated receptive fields have the same support.
    """
    mean = F.avg_pool2d(x, 5, stride=1, padding=2)
    var = (F.avg_pool2d(x*x, 5, stride=1, padding=2)-mean*mean).clamp_min(0.0036)
    contrast = (x-mean)/torch.sqrt(var)
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return torch.cat([x, (x*weights).sum(1, keepdim=True), contrast,
                      (contrast*weights).sum(1, keepdim=True)], dim=1)


def _safe_projective_denominator(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, x.clamp_min(1e-8), x.clamp_max(-1e-8))


def project_points_torch(h: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply a batched projective map preserving homogeneous depth sign."""
    with torch.amp.autocast(device_type=h.device.type, enabled=False):
        h, points = h.float(), points.float()
        p = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        q = p @ h.transpose(-1, -2)
        return q[..., :2]/_safe_projective_denominator(q[..., 2:3])


def _unit_square_homography(q: torch.Tensor) -> torch.Tensor:
    x0, y0 = q[..., 0, 0], q[..., 0, 1]
    x1, y1 = q[..., 1, 0], q[..., 1, 1]
    x2, y2 = q[..., 2, 0], q[..., 2, 1]
    x3, y3 = q[..., 3, 0], q[..., 3, 1]
    dx1, dx2, dx3 = x1-x2, x3-x2, x0-x1+x2-x3
    dy1, dy2, dy3 = y1-y2, y3-y2, y0-y1+y2-y3
    den = _safe_projective_denominator(dx1*dy2-dx2*dy1)
    g, h = (dx3*dy2-dx2*dy3)/den, (dx1*dy3-dx3*dy1)/den
    return torch.stack([x1-x0+g*x1, x3-x0+h*x3, x0,
                        y1-y0+g*y1, y3-y0+h*y3, y0,
                        g, h, torch.ones_like(g)], -1).reshape(q.shape[:-2]+(3, 3))


def solve_4pt_homography_torch(canon_corners: torch.Tensor, img_corners: torch.Tensor) -> torch.Tensor:
    """Exact batched four-point DLT by analytic elimination, ONNX opset 17.

    Maps arbitrary nondegenerate ordered quads using an adjugate in PGL(3),
    avoiding inverse/SVD/linear-solve export operators. Reject degenerate inputs.
    """
    with torch.amp.autocast(device_type=canon_corners.device.type, enabled=False):
        src = _unit_square_homography(canon_corners.float())
        dst = _unit_square_homography(img_corners.float())
        a, b, c = src[..., 0, 0], src[..., 0, 1], src[..., 0, 2]
        d, e, f = src[..., 1, 0], src[..., 1, 1], src[..., 1, 2]
        g, h, i = src[..., 2, 0], src[..., 2, 1], src[..., 2, 2]
        adj = torch.stack([e*i-f*h, c*h-b*i, b*f-c*e, f*g-d*i, a*i-c*g, c*d-a*f,
                           d*h-e*g, b*g-a*h, a*e-b*d], -1).reshape(src.shape)
        result = dst @ adj
        return result/_safe_projective_denominator(result[..., 2:3, 2:3])


def valid_quad_torch(corners: torch.Tensor) -> torch.Tensor:
    edges = torch.roll(corners, -1, dims=-2)-corners
    nxt = torch.roll(edges, -1, dims=-2)
    cross = edges[..., 0]*nxt[..., 1]-edges[..., 1]*nxt[..., 0]
    return (cross > 1e-8).all(-1) & torch.isfinite(corners).all(-1).all(-1)


def anchor_free_mode_init(cell_logits: torch.Tensor, relative_corners: torch.Tensor,
                          temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-map cell-weighted relative quads, kept separate for each mode.

    Inputs (B,M,H,W), (B,M,4,2,H,W). Unbounded ROI displacements allow
    amodal corners beyond any proposal edge. Cell grids enter only the readout.
    """
    with torch.amp.autocast(device_type=cell_logits.device.type, enabled=False):
        cell_logits, relative_corners = cell_logits.float(), relative_corners.float()
        b, m, h, w = cell_logits.shape
        yy, xx = torch.meshgrid((torch.arange(h, device=cell_logits.device, dtype=torch.float32)+.5)/h,
                                (torch.arange(w, device=cell_logits.device, dtype=torch.float32)+.5)/w, indexing="ij")
        grid = torch.stack([xx, yy], 0).reshape(1, 1, 1, 2, h*w)
        flat = cell_logits.reshape(b, m, h*w)/temperature
        logp = F.log_softmax(flat, -1)
        probs = torch.softmax(flat, -1).reshape(b, m, h, w)
        # Reduce along chart axes before normalizing the first moment. Shorter
        # reductions limit eager/ONNX roundoff without changing the measure.
        mass = probs.sum(-1).sum(-1)
        weighted = (relative_corners + grid.reshape(1, 1, 1, 2, h, w)) * probs[:, :, None, None]
        corners = weighted.sum(-1).sum(-1) / mass[:, :, None, None]
        log_cells = torch.ones_like(flat[0, 0]).sum().log()
        return torch.logsumexp(flat, -1)-log_cells, corners, logp


def plate_null_log_likelihood_ratio(logit: torch.Tensor, cal_temp: torch.Tensor | float,
                                    cal_bias: torch.Tensor | float,
                                    training_prior: torch.Tensor | float) -> torch.Tensor:
    prior = (training_prior.to(dtype=logit.dtype, device=logit.device) if isinstance(training_prior, torch.Tensor)
             else logit.new_tensor(training_prior))
    return logit/cal_temp+cal_bias-torch.log(prior/(1-prior))


def plate_null_cost_decision(p: torch.Tensor, c_fp: float = 1.0, c_fn: float = 1.0,
                             c_abstain: float = .25) -> torch.Tensor:
    """0=plate, 1=null, 2=abstain by minimum expected cost."""
    if min(c_fp, c_fn, c_abstain) < 0 or not all(math.isfinite(v) for v in (c_fp, c_fn, c_abstain)):
        raise ValueError("Decision costs must be finite and nonnegative")
    return torch.stack([c_fp*(1-p), c_fn*p, torch.full_like(p, c_abstain)], -1).argmin(-1)


@dataclass(frozen=True)
class ContainmentCropWindow:
    """Sensor window plus the unmodified posterior coverage requirement."""
    window_xyxy: tuple[float, float, float, float]
    required_window_xyxy: tuple[float, float, float, float]
    missing_support: bool
    alpha_crop: float

    @property
    def window_xywh(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.window_xyxy
        return x0, y0, x1-x0, y1-y0


def select_containment_crop_window(mu_corners_xy: np.ndarray, cov_corners_4x2x2: np.ndarray,
                                    img_hw: tuple[int, int], alpha_crop: float = 1e-3) -> ContainmentCropWindow:
    """Minimal rectangle enclosing symmetric Gaussian union-bound intervals.

    Accepts (4,2) or (M,4,2) means and matching covariances. Encloses all
    mixture modes, each with conditional coverage >=1-alpha, hence also their
    normalized mixture. Minimality is for these sufficient intervals and monotone
    cost. Sensor infeasibility is explicit; the sensor intersection claims no coverage.
    """
    mu, cov = np.asarray(mu_corners_xy, dtype=np.float64), np.asarray(cov_corners_4x2x2, dtype=np.float64)
    if mu.ndim not in (2, 3) or mu.shape[-2:] != (4, 2) or cov.shape != mu.shape[:-1]+(2, 2):
        raise ValueError("Expected (4,2) or (M,4,2) corners with matching 2x2 covariances")
    if not 0 < alpha_crop < 1 or min(img_hw) <= 0 or not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError("Invalid posterior, sensor size, or coverage probability")
    if not np.allclose(cov, np.swapaxes(cov, -1, -2)) or np.any(np.linalg.eigvalsh(cov) <= 0):
        raise ValueError("Corner covariance must be symmetric positive definite")
    k = NormalDist().inv_cdf(1-alpha_crop/16)
    radius = k*np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    lo, hi = (mu-radius).reshape(-1, 2).min(0), (mu+radius).reshape(-1, 2).max(0)
    bounds = np.array([img_hw[1], img_hw[0]], dtype=np.float64)
    missing = bool(np.any(lo < 0) or np.any(hi > bounds))
    return ContainmentCropWindow(tuple(np.concatenate([np.clip(lo, 0, bounds), np.clip(hi, 0, bounds)]).tolist()),
                                 tuple(np.concatenate([lo, hi]).tolist()), missing, alpha_crop)


class PlateQuadTrackNet(nn.Module):
    """Gen-6 conditional projective mixture with coarse-to-fine IC-STN updates.

    Stride-one CNNs contain no position channels, spatial gates, window priors
    or displacement clamp. Geometry is conditional on plate; null geometry is
    empty. Calibration/support buffers require held-out fitting per sampling policy;
    identity initialization is not a claim that an untrained model is calibrated.
    """
    is_dual_track_net: bool = True
    num_modes: int = 2

    def __init__(self, config: QuadTrackModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or QuadTrackModelConfig()
        c, sc = self.config.base_ch, self.config.surround_ch
        self.stem = nn.Sequential(ConvBNAct(8, c, k=3), DWConvBlock(c, k=5), DWConvBlock(c, k=5))
        self.ctx_down2 = DWConvBlock(c, k=5)
        self.ctx_down4 = DWConvBlock(c, k=7)
        self.ctx_down8 = DWConvBlock(c, k=7)
        self.surr_stem = nn.Sequential(ConvBNAct(8, sc, k=3), DWConvBlock(sc, k=3), DWConvBlock(sc, k=5))
        self.ctx_spatial = nn.Sequential(DWConvBlock(c, k=5), DWConvBlock(c, k=5), DWConvBlock(c, k=7), nn.Conv2d(c, c, 1))
        nn.init.zeros_(self.ctx_spatial[3].weight)
        nn.init.zeros_(self.ctx_spatial[3].bias)
        self.cell_head = nn.Conv2d(c, 2, 1)
        self.displacement_head = nn.Conv2d(c, 16, 1)
        nn.init.zeros_(self.displacement_head.weight)
        ratio = max(1.35, self.config.surround_scale / max(1e-3, self.config.fovea_scale))
        chart_scale = torch.tensor([ratio, ratio * (ROI_WIDTH / ROI_HEIGHT) / (SURROUND_WIDTH / SURROUND_HEIGHT)])
        self.register_buffer("coarse_chart_scale", chart_scale)
        local_shapes = torch.tensor([[[-0.25, -0.15], [0.25, -0.15], [0.25, 0.15], [-0.25, 0.15]]]).expand(2, 4, 2)
        with torch.no_grad():
            self.displacement_head.bias.copy_(local_shapes.reshape(-1))
        self.canonical_encoder = nn.Sequential(ConvBNAct(c+1, c, k=3), DWConvBlock(c, k=3), DWConvBlock(c, k=3))
        self.canon_spatial = nn.Conv2d(c, 2 * c, kernel_size=(4, 8), groups=c, bias=True)
        self.canon_mixer = nn.Sequential(nn.Linear(2 * c, 2 * c), nn.SiLU(), nn.Linear(2 * c, 2 * c))
        nn.init.zeros_(self.canon_mixer[2].weight)
        nn.init.zeros_(self.canon_mixer[2].bias)
        self.update_head = nn.Linear(2 * c, 8)
        nn.init.zeros_(self.update_head.weight)
        nn.init.zeros_(self.update_head.bias)
        self.step_update_heads = nn.ModuleList([nn.Linear(2 * c, 8) for _ in range(3)])
        for head in self.step_update_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.pyr_u2_proj = nn.Conv2d(c, c, 1)
        self.pyr_u4_proj = nn.Conv2d(c, c, 1)
        self.surr_pyr_s2_proj = nn.Conv2d(sc, sc, 1)
        self.surr_pyr_s4_proj = nn.Conv2d(sc, sc, 1)
        self.mode_score_head = nn.Linear(2 * c, 1)
        nn.init.zeros_(self.mode_score_head.weight)
        nn.init.zeros_(self.mode_score_head.bias)
        self.mode_cov_head = nn.Linear(2 * c, 12)
        self.visibility_head = nn.Linear(2 * c, 4)
        joint_dim = c + sc + 2 * c + 2 * 12 + 2 + 2 * 4
        self.meta_head = nn.Sequential(nn.Linear(joint_dim, c*2), nn.SiLU(), nn.Linear(c*2, 9))
        self.reid_head = nn.Linear(c+sc, self.config.reid_dim)
        self.layout_grid_head = nn.Conv2d(c, 1, 1)
        self.edge_hm_head = nn.Conv2d(c, 4, 1)
        self.register_buffer("canon_corners", torch.tensor([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]))
        self.register_buffer("default_fovea_quad", torch.tensor([[0.25, 0.35], [0.75, 0.35], [0.75, 0.65], [0.25, 0.65]]))
        yy, xx = torch.meshgrid((torch.arange(20)+.5)*2.5/20-1.25,
                                (torch.arange(40)+.5)*2.5/40-1.25, indexing="ij")
        self.register_buffer("canonical_grid", torch.stack([xx, yy], -1).reshape(1, 800, 2))
        # Each grid retains the same pixel-center convention as the final readout.
        self.register_buffer("step_canonical_grids", torch.stack([
            self.canonical_grid * (extent / 1.25) for extent in (1.65, 1.40, 1.25)]))
        self.register_buffer("cal_temp", torch.ones(4))
        self.register_buffer("cal_bias", torch.zeros(4))
        self.register_buffer("pi_train", torch.full((4,), .5))
        self.register_buffer("support_mean", torch.zeros(joint_dim))
        self.register_buffer("support_scale", torch.ones(joint_dim))
        # Preserve spatial evidence in the wider, concentric observation chart.
        self.surround_ctx_down2 = DWConvBlock(sc, k=5)
        self.surround_ctx_down4 = DWConvBlock(sc, k=7)
        self.surround_ctx_down8 = DWConvBlock(sc, k=7)
        self.ctx_u2_proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.ctx_u2_proj.weight)
        nn.init.zeros_(self.ctx_u2_proj.bias)
        self.surround_ctx_proj = nn.Conv2d(sc, c, 1)
        nn.init.zeros_(self.surround_ctx_proj.weight)
        nn.init.zeros_(self.surround_ctx_proj.bias)
        self.canonical_surround_proj = nn.Sequential(ConvBNAct(sc, c, k=3), DWConvBlock(c, k=3))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _pool_canonical(self, patch: torch.Tensor) -> torch.Tensor:
        raw_pool = F.silu(self.canon_spatial(F.avg_pool2d(patch, kernel_size=5, stride=5)).flatten(1))
        h_mix = F.dropout(self.canon_mixer[1](self.canon_mixer[0](raw_pool)), p=0.08, training=self.training)
        return raw_pool + self.canon_mixer[2](h_mix)

    def _align_surround_context(self, features: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        # Extraction fixes the footprint ratio and both chart aspect ratios.
        # Their common center gives a diagonal linear map in normalized grids.
        hh, ww = target_hw
        ratio = max(1.35, self.config.surround_scale / max(1e-3, self.config.fovea_scale))
        gx = ((torch.arange(ww, device=features.device, dtype=torch.float32) + 0.5) * (2.0 / ww) - 1.0) / ratio
        gy = ((torch.arange(hh, device=features.device, dtype=torch.float32) + 0.5) * (2.0 / hh) - 1.0) * ((SURROUND_WIDTH / SURROUND_HEIGHT) / (ROI_WIDTH / ROI_HEIGHT)) / ratio
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([xx, yy], -1)[None].expand(features.shape[0], -1, -1, -1)
        return F.grid_sample(features.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _align_fovea_context(self, features: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        # Inverse chart transport places the crisp fovea in the full surround field.
        hh, ww = target_hw
        ratio = max(1.35, self.config.surround_scale / max(1e-3, self.config.fovea_scale))
        # Scalar chart constants stay FP32 through ONNX weight quantization.
        scale_y = ratio * (ROI_WIDTH / ROI_HEIGHT) / (SURROUND_WIDTH / SURROUND_HEIGHT)
        gx = ((torch.arange(ww, device=features.device, dtype=torch.float32) + 0.5) * (2.0 / ww) - 1.0) * ratio
        gy = ((torch.arange(hh, device=features.device, dtype=torch.float32) + 0.5) * (2.0 / hh) - 1.0) * scale_y
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([xx, yy], -1)[None].expand(features.shape[0], -1, -1, -1)
        return F.grid_sample(features.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _feature_pyramids(self, fovea: torch.Tensor, surround: torch.Tensor) -> tuple[torch.Tensor, ...]:
        u2 = F.avg_pool2d(self.ctx_down2(fovea), 2)
        u4 = F.avg_pool2d(self.ctx_down4(u2), 2)
        u8 = F.avg_pool2d(self.ctx_down8(u4), 2)
        s2 = F.avg_pool2d(self.surround_ctx_down2(surround), 2)
        s4 = F.avg_pool2d(self.surround_ctx_down4(s2), 2)
        s8 = F.avg_pool2d(self.surround_ctx_down8(s4), 2)
        return u2, u4, u8, s2, s4, s8

    def _context_features(self, fovea: torch.Tensor, surround: torch.Tensor,
                          pyramid: tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
        u2, u4, u8, s2, s4, s8 = self._feature_pyramids(fovea, surround) if pyramid is None else pyramid
        original = 0.5 * (u4 + F.interpolate(u8, size=u4.shape[-2:], mode="nearest"))
        f_fpn = original + self.ctx_u2_proj(F.avg_pool2d(u2, 2))
        chart_hw = original.shape[-2:]
        s_fpn = (s2 + F.interpolate(s4, size=s2.shape[-2:], mode="bilinear", align_corners=False)
                 + F.interpolate(s8, size=s2.shape[-2:], mode="bilinear", align_corners=False))
        wide = F.interpolate(s_fpn, size=chart_hw, mode="bilinear", align_corners=False)
        aligned = self._align_fovea_context(f_fpn, chart_hw)
        support_ctx = self._align_fovea_context(torch.ones_like(f_fpn[:, :1]), chart_hw)
        return aligned + (1.0 - 0.5 * support_ctx) * self.surround_ctx_proj(wide)

    def _canonical_features(self, features: torch.Tensor, corners: torch.Tensor,
                            surround_features: torch.Tensor | None = None,
                            canonical_grid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = solve_4pt_homography_torch(self.canon_corners, corners)
        xy = project_points_torch(h, self.canonical_grid if canonical_grid is None else canonical_grid)
        grid = (2*xy-1).reshape(-1, 20, 40, 2)
        patch = F.grid_sample(features, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        support = F.grid_sample(torch.ones_like(features[:, :1]), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if surround_features is not None:
            # Both crops observe the same plane. Transport canonical samples to
            # the wider chart only where the crisp fovea has no observation.
            surround_xy = 0.5 + (xy - 0.5) / self.coarse_chart_scale
            surround_grid = (2 * surround_xy - 1).reshape(-1, 20, 40, 2)
            surround_patch = F.grid_sample(surround_features, surround_grid, mode="bilinear",
                                           padding_mode="zeros", align_corners=False)
            patch = patch + (1.0 - support) * surround_patch
        return self.canonical_encoder(torch.cat([patch, support], 1)), support, h

    def forward(self, fovea_rois: torch.Tensor, surround_rois: torch.Tensor | None = None,
                sampling_policy: torch.Tensor | None = None, *,
                _selected_mode_override: torch.Tensor | None = None) -> dict[str, Any]:
        b = fovea_rois.shape[0]
        if surround_rois is None:
            surround_rois = F.interpolate(fovea_rois, size=(SURROUND_HEIGHT, SURROUND_WIDTH), mode="bilinear", align_corners=False)
        if sampling_policy is None:
            sampling_policy = torch.zeros_like(fovea_rois[:, 0, 0, 0], dtype=torch.long)
        f = self.stem(photometric_channels(fovea_rois))
        s_features = self.surr_stem(photometric_channels(surround_rois))
        s = s_features.mean((2, 3))
        pyramid = self._feature_pyramids(f, s_features)
        u2, u4, _, s2, s4, _ = pyramid
        f_ctx = self._context_features(f, s_features, pyramid)
        f_ctx = f_ctx + self.ctx_spatial(f_ctx)
        f_fine = f + self.pyr_u2_proj(F.interpolate(u2, size=f.shape[-2:], mode="bilinear", align_corners=False))
        f_coarse = f_fine + self.pyr_u4_proj(F.interpolate(u4, size=f.shape[-2:], mode="bilinear", align_corners=False))
        hh, ww = f_ctx.shape[-2:]
        # Learn offsets in fovea units; only the integral readout uses the wide chart.
        relative_fovea = self.displacement_head(f_ctx).reshape(b, 2, 4, 2, hh, ww).float()
        relative_wide = relative_fovea / self.coarse_chart_scale.reshape(1, 1, 1, 2, 1, 1)
        init_mode_logits, init_corners, cell_logp = anchor_free_mode_init(
            self.cell_head(f_ctx), relative_wide, self.config.temperature)
        init_corners = 0.5 + (init_corners - 0.5) * self.coarse_chart_scale
        corners = init_corners.reshape(-1, 4, 2)
        def expand_modes(features: torch.Tensor) -> torch.Tensor:
            return features[:, None].expand(-1, 2, -1, -1, -1).reshape(
                -1, features.shape[1], features.shape[-2], features.shape[-1])
        mode_fine, mode_coarse = expand_modes(f_fine), expand_modes(f_coarse)
        s_fine_raw = s_features + self.surr_pyr_s2_proj(F.interpolate(s2, size=s_features.shape[-2:], mode="bilinear", align_corners=False))
        s_coarse_raw = s_fine_raw + self.surr_pyr_s4_proj(F.interpolate(s4, size=s_features.shape[-2:], mode="bilinear", align_corners=False))
        surr_fine = self.canonical_surround_proj(s_fine_raw)
        surr_coarse = self.canonical_surround_proj(s_coarse_raw)
        mode_surr_fine, mode_surr_coarse = expand_modes(surr_fine), expand_modes(surr_coarse)
        step_corners = [init_corners]
        for k, head in enumerate(self.step_update_heads):
            q_valid = valid_quad_torch(corners)
            warp_corners = torch.where(q_valid[:, None, None], corners.detach(), self.default_fovea_quad)
            patch, support, h = self._canonical_features(
                mode_coarse if k < 2 else mode_fine, warp_corners,
                mode_surr_coarse if k < 2 else mode_surr_fine, self.step_canonical_grids[k])
            pooled_k = self._pool_canonical(patch)
            delta = (self.update_head(pooled_k) + head(pooled_k)).reshape(-1, 4, 2)
            cand_canon = self.canon_corners + delta
            canon_valid = valid_quad_torch(cand_canon)
            with torch.amp.autocast(device_type=h.device.type, enabled=False):
                p_hom = torch.cat([cand_canon.float(), torch.ones_like(cand_canon[..., :1], dtype=torch.float32)], dim=-1) @ h.float().transpose(-1, -2)
                w0 = h.float()[..., 2:3, 2]
                # Preserve homogeneous depth sign and margin without a spatial clamp.
                cheiral = ((p_hom[..., 2] * w0) > 0.15 * w0.square()).all(dim=-1)
            updated = project_points_torch(h, cand_canon)
            step_ok = q_valid & canon_valid & cheiral & valid_quad_torch(updated)
            corners = torch.where(step_ok[:, None, None], updated, corners)
            step_corners.append(corners.reshape(b, 2, 4, 2))
        final_q_valid = valid_quad_torch(corners)
        final_warp_corners = torch.where(final_q_valid[:, None, None], corners.detach(), self.default_fovea_quad)
        patch, support, _ = self._canonical_features(mode_fine, final_warp_corners, mode_surr_fine)
        h_out = solve_4pt_homography_torch(
            self.canon_corners, torch.where(final_q_valid[:, None, None], corners, self.default_fovea_quad)
        )
        pooled = self._pool_canonical(patch)
        mode_logits = init_mode_logits + self.mode_score_head(pooled.detach()).reshape(b, 2)
        raw = self.mode_cov_head(pooled.detach()).reshape(b, 2, 4, 3).float()
        chol = torch.stack([F.softplus(raw[..., 0] - 3.0) + 2e-3, raw[..., 1].clamp(-0.25, 0.25), F.softplus(raw[..., 2] - 3.0) + 2e-3], -1)
        mode_cov = chol3_to_cov2x2(chol)
        vis = self.visibility_head(pooled).reshape(b, 2, 4)
        mode_corners = corners.reshape(b, 2, 4, 2)
        valid = valid_quad_torch(mode_corners)
        posterior_logits = torch.where(valid, mode_logits, torch.full_like(mode_logits, -1e9))
        weights = posterior_logits.softmax(-1)*valid.to(mode_logits.dtype)
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
        selected = posterior_logits.argmax(-1) if _selected_mode_override is None else _selected_mode_override.to(device=posterior_logits.device, dtype=torch.long)
        def pick(x: torch.Tensor) -> torch.Tensor:
            index = selected.reshape((b, 1)+(1,)*(x.ndim-2)).expand((b, 1)+x.shape[2:])
            return x.gather(1, index).squeeze(1)
        chosen = pick(mode_corners)
        residual = mode_corners-chosen[:, None]
        # Mixture second moment about the selected mode, without averaging quads.
        mixture_cov = (weights[:, :, None, None, None]*(mode_cov+residual[..., :, None]*residual[..., None, :])).sum(1)
        lx = mixture_cov[..., 0, 0].clamp_min(1e-12).sqrt()
        lyx = mixture_cov[..., 1, 0]/lx
        lyy = (mixture_cov[..., 1, 1]-lyx.square()).clamp_min(1e-12).sqrt()
        mixture_chol = torch.stack([lx, lyx, lyy], -1)
        canon_sel_feat = (weights[:, :, None] * pooled.reshape(b, 2, -1)).sum(1)
        joint = torch.cat([f_ctx.mean((2, 3)), s, canon_sel_feat, chol.detach().reshape(b, -1), weights.detach(), vis.detach().reshape(b, -1)], -1)
        if self.training:
            meta_hidden = F.dropout(self.meta_head[1](self.meta_head[0](joint)), p=0.10, training=self.training)
            meta = self.meta_head[2](meta_hidden)
        else:
            # Retain the Sequential export scope and its terminal Gemm names.
            meta = self.meta_head(joint)
        llr = plate_null_log_likelihood_ratio(meta[:, 0], self.cal_temp[sampling_policy],
                                               self.cal_bias[sampling_policy], self.pi_train[sampling_policy])
        out = {
            "corners_norm": chosen, "mode_logits": mode_logits, "mode_weights": weights,
            "mode_init_corners": init_corners, "mode_corners": mode_corners, "mode_valid": valid,
            "geometry_valid": valid.any(-1), "selected_mode": selected, "ic_corners": torch.stack(step_corners, 1),
            "mode_cov_chol": chol, "mode_corner_cov": mode_cov, "corner_cov_chol": mixture_chol, "corner_cov": mixture_cov,
            "corner_vis_logits": pick(vis), "homography_3x3": pick(h_out.reshape(b, 2, 3, 3)),
            "blur_vec_norm": meta[:, 7:9], "is_plate_logit": meta[:, 0], "plate_null_llr": llr,
            "side_logits": meta[:, 1:4], "quality_logit": meta[:, 4], "topology_logit": meta[:, 5], "is_large_logit": meta[:, 6],
            "reid_emb": F.normalize(self.reid_head(torch.cat([f.mean((2, 3)), s], -1)), dim=-1),
            "support_nonconformity": ((joint-self.support_mean)/self.support_scale.clamp_min(1e-6)).square().mean(-1).sqrt(),
            "canonical_support": pick(support.reshape(b, 2, 1, 20, 40)),
            "corner_log_probs": cell_logp.mean(1, keepdim=True).expand(-1, 4, -1), "grid_hw": (hh, ww),
        }
        if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
            out["joint_features"] = joint
            out["mode_cell_log_probs"] = cell_logp
            out["coarse_chart_scale"] = self.coarse_chart_scale
            selected_patch = pick(patch.reshape(b, 2, -1, 20, 40))
            out["layout_logits"] = self.layout_grid_head(F.adaptive_avg_pool2d(selected_patch[:, :, 2:18, 4:36], (4, 8)))
            out["edge_hm_logits"] = self.edge_hm_head(f)
        return out


STAGE2_INT8_MAX_BYTES = 1_572_864
STAGE2_OUTPUT_NAMES = ("corners_norm", "mode_logits", "mode_weights", "mode_init_corners", "mode_corners",
                       "mode_corner_cov", "corner_cov_chol", "corner_cov",
                       "corner_vis_logits", "homography_3x3", "blur_vec_norm", "reid_emb",
                       "support_nonconformity", "plate_null_llr", "geometry_valid")


class _Stage2ExportWrapper(nn.Module):
    def __init__(self, model: PlateQuadTrackNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, fovea_rois: torch.Tensor, surround_rois: torch.Tensor,
                sampling_policy: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model(fovea_rois, surround_rois, sampling_policy)
        return tuple(out[name] for name in STAGE2_OUTPUT_NAMES)


def fuse_conv_bn_inplace(module: nn.Module) -> nn.Module:
    """Fold eval Conv+BN pairs in float64, rounding the fused parameters once.

    Includes ConvBNAct and adjacent Sequential pairs (notably DWConvBlock.pw2).
    Export and eager parity must use precisely the same folded parameters.
    """
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    def fused(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        return fuse_conv_bn_eval(copy.deepcopy(conv).double(), copy.deepcopy(bn).double()).float()

    module.eval()
    for child in module.modules():
        if isinstance(getattr(child, "conv", None), nn.Conv2d) and isinstance(getattr(child, "bn", None), nn.BatchNorm2d):
            child.conv = fused(child.conv, child.bn)
            child.bn = nn.Identity()
        if isinstance(child, nn.Sequential):
            for index in range(len(child) - 1):
                if isinstance(child[index], nn.Conv2d) and isinstance(child[index + 1], nn.BatchNorm2d):
                    child[index] = fused(child[index], child[index + 1])
                    child[index + 1] = nn.Identity()
    return module


def export_stage2_onnx(model: PlateQuadTrackNet, out_dir: str | Path) -> dict[str, Any]:
    """Export the full Gen-6 contract and enforce the serialized INT8 allocation."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    fp32, int8 = od / "stage2.onnx", od / "stage2.int8.onnx"
    net = fuse_conv_bn_inplace(copy.deepcopy(model).cpu().eval())
    dummy = (torch.zeros(1, 3, ROI_HEIGHT, ROI_WIDTH), torch.zeros(1, 3, SURROUND_HEIGHT, SURROUND_WIDTH), torch.zeros(1, dtype=torch.long))
    inputs = ["fovea_rois", "surround_rois", "sampling_policy"]
    torch.onnx.export(_Stage2ExportWrapper(net), dummy, str(fp32), input_names=inputs,
                      output_names=list(STAGE2_OUTPUT_NAMES), opset_version=17, dynamo=False,
                      dynamic_axes={name: {0: "batch"} for name in inputs+list(STAGE2_OUTPUT_NAMES)})
    onnx.checker.check_model(str(fp32))
    graph = onnx.load(str(fp32))
    # The shared canonical encoder can export repeated weights through Identity
    # aliases. Resolve initializer aliases before ORT's ConvInteger quantizer.
    initializers = {item.name for item in graph.graph.initializer}
    aliases = {}
    retained = []
    graph_outputs = {item.name for item in graph.graph.output}
    for node in graph.graph.node:
        for j, name in enumerate(node.input):
            node.input[j] = aliases.get(name, name)
        if node.op_type == "Identity" and node.input[0] in initializers and node.output[0] not in graph_outputs:
            aliases[node.output[0]] = node.input[0]
        else:
            retained.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(retained)
    # ORT's Gemm rewrite transposes shared initializers in place once per
    # consumer. Canonicalize shared Linear weights once to avoid a double transpose.
    tensors = {item.name: item for item in graph.graph.initializer}
    transposed = {}
    for node in graph.graph.node:
        if node.op_type == "Gemm":
            for attr in node.attribute:
                if attr.name == "transB" and attr.i == 1 and node.input[1] in tensors:
                    name = node.input[1]
                    if name not in transposed:
                        tname = name+"_transposed"
                        graph.graph.initializer.append(onnx.numpy_helper.from_array(
                            onnx.numpy_helper.to_array(tensors[name]).T.copy(), tname))
                        transposed[name] = tname
                    node.input[1] = transposed[name]
                    attr.i = 0
    used = {name for node in graph.graph.node for name in node.input}
    kept = [item for item in graph.graph.initializer if item.name in used]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(kept)
    onnx.save(graph, str(fp32))
    terminal_gemm_nodes = [n.name for n in graph.graph.node if n.op_type == "Gemm"
                           and (n.name.startswith("/model/meta_head/")
                                or n.name.startswith("/model/reid_head/"))]
    if len(terminal_gemm_nodes) != 3:
        raise RuntimeError(f"Expected two metadata and one identity Gemm nodes: {terminal_gemm_nodes}")
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8, per_channel=True,
                     # ORT rewrites Gemm to named MatMul before node selection.
                     nodes_to_quantize=[name + "_MatMul" for name in terminal_gemm_nodes],
                     extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT, "MatMulConstBOnly": True})
    graph = onnx.load(str(int8))
    del graph.graph.value_info[:]
    onnx.save(graph, str(int8))
    onnx.checker.check_model(str(int8))
    integer_nodes = [node for node in graph.graph.node if node.op_type == "MatMulInteger"]
    if len(integer_nodes) != 3 or any(not node.name.startswith(("/model/meta_head/", "/model/reid_head/"))
                                      for node in integer_nodes):
        raise RuntimeError("Stage-2 must quantize exactly the three terminal metadata/identity projections")
    size = int8.stat().st_size
    if size > STAGE2_INT8_MAX_BYTES:
        int8.unlink()
        raise ValueError(f"Stage-2 INT8 allocation exceeded: {size} > {STAGE2_INT8_MAX_BYTES} bytes")
    return {"fp32_path": str(fp32), "int8_path": str(int8), "int8_bytes": size}


def chol3_to_cov2x2(chol: torch.Tensor) -> torch.Tensor:
    """Convert `(..., 3)` `[l_xx, l_yx, l_yy_orth]` into SPD `(..., 2, 2)` covariance `L @ L^T`."""
    l_xx = chol[..., 0]
    l_yx = chol[..., 1]
    l_yy = chol[..., 2]
    c00 = l_xx * l_xx
    c01 = l_xx * l_yx
    c11 = l_yx * l_yx + l_yy * l_yy
    row0 = torch.stack([c00, c01], dim=-1)
    row1 = torch.stack([c01, c11], dim=-1)
    return torch.stack([row0, row1], dim=-2)


class PlateQuadTrackLoss(nn.Module):
    """Proper plate/null BCE and conditional geometric mixture NLL.

    Visibility/blur/Re-ID and renderer stroke/edge targets are training-only.
    Geometric targets are never excluded or clamped to the proposal window.
    """
    def __init__(self, sigma_cells: float = 1.15) -> None:
        super().__init__()
        self.sigma_cells = sigma_cells

    def forward(self, out: dict[str, torch.Tensor], is_plate_gt: torch.Tensor,
                corners_norm_gt: torch.Tensor, plate_side_gt: torch.Tensor,
                quality_gt: torch.Tensor, *, topology_gt: torch.Tensor | None = None,
                layout_grid_gt: torch.Tensor | None = None, is_large_gt: torch.Tensor | None = None,
                blur_vec_norm_gt: torch.Tensor | None = None, reid_instance_ids: torch.Tensor | None = None,
                corner_visibility_gt: torch.Tensor | None = None,
                aux_warmup_scale: float = 1.0) -> dict[str, torch.Tensor]:
        logit, target = out["is_plate_logit"].float(), is_plate_gt.float()
        zero = logit.sum()*0
        loss_cls = F.binary_cross_entropy_with_logits(logit, target)
        loss_topo = F.binary_cross_entropy_with_logits(out["topology_logit"], target if topology_gt is None else topology_gt.float())
        loss_layout = zero
        if "layout_logits" in out:
            loss_layout = (out["layout_logits"].sum()*0 if layout_grid_gt is None else
                           F.binary_cross_entropy_with_logits(out["layout_logits"], layout_grid_gt.float()))
        pos = target > .5
        loss_coord = loss_cov = loss_blur = loss_vis = loss_edge = loss_side = loss_qual = loss_large = euclid = zero
        loss_center = zero
        if pos.any():
            truth = corners_norm_gt[pos].float()
            means = out["mode_corners"][pos].float()
            chol = out["mode_cov_chol"][pos].float()
            # Calibrate covariance without variance-weighted localization gradients.
            error = truth[:, None] - means.detach()
            z0 = error[..., 0]/chol[..., 0]
            z1 = (error[..., 1]-chol[..., 1]*z0)/chol[..., 2]
            log_density = (-.5*(z0*z0+z1*z1)-torch.log(chol[..., 0])-torch.log(chol[..., 2])-math.log(2*math.pi)).sum(-1)
            joint = F.log_softmax(out["mode_logits"][pos], -1)+log_density
            loss_nll = -torch.logsumexp(joint, -1).mean()
            ic = out["ic_corners"][pos].float()
            tgt_m = truth[:, None].expand_as(means)
            # Localization assignments depend only on geometry. Learned covariance
            # still receives proper mixture-NLL supervision, but cannot buy a
            # larger localization responsibility by inflating uncertainty.
            distance = torch.linalg.vector_norm(means - tgt_m, dim=-1).mean(-1)
            precision = (distance.detach() + 0.018).reciprocal()
            responsibilities = precision / precision.sum(-1, keepdim=True)
            ic_edges = torch.roll(ic, -1, dims=-2) - ic
            ic_next = torch.roll(ic_edges, -1, dims=-2)
            ic_cross = ic_edges[..., 0] * ic_next[..., 1] - ic_edges[..., 1] * ic_next[..., 0]
            loss_convex = 6.0 * (F.relu(1e-3 - ic_cross).mean(dim=(1, 3)) * responsibilities).sum(-1).mean()
            step_err = torch.zeros_like(distance)
            for k, weight in enumerate((0.25, 0.40, 0.70, 1.00)):
                wing = _wing_loss(ic[:, k], tgt_m, w=0.04, eps=0.006, reduction="none").mean((-1, -2))
                l2 = torch.linalg.vector_norm(ic[:, k] - tgt_m, dim=-1).mean(-1)
                step_err = step_err + weight * (wing + l2)
            target_probs = (-distance.detach() / 0.008).softmax(dim=-1)
            loss_mode_sel = -(target_probs * F.log_softmax(out["mode_logits"][pos].float(), dim=-1)).sum(dim=-1).mean()
            loss_coord = (0.02 * loss_nll + 24.0 * (step_err * responsibilities).sum(-1).mean()
                          + 8.0 * step_err.min(-1).values.mean() + 0.60 * loss_mode_sel + loss_convex)
            if "mode_cell_log_probs" in out:
                # A categorical observation locates the center independently of
                # the offset field. Bilinear labels preserve its first moment
                # between cell centers without an arbitrary Gaussian bandwidth.
                center = 0.5 + (truth.mean(-2) - 0.5) / out["coarse_chart_scale"]
                observed = ((center >= 0.0) & (center <= 1.0)).all(-1)
                if observed.any():
                    hh, ww = out["grid_hw"]
                    cell_logp = out["mode_cell_log_probs"][pos][observed].reshape(-1, 2, hh, ww).float()
                    grid_center = (2.0 * center[observed] - 1.0)[:, None, None]
                    sampled_center = F.grid_sample(cell_logp, grid_center, mode="bilinear",
                                                   padding_mode="border", align_corners=False).reshape(-1, 2)
                    loss_center = -(sampled_center * responsibilities[observed]).sum(-1).mean()
            edges = torch.roll(means, -1, dims=-2) - means
            gt_edges = (torch.roll(truth, -1, dims=-2) - truth)[:, None].expand_as(edges)
            loss_edge = (F.smooth_l1_loss(edges, gt_edges, reduction="none").mean((-1, -2)) * responsibilities).sum(-1).mean()
            euclid = torch.linalg.vector_norm(out["corners_norm"][pos]-truth, dim=-1).mean()
            loss_side = F.cross_entropy(out["side_logits"][pos], plate_side_gt[pos])
            loss_qual = F.binary_cross_entropy_with_logits(out["quality_logit"][pos], quality_gt[pos].float())
            if is_large_gt is not None:
                loss_large = F.binary_cross_entropy_with_logits(out["is_large_logit"][pos], is_large_gt[pos].float())
            if corner_visibility_gt is not None:
                loss_vis = F.binary_cross_entropy_with_logits(out["corner_vis_logits"][pos], corner_visibility_gt[pos].float())
            if blur_vec_norm_gt is not None:
                pred, gt = out["blur_vec_norm"][pos], blur_vec_norm_gt[pos]
                loss_blur = torch.minimum((pred-gt).square().sum(-1), (pred+gt).square().sum(-1)).mean()
        # Renderer-derived edge heatmaps remain auxiliary, with no inference fusion.
        loss_hm = zero
        if "edge_hm_logits" in out:
            edge_logits = out["edge_hm_logits"]
            loss_hm = edge_logits.sum()*0
            if pos.any():
                hh, ww = edge_logits.shape[-2:]
                yy, xx = torch.meshgrid((torch.arange(hh, device=logit.device)+.5)/hh,
                                        (torch.arange(ww, device=logit.device)+.5)/ww, indexing="ij")
                grid = torch.stack([xx, yy], -1).reshape(1, 1, hh*ww, 2)
                q = corners_norm_gt[pos].float()
                start = q[:, :, None]
                vec = (torch.roll(q, -1, dims=1)-q)[:, :, None]
                t = ((grid-start)*vec).sum(-1, keepdim=True)/vec.square().sum(-1, keepdim=True).clamp_min(1e-8)
                dist = ((grid-start-t.clamp(0, 1)*vec)*logit.new_tensor([ww, hh])).square().sum(-1)
                edge_target = torch.exp(-dist/(2*self.sigma_cells**2)).reshape(-1, 4, hh, ww)
                loss_hm = F.binary_cross_entropy_with_logits(edge_logits[pos], edge_target)
        loss_reid = zero
        if reid_instance_ids is not None:
            valid = (reid_instance_ids > 0) & pos
            emb, ids = out["reid_emb"][valid], reid_instance_ids[valid]
            if len(emb) > 1:
                sim = emb @ emb.T
                same = ids[:, None] == ids[None]
                mask = ~torch.eye(len(emb), dtype=torch.bool, device=emb.device)
                pair_loss = torch.where(same, 1-sim, F.relu(sim-.2))
                loss_reid = pair_loss[mask].mean()
        aux = float(max(0, min(1, aux_warmup_scale)))
        total = (1.5 * loss_cls + loss_coord + 0.8 * loss_center
                 + aux * (loss_topo + loss_edge + loss_blur + loss_vis + loss_reid
                          + 0.30 * (loss_layout + loss_hm + loss_side + loss_qual + loss_large)))
        return {"loss": total, "loss_cls": loss_cls.detach(), "loss_topo": loss_topo.detach(),
                "loss_layout": loss_layout.detach(), "loss_coord": loss_coord.detach(), "loss_cov": loss_cov.detach(),
                "loss_center": loss_center.detach(),
                "loss_blur": loss_blur.detach(), "loss_reid": loss_reid.detach(), "loss_visibility": loss_vis.detach(),
                "mean_corner_err_norm": euclid.detach()}


def _estimate_roi_surround_rgb(
    frame_rgb: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    rx0: float,
    ry0: float,
    rw: float,
    rh: float,
) -> tuple[int, int, int]:
    """Compute neutral constant border fill avoiding bright plate-rim `BORDER_REPLICATE` streaks."""
    h_f, w_f = frame_rgb.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    # Tight pre-cropped plate image occupying most of frame: use neutral dark matte bumper surround
    if (x1 - x0) >= 0.72 * w_f or (y1 - y0) >= 0.72 * h_f:
        return (36, 40, 46)
    if rx0 >= 0.0 and ry0 >= 0.0 and (rx0 + rw) <= w_f and (ry0 + rh) <= h_f:
        return (40, 44, 50)

    vx0 = max(0, min(w_f - 1, int(math.floor(rx0))))
    vy0 = max(0, min(h_f - 1, int(math.floor(ry0))))
    vx1 = max(vx0 + 1, min(w_f, int(math.ceil(rx0 + rw))))
    vy1 = max(vy0 + 1, min(h_f, int(math.ceil(ry0 + rh))))
    sub = frame_rgb[vy0:vy1, vx0:vx1]
    if sub.size == 0:
        return (40, 44, 50)

    strips: list[np.ndarray] = []
    if vy1 - vy0 >= 4:
        strips.append(sub[:2, :].reshape(-1, 3))
        strips.append(sub[-2:, :].reshape(-1, 3))
    if vx1 - vx0 >= 4:
        strips.append(sub[:, :2].reshape(-1, 3))
        strips.append(sub[:, -2:].reshape(-1, 3))
    if not strips:
        med = np.median(sub.reshape(-1, 3), axis=0)
    else:
        med = np.median(np.concatenate(strips, axis=0), axis=0)
    # Clamp luminance/channels so bright white sky or plate borders never create fake rectangular edges
    return (
        int(np.clip(round(float(med[0])), 24, 136)),
        int(np.clip(round(float(med[1])), 24, 136)),
        int(np.clip(round(float(med[2])), 24, 136)),
    )


def _warp_roi_window_exact(
    frame_rgb: np.ndarray,
    rx0: float,
    ry0: float,
    rw: float,
    rh: float,
    out_hw: tuple[int, int],
    border_val: tuple[int, int, int],
) -> np.ndarray:
    """Pixel-center-exact affine ROI extraction (`u_norm = (j_d + 0.5) / W_out -> x_s = rx0 + u_norm * rw`).

    For every supersampling factor `ss in {1, 2, 4}`, setting `tx = -rx0 * sx * ss - 0.5` maps
    the normalized midpoint `u_norm = 0.5` (`x_s = rx0 + 0.5 * rw`) identically to `(W_out - 1) / 2`
    with `0.000000 px` sub-pixel phase shift across `INTER_CUBIC`, `INTER_LINEAR`, and `INTER_AREA`.
    """
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    sx = float(out_w) / max(1e-6, float(rw))
    sy = float(out_h) / max(1e-6, float(rh))

    if sx < 0.85:
        ss = 4 if sx < 0.42 else 2
        M_ss = np.array(
            [
                [sx * ss, 0.0, -rx0 * sx * ss - 0.5],
                [0.0, sy * ss, -ry0 * sy * ss - 0.5],
            ],
            dtype=np.float32,
        )
        roi_hi = cv2.warpAffine(
            frame_rgb,
            M_ss,
            (out_w * ss, out_h * ss),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )
        return cv2.resize(roi_hi, (out_w, out_h), interpolation=cv2.INTER_AREA)

    M = np.array(
        [
            [sx, 0.0, -rx0 * sx - 0.5],
            [0.0, sy, -ry0 * sy - 0.5],
        ],
        dtype=np.float32,
    )
    interp = cv2.INTER_CUBIC if sx > 1.18 else cv2.INTER_LINEAR
    return cv2.warpAffine(
        frame_rgb,
        M,
        (out_w, out_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )


def extract_expanded_plate_roi(
    frame_rgb: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | np.ndarray,
    expand_scale: float = 1.85,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Extract an aspect-locked `(96, 160)` pixel-center-exact expanded context patch around `bbox_xyxy`.

    Uses `cv2.warpAffine` with neutral clamped `BORDER_CONSTANT` surround (preventing bright
    `BORDER_REPLICATE` streaks on tight cutouts & image edges) and true supersampled `INTER_AREA`
    anti-aliasing when minifying high-resolution frames.
    """
    b0, b1, b2, b3 = [float(v) for v in bbox_xyxy]
    bw = max(6.0, b2 - b0)
    bh = max(4.0, b3 - b1)
    cx = 0.5 * (b0 + b2)
    cy = 0.5 * (b1 + b3)

    target_aspect = float(ROI_WIDTH) / float(ROI_HEIGHT)  # 1.6667
    rw = max(bw * expand_scale, bh * expand_scale * target_aspect, 20.0)
    rh = rw / target_aspect
    rx0 = cx - 0.5 * rw
    ry0 = cy - 0.5 * rh

    border_val = _estimate_roi_surround_rgb(frame_rgb, (b0, b1, b2, b3), rx0, ry0, rw, rh)
    roi = _warp_roi_window_exact(frame_rgb, rx0, ry0, rw, rh, (ROI_HEIGHT, ROI_WIDTH), border_val)
    return roi, (rx0, ry0, rw, rh)


def concentric_roi_windows(
    bbox_xyxy: Any, gamma: float = 1.0,
    fovea_scale: float = FOVEA_BASE_SCALE, surround_scale: float = SURROUND_BASE_SCALE,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return the exact sensor windows used by concentric extraction."""
    b0, b1, b2, b3 = [float(v) for v in bbox_xyxy]
    bw = max(5.0, b2 - b0)
    bh = max(3.0, b3 - b1)
    cx = 0.5 * (b0 + b2)
    cy = 0.5 * (b1 + b3)
    g = max(1.0, float(gamma))

    rho_f = float(ROI_WIDTH) / float(ROI_HEIGHT)             # 160 / 96 = 1.6667
    rho_s = float(SURROUND_WIDTH) / float(SURROUND_HEIGHT)   # 96 / 64 = 1.5000

    eff_f = max(1.05, float(fovea_scale)) * g
    rw_f = max(bw * eff_f, bh * eff_f * rho_f, 18.0 * g)
    rh_f = rw_f / rho_f
    rx0_f = cx - 0.5 * rw_f
    ry0_f = cy - 0.5 * rh_f

    scale_ratio = max(1.35, float(surround_scale) / max(1e-3, float(fovea_scale)))
    rw_s = rw_f * scale_ratio
    rh_s = rw_s / rho_s
    rx0_s = cx - 0.5 * rw_s
    ry0_s = cy - 0.5 * rh_s

    return (rx0_f, ry0_f, rw_f, rh_f), (rx0_s, ry0_s, rw_s, rh_s)


def extract_dual_concentric_plate_rois(
    frame_rgb: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | np.ndarray,
    gamma: float = 1.0,
    fovea_scale: float = FOVEA_BASE_SCALE,
    surround_scale: float = SURROUND_BASE_SCALE,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Extract concentric pixel-center-exact Fovea `(96, 160)` and Surround `(64, 96)` crops around `bbox_xyxy`.

    Both patches expand symmetrically about the exact midpoint `(cx, cy)`
    with isotropic pixel scales and constant footprint ratio. No directional
    proposal-repair bias enters extraction.
    """
    b0, b1, b2, b3 = map(float, bbox_xyxy)
    win_f, win_s = concentric_roi_windows(bbox_xyxy, gamma, fovea_scale, surround_scale)
    rx0_f, ry0_f, rw_f, rh_f = win_f
    rx0_s, ry0_s, rw_s, rh_s = win_s

    border_f = _estimate_roi_surround_rgb(frame_rgb, (b0, b1, b2, b3), rx0_f, ry0_f, rw_f, rh_f)
    border_s = _estimate_roi_surround_rgb(frame_rgb, (b0, b1, b2, b3), rx0_s, ry0_s, rw_s, rh_s)
    fovea_u8 = _warp_roi_window_exact(frame_rgb, rx0_f, ry0_f, rw_f, rh_f, (ROI_HEIGHT, ROI_WIDTH), border_f)
    surround_u8 = _warp_roi_window_exact(frame_rgb, rx0_s, ry0_s, rw_s, rh_s, (SURROUND_HEIGHT, SURROUND_WIDTH), border_s)
    return fovea_u8, surround_u8, (rx0_f, ry0_f, rw_f, rh_f), (rx0_s, ry0_s, rw_s, rh_s)


def warp_plate_quad(
    frame_rgb: np.ndarray,
    corners_xy: np.ndarray,
    out_hw: tuple[int, int] = (64, 128),
    margin_ratio: float = 0.0,
    pixel_center_aligned: bool = False,
) -> np.ndarray:
    """Perspective-rectify exact 4-corner quad `[TL, TR, BR, BL]` from `frame_rgb` into `out_hw`.

    - `margin_ratio = 0.0`: Pure exact plate-only cutout (zero background outside the 4 corners).
    - `margin_ratio = 0.05..0.06`: Slight rim padding optimal for `MF-JPPlateOCR`.
    - `pixel_center_aligned = True`: Exact pixel-center mapping `[mx - 0.5, out_w - 0.5 - mx]`
      preserving strict 2.0000:1 physical aspect ratio.
    - Note: OpenCV `cv2.warpPerspective` silently converts `INTER_AREA -> INTER_LINEAR`, so when
      minifying (`max_ratio > 1.15`), we warp at `2x`/`4x` with zero-shift pixel-center alignment
      and apply true `cv2.resize(..., interpolation=cv2.INTER_AREA)` box integration.
    """
    out_h, out_w = out_hw
    pts = np.asarray(corners_xy, dtype=np.float32).reshape(4, 2)

    mx = float(margin_ratio) * out_w
    my = float(margin_ratio) * out_h

    w_top = float(np.linalg.norm(pts[1] - pts[0]))
    h_right = float(np.linalg.norm(pts[2] - pts[1]))
    w_bot = float(np.linalg.norm(pts[2] - pts[3]))
    h_left = float(np.linalg.norm(pts[3] - pts[0]))
    eff_w = max(1.0, float(out_w) * (1.0 - 2.0 * float(margin_ratio)))
    eff_h = max(1.0, float(out_h) * (1.0 - 2.0 * float(margin_ratio)))
    max_ratio = max(0.5 * (w_top + w_bot) / eff_w, 0.5 * (h_left + h_right) / eff_h)

    if max_ratio > 1.15:
        ss = 4 if max_ratio > 2.6 else 2
        if pixel_center_aligned:
            x0_s, x1_s = mx * ss - 0.5, (out_w - mx) * ss - 0.5
            y0_s, y1_s = my * ss - 0.5, (out_h - my) * ss - 0.5
        else:
            shift = 0.5 * float(ss - 1)
            x0_s, x1_s = mx * ss + shift, (out_w - 1.0 - mx) * ss + shift
            y0_s, y1_s = my * ss + shift, (out_h - 1.0 - my) * ss + shift
        dst_s = np.array(
            [
                [x0_s, y0_s],
                [x1_s, y0_s],
                [x1_s, y1_s],
                [x0_s, y1_s],
            ],
            dtype=np.float32,
        )
        H_s = cv2.getPerspectiveTransform(pts, dst_s)
        hi = cv2.warpPerspective(
            frame_rgb,
            H_s,
            (out_w * ss, out_h * ss),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return cv2.resize(hi, (out_w, out_h), interpolation=cv2.INTER_AREA)

    if pixel_center_aligned:
        x0, x1 = mx - 0.5, out_w - 0.5 - mx
        y0, y1 = my - 0.5, out_h - 0.5 - my
    else:
        x0, x1 = mx, out_w - 1.0 - mx
        y0, y1 = my, out_h - 1.0 - my
    dst = np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(pts, dst)
    interp = cv2.INTER_CUBIC if max_ratio < 0.85 else cv2.INTER_LINEAR
    return cv2.warpPerspective(
        frame_rgb,
        H,
        (out_w, out_h),
        flags=interp,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_plate_canonical(
    frame_rgb: np.ndarray,
    corners_xy: np.ndarray,
    out_hw: tuple[int, int] = (80, 160),
    margin_ratio: float = 0.05,
) -> np.ndarray:
    """Canonical pixel-center-aligned homography crop for `MF-JPPlateOCR v2`."""
    return warp_plate_quad(
        frame_rgb,
        corners_xy,
        out_hw=out_hw,
        margin_ratio=margin_ratio,
        pixel_center_aligned=True,
    )


def extract_quad_geometry_meta(
    corners_xy: np.ndarray | None,
    crop_rgb: np.ndarray,
    prev_corners_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Compute deterministic 10-D geometric & photometric metadata vector `g_t` (`float32[10]`).

    Indices:
      0: log2 horizontal pixel scale `clamp(0.5 * log2(w_mean / 64), -1, 1)`
      1: log2 vertical pixel scale `clamp(0.5 * log2(h_mean / 32), -1, 1)`
      2: anisotropic foreshortening `clamp(w_mean / (2 * h_mean), 0.25, 2.0) - 1.0` (<0 steep yaw, >0 steep pitch)
      3: signed yaw taper `tanh(2.5 * ln((h_left + eps) / (h_right + eps)))`
      4: signed pitch taper `tanh(2.5 * ln((w_bot + eps) / (w_top + eps)))`
      5: normalized inter-frame corner delta `clamp(||q_t - q_{t-1}||_F / (2 * w_mean), 0, 1)`
      6: log Laplacian sharpness `clamp(0.25 * ln(1 + 100 * Var(Lap(Y))), 0, 1)`
      7: mean plate luminance `mu_Y` in `[0, 1]`
      8: top-15% pixel chroma span `S_chroma` in `[0, 1]` (<0.015 indicates 850/940nm monochrome IR!)
      9: highlight/IR-halation clipping ratio `mean(Y > 0.96)` in `[0, 1]`
    """
    crop_f = np.asarray(crop_rgb, dtype=np.float32) / 255.0
    if crop_f.ndim == 2:
        crop_f = np.repeat(crop_f[..., None], 3, axis=-1)

    y_lum = 0.299 * crop_f[..., 0] + 0.587 * crop_f[..., 1] + 0.114 * crop_f[..., 2]
    mu_y = float(np.clip(np.mean(y_lum), 0.0, 1.0))
    r_clip = float(np.mean(y_lum > 0.96))

    chroma_pix = np.max(crop_f, axis=-1) - np.min(crop_f, axis=-1)
    flat_c = chroma_pix.reshape(-1)
    top_k = max(1, int(round(0.15 * flat_c.size)))
    top_vals = np.partition(flat_c, flat_c.size - top_k)[flat_c.size - top_k :]
    s_chroma = float(np.clip(np.mean(top_vals), 0.0, 1.0))

    lap = cv2.Laplacian(y_lum, cv2.CV_32F, ksize=3)
    sharp = float(np.clip(0.25 * math.log1p(100.0 * float(np.var(lap))), 0.0, 1.0))

    if corners_xy is None:
        h_c, w_c = crop_f.shape[:2]
        w_top = w_bot = w_mean = float(w_c) * 0.90
        h_left = h_right = h_mean = float(h_c) * 0.90
        delta_norm = 0.0
    else:
        q = np.asarray(corners_xy, dtype=np.float32).reshape(4, 2)
        w_top = float(np.linalg.norm(q[1] - q[0]))
        h_right = float(np.linalg.norm(q[2] - q[1]))
        w_bot = float(np.linalg.norm(q[2] - q[3]))
        h_left = float(np.linalg.norm(q[3] - q[0]))
        w_mean = max(2.0, 0.5 * (w_top + w_bot))
        h_mean = max(1.0, 0.5 * (h_left + h_right))
        if prev_corners_xy is not None:
            qp = np.asarray(prev_corners_xy, dtype=np.float32).reshape(4, 2)
            delta_norm = float(np.clip(np.linalg.norm(q - qp) / (2.0 * w_mean), 0.0, 1.0))
        else:
            delta_norm = 0.0

    g0 = float(np.clip(0.5 * math.log2(max(1e-3, w_mean / 64.0)), -1.0, 1.0))
    g1 = float(np.clip(0.5 * math.log2(max(1e-3, h_mean / 32.0)), -1.0, 1.0))
    g2 = float(np.clip(w_mean / max(1e-3, 2.0 * h_mean), 0.25, 2.0) - 1.0)
    g3 = float(math.tanh(2.5 * math.log((h_left + 1e-3) / (h_right + 1e-3))))
    g4 = float(math.tanh(2.5 * math.log((w_bot + 1e-3) / (w_top + 1e-3))))

    return np.array(
        [g0, g1, g2, g3, g4, delta_norm, sharp, mu_y, s_chroma, r_clip],
        dtype=np.float32,
    )


def is_valid_plate_quad(c_xy: np.ndarray) -> bool:
    """Verify physical projective geometry invariants of a 4-corner `[TL, TR, BR, BL]` plate quad."""
    if c_xy.shape != (4, 2) or not np.all(np.isfinite(c_xy)):
        return False

    # 1. Strict clockwise convexity in image coordinates (+x right, +y down):
    # Directed edges e_k = p_{(k+1)%4} - p_k must turn clockwise (dx1 * dy2 - dy1 * dx2 > 0) at all 4 corners.
    edges = np.roll(c_xy, -1, axis=0) - c_xy
    next_edges = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if np.any(cross <= 0.0):
        return False

    return True


def _polygon_area_xy(c_xy: np.ndarray) -> float:
    return 0.5 * abs(
        float(np.sum(c_xy[:, 0] * np.roll(c_xy[:, 1], -1) - np.roll(c_xy[:, 0], -1) * c_xy[:, 1]))
    )


def _quad_raster_iou_and_iomin(qa: np.ndarray, qb: np.ndarray) -> tuple[float, float]:
    """Deterministic sub-pixel 64x64 local raster polygon IoU & IoMin (immune to collinear-edge degenerate cases)."""
    ax0, ay0 = float(qa[:, 0].min()), float(qa[:, 1].min())
    ax1, ay1 = float(qa[:, 0].max()), float(qa[:, 1].max())
    bx0, by0 = float(qb[:, 0].min()), float(qb[:, 1].min())
    bx1, by1 = float(qb[:, 0].max()), float(qb[:, 1].max())

    if min(ax1, bx1) <= max(ax0, bx0) or min(ay1, by1) <= max(ay0, by0):
        return 0.0, 0.0

    ux0, uy0 = min(ax0, bx0), min(ay0, by0)
    ux1, uy1 = max(ax1, bx1), max(ay1, by1)
    uw = max(1e-3, ux1 - ux0)
    uh = max(1e-3, uy1 - uy0)

    grid = 64
    scale_x = float(grid - 2) / uw
    scale_y = float(grid - 2) / uh

    pa = np.empty((4, 2), dtype=np.float32)
    pb = np.empty((4, 2), dtype=np.float32)
    pa[:, 0] = (qa[:, 0] - ux0) * scale_x + 1.0
    pa[:, 1] = (qa[:, 1] - uy0) * scale_y + 1.0
    pb[:, 0] = (qb[:, 0] - ux0) * scale_x + 1.0
    pb[:, 1] = (qb[:, 1] - uy0) * scale_y + 1.0

    ma = np.zeros((grid, grid), dtype=np.uint8)
    mb = np.zeros((grid, grid), dtype=np.uint8)
    cv2.fillConvexPoly(ma, np.round(pa * 4.0).astype(np.int32), 1, shift=2)
    cv2.fillConvexPoly(mb, np.round(pb * 4.0).astype(np.int32), 1, shift=2)

    inter = float(np.count_nonzero(ma & mb))
    if inter <= 0.0:
        return 0.0, 0.0
    area_a = float(np.count_nonzero(ma))
    area_b = float(np.count_nonzero(mb))
    union = max(1.0, area_a + area_b - inter)
    iomin = inter / max(1.0, min(area_a, area_b))
    return inter / union, iomin


def post_quad_nms(
    detections: list[PlateDetection],
    iou_thresh: float = 0.45,
    iomin_thresh: float = 0.70,
) -> list[PlateDetection]:
    """Suppress duplicate / partial Pass-1+Pass-2 refined quads converging onto the same physical plate."""
    if len(detections) <= 1:
        return detections

    def _rank_key(idx: int) -> tuple[float, float, float, float]:
        d = detections[idx]
        return (-float(d.quad_is_plate_prob), -_polygon_area_xy(d.corners_xy),
                float(d.bbox_xyxy[0]), float(d.bbox_xyxy[1]))

    order = sorted(range(len(detections)), key=_rank_key)
    keep: list[int] = []
    for idx in order:
        cand = detections[idx]
        suppress = False
        for k_idx in keep:
            win = detections[k_idx]
            iou, iomin = _quad_raster_iou_and_iomin(cand.corners_xy, win.corners_xy)
            if iou >= iou_thresh or iomin >= iomin_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(idx)
    return [detections[i] for i in keep]


@torch.no_grad()
def refine_and_crop_detections(
    quad_model: PlateQuadNet | PlateQuadTrackNet,
    frame_rgb: np.ndarray,
    detections: list[PlateDetection],
    device: torch.device | str = "cpu",
    min_is_plate_prob: float = .34,
    ocr_margin_ratio: float = .05,
    ocr_out_hw: tuple[int, int] = (80, 160),
    pixel_center_aligned: bool = True,
    enable_post_nms: bool = True,
    roi_gammas: list[float] | None = None,
    vehicles: list[VehicleDetection] | None = None,
    *, deployment_prior: float = .5,
    decision_costs: tuple[float, float, float] | None = None,
    alpha_crop: float = 1e-3,
) -> list[PlateDetection]:
    """Refine with joint plate/null evidence and posterior containment.

    Stage-1 score is retained as provenance only. The tracker deployment prior
    enters exactly once, when converting the Stage-2 LLR to a decision posterior.
    ``min_is_plate_prob`` maps to FP/FN costs for compatibility; explicit costs
    additionally allow abstention. Missing sensor/crop support sets
    ``quad_edge_touch=True``, ``support_nonconformity=inf`` and withholds OCR.
    One posterior-planned symmetric recrop is allowed; persistent support loss
    requests reacquisition through those fields, never clipped amodal geometry.
    """
    if not detections:
        return []
    if not 0 < deployment_prior < 1 or not 0 <= min_is_plate_prob <= 1:
        raise ValueError("Invalid deployment prior or acceptance threshold")
    quad_model.eval()
    dev = torch.device(device)
    height, width = frame_rgb.shape[:2]
    dual = bool(getattr(quad_model, "is_dual_track_net", False))
    cfg = getattr(quad_model, "config", None)
    fov_scale = float(getattr(cfg, "fovea_scale", FOVEA_BASE_SCALE))
    sur_scale = float(getattr(cfg, "surround_scale", SURROUND_BASE_SCALE))
    gammas = [max(1., float(roi_gammas[i] if roi_gammas is not None and i < len(roi_gammas) else det.roi_gamma))
              for i, det in enumerate(detections)]
    boxes = [d.bbox_xyxy for d in detections]
    for i, det in enumerate(detections):
        if det.corner_cov_xy is not None:
            plan = select_containment_crop_window(det.corners_xy, det.corner_cov_xy, (height, width), alpha_crop)
            boxes[i] = plan.required_window_xyxy
    policies = list(ObservationProvenance)
    policy = torch.tensor([policies.index(ObservationProvenance(d.provenance)) for d in detections], device=dev)

    def infer(crop_boxes: list[tuple[float, float, float, float]]) -> tuple[dict[str, torch.Tensor], list[tuple[float, float, float, float]]]:
        foveas, surrounds, windows = [], [], []
        for i, box in enumerate(crop_boxes):
            if dual:
                fov, sur, win, _ = extract_dual_concentric_plate_rois(frame_rgb, box, gamma=gammas[i],
                                                                     fovea_scale=fov_scale, surround_scale=sur_scale)
                surrounds.append(sur)
            else:
                fov, win = extract_expanded_plate_roi(frame_rgb, box, expand_scale=1.85*gammas[i])
            foveas.append(fov)
            windows.append(win)
        def batch(items: list[np.ndarray]) -> torch.Tensor:
            return torch.from_numpy(np.stack(items)).to(dev, dtype=torch.float32).permute(0, 3, 1, 2)/255
        if isinstance(quad_model, PlateQuadTrackNet):
            output = quad_model(batch(foveas), batch(surrounds), policy)
        elif dual:
            output = quad_model(batch(foveas), batch(surrounds))
        else:
            output = quad_model(batch(foveas))
        return output, windows

    def posterior_plan(output: dict[str, torch.Tensor], i: int,
                       window: tuple[float, float, float, float]) -> tuple[ContainmentCropWindow | None, bool]:
        x0, y0, rw, rh = window
        if "geometry_valid" in output and not bool(output["geometry_valid"][i]):
            return None, True
        scale = np.array([rw, rh])
        if "mode_corners" in output and "mode_corner_cov" in output:
            means = output["mode_corners"][i].cpu().numpy()*scale+np.array([x0, y0])
            cov = output["mode_corner_cov"][i].cpu().numpy()*scale[:, None]*scale[None, :]
            if "mode_weights" in output:
                retained = output["mode_weights"][i].cpu().numpy() > 0
                means, cov = means[retained], cov[retained]
                if not retained.any():
                    return None, True
        elif "corner_cov" in output or "corner_cov_chol" in output:
            means = output["corners_norm"][i].cpu().numpy()*scale+np.array([x0, y0])
            cov_t = output["corner_cov"][i] if "corner_cov" in output else chol3_to_cov2x2(output["corner_cov_chol"][i])
            cov = cov_t.cpu().numpy()*scale[:, None]*scale[None, :]
        else:
            return None, False
        plan = select_containment_crop_window(means, cov, (height, width), alpha_crop)
        l, t, r, b = plan.required_window_xyxy
        needs_crop = l < x0 or t < y0 or r > x0+rw or b > y0+rh
        return plan, needs_crop

    out, windows = infer(boxes)
    recrop = False
    for i in range(len(detections)):
        plan, needs_crop = posterior_plan(out, i, windows[i])
        if plan is not None and needs_crop and not plan.missing_support:
            boxes[i] = plan.required_window_xyxy
            recrop = True
    if recrop:
        out, windows = infer(boxes)
    llr = out.get("plate_null_llr", out["is_plate_logit"])
    posterior = torch.sigmoid(llr+math.log(deployment_prior/(1-deployment_prior)))
    costs = decision_costs if decision_costs is not None else (min_is_plate_prob, 1-min_is_plate_prob, 1.)
    actions = plate_null_cost_decision(posterior, *costs).cpu().numpy()
    corners = out["corners_norm"].cpu().numpy()
    sides = out["side_logits"].argmax(-1).cpu().numpy()
    qualities = torch.sigmoid(out["quality_logit"]).cpu().numpy()
    refined = []
    for i, det in enumerate(detections):
        if actions[i] != 0 or ("geometry_valid" in out and not bool(out["geometry_valid"][i])):
            continue
        x0, y0, rw, rh = windows[i]
        scale = np.array([rw, rh], dtype=np.float32)
        c_xy = corners[i]*scale+np.array([x0, y0], dtype=np.float32)
        if not is_valid_plate_quad(c_xy):
            continue
        plan, needs_crop = posterior_plan(out, i, windows[i])
        missing = (plan is not None and (plan.missing_support or needs_crop)) or bool(
            np.any(c_xy < 0) or np.any(c_xy > np.array([width, height])))
        covariance = None
        if "corner_cov" in out or "corner_cov_chol" in out:
            norm_cov = out["corner_cov"][i] if "corner_cov" in out else chol3_to_cov2x2(out["corner_cov_chol"][i])
            covariance = norm_cov.cpu().numpy()*scale[:, None]*scale[None, :]
        homography = None
        if "homography_3x3" in out:
            crop_to_image = np.array([[rw, 0, x0], [0, rh, y0], [0, 0, 1]], dtype=np.float32)
            homography = crop_to_image @ out["homography_3x3"][i].cpu().numpy()
        raw_score = float(det.raw_det_score if det.raw_det_score > 0 else det.score)
        p_plate = float(posterior[i])
        tight = None if missing else warp_plate_quad(frame_rgb, c_xy, out_hw=(64, 128), margin_ratio=0,
                                                     pixel_center_aligned=pixel_center_aligned)
        ocr = None if missing else warp_plate_quad(frame_rgb, c_xy, out_hw=ocr_out_hw, margin_ratio=ocr_margin_ratio,
                                                   pixel_center_aligned=pixel_center_aligned)
        lo, hi = np.clip(c_xy.min(0), 0, [width, height]), np.clip(c_xy.max(0), 0, [width, height])
        refined.append(PlateDetection(
            bbox_xyxy=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])), score=p_plate,
            corners_xy=c_xy, quad_refined=True, quad_confidence=float(qualities[i]), quad_is_plate_prob=p_plate,
            plate_side=int(sides[i]), tight_crop_rgb=tight, ocr_crop_rgb=ocr, track_id=det.track_id,
            vehicle_bbox_xyxy=det.vehicle_bbox_xyxy, raw_corners_xy=c_xy.copy(), is_coasted=False,
            raw_det_score=raw_score, quad_edge_touch=bool(missing), corner_cov_xy=covariance,
            blur_vec_xy=(out["blur_vec_norm"][i].cpu().numpy()*scale if "blur_vec_norm" in out else None),
            reid_emb=(out["reid_emb"][i].cpu().numpy() if "reid_emb" in out else None),
            is_large_prob=(float(torch.sigmoid(out["is_large_logit"][i])) if "is_large_logit" in out else 0.),
            roi_gamma=gammas[i], parent_vehicle_track_id=det.parent_vehicle_track_id,
            plate_null_llr=float(llr[i]),
            corner_visibility_4=(torch.sigmoid(out["corner_vis_logits"][i]).cpu().numpy() if "corner_vis_logits" in out else None),
            homography_3x3=homography,
            support_nonconformity=(float("inf") if missing else float(out["support_nonconformity"][i]) if "support_nonconformity" in out else 0.),
            provenance=det.provenance, frame_key=det.frame_key,
        ))
    return post_quad_nms(refined) if enable_post_nms else refined
