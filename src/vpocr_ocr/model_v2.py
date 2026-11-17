"""Gen-2 `MF-JPPlateOCR v2` with RepMobile backbone, Asymmetric FPN, ContiguousBilinearGather,
and a block Gaussian static-latent/AR(1)-nuisance filter with terminal-only readout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from vpocr_ocr.vocab import NUM_SLOTS


@dataclass
class ModelConfigV2:
    """Architecture configuration for `MFJPPlateOCRV2` ('base' ~1.45M, 'nano' ~1.02M)."""

    vocab_size: int = 254
    num_plate_types: int = 5
    num_status_classes: int = 3
    input_hw: tuple[int, int] = (80, 160)
    canonical_margin: float = 0.05
    is_v2: bool = True
    gen5_arch: bool = True
    d_model: int = 128
    c1: int = 44
    c2: int = 88
    c3: int = 112
    blocks: tuple[int, int, int] = (2, 4, 2)
    d_color: int = 32
    num_heads: int = 4
    mlp_ratio: float = 2.0
    conv_expansion: int = 2
    geom_dim: int = 10
    dropout: float = 0.0
    deployed: bool = False
    latent_prior_variance: float = 100.0
    nuisance_variance_init: float = 1.0
    nuisance_time_constant_s_init: float = 1.0

    @classmethod
    def from_preset(cls, preset: str = "base", vocab_size: int = 254) -> ModelConfigV2:
        """Create validated Gen-5 preset ('base' or 'nano')."""
        p = preset.lower().strip()
        if p == "base":
            return cls(
                vocab_size=vocab_size,
                input_hw=(80, 160),
                canonical_margin=0.05,
                is_v2=True,
                gen5_arch=True,
                d_model=128,
                c1=44,
                c2=88,
                c3=112,
                blocks=(2, 4, 2),
                d_color=32,
                num_heads=4,
                mlp_ratio=2.0,
                conv_expansion=2,
            )
        if p == "nano":
            return cls(
                vocab_size=vocab_size,
                input_hw=(80, 160),
                canonical_margin=0.05,
                is_v2=True,
                gen5_arch=True,
                d_model=112,
                c1=36,
                c2=72,
                c3=96,
                blocks=(2, 4, 2),
                d_color=32,
                num_heads=4,
                mlp_ratio=2.0,
                conv_expansion=2,
            )
        raise ValueError(f"Unknown ModelConfigV2 preset: {preset!r} (expected 'base' or 'nano')")


class FrameEncodingV2(NamedTuple):
    """Per-frame decoupled latent observations & anisotropic diagonal precisions (no char logits)."""

    z: torch.Tensor                    # (N, 12, D) character latent observation
    lambda_prec: torch.Tensor          # (N, 12, D) >= 0 anisotropic channel-wise precision
    z_color: torch.Tensor              # (N, Dc) rim + mid-plate color latent
    lambda_color: torch.Tensor         # (N, Dc) >= 0 chroma-gated color precision (==0 on IR!)
    pol_logit: torch.Tensor            # (N,) luminance polarity logit (+bright bg / -dark bg)
    pol_prec: torch.Tensor             # (N,) >= 0 polarity precision
    frame_valid_logit: torch.Tensor    # (N,) raw logit for auxiliary frame validity supervision
    slot_quality_logits: torch.Tensor  # (N, 12) raw logits for auxiliary slot quality supervision
    frame_valid_gate: torch.Tensor     # (N,) in [0, 1] with exact zero-write floor


class MemoryStateV2(NamedTuple):
    """Continuous natural-parameter O(1) `EvidenceMemoryV2` state per vehicle track."""

    mem_h: torch.Tensor             # (B, 12, D) posterior information vector (one Gaussian prior)
    mem_P: torch.Tensor             # (B, 12, D) marginal posterior precision of the static latent
    mem_conflict: torch.Tensor      # (B, 12) normalized squared innovation statistic
    color_h: torch.Tensor           # (B, Dc) decoupled color natural-parameter vector
    color_P: torch.Tensor           # (B, Dc) decoupled color diagonal precision
    pol_h: torch.Tensor             # (B,) luminance polarity natural-parameter scalar
    pol_P: torch.Tensor             # (B,) luminance polarity precision scalar
    pose_ema: torch.Tensor          # (B, 4+3*N) geometry and nuisance mean/variance/cross covariance
    last_timestamp_s: torch.Tensor  # (B,) timestamp of last valid observation (-1.0 initially)

    @property
    def nuisance_mean(self) -> torch.Tensor:
        n = NUM_SLOTS*self.mem_h.shape[-1] + self.color_h.shape[-1] + 1
        return self.pose_ema[:, 4:4+n]

    @property
    def nuisance_variance(self) -> torch.Tensor:
        n = NUM_SLOTS*self.mem_h.shape[-1] + self.color_h.shape[-1] + 1
        return self.pose_ema[:, 4+n:4+2*n]

    @property
    def latent_nuisance_covariance(self) -> torch.Tensor:
        n = NUM_SLOTS*self.mem_h.shape[-1] + self.color_h.shape[-1] + 1
        return self.pose_ema[:, 4+2*n:4+3*n]



class MemoryReadoutV2(NamedTuple):
    """Single final readout predictions decoded from `EvidenceMemoryV2`."""

    slot_logits: torch.Tensor            # (B, 12, vocab_size)
    plate_type_logits: torch.Tensor      # (B, 5)
    polarity_logit: torch.Tensor         # (B,)
    slot_readability_pred: torch.Tensor  # (B, 12) in [0, 1]
    status_logits: torch.Tensor          # (B, 3) [READABLE, UNREADABLE, TRACK_MIXED]


class SequenceOutputV2(NamedTuple):
    """Complete output bundle of `forward_sequence` for `OCRLossV2` training and evaluation."""

    slot_logits: torch.Tensor            # (B, 12, vocab_size)
    plate_type_logits: torch.Tensor      # (B, 5)
    polarity_logit: torch.Tensor         # (B,)
    slot_readability_pred: torch.Tensor  # (B, 12)
    status_logits: torch.Tensor          # (B, 3)
    memory: MemoryStateV2
    frame_z: torch.Tensor                # (B, T, 12, D)
    frame_lambda_prec: torch.Tensor      # (B, T, 12, D)
    frame_z_color: torch.Tensor          # (B, T, Dc)
    frame_lambda_color: torch.Tensor     # (B, T, Dc)
    frame_pol_logit: torch.Tensor        # (B, T)
    frame_pol_prec: torch.Tensor         # (B, T)
    frame_valid_logit: torch.Tensor      # (B, T)
    slot_quality_logits: torch.Tensor    # (B, T, 12)
    frame_valid_gate: torch.Tensor       # (B, T)
    prefix_slot_logits: torch.Tensor | None = None  # Optional (B, 12, V) at early prefix t_early
    prefix_readout: MemoryReadoutV2 | None = None   # Optional full MemoryReadoutV2 at prefix_len
    prefix_len: int = 0                             # Prefix frame count k in {1..T-1} when computed
    stroke_mask_logits: torch.Tensor | None = None  # Optional (B, T, 1, H, W) Text Gestalt stroke mask logits
    reg_len_logits: torch.Tensor | None = None      # Optional (B, T, 4) statutory region length logits (1..4)
    cls_len_logits: torch.Tensor | None = None      # Optional (B, T, 3) statutory class length logits (1..3)
    prefix_memory: MemoryStateV2 | None = None
    next_prefix_memory: MemoryStateV2 | None = None
    prefix_risk: torch.Tensor | None = None
    next_prefix_readout: MemoryReadoutV2 | None = None
    prefix_next_risk: torch.Tensor | None = None
    innovation_nll: torch.Tensor | None = None


def _fuse_conv_bn_double(
    conv_w: torch.Tensor,
    bn: nn.BatchNorm2d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact float64 algebraic fusion of Conv2d (bias=False) + BatchNorm2d -> (weight, bias)."""
    std = torch.sqrt(bn.running_var.double() + bn.eps)
    scale = bn.weight.double() / std
    fused_w = conv_w.double() * scale.reshape(-1, 1, 1, 1)
    fused_b = bn.bias.double() - bn.running_mean.double() * scale
    return fused_w, fused_b


class PhotometricStem(nn.Module):
    """Zero-parameter 4-channel stem: `[2*RGB - 1, 0.5*(Y - mu_Y)/clamp(std_Y, min=0.05)]`."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        r = crops[:, 0:1]
        g = crops[:, 1:2]
        b = crops[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b

        y_pad = F.pad(y, (4, 4, 4, 4), mode="replicate")
        mu_y = F.avg_pool2d(y_pad, kernel_size=9, stride=1)
        mu_y2 = F.avg_pool2d(y_pad * y_pad, kernel_size=9, stride=1)
        var_y = torch.clamp(mu_y2 - mu_y * mu_y, min=0.0)
        std_y = torch.clamp(torch.sqrt(var_y + 1e-6), min=0.05)
        y_lcn = torch.clamp(0.5 * (y - mu_y) / std_y, -3.0, 3.0)
        return torch.cat([crops * 2.0 - 1.0, y_lcn], dim=1)


class RepConv3x3(nn.Module):
    """Structural re-parameterizable `3x3` conv (`3x3+BN` + center-aligned `1x1+BN` + `Identity+BN`).

    Fuses algebraically into a single `nn.Conv2d(..., bias=True)` with zero runtime BN (`<1e-6` diff).
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
        deployed: bool = False,
    ) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.groups = groups
        self.deployed = deployed
        self.act: nn.Module = nn.SiLU() if act else nn.Identity()

        if deployed:
            self.rbr_reparam = nn.Conv2d(
                c_in, c_out, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True
            )
        else:
            self.conv3x3 = nn.Conv2d(
                c_in, c_out, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False
            )
            self.bn3x3 = nn.BatchNorm2d(c_out)
            self.conv1x1 = nn.Conv2d(
                c_in, c_out, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False
            )
            self.bn1x1 = nn.BatchNorm2d(c_out)
            self.id_bn: nn.BatchNorm2d | None = (
                nn.BatchNorm2d(c_in) if (stride == 1 and c_in == c_out) else None
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.act(self.rbr_reparam(x))
        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        if self.id_bn is not None:
            out = out + self.id_bn(x)
        return self.act(out)

    @torch.no_grad()
    def reparameterize(self) -> None:
        """Algebraically fuse multi-branch training weights into single `rbr_reparam` Conv2d."""
        if self.deployed:
            return
        dev = self.conv3x3.weight.device
        dt = self.conv3x3.weight.dtype

        w3, b3 = _fuse_conv_bn_double(self.conv3x3.weight, self.bn3x3)
        w1_raw, b1 = _fuse_conv_bn_double(self.conv1x1.weight, self.bn1x1)
        w1 = F.pad(w1_raw, (1, 1, 1, 1))

        w_sum = w3 + w1
        b_sum = b3 + b1

        if self.id_bn is not None:
            in_per_group = self.c_in // self.groups
            id_k = torch.zeros(
                (self.c_in, in_per_group, 3, 3), dtype=torch.float64, device=dev
            )
            for i in range(self.c_in):
                id_k[i, i % in_per_group, 1, 1] = 1.0
            std_id = torch.sqrt(self.id_bn.running_var.double() + self.id_bn.eps)
            scale_id = self.id_bn.weight.double() / std_id
            w_id = id_k * scale_id.reshape(-1, 1, 1, 1)
            b_id = self.id_bn.bias.double() - self.id_bn.running_mean.double() * scale_id
            w_sum = w_sum + w_id
            b_sum = b_sum + b_id

        reparam = nn.Conv2d(
            self.c_in,
            self.c_out,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True,
        ).to(device=dev, dtype=dt)
        reparam.weight.data.copy_(w_sum.to(dtype=dt))
        assert reparam.bias is not None
        reparam.bias.data.copy_(b_sum.to(dtype=dt))

        self.rbr_reparam = reparam
        self.deployed = True
        del self.conv3x3, self.bn3x3, self.conv1x1, self.bn1x1
        if hasattr(self, "id_bn"):
            del self.id_bn


class RepConv1x1(nn.Module):
    """Re-parameterizable `1x1` conv + BN (+ optional Identity+BN) fusing to single `1x1` Conv2d."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        act: bool = True,
        use_identity: bool = True,
        deployed: bool = False,
    ) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.deployed = deployed
        self.act: nn.Module = nn.SiLU() if act else nn.Identity()

        if deployed:
            self.rbr_reparam = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(c_out)
            self.id_bn: nn.BatchNorm2d | None = (
                nn.BatchNorm2d(c_in) if (use_identity and c_in == c_out) else None
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.act(self.rbr_reparam(x))
        out = self.bn(self.conv(x))
        if self.id_bn is not None:
            out = out + self.id_bn(x)
        return self.act(out)

    @torch.no_grad()
    def reparameterize(self) -> None:
        if self.deployed:
            return
        dev = self.conv.weight.device
        dt = self.conv.weight.dtype
        w_sum, b_sum = _fuse_conv_bn_double(self.conv.weight, self.bn)

        if self.id_bn is not None:
            id_k = torch.eye(self.c_in, dtype=torch.float64, device=dev).reshape(
                self.c_in, self.c_in, 1, 1
            )
            std_id = torch.sqrt(self.id_bn.running_var.double() + self.id_bn.eps)
            scale_id = self.id_bn.weight.double() / std_id
            w_sum = w_sum + id_k * scale_id.reshape(-1, 1, 1, 1)
            b_sum = b_sum + (self.id_bn.bias.double() - self.id_bn.running_mean.double() * scale_id)

        reparam = nn.Conv2d(self.c_in, self.c_out, kernel_size=1, bias=True).to(
            device=dev, dtype=dt
        )
        reparam.weight.data.copy_(w_sum.to(dtype=dt))
        assert reparam.bias is not None
        reparam.bias.data.copy_(b_sum.to(dtype=dt))

        self.rbr_reparam = reparam
        self.deployed = True
        del self.conv, self.bn
        if hasattr(self, "id_bn"):
            del self.id_bn


class RepMobileBlock(nn.Module):
    """Depthwise `RepConv3x3` + inverted `RepConv1x1` bottleneck with zero inference BN."""

    def __init__(self, dim: int, expansion: int = 2, deployed: bool = False) -> None:
        super().__init__()
        hidden = dim * expansion
        self.dw = RepConv3x3(dim, dim, stride=1, groups=dim, act=True, deployed=deployed)
        self.pw1 = RepConv1x1(dim, hidden, act=True, use_identity=False, deployed=deployed)
        self.pw2 = RepConv1x1(hidden, dim, act=False, use_identity=False, deployed=deployed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(self.pw1(self.dw(x)))


def _build_1d_sincos_pos_pair_v2(
    d_model: int, grid_h: int, grid_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create exact separable `((1, D//2, grid_h, 1), (1, D//2, 1, grid_w))` 2D sincos factors.

    Mathematically bit-identical to a dense `(1, D, grid_h, grid_w)` 2D sinusoidal buffer while
    reducing constant initializer storage in `frame_encoder.onnx` / `.int8.onnx` by >97% (~1.25 MiB).
    """
    if d_model % 4 != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by 4")
    quarter = d_model // 4
    omega = torch.pow(10000.0, -torch.arange(quarter, dtype=torch.float32) / float(quarter))
    ys = (torch.arange(grid_h, dtype=torch.float32) + 0.5) / float(grid_h) * (2.0 * math.pi)
    xs = (torch.arange(grid_w, dtype=torch.float32) + 0.5) / float(grid_w) * (2.0 * math.pi)
    out_y = ys[:, None] * omega[None, :]  # (H, D//4)
    out_x = xs[:, None] * omega[None, :]  # (W, D//4)
    pos_y = (
        torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1)
        .transpose(0, 1)
        .reshape(1, d_model // 2, grid_h, 1)
        .contiguous()
    )
    pos_x = (
        torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1)
        .transpose(0, 1)
        .reshape(1, d_model // 2, 1, grid_w)
        .contiguous()
    )
    return pos_y, pos_x


def _add_separable_2d_pos_v2(
    feat: torch.Tensor, pos_y: torch.Tensor, pos_x: torch.Tensor
) -> torch.Tensor:
    """Add separable `(pos_y, pos_x)` onto `(N, D, H, W)` runtime features without 2D constant folding."""
    half_d = feat.shape[1] // 2
    py = pos_y.to(dtype=feat.dtype)
    px = pos_x.to(dtype=feat.dtype)
    if not torch.jit.is_tracing() and (feat.shape[-2] != py.shape[-2] or feat.shape[-1] != px.shape[-1]):
        py = F.interpolate(py, size=(feat.shape[-2], 1), mode="bilinear", align_corners=False)
        px = F.interpolate(px, size=(1, feat.shape[-1]), mode="bilinear", align_corners=False)
    return torch.cat([feat[:, :half_d] + py, feat[:, half_d:] + px], dim=1)


def _build_canonical_slot_anchors(margin_ratio: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `(12, 2)` canonical centers `(u', v')` and `(12, 2)` `3x3` step sizes `(du, dv)`.

    Calibrated directly from real-world balanced `330x165 mm` Medium and `440x220 mm` Large MLIT
    `PlateGeometry` mapped through `u' = margin + (1 - 2*margin) * u0`, `v' = margin + (1 - 2*margin) * v0`.
    """
    scale = 1.0 - 2.0 * float(margin_ratio)
    # Physical (u0, v0, step_u0, step_v0) on [0, 1] x [0, 1] plate face
    raw_specs = [
        # Slots 0..3: Upper-left Region Kanji (calibrated across Medium 330mm & Large 440mm 1..4-char dies)
        (0.304, 0.212, 0.046, 0.102),
        (0.398, 0.212, 0.048, 0.102),
        (0.442, 0.212, 0.042, 0.102),
        (0.474, 0.212, 0.038, 0.102),
        # Slots 4..6: Upper-right Classification (Medium 187.5/212.5/237.5mm & Large 238/271/303mm)
        (0.556, 0.212, 0.036, 0.102),
        (0.630, 0.212, 0.036, 0.102),
        (0.704, 0.212, 0.036, 0.102),
        # Slot 7: Lower-left Hiragana (Medium 40/330, 111.5/165 & Large 53.5/440, 152.5/220 + dakuten)
        (40.0 / 330.0, 112.0 / 165.0, 0.042, 0.095),
        # Slots 8..11: Lower-right 4-slot Serial (Medium 95, 150, 235, 290mm & Large 127.5, 197.5, 312.5, 382.5mm)
        (95.0 / 330.0, 111.0 / 165.0, 0.038, 0.155),
        (150.0 / 330.0, 111.0 / 165.0, 0.038, 0.155),
        (235.0 / 330.0, 111.0 / 165.0, 0.038, 0.155),
        (290.0 / 330.0, 111.0 / 165.0, 0.038, 0.155),
    ]
    centers = torch.tensor(
        [[margin_ratio + scale * u0, margin_ratio + scale * v0] for u0, v0, _, _ in raw_specs],
        dtype=torch.float32,
    )
    steps = torch.tensor(
        [[scale * su, scale * sv] for _, _, su, sv in raw_specs],
        dtype=torch.float32,
    )
    return centers, steps


def _build_statutory_length_anchor_tables(
    margin_ratio: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build length-conditioned MLIT statutory anchors & `3x3` steps for Region (`1..4`) & Class (`1..3`).

    Calibrated directly from `PlateGeometry` (`330x165 mm` Medium & `440x220 mm` Large dies):
    - 2-char regions place Kanji #0 at `u0 ~ 0.3089` and Kanji #1 at `u0 ~ 0.4275` (NOT `0.398`),
      while inactive `<PAD>` slots sit in the inter-field blank margin so `<PAD>` never bleeds.
    - 2-digit classifications place Digits #0..1 at `u0 ~ 0.5775, 0.6612` and `<PAD>` Slot 6 in
      the right blank margin (`u0 ~ 0.748`), eliminating `54 -> 544` bleed.
    """
    scale = 1.0 - 2.0 * float(margin_ratio)
    v0_up = 0.212
    sv0_up = 0.102

    reg_u0_su0 = [
        # L_reg = 1 (e.g. 柏, 堺, 津): Slot 0 centered; Slots 1..3 in blank inter-field margin
        [(0.3894, 0.052), (0.4680, 0.022), (0.4880, 0.020), (0.5040, 0.018)],
        # L_reg = 2 (e.g. 長崎, 岡山, 福山, 品川): Slots 0..1 at 0.3089, 0.4275; Slots 2..3 in gap
        [(0.3089, 0.048), (0.4275, 0.048), (0.4860, 0.020), (0.5040, 0.018)],
        # L_reg = 3 (e.g. 名古屋, 佐世保, 久留米): Slots 0..2 at 0.2879, 0.3735, 0.4591; Slot 3 in gap
        [(0.2879, 0.038), (0.3735, 0.038), (0.4591, 0.038), (0.5060, 0.018)],
        # L_reg = 4 (e.g. 尾張小牧, 伊勢志摩): Slots 0..3 condensed across region box
        [(0.2757, 0.030), (0.3403, 0.030), (0.4048, 0.030), (0.4694, 0.030)],
    ]
    cls_u0_su0 = [
        # L_cls = 1 (1-digit class): Slot 4 centered; Slots 5..6 in right blank margin
        [(0.6006, 0.042), (0.6920, 0.024), (0.7480, 0.022)],
        # L_cls = 2 (2-digit class e.g. 54, 33): Slots 4..5 active; Slot 6 in right blank margin
        [(0.5775, 0.040), (0.6612, 0.040), (0.7480, 0.022)],
        # L_cls = 3 (3-char class e.g. 330, 580, 35A): Slots 4..6 active
        [(0.5549, 0.036), (0.6296, 0.036), (0.7045, 0.036)],
    ]

    reg_anchors = torch.tensor(
        [
            [[margin_ratio + scale * u0, margin_ratio + scale * v0_up] for u0, _ in row]
            for row in reg_u0_su0
        ],
        dtype=torch.float32,
    )  # (4, 4, 2)
    reg_steps = torch.tensor(
        [[[scale * su0, scale * sv0_up] for _, su0 in row] for row in reg_u0_su0],
        dtype=torch.float32,
    )  # (4, 4, 2)
    cls_anchors = torch.tensor(
        [
            [[margin_ratio + scale * u0, margin_ratio + scale * v0_up] for u0, _ in row]
            for row in cls_u0_su0
        ],
        dtype=torch.float32,
    )  # (3, 3, 2)
    cls_steps = torch.tensor(
        [[[scale * su0, scale * sv0_up] for _, su0 in row] for row in cls_u0_su0],
        dtype=torch.float32,
    )  # (3, 3, 2)
    return reg_anchors, reg_steps, cls_anchors, cls_steps


class PatchTopologyMixer(nn.Module):
    """Nonlinear `3x3` spatial-channel mixer preserving 2D stroke crossings & enclosed counters.

    Uses strictly ONNX-whitelisted `MatMul`, `Add`, `Reshape`, `Sigmoid`, and `Mul` ops with
    hidden dimensions guaranteed `!= vocab_size`.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        d_bot = max(16, d_model // 4)
        if 9 * d_bot == 254:
            d_bot += 4
        self.d_bot = d_bot
        self.down = nn.Linear(d_model, d_bot)
        self.mix = nn.Linear(9 * d_bot, d_model)
        self.up = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, v_combined: torch.Tensor) -> torch.Tensor:
        v_b = self.down(v_combined)  # (N, 12, 9, d_bot)
        v_flat = v_b.reshape(-1, NUM_SLOTS, 9 * self.d_bot)  # (N, 12, 9 * d_bot)
        return self.up(F.silu(self.mix(v_flat)))  # (N, 12, D)


def _build_color_zone_masks(h: int, w: int, margin_ratio: float = 0.05) -> torch.Tensor:
    """Build `(1, 3, h, w)` normalized spatial pooling masks: `[outer_rim, mid_bg, char_band]`."""
    ys = (torch.arange(h, dtype=torch.float32) + 0.5) / float(h)
    xs = (torch.arange(w, dtype=torch.float32) + 0.5) / float(w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    # 1. Outer plate border ribbon strictly inside physical [margin_ratio, 1 - margin_ratio] face
    # (0% vehicle bumper surround; directly captures 6.5mm yellow/green graphic border ribbons!)
    rim_lo = margin_ratio + 0.003
    rim_hi_x = margin_ratio + 0.035
    rim_hi_y = margin_ratio + 0.055
    in_plate = (xx >= rim_lo) & (xx <= 1.0 - rim_lo) & (yy >= rim_lo) & (yy <= 1.0 - rim_lo)
    inner_core = (
        (xx >= rim_hi_x)
        & (xx <= 1.0 - rim_hi_x)
        & (yy >= rim_hi_y)
        & (yy <= 1.0 - rim_hi_y)
    )
    rim = (in_plate & (~inner_core)).float()

    # 2. Mid-plate background band between upper and lower character rows + left upper margin
    mid_gap = ((yy >= 0.41) & (yy <= 0.54) & (xx >= 0.08) & (xx <= 0.92)).float()

    # 3. Character stroke bands (upper + lower rows) for background-vs-stroke polarity
    upper_band = (yy >= 0.16) & (yy <= 0.36) & (xx >= 0.24) & (xx <= 0.80)
    lower_band = (yy >= 0.56) & (yy <= 0.84) & (xx >= 0.12) & (xx <= 0.90)
    char_band = (upper_band | lower_band).float()

    masks = torch.stack([rim, mid_gap, char_band], dim=0)  # (3, H, W)
    masks = masks / torch.clamp(masks.sum(dim=(1, 2), keepdim=True), min=1.0)
    return masks.unsqueeze(0).contiguous()


def contiguous_bilinear_gather(feat_map: torch.Tensor, coords_uv: torch.Tensor) -> torch.Tensor:
    """ONNX Opset 17 `ContiguousBilinearGather` (`Gather(axis=0)` on cache-contiguous `(N*HW, C)` rows).

    Args:
        feat_map: `(N, C, H_f, W_f)` feature map.
        coords_uv: `(N, P, 2)` normalized pixel-center `(u, v)` coordinates in `[0, 1]`.

    Returns:
        Sampled features of shape `(N, P, C)`.
    """
    n, c, hf, wf = feat_map.shape
    p = coords_uv.shape[1]
    hw = hf * wf

    u = torch.clamp(coords_uv[:, :, 0], 0.0, 1.0)
    v = torch.clamp(coords_uv[:, :, 1], 0.0, 1.0)

    wf_f = u.new_zeros(()) + wf
    hf_f = v.new_zeros(()) + hf
    gx = torch.minimum(torch.clamp(u * wf_f - 0.5, min=0.0), wf_f - (1.0 + 1e-4))
    gy = torch.minimum(torch.clamp(v * hf_f - 0.5, min=0.0), hf_f - (1.0 + 1e-4))

    x0 = torch.floor(gx)
    y0 = torch.floor(gy)
    wx = (gx - x0).to(dtype=feat_map.dtype)
    wy = (gy - y0).to(dtype=feat_map.dtype)

    ix0 = x0.to(dtype=torch.long)
    iy0 = y0.to(dtype=torch.long)
    ones_n = torch.ones_like(feat_map[:, :1, 0, 0], dtype=torch.long)  # (N, 1)
    max_ix = ones_n.new_zeros(()) + (wf - 1)
    max_iy = ones_n.new_zeros(()) + (hf - 1)
    ix1 = torch.minimum(ix0 + 1, max_ix)
    iy1 = torch.minimum(iy0 + 1, max_iy)

    # TracerWarning-free dynamic batch index [0, hw, 2*hw, ..., (N-1)*hw] via CumSum
    batch_base = (torch.cumsum(ones_n, dim=0) - 1) * hw  # (N, 1)

    idx00 = (batch_base + iy0 * wf + ix0).reshape(-1)
    idx01 = (batch_base + iy0 * wf + ix1).reshape(-1)
    idx10 = (batch_base + iy1 * wf + ix0).reshape(-1)
    idx11 = (batch_base + iy1 * wf + ix1).reshape(-1)

    feat_rows = feat_map.flatten(2).transpose(1, 2).contiguous().reshape(-1, c)
    v00 = torch.index_select(feat_rows, 0, idx00).reshape(n, p, c)
    v01 = torch.index_select(feat_rows, 0, idx01).reshape(n, p, c)
    v10 = torch.index_select(feat_rows, 0, idx10).reshape(n, p, c)
    v11 = torch.index_select(feat_rows, 0, idx11).reshape(n, p, c)

    w00 = ((1.0 - wy) * (1.0 - wx)).unsqueeze(-1)
    w01 = ((1.0 - wy) * wx).unsqueeze(-1)
    w10 = (wy * (1.0 - wx)).unsqueeze(-1)
    w11 = (wy * wx).unsqueeze(-1)
    return w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11


class IntraSlotAttention(nn.Module):
    """Lightweight `12x12` ONNX-clean multi-head self-attention + MLP across the 12 plate slots."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(round(d_model * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, s_len, _ = x.shape
        xn = self.norm1(x)
        qkv = self.qkv(xn).reshape(bsz, s_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probs = self.drop(F.softmax(scores, dim=-1))
        ctx = torch.matmul(probs, v).transpose(1, 2).reshape(bsz, s_len, self.d_model)
        x = x + self.proj(ctx)
        x = x + self.mlp(self.norm2(x))
        return x


class EvidenceMemoryV2(nn.Module):
    """Block-diagonal Gaussian filter for static z and correlated AR(1) nuisance b.

    Each channel has a full 2x2 covariance including Cov(z,b). The legacy
    pose_ema tensor carries geometry then [mean_b, var_b, cov_zb]. Parameters
    are learned using sequence innovation likelihood, without optical rules.
    """
    def __init__(self, d_model: int, d_color: int, prior_variance: float = 100.0,
                 nuisance_variance_init: float = 1.0, time_constant_s_init: float = 1.0) -> None:
        super().__init__()
        if any(not math.isfinite(v) or v <= 0 for v in (prior_variance, nuisance_variance_init, time_constant_s_init)):
            raise ValueError("Gaussian prior/noise variances and time constant must be positive")
        self.d_model, self.d_color = d_model, d_color
        self.measurement_dim = NUM_SLOTS * d_model + d_color + 1
        self.state_dim = 4 + 3 * self.measurement_dim
        self.log_nuisance_variance = nn.Parameter(torch.full((NUM_SLOTS + 2,), math.log(nuisance_variance_init)))
        self.log_time_constant = nn.Parameter(torch.full((NUM_SLOTS + 2,), math.log(time_constant_s_init)))
        self.register_buffer("prior_variance", torch.tensor(float(prior_variance)))
        self.register_buffer("channel_groups", torch.tensor(
            [i for i in range(NUM_SLOTS) for _ in range(d_model)]
            + [NUM_SLOTS]*d_color + [NUM_SLOTS+1], dtype=torch.long), persistent=False)

    def _expand_channels(self, x):
        return x[self.channel_groups]

    def init_state(self, batch_size=1, device="cpu", dtype=torch.float32):
        kw = dict(device=device, dtype=dtype)
        return MemoryStateV2(
            torch.zeros(batch_size, NUM_SLOTS, self.d_model, **kw),
            torch.zeros(batch_size, NUM_SLOTS, self.d_model, **kw),
            torch.zeros(batch_size, NUM_SLOTS, **kw),
            torch.zeros(batch_size, self.d_color, **kw),
            torch.zeros(batch_size, self.d_color, **kw),
            torch.zeros(batch_size, **kw), torch.zeros(batch_size, **kw),
            torch.zeros(batch_size, self.state_dim, **kw),
            torch.full((batch_size,), -1.0, **kw))

    def update_with_innovation(self, mem_h, mem_P, mem_conflict, color_h, color_P,
                              pol_h, pol_P, pose_ema, last_timestamp_s, z,
                              lambda_prec, z_color, lambda_color, pol_logit,
                              pol_prec, geom_meta, timestamp_s, step_mask=None):
        """Joseph covariance update; returns state, Gaussian NLL, active mask."""
        n, d = self.measurement_dim, self.d_model
        def pack(a, b, c):
            return torch.cat((a.flatten(1), b, c[:, None]), dim=1).float()
        h, prec = pack(mem_h, color_h, pol_h), pack(mem_P, color_P, pol_P)
        y, lam = pack(z, z_color, pol_logit), pack(lambda_prec, lambda_color, pol_prec)
        finite_y = (y > -1e6) & (y < 1e6)
        lam = torch.where((lam > 0) & (lam < 1e6) & finite_y, lam, torch.zeros_like(lam))
        y = torch.where(finite_y, y, torch.zeros_like(y))
        if step_mask is not None:
            lam = lam * (step_mask > 0.5).float()[:, None]
        active = lam > 0
        frame_active = active.any(dim=1)
        if not torch.jit.is_tracing() and pose_ema.shape[1] != self.state_dim:
            if bool((prec > 0).any()):
                raise ValueError("Legacy accumulated state lacks nuisance covariance")
            pose_ema = torch.zeros(h.shape[0], self.state_dim, device=h.device)
        packed = pose_ema.float()
        mb, vb, cross = packed[:, 4:4+n], packed[:, 4+n:4+2*n], packed[:, 4+2*n:4+3*n]
        nv = self._expand_channels(torch.exp(self.log_nuisance_variance.clamp(-10, 10)))[None]
        tau = self._expand_channels(torch.exp(self.log_time_constant.clamp(-10, 10)))[None]
        ts = torch.where((timestamp_s > -1e9) & (timestamp_s < 1e9), timestamp_s, last_timestamp_s)
        dt = torch.clamp(ts - last_timestamp_s, min=0)[:, None]
        a = torch.exp(-dt / tau)
        has = prec > 0
        mz = torch.where(has, h / prec.clamp_min(1e-8), torch.zeros_like(h))
        vz = torch.where(has, 1 / prec.clamp_min(1e-8), self.prior_variance.expand_as(prec))
        mbp = a * mb
        vbp = torch.where(has, a*a*vb + (1-a*a)*nv, nv.expand_as(vb))
        cp = a * cross
        r = 1 / lam.clamp_min(1e-8)
        innovation = y - mz - mbp
        total = (vz + 2*cp + vbp + r).clamp_min(1e-8)
        kz, kb = (vz + cp) / total, (vbp + cp) / total
        mz_new, mb_new = mz + kz*innovation, mbp + kb*innovation
        # Joseph form preserves covariance positivity under quantized arithmetic.
        vz_new = ((1-kz)**2*vz + kz*kz*vbp - 2*(1-kz)*kz*cp + kz*kz*r).clamp_min(1e-8)
        vb_new = (kb*kb*vz + (1-kb)**2*vbp - 2*kb*(1-kb)*cp + kb*kb*r).clamp_min(1e-8)
        cross_new = -(1-kz)*kb*vz - kz*(1-kb)*vbp + ((1-kz)*(1-kb)+kz*kb)*cp + kz*kb*r
        new_p = torch.where(active, 1/vz_new, prec)
        new_h = torch.where(active, mz_new/vz_new, h)
        mb_new = torch.where(active, mb_new, mb)
        vb_new = torch.where(active, vb_new, vb)
        cross_new = torch.where(active, cross_new, cross)
        nis = innovation*innovation/total
        slot_nis = nis[:, :NUM_SLOTS*d].reshape(-1, NUM_SLOTS, d).mean(-1)
        slot_active = active[:, :NUM_SLOTS*d].reshape(-1, NUM_SLOTS, d).any(-1)
        new_conflict = torch.where(slot_active, slot_nis, mem_conflict)
        geom = torch.where((geom_meta > -1e6) & (geom_meta < 1e6), geom_meta, torch.zeros_like(geom_meta))
        pose = torch.stack((geom[:, 0], geom[:, 2], geom[:, 3], geom[:, 4]), -1)
        pose = torch.where(frame_active[:, None], pose, packed[:, :4])
        packed_new = torch.cat((pose, mb_new, vb_new, cross_new), -1)
        cut = NUM_SLOTS*d
        state = MemoryStateV2(new_h[:, :cut].reshape(-1, NUM_SLOTS, d),
            new_p[:, :cut].reshape(-1, NUM_SLOTS, d), new_conflict,
            new_h[:, cut:cut+self.d_color], new_p[:, cut:cut+self.d_color],
            new_h[:, -1], new_p[:, -1], packed_new,
            torch.where(frame_active, ts, last_timestamp_s))
        nll = torch.where(active, 0.5*(torch.log(total) + nis), torch.zeros_like(total))
        return state, nll, active

    def step(self, mem_h, mem_P, mem_conflict, color_h, color_P, pol_h, pol_P,
             pose_ema, last_timestamp_s, z, lambda_prec, z_color, lambda_color,
             pol_logit, pol_prec, geom_meta, timestamp_s, step_mask=None):
        return self.update_with_innovation(mem_h, mem_P, mem_conflict, color_h, color_P,
            pol_h, pol_P, pose_ema, last_timestamp_s, z, lambda_prec, z_color,
            lambda_color, pol_logit, pol_prec, geom_meta, timestamp_s, step_mask)[0]


def noncharacter_quality_features(state, support, geometry_variance):
    """Precision, visibility, geometry and innovation only; never latent means."""
    p = state.mem_P.float().mean(-1)
    return torch.stack((torch.log1p(p.mean(-1)), torch.log1p(p.amin(-1)),
        torch.log1p(p.amax(-1)), (p > 0).float().mean(-1),
        torch.log1p(state.color_P.float().mean(-1)), torch.log1p(state.pol_P.float()),
        state.mem_conflict.float().mean(-1), state.mem_conflict.float().amax(-1),
        support.float(), geometry_variance.float()), -1)


class MFJPPlateOCRV2(nn.Module):
    """Gen-2 Multi-Frame Japanese License Plate OCR (`v2-Base` ~1.09M / `v2-Nano` ~0.41M)."""

    is_v2: bool = True

    def __init__(self, config: ModelConfigV2 | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfigV2()
        d = self.config.d_model
        dc = self.config.d_color
        c1, c2, c3 = self.config.c1, self.config.c2, self.config.c3
        b1, b2, b3 = self.config.blocks
        dep = self.config.deployed
        exp = self.config.conv_expansion

        h_in, w_in = self.config.input_hw
        h2, w2 = h_in // 2, w_in // 2
        h4, w4 = h_in // 4, w_in // 4
        h8, w8 = h_in // 8, w_in // 8

        # 1. Photometric stem + RepMobile backbone (zero runtime BN after reparameterize())
        self.photo_stem = PhotometricStem()
        if self.config.gen5_arch:
            c0 = max(16, c1 // 2)
            self.pre_s1: RepConv3x3 | None = RepConv3x3(
                4, c0, stride=1, groups=1, act=True, deployed=dep
            )
            self.ds1 = RepConv3x3(c0, c1, stride=2, groups=1, act=True, deployed=dep)
        else:
            self.pre_s1 = None
            self.ds1 = RepConv3x3(4, c1, stride=2, groups=1, act=True, deployed=dep)
        self.stage1 = nn.Sequential(
            *[RepMobileBlock(c1, expansion=exp, deployed=dep) for _ in range(b1)]
        )
        self.ds2 = RepConv3x3(c1, c2, stride=2, groups=1, act=True, deployed=dep)
        self.stage2 = nn.Sequential(
            *[RepMobileBlock(c2, expansion=exp, deployed=dep) for _ in range(b2)]
        )
        self.ds3 = RepConv3x3(c2, c3, stride=2, groups=1, act=True, deployed=dep)
        self.stage3 = nn.Sequential(
            *[RepMobileBlock(c3, expansion=exp, deployed=dep) for _ in range(b3)]
        )

        # 2. Asymmetric Multi-Scale FPN (F2: stride 2 upper-row high-acuity, F4: stride 4, F8: stride 8)
        self.f2_lat = RepConv1x1(c1, d, act=False, use_identity=False, deployed=dep)
        self.f4_lat = RepConv1x1(c2, d, act=False, use_identity=False, deployed=dep)
        self.f8_lat = RepConv1x1(c3, d, act=False, use_identity=False, deployed=dep)
        if self.config.gen5_arch:
            self.f2_refine: nn.Sequential | None = nn.Sequential(
                RepConv3x3(d, d, stride=1, groups=d, act=True, deployed=dep),
                RepConv1x1(d, d, act=True, use_identity=True, deployed=dep),
            )
        else:
            self.f2_refine = None
        self.f4_refine = nn.Sequential(
            RepConv3x3(d, d, stride=1, groups=d, act=True, deployed=dep),
            RepConv1x1(d, d, act=True, use_identity=True, deployed=dep),
        )
        self.f8_refine = nn.Sequential(
            RepConv3x3(d, d, stride=1, groups=d, act=True, deployed=dep),
            RepConv1x1(d, d, act=True, use_identity=True, deployed=dep),
        )
        p2_y, p2_x = _build_1d_sincos_pos_pair_v2(d, h2, w2)
        p4_y, p4_x = _build_1d_sincos_pos_pair_v2(d, h4, w4)
        p8_y, p8_x = _build_1d_sincos_pos_pair_v2(d, h8, w8)
        self.register_buffer("pos_f2_y", p2_y)
        self.register_buffer("pos_f2_x", p2_x)
        self.register_buffer("pos_f4_y", p4_y)
        self.register_buffer("pos_f4_x", p4_x)
        self.register_buffer("pos_f8_y", p8_y)
        self.register_buffer("pos_f8_x", p8_x)

        # 3. Row-strip alignment vector a_t + plate_side + 10-D geom_meta conditioning
        # Horizontal 8-bin normalized Gaussian projection for resolution-agnostic row profiles
        xs_w4 = (torch.arange(w4, dtype=torch.float32) + 0.5) / float(w4)
        bin_centers = (torch.arange(8, dtype=torch.float32) + 0.5) / 8.0
        gx = torch.exp(-0.5 * ((xs_w4[:, None] - bin_centers[None, :]) / 0.085).pow(2))
        gx = gx / torch.clamp(gx.sum(dim=0, keepdim=True), min=1e-6)
        self.register_buffer("horiz_basis_w4", gx.contiguous())  # (W4, 8)

        prof_ch = max(16, d // 4)
        self.upper_prof_proj = nn.Linear(8 * prof_ch, d // 2)
        self.lower_prof_proj = nn.Linear(8 * prof_ch, d // 2)
        self.upper_ch_reduce = nn.Linear(d, prof_ch)
        self.lower_ch_reduce = nn.Linear(d, prof_ch)

        self.plate_side_embed = nn.Embedding(3, d // 4)
        self.geom_mlp = nn.Sequential(
            nn.Linear(self.config.geom_dim, d // 2),
            nn.LayerNorm(d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, d // 2),
            nn.SiLU(),
        )
        align_in_dim = (d // 2) + (d // 2) + (d // 4) + (d // 2)
        self.align_mlp = nn.Sequential(
            nn.Linear(align_in_dim, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # 4. 2-Pass Local Deformable Slot Sampler + Depthwise 3x3 Point Weights + 12x12 SA
        anchors, steps = _build_canonical_slot_anchors(self.config.canonical_margin)
        self.register_buffer("slot_anchors_uv", anchors)  # (12, 2)
        self.register_buffer("slot_steps_uv", steps)      # (12, 2)
        if self.config.gen5_arch:
            reg_anc, reg_stp, cls_anc, cls_stp = _build_statutory_length_anchor_tables(
                self.config.canonical_margin
            )
            self.register_buffer("reg_len_anchors_uv", reg_anc)  # (4, 4, 2)
            self.register_buffer("reg_len_steps_uv", reg_stp)    # (4, 4, 2)
            self.register_buffer("cls_len_anchors_uv", cls_anc)  # (3, 3, 2)
            self.register_buffer("cls_len_steps_uv", cls_stp)    # (3, 3, 2)
            self.reg_len_head: nn.Linear | None = nn.Linear(d // 2, 4)
            self.cls_len_head: nn.Linear | None = nn.Linear(d // 2, 3)
            nn.init.normal_(self.reg_len_head.weight, std=0.01)
            self.reg_len_head.bias.data.copy_(
                torch.tensor([-1.2, 0.8, -0.3, -1.0], dtype=torch.float32)
            )
            nn.init.normal_(self.cls_len_head.weight, std=0.01)
            self.cls_len_head.bias.data.copy_(
                torch.tensor([-1.15, -0.35, 0.65], dtype=torch.float32)
            )
        else:
            self.reg_len_head = None
            self.cls_len_head = None

        grid_1d = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        gy, gx_3 = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
        grid_3x3 = torch.stack([gx_3.reshape(-1), gy.reshape(-1)], dim=-1)  # (9, 2)
        self.register_buffer("local_grid_3x3", grid_3x3)

        self.slot_embed = nn.Parameter(torch.randn(1, NUM_SLOTS, d) * 0.02)
        # Learned per-slot F2 vs F4 high-acuity gate for upper-row 40mm Kanji/Class slots 0..6
        self.upper_scale_gate = nn.Parameter(torch.full((1, 7, 1, d), 0.7, dtype=torch.float32))
        # Learned per-slot F4 vs F8 scale gate for lower-row slots 7..11 (init +0.6 -> favors F4 detail)
        self.lower_scale_gate = nn.Parameter(torch.full((1, 5, 1, d), 0.6, dtype=torch.float32))

        # Pass 1 deformable offset/scale head from (slot_embed + a_t)
        self.offset_head_1 = nn.Linear(d, 4)
        nn.init.zeros_(self.offset_head_1.weight)
        nn.init.zeros_(self.offset_head_1.bias)
        self.point_weights_1 = nn.Parameter(torch.full((1, NUM_SLOTS, 9, d), 1.0 / 9.0))
        self.patch_mixer_1: PatchTopologyMixer | None = (
            PatchTopologyMixer(d) if self.config.gen5_arch else None
        )
        self.sample_proj_1 = nn.Linear(d, d)
        self.intra_sa_1 = IntraSlotAttention(
            d, num_heads=self.config.num_heads, mlp_ratio=self.config.mlp_ratio, dropout=self.config.dropout
        )

        # Pass 2 residual deformable refinement from Pass-1 slot states
        self.offset_head_2 = nn.Linear(d, 4)
        nn.init.zeros_(self.offset_head_2.weight)
        nn.init.zeros_(self.offset_head_2.bias)
        self.point_weights_2 = nn.Parameter(torch.full((1, NUM_SLOTS, 9, d), 1.0 / 9.0))
        self.patch_mixer_2: PatchTopologyMixer | None = (
            PatchTopologyMixer(d) if self.config.gen5_arch else None
        )
        self.sample_proj_2 = nn.Linear(d, d)
        self.intra_sa_2 = IntraSlotAttention(
            d, num_heads=self.config.num_heads, mlp_ratio=self.config.mlp_ratio, dropout=self.config.dropout
        )

        # 5. Decoupled Frame Evidence Heads (z, lambda_prec, z_color, lambda_color, polarity, valid)
        self.evidence_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.evidence_norm = nn.LayerNorm(d)

        self.frame_ctx_proj = nn.Sequential(
            nn.Linear(d * 3 + d // 2, d),
            nn.LayerNorm(d),
            nn.SiLU(),
        )
        self.frame_valid_head = nn.Linear(d, 1)
        nn.init.constant_(self.frame_valid_head.bias, 1.5)

        self.slot_quality_head = nn.Sequential(
            nn.Linear(d * 2, d // 2),
            nn.LayerNorm(d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, 1),
        )
        nn.init.constant_(self.slot_quality_head[-1].bias, 1.0)

        self.prec_head = nn.Sequential(
            nn.Linear(d * 2 + d // 2, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        nn.init.constant_(self.prec_head[-1].bias, -0.35)

        # Learned observation-noise scale from geometry; no optical offset rules.
        self.measurement_scale_head = nn.Sequential(
            nn.Linear(8, 16),
            nn.SiLU(),
            nn.Linear(16, 4),
        )
        nn.init.zeros_(self.measurement_scale_head[-1].weight)
        nn.init.zeros_(self.measurement_scale_head[-1].bias)

        # Training-only Text Gestalt stroke-mask decoder (zero BN layers; excluded from ONNX export)
        self.stroke_mask_head = nn.Sequential(
            nn.Conv2d(d, 16, kernel_size=3, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
        )

        # Polarity-aware rim + mid-plate + stroke-zone color & luminance polarity heads
        self.register_buffer(
            "color_masks_s1", _build_color_zone_masks(h2, w2, self.config.canonical_margin)
        )  # (1, 3, H2, W2)
        pol_in_dim = c1 * 2 + 4  # mid_feat(c1) + char_feat(c1) + [y_mid, y_char, y_diff, r_clip]
        self.polarity_head = nn.Sequential(
            nn.Linear(pol_in_dim, dc),
            nn.SiLU(),
            nn.Linear(dc, 2),  # [pol_logit, raw_pol_prec]
        )
        # rim_feat(c1) + mid_feat(c1) + chr_feat(c1) + rim_rgb(3) + mid_rgb(3) + chr_rgb(3)
        # + [y_mid, y_chr, y_mid - y_chr, tanh(0.25 * pol_logit)] (4) = c1 * 3 + 13
        color_in_dim = c1 * 3 + 13
        self.color_feat_head = nn.Sequential(
            nn.Linear(color_in_dim, dc * 2),
            nn.LayerNorm(dc * 2),
            nn.SiLU(),
            nn.Linear(dc * 2, dc),
            nn.LayerNorm(dc),
        )
        self.color_prec_head = nn.Sequential(
            nn.Linear(color_in_dim + d // 2, dc),
            nn.SiLU(),
            nn.Linear(dc, dc),
        )
        nn.init.constant_(self.color_prec_head[-1].bias, 0.25)

        # Exact OpenCV ksize=3 Laplacian kernel [[2,0,2],[0,-8,0],[2,0,2]] matching sequence_engine & quad_cropper
        lap_k = torch.tensor(
            [[[[2.0, 0.0, 2.0], [0.0, -8.0, 0.0], [2.0, 0.0, 2.0]]]], dtype=torch.float32
        )
        self.register_buffer("laplacian_kernel", lap_k)

        # 6. Anisotropic EvidenceMemoryV2 + Single Final Readout Heads
        self.memory_filter = EvidenceMemoryV2(d_model=d, d_color=dc,
            prior_variance=self.config.latent_prior_variance,
            nuisance_variance_init=self.config.nuisance_variance_init,
            time_constant_s_init=self.config.nuisance_time_constant_s_init)
        self.stopping_risk_head = nn.Sequential(nn.Linear(10, 16), nn.SiLU(), nn.Linear(16, 2), nn.Sigmoid())

        self.readout_slot_embed = nn.Parameter(torch.randn(1, NUM_SLOTS, d) * 0.02)
        ro_hidden = int(round(d * 1.5))
        self.readout_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, ro_hidden),
            nn.SiLU(),
            nn.Linear(ro_hidden, d),
        )
        self.readout_norm = nn.LayerNorm(d)
        if self.config.gen5_arch:
            self.readout_color_proj: nn.Linear | None = nn.Linear(dc, d)
            nn.init.zeros_(self.readout_color_proj.weight)
            nn.init.zeros_(self.readout_color_proj.bias)
            self.readout_sa: IntraSlotAttention | None = IntraSlotAttention(
                d, num_heads=self.config.num_heads, mlp_ratio=1.5, dropout=self.config.dropout
            )
        else:
            self.readout_color_proj = None
            self.readout_sa = None
        self.slot_classifier = nn.Linear(d, self.config.vocab_size)

        self.color_classifier = nn.Sequential(
            nn.Linear(dc, dc * 2),
            nn.SiLU(),
            nn.Linear(dc * 2, self.config.num_plate_types),
        )

        self.slot_readability_head = nn.Sequential(
            nn.Linear(d + 6, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, 1),
        )
        self.status_head = nn.Sequential(
            nn.Linear(d + 64, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, self.config.num_status_classes),
        )

    @property
    def num_parameters(self) -> int:
        """Total parameter count in current state (training or reparameterized)."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def reparameterize(self) -> MFJPPlateOCRV2:
        """In-place fuse all `RepConv3x3` & `RepConv1x1` branches into single inference Conv2ds."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (RepConv3x3, RepConv1x1)):
                m.reparameterize()
        self.config.deployed = True
        return self

    @torch.no_grad()
    def clone_reparameterized(self) -> MFJPPlateOCRV2:
        """Return a deep-copied, algebraically fused inference copy without mutating `self`.

        Conv+BN folding runs deterministically on CPU in FP32 before moving back to `self`'s device
        so CUDA-exported ONNX graphs and CPU-loaded checkpoints share bit-identical weights.
        """
        orig_device = next(self.parameters()).device
        cloned = copy.deepcopy(self).cpu()
        cloned.eval()
        cloned.reparameterize()
        return cloned.to(orig_device)

    @staticmethod
    def apply_zero_write_floor(raw_valid_logit: torch.Tensor) -> torch.Tensor:
        """Exact zero-write floor: frames with `sigmoid(raw) <= 0.05` produce exact `0.0` weight."""
        return torch.relu(torch.sigmoid(raw_valid_logit) - 0.05) / 0.95

    @staticmethod
    def compute_chroma_gate(s_chroma: torch.Tensor, r_clip: torch.Tensor) -> torch.Tensor:
        """Physical chroma gate: `clamp((S_chroma - 0.025)/0.075, 0, 1) * (1 - R_clip)^2`.

        Guarantees `0.0` identically on 850nm/940nm monochrome IR (`S_chroma < 0.015`).
        """
        c_gate = torch.clamp((s_chroma - 0.025) / 0.075, 0.0, 1.0)
        clip_gate = torch.clamp(1.0 - r_clip, 0.0, 1.0).pow(2)
        return c_gate * clip_gate

    def _default_geom_meta_from_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """Fallback pure-PyTorch `10-D geom_meta` computation when caller omits `geom_meta`."""
        n, _, h, w = crops.shape
        cf = crops.float()
        y_lum = 0.299 * cf[:, 0:1] + 0.587 * cf[:, 1:2] + 0.114 * cf[:, 2:3]  # (N, 1, H, W)
        mu_y = torch.clamp(y_lum.mean(dim=(1, 2, 3)), 0.0, 1.0)
        r_clip = (y_lum > 0.96).float().mean(dim=(1, 2, 3))

        chroma_pix = (cf.max(dim=1).values - cf.min(dim=1).values).reshape(n, -1)
        top_k = max(1, (h * w * 15) // 100)
        s_chroma = torch.clamp(torch.topk(chroma_pix, k=top_k, dim=-1).values.mean(dim=-1), 0.0, 1.0)

        y_pad = F.pad(y_lum, (1, 1, 1, 1), mode="reflect")
        lap = F.conv2d(y_pad, self.laplacian_kernel.float())
        lap_var = lap.flatten(1).var(dim=-1, unbiased=False)
        sharp = torch.clamp(0.25 * torch.log1p(100.0 * lap_var), 0.0, 1.0)

        zeros = torch.zeros_like(mu_y)
        g0 = torch.full_like(mu_y, float(min(1.0, max(-1.0, 0.5 * math.log2(max(1e-3, w * 0.9 / 64.0))))))
        g1 = torch.full_like(mu_y, float(min(1.0, max(-1.0, 0.5 * math.log2(max(1e-3, h * 0.9 / 32.0))))))
        return torch.stack([g0, g1, zeros, zeros, zeros, zeros, sharp, mu_y, s_chroma, r_clip], dim=-1)

    def _gather_asymmetric_slots(
        self,
        f2: torch.Tensor,
        f4: torch.Tensor,
        f8: torch.Tensor,
        centers_uv: torch.Tensor,
        scales_uv: torch.Tensor,
        point_weights: torch.Tensor,
        proj: nn.Linear,
        base_steps_uv: torch.Tensor | None = None,
        patch_mixer: PatchTopologyMixer | None = None,
    ) -> torch.Tensor:
        """Sample `3x3` local grids: slots `0..6` from gated `(F2, F4)`, slots `7..11` from gated `(F4, F8)`."""
        n, d, _, _ = f4.shape
        ref_steps = (
            base_steps_uv
            if base_steps_uv is not None
            else self.slot_steps_uv.to(dtype=f4.dtype).unsqueeze(0)
        )
        steps = ref_steps * scales_uv  # (N, 12, 2)
        grid = self.local_grid_3x3.to(dtype=f4.dtype).reshape(1, 1, 9, 2)
        coords = centers_uv.unsqueeze(2) + grid * steps.unsqueeze(2)  # (N, 12, 9, 2)

        flat_all = coords.reshape(n, NUM_SLOTS * 9, 2)
        v_f4 = contiguous_bilinear_gather(f4, flat_all).reshape(n, NUM_SLOTS, 9, d)

        flat_upper = coords[:, 0:7].reshape(n, 7 * 9, 2)
        v_f2_upper = contiguous_bilinear_gather(f2, flat_upper).reshape(n, 7, 9, d)
        u_gate = torch.sigmoid(self.upper_scale_gate.to(dtype=f4.dtype))  # (1, 7, 1, D)
        v_upper = u_gate * v_f2_upper + (1.0 - u_gate) * v_f4[:, 0:7]

        flat_lower = coords[:, 7:12].reshape(n, 5 * 9, 2)
        v_f8_lower = contiguous_bilinear_gather(f8, flat_lower).reshape(n, 5, 9, d)

        gate = torch.sigmoid(self.lower_scale_gate.to(dtype=f4.dtype))  # (1, 5, 1, D)
        v_lower = gate * v_f4[:, 7:12] + (1.0 - gate) * v_f8_lower
        v_combined = torch.cat([v_upper, v_lower], dim=1)  # (N, 12, 9, D)

        pw = point_weights.to(dtype=f4.dtype)  # (1, 12, 9, D)
        weighted = (v_combined * pw).sum(dim=2)  # (N, 12, D)
        if patch_mixer is not None:
            weighted = weighted + patch_mixer(v_combined)
        return proj(weighted)

    def decode_stroke_mask(
        self, f4: torch.Tensor, out_hw: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """Training-only `Text Gestalt` stroke topology decoder on stride-4 `F4` features."""
        raw_s4 = self.stroke_mask_head(f4)
        if out_hw is not None and raw_s4.shape[-2:] != out_hw:
            return F.interpolate(raw_s4, size=out_hw, mode="bilinear", align_corners=False)
        return raw_s4

    def _encode_frames_impl(
        self,
        crops: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        geom_meta: torch.Tensor | None = None,
        return_f4: bool = False,
        return_len_logits: bool = False,
    ) -> tuple[FrameEncodingV2, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Internal encoder returning `(FrameEncodingV2, optional_f4, reg_len, cls_len)` cleanly."""
        n, _, h_img, w_img = crops.shape
        if plate_sides is None:
            plate_sides = torch.zeros(n, dtype=torch.long, device=crops.device)
        if geom_meta is None:
            geom_meta = self._default_geom_meta_from_crops(crops)

        # 1. Stem + Full-Res Stride-1 Pre-Stem + RepMobile Backbone
        x0 = self.photo_stem(crops)           # (N, 4, H, W)
        if self.pre_s1 is not None:
            x0 = self.pre_s1(x0)              # (N, c0, H, W) resolves 7.8px Kanji at 1.0px/cell!
        x2 = self.stage1(self.ds1(x0))        # (N, c1, H/2, W/2)
        x4 = self.stage2(self.ds2(x2))        # (N, c2, H/4, W/4)
        x8 = self.stage3(self.ds3(x4))        # (N, c3, H/8, W/8)

        # 2. Asymmetric Multi-Scale FPN (F2: stride 2, F4: stride 4, F8: stride 8)
        l8 = self.f8_lat(x8)
        l8_up = F.interpolate(l8, scale_factor=2.0, mode="bilinear", align_corners=False)
        f4_feat = self.f4_refine(self.f4_lat(x4) + l8_up)
        f8_feat = self.f8_refine(l8)
        f4_up = F.interpolate(f4_feat, scale_factor=2.0, mode="bilinear", align_corners=False)
        f2_raw = self.f2_lat(x2) + f4_up
        f2_feat = self.f2_refine(f2_raw) if self.f2_refine is not None else f2_raw

        f2 = _add_separable_2d_pos_v2(f2_feat, self.pos_f2_y, self.pos_f2_x)
        f4 = _add_separable_2d_pos_v2(f4_feat, self.pos_f4_y, self.pos_f4_x)
        f8 = _add_separable_2d_pos_v2(f8_feat, self.pos_f8_y, self.pos_f8_x)

        # 3. Row-Strip Alignment Vector a_t
        h4, w4 = f4.shape[-2], f4.shape[-1]
        half_h4 = h4 // 2
        basis = self.horiz_basis_w4.to(dtype=f4.dtype)
        if not torch.jit.is_tracing() and w4 != basis.shape[0]:
            basis = F.interpolate(
                basis.T.unsqueeze(0), size=w4, mode="linear", align_corners=False
            ).squeeze(0).T
            basis = basis / torch.clamp(basis.sum(dim=0, keepdim=True), min=1e-6)

        up_strip = torch.matmul(f4[:, :, :half_h4, :].mean(dim=2), basis)  # (N, D, 8)
        lo_strip = torch.matmul(f4[:, :, half_h4:, :].mean(dim=2), basis)  # (N, D, 8)
        up_red = self.upper_ch_reduce(up_strip.transpose(1, 2)).reshape(n, -1)
        lo_red = self.lower_ch_reduce(lo_strip.transpose(1, 2)).reshape(n, -1)
        up_feat = F.silu(self.upper_prof_proj(up_red))
        lo_feat = F.silu(self.lower_prof_proj(lo_red))

        sides_clamped = torch.clamp(plate_sides.long(), 0, 2)
        side_emb = self.plate_side_embed(sides_clamped)
        geom_in = geom_meta.to(dtype=f4.dtype)
        geom_emb = self.geom_mlp(geom_in)
        a_t = self.align_mlp(torch.cat([up_feat, lo_feat, side_emb, geom_emb], dim=-1))  # (N, D)

        # 4. Statutory Length-Conditioned Slot Anchor Mixing + 2-Pass Local Deformable Sampler
        reg_len_logits: torch.Tensor | None = None
        cls_len_logits: torch.Tensor | None = None
        if self.reg_len_head is not None and self.cls_len_head is not None:
            reg_len_logits = self.reg_len_head(up_feat)  # (N, 4)
            cls_len_logits = self.cls_len_head(up_feat)  # (N, 3)
            p_reg = F.softmax(reg_len_logits, dim=-1)
            p_cls = F.softmax(cls_len_logits, dim=-1)
            reg_anc = torch.matmul(
                p_reg, self.reg_len_anchors_uv.to(dtype=f4.dtype).reshape(4, 8)
            ).reshape(-1, 4, 2)
            reg_stp = torch.matmul(
                p_reg, self.reg_len_steps_uv.to(dtype=f4.dtype).reshape(4, 8)
            ).reshape(-1, 4, 2)
            cls_anc = torch.matmul(
                p_cls, self.cls_len_anchors_uv.to(dtype=f4.dtype).reshape(3, 6)
            ).reshape(-1, 3, 2)
            cls_stp = torch.matmul(
                p_cls, self.cls_len_steps_uv.to(dtype=f4.dtype).reshape(3, 6)
            ).reshape(-1, 3, 2)
            zero_n12 = reg_anc[:, :1, :] * 0.0  # (N, 1, 2) pure broadcast without Shape/Expand
            low_anc = zero_n12 + self.slot_anchors_uv[7:12].to(dtype=f4.dtype).unsqueeze(0)
            low_stp = zero_n12 + self.slot_steps_uv[7:12].to(dtype=f4.dtype).unsqueeze(0)
            base_anchors_uv = torch.cat([reg_anc, cls_anc, low_anc], dim=1)  # (N, 12, 2)
            base_steps_uv: torch.Tensor | None = torch.cat([reg_stp, cls_stp, low_stp], dim=1)
        else:
            base_anchors_uv = self.slot_anchors_uv.to(dtype=f4.dtype).unsqueeze(0)
            base_steps_uv = None

        slot_q0 = self.slot_embed.to(dtype=f4.dtype) + a_t.unsqueeze(1)  # (N, 12, D)
        raw_off1 = self.offset_head_1(slot_q0)  # (N, 12, 4)
        # +/-0.18 horizontal == +/-28.8 px on W=160; +/-0.10 vertical == +/-8 px on H=80 (recessed bumper resilience)
        dc1_x = 0.18 * torch.tanh(raw_off1[:, :, 0])
        dc1_y = 0.10 * torch.tanh(raw_off1[:, :, 1])
        ds1 = 1.0 + 0.25 * torch.tanh(raw_off1[:, :, 2:4])
        c1_uv = base_anchors_uv + torch.stack([dc1_x, dc1_y], dim=-1)

        s1_sampled = self._gather_asymmetric_slots(
            f2,
            f4,
            f8,
            c1_uv,
            ds1,
            self.point_weights_1,
            self.sample_proj_1,
            base_steps_uv=base_steps_uv,
            patch_mixer=self.patch_mixer_1,
        )
        h_pass1 = self.intra_sa_1(slot_q0 + s1_sampled)  # (N, 12, D)

        raw_off2 = self.offset_head_2(h_pass1)
        tot_dx = 0.18 * torch.tanh(raw_off1[:, :, 0] + raw_off2[:, :, 0])
        tot_dy = 0.10 * torch.tanh(raw_off1[:, :, 1] + raw_off2[:, :, 1])
        ds2 = 1.0 + 0.25 * torch.tanh(raw_off1[:, :, 2:4] + raw_off2[:, :, 2:4])
        c2_uv = base_anchors_uv + torch.stack([tot_dx, tot_dy], dim=-1)

        s2_sampled = self._gather_asymmetric_slots(
            f2,
            f4,
            f8,
            c2_uv,
            ds2,
            self.point_weights_2,
            self.sample_proj_2,
            base_steps_uv=base_steps_uv,
            patch_mixer=self.patch_mixer_2,
        )
        slot_feat = self.intra_sa_2(h_pass1 + s2_sampled)  # (N, 12, D)

        # 5. Decoupled latent measurements and learned anisotropic measurement precision
        z = self.evidence_norm(slot_feat + self.evidence_mlp(slot_feat))  # (N, 12, D)

        f4_mean = f4.mean(dim=(2, 3))
        f8_mean = f8.mean(dim=(2, 3))
        sl_mean = z.mean(dim=1)
        frame_ctx = self.frame_ctx_proj(
            torch.cat([f4_mean, f8_mean, sl_mean, geom_emb], dim=-1)
        )  # (N, D)

        frame_valid_logit = self.frame_valid_head(frame_ctx).squeeze(-1)  # (N,)
        frame_valid_gate = self.apply_zero_write_floor(frame_valid_logit)  # (N,)

        ctx_exp = frame_ctx.unsqueeze(1).expand(-1, NUM_SLOTS, -1)
        slot_quality_logits = self.slot_quality_head(
            torch.cat([slot_feat, ctx_exp], dim=-1)
        ).squeeze(-1)  # (N, 12)
        slot_q_gate = torch.sigmoid(slot_quality_logits)  # (N, 12)

        # Use whitelisted Neg + Max instead of Abs so ONNX symbolic prover remains 100% whitelisted
        abs_g2 = torch.maximum(geom_in[:, 2], -geom_in[:, 2])
        abs_g3 = torch.maximum(geom_in[:, 3], -geom_in[:, 3])
        abs_g4 = torch.maximum(geom_in[:, 4], -geom_in[:, 4])
        noise_features = torch.stack(
            [
                geom_in[:, 0],
                geom_in[:, 1],
                abs_g2,
                abs_g3,
                abs_g4,
                geom_in[:, 5],
                geom_in[:, 6],
                geom_in[:, 9],
            ],
            dim=-1,
        )  # (N, 8)
        noise_scale_group = torch.sigmoid(self.measurement_scale_head(noise_features))
        noise_scale_slots = torch.cat(
            [
                noise_scale_group[:, 0:1].expand(-1, 4),
                noise_scale_group[:, 1:2].expand(-1, 3),
                noise_scale_group[:, 2:3],
                noise_scale_group[:, 3:4].expand(-1, 4),
            ],
            dim=1,
        )  # (N, 12)

        a_exp = a_t.unsqueeze(1).expand(-1, NUM_SLOTS, -1)
        g_exp = geom_emb.unsqueeze(1).expand(-1, NUM_SLOTS, -1)
        raw_prec = torch.clamp(
            F.softplus(self.prec_head(torch.cat([slot_feat, a_exp, g_exp], dim=-1))),
            max=6.0,
        )
        lambda_prec = (
            raw_prec.float()
            * noise_scale_slots.float()[:, :, None]
            * frame_valid_gate.float()[:, None, None]
            * slot_q_gate.float()[:, :, None]
        )  # (N, 12, D)

        # Polarity-first extraction so color_feat_head & color_prec_head see stroke vs background
        # luminance contrast + polarity sign (separating dark commercial green + white strokes from
        # cool-lit white/pastel graphic plates with dark green strokes!).
        cmasks = self.color_masks_s1.to(dtype=x2.dtype)
        if not torch.jit.is_tracing() and x2.shape[-2:] != cmasks.shape[-2:]:
            cmasks = F.interpolate(cmasks, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            cmasks = cmasks / torch.clamp(cmasks.sum(dim=(2, 3), keepdim=True), min=1e-6)

        rgb_s1 = F.avg_pool2d(crops.to(dtype=x2.dtype), kernel_size=2, stride=2)
        rim_m = cmasks[:, 0:1]   # (1, 1, H2, W2)
        mid_m = cmasks[:, 1:2]
        chr_m = cmasks[:, 2:3]

        rim_feat = (x2 * rim_m).sum(dim=(2, 3))
        mid_feat = (x2 * mid_m).sum(dim=(2, 3))
        chr_feat = (x2 * chr_m).sum(dim=(2, 3))

        rim_rgb = (rgb_s1 * rim_m).sum(dim=(2, 3))
        mid_rgb = (rgb_s1 * mid_m).sum(dim=(2, 3))
        chr_rgb = (rgb_s1 * chr_m).sum(dim=(2, 3))

        s_chroma = geom_meta[:, 8].float()
        r_clip = geom_meta[:, 9].float()

        y_mid = 0.299 * mid_rgb[:, 0] + 0.587 * mid_rgb[:, 1] + 0.114 * mid_rgb[:, 2]
        y_chr = 0.299 * chr_rgb[:, 0] + 0.587 * chr_rgb[:, 1] + 0.114 * chr_rgb[:, 2]
        y_diff = y_mid - y_chr
        pol_stats = torch.stack(
            [y_mid, y_chr, y_diff, r_clip.to(dtype=x2.dtype)], dim=-1
        )
        pol_out = self.polarity_head(torch.cat([mid_feat, chr_feat, pol_stats], dim=-1))
        # Physical luminance difference residual anchors polarity sign; clamp to [-8, 8] (sigmoid=0.9997) for bounded FP32/INT8 state
        color_noise_scale = noise_scale_group[:, 3].float()
        pol_logit = torch.clamp(pol_out[:, 0].float() + 2.5 * y_diff.float(), -8.0, 8.0)
        raw_pol_prec = torch.clamp(F.softplus(pol_out[:, 1].float()), max=4.0)
        pol_prec = (
            raw_pol_prec
            * color_noise_scale
            * frame_valid_gate.float()
            * torch.clamp(1.0 - r_clip, 0.0, 1.0).pow(2)
        )

        pol_hint = torch.tanh(0.25 * pol_logit).to(dtype=x2.dtype)
        col_pol_stats = torch.stack([y_mid, y_chr, y_diff, pol_hint], dim=-1)
        col_in = torch.cat(
            [rim_feat, mid_feat, chr_feat, rim_rgb, mid_rgb, chr_rgb, col_pol_stats],
            dim=-1,
        )
        z_color = self.color_feat_head(col_in)  # (N, Dc)
        raw_col_prec = torch.clamp(
            F.softplus(self.color_prec_head(torch.cat([col_in, geom_emb], dim=-1))),
            max=6.0,
        )

        w_chroma = self.compute_chroma_gate(s_chroma, r_clip)  # == 0.0 on 850/940nm IR!
        lambda_color = (
            raw_col_prec.float()
            * color_noise_scale[:, None]
            * w_chroma[:, None]
            * frame_valid_gate.float()[:, None]
        )  # (N, Dc)

        enc_out = FrameEncodingV2(
            z=z.float(),
            lambda_prec=lambda_prec,
            z_color=z_color.float(),
            lambda_color=lambda_color,
            pol_logit=pol_logit,
            pol_prec=pol_prec,
            frame_valid_logit=frame_valid_logit.float(),
            slot_quality_logits=slot_quality_logits.float(),
            frame_valid_gate=frame_valid_gate.float(),
        )
        return (
            enc_out,
            (f4_feat if return_f4 else None),
            (reg_len_logits.float() if (return_len_logits and reg_len_logits is not None) else None),
            (cls_len_logits.float() if (return_len_logits and cls_len_logits is not None) else None),
        )

    def encode_frames(
        self,
        crops: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        geom_meta: torch.Tensor | None = None,
    ) -> FrameEncodingV2:
        """Encode `(N, 3, H, W)` canonical crops into decoupled latent & precision observations."""
        enc_out, _, _, _ = self._encode_frames_impl(
            crops, plate_sides, geom_meta, return_f4=False, return_len_logits=False
        )
        return enc_out

    def init_memory(
        self,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> MemoryStateV2:
        return self.memory_filter.init_state(batch_size=batch_size, device=device, dtype=dtype)

    def step_memory(
        self,
        mem_h: torch.Tensor,
        mem_P: torch.Tensor,
        mem_conflict: torch.Tensor,
        color_h: torch.Tensor,
        color_P: torch.Tensor,
        pol_h: torch.Tensor,
        pol_P: torch.Tensor,
        pose_ema: torch.Tensor,
        last_timestamp_s: torch.Tensor,
        z: torch.Tensor,
        lambda_prec: torch.Tensor,
        z_color: torch.Tensor,
        lambda_color: torch.Tensor,
        pol_logit: torch.Tensor,
        pol_prec: torch.Tensor,
        geom_meta: torch.Tensor,
        timestamp_s: torch.Tensor,
        step_mask: torch.Tensor | None = None,
    ) -> MemoryStateV2:
        return self.memory_filter.step(
            mem_h=mem_h,
            mem_P=mem_P,
            mem_conflict=mem_conflict,
            color_h=color_h,
            color_P=color_P,
            pol_h=pol_h,
            pol_P=pol_P,
            pose_ema=pose_ema,
            last_timestamp_s=last_timestamp_s,
            z=z,
            lambda_prec=lambda_prec,
            z_color=z_color,
            lambda_color=lambda_color,
            pol_logit=pol_logit,
            pol_prec=pol_prec,
            geom_meta=geom_meta,
            timestamp_s=timestamp_s,
            step_mask=step_mask,
        )

    def refine_slot_features(self, feat: torch.Tensor) -> torch.Tensor:
        slot_emb = self.readout_slot_embed.to(dtype=feat.dtype)
        return self.readout_norm(feat + self.readout_mlp(feat + slot_emb))

    def decode_slot_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Shared per-slot character classifier for readout and training-only auxiliary supervision."""
        ref = self.refine_slot_features(feat)
        if self.readout_sa is not None:
            orig_shape = ref.shape
            ref_3d = ref.reshape(-1, NUM_SLOTS, self.config.d_model)
            ref = self.readout_sa(ref_3d).reshape(orig_shape)
        return self.slot_classifier(ref)

    def decode_color_features(self, z_color: torch.Tensor) -> torch.Tensor:
        """Shared color classifier for decoupled `color_h / color_P` readout and RGB supervision."""
        return self.color_classifier(z_color)

    def decode_memory(
        self,
        mem_h: torch.Tensor,
        mem_P: torch.Tensor,
        mem_conflict: torch.Tensor,
        color_h: torch.Tensor,
        color_P: torch.Tensor,
        pol_h: torch.Tensor,
        pol_P: torch.Tensor,
    ) -> MemoryReadoutV2:
        """Single final readout from natural-parameter `EvidenceMemoryV2`."""
        mh = mem_h.float()
        mp = torch.clamp(mem_P.float(), min=0.0)
        mc = torch.clamp(mem_conflict.float(), min=0.0)
        ch = color_h.float()
        cp = torch.clamp(color_P.float(), min=0.0)

        color_mu = ch / (cp + 1e-5)
        color_obs = cp / (cp + 0.05)
        color_vec = (color_mu * color_obs).to(dtype=self.readout_slot_embed.dtype)
        plate_type_logits = self.color_classifier(color_vec)
        polarity_logit = pol_h.float() / (torch.clamp(pol_P.float(), min=0.0) + 1e-5)

        mu = mh / (mp + 1e-5)
        obs_gate = mp / (mp + 0.05)
        slot_in = (mu * obs_gate).to(dtype=self.readout_slot_embed.dtype)
        if self.readout_color_proj is not None and self.readout_sa is not None:
            color_ctx = self.readout_color_proj(color_vec).unsqueeze(1)  # (B, 1, D)
            h = self.readout_sa(self.refine_slot_features(slot_in + color_ctx))
        else:
            h = self.refine_slot_features(slot_in)
        slot_logits = self.slot_classifier(h)  # (B, 12, V)

        probs = F.softmax(slot_logits.float(), dim=-1)
        max_prob = probs.max(dim=-1).values
        pad_prob = probs[:, :, 0]
        log_v = math.log(float(self.config.vocab_size))
        norm_entropy = -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=-1) / log_v

        slot_mass = torch.clamp(mp.mean(dim=-1), 0.0, 6.0)
        slot_p_std = torch.clamp(mp.std(dim=-1, unbiased=False), 0.0, 4.0)
        log_mass = torch.log1p(slot_mass)
        clamped_conf = torch.clamp(mc, 0.0, 6.0)

        read_in = torch.cat(
            [
                h.float(),
                log_mass.unsqueeze(-1),
                slot_mass.unsqueeze(-1),
                slot_p_std.unsqueeze(-1),
                max_prob.unsqueeze(-1),
                norm_entropy.unsqueeze(-1),
                clamped_conf.unsqueeze(-1),
            ],
            dim=-1,
        )  # (B, 12, D + 6)
        slot_readability_pred = torch.sigmoid(
            self.slot_readability_head(read_in.to(dtype=h.dtype)).squeeze(-1)
        )

        active_prob = 1.0 - pad_prob
        inactive_boost = (pad_prob > 0.65).float()
        eff_slot_read = torch.clamp(slot_readability_pred.float() + inactive_boost, 0.0, 1.0)
        min_active_read = eff_slot_read.min(dim=-1, keepdim=True).values
        eff_slot_mass = slot_mass + inactive_boost * 6.0
        min_active_mass = eff_slot_mass.min(dim=-1, keepdim=True).values
        max_conflict = clamped_conf.max(dim=-1, keepdim=True).values
        mean_conflict = clamped_conf.mean(dim=-1, keepdim=True)

        status_in = torch.cat(
            [
                h.float().mean(dim=1),
                log_mass,
                slot_readability_pred.float(),
                clamped_conf,
                max_prob,
                active_prob,
                min_active_read,
                min_active_mass,
                max_conflict,
                mean_conflict,
            ],
            dim=-1,
        )  # (B, D + 64)
        status_logits = self.status_head(status_in.to(dtype=h.dtype))

        return MemoryReadoutV2(
            slot_logits=slot_logits,
            plate_type_logits=plate_type_logits,
            polarity_logit=polarity_logit,
            slot_readability_pred=slot_readability_pred,
            status_logits=status_logits,
        )

    @torch.no_grad()
    def reset_canonical_slot_anchors(self, reset_offset_heads: bool = True) -> None:
        """Refresh `slot_anchors_uv` and `slot_steps_uv` buffers to calibrated MLIT anchors.

        Call after warm-starting from an older checkpoint so `load_state_dict` does not leave stale
        anchor buffers or stale Pass-1/Pass-2 offset biases in place.
        """
        anchors, steps = _build_canonical_slot_anchors(self.config.canonical_margin)
        self.slot_anchors_uv.copy_(anchors.to(device=self.slot_anchors_uv.device, dtype=self.slot_anchors_uv.dtype))
        self.slot_steps_uv.copy_(steps.to(device=self.slot_steps_uv.device, dtype=self.slot_steps_uv.dtype))
        if self.config.gen5_arch and hasattr(self, "reg_len_anchors_uv"):
            reg_anc, reg_stp, cls_anc, cls_stp = _build_statutory_length_anchor_tables(
                self.config.canonical_margin
            )
            dev_a = self.slot_anchors_uv.device
            dt_a = self.slot_anchors_uv.dtype
            self.reg_len_anchors_uv.copy_(reg_anc.to(device=dev_a, dtype=dt_a))
            self.reg_len_steps_uv.copy_(reg_stp.to(device=dev_a, dtype=dt_a))
            self.cls_len_anchors_uv.copy_(cls_anc.to(device=dev_a, dtype=dt_a))
            self.cls_len_steps_uv.copy_(cls_stp.to(device=dev_a, dtype=dt_a))
        if reset_offset_heads:
            nn.init.zeros_(self.offset_head_1.weight)
            nn.init.zeros_(self.offset_head_1.bias)
            nn.init.zeros_(self.offset_head_2.weight)
            nn.init.zeros_(self.offset_head_2.bias)

    def forward_sequence(
        self,
        crops: torch.Tensor,
        timestamps_s: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        geom_meta: torch.Tensor | None = None,
        frame_pad_mask: torch.Tensor | None = None,
        compute_prefix_readout: bool = False,
        prefix_len: int | None = None,
        compute_stroke_mask: bool = False,
        init_state: MemoryStateV2 | None = None,
    ) -> SequenceOutputV2:
        """Run frame encoding across `(B, T, 3, H, W)` and O(1) natural-parameter rollout."""
        bsz, seq_len, ch, h, w = crops.shape
        if plate_sides is None:
            plate_sides = torch.zeros((bsz, seq_len), dtype=torch.long, device=crops.device)

        flat_crops = crops.reshape(bsz * seq_len, ch, h, w)
        flat_sides = plate_sides.reshape(bsz * seq_len)
        if geom_meta is None:
            flat_geom = self._default_geom_meta_from_crops(flat_crops)
            geom_meta = flat_geom.reshape(bsz, seq_len, self.config.geom_dim)
        else:
            flat_geom = geom_meta.reshape(bsz * seq_len, self.config.geom_dim)

        enc, f4_raw, flat_reg_len, flat_cls_len = self._encode_frames_impl(
            flat_crops,
            flat_sides,
            flat_geom,
            return_f4=bool(compute_stroke_mask),
            return_len_logits=True,
        )
        stroke_mask_logits: torch.Tensor | None = None
        if compute_stroke_mask and f4_raw is not None:
            sm_flat = self.decode_stroke_mask(f4_raw, out_hw=(h, w))
            stroke_mask_logits = sm_flat.reshape(bsz, seq_len, 1, h, w)
        reg_len_seq = (
            flat_reg_len.reshape(bsz, seq_len, 4) if flat_reg_len is not None else None
        )
        cls_len_seq = (
            flat_cls_len.reshape(bsz, seq_len, 3) if flat_cls_len is not None else None
        )

        d = self.config.d_model
        dc = self.config.d_color
        frame_z = enc.z.reshape(bsz, seq_len, NUM_SLOTS, d)
        frame_lam = enc.lambda_prec.reshape(bsz, seq_len, NUM_SLOTS, d)
        frame_zc = enc.z_color.reshape(bsz, seq_len, dc)
        frame_lam_c = enc.lambda_color.reshape(bsz, seq_len, dc)
        frame_pol = enc.pol_logit.reshape(bsz, seq_len)
        frame_pol_p = enc.pol_prec.reshape(bsz, seq_len)
        frame_v_log = enc.frame_valid_logit.reshape(bsz, seq_len)
        frame_q_log = enc.slot_quality_logits.reshape(bsz, seq_len, NUM_SLOTS)
        frame_v_gate = enc.frame_valid_gate.reshape(bsz, seq_len)

        if frame_pad_mask is not None:
            step_valid = (~frame_pad_mask).float()
        else:
            step_valid = torch.ones((bsz, seq_len), dtype=torch.float32, device=crops.device)

        state = (
            init_state
            if init_state is not None
            else self.init_memory(batch_size=bsz, device=crops.device, dtype=torch.float32)
        )
        if prefix_len is not None:
            t_prefix = max(1, min(int(prefix_len), max(1, seq_len - 1)))
        elif compute_prefix_readout and self.training and seq_len >= 2:
            # Sample K in {1 .. seq_len - 1} with strong K=1..2 emphasis so status_head and
            # slot_readability_pred are calibrated across single-photo and short-track regimes.
            u_rnd = float(torch.rand((), device="cpu").item())
            if u_rnd < 0.40:
                t_prefix = 1
            elif u_rnd < 0.65 and seq_len >= 3:
                t_prefix = 2
            else:
                hi = max(2, seq_len)
                t_prefix = int(torch.randint(1, hi, (), device="cpu").item())
        else:
            t_prefix = max(1, seq_len // 3)
        prefix_slot_logits: torch.Tensor | None = None
        prefix_readout: MemoryReadoutV2 | None = None
        used_prefix_len = 0

        innovation_losses = []
        prefix_risk = prefix_next_risk = next_prefix_readout = None
        prefix_memory = next_prefix_memory = None
        for t in range(seq_len):
            state, nll, active = self.memory_filter.update_with_innovation(
                mem_h=state.mem_h,
                mem_P=state.mem_P,
                mem_conflict=state.mem_conflict,
                color_h=state.color_h,
                color_P=state.color_P,
                pol_h=state.pol_h,
                pol_P=state.pol_P,
                pose_ema=state.pose_ema,
                last_timestamp_s=state.last_timestamp_s,
                z=frame_z[:, t],
                lambda_prec=frame_lam[:, t],
                z_color=frame_zc[:, t],
                lambda_color=frame_lam_c[:, t],
                pol_logit=frame_pol[:, t],
                pol_prec=frame_pol_p[:, t],
                geom_meta=geom_meta[:, t],
                timestamp_s=timestamps_s[:, t],
                step_mask=step_valid[:, t],
            )
            innovation_losses.append(nll.sum() / active.float().sum().clamp_min(1))
            if compute_prefix_readout and (t + 1) == t_prefix:
                q = noncharacter_quality_features(state, (frame_lam[:, :t+1].mean((2, 3)) > 0).float().mean(1), geom_meta[:, :t+1, :6].float().var(1, unbiased=False).mean(-1))
                prefix_memory = state
                risks = self.stopping_risk_head(q)
                prefix_risk, prefix_next_risk = risks[:, 0], risks[:, 1]
            if compute_prefix_readout and seq_len >= 2 and (t + 1) == t_prefix+1:
                next_prefix_memory = state
                next_prefix_readout = self.decode_memory(*state[:7])
            if compute_prefix_readout and (t + 1) == t_prefix:
                prefix_readout = self.decode_memory(
                    state.mem_h,
                    state.mem_P,
                    state.mem_conflict,
                    state.color_h,
                    state.color_P,
                    state.pol_h,
                    state.pol_P,
                )
                prefix_slot_logits = prefix_readout.slot_logits
                used_prefix_len = t_prefix

        ro = self.decode_memory(
            state.mem_h,
            state.mem_P,
            state.mem_conflict,
            state.color_h,
            state.color_P,
            state.pol_h,
            state.pol_P,
        )
        return SequenceOutputV2(
            slot_logits=ro.slot_logits,
            plate_type_logits=ro.plate_type_logits,
            polarity_logit=ro.polarity_logit,
            slot_readability_pred=ro.slot_readability_pred,
            status_logits=ro.status_logits,
            memory=state,
            frame_z=frame_z,
            frame_lambda_prec=frame_lam,
            frame_z_color=frame_zc,
            frame_lambda_color=frame_lam_c,
            frame_pol_logit=frame_pol,
            frame_pol_prec=frame_pol_p,
            frame_valid_logit=frame_v_log,
            slot_quality_logits=frame_q_log,
            frame_valid_gate=frame_v_gate,
            prefix_slot_logits=prefix_slot_logits,
            prefix_readout=prefix_readout,
            prefix_len=used_prefix_len,
            stroke_mask_logits=stroke_mask_logits,
            reg_len_logits=reg_len_seq,
            cls_len_logits=cls_len_seq,
            prefix_memory=prefix_memory, next_prefix_memory=next_prefix_memory,
            prefix_risk=prefix_risk, prefix_next_risk=prefix_next_risk,
            next_prefix_readout=next_prefix_readout,
            innovation_nll=torch.stack(innovation_losses).mean(),
        )

    def forward(
        self,
        crops: torch.Tensor,
        timestamps_s: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        geom_meta: torch.Tensor | None = None,
        frame_pad_mask: torch.Tensor | None = None,
    ) -> SequenceOutputV2:
        return self.forward_sequence(
            crops=crops,
            timestamps_s=timestamps_s,
            plate_sides=plate_sides,
            geom_meta=geom_meta,
            frame_pad_mask=frame_pad_mask,
        )
