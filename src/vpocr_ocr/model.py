"""Core MF-JPPlateOCR architecture with latent O(1) EvidenceMemory (patent-clean)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from vpocr_ocr.vocab import NUM_SLOTS


@dataclass
class ModelConfig:
    """Hyperparameters for MFJPPlateOCR (supports 'base', 'small', and 'nano' presets)."""

    vocab_size: int = 254
    num_plate_types: int = 5
    num_status_classes: int = 3
    d_model: int = 160
    num_heads: int = 5
    num_decoder_layers: int = 2
    mlp_ratio: float = 3.0
    dropout: float = 0.0
    stem_dim: int = 48
    stage2_dim: int = 96
    stem_blocks: int = 2
    stage2_blocks: int = 3
    stage3_blocks: int = 4
    conv_expansion: int = 2

    @classmethod
    def from_preset(cls, preset: str = "base", vocab_size: int = 254) -> ModelConfig:
        """Create architecture configuration from named preset ('base', 'small', 'nano')."""
        p = preset.lower().strip()
        if p == "base":
            return cls(
                vocab_size=vocab_size,
                d_model=160,
                num_heads=5,
                num_decoder_layers=2,
                mlp_ratio=3.0,
                stem_dim=48,
                stage2_dim=96,
                stem_blocks=2,
                stage2_blocks=3,
                stage3_blocks=4,
                conv_expansion=2,
            )
        if p == "small":
            return cls(
                vocab_size=vocab_size,
                d_model=128,
                num_heads=4,
                num_decoder_layers=2,
                mlp_ratio=2.5,
                stem_dim=32,
                stage2_dim=64,
                stem_blocks=1,
                stage2_blocks=2,
                stage3_blocks=3,
                conv_expansion=2,
            )
        if p == "nano":
            return cls(
                vocab_size=vocab_size,
                d_model=96,
                num_heads=4,
                num_decoder_layers=1,
                mlp_ratio=2.0,
                stem_dim=24,
                stage2_dim=48,
                stem_blocks=1,
                stage2_blocks=2,
                stage3_blocks=2,
                conv_expansion=2,
            )
        raise ValueError(f"Unknown ModelConfig preset: {preset!r} (expected 'base', 'small', or 'nano')")


class FrameEncoding(NamedTuple):
    """Per-frame spatial slot encoding outputs."""

    z: torch.Tensor                        # (N, 12, D)
    slot_quality: torch.Tensor             # (N, 12) in [0, 1] with exact zero floor
    frame_valid: torch.Tensor              # (N, 1) in [0, 1] with exact zero-write floor
    raw_valid_logit: torch.Tensor          # (N, 1) pre-sigmoid logit for calibration loss
    raw_slot_quality_logit: torch.Tensor   # (N, 12) pre-sigmoid logit for calibration loss


class MemoryState(NamedTuple):
    """Fixed-size O(1) latent EvidenceMemory state per vehicle track."""

    mem_feat: torch.Tensor          # (B, 12, D)
    mem_mass: torch.Tensor          # (B, 12)
    mem_conflict: torch.Tensor      # (B, 12)
    last_timestamp_s: torch.Tensor  # (B,)


class MemoryReadout(NamedTuple):
    """Structured predictions decoded from EvidenceMemory."""

    slot_logits: torch.Tensor            # (B, 12, vocab_size)
    plate_type_logits: torch.Tensor      # (B, 5)
    slot_readability_pred: torch.Tensor  # (B, 12) in [0, 1]
    status_logits: torch.Tensor          # (B, 3) [READABLE, UNREADABLE, TRACK_MIXED]


class SequenceOutput(NamedTuple):
    """Complete output bundle of forward_sequence for training and evaluation."""

    slot_logits: torch.Tensor                  # (B, 12, vocab_size)
    plate_type_logits: torch.Tensor            # (B, 5)
    slot_readability_pred: torch.Tensor        # (B, 12)
    status_logits: torch.Tensor                # (B, 3)
    memory: MemoryState
    frame_z: torch.Tensor                      # (B, T, 12, D)
    frame_slot_quality: torch.Tensor           # (B, T, 12)
    frame_valid: torch.Tensor                  # (B, T, 1)
    frame_raw_valid_logit: torch.Tensor        # (B, T, 1)
    frame_raw_slot_quality_logit: torch.Tensor # (B, T, 12)


class ChannelGroupNorm2d(nn.Module):
    """ONNX-clean, TracerWarning-free 2D GroupNorm using standard tensor operations."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) % num_groups ({num_groups}) != 0")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            return F.group_norm(
                x.float(),
                self.num_groups,
                self.weight.reshape(-1).float(),
                self.bias.reshape(-1).float(),
                self.eps,
            ).to(dtype=x.dtype)
        in_dtype = x.dtype
        xf = x.float()
        n, c, h, w = xf.shape
        xg = xf.reshape(n, self.num_groups, -1)
        mean = xg.mean(dim=-1, keepdim=True)
        var = (xg - mean).pow(2).mean(dim=-1, keepdim=True)
        xn = ((xg - mean) * torch.rsqrt(var + self.eps)).reshape(n, c, h, w)
        out = xn * self.weight.float() + self.bias.float()
        return out.to(dtype=in_dtype)


class ConvNeXtLiteBlock(nn.Module):
    """ONNX-clean depthwise-separable inverted residual block with batch-independent GroupNorm."""

    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = dim * expansion
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.gn0 = ChannelGroupNorm2d(8, dim)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.gn1 = ChannelGroupNorm2d(8, hidden)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.gn2 = ChannelGroupNorm2d(8, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.gn0(self.dwconv(x))
        out = self.act(self.gn1(self.pw1(out)))
        out = self.gn2(self.pw2(out))
        return residual + out


def _build_2d_sincos_pos_embed(d_model: int, grid_h: int = 8, grid_w: int = 16) -> torch.Tensor:
    """Create (1, D, H, W) 2D sinusoidal positional embedding."""
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4 for 2D sincos, got {d_model}")
    quarter = d_model // 4
    omega = torch.pow(10000.0, -torch.arange(quarter, dtype=torch.float32) / float(quarter))

    ys = (torch.arange(grid_h, dtype=torch.float32) + 0.5) / float(grid_h) * (2.0 * math.pi)
    xs = (torch.arange(grid_w, dtype=torch.float32) + 0.5) / float(grid_w) * (2.0 * math.pi)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    out_y = yy.unsqueeze(-1) * omega  # (H, W, quarter)
    out_x = xx.unsqueeze(-1) * omega  # (H, W, quarter)

    emb = torch.cat(
        [torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)],
        dim=-1,
    )  # (H, W, D)
    return emb.permute(2, 0, 1).unsqueeze(0).contiguous()


def _build_slot_spatial_prior_dist2(grid_h: int = 8, grid_w: int = 16) -> torch.Tensor:
    """Build (1, 1, 12, H*W) normalized squared distance bias for the 12 physical plate slots.

    Mirrors official MLIT 165x330mm plate geometry:
      - Slots 0..3 (upper-left): region
      - Slots 4..6 (upper-right): classification
      - Slot 7 (lower-left): hiragana
      - Slots 8..11 (lower-right): 4-slot serial
    """
    # Normalized (cx, cy, sigma_x, sigma_y) on [0, 1] x [0, 1]
    slot_priors = [
        # Region (upper-left quadrant)
        (0.27, 0.22, 0.16, 0.22),
        (0.34, 0.22, 0.16, 0.22),
        (0.41, 0.22, 0.16, 0.22),
        (0.48, 0.22, 0.16, 0.22),
        # Classification (upper-right quadrant)
        (0.60, 0.22, 0.15, 0.22),
        (0.68, 0.22, 0.15, 0.22),
        (0.76, 0.22, 0.15, 0.22),
        # Hiragana (lower-left quadrant)
        (0.12, 0.67, 0.15, 0.24),
        # Serial (lower-right quadrant)
        (0.29, 0.67, 0.15, 0.25),
        (0.45, 0.67, 0.15, 0.25),
        (0.71, 0.67, 0.15, 0.25),
        (0.88, 0.67, 0.15, 0.25),
    ]
    ys = (torch.arange(grid_h, dtype=torch.float32) + 0.5) / float(grid_h)
    xs = (torch.arange(grid_w, dtype=torch.float32) + 0.5) / float(grid_w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gx = xx.reshape(-1)  # (128,)
    gy = yy.reshape(-1)  # (128,)

    dists = []
    for cx, cy, sx, sy in slot_priors:
        d2 = ((gx - cx) / sx).pow(2) + ((gy - cy) / sy).pow(2)
        dists.append(d2)
    dist_tensor = torch.stack(dists, dim=0)  # (12, 128)
    return dist_tensor.unsqueeze(0).unsqueeze(0).contiguous()


class FrameSpatialEncoder(nn.Module):
    """Lightweight 8x-downsampling 2D CNN preserving (8, 16) upper/lower row layout."""

    def __init__(
        self,
        d_model: int = 160,
        stem_dim: int = 48,
        stage2_dim: int = 96,
        stem_blocks: int = 2,
        stage2_blocks: int = 3,
        stage3_blocks: int = 4,
        conv_expansion: int = 2,
    ) -> None:
        super().__init__()
        g_stem = 8 if stem_dim % 8 == 0 else 4
        g_s2 = 8 if stage2_dim % 8 == 0 else 4
        g_s3 = 8 if d_model % 8 == 0 else 4

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            ChannelGroupNorm2d(g_stem, stem_dim),
            nn.SiLU(),
            *[ConvNeXtLiteBlock(stem_dim, expansion=conv_expansion) for _ in range(stem_blocks)],
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(stem_dim, stage2_dim, kernel_size=3, stride=2, padding=1, bias=False),
            ChannelGroupNorm2d(g_s2, stage2_dim),
            nn.SiLU(),
            *[ConvNeXtLiteBlock(stage2_dim, expansion=conv_expansion) for _ in range(stage2_blocks)],
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(stage2_dim, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            ChannelGroupNorm2d(g_s3, d_model),
            nn.SiLU(),
            *[ConvNeXtLiteBlock(d_model, expansion=conv_expansion) for _ in range(stage3_blocks)],
        )
        # Learned sub-cell skip-detail projection from 16x32 stage for fine dots ('・') & dakuten
        self.detail_skip = nn.Sequential(
            nn.Conv2d(stage2_dim, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            ChannelGroupNorm2d(g_s3, d_model),
        )

        self.register_buffer("pos_embed_2d", _build_2d_sincos_pos_embed(d_model, 8, 16))
        self.plate_side_embed = nn.Embedding(3, d_model)

    def forward(self, crops: torch.Tensor, plate_sides: torch.Tensor) -> torch.Tensor:
        """Encode (N, 3, 64, 128) float32 [0, 1] crops to (N, D, 8, 16) feature maps."""
        x = (crops - 0.5) * 2.0
        x1 = self.stem(x)     # (N, 48, 32, 64)
        x2 = self.stage2(x1)  # (N, 96, 16, 32)
        x3 = self.stage3(x2) + 0.35 * self.detail_skip(x2)  # (N, D, 8, 16)

        sides = torch.clamp(plate_sides.long(), 0, 2)
        side_emb = self.plate_side_embed(sides).unsqueeze(-1).unsqueeze(-1)
        return x3 + self.pos_embed_2d.to(dtype=x3.dtype) + side_emb


class ExplicitMultiHeadAttention(nn.Module):
    """Pure Linear + MatMul + Softmax multi-head attention for exact ONNX parity."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = query.shape
        kv_len = key_value.shape[1]

        q = self.q_proj(query).reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, Q, KV)
        if attn_bias is not None:
            scores = scores + attn_bias.to(dtype=scores.dtype)

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        ctx = torch.matmul(probs, v)  # (B, H, Q, head_dim)
        ctx = ctx.transpose(1, 2).reshape(bsz, q_len, self.d_model)
        return self.out_proj(ctx)


class SlotDecoderLayer(nn.Module):
    """Cross-attention over 8x16 spatial tokens + inter-slot self-attention + MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = ExplicitMultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = ExplicitMultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.norm_mlp = nn.LayerNorm(d_model)
        hidden = int(round(d_model * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(
        self,
        slots: torch.Tensor,
        spatial_tokens: torch.Tensor,
        spatial_bias: torch.Tensor,
    ) -> torch.Tensor:
        slots = slots + self.cross_attn(
            self.norm_q(slots),
            self.norm_kv(spatial_tokens),
            attn_bias=spatial_bias,
        )
        sa_in = self.norm_sa(slots)
        slots = slots + self.self_attn(sa_in, sa_in)
        slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class MFJPPlateOCR(nn.Module):
    """Multi-Frame Japanese License Plate OCR with patent-clean latent EvidenceMemory."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.d_model

        # 1. Shared 2D spatial encoder
        self.spatial_encoder = FrameSpatialEncoder(
            d_model=d,
            stem_dim=self.config.stem_dim,
            stage2_dim=self.config.stage2_dim,
            stem_blocks=self.config.stem_blocks,
            stage2_blocks=self.config.stage2_blocks,
            stage3_blocks=self.config.stage3_blocks,
            conv_expansion=self.config.conv_expansion,
        )

        # 2. 12-Slot 2D cross-attention extractor
        self.slot_queries = nn.Parameter(torch.randn(1, NUM_SLOTS, d) * 0.02)
        self.register_buffer("spatial_prior_dist2", _build_slot_spatial_prior_dist2(8, 16))
        self.raw_prior_scale = nn.Parameter(
            torch.full((1, self.config.num_heads, 1, 1), 0.55, dtype=torch.float32)
        )
        self.decoder_layers = nn.ModuleList(
            [
                SlotDecoderLayer(
                    d_model=d,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_decoder_layers)
            ]
        )

        # Latent slot evidence projection (shared latent space for z and mem_feat)
        self.evidence_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.evidence_norm = nn.LayerNorm(d)

        # Global frame validity gate (with exact zero-write floor below 0.05)
        self.frame_ctx_proj = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.LayerNorm(d),
            nn.SiLU(),
        )
        self.frame_valid_head = nn.Linear(d, 1)
        nn.init.constant_(self.frame_valid_head.bias, 1.5)

        # Per-slot quality estimator conditioned on slot feature + global frame context
        self.slot_quality_head = nn.Sequential(
            nn.Linear(d * 2, d // 2),
            nn.LayerNorm(d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, 1),
        )
        nn.init.constant_(self.slot_quality_head[-1].bias, 1.0)

        # 3. Structured memory readout heads
        self.readout_slot_embed = nn.Parameter(torch.randn(1, NUM_SLOTS, d) * 0.02)
        self.readout_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, d),
        )
        self.readout_norm = nn.LayerNorm(d)

        self.slot_classifier = nn.Linear(d, self.config.vocab_size)
        self.plate_type_head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.SiLU(),
            nn.Linear(d, self.config.num_plate_types),
        )
        self.slot_readability_head = nn.Sequential(
            nn.Linear(d + 5, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, 1),
        )
        # Status head: pooled h (D) + 12*5 per-slot stats (60) + 4 summary scalars = D + 64
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
        """Total parameter count."""
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def apply_zero_write_floor(raw_valid_logit: torch.Tensor) -> torch.Tensor:
        """Exact zero-write floor: frames with sigmoid(raw) <= 0.05 produce exact 0.0 weight."""
        return torch.relu(torch.sigmoid(raw_valid_logit) - 0.05) / 0.95

    def encode_frames(
        self,
        crops: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
    ) -> FrameEncoding:
        """Encode (N, 3, 64, 128) RGB crops in [0, 1] into per-frame latent slot observations."""
        n = crops.shape[0]
        if plate_sides is None:
            plate_sides = torch.zeros(n, dtype=torch.long, device=crops.device)

        feat_map = self.spatial_encoder(crops, plate_sides)  # (N, D, 8, 16)
        spatial_tokens = feat_map.flatten(2).transpose(1, 2)  # (N, 128, D)

        prior_scale = F.softplus(self.raw_prior_scale)
        spatial_bias = -prior_scale * self.spatial_prior_dist2  # (1, H, 12, 128)

        slots = self.slot_queries.expand(n, -1, -1).to(dtype=spatial_tokens.dtype)
        for layer in self.decoder_layers:
            slots = layer(slots, spatial_tokens, spatial_bias)

        # Calibrated latent slot evidence vector z in R^D
        z = self.evidence_norm(slots + self.evidence_mlp(slots))

        # Global frame context + zero-write validity floor
        sp_mean = spatial_tokens.mean(dim=1)
        sp_max = spatial_tokens.max(dim=1).values
        sl_mean = z.mean(dim=1)
        frame_ctx = self.frame_ctx_proj(torch.cat([sp_mean, sp_max, sl_mean], dim=-1))  # (N, D)
        raw_valid_logit = self.frame_valid_head(frame_ctx)  # (N, 1)
        frame_valid = self.apply_zero_write_floor(raw_valid_logit)

        # Per-slot readability quality in [0, 1] with zero floor for glared/occluded slots
        ctx_expanded = frame_ctx.unsqueeze(1).expand(-1, NUM_SLOTS, -1)
        raw_slot_q = self.slot_quality_head(torch.cat([z, ctx_expanded], dim=-1)).squeeze(-1)
        slot_quality = torch.relu(torch.sigmoid(raw_slot_q) - 0.03) / 0.97

        return FrameEncoding(
            z=z,
            slot_quality=slot_quality,
            frame_valid=frame_valid,
            raw_valid_logit=raw_valid_logit,
            raw_slot_quality_logit=raw_slot_q,
        )

    @staticmethod
    def fuse_views(
        z_views: torch.Tensor,
        slot_quality_views: torch.Tensor,
        frame_valid_views: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Consolidate V same-frame crop variants into a single observation (no double-voting).

        Accepts either (B, V, ...) or unbatched (V, ...).
        """
        unbatched = z_views.ndim == 3
        if unbatched:
            z_views = z_views.unsqueeze(0)
            slot_quality_views = slot_quality_views.unsqueeze(0)
            frame_valid_views = frame_valid_views.unsqueeze(0)
            if view_mask is not None:
                view_mask = view_mask.unsqueeze(0)

        #z_views: (B, V, 12, D), slot_quality_views: (B, V, 12), frame_valid_views: (B, V, 1)
        if view_mask is not None:
            vm = view_mask.to(dtype=z_views.dtype)  # (B, V)
            q_masked = slot_quality_views * vm.unsqueeze(-1)
            fv_masked = frame_valid_views * vm.unsqueeze(-1)
            uniform_w = vm / torch.clamp(vm.sum(dim=1, keepdim=True), min=1.0)
        else:
            q_masked = slot_quality_views
            fv_masked = frame_valid_views
            v_count = float(z_views.shape[1])
            uniform_w = torch.full(
                (z_views.shape[0], z_views.shape[1]),
                1.0 / max(v_count, 1.0),
                dtype=z_views.dtype,
                device=z_views.device,
            )

        raw_w = fv_masked * q_masked  # (B, V, 12)
        w_sum = raw_w.sum(dim=1, keepdim=True)  # (B, 1, 12)
        w_norm = torch.where(
            w_sum > 1e-6,
            raw_w / torch.clamp(w_sum, min=1e-6),
            uniform_w.unsqueeze(-1).expand_as(raw_w),
        )
        z_step = (z_views * w_norm.unsqueeze(-1)).sum(dim=1)  # (B, 12, D)
        slot_quality_step = q_masked.max(dim=1).values        # (B, 12)
        frame_valid_step = fv_masked.max(dim=1).values        # (B, 1)

        if unbatched:
            return z_step.squeeze(0), slot_quality_step.squeeze(0), frame_valid_step.squeeze(0)
        return z_step, slot_quality_step, frame_valid_step

    def init_memory(
        self,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> MemoryState:
        """Create fresh empty O(1) EvidenceMemory state for `batch_size` tracks."""
        dev = torch.device(device)
        return MemoryState(
            mem_feat=torch.zeros((batch_size, NUM_SLOTS, self.config.d_model), device=dev, dtype=dtype),
            mem_mass=torch.zeros((batch_size, NUM_SLOTS), device=dev, dtype=dtype),
            mem_conflict=torch.zeros((batch_size, NUM_SLOTS), device=dev, dtype=dtype),
            last_timestamp_s=torch.full((batch_size,), -1.0, device=dev, dtype=dtype),
        )

    def step_memory(
        self,
        mem_feat: torch.Tensor,
        mem_mass: torch.Tensor,
        mem_conflict: torch.Tensor,
        last_timestamp_s: torch.Tensor,
        z: torch.Tensor,
        slot_quality: torch.Tensor,
        frame_valid: torch.Tensor,
        timestamp_s: torch.Tensor,
        step_mask: torch.Tensor | None = None,
    ) -> MemoryState:
        """Pure tensor-in / tensor-out O(1) EvidenceMemory update step (ONNX-clean)."""
        if z.dtype != torch.float32:
            z = z.float()
        if slot_quality.dtype != torch.float32:
            slot_quality = slot_quality.float()
        if frame_valid.dtype != torch.float32:
            frame_valid = frame_valid.float()
        if timestamp_s.dtype != torch.float32:
            timestamp_s = timestamp_s.float()
        if mem_feat.dtype != torch.float32:
            mem_feat = mem_feat.float()
            mem_mass = mem_mass.float()
            mem_conflict = mem_conflict.float()
            last_timestamp_s = last_timestamp_s.float()

        # Cosine similarity between current slot memory and incoming slot observation
        mem_norm = mem_feat / (torch.norm(mem_feat, p=2, dim=-1, keepdim=True) + 1e-6)
        z_norm = z / (torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-6)
        cos_sim = torch.clamp((mem_norm * z_norm).sum(dim=-1), -1.0, 1.0)  # (B, 12)
        feat_diff = torch.clamp(0.5 * (1.0 - cos_sim), 0.0, 1.0)           # (B, 12)

        # Temporal delta_t & novelty gate in [0.15, 1.0]
        raw_dt = torch.clamp(timestamp_s - last_timestamp_s, 0.0, 10.0)
        delta_t = torch.where(
            last_timestamp_s >= 0.0,
            raw_dt,
            torch.full_like(raw_dt, 10.0),
        )
        time_factor = torch.clamp(delta_t / 0.25, 0.0, 1.0).unsqueeze(-1)  # (B, 1)
        nov_signal = torch.clamp(0.5 * time_factor + 2.5 * feat_diff, 0.0, 1.0)
        novelty_prior = 0.15 + 0.85 * nov_signal
        novelty = torch.where(mem_mass > 1e-3, novelty_prior, torch.ones_like(novelty_prior))

        # Effective slot write weight (exact 0.0 whenever frame_valid == 0.0)
        alpha = frame_valid * slot_quality * novelty  # (B, 12)
        if step_mask is not None:
            sm = step_mask.to(dtype=alpha.dtype)
            alpha = alpha * sm.unsqueeze(-1)
            eff_valid = frame_valid.squeeze(-1) * sm
        else:
            eff_valid = frame_valid.squeeze(-1)

        # High-quality cross-frame feature disagreement accumulator (patent-clean latent conflict)
        # Deadzone of 0.07 ignores benign intra-character blur/perspective jitter on long 24-60f tracks
        both_active = (
            torch.clamp((mem_mass - 0.18) / 0.22, 0.0, 1.0)
            * torch.clamp((alpha - 0.22) / 0.22, 0.0, 1.0)
        )
        meaningful_disagreement = torch.relu((1.0 - cos_sim) - 0.07) / 0.93
        conflict_step = (
            both_active
            * meaningful_disagreement
            * torch.minimum(torch.clamp(mem_mass, max=2.0), alpha)
        )
        new_conflict = mem_conflict + conflict_step

        # Exact convex evidence update (guarantees bit-exact zero drift when alpha == 0.0)
        new_mass = mem_mass + alpha
        denom = torch.clamp(new_mass, min=1e-6)
        step_ratio = alpha / denom  # (B, 12)
        new_feat = mem_feat + step_ratio.unsqueeze(-1) * (z - mem_feat)

        new_last_ts = torch.where(eff_valid > 0.0, timestamp_s, last_timestamp_s)

        return MemoryState(
            mem_feat=new_feat,
            mem_mass=new_mass,
            mem_conflict=new_conflict,
            last_timestamp_s=new_last_ts,
        )

    def refine_slot_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply shared readout MLP + LayerNorm to (..., 12, D) latent slot features."""
        slot_emb = self.readout_slot_embed.to(dtype=feat.dtype)
        return self.readout_norm(feat + self.readout_mlp(feat + slot_emb))

    def decode_slot_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Shared per-slot character classifier for both memory and auxiliary per-frame loss."""
        return self.slot_classifier(self.refine_slot_features(feat))

    def decode_memory(
        self,
        mem_feat: torch.Tensor,
        mem_mass: torch.Tensor,
        mem_conflict: torch.Tensor,
    ) -> MemoryReadout:
        """Read out structured 12-slot predictions, readability, and track status from memory."""
        h = self.refine_slot_features(mem_feat)  # (B, 12, D)
        slot_logits = self.slot_classifier(h)    # (B, 12, V)

        # Plate type from uniform + mass-weighted pooled slot embeddings
        h_mean = h.mean(dim=1)
        mass_w = mem_mass / torch.clamp(mem_mass.sum(dim=1, keepdim=True), min=1e-6)
        h_mass = (h * mass_w.unsqueeze(-1)).sum(dim=1)
        plate_type_logits = self.plate_type_head(torch.cat([h_mean, h_mass], dim=-1))

        # Raw slot uncertainty features (computed on raw unmasked probabilities!)
        probs = F.softmax(slot_logits.float(), dim=-1)
        max_prob = probs.max(dim=-1).values  # (B, 12)
        pad_prob = probs[:, :, 0]            # (B, 12)
        log_v = math.log(float(self.config.vocab_size))
        norm_entropy = -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=-1) / log_v

        clamped_mass = torch.clamp(mem_mass.float(), 0.0, 4.5)
        log_mass = torch.log(clamped_mass + 1.0)
        clamped_conf = torch.clamp(mem_conflict.float(), 0.0, 5.0)

        read_in = torch.cat(
            [
                h.float(),
                log_mass.unsqueeze(-1),
                clamped_mass.unsqueeze(-1),
                max_prob.unsqueeze(-1),
                norm_entropy.unsqueeze(-1),
                clamped_conf.unsqueeze(-1),
            ],
            dim=-1,
        )  # (B, 12, D + 5)
        slot_readability_pred = torch.sigmoid(
            self.slot_readability_head(read_in.to(dtype=h.dtype)).squeeze(-1)
        )

        # Active-slot summary statistics (down-weighting slots predicted as <PAD>)
        active_prob = 1.0 - pad_prob  # (B, 12)
        # Soft minimum readability & mass over active slots (push inactive slots toward 1.0/5.0)
        inactive_boost = (pad_prob > 0.65).to(dtype=torch.float32)
        eff_slot_read = torch.clamp(
            slot_readability_pred.float() + inactive_boost, 0.0, 1.0
        )
        min_active_read = eff_slot_read.min(dim=-1, keepdim=True).values  # (B, 1)
        eff_slot_mass = clamped_mass + inactive_boost * 5.0
        min_active_mass = eff_slot_mass.min(dim=-1, keepdim=True).values  # (B, 1)
        max_conflict = clamped_conf.max(dim=-1, keepdim=True).values      # (B, 1)
        mean_conflict = clamped_conf.mean(dim=-1, keepdim=True)           # (B, 1)

        status_in = torch.cat(
            [
                h_mean.float(),                  # D
                log_mass,                        # 12
                slot_readability_pred.float(),   # 12
                clamped_conf,                    # 12
                max_prob,                        # 12
                active_prob,                     # 12
                min_active_read,                 # 1
                min_active_mass,                 # 1
                max_conflict,                    # 1
                mean_conflict,                   # 1
            ],
            dim=-1,
        )  # (B, D + 64)
        status_logits = self.status_head(status_in.to(dtype=h.dtype))

        return MemoryReadout(
            slot_logits=slot_logits,
            plate_type_logits=plate_type_logits,
            slot_readability_pred=slot_readability_pred,
            status_logits=status_logits,
        )

    @staticmethod
    def _has_duplicate_frame_groups(
        camera_ids: torch.Tensor | None,
        source_frame_ids: torch.Tensor | None,
        frame_pad_mask: torch.Tensor | None,
    ) -> bool:
        """Check whether any sequence in batch contains multi-view crops from same physical frame."""
        if camera_ids is None or source_frame_ids is None:
            return False
        bsz, seq_len = source_frame_ids.shape
        if seq_len <= 1:
            return False
        cam_cpu = camera_ids.detach().cpu().tolist()
        src_cpu = source_frame_ids.detach().cpu().tolist()
        pad_cpu = (
            frame_pad_mask.detach().cpu().tolist()
            if frame_pad_mask is not None
            else [[False] * seq_len for _ in range(bsz)]
        )
        for b in range(bsz):
            seen: set[tuple[int, int]] = set()
            for t in range(seq_len):
                if pad_cpu[b][t]:
                    continue
                sfid = int(src_cpu[b][t])
                if sfid < 0:
                    continue
                key = (int(cam_cpu[b][t]), sfid)
                if key in seen:
                    return True
                seen.add(key)
        return False

    def forward_sequence(
        self,
        crops: torch.Tensor,
        timestamps_s: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        frame_pad_mask: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
        source_frame_ids: torch.Tensor | None = None,
    ) -> SequenceOutput:
        """Run spatial encoding, optional same-frame view fusion, and O(1) memory rollout.

        Args:
            crops: (B, T, 3, 64, 128) float32 in [0, 1].
            timestamps_s: (B, T) float32 timestamps in seconds.
            plate_sides: Optional (B, T) int64 in {0, 1, 2}.
            frame_pad_mask: Optional (B, T) bool where True indicates padded timestep to ignore.
            camera_ids: Optional (B, T) int64 camera IDs for same-frame view deduplication.
            source_frame_ids: Optional (B, T) int64 physical frame IDs for view deduplication.
        """
        bsz, seq_len, ch, h, w = crops.shape
        if plate_sides is None:
            plate_sides = torch.zeros((bsz, seq_len), dtype=torch.long, device=crops.device)

        flat_crops = crops.reshape(bsz * seq_len, ch, h, w)
        flat_sides = plate_sides.reshape(bsz * seq_len)
        enc = self.encode_frames(flat_crops, flat_sides)

        frame_z = enc.z.reshape(bsz, seq_len, NUM_SLOTS, self.config.d_model)
        frame_q = enc.slot_quality.reshape(bsz, seq_len, NUM_SLOTS)
        frame_v = enc.frame_valid.reshape(bsz, seq_len, 1)
        frame_raw_v = enc.raw_valid_logit.reshape(bsz, seq_len, 1)
        frame_raw_q = enc.raw_slot_quality_logit.reshape(bsz, seq_len, NUM_SLOTS)

        if frame_pad_mask is not None:
            step_valid = (~frame_pad_mask).to(dtype=torch.float32)
        else:
            step_valid = torch.ones((bsz, seq_len), dtype=torch.float32, device=crops.device)

        if not self._has_duplicate_frame_groups(camera_ids, source_frame_ids, frame_pad_mask):
            state = self.init_memory(batch_size=bsz, device=crops.device, dtype=torch.float32)
            for t in range(seq_len):
                state = self.step_memory(
                    state.mem_feat,
                    state.mem_mass,
                    state.mem_conflict,
                    state.last_timestamp_s,
                    frame_z[:, t],
                    frame_q[:, t],
                    frame_v[:, t],
                    timestamps_s[:, t],
                    step_mask=step_valid[:, t],
                )
        else:
            # Consolidate same-(camera_id, source_frame_id) views per track BEFORE memory update
            assert camera_ids is not None and source_frame_ids is not None
            cam_cpu = camera_ids.detach().cpu().tolist()
            src_cpu = source_frame_ids.detach().cpu().tolist()
            pad_cpu = (
                frame_pad_mask.detach().cpu().tolist()
                if frame_pad_mask is not None
                else [[False] * seq_len for _ in range(bsz)]
            )

            feats_list, mass_list, conf_list, ts_list = [], [], [], []
            for b in range(bsz):
                groups_order: list[tuple[int, int]] = []
                group_indices: dict[tuple[int, int], list[int]] = {}
                unique_counter = -1
                for t in range(seq_len):
                    if pad_cpu[b][t]:
                        continue
                    sfid = int(src_cpu[b][t])
                    if sfid < 0:
                        key = (-1, unique_counter)
                        unique_counter -= 1
                    else:
                        key = (int(cam_cpu[b][t]), sfid)
                    if key not in group_indices:
                        groups_order.append(key)
                        group_indices[key] = []
                    group_indices[key].append(t)

                st_b = self.init_memory(batch_size=1, device=crops.device, dtype=torch.float32)
                for key in groups_order:
                    idxs = group_indices[key]
                    if len(idxs) == 1:
                        t0 = idxs[0]
                        z_s = frame_z[b : b + 1, t0]
                        q_s = frame_q[b : b + 1, t0]
                        v_s = frame_v[b : b + 1, t0]
                        ts_s = timestamps_s[b : b + 1, t0]
                    else:
                        z_views = frame_z[b, idxs].unsqueeze(0)
                        q_views = frame_q[b, idxs].unsqueeze(0)
                        v_views = frame_v[b, idxs].unsqueeze(0)
                        z_s, q_s, v_s = self.fuse_views(z_views, q_views, v_views)
                        ts_s = timestamps_s[b : b + 1, idxs[0]]
                    st_b = self.step_memory(
                        st_b.mem_feat,
                        st_b.mem_mass,
                        st_b.mem_conflict,
                        st_b.last_timestamp_s,
                        z_s,
                        q_s,
                        v_s,
                        ts_s,
                    )
                feats_list.append(st_b.mem_feat)
                mass_list.append(st_b.mem_mass)
                conf_list.append(st_b.mem_conflict)
                ts_list.append(st_b.last_timestamp_s)

            state = MemoryState(
                mem_feat=torch.cat(feats_list, dim=0),
                mem_mass=torch.cat(mass_list, dim=0),
                mem_conflict=torch.cat(conf_list, dim=0),
                last_timestamp_s=torch.cat(ts_list, dim=0),
            )

        readout = self.decode_memory(state.mem_feat, state.mem_mass, state.mem_conflict)
        return SequenceOutput(
            slot_logits=readout.slot_logits,
            plate_type_logits=readout.plate_type_logits,
            slot_readability_pred=readout.slot_readability_pred,
            status_logits=readout.status_logits,
            memory=state,
            frame_z=frame_z,
            frame_slot_quality=frame_q,
            frame_valid=frame_v,
            frame_raw_valid_logit=frame_raw_v,
            frame_raw_slot_quality_logit=frame_raw_q,
        )

    def forward(
        self,
        crops: torch.Tensor,
        timestamps_s: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        frame_pad_mask: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
        source_frame_ids: torch.Tensor | None = None,
    ) -> SequenceOutput:
        return self.forward_sequence(
            crops=crops,
            timestamps_s=timestamps_s,
            plate_sides=plate_sides,
            frame_pad_mask=frame_pad_mask,
            camera_ids=camera_ids,
            source_frame_ids=source_frame_ids,
        )
