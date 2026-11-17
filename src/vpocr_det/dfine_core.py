"""Self-contained Apache-2.0 `D-FINE` (`HGNetV2` + `HybridEncoder` + `FDR Decoder`) core.

Adapted from:
  - `Peterande/D-FINE` (Copyright 2024 The D-FINE Authors, Apache-2.0)
  - `huggingface/transformers` `v4.52.4` `hgnet_v2` & `d_fine` (Copyright 2025 Baidu Inc. &
    The HuggingFace Inc. team, Apache-2.0)

Key properties for vehicle-plate-ocr-jp-v1 Gen-4 (`v0.4.0`):
  1. Zero external runtime dependencies (`transformers` / `safetensors` not required).
  2. Cryptographic SHA-256 verification + `strict=True` loading of pinned Apache-2.0
     `.safetensors` checkpoints (`dfine-nano-coco` & `dfine-small-obj2coco`).
  3. Exposes `[C2 (s=4), C3 (s=8), C4 (s=16), C5 (s=32)]` `HGNetV2` features, `PAN` encoder
     features, and `300` `D-FINE` object queries `(logits, boxes_cxcywh, query_embeds)` in a
     single forward pass.
  4. 100% ONNX Opset-17 exportable (`float32` sinusoidal embeddings, `F.grid_sample`
     deformable attention, zero `Float64` or dynamic `ScatterND` tensor mutations).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# COCO-80 0-indexed road-vehicle label IDs in D-FINE checkpoints:
#   2: car, 3: motorbike, 5: bus, 6: train, 7: truck
COCO_VEHICLE_CLASS_IDS: Tuple[int, ...] = (2, 3, 5, 6, 7)
# Heavy/four-wheel motor vehicles prioritized for heavy-vehicle construction & gate monitoring:
#   2: car, 5: bus, 7: truck (plus 3: motorbike / 6: train at slightly lower weight if desired)
COCO_PRIMARY_MOTOR_VEHICLE_IDS: Tuple[int, ...] = (2, 5, 7)


@dataclass(frozen=True)
class DFinePretrainedSpec:
    variant: str
    repo_id: str
    revision: str
    license_spdx: str
    filename: str
    sha256: str
    num_bytes: int


DFINE_PRETRAINED_SPECS: Dict[str, DFinePretrainedSpec] = {
    "nano": DFinePretrainedSpec(
        variant="nano",
        repo_id="ustc-community/dfine-nano-coco",
        revision="066438d3d8f0da137a37b38fdf3368fd4afceced",
        license_spdx="Apache-2.0",
        filename="dfine_nano_coco.safetensors",
        sha256="19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b",
        num_bytes=15_278_996,
    ),
    "small": DFinePretrainedSpec(
        variant="small",
        repo_id="ustc-community/dfine-small-obj2coco",
        revision="2b548d12a7623d674bb2aa2bf24022ce090b7736",
        license_spdx="Apache-2.0",
        filename="dfine_small_obj2coco.safetensors",
        sha256="8ff4a8332cf9507209fa443058d00752ebfede3f8cb497b91d4238369e3a57f7",
        num_bytes=41_497_620,
    ),
}


@dataclass(frozen=True)
class DFineCoreConfig:
    variant: str = "nano"
    num_labels: int = 80
    # HGNetV2 backbone
    stem_channels: Tuple[int, int, int] = (3, 16, 16)
    stem_strides: Tuple[int, int, int, int, int] = (2, 1, 1, 2, 1)
    stage_in_channels: Tuple[int, int, int, int] = (16, 64, 256, 512)
    stage_mid_channels: Tuple[int, int, int, int] = (16, 32, 64, 128)
    stage_out_channels: Tuple[int, int, int, int] = (64, 256, 512, 1024)
    stage_num_blocks: Tuple[int, int, int, int] = (1, 1, 2, 1)
    stage_numb_of_layers: Tuple[int, int, int, int] = (3, 3, 3, 3)
    stage_downsample: Tuple[bool, bool, bool, bool] = (False, True, True, True)
    stage_downsample_strides: Tuple[int, int, int, int] = (2, 2, 2, 2)
    stage_light_block: Tuple[bool, bool, bool, bool] = (False, False, True, True)
    stage_kernel_size: Tuple[int, int, int, int] = (3, 3, 5, 5)
    out_indices: Tuple[int, ...] = (3, 4)
    use_learnable_affine_block: bool = True
    # Hybrid encoder
    encoder_in_channels: Tuple[int, ...] = (512, 1024)
    feat_strides: Tuple[int, ...] = (16, 32)
    encoder_hidden_dim: int = 128
    encoder_layers: int = 1
    encoder_ffn_dim: int = 512
    encoder_attention_heads: int = 8
    encode_proj_layers: Tuple[int, ...] = (1,)
    hidden_expansion: float = 0.34
    depth_mult: float = 0.5
    positional_encoding_temperature: float = 10000.0
    # FDR Decoder
    d_model: int = 128
    decoder_in_channels: Tuple[int, ...] = (128, 128)
    num_feature_levels: int = 2
    decoder_layers: int = 3
    decoder_ffn_dim: int = 512
    decoder_attention_heads: int = 8
    decoder_n_points: Tuple[int, ...] = (6, 6)
    decoder_offset_scale: float = 0.5
    num_queries: int = 300
    num_denoising: int = 100
    max_num_bins: int = 32
    reg_scale: float = 4.0
    up: float = 0.5
    top_prob_values: int = 4
    lqe_hidden_dim: int = 64
    lqe_layers: int = 2
    batch_norm_eps: float = 1e-5
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_variant(cls, variant: str) -> "DFineCoreConfig":
        v = str(variant).lower().strip()
        if v == "nano":
            return cls(variant="nano")
        if v == "small":
            return cls(
                variant="small",
                out_indices=(2, 3, 4),
                encoder_in_channels=(256, 512, 1024),
                feat_strides=(8, 16, 32),
                encoder_hidden_dim=256,
                encoder_ffn_dim=1024,
                encode_proj_layers=(2,),
                hidden_expansion=0.5,
                depth_mult=0.34,
                d_model=256,
                decoder_in_channels=(256, 256, 256),
                num_feature_levels=3,
                decoder_ffn_dim=1024,
                decoder_n_points=(3, 6, 3),
            )
        raise ValueError(f"Unsupported DFineCoreConfig variant: {variant!r}")


def default_pretrained_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "out" / "models" / "pretrained_apache20"


def compute_file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


_SAFETENSORS_DTYPE_MAP: Dict[str, torch.dtype] = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def load_safetensors_state_dict(
    path: str | Path,
    *,
    expected_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Tensor], str]:
    """Zero-dependency `.safetensors` reader with optional SHA-256 verification."""
    fpath = Path(path)
    if not fpath.is_file():
        raise FileNotFoundError(f"Missing pretrained Apache-2.0 safetensors file: {fpath}")
    raw = fpath.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {fpath}: expected {expected_sha256}, got {actual_sha}"
        )
    if len(raw) < 8:
        raise ValueError(f"Corrupt safetensors file (too short): {fpath}")
    header_len = struct.unpack("<Q", raw[:8])[0]
    header_end = 8 + int(header_len)
    if header_end > len(raw):
        raise ValueError(f"Corrupt safetensors header length in {fpath}")
    header = json.loads(raw[8:header_end].decode("utf-8"))
    body = memoryview(raw)[header_end:]
    state: Dict[str, Tensor] = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        dtype_str = str(info["dtype"])
        if dtype_str not in _SAFETENSORS_DTYPE_MAP:
            raise ValueError(f"Unsupported safetensors dtype {dtype_str!r} for {key}")
        torch_dtype = _SAFETENSORS_DTYPE_MAP[dtype_str]
        shape = tuple(int(s) for s in info["shape"])
        start, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        if end == start:
            tensor = torch.empty(shape, dtype=torch_dtype)
        else:
            # Copy buffer slice into writable contiguous CPU tensor
            buf = bytearray(body[start:end])
            tensor = torch.frombuffer(buf, dtype=torch_dtype).reshape(shape).clone()
        state[key] = tensor
    return state, actual_sha


def _act_fn(name: Optional[str]) -> nn.Module:
    if name is None or name == "none":
        return nn.Identity()
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "silu":
        return nn.SiLU()
    if n == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation {name!r}")


def _corners_to_center_format(boxes_xyxy: Tensor) -> Tensor:
    x0, y0, x1, y1 = boxes_xyxy.unbind(-1)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w = x1 - x0
    h = y1 - y0
    return torch.stack([cx, cy, w, h], dim=-1)


def center_to_corners_format_torch(boxes_cxcywh: Tensor) -> Tensor:
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


class DFineFrozenBatchNorm2d(nn.Module):
    """Fixed-statistics affine BatchNorm2d matching `DFineFrozenBatchNorm2d`."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        tracked_key = prefix + "num_batches_tracked"
        if tracked_key in state_dict:
            del state_dict[tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# ---------------------------------------------------------------------------
# HGNetV2 Backbone (with FrozenBatchNorm2d & LearnableAffineBlock)
# ---------------------------------------------------------------------------


class HGNetV2LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([bias_value], dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.bias


class HGNetV2ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = "relu",
        use_learnable_affine_block: bool = False,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.normalization = DFineFrozenBatchNorm2d(out_channels)
        self.activation = _act_fn(activation)
        if activation is not None and use_learnable_affine_block:
            self.lab: nn.Module = HGNetV2LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.lab(x)
        return x


class HGNetV2ConvLayerLight(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_learnable_affine_block: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = HGNetV2ConvLayer(
            in_channels,
            out_channels,
            kernel_size=1,
            activation=None,
            use_learnable_affine_block=use_learnable_affine_block,
        )
        self.conv2 = HGNetV2ConvLayer(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            activation="relu",
            use_learnable_affine_block=use_learnable_affine_block,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.conv1(x))


class HGNetV2Embeddings(nn.Module):
    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        u = cfg.use_learnable_affine_block
        c0, c1, c2 = cfg.stem_channels
        s0, s1, s2, s3, s4 = cfg.stem_strides
        self.stem1 = HGNetV2ConvLayer(c0, c1, kernel_size=3, stride=s0, activation="relu", use_learnable_affine_block=u)
        self.stem2a = HGNetV2ConvLayer(
            c1, c1 // 2, kernel_size=2, stride=s1, activation="relu", use_learnable_affine_block=u
        )
        self.stem2b = HGNetV2ConvLayer(
            c1 // 2, c1, kernel_size=2, stride=s2, activation="relu", use_learnable_affine_block=u
        )
        self.stem3 = HGNetV2ConvLayer(
            c1 * 2, c1, kernel_size=3, stride=s3, activation="relu", use_learnable_affine_block=u
        )
        self.stem4 = HGNetV2ConvLayer(c1, c2, kernel_size=1, stride=s4, activation="relu", use_learnable_affine_block=u)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, pixel_values: Tensor) -> Tensor:
        emb = self.stem1(pixel_values)
        emb_pad = F.pad(emb, (0, 1, 0, 1))
        s2 = self.stem2a(emb_pad)
        s2 = F.pad(s2, (0, 1, 0, 1))
        s2 = self.stem2b(s2)
        # Note: F.pad(emb, (0, 1, 0, 1)) with MaxPool2d(k=2, s=1) without ceil_mode is exact
        # ONNX-friendly equivalent of MaxPool2d(k=2, s=1, ceil_mode=True) on F.pad(emb, (0,1,0,1))!
        # Let's double check: in upstream HGNetV2Embeddings:
        #   embedding = F.pad(embedding, (0, 1, 0, 1))
        #   pooled_emb = self.pool(embedding) -> Wait! With k=2, s=1 on (H+1, W+1), output is (H, W)
        #   whether ceil_mode is True or False because (H+1 - 2)/1 + 1 = H (exact integer)!
        # Using ceil_mode=False on already-padded emb_pad avoids ONNX MaxPool ceil_mode quirks while
        # producing bit-identical (H, W) output!
        pooled = F.max_pool2d(emb_pad, kernel_size=2, stride=1, padding=0, ceil_mode=False)
        out = torch.cat([pooled, s2], dim=1)
        out = self.stem3(out)
        out = self.stem4(out)
        return out


class HGNetV2BasicLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        layer_num: int,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        use_learnable_affine_block: bool = False,
    ) -> None:
        super().__init__()
        self.residual = bool(residual)
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            c_in = in_channels if i == 0 else middle_channels
            if light_block:
                blk: nn.Module = HGNetV2ConvLayerLight(
                    c_in, middle_channels, kernel_size=kernel_size, use_learnable_affine_block=use_learnable_affine_block
                )
            else:
                blk = HGNetV2ConvLayer(
                    c_in,
                    middle_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    activation="relu",
                    use_learnable_affine_block=use_learnable_affine_block,
                )
            self.layers.append(blk)
        total_ch = in_channels + layer_num * middle_channels
        self.aggregation = nn.Sequential(
            HGNetV2ConvLayer(
                total_ch,
                out_channels // 2,
                kernel_size=1,
                stride=1,
                activation="relu",
                use_learnable_affine_block=use_learnable_affine_block,
            ),
            HGNetV2ConvLayer(
                out_channels // 2,
                out_channels,
                kernel_size=1,
                stride=1,
                activation="relu",
                use_learnable_affine_block=use_learnable_affine_block,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        outs = [x]
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        x = self.aggregation(torch.cat(outs, dim=1))
        if self.residual:
            x = x + identity
        return x


class HGNetV2Stage(nn.Module):
    def __init__(self, cfg: DFineCoreConfig, stage_index: int) -> None:
        super().__init__()
        c_in = cfg.stage_in_channels[stage_index]
        c_mid = cfg.stage_mid_channels[stage_index]
        c_out = cfg.stage_out_channels[stage_index]
        n_blocks = cfg.stage_num_blocks[stage_index]
        n_layers = cfg.stage_numb_of_layers[stage_index]
        downsample = cfg.stage_downsample[stage_index]
        stride = cfg.stage_downsample_strides[stage_index]
        light = cfg.stage_light_block[stage_index]
        k = cfg.stage_kernel_size[stage_index]
        u = cfg.use_learnable_affine_block

        if downsample:
            self.downsample: nn.Module = HGNetV2ConvLayer(
                c_in, c_in, kernel_size=3, stride=stride, groups=c_in, activation=None, use_learnable_affine_block=False
            )
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                HGNetV2BasicLayer(
                    c_in if i == 0 else c_out,
                    c_mid,
                    c_out,
                    n_layers,
                    kernel_size=k,
                    residual=(i != 0),
                    light_block=light,
                    use_learnable_affine_block=u,
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class HGNetV2Encoder(nn.Module):
    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        self.stages = nn.ModuleList([HGNetV2Stage(cfg, idx) for idx in range(len(cfg.stage_in_channels))])

    def forward(self, stem_feat: Tensor) -> Tuple[Tensor, ...]:
        # hidden_states[0] = stem (s=4, 16ch)
        # hidden_states[1] = stage1 / C2 (s=4, 64ch)
        # hidden_states[2] = stage2 / C3 (s=8, 256ch)
        # hidden_states[3] = stage3 / C4 (s=16, 512ch)
        # hidden_states[4] = stage4 / C5 (s=32, 1024ch)
        states: List[Tensor] = [stem_feat]
        x = stem_feat
        for stage in self.stages:
            x = stage(x)
            states.append(x)
        return tuple(states)


class HGNetV2BackboneModule(nn.Module):
    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        self.out_indices = cfg.out_indices
        self.embedder = HGNetV2Embeddings(cfg)
        self.encoder = HGNetV2Encoder(cfg)

    def forward_all_stages(self, pixel_values: Tensor) -> Tuple[Tuple[Tensor, ...], List[Tensor]]:
        stem = self.embedder(pixel_values)
        states = self.encoder(stem)
        selected = [states[idx] for idx in self.out_indices]
        return states, selected


class DFineConvEncoderWrapper(nn.Module):
    """Preserves exact `backbone.model.*` parameter path hierarchy."""

    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        self.model = HGNetV2BackboneModule(cfg)


# ---------------------------------------------------------------------------
# D-FINE Hybrid Encoder & FDR Transformer Decoder (v4.52 checkpoint layout)
# ---------------------------------------------------------------------------


class DFineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, act: str = "relu") -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = _act_fn(act)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DFineConvNormLayer(nn.Module):
    def __init__(
        self,
        cfg: DFineCoreConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        padding: Optional[int] = None,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, eps=cfg.batch_norm_eps)
        self.activation = _act_fn(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class DFineRepVggBlock(nn.Module):
    def __init__(self, cfg: DFineCoreConfig, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = DFineConvNormLayer(cfg, in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = DFineConvNormLayer(cfg, in_channels, out_channels, 1, 1, padding=0)
        self.activation = _act_fn("silu")

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv1(x) + self.conv2(x))


class DFineCSPRepLayer(nn.Module):
    def __init__(
        self, cfg: DFineCoreConfig, in_channels: int, out_channels: int, num_blocks: int, expansion: float = 1.0
    ) -> None:
        super().__init__()
        hidden_ch = int(out_channels * expansion)
        self.conv1 = DFineConvNormLayer(cfg, in_channels, hidden_ch, 1, 1, activation="silu")
        self.conv2 = DFineConvNormLayer(cfg, in_channels, hidden_ch, 1, 1, activation="silu")
        self.bottlenecks = nn.ModuleList([DFineRepVggBlock(cfg, hidden_ch, hidden_ch) for _ in range(num_blocks)])
        if hidden_ch != out_channels:
            self.conv3: nn.Module = DFineConvNormLayer(cfg, hidden_ch, out_channels, 1, 1, activation="silu")
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        for b in self.bottlenecks:
            x1 = b(x1)
        x2 = self.conv2(x)
        return self.conv3(x1 + x2)


class DFineRepNCSPELAN4(nn.Module):
    def __init__(self, cfg: DFineCoreConfig, act: str = "silu", numb_blocks: int = 3) -> None:
        super().__init__()
        conv1_dim = cfg.encoder_hidden_dim * 2
        conv2_dim = cfg.encoder_hidden_dim
        conv3_dim = cfg.encoder_hidden_dim * 2
        # Exact upstream Python precedence: (hidden_expansion * encoder_hidden_dim) // 2
        conv4_dim = round(cfg.hidden_expansion * cfg.encoder_hidden_dim // 2)
        self.conv_dim = conv3_dim // 2
        self.conv1 = DFineConvNormLayer(cfg, conv1_dim, conv3_dim, 1, 1, activation=act)
        self.csp_rep1 = DFineCSPRepLayer(cfg, conv3_dim // 2, conv4_dim, num_blocks=numb_blocks)
        self.conv2 = DFineConvNormLayer(cfg, conv4_dim, conv4_dim, 3, 1, activation=act)
        self.csp_rep2 = DFineCSPRepLayer(cfg, conv4_dim, conv4_dim, num_blocks=numb_blocks)
        self.conv3 = DFineConvNormLayer(cfg, conv4_dim, conv4_dim, 3, 1, activation=act)
        self.conv4 = DFineConvNormLayer(cfg, conv3_dim + (2 * conv4_dim), conv2_dim, 1, 1, activation=act)

    def forward(self, x: Tensor) -> Tensor:
        s0, s1 = self.conv1(x).split((self.conv_dim, self.conv_dim), dim=1)
        b1 = self.conv2(self.csp_rep1(s1))
        b2 = self.conv3(self.csp_rep2(b1))
        return self.conv4(torch.cat([s0, s1, b1, b2], dim=1))


class DFineSCDown(nn.Module):
    def __init__(self, cfg: DFineCoreConfig, kernel_size: int, stride: int) -> None:
        super().__init__()
        c = cfg.encoder_hidden_dim
        self.conv1 = DFineConvNormLayer(cfg, c, c, 1, 1)
        self.conv2 = DFineConvNormLayer(cfg, c, c, kernel_size, stride, groups=c)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.conv1(x))


class DFineMultiheadAttention(nn.Module):
    """Matches `v4.52` key names (`q_proj`, `k_proj`, `v_proj`, `out_proj`)."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: Tensor, position_embeddings: Optional[Tensor] = None) -> Tensor:
        bsz, seq_len, embed_dim = hidden_states.shape
        qk_in = hidden_states if position_embeddings is None else hidden_states + position_embeddings
        q = (self.q_proj(qk_in) * self.scaling).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(qk_in).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
        return self.out_proj(out)


class DFineEncoderLayer(nn.Module):
    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        c = cfg.encoder_hidden_dim
        self.self_attn = DFineMultiheadAttention(c, cfg.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(c, eps=cfg.layer_norm_eps)
        self.activation_fn = _act_fn("gelu")
        self.fc1 = nn.Linear(c, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, c)
        self.final_layer_norm = nn.LayerNorm(c, eps=cfg.layer_norm_eps)

    def forward(self, x: Tensor, pos_embed: Optional[Tensor] = None) -> Tensor:
        x = self.self_attn_layer_norm(x + self.self_attn(x, position_embeddings=pos_embed))
        x = self.final_layer_norm(x + self.fc2(self.activation_fn(self.fc1(x))))
        return x


class DFineEncoderBlock(nn.Module):
    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DFineEncoderLayer(cfg) for _ in range(cfg.encoder_layers)])

    def forward(self, x: Tensor, pos_embed: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed)
        return x


def build_dfine_2d_sincos_position_embedding(
    width: int,
    height: int,
    embed_dim: int,
    temperature: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Exact `Peterande/D-FINE` & `transformers v4.52` `float32` 2D sin-cos embedding."""
    grid_w = torch.arange(int(width), device=device, dtype=torch.float32)
    grid_h = torch.arange(int(height), device=device, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, device=device, dtype=torch.float32) / float(pos_dim)
    omega = 1.0 / (float(temperature) ** omega)
    out_w = grid_w.reshape(-1, 1) * omega.reshape(1, -1)
    out_h = grid_h.reshape(-1, 1) * omega.reshape(1, -1)
    emb = torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1).unsqueeze(0)
    return emb.to(dtype=dtype)


class DFineHybridEncoder(nn.Module):
    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_fpn_stages = len(cfg.encoder_in_channels) - 1
        self.encoder_hidden_dim = cfg.encoder_hidden_dim
        self.encode_proj_layers = cfg.encode_proj_layers
        self.positional_encoding_temperature = cfg.positional_encoding_temperature

        # Named `encoder` to match safetensors `model.encoder.encoder.{i}.layers.{j}.*`
        self.encoder = nn.ModuleList([DFineEncoderBlock(cfg) for _ in range(len(cfg.encode_proj_layers))])
        num_blocks = round(3 * cfg.depth_mult)
        self.lateral_convs = nn.ModuleList(
            [DFineConvNormLayer(cfg, self.encoder_hidden_dim, self.encoder_hidden_dim, 1, 1) for _ in range(self.num_fpn_stages)]
        )
        self.fpn_blocks = nn.ModuleList(
            [DFineRepNCSPELAN4(cfg, numb_blocks=num_blocks) for _ in range(self.num_fpn_stages)]
        )
        self.downsample_convs = nn.ModuleList(
            [DFineSCDown(cfg, 3, 2) for _ in range(self.num_fpn_stages)]
        )
        self.pan_blocks = nn.ModuleList(
            [DFineRepNCSPELAN4(cfg, numb_blocks=num_blocks) for _ in range(self.num_fpn_stages)]
        )

    def forward(self, proj_feats: List[Tensor]) -> List[Tensor]:
        feats = list(proj_feats)
        if self.cfg.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feat = feats[enc_ind]
                bsz, ch, h, w = feat.shape
                flat = feat.flatten(2).permute(0, 2, 1)
                pos = build_dfine_2d_sincos_position_embedding(
                    width=int(w),
                    height=int(h),
                    embed_dim=self.encoder_hidden_dim,
                    temperature=self.positional_encoding_temperature,
                    device=flat.device,
                    dtype=flat.dtype,
                )
                flat = self.encoder[i](flat, pos_embed=pos)
                feats[enc_ind] = flat.permute(0, 2, 1).reshape(bsz, ch, h, w).contiguous()

        fpn_maps: List[Tensor] = [feats[-1]]
        for idx, (lat_conv, fpn_blk) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            back_map = feats[self.num_fpn_stages - idx - 1]
            top_map = lat_conv(fpn_maps[-1])
            fpn_maps[-1] = top_map
            up_map = F.interpolate(top_map, scale_factor=2.0, mode="nearest")
            fpn_maps.append(fpn_blk(torch.cat([up_map, back_map], dim=1)))

        fpn_maps = fpn_maps[::-1]
        pan_maps: List[Tensor] = [fpn_maps[0]]
        for idx, (down_conv, pan_blk) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            down_map = down_conv(pan_maps[-1])
            pan_maps.append(pan_blk(torch.cat([down_map, fpn_maps[idx + 1]], dim=1)))
        return pan_maps


class DFineGate(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, second_residual: Tensor, hidden_states: Tensor) -> Tensor:
        gates = torch.sigmoid(self.gate(torch.cat([second_residual, hidden_states], dim=-1)))
        g1, g2 = gates.chunk(2, dim=-1)
        return self.norm(g1 * second_residual + g2 * hidden_states)


class DFineMultiscaleDeformableAttention(nn.Module):
    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.decoder_attention_heads
        self.n_levels = cfg.num_feature_levels
        self.offset_scale = float(cfg.decoder_offset_scale)
        self.num_points_list = list(cfg.decoder_n_points)
        num_points_scale = [1.0 / n for n in self.num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))
        self.total_points = self.n_heads * sum(self.num_points_list)
        self.sampling_offsets = nn.Linear(self.d_model, self.total_points * 2)
        self.attention_weights = nn.Linear(self.d_model, self.total_points)

    def forward(
        self,
        hidden_states: Tensor,
        reference_points: Tensor,
        encoder_hidden_states: Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
    ) -> Tensor:
        bsz, num_queries, _ = hidden_states.shape
        _, seq_len, _ = encoder_hidden_states.shape
        head_dim = self.d_model // self.n_heads
        total_pts_per_head = sum(self.num_points_list)

        value = encoder_hidden_states.reshape(bsz, seq_len, self.n_heads, head_dim)
        sampling_offsets = self.sampling_offsets(hidden_states).reshape(
            bsz, num_queries, self.n_heads, total_pts_per_head, 2
        )
        attn_weights = F.softmax(
            self.attention_weights(hidden_states).reshape(bsz, num_queries, self.n_heads, total_pts_per_head),
            dim=-1,
        )

        # reference_points: (B, Q, 1, 4)
        pts_scale = self.num_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
        offset = sampling_offsets * pts_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
        sampling_locations = reference_points[:, :, None, :, :2] + offset

        value_list = (
            value.permute(0, 2, 3, 1)
            .flatten(0, 1)
            .split([int(h) * int(w) for h, w in spatial_shapes_list], dim=-1)
        )
        sampling_grids = (2.0 * sampling_locations - 1.0).permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampling_grids_split = sampling_grids.split(self.num_points_list, dim=-2)

        sampled_levels: List[Tensor] = []
        for lvl, (h, w) in enumerate(spatial_shapes_list):
            val_l = value_list[lvl].reshape(bsz * self.n_heads, head_dim, int(h), int(w))
            sampled_l = F.grid_sample(
                val_l,
                sampling_grids_split[lvl],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_levels.append(sampled_l)

        attn_flat = attn_weights.permute(0, 2, 1, 3).reshape(bsz * self.n_heads, 1, num_queries, total_pts_per_head)
        out = (torch.cat(sampled_levels, dim=-1) * attn_flat).sum(dim=-1).reshape(bsz, self.d_model, num_queries)
        return out.transpose(1, 2).contiguous()


class DFineDecoderLayer(nn.Module):
    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.self_attn = DFineMultiheadAttention(d, cfg.decoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.encoder_attn = DFineMultiscaleDeformableAttention(cfg)
        self.activation_fn = _act_fn("relu")
        self.fc1 = nn.Linear(d, cfg.decoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.decoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.gateway = DFineGate(d, eps=cfg.layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tensor,
        reference_points: Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        encoder_hidden_states: Tensor,
    ) -> Tensor:
        h2 = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = self.self_attn_layer_norm(hidden_states + h2)
        residual = hidden_states
        h_cross = self.encoder_attn(
            hidden_states=hidden_states + position_embeddings,
            reference_points=reference_points,
            encoder_hidden_states=encoder_hidden_states,
            spatial_shapes_list=spatial_shapes_list,
        )
        hidden_states = self.gateway(residual, h_cross)
        h_ffn = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        hidden_states = self.final_layer_norm((hidden_states + h_ffn).clamp(min=-65504.0, max=65504.0))
        return hidden_states


class DFineLQE(nn.Module):
    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        self.top_prob_values = cfg.top_prob_values
        self.max_num_bins = cfg.max_num_bins
        self.reg_conf = DFineMLP(4 * (self.top_prob_values + 1), cfg.lqe_hidden_dim, 1, cfg.lqe_layers, act="relu")

    def forward(self, scores: Tensor, pred_corners: Tensor) -> Tensor:
        bsz, length, _ = pred_corners.shape
        prob = F.softmax(pred_corners.reshape(bsz, length, 4, self.max_num_bins + 1), dim=-1)
        prob_topk, _ = prob.topk(self.top_prob_values, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        return scores + self.reg_conf(stat.reshape(bsz, length, -1))


def _inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _weighting_function(max_num_bins: int, up: Tensor, reg_scale: Tensor) -> Tensor:
    up0 = up[0].abs()
    rs0 = reg_scale[0].abs()
    upper1 = up0 * rs0
    upper2 = upper1 * 2.0
    step = (upper1 + 1.0) ** (2.0 / float(max_num_bins - 2))
    left_vals = [-(step**i) + 1.0 for i in range(max_num_bins // 2 - 1, 0, -1)]
    right_vals = [(step**i) - 1.0 for i in range(1, max_num_bins // 2)]
    vals = [-upper2] + left_vals + [torch.zeros_like(up0)] + right_vals + [upper2]
    return torch.stack(vals, dim=0)


def _distance2bbox(points: Tensor, distance: Tensor, reg_scale: Tensor) -> Tensor:
    rs = reg_scale[0].abs()
    x0 = points[..., 0] - (0.5 * rs + distance[..., 0]) * (points[..., 2] / rs)
    y0 = points[..., 1] - (0.5 * rs + distance[..., 1]) * (points[..., 3] / rs)
    x1 = points[..., 0] + (0.5 * rs + distance[..., 2]) * (points[..., 2] / rs)
    y1 = points[..., 1] + (0.5 * rs + distance[..., 3]) * (points[..., 3] / rs)
    return _corners_to_center_format(torch.stack([x0, y0, x1, y1], dim=-1))


class DFineDecoder(nn.Module):
    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.eval_idx = cfg.decoder_layers - 1
        self.max_num_bins = cfg.max_num_bins
        self.layers = nn.ModuleList([DFineDecoderLayer(cfg) for _ in range(cfg.decoder_layers)])
        self.query_pos_head = DFineMLP(4, 2 * cfg.d_model, cfg.d_model, num_layers=2, act="relu")
        self.reg_scale = nn.Parameter(torch.tensor([cfg.reg_scale], dtype=torch.float32), requires_grad=False)
        self.up = nn.Parameter(torch.tensor([cfg.up], dtype=torch.float32), requires_grad=False)
        self.pre_bbox_head = DFineMLP(cfg.d_model, cfg.d_model, 4, num_layers=3, act="relu")
        self.lqe_layers = nn.ModuleList([DFineLQE(cfg) for _ in range(cfg.decoder_layers)])
        self.class_embed = nn.ModuleList([nn.Linear(cfg.d_model, cfg.num_labels) for _ in range(cfg.decoder_layers)])
        self.bbox_embed = nn.ModuleList(
            [DFineMLP(cfg.d_model, cfg.d_model, 4 * (cfg.max_num_bins + 1), num_layers=3, act="relu") for _ in range(cfg.decoder_layers)]
        )

    def forward(
        self,
        inputs_embeds: Tensor,
        encoder_hidden_states: Tensor,
        reference_points_unact: Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns `(final_logits, final_boxes_cxcywh, last_hidden_state)`."""
        hidden_states = inputs_embeds
        project = _weighting_function(self.max_num_bins, self.up, self.reg_scale).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        ref_points_detach = torch.sigmoid(reference_points_unact)

        output_detach: Tensor | float = 0.0
        pred_corners_undetach: Tensor | float = 0.0
        ref_points_initial = ref_points_detach
        final_logits = hidden_states.new_zeros((hidden_states.shape[0], hidden_states.shape[1], self.cfg.num_labels))
        final_boxes = ref_points_detach

        for i, decoder_layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = self.query_pos_head(ref_points_detach).clamp(min=-10.0, max=10.0)
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=query_pos_embed,
                reference_points=ref_points_input,
                spatial_shapes_list=spatial_shapes_list,
                encoder_hidden_states=encoder_hidden_states,
            )
            if i == 0:
                ref_points_initial = torch.sigmoid(
                    self.pre_bbox_head(hidden_states) + _inverse_sigmoid(ref_points_detach)
                ).detach()

            pred_corners = self.bbox_embed[i](hidden_states + output_detach) + pred_corners_undetach
            bsz, nq, _ = pred_corners.shape
            prob_bins = F.softmax(pred_corners.reshape(-1, self.max_num_bins + 1), dim=1)
            dist = torch.matmul(prob_bins, project).reshape(bsz, nq, 4)
            inter_ref_bbox = _distance2bbox(ref_points_initial, dist, self.reg_scale)

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = hidden_states.detach()

            if i == self.eval_idx:
                scores = self.class_embed[i](hidden_states)
                final_logits = self.lqe_layers[i](scores, pred_corners)
                final_boxes = inter_ref_bbox

        return final_logits, final_boxes, hidden_states


class _DFineInnerModel(nn.Module):
    """Matches exact `model.*` state-dict prefix of upstream safetensors."""

    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DFineConvEncoderWrapper(cfg)
        self.encoder_input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, cfg.encoder_hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(cfg.encoder_hidden_dim, eps=cfg.batch_norm_eps),
                )
                for in_ch in cfg.encoder_in_channels
            ]
        )
        self.encoder = DFineHybridEncoder(cfg)
        if cfg.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(cfg.num_labels + 1, cfg.d_model, padding_idx=cfg.num_labels)
        self.enc_output = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(cfg.d_model, cfg.num_labels)
        self.enc_bbox_head = DFineMLP(cfg.d_model, cfg.d_model, 4, num_layers=3, act="relu")
        self.decoder = DFineDecoder(cfg)
        decoder_input_proj: List[nn.Module] = []
        for in_ch in cfg.decoder_in_channels:
            if cfg.d_model == in_ch:
                decoder_input_proj.append(nn.Identity())
            else:
                decoder_input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, cfg.d_model, kernel_size=1, bias=False),
                        nn.BatchNorm2d(cfg.d_model, eps=cfg.batch_norm_eps),
                    )
                )
        self.decoder_input_proj = nn.ModuleList(decoder_input_proj)

    @staticmethod
    def _generate_anchors(
        spatial_shapes_list: List[Tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype,
        grid_size: float = 0.05,
    ) -> Tuple[Tensor, Tensor]:
        anchors: List[Tensor] = []
        for lvl, (h, w) in enumerate(spatial_shapes_list):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(int(h), device=device, dtype=dtype),
                torch.arange(int(w), device=device, dtype=dtype),
                indexing="ij",
            )
            grid_xy = (torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0) + 0.5)
            grid_xy_x = grid_xy[..., 0] / float(w)
            grid_xy_y = grid_xy[..., 1] / float(h)
            grid_norm = torch.stack([grid_xy_x, grid_xy_y], dim=-1)
            wh = torch.full_like(grid_norm, float(grid_size) * (2.0**lvl))
            anchors.append(torch.cat([grid_norm, wh], dim=-1).reshape(1, int(h) * int(w), 4))

        eps = 1e-2
        all_anchors = torch.cat(anchors, dim=1)
        valid_mask = ((all_anchors > eps) & (all_anchors < (1.0 - eps))).all(dim=-1, keepdim=True)
        safe_anchors = all_anchors.clamp(min=1e-4, max=1.0 - 1e-4)
        logit_anchors = torch.log(safe_anchors / (1.0 - safe_anchors))
        max_val = float(torch.finfo(torch.float16).max) if dtype in (torch.float16, torch.bfloat16) else 3.4e38
        logit_anchors = torch.where(
            valid_mask,
            logit_anchors,
            torch.full_like(logit_anchors, max_val),
        )
        return logit_anchors, valid_mask

    def forward(self, pixel_values: Tensor) -> Dict[str, object]:
        # pixel_values: [0, 1] RGB (B, 3, H, W)
        states, selected = self.backbone.model.forward_all_stages(pixel_values)
        proj_feats = [self.encoder_input_proj[i](feat) for i, feat in enumerate(selected)]
        pan_feats = self.encoder(proj_feats)

        sources = [self.decoder_input_proj[i](src) for i, src in enumerate(pan_feats)]
        spatial_shapes_list: List[Tuple[int, int]] = []
        source_flatten_list: List[Tensor] = []
        for src in sources:
            h, w = int(src.shape[-2]), int(src.shape[-1])
            spatial_shapes_list.append((h, w))
            source_flatten_list.append(src.flatten(2).transpose(1, 2))
        source_flatten = torch.cat(source_flatten_list, dim=1)

        anchors, valid_mask = self._generate_anchors(
            spatial_shapes_list, device=source_flatten.device, dtype=source_flatten.dtype
        )
        memory = valid_mask.to(source_flatten.dtype) * source_flatten
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(dim=-1).values, self.cfg.num_queries, dim=1)
        idx_box = topk_ind.unsqueeze(-1).expand(-1, -1, 4)
        idx_mem = topk_ind.unsqueeze(-1).expand(-1, -1, output_memory.shape[-1])
        ref_points_unact = enc_outputs_coord_logits.gather(dim=1, index=idx_box).detach()
        target = output_memory.gather(dim=1, index=idx_mem).detach()

        logits, pred_boxes_cxcywh, last_hidden_state = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            reference_points_unact=ref_points_unact,
            spatial_shapes_list=spatial_shapes_list,
        )
        return {
            # Backbone stages: stem (s4, 16), c2 (s4, 64), c3 (s8, 256), c4 (s16, 512), c5 (s32, 1024)
            "stem": states[0],
            "c2": states[1],
            "c3": states[2],
            "c4": states[3],
            "c5": states[4],
            "pan_feats": pan_feats,
            "logits": logits,
            "pred_boxes": pred_boxes_cxcywh,
            "query_embeds": last_hidden_state,
        }


class DFineCore(nn.Module):
    """Frozen-capable Apache-2.0 `D-FINE` core (`nano` / `small`)."""

    def __init__(self, cfg: DFineCoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _DFineInnerModel(cfg)
        self.loaded_pretrained_spec: Optional[DFinePretrainedSpec] = None
        self.loaded_sha256: Optional[str] = None

    @classmethod
    def from_pretrained_apache20(
        cls,
        variant: str = "nano",
        pretrained_dir: Optional[str | Path] = None,
        *,
        freeze: bool = True,
    ) -> "DFineCore":
        v = str(variant).lower().strip()
        if v not in DFINE_PRETRAINED_SPECS:
            raise ValueError(f"Unknown pretrained D-FINE variant {variant!r}")
        spec = DFINE_PRETRAINED_SPECS[v]
        cfg = DFineCoreConfig.from_variant(v)
        core = cls(cfg)
        base_dir = Path(pretrained_dir) if pretrained_dir is not None else default_pretrained_dir()
        st_path = base_dir / spec.filename
        state, sha = load_safetensors_state_dict(st_path, expected_sha256=spec.sha256)
        core.load_state_dict(state, strict=True)
        core.loaded_pretrained_spec = spec
        core.loaded_sha256 = sha
        if freeze:
            core.freeze_all()
        return core

    def freeze_all(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, images_rgb_01: Tensor) -> Dict[str, object]:
        return self.model(images_rgb_01)
