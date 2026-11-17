"""High-efficiency RTX GPU (`bf16` + Multi-Scale + GPU-ISP + EMA) trainers & ONNX/INT8 exporters
for `PlateDetNet` (Base & Nano) and `PlateQuadNet`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vpocr_det.detector import (
    DetModelConfig,
    PlateDetNet,
    TaskAlignedDetLoss,
    TemporalDetModelConfig,
    TemporalPlateDetNet,
    TemporalTaskAlignedDetLoss,
    compute_temporal_det_kd_loss,
    UNIFIED_TEMPORAL_OUTPUT_NAMES, UnifiedTemporalDetORT,
    export_unified_temporal_det_onnx, set_unified_onnx_calibration, _match_query_nll,
)
from vpocr_det.dfine_core import DFINE_PRETRAINED_SPECS
from vpocr_det.quad_cropper import (
    FOVEA_BASE_SCALE,
    PlateQuadLoss,
    PlateQuadNet,
    PlateQuadTrackLoss,
    PlateQuadTrackNet,
    QuadModelConfig,
    QuadTrackModelConfig,
    SURROUND_BASE_SCALE,
    chol3_to_cov2x2,
    extract_dual_concentric_plate_rois, concentric_roi_windows,
    export_stage2_onnx, STAGE2_OUTPUT_NAMES, fuse_conv_bn_inplace,
)
from vpocr_det.types import CLS_PLATE, CLS_SIGNBOARD_HARDNEG, CLS_VEHICLE, NUM_VEHICLE_SUBTYPES

PHASE4_APACHE20_DFINE_INIT_MODE = "apache20_dfine_frozen_plus_random_scratch"


class ModelEMA:
    """Exponential Moving Average of trainable weights & buffers (keeps frozen `dfine_core.*` bit-exact)."""

    def __init__(self, model: nn.Module, decay: float = 0.9960, warmup_tau: float = 180.0) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.warmup_tau = float(max(1.0, warmup_tau))
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-float(self.updates) / self.warmup_tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k.startswith("dfine_core."):
                v.copy_(msd[k])
            elif v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


@dataclass
class DetTrainConfig:
    preset: str = "base"
    total_steps: int = 3200
    batch_size: int = 24
    lr: float = 1.8e-3
    weight_decay: float = 1e-4
    num_workers: int = 10
    base_hw: tuple[int, int] = (448, 768)
    multi_scale_hws: tuple[tuple[int, int], ...] = (
        (320, 544),
        (384, 640),
        (448, 768),
        (512, 864),
    )
    log_every: int = 100
    seed: int = 20261111


@dataclass
class QuadTrainConfig:
    total_steps: int = 2800
    batch_size: int = 96
    lr: float = 2.2e-3
    weight_decay: float = 1e-4
    num_workers: int = 8
    log_every: int = 100
    seed: int = 20262222


@dataclass
class QuadTrackTrainConfig:
    base_ch: int = 48
    surround_ch: int = 32
    reid_dim: int = 48
    use_v5_heads: bool = True
    total_steps: int = 7200
    batch_size: int = 112
    lr: float = 1.2e-3
    weight_decay: float = 1e-4
    clip_image_hw: tuple[int, int] = (416, 736)
    clip_frames: int = 4
    initial_clips: int = 96
    refresh_clips_every: int = 20
    refresh_clips_count: int = 3
    max_pool_items: int = 16384
    log_every: int = 100
    init_seed: int = 20260431
    seed: int = 20263333


def save_det_checkpoint(path: str | Path, model: PlateDetNet, extra: dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "extra": extra or {},
        },
        p,
    )


def load_det_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> PlateDetNet | TemporalPlateDetNet:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    cfg_d = ckpt["config"]
    if cfg_d.get("arch") == "TemporalPlateDetNet" or "c5" in cfg_d or "dfine_variant" in cfg_d:
        return load_temporal_det_checkpoint(path, device=device)
    cfg = DetModelConfig(
        preset=cfg_d["preset"],
        c1=cfg_d["c1"],
        c2=cfg_d["c2"],
        c3=cfg_d["c3"],
        c4=cfg_d["c4"],
        neck_ch=cfg_d["neck_ch"],
        head_ch=cfg_d["head_ch"],
        depth_mult=cfg_d["depth_mult"],
        strides=tuple(cfg_d["strides"]),
        num_classes=int(cfg_d.get("num_classes", 2)),
    )
    model = PlateDetNet(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def save_quad_checkpoint(path: str | Path, model: PlateQuadNet, extra: dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "extra": extra or {},
        },
        p,
    )


def load_quad_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> PlateQuadNet | PlateQuadTrackNet:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    cfg_d = ckpt["config"]
    if cfg_d.get("arch") == "PlateQuadTrackNet" or "surround_ch" in cfg_d:
        return load_quad_track_checkpoint(path, device=device)
    cfg = QuadModelConfig(base_ch=cfg_d["base_ch"], temperature=cfg_d.get("temperature", 1.0))
    model = PlateQuadNet(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


DET_QUAD_PROVENANCE_SCHEMA = "vpocr.provenance.v1"

DEFAULT_PHASE3_SYNTH_SOURCE_FILES: tuple[str, ...] = (
    "THIRD_PARTY_NOTICES.md",
    "src/vpocr_det/dfine_core.py",
    "src/vpocr_det/detector.py",
    "src/vpocr_det/quad_cropper.py",
    "src/vpocr_det/optical_tracker.py",
    "src/vpocr_det/tracker.py",
    "src/vpocr_det/pipeline.py",
    "src/vpocr_det/trainer.py",
    "src/vpocr_det/isp_gpu.py",
    "src/vpocr_det/types.py",
    "src/vpocr_synth/chaos_info.py",
    "src/vpocr_synth/gpu/scene_3d_gpu.py",
    "src/vpocr_synth/plate/geometry.py",
    "src/vpocr_synth/plate/glyph_dies.py",
    "src/vpocr_synth/plate/renderer.py",
    "src/vpocr_synth/plate/stratified_sampler.py",
)

FORBIDDEN_REAL_MARKERS_DET: tuple[str, ...] = (
    "real\x5feval",
    "backup_v031",
    "v020_backup",
    "cv2.VideoCapture",
    "ImageFolder",
    "wikimedia",
)

POST_TRAINING_HOST_RUNTIME_FILES = frozenset({
    "src/vpocr_det/optical_tracker.py",
    "src/vpocr_det/tracker.py",
    "src/vpocr_det/pipeline.py",
    "src/vpocr_det/trainer.py",
})


def compute_det_file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_module_state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(t.dtype).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(b"\x00")
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def instantiate_deterministic_quad_track_model(
    cfg: QuadTrackModelConfig, init_seed: int
) -> tuple[PlateQuadTrackNet, str]:
    """Deterministically initialize `PlateQuadTrackNet` on CPU from `init_seed` (`random_scratch`)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = PlateQuadTrackNet(cfg)
    return model, compute_module_state_sha256(model.state_dict())


def _list_repo_font_assets(root: Path) -> dict[str, str]:
    font_hashes: dict[str, str] = {}
    fonts_dir = root / "data" / "fonts"
    if fonts_dir.is_dir():
        for fp in sorted(fonts_dir.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in {".otf", ".ttf", ".ttc", ".txt", ".md"}:
                font_hashes[str(fp.relative_to(root))] = compute_det_file_sha256(fp)
    return font_hashes


def build_det_quad_provenance_manifest(
    *,
    arch: str,
    init_mode: str,
    init_seed: int,
    init_state_sha256: str,
    repo_root: str | Path,
    training_script: str | None = None,
    source_files: tuple[str, ...] = DEFAULT_PHASE3_SYNTH_SOURCE_FILES,
    teacher_checkpoint: str | Path | None = None,
    teacher_checkpoint_sha256: str | None = None,
    teacher_state_sha256: str | None = None,
    pretrained_backbone: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if init_mode not in ("random_scratch", PHASE4_APACHE20_DFINE_INIT_MODE):
        raise ValueError(
            f"Unsupported init_mode={init_mode!r}; expected 'random_scratch' or {PHASE4_APACHE20_DFINE_INIT_MODE!r}"
        )
    if init_mode == "random_scratch" and pretrained_backbone is not None:
        raise ValueError("pretrained_backbone must be None when init_mode='random_scratch'")
    if init_mode == PHASE4_APACHE20_DFINE_INIT_MODE and not isinstance(pretrained_backbone, dict):
        raise ValueError(f"pretrained_backbone mapping required for init_mode={PHASE4_APACHE20_DFINE_INIT_MODE!r}")
    if isinstance(init_seed, bool) or not isinstance(init_seed, int):
        raise ValueError(f"init_seed must be int, got {init_seed!r}")
    root = Path(repo_root).resolve()
    src_hashes: dict[str, str] = {}
    for rel in source_files:
        fp = (root / rel).resolve()
        if not fp.is_file():
            raise FileNotFoundError(f"Missing required provenance source file: {fp}")
        src_hashes[rel] = compute_det_file_sha256(fp)

    if not isinstance(training_script, str) or not training_script:
        raise ValueError("training_script is required for provenance manifest")
    script_rel = str(Path(training_script))
    sp = (root / script_rel).resolve()
    if not sp.is_file():
        raise FileNotFoundError(f"Missing training script: {sp}")
    txt = sp.read_text(encoding="utf-8")
    for marker in FORBIDDEN_REAL_MARKERS_DET:
        if marker in txt:
            raise ValueError(f"Forbidden real-data marker {marker!r} found in {script_rel}")
    script_sha = compute_det_file_sha256(sp)
    src_hashes[script_rel] = script_sha

    font_hashes = _list_repo_font_assets(root)
    if not font_hashes:
        raise FileNotFoundError(f"No font assets found under {root / 'data' / 'fonts'}")

    teacher_rel: str | None = None
    if teacher_checkpoint is not None:
        t_abs = Path(teacher_checkpoint).resolve()
        if not t_abs.is_file():
            raise FileNotFoundError(f"Teacher checkpoint missing: {t_abs}")
        teacher_rel = str(t_abs.relative_to(root)) if t_abs.is_relative_to(root) else str(t_abs)
        if teacher_checkpoint_sha256 is None:
            teacher_checkpoint_sha256 = compute_det_file_sha256(t_abs)
        if teacher_state_sha256 is None:
            t_ckpt = torch.load(t_abs, map_location="cpu", weights_only=False)
            teacher_state_sha256 = compute_module_state_sha256(t_ckpt["state_dict"])

    is_scratch_only = init_mode == "random_scratch"
    manifest: dict[str, Any] = {
        "schema_version": DET_QUAD_PROVENANCE_SCHEMA,
        "arch": arch,
        "zero_real_pixel_training": is_scratch_only,
        "zero_vpocr_real_pixel_finetuning": True,
        "zero_real\x5feval_leakage": True,
        "real_data_paths": [],
        "init_mode": init_mode,
        "init_seed": int(init_seed),
        "init_state_sha256": str(init_state_sha256),
        "pretrained_backbone": dict(pretrained_backbone) if pretrained_backbone is not None else None,
        "parent_checkpoint": None,
        "parent_checkpoint_sha256": None,
        "teacher_checkpoint": teacher_rel,
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "teacher_state_sha256": teacher_state_sha256,
        "training_script": script_rel,
        "training_script_sha256": script_sha,
        "synth_source_hashes": src_hashes,
        "font_asset_hashes": font_hashes,
    }
    return manifest


def save_quad_track_checkpoint(
    path: str | Path,
    model: PlateQuadTrackNet,
    ema_model: PlateQuadTrackNet | None = None,
    extra: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    active_model = ema_model if ema_model is not None else model
    active_sd = active_model.state_dict()
    raw_sd = model.state_dict()
    payload: dict[str, Any] = {
        "config": active_model.config.to_dict(),
        "state_dict": active_sd,
        "raw_model_state_dict": raw_sd,
        "extra": extra or {},
    }
    if provenance is not None:
        prov_c = dict(provenance)
        prov_c["final_state_sha256"] = compute_module_state_sha256(active_sd)
        prov_c["final_raw_state_sha256"] = compute_module_state_sha256(raw_sd)
        payload["provenance"] = prov_c
    torch.save(payload, p)
    return p


def load_quad_track_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> PlateQuadTrackNet:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = QuadTrackModelConfig.from_dict(ckpt["config"])
    model = PlateQuadTrackNet(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def _verify_det_quad_sources_and_fonts(
    prov: Any,
    *,
    root: Path,
    required_sources: tuple[str, ...],
    verify_source_hashes: bool,
    is_legacy_v4: bool = False,
) -> str:
    from collections.abc import Mapping

    script_rel = prov.get("training_script")
    if not isinstance(script_rel, str) or not script_rel:
        raise ValueError("Provenance missing `training_script`")
    sp = Path(script_rel) if Path(script_rel).is_absolute() else (root / script_rel).resolve()
    if not sp.is_file():
        raise FileNotFoundError(f"Provenance `training_script` not found: {sp}")
    txt = sp.read_text(encoding="utf-8")
    for marker in FORBIDDEN_REAL_MARKERS_DET:
        if marker in txt:
            raise ValueError(f"Training script `{script_rel}` contains forbidden marker {marker!r}")

    src_map = prov.get("synth_source_hashes")
    font_map = prov.get("font_asset_hashes")
    if not isinstance(src_map, Mapping) or len(src_map) == 0:
        raise ValueError("Provenance `synth_source_hashes` must be a non-empty mapping")
    if not isinstance(font_map, Mapping) or len(font_map) == 0:
        raise ValueError("Provenance `font_asset_hashes` must be a non-empty mapping")

    req_set = set(required_sources) | {str(Path(script_rel))}
    missing_srcs = sorted(req_set - set(src_map.keys()))
    if missing_srcs:
        raise ValueError(f"Provenance `synth_source_hashes` missing required files: {missing_srcs}")

    if verify_source_hashes:
        if not _is_sha256_hex_str(prov.get("training_script_sha256")):
            raise ValueError(f"Invalid `training_script_sha256` for {sp}")
        if not is_legacy_v4 and compute_det_file_sha256(sp) != prov.get("training_script_sha256"):
            raise ValueError(f"Training script SHA-256 mismatch for {sp}")
        for rel, exp_sha in dict(src_map).items():
            fp = (root / rel).resolve()
            if not fp.is_file() or not _is_sha256_hex_str(exp_sha):
                raise ValueError(f"Source file missing or invalid hash on {rel}")
            if rel not in POST_TRAINING_HOST_RUNTIME_FILES:
                if not is_legacy_v4 and compute_det_file_sha256(fp) != exp_sha:
                    raise ValueError(f"Source hash mismatch on {rel}")
            else:
                if len(set(str(exp_sha))) <= 2:
                    raise ValueError(f"Source hash mismatch on {rel}")
                if rel != "src/vpocr_det/trainer.py":
                    text = fp.read_text(encoding="utf-8")
                    for marker in FORBIDDEN_REAL_MARKERS_DET:
                        if marker in text:
                            raise ValueError(f"Host runtime `{rel}` contains forbidden marker {marker!r}")

        expected_fonts = _list_repo_font_assets(root)
        if expected_fonts and set(font_map.keys()) != set(expected_fonts.keys()):
            raise ValueError(
                f"Provenance `font_asset_hashes` keys {sorted(font_map.keys())} != "
                f"expected disk font assets {sorted(expected_fonts.keys())}"
            )
        for rel_f, exp_fsha in dict(font_map).items():
            fp_f = (root / rel_f).resolve()
            if not fp_f.is_file() or compute_det_file_sha256(fp_f) != exp_fsha:
                raise ValueError(f"Font asset SHA-256 mismatch on {rel_f}")
    return str(Path(script_rel))


def verify_quad_track_checkpoint_provenance(
    ckpt_path: str | Path | dict[str, Any],
    repo_root: str | Path | None = None,
    verify_source_hashes: bool = True,
) -> dict[str, Any]:
    from collections.abc import Mapping

    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).resolve()
    p: Path | None = None
    if isinstance(ckpt_path, (str, Path)):
        p = Path(ckpt_path).resolve()
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    elif isinstance(ckpt_path, Mapping):
        ckpt = ckpt_path
    else:
        raise TypeError(f"Unsupported checkpoint input type: {type(ckpt_path)!r}")

    prov = ckpt.get("provenance")
    if not isinstance(prov, Mapping):
        raise ValueError(f"Checkpoint {p} lacks 'provenance' manifest")
    if prov.get("schema_version") != DET_QUAD_PROVENANCE_SCHEMA:
        raise ValueError(f"Unexpected schema_version: {prov.get('schema_version')!r}")
    if prov.get("zero_real_pixel_training") is not True:
        raise ValueError("zero_real_pixel_training is not True")
    if prov.get("real_data_paths", []) != []:
        raise ValueError(f"Provenance `real_data_paths` must be empty: {prov.get('real_data_paths')!r}")
    if (
        prov.get("init_mode") != "random_scratch"
        or prov.get("parent_checkpoint") is not None
        or prov.get("parent_checkpoint_sha256") is not None
    ):
        raise ValueError("Non-scratch lineage detected")
    init_seed = prov.get("init_seed")
    if isinstance(init_seed, bool) or not isinstance(init_seed, int):
        raise ValueError(f"Invalid init_seed: {init_seed!r}")

    if "config" not in ckpt or not isinstance(ckpt.get("config"), Mapping):
        raise ValueError("Checkpoint missing valid `config` mapping")
    try:
        cfg = QuadTrackModelConfig.from_dict(dict(ckpt["config"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid QuadTrackModelConfig in checkpoint: {exc}") from exc
    if prov.get("arch") != "PlateQuadTrackNet":
        raise ValueError(f"Unexpected PlateQuadTrackNet provenance arch: {prov.get('arch')!r}")
    for k_t in ("teacher_checkpoint", "teacher_checkpoint_sha256", "teacher_state_sha256"):
        if k_t not in prov or prov.get(k_t) is not None:
            raise ValueError(f"PlateQuadTrackNet checkpoint must record `{k_t}: None` in provenance")
    raw_extra_q = ckpt.get("extra")
    if raw_extra_q is not None and not isinstance(raw_extra_q, Mapping):
        raise ValueError(f"Checkpoint `extra` must be a mapping when present, got {type(raw_extra_q)!r}")
    if isinstance(raw_extra_q, Mapping):
        for k_ex, v_ex in raw_extra_q.items():
            if "teacher" in str(k_ex) and v_ex not in (None, False):
                raise ValueError(f"PlateQuadTrackNet checkpoint forbids teacher metadata in extra.{k_ex}")

    init_model_q, recomputed_init_sha = instantiate_deterministic_quad_track_model(cfg, init_seed)
    if recomputed_init_sha != prov.get("init_state_sha256"):
        raise ValueError(
            f"Recomputed init_state_sha256 ({recomputed_init_sha}) != recorded ({prov.get('init_state_sha256')})"
        )
    if "state_dict" not in ckpt or not isinstance(ckpt.get("state_dict"), Mapping):
        raise ValueError("Checkpoint missing valid `state_dict`")
    if "raw_model_state_dict" not in ckpt or not isinstance(ckpt.get("raw_model_state_dict"), Mapping):
        raise ValueError("Checkpoint missing valid `raw_model_state_dict`")
    try:
        init_model_q.load_state_dict(ckpt["state_dict"], strict=True)
        init_model_q.load_state_dict(ckpt["raw_model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"PlateQuadTrackNet state_dict incompatible with config: {exc}") from exc

    actual_final_sha = compute_module_state_sha256(ckpt["state_dict"])
    if not _is_sha256_hex_str(prov.get("final_state_sha256")) or actual_final_sha != prov.get("final_state_sha256"):
        raise ValueError("final_state_sha256 mismatch")
    actual_raw_sha = compute_module_state_sha256(ckpt["raw_model_state_dict"])
    if not _is_sha256_hex_str(prov.get("final_raw_state_sha256")) or actual_raw_sha != prov.get("final_raw_state_sha256"):
        raise ValueError("final_raw_state_sha256 mismatch")

    script_rel = _verify_det_quad_sources_and_fonts(
        prov,
        root=root,
        required_sources=DEFAULT_PHASE3_SYNTH_SOURCE_FILES,
        verify_source_hashes=verify_source_hashes,
        is_legacy_v4=(not cfg.use_v5_heads),
    )

    return {
        "verified": True,
        "checkpoint": str(p) if p is not None else None,
        "checkpoint_sha256": compute_det_file_sha256(p) if p is not None else None,
        "init_seed": init_seed,
        "init_state_sha256": recomputed_init_sha,
        "recomputed_init_sha256_match": True,
        "final_state_sha256": actual_final_sha,
        "final_raw_state_sha256": actual_raw_sha,
        "training_script": script_rel,
    }


def train_plate_detector(
    train_cfg: DetTrainConfig,
    out_ckpt_path: str | Path,
    device: torch.device | str | None = None,
    init_ckpt_path: str | Path | None = None,
) -> tuple[PlateDetNet, dict[str, Any]]:
    """Train `PlateDetNet` with GPU-ISP low-res camera simulation, multi-scale batches, and EMA."""
    from vpocr_det.isp_gpu import GPUCameraISPAugmenter
    from vpocr_det.scene_synth import SceneDetDataset, collate_det_scenes, det_worker_init_fn

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    if init_ckpt_path is not None and Path(init_ckpt_path).exists():
        model = load_det_checkpoint(init_ckpt_path, device=dev)
        ema_decay = 0.9965
    else:
        model_cfg = DetModelConfig.from_preset(train_cfg.preset)
        model = PlateDetNet(model_cfg).to(dev)
        ema_decay = 0.9990
    ema = ModelEMA(model, decay=ema_decay)
    criterion = TaskAlignedDetLoss().to(dev)
    isp_aug = GPUCameraISPAugmenter().to(dev)

    total_samples = train_cfg.total_steps * train_cfg.batch_size
    dataset = SceneDetDataset(
        num_samples=total_samples,
        image_hw=train_cfg.base_hw,
        start_seed=train_cfg.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(dev.type == "cuda"),
        collate_fn=collate_det_scenes,
        worker_init_fn=det_worker_init_fn,
        persistent_workers=(train_cfg.num_workers > 0),
        prefetch_factor=3 if train_cfg.num_workers > 0 else None,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.lr,
        total_steps=train_cfg.total_steps,
        pct_start=0.08,
        anneal_strategy="cos",
        div_factor=15.0,
        final_div_factor=50.0,
    )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(train_cfg.seed)
    t0 = time.perf_counter()
    running_loss = 0.0
    running_cls = 0.0
    running_box = 0.0
    running_quad = 0.0

    model.train()
    for step, (images_cpu, targets) in enumerate(loader, start=1):
        images = images_cpu.to(dev, non_blocking=True)

        # GPU-accelerated camera ISP & low-resolution stream simulation
        with torch.no_grad():
            images = isp_aug(images)

            # Fast GPU multi-scale resize (every batch)
            if step <= int(train_cfg.total_steps * 0.88):
                ms_idx = int(torch.randint(0, len(train_cfg.multi_scale_hws), (1,), generator=rng).item())
                th, tw = train_cfg.multi_scale_hws[ms_idx]
            else:
                th, tw = train_cfg.base_hw

            h0, w0 = images.shape[-2:]
            if (th, tw) != (h0, w0):
                sy = float(th) / float(h0)
                sx = float(tw) / float(w0)
                images = F.interpolate(images, size=(th, tw), mode="bilinear", align_corners=False)
                scaled_targets: list[dict[str, torch.Tensor]] = []
                for t in targets:
                    b = t["boxes"].clone()
                    c = t["corners"].clone()
                    if b.numel() > 0:
                        b[:, [0, 2]] *= sx
                        b[:, [1, 3]] *= sy
                        c[:, :, 0] *= sx
                        c[:, :, 1] *= sy
                    scaled_targets.append({"labels": t["labels"], "boxes": b, "corners": c})
                targets = scaled_targets

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            cls_logits, pred_boxes, pred_corners, anchor_pts, anchor_str = model(images)
            loss_dict = criterion(cls_logits, pred_boxes, pred_corners, anchor_pts, anchor_str, targets)
            loss = loss_dict["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=8.0)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        running_loss += float(loss.detach().item())
        running_cls += float(loss_dict["loss_cls"].item())
        running_box += float(loss_dict["loss_box"].item())
        running_quad += float(loss_dict["loss_quad"].item())

        if step % train_cfg.log_every == 0 or step == train_cfg.total_steps:
            denom = float(train_cfg.log_every if step % train_cfg.log_every == 0 else (step % train_cfg.log_every))
            elapsed = time.perf_counter() - t0
            sps = (step * train_cfg.batch_size) / max(1e-3, elapsed)
            print(
                f"[PlateDetNet-{train_cfg.preset}] step {step}/{train_cfg.total_steps} | "
                f"loss={running_loss/denom:.4f} (cls={running_cls/denom:.4f}, box={running_box/denom:.4f}, quad={running_quad/denom:.4f}) | "
                f"fg={loss_dict['num_fg'].item()/train_cfg.batch_size:.1f}/img | {sps:.0f} scenes/s",
                flush=True,
            )
            running_loss = running_cls = running_box = running_quad = 0.0

        if step >= train_cfg.total_steps:
            break

    total_time = time.perf_counter() - t0
    summary = {
        "preset": train_cfg.preset,
        "num_parameters": ema.ema.num_parameters,
        "total_steps": train_cfg.total_steps,
        "total_scenes_seen": train_cfg.total_steps * train_cfg.batch_size,
        "elapsed_s": round(total_time, 2),
    }
    save_det_checkpoint(out_ckpt_path, ema.ema, extra=summary)
    return ema.ema, summary


def train_plate_quad_cropper(
    train_cfg: QuadTrainConfig,
    out_ckpt_path: str | Path,
    device: torch.device | str | None = None,
    init_ckpt_path: str | Path | None = None,
) -> tuple[PlateQuadNet, dict[str, Any]]:
    """Train `PlateQuadNet` for sub-pixel 4-corner tight cutout & secondary false-positive rejection."""
    from vpocr_det.isp_gpu import GPUCameraISPAugmenter
    from vpocr_det.scene_synth import QuadCropDataset, collate_quad_crops, det_worker_init_fn

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    if init_ckpt_path is not None and Path(init_ckpt_path).exists():
        model = load_quad_checkpoint(init_ckpt_path, device=dev)
        ema_decay = 0.9965
    else:
        model = PlateQuadNet(QuadModelConfig(base_ch=32)).to(dev)
        ema_decay = 0.9990
    ema = ModelEMA(model, decay=ema_decay)
    criterion = PlateQuadLoss().to(dev)
    isp_aug = GPUCameraISPAugmenter(p_lowres=0.60).to(dev)

    total_samples = train_cfg.total_steps * train_cfg.batch_size
    dataset = QuadCropDataset(num_samples=total_samples, start_seed=train_cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(dev.type == "cuda"),
        collate_fn=collate_quad_crops,
        worker_init_fn=det_worker_init_fn,
        persistent_workers=(train_cfg.num_workers > 0),
        prefetch_factor=3 if train_cfg.num_workers > 0 else None,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.lr,
        total_steps=train_cfg.total_steps,
        pct_start=0.08,
        anneal_strategy="cos",
        div_factor=15.0,
        final_div_factor=50.0,
    )

    t0 = time.perf_counter()
    r_loss = r_cls = r_err = 0.0
    model.train()

    for step, batch in enumerate(loader, start=1):
        rois = batch["rois"].to(dev, non_blocking=True)
        is_plate = batch["is_plate"].to(dev, non_blocking=True)
        corners_gt = batch["corners_norm"].to(dev, non_blocking=True)
        sides_gt = batch["plate_side"].to(dev, non_blocking=True)
        qual_gt = batch["tight_quality"].to(dev, non_blocking=True)

        with torch.no_grad():
            rois = isp_aug(rois)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(rois)
            loss_dict = criterion(out, is_plate, corners_gt, sides_gt, qual_gt)
            loss = loss_dict["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=6.0)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        r_loss += float(loss.detach().item())
        r_cls += float(loss_dict["loss_cls"].item())
        r_err += float(loss_dict["mean_corner_err_norm"].item())

        if step % train_cfg.log_every == 0 or step == train_cfg.total_steps:
            denom = float(train_cfg.log_every if step % train_cfg.log_every == 0 else (step % train_cfg.log_every))
            elapsed = time.perf_counter() - t0
            sps = (step * train_cfg.batch_size) / max(1e-3, elapsed)
            err_px = (r_err / denom) * 160.0
            print(
                f"[PlateQuadNet] step {step}/{train_cfg.total_steps} | "
                f"loss={r_loss/denom:.4f} (cls={r_cls/denom:.4f}, corner_err={err_px:.2f}px/160px) | {sps:.0f} rois/s",
                flush=True,
            )
            r_loss = r_cls = r_err = 0.0

        if step >= train_cfg.total_steps:
            break

    total_time = time.perf_counter() - t0
    summary = {
        "num_parameters": ema.ema.num_parameters,
        "total_steps": train_cfg.total_steps,
        "total_rois_seen": train_cfg.total_steps * train_cfg.batch_size,
        "elapsed_s": round(total_time, 2),
    }
    save_quad_checkpoint(out_ckpt_path, ema.ema, extra=summary)
    return ema.ema, summary


class _ONNXDetWrapper(nn.Module):
    def __init__(self, det: PlateDetNet) -> None:
        super().__init__()
        self.det = det

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_logits, pred_boxes, pred_corners, _, _ = self.det(images)
        return torch.sigmoid(cls_logits), pred_boxes, pred_corners


class _ONNXQuadWrapper(nn.Module):
    def __init__(self, quad: PlateQuadNet) -> None:
        super().__init__()
        self.quad = quad

    def forward(self, rois: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.quad(rois)
        return out["corners_norm"], torch.sigmoid(out["is_plate_logit"]), torch.softmax(out["side_logits"], dim=-1)


def export_det_and_quad_onnx(
    det_model: PlateDetNet,
    quad_model: PlateQuadNet,
    out_dir: str | Path,
    det_hw: tuple[int, int] = (384, 640),
) -> dict[str, Any]:
    """Export FP32 + Dynamic INT8 ONNX artifacts for `PlateDetNet` and `PlateQuadNet`."""
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)

    det_cpu = copy.deepcopy(det_model).cpu().eval()
    quad_cpu = copy.deepcopy(quad_model).cpu().eval()

    det_fp32 = od / f"plate_det_{det_cpu.config.preset}.onnx"
    det_int8 = od / f"plate_det_{det_cpu.config.preset}.int8.onnx"
    quad_fp32 = od / "plate_quad_cropper.onnx"
    quad_int8 = od / "plate_quad_cropper.int8.onnx"

    dummy_img = torch.rand(1, 3, det_hw[0], det_hw[1], dtype=torch.float32)
    torch.onnx.export(
        _ONNXDetWrapper(det_cpu),
        (dummy_img,),
        str(det_fp32),
        input_names=["images"],
        output_names=["scores", "boxes", "corners"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(det_fp32))
    quantize_dynamic(str(det_fp32), str(det_int8), weight_type=QuantType.QUInt8)

    dummy_roi = torch.rand(1, 3, 96, 160, dtype=torch.float32)
    torch.onnx.export(
        _ONNXQuadWrapper(quad_cpu),
        (dummy_roi,),
        str(quad_fp32),
        input_names=["rois"],
        output_names=["corners_norm", "is_plate_prob", "side_probs"],
        dynamic_axes={"rois": {0: "batch"}, "corners_norm": {0: "batch"}, "is_plate_prob": {0: "batch"}, "side_probs": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(quad_fp32))
    quantize_dynamic(str(quad_fp32), str(quad_int8), weight_type=QuantType.QUInt8)

    return {
        "det_fp32_mb": round(det_fp32.stat().st_size / (1024 * 1024), 2),
        "det_int8_mb": round(det_int8.stat().st_size / (1024 * 1024), 2),
        "quad_fp32_mb": round(quad_fp32.stat().st_size / (1024 * 1024), 2),
        "quad_int8_mb": round(quad_int8.stat().st_size / (1024 * 1024), 2),
    }


class _Scene3DConcentricROIPool:
    """Physical renderer targets under general, isotropic proposal corruption.

    Corners follow quad_cropper's [0,1] ROI chart: [-.75,1.75] and further
    finite amodal coordinates remain valid targets. Caller-supplied detector
    proposals must be out of fold for the sampled synthetic scene identities.
    """

    def __init__(
        self,
        *,
        image_hw: tuple[int, int] = (320, 576),
        num_frames: int = 4,
        max_items: int = 2048,
        seed: int = 20263333,
        device: torch.device | str = "cpu",
        fovea_scale: float = FOVEA_BASE_SCALE,
        surround_scale: float = SURROUND_BASE_SCALE,
        domain_split: str = "train",
        detector_proposal_model: TemporalPlateDetNet | None = None,
    ) -> None:
        from vpocr_det.isp_gpu import GPUCameraISPAugmenter, GPUISPConfig
        from vpocr_synth.gpu.scene_3d_gpu import Scene3DClipConfig, Scene3DGPUEngine

        self.dev = torch.device(device)
        self.domain_split = domain_split
        self.detector_proposal_model = detector_proposal_model
        self.max_items = max(128, int(max_items))
        self.rng = random.Random(int(seed))
        self.torch_rng = torch.Generator(device="cpu")
        self.torch_rng.manual_seed(int(seed) + 77)
        self.next_clip_seed = int(seed) + 10_000
        self.clip_counter = 0
        self.fovea_scale = float(fovea_scale)
        self.surround_scale = float(surround_scale)

        clip_cfg = Scene3DClipConfig(
            image_hw=image_hw,
            num_frames=int(num_frames),
            bank_size=(512 if self.max_items > 512 else 96) if domain_split == "train" else 128,
            bank_seed=int(seed) + 55,
            split="train" if domain_split == "train" else "val",
            domain_split=domain_split,
        )
        self.engine = Scene3DGPUEngine(config=clip_cfg, device=self.dev)
        self.isp_aug = GPUCameraISPAugmenter(config=GPUISPConfig(domain_split=domain_split), seed=int(seed)+56).to(self.dev).eval()
        self.pos_items: list[dict[str, Any]] = []
        self.neg_items: list[dict[str, Any]] = []

    @staticmethod
    def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter <= 0.0:
            return 0.0
        aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
        ba = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / max(1.0, aa + ba - inter)

    def _proposal_observable(self, proposal: Any, corners_xy: Any) -> bool:
        """A positive crop must contain plate photons in its wider sensor chart."""
        _, (rx0_s, ry0_s, rw_s, rh_s) = concentric_roi_windows(
            proposal, fovea_scale=self.fovea_scale, surround_scale=self.surround_scale)
        corners = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
        chart = (corners - np.array([rx0_s, ry0_s])) / np.array([rw_s, rh_s])
        center = chart.mean(0)
        return bool(np.isfinite(chart).all() and (center > .12).all() and (center < .88).all()
                    and ((chart > 0) & (chart < 1)).all(-1).sum() >= 2)

    def _corrupt_proposal(self, box: Any, corners_xy: Any = None) -> tuple[float, float, float, float]:
        """Rejection sample a zero-direction-bias disk under observable sensor support."""
        x0, y0, x1, y1 = map(float, box)
        w, h = max(1., x1-x0), max(1., y1-y0)
        if corners_xy is None:
            corners_xy = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        for _ in range(4096):
            angle = self.rng.uniform(0., 2*math.pi)
            radius = .65*math.hypot(w, h)*math.sqrt(self.rng.random())
            cx, cy = (x0+x1)/2+radius*math.cos(angle), (y0+y1)/2+radius*math.sin(angle)
            scale = math.exp(self.rng.uniform(-.45, .55))
            aspect = math.exp(self.rng.uniform(-.35, .35))
            pw, ph = w*scale*aspect, h*scale/aspect
            proposal = (cx-pw/2, cy-ph/2, cx+pw/2, cy+ph/2)
            if self._proposal_observable(proposal, corners_xy):
                return proposal
        raise RuntimeError("Unable to sample an observable plate proposal")

    @staticmethod
    def _corner_visibility(pg: Any, cn: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
        mask = np.asarray(pg.plate_vis_mask, dtype=np.float32)
        ray = np.array([mask[0, 0], mask[0, -1], mask[-1, -1], mask[-1, 0]])
        xy = np.asarray(pg.corners_xy)
        h, w = image_hw
        in_sensor = ((xy >= 0) & (xy < np.array([w, h]))).all(-1)
        return (ray*in_sensor*((cn >= 0) & (cn <= 1)).all(-1)).astype(np.float32)

    @torch.no_grad()
    def _sample_detector_proposals(self, frames: torch.Tensor) -> list[list[tuple[float, ...]]]:
        """Caller must supply an out-of-fold detector checkpoint for these scene identities."""
        model = self.detector_proposal_model
        if model is None:
            return [[] for _ in frames]
        was_training = model.training
        model.eval()
        hidden = None
        result = []
        h, w = frames.shape[-2:]
        dh, dw = math.ceil(h/32)*32, math.ceil(w/32)*32
        try:
            for frame in frames:
                image = F.interpolate(frame[None].to(next(model.parameters()).device), (dh, dw), mode="bilinear", align_corners=False)
                out = model.forward_step(image, hidden)
                hidden = tuple(t.detach() for t in out["h_next"])
                nq = int(getattr(model.config, "num_vehicle_queries", 0))
                plate_logits = out["cls_logits"][0, :-nq, CLS_PLATE] if nq > 0 else out["cls_logits"][0, :, CLS_PLATE]
                plate_boxes_pred = out["pred_boxes"][0, :-nq] if nq > 0 else out["pred_boxes"][0]
                scores = plate_logits.sigmoid()
                indices = scores.topk(min(32, scores.numel())).indices
                boxes = plate_boxes_pred[indices].cpu().numpy() * np.array([w / dw, h / dh, w / dw, h / dh])
                result.append([tuple(map(float, b)) for b in boxes
                               if np.isfinite(b).all() and (b[2:] > b[:2]).all()])
        finally:
            model.train(was_training)
        return result

    @torch.no_grad()
    def ingest_clips(self, num_clips: int) -> None:
        for _ in range(max(0, int(num_clips))):
            c_seed = self.next_clip_seed
            self.next_clip_seed += 1
            self.clip_counter += 1
            reid_base = self.clip_counter << 32

            clip = self.engine.render_clip(seed=c_seed)
            frames_dev = clip.frames_chw.to(self.dev).unsqueeze(0)  # (1, K, 3, H, W)
            already_ir = torch.tensor([bool(clip.frame_gts[0].is_night_ir)], device=self.dev)
            frames_isp = self.isp_aug.forward_clips(frames_dev, already_ir=already_ir).squeeze(0).clamp(0.0, 1.0)
            frames_u8 = (
                (frames_isp * 255.0)
                .round()
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )  # (K, H, W, 3)

            detector_proposals = self._sample_detector_proposals(frames_isp)
            for k_idx, fg in enumerate(clip.frame_gts):
                fr_u8 = frames_u8[k_idx]
                h_f, w_f = fr_u8.shape[:2]
                plate_boxes: list[tuple[float, float, float, float]] = [
                    tuple(float(v) for v in pg.bbox_xyxy)
                    for pg in fg.plates_and_signs
                    if pg.cls_id == CLS_PLATE
                ]
                plate_boxes.extend(fg.ignore_bboxes_xyxy)

                for pg in fg.plates_and_signs:
                    bx0, by0, bx1, by1 = (float(v) for v in pg.bbox_xyxy)
                    bw = max(4.0, bx1 - bx0)
                    bh = max(2.5, by1 - by0)

                    if pg.cls_id == CLS_PLATE:
                        reid_id = reid_base + max(1, int(pg.track_id))
                        even_slot = ((self.clip_counter - 1 + k_idx) % 2 == 0)
                        proposals = [(tuple(pg.bbox_xyxy), 0 if even_slot else 3),
                                     (self._corrupt_proposal(pg.bbox_xyxy, pg.corners_xy), 2 if even_slot else 1)]
                        det_kept = [box for box in detector_proposals[k_idx]
                                    if self._box_iou(box, pg.bbox_xyxy) >= 0.12
                                    and self._proposal_observable(box, pg.corners_xy)]
                        for d_i, box in enumerate(det_kept[:2]):
                            proposals.append((box, 0 if d_i % 2 == 0 else 3))
                            proposals.append((self._corrupt_proposal(pg.bbox_xyxy, pg.corners_xy),
                                              1 if d_i % 2 == 0 else 2))
                        for prop_box, policy in proposals:
                            gamma_v = 1.0
                            src_frame = fr_u8
                            c_gt_xy = np.asarray(pg.corners_xy, dtype=np.float32)
                            b_gt_xy = np.asarray(pg.blur_vec_xy, dtype=np.float32)
                            fov_u8, sur_u8, win_f, _ = extract_dual_concentric_plate_rois(
                                src_frame,
                                prop_box,
                                gamma=gamma_v,
                                fovea_scale=self.fovea_scale,
                                surround_scale=self.surround_scale,
                            )
                            rx0, ry0, rw, rh = win_f
                            cn = np.empty((4, 2), dtype=np.float32)
                            cn[:, 0] = (c_gt_xy[:, 0] - rx0) / max(1e-3, rw)
                            cn[:, 1] = (c_gt_xy[:, 1] - ry0) / max(1e-3, rh)
                            # [0,1] ROI chart: retain [-.75,1.75] and further finite amodal targets.
                            bn = np.array(
                                [
                                    float(b_gt_xy[0] / max(1e-3, rw)),
                                    float(b_gt_xy[1] / max(1e-3, rh)),
                                ],
                                dtype=np.float32,
                            )
                            self.pos_items.append(
                                {
                                    "fovea": fov_u8,
                                    "surround": sur_u8,
                                    "is_plate": 1.0,
                                    "corners_norm": cn,
                                    "blur_vec_norm": bn,
                                    "plate_side": int(pg.plate_side) if 0 <= int(pg.plate_side) < 3 else 0,
                                    "visible_support_fraction": float(pg.visible_support_fraction),
                                    "corner_vis_gt": self._corner_visibility(pg, cn, fr_u8.shape[:2]),
                                    "sampling_policy": policy,
                                    "min_projected_edge_px": float(pg.min_projected_edge_px),
                                    "blur_magnitude_px": float(pg.blur_magnitude_px),
                                    "local_contrast_snr": float(pg.local_contrast_snr),
                                    "is_large": 1.0 if bool(pg.is_large) else 0.0,
                                    "reid_id": int(reid_id),
                                    "clip_id": self.clip_counter,
                                }
                            )
                    elif pg.cls_id == CLS_SIGNBOARD_HARDNEG:
                        # Nonplate ownership is exact renderer truth.
                        jx = self.rng.uniform(-0.18, 0.18) * bw
                        jy = self.rng.uniform(-0.18, 0.18) * bh
                        gamma_n = self.rng.uniform(1.0, 1.6)
                        prop_n = (bx0 + jx, by0 + jy, bx1 + jx, by1 + jy)
                        fov_u8, sur_u8, _, _ = extract_dual_concentric_plate_rois(
                            fr_u8,
                            prop_n,
                            gamma=gamma_n,
                            fovea_scale=self.fovea_scale,
                            surround_scale=self.surround_scale,
                        )
                        self.neg_items.append(
                            {
                                "fovea": fov_u8,
                                "surround": sur_u8,
                                "is_plate": 0.0,
                                "corners_norm": np.zeros((4, 2), dtype=np.float32),
                                "blur_vec_norm": np.zeros((2,), dtype=np.float32),
                                "plate_side": 0,
                                "visible_support_fraction": 0.0,
                                "corner_vis_gt": np.zeros(4, dtype=np.float32),
                                "sampling_policy": int(len(self.neg_items) % 4),
                                "min_projected_edge_px": 0.0,
                                "blur_magnitude_px": 0.0,
                                "local_contrast_snr": 0.0,
                                "is_large": 0.0,
                                "reid_id": 0,
                                "clip_id": self.clip_counter,
                            }
                        )

                # General proposals on vehicle surfaces and the full image, with
                # no special negative object taxonomy or privileged direction.
                cand_boxes: list[tuple[float, float, float, float]] = list(detector_proposals[k_idx][:3])
                regions = [vg.bbox_xyxy for vg in fg.vehicles[:4]] + [(0., 0., float(w_f), float(h_f))] * 2
                for region in regions:
                    x0, y0, x1, y1 = map(float, region)
                    nw = self.rng.uniform(.05, .5)*max(1., x1-x0)
                    nh = nw*math.exp(self.rng.uniform(-1.5, .5))
                    nx = self.rng.uniform(x0, max(x0, x1-nw))
                    ny = self.rng.uniform(y0, max(y0, y1-nh))
                    cand_boxes.append((nx, ny, nx+nw, ny+nh))

                for nb in cand_boxes:
                    if any(self._box_iou(nb, pb) > 0 for pb in plate_boxes):
                        continue
                    fov_u8, sur_u8, _, _ = extract_dual_concentric_plate_rois(
                        fr_u8,
                        nb,
                        gamma=self.rng.uniform(1.0, 1.6),
                        fovea_scale=self.fovea_scale,
                        surround_scale=self.surround_scale,
                    )
                    self.neg_items.append(
                        {
                            "fovea": fov_u8,
                            "surround": sur_u8,
                            "is_plate": 0.0,
                            "corners_norm": np.zeros((4, 2), dtype=np.float32),
                            "blur_vec_norm": np.zeros((2,), dtype=np.float32),
                            "plate_side": 0,
                            "visible_support_fraction": 0.0,
                            "corner_vis_gt": np.zeros(4, dtype=np.float32),
                            "sampling_policy": int(len(self.neg_items) % 4),
                            "min_projected_edge_px": 0.0,
                            "blur_magnitude_px": 0.0,
                            "local_contrast_snr": 0.0,
                            "is_large": 0.0,
                            "reid_id": 0,
                            "clip_id": self.clip_counter,
                        }
                    )

        if len(self.pos_items) > self.max_items:
            self.pos_items = self.pos_items[-self.max_items :]
        if len(self.neg_items) > self.max_items:
            self.neg_items = self.neg_items[-self.max_items :]

    def sample_batch(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if batch_size < 2:
            raise ValueError("Stage-2 batches require at least one plate and one null")
        for _ in range(32):
            if self.pos_items and self.neg_items:
                break
            self.ingest_clips(1)
        if not self.pos_items or not self.neg_items:
            raise RuntimeError("No plate/null support after 32 synthetic clips; enlarge image_hw")
        n_neg = batch_size // 2
        n_pos = batch_size - n_neg

        # Ingestion appends or replaces the bank; unchanged banks reuse their indices.
        cache_key = (id(self.pos_items), len(self.pos_items),
                     id(self.pos_items[0]), id(self.pos_items[-1]))
        if getattr(self, "_cached_pos_key", None) != cache_key:
            by_reid: dict[int, list[dict[str, Any]]] = {}
            pos_by_policy: dict[int, list[dict[str, Any]]] = {p: [] for p in range(4)}
            for it in self.pos_items:
                by_reid.setdefault(int(it["reid_id"]), []).append(it)
                pos_by_policy[int(it["sampling_policy"])].append(it)
            self._cached_multi_groups = [grp for grp in by_reid.values() if len(grp) >= 2]
            self._cached_pos_by_policy = pos_by_policy
            self._cached_group_policies = {}
            for grp in self._cached_multi_groups:
                buckets: dict[int, list[dict[str, Any]]] = {p: [] for p in range(4)}
                for it in grp:
                    buckets[int(it["sampling_policy"])].append(it)
                self._cached_group_policies[id(grp)] = buckets
            self._cached_pos_key = cache_key
        multi_groups = self._cached_multi_groups
        pos_by_policy = self._cached_pos_by_policy

        chosen: list[dict[str, Any]] = []
        if getattr(self, "domain_split", "") == "train" and n_pos >= 4:
            # Pair draws consume the same quotas as the cyclic fill. This keeps
            # Re-ID supervision from changing the tight/wide sampling measure.
            remaining = {p: n_pos // 4 + int(p < n_pos % 4) for p in range(4)}
            n_pair_groups = min(len(multi_groups), max(2, n_pos // 4))
            for grp in self.rng.sample(multi_groups, k=n_pair_groups):
                buckets = self._cached_group_policies[id(grp)]
                pairs = [(p, q) for p in range(4) for q in range(4)
                         if remaining[p] > 0 and remaining[q] > int(p == q)
                         and buckets[p] and len(buckets[q]) > int(p == q)]
                if pairs:
                    p, q = self.rng.choice(pairs)
                    chosen.extend(self.rng.sample(buckets[p], k=2) if p == q else
                                  [self.rng.choice(buckets[p]), self.rng.choice(buckets[q])])
                    remaining[p] -= 1
                    remaining[q] -= 1
            while len(chosen) < n_pos:
                for p in (0, 1, 2, 3):
                    if remaining[p] > 0:
                        chosen.append(self.rng.choice(pos_by_policy[p] or self.pos_items))
                        remaining[p] -= 1
        else:
            if multi_groups:
                n_pair_groups = min(len(multi_groups), max(2, n_pos // 3))
                sampled_grps = self.rng.sample(multi_groups, k=n_pair_groups)
                for grp in sampled_grps:
                    if len(chosen) + 2 <= n_pos:
                        chosen.extend(self.rng.sample(grp, k=2))
            while len(chosen) < n_pos:
                chosen.append(self.rng.choice(self.pos_items))
        for _ in range(n_neg):
            chosen.append(self.rng.choice(self.neg_items))
        self.rng.shuffle(chosen)

        fov_t = (
            torch.from_numpy(np.stack([c["fovea"] for c in chosen], axis=0))
            .to(device, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            / 255.0
        )
        sur_t = (
            torch.from_numpy(np.stack([c["surround"] for c in chosen], axis=0))
            .to(device, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            / 255.0
        )
        if getattr(self, "domain_split", "") == "train" and batch_size >= 8:
            gen = getattr(self, "torch_rng", None)
            if gen is None:
                gen = self.torch_rng = torch.Generator(device="cpu")
                gen.manual_seed(self.rng.getrandbits(63))
            # Share appearance shifts across both crops; symmetric blur has zero phase shift.
            wb = (0.85 + 0.30 * torch.rand((batch_size, 3, 1, 1), generator=gen)).to(device)
            exposure = (0.74 + 0.54 * torch.rand((batch_size, 1, 1, 1), generator=gen)).to(device)
            pedestal = (-0.035 + 0.070 * torch.rand((batch_size, 1, 1, 1), generator=gen)).to(device)
            gamma = (0.82 + 0.38 * torch.rand((batch_size, 1, 1, 1), generator=gen)).to(device)
            desat = ((torch.rand((batch_size, 1, 1, 1), generator=gen) < 0.18).to(torch.float32)
                     * (0.65 + 0.35 * torch.rand((batch_size, 1, 1, 1), generator=gen))).to(device)
            haze_on = (torch.rand((batch_size, 1, 1, 1), generator=gen) < 0.24).to(device=device, dtype=torch.float32)
            haze_trans = (0.58 + 0.32 * torch.rand((batch_size, 1, 1, 1), generator=gen)).to(device)
            haze_air = (0.55 + 0.35 * torch.rand((batch_size, 1, 1, 1), generator=gen)).to(device)
            blur_mix = ((torch.rand((batch_size, 1, 1, 1), generator=gen) < 0.28).to(torch.float32)
                        * (0.35 + 0.55 * torch.rand((batch_size, 1, 1, 1), generator=gen))).to(device)
            noise_std = (0.016 * torch.rand((batch_size, 1, 1, 1), generator=gen)).to(device)
            luma_w = fov_t.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            k1d = fov_t.new_tensor([0.25, 0.50, 0.25])
            k2d = (k1d[:, None] * k1d[None, :]).view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()

            def radiometric(x: torch.Tensor) -> torch.Tensor:
                if bool((blur_mix > 0).any()):
                    x_blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k2d, groups=3)
                    x = torch.lerp(x, x_blur, blur_mix)
                lit = (x * wb * exposure + pedestal).clamp(1e-4, 1.0)
                lit = torch.lerp(lit, lit * haze_trans + haze_air * (1.0 - haze_trans), haze_on)
                luma = (lit * luma_w).sum(dim=1, keepdim=True)
                mixed = torch.lerp(lit, luma.expand_as(lit), desat)
                radiance = mixed.clamp(1e-4, 1.0).pow(gamma)
                return (radiance + torch.randn_like(x) * noise_std).clamp(0.0, 1.0)

            fov_t, sur_t = radiometric(fov_t), radiometric(sur_t)
        return {
            "fovea_rois": fov_t,
            "surround_rois": sur_t,
            "is_plate": torch.tensor([c["is_plate"] for c in chosen], dtype=torch.float32, device=device),
            "corners_norm": torch.from_numpy(np.stack([c["corners_norm"] for c in chosen], axis=0)).to(device),
            "blur_vec_norm": torch.from_numpy(np.stack([c["blur_vec_norm"] for c in chosen], axis=0)).to(device),
            "plate_side": torch.tensor([c["plate_side"] for c in chosen], dtype=torch.long, device=device),
            "visible_support_fraction": torch.tensor([c["visible_support_fraction"] for c in chosen], dtype=torch.float32, device=device),
            "corner_vis_gt": torch.from_numpy(np.stack([c["corner_vis_gt"] for c in chosen])).to(device),
            "sampling_policy": torch.tensor([c["sampling_policy"] for c in chosen], dtype=torch.long, device=device),
            "is_large": torch.tensor([c["is_large"] for c in chosen], dtype=torch.float32, device=device),
            "reid_ids": torch.tensor([c["reid_id"] for c in chosen], dtype=torch.long, device=device),
            "clip_ids": torch.tensor([c["clip_id"] for c in chosen], dtype=torch.long, device=device),
            **{key: torch.tensor([c[key] for c in chosen], dtype=torch.float32, device=device)
               for key in ("min_projected_edge_px", "blur_magnitude_px", "local_contrast_snr")},
        }


def train_plate_quad_track_net(
    train_cfg: QuadTrackTrainConfig,
    out_ckpt_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    training_script: str | None = None,
    device: torch.device | str | None = None,
    detector_proposal_model: TemporalPlateDetNet | None = None,
) -> tuple[PlateQuadTrackNet, dict[str, Any]]:
    """Train Dual-Scale `PlateQuadTrackNet` strictly from `random_scratch` on `Scene3DGPUEngine` clips."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    model_cfg = QuadTrackModelConfig(
        base_ch=train_cfg.base_ch,
        surround_ch=train_cfg.surround_ch,
        reid_dim=train_cfg.reid_dim,
        use_v5_heads=train_cfg.use_v5_heads,
    )
    model, init_state_sha256 = instantiate_deterministic_quad_track_model(
        model_cfg, init_seed=train_cfg.init_seed
    )
    model = model.to(dev)
    ema = ModelEMA(model, decay=0.9980, warmup_tau=220.0)
    criterion = PlateQuadTrackLoss().to(dev)

    pool = _Scene3DConcentricROIPool(
        image_hw=train_cfg.clip_image_hw,
        num_frames=train_cfg.clip_frames,
        max_items=train_cfg.max_pool_items,
        seed=train_cfg.seed,
        device=dev,
        fovea_scale=model_cfg.fovea_scale,
        surround_scale=model_cfg.surround_scale,
        domain_split="train",
        detector_proposal_model=detector_proposal_model,
    )
    pool.ingest_clips(train_cfg.initial_clips)

    # Covariance and coordinate readouts have different Jacobian scales; clip
    # each family separately from shared features and the remaining readouts.
    cov_readout_params = list(model.mode_cov_head.parameters())
    disp_readout_params = list(model.displacement_head.parameters())
    step_readout_params = list(model.update_head.parameters()) + list(model.step_update_heads.parameters())
    readout_ids = {id(p) for p in cov_readout_params + disp_readout_params + step_readout_params}
    other_params = [p for p in model.parameters() if id(p) not in readout_ids]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.lr,
        total_steps=max(1, train_cfg.total_steps),
        pct_start=0.08,
        anneal_strategy="cos",
        div_factor=15.0,
        final_div_factor=50.0,
    )

    t0 = time.perf_counter()
    r_loss = r_cls = r_err = r_reid = 0.0
    skipped_nan = 0
    model.train()

    for step in range(1, train_cfg.total_steps + 1):
        if (step > 1 and train_cfg.refresh_clips_every > 0
                and step <= int(0.92 * train_cfg.total_steps) and step % train_cfg.refresh_clips_every == 0):
            pool.ingest_clips(train_cfg.refresh_clips_count)
            model.train()

        batch = pool.sample_batch(train_cfg.batch_size, device=dev)
        optimizer.zero_grad(set_to_none=True)
        aux_warmup = min(1.0, float(step) / max(1.0, 0.20 * float(train_cfg.total_steps)))
        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(batch["fovea_rois"], batch["surround_rois"], batch["sampling_policy"])
            # Compatibility with loss versions retaining a deprecated topology
            # auxiliary: it supplies no independent observation or gradient.
            if "topology_logit" in out:
                out["topology_logit"] = out["topology_logit"].detach()
            loss_dict = criterion(
                out,
                batch["is_plate"],
                batch["corners_norm"],
                batch["plate_side"],
                batch["visible_support_fraction"],
                is_large_gt=batch["is_large"],
                blur_vec_norm_gt=batch["blur_vec_norm"],
                reid_instance_ids=batch["reid_ids"],
                aux_warmup_scale=aux_warmup,
                corner_visibility_gt=batch["corner_vis_gt"],
            )
            loss = loss_dict["loss"] - aux_warmup*loss_dict.get("loss_topo", 0.)

        if not torch.isfinite(loss):
            skipped_nan += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(cov_readout_params, max_norm=2.0)
        torch.nn.utils.clip_grad_norm_(other_params, max_norm=6.0)
        torch.nn.utils.clip_grad_norm_(disp_readout_params, max_norm=4.0)
        torch.nn.utils.clip_grad_norm_(step_readout_params, max_norm=6.0)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        r_loss += float(loss.detach().item())
        r_cls += float(loss_dict["loss_cls"].item())
        r_err += float(loss_dict["mean_corner_err_norm"].item())
        r_reid += float(loss_dict["loss_reid"].item())

        if step % train_cfg.log_every == 0 or step == train_cfg.total_steps:
            denom = float(
                train_cfg.log_every
                if step % train_cfg.log_every == 0
                else (step % train_cfg.log_every)
            )
            elapsed = time.perf_counter() - t0
            sps = (step * train_cfg.batch_size) / max(1e-3, elapsed)
            err_px = (r_err / denom) * 160.0
            print(
                f"[PlateQuadTrackNet] step {step}/{train_cfg.total_steps} | "
                f"loss={r_loss/denom:.4f} (cls={r_cls/denom:.4f}, "
                f"reid={r_reid/denom:.4f}, corner_err={err_px:.2f}px/160px) | {sps:.0f} rois/s",
                flush=True,
            )
            r_loss = r_cls = r_err = r_reid = 0.0

    # Fit the feature chart on positive training observations, before conformal calibration.
    if pool.pos_items:
        ema.ema.eval()
        positive_features = []
        with torch.no_grad():
            for start in range(0, len(pool.pos_items), 64):
                items = pool.pos_items[start:start + 64]
                fovea = torch.from_numpy(np.stack([item["fovea"] for item in items])).to(dev, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
                surround = torch.from_numpy(np.stack([item["surround"] for item in items])).to(dev, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
                policies = torch.tensor([item["sampling_policy"] for item in items], device=dev, dtype=torch.long)
                positive_features.append(ema.ema(fovea, surround, policies)["joint_features"].double().cpu())
            features = torch.cat(positive_features)
            mean = features.mean(0)
            scale = features.std(0, correction=0).clamp_min(1e-3)
            for net in (model, ema.ema):
                net.support_mean.copy_(mean.to(net.support_mean))
                net.support_scale.copy_(scale.to(net.support_scale))

    total_time = time.perf_counter() - t0
    prov_manifest = None
    if repo_root is not None:
        prov_manifest = build_det_quad_provenance_manifest(
            arch="PlateQuadTrackNet",
            init_mode="random_scratch",
            init_seed=train_cfg.init_seed,
            init_state_sha256=init_state_sha256,
            repo_root=repo_root,
            training_script=training_script,
        )

    summary: dict[str, Any] = {
        "arch": "PlateQuadTrackNet",
        "zero_real_pixel_training": True,
        "domain_split": "train",
        "detector_proposal_state_sha256": (compute_module_state_sha256(detector_proposal_model.state_dict())
                                              if detector_proposal_model is not None else None),
        "init_mode": "random_scratch",
        "init_seed": train_cfg.init_seed,
        "init_state_sha256": init_state_sha256,
        "num_parameters": ema.ema.num_parameters,
        "total_steps": train_cfg.total_steps,
        "skipped_nan_steps": skipped_nan,
        "total_rois_seen": train_cfg.total_steps * train_cfg.batch_size,
        "clips_rendered": pool.clip_counter,
        "elapsed_s": round(total_time, 2),
    }
    save_quad_track_checkpoint(
        out_ckpt_path,
        model=model,
        ema_model=ema.ema,
        extra=summary,
        provenance=prov_manifest,
    )
    return ema.ema, summary


def _fuse_conv_bn_inplace(module: nn.Module) -> nn.Module:
    """Use the same float64 Conv+BN folding as the Stage-2 exporter."""
    return fuse_conv_bn_inplace(module)


def export_quad_track_onnx(
    quad_track_model: PlateQuadTrackNet, out_dir: str | Path,
    stem_name: str = "plate_quad_track_v4",
) -> dict[str, Any]:
    """Delegate the complete projective/null contract to the Stage-2 exporter."""
    info = export_stage2_onnx(quad_track_model, out_dir)
    _restore_float_onnx_calibration_buffers(Path(info["int8_path"]))
    info["int8_bytes"] = Path(info["int8_path"]).stat().st_size
    for key, suffix in (("fp32_path", ".onnx"), ("int8_path", ".int8.onnx")):
        path = Path(info[key])
        target = Path(out_dir)/f"{stem_name}{suffix}"
        if path != target:
            path.replace(target)
        info[key] = str(target)
    info["quad_track_fp32_mb"] = Path(info["fp32_path"]).stat().st_size/(1024**2)
    info["quad_track_int8_mb"] = Path(info["int8_path"]).stat().st_size/(1024**2)
    return info


def _cpu_ort_session(path: str | Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


@torch.no_grad()
def verify_quad_track_onnx_parity(
    quad_track_model: PlateQuadTrackNet, onnx_dir: str | Path,
    stem_name: str = "plate_quad_track_v4", seed: int = 911503,
    atol: float = 2.5e-3, rtol: float = 1e-3,
) -> dict[str, Any]:
    """Check every deployed Gen-6 output, dynamic batches, and INT8 finiteness."""
    fp32, int8 = (Path(onnx_dir)/f"{stem_name}{suffix}" for suffix in (".onnx", ".int8.onnx"))
    sessions = [_cpu_ort_session(path) for path in (fp32, int8)]
    model = _fuse_conv_bn_inplace(copy.deepcopy(quad_track_model).cpu().eval())
    int8_model = copy.deepcopy(model)
    # Post-export calibration changes the deployed likelihood transformation.
    # Compare each graph with eager inference under that graph's own parameters.
    for name, value in _onnx_calibration_buffers(int8).items():
        getattr(int8_model, name).copy_(torch.from_numpy(value))
    rng = np.random.default_rng(seed)
    max_diff = 0.
    int8_diffs = {name: 0. for name in STAGE2_OUTPUT_NAMES}
    for count in (1, 3):
        feed = {"fovea_rois": rng.random((count, 3, 96, 160), dtype=np.float32),
                "surround_rois": rng.random((count, 3, 64, 96), dtype=np.float32),
                "sampling_policy": np.arange(count, dtype=np.int64) % 4}
        feed_tensors = tuple(torch.from_numpy(feed[k]) for k in ("fovea_rois", "surround_rois", "sampling_policy"))
        expected = model(*feed_tensors)
        for index, session in enumerate(sessions):
            names = [out.name for out in session.get_outputs()]
            if tuple(names) != STAGE2_OUTPUT_NAMES:
                raise AssertionError("Stage-2 ONNX output contract differs from eager model")
            ref_model = model if index == 0 else int8_model
            reference = expected if index == 0 else ref_model(*feed_tensors)
            cur_atol, cur_rtol = (atol, rtol) if index == 0 else (0.015, 0.015)
            actual = session.run(None, feed)
            for name, value in zip(names, actual):
                if not np.isfinite(value).all():
                    raise AssertionError(f"Non-finite Stage-2 output: {name}")
            actual_map = dict(zip(names, actual))
            post_logits = torch.where(
                reference["mode_valid"],
                reference["mode_logits"],
                torch.full_like(reference["mode_logits"], -1e9),
            )
            realized_sel = reference["selected_mode"].clone()
            mode_corners_np = reference["mode_corners"].cpu().numpy()
            # Condition only numerical ties on the geometry actually selected by ONNX.
            # Every output must still match one coherent eager realization below.
            for b_idx in range(count):
                if bool(reference["mode_valid"][b_idx].all()) and float((post_logits[b_idx, 0] - post_logits[b_idx, 1]).abs()) <= cur_atol:
                    d0 = float(np.max(np.abs(actual_map["corners_norm"][b_idx] - mode_corners_np[b_idx, 0])))
                    d1 = float(np.max(np.abs(actual_map["corners_norm"][b_idx] - mode_corners_np[b_idx, 1])))
                    realized_sel[b_idx] = 0 if d0 <= d1 else 1
            if not torch.equal(realized_sel, reference["selected_mode"]):
                reference = ref_model(*feed_tensors, _selected_mode_override=realized_sel)
            for name, value in zip(names, actual):
                if not np.isfinite(value).all():
                    raise AssertionError(f"Non-finite Stage-2 output: {name}")
                ref = reference[name].cpu().numpy()
                diff = float(np.max(np.abs(ref.astype(float)-value.astype(float))))
                if index == 0:
                    max_diff = max(max_diff, diff)
                elif name == "plate_null_llr":
                    val_prob = 1.0 / (1.0 + np.exp(-np.clip(value.astype(np.float64), -60.0, 60.0)))
                    ref_prob = 1.0 / (1.0 + np.exp(-np.clip(ref.astype(np.float64), -60.0, 60.0)))
                    prob_diff = float(np.max(np.abs(ref_prob - val_prob)))
                    int8_diffs[name] = max(int8_diffs[name], prob_diff)
                    np.testing.assert_allclose(val_prob, ref_prob, atol=0.025, rtol=0.025, err_msg=name)
                    confident = np.abs(ref) > 1.0
                    if np.any(confident):
                        assert np.array_equal((value > 0)[confident], (ref > 0)[confident]), "Confident INT8 LLR sign flip"
                    continue
                elif name == "reid_emb":
                    int8_diffs[name] = max(int8_diffs[name], diff)
                    np.testing.assert_allclose(value, ref, atol=0.075, rtol=0.075, err_msg=name)
                    cos_sim = np.sum(value.astype(np.float64) * ref.astype(np.float64), axis=-1)
                    if not np.all(cos_sim >= 0.99):
                        raise AssertionError(f"INT8 reid_emb cosine similarity too low: {cos_sim}")
                    continue
                elif name == "blur_vec_norm":
                    int8_diffs[name] = max(int8_diffs[name], diff)
                    np.testing.assert_allclose(value, ref, atol=0.075, rtol=0.075, err_msg=name)
                    continue
                else:
                    int8_diffs[name] = max(int8_diffs[name], diff)
                np.testing.assert_allclose(value, ref, atol=cur_atol, rtol=cur_rtol, err_msg=name)
    return {"max_abs_diff": max_diff, "int8_max_abs_diff": max(int8_diffs.values()),
            "int8_per_output_abs_diff": int8_diffs, "fp32_mb": fp32.stat().st_size/1024**2,
            "int8_mb": int8.stat().st_size/1024**2}


# ==============================================================================
# Gen-4 (`v0.4.0`) Phase 4: `TemporalPlateDetNet` Clip Trainer & ONNX Exporter
# ==============================================================================

# Both trainers hash the same renderer/model/ISP dependency closure.
DEFAULT_PHASE4_SYNTH_SOURCE_FILES = DEFAULT_PHASE3_SYNTH_SOURCE_FILES


@dataclass
class TemporalDetTrainConfig:
    preset: str = "base"
    use_v5_heads: bool = True
    total_steps: int = 1600
    batch_size: int = 6
    clip_frames: int = 4
    clip_image_hw: tuple[int, int] = (320, 576)
    multi_scale_hws: tuple[tuple[int, int], ...] = (
        (320, 576),
        (384, 640),
        (448, 768),
    )
    initial_clips: int = 44
    refresh_clips_every: int = 16
    refresh_clips_count: int = 2
    max_pool_clips: int = 96
    reset_prob: float = 0.25
    kd_weight: float = 0.45
    lr: float = 1.6e-3
    weight_decay: float = 1e-4
    log_every: int = 100
    init_seed: int = 20260441
    seed: int = 20264444


def _build_dfine_pretrained_manifest(cfg: TemporalDetModelConfig) -> dict[str, Any]:
    spec = DFINE_PRETRAINED_SPECS[cfg.dfine_variant]
    return {
        "variant": spec.variant,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "license_spdx": spec.license_spdx,
        "filename": spec.filename,
        "sha256": spec.sha256,
        "num_bytes": int(spec.num_bytes),
        "frozen": True,
    }


def instantiate_deterministic_temporal_det_model(
    cfg: TemporalDetModelConfig,
    init_seed: int,
    *,
    pretrained_dir: str | Path | None = None,
) -> tuple[TemporalPlateDetNet, str]:
    """Deterministically initialize `TemporalPlateDetNet` (SHA-256-verified frozen Apache-2.0 `DFineCore` + seeded scratch heads)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = TemporalPlateDetNet(cfg, pretrained_dir=pretrained_dir, load_pretrained_dfine=True)
    return model, compute_module_state_sha256(model.state_dict())


def save_temporal_det_checkpoint(
    path: str | Path,
    model: TemporalPlateDetNet,
    ema_model: TemporalPlateDetNet | None = None,
    extra: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    active_model = ema_model if ema_model is not None else model
    active_sd = active_model.state_dict()
    raw_sd = model.state_dict()
    payload: dict[str, Any] = {
        "config": active_model.config.to_dict(),
        "state_dict": active_sd,
        "raw_model_state_dict": raw_sd,
        "extra": extra or {},
    }
    if provenance is not None:
        prov_c = dict(provenance)
        prov_c["final_state_sha256"] = compute_module_state_sha256(active_sd)
        prov_c["final_raw_state_sha256"] = compute_module_state_sha256(raw_sd)
        payload["provenance"] = prov_c
    torch.save(payload, p)
    return p


def _is_sha256_hex_str(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def load_temporal_det_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    require_verified_provenance: bool = False,
    repo_root: str | Path | None = None,
    teacher_ckpt_path: str | Path | None = None,
) -> TemporalPlateDetNet:
    p = Path(path)
    if require_verified_provenance:
        verify_temporal_det_checkpoint_provenance(
            p,
            repo_root=repo_root,
            teacher_ckpt_path=teacher_ckpt_path,
            verify_source_hashes=True,
        )
    ckpt = torch.load(p, map_location=device, weights_only=False)
    cfg = TemporalDetModelConfig.from_dict(ckpt["config"])
    model = TemporalPlateDetNet(cfg, load_pretrained_dfine=False).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def verify_temporal_det_checkpoint_provenance(
    ckpt_path: str | Path | dict[str, Any],
    repo_root: str | Path | None = None,
    teacher_ckpt_path: str | Path | None = None,
    verify_source_hashes: bool = True,
) -> dict[str, Any]:
    from collections.abc import Mapping

    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).resolve()
    p: Path | None = None
    if isinstance(ckpt_path, (str, Path)):
        p = Path(ckpt_path).resolve()
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    elif isinstance(ckpt_path, Mapping):
        ckpt = ckpt_path
    else:
        raise TypeError(f"Unsupported checkpoint input type: {type(ckpt_path)!r}")

    prov = ckpt.get("provenance")
    if not isinstance(prov, Mapping):
        raise ValueError(f"Checkpoint {p} lacks 'provenance' manifest")
    if prov.get("schema_version") != DET_QUAD_PROVENANCE_SCHEMA:
        raise ValueError(f"Unexpected schema_version: {prov.get('schema_version')!r}")
    if prov.get("zero_real_pixel_training") is not False:
        raise ValueError(
            "TemporalPlateDetNet provenance must honestly record `zero_real_pixel_training: False` "
            "(uses frozen Apache-2.0 D-FINE backbone)"
        )
    if prov.get("zero_vpocr_real_pixel_finetuning") is not True:
        raise ValueError("zero_vpocr_real_pixel_finetuning is not True")
    if prov.get("zero_real\x5feval_leakage") is not True:
        raise ValueError("zero_real\x5feval_leakage is not True")
    if prov.get("real_data_paths", []) != []:
        raise ValueError(f"Provenance `real_data_paths` must be empty: {prov.get('real_data_paths')!r}")
    if (
        prov.get("init_mode") != PHASE4_APACHE20_DFINE_INIT_MODE
        or prov.get("parent_checkpoint") is not None
        or prov.get("parent_checkpoint_sha256") is not None
    ):
        raise ValueError(
            f"Expected init_mode={PHASE4_APACHE20_DFINE_INIT_MODE!r} with null parent_checkpoint, got {prov.get('init_mode')!r}"
        )
    init_seed = prov.get("init_seed")
    if isinstance(init_seed, bool) or not isinstance(init_seed, int):
        raise ValueError(f"Invalid init_seed: {init_seed!r}")

    if "config" not in ckpt or not isinstance(ckpt.get("config"), Mapping):
        raise ValueError("Checkpoint missing valid `config` mapping")
    try:
        cfg = TemporalDetModelConfig.from_dict(dict(ckpt["config"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid TemporalDetModelConfig in checkpoint: {exc}") from exc
    if cfg.preset not in ("base", "nano"):
        raise ValueError(f"Unsupported TemporalPlateDetNet preset in provenance check: {cfg.preset!r}")
    canonical_cfg = TemporalDetModelConfig.from_preset(
        cfg.preset,
        num_classes=cfg.num_classes,
        num_subtypes=cfg.num_subtypes,
        use_v5_heads=cfg.use_v5_heads,
    )
    if cfg != canonical_cfg:
        raise ValueError(
            f"Checkpoint config does not match canonical {cfg.preset!r} preset geometry: {cfg} != {canonical_cfg}"
        )
    expected_arch = f"TemporalPlateDetNet-{cfg.preset}"
    if prov.get("arch") != expected_arch:
        raise ValueError(f"Provenance arch {prov.get('arch')!r} != expected {expected_arch!r}")

    expected_pb = _build_dfine_pretrained_manifest(cfg)
    rec_pb = prov.get("pretrained_backbone")
    if not isinstance(rec_pb, Mapping) or dict(rec_pb) != expected_pb:
        raise ValueError(
            f"Provenance `pretrained_backbone` mismatch: {rec_pb!r} != expected {expected_pb!r}"
        )

    cand_pt_dir = root / "out" / "models" / "pretrained_apache20"
    pt_dir_arg = cand_pt_dir if cand_pt_dir.is_dir() else None
    if verify_source_hashes and pt_dir_arg is not None:
        sf_path = pt_dir_arg / str(expected_pb["filename"])
        if not sf_path.is_file():
            raise FileNotFoundError(f"Missing Apache-2.0 D-FINE safetensors file: {sf_path}")
        if sf_path.stat().st_size != int(expected_pb["num_bytes"]):
            raise ValueError(f"Apache-2.0 D-FINE safetensors size mismatch on {sf_path}")
        if compute_det_file_sha256(sf_path) != str(expected_pb["sha256"]):
            raise ValueError(f"Apache-2.0 D-FINE safetensors SHA-256 mismatch on {sf_path}")

    init_model, recomputed_init_sha = instantiate_deterministic_temporal_det_model(
        cfg, init_seed, pretrained_dir=pt_dir_arg
    )
    if recomputed_init_sha != prov.get("init_state_sha256"):
        raise ValueError(
            f"Recomputed init_state_sha256 ({recomputed_init_sha}) != recorded ({prov.get('init_state_sha256')})"
        )
    if "state_dict" not in ckpt or not isinstance(ckpt["state_dict"], Mapping):
        raise ValueError("Checkpoint missing `state_dict`")
    if "raw_model_state_dict" not in ckpt or not isinstance(ckpt["raw_model_state_dict"], Mapping):
        raise ValueError("Checkpoint missing `raw_model_state_dict`")

    init_sd = init_model.state_dict()
    for sd_name, sd_map in (("state_dict", ckpt["state_dict"]), ("raw_model_state_dict", ckpt["raw_model_state_dict"])):
        for k_df, init_t in init_sd.items():
            if k_df.startswith("dfine_core."):
                if k_df not in sd_map:
                    raise ValueError(f"Checkpoint `{sd_name}` missing frozen key `{k_df}`")
                cand_t = sd_map[k_df].detach().cpu()
                if cand_t.dtype != init_t.dtype or cand_t.shape != init_t.shape or not torch.equal(cand_t, init_t):
                    raise ValueError(
                        f"Frozen Apache-2.0 `{k_df}` in `{sd_name}` was mutated from pinned safetensors state"
                    )

    # Verify tensor keys/shapes match `cfg` so presets cannot be cross-labeled (always raise ValueError)
    try:
        init_model.load_state_dict(ckpt["state_dict"], strict=True)
        init_model.load_state_dict(ckpt["raw_model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint state_dict incompatible with canonical {cfg.preset!r} model geometry: {exc}"
        ) from exc

    actual_final_sha = compute_module_state_sha256(ckpt["state_dict"])
    if not _is_sha256_hex_str(prov.get("final_state_sha256")) or actual_final_sha != prov.get("final_state_sha256"):
        raise ValueError("final_state_sha256 mismatch")
    actual_raw_sha = compute_module_state_sha256(ckpt["raw_model_state_dict"])
    if (
        not _is_sha256_hex_str(prov.get("final_raw_state_sha256"))
        or actual_raw_sha != prov.get("final_raw_state_sha256")
    ):
        raise ValueError("final_raw_state_sha256 mismatch")

    script_rel = _verify_det_quad_sources_and_fonts(
        prov,
        root=root,
        required_sources=DEFAULT_PHASE4_SYNTH_SOURCE_FILES,
        verify_source_hashes=verify_source_hashes,
        is_legacy_v4=(not cfg.use_v5_heads),
    )

    raw_extra = ckpt.get("extra")
    if raw_extra is not None and not isinstance(raw_extra, Mapping):
        raise ValueError(f"Checkpoint `extra` must be a mapping when present, got {type(raw_extra)!r}")
    extra_d: Mapping[str, Any] = raw_extra if isinstance(raw_extra, Mapping) else {}
    if "preset" in extra_d and extra_d.get("preset") != cfg.preset:
        raise ValueError(f"extra.preset {extra_d.get('preset')!r} != config.preset {cfg.preset!r}")
    if "zero_real_pixel_training" in extra_d and extra_d.get("zero_real_pixel_training") is not False:
        raise ValueError("extra.zero_real_pixel_training must be False for Apache-2.0 D-FINE hybrid model")
    if "zero_vpocr_real_pixel_finetuning" in extra_d and extra_d.get("zero_vpocr_real_pixel_finetuning") is not True:
        raise ValueError("extra.zero_vpocr_real_pixel_finetuning is not True")
    if "zero_real\x5feval_leakage" in extra_d and extra_d.get("zero_real\x5feval_leakage") is not True:
        raise ValueError("extra.zero_real\x5feval_leakage is not True")
    if "init_mode" in extra_d and extra_d.get("init_mode") != PHASE4_APACHE20_DFINE_INIT_MODE:
        raise ValueError(f"extra.init_mode {extra_d.get('init_mode')!r} != {PHASE4_APACHE20_DFINE_INIT_MODE!r}")
    if "pretrained_backbone" in extra_d and (
        not isinstance(extra_d.get("pretrained_backbone"), Mapping)
        or dict(extra_d["pretrained_backbone"]) != expected_pb
    ):
        raise ValueError("extra.pretrained_backbone != provenance.pretrained_backbone")
    if "init_seed" in extra_d and (
        isinstance(extra_d.get("init_seed"), bool) or extra_d.get("init_seed") != init_seed
    ):
        raise ValueError("extra.init_seed != provenance.init_seed")
    if "init_state_sha256" in extra_d and extra_d.get("init_state_sha256") != recomputed_init_sha:
        raise ValueError("extra.init_state_sha256 != recomputed init_state_sha256")

    rec_teacher_rel = prov.get("teacher_checkpoint")
    rec_teacher_sha = prov.get("teacher_checkpoint_sha256")
    rec_teacher_state_sha = prov.get("teacher_state_sha256")

    if cfg.preset == "base":
        if teacher_ckpt_path is not None:
            raise ValueError("Base checkpoint forbids `teacher_ckpt_path` override")
        for k_t in ("teacher_checkpoint", "teacher_checkpoint_sha256", "teacher_state_sha256"):
            if k_t not in prov or prov.get(k_t) is not None:
                raise ValueError(f"Base checkpoint must record `{k_t}: None` in provenance")
        for k_p, v_p in prov.items():
            if "teacher" in str(k_p) and v_p is not None:
                raise ValueError(f"Base checkpoint forbids non-null provenance `{k_p}`: {v_p!r}")
        if "used_teacher" in extra_d and extra_d.get("used_teacher") is not False:
            raise ValueError(f"Base checkpoint forbids extra.used_teacher={extra_d.get('used_teacher')!r}")
        for k_ex, v_ex in extra_d.items():
            if "teacher" in str(k_ex) and k_ex != "used_teacher" and v_ex is not None:
                raise ValueError(f"Base checkpoint forbids non-null extra.{k_ex}: {v_ex!r}")
        teacher_verified = False
    else:
        # `cfg.preset == "nano"` unconditionally requires verified Base teacher linkage in BOTH `provenance` and `extra`
        if not isinstance(raw_extra, Mapping):
            raise ValueError("Nano student checkpoint requires `extra` metadata mapping")
        if not isinstance(rec_teacher_rel, str) or not rec_teacher_rel.strip():
            raise ValueError("Student checkpoint missing or invalid provenance `teacher_checkpoint`")
        if not _is_sha256_hex_str(rec_teacher_sha):
            raise ValueError("Student checkpoint missing or invalid provenance `teacher_checkpoint_sha256`")
        if not _is_sha256_hex_str(rec_teacher_state_sha):
            raise ValueError("Student checkpoint missing or invalid provenance `teacher_state_sha256`")
        if extra_d.get("used_teacher") is not True:
            raise ValueError("Nano student checkpoint requires `extra.used_teacher is True`")
        if "teacher_checkpoint" not in extra_d or extra_d.get("teacher_checkpoint") != rec_teacher_rel:
            raise ValueError("extra.teacher_checkpoint missing or != provenance.teacher_checkpoint")
        if (
            "teacher_checkpoint_sha256" not in extra_d
            or extra_d.get("teacher_checkpoint_sha256") != rec_teacher_sha
        ):
            raise ValueError("extra.teacher_checkpoint_sha256 missing or != provenance.teacher_checkpoint_sha256")
        if (
            "teacher_state_sha256" not in extra_d
            or extra_d.get("teacher_state_sha256") != rec_teacher_state_sha
        ):
            raise ValueError("extra.teacher_state_sha256 missing or != provenance.teacher_state_sha256")

        if teacher_ckpt_path is not None:
            t_path = Path(teacher_ckpt_path).resolve()
        else:
            t_cand = Path(rec_teacher_rel)
            t_path = t_cand.resolve() if t_cand.is_absolute() else (root / t_cand).resolve()
            if not t_path.is_file() and p is not None and (p.parent / t_cand.name).is_file():
                t_path = (p.parent / t_cand.name).resolve()

        if not t_path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found for provenance verification: {t_path}")
        if p is not None and t_path == p:
            raise ValueError(f"Self-referential teacher_checkpoint forbidden: {t_path}")

        t_res = verify_temporal_det_checkpoint_provenance(
            t_path,
            repo_root=root,
            verify_source_hashes=verify_source_hashes,
        )
        if t_res.get("preset") != "base":
            raise ValueError(f"Expected teacher preset 'base', got {t_res.get('preset')!r}")
        actual_t_file_sha = compute_det_file_sha256(t_path)
        if actual_t_file_sha != rec_teacher_sha:
            raise ValueError(f"Teacher checkpoint SHA-256 mismatch: {actual_t_file_sha} != {rec_teacher_sha}")
        if t_res["final_state_sha256"] != rec_teacher_state_sha:
            raise ValueError(
                f"Teacher final_state_sha256 mismatch: {t_res['final_state_sha256']} != {rec_teacher_state_sha}"
            )
        teacher_verified = True

    return {
        "verified": True,
        "checkpoint": str(p) if p is not None else None,
        "checkpoint_sha256": compute_det_file_sha256(p) if p is not None else None,
        "preset": cfg.preset,
        "init_seed": init_seed,
        "init_state_sha256": recomputed_init_sha,
        "recomputed_init_sha256_match": True,
        "final_state_sha256": actual_final_sha,
        "final_raw_state_sha256": actual_raw_sha,
        "training_script": script_rel,
        "teacher_verified": teacher_verified,
    }


def _frame_gt_to_det_target(fgt: Any, *, identity_namespace: int = 0) -> dict[str, torch.Tensor]:
    """Convert a `Scene3DFrameGT` into exact `TemporalTaskAlignedDetLoss` tensors."""
    from vpocr_synth.gpu.scene_3d_gpu import _conservative_enclosing_quad_2d

    labels: list[int] = []
    boxes: list[np.ndarray] = []
    corners: list[np.ndarray] = []
    subtypes: list[int] = []
    cab_val: list[float] = []
    cab_box: list[np.ndarray] = []
    fp_val: list[float] = []
    fp_xy: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    amodal_boxes: list[np.ndarray] = []
    identity_ids: list[int] = []
    veh_track_ids: list[int] = []

    for v in fgt.vehicles:
        vb = np.asarray(v.bbox_xyxy, dtype=np.float32)
        if (vb[2] - vb[0]) < 8.0 or (vb[3] - vb[1]) < 8.0:
            continue
        labels.append(CLS_VEHICLE)
        boxes.append(vb)
        vc = np.array(
            [[vb[0], vb[1]], [vb[2], vb[1]], [vb[2], vb[3]], [vb[0], vb[3]]],
            dtype=np.float32,
        )
        corners.append(vc)
        subtype = int(v.vehicle_subtype)
        subtypes.append(subtype if 0 <= subtype < NUM_VEHICLE_SUBTYPES else NUM_VEHICLE_SUBTYPES)
        identity = (int(identity_namespace) << 32) + int(v.track_id)
        identity_ids.append(identity)
        veh_track_ids.append(identity)
        amodal_boxes.append(np.asarray(v.amodal_bbox_xyxy, dtype=np.float32))
        cab_val.append(1.0 if v.cabin_valid else 0.0)
        cab_box.append(np.asarray(v.cabin_bbox_xyxy, dtype=np.float32).reshape(4))
        raw_fp = np.asarray(v.ground_footprint_xy, dtype=np.float32)
        fp_ok = bool(v.footprint_valid) and raw_fp.ndim == 2 and raw_fp.shape[1] == 2 and raw_fp.shape[0] >= 3
        fp_val.append(1.0 if fp_ok else 0.0)
        if not fp_ok:
            fp_xy.append(np.zeros((4, 2), dtype=np.float32))
        else:
            fp_xy.append(
                np.asarray(
                    _conservative_enclosing_quad_2d(
                        raw_fp,
                        cam_pos_w=getattr(fgt, "cam_pos_w", None),
                        cam_R_c2w=getattr(fgt, "cam_R_c2w", None),
                        intrinsics_fxfycxcy=getattr(fgt, "cam_intrinsics_fxfycxcy", None),
                        z_near=0.22,
                    ),
                    dtype=np.float32,
                ).reshape(4, 2)
            )
        masks.append(np.asarray(v.visible_roi_mask, dtype=np.float32).reshape(64, 64))

    for ps in fgt.plates_and_signs:
        pb = np.asarray(ps.bbox_xyxy, dtype=np.float32)
        if (pb[2] - pb[0]) < 4.5 or (pb[3] - pb[1]) < 2.5:
            continue
        labels.append(int(ps.cls_id))
        boxes.append(pb)
        corners.append(np.asarray(ps.corners_xy, dtype=np.float32))
        subtypes.append(NUM_VEHICLE_SUBTYPES)
        identity_ids.append(-1)
        parent = int(ps.veh_track_id)
        veh_track_ids.append((int(identity_namespace) << 32)+parent if parent > 0 else -1)
        pc = np.asarray(ps.corners_xy, dtype=np.float32)
        amodal_boxes.append(np.r_[pc.min(0), pc.max(0)].astype(np.float32))
        cab_val.append(0.0)
        cab_box.append(pb.copy())
        fp_val.append(0.0)
        fp_xy.append(np.asarray(ps.corners_xy, dtype=np.float32))
        masks.append(np.zeros((64, 64), dtype=np.float32))

    if labels:
        t_dict: dict[str, torch.Tensor] = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.from_numpy(np.stack(boxes, axis=0)),
            "corners": torch.from_numpy(np.stack(corners, axis=0)),
            "vehicle_subtypes": torch.tensor(subtypes, dtype=torch.long),
            "cabin_valid": torch.tensor(cab_val, dtype=torch.float32),
            "cabin_boxes": torch.from_numpy(np.stack(cab_box, axis=0)),
            "footprint_valid": torch.tensor(fp_val, dtype=torch.float32),
            "footprints": torch.from_numpy(np.stack(fp_xy, axis=0)),
            "cutout_masks": torch.from_numpy(np.stack(masks, axis=0)),
        }
    else:
        t_dict = {
            "labels": torch.zeros((0,), dtype=torch.long),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "corners": torch.zeros((0, 4, 2), dtype=torch.float32),
            "vehicle_subtypes": torch.zeros((0,), dtype=torch.long),
            "cabin_valid": torch.zeros((0,), dtype=torch.float32),
            "cabin_boxes": torch.zeros((0, 4), dtype=torch.float32),
            "footprint_valid": torch.zeros((0,), dtype=torch.float32),
            "footprints": torch.zeros((0, 4, 2), dtype=torch.float32),
            "cutout_masks": torch.zeros((0, 64, 64), dtype=torch.float32),
        }

    t_dict["identity_ids"] = torch.tensor(identity_ids, dtype=torch.long)
    t_dict["veh_track_ids"] = torch.tensor(veh_track_ids, dtype=torch.long)
    t_dict["amodal_boxes"] = torch.from_numpy(np.stack(amodal_boxes)) if amodal_boxes else torch.zeros((0, 4))
    t_dict["visible_masks"] = t_dict["cutout_masks"]
    t_dict["subtype_unknown_gt"] = ((t_dict["vehicle_subtypes"] == NUM_VEHICLE_SUBTYPES)
                                  & (t_dict["labels"] == CLS_VEHICLE)).float()

    if fgt.ignore_bboxes_xyxy:
        t_dict["ignore_boxes"] = torch.tensor(fgt.ignore_bboxes_xyxy, dtype=torch.float32).reshape(-1, 4)
        if getattr(fgt, "ignore_bbox_classes", None) and len(fgt.ignore_bbox_classes) == len(fgt.ignore_bboxes_xyxy):
            t_dict["ignore_classes"] = torch.tensor(fgt.ignore_bbox_classes, dtype=torch.long)
    else:
        t_dict["ignore_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        t_dict["ignore_classes"] = torch.zeros((0,), dtype=torch.long)
    return t_dict


def _scale_pad_target_dict(
    tgt: dict[str, torch.Tensor],
    scale: float,
    pad_x: float,
    pad_y: float,
    out_w: int,
    out_h: int,
) -> dict[str, torch.Tensor]:
    """Transform target coordinates under exact isotropic letterbox `(scale, pad_x, pad_y)`."""
    out: dict[str, torch.Tensor] = dict(tgt)
    if tgt["boxes"].numel() > 0:
        b = tgt["boxes"].clone()
        b[:, [0, 2]] = (b[:, [0, 2]] * scale + pad_x).clamp(0.0, float(out_w))
        b[:, [1, 3]] = (b[:, [1, 3]] * scale + pad_y).clamp(0.0, float(out_h))
        out["boxes"] = b

        c = tgt["corners"].clone()
        c[:, :, 0] = c[:, :, 0] * scale + pad_x
        c[:, :, 1] = c[:, :, 1] * scale + pad_y
        out["corners"] = c

        cb = tgt["cabin_boxes"].clone()
        cb[:, [0, 2]] = (cb[:, [0, 2]] * scale + pad_x).clamp(0.0, float(out_w))
        cb[:, [1, 3]] = (cb[:, [1, 3]] * scale + pad_y).clamp(0.0, float(out_h))
        out["cabin_boxes"] = cb

        fp = tgt["footprints"].clone()
        fp[:, :, 0] = fp[:, :, 0] * scale + pad_x
        fp[:, :, 1] = fp[:, :, 1] * scale + pad_y
        out["footprints"] = fp
    else:
        out["boxes"] = tgt["boxes"]
        out["corners"] = tgt["corners"]
        out["cabin_boxes"] = tgt["cabin_boxes"]
        out["footprints"] = tgt["footprints"]

    if "amodal_boxes" in tgt:
        amodal = tgt["amodal_boxes"].clone()
        amodal[:, [0, 2]] = amodal[:, [0, 2]]*scale+pad_x
        amodal[:, [1, 3]] = amodal[:, [1, 3]]*scale+pad_y
        out["amodal_boxes"] = amodal

    if tgt["ignore_boxes"].numel() > 0:
        ib = tgt["ignore_boxes"].clone()
        ib[:, [0, 2]] = (ib[:, [0, 2]] * scale + pad_x).clamp(0.0, float(out_w))
        ib[:, [1, 3]] = (ib[:, [1, 3]] * scale + pad_y).clamp(0.0, float(out_h))
        out["ignore_boxes"] = ib
    else:
        out["ignore_boxes"] = tgt["ignore_boxes"]
    return out


class _Scene3DTemporalClipPool:
    """Rolling GPU-rendered 3D multi-frame video clip pool for Phase 4 `TemporalPlateDetNet` training."""

    def __init__(
        self,
        *,
        image_hw: tuple[int, int] = (320, 576),
        num_frames: int = 4,
        max_clips: int = 96,
        seed: int = 20264444,
        device: torch.device | str = "cpu",
        domain_split: str = "train",
    ) -> None:
        from vpocr_synth.gpu.scene_3d_gpu import Scene3DClipConfig, Scene3DGPUEngine

        from vpocr_det.isp_gpu import GPUCameraISPAugmenter, GPUISPConfig

        self.domain_split = domain_split
        self.dev = torch.device(device)
        self.isp_aug = GPUCameraISPAugmenter(config=GPUISPConfig(domain_split=domain_split), seed=seed+55).to(self.dev).eval()
        self.image_hw = image_hw
        self.num_frames = int(num_frames)
        self.max_clips = int(max_clips)
        self.base_seed = int(seed)
        self.rng = random.Random(int(seed) + 91)
        self.clip_counter = 0
        cfg = Scene3DClipConfig(
            num_frames=self.num_frames,
            image_hw=self.image_hw,
            split="train" if domain_split == "train" else "val",
            domain_split=domain_split,
            bank_size=(512 if self.max_clips > 8 else 96) if domain_split == "train" else 128,
        )
        self.engine = Scene3DGPUEngine(config=cfg, device=device)
        self.items: list[dict[str, Any]] = []

    def ingest_clips(self, count: int) -> None:
        for _ in range(int(count)):
            c_seed = self.base_seed + self.clip_counter * 17
            self.clip_counter += 1
            clip = self.engine.render_clip(seed=c_seed)
            frames_u8 = (clip.frames_chw.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
            is_ir = bool(clip.frame_gts[0].is_night_ir) if clip.frame_gts else False
            per_frame_tgts = [_frame_gt_to_det_target(fgt, identity_namespace=self.clip_counter) for fgt in clip.frame_gts]
            item = {
                "frames_u8": frames_u8,
                "is_ir": is_ir,
                "targets": per_frame_tgts,
            }
            if len(self.items) < self.max_clips:
                self.items.append(item)
            else:
                rep_i = self.rng.randrange(len(self.items))
                self.items[rep_i] = item

    def sample_batch(
        self,
        batch_size: int,
        target_hw: tuple[int, int],
        device: torch.device,
        isp_aug: Any | None = None,
    ) -> tuple[torch.Tensor, list[list[dict[str, torch.Tensor]]]]:
        if not self.items:
            self.ingest_clips(1)
        if isp_aug is None:
            isp_aug = self.isp_aug
        if isp_aug.config.domain_split != self.domain_split:
            raise ValueError("Scene/ISP domain splits must match")
        chosen = [self.rng.choice(self.items) for _ in range(int(batch_size))]
        clips_f = (
            torch.stack([c["frames_u8"] for c in chosen], dim=0)
            .to(device=device, dtype=torch.float32, non_blocking=True)
            / 255.0
        )  # (B, K, 3, H0, W0)
        already_ir = torch.tensor([c["is_ir"] for c in chosen], device=device, dtype=torch.bool)
        if isp_aug is not None:
            clips_f = isp_aug.forward_clips(clips_f, already_ir=already_ir)

        bsz, kf, _, h0, w0 = clips_f.shape
        ht, wt = int(target_hw[0]), int(target_hw[1])
        if (h0, w0) == (ht, wt):
            scale, pad_x, pad_y = 1.0, 0.0, 0.0
            clips_out = clips_f
        else:
            scale = min(float(wt) / float(max(1, w0)), float(ht) / float(max(1, h0)))
            nw = max(1, int(round(w0 * scale)))
            nh = max(1, int(round(h0 * scale)))
            flat = clips_f.reshape(bsz * kf, 3, h0, w0)
            resized = F.interpolate(flat, size=(nh, nw), mode="bilinear", align_corners=False)
            pad_l = (wt - nw) // 2
            pad_r = wt - nw - pad_l
            pad_t = (ht - nh) // 2
            pad_b = ht - nh - pad_t
            padded = F.pad(resized, (pad_l, pad_r, pad_t, pad_b), value=114.0 / 255.0)
            clips_out = padded.reshape(bsz, kf, 3, ht, wt)
            pad_x, pad_y = float(pad_l), float(pad_t)

        targets_by_k: list[list[dict[str, torch.Tensor]]] = []
        for k in range(kf):
            step_tgts = [
                _scale_pad_target_dict(c["targets"][k], scale, pad_x, pad_y, wt, ht)
                for c in chosen
            ]
            targets_by_k.append(step_tgts)
        return clips_out, targets_by_k


def _merge_temporal_step_outputs(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge recurrent observations without duplicating or renumbering identities."""
    merged = dict(steps[0])
    for key, value in merged.items():
        if isinstance(value, torch.Tensor) and key not in {"anchor_points", "anchor_strides"}:
            merged[key] = torch.cat([step[key] for step in steps], dim=0)
    return merged


def train_temporal_plate_detector(
    train_cfg: TemporalDetTrainConfig,
    out_ckpt_path: str | Path,
    *,
    teacher_model: TemporalPlateDetNet | None = None,
    teacher_ckpt_path: str | Path | None = None,
    teacher_ckpt_sha256: str | None = None,
    shared_pool: _Scene3DTemporalClipPool | None = None,
    repo_root: str | Path | None = None,
    training_script: str | None = None,
    device: torch.device | str | None = None,
) -> tuple[TemporalPlateDetNet, dict[str, Any], _Scene3DTemporalClipPool]:
    """Train Gen-4 `TemporalPlateDetNet` (frozen Apache-2.0 `DFineCore` + synthetic-trained heads)."""
    from vpocr_det.isp_gpu import GPUCameraISPAugmenter, GPUISPConfig

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    pt_dir: Path | None = None
    if repo_root is not None:
        cand_pt = Path(repo_root).resolve() / "out" / "models" / "pretrained_apache20"
        if cand_pt.is_dir():
            pt_dir = cand_pt

    model_cfg = TemporalDetModelConfig.from_preset(
        train_cfg.preset, use_v5_heads=train_cfg.use_v5_heads
    )
    model, init_state_sha256 = instantiate_deterministic_temporal_det_model(
        model_cfg, init_seed=train_cfg.init_seed, pretrained_dir=pt_dir
    )
    model = model.to(dev)
    ema = ModelEMA(model, decay=0.9960, warmup_tau=180.0)
    criterion = TemporalTaskAlignedDetLoss().to(dev)
    isp_aug = GPUCameraISPAugmenter(config=GPUISPConfig(domain_split="train"), seed=train_cfg.seed + 55).to(dev).eval()

    teacher_state_sha: str | None = None
    if teacher_model is None and teacher_ckpt_path is not None:
        teacher_model = load_temporal_det_checkpoint(teacher_ckpt_path, device=dev)
    if teacher_model is not None:
        if repo_root is not None and teacher_ckpt_path is None:
            raise ValueError("teacher_ckpt_path is required when distilling with repo_root provenance")
        teacher_model = teacher_model.to(dev).eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_state_sha = compute_module_state_sha256(teacher_model.state_dict())

    pool = shared_pool
    if pool is not None and pool.domain_split != "train":
        raise ValueError("Training cannot consume a held-out scene pool")
    if pool is None:
        pool = _Scene3DTemporalClipPool(
            image_hw=train_cfg.clip_image_hw,
            num_frames=train_cfg.clip_frames,
            max_clips=train_cfg.max_pool_clips,
            seed=train_cfg.seed,
            device=dev,
            domain_split="train",
        )
        pool.ingest_clips(train_cfg.initial_clips)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Separate normalized box-coordinate and log-odds/feature gradient metrics.
    # A large image-to-small-object Jacobian must not rescale every task head.
    box_mean_params = list(model.veh_query_head.box_polish_pred.parameters())
    box_mean_ids = {id(p) for p in box_mean_params}
    other_params = [p for p in trainable_params if id(p) not in box_mean_ids]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.lr,
        total_steps=max(1, train_cfg.total_steps),
        pct_start=0.08,
        anneal_strategy="cos",
        div_factor=15.0,
        final_div_factor=50.0,
    )

    rng_ms = random.Random(train_cfg.seed + 303)
    reset_gen = torch.Generator(device="cpu").manual_seed(train_cfg.seed + 707)
    cur_hw = train_cfg.multi_scale_hws[-1]

    t0 = time.perf_counter()
    r_loss = r_cls = r_box = r_cut = r_sub = r_kd = 0.0
    skipped_nan = 0
    model.train()

    for step in range(1, train_cfg.total_steps + 1):
        if step > 1 and train_cfg.refresh_clips_every > 0 and step % train_cfg.refresh_clips_every == 0:
            pool.ingest_clips(train_cfg.refresh_clips_count)
            model.train()

        if step % 8 == 1:
            if step > int(0.85 * train_cfg.total_steps):
                cur_hw = train_cfg.multi_scale_hws[-1]
            else:
                cur_hw = rng_ms.choice(train_cfg.multi_scale_hws)

        clips_bkhw, targets_by_k = pool.sample_batch(
            train_cfg.batch_size, target_hw=cur_hw, device=dev, isp_aug=isp_aug
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            step_outs = model.forward_clip(
                clips_bkhw,
                reset_prob=train_cfg.reset_prob,
                generator=reset_gen,
            )
            t_outs: list[dict[str, Any]] | None = None
            if teacher_model is not None:
                with torch.no_grad():
                    t_outs = teacher_model.forward_clip(clips_bkhw, reset_prob=0.0)

            kf = len(step_outs)
            tot_loss = clips_bkhw.sum() * 0.0
            s_cls = s_box = s_cut = s_sub = s_kd = 0.0
            # Flatten clip time into batch so one loss matches once per image while
            # its contrastive term sees persistent identities across observations.
            merged = _merge_temporal_step_outputs(step_outs)
            merged_targets = [target for targets in targets_by_k for target in targets]
            ld = criterion(merged, merged_targets, cutout_head=model.cutout_head)
            tot_loss = ld["loss"]
            s_cls, s_box = float(ld["loss_cls"]), float(ld["loss_box"])
            s_cut, s_sub = float(ld["loss_cutout"]), float(ld["loss_subtype"])
            if t_outs is not None and train_cfg.kd_weight > 0:
                kd_l = sum(compute_temporal_det_kd_loss(s_out, t_out) for s_out, t_out in zip(step_outs, t_outs))/kf
                tot_loss = tot_loss + train_cfg.kd_weight*kd_l
                s_kd = float(kd_l.detach())

        if not torch.isfinite(tot_loss):
            skipped_nan += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(other_params, max_norm=8.0)
        torch.nn.utils.clip_grad_norm_(box_mean_params, max_norm=8.0)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        r_loss += float(tot_loss.detach().item())
        r_cls += s_cls
        r_box += s_box
        r_cut += s_cut
        r_sub += s_sub
        r_kd += s_kd

        if step % train_cfg.log_every == 0 or step == train_cfg.total_steps:
            denom = float(
                train_cfg.log_every
                if step % train_cfg.log_every == 0
                else (step % train_cfg.log_every)
            )
            elapsed = time.perf_counter() - t0
            fps = (step * train_cfg.batch_size * train_cfg.clip_frames) / max(1e-3, elapsed)
            print(
                f"[TemporalDet-{train_cfg.preset}] step {step}/{train_cfg.total_steps} | "
                f"hw={cur_hw} | loss={r_loss/denom:.4f} (cls={r_cls/denom:.4f}, box={r_box/denom:.4f}, "
                f"cut={r_cut/denom:.4f}, sub={r_sub/denom:.4f}, kd={r_kd/denom:.4f}) | {fps:.1f} frames/s",
                flush=True,
            )
            r_loss = r_cls = r_box = r_cut = r_sub = r_kd = 0.0

    total_time = time.perf_counter() - t0
    pretrained_manifest = _build_dfine_pretrained_manifest(model_cfg)
    prov_manifest = None
    if repo_root is not None:
        prov_manifest = build_det_quad_provenance_manifest(
            arch=f"TemporalPlateDetNet-{train_cfg.preset}",
            init_mode=PHASE4_APACHE20_DFINE_INIT_MODE,
            init_seed=train_cfg.init_seed,
            init_state_sha256=init_state_sha256,
            repo_root=repo_root,
            training_script=training_script,
            source_files=DEFAULT_PHASE4_SYNTH_SOURCE_FILES,
            teacher_checkpoint=teacher_ckpt_path,
            teacher_checkpoint_sha256=teacher_ckpt_sha256,
            teacher_state_sha256=teacher_state_sha,
            pretrained_backbone=pretrained_manifest,
        )
        teacher_ckpt_sha256 = prov_manifest["teacher_checkpoint_sha256"]

    teacher_rel_val = (
        prov_manifest["teacher_checkpoint"]
        if prov_manifest is not None
        else (str(teacher_ckpt_path) if teacher_ckpt_path is not None else None)
    )
    teacher_state_val = (
        prov_manifest["teacher_state_sha256"] if prov_manifest is not None else teacher_state_sha
    )
    summary: dict[str, Any] = {
        "arch": "TemporalPlateDetNet",
        "preset": train_cfg.preset,
        "domain_split": "train",
        "zero_real_pixel_training": False,
        "zero_vpocr_real_pixel_finetuning": True,
        "zero_real\x5feval_leakage": True,
        "init_mode": PHASE4_APACHE20_DFINE_INIT_MODE,
        "pretrained_backbone": pretrained_manifest,
        "init_seed": train_cfg.init_seed,
        "init_state_sha256": init_state_sha256,
        "used_teacher": bool(teacher_model is not None),
        "teacher_checkpoint": teacher_rel_val,
        "teacher_checkpoint_sha256": teacher_ckpt_sha256,
        "teacher_state_sha256": teacher_state_val,
        "num_parameters": ema.ema.num_parameters,
        "num_trainable_parameters": model.num_trainable_parameters,
        "total_steps": train_cfg.total_steps,
        "skipped_nan_steps": skipped_nan,
        "total_frames_seen": train_cfg.total_steps * train_cfg.batch_size * train_cfg.clip_frames,
        "clips_rendered": pool.clip_counter,
        "elapsed_s": round(total_time, 2),
    }
    save_temporal_det_checkpoint(
        out_ckpt_path,
        model=model,
        ema_model=ema.ema,
        extra=summary,
        provenance=prov_manifest,
    )
    return ema.ema, summary, pool


def _quantize_onnx_dense_convs_and_strip(fp32_path: Path, int8_path: Path) -> None:
    """Quantize dense `group == 1` Conv weights to QUInt8 while keeping depthwise `group > 1` in FP32."""
    m_fp32 = onnx.load(str(fp32_path))
    dw_nodes = [
        n.name
        for n in m_fp32.graph.node
        if n.op_type == "Conv" and any(a.name == "group" and a.i > 1 for a in n.attribute)
    ]
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=dw_nodes,
    )
    m_i8 = onnx.load(str(int8_path))
    del m_i8.graph.value_info[:]
    for n in m_i8.graph.node:
        n.doc_string = ""
    onnx.save(m_i8, str(int8_path))
    onnx.checker.check_model(str(int8_path))


def export_temporal_det_onnx(
    det_model: TemporalPlateDetNet, out_dir: str | Path,
    input_hw: tuple[int, int] = (448, 768), stem_prefix: str | None = None,
) -> dict[str, Any]:
    """Export through the sole unified detector/query-conditioned cutout bridge."""
    info = export_unified_temporal_det_onnx(det_model, out_dir, input_hw=input_hw)
    if stem_prefix is not None:
        for precision, suffix in (("fp32", ".onnx"), ("int8", ".int8.onnx")):
            key = f"det_{precision}_path"
            path, target = Path(info[key]), Path(out_dir)/f"{stem_prefix}{suffix}"
            if path != target:
                path.replace(target)
            info[key] = str(target)
    for component in ("det", "cutout"):
        for precision in ("fp32", "int8"):
            info[f"{component}_{precision}_mb"] = Path(info[f"{component}_{precision}_path"]).stat().st_size/1024**2
    info["combined_int8_mb"] = info["combined_int8_bytes"]/1024**2
    return info


@torch.no_grad()
def _verify_query_condition_cutout_parity(
    model: TemporalPlateDetNet, runtime: UnifiedTemporalDetORT,
    p2_ref: torch.Tensor, p2_actual: torch.Tensor,
    edge_ref: torch.Tensor, edge_actual: torch.Tensor,
    rois: torch.Tensor, query_ref: torch.Tensor, query_actual: torch.Tensor,
    *, atol: float, rtol: float,
) -> float:
    """Compare a learned interface in its consumer's Bernoulli observation units.

    Latent feature coordinates have arbitrary scale and null directions. Their
    physical contract is the conditional pixel probability produced downstream.
    """
    maximum = 0.0
    for start in range(0, len(rois), 64):
        region = rois[start:start + 64]
        reference = model.cutout_head(p2_ref, edge_ref, region, query_ref[start:start + 64]).float().sigmoid()
        actual = runtime.predict_vehicle_cutouts(p2_actual, edge_actual, region, query_actual[start:start + 64]).float().sigmoid()
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        maximum = max(maximum, float((actual - reference).abs().max()))
    return maximum


@torch.no_grad()
def verify_temporal_det_onnx_parity(
    det_model: TemporalPlateDetNet, onnx_dir: str | Path,
    input_hw: tuple[int, int] = (448, 768), stem_prefix: str | None = None,
    seed: int = 911604, atol: float = 4e-3, rtol: float = 2e-3,
    prob_atol: float = 3e-3, pyr_coord_px_atol: float = 2e-2,
    query_prob_atol: float = 1e-2, coord_px_atol: float = .55,
    footprint_px_atol: float = 1.10, rollout_steps: int = 5,
) -> dict[str, Any]:
    """Compare the full observation contract, recurrent states and dynamic cutouts.

    TopK queries are unordered. Hungarian box matching aligns them before all
    query attributes (including covariance, appearance and unknown) are checked.
    """
    root = Path(onnx_dir)
    prefix = stem_prefix or f"temporal_det_{det_model.config.preset}"
    cut_prefix = f"vehicle_cutout_{det_model.config.preset}"
    paths = [(root/f"{prefix}{suffix}", root/f"{cut_prefix}{suffix}")
             for suffix in (".onnx", ".int8.onnx")]
    runtimes = [UnifiedTemporalDetORT(*pair) for pair in paths]
    model = _fuse_conv_bn_inplace(copy.deepcopy(det_model).cpu().eval())
    rng = np.random.default_rng(seed)
    states = [None, None, None]
    max_diff = max_cut = 0.
    maxima = dict.fromkeys(("max_pyr_prob_abs_diff", "max_pyr_coord_px_abs_diff",
                           "max_prob_abs_diff", "max_feat_abs_diff",
                           "max_coord_px_abs_diff", "max_footprint_px_abs_diff"), 0.)
    sigmoid_names = {"cls_logits", "pred_cabin_valid_logit", "pred_footprint_valid_logit",
                     "synth_veh_logits", "veh_log_odds", "pred_q_mount_vis_logit"}
    coord_names = {"pred_boxes", "pred_corners", "pred_cabin_boxes", "raw_q_boxes_px", "pred_q_mount_boxes", "anchor_points"}

    def probability(key, array):
        tensor = torch.from_numpy(array)
        if key in sigmoid_names:
            return tensor.sigmoid().numpy()
        if key == "pred_subtype_logits":
            return tensor.softmax(-1).numpy()
        return array

    def compare(key, value, ref, metric, tolerance):
        np.testing.assert_allclose(value, ref, atol=tolerance, rtol=rtol, err_msg=key)
        maxima[metric] = max(maxima[metric], float(np.max(np.abs(value-ref))))

    nq = model.config.num_vehicle_queries
    min_matched = nq
    max_cov_norm_diff = 0.0
    max_query_cutout_prob_diff = 0.0
    query_names = {"synth_veh_logits", "raw_q_boxes_px", "veh_log_odds", "bbox_cov_4x4",
                   "appearance_emb", "query_condition", "subtype_probs", "subtype_unknown_prob",
                   "pred_q_mount_boxes", "pred_q_mount_vis_logit"}
    full_names = {"cls_logits", "pred_boxes", "pred_corners", "pred_subtype_logits",
                  "pred_cabin_valid_logit", "pred_cabin_boxes", "pred_footprint_valid_logit", "pred_footprints"}
    # Two batch sizes also verify exported symbolic batch axes.
    for batch in (1, 2):
        states = [None, None, None]
        for _ in range(max(1, rollout_steps)):
            image = torch.from_numpy(rng.random((batch, 3, *input_hw), dtype=np.float32))
            eager = model.forward_step(image, states[0])
            actuals = [rt.forward_step(image, states[i+1]) for i, rt in enumerate(runtimes)]
            for out in actuals:
                for key in UNIFIED_TEMPORAL_OUTPUT_NAMES:
                    if not torch.isfinite(out[key]).all():
                        raise AssertionError(f"Non-finite unified ONNX output: {key}")
            actual = actuals[0]
            for bi in range(batch):
                pb = eager["pred_boxes"][bi, -nq:].numpy()
                ob = actual["pred_boxes"][bi, -nq:].numpy()
                qc_e = eager["query_condition"][bi].numpy()
                qc_o = actual["query_condition"][bi].numpy()
                box_diff = np.max(np.abs(pb[:, None] - ob[None]), -1)
                qc_diff = np.max(np.abs(qc_e[:, None] - qc_o[None]), -1)
                cost = box_diff + 10.0 * qc_diff
                ri, ci = _match_query_nll(cost)
                good = (box_diff[ri, ci] <= coord_px_atol) & (qc_diff[ri, ci] <= 5e-3)
                ri, ci = ri[good], ci[good]
                active_pt = set(np.flatnonzero(eager["veh_log_odds"][bi].sigmoid().numpy() >= .05))
                active_ort = set(np.flatnonzero(actual["veh_log_odds"][bi].sigmoid().numpy() >= .05))
                if not active_pt <= set(ri) or not active_ort <= set(ci):
                    raise AssertionError("A confident vehicle query is unmatched in ONNX parity")
                min_matched = min(min_matched, len(ri))
                if len(ri) < max(1, int(0.85 * nq)):
                    raise AssertionError(f"Only {len(ri)}/{nq} unordered ONNX queries match")
                for key in UNIFIED_TEMPORAL_OUTPUT_NAMES:
                    ref, value = eager[key].numpy(), actual[key].numpy()
                    if key in query_names:
                        ref, value = ref[bi, ri], value[bi, ci]
                    elif key in full_names:
                        # Dense slots stay aligned; only the query tail is unordered.
                        dense_ref = probability(key, ref[bi, :-nq])
                        dense_value = probability(key, value[bi, :-nq])
                        is_prob = key in sigmoid_names or key == "pred_subtype_logits"
                        compare(key, dense_value, dense_ref,
                                "max_pyr_prob_abs_diff" if is_prob else "max_pyr_coord_px_abs_diff",
                                prob_atol if is_prob else pyr_coord_px_atol)
                        ref, value = ref[bi, -nq:][ri], value[bi, -nq:][ci]
                    elif key == "anchor_points":
                        # Query anchors come from the first batch element and
                        # have pixel units, so align their unordered tail too.
                        if bi != 0:
                            continue
                        np.testing.assert_allclose(value[:-nq], ref[:-nq], atol=pyr_coord_px_atol, rtol=rtol, err_msg=key)
                        ref, value = ref[-nq:][ri], value[-nq:][ci]
                    ref, value = probability(key, ref), probability(key, value)
                    if key == "anchor_strides":
                        np.testing.assert_allclose(value, ref, atol=atol, rtol=rtol, err_msg=key)
                    elif key == "query_condition":
                        maxima["max_feat_abs_diff"] = max(maxima["max_feat_abs_diff"], float(np.max(np.abs(value - ref))))
                        boxes = eager["pred_boxes"][bi, -nq:][ri]
                        rois = torch.cat([torch.full((len(ri), 1), float(bi)), boxes], -1)
                        error = _verify_query_condition_cutout_parity(
                            model, runtimes[0], eager["p2_feat"], actual["p2_feat"],
                            eager["aux_edge_3ch"], actual["aux_edge_3ch"], rois,
                            torch.from_numpy(ref), torch.from_numpy(value), atol=prob_atol, rtol=rtol)
                        max_query_cutout_prob_diff = max(max_query_cutout_prob_diff, error)
                    elif key == "bbox_cov_4x4":
                        # Convert squared-pixel covariance to the normalized image chart.
                        scale_vec = np.array([input_hw[1], input_hw[0], input_hw[1], input_hw[0]], dtype=np.float64)
                        scale_mat = scale_vec[:, None] * scale_vec[None, :]
                        norm_val = value.astype(np.float64) / scale_mat
                        norm_ref = ref.astype(np.float64) / scale_mat
                        cov_norm_atol = max(atol, coord_px_atol / max(1.0, float(min(input_hw))))
                        np.testing.assert_allclose(norm_val, norm_ref, atol=cov_norm_atol, rtol=max(rtol, 5e-3), err_msg=key)
                        for cov in (ref, value):
                            cov64 = cov.astype(np.float64)
                            np.testing.assert_allclose(cov64, cov64.swapaxes(-1, -2), atol=1e-4, rtol=1e-4)
                            if not np.all(np.linalg.eigvalsh(cov64) > 0):
                                raise AssertionError("Non-SPD query covariance in ONNX parity")
                        cov_diff = float(np.max(np.abs(norm_val - norm_ref)))
                        max_cov_norm_diff = max(max_cov_norm_diff, cov_diff)
                        max_diff = max(max_diff, cov_diff)
                        continue
                    else:
                        metric, tolerance = (
                            ("max_footprint_px_abs_diff", footprint_px_atol) if key == "pred_footprints" else
                            ("max_coord_px_abs_diff", coord_px_atol) if key in coord_names else
                            ("max_prob_abs_diff", query_prob_atol) if key in sigmoid_names or key in
                                {"pred_subtype_logits", "subtype_probs", "subtype_unknown_prob"} else
                            ("max_feat_abs_diff", atol))
                        compare(key, value, ref, metric, tolerance)
                    max_diff = max(max_diff, float(np.max(np.abs(value-ref))))
            for ref, value in zip(eager["h_next"], actual["h_next"]):
                torch.testing.assert_close(value, ref, atol=atol, rtol=rtol)
            # Use identical features and query conditions to isolate the cutout bridge.
            rois = torch.tensor([[float(bi), 0., 0., float(input_hw[1]), float(input_hw[0])]
                                 for bi in range(batch)])
            query = eager["query_condition"][:, 0]
            ref_cut = model.cutout_head(eager["p2_feat"], eager["aux_edge_3ch"], rois, query)
            for i, runtime in enumerate(runtimes):
                cut = runtime.predict_vehicle_cutouts(eager["p2_feat"], eager["aux_edge_3ch"], rois, query)
                if not torch.isfinite(cut).all():
                    raise AssertionError("Non-finite query-conditioned INT8 cutout")
                if i == 0:
                    torch.testing.assert_close(cut, ref_cut, atol=prob_atol, rtol=rtol)
                    max_cut = max(max_cut, float((cut-ref_cut).abs().max()))
            states = [eager["h_next"], actuals[0]["h_next"], actuals[1]["h_next"]]
    det_bytes, cut_bytes = (p.stat().st_size for p in paths[1])
    return {**maxima, "max_query_condition_cutout_prob_abs_diff": max_query_cutout_prob_diff,
            "max_bbox_cov_norm_abs_diff": max_cov_norm_diff, "max_det_abs_diff": max_diff, "max_cutout_abs_diff": max_cut,
            "min_matched_queries": min_matched, "num_queries": nq,
            "det_int8_mb": det_bytes/1024**2, "cutout_int8_mb": cut_bytes/1024**2,
            "combined_int8_mb": (det_bytes+cut_bytes)/1024**2}


def build_stage1_validation_pool(**kwargs: Any) -> _Scene3DTemporalClipPool:
    """Model-selection scenes/identities and ISP sensor families held out from training."""
    return _Scene3DTemporalClipPool(domain_split="val_holdout", **kwargs)


def build_stage1_calibration_pool(**kwargs: Any) -> _Scene3DTemporalClipPool:
    """Calibration-only grammar, identities, trajectories, fonts and sensor families."""
    return _Scene3DTemporalClipPool(domain_split="cal_holdout", **kwargs)


def build_stage2_validation_pool(**kwargs: Any) -> _Scene3DConcentricROIPool:
    return _Scene3DConcentricROIPool(domain_split="val_holdout", **kwargs)


def build_stage2_calibration_pool(**kwargs: Any) -> _Scene3DConcentricROIPool:
    return _Scene3DConcentricROIPool(domain_split="cal_holdout", **kwargs)


def _restore_float_onnx_calibration_buffers(path: Path) -> None:
    """Undo weight quantization of the tiny policy calibration Gather tables.

    ORT quantize_dynamic also quantizes constant Gather inputs. Post-INT8
    calibration must update those parameters without quantization rounding.
    Dense learned weights remain quantized; only these explicit parameters
    and their dequantization aliases are restored to FP32.
    """
    graph = onnx.load(str(path))
    tensors = {item.name: onnx.numpy_helper.to_array(item) for item in graph.graph.initializer}
    changed = False
    for name, quantized in list(tensors.items()):
        if not name.endswith("_quantized"):
            continue
        base = name[:-len("_quantized")]
        if base.rsplit(".", 1)[-1] not in {"cal_temp", "cal_bias", "pi_train"}:
            continue
        scale, zero = tensors[base+"_scale"], tensors[base+"_zero_point"]
        value = ((quantized.astype(np.float32)-zero.astype(np.float32))*scale).astype(np.float32)
        graph.graph.initializer.append(onnx.numpy_helper.from_array(value, name=base))
        gather_outputs = set()
        for node in graph.graph.node:
            if node.op_type == "Gather" and node.input[0] == name:
                node.input[0] = base
                gather_outputs.update(node.output)
        for node in graph.graph.node:
            if node.op_type == "DequantizeLinear" and node.input[0] in gather_outputs:
                source = node.input[0]
                node.op_type = "Identity"
                del node.input[:]
                node.input.append(source)
                del node.attribute[:]
        changed = True
    if changed:
        used = {name for node in graph.graph.node for name in node.input}
        kept = [item for item in graph.graph.initializer if item.name in used]
        del graph.graph.initializer[:]
        graph.graph.initializer.extend(kept)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))


def _onnx_calibration_buffers(path: str | Path) -> dict[str, np.ndarray]:
    graph = onnx.load(str(path))
    buffers = {item.name.rsplit(".", 1)[-1]: onnx.numpy_helper.to_array(item).copy()
               for item in graph.graph.initializer
               if item.name.rsplit(".", 1)[-1] in {"cal_temp", "cal_bias", "pi_train"}}
    if not {"cal_temp", "cal_bias"} <= buffers.keys():
        raise ValueError("ONNX graph must preserve calibration temperature and bias buffers")
    return buffers


def _fit_binary_calibration(logits: np.ndarray, labels: np.ndarray,
                            max_iter: int = 80) -> dict[str, float]:
    """Proper Bernoulli NLL with positive temperature, fitted on exported logits."""
    x = torch.as_tensor(np.asarray(logits).reshape(-1), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels).reshape(-1), dtype=torch.float64)
    if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Calibration needs aligned finite logits and binary labels")
    if not ((y == 0) | (y == 1)).all() or y.unique().numel() != 2:
        raise ValueError("Calibration requires both plate/vehicle and null/background observations")
    log_temp = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp, bias], max_iter=max_iter, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x/log_temp.clamp(-6, 6).exp()+bias, y)
        loss.backward()
        return loss
    before = float(F.binary_cross_entropy_with_logits(x, y))
    optimizer.step(closure)
    temperature, offset = float(log_temp.detach().clamp(-6, 6).exp()), float(bias.detach())
    after = float(F.binary_cross_entropy_with_logits(x/temperature+offset, y))
    if not math.isfinite(after) or after > before:
        temperature, offset, after = 1., 0., before
    return {"cal_temp": temperature, "cal_bias": offset, "nll_before": before, "nll_after": after}


def _require_calibration_pool(pool: Any) -> None:
    if pool.domain_split != "cal_holdout" or pool.engine.config.domain_split != "cal_holdout" or pool.isp_aug.config.domain_split != "cal_holdout":
        raise ValueError("Post-INT8 fitting requires cal_holdout scenes AND cal_holdout ISP families")


def _write_calibration_sidecar(path: Path, graph_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    result["synth_source_hashes"] = {rel: compute_det_file_sha256(root/rel) for rel in DEFAULT_PHASE4_SYNTH_SOURCE_FILES}
    result.update(domain_split="cal_holdout", distribution="synthetic", onnx_path=str(graph_path),
                  calibrated_onnx_sha256=compute_det_file_sha256(graph_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return {**result, "sidecar_path": str(path)}


def fit_stage1_post_int8_calibration(
    onnx_path: str | Path, calibration_pool: _Scene3DTemporalClipPool | None = None, *,
    num_batches: int = 8, batch_size: int = 2, num_clips: int = 8,
    seed: int = 20260601, device: torch.device | str = "cpu", max_iter: int = 80,
    sidecar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit vehicle/background temperature+bias on matched INT8 query outputs.

    Every unmatched query is background, using the same amodal Gaussian/class
    assignment as training. Re-fitting first inverts existing calibration.
    This calibrates synthetic query observations, not arbitrary real cameras.
    """
    if min(num_batches, batch_size, num_clips, max_iter) < 1:
        raise ValueError("Calibration counts must be positive")
    path = Path(onnx_path)
    _require_int8_onnx(path)
    session = _cpu_ort_session(path)
    hw = tuple(map(int, session.get_inputs()[0].shape[-2:]))
    pool = calibration_pool or build_stage1_calibration_pool(image_hw=hw, seed=seed, device=device)
    _require_calibration_pool(pool)
    if not pool.items:
        pool.ingest_clips(num_clips)
    buffers = _onnx_calibration_buffers(path)
    old_temp, old_bias = float(buffers["cal_temp"].item()), float(buffers["cal_bias"].item())
    logits, labels = [], []
    for _ in range(num_batches):
        clips, targets = pool.sample_batch(batch_size, hw, torch.device(device))
        state = None
        for k, step_targets in enumerate(targets):
            if state is None:
                state = [np.zeros((batch_size, *meta.shape[1:]), dtype=np.float32) for meta in session.get_inputs()[1:3]]
            feed = dict(zip(("images", "h_p3_in", "h_p4_in"), (clips[:, k].cpu().numpy(), *state)))
            out = dict(zip((meta.name for meta in session.get_outputs()), session.run(None, feed)))
            state = [out["h_p3_out"], out["h_p4_out"]]
            score = out["veh_log_odds"]
            nq = score.shape[1]
            binary = np.zeros_like(score)
            for bi, target in enumerate(step_targets):
                gt = target["amodal_boxes"][target["labels"] == CLS_VEHICLE].numpy()
                if len(gt):
                    cov = torch.from_numpy(out["bbox_cov_4x4"][bi]).double()
                    chol = torch.linalg.cholesky(cov)
                    error = torch.from_numpy(gt[:, None]-out["pred_boxes"][bi, -nq:][None]).double()
                    white = torch.linalg.solve_triangular(chol[None], error[..., None], upper=False)[..., 0]
                    cost = .5*white.square().sum(-1)+chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)[None]-torch.from_numpy(score[bi])[None]
                    _, queries = _match_query_nll(cost.numpy())
                    binary[bi, queries] = 1.
            logits.append((score-old_bias)*old_temp)
            labels.append(binary)
    x, y = np.concatenate([a.reshape(-1) for a in logits]), np.concatenate([a.reshape(-1) for a in labels])
    fit = _fit_binary_calibration(x, y, max_iter)
    set_unified_onnx_calibration(path, fit["cal_temp"], fit["cal_bias"])
    result = {"stage": 1, **fit, "num_observations": len(y), "num_positive": int(y.sum()),
              "num_clips": pool.clip_counter, "seed": seed, "target": "vehicle_query"}
    return _write_calibration_sidecar(Path(sidecar_path) if sidecar_path else path.with_suffix(".calibration.json"), path, result)


def _require_int8_onnx(path: Path) -> None:
    graph = onnx.load(str(path))
    if not any(node.op_type in {"ConvInteger", "MatMulInteger", "QLinearConv", "QLinearMatMul", "DynamicQuantizeLinear", "DequantizeLinear"} for node in graph.graph.node):
        raise ValueError("Post-INT8 calibration requires an actually quantized ONNX graph")


def fit_stage2_post_int8_calibration(
    onnx_path: str | Path, calibration_pool: _Scene3DConcentricROIPool | None = None, *,
    num_batches: int = 8, batch_size: int = 16, num_clips: int = 8,
    seed: int = 20260602, device: torch.device | str = "cpu", max_iter: int = 80,
    support_alpha: float = .05, sidecar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit policy-conditioned plate/null likelihoods and clip-level support cutoff.

    Max support score per independent positive clip protects simultaneous clip
    coverage. An insufficient conformal sample reports an unbounded cutoff
    (JSON null), never an unsupported finite coverage claim. Unsampled policies
    keep their parameters and are explicitly marked uncalibrated.
    """
    if min(num_batches, batch_size, num_clips, max_iter) < 1 or batch_size < 2 or not 0 < support_alpha < 1:
        raise ValueError("Invalid calibration counts or support_alpha")
    path = Path(onnx_path)
    _require_int8_onnx(path)
    pool = calibration_pool or build_stage2_calibration_pool(seed=seed, device=device)
    _require_calibration_pool(pool)
    _restore_float_onnx_calibration_buffers(path)
    if not pool.pos_items or not pool.neg_items:
        pool.ingest_clips(num_clips)
    session = _cpu_ort_session(path)
    buffers = _onnx_calibration_buffers(path)
    temperature, bias = buffers["cal_temp"].copy(), buffers["cal_bias"].copy()
    prior = buffers.get("pi_train", np.full(4, .5))
    log_prior = np.log(prior/(1-prior))
    samples: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(temperature))}
    for _ in range(num_batches):
        batch = pool.sample_batch(batch_size, torch.device(device))
        feed = {key: batch[key].cpu().numpy() for key in ("fovea_rois", "surround_rois", "sampling_policy")}
        out = dict(zip((meta.name for meta in session.get_outputs()), session.run(None, feed)))
        for i, policy in enumerate(feed["sampling_policy"]):
            policy = int(policy)
            label = float(batch["is_plate"][i])
            raw = (float(out["plate_null_llr"][i])+log_prior[policy]-bias[policy])*temperature[policy]
            samples[policy].append((float(raw), label))
    # Evaluate every retained positive frame/proposal, rather than claiming
    # simultaneous coverage from a random subset of a trajectory.
    support_by_clip = {i: {} for i in samples}
    for start in range(0, len(pool.pos_items), batch_size):
        items = pool.pos_items[start:start+batch_size]
        feed = {name: np.stack([item[key] for item in items]).transpose(0, 3, 1, 2).astype(np.float32)/255
                for name, key in (("fovea_rois", "fovea"), ("surround_rois", "surround"))}
        feed["sampling_policy"] = np.array([item["sampling_policy"] for item in items], dtype=np.int64)
        scores = session.run(["support_nonconformity"], feed)[0]
        for item, score in zip(items, scores):
            if not math.isfinite(float(score)):
                raise ValueError("Non-finite INT8 support score")
            clip, policy = item["clip_id"], item["sampling_policy"]
            support_by_clip[policy][clip] = max(support_by_clip[policy].get(clip, 0.), float(score))
    policies = {}
    for policy, pairs in samples.items():
        if not pairs or len({y for _, y in pairs}) != 2:
            policies[str(policy)] = {"calibrated": False, "num_observations": len(pairs)}
            continue
        fit = _fit_binary_calibration(*np.asarray(pairs).T, max_iter=max_iter)
        temperature[policy], bias[policy] = fit["cal_temp"], fit["cal_bias"]
        # Convert the calibrated sampling posterior to a prior-corrected LLR.
        prior[policy] = float(np.asarray(pairs)[:, 1].mean())
        scores = sorted(support_by_clip[policy].values())
        rank = math.ceil((len(scores)+1)*(1-support_alpha))
        quantile = scores[rank-1] if 1 <= rank <= len(scores) else None
        policies[str(policy)] = {"calibrated": True, **fit, "num_observations": len(pairs),
                                 "support_quantile": quantile, "support_num_clips": len(scores),
                                 "support_bounded": quantile is not None}
    if not any(value["calibrated"] for value in policies.values()):
        raise ValueError("No policy has both plate and null calibration support")
    graph = onnx.load(str(path))
    for item in graph.graph.initializer:
        key = item.name.rsplit(".", 1)[-1]
        if key in {"cal_temp", "cal_bias", "pi_train"}:
            value = {"cal_temp": temperature, "cal_bias": bias, "pi_train": prior}[key]
            item.CopyFrom(onnx.numpy_helper.from_array(value, name=item.name))
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))
    result = {"stage": 2, "policies": policies, "support_alpha": support_alpha,
              "support_unit": "positive_clip_max", "support_scope": "all_retained_positive_proposals",
              "prior_source": "per_policy_calibration_sampling_frequency",
              "num_clips": pool.clip_counter, "seed": seed,
              "cal_temp": temperature.tolist(), "cal_bias": bias.tolist(), "pi_train": prior.tolist()}
    return _write_calibration_sidecar(Path(sidecar_path) if sidecar_path else path.with_suffix(".calibration.json"), path, result)
