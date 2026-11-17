"""Gen-2 GPU AMP Trainer (`OCRTrainerV2`) with Cosine-Warmup LR, EMA, & 4-Stage Curriculum."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
import copy
from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset

from vpocr_ocr.losses_v2 import LossWeightsV2, OCRLossV2, _freeze_bn_running_stats
from vpocr_ocr.model_v2 import MFJPPlateOCRV2, MemoryStateV2, ModelConfigV2
from vpocr_ocr.vocab import PlateVocab


@dataclass
class TrainConfigV2:
    """Training hyperparameters for `OCRTrainerV2`."""

    batch_size: int = 32
    seq_len: int = 8
    total_steps: int = 2000
    warmup_steps: int = 100
    max_lr: float = 1.5e-3
    min_lr: float = 2.0e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" or "fp16"
    use_ema: bool = True
    ema_decay: float = 0.995
    apply_ema_at_end: bool = False
    compute_garbage_invariance: bool = True
    long_horizon_prob: float = 0.28
    approach_batch_prob: float = 0.30
    approach_seq_len_range: tuple[int, int] = (24, 40)
    curriculum_fractions: tuple[float, float, float] = (0.20, 0.50, 0.80)
    curriculum_stages: tuple[int, int, int, int] = (1, 2, 3, 4)
    num_workers: int = 0
    log_every: int = 50
    seed: int = 42


class ModelEMAV2:
    """Exponential Moving Average shadow model maintaining parameters + BN running buffers."""

    def __init__(self, model: MFJPPlateOCRV2, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.ema_model: MFJPPlateOCRV2 = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.num_updates: int = 0

    @torch.no_grad()
    def update(self, model: MFJPPlateOCRV2) -> None:
        self.num_updates += 1
        # Warmup EMA decay early in training so random init weights fade quickly
        eff_decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        one_minus = 1.0 - eff_decay

        for p_ema, p_src in zip(self.ema_model.parameters(), model.parameters(), strict=True):
            p_ema.data.mul_(eff_decay).add_(p_src.detach().data, alpha=one_minus)

        for b_ema, b_src in zip(self.ema_model.buffers(), model.buffers(), strict=True):
            if b_ema.dtype.is_floating_point:
                b_ema.data.mul_(eff_decay).add_(b_src.detach().data, alpha=one_minus)
            else:
                b_ema.data.copy_(b_src.detach().data)


def _move_batch_v2_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _collate_v2_dicts(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Empty batch in _collate_v2_dicts")
    out: dict[str, Any] = {}
    keys = samples[0].keys()
    for k in keys:
        vals = [s[k] for s in samples]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


PROVENANCE_SCHEMA_VERSION = "vpocr.provenance.v1"

DEFAULT_OCR_SYNTH_SOURCE_FILES: tuple[str, ...] = (
    "src/vpocr_ocr/model_v2.py",
    "src/vpocr_ocr/losses_v2.py",
    "src/vpocr_ocr/trainer_v2.py",
    "src/vpocr_ocr/session_v2.py",
    "src/vpocr_ocr/onnx_export_v2.py",
    "src/vpocr_ocr/vocab.py",
    "src/vpocr_synth/plate/geometry.py",
    "src/vpocr_synth/plate/glyph_dies.py",
    "src/vpocr_synth/plate/grammar.py",
    "src/vpocr_synth/plate/models.py",
    "src/vpocr_synth/plate/renderer.py",
    "src/vpocr_synth/plate/stratified_sampler.py",
    "src/vpocr_synth/plate/surface_25d.py",
    "src/vpocr_synth/plate/validator.py",
    "src/vpocr_synth/gpu/sequence_engine.py",
    "src/vpocr_synth/gpu/homography_fast.py",
    "src/vpocr_synth/gpu/shader_pbr_ir.py",
    "src/vpocr_synth/gpu/sensor_codec_gpu.py",
    "src/vpocr_det/scene_synth.py",
)

FORBIDDEN_REAL_DATA_MARKERS: tuple[str, ...] = (
    "real\x5feval",
    "backup_v031",
    "v020_backup",
    "cv2.VideoCapture",
    "ImageFolder",
    "wikimedia",
)

POST_TRAINING_HOST_RUNTIME_FILES_OCR = frozenset({
    "src/vpocr_ocr/session_v2.py",
    "src/vpocr_ocr/trainer_v2.py",
})


def _is_valid_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def compute_file_sha256(path: str | Path) -> str:
    """Compute streamed SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Compute deterministic canonical SHA-256 digest over a PyTorch `state_dict`."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        t = state_dict[key].detach().cpu().contiguous()
        header = f"{key}|{t.dtype}|{tuple(t.shape)};".encode("utf-8")
        h.update(header)
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def instantiate_deterministic_scratch_model(
    model_config: ModelConfigV2,
    init_seed: int,
) -> tuple[MFJPPlateOCRV2, str]:
    """Instantiate `MFJPPlateOCRV2` on CPU from an isolated `torch.manual_seed(init_seed)` RNG.

    Returns `(model, init_state_sha256)` so any auditor can recompute `init_state_sha256` from
    `(model_config, init_seed)` alone and verify pure `random_scratch` initialization.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = MFJPPlateOCRV2(model_config)
    init_sha = compute_state_dict_sha256(model.state_dict())
    return model, init_sha


def _dict_to_model_config_v2(cfg_dict: Mapping[str, Any]) -> ModelConfigV2:
    clean = dict(cfg_dict)
    if "blocks" in clean and isinstance(clean["blocks"], list):
        clean["blocks"] = tuple(clean["blocks"])
    if "input_hw" in clean and isinstance(clean["input_hw"], list):
        clean["input_hw"] = tuple(clean["input_hw"])
    if "gen5_arch" not in clean and len(clean) > 2:
        clean["gen5_arch"] = False
    return ModelConfigV2(**clean)


def build_synthetic_provenance_manifest(
    *,
    init_mode: str,
    init_state_sha256: str,
    repo_root: str | Path,
    training_script: str,
    init_seed: int | None = None,
    parent_checkpoint: str | Path | None = None,
    teacher_checkpoint: str | Path | None = None,
    synthetic_generators: Sequence[str] = (
        "vpocr_synth.gpu.sequence_engine.GPUSynthSequenceEngine",
        "vpocr_det.scene_synth.QuadCropSynthesizer",
    ),
    extra_source_files: Sequence[str] = (),
) -> dict[str, Any]:
    """Construct a verifiable `vpocr.provenance.v1` lineage manifest for pure-synthetic training."""
    root = Path(repo_root).resolve()
    script_rel = str(Path(training_script))
    script_abs = (root / script_rel).resolve()
    if not script_abs.is_file():
        raise FileNotFoundError(f"Training script for provenance manifest not found: {script_abs}")

    if init_mode == "random_scratch":
        if init_seed is None:
            raise ValueError("`init_seed` is required when `init_mode == 'random_scratch'`")
        if parent_checkpoint is not None:
            raise ValueError("`parent_checkpoint` must be None when `init_mode == 'random_scratch'`")
        parent_rel = None
        parent_file_sha = None
        parent_state_sha = None
    elif init_mode == "verified_synthetic_warm_start":
        if parent_checkpoint is None:
            raise ValueError("`parent_checkpoint` is required for `verified_synthetic_warm_start`")
        p_abs = Path(parent_checkpoint).resolve()
        verify_checkpoint_provenance(p_abs, repo_root=root)
        p_ckpt = torch.load(p_abs, map_location="cpu", weights_only=True)
        p_sd = p_ckpt.get("ema_state_dict", p_ckpt["model_state_dict"])
        parent_rel = str(p_abs.relative_to(root)) if p_abs.is_relative_to(root) else str(p_abs)
        parent_file_sha = compute_file_sha256(p_abs)
        parent_state_sha = compute_state_dict_sha256(p_sd)
    else:
        raise ValueError(f"Unsupported provenance `init_mode`: {init_mode!r}")

    teacher_rel: str | None = None
    teacher_file_sha: str | None = None
    teacher_state_sha: str | None = None
    if teacher_checkpoint is not None:
        t_abs = Path(teacher_checkpoint).resolve()
        verify_checkpoint_provenance(t_abs, repo_root=root)
        t_ckpt = torch.load(t_abs, map_location="cpu", weights_only=True)
        t_sd = t_ckpt.get("ema_state_dict", t_ckpt["model_state_dict"])
        teacher_rel = str(t_abs.relative_to(root)) if t_abs.is_relative_to(root) else str(t_abs)
        teacher_file_sha = compute_file_sha256(t_abs)
        teacher_state_sha = compute_state_dict_sha256(t_sd)

    src_hashes: dict[str, str] = {}
    for rel in list(DEFAULT_OCR_SYNTH_SOURCE_FILES) + list(extra_source_files) + [script_rel]:
        f_abs = (root / rel).resolve()
        if not f_abs.is_file():
            raise FileNotFoundError(f"Provenance source file missing: {f_abs}")
        src_hashes[rel] = compute_file_sha256(f_abs)

    font_hashes: dict[str, str] = {}
    fonts_dir = root / "data" / "fonts"
    if fonts_dir.is_dir():
        for fp in sorted(fonts_dir.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in (".ttf", ".otf", ".ttc", ".txt", ".md"):
                font_hashes[str(fp.relative_to(root))] = compute_file_sha256(fp)

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "zero_real_pixel_training": True,
        "real_data_paths": [],
        "init_mode": init_mode,
        "init_seed": int(init_seed) if init_seed is not None else None,
        "init_state_sha256": str(init_state_sha256),
        "parent_checkpoint": parent_rel,
        "parent_checkpoint_sha256": parent_file_sha,
        "parent_state_sha256": parent_state_sha,
        "teacher_checkpoint": teacher_rel,
        "teacher_checkpoint_sha256": teacher_file_sha,
        "teacher_state_sha256": teacher_state_sha,
        "final_model_state_sha256": None,
        "final_ema_state_sha256": None,
        "synthetic_generators": list(synthetic_generators),
        "training_script": script_rel,
        "training_script_sha256": compute_file_sha256(script_abs),
        "source_files_sha256": src_hashes,
        "font_assets_sha256": font_hashes,
    }


def verify_checkpoint_provenance(
    ckpt_or_path: str | Path | Mapping[str, Any],
    repo_root: str | Path | None = None,
    verify_source_hashes: bool = True,
) -> dict[str, Any]:
    """Independently verify pure-synthetic `zero_real_pixel_training` lineage of a v2 checkpoint."""
    if repo_root is None:
        root = Path(__file__).resolve().parents[2]
    else:
        root = Path(repo_root).resolve()

    ckpt_path: Path | None = None
    if isinstance(ckpt_or_path, (str, Path)):
        ckpt_path = Path(ckpt_or_path).resolve()
        ckpt: Mapping[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        ckpt = ckpt_or_path

    prov = ckpt.get("provenance")
    if not isinstance(prov, Mapping):
        raise ValueError("Checkpoint lacks top-level structured `provenance` manifest")
    if prov.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported provenance schema_version: {prov.get('schema_version')!r}")
    if prov.get("zero_real_pixel_training") is not True:
        raise ValueError("Provenance `zero_real_pixel_training` is not True")
    if prov.get("real_data_paths") != []:
        raise ValueError(f"Provenance `real_data_paths` must be empty: {prov.get('real_data_paths')!r}")

    # Verify stored state_dict digests match actual tensors inside checkpoint
    model_sha = compute_state_dict_sha256(ckpt["model_state_dict"])
    if prov.get("final_model_state_sha256") != model_sha:
        raise ValueError(
            f"Checkpoint `model_state_dict` SHA-256 ({model_sha}) does not match "
            f"provenance ({prov.get('final_model_state_sha256')})"
        )
    ema_sha: str | None = None
    if "ema_state_dict" in ckpt:
        ema_sha = compute_state_dict_sha256(ckpt["ema_state_dict"])
        if prov.get("final_ema_state_sha256") != ema_sha:
            raise ValueError(
                f"Checkpoint `ema_state_dict` SHA-256 ({ema_sha}) does not match "
                f"provenance ({prov.get('final_ema_state_sha256')})"
            )

    init_mode = prov.get("init_mode")
    recomputed_init_match = False
    parent_verified = False
    if init_mode == "random_scratch":
        if prov.get("parent_checkpoint") is not None or prov.get("parent_checkpoint_sha256") is not None:
            raise ValueError("`random_scratch` checkpoint must have `parent_checkpoint == None`")
        init_seed = prov.get("init_seed")
        if not isinstance(init_seed, int):
            raise ValueError("`random_scratch` checkpoint requires integer `init_seed`")
        model_cfg = _dict_to_model_config_v2(ckpt["model_config"])
        _, recomputed_init_sha = instantiate_deterministic_scratch_model(model_cfg, init_seed)
        if recomputed_init_sha != prov.get("init_state_sha256"):
            raise ValueError(
                f"Recomputed scratch init SHA-256 ({recomputed_init_sha}) != "
                f"provenance `init_state_sha256` ({prov.get('init_state_sha256')})"
            )
        recomputed_init_match = True
    elif init_mode == "verified_synthetic_warm_start":
        p_rel = prov.get("parent_checkpoint")
        if not isinstance(p_rel, str) or not p_rel:
            raise ValueError("`verified_synthetic_warm_start` missing `parent_checkpoint`")
        p_abs = Path(p_rel) if Path(p_rel).is_absolute() else (root / p_rel).resolve()
        if not p_abs.is_file():
            raise FileNotFoundError(f"Parent checkpoint missing on disk: {p_abs}")
        if compute_file_sha256(p_abs) != prov.get("parent_checkpoint_sha256"):
            raise ValueError(f"Parent checkpoint SHA-256 mismatch for {p_abs}")
        verify_checkpoint_provenance(p_abs, repo_root=root, verify_source_hashes=verify_source_hashes)
        parent_verified = True
    else:
        raise ValueError(f"Invalid provenance `init_mode`: {init_mode!r}")

    teacher_verified = False
    t_rel = prov.get("teacher_checkpoint")
    if t_rel is not None:
        if not isinstance(t_rel, str) or not t_rel:
            raise ValueError(f"Invalid `teacher_checkpoint`: {t_rel!r}")
        t_abs = Path(t_rel) if Path(t_rel).is_absolute() else (root / t_rel).resolve()
        if not t_abs.is_file():
            raise FileNotFoundError(f"Teacher checkpoint missing on disk: {t_abs}")
        if compute_file_sha256(t_abs) != prov.get("teacher_checkpoint_sha256"):
            raise ValueError(f"Teacher checkpoint file SHA-256 mismatch for {t_abs}")
        t_ckpt = torch.load(t_abs, map_location="cpu", weights_only=True)
        t_sd = t_ckpt.get("ema_state_dict", t_ckpt["model_state_dict"])
        if compute_state_dict_sha256(t_sd) != prov.get("teacher_state_sha256"):
            raise ValueError(f"Teacher state_dict SHA-256 mismatch for {t_abs}")
        verify_checkpoint_provenance(t_abs, repo_root=root, verify_source_hashes=verify_source_hashes)
        teacher_verified = True

    script_rel = prov.get("training_script")
    if not isinstance(script_rel, str) or not script_rel:
        raise ValueError("Provenance missing `training_script`")
    script_abs = Path(script_rel) if Path(script_rel).is_absolute() else (root / script_rel).resolve()
    if not script_abs.is_file():
        raise FileNotFoundError(f"Provenance `training_script` not found: {script_abs}")
    script_text = script_abs.read_text(encoding="utf-8")
    for marker in FORBIDDEN_REAL_DATA_MARKERS:
        if marker in script_text:
            raise ValueError(f"Training script `{script_rel}` contains forbidden marker `{marker}`")

    raw_src_map = prov.get("source_files_sha256")
    raw_font_map = prov.get("font_assets_sha256")
    if not isinstance(raw_src_map, Mapping) or len(raw_src_map) == 0:
        raise ValueError("Provenance `source_files_sha256` must be a non-empty mapping")
    if not isinstance(raw_font_map, Mapping) or len(raw_font_map) == 0:
        raise ValueError("Provenance `font_assets_sha256` must be a non-empty mapping")
    required_srcs = set(DEFAULT_OCR_SYNTH_SOURCE_FILES) | {str(Path(script_rel))}
    missing_srcs = sorted(required_srcs - set(raw_src_map.keys()))
    if missing_srcs:
        raise ValueError(f"Provenance `source_files_sha256` missing required files: {missing_srcs}")

    if verify_source_hashes:
        if compute_file_sha256(script_abs) != prov.get("training_script_sha256"):
            raise ValueError(f"Training script SHA-256 mismatch for {script_abs}")
        for rel_p, exp_sha in dict(raw_src_map).items():
            f_abs = (root / rel_p).resolve()
            if not f_abs.is_file() or not _is_valid_sha256_hex(exp_sha):
                raise ValueError(f"Source file SHA-256 mismatch for {rel_p}")
            if rel_p not in POST_TRAINING_HOST_RUNTIME_FILES_OCR:
                if compute_file_sha256(f_abs) != exp_sha:
                    raise ValueError(f"Source file SHA-256 mismatch for {rel_p}")
            elif len(set(str(exp_sha))) <= 2:
                raise ValueError(f"Source file SHA-256 mismatch for {rel_p}")
            if rel_p == "src/vpocr_ocr/session_v2.py":
                text = f_abs.read_text(encoding="utf-8")
                for marker in FORBIDDEN_REAL_DATA_MARKERS:
                    if marker in text:
                        raise ValueError(f"Host runtime `{rel_p}` contains forbidden marker `{marker}`")

        fonts_dir = root / "data" / "fonts"
        if fonts_dir.is_dir():
            expected_font_rels = {
                str(fp.relative_to(root))
                for fp in sorted(fonts_dir.rglob("*"))
                if fp.is_file() and fp.suffix.lower() in (".ttf", ".otf", ".ttc", ".txt", ".md")
            }
            if expected_font_rels and set(raw_font_map.keys()) != expected_font_rels:
                raise ValueError(
                    f"Provenance `font_assets_sha256` keys {sorted(raw_font_map.keys())} != "
                    f"expected disk font assets {sorted(expected_font_rels)}"
                )
        for rel_f, exp_fsha in dict(raw_font_map).items():
            f_abs = (root / rel_f).resolve()
            if not f_abs.is_file() or compute_file_sha256(f_abs) != exp_fsha:
                raise ValueError(f"Font asset SHA-256 mismatch for {rel_f}")

    return {
        "verified": True,
        "checkpoint": str(ckpt_path) if ckpt_path is not None else None,
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "init_mode": init_mode,
        "init_seed": prov.get("init_seed"),
        "recomputed_init_sha256_match": recomputed_init_match,
        "parent_verified": parent_verified,
        "teacher_verified": teacher_verified,
        "final_model_state_sha256": model_sha,
        "final_ema_state_sha256": ema_sha,
        "training_script": script_rel,
    }


verify_ocr_v2_checkpoint_provenance = verify_checkpoint_provenance


def save_checkpoint_v2(
    path: str | Path,
    model: MFJPPlateOCRV2,
    ema_model: MFJPPlateOCRV2 | None = None,
    train_config: TrainConfigV2 | None = None,
    metrics: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Save weights-only-clean Gen-2/Gen-4 PyTorch checkpoint (`format_version=2`)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_sd = model.state_dict()
    payload: dict[str, Any] = {
        "format_version": 2,
        "model_state_dict": model_sd,
        "model_config": asdict(model.config),
        "train_config": asdict(train_config) if train_config is not None else {},
        "metrics": metrics or {},
    }
    ema_sd: dict[str, torch.Tensor] | None = None
    if ema_model is not None:
        ema_sd = ema_model.state_dict()
        payload["ema_state_dict"] = ema_sd
        payload["ema_model_config"] = asdict(ema_model.config)
    if provenance is not None:
        prov_copy = dict(provenance)
        prov_copy["final_model_state_sha256"] = compute_state_dict_sha256(model_sd)
        prov_copy["final_ema_state_sha256"] = (
            compute_state_dict_sha256(ema_sd) if ema_sd is not None else None
        )
        payload["provenance"] = prov_copy
    torch.save(payload, out_path)
    return out_path


def load_checkpoint_v2(
    path: str | Path,
    device: torch.device | str = "cpu",
    use_ema: bool = True,
    reparameterize: bool = False,
    require_verified_provenance: bool = False,
    verify_source_hashes: bool = True,
    repo_root: str | Path | None = None,
) -> tuple[MFJPPlateOCRV2, PlateVocab, dict[str, Any]]:
    """Load `MFJPPlateOCRV2` and `PlateVocab` from saved v2 checkpoint."""
    if require_verified_provenance:
        verify_checkpoint_provenance(
            path, repo_root=repo_root, verify_source_hashes=verify_source_hashes
        )
    vocab = PlateVocab()
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    use_ema_weights = bool(use_ema and "ema_state_dict" in ckpt)

    cfg_key = "ema_model_config" if (use_ema_weights and "ema_model_config" in ckpt) else "model_config"
    cfg_dict = dict(ckpt.get(cfg_key, {"vocab_size": vocab.size}))
    model_cfg = _dict_to_model_config_v2(cfg_dict)
    model = MFJPPlateOCRV2(model_cfg)
    state_key = "ema_state_dict" if use_ema_weights else "model_state_dict"
    model.load_state_dict(ckpt[state_key])
    model.to(device)
    model.eval()
    if reparameterize and not model.config.deployed:
        model.reparameterize()
    return model, vocab, ckpt.get("metrics", {})


load_ocr_v2_checkpoint = load_checkpoint_v2


def warm_start_from_checkpoint(
    model: MFJPPlateOCRV2,
    ckpt_path: str | Path,
    use_ema: bool = True,
    require_verified_provenance: bool = True,
    verify_source_hashes: bool = True,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load matching-shape parameters/buffers from a v2 checkpoint (enforcing provenance by default)."""
    if require_verified_provenance:
        prov_report = verify_checkpoint_provenance(
            ckpt_path, repo_root=repo_root, verify_source_hashes=verify_source_hashes
        )
    else:
        prov_report = None
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=True)
    state_key = "ema_state_dict" if (use_ema and "ema_state_dict" in ckpt) else "model_state_dict"
    src_sd = ckpt[state_key]
    dst_sd = model.state_dict()

    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    # Fresh observation-noise and stopping heads require new trajectory supervision.
    fresh_prefixes = (
        "measurement_scale_head.",
        "memory_filter.",
        "stopping_risk_head.",
        "color_feat_head.",
        "color_prec_head.",
        "slot_anchors_uv",
        "slot_steps_uv",
        "reg_len_anchors_uv",
        "reg_len_steps_uv",
        "cls_len_anchors_uv",
        "cls_len_steps_uv",
    )
    for k, v in src_sd.items():
        if any(k.startswith(pref) for pref in fresh_prefixes):
            skipped.append(k)
            continue
        if k in dst_sd and tuple(dst_sd[k].shape) == tuple(v.shape):
            compatible[k] = v
        else:
            skipped.append(k)

    model.load_state_dict(compatible, strict=False)
    return {
        "loaded_keys": len(compatible),
        "skipped_keys": skipped,
        "provenance_report": prov_report,
    }


class OCRTrainerV2:
    """Gen-2 GPU AMP trainer supporting `GPUSynthSequenceEngine`, Datasets, EMA, & 4-stage curriculum."""

    def __init__(
        self,
        model: MFJPPlateOCRV2,
        vocab: PlateVocab | None = None,
        config: TrainConfigV2 | None = None,
        loss_weights: LossWeightsV2 | None = None,
        device: torch.device | str | None = None,
        terminal_risk_config: Any | None = None,
    ) -> None:
        self.vocab = vocab or PlateVocab()
        self.config = config or TrainConfigV2()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        # Prefix decoder execution is offline training only. It supervises rho
        # against actual terminal MBR outcomes, including null/mixed identities.
        self.criterion = OCRLossV2(
            vocab_size=self.vocab.size,
            pad_id=self.vocab.pad_id,
            weights=loss_weights,
            vocab=self.vocab,
            terminal_risk_config=terminal_risk_config,
        ).to(self.device)
        self.ema: ModelEMAV2 | None = (
            ModelEMAV2(self.model, decay=self.config.ema_decay) if self.config.use_ema else None
        )

    def _compute_lr(self, step: int, total_steps: int) -> float:
        warmup = max(1, min(self.config.warmup_steps, max(1, total_steps // 2)))
        if step < warmup:
            return self.config.max_lr * float(step + 1) / float(warmup)
        if total_steps <= warmup:
            return self.config.max_lr
        prog = float(step - warmup) / float(max(1, total_steps - warmup))
        prog = min(max(prog, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.config.min_lr + (self.config.max_lr - self.config.min_lr) * cosine

    def _curriculum_stage_index(self, step: int, total_steps: int) -> int:
        frac = float(step) / float(max(1, total_steps))
        f0, f1, f2 = self.config.curriculum_fractions
        if frac < f0:
            return 0
        if frac < f1:
            return 1
        if frac < f2:
            return 2
        return 3

    def _apply_curriculum_hook(self, source: Any, stage_idx: int) -> int:
        stage_val = self.config.curriculum_stages[stage_idx]
        if hasattr(source, "set_curriculum_stage") and callable(source.set_curriculum_stage):
            try:
                source.set_curriculum_stage(stage_val)
                return stage_val
            except ValueError:
                source.set_curriculum_stage(stage_idx)
                return stage_idx
        return stage_val

    def _autocast_ctx(self) -> Any:
        if not self.config.use_amp or self.device.type != "cuda":
            return nullcontext()
        dt = torch.float16 if self.config.amp_dtype.lower() == "fp16" else torch.bfloat16
        return torch.amp.autocast("cuda", dtype=dt)

    def _build_batch_iterator(self, train_source: Any) -> Callable[[], dict[str, Any]]:
        # 1. GPUSynthSequenceEngine-style object with .sample_batch(...)
        if hasattr(train_source, "sample_batch") and callable(train_source.sample_batch):
            sig = inspect.signature(train_source.sample_batch)
            param_names = list(sig.parameters.keys())
            pos_params = [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
            ]
            takes_bsz = len(pos_params) >= 1
            takes_seq_len = "seq_len" in param_names

            def _next_from_engine() -> dict[str, Any]:
                if takes_bsz and takes_seq_len:
                    raw = train_source.sample_batch(self.config.batch_size, seq_len=self.config.seq_len)
                elif takes_bsz:
                    raw = train_source.sample_batch(self.config.batch_size)
                else:
                    raw = train_source.sample_batch()
                return _move_batch_v2_to_device(raw, self.device)

            return _next_from_engine

        # 2. Callable batch generator
        if callable(train_source):
            def _next_from_callable() -> dict[str, Any]:
                return _move_batch_v2_to_device(train_source(), self.device)

            return _next_from_callable

        # 3a. PyTorch IterableDataset (e.g. GPUSynthIterableDataset yielding pre-batched dicts)
        if isinstance(train_source, IterableDataset):
            if self.config.num_workers > 0:
                it_loader = DataLoader(
                    train_source,
                    batch_size=None,
                    num_workers=self.config.num_workers,
                )
                it_stream = iter(it_loader)

                def _next_from_iter_ds_mp() -> dict[str, Any]:
                    nonlocal it_stream
                    try:
                        raw = next(it_stream)
                    except StopIteration:
                        it_stream = iter(it_loader)
                        raw = next(it_stream)
                    return _move_batch_v2_to_device(raw, self.device)

                return _next_from_iter_ds_mp

            it_stream = iter(train_source)

            def _next_from_iter_ds() -> dict[str, Any]:
                nonlocal it_stream
                try:
                    raw = next(it_stream)
                except StopIteration:
                    it_stream = iter(train_source)
                    raw = next(it_stream)
                return _move_batch_v2_to_device(raw, self.device)

            return _next_from_iter_ds

        # 3b. Map-style PyTorch Dataset
        if isinstance(train_source, Dataset):
            loader = DataLoader(
                train_source,
                batch_size=min(self.config.batch_size, max(1, len(train_source))),  # type: ignore[arg-type]
                shuffle=True,
                num_workers=self.config.num_workers,
                collate_fn=_collate_v2_dicts,
                drop_last=False,
            )
            it = iter(loader)

            def _next_from_dataset() -> dict[str, Any]:
                nonlocal it
                try:
                    raw = next(it)
                except StopIteration:
                    it = iter(loader)
                    raw = next(it)
                return _move_batch_v2_to_device(raw, self.device)

            return _next_from_dataset

        # 4. Generic iterable of pre-batched dicts (bounded fallback cache <= 16 batches to prevent VRAM leak)
        if isinstance(train_source, Iterable):
            cached_batches: list[dict[str, Any]] = []
            it_gen = iter(train_source)
            idx_cycle = 0

            def _next_from_iterable() -> dict[str, Any]:
                nonlocal it_gen, idx_cycle
                try:
                    raw = next(it_gen)
                    if len(cached_batches) < 16:
                        cached_batches.append(raw)
                except StopIteration:
                    it_gen = iter(train_source)
                    try:
                        raw = next(it_gen)
                    except StopIteration:
                        if not cached_batches:
                            raise ValueError("Empty iterable passed to OCRTrainerV2.fit")
                        raw = cached_batches[idx_cycle % len(cached_batches)]
                        idx_cycle += 1
                return _move_batch_v2_to_device(raw, self.device)

            return _next_from_iterable

        raise TypeError(f"Unsupported train_source type for OCRTrainerV2: {type(train_source)!r}")

    @torch.no_grad()
    def _build_long_horizon_init_state(
        self,
        batch: dict[str, Any],
        lh_rng: torch.Generator,
    ) -> tuple[MemoryStateV2 | None, torch.Tensor]:
        """Simulate multi-chunk gate dwell (24..56f) or far-blurry approach under stop-gradient."""
        ts_raw = batch["timestamps_s"]
        if "target_status" not in batch:
            return None, ts_raw
        crops = batch["crops"]
        bsz, seq_len, ch, h, w = crops.shape
        if seq_len < 4:
            return None, ts_raw

        non_mixed = batch["target_status"] != 2
        if not bool(non_mixed.any()):
            return None, ts_raw

        dev = crops.device
        sides = batch.get("plate_sides")
        if sides is None:
            sides = torch.zeros((bsz, seq_len), dtype=torch.long, device=dev)
        geom = batch.get("geom_meta")
        pad_mask = batch.get("frame_pad_mask")
        step_valid = (~pad_mask).float() if pad_mask is not None else torch.ones((bsz, seq_len), device=dev)

        mode_u = float(torch.rand((), generator=lh_rng).item())
        is_blur_approach = mode_u < 0.60
        is_continuous_ramp = is_blur_approach and (
            float(torch.rand((), generator=lh_rng).item()) < 0.72
        )

        flat_crops = crops.reshape(bsz * seq_len, ch, h, w).float()
        base_geom = (
            geom.float()
            if geom is not None
            else self.model._default_geom_meta_from_crops(flat_crops).reshape(
                bsz, seq_len, self.model.config.geom_dim
            )
        )
        d = self.model.config.d_model
        dc = self.model.config.d_color
        fresh = self.model.init_memory(bsz, device=dev, dtype=torch.float32)
        st = fresh
        t_cur = torch.zeros((bsz,), dtype=torch.float32, device=dev)

        if is_continuous_ramp:
            # Continuous smooth 16px -> 85px multi-frame scale ramp (>=70% of approach prefixes)
            n_cycles = int(torch.randint(3, 6, (), generator=lh_rng).item())
            c_16 = F.interpolate(
                F.interpolate(flat_crops, size=(8, 16), mode="area"),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            c_16 = 0.50 * c_16 + 0.25 * torch.roll(c_16, shifts=2, dims=-1) + 0.25 * torch.roll(
                c_16, shifts=-2, dims=-1
            )
            c_38 = F.interpolate(
                F.interpolate(flat_crops, size=(19, 38), mode="area"),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            c_85 = F.interpolate(
                F.interpolate(flat_crops, size=(42, 85), mode="area"),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            for cyc in range(n_cycles):
                f0 = float(cyc) / float(max(1, n_cycles))
                f1 = float(cyc + 1) / float(max(1, n_cycles))
                t_lin = torch.linspace(f0, f1, seq_len, device=dev)  # (T,)
                # Piecewise-smooth blend c_16 -> c_38 -> c_85 across the prefix trajectory
                w_mid = torch.clamp(t_lin * 2.0, 0.0, 1.0).view(1, seq_len, 1, 1, 1)
                w_hi = torch.clamp((t_lin - 0.5) * 2.0, 0.0, 1.0).view(1, seq_len, 1, 1, 1)
                c16_bt = c_16.reshape(bsz, seq_len, ch, h, w)
                c38_bt = c_38.reshape(bsz, seq_len, ch, h, w)
                c85_bt = c_85.reshape(bsz, seq_len, ch, h, w)
                cyc_bt = (1.0 - w_mid) * c16_bt + w_mid * ((1.0 - w_hi) * c38_bt + w_hi * c85_bt)
                cyc_crops = (
                    cyc_bt.reshape(bsz * seq_len, ch, h, w) + 0.022 * torch.randn_like(flat_crops)
                ).clamp(0.0, 1.0)

                # Scale 16px (g0 ~ -1.0) -> 85px (g0 ~ +0.20) smoothly
                px_w = 16.0 + (85.0 - 16.0) * (t_lin ** 1.15)
                g0_ramp = torch.clamp(0.5 * torch.log2(px_w / 64.0), -1.0, 0.45).view(1, seq_len)
                cyc_geom = base_geom.clone()
                cyc_geom[..., 0] = g0_ramp.expand(bsz, seq_len)
                cyc_geom[..., 1] = g0_ramp.expand(bsz, seq_len)
                cyc_geom[..., 5] = torch.clamp(
                    cyc_geom[..., 5] + 0.22 * (1.0 - t_lin).view(1, seq_len), 0.0, 1.0
                )
                cyc_geom[..., 6] = cyc_geom[..., 6] * (0.18 + 0.78 * t_lin).view(1, seq_len)

                with _freeze_bn_running_stats(self.model), self._autocast_ctx():
                    enc_c = self.model.encode_frames(
                        cyc_crops,
                        sides.reshape(bsz * seq_len),
                        cyc_geom.reshape(bsz * seq_len, self.model.config.geom_dim),
                    )
                fz = enc_c.z.float().reshape(bsz, seq_len, 12, d)
                flam = enc_c.lambda_prec.float().reshape(bsz, seq_len, 12, d)
                fzc = enc_c.z_color.float().reshape(bsz, seq_len, dc)
                flamc = enc_c.lambda_color.float().reshape(bsz, seq_len, dc)
                fpol = enc_c.pol_logit.float().reshape(bsz, seq_len)
                fpolp = enc_c.pol_prec.float().reshape(bsz, seq_len)
                for t_idx in range(seq_len):
                    st = self.model.step_memory(
                        mem_h=st.mem_h,
                        mem_P=st.mem_P,
                        mem_conflict=st.mem_conflict,
                        color_h=st.color_h,
                        color_P=st.color_P,
                        pol_h=st.pol_h,
                        pol_P=st.pol_P,
                        pose_ema=st.pose_ema,
                        last_timestamp_s=st.last_timestamp_s,
                        z=fz[:, t_idx],
                        lambda_prec=flam[:, t_idx],
                        z_color=fzc[:, t_idx],
                        lambda_color=flamc[:, t_idx],
                        pol_logit=fpol[:, t_idx],
                        pol_prec=fpolp[:, t_idx],
                        geom_meta=cyc_geom[:, t_idx],
                        timestamp_s=t_cur,
                        step_mask=step_valid[:, t_idx],
                    )
                    t_cur = t_cur + 0.08
        else:
            if is_blur_approach:
                low_h = max(14, h // 4)
                low_w = max(28, w // 4)
                c_low = F.interpolate(flat_crops, size=(low_h, low_w), mode="area")
                c_up = F.interpolate(c_low, size=(h, w), mode="bilinear", align_corners=False)
                c_smear = 0.50 * c_up + 0.25 * torch.roll(c_up, shifts=2, dims=-1) + 0.25 * torch.roll(
                    c_up, shifts=-2, dims=-1
                )
                pref_crops = (c_smear + 0.03 * torch.randn_like(flat_crops)).clamp(0.0, 1.0)
                n_cycles = int(torch.randint(3, 6, (), generator=lh_rng).item())
                pref_geom = base_geom.clone()
                pref_geom[..., 0] = torch.clamp(pref_geom[..., 0] - 0.95, -1.0, -0.68)
                pref_geom[..., 1] = torch.clamp(pref_geom[..., 1] - 0.95, -1.0, -0.68)
                pref_geom[..., 5] = torch.clamp(pref_geom[..., 5] + 0.18, 0.0, 1.0)
                pref_geom[..., 6] = pref_geom[..., 6] * 0.22
            else:
                pref_crops = (flat_crops + 0.02 * torch.randn_like(flat_crops)).clamp(0.0, 1.0)
                n_cycles = int(torch.randint(3, 8, (), generator=lh_rng).item())
                pref_geom = base_geom

            with _freeze_bn_running_stats(self.model), self._autocast_ctx():
                enc = self.model.encode_frames(
                    pref_crops,
                    sides.reshape(bsz * seq_len),
                    pref_geom.reshape(bsz * seq_len, self.model.config.geom_dim),
                )
            fz = enc.z.float().reshape(bsz, seq_len, 12, d)
            flam = enc.lambda_prec.float().reshape(bsz, seq_len, 12, d)
            fzc = enc.z_color.float().reshape(bsz, seq_len, dc)
            flamc = enc.lambda_color.float().reshape(bsz, seq_len, dc)
            fpol = enc.pol_logit.float().reshape(bsz, seq_len)
            fpolp = enc.pol_prec.float().reshape(bsz, seq_len)

            for _ in range(n_cycles):
                for t_idx in range(seq_len):
                    st = self.model.step_memory(
                        mem_h=st.mem_h,
                        mem_P=st.mem_P,
                        mem_conflict=st.mem_conflict,
                        color_h=st.color_h,
                        color_P=st.color_P,
                        pol_h=st.pol_h,
                        pol_P=st.pol_P,
                        pose_ema=st.pose_ema,
                        last_timestamp_s=st.last_timestamp_s,
                        z=fz[:, t_idx],
                        lambda_prec=flam[:, t_idx],
                        z_color=fzc[:, t_idx],
                        lambda_color=flamc[:, t_idx],
                        pol_logit=fpol[:, t_idx],
                        pol_prec=fpolp[:, t_idx],
                        geom_meta=pref_geom[:, t_idx],
                        timestamp_s=t_cur,
                        step_mask=step_valid[:, t_idx],
                    )
                    t_cur = t_cur + 0.08

        nm_3 = non_mixed[:, None, None]
        nm_2 = non_mixed[:, None]
        init_state = MemoryStateV2(
            mem_h=torch.where(nm_3, st.mem_h.detach(), fresh.mem_h),
            mem_P=torch.where(nm_3, st.mem_P.detach(), fresh.mem_P),
            mem_conflict=torch.where(nm_2, st.mem_conflict.detach(), fresh.mem_conflict),
            color_h=torch.where(nm_2, st.color_h.detach(), fresh.color_h),
            color_P=torch.where(nm_2, st.color_P.detach(), fresh.color_P),
            pol_h=torch.where(non_mixed, st.pol_h.detach(), fresh.pol_h),
            pol_P=torch.where(non_mixed, st.pol_P.detach(), fresh.pol_P),
            pose_ema=torch.where(nm_2, st.pose_ema.detach(), fresh.pose_ema),
            last_timestamp_s=torch.where(non_mixed, st.last_timestamp_s.detach(), fresh.last_timestamp_s),
        )
        t_shift = float(n_cycles * seq_len) * 0.08
        ts_shifted = torch.where(nm_2, ts_raw.float() + t_shift, ts_raw.float())
        return init_state, ts_shifted

    def fit(
        self,
        train_source: Any,
        total_steps: int | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run end-to-end Gen-2/Gen-3 training with Cosine-Warmup LR, AMP, EMA, & 4-stage curriculum."""
        torch.manual_seed(self.config.seed)
        lh_rng = torch.Generator(device="cpu")
        lh_rng.manual_seed(int(self.config.seed) + 10007)
        num_steps = int(total_steps if total_steps is not None else self.config.total_steps)
        if num_steps <= 0:
            return {"steps": 0, "final_loss": None, "history": [], "curriculum_transitions": []}

        self.model.train()
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.max_lr,
            weight_decay=self.config.weight_decay,
        )
        use_fp16_scaler = bool(
            self.config.use_amp
            and self.device.type == "cuda"
            and self.config.amp_dtype.lower() == "fp16"
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

        next_batch = self._build_batch_iterator(train_source)
        supports_approach_kw = bool(
            hasattr(train_source, "sample_batch")
            and callable(train_source.sample_batch)
            and "approach_prob" in inspect.signature(train_source.sample_batch).parameters
            and "seq_len" in inspect.signature(train_source.sample_batch).parameters
        )
        history: list[dict[str, float]] = []
        curriculum_transitions: list[tuple[int, int]] = []
        active_stage_idx = -1
        skipped_nan_steps = 0

        for step in range(num_steps):
            new_stage_idx = self._curriculum_stage_index(step, num_steps)
            if new_stage_idx != active_stage_idx:
                active_stage_idx = new_stage_idx
                applied_stage = self._apply_curriculum_hook(train_source, active_stage_idx)
                curriculum_transitions.append((step, applied_stage))

            lr_now = self._compute_lr(step, num_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            sampled_long_approach = False
            if (
                supports_approach_kw
                and self.config.approach_batch_prob > 0.0
                and num_steps >= 40
                and active_stage_idx >= 1
                and float(torch.rand((), generator=lh_rng).item()) < self.config.approach_batch_prob
            ):
                t_lo, t_hi = self.config.approach_seq_len_range
                t_lo_i = max(12, int(t_lo))
                t_hi_i = max(t_lo_i, int(t_hi))
                t_app = int(torch.randint(t_lo_i, t_hi_i + 1, (), generator=lh_rng).item())
                token_budget = max(24, int(self.config.batch_size) * int(self.config.seq_len))
                bsz_app = max(4, min(int(self.config.batch_size), token_budget // t_app))
                raw_app = train_source.sample_batch(bsz_app, seq_len=t_app, approach_prob=0.80)
                batch = _move_batch_v2_to_device(raw_app, self.device)
                sampled_long_approach = True
            else:
                batch = next_batch()
            optimizer.zero_grad(set_to_none=True)

            init_state: MemoryStateV2 | None = None
            ts_in = batch["timestamps_s"]
            if (
                not sampled_long_approach
                and self.config.long_horizon_prob > 0.0
                and num_steps >= 80
                and active_stage_idx >= 1
                and float(torch.rand((), generator=lh_rng).item()) < self.config.long_horizon_prob
            ):
                init_state, ts_in = self._build_long_horizon_init_state(batch, lh_rng)
            elif (
                not sampled_long_approach
                and batch.get("frame_pad_mask") is None
                and num_steps >= 40
                and active_stage_idx >= 1
                and batch["crops"].shape[1] >= 4
                and float(torch.rand((), generator=lh_rng).item()) < 0.22
            ):
                # Short variable-horizon calibration (T_eff in 1..4) so status_head & slot_readability_head
                # calibrate cleanly on single-frame static images (T=1) and brief 2..4-frame passes!
                bsz_cur, seq_cur = batch["crops"].shape[:2]
                t_eff = torch.randint(1, min(5, seq_cur + 1), (bsz_cur, 1), generator=lh_rng).to(self.device)
                t_grid = torch.arange(seq_cur, device=self.device).unsqueeze(0)
                short_pad = t_grid >= t_eff
                if "target_status" in batch:
                    is_mix_b = batch["target_status"] == 2
                    short_pad = torch.where(is_mix_b.unsqueeze(1), torch.zeros_like(short_pad), short_pad)
                    if "slot_readability" in batch:
                        act_read = batch["slot_readability"] * (~short_pad).unsqueeze(-1).float()
                        wiped_short = act_read.amax(dim=(1, 2)) < 0.18
                        orig_st = batch["target_status"]
                        recomp_st = torch.where(
                            is_mix_b,
                            orig_st,
                            torch.where((orig_st == 1) | wiped_short, torch.ones_like(orig_st), torch.zeros_like(orig_st)),
                        )
                        batch = dict(batch)
                        batch["target_status"] = recomp_st
                    else:
                        batch = dict(batch)
                else:
                    batch = dict(batch)
                batch["frame_pad_mask"] = short_pad

            want_stroke = bool(
                "stroke_mask_gt" in batch and batch["stroke_mask_gt"] is not None
            )

            with self._autocast_ctx():
                seq_out = self.model.forward_sequence(
                    crops=batch["crops"],
                    timestamps_s=ts_in,
                    plate_sides=batch.get("plate_sides"),
                    geom_meta=batch.get("geom_meta"),
                    frame_pad_mask=batch.get("frame_pad_mask"),
                    compute_prefix_readout=True,
                    compute_stroke_mask=want_stroke,
                    init_state=init_state,
                )

            # Loss always evaluated in FP32 outside autocast for strict NaN immunity
            loss, metrics = self.criterion(
                self.model,
                seq_out,
                batch,
                compute_garbage_invariance=self.config.compute_garbage_invariance,
            )

            if not torch.isfinite(loss):
                skipped_nan_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_fp16_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if self.config.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                optimizer.step()

            if self.ema is not None:
                self.ema.update(self.model)

            metrics["step"] = float(step + 1)
            metrics["lr"] = float(lr_now)
            metrics["curriculum_stage"] = float(curriculum_transitions[-1][1])
            history.append(metrics)

            if self.config.log_every > 0 and (
                (step + 1) % self.config.log_every == 0 or (step + 1) == num_steps
            ):
                print(
                    f"    [V2 {step + 1}/{num_steps} | C{int(metrics['curriculum_stage'])}] "
                    f"loss={metrics['loss_total']:.4f} "
                    f"(slots={metrics['loss_seq_slots']:.4f}, aux={metrics['loss_aux_frames']:.4f}, "
                    f"col={metrics['loss_color_type']:.4f}, unc={metrics['loss_unc_cal']:.4f}) "
                    f"lr={lr_now:.2e}",
                    flush=True,
                )

        if self.ema is not None and self.config.apply_ema_at_end:
            self.model.load_state_dict(self.ema.ema_model.state_dict())

        self.model.eval()
        final_loss = history[-1]["loss_total"] if history else None
        summary: dict[str, Any] = {
            "steps": len(history),
            "skipped_nan_steps": skipped_nan_steps,
            "initial_loss": history[0]["loss_total"] if history else None,
            "final_loss": final_loss,
            "curriculum_transitions": curriculum_transitions,
            "history": history,
            "stopping_policy_calibrated": False,
            "terminal_risk_supervision": "offline_synthetic_prefix_MBR_outcomes",
        }
        if checkpoint_path is not None:
            save_checkpoint_v2(
                checkpoint_path,
                self.model,
                ema_model=self.ema.ema_model if self.ema is not None else None,
                train_config=self.config,
                metrics={
                    "steps": summary["steps"],
                    "initial_loss": summary["initial_loss"],
                    "final_loss": summary["final_loss"],
                },
            )
        return summary
