"""ONNX exporter (`frame_encoder.onnx`, `memory_step_decode.onnx`) and ONNX Runtime session."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
from pathlib import Path
from typing import Any
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from vpocr_ocr.model import MFJPPlateOCR
from vpocr_ocr.session import (
    MFPlateOCRSession,
    OCRDecision,
    SessionConfig,
    build_decision_from_arrays,
)
from vpocr_ocr.vocab import JPPlateGrammar, NUM_SLOTS, PlateVocab


class _FrameEncoderExportWrapper(nn.Module):
    """ONNX wrapper for MFJPPlateOCR.encode_frames."""

    def __init__(self, model: MFJPPlateOCR) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, crops: torch.Tensor, plate_sides: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.model.encode_frames(crops, plate_sides)
        return enc.z, enc.slot_quality, enc.frame_valid


class _MemoryStepDecodeExportWrapper(nn.Module):
    """ONNX wrapper combining MFJPPlateOCR.step_memory and decode_memory."""

    def __init__(self, model: MFJPPlateOCR) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        mem_feat: torch.Tensor,
        mem_mass: torch.Tensor,
        mem_conflict: torch.Tensor,
        last_timestamp_s: torch.Tensor,
        z: torch.Tensor,
        slot_quality: torch.Tensor,
        frame_valid: torch.Tensor,
        timestamp_s: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        next_state = self.model.step_memory(
            mem_feat=mem_feat,
            mem_mass=mem_mass,
            mem_conflict=mem_conflict,
            last_timestamp_s=last_timestamp_s,
            z=z,
            slot_quality=slot_quality,
            frame_valid=frame_valid,
            timestamp_s=timestamp_s,
        )
        ro = self.model.decode_memory(
            next_state.mem_feat, next_state.mem_mass, next_state.mem_conflict
        )
        return (
            next_state.mem_feat,
            next_state.mem_mass,
            next_state.mem_conflict,
            next_state.last_timestamp_s,
            ro.slot_logits,
            ro.plate_type_logits,
            ro.slot_readability_pred,
            ro.status_logits,
        )


def fuse_views_numpy(
    z_views: np.ndarray,
    slot_quality_views: np.ndarray,
    frame_valid_views: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact numpy equivalent of MFJPPlateOCR.fuse_views for (V, 12, D) inputs."""
    if z_views.shape[0] == 1:
        return z_views[0], slot_quality_views[0], frame_valid_views[0]

    raw_w = frame_valid_views * slot_quality_views  # (V, 12)
    w_sum = raw_w.sum(axis=0, keepdims=True)        # (1, 12)
    uniform_w = np.full_like(raw_w, 1.0 / float(z_views.shape[0]))
    w_norm = np.where(w_sum > 1e-6, raw_w / np.maximum(w_sum, 1e-6), uniform_w)

    z_step = (z_views * w_norm[:, :, None]).sum(axis=0).astype(np.float32)
    q_step = slot_quality_views.max(axis=0).astype(np.float32)
    v_step = frame_valid_views.max(axis=0).astype(np.float32)
    return z_step, q_step, v_step


def _normalize_crop_views_numpy(
    crop_rgb: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor],
) -> np.ndarray:
    """Convert single crop or sequence of views into (V, 3, 64, 128) float32 [0, 1] numpy array."""
    if isinstance(crop_rgb, (list, tuple)):
        return np.concatenate([_normalize_crop_views_numpy(c) for c in crop_rgb], axis=0)

    if isinstance(crop_rgb, torch.Tensor):
        arr = crop_rgb.detach().cpu().numpy()
    else:
        arr = np.asarray(crop_rgb)

    if arr.ndim == 3:
        if arr.shape[-1] == 3 and arr.shape[0] != 3:
            arr = np.transpose(arr, (2, 0, 1))
        arr = arr[None, ...]
    elif arr.ndim == 4:
        if arr.shape[-1] == 3 and arr.shape[1] != 3:
            arr = np.transpose(arr, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unsupported crop shape: {arr.shape}")

    if arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    else:
        out = arr.astype(np.float32)
        if out.max() > 1.5:
            out = out / 255.0
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0), dtype=np.float32)


def export_ocr_to_onnx(
    model: MFJPPlateOCR,
    output_dir: str | Path,
    opset_version: int = 17,
) -> tuple[Path, Path]:
    """Export frame_encoder.onnx and memory_step_decode.onnx (opset 17, dynamo=False)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_model = copy.deepcopy(model).cpu().eval()
    d = cpu_model.config.d_model

    encoder_path = out_dir / "frame_encoder.onnx"
    mem_path = out_dir / "memory_step_decode.onnx"

    # 1. Export frame_encoder.onnx
    enc_wrapper = _FrameEncoderExportWrapper(cpu_model).eval()
    dummy_crops = torch.rand(2, 3, 64, 128, dtype=torch.float32)
    dummy_sides = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            enc_wrapper,
            (dummy_crops, dummy_sides),
            str(encoder_path),
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["crops", "plate_sides"],
            output_names=["z", "slot_quality", "frame_valid"],
            dynamic_axes={
                "crops": {0: "num_views"},
                "plate_sides": {0: "num_views"},
                "z": {0: "num_views"},
                "slot_quality": {0: "num_views"},
                "frame_valid": {0: "num_views"},
            },
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(encoder_path)))

    # 2. Export memory_step_decode.onnx
    mem_wrapper = _MemoryStepDecodeExportWrapper(cpu_model).eval()
    dummy_mem_feat = torch.randn(1, NUM_SLOTS, d, dtype=torch.float32)
    dummy_mem_mass = torch.rand(1, NUM_SLOTS, dtype=torch.float32)
    dummy_mem_conf = torch.rand(1, NUM_SLOTS, dtype=torch.float32) * 0.2
    dummy_last_ts = torch.tensor([0.1], dtype=torch.float32)
    dummy_z = torch.randn(1, NUM_SLOTS, d, dtype=torch.float32)
    dummy_q = torch.rand(1, NUM_SLOTS, dtype=torch.float32)
    dummy_v = torch.rand(1, 1, dtype=torch.float32)
    dummy_ts = torch.tensor([0.2], dtype=torch.float32)

    mem_inputs = (
        dummy_mem_feat,
        dummy_mem_mass,
        dummy_mem_conf,
        dummy_last_ts,
        dummy_z,
        dummy_q,
        dummy_v,
        dummy_ts,
    )
    mem_in_names = [
        "mem_feat",
        "mem_mass",
        "mem_conflict",
        "last_timestamp_s",
        "z",
        "slot_quality",
        "frame_valid",
        "timestamp_s",
    ]
    mem_out_names = [
        "new_mem_feat",
        "new_mem_mass",
        "new_mem_conflict",
        "new_last_timestamp_s",
        "slot_logits",
        "plate_type_logits",
        "slot_readability_pred",
        "status_logits",
    ]
    mem_dyn = {name: {0: "batch"} for name in (mem_in_names + mem_out_names)}

    with torch.no_grad():
        torch.onnx.export(
            mem_wrapper,
            mem_inputs,
            str(mem_path),
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=mem_in_names,
            output_names=mem_out_names,
            dynamic_axes=mem_dyn,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(mem_path)))

    return encoder_path, mem_path


def _try_preload_torch_cuda_libs() -> None:
    """Preload bundled nvidia/* CUDA/cuDNN shared libraries from site-packages if present."""
    import ctypes

    site_root = Path(ort.__file__).resolve().parent.parent
    nvidia_root = site_root / "nvidia"
    if not nvidia_root.is_dir():
        return
    lib_order = [
        "libcudart.so",
        "libnvJitLink.so",
        "libcublasLt.so",
        "libcublas.so",
        "libcufft.so",
        "libcurand.so",
        "libcudnn.so",
    ]
    lib_dirs = sorted(nvidia_root.glob("*/lib"))
    for prefix in lib_order:
        for ldir in lib_dirs:
            for candidate in sorted(ldir.glob(f"{prefix}*")):
                if candidate.is_file():
                    try:
                        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass


def _resolve_ort_providers(
    providers: Sequence[str | tuple[str, dict[str, Any]]] | None = None,
    use_tf32: bool = False,
) -> list[str | tuple[str, dict[str, Any]]]:
    _try_preload_torch_cuda_libs()
    available = set(ort.get_available_providers())
    if providers is not None:
        resolved: list[str | tuple[str, dict[str, Any]]] = []
        for p in providers:
            name = p[0] if isinstance(p, tuple) else p
            if name in available:
                if name == "CUDAExecutionProvider" and isinstance(p, str) and not use_tf32:
                    resolved.append(("CUDAExecutionProvider", {"use_tf32": 0}))
                else:
                    resolved.append(p)
        if "CPUExecutionProvider" not in [
            (x[0] if isinstance(x, tuple) else x) for x in resolved
        ]:
            resolved.append("CPUExecutionProvider")
        if resolved:
            return resolved

    if "CUDAExecutionProvider" in available:
        cuda_opt = {"use_tf32": 1 if use_tf32 else 0}
        return [("CUDAExecutionProvider", cuda_opt), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OrtMFPlateOCRSession:
    """Streaming multi-frame OCR session backed by ONNX Runtime (CUDA or CPU)."""

    def __init__(
        self,
        onnx_dir: str | Path,
        vocab: PlateVocab | None = None,
        config: SessionConfig | None = None,
        grammar_checker: Callable[..., bool] | None = None,
        providers: Sequence[str | tuple[str, dict[str, Any]]] | None = None,
        use_tf32: bool = False,
    ) -> None:
        onnx_path = Path(onnx_dir)
        enc_file = onnx_path / "frame_encoder.onnx"
        mem_file = onnx_path / "memory_step_decode.onnx"
        if not enc_file.is_file() or not mem_file.is_file():
            raise FileNotFoundError(f"Missing ONNX files in {onnx_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = _resolve_ort_providers(providers, use_tf32=use_tf32)

        self.enc_session = ort.InferenceSession(
            str(enc_file), sess_options=sess_opts, providers=prov
        )
        self.mem_session = ort.InferenceSession(
            str(mem_file), sess_options=sess_opts, providers=prov
        )

        # Infer d_model directly from exported ONNX input shape (B, 12, D)
        mem_feat_shape = self.mem_session.get_inputs()[0].shape
        self.d_model: int = int(mem_feat_shape[2])

        self.vocab = vocab or PlateVocab()
        self.config = config or SessionConfig()
        self.grammar = JPPlateGrammar(self.vocab)
        self.grammar_checker = grammar_checker
        self.reset()

    @property
    def active_providers(self) -> list[str]:
        return list(self.enc_session.get_providers())

    def _empty_state_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((1, NUM_SLOTS, self.d_model), dtype=np.float32),
            np.zeros((1, NUM_SLOTS), dtype=np.float32),
            np.zeros((1, NUM_SLOTS), dtype=np.float32),
            np.array([-1.0], dtype=np.float32),
        )

    def reset(self) -> None:
        """Reset ONNX Runtime track EvidenceMemory state."""
        self._committed_state = self._empty_state_arrays()
        self._active_state = self._committed_state
        self._last_readout: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._current_frame_key: tuple[int, int] | None = None
        self._current_frame_ts: float = 0.0
        self._current_z_views: list[np.ndarray] = []
        self._current_q_views: list[np.ndarray] = []
        self._current_v_views: list[np.ndarray] = []
        self._auto_frame_counter: int = -1
        self.num_frames_seen: int = 0

    def push(
        self,
        crop_rgb: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor],
        timestamp_s: float,
        camera_id: int = 0,
        source_frame_id: int | None = None,
        view_variant: str | Sequence[str] = "raw",
        plate_side: int | Sequence[int] = 0,
    ) -> OCRDecision:
        """Encode crop(s) via frame_encoder.onnx, fuse views, and step memory_step_decode.onnx."""
        del view_variant
        crops_np = _normalize_crop_views_numpy(crop_rgb)  # (V, 3, 64, 128)
        num_views = crops_np.shape[0]

        if isinstance(plate_side, Sequence) and not isinstance(plate_side, (str, bytes)):
            sides_np = np.asarray(list(plate_side), dtype=np.int64)
        else:
            sides_np = np.full((num_views,), int(plate_side), dtype=np.int64)

        z_out, q_out, v_out = self.enc_session.run(
            None, {"crops": crops_np, "plate_sides": sides_np}
        )

        if source_frame_id is None:
            frame_key = (int(camera_id), self._auto_frame_counter)
            self._auto_frame_counter -= 1
        else:
            frame_key = (int(camera_id), int(source_frame_id))

        if self._current_frame_key == frame_key:
            self._current_z_views.append(z_out)
            self._current_q_views.append(q_out)
            self._current_v_views.append(v_out)
        else:
            self._committed_state = self._active_state
            self._current_frame_key = frame_key
            self._current_frame_ts = float(timestamp_s)
            self._current_z_views = [z_out]
            self._current_q_views = [q_out]
            self._current_v_views = [v_out]
            self.num_frames_seen += 1

        all_z = np.concatenate(self._current_z_views, axis=0)
        all_q = np.concatenate(self._current_q_views, axis=0)
        all_v = np.concatenate(self._current_v_views, axis=0)

        z_s, q_s, v_s = fuse_views_numpy(all_z, all_q, all_v)

        mf, mm, mc, lts = self._committed_state
        ts_arr = np.array([self._current_frame_ts], dtype=np.float32)

        outputs = self.mem_session.run(
            None,
            {
                "mem_feat": mf,
                "mem_mass": mm,
                "mem_conflict": mc,
                "last_timestamp_s": lts,
                "z": z_s[None, ...],
                "slot_quality": q_s[None, ...],
                "frame_valid": v_s[None, ...],
                "timestamp_s": ts_arr,
            },
        )
        new_mf, new_mm, new_mc, new_lts, s_log, pt_log, sr_pred, st_log = outputs
        self._active_state = (new_mf, new_mm, new_mc, new_lts)
        self._last_readout = (s_log, pt_log, sr_pred, st_log)
        return self.state()

    def _build_decision(self, is_final: bool) -> OCRDecision:
        mf, mm, mc, _ = self._active_state
        if self._last_readout is None:
            s_log = np.zeros((1, NUM_SLOTS, self.vocab.size), dtype=np.float32)
            pt_log = np.zeros((1, len(self.vocab.plate_types)), dtype=np.float32)
            sr_pred = np.zeros((1, NUM_SLOTS), dtype=np.float32)
            st_log = np.zeros((1, 3), dtype=np.float32)
        else:
            s_log, pt_log, sr_pred, st_log = self._last_readout

        return build_decision_from_arrays(
            slot_logits=s_log[0],
            plate_type_logits=pt_log[0],
            slot_readability_pred=sr_pred[0],
            status_logits=st_log[0],
            mem_mass=mm[0],
            mem_conflict=mc[0],
            num_frames_seen=self.num_frames_seen,
            vocab=self.vocab,
            config=self.config,
            grammar=self.grammar,
            grammar_checker=self.grammar_checker,
            is_final=is_final,
        )

    def state(self) -> OCRDecision:
        return self._build_decision(is_final=False)

    def finish(self) -> OCRDecision:
        return self._build_decision(is_final=True)


@torch.no_grad()
def verify_onnx_parity(
    model: MFJPPlateOCR,
    onnx_dir: str | Path,
    providers: Sequence[str | tuple[str, dict[str, Any]]] | None = None,
    num_steps: int = 6,
    seed: int = 123,
) -> dict[str, Any]:
    """Compare PyTorch vs ONNX Runtime outputs across multi-step rollout (< 1e-4 max_abs_diff)."""
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        model_eval = copy.deepcopy(model).cpu().eval()
        vocab = PlateVocab()
        pt_sess = MFPlateOCRSession(model_eval, vocab=vocab, device="cpu")
        ort_sess = OrtMFPlateOCRSession(
            onnx_dir=onnx_dir, vocab=vocab, providers=providers, use_tf32=False
        )

        rng = np.random.default_rng(seed)
        max_enc_diff = 0.0
        max_feat_diff = 0.0
        max_mass_diff = 0.0
        max_logits_diff = 0.0

        for step in range(num_steps):
            crop = rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
            side = int(step % 3)
            ts = float(step) * 0.15

            # Direct encoder check
            crop_f = (crop.transpose(2, 0, 1)[None, ...].astype(np.float32)) / 255.0
            enc_pt = model_eval.encode_frames(
                torch.from_numpy(crop_f), torch.tensor([side], dtype=torch.long)
            )
            z_ort, q_ort, v_ort = ort_sess.enc_session.run(
                None,
                {"crops": crop_f, "plate_sides": np.array([side], dtype=np.int64)},
            )
            max_enc_diff = max(
                max_enc_diff,
                float(np.abs(enc_pt.z.numpy() - z_ort).max()),
                float(np.abs(enc_pt.slot_quality.numpy() - q_ort).max()),
                float(np.abs(enc_pt.frame_valid.numpy() - v_ort).max()),
            )

            # Session step check
            pt_sess.push(crop, timestamp_s=ts, camera_id=0, source_frame_id=step, plate_side=side)
            ort_sess.push(crop, timestamp_s=ts, camera_id=0, source_frame_id=step, plate_side=side)

            pt_st = pt_sess.memory_state
            pt_ro = model_eval.decode_memory(pt_st.mem_feat, pt_st.mem_mass, pt_st.mem_conflict)
            ort_mf, ort_mm, _, _ = ort_sess._active_state
            assert ort_sess._last_readout is not None
            ort_slog = ort_sess._last_readout[0]

            max_feat_diff = max(max_feat_diff, float(np.abs(pt_st.mem_feat.numpy() - ort_mf).max()))
            max_mass_diff = max(max_mass_diff, float(np.abs(pt_st.mem_mass.numpy() - ort_mm).max()))
            max_logits_diff = max(
                max_logits_diff, float(np.abs(pt_ro.slot_logits.numpy() - ort_slog).max())
            )

        overall_max = max(max_enc_diff, max_feat_diff, max_mass_diff, max_logits_diff)
        return {
            "providers": ort_sess.active_providers,
            "max_encoder_abs_diff": max_enc_diff,
            "max_mem_feat_abs_diff": max_feat_diff,
            "max_mem_mass_abs_diff": max_mass_diff,
            "max_slot_logits_abs_diff": max_logits_diff,
            "max_abs_diff": overall_max,
        }
    finally:
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
