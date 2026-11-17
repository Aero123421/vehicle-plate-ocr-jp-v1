"""Weak-CPU optimization suite: Nano distillation, INT8 ONNX, pre-encoder pixel gate, and adaptive saturation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from vpocr_ocr.dataset import collate_plate_sequences
from vpocr_ocr.losses import LossWeights, MFPlateLoss
from vpocr_ocr.model import MFJPPlateOCR
from vpocr_ocr.onnx_export import _normalize_crop_views_numpy, fuse_views_numpy
from vpocr_ocr.session import OCRDecision, SessionConfig, build_decision_from_arrays
from vpocr_ocr.trainer import _make_stage1_single_frame_batch, _move_batch_to_device, save_checkpoint
from vpocr_ocr.vocab import JPPlateGrammar, NUM_SLOTS, PlateVocab


@dataclass
class EdgeCPUConfig:
    """Runtime settings for weak edge CPUs (e.g. 1-2 core Atom / N100 / Cortex-A76)."""

    intra_op_threads: int = 1
    inter_op_threads: int = 1
    enable_pregate: bool = True
    enable_saturation_skip: bool = True
    whiteout_mean_thresh: float = 247.0
    whiteout_std_thresh: float = 7.5
    blackout_mean_thresh: float = 9.0
    blackout_std_thresh: float = 5.5
    flat_std_thresh: float = 3.2
    static_diff_thresh: float = 2.2
    static_skip_min_mass: float = 0.75
    sentinel_stride: int = 3
    scene_change_diff_thresh: float = 25.0


class FastPixelPreGate:
    """Microsecond-level NumPy pixel gate rejecting empty/whiteout/static-duplicate crops before CNN."""

    def __init__(self, config: EdgeCPUConfig | None = None) -> None:
        self.config = config or EdgeCPUConfig()

    def check_garbage_crop(self, crop_u8: np.ndarray) -> bool:
        """Return True if crop is featureless whiteout, blackout, or flat solid color."""
        # Downsample 2x stride slice for <0.015ms statistics
        sub = crop_u8[::2, ::2].astype(np.float32)
        mean_val = float(sub.mean())
        std_val = float(sub.std())

        if std_val <= self.config.flat_std_thresh:
            return True
        if mean_val >= self.config.whiteout_mean_thresh and std_val <= self.config.whiteout_std_thresh:
            return True
        if mean_val <= self.config.blackout_mean_thresh and std_val <= self.config.blackout_std_thresh:
            return True
        return False

    @staticmethod
    def mean_abs_diff(crop_a_u8: np.ndarray, crop_b_u8: np.ndarray) -> float:
        """Fast strided L1 pixel difference in [0, 255]."""
        sa = crop_a_u8[::2, ::2].astype(np.int16)
        sb = crop_b_u8[::2, ::2].astype(np.int16)
        return float(np.abs(sa - sb).mean())

    @staticmethod
    def coarse_grid_diff(crop_a_u8: np.ndarray, crop_b_u8: np.ndarray) -> float:
        """4x8 spatial block-mean L1 difference immune to 2-4px detector BBox subpixel jitter."""
        ga = crop_a_u8.reshape(4, 16, 8, 16, 3).mean(axis=(1, 3))
        gb = crop_b_u8.reshape(4, 16, 8, 16, 3).mean(axis=(1, 3))
        return float(np.abs(ga - gb).mean())


class AdaptiveEdgeOCRSession:
    """Weak-CPU optimized ONNX Runtime streaming session with pre-gate & saturation sentinel skipping."""

    def __init__(
        self,
        onnx_dir: str | Path,
        vocab: PlateVocab | None = None,
        session_config: SessionConfig | None = None,
        edge_config: EdgeCPUConfig | None = None,
        use_int8: bool = False,
    ) -> None:
        self.onnx_dir = Path(onnx_dir)
        self.vocab = vocab or PlateVocab()
        self.session_config = session_config or SessionConfig()
        self.edge_config = edge_config or EdgeCPUConfig()
        self.grammar = JPPlateGrammar(self.vocab)
        self.pregate = FastPixelPreGate(self.edge_config)

        enc_name = "frame_encoder.int8.onnx" if use_int8 else "frame_encoder.onnx"
        mem_name = "memory_step_decode.int8.onnx" if use_int8 else "memory_step_decode.onnx"
        enc_path = self.onnx_dir / enc_name
        mem_path = self.onnx_dir / mem_name
        if not enc_path.is_file():
            enc_path = self.onnx_dir / "frame_encoder.onnx"
        if not mem_path.is_file():
            mem_path = self.onnx_dir / "memory_step_decode.onnx"

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, int(self.edge_config.intra_op_threads))
        sess_opts.inter_op_num_threads = max(1, int(self.edge_config.inter_op_threads))
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.enc_sess = ort.InferenceSession(
            str(enc_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self.mem_sess = ort.InferenceSession(
            str(mem_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )

        # Infer latent dimension D from memory input shape [batch, 12, D]
        mem_shape = self.mem_sess.get_inputs()[0].shape
        self.d_model = int(mem_shape[2]) if isinstance(mem_shape[2], int) else 96
        self.reset()

    def reset(self) -> None:
        """Reset streaming track state and telemetry counters."""
        self.mem_feat = np.zeros((1, NUM_SLOTS, self.d_model), dtype=np.float32)
        self.mem_mass = np.zeros((1, NUM_SLOTS), dtype=np.float32)
        self.mem_conflict = np.zeros((1, NUM_SLOTS), dtype=np.float32)
        self.last_timestamp_s = np.array([-1.0], dtype=np.float32)

        self._last_slot_logits = np.zeros((NUM_SLOTS, self.vocab.size), dtype=np.float32)
        self._last_type_logits = np.zeros((5,), dtype=np.float32)
        self._last_readability = np.zeros((NUM_SLOTS,), dtype=np.float32)
        self._last_status_logits = np.array([0.0, 2.0, -2.0], dtype=np.float32)
        self._last_decision: OCRDecision | None = None
        self._last_encoded_crop_u8: np.ndarray | None = None
        self._prev_pushed_crop_u8: np.ndarray | None = None
        self._sentinel_counter: int = 0

        self.frames_pushed: int = 0
        self.frames_encoded_cnn: int = 0
        self.frames_skipped_pregate: int = 0
        self.frames_skipped_saturated: int = 0

    def _to_u8_hwc(self, crop_rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(crop_rgb)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            scale = 255.0 if arr.max() <= 1.5 else 1.0
            arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
        return arr

    def push(
        self,
        crop_rgb: np.ndarray,
        timestamp_s: float,
        plate_side: int = 0,
    ) -> OCRDecision:
        """Process incoming frame crop with fast pre-gating and adaptive saturation skipping."""
        self.frames_pushed += 1
        crop_u8 = self._to_u8_hwc(crop_rgb)

        # 1. Fast microsecond pre-gate
        if self.edge_config.enable_pregate:
            if self.pregate.check_garbage_crop(crop_u8):
                self.frames_skipped_pregate += 1
                return self.state(is_final=False)

            if self._last_encoded_crop_u8 is not None:
                static_diff = self.pregate.mean_abs_diff(crop_u8, self._last_encoded_crop_u8)
                # Static vehicle duplicate skip once moderate mass is accumulated
                if (
                    static_diff < self.edge_config.static_diff_thresh
                    and float(np.min(self.mem_mass[:, [0, 4, 7, 11]])) >= self.edge_config.static_skip_min_mass
                ):
                    self._prev_pushed_crop_u8 = crop_u8
                    self.frames_skipped_pregate += 1
                    return self.state(is_final=False)

        coarse_diff = (
            self.pregate.coarse_grid_diff(crop_u8, self._prev_pushed_crop_u8)
            if self._prev_pushed_crop_u8 is not None
            else 255.0
        )
        self._prev_pushed_crop_u8 = crop_u8

        # 2. Adaptive sentinel stride after saturation (wakes immediately on coarse scene change!)
        if (
            self.edge_config.enable_saturation_skip
            and self._last_decision is not None
            and self._last_decision.is_saturated
            and coarse_diff < self.edge_config.scene_change_diff_thresh
        ):
            self._sentinel_counter += 1
            if (self._sentinel_counter % max(1, self.edge_config.sentinel_stride)) != 0:
                self.frames_skipped_saturated += 1
                return self.state(is_final=False)

        # 3. Full ONNX frame encoder + latent EvidenceMemory update
        views_f = _normalize_crop_views_numpy(crop_u8)
        sides_arr = np.full((views_f.shape[0],), int(plate_side), dtype=np.int64)
        z_v, q_v, v_v = self.enc_sess.run(None, {"crops": views_f, "plate_sides": sides_arr})
        z_step, q_step, v_step = fuse_views_numpy(z_v, q_v, v_v)

        mem_outs = self.mem_sess.run(
            None,
            {
                "mem_feat": self.mem_feat,
                "mem_mass": self.mem_mass,
                "mem_conflict": self.mem_conflict,
                "last_timestamp_s": self.last_timestamp_s,
                "z": z_step[None, ...],
                "slot_quality": q_step[None, ...],
                "frame_valid": v_step[None, ...],
                "timestamp_s": np.array([float(timestamp_s)], dtype=np.float32),
            },
        )
        (
            self.mem_feat,
            self.mem_mass,
            self.mem_conflict,
            self.last_timestamp_s,
            slot_logits,
            type_logits,
            read_pred,
            status_logits,
        ) = mem_outs

        self._last_slot_logits = slot_logits[0]
        self._last_type_logits = type_logits[0]
        self._last_readability = read_pred[0]
        self._last_status_logits = status_logits[0]
        self._last_encoded_crop_u8 = crop_u8
        self.frames_encoded_cnn += 1
        self._last_decision = self.state(is_final=False)
        return self._last_decision

    def state(self, is_final: bool = False) -> OCRDecision:
        """Compute current structured OCRDecision."""
        dec = build_decision_from_arrays(
            slot_logits=self._last_slot_logits,
            plate_type_logits=self._last_type_logits,
            slot_readability_pred=self._last_readability,
            status_logits=self._last_status_logits,
            mem_mass=self.mem_mass[0],
            mem_conflict=self.mem_conflict[0],
            num_frames_seen=self.frames_pushed,
            vocab=self.vocab,
            config=self.session_config,
            grammar=self.grammar,
            is_final=is_final,
        )
        self._last_decision = dec
        return dec

    def finish(self) -> OCRDecision:
        """Finalize vehicle pass decision."""
        return self.state(is_final=True)


def quantize_onnx_for_cpu(onnx_dir: str | Path) -> dict[str, Path]:
    """Produce dynamic INT8 quantized ONNX models alongside FP32 models for weak CPUs."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    base_dir = Path(onnx_dir)
    enc_fp32 = base_dir / "frame_encoder.onnx"
    mem_fp32 = base_dir / "memory_step_decode.onnx"
    enc_int8 = base_dir / "frame_encoder.int8.onnx"
    mem_int8 = base_dir / "memory_step_decode.int8.onnx"

    # Quantize Linear/MatMul ops cleanly to QUInt8 (ConvInteger on CPU lacks kernel on some x86 builds unless MatMul-only)
    quantize_dynamic(
        model_input=str(enc_fp32),
        model_output=str(enc_int8),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QUInt8,
    )
    quantize_dynamic(
        model_input=str(mem_fp32),
        model_output=str(mem_int8),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QUInt8,
    )
    return {"encoder_int8": enc_int8, "memory_int8": mem_int8}


@dataclass
class DistillConfig:
    """Configuration for Teacher -> Student (Base -> Nano) knowledge distillation."""

    batch_size: int = 48
    stage1_steps: int = 800
    stage2_steps: int = 1600
    lr_stage1: float = 2.0e-3
    lr_stage2: float = 1.0e-3
    temperature: float = 2.0
    kd_slot_weight: float = 1.25
    kd_quality_weight: float = 0.60
    num_workers: int = 8
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    log_every: int = 200
    seed: int = 42


class DistillationTrainer:
    """Distills a trained Teacher (`MF-JPPlateOCR-Base`) into a compact Student (`Nano` / `Small`)."""

    def __init__(
        self,
        teacher: MFJPPlateOCR,
        student: MFJPPlateOCR,
        vocab: PlateVocab | None = None,
        config: DistillConfig | None = None,
        loss_weights: LossWeights | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.vocab = vocab or PlateVocab()
        self.config = config or DistillConfig()
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.teacher = teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student = student.to(self.device)
        self.criterion = MFPlateLoss(
            vocab_size=self.vocab.size,
            pad_id=self.vocab.pad_id,
            weights=loss_weights,
        ).to(self.device)

    def _autocast_ctx(self) -> Any:
        if self.config.use_amp and self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _kd_kl_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
        t_prob = F.softmax(teacher_logits.float() / temperature, dim=-1)
        kl = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=-1) * (temperature * temperature)
        if mask is not None:
            m = mask.float()
            return (kl * m).sum() / m.sum().clamp_min(1.0)
        return kl.mean()

    def _run_stage(
        self,
        dataset: Dataset[dict[str, Any]],
        num_steps: int,
        max_lr: float,
        is_stage1: bool,
    ) -> list[dict[str, float]]:
        if num_steps <= 0:
            return []
        self.student.train()
        loader = DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, max(1, len(dataset))),  # type: ignore[arg-type]
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_plate_sequences,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(self.config.num_workers > 0),
            multiprocessing_context=("forkserver" if self.config.num_workers > 0 else None),
        )
        optimizer = AdamW(self.student.parameters(), lr=max_lr, weight_decay=1e-4)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max(2, num_steps),
            pct_start=0.15,
            anneal_strategy="cos",
        )
        temp = self.config.temperature
        history: list[dict[str, float]] = []
        step = 0

        while step < num_steps:
            for raw_batch in loader:
                if step >= num_steps:
                    break
                batch = _move_batch_to_device(raw_batch, self.device)
                if is_stage1:
                    batch = _make_stage1_single_frame_batch(batch, max_frames=128)

                optimizer.zero_grad(set_to_none=True)
                with self._autocast_ctx():
                    with torch.no_grad():
                        t_out = self.teacher.forward_sequence(
                            crops=batch["crops"],
                            timestamps_s=batch["timestamps_s"],
                            plate_sides=batch["plate_sides"],
                            frame_pad_mask=batch["frame_pad_mask"],
                            camera_ids=batch["camera_ids"],
                            source_frame_ids=batch["source_frame_ids"],
                        )
                        t_aux = self.teacher.decode_slot_features(t_out.frame_z)

                    s_out = self.student.forward_sequence(
                        crops=batch["crops"],
                        timestamps_s=batch["timestamps_s"],
                        plate_sides=batch["plate_sides"],
                        frame_pad_mask=batch["frame_pad_mask"],
                        camera_ids=batch["camera_ids"],
                        source_frame_ids=batch["source_frame_ids"],
                    )
                    s_aux = self.student.decode_slot_features(s_out.frame_z)

                    gt_loss, metrics = self.criterion(
                        self.student, s_out, batch, compute_garbage_invariance=True
                    )

                    non_mixed = (batch["target_status"] != 2) & batch["has_target_plate"]
                    kd_seq = self._kd_kl_loss(
                        s_out.slot_logits,
                        t_out.slot_logits,
                        temperature=temp,
                        mask=non_mixed.unsqueeze(1).expand(-1, NUM_SLOTS),
                    )
                    non_pad = (~batch["frame_pad_mask"]).unsqueeze(-1).expand_as(s_out.frame_slot_quality)
                    kd_aux = self._kd_kl_loss(s_aux, t_aux, temperature=temp, mask=non_pad)

                    kd_qual = (
                        (s_out.frame_slot_quality.float() - t_out.frame_slot_quality.float()).pow(2).mean()
                        + (s_out.frame_valid.float() - t_out.frame_valid.float()).pow(2).mean()
                    )

                    loss = (
                        gt_loss
                        + self.config.kd_slot_weight * (0.6 * kd_seq + 0.4 * kd_aux)
                        + self.config.kd_quality_weight * kd_qual
                    )

                loss.backward()
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip_norm)
                optimizer.step()
                if step + 1 < max(2, num_steps):
                    scheduler.step()

                metrics["kd_seq"] = float(kd_seq.detach().item())
                metrics["total_distill_loss"] = float(loss.detach().item())
                history.append(metrics)
                step += 1
                if self.config.log_every > 0 and (
                    step % self.config.log_every == 0 or step == num_steps
                ):
                    st_label = "KD-S1" if is_stage1 else "KD-S2"
                    print(
                        f"    [{st_label} {step}/{num_steps}] "
                        f"distill_loss={metrics['total_distill_loss']:.4f} "
                        f"(gt={metrics['loss_total']:.4f}, kd_seq={metrics['kd_seq']:.4f})",
                        flush=True,
                    )

        return history

    def fit(
        self,
        train_dataset: Dataset[dict[str, Any]],
        stage1_dataset: Dataset[dict[str, Any]] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        s1_ds = stage1_dataset if stage1_dataset is not None else train_dataset
        h1 = self._run_stage(s1_ds, self.config.stage1_steps, self.config.lr_stage1, is_stage1=True)
        h2 = self._run_stage(train_dataset, self.config.stage2_steps, self.config.lr_stage2, is_stage1=False)
        summary = {
            "student_params": self.student.num_parameters,
            "teacher_params": self.teacher.num_parameters,
            "stage1_final_loss": h1[-1]["total_distill_loss"] if h1 else 0.0,
            "stage2_final_loss": h2[-1]["total_distill_loss"] if h2 else 0.0,
        }
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, self.student, metrics=summary)
        return summary


def benchmark_weak_cpu_pass(
    onnx_dir: str | Path,
    sequences: list[dict[str, Any]],
    intra_threads: int = 1,
    use_int8: bool = False,
    enable_adaptive: bool = True,
) -> dict[str, float]:
    """Profile end-to-end 1-thread / 2-thread weak-CPU vehicle-pass latency and accuracy."""
    edge_cfg = EdgeCPUConfig(
        intra_op_threads=intra_threads,
        inter_op_threads=1,
        enable_pregate=enable_adaptive,
        enable_saturation_skip=enable_adaptive,
    )
    sess = AdaptiveEdgeOCRSession(
        onnx_dir=onnx_dir,
        edge_config=edge_cfg,
        use_int8=use_int8,
    )

    # Warmup 1 sequence
    if sequences:
        sess.reset()
        w_seq = sequences[0]
        for t in range(min(4, w_seq["crops"].shape[0])):
            sess.push(w_seq["crops"][t], float(w_seq["timestamps_s"][t]), int(w_seq["plate_sides"][t]))
        sess.finish()

    total_frames = 0
    total_cnn_calls = 0
    total_pregate_skips = 0
    total_sat_skips = 0
    exact_matches = 0
    readable_count = 0

    t0 = time.perf_counter()
    for seq in sequences:
        sess.reset()
        crops = seq["crops"]  # (T, 3, 64, 128) or (T, 64, 128, 3)
        ts = seq["timestamps_s"]
        sides = seq["plate_sides"]
        for t in range(crops.shape[0]):
            sess.push(crops[t], float(ts[t]), int(sides[t]))
        dec = sess.finish()

        total_frames += sess.frames_pushed
        total_cnn_calls += sess.frames_encoded_cnn
        total_pregate_skips += sess.frames_skipped_pregate
        total_sat_skips += sess.frames_skipped_saturated

        gt = seq.get("plate_text")
        if gt is not None and seq.get("sequence_target_status", "READABLE") == "READABLE":
            readable_count += 1
            if dec.plate_text is not None and dec.plate_text.ocr_string == gt.ocr_string:
                exact_matches += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    n_seqs = max(1, len(sequences))
    return {
        "intra_threads": float(intra_threads),
        "use_int8": float(use_int8),
        "enable_adaptive": float(enable_adaptive),
        "ms_per_frame_effective": elapsed_ms / max(1, total_frames),
        "ms_per_pass_total": elapsed_ms / n_seqs,
        "ms_per_cnn_call": elapsed_ms / max(1, total_cnn_calls),
        "cnn_call_ratio": total_cnn_calls / max(1, total_frames),
        "pregate_skip_ratio": total_pregate_skips / max(1, total_frames),
        "saturation_skip_ratio": total_sat_skips / max(1, total_frames),
        "exact_match": exact_matches / max(1, readable_count),
    }
