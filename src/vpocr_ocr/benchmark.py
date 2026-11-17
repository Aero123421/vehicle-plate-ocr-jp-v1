"""Comprehensive 5-suite benchmark engine for MF-JPPlateOCR vs single-frame & pixel baselines."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader

from vpocr_ocr.baselines import (
    EvalPrediction,
    MultiFrameEvaluator,
    PixelAverageEvaluator,
    SingleFrameBestQualityEvaluator,
    SingleFrameLastEvaluator,
)
from vpocr_ocr.dataset import MicroPlateSequenceDataset, collate_plate_sequences
from vpocr_ocr.model import MFJPPlateOCR
from vpocr_ocr.onnx_export import OrtMFPlateOCRSession, export_ocr_to_onnx, verify_onnx_parity
from vpocr_ocr.session import MFPlateOCRSession, SessionConfig
from vpocr_ocr.vocab import JPPlateGrammar, PlateVocab, canonicalize_region
from vpocr_synth.plate.models import PlateText


@dataclass
class BenchmarkConfig:
    """Configuration for BenchmarkEngine evaluation suites."""

    num_sequences: int = 64
    batch_size: int = 32
    k_frames: tuple[int, ...] = (1, 3, 5, 10, 20)
    complementary_seq_len: int = 8
    clean_prefix_frames: int = 5
    appended_garbage_frames: int = 10
    latency_warmup_iters: int = 15
    latency_measure_iters: int = 60
    seed: int = 2026


def _compute_accuracy_metrics(
    preds: list[EvalPrediction],
    targets: list[PlateText | None],
    target_types: list[str],
) -> dict[str, float]:
    n = 0
    exact = 0
    reg_ok = 0
    cls_ok = 0
    kana_ok = 0
    ser_ok = 0
    ptype_ok = 0

    for pred, gt, gt_type in zip(preds, targets, target_types):
        if gt is None:
            continue
        n += 1
        if pred.plate_type == gt_type:
            ptype_ok += 1
        pt = pred.plate_text
        if pt is None:
            continue

        r_match = canonicalize_region(pt.region) == canonicalize_region(gt.region)
        c_match = pt.classification == gt.classification
        h_match = pt.hiragana == gt.hiragana
        s_match = pt.serial == gt.serial

        reg_ok += int(r_match)
        cls_ok += int(c_match)
        kana_ok += int(h_match)
        ser_ok += int(s_match)
        exact += int(r_match and c_match and h_match and s_match)

    denom = float(max(1, n))
    return {
        "count": float(n),
        "exact_match": exact / denom,
        "region_acc": reg_ok / denom,
        "classification_acc": cls_ok / denom,
        "hiragana_acc": kana_ok / denom,
        "serial_acc": ser_ok / denom,
        "plate_type_acc": ptype_ok / denom,
    }


class BenchmarkEngine:
    """Executes all 5 scientific & latency benchmark suites and renders JSON + Markdown reports."""

    def __init__(
        self,
        model: MFJPPlateOCR,
        vocab: PlateVocab | None = None,
        config: BenchmarkConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.vocab = vocab or PlateVocab()
        self.config = config or BenchmarkConfig()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.grammar = JPPlateGrammar(self.vocab)

    @torch.no_grad()
    def benchmark_k_frames_curve(
        self,
        plate_pool_size: int = 128,
    ) -> dict[str, dict[str, float]]:
        """1. Full-plate exact match & per-field accuracy at K in {1, 3, 5, 10, 20} frames."""
        max_k = max(self.config.k_frames)
        ds = MicroPlateSequenceDataset(
            num_sequences=self.config.num_sequences,
            seq_len=max_k,
            mode="degraded",
            plate_pool_size=plate_pool_size,
            vocab=self.vocab,
            seed=self.config.seed + 10,
        )
        loader = DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_plate_sequences,
        )
        mf_eval = MultiFrameEvaluator(self.model, self.vocab, self.grammar, constrained=True)

        preds_by_k: dict[int, list[EvalPrediction]] = {k: [] for k in self.config.k_frames}
        all_targets: list[PlateText | None] = []
        all_types: list[str] = []

        for batch in loader:
            crops = batch["crops"].to(self.device)
            ts = batch["timestamps_s"].to(self.device)
            sides = batch["plate_sides"].to(self.device)
            all_targets.extend(batch["raw_plate_texts"])
            all_types.extend(batch["raw_plate_types"])

            for k in self.config.k_frames:
                k_eff = min(k, crops.shape[1])
                preds = mf_eval.evaluate_batch(
                    crops=crops[:, :k_eff],
                    timestamps_s=ts[:, :k_eff],
                    plate_sides=sides[:, :k_eff],
                )
                preds_by_k[k].extend(preds)

        return {
            f"K={k}": _compute_accuracy_metrics(preds_by_k[k], all_targets, all_types)
            for k in self.config.k_frames
        }

    @torch.no_grad()
    def benchmark_complementary_glare(
        self,
        plate_pool_size: int = 128,
    ) -> dict[str, dict[str, float]]:
        """2. Complementary Partial-Glare Benchmark (lower glared early, upper glared late)."""
        ds = MicroPlateSequenceDataset(
            num_sequences=self.config.num_sequences,
            seq_len=self.config.complementary_seq_len,
            mode="complementary_glare",
            plate_pool_size=plate_pool_size,
            vocab=self.vocab,
            seed=self.config.seed + 20,
        )
        loader = DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_plate_sequences,
        )

        evaluators = {
            "SingleFrame-Last": SingleFrameLastEvaluator(self.model, self.vocab, self.grammar),
            "SingleFrame-Best": SingleFrameBestQualityEvaluator(
                self.model, self.vocab, self.grammar
            ),
            "PixelAverage": PixelAverageEvaluator(self.model, self.vocab, self.grammar),
            "MF-JPPlateOCR": MultiFrameEvaluator(self.model, self.vocab, self.grammar),
        }
        preds_map: dict[str, list[EvalPrediction]] = {name: [] for name in evaluators}
        all_targets: list[PlateText | None] = []
        all_types: list[str] = []

        for batch in loader:
            crops = batch["crops"].to(self.device)
            ts = batch["timestamps_s"].to(self.device)
            sides = batch["plate_sides"].to(self.device)
            all_targets.extend(batch["raw_plate_texts"])
            all_types.extend(batch["raw_plate_types"])

            for name, ev in evaluators.items():
                preds_map[name].extend(ev.evaluate_batch(crops, ts, sides))

        return {
            name: _compute_accuracy_metrics(preds_map[name], all_targets, all_types)
            for name in evaluators
        }

    @torch.no_grad()
    def benchmark_garbage_stability(
        self,
        plate_pool_size: int = 128,
    ) -> dict[str, float]:
        """3. Garbage-Frame Appending Stability (5 clean frames vs 5 clean + 10 garbage frames)."""
        n_clean = self.config.clean_prefix_frames
        n_garb = self.config.appended_garbage_frames
        ds = MicroPlateSequenceDataset(
            num_sequences=self.config.num_sequences,
            seq_len=n_clean,
            mode="clean",
            plate_pool_size=plate_pool_size,
            vocab=self.vocab,
            seed=self.config.seed + 30,
        )
        loader = DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_plate_sequences,
        )

        mf_eval = MultiFrameEvaluator(self.model, self.vocab, self.grammar)
        preds_before: list[EvalPrediction] = []
        preds_after: list[EvalPrediction] = []
        all_targets: list[PlateText | None] = []
        all_types: list[str] = []

        max_feat_drift = 0.0
        garb_valid_vals: list[float] = []
        rng = np.random.default_rng(self.config.seed + 300)

        for batch in loader:
            crops_clean = batch["crops"].to(self.device)  # (B, 5, 3, 64, 128)
            ts_clean = batch["timestamps_s"].to(self.device)
            sides_clean = batch["plate_sides"].to(self.device)
            bsz = crops_clean.shape[0]

            all_targets.extend(batch["raw_plate_texts"])
            all_types.extend(batch["raw_plate_types"])

            #Synthesize 10 realistic numpy garbage crops per sample (whiteout/noise/side-smear)
            garb_np = np.stack(
                [
                    np.stack(
                        [
                            MicroPlateSequenceDataset.make_garbage_crop(rng, kind=g)
                            for g in range(n_garb)
                        ],
                        axis=0,
                    )
                    for _ in range(bsz)
                ],
                axis=0,
            )  # (B, 10, 64, 128, 3) uint8
            garb_t = (
                torch.from_numpy(garb_np).permute(0, 1, 4, 2, 3).to(self.device, torch.float32)
                / 255.0
            )
            garb_sides = torch.full((bsz, n_garb), 2, dtype=torch.long, device=self.device)
            garb_ts = ts_clean[:, -1:] + (
                torch.arange(1, n_garb + 1, device=self.device, dtype=torch.float32).unsqueeze(0)
                * 0.12
            )

            out_before = self.model.forward_sequence(crops_clean, ts_clean, sides_clean)

            crops_comb = torch.cat([crops_clean, garb_t], dim=1)
            ts_comb = torch.cat([ts_clean, garb_ts], dim=1)
            sides_comb = torch.cat([sides_clean, garb_sides], dim=1)
            out_after = self.model.forward_sequence(crops_comb, ts_comb, sides_comb)

            drift = float(
                (out_after.memory.mem_feat - out_before.memory.mem_feat).abs().max().item()
            )
            max_feat_drift = max(max_feat_drift, drift)
            garb_valid_vals.append(float(out_after.frame_valid[:, n_clean:].mean().item()))

            preds_before.extend(mf_eval.evaluate_batch(crops_clean, ts_clean, sides_clean))
            preds_after.extend(mf_eval.evaluate_batch(crops_comb, ts_comb, sides_comb))

        acc_before = _compute_accuracy_metrics(preds_before, all_targets, all_types)
        acc_after = _compute_accuracy_metrics(preds_after, all_targets, all_types)
        flips = sum(
            1 for pb, pa in zip(preds_before, preds_after) if pb.slot_ids != pa.slot_ids
        )
        flip_rate = flips / float(max(1, len(preds_before)))

        return {
            "clean_5_exact_match": acc_before["exact_match"],
            "clean_5_plus_10_garbage_exact_match": acc_after["exact_match"],
            "prediction_flip_rate": flip_rate,
            "mean_garbage_frame_valid": float(np.mean(garb_valid_vals)) if garb_valid_vals else 0.0,
            "max_mem_feat_abs_drift": max_feat_drift,
        }

    @torch.no_grad()
    def benchmark_abstention_and_conflict(
        self,
        plate_pool_size: int = 128,
    ) -> dict[str, float]:
        """4. Abstention (persistent 1-slot occlusion) & Conflict Detection (foreign plate splice)."""
        n_each = max(16, self.config.num_sequences // 2)
        ds_clean = MicroPlateSequenceDataset(
            num_sequences=n_each,
            seq_len=6,
            mode="clean",
            plate_pool_size=plate_pool_size,
            vocab=self.vocab,
            seed=self.config.seed + 40,
        )
        ds_occ = MicroPlateSequenceDataset(
            num_sequences=n_each,
            seq_len=6,
            mode="persistent_occlusion",
            plate_pool_size=plate_pool_size,
            vocab=self.vocab,
            seed=self.config.seed + 41,
        )
        ds_mixed = MicroPlateSequenceDataset(
            num_sequences=n_each,
            seq_len=6,
            mode="track_mixed",
            plate_pool_size=plate_pool_size,
            vocab=self.vocab,
            seed=self.config.seed + 42,
        )

        sess = MFPlateOCRSession(
            self.model,
            vocab=self.vocab,
            config=SessionConfig(),
            device=self.device,
        )

        def _run_ds(ds: MicroPlateSequenceDataset) -> list[tuple[Any, PlateText | None]]:
            out_list = []
            for i in range(len(ds)):
                item = ds[i]
                sess.reset()
                crops_chw = item["crops"]
                ts = item["timestamps_s"]
                sides = item["plate_sides"]
                for t in range(crops_chw.shape[0]):
                    sess.push(
                        torch.from_numpy(crops_chw[t]),
                        timestamp_s=float(ts[t]),
                        camera_id=0,
                        source_frame_id=t,
                        plate_side=int(sides[t]),
                    )
                dec = sess.finish()
                out_list.append((dec, item["plate_text"]))
            return out_list

        res_clean = _run_ds(ds_clean)
        res_occ = _run_ds(ds_occ)
        res_mixed = _run_ds(ds_mixed)

        unreadable_hits = sum(
            1 for dec, _ in res_occ if dec.status in ("UNREADABLE", "CONFLICT", "TRACK_MIXED")
        )
        unreadable_strict_hits = sum(1 for dec, _ in res_occ if dec.status == "UNREADABLE")
        track_mixed_hits = sum(
            1 for dec, _ in res_mixed if dec.status in ("TRACK_MIXED", "CONFLICT")
        )

        # Selective ACCEPTED precision & coverage across combined pool
        combined = (
            [(dec, gt, "READABLE") for dec, gt in res_clean]
            + [(dec, gt, "UNREADABLE") for dec, gt in res_occ]
            + [(dec, gt, "TRACK_MIXED") for dec, gt in res_mixed]
        )
        accepted = [(dec, gt, kind) for dec, gt, kind in combined if dec.status == "ACCEPTED"]
        accepted_correct = sum(
            1
            for dec, gt, kind in accepted
            if kind == "READABLE" and gt is not None and dec.plate_text == gt
        )
        clean_accepted = sum(1 for dec, _ in res_clean if dec.status == "ACCEPTED")

        return {
            "unreadable_abstention_recall": unreadable_hits / float(len(res_occ)),
            "unreadable_exact_label_recall": unreadable_strict_hits / float(len(res_occ)),
            "track_mixed_detection_recall": track_mixed_hits / float(len(res_mixed)),
            "clean_accepted_coverage": clean_accepted / float(len(res_clean)),
            "overall_accepted_coverage": len(accepted) / float(len(combined)),
            "accepted_selective_precision": (
                accepted_correct / float(len(accepted)) if accepted else 1.0
            ),
            "mean_clean_conflict_score": float(
                np.mean([dec.conflict_score for dec, _ in res_clean])
            ),
            "mean_track_mixed_conflict_score": float(
                np.mean([dec.conflict_score for dec, _ in res_mixed])
            ),
        }

    @torch.no_grad()
    def benchmark_latency(
        self,
        onnx_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """5. Inference Latency (PyTorch RTX 4070 GPU, ONNX Runtime CUDA, ONNX Runtime CPU)."""
        cleanup_dir: Path | None = None
        if onnx_dir is None:
            cleanup_dir = Path(tempfile.mkdtemp(prefix="vpocr_ocr_bench_"))
            export_ocr_to_onnx(self.model, cleanup_dir)
            eff_onnx_dir = cleanup_dir
        else:
            eff_onnx_dir = Path(onnx_dir)
            if not (eff_onnx_dir / "frame_encoder.onnx").is_file():
                export_ocr_to_onnx(self.model, eff_onnx_dir)

        try:
            parity = verify_onnx_parity(
                self.model, eff_onnx_dir, providers=["CPUExecutionProvider"]
            )
            warmup = self.config.latency_warmup_iters
            iters = self.config.latency_measure_iters

            rng = np.random.default_rng(self.config.seed + 50)
            crop_np = rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)

            results: dict[str, Any] = {
                "onnx_parity_max_abs_diff": parity["max_abs_diff"],
            }

            # 1. PyTorch CUDA streaming (B=1) & batched (B=32)
            if torch.cuda.is_available():
                cuda_dev = torch.device("cuda")
                m_cuda = self.model.to(cuda_dev).eval()
                crop_cuda = torch.rand((1, 3, 64, 128), device=cuda_dev, dtype=torch.float32)
                side_cuda = torch.zeros((1,), device=cuda_dev, dtype=torch.long)
                ts_cuda = torch.tensor([0.1], device=cuda_dev, dtype=torch.float32)
                st = m_cuda.init_memory(1, device=cuda_dev)

                for i in range(warmup):
                    enc = m_cuda.encode_frames(crop_cuda, side_cuda)
                    st = m_cuda.step_memory(
                        *st, enc.z, enc.slot_quality, enc.frame_valid, ts_cuda + i * 0.1
                    )
                    _ = m_cuda.decode_memory(st.mem_feat, st.mem_mass, st.mem_conflict)
                torch.cuda.synchronize()

                t0 = time.perf_counter()
                for i in range(iters):
                    enc = m_cuda.encode_frames(crop_cuda, side_cuda)
                    st = m_cuda.step_memory(
                        *st, enc.z, enc.slot_quality, enc.frame_valid, ts_cuda + i * 0.1
                    )
                    _ = m_cuda.decode_memory(st.mem_feat, st.mem_mass, st.mem_conflict)
                torch.cuda.synchronize()
                results["pytorch_gpu_stream_ms_per_frame"] = (
                    (time.perf_counter() - t0) * 1000.0 / float(iters)
                )

                # Batched B=32 throughput
                b32 = 32
                crops_b32 = torch.rand((b32, 3, 64, 128), device=cuda_dev, dtype=torch.float32)
                sides_b32 = torch.zeros((b32,), device=cuda_dev, dtype=torch.long)
                ts_b32 = torch.full((b32,), 0.1, device=cuda_dev, dtype=torch.float32)
                st_b32 = m_cuda.init_memory(b32, device=cuda_dev)
                for _ in range(warmup):
                    enc = m_cuda.encode_frames(crops_b32, sides_b32)
                    st_b32 = m_cuda.step_memory(
                        *st_b32, enc.z, enc.slot_quality, enc.frame_valid, ts_b32
                    )
                    _ = m_cuda.decode_memory(
                        st_b32.mem_feat, st_b32.mem_mass, st_b32.mem_conflict
                    )
                torch.cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(iters):
                    enc = m_cuda.encode_frames(crops_b32, sides_b32)
                    st_b32 = m_cuda.step_memory(
                        *st_b32, enc.z, enc.slot_quality, enc.frame_valid, ts_b32
                    )
                    _ = m_cuda.decode_memory(
                        st_b32.mem_feat, st_b32.mem_mass, st_b32.mem_conflict
                    )
                torch.cuda.synchronize()
                results["pytorch_gpu_batch32_ms_per_frame"] = (
                    (time.perf_counter() - t0) * 1000.0 / float(iters * b32)
                )
                self.model.to(self.device)

            # 2. ONNX Runtime CUDAExecutionProvider (with fallback tracking)
            ort_cuda_sess = OrtMFPlateOCRSession(
                eff_onnx_dir,
                vocab=self.vocab,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            results["ort_cuda_active_provider"] = ort_cuda_sess.active_providers[0]
            for i in range(warmup):
                ort_cuda_sess.push(crop_np, timestamp_s=float(i) * 0.1, source_frame_id=i)
            ort_cuda_sess.reset()
            t0 = time.perf_counter()
            for i in range(iters):
                ort_cuda_sess.push(crop_np, timestamp_s=float(i) * 0.1, source_frame_id=i)
            results["ort_cuda_ms_per_frame"] = (
                (time.perf_counter() - t0) * 1000.0 / float(iters)
            )

            # 3. ONNX Runtime CPUExecutionProvider
            ort_cpu_sess = OrtMFPlateOCRSession(
                eff_onnx_dir,
                vocab=self.vocab,
                providers=["CPUExecutionProvider"],
            )
            for i in range(warmup):
                ort_cpu_sess.push(crop_np, timestamp_s=float(i) * 0.1, source_frame_id=i)
            ort_cpu_sess.reset()
            t0 = time.perf_counter()
            for i in range(iters):
                ort_cpu_sess.push(crop_np, timestamp_s=float(i) * 0.1, source_frame_id=i)
            results["ort_cpu_ms_per_frame"] = (
                (time.perf_counter() - t0) * 1000.0 / float(iters)
            )

            return results
        finally:
            if cleanup_dir is not None and cleanup_dir.exists():
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def run_all(
        self,
        onnx_dir: str | Path | None = None,
        plate_pool_size: int = 128,
    ) -> dict[str, Any]:
        """Run all 5 benchmark suites and return structured report dictionary."""
        return {
            "model": {
                "architecture": "MF-JPPlateOCR",
                "num_parameters": self.model.num_parameters,
                "d_model": self.model.config.d_model,
                "vocab_size": self.vocab.size,
                "device": str(self.device),
            },
            "k_frames_curve": self.benchmark_k_frames_curve(plate_pool_size=plate_pool_size),
            "complementary_glare": self.benchmark_complementary_glare(
                plate_pool_size=plate_pool_size
            ),
            "garbage_stability": self.benchmark_garbage_stability(
                plate_pool_size=plate_pool_size
            ),
            "abstention_and_conflict": self.benchmark_abstention_and_conflict(
                plate_pool_size=plate_pool_size
            ),
            "latency": self.benchmark_latency(onnx_dir=onnx_dir),
        }

    @staticmethod
    def format_markdown(report: dict[str, Any]) -> str:
        """Render complete benchmark report dictionary as GitHub-Flavored Markdown."""
        lines: list[str] = []
        m = report["model"]
        lines.append("# MF-JPPlateOCR Benchmark Report")
        lines.append("")
        lines.append(
            f"- **Parameters**: `{m['num_parameters']:,}` (`d_model={m['d_model']}`, "
            f"`vocab_size={m['vocab_size']}`)"
        )
        lines.append(f"- **Device**: `{m['device']}`")
        lines.append("")

        # 1. K-frames curve
        lines.append("## 1. Multi-Frame Evidence Accumulation (`K` Frames)")
        lines.append("")
        lines.append("| Frames `K` | Exact Match | Region | Classification | Hiragana | Serial | Plate Type |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for k_name, stats in report["k_frames_curve"].items():
            lines.append(
                f"| `{k_name}` | {stats['exact_match']*100:.1f}% | "
                f"{stats['region_acc']*100:.1f}% | {stats['classification_acc']*100:.1f}% | "
                f"{stats['hiragana_acc']*100:.1f}% | {stats['serial_acc']*100:.1f}% | "
                f"{stats['plate_type_acc']*100:.1f}% |"
            )
        lines.append("")

        # 2. Complementary glare
        lines.append("## 2. Complementary Partial-Glare Benchmark")
        lines.append("")
        lines.append("| Method | Exact Match | Region | Classification | Hiragana | Serial |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for name, stats in report["complementary_glare"].items():
            lines.append(
                f"| **{name}** | {stats['exact_match']*100:.1f}% | "
                f"{stats['region_acc']*100:.1f}% | {stats['classification_acc']*100:.1f}% | "
                f"{stats['hiragana_acc']*100:.1f}% | {stats['serial_acc']*100:.1f}% |"
            )
        lines.append("")

        # 3. Garbage stability
        g = report["garbage_stability"]
        lines.append("## 3. Garbage-Frame Appending Stability (`5` Clean vs `5` Clean + `10` Garbage)")
        lines.append("")
        lines.append(f"- **5 Clean Exact Match**: `{g['clean_5_exact_match']*100:.1f}%`")
        lines.append(
            f"- **5 Clean + 10 Garbage Exact Match**: `{g['clean_5_plus_10_garbage_exact_match']*100:.1f}%`"
        )
        lines.append(f"- **Prediction Flip Rate**: `{g['prediction_flip_rate']*100:.2f}%`")
        lines.append(f"- **Mean Garbage `frame_valid` Weight**: `{g['mean_garbage_frame_valid']:.6f}`")
        lines.append(f"- **Max Latent `mem_feat` Drift**: `{g['max_mem_feat_abs_drift']:.6e}`")
        lines.append("")

        # 4. Abstention & conflict
        ac = report["abstention_and_conflict"]
        lines.append("## 4. Abstention & Track-Mixed Conflict Detection")
        lines.append("")
        lines.append(
            f"- **Persistent 1-Slot Occlusion Abstention Recall**: `{ac['unreadable_abstention_recall']*100:.1f}%`"
        )
        lines.append(
            f"- **Spliced Foreign Plate (`TRACK_MIXED`) Recall**: `{ac['track_mixed_detection_recall']*100:.1f}%`"
        )
        lines.append(
            f"- **`ACCEPTED` Selective Precision**: `{ac['accepted_selective_precision']*100:.1f}%` "
            f"(Clean Coverage: `{ac['clean_accepted_coverage']*100:.1f}%`)"
        )
        lines.append(
            f"- **Conflict Score Separation**: Clean `{ac['mean_clean_conflict_score']:.3f}` vs "
            f"Mixed `{ac['mean_track_mixed_conflict_score']:.3f}`"
        )
        lines.append("")

        # 5. Latency
        lat = report["latency"]
        lines.append("## 5. Streaming Edge & GPU Inference Latency")
        lines.append("")
        if "pytorch_gpu_stream_ms_per_frame" in lat:
            lines.append(
                f"- **PyTorch GPU Streaming (`B=1`)**: `{lat['pytorch_gpu_stream_ms_per_frame']:.3f} ms/frame`"
            )
            lines.append(
                f"- **PyTorch GPU Batched (`B=32`)**: `{lat['pytorch_gpu_batch32_ms_per_frame']:.3f} ms/frame`"
            )
        lines.append(
            f"- **ONNX Runtime (`{lat['ort_cuda_active_provider']}`)**: `{lat['ort_cuda_ms_per_frame']:.3f} ms/frame`"
        )
        lines.append(
            f"- **ONNX Runtime (`CPUExecutionProvider`)**: `{lat['ort_cpu_ms_per_frame']:.3f} ms/frame`"
        )
        lines.append(f"- **ONNX Parity `max_abs_diff`**: `{lat['onnx_parity_max_abs_diff']:.6e}`")
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def save_reports(
        cls,
        report: dict[str, Any],
        json_path: str | Path | None = None,
        markdown_path: str | Path | None = None,
    ) -> None:
        if json_path is not None:
            jp = Path(json_path)
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if markdown_path is not None:
            mp = Path(markdown_path)
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(cls.format_markdown(report), encoding="utf-8")
