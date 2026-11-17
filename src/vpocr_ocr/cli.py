"""Command-line interface for MF-JPPlateOCR (`vpocr-ocr`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
import torch

from vpocr_ocr.benchmark import BenchmarkConfig, BenchmarkEngine
from vpocr_ocr.dataset import (
    CropSequenceDataset,
    MicroPlateSequenceDataset,
    StreamingFastSynthDataset,
    adapt_sequence_sample,
)
from vpocr_ocr.edge_cpu import (
    DistillConfig,
    DistillationTrainer,
    benchmark_weak_cpu_pass,
    quantize_onnx_for_cpu,
)
from vpocr_ocr.model import MFJPPlateOCR, ModelConfig
from vpocr_ocr.onnx_export import export_ocr_to_onnx, verify_onnx_parity
from vpocr_ocr.trainer import OCRTrainer, TrainConfig, load_checkpoint
from vpocr_ocr.vocab import PlateVocab
from vpocr_synth.export.fast_synthesizer import FastCropSequenceSynthesizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vpocr-ocr",
        description="MF-JPPlateOCR: Multi-Frame Japanese License Plate OCR CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. train
    p_train = sub.add_parser("train", help="Train MF-JPPlateOCR (Stage 1 + Stage 2 GPU AMP)")
    p_train.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output checkpoint path (.pt)",
    )
    p_train.add_argument(
        "--preset",
        type=str,
        choices=["base", "small", "nano"],
        default=None,
        help="Architecture preset ('base' 2.02M, 'small' 1.11M, 'nano' 0.47M)",
    )
    p_train.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional .npz shard file/directory",
    )
    p_train.add_argument(
        "--fast-synth",
        action="store_true",
        help="Use infinite stratified StreamingFastSynthDataset across all 138 regions",
    )
    p_train.add_argument("--num-sequences", type=int, default=1024, help="Synthetic dataset size")
    p_train.add_argument("--plate-pool", type=int, default=256, help="Distinct base plates")
    p_train.add_argument("--seq-len", type=int, default=8, help="Frames per sequence")
    p_train.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_train.add_argument("--stage1-steps", type=int, default=300, help="Stage 1 single-frame steps")
    p_train.add_argument("--stage2-steps", type=int, default=500, help="Stage 2 multi-frame steps")
    p_train.add_argument("--lr-stage1", type=float, default=1.5e-3, help="Stage 1 max learning rate")
    p_train.add_argument("--lr-stage2", type=float, default=8.0e-4, help="Stage 2 max learning rate")
    p_train.add_argument("--num-workers", type=int, default=0, help="DataLoader CPU workers")
    p_train.add_argument("--d-model", type=int, default=160, help="Latent slot embedding width")
    p_train.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")

    # 2. distill
    p_distill = sub.add_parser(
        "distill",
        help="Distill trained Teacher checkpoint (Base) into compact Student (Nano/Small)",
    )
    p_distill.add_argument("--teacher", type=str, required=True, help="Teacher .pt checkpoint path")
    p_distill.add_argument("--output", type=str, required=True, help="Student .pt output path")
    p_distill.add_argument(
        "--student-preset",
        type=str,
        choices=["nano", "small"],
        default="nano",
        help="Student preset ('nano' 0.47M or 'small' 1.11M)",
    )
    p_distill.add_argument("--data-dir", type=str, default=None, help="Optional .npz shard dir")
    p_distill.add_argument(
        "--fast-synth",
        action="store_true",
        default=True,
        help="Use StreamingFastSynthDataset for distillation (default: enabled)",
    )
    p_distill.add_argument("--num-sequences", type=int, default=50000, help="Virtual dataset size")
    p_distill.add_argument("--seq-len", type=int, default=8, help="Frames per sequence")
    p_distill.add_argument("--batch-size", type=int, default=48, help="Batch size")
    p_distill.add_argument("--stage1-steps", type=int, default=800, help="Stage 1 distill steps")
    p_distill.add_argument("--stage2-steps", type=int, default=1600, help="Stage 2 distill steps")
    p_distill.add_argument("--temperature", type=float, default=2.0, help="KD softmax temperature")
    p_distill.add_argument("--num-workers", type=int, default=4, help="DataLoader CPU workers")
    p_distill.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    p_distill.add_argument("--seed", type=int, default=42, help="Random seed")

    # 3. export-onnx
    p_export = sub.add_parser(
        "export-onnx",
        help="Export frame_encoder.onnx and memory_step_decode.onnx (opset 17 + optional INT8)",
    )
    p_export.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained .pt checkpoint (exports initialized model if omitted)",
    )
    p_export.add_argument(
        "--preset",
        type=str,
        choices=["base", "small", "nano"],
        default="base",
        help="Preset when --checkpoint is omitted",
    )
    p_export.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write frame_encoder.onnx and memory_step_decode.onnx",
    )
    p_export.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p_export.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Also emit dynamic QUInt8 quantized ONNX models for weak edge CPUs",
    )
    p_export.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify numerical parity (< 1e-4) between PyTorch and ONNX Runtime",
    )

    # 4. benchmark
    p_bench = sub.add_parser(
        "benchmark",
        help="Run all 5 MF-JPPlateOCR accuracy, safety, and latency benchmark suites",
    )
    p_bench.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained .pt checkpoint (evaluates initialized model if omitted)",
    )
    p_bench.add_argument("--onnx-dir", type=str, default=None, help="Optional pre-exported ONNX dir")
    p_bench.add_argument("--num-sequences", type=int, default=64, help="Sequences per suite")
    p_bench.add_argument("--plate-pool", type=int, default=128, help="Distinct plates in pool")
    p_bench.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    p_bench.add_argument("--output-json", type=str, default=None, help="Path to write JSON report")
    p_bench.add_argument("--output-md", type=str, default=None, help="Path to write Markdown report")
    p_bench.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    p_bench.add_argument("--seed", type=int, default=2026, help="Benchmark random seed")

    # 5. benchmark-cpu
    p_bcpu = sub.add_parser(
        "benchmark-cpu",
        help="Profile 1-thread / 2-thread weak-CPU edge latency, skip ratios, and pass accuracy",
    )
    p_bcpu.add_argument("--onnx-dir", type=str, required=True, help="Exported ONNX directory")
    p_bcpu.add_argument("--num-sequences", type=int, default=32, help="Number of vehicle passes")
    p_bcpu.add_argument("--frames", type=int, default=16, help="Frames per vehicle pass")
    p_bcpu.add_argument("--threads", type=int, default=1, help="Intra-op CPU threads (e.g. 1 or 2)")
    p_bcpu.add_argument("--int8", action="store_true", help="Use .int8.onnx quantized graphs")
    p_bcpu.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable FastPixelPreGate and post-saturation sentinel skipping",
    )
    p_bcpu.add_argument("--seed", type=int, default=9200, help="Benchmark random seed")

    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    vocab = PlateVocab()
    if args.data_dir is not None:
        train_ds = CropSequenceDataset(args.data_dir, vocab=vocab)
    elif args.fast_synth:
        train_ds = StreamingFastSynthDataset(
            virtual_size=max(args.num_sequences, args.batch_size * (args.stage1_steps + args.stage2_steps)),
            seq_len=args.seq_len,
            mode="mixed",
            base_seed=args.seed * 10000 + 1000,
            variable_length=True,
            vocab=vocab,
        )
    else:
        train_ds = MicroPlateSequenceDataset(
            num_sequences=args.num_sequences,
            seq_len=args.seq_len,
            mode="mixed",
            plate_pool_size=args.plate_pool,
            vocab=vocab,
            seed=args.seed,
            stochastic=True,
        )

    if args.preset is not None:
        m_cfg = ModelConfig.from_preset(args.preset, vocab_size=vocab.size)
    else:
        m_cfg = ModelConfig(vocab_size=vocab.size, d_model=args.d_model)

    model = MFJPPlateOCR(m_cfg)
    t_cfg = TrainConfig(
        batch_size=args.batch_size,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    trainer = OCRTrainer(model=model, vocab=vocab, config=t_cfg, device=args.device)
    summary = trainer.fit(train_dataset=train_ds, checkpoint_path=args.output)
    print(
        json.dumps(
            {
                "checkpoint": str(Path(args.output).resolve()),
                "preset": args.preset or "custom",
                "num_parameters": model.num_parameters,
                "stage1_steps": summary["stage1_steps"],
                "stage2_steps": summary["stage2_steps"],
                "stage1_final_loss": summary["stage1_final_loss"],
                "stage2_final_loss": summary["stage2_final_loss"],
            },
            indent=2,
        )
    )
    return 0


def _cmd_distill(args: argparse.Namespace) -> int:
    teacher, vocab, _ = load_checkpoint(args.teacher, device=args.device or "cpu")
    if args.data_dir is not None:
        train_ds = CropSequenceDataset(args.data_dir, vocab=vocab)
    else:
        train_ds = StreamingFastSynthDataset(
            virtual_size=args.num_sequences,
            seq_len=args.seq_len,
            mode="mixed",
            base_seed=args.seed * 10000 + 5000,
            variable_length=True,
            vocab=vocab,
        )

    student = MFJPPlateOCR(ModelConfig.from_preset(args.student_preset, vocab_size=vocab.size))
    d_cfg = DistillConfig(
        batch_size=args.batch_size,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        temperature=args.temperature,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    distiller = DistillationTrainer(
        teacher=teacher,
        student=student,
        vocab=vocab,
        config=d_cfg,
        device=args.device,
    )
    summary = distiller.fit(train_dataset=train_ds, checkpoint_path=args.output)
    summary["checkpoint"] = str(Path(args.output).resolve())
    summary["student_preset"] = args.student_preset
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_export_onnx(args: argparse.Namespace) -> int:
    if args.checkpoint is not None:
        model, _, _ = load_checkpoint(args.checkpoint, device="cpu")
    else:
        vocab = PlateVocab()
        model = MFJPPlateOCR(ModelConfig.from_preset(args.preset, vocab_size=vocab.size)).eval()

    enc_path, mem_path = export_ocr_to_onnx(
        model, output_dir=args.output_dir, opset_version=args.opset
    )
    result: dict[str, object] = {
        "frame_encoder_onnx": str(enc_path.resolve()),
        "memory_step_decode_onnx": str(mem_path.resolve()),
    }
    if args.verify:
        parity = verify_onnx_parity(model, args.output_dir)
        result["parity"] = parity
    if args.quantize_int8:
        q_paths = quantize_onnx_for_cpu(args.output_dir)
        result["frame_encoder_int8_onnx"] = str(q_paths["encoder_int8"].resolve())
        result["memory_step_decode_int8_onnx"] = str(q_paths["memory_int8"].resolve())
    print(json.dumps(result, indent=2))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    dev = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.checkpoint is not None:
        model, vocab, _ = load_checkpoint(args.checkpoint, device=dev)
    else:
        vocab = PlateVocab()
        model = MFJPPlateOCR(ModelConfig(vocab_size=vocab.size)).to(dev).eval()

    b_cfg = BenchmarkConfig(
        num_sequences=args.num_sequences,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    engine = BenchmarkEngine(model=model, vocab=vocab, config=b_cfg, device=dev)
    report = engine.run_all(onnx_dir=args.onnx_dir, plate_pool_size=args.plate_pool)
    engine.save_reports(report, json_path=args.output_json, markdown_path=args.output_md)
    print(engine.format_markdown(report))
    return 0


def _cmd_benchmark_cpu(args: argparse.Namespace) -> int:
    vocab = PlateVocab()
    synth = FastCropSequenceSynthesizer()
    samples = synth.synthesize_batch(
        num_sequences=args.num_sequences,
        start_seed=args.seed,
        num_frames=args.frames,
        mode="readable",
    )
    adapted = [adapt_sequence_sample(s, vocab) for s in samples]
    metrics = benchmark_weak_cpu_pass(
        onnx_dir=args.onnx_dir,
        sequences=adapted,
        intra_threads=args.threads,
        use_int8=args.int8,
        enable_adaptive=not args.no_adaptive,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "distill":
        return _cmd_distill(args)
    if args.command == "export-onnx":
        return _cmd_export_onnx(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "benchmark-cpu":
        return _cmd_benchmark_cpu(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
