"""CLI entry point for vehicle-plate-ocr-jp-v1 PlateDetNet & PlateQuadNet training and export."""

from __future__ import annotations

import argparse
from pathlib import Path

from vpocr_det.trainer import (
    DetTrainConfig,
    QuadTrainConfig,
    export_det_and_quad_onnx,
    load_det_checkpoint,
    load_quad_checkpoint,
    train_plate_detector,
    train_plate_quad_cropper,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="vehicle-plate-ocr-jp-v1 Plate Detector & Tight Quad Cropper CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_det = sub.add_parser("train-det", help="Train PlateDetNet (base or nano)")
    p_det.add_argument("--preset", default="base", choices=["base", "nano"])
    p_det.add_argument("--steps", type=int, default=3200)
    p_det.add_argument("--batch-size", type=int, default=24)
    p_det.add_argument("--out", default="out/models/plate_det_base.pt")

    p_quad = sub.add_parser("train-quad", help="Train PlateQuadNet tight plate-only cropper")
    p_quad.add_argument("--steps", type=int, default=2800)
    p_quad.add_argument("--batch-size", type=int, default=96)
    p_quad.add_argument("--out", default="out/models/plate_quad_cropper.pt")

    p_exp = sub.add_parser("export-onnx", help="Export PlateDetNet & PlateQuadNet to FP32/INT8 ONNX")
    p_exp.add_argument("--det-ckpt", required=True)
    p_exp.add_argument("--quad-ckpt", required=True)
    p_exp.add_argument("--out-dir", default="out/models/onnx_det")

    args = parser.parse_args()
    if args.cmd == "train-det":
        cfg = DetTrainConfig(preset=args.preset, total_steps=args.steps, batch_size=args.batch_size)
        train_plate_detector(cfg, Path(args.out))
    elif args.cmd == "train-quad":
        cfg = QuadTrainConfig(total_steps=args.steps, batch_size=args.batch_size)
        train_plate_quad_cropper(cfg, Path(args.out))
    elif args.cmd == "export-onnx":
        det = load_det_checkpoint(args.det_ckpt, device="cpu")
        quad = load_quad_checkpoint(args.quad_ckpt, device="cpu")
        info = export_det_and_quad_onnx(det, quad, args.out_dir)
        print("Exported ONNX:", info)


if __name__ == "__main__":
    main()
