"""Command-line interface for the public INT8 release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import MODEL_NAME, __version__, _resolve_root, create_predictor

GRAPH_NAMES = (
    "temporal_det_nano", "vehicle_cutout_nano", "stage2", "frame_encoder",
    "memory_step", "memory_readout", "stopping_risk",
)
CALIBRATION_NAMES = ("stage1_int8_cal.json", "stage2_int8_cal.json", "gen6.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=MODEL_NAME, description="Japanese vehicle and terminal plate OCR")
    parser.add_argument("--version", action="version", version=f"{MODEL_NAME} {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Show the model bundle and inference contract")
    predict = sub.add_parser("predict", help="Recognize plates when the supplied clip ends")
    predict.add_argument("video_path", type=Path)
    predict.add_argument("--json", action="store_true", help="Print compact JSON (default: indented JSON)")
    args = parser.parse_args(argv)
    if args.command == "info":
        directory = _resolve_root() / "web/models"
        graphs = {f"{name}.int8.onnx": (directory / f"{name}.int8.onnx").stat().st_size for name in GRAPH_NAMES}
        calibration = {name: (directory / name).stat().st_size for name in CALIBRATION_NAMES}
        print(json.dumps({
            "model": MODEL_NAME, "version": __version__, "license": "Apache-2.0",
            "graphs_bytes": graphs, "calibration_bytes": calibration,
            "total_bytes": sum(graphs.values()) + sum(calibration.values()),
            "contract": {
                "states": ["COLLECTING", "SEALED", "FINALIZED"],
                "streaming_character_readouts": 0,
                "terminal_readouts_per_collector": 1,
                "backend": "all-INT8 ONNX Runtime / CPU",
                "design_around": "JP7029732B2: vocabulary-free collection, one terminal Bayesian decision",
            },
        }, ensure_ascii=False, indent=2))
        return 0
    if not args.video_path.is_file():
        parser.error(f"Video not found: {args.video_path}")
    try:
        results = create_predictor().predict_clip(args.video_path)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{MODEL_NAME}: {exc}\n")
    print(json.dumps(results, ensure_ascii=False, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
