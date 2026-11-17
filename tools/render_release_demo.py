"""Deterministically select a synthetic, terminal-verified release illustration.

No evaluation/calibration inputs are used. Only Scene3DGPUEngine pixels are
encoded; the predictor sees the resulting H.264, never synthesis labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

import torch
from vpocr_synth.gpu.scene_3d_gpu import Scene3DClipConfig, Scene3DGPUEngine
import vehicle_plate_ocr_jp_v1


def close_gate_seed(seed: int) -> bool:
    # The engine samples the rig after these six nuisance draws when night=0.
    # Filter the deterministic sequence without changing the engine or its RNG.
    rng = random.Random(seed)
    for _ in range(6):
        rng.random()
    return rng.choices(["close_gate", "mid_gate", "high_pole_wide", "top_edge_pitch"],
                       weights=[.38, .34, .16, .12], k=1)[0] == "close_gate"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=20265115)
    parser.add_argument("--attempts", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    config = Scene3DClipConfig(
        image_hw=(448, 768), num_frames=32, bank_size=64, split="val",
        domain_split="val_holdout", p_negative_clip=0, p_night_scene=0,
        p_rain_wet=0, p_chainlink_fence=0, p_gate_pillar_occlusion=0,
        p_stationary_clip=0, max_substeps=1,
    )
    output_frames = 28  # Short excerpt from a 32-frame synthesized trajectory.
    engine = Scene3DGPUEngine(config, device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = vehicle_plate_ocr_jp_v1.create_predictor()
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        import imageio_ffmpeg
        encoder = imageio_ffmpeg.get_ffmpeg_exe()
    dest = root / "web/samples/synthetic_gate_demo.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="release-demo-") as tmp:
        candidate = Path(tmp) / "candidate.mp4"
        for seed in range(args.start_seed, args.start_seed + args.attempts):
            if not close_gate_seed(seed):
                continue
            print(f"Rendering close_gate seed={seed} val_holdout", flush=True)
            clip = engine.render_clip(seed)
            if clip.scene_mode != "close_gate":
                continue
            frames = (clip.frames_chw[:output_frames].clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            subprocess.run([
                encoder, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s:v", "768x448", "-r", "12", "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(candidate),
            ], input=frames.tobytes(), check=True)
            if candidate.stat().st_size > 1_500_000:
                continue
            predictions = predictor.predict_clip(candidate)
            print(f"Terminal predictions: {predictions}", flush=True)
            if not predictions:
                continue
            # Public predictor returns ACCEPTED terminal predictions only.
            record = {
                "generator": "Scene3DGPUEngine", "seed": seed,
                "domain_split": clip.domain_split, "split": config.split,
                "bank_size": config.bank_size, "bank_seed": config.bank_seed,
                "scene_mode": clip.scene_mode, "grammar_combo_id": clip.grammar_combo_id,
                "font_family_id": clip.font_family_id,
                "camera_motion_family": clip.camera_motion_family,
                "width": 768, "height": 448, "frames": output_frames, "render_frames": config.num_frames, "fps": 12,
                "codec": "H.264", "pixel_format": "yuv420p", "crf": 20,
                "bytes": candidate.stat().st_size, "expected_predictions": predictions,
                "expected_serial": predictions[0]["serial"],
                "verification": "create_predictor().predict_clip(encoded_mp4): ACCEPTED",
                "render_options": {k: getattr(config, k) for k in (
                    "p_negative_clip", "p_night_scene", "p_rain_wet", "p_chainlink_fence",
                    "p_gate_pillar_occlusion", "p_stationary_clip", "max_substeps")},
            }
            shutil.copyfile(candidate, dest)
            dest.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            return
    raise RuntimeError("No clean ACCEPTED close_gate seed found; existing demo preserved")


if __name__ == "__main__":
    main()
