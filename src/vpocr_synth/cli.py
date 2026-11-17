"""Command line interface for vehicle-plate-ocr-jp-v1 synthetic data generator."""

import argparse
import json
from pathlib import Path
import sys
import time
from PIL import Image, ImageDraw

from vpocr_synth.camera.simulator import CameraSimulator
from vpocr_synth.crop.extractor import CropExtractor
from vpocr_synth.export.fast_synthesizer import FastCropSequenceSynthesizer
from vpocr_synth.export.shard_io import ShardExporter
from vpocr_synth.motion.generator import SequenceGenerator
from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.models import PlateConfig, PlateType
from vpocr_synth.plate.renderer import PlateRenderer
from vpocr_synth.plate.sampler import PlateSampler
from vpocr_synth.plate.validator import PlateValidator, ValidationError


def cmd_degrade(args: argparse.Namespace) -> int:
    """Degrade clean motion sequences through realistic physical camera and scene simulation."""
    in_dir = Path(getattr(args, "in"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_in = in_dir / "sequences.jsonl"
    if not index_in.exists():
        print(f"Error: Input sequence index not found: {index_in}", file=sys.stderr)
        return 1

    seq_gen = SequenceGenerator()
    cam_sim = CameraSimulator()

    available_profile_ids = cam_sim.get_available_profiles()
    num_profiles = min(len(available_profile_ids), max(1, args.profiles))

    # Read input sequences
    with open(index_in, "r", encoding="utf-8") as f:
        clean_records = [json.loads(line) for line in f if line.strip()]

    print(f"Degrading {len(clean_records)} sequences with {num_profiles} profiles each -> {out_dir}")
    index_out = out_dir / "degraded_sequences.jsonl"

    comparison_saved = False
    with open(index_out, "w", encoding="utf-8") as f_out:
        for s_idx, rec in enumerate(clean_records):
            clean_seed = rec["seed"]
            clean_seq = seq_gen.generate_sequence(seed=clean_seed)

            for p_idx in range(num_profiles):
                prof_id = available_profile_ids[p_idx % len(available_profile_ids)]
                prof_seed = args.seed + s_idx * 100 + p_idx * 17
                profile = cam_sim.create_profile_by_id(prof_id, seed=prof_seed)

                deg_seq = cam_sim.degrade_sequence(clean_seq, profile)

                deg_dir = out_dir / deg_seq.sequence_id
                deg_dir.mkdir(parents=True, exist_ok=True)
                frames_dir = deg_dir / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)

                for f_idx, img in enumerate(deg_seq.images):
                    img.save(frames_dir / f"frame_{f_idx:04d}.png")

                with open(deg_dir / "meta.json", "w", encoding="utf-8") as f_m:
                    json.dump(deg_seq.to_dict(), f_m, ensure_ascii=False, indent=2)

                # Comparison filmstrip
                comp_img = cam_sim.create_comparison_filmstrip(clean_seq, deg_seq, max_samples=6)
                comp_img.save(deg_dir / "comparison.png")

                if not comparison_saved:
                    comp_img.save(out_dir / "comparison_preview.png")
                    comparison_saved = True

                summary = {
                    "sequence_id": deg_seq.sequence_id,
                    "parent_sequence_id": deg_seq.parent_sequence_id,
                    "profile_id": profile.id,
                    "profile_name": profile.name,
                    "illumination": profile.illumination.value,
                    "text": deg_seq.ground_truth_text,
                    "ocr_string": deg_seq.ocr_string,
                    "num_frames": deg_seq.num_frames,
                    "fps": round(deg_seq.fps, 2),
                    "path": str(deg_dir.relative_to(out_dir)),
                    "seed": prof_seed,
                }
                f_out.write(json.dumps(summary, ensure_ascii=False) + "\n")

                print(f"  [{s_idx + 1}/{len(clean_records)}] Generated: {deg_seq.sequence_id} ({profile.name})")

    print(f"Completed! Degraded sequences saved to {out_dir}")
    print(f"Comparison preview saved to: {out_dir / 'comparison_preview.png'}")
    return 0


def cmd_sequences(args: argparse.Namespace) -> int:
    """Generate multi-frame continuous vehicle approach sequences with labels."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_gen = SequenceGenerator()
    index_file = out_dir / "sequences.jsonl"

    print(f"Generating {args.n} multi-frame motion sequences (seed={args.seed}) -> {out_dir}")

    with open(index_file, "w", encoding="utf-8") as f_out:
        for idx in range(args.n):
            seq_seed = args.seed + idx
            seq_res = seq_gen.generate_sequence(seed=seq_seed)

            # Sequence directory
            seq_dir = out_dir / seq_res.sequence_id
            seq_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = seq_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            # Save individual frame images
            for f_idx, img in enumerate(seq_res.images):
                img.save(frames_dir / f"frame_{f_idx:04d}.png")

            # Save sequence metadata
            meta_path = seq_dir / "meta.json"
            seq_dict = seq_res.to_dict()
            with open(meta_path, "w", encoding="utf-8") as f_meta:
                json.dump(seq_dict, f_meta, ensure_ascii=False, indent=2)

            # Export preview filmstrip and GIF for first few sequences
            if idx < args.preview_count:
                filmstrip = seq_gen.create_filmstrip(seq_res, max_samples=8)
                filmstrip.save(seq_dir / "filmstrip.png")
                seq_gen.export_animated_preview(seq_res, seq_dir / "preview.gif")

                # If first sequence, also save to top-level for convenience
                if idx == 0:
                    filmstrip.save(out_dir / "preview_filmstrip.png")
                    seq_gen.export_animated_preview(seq_res, out_dir / "preview.gif")

            # Write record to aggregate index
            summary_record = {
                "sequence_id": seq_res.sequence_id,
                "scenario": seq_res.scenario.value,
                "text": seq_res.plate_text.full_text,
                "ocr_string": seq_res.plate_text.ocr_string,
                "num_frames": seq_res.num_frames,
                "fps": round(seq_res.fps, 2),
                "resolution": [seq_res.camera_params.width, seq_res.camera_params.height],
                "path": str(seq_dir.relative_to(out_dir)),
                "seed": seq_seed,
            }
            f_out.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

            print(
                f"  [{idx + 1}/{args.n}] {seq_res.sequence_id}: {seq_res.scenario.value} | "
                f"{seq_res.num_frames} frames ({seq_res.fps:.1f} fps) | {seq_res.plate_text.full_text}"
            )

    print(f"Completed! {args.n} sequences saved to {out_dir}")
    print(f"Top-level preview filmstrip: {out_dir / 'preview_filmstrip.png'}")
    return 0


def cmd_plates(args: argparse.Namespace) -> int:
    """Generate synthetic plate images and JSONL labels."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels_file = out_dir / "labels.jsonl"

    sampler = PlateSampler()
    renderer = PlateRenderer()

    print(f"Generating {args.n} plates (seed={args.seed}, scale={args.scale}) -> {out_dir}")

    with open(labels_file, "w", encoding="utf-8") as f_lbl:
        for idx in range(args.n):
            sample_seed = args.seed + idx
            plate_text, config = sampler.sample(seed=sample_seed)
            config.scale = args.scale

            result = renderer.render(plate_text, config)

            img_filename = f"{idx:06d}.{args.format}"
            img_path = images_dir / img_filename
            result.image.save(img_path)

            record = result.to_dict()
            record["sample_id"] = idx
            record["image_path"] = str(img_path.relative_to(out_dir))
            f_lbl.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % max(1, args.n // 10) == 0 or idx + 1 == args.n:
                print(f"  [{idx + 1}/{args.n}] Generated: {plate_text.full_text}")

    print(f"Completed! Images saved to {images_dir}, labels written to {labels_file}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Generate a tiled contact sheet PNG previewing N plates."""
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sampler = PlateSampler()
    renderer = PlateRenderer()

    n = args.n
    cols = args.cols
    rows = (n + cols - 1) // cols

    samples = []
    print(f"Generating {n} plates for contact sheet preview (seed={args.seed})...")
    for i in range(n):
        sample_seed = args.seed + i
        pt, cfg = sampler.sample(seed=sample_seed)
        cfg.scale = args.scale
        res = renderer.render(pt, cfg)
        samples.append(res)

    sample_w = max(s.image.width for s in samples)
    sample_h = max(s.image.height for s in samples)

    margin = int(round(16 * (args.scale / 2.0)))
    sheet_w = cols * sample_w + (cols + 1) * margin
    sheet_h = rows * sample_h + (rows + 1) * margin

    contact_sheet = Image.new("RGB", (sheet_w, sheet_h), (35, 38, 42))  # Dark neutral background
    draw = ImageDraw.Draw(contact_sheet)

    for idx, res in enumerate(samples):
        r = idx // cols
        c = idx % cols
        x = margin + c * (sample_w + margin) + (sample_w - res.image.width) // 2
        y = margin + r * (sample_h + margin) + (sample_h - res.image.height) // 2

        tile_img = res.image.copy()

        if args.with_boxes:
            # Draw bounding boxes overlay on tile
            overlay_draw = ImageDraw.Draw(tile_img)
            # Character boxes in cyan
            for cb in res.character_boxes:
                b = cb.bbox
                overlay_draw.rectangle([b.x_min, b.y_min, b.x_max, b.y_max], outline=(0, 220, 255), width=1)
            # Field boxes in magenta
            for fname, fb in res.field_boxes.items():
                overlay_draw.rectangle([fb.x_min, fb.y_min, fb.x_max, fb.y_max], outline=(255, 0, 128), width=2)

        contact_sheet.paste(tile_img, (x, y))

    contact_sheet.save(out_file)
    print(f"Preview contact sheet saved: {out_file} ({sheet_w}x{sheet_h} px, {n} samples)")
    return 0


def cmd_charset(args: argparse.Namespace) -> int:
    """Export the full OCR character set."""
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chars = PlateDataLoader.load_charset()
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chars:
            f.write(c + "\n")

    print(f"Exported {len(chars)} OCR characters to {out_path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a license plate string."""
    text = args.text
    print(f"Validating: '{text}'")
    try:
        pt = PlateValidator.parse_string(text)
        print(f"  Valid! Parsed: {pt.to_dict()}")
        return 0
    except ValidationError as e:
        print(f"  INVALID: {e}", file=sys.stderr)
        return 1


def cmd_multicam(args: argparse.Namespace) -> int:
    """Generate arbitrary 360-degree multi-camera gate pass sequences."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_gen = SequenceGenerator()
    index_path = out_dir / "multicam_passes.jsonl"
    num_passes = max(1, int(args.n))
    num_cameras = max(1, min(4, int(getattr(args, "cameras", 3))))
    rig_preset = getattr(args, "rig_preset", None)
    apply_deg = not bool(getattr(args, "no_degrade", False))

    print(f"Generating {num_passes} multi-camera passes ({num_cameras} cameras, preset={rig_preset}) -> {out_dir}")
    with open(index_path, "w", encoding="utf-8") as f_out:
        for idx in range(num_passes):
            pass_seed = args.seed + idx
            mc_pass = seq_gen.generate_multicam_pass(
                seed=pass_seed,
                num_cameras=num_cameras,
                rig_preset=rig_preset,
                apply_degradation=apply_deg,
            )
            pass_dir = out_dir / mc_pass.pass_id
            pass_dir.mkdir(parents=True, exist_ok=True)
            with open(pass_dir / "meta.json", "w", encoding="utf-8") as f_m:
                json.dump(mc_pass.to_dict(), f_m, ensure_ascii=False, indent=2)

            strip = seq_gen.create_multicam_filmstrip(mc_pass, max_samples=6)
            strip.save(pass_dir / "multicam_filmstrip.png")
            if idx == 0:
                strip.save(out_dir / "multicam_preview.png")

            f_out.write(json.dumps(mc_pass.to_dict(), ensure_ascii=False) + "\n")
            print(f"  [{idx + 1}/{num_passes}] {mc_pass.pass_id}: {mc_pass.plate_text.full_text} ({len(mc_pass.streams)} cams)")

    print(f"Completed! Multi-camera passes saved to {out_dir}")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    """Synthesize and export high-throughput OCR `.npz` + `.jsonl` dataset shards."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seq = max(1, int(args.n))
    num_frames = max(4, int(getattr(args, "frames", 8)))
    split = getattr(args, "split", "train")
    shard_size = max(1, int(getattr(args, "shard_size", 256)))
    mode = getattr(args, "mode", "mixed")

    synth = FastCropSequenceSynthesizer()
    exporter = ShardExporter(shard_size=shard_size, compressed=False)

    t0 = time.perf_counter()
    samples = synth.synthesize_batch(
        num_sequences=n_seq,
        start_seed=args.seed,
        num_frames=num_frames,
        mode=mode,
    )
    elapsed = max(1e-6, time.perf_counter() - t0)
    seq_per_sec = n_seq / elapsed

    split_dir = exporter.export(out_dir, samples, split=split)

    # Save visual preview strip of first sample
    preview_strip = CropExtractor.create_crop_sequence_strip(samples[0], max_cols=8)
    preview_strip.save(out_dir / "crop_sequences_preview.png")

    print(
        f"Exported {n_seq} sequences to {split_dir} "
        f"({seq_per_sec:.1f} sequences/sec on CPU)"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vpocr-synth",
        description="vehicle-plate-ocr-jp-v1 Synthetic Data Generator for Japanese License Plate OCR",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # plates
    p_plates = subparsers.add_parser("plates", help="Generate synthetic plate dataset")
    p_plates.add_argument("--n", type=int, default=100, help="Number of samples to generate")
    p_plates.add_argument("--out", type=str, default="out/plates", help="Output directory")
    p_plates.add_argument("--seed", type=int, default=0, help="Base random seed")
    p_plates.add_argument("--scale", type=float, default=2.0, help="Pixel scale (px/mm, default: 2.0 -> 660x330)")
    p_plates.add_argument("--format", type=str, default="png", choices=["png", "jpg"], help="Image format")
    p_plates.set_defaults(func=cmd_plates)

    # preview
    p_preview = subparsers.add_parser("preview", help="Generate tiled contact sheet preview")
    p_preview.add_argument("--n", type=int, default=16, help="Number of samples to preview")
    p_preview.add_argument("--out", type=str, default="out/preview.png", help="Output PNG path")
    p_preview.add_argument("--seed", type=int, default=42, help="Random seed")
    p_preview.add_argument("--cols", type=int, default=4, help="Number of columns in grid")
    p_preview.add_argument("--scale", type=float, default=1.5, help="Pixel scale (px/mm, default: 1.5)")
    p_preview.add_argument("--with-boxes", action="store_true", help="Overlay bounding boxes")
    p_preview.set_defaults(func=cmd_preview)

    # charset
    p_charset = subparsers.add_parser("charset", help="Export full OCR character set")
    p_charset.add_argument("--out", type=str, default="data/charset.txt", help="Output charset file path")
    p_charset.set_defaults(func=cmd_charset)

    # sequences
    p_seq = subparsers.add_parser("sequences", help="Generate multi-frame vehicle approach motion sequences")
    p_seq.add_argument("--n", type=int, default=20, help="Number of sequences to generate")
    p_seq.add_argument("--out", type=str, default="out/seq", help="Output directory")
    p_seq.add_argument("--seed", type=int, default=0, help="Base random seed")
    p_seq.add_argument("--preview-count", type=int, default=3, help="Number of preview filmstrips/GIFs to generate")
    p_seq.set_defaults(func=cmd_sequences)

    # degrade
    p_deg = subparsers.add_parser("degrade", help="Apply camera simulator degradation to clean sequences")
    p_deg.add_argument("--in", dest="in_dir", type=str, default="out/seq", help="Input sequences directory")
    p_deg.add_argument("--out", type=str, default="out/seq_cam", help="Output directory")
    p_deg.add_argument("--profiles", type=int, default=3, help="Number of camera profiles per sequence")
    p_deg.add_argument("--seed", type=int, default=0, help="Random seed")
    p_deg.set_defaults(func=lambda args: cmd_degrade(argparse.Namespace(**{**vars(args), "in": args.in_dir})))

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate plate text string")
    p_validate.add_argument("text", type=str, help="Plate string e.g. '品川 599 あ 12-34'")
    p_validate.set_defaults(func=cmd_validate)

    # multicam
    p_mc = subparsers.add_parser("multicam", help="Generate 360-degree multi-camera gate passes")
    p_mc.add_argument("--n", type=int, default=5, help="Number of vehicle passes")
    p_mc.add_argument("--cameras", type=int, default=3, help="Number of cameras (1..4)")
    p_mc.add_argument("--rig-preset", dest="rig_preset", type=str, default=None, help="Optional rig preset")
    p_mc.add_argument("--out", type=str, default="out/multicam", help="Output directory")
    p_mc.add_argument("--seed", type=int, default=0, help="Random seed")
    p_mc.add_argument("--no-degrade", action="store_true", help="Skip camera ISP degradation")
    p_mc.set_defaults(func=cmd_multicam)

    # dataset
    p_ds = subparsers.add_parser("dataset", help="Generate high-throughput OCR crop sequence shards (.npz + .jsonl)")
    p_ds.add_argument("--n", type=int, default=512, help="Number of sequences to synthesize")
    p_ds.add_argument("--frames", type=int, default=8, help="Frames per sequence")
    p_ds.add_argument("--split", type=str, default="train", help="Split name (e.g. train, val, test)")
    p_ds.add_argument("--shard-size", dest="shard_size", type=int, default=256, help="Sequences per shard")
    p_ds.add_argument("--mode", type=str, default="mixed", help="Synthesis curriculum mode")
    p_ds.add_argument("--out", type=str, default="out/dataset", help="Output dataset directory")
    p_ds.add_argument("--seed", type=int, default=0, help="Base random seed")
    p_ds.set_defaults(func=cmd_dataset)

    parsed = parser.parse_args()
    sys.exit(parsed.func(parsed))


if __name__ == "__main__":
    main()
