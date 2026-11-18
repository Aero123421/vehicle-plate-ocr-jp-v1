"""Reproducible Gen-6 synthetic detector, quad, export and calibration pipeline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

# Help and invalid arguments exit before importing training dependencies.
sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--stage1-only", action="store_true", help="Train, export and calibrate Stage-1 only")
    mode_group.add_argument("--stage2-only", action="store_true", help="Train, export and calibrate Stage-2 only")
    parser.add_argument("--stage1-ckpt", type=Path, default=None, help="Optional Stage-1 checkpoint for Stage-2 proposals")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out/models")
    parser.add_argument("--cal-dir", type=Path, default=REPO_ROOT / "out/calibration")
    parser.add_argument("--base-steps", type=int, default=3600)
    parser.add_argument("--nano-steps", type=int, default=4400)
    parser.add_argument("--quad-steps", type=int, default=11000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--clip-frames", type=int, default=8)
    parser.add_argument("--max-pool-clips", type=int, default=None)
    parser.add_argument("--initial-clips", type=int, default=None)
    args = parser.parse_args(argv)
    if args.clip_frames < 1:
        parser.error("--clip-frames must be >= 1")
    if args.max_pool_clips is not None and args.max_pool_clips < 1:
        parser.error("--max-pool-clips must be >= 1")
    if args.initial_clips is not None and args.initial_clips < 1:
        parser.error("--initial-clips must be >= 1")
    return args


if __name__ == "__main__":
    parse_args()

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(REPO_ROOT / "src"))

from vpocr_det.detector import TemporalDetModelConfig, TemporalPlateDetNet, _match_query_nll
from vpocr_det.trainer import (
    FORBIDDEN_REAL_MARKERS_DET, DEFAULT_PHASE4_SYNTH_SOURCE_FILES,
    TemporalDetTrainConfig, QuadTrackTrainConfig,
    build_stage1_validation_pool, build_stage2_validation_pool,
    build_stage1_calibration_pool, build_stage2_calibration_pool,
    compute_det_file_sha256, save_temporal_det_checkpoint, save_quad_track_checkpoint,
    load_temporal_det_checkpoint,
    train_temporal_plate_detector, train_plate_quad_track_net,
    verify_temporal_det_checkpoint_provenance, verify_quad_track_checkpoint_provenance,
    export_temporal_det_onnx, export_quad_track_onnx,
    verify_temporal_det_onnx_parity, verify_quad_track_onnx_parity,
    fit_stage1_post_int8_calibration, fit_stage2_post_int8_calibration,
)
from vpocr_det.quad_cropper import PlateQuadTrackNet, QuadTrackModelConfig
from vpocr_det.types import CLS_PLATE, CLS_VEHICLE

TRAINING_SCRIPT = "tools/train_det_quad.py"


def source_snapshot():
    """Freeze the trainer's provenance inputs, this entry point and source tree."""
    paths = {REPO_ROOT / rel for rel in DEFAULT_PHASE4_SYNTH_SOURCE_FILES}
    paths.add(REPO_ROOT / TRAINING_SCRIPT)
    paths.update((REPO_ROOT / "src").rglob("*.py"))
    paths.update(p for p in (REPO_ROOT / "data/fonts").rglob("*") if p.is_file())
    return {p.relative_to(REPO_ROOT).as_posix(): compute_det_file_sha256(p)
            for p in sorted(paths)}


def assert_frozen(snapshot):
    if source_snapshot() != snapshot:
        raise RuntimeError("Source changed after the pre-training freeze")


def box_iou(a, b):
    inter = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
             torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter).clamp_min(torch.finfo(inter.dtype).tiny)


@torch.no_grad()
def evaluate_stage1(model, *, seed, device, clips, dry_run):
    """One-to-one recall at IoU .5 and equal-cost Bernoulli decisions; soft mask IoU."""
    hw = (256, 320) if dry_run else (384, 640)
    pool = build_stage1_validation_pool(image_hw=hw, num_frames=2 if dry_run else 4,
                                       seed=seed, device=device)
    pool.ingest_clips(clips)
    items = pool.items[:]
    counts = {"plate": [0, 0], "vehicle": [0, 0]}
    soft_ious = []
    model.eval()
    for item in items:
        pool.items = [item]
        frames, targets = pool.sample_batch(1, hw, device)
        hidden = None
        for k, truth in enumerate(targets):
            out = model.forward_step(frames[:, k], hidden)
            hidden = out["h_next"]
            target = truth[0]
            nq = model.config.num_vehicle_queries
            for name, cls in (("plate", CLS_PLATE), ("vehicle", CLS_VEHICLE)):
                selected = target["labels"] == cls
                gt = target["amodal_boxes" if name == "vehicle" else "boxes"][selected].to(device)
                counts[name][1] += len(gt)
                if not len(gt):
                    continue
                boxes = out["pred_boxes"][0, -nq:] if name == "vehicle" else out["pred_boxes"][0, :-nq]
                scores = out["veh_log_odds"][0].sigmoid() if name == "vehicle" else out["cls_logits"][0, :-nq, cls].sigmoid()
                overlaps = box_iou(gt, boxes)
                # Equal-cost decisions define the detected set before association:
                # a rejected duplicate cannot consume an accepted detection's GT.
                accepted = torch.nonzero(scores >= .5, as_tuple=False).squeeze(1)
                if len(accepted):
                    hit_ri, hit_ci = _match_query_nll((-overlaps[:, accepted]).cpu().numpy())
                    counts[name][0] += int((overlaps[hit_ri, accepted[hit_ci]] >= .5).sum())
                if name == "vehicle":
                    # Mask quality remains unconditional on detection acceptance.
                    ri, ci = _match_query_nll((-overlaps).cpu().numpy())
                    # Fractional masks are expressed in the visible-box chart.
                    # Amodal localization does not change their sampling map.
                    visible_boxes = target["boxes"][selected][ri].to(device)
                    rois = torch.cat((torch.zeros((len(ci), 1), device=device), visible_boxes), -1)
                    masks = model.cutout_head(out["p2_feat"], out["aux_edge_3ch"], rois,
                                              out["query_condition"][0, ci])
                    actual = masks.sigmoid().reshape(len(ci), -1)
                    expected = target["cutout_masks"][selected][ri].to(device).reshape(len(ci), -1)
                    inter = (actual * expected).sum(-1)
                    union = (actual + expected - actual * expected).sum(-1)
                    soft_ious.extend((inter / union.clamp_min(torch.finfo(union.dtype).tiny)).cpu().tolist())
    return {"domain_split": pool.domain_split, "num_clips": clips,
            **{f"{key}_recall_iou50_equal_cost": hit / total if total else 0.
               for key, (hit, total) in counts.items()},
            "soft_cutout_iou": float(np.mean(soft_ious)) if soft_ious else 0.,
            "num_plates": counts["plate"][1], "num_vehicles": counts["vehicle"][1]}


def select_stage1(path, model, *, seed, device, clips, dry_run):
    ema_metrics = evaluate_stage1(model, seed=seed, device=device, clips=clips, dry_run=dry_run)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = TemporalPlateDetNet(TemporalDetModelConfig.from_dict(payload["config"]),
                              load_pretrained_dfine=False).to(device).eval()
    raw.load_state_dict(payload["raw_model_state_dict"], strict=True)
    raw_metrics = evaluate_stage1(raw, seed=seed, device=device, clips=clips, dry_run=dry_run)
    keys = ("plate_recall_iou50_equal_cost", "vehicle_recall_iou50_equal_cost", "soft_cutout_iou")
    # Pareto improvement avoids arbitrary weights or promotion margins.
    promote = all(raw_metrics[k] >= ema_metrics[k] for k in keys) and any(raw_metrics[k] > ema_metrics[k] for k in keys)
    if promote:
        save_temporal_det_checkpoint(path, model=raw, extra=payload["extra"], provenance=payload["provenance"])
        model = raw
    report = {"selected": "raw" if promote else "ema", "ema": ema_metrics, "raw": raw_metrics}
    print(f"[validation {path.stem}] {json.dumps(report)}", flush=True)
    return model.eval(), report


@torch.no_grad()
def evaluate_stage2(model, *, seed, device, dry_run):
    pool = build_stage2_validation_pool(seed=seed, device=device,
                image_hw=(256, 320) if dry_run else (416, 736),
                num_frames=2 if dry_run else 4)
    pool.ingest_clips(1 if dry_run else 16)
    batch = pool.sample_batch(8 if dry_run else 256, device)
    model.eval()
    out = model(batch["fovea_rois"], batch["surround_rois"], batch["sampling_policy"])
    positive = batch["is_plate"].bool()
    logits = out["is_plate_logit"].reshape(-1)
    error = ((out["corners_norm"][positive] - batch["corners_norm"][positive]) *
             torch.tensor([160., 96.], device=device)).norm(dim=-1).mean()
    return {"domain_split": pool.domain_split, "num_clips": pool.clip_counter,
            "num_observations": len(positive), "corner_error_px": float(error),
            "plate_nll": float(F.binary_cross_entropy_with_logits(logits, batch["is_plate"])),
            "plate_recall_equal_cost": float((logits[positive] >= 0).float().mean()),
            "null_rejection_equal_cost": float((logits[~positive] < 0).float().mean())}


def select_stage2(path, model, *, seed, device, dry_run):
    ema_metrics = evaluate_stage2(model, seed=seed, device=device, dry_run=dry_run)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = PlateQuadTrackNet(QuadTrackModelConfig.from_dict(payload["config"])).to(device).eval()
    raw.load_state_dict(payload["raw_model_state_dict"], strict=True)
    raw_metrics = evaluate_stage2(raw, seed=seed, device=device, dry_run=dry_run)
    objectives = (("corner_error_px", -1), ("plate_nll", -1),
                  ("plate_recall_equal_cost", 1), ("null_rejection_equal_cost", 1))
    promote = (all(direction * raw_metrics[k] >= direction * ema_metrics[k] for k, direction in objectives)
               and any(direction * raw_metrics[k] > direction * ema_metrics[k] for k, direction in objectives))
    if promote:
        save_quad_track_checkpoint(path, model=raw, extra=payload["extra"], provenance=payload["provenance"])
        model = raw
    report = {"selected": "raw" if promote else "ema", "ema": ema_metrics, "raw": raw_metrics}
    print(f"[validation {path.stem}] {json.dumps(report)}", flush=True)
    return model.eval(), report


def main():
    args = parse_args()
    text = Path(__file__).read_text(encoding="utf-8")
    if any(marker in text for marker in FORBIDDEN_REAL_MARKERS_DET):
        raise ValueError("Training script failed the data-isolation scan")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This run requires CUDA with bfloat16 support")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    random.seed(20260600)
    np.random.seed(20260600)
    torch.manual_seed(20260600)
    device = torch.device("cuda")
    frozen = source_snapshot()
    start = time.perf_counter()
    out_dir, cal_dir = args.out_dir.resolve(), args.cal_dir.resolve()
    if args.dry_run and (out_dir == (REPO_ROOT / "out/models").resolve() or
                         cal_dir == (REPO_ROOT / "out/calibration").resolve()):
        raise ValueError("Dry run requires isolated output and calibration directories")
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)
    dry = args.dry_run
    run_stage1 = not args.stage2_only
    run_stage2 = not args.stage1_only
    s1_initial = args.initial_clips if args.initial_clips is not None else (1 if dry else 128)
    s1_max_pool = args.max_pool_clips if args.max_pool_clips is not None else (2 if dry else max(320, s1_initial))
    dry_config = dict(
        total_steps=2,
        batch_size=1,
        clip_frames=2,
        clip_image_hw=(256, 320),
        multi_scale_hws=((256, 320),),
        initial_clips=s1_initial,
        max_pool_clips=s1_max_pool,
        refresh_clips_every=0,
        log_every=2,
    ) if dry else {}
    full_det_common = dict(
        batch_size=args.batch_size,
        clip_frames=int(args.clip_frames),
        clip_image_hw=(448, 768),
        multi_scale_hws=((384, 640), (416, 736), (448, 768)),
        initial_clips=s1_initial,
        max_pool_clips=s1_max_pool,
    )
    mode_name = "stage1_only" if args.stage1_only else ("stage2_only" if args.stage2_only else "full")
    report = {
        "version": "0.7.0",
        "mode": mode_name,
        "dry_run": dry,
        "training_script": TRAINING_SCRIPT,
        "training_script_sha256": compute_det_file_sha256(Path(__file__)),
        "source_freeze": frozen,
        "device": torch.cuda.get_device_name(),
        "precision": "bf16",
    }
    artifacts: list[Path] = []
    base: TemporalPlateDetNet | None = None
    base_path = out_dir / "temporal_plate_det_v4_base.pt"
    nano_path = out_dir / "temporal_plate_det_v4_nano.pt"

    if run_stage1:
        base_cfg = TemporalDetTrainConfig(
            preset="base",
            **(dry_config or dict(total_steps=args.base_steps, refresh_clips_every=15, refresh_clips_count=2, **full_det_common)),
            init_seed=20260641,
            seed=20266441,
        )
        nano_cfg = TemporalDetTrainConfig(
            preset="nano",
            **(dry_config or dict(total_steps=args.nano_steps, refresh_clips_every=24, refresh_clips_count=2, **full_det_common)),
            lr=1.9e-3,
            kd_weight=0.55,
            init_seed=20260642,
            seed=20266442,
        )
        print("[Gen6] Training Base", flush=True)
        base, summary, shared_pool = train_temporal_plate_detector(
            base_cfg, base_path, repo_root=REPO_ROOT, training_script=TRAINING_SCRIPT, device=device
        )
        assert_frozen(frozen)
        base, validation = select_stage1(base_path, base, seed=20269641, device=device, clips=1 if dry else 16, dry_run=dry)
        base_prov = verify_temporal_det_checkpoint_provenance(base_path, repo_root=REPO_ROOT, verify_source_hashes=True)
        report["base"] = dict(config=asdict(base_cfg), training=summary, validation=validation, provenance=base_prov)
        print("[Gen6] Training Nano with Base synthetic distillation", flush=True)
        nano, summary, _ = train_temporal_plate_detector(
            nano_cfg,
            nano_path,
            teacher_model=base,
            teacher_ckpt_path=base_path,
            teacher_ckpt_sha256=base_prov["checkpoint_sha256"],
            shared_pool=shared_pool,
            repo_root=REPO_ROOT,
            training_script=TRAINING_SCRIPT,
            device=device,
        )
        assert_frozen(frozen)
        nano, validation = select_stage1(nano_path, nano, seed=20269641, device=device, clips=1 if dry else 16, dry_run=dry)
        nano_prov = verify_temporal_det_checkpoint_provenance(
            nano_path, repo_root=REPO_ROOT, teacher_ckpt_path=base_path, verify_source_hashes=True
        )
        report["nano"] = dict(config=asdict(nano_cfg), training=summary, validation=validation, provenance=nano_prov)
        hw = (256, 320) if dry else (448, 768)
        for name, model in (("base", base), ("nano", nano)):
            directory = out_dir / f"onnx_temporal_det_{name}"
            print(f"[Gen6] Export and pre-calibration parity: {name}", flush=True)
            report[name]["export"] = export_temporal_det_onnx(model, directory, input_hw=hw)
            report[name]["parity_before_calibration"] = verify_temporal_det_onnx_parity(
                model, directory, input_hw=hw, rollout_steps=1 if dry else 5
            )
        del shared_pool, _
        gc.collect()
        torch.cuda.empty_cache()
        calibration_pool = build_stage1_calibration_pool(
            image_hw=hw, num_frames=2 if dry else min(int(args.clip_frames), 8), seed=20260601, device=device
        )
        report["stage1_calibration"] = fit_stage1_post_int8_calibration(
            out_dir / "onnx_temporal_det_nano/temporal_det_nano.int8.onnx",
            calibration_pool=calibration_pool,
            num_clips=1 if dry else 24,
            num_batches=1 if dry else 16,
            batch_size=1 if dry else 4,
            seed=20260601,
            device=device,
            sidecar_path=cal_dir / "stage1_int8_cal.json",
        )
        del calibration_pool
        for name, model in (("base", base), ("nano", nano)):
            parity = verify_temporal_det_onnx_parity(
                model, out_dir / f"onnx_temporal_det_{name}", input_hw=hw, rollout_steps=1 if dry else 5
            )
            if (name == "nano" and parity["combined_int8_mb"] > 8.0) or parity["min_matched_queries"] < 298:
                raise AssertionError(f"Stage-1 size/query parity gate failed: {parity}")
            report[name]["parity_after_calibration"] = parity
        del nano, model
        gc.collect()
        torch.cuda.empty_cache()
        assert_frozen(frozen)
        artifacts.extend([base_path, nano_path, cal_dir / "stage1_int8_cal.json"])
        for directory in (out_dir / "onnx_temporal_det_base", out_dir / "onnx_temporal_det_nano"):
            artifacts.extend(sorted(directory.glob("*.onnx")))

    if run_stage2:
        proposal_model: TemporalPlateDetNet | None = base.eval() if base is not None else None
        if proposal_model is None:
            cand_s1 = args.stage1_ckpt.resolve() if args.stage1_ckpt is not None else (base_path if base_path.is_file() else None)
            if cand_s1 is not None:
                if not cand_s1.is_file():
                    raise FileNotFoundError(f"--stage1-ckpt not found: {cand_s1}")
                proposal_model = load_temporal_det_checkpoint(cand_s1, device=device).eval()
        s2_initial = args.initial_clips if args.initial_clips is not None else (1 if dry else 96)
        s2_max_items = (
            max(256, int(args.max_pool_clips) * 64)
            if args.max_pool_clips is not None
            else (256 if dry else 20480)
        )
        quad_cfg = QuadTrackTrainConfig(
            total_steps=2 if dry else args.quad_steps,
            batch_size=8 if dry else 144,
            base_ch=48,
            surround_ch=32,
            lr=1.3e-3,
            weight_decay=3e-4,
            max_pool_items=s2_max_items,
            clip_image_hw=(256, 320) if dry else (448, 768),
            clip_frames=2 if dry else int(args.clip_frames),
            initial_clips=s2_initial,
            refresh_clips_every=0 if dry else 28,
            refresh_clips_count=2,
            log_every=2 if dry else 200,
            init_seed=20260631,
            seed=20266331,
        )
        quad_path = out_dir / "plate_quad_track_v4.pt"
        print("[Gen6] Training Stage-2 with independent scene-bank proposals", flush=True)
        quad, summary = train_plate_quad_track_net(
            quad_cfg,
            quad_path,
            repo_root=REPO_ROOT,
            training_script=TRAINING_SCRIPT,
            device=device,
            detector_proposal_model=proposal_model,
        )
        assert_frozen(frozen)
        quad, quad_val_report = select_stage2(quad_path, quad, seed=20269631, device=device, dry_run=dry)
        report["quad"] = dict(
            config=asdict(quad_cfg),
            training=summary,
            validation=quad_val_report,
            provenance=verify_quad_track_checkpoint_provenance(quad_path, repo_root=REPO_ROOT, verify_source_hashes=True),
        )
        directory = out_dir / "onnx_det_v4"
        report["quad"]["export"] = export_quad_track_onnx(quad, directory, stem_name="plate_quad_track_v4")
        report["quad"]["parity_before_calibration"] = verify_quad_track_onnx_parity(quad, directory)
        pool = build_stage2_calibration_pool(
            seed=20260602,
            device=device,
            image_hw=(256, 320) if dry else (320, 576),
            num_frames=2 if dry else 4,
            max_items=256 if dry else 4096,
        )
        report["stage2_calibration"] = fit_stage2_post_int8_calibration(
            directory / "plate_quad_track_v4.int8.onnx",
            calibration_pool=pool,
            num_clips=1 if dry else 64,
            num_batches=2 if dry else 48,
            batch_size=16 if dry else 32,
            seed=20260602,
            device=device,
            sidecar_path=cal_dir / "stage2_int8_cal.json",
        )
        for suffix in (".onnx", ".int8.onnx"):
            shutil.copyfile(directory / f"plate_quad_track_v4{suffix}", directory / f"stage2{suffix}")
            assert compute_det_file_sha256(directory / f"stage2{suffix}") == compute_det_file_sha256(
                directory / f"plate_quad_track_v4{suffix}"
            )
        report["quad"]["parity_after_calibration"] = verify_quad_track_onnx_parity(quad, directory)
        assert_frozen(frozen)
        artifacts.extend([quad_path, cal_dir / "stage2_int8_cal.json"])
        artifacts.extend(sorted(directory.glob("*.onnx")))

    report["artifacts"] = {str(p): {"bytes": p.stat().st_size, "sha256": compute_det_file_sha256(p)} for p in artifacts}
    report["elapsed_s"] = time.perf_counter() - start
    if not dry:
        if run_stage1:
            for name in ("base", "nano"):
                validation = report[name]["validation"]
                metrics = validation[validation["selected"]]
                for key, floor in (
                    ("plate_recall_iou50_equal_cost", 0.65),
                    ("vehicle_recall_iou50_equal_cost", 0.60),
                    ("soft_cutout_iou", 0.65),
                ):
                    assert math.isfinite(metrics[key]) and metrics[key] >= floor, (name, key, metrics[key], floor)
                assert report[name]["training"]["skipped_nan_steps"] == 0, name
        if run_stage2:
            validation = report["quad"]["validation"]
            metrics = validation[validation["selected"]]
            for key, ceiling in (("corner_error_px", 4.0), ("plate_nll", 0.30)):
                assert math.isfinite(metrics[key]) and metrics[key] <= ceiling, (key, metrics[key], ceiling)
            for key in ("plate_recall_equal_cost", "null_rejection_equal_cost"):
                assert math.isfinite(metrics[key]) and metrics[key] >= 0.88, (key, metrics[key])
            for policy in ("0", "1", "2", "3"):
                calibration = report["stage2_calibration"]["policies"][policy]
                assert calibration["calibrated"] is True and calibration["support_bounded"] is True, (policy, calibration)
                assert calibration["support_quantile"] is not None and math.isfinite(calibration["support_quantile"]), (
                    policy,
                    calibration,
                )
                assert calibration["support_num_clips"] >= 30, (policy, calibration)
            assert report["quad"]["training"]["skipped_nan_steps"] == 0, "quad"
    manifest_name = (
        "gen6_stage1_manifest.json"
        if args.stage1_only
        else ("gen6_stage2_manifest.json" if args.stage2_only else "gen6_stage1_stage2_manifest.json")
    )
    manifest = out_dir / manifest_name
    manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Gen6] Completed in {report['elapsed_s']:.1f}s: {manifest}", flush=True)


if __name__ == "__main__":
    main()
