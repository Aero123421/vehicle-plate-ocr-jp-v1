#!/usr/bin/env python3
"""Pure-synthetic terminal/tracker calibration and deployed INT8 audit.

No calibration, model loading or output access occurs when requesting --help.
Run after all exported graphs and calibration sidecars have finished updating.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile


# Keep imports and the final pure verification from changing the frozen inventory.
sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = "web/models/gen6.json"
READOUT_PRECISION_CEILING = 48.0
SIZE_LIMIT_BYTES = 15 * 1024 * 1024
GRAPH_NAMES = (
    "temporal_det_nano", "vehicle_cutout_nano", "stage2", "frame_encoder",
    "memory_step", "memory_readout", "stopping_risk",
)
STATE_NAMES = ("mem_h", "mem_P", "mem_conflict", "color_h", "color_P",
               "pol_h", "pol_P", "pose_ema", "last_timestamp_s")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("web/models"),
                        help="Flat INT8 bundle, or nested training output (out/models).")
    parser.add_argument("--calibration-out", type=Path, default=Path(DEFAULT_CALIBRATION),
                        help="Policy JSON to write, or reuse with --audit-only.")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--audit-only", action="store_true",
                        help="Reuse the policy; validate, check size and smoke-test without fitting.")
    return parser.parse_args(argv)


def repository_path(raw: Path) -> Path:
    """Keep inputs, outputs and temporary files inside this checkout."""
    path = raw if raw.is_absolute() else REPO_ROOT / raw
    if ".." in path.parts or not path.is_relative_to(REPO_ROOT):
        raise ValueError("Paths must stay inside the repository")
    for ancestor in (path, *path.parents):
        if ancestor == REPO_ROOT:
            break
        if ancestor.is_symlink():
            raise ValueError("Symlinked artifact paths are forbidden")
    return path


def deployment_paths(models_dir):
    """Resolve the flat public bundle or the detector/OCR training directories."""
    directory = repository_path(models_dir)
    flat = (directory / "temporal_det_nano.int8.onnx").is_file()
    detector_dir = directory if flat else directory / "onnx_temporal_det_nano"
    stage2_dir = directory if flat else directory / "onnx_det_v4"
    ocr_dir = directory if flat else directory / "onnx_v2_nano"
    graphs = (
        detector_dir / "temporal_det_nano.int8.onnx",
        detector_dir / "vehicle_cutout_nano.int8.onnx",
        stage2_dir / "stage2.int8.onnx",
        *(ocr_dir / f"{name}.int8.onnx" for name in GRAPH_NAMES[3:]),
    )
    calibration_dir = directory if flat else directory.parent / "calibration"
    sidecars = tuple(calibration_dir / f"stage{i}_int8_cal.json" for i in (1, 2))
    for path in (*graphs, *sidecars):
        repository_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing INT8 deployment artifact: {path.relative_to(REPO_ROOT)}")
    return graphs, sidecars


def atomic_json(path: Path, value: dict) -> None:
    # Serialize first: invalid floats cannot replace an existing good policy.
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class CalibrationPlateSampler:
    """Supply only identities from the cal_holdout scene bank to the 2.5D engine.

    GPUSynthSequenceEngine has no domain_split argument. Using the partitioned
    bank here avoids silently using its default training identity sampler,
    including when the engine requests secondary TRACK_MIXED identities.
    """

    def __init__(self, bank, seed):
        from vpocr_synth.plate.models import PlateConfig, PlateSize, PlateText, PlateType

        if bank.domain_split != "cal_holdout":
            raise ValueError("Calibration identities must come from cal_holdout")
        self.seed = seed
        self.index = 0
        self.plates = []
        for row in bank.plate_meta:
            region, classification, hiragana, serial = row["plate_text_str"].split()
            digits = "".join(c for c in serial if c.isdecimal())
            text = PlateText(region, classification, hiragana, int(digits))
            config = PlateConfig(plate_type=PlateType(row["plate_type_str"]),
                                plate_size=PlateSize.LARGE if row["is_large"] else PlateSize.MEDIUM,
                                is_graphic=row["plate_type_str"] == "graphic")
            self.plates.append((text, config))
        if len(self.plates) < 2:
            raise ValueError("Calibration identity bank is empty")

    def sample(self, seed=None):
        if seed is None:
            seed = self.seed + self.index
            self.index += 1
        return self.plates[int(seed) % len(self.plates)]

    def sample_hard_negative_pair(self, seed=None):
        first = self.sample(seed)
        start = (self.seed if seed is None else int(seed)) + 1
        for offset in range(len(self.plates)):
            second = self.sample(start + offset)
            if first[0] != second[0]:
                return first, second
        raise ValueError("Calibration bank needs distinct identities")


def encode(runtime, crops, sides, geometry):
    import numpy as np

    return tuple(runtime.enc_session.run(None, {
        "crops": np.ascontiguousarray(crops, dtype=np.float32),
        "plate_sides": np.ascontiguousarray(sides, dtype=np.int64),
        "geom_meta": np.ascontiguousarray(geometry, dtype=np.float32),
    }))


def step(runtime, state, observation, geometry, timestamp):
    import numpy as np

    feed = dict(zip(STATE_NAMES, state))
    names = ("z", "lambda_prec", "z_color", "lambda_color", "pol_logit", "pol_prec")
    feed.update((name, np.ascontiguousarray(value[None], dtype=np.float32))
                for name, value in zip(names, observation))
    feed["geom_meta"] = np.ascontiguousarray(geometry[None], dtype=np.float32)
    feed["timestamp_s"] = np.array([timestamp], dtype=np.float32)
    return tuple(runtime.step_session.run(None, feed))


def _conformal_damping(nc, llr, vis, tau_nc=1.65):
    nc_excess = max(0.0, float(nc) - float(tau_nc))
    neg_llr = max(0.0, -float(llr) - 0.25)
    return math.exp(-min(20.0, 0.55 * nc_excess + 0.75 * neg_llr)) * (0.15 + 0.85 * float(vis))


def _scale_obs_damping(observation, damping):
    import numpy as np

    d = np.float32(damping)
    return tuple(
        np.asarray(value * d, dtype=np.float32) if i in (1, 3, 5) else np.asarray(value).copy()
        for i, value in enumerate(observation)
    )


def terminal_record(runtime, state, slots, plate_type, status, rgb=True, *, support=1.0, is_non_plate=False):
    import numpy as np
    from vpocr_ocr.session_v2 import _apply_sealed_precision_ceiling

    state = _apply_sealed_precision_ceiling(
        tuple(value.copy() for value in state), READOUT_PRECISION_CEILING)
    values = runtime.read_session.run(None, dict(zip(STATE_NAMES[:7], state[:7])))
    slot_logits, type_logits, _, readability, status_logits = (a[0] for a in values)
    return {
        "slot_logits": slot_logits, "plate_type_logits": type_logits,
        "status_logits": status_logits, "slot_readability_pred": readability,
        "mem_P": state[1][0].copy(), "mem_conflict": state[2][0].copy(),
        "color_P": state[4][0].copy(), "target_slots": np.asarray(slots, dtype=np.int64),
        "target_plate_type": int(plate_type), "target_status": int(status), "rgb": bool(rgb),
        "support": float(support),
        "is_non_plate": bool(is_non_plate),
    }


def collect_terminal_samples(runtime, pool, seed):
    import cv2
    import numpy as np
    import torch
    from vpocr_det.quad_cropper import extract_quad_geometry_meta
    from vpocr_det.scene_synth import QuadCropSynthConfig, QuadCropSynthesizer
    from vpocr_synth.gpu.sequence_engine import GPUSynthSequenceEngine

    engine = GPUSynthSequenceEngine(device="cpu", train_mode=False, curriculum_stage=4,
                                    seed=seed, sampler=CalibrationPlateSampler(pool.engine.bank, seed),
                                    out_hw=tuple(runtime.enc_session.get_inputs()[0].shape[-2:]))
    rng_s2 = np.random.default_rng(seed + 15000)
    records = []
    crisp = []
    seq_bank = []
    for batch_index in range(24):
        batch = engine.sample_batch(batch_size=16, seq_len=8, approach_prob=0.30)
        arrays = {key: value.detach().cpu().numpy() for key, value in batch.items()
                  if isinstance(value, torch.Tensor)}
        for b in range(16):
            crops, sides, geometry = (arrays[key][b] for key in ("crops", "plate_sides", "geom_meta"))
            observations = encode(runtime, crops, sides, geometry)
            tgt_st = int(arrays["target_status"][b])
            state = runtime._empty_state()
            for t in range(8):
                if arrays["frame_valid_mask"][b, t] > 0:
                    obs_t = tuple(a[t] for a in observations)
                    if tgt_st in (0, 2):
                        nc_t = float(rng_s2.uniform(0.15, 2.15))
                        vis_t = float(rng_s2.uniform(0.55, 0.99))
                        llr_t = float(rng_s2.uniform(-0.10, 4.20))
                    else:
                        nc_t = float(rng_s2.uniform(0.80, 3.60))
                        vis_t = float(rng_s2.uniform(0.25, 0.85))
                        llr_t = float(rng_s2.uniform(-2.50, 1.20))
                    damp_t = _conformal_damping(nc_t, llr_t, vis_t)
                    state = step(runtime, state, _scale_obs_damping(obs_t, damp_t),
                                 geometry[t], arrays["timestamps_s"][b, t])
            if tgt_st in (0, 2):
                sup_b = (
                    float(rng_s2.uniform(0.92, 0.995))
                    if (b % 2 == 0)
                    else float(rng_s2.uniform(0.45, 0.95))
                )
            else:
                sup_b = float(rng_s2.uniform(0.20, 0.88))
            records.append(terminal_record(runtime, state, arrays["target_slots"][b],
                           arrays["target_plate_type"][b], tgt_st,
                           rgb=not bool(np.all(arrays["is_monochrome_ir"][b])),
                           support=sup_b))
            if tgt_st == 0:
                seq_views = [(tuple(a[t] for a in observations), geometry[t])
                             for t in range(8) if arrays["frame_valid_mask"][b, t] > 0]
                if len(seq_views) >= 4:
                    seq_bank.append((seq_views, arrays["target_slots"][b].copy(),
                                     int(arrays["target_plate_type"][b]),
                                     not bool(np.all(arrays["is_monochrome_ir"][b]))))
            if len(crisp) < 64 and tgt_st == 0:
                # Clean canonical pixels, rather than a blurred frame labelled crisp.
                crop = arrays["canon_clean_crop"][b]
                rgb = (crop.transpose(1, 2, 0).clip(0, 1) * 255).round().astype(np.uint8)
                h, w = rgb.shape[:2]
                quad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
                geom = extract_quad_geometry_meta(quad, rgb)
                obs = encode(runtime, crop[None], sides[:1], geom[None])
                crisp.append((tuple(a[0] for a in obs), geom, arrays["target_slots"][b].copy(),
                              int(arrays["target_plate_type"][b])))
        print(f"Terminal cal_holdout: {batch_index + 1}/24 batches", flush=True)

    negatives = QuadCropSynthesizer(QuadCropSynthConfig(p_positive=0.0, domain_split="cal_holdout"))
    h, w = engine.out_hw
    for index in range(128):
        sample = negatives.synthesize_sample(seed=seed + 10000 + index)
        if sample.is_plate != 0.0:
            raise AssertionError("Non-plate calibration sample has a positive label")
        rgb = cv2.resize(sample.roi_rgb, (w, h))
        quad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
        geom = extract_quad_geometry_meta(quad, rgb)
        crop = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        observation = tuple(a[0] for a in encode(runtime, crop[None], np.array([2]), geom[None]))
        n_neg_steps = (1, 4, 8, 12)[index % 4]
        state = runtime._empty_state()
        for t in range(n_neg_steps):
            nc_n = float(rng_s2.uniform(1.10, 4.80))
            vis_n = float(rng_s2.uniform(0.18, 0.88))
            llr_n = float(rng_s2.uniform(-4.20, 0.65))
            damp_n = _conformal_damping(nc_n, llr_n, vis_n)
            state = step(runtime, state, _scale_obs_damping(observation, damp_n), geom, t / 15.0)
        neg_support = float(rng_s2.uniform(0.02, 0.65))
        records.append(
            terminal_record(
                runtime,
                state,
                np.full(12, runtime.vocab.pad_id),
                0,
                1,
                support=neg_support,
                is_non_plate=True,
            )
        )

    if len(crisp) < 2:
        raise RuntimeError("Insufficient readable cal_holdout identities for crisp splices")
    rng = np.random.default_rng(seed + 20000)
    for index in range(32):
        a = crisp[index % len(crisp)]
        candidates = [b for b in crisp if np.any(a[2] != b[2])]
        if not candidates:
            raise RuntimeError("Crisp splice requires distinct identity targets")
        b = candidates[int(rng.integers(len(candidates)))]
        if index % 2 == 0:
            # Exactly one changed latent slot; retain A's color and polarity.
            changed = np.flatnonzero(a[2] != b[2])
            slot = int(rng.choice(changed))
            spliced = [value.copy() for value in a[0]]
            spliced[0][slot] = b[0][0][slot]
            spliced[1][slot] = b[0][1][slot]
            splice_geom = a[1]
        else:
            spliced, splice_geom = b[0], b[1]
        state = runtime._empty_state()
        # Match the training crisp-dwell stressor, using only INT8 ORT evidence.
        for t in range(40):
            state = step(runtime, state, a[0] if t < 36 else spliced,
                         a[1] if t < 36 else splice_geom, t * 0.08)
        records.append(
            terminal_record(runtime, state, a[2], a[3], 2, support=float(rng_s2.uniform(0.82, 0.995)))
        )
    if len(seq_bank) < 2:
        raise RuntimeError("Insufficient readable cal_holdout sequences for jittered dwells")
    # Revisit multiple measured views of one identity to retain natural crop jitter.
    for index in range(48):
        seq_views, slots, plate_type, rgb = seq_bank[index % len(seq_bank)]
        state = runtime._empty_state()
        for t in range(96):
            observation, geometry = seq_views[t % len(seq_views)]
            if index % 2 == 1:
                damp_dw = _conformal_damping(
                    float(rng_s2.uniform(0.60, 2.10)),
                    float(rng_s2.uniform(0.20, 2.50)),
                    float(rng_s2.uniform(0.55, 0.92)),
                )
                observation = _scale_obs_damping(observation, damp_dw)
            state = step(runtime, state, observation, geometry, t * 0.08)
        records.append(
            terminal_record(
                runtime,
                state,
                slots,
                plate_type,
                0,
                rgb=rgb,
                support=float(rng_s2.uniform(0.75, 0.995)),
            )
        )
    for index in range(48):
        a = seq_bank[index % len(seq_bank)]
        candidates = [b for b in seq_bank if np.any(a[1] != b[1])]
        if not candidates:
            raise RuntimeError("Jittered splice requires distinct identity targets")
        b = candidates[int(rng.integers(len(candidates)))]
        slot = int(rng.choice(np.flatnonzero(a[1] != b[1]))) if index % 2 == 0 else None
        state = runtime._empty_state()
        for t in range(96):
            observation, geometry = a[0][t % len(a[0])]
            if t >= 84:
                other, other_geometry = b[0][t % len(b[0])]
                if slot is not None:
                    # Change one identity slot while retaining A's other evidence.
                    observation = [value.copy() for value in observation]
                    observation[0][slot] = other[0][slot]
                    observation[1][slot] = other[1][slot]
                else:
                    observation, geometry = other, other_geometry
            state = step(runtime, state, observation, geometry, t * 0.08)
        records.append(
            terminal_record(
                runtime,
                state,
                a[1],
                a[2],
                2,
                rgb=a[3] and b[3],
                support=float(rng_s2.uniform(0.75, 0.995)),
            )
        )
    print(f"Terminal samples: {len(records)} (384 sequences + 128 non-plates + 32 crisp splices"
          " + 48 jittered dwells + 48 jittered splices)", flush=True)
    return records


def fit_terminal(records, runtime):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from vpocr_ocr.session_v2 import TerminalCalibrationV2, _terminal_tables

    def tensor(key):
        return torch.as_tensor(np.stack([r[key] for r in records]), dtype=torch.float64)

    slots = torch.nan_to_num(tensor("slot_logits"), nan=0.0, posinf=50.0, neginf=-50.0)
    precision = torch.nan_to_num(tensor("mem_P"), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    slots = torch.where(precision.mean(-1)[..., None] > 0, slots, 0)
    types = torch.nan_to_num(tensor("plate_type_logits"), nan=0.0, posinf=50.0, neginf=-50.0)
    color_supported = tensor("color_P").mean(-1) > 0
    types = torch.where(color_supported[:, None], types, 0)
    targets = tensor("target_slots").long()
    type_targets = tensor("target_plate_type").long()
    status_targets = tensor("target_status").long()
    readable = status_targets == 0
    rgb = readable & torch.tensor([r["rgb"] for r in records]) & color_supported
    if not bool(readable.any()):
        raise RuntimeError("No readable synthetic calibration sequences")

    def slot_nll(parameters):
        temperature = parameters[0].clamp(0.5, 3.0)
        loss = F.cross_entropy((slots[readable] / temperature).flatten(0, 1),
                               targets[readable].flatten())
        if bool(rgb.any()):
            loss = loss + F.cross_entropy(types[rgb] / temperature, type_targets[rgb])
        return loss

    def minimize(objective, initial, project):
        parameters = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        baseline = float(objective(parameters).detach())
        if not math.isfinite(baseline):
            raise ValueError("Non-finite identity calibration NLL")
        optimizer = torch.optim.LBFGS([parameters], max_iter=80, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = objective(parameters)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
            candidate = project(parameters.detach())
            fitted = float(objective(candidate))
        except (RuntimeError, ValueError):
            candidate, fitted = torch.tensor(initial, dtype=torch.float64), baseline
        if not math.isfinite(fitted) or not torch.isfinite(candidate).all() or fitted >= baseline:
            candidate, fitted = torch.tensor(initial, dtype=torch.float64), baseline
        return candidate, baseline, fitted

    temp, before, after = minimize(slot_nll, [1.0], lambda p: p.clamp(0.5, 3.0))
    temperature = float(temp[0])
    print(f"Slot/RGB type NLL: {before:.8f} -> {after:.8f}; temperature={temperature:.8f}")

    # Exact serial partition (including the all-PAD alternative), as in the
    # deployed terminal equations. Compute once: temperature is now fixed.
    lp = F.log_softmax(slots / temperature, dim=-1)
    serials = torch.as_tensor(_terminal_tables(runtime.grammar)[3], dtype=torch.long)
    serial_partitions = []
    for chunk in lp.split(32):
        scores = sum(chunk[:, 8 + i, serials[:, i]] for i in range(4))
        serial_partitions.append(torch.logaddexp(torch.logsumexp(scores, -1),
                                                 chunk[:, 8:12, runtime.vocab.pad_id].sum(-1)))
    zs_aug = torch.cat(serial_partitions)
    status = torch.nan_to_num(tensor("status_logits"), nan=0.0, posinf=50.0, neginf=-50.0)
    conflict = torch.nan_to_num(tensor("mem_conflict"), nan=0.0, posinf=0.0,
                               neginf=0.0).clamp_min(0).amax(-1)
    visibility = torch.nan_to_num(tensor("slot_readability_pred")).clamp(0, 1).mean(-1)
    support = tensor("support").clamp(1e-6, 1.0 - 1e-6)
    non_plate = torch.tensor([bool(r.get("is_non_plate", False)) for r in records], dtype=torch.bool)
    evidence = (precision > 0).flatten(1).any(-1)

    # Saerens-Latinne-Decaestecker label-shift correction to the episode prior.
    pi_target = torch.tensor([0.82, 0.16, 0.02], dtype=torch.float64)
    counts = torch.bincount(status_targets, minlength=3).to(torch.float64)
    if torch.any(counts <= 0):
        raise RuntimeError("All three terminal status classes must appear in cal_holdout")
    sample_weights = pi_target[status_targets] / (counts[status_targets] / counts.sum())
    sample_weights = sample_weights / sample_weights.mean()

    def eval_weighted_null_nll(nt, nb, rw, sw, mb, cw):
        adjusted = torch.stack((status[:, 0], status[:, 1],
                                status[:, 2] + mb + cw * conflict), -1)
        log_status = F.log_softmax(adjusted, dim=-1)
        log_single = torch.logsumexp(log_status[:, :2], dim=-1)
        odds = (status[:, 0] - status[:, 1] + zs_aug + nb
                + rw * visibility + sw * support) / nt
        # P(READABLE)=support*(1-P(mixed))*sigmoid(odds), identity=1.
        # P(UNREADABLE)=1-P(mixed)-P(READABLE), computed in log space.
        log_readable = support.log() + log_single + F.logsigmoid(odds)
        log_unreadable = log_single + torch.logaddexp(torch.log1p(-support),
                                                      support.log() + F.logsigmoid(-odds))
        log_readable = torch.where(evidence, log_readable, -torch.inf)
        log_unreadable = torch.where(evidence, log_unreadable, log_single)
        log_probs = torch.stack((log_readable, log_unreadable, log_status[:, 2]), -1)
        per_sample_nll = -log_probs.gather(1, status_targets[:, None]).squeeze(1).clamp_min(math.log(1e-12))
        loss = (sample_weights * per_sample_nll).mean()
        if bool(non_plate.any()):
            p_np = torch.where(evidence[non_plate], log_readable[non_plate].exp(), torch.zeros_like(support[non_plate]))
            odds_stress = (status[non_plate, 0] - status[non_plate, 1] + zs_aug[non_plate]
                           + nb + rw * visibility[non_plate] + sw * 0.96) / nt
            p_stress = torch.where(
                evidence[non_plate],
                0.96 * log_single[non_plate].exp() * torch.sigmoid(odds_stress),
                torch.zeros_like(support[non_plate]),
            )
            fp_barrier = (
                0.25 * F.softplus((p_np - 0.55) / 0.025).square().mean()
                + 0.35 * F.softplus((p_stress - 0.90) / 0.02).square().mean()
                + 2.0 * F.softplus((p_stress.amax() - 0.94) / 0.01).square()
            )
            loss = loss + fp_barrier
        return loss

    def decode_null_params(u):
        return (0.5 + 2.5 * torch.sigmoid(u[0]), u[1], F.softplus(u[2]),
                F.softplus(u[3]), -10.0 * torch.sigmoid(u[4]), F.softplus(u[5]))

    identity = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    baseline_loss = float(eval_weighted_null_nll(*identity))
    if not math.isfinite(baseline_loss):
        raise ValueError("Non-finite identity calibration NLL")
    u_init = [math.log(0.2 / 0.8), 0.0, -4.0, -4.0, -5.0, -3.0]
    u = torch.tensor(u_init, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([u], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = eval_weighted_null_nll(*decode_null_params(u))
        loss.backward()
        return loss

    optimizer.step(closure)
    nt, nb, rw, sw, mb, cw = decode_null_params(u.detach())
    fitted_loss = float(eval_weighted_null_nll(nt, nb, rw, sw, mb, cw))
    if (not torch.isfinite(torch.stack((nt, nb, rw, sw, mb, cw))).all()
            or not math.isfinite(fitted_loss) or fitted_loss >= baseline_loss):
        nt, nb, rw, sw, mb, cw = identity
        fitted_loss = baseline_loss
    print(f"Three-way terminal weighted NLL: {baseline_loss:.8f} -> {fitted_loss:.8f}")
    return TerminalCalibrationV2(temperature=temperature, null_temperature=float(nt),
        null_bias=float(nb), readability_weight=float(rw),
        support_weight=float(sw), identity_probability=1.0,
        mixed_logit_scale=1.0, mixed_bias=float(mb), conflict_weight=float(cw),
        readout_precision_ceiling=READOUT_PRECISION_CEILING)


def fit_tracker(pool, graphs, stage2_calibration):
    import numpy as np
    import onnxruntime as ort
    from vpocr_det.detector import UnifiedTemporalDetORT, _match_query_nll, decode_temporal_det_predictions
    from vpocr_det.pipeline import _inference_model
    from vpocr_det.quad_cropper import refine_and_crop_detections
    from vpocr_det.tracker import TrackerCalibration
    from vpocr_det.types import CLS_PLATE, ObservationProvenance
    import torch

    detector = UnifiedTemporalDetORT(graphs[0], graphs[1])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    stage2 = ort.InferenceSession(str(graphs[2]), options,
                                  providers=["CPUExecutionProvider"])
    quad_model = _inference_model(stage2, torch.device("cpu"), stage2=True)
    pool.ingest_clips(12)
    hits = total = 0
    for item in pool.items:
        hidden = None
        for t, frame in enumerate(item["frames_u8"]):
            rgb = frame.permute(1, 2, 0).numpy()
            raw, hidden = decode_temporal_det_predictions(detector, rgb, device="cpu",
                input_hw=detector.input_hw, hidden_state=hidden, plate_thresh=0.22,
                enable_far_zoom=True, frame_key=f"cal:{total}:{t}")
            refined = refine_and_crop_detections(quad_model, rgb, raw.plates, device=torch.device("cpu"),
                min_is_plate_prob=0.34, vehicles=raw.vehicles)
            target = item["targets"][t]
            boxes = target["boxes"][target["labels"] == CLS_PLATE].numpy()
            total += len(boxes)
            if len(boxes) and refined:
                predicted = np.asarray([p.bbox_xyxy for p in refined], dtype=np.float64)
                lo = np.maximum(boxes[:, None, :2], predicted[None, :, :2])
                hi = np.minimum(boxes[:, None, 2:], predicted[None, :, 2:])
                intersection = np.maximum(hi - lo, 0).prod(-1)
                area_gt = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(-1)
                area_pred = np.maximum(predicted[:, 2:] - predicted[:, :2], 0).prod(-1)
                iou = intersection / np.maximum(area_gt[:, None] + area_pred - intersection, 1e-12)
                # Maximize the number of >=0.5 IoU matches before breaking ties.
                rows, cols = _match_query_nll(-(iou >= 0.5).astype(float) - iou * 1e-6)
                hits += int((iou[rows, cols] >= 0.5).sum())
    if total == 0:
        raise RuntimeError("No plate GT in the 12 cal_holdout detection clips")
    p_D = float(np.clip(hits / total, 0.65, 0.95))
    sidecar = json.loads(stage2_calibration.read_text())
    prior = np.asarray(sidecar["pi_train"], dtype=np.float64)
    if prior.shape != (4,) or not np.isfinite(prior).all() or np.any((prior <= 0) | (prior >= 1)):
        raise ValueError("Stage-2 pi_train must contain four finite probabilities in (0,1)")
    policies = (ObservationProvenance.DETECTOR_FULL, ObservationProvenance.DETECTOR_FAR_ZOOM,
                ObservationProvenance.SEARCH_MEASUREMENT, ObservationProvenance.STANDALONE_PRECROP)
    raw_odds = prior / (1.0 - prior)
    # Stage-2 already removes its crop class prior from plate_null_llr. Only
    # relative policy rates modulate the spatial Poisson birth prior here.
    policy_odds = np.clip(0.05 * (raw_odds / np.mean(raw_odds)), 0.02, 0.08)
    print(f"Detection availability: {hits}/{total}={hits/total:.8f}; clamped p_D={p_D:.8f}")
    return TrackerCalibration(detection_probability=p_D, birth_odds=float(policy_odds[0]),
                              policy_birth_odds={str(p): float(o) for p, o in zip(policies, policy_odds)})


def validate_policy(policy):
    from vpocr_det.tracker import TrackerCalibration
    from vpocr_ocr.session_v2 import SessionConfigV2, TerminalCalibrationV2

    if not isinstance(policy, dict) or set(policy) != {"terminal", "tracker", "session", "pipeline"}:
        raise ValueError("Invalid unified Gen-6 calibration schema")
    if not all(isinstance(section, dict) for section in policy.values()):
        raise ValueError("Calibration sections must be objects")
    allowed = {"plate_thresh", "min_is_plate_prob", "enable_far_zoom", "heartbeat_interval",
               "max_heartbeat_interval", "overload_track_threshold", "max_retired_history"}
    if policy["pipeline"].keys() - allowed:
        raise ValueError("Invalid calibration pipeline options")
    terminal = TerminalCalibrationV2(**policy["terminal"])
    TrackerCalibration(**policy["tracker"])
    config = SessionConfigV2(**policy["session"], calibration=terminal)
    if config.risk_calibrated:
        raise ValueError("Calibration requires terminal-only readouts")
    json.dumps(policy, allow_nan=False)


def fit_session_costs(records, terminal, runtime):
    import numpy as np
    from vpocr_ocr.session_v2 import SessionConfigV2, build_decision_v2_from_arrays

    def eval_costs(ca_ser, ca_up, ca_full):
        cfg = SessionConfigV2(
            C_wrong=1.0, C_abstain=ca_up,
            field_abstain_costs={"serial": ca_ser, "region": ca_up, "classification": ca_up, "hiragana": ca_up},
            C_wrong_full=1.0, C_abstain_full=ca_full,
            calibration=terminal,
        )
        n_acc = n_ser_ok = n_up_ver = n_up_ok = n_full_ver = n_full_ok = n_fp_np = 0
        field_slices = {"region": slice(0, 4), "classification": slice(4, 7), "hiragana": slice(7, 8)}
        for r in records:
            dec = build_decision_v2_from_arrays(
                slot_logits=r["slot_logits"], plate_type_logits=r["plate_type_logits"],
                polarity_logit=0.0, slot_readability_pred=r["slot_readability_pred"],
                status_logits=r["status_logits"], mem_P=r["mem_P"], mem_conflict=r["mem_conflict"],
                color_P=r["color_P"], pol_P=1.0, num_frames_seen=8,
                vocab=runtime.vocab, grammar=runtime.grammar, config=cfg,
                quad_prob=r["support"],
            )
            if r["is_non_plate"]:
                if dec.status == "ACCEPTED":
                    n_fp_np += 1
            elif r["target_status"] == 0 and dec.status == "ACCEPTED":
                n_acc += 1
                pred_ids = runtime.vocab.encode(dec.plate_text)
                if np.array_equal(pred_ids[8:12], r["target_slots"][8:12]):
                    n_ser_ok += 1
                for fn, sl in field_slices.items():
                    if fn in dec.verified_fields:
                        n_up_ver += 1
                        if np.array_equal(pred_ids[sl], r["target_slots"][sl]):
                            n_up_ok += 1
                if dec.confidence_tier == "FULL_4FIELD":
                    n_full_ver += 1
                    if np.array_equal(pred_ids, r["target_slots"]):
                        n_full_ok += 1
        ser_prec = n_ser_ok / max(1, n_acc)
        up_prec = n_up_ok / max(1, n_up_ver)
        full_prec = n_full_ok / max(1, n_full_ver)
        return n_fp_np, n_acc, ser_prec, n_up_ver, up_prec, n_full_ver, full_prec

    best_ser = 0.05
    for cand in (0.50, 0.48, 0.45, 0.40, 0.35, 0.30, 0.25, 0.15, 0.05):
        fp_np, n_acc, ser_prec, *_ = eval_costs(cand, 0.25, 0.05)
        if fp_np == 0 and n_acc >= 10 and ser_prec >= 0.980:
            best_ser = cand
            break

    best_up = 0.05
    for cand in (0.32, 0.28, 0.25, 0.22, 0.18, 0.12, 0.05):
        if cand > best_ser:
            continue
        fp_np, _, _, n_up_ver, up_prec, *_ = eval_costs(best_ser, cand, 0.05)
        if fp_np == 0 and n_up_ver >= 10 and up_prec >= 0.850:
            best_up = cand
            break

    best_full = 0.05
    for cand in (0.30, 0.25, 0.22, 0.18, 0.14, 0.10, 0.05):
        if cand > best_ser:
            continue
        fp_np, _, _, _, _, n_full_ver, full_prec = eval_costs(best_ser, best_up, cand)
        if fp_np == 0 and n_full_ver >= 5 and full_prec >= 0.800:
            best_full = cand
            break

    fp_np, n_acc, ser_prec, n_up_ver, up_prec, n_full_ver, full_prec = eval_costs(best_ser, best_up, best_full)
    print(
        f"Calibrated session costs: C_abstain_serial={best_ser:.3f} (prec={ser_prec:.4f}, n={n_acc}), "
        f"C_abstain_upper={best_up:.3f} (prec={up_prec:.4f}), "
        f"C_abstain_full={best_full:.3f} (prec={full_prec:.4f}, n={n_full_ver}), fp_nonplate={fp_np}"
    )
    return {
        "risk_calibrated": False,
        "C_wrong": 1.0,
        "C_abstain": float(best_up),
        "field_abstain_costs": {
            "serial": float(best_ser),
            "region": float(best_up),
            "classification": float(best_up),
            "hiragana": float(best_up),
        },
        "C_wrong_full": 1.0,
        "C_abstain_full": float(best_full),
    }


def calibrate(seed, graphs, sidecars):
    from vpocr_det.trainer import build_stage1_calibration_pool
    from vpocr_ocr.onnx_export_v2 import OrtMFPlateOCRSessionV2

    pool = build_stage1_calibration_pool(image_hw=(448, 768), num_frames=4,
                                        seed=seed + 100, device="cpu")
    runtime = OrtMFPlateOCRSessionV2(graphs[3].parent, use_int8=True,
                                    providers=["CPUExecutionProvider"], intra_op_threads=1)
    records = collect_terminal_samples(runtime, pool, seed)
    terminal = fit_terminal(records, runtime)
    session_costs = fit_session_costs(records, terminal, runtime)
    tracker = fit_tracker(pool, graphs, sidecars[1])
    policy = {"terminal": asdict(terminal), "tracker": asdict(tracker),
              "session": session_costs,
              "pipeline": {"plate_thresh": 0.18, "enable_far_zoom": True,
                           "heartbeat_interval": 6, "max_heartbeat_interval": 12}}
    validate_policy(policy)
    return policy


def smoke_test(seed, graphs, sidecars, calibration_path):
    """Exercise the public factory and MP4 decoding on two procedural frames.

    Stage the chosen graphs as a flat bundle so both supported layouts exercise
    exactly the same public predictor and selected policy. No training runs here.
    """
    import cv2
    import numpy as np
    import vehicle_plate_ocr_jp_v1

    with tempfile.TemporaryDirectory(prefix=".calibration-smoke-", dir=REPO_ROOT) as directory:
        root = Path(directory)
        bundle = root / "web/models"
        bundle.mkdir(parents=True)
        for source in (*graphs, *sidecars):
            os.link(source, bundle / source.name)
        shutil.copyfile(calibration_path, bundle / "gen6.json")
        predictor = vehicle_plate_ocr_jp_v1.create_predictor(root=root)
        mp4_path = root / "synthetic.mp4"
        height, width = 96, 160
        rng = np.random.default_rng(seed + 500)
        writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 15.0, (width, height))
        try:
            if not writer.isOpened():
                raise RuntimeError("Cannot open synthetic MP4 writer")
            for index in range(2):
                frame = rng.integers(18, 28, (height, width, 3), dtype=np.uint8)
                cv2.rectangle(frame, (24 + index, 24), (136 + index, 84), (120, 110, 100), -1)
                cv2.rectangle(frame, (58 + index, 64), (102 + index, 78), (240, 240, 240), -1)
                cv2.putText(frame, "1234", (60 + index, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (20, 20, 20), 1)
                writer.write(frame)
        finally:
            writer.release()
        predictions = predictor.predict_clip(mp4_path)
        if not isinstance(predictions, list):
            raise AssertionError("Public predictor must return a list")
    print(f"Synthetic MP4 smoke passed: {len(predictions)} terminal predictions; zero streaming readouts.")


def audit_size(graphs, sidecars, calibration_path):
    total = 0
    for path in (*graphs, *sidecars, calibration_path):
        size = path.stat().st_size
        total += size
        print(f"{path.relative_to(REPO_ROOT).as_posix()}: {size} bytes ({size / 1024**2:.9f} MiB)")
    print(f"TOTAL: {total} bytes ({total / 1024**2:.9f} MiB); limit {SIZE_LIMIT_BYTES} bytes (15.000 MiB)")
    if total > SIZE_LIMIT_BYTES:
        raise ValueError(f"Deployed INT8 size exceeds 15.000 MiB: {total} bytes")
    return total


def clean_bytecode():
    for relative in ("src", "tools", "tests", "out/models", "out/calibration", "web/models"):
        base = REPO_ROOT / relative
        if base.is_symlink():
            raise ValueError(f"Refusing symlinked cleanup root: {relative}")
        for directory, dirs, files in os.walk(base, followlinks=False):
            for name in list(dirs):
                path = Path(directory) / name
                if name == "__pycache__" and not path.is_symlink():
                    shutil.rmtree(path)
                    dirs.remove(name)
            for name in files:
                if name.endswith((".pyc", ".pyo")):
                    (Path(directory) / name).unlink()


def main(argv=None):
    args = parse_args(argv)
    calibration_path = repository_path(args.calibration_out)
    if calibration_path.suffix != ".json":
        raise ValueError("Calibration output must be a JSON file")
    graphs, sidecars = deployment_paths(args.models_dir)
    sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    import cv2
    import torch

    previous_threads = torch.get_num_threads()
    previous_cv_threads = cv2.getNumThreads()
    try:
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        if args.audit_only:
            validate_policy(json.loads(calibration_path.read_text(encoding="utf-8")))
            print(f"Reusing {calibration_path.relative_to(REPO_ROOT)}")
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed)
                with torch.inference_mode(False), torch.enable_grad():
                    atomic_json(calibration_path, calibrate(args.seed, graphs, sidecars))
        audit_size(graphs, sidecars, calibration_path)
        with torch.inference_mode():
            smoke_test(args.seed, graphs, sidecars, calibration_path)
    finally:
        torch.set_num_threads(previous_threads)
        cv2.setNumThreads(previous_cv_threads)
        clean_bytecode()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
