#!/usr/bin/env python3
"""Gen-6 pure-synthetic scratch training, innovation calibration and four-graph export."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import sys
import os

# Help and invalid arguments exit before importing training dependencies.
sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "out/models"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Run 4-step smoke test into --out-dir")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--base-steps", type=int, default=5200)
    parser.add_argument("--nano-steps", type=int, default=5800)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--base-init-seed", type=int, default=20260411)
    parser.add_argument("--nano-init-seed", type=int, default=20260412)
    return parser.parse_args(argv)


if __name__ == "__main__":
    parse_args()

sys.path.insert(0, str(REPO_ROOT / "src"))
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from vpocr_det.quad_cropper import extract_quad_geometry_meta
from vpocr_det.scene_synth import QuadCropSynthConfig, QuadCropSynthesizer
from vpocr_ocr.losses_v2 import (
    LossWeightsV2,
    OCRLossV2,
    _freeze_bn_running_stats,
    _safe_bce_prob_v2,
    compute_effective_slot_readability_v2,
)
from vpocr_ocr.model_v2 import MFJPPlateOCRV2, MemoryStateV2, ModelConfigV2
from vpocr_ocr.onnx_export_v2 import (
    export_ocr_v2_to_onnx,
    quantize_onnx_v2_for_cpu,
    verify_onnx_patent_hygiene,
    verify_onnx_v2_parity,
)
from vpocr_ocr.trainer_v2 import (
    FORBIDDEN_REAL_DATA_MARKERS,
    PROVENANCE_SCHEMA_VERSION,
    ModelEMAV2,
    TrainConfigV2,
    _move_batch_v2_to_device,
    build_synthetic_provenance_manifest,
    compute_file_sha256,
    instantiate_deterministic_scratch_model,
    save_checkpoint_v2,
    verify_checkpoint_provenance,
)
from vpocr_ocr.vocab import NUM_SLOTS, PlateVocab
from vpocr_synth.gpu.homography_fast import REGIME_NAMES
from vpocr_synth.gpu.sequence_engine import GPUSynthSequenceEngine


TRAINING_SCRIPT_REL = "tools/train_ocr.py"
BASE_DATA_SEED = 610_001
NANO_DATA_SEED = 620_001
VAL_SEED = 750_001


def _cosine_lr(step: int, total_steps: int, warmup: int, max_lr: float, min_lr: float) -> float:
    w = max(1, min(warmup, max(1, total_steps // 2)))
    if step < w:
        return max_lr * float(step + 1) / float(w)
    prog = min(max(float(step - w) / float(max(1, total_steps - w)), 0.0), 1.0)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))


def _stage_for_step(step: int, total_steps: int, fracs: tuple[float, float, float]) -> int:
    f = float(step) / float(max(1, total_steps))
    if f < fracs[0]:
        return 1
    if f < fracs[1]:
        return 2
    if f < fracs[2]:
        return 3
    return 4


def build_synthetic_hard_negative_bank(
    num_samples: int,
    out_hw: tuple[int, int],
    device: torch.device,
    seed: int = 999_111,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize `(N, 3, H, W)` non-plate procedural barrier/placard/sign/grille crops."""
    qsynth = QuadCropSynthesizer(QuadCropSynthConfig(p_positive=0.0, domain_split="train"))
    crops_list: list[np.ndarray] = []
    geoms_list: list[np.ndarray] = []
    h_out, w_out = out_hw
    for i in range(num_samples):
        sample = qsynth.synthesize_sample(seed=seed + i)
        assert sample.is_plate == 0.0
        resized = cv2.resize(sample.roi_rgb, (w_out, h_out), interpolation=cv2.INTER_AREA)
        gm = extract_quad_geometry_meta(None, resized)
        crops_list.append(resized.transpose(2, 0, 1).astype(np.float32) / 255.0)
        geoms_list.append(gm.astype(np.float32))
    crops_t = torch.from_numpy(np.stack(crops_list, axis=0)).to(device)
    geoms_t = torch.from_numpy(np.stack(geoms_list, axis=0)).to(device)
    return crops_t, geoms_t


def compute_hard_negative_aux_loss(
    model: MFJPPlateOCRV2,
    neg_crops_bank: torch.Tensor,
    neg_geoms_bank: torch.Tensor,
    gen: torch.Generator,
    batch_n: int = 16,
    use_amp: bool = False,
) -> torch.Tensor:
    """Supervise `frame_valid_gate -> 0`, `status -> UNREADABLE`, `readability -> 0` on non-plates."""
    dev = neg_crops_bank.device
    total_bank = neg_crops_bank.shape[0]
    idx = torch.randint(0, total_bank, (batch_n,), generator=gen).to(dev)
    c_neg = neg_crops_bank.index_select(0, idx)
    g_neg = neg_geoms_bank.index_select(0, idx)
    s_neg = torch.full((batch_n,), 2, dtype=torch.long, device=dev)

    with _freeze_bn_running_stats(model):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            enc = model.encode_frames(c_neg, s_neg, g_neg)
        st0 = model.init_memory(batch_size=batch_n, device=dev, dtype=torch.float32)
        st1 = model.step_memory(
            mem_h=st0.mem_h,
            mem_P=st0.mem_P,
            mem_conflict=st0.mem_conflict,
            color_h=st0.color_h,
            color_P=st0.color_P,
            pol_h=st0.pol_h,
            pol_P=st0.pol_P,
            pose_ema=st0.pose_ema,
            last_timestamp_s=st0.last_timestamp_s,
            z=enc.z.float(),
            lambda_prec=enc.lambda_prec.float(),
            z_color=enc.z_color.float(),
            lambda_color=enc.lambda_color.float(),
            pol_logit=enc.pol_logit.float(),
            pol_prec=enc.pol_prec.float(),
            geom_meta=g_neg.float(),
            timestamp_s=torch.zeros((batch_n,), dtype=torch.float32, device=dev),
        )
        ro = model.decode_memory(
            st1.mem_h,
            st1.mem_P,
            st1.mem_conflict,
            st1.color_h,
            st1.color_P,
            st1.pol_h,
            st1.pol_P,
        )

    fv_bce = F.binary_cross_entropy_with_logits(
        enc.frame_valid_logit.float(),
        torch.zeros_like(enc.frame_valid_logit.float()),
    )
    gate_pen = enc.frame_valid_gate.float().pow(2).mean()
    tgt_unread = torch.ones((batch_n,), dtype=torch.long, device=dev)
    stat_ce = F.cross_entropy(ro.status_logits.float(), tgt_unread)
    read_bce = _safe_bce_prob_v2(
        ro.slot_readability_pred.float(),
        torch.zeros_like(ro.slot_readability_pred.float()),
    ).mean()
    return fv_bce + gate_pen + 0.5 * stat_ce + 0.5 * read_bce


def compute_crisp_dwell_splice_aux_loss(
    model: MFJPPlateOCRV2,
    seq_out: Any,
    batch: dict[str, Any],
    gen: torch.Generator,
    max_pairs: int = 8,
) -> torch.Tensor:
    """Supervise `TRACK_MIXED` & `mem_conflict >= 1.65` on 36-frame crisp dwell + 1-slot/full splice."""
    dev = seq_out.frame_z.device
    readable_mask = (batch["target_status"] == 0) & (
        batch["slot_readability"].mean(dim=(1, 2)) > 0.48
    )
    valid_idx = torch.nonzero(readable_mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() < 2:
        return seq_out.slot_logits.sum() * 0.0

    k = min(int(max_pairs), int(valid_idx.numel()))
    idx_a = valid_idx[:k]
    idx_b = torch.roll(valid_idx, shifts=1)[:k]

    tgt_a = batch["target_slots"].index_select(0, idx_a)  # (K, 12)
    tgt_b = batch["target_slots"].index_select(0, idx_b)  # (K, 12)
    diff_mask = tgt_a != tgt_b  # (K, 12)
    has_diff = diff_mask.any(dim=-1)
    if not bool(has_diff.any()):
        return seq_out.slot_logits.sum() * 0.0

    best_frame = batch["slot_readability"].mean(-1).argmax(-1)
    precision = seq_out.frame_lambda_prec.float().mean((-1, -2))
    readable_prec = precision[torch.arange(len(precision), device=dev), best_frame] > .02
    has_diff = has_diff & readable_prec[idx_a] & readable_prec[idx_b]
    if not bool(has_diff.any()):
        return seq_out.slot_logits.sum() * 0.0

    idx_a = idx_a[has_diff]
    idx_b = idx_b[has_diff]
    diff_mask = diff_mask[has_diff]
    k_eff = int(idx_a.numel())

    # Pick sharpest frame of vehicle A and vehicle B
    best_ta = batch["slot_readability"].index_select(0, idx_a).mean(dim=-1).argmax(dim=-1)
    best_tb = batch["slot_readability"].index_select(0, idx_b).mean(dim=-1).argmax(dim=-1)

    za = seq_out.frame_z[idx_a, best_ta].float()
    lama = seq_out.frame_lambda_prec[idx_a, best_ta].float()
    zca = seq_out.frame_z_color[idx_a, best_ta].float().detach()
    lamca = seq_out.frame_lambda_color[idx_a, best_ta].float().detach()
    pola = seq_out.frame_pol_logit[idx_a, best_ta].float().detach()
    polpa = seq_out.frame_pol_prec[idx_a, best_ta].float().detach()
    gma = batch["geom_meta"][idx_a, best_ta].float().clone()

    zb = seq_out.frame_z[idx_b, best_tb].float()
    lamb = seq_out.frame_lambda_prec[idx_b, best_tb].float()
    zcb = seq_out.frame_z_color[idx_b, best_tb].float()
    lamcb = seq_out.frame_lambda_color[idx_b, best_tb].float()
    polb = seq_out.frame_pol_logit[idx_b, best_tb].float()
    polpb = seq_out.frame_pol_prec[idx_b, best_tb].float()
    gmb = batch["geom_meta"][idx_b, best_tb].float().clone()

    # Uniform slot-group sampling covers region, class, kana and serial identity changes.
    groups = torch.tensor([0]*4 + [1]*3 + [2] + [3]*4, device=dev)
    group_available = torch.stack([diff_mask[:, groups == g].any(-1) for g in range(4)], -1)
    group_u = torch.rand((k_eff, 4), generator=gen).to(dev)
    chosen_group = (group_u * group_available).argmax(-1)
    slot_u = torch.rand((k_eff, NUM_SLOTS), generator=gen).to(dev)
    chosen_diff_idx = (slot_u * diff_mask * (groups[None] == chosen_group[:, None])).argmax(-1)
    one_slot_mask = F.one_hot(chosen_diff_idx, num_classes=NUM_SLOTS).bool()
    use_one_slot = (torch.rand((k_eff, 1), generator=gen) < .70).to(dev)
    swap_slot_mask = torch.where(use_one_slot, one_slot_mask, diff_mask).unsqueeze(-1)

    z_sp = torch.where(swap_slot_mask, zb, za)
    lam_sp = torch.where(swap_slot_mask, lamb, lama)
    zc_sp = torch.where(use_one_slot, zca, zcb)
    lamc_sp = torch.where(use_one_slot, lamca, lamcb)
    pol_sp = torch.where(use_one_slot.squeeze(-1), pola, polb)
    polp_sp = torch.where(use_one_slot.squeeze(-1), polpa, polpb)

    st = model.init_memory(batch_size=k_eff, device=dev, dtype=torch.float32)
    n_dwell = int(torch.randint(36, 41, (), generator=gen).item())
    for t_d in range(n_dwell):
        with torch.set_grad_enabled(t_d >= n_dwell - 4):
            st = model.step_memory(
                mem_h=st.mem_h,
                mem_P=st.mem_P,
                mem_conflict=st.mem_conflict,
                color_h=st.color_h,
                color_P=st.color_P,
                pol_h=st.pol_h,
                pol_P=st.pol_P,
                pose_ema=st.pose_ema,
                last_timestamp_s=st.last_timestamp_s,
                z=za,
                lambda_prec=lama,
                z_color=zca,
                lambda_color=lamca,
                pol_logit=pola,
                pol_prec=polpa,
                geom_meta=gma,
                timestamp_s=torch.full((k_eff,), float(t_d) * 0.08, device=dev),
            )

    n_splice = int(torch.randint(4, 6, (), generator=gen).item())
    for t_s in range(n_splice):
        st = model.step_memory(
            mem_h=st.mem_h,
            mem_P=st.mem_P,
            mem_conflict=st.mem_conflict,
            color_h=st.color_h,
            color_P=st.color_P,
            pol_h=st.pol_h,
            pol_P=st.pol_P,
            pose_ema=st.pose_ema,
            last_timestamp_s=st.last_timestamp_s,
            z=z_sp,
            lambda_prec=lam_sp,
            z_color=zc_sp,
            lambda_color=lamc_sp,
            pol_logit=pol_sp,
            pol_prec=polp_sp,
            geom_meta=gmb,
            timestamp_s=torch.full((k_eff,), float(n_dwell + t_s) * 0.08, device=dev),
        )

    ro_sp = model.decode_memory(
        st.mem_h,
        st.mem_P,
        st.mem_conflict,
        st.color_h,
        st.color_P,
        st.pol_h,
        st.pol_P,
    )
    max_conf_sp = st.mem_conflict.float().max(dim=-1).values
    loss_conf_hinge = F.relu(1.65 - max_conf_sp).mean()
    tgt_mixed = torch.full((k_eff,), 2, dtype=torch.long, device=dev)
    loss_mixed_ce = F.cross_entropy(ro_sp.status_logits.float(), tgt_mixed)

    # Paired 40-frame clean multi-frame rollout across real sequence frames of `idx_a`:
    # explicitly calibrates `status_head -> ACCEPTED (0)` and `mem_conflict < 0.25` on long clean passes!
    seq_len_a = int(seq_out.frame_z.shape[1])
    fz_a = seq_out.frame_z.index_select(0, idx_a).float()
    flam_a = seq_out.frame_lambda_prec.index_select(0, idx_a).float()
    fzc_a = seq_out.frame_z_color.index_select(0, idx_a).float()
    flamc_a = seq_out.frame_lambda_color.index_select(0, idx_a).float()
    fpol_a = seq_out.frame_pol_logit.index_select(0, idx_a).float()
    fpolp_a = seq_out.frame_pol_prec.index_select(0, idx_a).float()
    fgm_a = batch["geom_meta"].index_select(0, idx_a).float()
    pad_m_b = batch.get("frame_pad_mask")
    fstep_a = (
        (~pad_m_b.index_select(0, idx_a)).float()
        if pad_m_b is not None
        else torch.ones((k_eff, seq_len_a), dtype=torch.float32, device=dev)
    )
    st_cl = model.init_memory(batch_size=k_eff, device=dev, dtype=torch.float32)
    for t_c in range(40):
        ti = t_c % seq_len_a
        use_grad_c = t_c >= 34
        z_t = fz_a[:, ti] if use_grad_c else fz_a[:, ti].detach()
        lam_t = flam_a[:, ti] if use_grad_c else flam_a[:, ti].detach()
        zc_t = fzc_a[:, ti] if use_grad_c else fzc_a[:, ti].detach()
        lamc_t = flamc_a[:, ti] if use_grad_c else flamc_a[:, ti].detach()
        pol_t = fpol_a[:, ti] if use_grad_c else fpol_a[:, ti].detach()
        polp_t = fpolp_a[:, ti] if use_grad_c else fpolp_a[:, ti].detach()
        st_cl = model.step_memory(
            mem_h=st_cl.mem_h,
            mem_P=st_cl.mem_P,
            mem_conflict=st_cl.mem_conflict,
            color_h=st_cl.color_h,
            color_P=st_cl.color_P,
            pol_h=st_cl.pol_h,
            pol_P=st_cl.pol_P,
            pose_ema=st_cl.pose_ema,
            last_timestamp_s=st_cl.last_timestamp_s,
            z=z_t,
            lambda_prec=lam_t,
            z_color=zc_t,
            lambda_color=lamc_t,
            pol_logit=pol_t,
            pol_prec=polp_t,
            geom_meta=fgm_a[:, ti],
            timestamp_s=torch.full((k_eff,), float(t_c) * 0.08, device=dev),
            step_mask=fstep_a[:, ti],
        )
    ro_cl = model.decode_memory(
        st_cl.mem_h,
        st_cl.mem_P,
        st_cl.mem_conflict,
        st_cl.color_h,
        st_cl.color_P,
        st_cl.pol_h,
        st_cl.pol_P,
    )
    loss_clean_conf = F.relu(st_cl.mem_conflict.float().max(dim=-1).values - 0.25).mean()
    tgt_acc = torch.zeros((k_eff,), dtype=torch.long, device=dev)
    loss_clean_acc_ce = F.cross_entropy(ro_cl.status_logits.float(), tgt_acc)
    return loss_conf_hinge + loss_mixed_ce + 0.75 * (loss_clean_conf + loss_clean_acc_ce)


@torch.no_grad()
def _build_approach_or_dwell_prefix_state(
    model: MFJPPlateOCRV2,
    batch: dict[str, Any],
    lh_rng: torch.Generator,
    use_amp: bool,
) -> tuple[MemoryStateV2 | None, torch.Tensor]:
    """Stop-gradient long-horizon rollout (55% 24..44f blurry approach, 45% 24..56f gate dwell)."""
    ts_raw = batch["timestamps_s"]
    non_mixed = batch["target_status"] != 2
    if not bool(non_mixed.any()):
        return None, ts_raw

    crops = batch["crops"]
    bsz, seq_len, ch, h, w = crops.shape
    dev = crops.device
    sides = batch.get("plate_sides")
    if sides is None:
        sides = torch.zeros((bsz, seq_len), dtype=torch.long, device=dev)
    geom = batch["geom_meta"].float()
    pad_mask = batch.get("frame_pad_mask")
    step_valid = (
        (~pad_mask).float() if pad_mask is not None else torch.ones((bsz, seq_len), device=dev)
    )

    is_blur_approach = float(torch.rand((), generator=lh_rng).item()) < 0.65
    flat_crops = crops.reshape(bsz * seq_len, ch, h, w).float()
    d = model.config.d_model
    dc = model.config.d_color

    fresh = model.init_memory(bsz, device=dev, dtype=torch.float32)
    st = fresh
    t_cur = torch.zeros((bsz,), dtype=torch.float32, device=dev)

    if is_blur_approach and float(torch.rand((), generator=lh_rng).item()) < 0.75:
        # Continuous smooth 16px -> 85px multi-frame scale ramp (>=60% of approach prefixes!)
        n_cycles = int(torch.randint(4, 7, (), generator=lh_rng).item())
        total_pref_steps = n_cycles * seq_len

        c_far = F.interpolate(
            F.interpolate(flat_crops, size=(max(10, h // 5), max(16, w // 10)), mode="area"),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        c_far = (
            0.50 * c_far
            + 0.25 * torch.roll(c_far, shifts=2, dims=-1)
            + 0.25 * torch.roll(c_far, shifts=-2, dims=-1)
        )
        c_mid = F.interpolate(
            F.interpolate(flat_crops, size=(max(20, int(h * 0.45)), max(38, int(w * 0.45))), mode="area"),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        c_near = F.interpolate(
            F.interpolate(flat_crops, size=(max(44, int(h * 0.85)), max(85, int(w * 0.85))), mode="area"),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        g_far = geom.clone()
        g_far[..., 0] = -1.0
        g_far[..., 1] = -1.0
        g_far[..., 5] = torch.clamp(g_far[..., 5] + 0.22, 0.0, 1.0)
        g_far[..., 6] = g_far[..., 6] * 0.16

        g_mid = geom.clone()
        g_mid[..., 0] = -0.75
        g_mid[..., 1] = -0.75
        g_mid[..., 5] = torch.clamp(g_mid[..., 5] + 0.10, 0.0, 1.0)
        g_mid[..., 6] = g_mid[..., 6] * 0.48

        g_near = geom.clone()
        g_near[..., 0] = torch.clamp(g_near[..., 0], min=0.18)
        g_near[..., 1] = torch.clamp(g_near[..., 1], min=0.12)
        g_near[..., 6] = torch.clamp(g_near[..., 6], min=0.58)

        cat_crops = torch.cat(
            [
                (c_far + 0.03 * torch.randn_like(c_far)).clamp(0.0, 1.0),
                (c_mid + 0.02 * torch.randn_like(c_mid)).clamp(0.0, 1.0),
                (c_near + 0.01 * torch.randn_like(c_near)).clamp(0.0, 1.0),
            ],
            dim=0,
        )
        cat_sides = sides.reshape(bsz * seq_len).repeat(3)
        cat_geom = torch.cat(
            [
                g_far.reshape(bsz * seq_len, model.config.geom_dim),
                g_mid.reshape(bsz * seq_len, model.config.geom_dim),
                g_near.reshape(bsz * seq_len, model.config.geom_dim),
            ],
            dim=0,
        )
        with _freeze_bn_running_stats(model), torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            enc3 = model.encode_frames(cat_crops, cat_sides, cat_geom)

        n_flat = bsz * seq_len
        z0, z1, z2 = [t.reshape(bsz, seq_len, 12, d) for t in enc3.z.float().split(n_flat, dim=0)]
        l0, l1, l2 = [t.reshape(bsz, seq_len, 12, d) for t in enc3.lambda_prec.float().split(n_flat, dim=0)]
        zc0, zc1, zc2 = [t.reshape(bsz, seq_len, dc) for t in enc3.z_color.float().split(n_flat, dim=0)]
        lc0, lc1, lc2 = [t.reshape(bsz, seq_len, dc) for t in enc3.lambda_color.float().split(n_flat, dim=0)]
        p0, p1, p2 = [t.reshape(bsz, seq_len) for t in enc3.pol_logit.float().split(n_flat, dim=0)]
        pp0, pp1, pp2 = [t.reshape(bsz, seq_len) for t in enc3.pol_prec.float().split(n_flat, dim=0)]

        for s_idx in range(total_pref_steps):
            t_idx = s_idx % seq_len
            frac = float(s_idx) / float(max(1, total_pref_steps - 1))
            px_w = 16.0 * ((85.0 / 16.0) ** frac)
            g0_s = max(-1.0, min(1.0, math.log2(px_w / 64.0)))
            if frac < 0.55:
                a = frac / 0.55
                zt = (1.0 - a) * z0[:, t_idx] + a * z1[:, t_idx]
                lt = (1.0 - a) * l0[:, t_idx] + a * l1[:, t_idx]
                zct = (1.0 - a) * zc0[:, t_idx] + a * zc1[:, t_idx]
                lct = (1.0 - a) * lc0[:, t_idx] + a * lc1[:, t_idx]
                pt = (1.0 - a) * p0[:, t_idx] + a * p1[:, t_idx]
                ppt = (1.0 - a) * pp0[:, t_idx] + a * pp1[:, t_idx]
                gmt = ((1.0 - a) * g_far[:, t_idx] + a * g_mid[:, t_idx]).clone()
            else:
                a = (frac - 0.55) / 0.45
                zt = (1.0 - a) * z1[:, t_idx] + a * z2[:, t_idx]
                lt = (1.0 - a) * l1[:, t_idx] + a * l2[:, t_idx]
                zct = (1.0 - a) * zc1[:, t_idx] + a * zc2[:, t_idx]
                lct = (1.0 - a) * lc1[:, t_idx] + a * lc2[:, t_idx]
                pt = (1.0 - a) * p1[:, t_idx] + a * p2[:, t_idx]
                ppt = (1.0 - a) * pp1[:, t_idx] + a * pp2[:, t_idx]
                gmt = ((1.0 - a) * g_mid[:, t_idx] + a * g_near[:, t_idx]).clone()
            gmt[:, 0] = g0_s
            gmt[:, 1] = g0_s
            st = model.step_memory(
                mem_h=st.mem_h,
                mem_P=st.mem_P,
                mem_conflict=st.mem_conflict,
                color_h=st.color_h,
                color_P=st.color_P,
                pol_h=st.pol_h,
                pol_P=st.pol_P,
                pose_ema=st.pose_ema,
                last_timestamp_s=st.last_timestamp_s,
                z=zt,
                lambda_prec=lt,
                z_color=zct,
                lambda_color=lct,
                pol_logit=pt,
                pol_prec=ppt,
                geom_meta=gmt,
                timestamp_s=t_cur,
                step_mask=step_valid[:, t_idx],
            )
            t_cur = t_cur + 0.06
    else:
        if is_blur_approach:
            low_h, low_w = max(14, h // 4), max(28, w // 4)
            c_low = F.interpolate(flat_crops, size=(low_h, low_w), mode="area")
            c_up = F.interpolate(c_low, size=(h, w), mode="bilinear", align_corners=False)
            c_smear = (
                0.50 * c_up
                + 0.25 * torch.roll(c_up, shifts=2, dims=-1)
                + 0.25 * torch.roll(c_up, shifts=-2, dims=-1)
            )
            pref_crops = (c_smear + 0.03 * torch.randn_like(flat_crops)).clamp(0.0, 1.0)
            n_cycles = int(torch.randint(3, 6, (), generator=lh_rng).item())
            pref_geom = geom.clone()
            pref_geom[..., 0] = torch.clamp(pref_geom[..., 0] - 0.95, -1.0, -0.68)
            pref_geom[..., 1] = torch.clamp(pref_geom[..., 1] - 0.95, -1.0, -0.68)
            pref_geom[..., 5] = torch.clamp(pref_geom[..., 5] + 0.18, 0.0, 1.0)
            pref_geom[..., 6] = pref_geom[..., 6] * 0.22
        else:
            pref_crops = (flat_crops + 0.02 * torch.randn_like(flat_crops)).clamp(0.0, 1.0)
            n_cycles = int(torch.randint(3, 8, (), generator=lh_rng).item())
            pref_geom = geom

        with _freeze_bn_running_stats(model), torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            enc = model.encode_frames(
                pref_crops,
                sides.reshape(bsz * seq_len),
                pref_geom.reshape(bsz * seq_len, model.config.geom_dim),
            )

        fz = enc.z.float().reshape(bsz, seq_len, 12, d)
        flam = enc.lambda_prec.float().reshape(bsz, seq_len, 12, d)
        fzc = enc.z_color.float().reshape(bsz, seq_len, dc)
        flamc = enc.lambda_color.float().reshape(bsz, seq_len, dc)
        fpol = enc.pol_logit.float().reshape(bsz, seq_len)
        fpolp = enc.pol_prec.float().reshape(bsz, seq_len)

        for _ in range(n_cycles):
            for t_idx in range(seq_len):
                st = model.step_memory(
                    mem_h=st.mem_h,
                    mem_P=st.mem_P,
                    mem_conflict=st.mem_conflict,
                    color_h=st.color_h,
                    color_P=st.color_P,
                    pol_h=st.pol_h,
                    pol_P=st.pol_P,
                    pose_ema=st.pose_ema,
                    last_timestamp_s=st.last_timestamp_s,
                    z=fz[:, t_idx],
                    lambda_prec=flam[:, t_idx],
                    z_color=fzc[:, t_idx],
                    lambda_color=flamc[:, t_idx],
                    pol_logit=fpol[:, t_idx],
                    pol_prec=fpolp[:, t_idx],
                    geom_meta=pref_geom[:, t_idx],
                    timestamp_s=t_cur,
                    step_mask=step_valid[:, t_idx],
                )
                t_cur = t_cur + 0.08

    nm_3 = non_mixed[:, None, None]
    nm_2 = non_mixed[:, None]
    init_state = MemoryStateV2(
        mem_h=torch.where(nm_3, st.mem_h.detach(), fresh.mem_h),
        mem_P=torch.where(nm_3, st.mem_P.detach(), fresh.mem_P),
        mem_conflict=torch.where(nm_2, st.mem_conflict.detach(), fresh.mem_conflict),
        color_h=torch.where(nm_2, st.color_h.detach(), fresh.color_h),
        color_P=torch.where(nm_2, st.color_P.detach(), fresh.color_P),
        pol_h=torch.where(non_mixed, st.pol_h.detach(), fresh.pol_h),
        pol_P=torch.where(non_mixed, st.pol_P.detach(), fresh.pol_P),
        pose_ema=torch.where(nm_2, st.pose_ema.detach(), fresh.pose_ema),
        last_timestamp_s=torch.where(non_mixed, st.last_timestamp_s.detach(), fresh.last_timestamp_s),
    )
    t_shift = float(n_cycles * seq_len) * 0.08
    ts_shifted = torch.where(nm_2, ts_raw.float() + t_shift, ts_raw.float())
    return init_state, ts_shifted


def _compute_synthetic_kd_loss(
    student_model: MFJPPlateOCRV2,
    s_out: Any,
    teacher_model: MFJPPlateOCRV2,
    batch: dict[str, Any],
    ts_in: torch.Tensor,
    use_amp: bool,
    pad_id: int = 0,
    has_prefill: bool = False,
    temp: float = 1.5,
) -> torch.Tensor:
    """Pure-synthetic Base -> Nano distillation across readable slots, frame logits, & color."""
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        t_out = teacher_model.forward_sequence(
            crops=batch["crops"],
            timestamps_s=ts_in,
            plate_sides=batch.get("plate_sides"),
            geom_meta=batch.get("geom_meta"),
            frame_pad_mask=batch.get("frame_pad_mask"),
            compute_prefix_readout=False,
            compute_stroke_mask=False,
        )
        bsz, seq_len, _, d_t = t_out.frame_z.shape
        t_frame_logits = teacher_model.decode_slot_features(
            t_out.frame_z.reshape(bsz * seq_len, NUM_SLOTS, d_t)
        ).reshape(bsz, seq_len, NUM_SLOTS, -1)

    d_s = s_out.frame_z.shape[-1]
    s_frame_logits = student_model.decode_slot_features(
        s_out.frame_z.reshape(bsz * seq_len, NUM_SLOTS, d_s)
    ).reshape(bsz, seq_len, NUM_SLOTS, -1)

    dev = s_out.slot_logits.device
    pad_mask = batch.get("frame_pad_mask")
    if pad_mask is None:
        pad_mask_bool = torch.zeros((bsz, seq_len), dtype=torch.bool, device=dev)
    else:
        pad_mask_bool = pad_mask.bool()
    target_slots = batch["target_slots"].long()
    frame_slot_targets = batch.get(
        "frame_slot_targets",
        target_slots.unsqueeze(1).expand(bsz, seq_len, NUM_SLOTS),
    ).long()
    eff_read = compute_effective_slot_readability_v2(
        slot_readability=batch["slot_readability"].float(),
        frame_slot_targets=frame_slot_targets,
        frame_valid_mask=batch["frame_valid_mask"].float(),
        frame_pad_mask=pad_mask_bool,
        slot_glare=batch.get("slot_glare", None),
        pad_id=pad_id,
    )  # (B, T, 12)

    valid_f = (~pad_mask_bool) & (batch["frame_valid_mask"].float() > 0.5)
    same_as_primary = (frame_slot_targets == target_slots.unsqueeze(1)).all(dim=-1)
    primary_read = eff_read * (valid_f & same_as_primary).unsqueeze(-1).float()
    seq_max_read = primary_read.max(dim=1).values  # (B, 12)

    t2 = temp * temp
    slot_kd_w = torch.tensor(
        [3.6, 3.6, 1.8, 1.2, 1.25, 1.25, 1.15, 1.45, 1.25, 1.25, 1.25, 1.25],
        device=dev,
        dtype=torch.float32,
    )
    slot_kd_w = slot_kd_w / slot_kd_w.mean()
    non_mixed = batch["target_status"] != 2
    zero_kd = s_out.slot_logits.float().sum() * 0.0

    # Sequence-level distillation only when teacher & student share identical initial state
    seq_mask = (seq_max_read >= 0.30) & non_mixed.unsqueeze(1)
    if (not has_prefill) and bool(seq_mask.any()):
        s_logp = F.log_softmax(s_out.slot_logits.float() / temp, dim=-1)
        t_prob = F.softmax(t_out.slot_logits.float() / temp, dim=-1)
        kl_per_slot = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=-1)  # (B, 12)
        seq_w = seq_max_read * seq_mask.float() * slot_kd_w.unsqueeze(0)
        kd_seq = ((kl_per_slot * seq_w).sum() / seq_w.sum().clamp_min(1e-6)) * t2
    else:
        kd_seq = zero_kd

    # Per-frame encoder slot distillation strictly on optically resolved slots (eff_read >= 0.28)
    aux_mask = valid_f.unsqueeze(-1) & (eff_read >= 0.28)
    if bool(aux_mask.any()):
        sf_logp = F.log_softmax(s_frame_logits.float() / temp, dim=-1)
        tf_prob = F.softmax(t_frame_logits.float() / temp, dim=-1)
        kl_f_slot = F.kl_div(sf_logp, tf_prob, reduction="none").sum(dim=-1)  # (B, T, 12)
        aux_w = eff_read * aux_mask.float() * slot_kd_w.view(1, 1, NUM_SLOTS)
        kd_aux = ((kl_f_slot * aux_w).sum() / aux_w.sum().clamp_min(1e-6)) * t2
    else:
        kd_aux = zero_kd

    rgb_nm = non_mixed & ((batch["is_monochrome_ir"] < 0.5).any(dim=1)) & (valid_f.any(dim=1))
    if (not has_prefill) and bool(rgb_nm.any()):
        sc_logp = F.log_softmax(s_out.plate_type_logits[rgb_nm].float() / temp, dim=-1)
        tc_prob = F.softmax(t_out.plate_type_logits[rgb_nm].float() / temp, dim=-1)
        kd_col = F.kl_div(sc_logp, tc_prob, reduction="batchmean") * t2
    else:
        kd_col = zero_kd

    if not has_prefill:
        st_logp = F.log_softmax(s_out.status_logits.float() / temp, dim=-1)
        tst_prob = F.softmax(t_out.status_logits.float() / temp, dim=-1)
        kd_st = F.kl_div(st_logp, tst_prob, reduction="batchmean") * t2
    else:
        kd_st = zero_kd

    kd_qual = F.mse_loss(
        torch.sigmoid(s_out.slot_quality_logits.float()),
        torch.sigmoid(t_out.slot_quality_logits.float()),
    )
    s_rl = getattr(s_out, "reg_len_logits", None)
    t_rl = getattr(t_out, "reg_len_logits", None)
    s_cl = getattr(s_out, "cls_len_logits", None)
    t_cl = getattr(t_out, "cls_len_logits", None)
    if s_rl is not None and t_rl is not None and s_cl is not None and t_cl is not None and bool(valid_f.any()):
        kd_rlen = F.kl_div(
            F.log_softmax(s_rl[valid_f].float() / temp, dim=-1),
            F.softmax(t_rl[valid_f].float() / temp, dim=-1),
            reduction="batchmean",
        ) * t2
        kd_clen = F.kl_div(
            F.log_softmax(s_cl[valid_f].float() / temp, dim=-1),
            F.softmax(t_cl[valid_f].float() / temp, dim=-1),
            reduction="batchmean",
        ) * t2
        kd_len = 0.5 * (kd_rlen + kd_clen)
    else:
        kd_len = zero_kd
    if has_prefill:
        return 0.70 * kd_aux + 0.25 * kd_qual + 0.15 * kd_len
    return 0.55 * kd_seq + 0.40 * kd_aux + 0.18 * kd_col + 0.12 * kd_st + 0.20 * kd_qual + 0.15 * kd_len


@torch.no_grad()
def evaluate_synthetic_val(
    model: MFJPPlateOCRV2,
    vocab: PlateVocab,
    device: torch.device,
    seed: int = VAL_SEED,
    num_batches: int = 6,
    val_engine: GPUSynthSequenceEngine | None = None,
) -> tuple[dict[str, float], torch.Tensor]:
    """Evaluate on 100% synthetic validation batches (standard + T=32 approach) & build confusion vector."""
    was_training = model.training
    model.eval()

    if val_engine is None:
        val_engine = GPUSynthSequenceEngine(
            vocab=vocab,
            device=device,
            out_hw=model.config.input_hw,
            margin_ratio=model.config.canonical_margin,
            curriculum_stage=4,
            train_mode=False,
            include_stroke_mask=False,
            seed=seed,
        )

    bsz_std = 32 if num_batches >= 2 else 8
    tot_read_seq = 0
    exact_4f = 0
    reg_ok = 0
    ser_ok = 0
    slot_ok = 0
    slot_tot = 0
    status_ok = 0
    status_tot = 0
    rgb_type_ok = 0
    rgb_type_tot = 0
    kl_surprise_sum = torch.zeros(vocab.size, dtype=torch.float32, device=device)
    tok_occ_counts = torch.zeros(vocab.size, dtype=torch.float32, device=device)

    for bi_std in range(num_batches):
        b = _move_batch_v2_to_device(
            val_engine.sample_batch(bsz_std, seq_len=8, approach_prob=0.25, seed=seed + bi_std), device
        )
        out = model.forward_sequence(
            crops=b["crops"],
            timestamps_s=b["timestamps_s"],
            plate_sides=b.get("plate_sides"),
            geom_meta=b.get("geom_meta"),
            frame_pad_mask=b.get("frame_pad_mask"),
            compute_prefix_readout=False,
            compute_stroke_mask=False,
        )
        pred_status = out.status_logits.argmax(dim=-1)
        tgt_status = b["target_status"]
        status_ok += int((pred_status == tgt_status).sum().item())
        status_tot += int(tgt_status.numel())

        readable = tgt_status == 0
        if readable.any():
            ps = out.slot_logits[readable].argmax(dim=-1)
            ts = b["target_slots"][readable]
            eq = ps == ts
            tot_read_seq += int(readable.sum().item())
            exact_4f += int(eq.all(dim=-1).sum().item())
            reg_ok += int(eq[:, 0:4].all(dim=-1).sum().item())
            ser_ok += int(eq[:, 8:12].all(dim=-1).sum().item())
            slot_ok += int(eq.sum().item())
            slot_tot += int(ts.numel())

            # Information-theoretic per-token conditional surprise E[-log p(y | x)] (unbiased by digit frequency!)
            log_probs_r = F.log_softmax(out.slot_logits[readable].float(), dim=-1)
            safe_ts = ts.clamp(0, vocab.size - 1)
            nll_r = (-log_probs_r.gather(-1, safe_ts.unsqueeze(-1)).squeeze(-1)).clamp(0.0, 8.0)
            flat_tok = safe_ts.reshape(-1)
            kl_surprise_sum.scatter_add_(0, flat_tok, nll_r.reshape(-1))
            tok_occ_counts.scatter_add_(0, flat_tok, torch.ones_like(flat_tok, dtype=torch.float32))

            rgb_seq = readable & ((b["is_monochrome_ir"] < 0.5).any(dim=1))
            if rgb_seq.any():
                pt_pred = out.plate_type_logits[rgb_seq].argmax(dim=-1)
                pt_tgt = b["target_plate_type"][rgb_seq]
                rgb_type_ok += int((pt_pred == pt_tgt).sum().item())
                rgb_type_tot += int(pt_tgt.numel())

    app_tot = 0
    app_4f = 0
    app_ser = 0
    app_clean_conflict = 0
    n_app_batches = 4 if num_batches >= 2 else 1
    bsz_app = 16 if num_batches >= 2 else 4
    t_app_eval = 32 if num_batches >= 2 else 12
    for bi_app in range(n_app_batches):
        ba = _move_batch_v2_to_device(
            val_engine.sample_batch(
                bsz_app,
                seq_len=t_app_eval,
                approach_prob=1.0,
                regime_override=REGIME_NAMES[0],
                seed=seed + 1000 + bi_app,
            ),
            device,
        )
        out_a = model.forward_sequence(
            crops=ba["crops"],
            timestamps_s=ba["timestamps_s"],
            plate_sides=ba.get("plate_sides"),
            geom_meta=ba.get("geom_meta"),
            frame_pad_mask=ba.get("frame_pad_mask"),
            compute_prefix_readout=False,
            compute_stroke_mask=False,
        )
        rd = ba["target_status"] == 0
        if rd.any():
            psa = out_a.slot_logits[rd].argmax(dim=-1)
            tsa = ba["target_slots"][rd]
            eqa = psa == tsa
            app_tot += int(rd.sum().item())
            app_4f += int(eqa.all(dim=-1).sum().item())
            app_ser += int(eqa[:, 8:12].all(dim=-1).sum().item())
            max_conf = out_a.memory.mem_conflict[rd].max(dim=-1).values
            app_clean_conflict += int((max_conf < 0.95).sum().item())

    err_counts = kl_surprise_sum / (tok_occ_counts + 1.5)
    err_counts[0] = 0.0
    if was_training:
        model.train()

    metrics = {
        "val_exact_4field": round(exact_4f / max(1, tot_read_seq), 4),
        "val_region_acc": round(reg_ok / max(1, tot_read_seq), 4),
        "val_serial_acc": round(ser_ok / max(1, tot_read_seq), 4),
        "val_slot_acc": round(slot_ok / max(1, slot_tot), 4),
        "val_status_acc": round(status_ok / max(1, status_tot), 4),
        "val_rgb_type_acc": round(rgb_type_ok / max(1, rgb_type_tot), 4),
        "val_approach_4field": round(app_4f / max(1, app_tot), 4),
        "val_approach_serial": round(app_ser / max(1, app_tot), 4),
        "val_approach_no_false_conflict": round(app_clean_conflict / max(1, app_tot), 4),
    }
    return metrics, err_counts.detach().cpu()


def train_preset_from_scratch(
    *,
    preset: str,
    total_steps: int,
    batch_size: int,
    seq_len: int,
    max_lr: float,
    init_seed: int,
    data_seed: int,
    out_models_dir: Path,
    device: torch.device,
    teacher_model: MFJPPlateOCRV2 | None = None,
    teacher_ckpt_path: Path | None = None,
    eval_every: int = 500,
    val_batches: int = 6,
) -> tuple[MFJPPlateOCRV2, Path, dict[str, Any]]:
    vocab = PlateVocab()
    cfg = ModelConfigV2.from_preset(preset, vocab_size=vocab.size)

    # 1. Deterministic `random_scratch` initialization on CPU with verifiable step-0 SHA-256
    model, init_state_sha256 = instantiate_deterministic_scratch_model(cfg, init_seed=init_seed)
    model.to(device)
    ema = ModelEMAV2(model, decay=0.996)

    train_params = model.num_parameters
    fused_probe = model.clone_reparameterized()
    fused_params = fused_probe.num_parameters
    del fused_probe
    print(
        f"\n=== [{preset.upper()} RANDOM-SCRATCH] init_seed={init_seed}, "
        f"init_sha256={init_state_sha256[:16]}..., train_params={train_params:,}, "
        f"fused_deploy_params={fused_params:,}, steps={total_steps}, batch={batch_size}x{seq_len} ===",
        flush=True,
    )

    use_amp = device.type == "cuda"
    warmup_steps = max(20, min(180, total_steps // 16))
    fracs = (0.14, 0.38, 0.68)

    loss_weights = LossWeightsV2(
        seq_slots=1.0,
        aux_frames=0.85,
        status=0.75,
        calibration=0.75,
        color_type=0.45,
        polarity=0.30,
        unc_cal=0.35,
        mv_cons=0.45,
        mono=0.35,
        cf_conflict=0.60,
        garbage_invariance=0.70,
        stroke_gestalt=0.25,
        statutory_len=0.65,
        innovation_nll=0.02,
        terminal_risk=0.25,
    )
    criterion = OCRLossV2(
        vocab_size=vocab.size,
        pad_id=vocab.pad_id,
        weights=loss_weights,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
    engine = GPUSynthSequenceEngine(
        vocab=vocab,
        device=device,
        out_hw=cfg.input_hw,
        margin_ratio=cfg.canonical_margin,
        curriculum_stage=1,
        train_mode=True,
        include_stroke_mask=True,
        seed=data_seed,
    )
    neg_bank_size = 96 if total_steps >= 100 else 16
    neg_crops_bank, neg_geoms_bank = build_synthetic_hard_negative_bank(
        num_samples=neg_bank_size,
        out_hw=cfg.input_hw,
        device=device,
        seed=data_seed + 333_333,
    )

    if teacher_model is not None:
        teacher_model = teacher_model.to(device).eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    lh_rng = torch.Generator(device="cpu")
    lh_rng.manual_seed(data_seed + 7777)
    val_engine = GPUSynthSequenceEngine(
        vocab=vocab,
        device=device,
        out_hw=cfg.input_hw,
        margin_ratio=cfg.canonical_margin,
        curriculum_stage=4,
        train_mode=False,
        include_stroke_mask=False,
        seed=VAL_SEED,
    )
    t0 = time.perf_counter()
    val_snapshots: list[dict[str, Any]] = []
    skipped_nan = 0
    cached_raw_batch: dict[str, Any] | None = None
    seq_time_keys = (
        "crops",
        "geom_meta",
        "camera_ids",
        "plate_sides",
        "frame_valid_mask",
        "slot_readability",
        "slot_glare",
        "frame_slot_targets",
        "is_monochrome_ir",
        "stroke_mask_gt",
    )

    for step in range(total_steps):
        stage = _stage_for_step(step, total_steps, fracs)
        if engine.curriculum_stage != stage:
            engine.set_curriculum_stage(stage)
            cached_raw_batch = None

        lr_now = _cosine_lr(step, total_steps, warmup=warmup_steps, max_lr=max_lr, min_lr=2.0e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        u_mode = float(torch.rand((), generator=lh_rng).item())
        init_state: MemoryStateV2 | None = None
        if step % 4 != 3 or cached_raw_batch is None:
            if stage >= 2 and total_steps >= 40 and u_mode < 0.24:
                t_app = int(torch.randint(24, 41, (), generator=lh_rng).item())
                bsz_cur = max(6, (batch_size * seq_len) // t_app)
                raw_batch = _move_batch_v2_to_device(
                    engine.sample_batch(bsz_cur, seq_len=t_app, approach_prob=0.85), device
                )
            else:
                raw_batch = _move_batch_v2_to_device(engine.sample_batch(batch_size, seq_len=seq_len), device)
            cached_raw_batch = dict(raw_batch)
            batch = dict(raw_batch)
            ts_in = batch["timestamps_s"]
            if batch["crops"].shape[1] == seq_len and stage >= 2 and total_steps >= 40 and 0.24 <= u_mode < 0.46:
                init_state, ts_in = _build_approach_or_dwell_prefix_state(
                    model, batch, lh_rng, use_amp=use_amp
                )
            elif stage >= 2 and total_steps >= 40 and batch["crops"].shape[1] >= 4 and u_mode >= 0.84:
                # Calibrate K=1..5 single-frame & short-burst passes on fresh plates too
                bsz_f, seq_f = batch["crops"].shape[:2]
                t_eff = torch.randint(1, min(6, seq_f + 1), (bsz_f, 1), generator=lh_rng).to(device)
                t_grid = torch.arange(seq_f, device=device).unsqueeze(0)
                short_pad = t_grid >= t_eff
                is_mix_b = batch["target_status"] == 2
                short_pad = torch.where(is_mix_b.unsqueeze(1), torch.zeros_like(short_pad), short_pad)
                act_read = batch["slot_readability"] * (~short_pad).unsqueeze(-1).float()
                wiped_short = act_read.amax(dim=(1, 2)) < 0.18
                orig_st = batch["target_status"]
                batch["target_status"] = torch.where(
                    is_mix_b,
                    orig_st,
                    torch.where((orig_st == 1) | wiped_short, torch.ones_like(orig_st), torch.zeros_like(orig_st)),
                )
                batch["frame_pad_mask"] = short_pad
        else:
            batch = dict(cached_raw_batch)
            ts_in = batch["timestamps_s"]
            bsz_cur, seq_cur = batch["crops"].shape[:2]
            if u_mode < 0.65 and seq_cur >= 2:
                # Reverse non-mixed sequences along time (`near -> far` receding pass supervision!)
                rev_mask = batch["target_status"] != 2
                if bool(rev_mask.any()):
                    for tk in seq_time_keys:
                        tv = batch.get(tk)
                        if isinstance(tv, torch.Tensor) and tv.ndim >= 2 and tv.shape[1] == seq_cur:
                            flipped = torch.flip(tv, dims=[1])
                            m_shape = (-1,) + (1,) * (tv.ndim - 1)
                            batch[tk] = torch.where(rev_mask.view(m_shape), flipped, tv)
            elif stage >= 2 and total_steps >= 40 and seq_cur >= 4:
                # Calibrate K=1..5 single-frame & short-burst passes explicitly
                t_eff = torch.randint(
                    1, min(6, seq_cur + 1), (bsz_cur, 1), generator=lh_rng
                ).to(device)
                t_grid = torch.arange(seq_cur, device=device).unsqueeze(0)
                short_pad = t_grid >= t_eff
                is_mix_b = batch["target_status"] == 2
                short_pad = torch.where(is_mix_b.unsqueeze(1), torch.zeros_like(short_pad), short_pad)
                act_read = batch["slot_readability"] * (~short_pad).unsqueeze(-1).float()
                wiped_short = act_read.amax(dim=(1, 2)) < 0.18
                orig_st = batch["target_status"]
                recomp_st = torch.where(
                    is_mix_b,
                    orig_st,
                    torch.where(
                        (orig_st == 1) | wiped_short,
                        torch.ones_like(orig_st),
                        torch.zeros_like(orig_st),
                    ),
                )
                batch["target_status"] = recomp_st
                batch["frame_pad_mask"] = short_pad

        want_stroke = bool("stroke_mask_gt" in batch and batch["stroke_mask_gt"] is not None)
        do_aux_pair = (step % 2 == 0) or (step < 120)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            seq_out = model.forward_sequence(
                crops=batch["crops"],
                timestamps_s=ts_in,
                plate_sides=batch.get("plate_sides"),
                geom_meta=batch.get("geom_meta"),
                frame_pad_mask=batch.get("frame_pad_mask"),
                compute_prefix_readout=True,
                compute_stroke_mask=want_stroke,
                init_state=init_state,
            )

        loss, metrics = criterion(model, seq_out, batch, compute_garbage_invariance=do_aux_pair)

        if do_aux_pair:
            loss_hn = compute_hard_negative_aux_loss(
                model,
                neg_crops_bank,
                neg_geoms_bank,
                gen=lh_rng,
                batch_n=min(16, batch_size),
                use_amp=use_amp,
            )
            loss = loss + 0.35 * loss_hn
            if stage >= 2:
                loss_sp = compute_crisp_dwell_splice_aux_loss(
                    model, seq_out, batch, gen=lh_rng, max_pairs=8
                )
                loss = loss + 0.45 * loss_sp

        if teacher_model is not None:
            loss_kd = _compute_synthetic_kd_loss(
                student_model=model,
                s_out=seq_out,
                teacher_model=teacher_model,
                batch=batch,
                ts_in=ts_in,
                use_amp=use_amp,
                pad_id=vocab.pad_id,
                has_prefill=(init_state is not None),
                temp=1.5,
            )
            loss = loss + 0.85 * loss_kd

        if not torch.isfinite(loss):
            skipped_nan += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)

        if (step + 1) % 200 == 0 or (step + 1) == total_steps:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{preset.upper()} {step + 1}/{total_steps} | C{stage} | {elapsed:.0f}s] "
                f"loss={metrics['loss_total']:.4f} "
                f"(slots={metrics['loss_seq_slots']:.4f}, col={metrics['loss_color_type']:.4f}, "
                f"unc={metrics['loss_unc_cal']:.4f}, cf={metrics['loss_cf_conflict']:.4f}) "
                f"lr={lr_now:.2e}",
                flush=True,
            )

        if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
            val_m, err_vec = evaluate_synthetic_val(
                ema.ema_model,
                vocab=vocab,
                device=device,
                seed=VAL_SEED,
                num_batches=val_batches,
                val_engine=val_engine,
            )
            val_m["step"] = step + 1
            val_snapshots.append(val_m)
            engine.update_confusion_boost(err_vec.numpy(), boost_prob=0.30)
            print(f"    -> [VAL {preset.upper()} @{step + 1}] {val_m}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    wall_s = round(time.perf_counter() - t0, 2)
    ckpt_path = out_models_dir / f"mf_plate_ocr_v2_{preset}.pt"
    train_cfg_record = TrainConfigV2(
        batch_size=batch_size,
        seq_len=seq_len,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        max_lr=max_lr,
        min_lr=2.0e-5,
        long_horizon_prob=0.24,
        approach_batch_prob=0.28,
        approach_seq_len_range=(24, 40),
        curriculum_fractions=fracs,
        seed=data_seed,
    )

    prov_manifest = build_synthetic_provenance_manifest(
        init_mode="random_scratch",
        init_seed=init_seed,
        init_state_sha256=init_state_sha256,
        repo_root=REPO_ROOT,
        training_script=TRAINING_SCRIPT_REL,
        parent_checkpoint=None,
        teacher_checkpoint=teacher_ckpt_path,
    )

    meta_summary: dict[str, Any] = {
        "version": "0.6.0",
        "zero_real_pixel_training": True,
        "init_mode": "random_scratch",
        "init_seed": init_seed,
        "init_state_sha256": init_state_sha256,
        "preset": preset,
        "train_params": train_params,
        "fused_deploy_params": fused_params,
        "run_steps": total_steps,
        "skipped_nan_steps": skipped_nan,
        "wall_time_s": wall_s,
        "domain_split": "val_holdout",
        "data_seed": data_seed,
        "val_seed": VAL_SEED,
        "final_val": val_snapshots[-1] if val_snapshots else {},
        "val_snapshots": val_snapshots,
        "checkpoint": str(ckpt_path),
    }

    save_checkpoint_v2(
        ckpt_path,
        model=model,
        ema_model=ema.ema_model,
        train_config=train_cfg_record,
        metrics=meta_summary,
        provenance=prov_manifest,
    )

    # Verify cryptographic provenance immediately against repo sources and fonts
    prov_report = verify_checkpoint_provenance(
        ckpt_path, repo_root=REPO_ROOT, verify_source_hashes=True
    )
    assert prov_report["verified"] is True

    # Export FP32 & INT8 ONNX from EMA model + verify patent hygiene & numerical parity
    onnx_dir = out_models_dir / f"onnx_v2_{preset}"
    export_ocr_v2_to_onnx(ema.ema_model, onnx_dir)
    quantize_onnx_v2_for_cpu(onnx_dir)
    hyg_fp32 = verify_onnx_patent_hygiene(onnx_dir, use_int8=False)
    hyg_int8 = verify_onnx_patent_hygiene(onnx_dir, use_int8=True)
    assert hyg_fp32["patent_clean"] is True, f"{preset} FP32 patent hygiene failed: {hyg_fp32}"
    assert hyg_int8["patent_clean"] is True, f"{preset} INT8 patent hygiene failed: {hyg_int8}"
    parity = verify_onnx_v2_parity(
        ema.ema_model, onnx_dir, providers=["CPUExecutionProvider"], atol=2.5e-4
    )

    assert parity["max_abs_diff"] < 2.5e-4, parity
    graph_names = ("frame_encoder", "memory_step", "memory_readout", "stopping_risk")
    artifact_sizes = {f"{name}.int8.onnx": (onnx_dir / f"{name}.int8.onnx").stat().st_size for name in graph_names}
    if preset == "nano":
        assert sum(artifact_sizes.values()) <= 4 * 1024**2, artifact_sizes
    meta_summary["int8_bytes"] = artifact_sizes
    meta_summary["checkpoint_sha256"] = compute_file_sha256(ckpt_path)
    meta_summary["provenance_verification"] = prov_report
    meta_summary["onnx_dir"] = str(onnx_dir)
    meta_summary["onnx_patent_clean_fp32"] = hyg_fp32["patent_clean"]
    meta_summary["onnx_patent_clean_int8"] = hyg_int8["patent_clean"]
    meta_summary["onnx_max_abs_diff"] = parity["max_abs_diff"]
    print(
        f"=== [{preset.upper()} EXPORTED & PROVENANCE-VERIFIED] "
        f"max_abs_diff={parity['max_abs_diff']:.2e}, "
        f"recomputed_init_match={prov_report['recomputed_init_sha256_match']}, "
        f"teacher_verified={prov_report['teacher_verified']} ===",
        flush=True,
    )
    del model
    del optimizer
    del neg_crops_bank
    del neg_geoms_bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ema.ema_model, ckpt_path, meta_summary


def main() -> None:
    args = parse_args()

    if any(marker in Path(__file__).read_text() for marker in FORBIDDEN_REAL_DATA_MARKERS):
        raise ValueError("Training script failed the data-isolation scan")
    if args.dry_run and args.out_dir.resolve() == DEFAULT_MODELS_DIR.resolve():
        raise ValueError("Dry run requires an isolated output directory")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_steps = 4 if args.dry_run else args.base_steps
    nano_steps = 4 if args.dry_run else args.nano_steps
    bsz = 8 if args.dry_run else args.batch_size
    seq_l = 4 if args.dry_run else args.seq_len
    eval_ev = 4 if args.dry_run else 600
    val_b = 1 if args.dry_run else 6

    base_ema, base_ckpt, base_summary = train_preset_from_scratch(
        preset="base",
        total_steps=base_steps,
        batch_size=bsz,
        seq_len=seq_l,
        max_lr=1.75e-3,
        init_seed=args.base_init_seed,
        data_seed=BASE_DATA_SEED,
        out_models_dir=out_dir,
        device=device,
        teacher_model=None,
        teacher_ckpt_path=None,
        eval_every=eval_ev,
        val_batches=val_b,
    )

    nano_ema, _, nano_summary = train_preset_from_scratch(
        preset="nano",
        total_steps=nano_steps,
        batch_size=bsz,
        seq_len=seq_l,
        max_lr=1.90e-3,
        init_seed=args.nano_init_seed,
        data_seed=NANO_DATA_SEED,
        out_models_dir=out_dir,
        device=device,
        teacher_model=base_ema,
        teacher_ckpt_path=base_ckpt,
        eval_every=eval_ev,
        val_batches=val_b,
    )
    del nano_ema
    del base_ema

    manifest_out = out_dir / "ocr_v4_phase1_provenance_manifest.json"
    manifest_payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "zero_real_pixel_training": True,
        "training_script": TRAINING_SCRIPT_REL,
        "training_script_sha256": compute_file_sha256(REPO_ROOT / TRAINING_SCRIPT_REL),
        "base": base_summary,
        "nano": nano_summary,
    }
    for name in ("gen6_stage3_ocr_manifest.json", manifest_out.name):
        target = out_dir / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(tmp, target)
    print(f"\nSaved verified Phase 1 provenance manifest -> {manifest_out}", flush=True)


if __name__ == "__main__":
    main()
