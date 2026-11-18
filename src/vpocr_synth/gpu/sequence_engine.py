"""GPU-accelerated 2.5D Sim-to-Real multi-frame sequence engine & IterableDataset.

Implements `GPUSynthSequenceEngine`, `GPUSynthIterableDataset`, and exact GPU
`compute_geom_meta_gpu` matching `vpocr_det.quad_cropper.extract_quad_geometry_meta`
and `/scratch/v2_shared_contract.md` 100% key-for-key, shape-for-shape, dtype-for-dtype.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import math
import os
import random
from typing import Any

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from vpocr_ocr.vocab import NUM_SLOTS, PlateVocab
from vpocr_synth.chaos_info import boltzmann_kl_token_weights, chaotic_intermittent_corner_burst
from vpocr_synth.gpu.homography_fast import (
    REGIME_NAMES,
    REGIME_TO_ID,
    TrajectoryBatch,
    sample_se3_trajectories,
    warp_quad_to_canonical_gpu,
)
from vpocr_synth.gpu.sensor_codec_gpu import SensorCodecGPU
from vpocr_synth.gpu.shader_pbr_ir import ReliefPBRShaderGPU
from vpocr_synth.plate.glyph_dies import GlyphSDFAtlas, get_default_glyph_atlas
from vpocr_synth.plate.models import PlateConfig, PlateText, PlateType
from vpocr_synth.plate.stratified_sampler import StratifiedPlateSampler
from vpocr_synth.plate.surface_25d import EXT_PLATE_CORNERS_XY, PlateSurface25DBuilder, PlateSurface25DPacket
from vpocr_synth.plate.validator import PlateValidator

# Exact OpenCV cv2.Laplacian(..., ksize=3) 3x3 kernel (cached per device, zero H2D sync!)
_OPENCV_LAPLACIAN_K3 = (
    (2.0, 0.0, 2.0),
    (0.0, -8.0, 0.0),
    (2.0, 0.0, 2.0),
)
_LAP_KERNEL_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_lap_kernel(dev: torch.device) -> torch.Tensor:
    k = _LAP_KERNEL_CACHE.get(dev)
    if k is None:
        k = torch.tensor(_OPENCV_LAPLACIAN_K3, device=dev, dtype=torch.float32).view(1, 1, 3, 3)
        _LAP_KERNEL_CACHE[dev] = k
    return k


_SEQ_CUDA_CACHE: dict[str, Callable[..., Any]] = {}
_AR_MAT_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _dispatch_seq_cuda(name: str, fn: Callable[..., Any], *args: Any) -> Any:
    first = args[0]
    if isinstance(first, torch.Tensor) and first.is_cuda:
        c_fn = _SEQ_CUDA_CACHE.get(name)
        if c_fn is None:
            c_fn = torch.compile(fn, dynamic=True)
            _SEQ_CUDA_CACHE[name] = c_fn
        return c_fn(*args)
    return fn(*args)


def _get_ar1_matrix(dev: torch.device, t_len: int) -> torch.Tensor:
    key = (dev, t_len)
    mat = _AR_MAT_CACHE.get(key)
    if mat is None:
        m = np.zeros((t_len, t_len), dtype=np.float32)
        for t in range(t_len):
            for s in range(t + 1):
                m[t, s] = (0.62 ** (t - s)) * (1.0 if s == 0 else 0.78)
        mat = torch.from_numpy(m).to(dev)
        _AR_MAT_CACHE[key] = mat
    return mat


def _clean_canon_lit_impl(albedo_rgb: torch.Tensor, height_mm: torch.Tensor) -> torch.Tensor:
    h_pad = F.pad(height_mm, (1, 1, 1, 1), mode="replicate")
    dh_dx = (h_pad[:, :, 1:-1, 2:] - h_pad[:, :, 1:-1, :-2]) * (1.0 / (2.0 * 1.03125))
    dh_dy = (h_pad[:, :, 2:, 1:-1] - h_pad[:, :, :-2, 1:-1]) * (1.0 / (2.0 * 1.03125))
    normals = F.normalize(torch.cat([-dh_dx, -dh_dy, torch.ones_like(dh_dx)], dim=1), dim=1)
    ndotl = (-0.25 * normals[:, 0:1] - 0.45 * normals[:, 1:2] + 0.86 * normals[:, 2:3]).clamp(0.0, 1.0)
    return (albedo_rgb * (0.36 + 0.68 * ndotl)).clamp(0.0, 1.0)


def _perturb_corners_core_impl(
    gt_roi: torch.Tensor,
    roi_origin_xy: torch.Tensor,
    has_seal: torch.Tensor,
    plate_sides: torch.Tensor,
    l_ar: torch.Tensor,
    eps_unit: torch.Tensor,
    dof_seq_u: torch.Tensor,
    dof_bt_n: torch.Tensor,
    top_recess_u: torch.Tensor,
    sigma_mult: float,
    stage_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Structured 8-DOF Stage-2 quad homography perturbation in normalized `(u, v)` + ROI space.

    Combines:
      1. Global canonical translation jitter (`±6.5px` X, `±5.0px` Y in `160x80` space),
      2. Independent 4-edge margins `[m_left, m_right, m_top, m_bot]` in `[-0.03, +0.12]`,
      3. Residual horizontal/vertical keystone (trapezoid) + parallelogram shear skew,
      4. Temporal AR(1) per-corner sub-pixel drift + rear-seal / recessed top-lip bias.
    """
    bsz, t_len, _, _ = gt_roi.shape
    u_top = gt_roi[:, :, 1] - gt_roi[:, :, 0]
    u_bot = gt_roi[:, :, 2] - gt_roi[:, :, 3]
    v_left = gt_roi[:, :, 3] - gt_roi[:, :, 0]
    v_right = gt_roi[:, :, 2] - gt_roi[:, :, 1]
    u_vec = 0.5 * (u_top + u_bot)
    v_vec = 0.5 * (v_left + v_right)
    w_px = (0.5 * (torch.linalg.norm(u_top, dim=-1) + torch.linalg.norm(u_bot, dim=-1))).clamp_min(8.0)

    # 1. Temporal AR(1) correlated 8-DOF noise + per-corner sub-pixel jitter
    dof_ar = torch.matmul(l_ar, dof_bt_n) * (0.34 * stage_scale)  # (B, T, 9)

    # Global translation jitter: ±6.5px / 160 in X, ±5.0px / 80 in Y
    tx_seq = ((dof_seq_u[:, 0:1] * 2.0 - 1.0) * (6.5 / 160.0)) * stage_scale
    ty_seq = ((dof_seq_u[:, 1:2] * 2.0 - 1.0) * (5.0 / 80.0)) * stage_scale
    tx_bt = (tx_seq + dof_ar[:, :, 0] * (3.2 / 160.0)).clamp(-7.5 / 160.0, 7.5 / 160.0)
    ty_bt = (ty_seq + dof_ar[:, :, 1] * (2.6 / 80.0)).clamp(-6.0 / 80.0, 6.0 / 80.0)

    # Independent 4-edge margins relative to base margin_ratio=0.05:
    # 55% full wide [-0.03, +0.12], 45% tight [0.025, 0.075]
    use_wide = (dof_seq_u[:, 2:3] < 0.55) & (stage_scale > 0.6)
    m_wide = -0.03 + 0.15 * dof_seq_u[:, 3:7]   # (B, 4) in [-0.03, +0.12]
    m_tight = 0.025 + 0.050 * dof_seq_u[:, 3:7]  # (B, 4) in [0.025, 0.075]
    m_edges = torch.where(use_wide, m_wide, m_tight)
    dm_seq = (m_edges - 0.05) * stage_scale  # (B, 4): [left, right, top, bot]
    dm_l = (dm_seq[:, 0:1] + dof_ar[:, :, 2] * 0.018).clamp(-0.082, 0.075)
    dm_r = (dm_seq[:, 1:2] + dof_ar[:, :, 3] * 0.018).clamp(-0.082, 0.075)
    dm_t = (dm_seq[:, 2:3] + dof_ar[:, :, 4] * 0.020).clamp(-0.082, 0.075)
    dm_b = (dm_seq[:, 3:4] + dof_ar[:, :, 5] * 0.020).clamp(-0.082, 0.075)

    # Residual keystone / trapezoid & shear skew
    kh = (((dof_seq_u[:, 7:8] * 2.0 - 1.0) * 0.026) * stage_scale + dof_ar[:, :, 6] * 0.014).clamp(-0.036, 0.036)
    kv = (((dof_seq_u[:, 8:9] * 2.0 - 1.0) * 0.032) * stage_scale + dof_ar[:, :, 7] * 0.016).clamp(-0.042, 0.042)
    sh = (((dof_seq_u[:, 9:10] * 2.0 - 1.0) * 0.020) * stage_scale + dof_ar[:, :, 8] * 0.010).clamp(-0.028, 0.028)

    # Per-corner independent AR(1) sub-pixel residual normalized to plate width
    sigma_u = (sigma_mult * (w_px / w_px.clamp_min(12.0))).clamp(0.004, 0.022)[:, :, None, None]
    ar_c = torch.matmul(l_ar, (eps_unit * sigma_u).reshape(bsz, t_len, 8)).reshape(bsz, t_len, 4, 2)

    has_rear_seal = (has_seal[:, None] & (plate_sides == 1)).float()
    recess_frac = ((top_recess_u - 0.82) / 0.18).clamp(0.0, 1.0) * 0.082 * stage_scale

    # Assemble [TL, TR, BR, BL] normalized (du, dv)
    du_tl = tx_bt - dm_l - kh - 0.5 * sh + has_rear_seal * 0.012
    du_tr = tx_bt + dm_r + kh - 0.5 * sh
    du_br = tx_bt + dm_r - kh + 0.5 * sh
    du_bl = tx_bt - dm_l + kh + 0.5 * sh

    dv_tl = ty_bt - dm_t - kv + has_rear_seal * 0.016 + recess_frac
    dv_tr = ty_bt - dm_t + kv + recess_frac
    dv_br = ty_bt + dm_b - kv
    dv_bl = ty_bt + dm_b + kv

    du_struct = torch.stack([du_tl, du_tr, du_br, du_bl], dim=-1)
    dv_struct = torch.stack([dv_tl, dv_tr, dv_br, dv_bl], dim=-1)
    uv_delta = torch.stack([du_struct, dv_struct], dim=-1) + torch.stack(
        [ar_c[..., 0], ar_c[..., 1] * 1.4], dim=-1
    )

    delta_roi = uv_delta[..., 0:1] * u_vec[:, :, None, :] + uv_delta[..., 1:2] * v_vec[:, :, None, :]
    noisy_roi = gt_roi + delta_roi
    noisy_full = noisy_roi + roi_origin_xy[:, :, None, :]
    return noisy_roi, noisy_full, uv_delta


def _overhang_umbra_roi_impl(
    hdr_roi_rgb: torch.Tensor,
    contrast_survival: torch.Tensor,
    corners_roi_gt: torch.Tensor,
    is_monochrome_ir: torch.Tensor,
    is_temporary: torch.Tensor,
    regime_ids: torch.Tensor,
    u_overhang: torch.Tensor,
    v_jitter_bt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dump-truck tailgate overhang upper-half umbra shadow + lower-half asphalt bounce light (~16% of RGB seqs)."""
    bsz, t_len, _, h_r, w_r = hdr_roi_rgb.shape
    dev = hdr_roi_rgb.device

    valid_rgb = (
        (~is_temporary)
        & (regime_ids != 11)
        & (regime_ids != 13)
        & ((is_monochrome_ir < 0.5).all(dim=1))
    )
    has_overhang = valid_rgb & (u_overhang[:, 0] < 0.16)

    p_top = 0.5 * (corners_roi_gt[:, :, 0] + corners_roi_gt[:, :, 1])
    p_bot = 0.5 * (corners_roi_gt[:, :, 3] + corners_roi_gt[:, :, 2])
    axis_v = p_bot - p_top
    len2 = (axis_v[..., 0] ** 2 + axis_v[..., 1] ** 2).clamp_min(4.0)

    u_grid = torch.arange(w_r, device=dev, dtype=torch.float32).view(1, 1, 1, w_r) + 0.5
    v_grid = torch.arange(h_r, device=dev, dtype=torch.float32).view(1, 1, h_r, 1) + 0.5
    v_plate = (
        (u_grid - p_top[..., 0:1, None]) * axis_v[..., 0:1, None]
        + (v_grid - p_top[..., 1:2, None]) * axis_v[..., 1:2, None]
    ) / len2[:, :, None, None]

    v_cut = (0.32 + 0.20 * u_overhang[:, 1:2] + 0.025 * (v_jitter_bt - 0.5))[:, :, None, None]
    u_dark = (0.06 + 0.12 * u_overhang[:, 2:3])[:, :, None, None]
    w_pen = (0.035 + 0.050 * u_overhang[:, 3:4])[:, :, None, None]
    b_asph = (1.03 + 0.09 * u_overhang[:, 4:5])[:, :, None, None]

    s_lit = torch.sigmoid((v_plate - v_cut) / (w_pen * 0.35)).unsqueeze(2)  # (B, T, 1, H, W)
    umbra_tint = torch.tensor([0.92, 0.97, 1.08], device=dev, dtype=torch.float32).view(1, 1, 3, 1, 1)
    bounce_tint = torch.tensor([1.02, 1.00, 0.97], device=dev, dtype=torch.float32).view(1, 1, 3, 1, 1)
    sh_mult = (u_dark.unsqueeze(2) * umbra_tint) * (1.0 - s_lit) + (b_asph.unsqueeze(2) * bounce_tint) * s_lit

    hdr_out = torch.where(has_overhang[:, None, None, None, None], hdr_roi_rgb * sh_mult, hdr_roi_rgb)

    # Lower-row serial slots (7..11) stay 100% un-attenuated; upper-row slots attenuate only under deepest 0.06x umbra
    upper_surv = (u_overhang[:, 2:3] * (0.12 / 0.14) + (0.06 / 0.14)).clamp(0.42, 1.0)[:, :, None]
    cs_upper = torch.where(has_overhang[:, None, None], contrast_survival[:, :, :7] * upper_surv, contrast_survival[:, :, :7])
    cs_out = torch.cat([cs_upper, contrast_survival[:, :, 7:]], dim=-1)
    return hdr_out, cs_out, has_overhang


def _assemble_targets_tail_impl(
    is_temp_seq: torch.Tensor,
    regime_ids: torch.Tensor,
    frame_slot_targets: torch.Tensor,
    occ_total: torch.Tensor,
    contrast_survival: torch.Tensor,
    isp_readability_mult: torch.Tensor,
    shaded_slot_glare: torch.Tensor,
    geom_g0: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, t_len, _ = frame_slot_targets.shape
    is_garb_seq = regime_ids == 13
    is_mixed_seq = regime_ids == 11

    valid_seq = (~is_temp_seq) & (~is_garb_seq)
    frame_valid_mask = valid_seq[:, None].expand(bsz, t_len).float()

    # Attenuate single-frame optical readability on distant sub-Nyquist frames (`0.06` floor when `g0 < -0.75`)
    res_upper = ((geom_g0 + 0.78) / 0.93).clamp(0.05, 1.0).unsqueeze(-1)
    res_lower = torch.where(
        geom_g0 < -0.75,
        0.06 + 0.08 * ((geom_g0 + 1.25) / 0.50).clamp(0.0, 1.0),
        ((geom_g0 + 0.83) / 0.58).clamp(0.14, 1.0),
    ).clamp(0.06, 1.0).unsqueeze(-1)
    res_mult = torch.cat([res_upper.expand(bsz, t_len, 7), res_lower.expand(bsz, t_len, 5)], dim=-1)

    raw_read = ((1.0 - occ_total) * contrast_survival * isp_readability_mult * res_mult).clamp(0.0, 1.0)
    is_pad_ft = frame_slot_targets == pad_id
    reg_part = torch.where(is_pad_ft[:, :, 0:4], raw_read[:, :, 0:1].expand(-1, -1, 4), raw_read[:, :, 0:4])
    cls_part = torch.where(is_pad_ft[:, :, 4:7], raw_read[:, :, 4:5].expand(-1, -1, 3), raw_read[:, :, 4:7])
    raw_propagated = torch.cat([reg_part, cls_part, raw_read[:, :, 7:12]], dim=-1)

    slot_readability = (raw_propagated * frame_valid_mask[:, :, None]).clamp(0.0, 1.0)
    slot_glare = (shaded_slot_glare * frame_valid_mask[:, :, None]).clamp(0.0, 1.0)

    # Multi-frame L2 SNR integration so persistent 12..26px SR bursts (`T >= 4`) stay READABLE (`status = 0`)
    # while genuinely wiped-out or single-frame sub-Nyquist crops become UNREADABLE (`status = 1`).
    wiped_out = torch.linalg.vector_norm(slot_readability, ord=2, dim=1).amax(dim=-1) < 0.095
    target_status = torch.zeros_like(regime_ids)
    target_status = torch.where(is_temp_seq | is_garb_seq | wiped_out, torch.ones_like(target_status), target_status)
    target_status = torch.where(is_mixed_seq, torch.full_like(target_status, 2), target_status)
    return frame_valid_mask, slot_readability, slot_glare, target_status


def _livecam_multiscale_tone_impl(
    crops_flat: torch.Tensor,
    noisy_full_corners: torch.Tensor,
    c_crisp: torch.Tensor,
    is_monochrome_ir: torch.Tensor,
    regime_ids: torch.Tensor,
    is_approach: torch.Tensor,
    is_lowres_sr: torch.Tensor,
    has_overhang: torch.Tensor,
    wb_b: torch.Tensor,
    u_seq: torch.Tensor,
    tone_rands: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Photometric RGB white-balance `[0.72, 1.28]` + tone/shadow curves + close-up `c_crisp` branch.

    NOTE: All optical resolution bottlenecking (`12..32px`) now runs BEFORE `SensorCodecGPU`
    so H.264 `8x8`/`4x4` DCT blocks, P-frame residuals, `YUV420` chroma bleed, USM halos,
    and sensor shot/read noise survive intact into the rectified canonical crop!
    """
    bsz, t_len = is_monochrome_ir.shape
    n_bt, _, h_t, w_t = crops_flat.shape
    dev = crops_flat.device
    del h_t, w_t

    is_mono_flat = (is_monochrome_ir > 0.5).reshape(n_bt, 1, 1, 1)
    wb_norm_b = wb_b / wb_b.mean(dim=1, keepdim=True).clamp_min(1e-4)
    wb_bt = wb_norm_b[:, None].expand(bsz, t_len, 3, 1, 1).reshape(n_bt, 3, 1, 1)
    do_wb = (u_seq < 0.68).expand(bsz, t_len).reshape(n_bt, 1, 1, 1) & (~is_mono_flat)
    crops_wb = torch.where(do_wb, (crops_flat * wb_bt).clamp(0.0, 1.0), crops_flat)

    t_idx = torch.arange(t_len, device=dev).unsqueeze(0)
    half_t = max(1, t_len // 2)
    reg_2d = regime_ids[:, None].expand(bsz, t_len)
    not_app_bt = (~is_approach)[:, None].expand(bsz, t_len)
    not_lr_bt = (~is_lowres_sr)[:, None].expand(bsz, t_len)
    not_ovh_bt = (~has_overhang)[:, None].expand(bsz, t_len)
    can_crisp_bt = (
        (is_monochrome_ir < 0.5)
        & (reg_2d != 6)
        & (reg_2d != 7)
        & (reg_2d != 11)
        & (reg_2d != 13)
        & not_app_bt
        & not_lr_bt
        & not_ovh_bt
    )
    use_crisp_bt = can_crisp_bt & (
        ((u_seq >= 0.78) & (u_seq < 0.88) & (t_idx >= half_t)) | (u_seq >= 0.88)
    )
    use_crisp_flat = use_crisp_bt.reshape(n_bt, 1, 1, 1)
    crops_sel = torch.where(use_crisp_flat, c_crisp, crops_wb)

    ones_bt = torch.ones((bsz, t_len), device=dev, dtype=noisy_full_corners.dtype)
    scale_bt = torch.where(use_crisp_bt, torch.full_like(ones_bt, 2.15), ones_bt)
    c_center = noisy_full_corners.mean(dim=(1, 2), keepdim=True)
    scaled_corners = c_center + (noisy_full_corners - c_center) * scale_bt[:, :, None, None]

    do_tone = tone_rands[:, 0:1] < 0.40
    do_shadow = (tone_rands[:, 0:1] >= 0.40) & (tone_rands[:, 0:1] < 0.54) & (~is_mono_flat)
    gamma = 0.78 + 0.46 * tone_rands[:, 1:2]
    gain = 0.78 + 0.38 * tone_rands[:, 2:3]
    pedestal = -0.03 + 0.10 * tone_rands[:, 3:4]
    c_gam = crops_sel.clamp(1e-4, 1.0).pow(gamma)
    toned = (0.5 + gain * (c_gam - 0.5) + pedestal).clamp(0.0, 1.0)
    # Daytime shadowed rear-door exposure mode (e.g. shadowed microbus/truck plate in bright outdoor scene)
    sh_gain = 0.26 + 0.26 * tone_rands[:, 2:3]
    sh_tint = torch.tensor([0.93, 0.98, 1.07], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    shadowed = (
        crops_sel.clamp(1e-4, 1.0).pow(1.12 + 0.28 * tone_rands[:, 1:2]) * sh_gain * sh_tint
        + 0.015 * tone_rands[:, 3:4]
    ).clamp(0.0, 1.0)
    crops_out = torch.where(do_tone, toned, torch.where(do_shadow, shadowed, crops_sel))
    return crops_out, scaled_corners


_DRIZZLE_TABLES_CACHE: dict[tuple[torch.device, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_drizzle_tables(dev: torch.device, h_c: int, w_c: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (dev, h_c, w_c)
    cached = _DRIZZLE_TABLES_CACHE.get(key)
    if cached is not None:
        return cached
    yy = (torch.arange(h_c, device=dev, dtype=torch.float32) + 0.5) * (2.0 / float(h_c)) - 1.0
    xx = (torch.arange(w_c, device=dev, dtype=torch.float32) + 0.5) * (2.0 / float(w_c)) - 1.0
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    dx_list: list[float] = []
    dy_list: list[float] = []
    for dy_i in (-1, 0, 1):
        for dx_i in (-2, -1, 0, 1, 2):
            dx_list.append(float(dx_i))
            dy_list.append(float(dy_i))
    dx_tab = torch.tensor(dx_list, device=dev, dtype=torch.float32).view(1, 15)
    dy_tab = torch.tensor(dy_list, device=dev, dtype=torch.float32).view(1, 15)
    _DRIZZLE_TABLES_CACHE[key] = (base_grid, dx_tab, dy_tab)
    return base_grid, dx_tab, dy_tab


def _rolling_canonical_drizzle_impl(
    crops: torch.Tensor,
    noisy_full_corners: torch.Tensor,
    camera_ids: torch.Tensor,
    enable_seq: torch.Tensor,
) -> torch.Tensor:
    """Rolling multi-frame sub-pixel canonical Drizzle super-resolution accumulator (`w_t < 54px`).

    Gates accumulation by inner-plate luminance `zncc >= 0.56` + same-camera check, and decays
    stale low-res history by `(w_ema / w_t)^3` during rapid far-to-near vehicle approaches.
    """
    bsz, t_len, _, h_c, w_c = crops.shape
    if t_len < 2 or h_c < 32 or w_c < 64:
        return crops
    dev = crops.device
    base_grid, dx_tab, dy_tab = _get_drizzle_tables(dev, h_c, w_c)

    w_top = torch.linalg.norm(noisy_full_corners[:, :, 1] - noisy_full_corners[:, :, 0], dim=-1)
    w_bot = torch.linalg.norm(noisy_full_corners[:, :, 2] - noisy_full_corners[:, :, 3], dim=-1)
    w_t = (0.5 * (w_top + w_bot)).clamp_min(4.0)

    y0 = max(2, int(round(0.125 * h_c)))
    y1 = min(h_c - 2, int(round(0.875 * h_c)))
    x0 = max(3, int(round(0.10 * w_c)))
    x1 = min(w_c - 3, int(round(0.90 * w_c)))

    driz_img = crops[:, 0]
    driz_wt = torch.ones((bsz,), device=dev, dtype=torch.float32)
    w_ema = w_t[:, 0]
    out_frames = [driz_img]
    bx0 = base_grid[..., 0]
    by0 = base_grid[..., 1]

    for t_i in range(1, t_len):
        cur = crops[:, t_i]
        wt_cur = w_t[:, t_i]
        do_driz = enable_seq & (wt_cur < 54.0) & (camera_ids[:, t_i] == camera_ids[:, t_i - 1])

        lum_cur = (0.299 * cur[:, 0] + 0.587 * cur[:, 1] + 0.114 * cur[:, 2])[:, y0:y1:2, x0:x1:2]
        lc_flat = lum_cur.reshape(bsz, 1, -1)
        lc_zm = lc_flat - lc_flat.mean(dim=-1, keepdim=True)
        lc_u = lc_zm / lc_zm.norm(dim=-1, keepdim=True).clamp_min(1e-4)

        lum_prev = 0.299 * driz_img[:, 0] + 0.587 * driz_img[:, 1] + 0.114 * driz_img[:, 2]
        lp_pad = F.pad(lum_prev.unsqueeze(1), (2, 2, 1, 1), mode="replicate").squeeze(1)
        cand_patches: list[torch.Tensor] = []
        for dy_i in (-1, 0, 1):
            sy0 = y0 + 1 - dy_i
            sy1 = y1 + 1 - dy_i
            for dx_i in (-2, -1, 0, 1, 2):
                sx0 = x0 + 2 - dx_i
                sx1 = x1 + 2 - dx_i
                cand_patches.append(lp_pad[:, sy0:sy1:2, sx0:sx1:2].reshape(bsz, -1))
        lp_stack = torch.stack(cand_patches, dim=1)  # (B, 15, P)
        lp_zm = lp_stack - lp_stack.mean(dim=-1, keepdim=True)
        lp_u = lp_zm / lp_zm.norm(dim=-1, keepdim=True).clamp_min(1e-4)

        zncc_all = (lc_u * lp_u).sum(dim=-1)  # (B, 15)
        best_zncc = zncc_all.amax(dim=-1)
        w_soft = F.softmax((zncc_all - best_zncc[:, None]) * 28.0, dim=-1)
        dx_sub = (w_soft * dx_tab).sum(dim=-1)
        dy_sub = (w_soft * dy_tab).sum(dim=-1)

        g_shift = torch.stack(
            [
                bx0 - dx_sub[:, None, None] * (2.0 / float(w_c)),
                by0 - dy_sub[:, None, None] * (2.0 / float(h_c)),
            ],
            dim=-1,
        )
        driz_aligned = F.grid_sample(
            driz_img, g_shift, mode="bilinear", padding_mode="border", align_corners=False
        )

        gate_ok = do_driz & (best_zncc >= 0.56)
        decay = ((w_ema / wt_cur).clamp(0.0, 1.15) ** 3) * 0.88
        w_prior = (driz_wt * decay).clamp(0.0, 4.5)
        new_wt = w_prior + 1.0
        blended = (w_prior[:, None, None, None] * driz_aligned + cur) / new_wt[:, None, None, None]
        blur3 = F.avg_pool2d(F.pad(blended, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        sr_gain = (0.16 * (w_prior / new_wt))[:, None, None, None]
        blended_sr = (blended + sr_gain * (blended - blur3)).clamp(0.0, 1.0)

        driz_img = torch.where(gate_ok[:, None, None, None], blended_sr, cur)
        driz_wt = torch.where(gate_ok, new_wt, torch.ones_like(driz_wt))
        w_ema = torch.where(gate_ok, (w_prior * w_ema + wt_cur) / new_wt, wt_cur)
        out_frames.append(torch.where(do_driz[:, None, None, None], driz_img, cur))

    return torch.stack(out_frames, dim=1)


def _compute_geom_photo_core_impl(
    c_flat: torch.Tensor,
    lap_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_items, _, h_c, w_c = c_flat.shape
    y_lum = 0.299 * c_flat[:, 0] + 0.587 * c_flat[:, 1] + 0.114 * c_flat[:, 2]
    mu_y = y_lum.mean(dim=(-2, -1)).clamp(0.0, 1.0)
    r_clip = (y_lum > 0.96).float().mean(dim=(-2, -1)).clamp(0.0, 1.0)

    chroma_px = (c_flat.amax(dim=1) - c_flat.amin(dim=1)).reshape(n_items, h_c * w_c)
    top_k = max(1, int(round(0.15 * float(h_c * w_c))))
    s_chroma = (
        torch.topk(chroma_px, k=top_k, dim=-1, largest=True, sorted=False)
        .values.mean(dim=-1)
        .clamp(0.0, 1.0)
    )

    y_pad = F.pad(y_lum.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
    lap = F.conv2d(y_pad, lap_k).reshape(n_items, h_c * w_c)
    lap_var = lap.var(dim=-1, correction=0)
    sharp = (0.25 * torch.log1p(100.0 * lap_var)).clamp(0.0, 1.0)
    return mu_y, r_clip, s_chroma, sharp


def compute_geom_meta_gpu(
    corners_xy: torch.Tensor | None,
    crops: torch.Tensor,
    prev_corners_xy: torch.Tensor | None = None,
    camera_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute exact 10-D geometric & photometric metadata vector `geom_meta` on GPU/CPU.

    Matches `vpocr_det.quad_cropper.extract_quad_geometry_meta` element-for-element:
      - `crops`: `float32 (..., 3, H, W)` in `[0.0, 1.0]`
      - `corners_xy`: `float32 (..., 4, 2)` `[TL, TR, BR, BL]` (or `None`)
      - Returns: `float32 (..., 10)`
    """
    lead_shape = crops.shape[:-3]
    h_c, w_c = crops.shape[-2], crops.shape[-1]
    n_items = int(math.prod(lead_shape)) if lead_shape else 1
    dev = crops.device

    c_flat = crops.reshape(n_items, 3, h_c, w_c).float().clamp(0.0, 1.0)
    lap_k = _get_lap_kernel(dev)
    mu_y, r_clip, s_chroma, sharp = _dispatch_seq_cuda(
        "geom_photo_core", _compute_geom_photo_core_impl, c_flat, lap_k
    )

    if corners_xy is None:
        w_mean = torch.full((n_items,), float(w_c) * 0.90, device=dev, dtype=torch.float32)
        h_mean = torch.full((n_items,), float(h_c) * 0.90, device=dev, dtype=torch.float32)
        w_top = w_bot = w_mean
        h_left = h_right = h_mean
        delta_norm = torch.zeros((n_items,), device=dev, dtype=torch.float32)
    else:
        q = corners_xy.reshape(n_items, 4, 2).float()
        w_top = torch.linalg.norm(q[:, 1] - q[:, 0], dim=-1)
        h_right = torch.linalg.norm(q[:, 2] - q[:, 1], dim=-1)
        w_bot = torch.linalg.norm(q[:, 2] - q[:, 3], dim=-1)
        h_left = torch.linalg.norm(q[:, 3] - q[:, 0], dim=-1)
        w_mean = (0.5 * (w_top + w_bot)).clamp_min(2.0)
        h_mean = (0.5 * (h_left + h_right)).clamp_min(1.0)

        if prev_corners_xy is not None:
            qp = prev_corners_xy.reshape(n_items, 4, 2).float()
            frob = torch.linalg.norm((q - qp).reshape(n_items, 8), dim=-1)
            delta_norm = (frob / (2.0 * w_mean)).clamp(0.0, 1.0)
        elif len(lead_shape) >= 2 and lead_shape[-1] >= 2:
            # Infer previous frame along T axis (-1 of lead_shape)
            bsz = int(math.prod(lead_shape[:-1]))
            seq_t = lead_shape[-1]
            q_bt = q.reshape(bsz, seq_t, 4, 2)
            wm_bt = w_mean.reshape(bsz, seq_t)
            frob_t1 = torch.linalg.norm((q_bt[:, 1:] - q_bt[:, :-1]).reshape(bsz, seq_t - 1, 8), dim=-1)
            d_t1 = (frob_t1 / (2.0 * wm_bt[:, 1:])).clamp(0.0, 1.0)
            if camera_ids is not None:
                cam_bt = camera_ids.reshape(bsz, seq_t)
                same_cam = (cam_bt[:, 1:] == cam_bt[:, :-1]).float()
                d_t1 = d_t1 * same_cam
            delta_norm = torch.cat([torch.zeros((bsz, 1), device=dev), d_t1], dim=1).reshape(n_items)
        else:
            delta_norm = torch.zeros((n_items,), device=dev, dtype=torch.float32)

    g0 = (0.5 * torch.log2((w_mean / 64.0).clamp_min(1e-3))).clamp(-1.0, 1.0)
    g1 = (0.5 * torch.log2((h_mean / 32.0).clamp_min(1e-3))).clamp(-1.0, 1.0)
    g2 = (w_mean / (2.0 * h_mean).clamp_min(1e-3)).clamp(0.25, 2.0) - 1.0
    g3 = torch.tanh(2.5 * torch.log((h_left + 1e-3) / (h_right + 1e-3)))
    g4 = torch.tanh(2.5 * torch.log((w_bot + 1e-3) / (w_top + 1e-3)))

    meta = torch.stack([g0, g1, g2, g3, g4, delta_norm, sharp, mu_y, s_chroma, r_clip], dim=-1)
    return meta.reshape(*lead_shape, 10)


class GPUSynthSequenceEngine:
    """On-the-fly 2.5D Physics & GPU Sim-to-Real Sequence Batch Synthesizer."""

    def __init__(
        self,
        vocab: PlateVocab | None = None,
        sampler: StratifiedPlateSampler | None = None,
        atlas: GlyphSDFAtlas | None = None,
        device: torch.device | str | None = None,
        out_hw: tuple[int, int] = (80, 160),
        margin_ratio: float = 0.05,
        curriculum_stage: int = 4,
        train_mode: bool = True,
        include_stroke_mask: bool = False,
        seed: int = 42,
    ) -> None:
        self.vocab = vocab or PlateVocab()
        self.atlas = atlas or get_default_glyph_atlas(self.vocab)
        self.sampler = sampler or StratifiedPlateSampler(seed=seed)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        self.margin_ratio = float(margin_ratio)
        self.curriculum_stage = 4
        self.set_curriculum_stage(curriculum_stage)
        self.train_mode = bool(train_mode)
        self.include_stroke_mask = bool(include_stroke_mask)
        self._seed = int(seed)
        self._step_counter = 0
        self._rng = random.Random(seed)

        self.surface_builder = PlateSurface25DBuilder(
            vocab=self.vocab, atlas=self.atlas, device=self.device
        )
        self.shader = ReliefPBRShaderGPU(device=self.device)
        self.codec = SensorCodecGPU(device=self.device)

        # Online token confusion boost table (chars / regions dynamically upweighted during training)
        self._boosted_chars: list[str] = []
        self._boost_prob: float = 0.0
        self._regions_by_char: dict[str, list[str]] = {}
        for reg in self.vocab.regions:
            for ch in set(reg):
                self._regions_by_char.setdefault(ch, []).append(reg)

        self._ext_corners_cache: dict[torch.device, torch.Tensor] = {}

    def to(self, device: torch.device | str) -> GPUSynthSequenceEngine:
        self.device = torch.device(device)
        self.surface_builder.to(self.device)
        self.shader.to(self.device)
        self.codec.to(self.device)
        return self

    def train(self, mode: bool = True) -> GPUSynthSequenceEngine:
        self.train_mode = bool(mode)
        return self

    def eval(self) -> GPUSynthSequenceEngine:
        self.train_mode = False
        return self

    def set_curriculum_stage(self, stage: int) -> None:
        """Configure progressive curriculum stage (`1, 2, 3, 4`)."""
        st = int(stage)
        if st not in (1, 2, 3, 4):
            raise ValueError(f"curriculum_stage must be in {{1, 2, 3, 4}}, got {stage}")
        self.curriculum_stage = st

    def update_confusion_boost(
        self,
        confusion_or_weights: torch.Tensor | np.ndarray | Mapping[str, float] | None,
        boost_prob: float = 0.25,
    ) -> None:
        """Update online hard-character boosting from a `(V, V)` confusion matrix or `(V,)` error vector."""
        if confusion_or_weights is None:
            self._boosted_chars = []
            self._boost_prob = 0.0
            return

        scores: dict[str, float] = {}
        if isinstance(confusion_or_weights, Mapping):
            for k, v in confusion_or_weights.items():
                if float(v) > 0 and k in self.vocab.token_to_id and k != "<PAD>":
                    scores[str(k)] = float(v)
        else:
            arr = (
                confusion_or_weights.detach().float().cpu().numpy()
                if isinstance(confusion_or_weights, torch.Tensor)
                else np.asarray(confusion_or_weights, dtype=np.float32)
            )
            if arr.ndim == 2 and arr.shape[0] == self.vocab.size:
                off_diag = arr.copy()
                np.fill_diagonal(off_diag, 0.0)
                err = off_diag.sum(axis=1)
            elif arr.ndim == 1 and arr.shape[0] == self.vocab.size:
                err = arr
            else:
                raise ValueError(f"Unexpected confusion_or_weights shape: {arr.shape}")
            for tid in range(1, self.vocab.size):
                if float(err[tid]) > 0.0:
                    scores[self.vocab.id_to_token[tid]] = float(err[tid])

        if not scores:
            self._boosted_chars = []
            self._boost_prob = 0.0
            return

        # Jaynes Maximum-Entropy / Boltzmann active curriculum over top-48 high-surprise tokens
        raw_vec = np.zeros((self.vocab.size,), dtype=np.float32)
        for ch, sc in scores.items():
            raw_vec[self.vocab.token_to_id[ch]] = float(sc)
        gibbs_w = boltzmann_kl_token_weights(raw_vec, self.vocab.size, beta=1.35, uniform_mix=0.14)
        ranked = sorted(
            ((self.vocab.id_to_token[tid], float(gibbs_w[tid])) for tid in range(1, self.vocab.size) if gibbs_w[tid] > 0),
            key=lambda kv: kv[1],
            reverse=True,
        )[:48]
        pool: list[str] = []
        max_sc = max(v for _, v in ranked)
        for ch, sc in ranked:
            reps = max(1, int(round(8.0 * sc / max(1e-6, max_sc))))
            pool.extend([ch] * reps)
        self._boosted_chars = pool
        self._boost_prob = float(np.clip(boost_prob, 0.0, 0.60))

    def _apply_online_boost(
        self,
        pt: PlateText,
        cfg: PlateConfig,
        rng: random.Random,
    ) -> PlateText:
        if not self._boosted_chars or rng.random() >= self._boost_prob:
            return pt
        ch = rng.choice(self._boosted_chars)
        if ch in self._regions_by_char:
            reg = rng.choice(self._regions_by_char[ch])
            cand = PlateText(region=reg, classification=pt.classification, hiragana=pt.hiragana, serial=pt.serial)
            if PlateValidator.is_valid(cand, cfg.plate_type, cfg.is_graphic):
                return cand
        if ch.isdigit() and ch != "0":
            s_list = list(str(pt.serial))
            s_list[-1] = ch
            cand = PlateText(region=pt.region, classification=pt.classification, hiragana=pt.hiragana, serial=int("".join(s_list)))
            if PlateValidator.is_valid(cand, cfg.plate_type, cfg.is_graphic):
                return cand
        if len(ch) == 1 and ch in PlateValidator.get_valid_hiragana_for_type(cfg.plate_type):
            cand = PlateText(region=pt.region, classification=pt.classification, hiragana=ch, serial=pt.serial)
            if PlateValidator.is_valid(cand, cfg.plate_type, cfg.is_graphic):
                return cand
        return pt

    def _get_ext_plate_corners(self, bsz: int, dev: torch.device) -> torch.Tensor:
        base = self._ext_corners_cache.get(dev)
        if base is None:
            base = torch.tensor(EXT_PLATE_CORNERS_XY, device=dev, dtype=torch.float32).unsqueeze(0)
            self._ext_corners_cache[dev] = base
        return base.expand(bsz, 4, 2)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int = 32,
        seq_len: int = 8,
        *,
        out_hw: tuple[int, int] | None = None,
        regime_override: str | Sequence[str] | None = None,
        plate_overrides: Sequence[tuple[PlateText, PlateConfig]] | None = None,
        plate_side_override: int | Sequence[int] | None = None,
        seed: int | None = None,
        coaxial_ir_power: float | None = None,
        glare_boost: float | None = None,
        jpeg_quality_override: float | None = None,
        halation_boost_override: float | None = None,
        include_stroke_mask: bool | None = None,
        approach_prob: float | None = None,
        lowres_sr_prob: float | None = None,
    ) -> dict[str, Any]:
        """Synthesize a multi-frame batch conforming 100% to `v2_shared_contract.md`."""
        bsz = int(batch_size)
        t_len = int(seq_len)
        target_hw = (int(out_hw[0]), int(out_hw[1])) if out_hw is not None else self.out_hw
        dev = self.device
        stage = self.curriculum_stage

        if seed is None:
            step_seed = self._seed + self._step_counter * 7919
            self._step_counter += 1
        else:
            step_seed = int(seed)
        rng = random.Random(step_seed)

        # 1. Sample 3D SE(3) trajectories & per-sequence regime assignments first (zero GPU sync!)
        traj = sample_se3_trajectories(
            batch_size=bsz,
            seq_len=t_len,
            curriculum_stage=stage,
            regime_override=regime_override,
            primary_sides=plate_side_override,
            device=dev,
            seed=step_seed + 11,
            approach_prob=approach_prob if approach_prob is not None else (None if self.train_mode else 0.0),
            lowres_sr_prob=lowres_sr_prob if lowres_sr_prob is not None else (None if self.train_mode else 0.0),
        )

        # 2. Sample primary (and sparse secondary ONLY for track_mixed) PlateText + PlateConfig
        pri_texts: list[PlateText] = []
        pri_cfgs: list[PlateConfig] = []
        sec_texts: list[PlateText] = []
        sec_cfgs: list[PlateConfig] = []
        sec_batch_indices: list[int] = []
        sec_sides: list[int] = []
        sides_cpu = traj.primary_sides_cpu

        for b in range(bsz):
            tag = traj.regime_tags[b]
            if plate_overrides is not None and b < len(plate_overrides):
                pt, cfg = plate_overrides[b]
                cfg = PlateConfig(**{**cfg.__dict__})
                if tag == "track_mixed":
                    _, (pt2, cfg2) = self.sampler.sample_hard_negative_pair(seed=step_seed * 131 + b)
                    sec_texts.append(pt2)
                    sec_cfgs.append(cfg2)
                    sec_batch_indices.append(b)
                    sec_sides.append(sides_cpu[b])
            elif tag == "track_mixed":
                (pt, cfg), (pt2, cfg2) = self.sampler.sample_hard_negative_pair(seed=step_seed * 131 + b)
                sec_texts.append(pt2)
                sec_cfgs.append(cfg2)
                sec_batch_indices.append(b)
                sec_sides.append(sides_cpu[b])
            else:
                pt, cfg = self.sampler.sample(seed=step_seed * 131 + b if seed is not None else None)
                pt = self._apply_online_boost(pt, cfg, rng)

            if tag == "illuminated_night":
                cfg.is_illuminated = True
                cfg.is_temporary_plate = False
            elif tag == "temporary_plate":
                cfg.is_temporary_plate = True
                cfg.is_illuminated = False
                cfg.with_seal = False
            elif (
                tag in ("heavy_rain_droplets", "steep_oblique", "day_clear", "day_harsh_shadow", "dusk_wdr", "turn_in_sweep")
                and stage >= 3
                and plate_overrides is None
            ):
                p_mud = 0.34 if tag in ("heavy_rain_droplets", "steep_oblique") else 0.16
                if rng.random() < p_mud:
                    cfg.mud_level = rng.uniform(0.10, 0.52)

            if sides_cpu[b] == 1 and not cfg.is_temporary_plate and plate_overrides is None:
                if cfg.plate_type in (PlateType.STANDARD_PRIVATE, PlateType.STANDARD_COMMERCIAL):
                    if rng.random() < 0.80:
                        cfg.with_seal = True

            pri_texts.append(pt)
            pri_cfgs.append(cfg)

        do_layout_jitter = bool(self.train_mode and stage >= 2 and plate_overrides is None)
        surface_pri = self.surface_builder.build_batch(
            plate_texts=pri_texts,
            configs=pri_cfgs,
            plate_sides=sides_cpu,
            seed=step_seed + 23,
            layout_jitter=do_layout_jitter,
        )
        if sec_batch_indices:
            if len(sec_batch_indices) == 1 and dev.type == "cuda":
                # Pad B_sec=1 -> 2 on CUDA so Triton dynamic=True never recompiles for dim==1
                surface_sec = self.surface_builder.build_batch(
                    plate_texts=[sec_texts[0], sec_texts[0]],
                    configs=[sec_cfgs[0], sec_cfgs[0]],
                    plate_sides=[sec_sides[0], sec_sides[0]],
                    seed=step_seed + 37,
                    layout_jitter=do_layout_jitter,
                )
            else:
                surface_sec = self.surface_builder.build_batch(
                    plate_texts=sec_texts,
                    configs=sec_cfgs,
                    plate_sides=sec_sides,
                    seed=step_seed + 37,
                    layout_jitter=do_layout_jitter,
                )
        else:
            surface_sec = None

        # 3. Clean Frontal Daylight Canonical Reference `canon_clean_crop` (B, 3, H, W) + lit (192, 384)
        canon_clean, lit_ext = self._render_clean_canonical_reference_with_lit(surface_pri, target_hw)

        # 4. 2.5D Relief Raymarch & Multi-Spectral Physical Shader in (96, 160) Sensor ROI
        shaded = self.shader.shade_sequences(
            surface=surface_pri,
            traj=traj,
            surface_secondary=surface_sec,
            secondary_batch_indices=sec_batch_indices,
            seed=step_seed + 53,
            coaxial_ir_power=coaxial_ir_power,
            glare_boost=glare_boost,
        )

        gen = torch.Generator(device=dev)
        gen.manual_seed((step_seed * 1103515245 + 12345) & 0x7FFFFFFF)

        # 4b. Dump-truck tailgate overhang upper-half umbra shadow + lower-half asphalt bounce (~16% in stage >= 2)
        if self.train_mode and stage >= 2 and regime_override is None:
            u_overhang = torch.rand((bsz, 5), generator=gen, device=dev)
            v_jitter_bt = torch.rand((bsz, t_len), generator=gen, device=dev)
            shaded.hdr_roi_rgb, shaded.contrast_survival, has_overhang = _dispatch_seq_cuda(
                "overhang_umbra",
                _overhang_umbra_roi_impl,
                shaded.hdr_roi_rgb,
                shaded.contrast_survival,
                traj.corners_roi_gt,
                shaded.is_monochrome_ir,
                surface_pri.is_temporary,
                traj.regime_ids,
                u_overhang,
                v_jitter_bt,
            )
        else:
            has_overhang = torch.zeros((bsz,), dtype=torch.bool, device=dev)

        # 5. Sensor Optics (9-tap motion blur + defocus), Rain, NIR Noise, USM, YUV420 & H.264 8x8/4x4 DCT
        degraded = self.codec.process_rois(
            shaded=shaded,
            traj=traj,
            curriculum_stage=stage,
            seed=step_seed + 71,
            jpeg_quality_override=jpeg_quality_override,
            halation_boost_override=halation_boost_override,
        )

        # 6. Structured 8-DOF Stage-2 Perturbation + Closed-Form O(1) Rectification to out_hw
        noisy_roi_corners, noisy_full_corners, uv_delta = self._perturb_quad_corners(
            traj=traj,
            surface=surface_pri,
            stage=stage,
            gen=gen,
        )

        n_bt = bsz * t_len
        mr = self.margin_ratio

        crops_flat = warp_quad_to_canonical_gpu(
            src_imgs=degraded.roi_rgb.reshape(n_bt, 3, degraded.roi_rgb.shape[-2], degraded.roi_rgb.shape[-1]),
            quad_xy=noisy_roi_corners.reshape(n_bt, 4, 2),
            out_hw=target_hw,
            margin_ratio=mr,
        ).clamp(0.0, 1.0)

        ext_corners_base = self._get_ext_plate_corners(bsz, dev)
        ext_corners_bt = ext_corners_base[:, None, :, :] + torch.stack(
            [uv_delta[..., 0] * 352.0, uv_delta[..., 1] * 176.0], dim=-1
        )

        # 6b. Photometric RGB WB [0.72, 1.28] + tone/shadow curves + 8-DOF-jittered `c_crisp` with light DCT
        if self.train_mode and stage >= 2:
            is_app_t = (
                traj.is_approach
                if traj.is_approach is not None
                else torch.zeros((bsz,), dtype=torch.bool, device=dev)
            )
            is_lr_t = (
                traj.is_lowres_sr
                if traj.is_lowres_sr is not None
                else torch.zeros((bsz,), dtype=torch.bool, device=dev)
            )
            crops_flat, noisy_full_corners = self._apply_livecam_multiscale_and_tone_jitter(
                crops_flat=crops_flat,
                noisy_full_corners=noisy_full_corners,
                lit_ext=lit_ext,
                ext_corners_bt=ext_corners_bt,
                mr_val=mr,
                codec=self.codec,
                ir_mode=traj.sensor_ir_mode,
                is_monochrome_ir=shaded.is_monochrome_ir,
                regime_ids=traj.regime_ids,
                is_approach=is_app_t,
                is_lowres_sr=is_lr_t,
                has_overhang=has_overhang,
                bsz=bsz,
                t_len=t_len,
                target_hw=target_hw,
                gen=gen,
            )

        # 7. Phase-Preserving Fourier Amplitude Spectrum Perturbation (p=0.25 marginal frame prob in train mode)
        if self.train_mode and rng.random() < 0.50:
            crops_flat = self._apply_fourier_amplitude_perturbation(
                crops_flat=crops_flat,
                prob=0.50,
                gen=gen,
            )

        # 7b. Recessed dump-truck / bumper top-lip shadow & partial top overhang (~14% in train stage >= 2)
        top_lip_rows = torch.zeros((n_bt, 1, 1, 1), dtype=torch.long, device=dev)
        if self.train_mode and stage >= 2:
            has_lip = (torch.rand((bsz, 1), generator=gen, device=dev) < 0.14).expand(bsz, t_len).reshape(n_bt, 1, 1, 1)
            lip_h = torch.randint(2, max(3, int(round(target_hw[0] * 0.10)) + 1), (bsz, 1), generator=gen, device=dev)
            top_lip_rows = torch.where(has_lip, lip_h.expand(bsz, t_len).reshape(n_bt, 1, 1, 1), top_lip_rows)
            row_idx = torch.arange(target_hw[0], device=dev).view(1, 1, target_hw[0], 1)
            in_lip = (row_idx < top_lip_rows).float()
            lip_lum = (0.04 + 0.14 * torch.rand((bsz, 1, 1, 1), generator=gen, device=dev))[:, None].expand(
                bsz, t_len, 1, 1, 1
            ).reshape(n_bt, 1, 1, 1)
            lip_rgb = (lip_lum + 0.02 * (torch.rand((bsz, 3, 1, 1), generator=gen, device=dev) - 0.5)[:, None].expand(
                bsz, t_len, 3, 1, 1
            ).reshape(n_bt, 3, 1, 1)).clamp(0.0, 1.0)
            is_ir_flat = (shaded.is_monochrome_ir > 0.5).reshape(n_bt, 1, 1, 1)
            lip_col = torch.where(is_ir_flat, lip_lum.expand(n_bt, 3, 1, 1), lip_rgb)
            crops_flat = crops_flat * (1.0 - in_lip) + lip_col * in_lip

        crops = crops_flat.reshape(bsz, t_len, 3, target_hw[0], target_hw[1])

        # 7c. Rolling multi-frame sub-pixel canonical Drizzle super-resolution on `w_t < 54px` sequences
        if self.train_mode and stage >= 2 and t_len >= 2:
            enable_drizzle = torch.rand((bsz,), generator=gen, device=dev) < 0.75
            crops = _rolling_canonical_drizzle_impl(
                crops=crops,
                noisy_full_corners=noisy_full_corners,
                camera_ids=traj.camera_ids,
                enable_seq=enable_drizzle,
            )

        # 8. Exact GPU 10-D Geometric & Photometric Metadata `geom_meta` (B, T, 10)
        geom_meta = compute_geom_meta_gpu(
            corners_xy=noisy_full_corners,
            crops=crops,
            camera_ids=traj.camera_ids,
        )

        # 9. Calibrated Masks, Readability, Frame Slot Targets & Sequence Target Status
        frame_slot_targets = surface_pri.target_slots[:, None, :].expand(bsz, t_len, NUM_SLOTS).clone()
        if surface_sec is not None and sec_batch_indices:
            n_sec = len(sec_batch_indices)
            idx_sec = torch.from_numpy(np.asarray(sec_batch_indices, dtype=np.int64)).to(dev)
            sp_sub = traj.splice_second_plate.index_select(0, idx_sec)
            frame_slot_targets[idx_sec] = torch.where(
                sp_sub[:, :, None],
                surface_sec.target_slots[:n_sec, None, :].expand(-1, t_len, NUM_SLOTS),
                frame_slot_targets[idx_sec],
            )

        frame_valid_mask, slot_readability, slot_glare, target_status = _dispatch_seq_cuda(
            "targets_tail",
            _assemble_targets_tail_impl,
            surface_pri.is_temporary,
            traj.regime_ids,
            frame_slot_targets,
            shaded.occ_total,
            shaded.contrast_survival,
            degraded.isp_readability_mult,
            shaded.slot_glare,
            geom_meta[:, :, 0],
            int(self.vocab.pad_id),
        )

        # Train 30% of valid positive plates with `plate_sides = 2` (unknown side) so `plate_sides = 2`
        # is never learned as a spurious non-plate shortcut!
        out_plate_sides = traj.plate_sides
        if self.train_mode and plate_side_override is None:
            unk_side_seq = (torch.rand((bsz, 1), generator=gen, device=dev) < 0.30) & (
                (target_status != 1)[:, None]
            )
            out_plate_sides = torch.where(
                unk_side_seq.expand(bsz, t_len),
                torch.full_like(out_plate_sides, 2),
                out_plate_sides,
            )

        out_dict: dict[str, Any] = {
            "crops": crops,
            "canon_clean_crop": canon_clean,
            "geom_meta": geom_meta,
            "timestamps_s": traj.timestamps_s,
            "camera_ids": traj.camera_ids,
            "plate_sides": out_plate_sides,
            "frame_valid_mask": frame_valid_mask,
            "slot_readability": slot_readability,
            "slot_glare": slot_glare,
            "target_slots": surface_pri.target_slots,
            "frame_slot_targets": frame_slot_targets,
            "target_plate_type": surface_pri.target_plate_type,
            "is_monochrome_ir": shaded.is_monochrome_ir,
            "polarity_target": shaded.polarity_target,
            "target_status": target_status,
            "regime_tags": traj.regime_tags,
        }
        want_sm = self.include_stroke_mask if include_stroke_mask is None else bool(include_stroke_mask)
        if want_sm:
            ext_c_bt = ext_corners_bt.reshape(n_bt, 4, 2)
            sm_src = surface_pri.stroke_mask[:, None, :, :, :].expand(
                bsz, t_len, 1, surface_pri.stroke_mask.shape[-2], surface_pri.stroke_mask.shape[-1]
            ).reshape(
                n_bt, 1, surface_pri.stroke_mask.shape[-2], surface_pri.stroke_mask.shape[-1]
            )
            sm_warped = warp_quad_to_canonical_gpu(
                src_imgs=sm_src,
                quad_xy=ext_c_bt,
                out_hw=target_hw,
                margin_ratio=mr,
            ).clamp(0.0, 1.0)
            if self.train_mode and stage >= 2:
                r_idx = torch.arange(target_hw[0], device=dev).view(1, 1, target_hw[0], 1)
                sm_warped = sm_warped * (r_idx >= top_lip_rows).float()
            sm_bt = sm_warped.reshape(bsz, t_len, 1, target_hw[0], target_hw[1]) * frame_valid_mask[:, :, None, None, None]
            out_dict["stroke_mask_gt"] = sm_bt
        return out_dict

    def _render_clean_canonical_reference_with_lit(
        self,
        surface: PlateSurface25DPacket,
        out_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render clean frontal daylight reference `(B, 3, H, W)` + `(192, 384)` lit surface."""
        bsz = surface.albedo_rgb.shape[0]
        dev = surface.albedo_rgb.device
        lit = _dispatch_seq_cuda("canon_lit", _clean_canon_lit_impl, surface.albedo_rgb, surface.height_mm)
        ext_corners = self._get_ext_plate_corners(bsz, dev)
        canon = warp_quad_to_canonical_gpu(
            src_imgs=lit,
            quad_xy=ext_corners,
            out_hw=out_hw,
            margin_ratio=0.05,
        ).clamp(0.0, 1.0)
        return canon, lit

    def _render_clean_canonical_reference(
        self,
        surface: PlateSurface25DPacket,
        out_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Render clean frontal daylight reference `(B, 3, H, W)` with exact `margin_ratio = 0.05`."""
        canon, _ = self._render_clean_canonical_reference_with_lit(surface, out_hw)
        return canon

    def _perturb_quad_corners(
        self,
        traj: TrajectoryBatch,
        surface: PlateSurface25DPacket,
        stage: int,
        gen: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate structured 8-DOF Stage-2 quad homography jitter + AR(1) drift + coasting."""
        gt_roi = traj.corners_roi_gt
        bsz, t_len, _, _ = gt_roi.shape
        dev = gt_roi.device
        if not self.train_mode:
            return gt_roi, traj.corners_full_gt, torch.zeros((bsz, t_len, 4, 2), device=dev, dtype=gt_roi.dtype)

        l_ar = _get_ar1_matrix(dev, t_len)
        eps_unit = torch.randn((bsz, t_len, 4, 2), generator=gen, device=dev)
        dof_seq_u = torch.rand((bsz, 10), generator=gen, device=dev)
        dof_bt_n = torch.randn((bsz, t_len, 9), generator=gen, device=dev)
        top_recess_u = (
            torch.rand((bsz, 1), generator=gen, device=dev)
            if stage >= 2
            else torch.zeros((bsz, 1), device=dev)
        )
        sigma_mult = float(0.004 + 0.003 * stage)
        stage_scale = 1.0 if stage >= 2 else 0.35

        noisy_roi, noisy_full, uv_delta = _dispatch_seq_cuda(
            f"perturb_q_{stage}",
            _perturb_corners_core_impl,
            gt_roi,
            traj.roi_origin_xy,
            surface.has_seal,
            traj.plate_sides,
            l_ar,
            eps_unit,
            dof_seq_u,
            dof_bt_n,
            top_recess_u,
            sigma_mult,
            stage_scale,
        )

        if t_len >= 2 and stage >= 2:
            chaos_u = torch.rand((bsz, 8), generator=gen, device=dev)
            chaos_burst = chaotic_intermittent_corner_burst(chaos_u, bsz, t_len)
            w_approx = (gt_roi[:, :, 1, 0] - gt_roi[:, :, 0, 0]).abs().clamp(8.0, 140.0)[:, :, None, None]
            chaos_delta = chaos_burst * (0.0042 * w_approx)
            noisy_roi = noisy_roi + chaos_delta
            noisy_full = noisy_roi + traj.roi_origin_xy[:, :, None, :]

        if t_len >= 2 and stage >= 3:
            coast = (
                (torch.rand((bsz, t_len - 1), generator=gen, device=dev) < 0.05)
                & (traj.camera_ids[:, 1:] == traj.camera_ids[:, :-1])
            )[:, :, None, None]
            coasted = noisy_roi[:, :-1] + (
                torch.randn((bsz, t_len - 1, 4, 2), generator=gen, device=dev) * 0.6
            )
            noisy_roi = torch.cat([noisy_roi[:, 0:1], torch.where(coast, coasted, noisy_roi[:, 1:])], dim=1)
            noisy_full = noisy_roi + traj.roi_origin_xy[:, :, None, :]

        return noisy_roi, noisy_full, uv_delta

    @staticmethod
    def _apply_livecam_multiscale_and_tone_jitter(
        crops_flat: torch.Tensor,
        noisy_full_corners: torch.Tensor,
        lit_ext: torch.Tensor,
        ext_corners_bt: torch.Tensor,
        mr_val: float,
        codec: SensorCodecGPU,
        ir_mode: torch.Tensor,
        is_monochrome_ir: torch.Tensor,
        regime_ids: torch.Tensor,
        is_approach: torch.Tensor,
        is_lowres_sr: torch.Tensor,
        has_overhang: torch.Tensor,
        bsz: int,
        t_len: int,
        target_hw: tuple[int, int],
        gen: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """8-DOF-perturbed close-up `c_crisp` + light H.264/DCT + RGB WB `[0.72, 1.28]` + tone curves."""
        dev = crops_flat.device
        n_bt = bsz * t_len
        h_t, w_t = target_hw

        # Warp `lit_ext` across ALL (B, T) frames using exact structured 8-DOF homography jitter
        # (zero GPU-CPU `.item()` sync, identical jitter richness on both CPU and CUDA!).
        lit_bt = lit_ext[:, None].expand(bsz, t_len, 3, lit_ext.shape[-2], lit_ext.shape[-1]).reshape(
            n_bt, 3, lit_ext.shape[-2], lit_ext.shape[-1]
        )
        crisp_warped = warp_quad_to_canonical_gpu(
            src_imgs=lit_bt,
            quad_xy=ext_corners_bt.reshape(n_bt, 4, 2),
            out_hw=(h_t, w_t),
            margin_ratio=mr_val,
        ).clamp(0.0, 1.0)

        wb_b = 0.72 + 0.56 * torch.rand((bsz, 3, 1, 1), generator=gen, device=dev)
        wb_norm_bt = (wb_b / wb_b.mean(dim=1, keepdim=True).clamp_min(1e-4))[:, None].expand(
            bsz, t_len, 3, 1, 1
        ).reshape(n_bt, 3, 1, 1)
        illum_coefs = torch.rand((bsz, 3, 1, 1), generator=gen, device=dev)[:, None].expand(
            bsz, t_len, 3, 1, 1
        ).reshape(n_bt, 3, 1, 1)
        gx = torch.linspace(-0.5, 0.5, w_t, device=dev, dtype=torch.float32).view(1, 1, 1, w_t)
        gy = torch.linspace(-0.5, 0.5, h_t, device=dev, dtype=torch.float32).view(1, 1, h_t, 1)
        illum_bt = (
            (0.94 + 0.12 * illum_coefs[:, 0:1])
            + (-0.08 + 0.16 * illum_coefs[:, 1:2]) * gx
            + (-0.08 + 0.16 * illum_coefs[:, 2:3]) * gy
        )
        usm_hi_bt = (0.20 + 0.65 * torch.rand((bsz, 1, 1, 1), generator=gen, device=dev))[:, None].expand(
            bsz, t_len, 1, 1, 1
        ).reshape(n_bt, 1, 1, 1)
        n_dt = torch.float16 if dev.type == "cuda" else torch.float32
        sigma_hi = (0.006 + 0.018 * torch.rand((n_bt, 1, 1, 1), generator=gen, device=dev)).to(dtype=n_dt)
        hi_noise = torch.empty((n_bt, 3, h_t, w_t), device=dev, dtype=n_dt).normal_(generator=gen)

        crisp_wb = (crisp_warped * wb_norm_bt * illum_bt).clamp(0.0, 1.0)
        blur3 = F.avg_pool2d(F.pad(crisp_wb, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        crisp_pre = (crisp_wb + usm_hi_bt * (crisp_wb - blur3) + (hi_noise * sigma_hi).float()).clamp(0.0, 1.0)
        if (h_t % 16) == 0 and (w_t % 16) == 0:
            c_crisp = codec.apply_light_dct_crops(
                crisp_pre, ir_mode.reshape(n_bt), gen=gen, q_lo=42.0, q_hi=84.0
            )
        else:
            c_crisp = crisp_pre

        u_seq = torch.rand((bsz, 1), generator=gen, device=dev)
        tone_rands = torch.rand((n_bt, 4, 1, 1), generator=gen, device=dev)

        return _dispatch_seq_cuda(
            "livecam_tone",
            _livecam_multiscale_tone_impl,
            crops_flat,
            noisy_full_corners,
            c_crisp,
            is_monochrome_ir,
            regime_ids,
            is_approach,
            is_lowres_sr,
            has_overhang,
            wb_b,
            u_seq,
            tone_rands,
        )

    @staticmethod
    def _apply_fourier_amplitude_perturbation(
        crops_flat: torch.Tensor,
        prob: float,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """Phase-preserving 2D luminance Fourier amplitude MTF perturbation (preserves R==G==B on IR!)."""
        n_bt, _, h_c, w_c = crops_flat.shape
        dev = crops_flat.device
        k_sel = int(round(float(n_bt) * float(prob)))
        if k_sel <= 0:
            return crops_flat
        k_sel = min(k_sel, n_bt)

        # Sync-free static-shape random subset selection (zero cudaStreamSynchronize, cached cuFFT plan!)
        idx = torch.randperm(n_bt, generator=gen, device=dev)[:k_sel]
        sub = crops_flat.index_select(0, idx)
        w_rfft = w_c // 2 + 1
        gain_low = 0.72 + 0.56 * torch.rand((k_sel, 1, 5, 5), generator=gen, device=dev)
        gain_freq = F.interpolate(gain_low, size=(h_c, w_rfft), mode="bilinear", align_corners=False)
        gain_freq[:, :, 0, 0] = 1.0

        lum = sub.mean(dim=1, keepdim=True)
        spec = torch.fft.rfft2(lum, dim=(-2, -1))
        lum_pert = torch.fft.irfft2(spec * gain_freq, s=(h_c, w_c), dim=(-2, -1))
        pert = (sub + (lum_pert - lum)).clamp(0.0, 1.0)
        out = crops_flat.clone()
        out.index_copy_(0, idx, pert)
        return out


class GPUSynthIterableDataset(IterableDataset):
    """PyTorch `IterableDataset` streaming on-the-fly GPU sequence batches for `OCRTrainerV2`."""

    def __init__(
        self,
        engine: GPUSynthSequenceEngine | None = None,
        batch_size: int = 32,
        seq_len: int = 8,
        steps_per_epoch: int | None = None,
        **engine_kwargs: Any,
    ) -> None:
        super().__init__()
        self.engine = engine or GPUSynthSequenceEngine(**engine_kwargs)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else None

    def set_curriculum_stage(self, stage: int) -> None:
        self.engine.set_curriculum_stage(stage)

    def update_confusion_boost(
        self,
        confusion_or_weights: torch.Tensor | np.ndarray | Mapping[str, float] | None,
        boost_prob: float = 0.25,
    ) -> None:
        self.engine.update_confusion_boost(confusion_or_weights, boost_prob=boost_prob)

    def __len__(self) -> int:
        if self.steps_per_epoch is None:
            raise TypeError("Infinite GPUSynthIterableDataset has no finite len(); pass steps_per_epoch.")
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        step = 0
        while self.steps_per_epoch is None or step < self.steps_per_epoch:
            yield self.engine.sample_batch(batch_size=self.batch_size, seq_len=self.seq_len)
            step += 1
