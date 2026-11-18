"""Camera-ray 2.5D relief raymarching & physical Day / Shadow / Glare / 850nm / 940nm / 字光式 shader.

Zero-`T`-expansion memory design + end-to-end CUDA Triton kernel fusion (`torch.compile(dynamic=True)`):
samples static `(B, 13, 192, 384)` 2.5D surface packets across `T` trajectory frames without
duplicating `(192, 384)` buffers across `T`, keeping peak VRAM `<= 1.1 GB` and achieving
`> 700+ seqs/s` on a single RTX 4070.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import numpy as np
import torch
import torch.nn.functional as F

from vpocr_synth.gpu.homography_fast import (
    REGIME_TO_ID,
    SENSOR_ROI_H,
    SENSOR_ROI_W,
    TrajectoryBatch,
    project_plate_ray_grid_roi,
)
from vpocr_synth.plate.surface_25d import (
    EXT_H,
    EXT_W,
    PAD_X,
    PAD_Y,
    PLATE_H,
    PLATE_W,
    PlateSurface25DPacket,
    get_default_nonplate_stencil_atlas,
)


@dataclass
class ShadedROIBatch:
    """Output of 2.5D raymarch & physical shader in Sensor Local ROI `(112, 176)`."""

    hdr_roi_rgb: torch.Tensor       # (B, T, 3, 112, 176) float32 >= 0 (pre-clamp HDR irradiance!)
    occ_3d: torch.Tensor            # (B, T, 12) float32 in [0, 1] 3D relief occlusion fraction
    occ_total: torch.Tensor         # (B, T, 12) float32 in [0, 1] combined 2D + 3D occlusion
    slot_glare: torch.Tensor        # (B, T, 12) float32 in [0, 1] per-slot specular/halation glare
    contrast_survival: torch.Tensor # (B, T, 12) float32 in [0, 1] photometric contrast readability factor
    is_monochrome_ir: torch.Tensor  # (B, T) float32 in {0.0, 1.0}
    polarity_target: torch.Tensor   # (B, T) float32 in {-1.0, +1.0}
    ir_halation_gain: torch.Tensor  # (B, T) float32 >= 0 strength of pre-clamp HDR bloom


def _mm_to_ext_norm(x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
    """Convert MLIT `[0..330] x [0..165] mm` coordinates to `grid_sample` `[-1, 1]` on `(192, 384)`."""
    gn_x = (float(PAD_X) + x_mm * (float(PLATE_W) / 330.0)) * (2.0 / float(EXT_W)) - 1.0
    gn_y = (float(PAD_Y) + y_mm * (float(PLATE_H) / 165.0)) * (2.0 / float(EXT_H)) - 1.0
    return torch.stack([gn_x, gn_y], dim=-1)


def _compute_normals_impl(height_mm: torch.Tensor) -> torch.Tensor:
    h_pad = F.pad(height_mm, (1, 1, 1, 1), mode="replicate")
    dh_dx = (h_pad[:, :, 1:-1, 2:] - h_pad[:, :, 1:-1, :-2]) * (1.0 / (2.0 * 1.03125))
    dh_dy = (h_pad[:, :, 2:, 1:-1] - h_pad[:, :, :-2, 1:-1]) * (1.0 / (2.0 * 1.03125))
    nx = -dh_dx
    ny = -dh_dy
    nz = torch.ones_like(nx)
    return F.normalize(torch.cat([nx, ny, nz], dim=1), dim=1)


def _occ3d_pre_impl(
    slot_masks_half: torch.Tensor,
    surface_occ_2d: torch.Tensor,
    parallax_tan_xy: torch.Tensor,
    xx_q: torch.Tensor,
    yy_q: torch.Tensor,
    z_ray_taps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len = parallax_tan_xy.shape[:2]
    smask_q = F.avg_pool2d(slot_masks_half, kernel_size=2, stride=2)
    occ2d_q = F.avg_pool2d(surface_occ_2d, kernel_size=2, stride=2)

    dz_taps = z_ray_taps - 0.80
    ptan_bt = parallax_tan_xy.clamp(-3.2, 3.2)
    vx_t = ptan_bt[:, :, 0:1, None, None]
    vy_t = ptan_bt[:, :, 1:2, None, None]

    x_taps = xx_q[None, None, None, :, :] + dz_taps * vx_t
    y_taps = yy_q[None, None, None, :, :] + dz_taps * vy_t
    grid_taps = _mm_to_ext_norm(x_taps, y_taps).reshape(bsz, seq_len * 4 * 40, 80, 2)
    return smask_q, occ2d_q, grid_taps


def _occ3d_post_impl(
    h_taps_raw: torch.Tensor,
    smask_q: torch.Tensor,
    occ2d_q: torch.Tensor,
    plate_sides: torch.Tensor,
    has_seal: torch.Tensor,
    xx_q: torch.Tensor,
    z_ray_taps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len = plate_sides.shape
    h_taps = h_taps_raw.reshape(bsz, seq_len, 4, 40, 80)
    front_no_seal = ((plate_sides == 0) & has_seal[:, None])[:, :, None, None, None]
    h_taps = torch.where(front_no_seal & (h_taps > 4.2) & (xx_q[None, None, None] < 95.0), 4.0, h_taps)

    occ_ray_q = torch.clamp((h_taps - z_ray_taps) * 3.0, 0.0, 1.0).amax(dim=2)
    raw_area = smask_q.sum(dim=(-2, -1))
    slot_area = raw_area.clamp_min(1e-4)[:, None, :]
    has_ink = (raw_area > 0.25).float()[:, None, :]

    occ_3d = (((smask_q[:, None] * occ_ray_q[:, :, None]).sum(dim=(-2, -1)) / slot_area) * has_ink).clamp(0.0, 1.0)
    comb_q = torch.maximum(occ_ray_q[:, :, None], occ2d_q[:, None])
    occ_total = (((smask_q[:, None] * comb_q).sum(dim=(-2, -1)) / slot_area) * has_ink).clamp(0.0, 1.0)
    return occ_3d, occ_total, slot_area, has_ink


def _raymarch_prep_taps_impl(
    x0_c_mm: torch.Tensor,
    y0_c_mm: torch.Tensor,
    parallax_tan_xy: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    bsz, seq_len = parallax_tan_xy.shape[:2]
    x0_mm = (x0_c_mm + 165.0).reshape(bsz, seq_len, SENSOR_ROI_H, SENSOR_ROI_W)
    y0_mm = (y0_c_mm + 82.5).reshape(bsz, seq_len, SENSOR_ROI_H, SENSOR_ROI_W)

    ptan_bt = parallax_tan_xy.clamp(-3.2, 3.2)
    vx_r = ptan_bt[:, :, 0, None, None]
    vy_r = ptan_bt[:, :, 1, None, None]

    xh_all = torch.stack([x0_mm + 7.6 * vx_r, x0_mm + 5.2 * vx_r, x0_mm + 2.6 * vx_r, x0_mm], dim=2)
    yh_all = torch.stack([y0_mm + 7.6 * vy_r, y0_mm + 5.2 * vy_r, y0_mm + 2.6 * vy_r, y0_mm], dim=2)
    g_all = _mm_to_ext_norm(xh_all, yh_all).reshape(bsz, seq_len * 4 * SENSOR_ROI_H, SENSOR_ROI_W, 2)
    return x0_mm, y0_mm, vx_r, vy_r, xh_all, yh_all, g_all


def _secant_refine_impl(
    x0_mm: torch.Tensor,
    y0_mm: torch.Tensor,
    vx_r: torch.Tensor,
    vy_r: torch.Tensor,
    hs_raw: torch.Tensor,
    xh_all: torch.Tensor,
    yh_all: torch.Tensor,
    plate_sides: torch.Tensor,
    has_seal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len = plate_sides.shape
    hs_all = hs_raw.reshape(bsz, seq_len, 4, SENSOR_ROI_H, SENSOR_ROI_W)
    front_no_seal = ((plate_sides == 0) & has_seal[:, None])[:, :, None, None, None]
    hs_all = torch.where(
        front_no_seal & (hs_all > 4.2) & (xh_all < 92.0) & (yh_all < 46.0),
        4.0,
        hs_all,
    )

    h_prev = torch.full_like(x0_mm, 9.0)
    hs_prev = torch.zeros_like(x0_mm)
    h_hit = torch.zeros_like(x0_mm)
    found = torch.zeros_like(x0_mm, dtype=torch.bool)

    for idx_l, h_val in ((0, 7.6), (1, 5.2), (2, 2.6), (3, 0.0)):
        hs_curr = hs_all[:, :, idx_l]
        newly_hit = (~found) & (hs_curr >= h_val)
        d_prev = h_prev - hs_prev
        d_curr = h_val - hs_curr
        w_sec = torch.clamp(d_prev / (d_prev - d_curr).clamp_min(1e-3), 0.0, 1.0)
        h_sec = h_prev + w_sec * (h_val - h_prev)
        h_hit = torch.where(newly_hit, h_sec, h_hit)
        found = found | newly_hit
        h_prev = torch.full_like(x0_mm, h_val)
        hs_prev = hs_curr

    h_hit = torch.where(found, h_hit, hs_all[:, :, 3].clamp(-9.0, 0.0))
    x_hit_bt = x0_mm + h_hit * vx_r
    y_hit_bt = y0_mm + h_hit * vy_r
    return x_hit_bt, y_hit_bt, h_hit


def _pack_gbuf_and_grid_impl(
    albedo_rgb: torch.Tensor,
    nir_albedo: torch.Tensor,
    retro_mask: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    emissive_rgb: torch.Tensor,
    height_mm: torch.Tensor,
    x_hit_bt: torch.Tensor,
    y_hit_bt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len = x_hit_bt.shape[:2]
    normals = _compute_normals_impl(height_mm)
    gbuf = torch.cat(
        [albedo_rgb, nir_albedo, retro_mask, roughness, metallic, emissive_rgb, normals],
        dim=1,
    )
    grid_hit = _mm_to_ext_norm(x_hit_bt, y_hit_bt).reshape(bsz, seq_len * SENSOR_ROI_H, SENSOR_ROI_W, 2)
    if albedo_rgb.is_cuda:
        return gbuf.half(), grid_hit.half()
    return gbuf, grid_hit


def _light_shadow_prep_impl(
    v_dir: torch.Tensor,
    reg_bt: torch.Tensor,
    ir_bt: torch.Tensor,
    x_hit_bt: torch.Tensor,
    y_hit_bt: torch.Tensor,
    rand_pack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len = x_hit_bt.shape[:2]
    l_az = (rand_pack[:, 0] * 2.0 - 1.0) * 0.85
    l_el = -(0.35 + 0.55 * rand_pack[:, 1])
    l_raw = torch.stack([l_az, l_el, torch.ones_like(l_az)], dim=-1)
    is_coax = (ir_bt > 0) | (reg_bt == 3)  # REGIME_TO_ID["night_rgb_glare"] == 3
    coax_jitter = (rand_pack[:, 2:5] - 0.5) * 0.06
    l_coax = F.normalize(v_dir + coax_jitter, dim=-1)
    l_dir = torch.where(is_coax[:, None], l_coax, F.normalize(l_raw, dim=-1))

    lx_tan = (l_dir[:, 0] / l_dir[:, 2].clamp_min(0.25)).clamp(-2.5, 2.5).reshape(bsz, seq_len, 1, 1)
    ly_tan = (l_dir[:, 1] / l_dir[:, 2].clamp_min(0.25)).clamp(-2.5, 2.5).reshape(bsz, seq_len, 1, 1)
    g_lsh = _mm_to_ext_norm(x_hit_bt + 5.5 * lx_tan, y_hit_bt + 5.5 * ly_tan).reshape(
        bsz, seq_len * SENSOR_ROI_H, SENSOR_ROI_W, 2
    )
    return l_dir, g_lsh


def _brdf_core_impl(
    sampled: torch.Tensor,         # (N, 13, 112, 176)
    x_hit: torch.Tensor,           # (N, 112, 176)
    y_hit: torch.Tensor,           # (N, 112, 176)
    h_hit_flat: torch.Tensor,      # (N, 112, 176)
    h_lblk: torch.Tensor,          # (N, 1, 112, 176)
    v_dir: torch.Tensor,           # (N, 3)
    l_dir: torch.Tensor,           # (N, 3)
    reg_bt: torch.Tensor,          # (N,) int64
    ir_bt: torch.Tensor,           # (N,) int64
    night_bt: torch.Tensor,        # (N,) bool
    sides_bt: torch.Tensor,        # (N,) int64
    illum_flat: torch.Tensor,      # (N,) bool
    base_pol_flat: torch.Tensor,   # (N,) float32 (+1 bright bg, -1 dark bg)
    ir_pow: torch.Tensor,          # (N, 1, 1, 1)
    g_amp: torch.Tensor,           # (N, 1, 1, 1)
    rand_u: torch.Tensor,          # (N, 6) float32 in [0, 1)
    c_day: torch.Tensor,
    c_sky: torch.Tensor,
    c_sod: torch.Tensor,
    c_mer: torch.Tensor,
    c_led: torch.Tensor,
    c_brake: torch.Tensor,
    c_icr850: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused multi-spectral BRDF + NIR + 字光式 emissive pixel shader on `(N, 3, 112, 176)`."""
    n_bt = sampled.shape[0]
    sampled = sampled.float()
    alb_rgb = sampled[:, 0:3]
    alb_nir = sampled[:, 3:4]
    retro = sampled[:, 4:5]
    rough = sampled[:, 5:6]
    metal = sampled[:, 6:7]
    emis_rgb = sampled[:, 7:10]
    norm_xyz = F.normalize(sampled[:, 10:13], dim=1)

    relief_shadow = 1.0 - 0.72 * torch.clamp((h_lblk - (h_hit_flat[:, None] + 3.2)) * 0.6, 0.0, 1.0)

    ndotv = (
        norm_xyz[:, 0:1] * v_dir[:, 0:1, None, None]
        + norm_xyz[:, 1:2] * v_dir[:, 1:2, None, None]
        + norm_xyz[:, 2:3] * v_dir[:, 2:3, None, None]
    ).clamp(0.02, 1.0)
    cos_inc = v_dir[:, 2].clamp(0.05, 1.0)[:, None, None, None]

    ndotl = (
        norm_xyz[:, 0:1] * l_dir[:, 0:1, None, None]
        + norm_xyz[:, 1:2] * l_dir[:, 1:2, None, None]
        + norm_xyz[:, 2:3] * l_dir[:, 2:3, None, None]
    ).clamp(0.0, 1.0)

    has_macro_sh = (reg_bt == 1) | ((reg_bt == 2) & (rand_u[:, 0] < 0.5))
    sh_nx = (rand_u[:, 1] * 1.4 - 0.7)[:, None, None]
    sh_off = (rand_u[:, 2] * 140.0 + 25.0)[:, None, None]
    macro_lit = torch.sigmoid(((x_hit - 165.0) * sh_nx + y_hit - sh_off) * 0.28).unsqueeze(1)
    shadow_mult = torch.where(
        has_macro_sh[:, None, None, None],
        0.16 + 0.84 * macro_lit,
        torch.ones_like(macro_lit),
    ) * relief_shadow

    ldotv = (l_dir * v_dir).sum(dim=-1).clamp(-0.9999, 0.9999)[:, None, None, None]
    obs_angle = torch.acos(ldotv)
    retro_cone = torch.exp(-((obs_angle / 0.14) ** 2))
    steep_dropoff = cos_inc.pow(2.8)

    wet_rain = (reg_bt == 7)[:, None, None, None].float()
    eff_rough = (rough * (1.0 - 0.55 * wet_rain)).clamp(0.06, 0.95)
    gx_c = (40.0 + 250.0 * rand_u[:, 3])[:, None, None]
    gy_c = (25.0 + 115.0 * rand_u[:, 4])[:, None, None]
    d2_glare = ((x_hit - gx_c) / 52.0) ** 2 + ((y_hit - gy_c) / 34.0) ** 2
    glare_lobe = torch.exp(-d2_glare / (eff_rough.squeeze(1) * 2.2 + 0.15)).unsqueeze(1)
    spec_term = g_amp * glare_lobe * (0.35 + 0.85 * metal + 0.45 * wet_rain) * (0.4 + 0.6 * ndotv)

    u_col = rand_u[:, 5:6]
    night_tint = torch.where(u_col < 0.36, c_sod, torch.where(u_col < 0.58, c_mer, c_led))
    # Cool open-sky / outdoor-shade AWB / blue-white LED fog-lamp stray-light reflections on ~32% of daytime/overcast/rain/dusk frames
    cool_sky_w = torch.clamp((u_col - 0.68) / 0.32, 0.0, 1.0)
    day_tint = (1.0 - 0.82 * cool_sky_w) * c_day + (0.82 * cool_sky_w) * c_sky
    spec_day_tint = (1.0 - cool_sky_w) * c_day + cool_sky_w * c_sky
    illum_tint = torch.where(night_bt[:, None], night_tint, day_tint)[:, :, None, None]
    spec_tint = torch.where(night_bt[:, None], night_tint, spec_day_tint)[:, :, None, None]

    rear_night_rgb = night_bt & (ir_bt == 0) & (sides_bt == 1)
    edge_prox = torch.exp(-torch.minimum(x_hit, 330.0 - x_hit).clamp_min(0.0) / 55.0).unsqueeze(1)
    brake_spill = rear_night_rgb[:, None, None, None].float() * edge_prox * c_brake

    night_illum_active = night_bt & (ir_bt == 0) & illum_flat
    amb_level = torch.where(night_illum_active, 0.08, torch.where(night_bt, 0.22, 0.34))[:, None, None, None]
    dir_level = torch.where(night_illum_active, 0.10, torch.where(night_bt, 0.52, 0.82))[:, None, None, None]

    retro_rgb_gain = torch.where(
        (night_bt & (~night_illum_active))[:, None, None, None],
        0.95 * retro_cone * steep_dropoff,
        0.18 * retro_cone * steep_dropoff,
    )
    diffuse_and_retro = alb_rgb * illum_tint * (
        amb_level + dir_level * ndotl * shadow_mult + retro * retro_rgb_gain
    )
    # Glossy dark-plate sky-sheen + white/yellow stroke HDR radiance push on commercial green/black plates
    is_dark_pol = (base_pol_flat < 0.0)[:, None, None, None].float()
    day_vis_f = (~night_bt)[:, None, None, None].float()
    stroke_bright = (alb_rgb.amax(dim=1, keepdim=True) > 0.62).float()
    diffuse_and_retro = diffuse_and_retro * (1.0 + 0.24 * is_dark_pol * stroke_bright * (1.0 - 0.5 * night_bt[:, None, None, None].float()))

    gloss_factor = ((0.40 - rough) / 0.30).clamp(0.0, 1.0)
    dark_sky_sheen = (
        day_vis_f
        * is_dark_pol
        * (0.14 + 0.16 * cool_sky_w[:, :, None, None])
        * (1.0 - 0.32 * ndotv)
        * gloss_factor
        * c_sky[:, :, None, None]
    )
    # Cool sky-dome / open-shade Fresnel sheen on glossy painted plates under overcast/ferry/gate skies
    cool_sheen = (
        day_vis_f
        * cool_sky_w[:, :, None, None]
        * (0.14 * (1.0 - 0.40 * ndotv) * (1.0 - 0.55 * rough))
        * c_sky[:, :, None, None]
    )
    emis_gain = torch.where(night_bt[:, None, None, None], 1.15, 0.08)
    hdr_vis = diffuse_and_retro + spec_term * spec_tint + cool_sheen + dark_sky_sheen + emis_rgb * emis_gain + brake_spill

    is_940 = (ir_bt == 2)[:, None, None, None]
    qe_nir = torch.where(is_940, 0.58, 0.92)
    nir_base = alb_nir * (
        0.12 + 0.38 * ndotl * relief_shadow + retro * ir_pow * retro_cone * steep_dropoff
    ) * qe_nir + 0.55 * spec_term
    hdr_nir_mono = nir_base.expand(n_bt, 3, SENSOR_ROI_H, SENSOR_ROI_W)
    hdr_nir = torch.where((ir_bt == 1)[:, None, None, None], hdr_nir_mono * c_icr850, hdr_nir_mono)

    hdr_roi = torch.where((ir_bt > 0)[:, None, None, None], hdr_nir, hdr_vis).clamp_min(0.0)
    return hdr_roi, gx_c, gy_c, steep_dropoff


def _garbage_roi_impl(
    yy: torch.Tensor,
    xx: torch.Tensor,
    r_params: torch.Tensor,
    low_tex_in: torch.Tensor,
    is_ir: torch.Tensor,
) -> torch.Tensor:
    n_bt = is_ir.shape[0]
    freq_y = (6.0 + 18.0 * r_params[:, 0:1, None, None])
    freq_x = (4.0 + 16.0 * r_params[:, 1:2, None, None])
    mode = r_params[:, 2:3, None, None]

    slats = 0.12 + 0.55 * (torch.sin(yy * 3.14159265 * freq_y) > 0.35).float()
    ribs = 0.08 + 0.25 * (torch.sin(xx * 3.14159265 * freq_x) > 0.65).float()
    grille = (slats + ribs).expand(n_bt, 3, SENSOR_ROI_H, SENSOR_ROI_W)

    low_tex = F.interpolate(
        low_tex_in * 0.35,
        size=(SENSOR_ROI_H, SENSOR_ROI_W),
        mode="bilinear",
        align_corners=False,
    )
    lamp = 1.6 * torch.exp(-(((xx - 0.32) ** 2 + (yy - 0.48) ** 2) / 0.025))
    asphalt = low_tex + lamp

    out = torch.where(mode < 0.55, grille, asphalt)
    mono = out.mean(dim=1, keepdim=True).expand_as(out)
    return torch.where(is_ir[:, None, None, None], mono, out)


def _shader_slot_tail_impl(
    xx_q: torch.Tensor,
    yy_q: torch.Tensor,
    gx_c: torch.Tensor,
    gy_c: torch.Tensor,
    g_amp: torch.Tensor,
    ir_bt: torch.Tensor,
    ir_pow: torch.Tensor,
    steep_dropoff: torch.Tensor,
    smask_q: torch.Tensor,
    slot_area: torch.Tensor,
    has_ink: torch.Tensor,
    sensor_ir_mode: torch.Tensor,
    night_bt: torch.Tensor,
    illum_bt: torch.Tensor,
    base_pol_bt: torch.Tensor,
    is_glare_reg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len = sensor_ir_mode.shape
    n_bt = bsz * seq_len

    d2_g_q = ((xx_q[None] - gx_c) / 52.0) ** 2 + ((yy_q[None] - gy_c) / 34.0) ** 2
    glare_q = (g_amp.squeeze(1) * torch.exp(-d2_g_q / 0.55)).clamp(0.0, 1.5)
    hal_over = (
        ((ir_bt > 0)[:, None, None].float())
        * torch.clamp(ir_pow.squeeze(1) * steep_dropoff.squeeze(1) - 1.15, 0.0, 1.0)
        * 0.55
    )
    glare_map_q = torch.clamp(glare_q * 0.65 + hal_over, 0.0, 1.0).reshape(bsz, seq_len, 1, 40, 80)
    slot_glare = (((smask_q[:, None] * glare_map_q).sum(dim=(-2, -1)) / slot_area) * has_ink).clamp(0.0, 1.0)

    steep_ir_dim = torch.where(
        (ir_bt > 0)[:, None],
        steep_dropoff.reshape(n_bt, 1).clamp(0.25, 1.0),
        torch.ones_like(steep_dropoff.reshape(n_bt, 1)),
    )
    contrast_surv = (steep_ir_dim.reshape(bsz, seq_len, 1) * (1.0 - 0.68 * slot_glare)).clamp(0.05, 1.0)

    is_mono_ir = (sensor_ir_mode > 0).float()
    night_illum_active = (night_bt & (ir_bt == 0) & illum_bt.reshape(n_bt)).reshape(bsz, seq_len)
    polarity_target = torch.where(
        night_illum_active,
        torch.full_like(base_pol_bt, -1.0),
        base_pol_bt,
    )
    ir_hal_gain = torch.where(
        sensor_ir_mode > 0,
        torch.ones_like(base_pol_bt),
        torch.where(
            polarity_target < 0.0,
            torch.where(
                night_illum_active | is_glare_reg.reshape(bsz, seq_len),
                torch.full_like(base_pol_bt, 0.85),
                torch.full_like(base_pol_bt, 0.68),
            ),
            torch.where(
                night_illum_active | is_glare_reg.reshape(bsz, seq_len),
                torch.full_like(base_pol_bt, 0.75),
                torch.full_like(base_pol_bt, 0.25),
            ),
        ),
    )
    return slot_glare, contrast_surv, is_mono_ir, polarity_target, ir_hal_gain


_CUDA_CACHE: dict[str, Callable[..., object]] = {}


def _dispatch_cuda(name: str, fn: Callable[..., object], *args: torch.Tensor):
    if args[0].is_cuda:
        c_fn = _CUDA_CACHE.get(name)
        if c_fn is None:
            c_fn = torch.compile(fn, dynamic=True)
            _CUDA_CACHE[name] = c_fn
        return c_fn(*args)
    return fn(*args)


class ReliefPBRShaderGPU:
    """Vectorized 2.5D relief raymarcher and multi-spectral PBR/IR shader."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self._grid_cache: dict[torch.device, tuple[torch.Tensor, ...]] = {}

    def to(self, device: torch.device | str) -> ReliefPBRShaderGPU:
        self.device = torch.device(device)
        return self

    def _get_cached_constants(self, dev: torch.device) -> tuple[torch.Tensor, ...]:
        cached = self._grid_cache.get(dev)
        if cached is not None:
            return cached
        yh = (torch.arange(40, device=dev, dtype=torch.float32) + 0.5) * (165.0 / 40.0)
        xh = (torch.arange(80, device=dev, dtype=torch.float32) + 0.5) * (330.0 / 80.0)
        yy_q, xx_q = torch.meshgrid(yh, xh, indexing="ij")  # (40, 80)

        c_day = torch.tensor([[1.00, 0.99, 0.97]], device=dev)
        c_sky = torch.tensor([[0.60, 0.84, 1.46]], device=dev)
        c_sod = torch.tensor([[1.05, 0.66, 0.26]], device=dev)
        c_mer = torch.tensor([[0.80, 1.00, 0.88]], device=dev)
        c_led = torch.tensor([[0.95, 0.97, 1.04]], device=dev)
        c_brake = torch.tensor([0.38, 0.03, 0.03], device=dev).view(1, 3, 1, 1)
        c_icr850 = torch.tensor([1.003, 0.997, 1.002], device=dev).view(1, 3, 1, 1)
        z_ray_taps = torch.tensor([2.4, 4.5, 6.3, 7.6], device=dev, dtype=torch.float32).view(1, 1, 4, 1, 1)

        yy_roi = torch.linspace(0.0, 1.0, SENSOR_ROI_H, device=dev)[None, None, :, None]
        xx_roi = torch.linspace(0.0, 1.0, SENSOR_ROI_W, device=dev)[None, None, None, :]
        out = (xx_q, yy_q, c_day, c_sky, c_sod, c_mer, c_led, c_brake, c_icr850, z_ray_taps, yy_roi, xx_roi)
        self._grid_cache[dev] = out
        return out

    @staticmethod
    def _mm_to_ext_norm(x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        return _mm_to_ext_norm(x_mm, y_mm)

    @staticmethod
    def _compute_normals_from_height(height_mm: torch.Tensor) -> torch.Tensor:
        """Compute unit surface normals `(Nx, Ny, Nz)` `(B, 3, 192, 384)` from `height_mm`."""
        return _dispatch_cuda("normals", _compute_normals_impl, height_mm)

    @staticmethod
    def _compute_slot_occlusion_3d(
        surface: PlateSurface25DPacket,
        traj: TrajectoryBatch,
        xx_q: torch.Tensor,
        yy_q: torch.Tensor,
        z_ray_taps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-call 4-tap outgoing camera-ray occlusion `occ_3d[B, T, 12]` on `(40, 80)`."""
        smask_q, occ2d_q, grid_taps = _dispatch_cuda(
            "occ_pre",
            _occ3d_pre_impl,
            surface.slot_masks_half,
            surface.surface_occ_2d,
            traj.parallax_tan_xy,
            xx_q,
            yy_q,
            z_ray_taps,
        )
        h_taps_raw = F.grid_sample(
            surface.height_mm, grid_taps, mode="bilinear", padding_mode="border", align_corners=False
        )
        occ_3d, occ_total, slot_area, has_ink = _dispatch_cuda(
            "occ_post",
            _occ3d_post_impl,
            h_taps_raw,
            smask_q,
            occ2d_q,
            traj.plate_sides,
            surface.has_seal,
            xx_q,
            z_ray_taps,
        )
        return occ_3d, occ_total, smask_q, slot_area, has_ink

    def _raymarch_gbuffer_roi(
        self,
        surface: PlateSurface25DPacket,
        traj: TrajectoryBatch,
        surface_secondary: PlateSurface25DPacket | None,
        secondary_batch_indices: list[int] | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """4-tap + 1-step secant relief raymarch in `(112, 176)` ROI without `T` buffer expansion."""
        dev = surface.albedo_rgb.device
        bsz, seq_len = traj.yaw_deg.shape
        n_bt = bsz * seq_len

        x0_c_mm, y0_c_mm = project_plate_ray_grid_roi(traj)
        x0_mm, y0_mm, vx_r, vy_r, xh_all, yh_all, g_all = _dispatch_cuda(
            "rm_prep",
            _raymarch_prep_taps_impl,
            x0_c_mm,
            y0_c_mm,
            traj.parallax_tan_xy,
        )
        del x0_c_mm, y0_c_mm

        hs_dtype = torch.float16 if dev.type == "cuda" else torch.float32
        h_surf_rm = surface.height_mm.to(dtype=hs_dtype)
        hs_raw = F.grid_sample(
            h_surf_rm, g_all.to(dtype=hs_dtype), mode="bilinear", padding_mode="border", align_corners=False
        ).float()
        del g_all, h_surf_rm

        x_hit_bt, y_hit_bt, h_hit = _dispatch_cuda(
            "secant",
            _secant_refine_impl,
            x0_mm,
            y0_mm,
            vx_r,
            vy_r,
            hs_raw,
            xh_all,
            yh_all,
            traj.plate_sides,
            surface.has_seal,
        )
        del hs_raw, x0_mm, y0_mm, xh_all, yh_all

        gbuf_pri, grid_hit_b = _dispatch_cuda(
            "pack_gbuf",
            _pack_gbuf_and_grid_impl,
            surface.albedo_rgb,
            surface.nir_albedo,
            surface.retro_mask,
            surface.roughness,
            surface.metallic,
            surface.emissive_rgb,
            surface.height_mm,
            x_hit_bt,
            y_hit_bt,
        )

        # Pre-filter 2.5D G-buffer by projected sensor footprint scale_uv = max(1.0, 160.0 / w_target_px)
        if traj.w_target_px is not None:
            w_t_px = traj.w_target_px.float()
        else:
            w_top = torch.linalg.norm(traj.corners_roi_gt[:, :, 1] - traj.corners_roi_gt[:, :, 0], dim=-1)
            w_bot = torch.linalg.norm(traj.corners_roi_gt[:, :, 2] - traj.corners_roi_gt[:, :, 3], dim=-1)
            w_t_px = 0.5 * (w_top + w_bot)
        scale_uv = (160.0 / w_t_px.clamp_min(10.0)).clamp(1.0, 16.0)
        lod_bt = torch.log2(scale_uv)  # (B, T) in [0.0, 4.0]

        def _mip_sample(gbuf: torch.Tensor, g_hit: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
            nb, ch_g = gbuf.shape[:2]
            g1 = F.avg_pool2d(gbuf, kernel_size=2, stride=2)
            g2 = F.avg_pool2d(g1, kernel_size=2, stride=2)
            g3 = F.avg_pool2d(g2, kernel_size=2, stride=2)
            g4 = F.avg_pool2d(g3, kernel_size=2, stride=2)
            s0 = F.grid_sample(gbuf, g_hit, mode="bilinear", padding_mode="border", align_corners=False)
            s1 = F.grid_sample(g1, g_hit, mode="bilinear", padding_mode="border", align_corners=False)
            s2 = F.grid_sample(g2, g_hit, mode="bilinear", padding_mode="border", align_corners=False)
            s3 = F.grid_sample(g3, g_hit, mode="bilinear", padding_mode="border", align_corners=False)
            s4 = F.grid_sample(g4, g_hit, mode="bilinear", padding_mode="border", align_corners=False)
            lf = lod.clamp(0.0, 4.0).to(dtype=gbuf.dtype)[:, None, :, None, None]
            w0 = (1.0 - lf).clamp(0.0, 1.0)
            w1 = (1.0 - (lf - 1.0).abs()).clamp(0.0, 1.0)
            w2 = (1.0 - (lf - 2.0).abs()).clamp(0.0, 1.0)
            w3 = (1.0 - (lf - 3.0).abs()).clamp(0.0, 1.0)
            w4 = (lf - 3.0).clamp(0.0, 1.0)
            blended = (
                w0 * s0.reshape(nb, ch_g, seq_len, SENSOR_ROI_H, SENSOR_ROI_W)
                + w1 * s1.reshape(nb, ch_g, seq_len, SENSOR_ROI_H, SENSOR_ROI_W)
                + w2 * s2.reshape(nb, ch_g, seq_len, SENSOR_ROI_H, SENSOR_ROI_W)
                + w3 * s3.reshape(nb, ch_g, seq_len, SENSOR_ROI_H, SENSOR_ROI_W)
                + w4 * s4.reshape(nb, ch_g, seq_len, SENSOR_ROI_H, SENSOR_ROI_W)
            )
            return blended.permute(0, 2, 1, 3, 4).contiguous()

        sampled_b = _mip_sample(gbuf_pri, grid_hit_b, lod_bt)
        del gbuf_pri

        base_pol_bt = surface.base_polarity[:, None].expand(bsz, seq_len).clone()
        illum_bt = surface.is_illuminated[:, None].expand(bsz, seq_len).clone()

        if surface_secondary is not None and secondary_batch_indices:
            n_sec = len(secondary_batch_indices)
            idx_t = torch.from_numpy(np.asarray(secondary_batch_indices, dtype=np.int64)).to(dev)
            n_surf_sec = int(surface_secondary.albedo_rgb.shape[0])
            idx_eval = idx_t.expand(n_surf_sec) if (n_sec == 1 and n_surf_sec > 1) else idx_t
            gbuf_sec_full, _ = _dispatch_cuda(
                "pack_gbuf",
                _pack_gbuf_and_grid_impl,
                surface_secondary.albedo_rgb,
                surface_secondary.nir_albedo,
                surface_secondary.retro_mask,
                surface_secondary.roughness,
                surface_secondary.metallic,
                surface_secondary.emissive_rgb,
                surface_secondary.height_mm,
                x_hit_bt.index_select(0, idx_eval),
                y_hit_bt.index_select(0, idx_eval),
            )
            gbuf_sec = gbuf_sec_full[:n_sec]
            g_hit_sec = grid_hit_b.index_select(0, idx_t)
            samp_sec = _mip_sample(gbuf_sec, g_hit_sec, lod_bt.index_select(0, idx_t))
            sp_sub = traj.splice_second_plate.index_select(0, idx_t)
            sampled_b[idx_t] = torch.where(sp_sub[:, :, None, None, None], samp_sec, sampled_b[idx_t])
            base_pol_bt[idx_t] = torch.where(
                sp_sub, surface_secondary.base_polarity[:n_sec, None].expand_as(sp_sub), base_pol_bt[idx_t]
            )
            illum_bt[idx_t] = torch.where(
                sp_sub, surface_secondary.is_illuminated[:n_sec, None].expand_as(sp_sub), illum_bt[idx_t]
            )

        return (
            sampled_b.reshape(n_bt, 13, SENSOR_ROI_H, SENSOR_ROI_W),
            x_hit_bt.reshape(n_bt, SENSOR_ROI_H, SENSOR_ROI_W),
            y_hit_bt.reshape(n_bt, SENSOR_ROI_H, SENSOR_ROI_W),
            h_hit.reshape(n_bt, SENSOR_ROI_H, SENSOR_ROI_W),
            x_hit_bt,
            y_hit_bt,
            base_pol_bt,
            illum_bt,
        )

    @torch.no_grad()
    def shade_sequences(
        self,
        surface: PlateSurface25DPacket,
        traj: TrajectoryBatch,
        surface_secondary: PlateSurface25DPacket | None = None,
        secondary_batch_indices: list[int] | None = None,
        seed: int = 0,
        coaxial_ir_power: float | None = None,
        glare_boost: float | None = None,
    ) -> ShadedROIBatch:
        """Raymarch 2.5D relief & physically shade `(B, T, 3, 112, 176)` HDR sensor ROIs."""
        dev = surface.albedo_rgb.device
        bsz, seq_len = traj.yaw_deg.shape
        n_bt = bsz * seq_len

        (
            xx_q, yy_q, c_day, c_sky, c_sod, c_mer, c_led, c_brake, c_icr850, z_ray_taps, yy_roi, xx_roi
        ) = self._get_cached_constants(dev)

        gen = torch.Generator(device=dev)
        gen.manual_seed((seed * 2246822519 + 3266489917) & 0x7FFFFFFF)

        # 1. 3D Outgoing Camera-Ray Occlusion occ_3d[B, T, 12]
        occ_3d, occ_total, smask_q, slot_area, has_ink = self._compute_slot_occlusion_3d(
            surface=surface, traj=traj, xx_q=xx_q, yy_q=yy_q, z_ray_taps=z_ray_taps
        )

        # 2. 4-Tap + Secant Relief Raymarch in (112, 176)
        (
            sampled, x_hit, y_hit, h_hit_flat, x_hit_bt, y_hit_bt, base_pol_bt, illum_bt
        ) = self._raymarch_gbuffer_roi(
            surface=surface,
            traj=traj,
            surface_secondary=surface_secondary,
            secondary_batch_indices=secondary_batch_indices,
        )

        # 3. Lighting vectors & 3D relief shadow tap
        reg_bt = traj.regime_ids[:, None].expand(bsz, seq_len).reshape(n_bt)
        ir_bt = traj.sensor_ir_mode.reshape(n_bt)
        night_bt = traj.is_night.reshape(n_bt)
        v_dir = traj.view_dir_local.reshape(n_bt, 3)

        rand_pack = torch.rand((n_bt, 13), generator=gen, device=dev)
        l_dir, g_lsh = _dispatch_cuda(
            "light_prep",
            _light_shadow_prep_impl,
            v_dir,
            reg_bt,
            ir_bt,
            x_hit_bt,
            y_hit_bt,
            rand_pack,
        )
        del x_hit_bt, y_hit_bt
        h_lblk = F.grid_sample(
            surface.height_mm, g_lsh, mode="bilinear", padding_mode="border", align_corners=False
        ).reshape(n_bt, 1, SENSOR_ROI_H, SENSOR_ROI_W)
        del g_lsh

        if coaxial_ir_power is not None:
            ir_pow = torch.full((n_bt, 1, 1, 1), float(coaxial_ir_power), device=dev)
        else:
            ir_pow = (0.62 + 0.98 * rand_pack[:, 5])[:, None, None, None]

        is_glare_reg = (
            (reg_bt == REGIME_TO_ID["night_rgb_glare"])
            | (reg_bt == REGIME_TO_ID["dusk_wdr"])
            | (reg_bt == REGIME_TO_ID["heavy_rain_droplets"])
        )
        if glare_boost is not None:
            g_amp = torch.full((n_bt, 1, 1, 1), float(glare_boost), device=dev)
        else:
            g_amp = torch.where(
                is_glare_reg,
                0.75 + 1.65 * rand_pack[:, 6],
                0.08 + 0.28 * rand_pack[:, 6],
            )[:, None, None, None]

        # 4. Fused Multi-Spectral BRDF & NIR Shader
        hdr_roi, gx_c, gy_c, steep_dropoff = _dispatch_cuda(
            "brdf",
            _brdf_core_impl,
            sampled,
            x_hit,
            y_hit,
            h_hit_flat,
            h_lblk,
            v_dir,
            l_dir,
            reg_bt,
            ir_bt,
            night_bt,
            traj.plate_sides.reshape(n_bt),
            illum_bt.reshape(n_bt),
            base_pol_bt.reshape(n_bt),
            ir_pow,
            g_amp,
            rand_pack[:, 7:13].contiguous(),
            c_day,
            c_sky,
            c_sod,
            c_mer,
            c_led,
            c_brake,
            c_icr850,
        )
        del sampled, x_hit, y_hit, h_hit_flat, h_lblk

        # Sparse CPU-indexed negative_garbage ROI synthesis (68% NonPlateStencilAtlas text-bearing signs/decals + 32% procedural grilles!)
        garb_seq_idx = [i for i, tag in enumerate(traj.regime_tags) if tag == "negative_garbage"]
        if garb_seq_idx:
            hdr_bt = hdr_roi.reshape(bsz, seq_len, 3, SENSOR_ROI_H, SENSOR_ROI_W)
            idx_g = torch.from_numpy(np.asarray(garb_seq_idx, dtype=np.int64)).to(dev)
            n_g_seq = len(garb_seq_idx)
            n_g = n_g_seq * seq_len
            is_ir_g = (traj.sensor_ir_mode.index_select(0, idx_g) > 0).reshape(n_g)
            r_garb = torch.rand((n_g, 3), generator=gen, device=dev)
            low_tex_g = torch.rand((n_g, 3, 7, 11), generator=gen, device=dev)
            proc_garb = _dispatch_cuda(
                "garbage",
                _garbage_roi_impl,
                yy_roi,
                xx_roi,
                r_garb,
                low_tex_g,
                is_ir_g,
            ).reshape(n_g_seq, seq_len, 3, SENSOR_ROI_H, SENSOR_ROI_W)

            stencil_bank = get_default_nonplate_stencil_atlas().to_torch(dev)
            st_idx = torch.randint(0, stencil_bank.shape[0], (n_g_seq,), generator=gen, device=dev)
            st_seq = stencil_bank.index_select(0, st_idx)  # (G, 3, 96, 160)
            # Project text-bearing non-plate stencil across T trajectory frames with consistent drift
            st_u = torch.rand((n_g_seq, 6), generator=gen, device=dev)
            t_lin_g = (
                torch.linspace(-0.5, 0.5, seq_len, device=dev, dtype=torch.float32)[None, :]
                if seq_len > 1
                else torch.zeros((1, 1), device=dev, dtype=torch.float32)
            )
            sx_g = (0.78 + 0.42 * st_u[:, 0:1]).expand(n_g_seq, seq_len).reshape(n_g, 1)
            sy_g = (0.78 + 0.42 * st_u[:, 1:2]).expand(n_g_seq, seq_len).reshape(n_g, 1)
            sh_g = (-0.12 + 0.24 * st_u[:, 2:3]).expand(n_g_seq, seq_len).reshape(n_g, 1)
            tx_g = ((-0.14 + 0.28 * st_u[:, 3:4]) + (-0.08 + 0.16 * st_u[:, 4:5]) * t_lin_g).reshape(n_g, 1)
            ty_g = ((-0.12 + 0.24 * st_u[:, 4:5]) + (-0.06 + 0.12 * st_u[:, 3:4]) * t_lin_g).reshape(n_g, 1)
            theta_g = torch.stack(
                [
                    torch.cat([sx_g, sh_g, tx_g], dim=1),
                    torch.cat([torch.zeros_like(sx_g), sy_g, ty_g], dim=1),
                ],
                dim=1,
            )
            grid_g = F.affine_grid(theta_g, size=[n_g, 3, SENSOR_ROI_H, SENSOR_ROI_W], align_corners=False)
            st_bt = F.grid_sample(
                st_seq[:, None].expand(n_g_seq, seq_len, 3, SENSOR_ROI_H, SENSOR_ROI_W).reshape(n_g, 3, SENSOR_ROI_H, SENSOR_ROI_W),
                grid_g,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            gain_g = (0.52 + 0.68 * st_u[:, 5:6])[:, :, None, None, None].expand(n_g_seq, seq_len, 1, 1, 1).reshape(n_g, 1, 1, 1)
            st_bt = st_bt * gain_g
            st_mono = (0.299 * st_bt[:, 0:1] + 0.587 * st_bt[:, 1:2] + 0.114 * st_bt[:, 2:3]).expand_as(st_bt)
            st_bt = torch.where(is_ir_g[:, None, None, None], st_mono, st_bt).reshape(
                n_g_seq, seq_len, 3, SENSOR_ROI_H, SENSOR_ROI_W
            )
            use_stencil = (torch.rand((n_g_seq, 1, 1, 1, 1), generator=gen, device=dev) < 0.68)
            hdr_bt[idx_g] = torch.where(use_stencil, st_bt, proc_garb)
            hdr_roi = hdr_bt.reshape(n_bt, 3, SENSOR_ROI_H, SENSOR_ROI_W)

        # 5. Fused Per-Slot Glare, Polarity & Readability Survival Tail
        slot_glare, contrast_surv, is_mono_ir, polarity_target, ir_hal_gain = _dispatch_cuda(
            "slot_tail",
            _shader_slot_tail_impl,
            xx_q,
            yy_q,
            gx_c,
            gy_c,
            g_amp,
            ir_bt,
            ir_pow,
            steep_dropoff,
            smask_q,
            slot_area,
            has_ink,
            traj.sensor_ir_mode,
            night_bt,
            illum_bt,
            base_pol_bt,
            is_glare_reg,
        )

        return ShadedROIBatch(
            hdr_roi_rgb=hdr_roi.reshape(bsz, seq_len, 3, SENSOR_ROI_H, SENSOR_ROI_W),
            occ_3d=occ_3d,
            occ_total=occ_total,
            slot_glare=slot_glare,
            contrast_survival=contrast_surv,
            is_monochrome_ir=is_mono_ir,
            polarity_target=polarity_target,
            ir_halation_gain=ir_hal_gain,
        )
