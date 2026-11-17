"""Physical Camera Sensor, Rain/Droplet Optics, ISP & Exact 8x8 DCT Codec in `(112, 176)` ROI.

All degradations execute IN NATIVE CAMERA SENSOR ROI `(112, 176)` BEFORE
`PlateQuadNet` 4-corner perspective rectification so steep yaw/pitch produces
authentic anisotropic horizontal blur, rotated 8x8 DCT block artifacts, and
2x2 `YUV420` (`NV12`) chroma bleeding in the final `(80, 160)` crop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import os
import random

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import numpy as np
import torch
import torch.nn.functional as F

from vpocr_synth.gpu.homography_fast import REGIME_TO_ID, SENSOR_ROI_H, SENSOR_ROI_W, TrajectoryBatch
from vpocr_synth.gpu.shader_pbr_ir import ShadedROIBatch

_JPEG_LUMA_Q50 = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]

_JPEG_CHROMA_Q50 = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
]


def _build_dct8_matrix(dev: torch.device) -> torch.Tensor:
    """Build exact `8x8` orthonormal DCT-II matrix `D` (`D @ D^T == I`)."""
    n = 8
    k = torch.arange(n, device=dev, dtype=torch.float32)[:, None]
    i = torch.arange(n, device=dev, dtype=torch.float32)[None, :]
    mat = torch.cos((math.pi / float(n)) * (i + 0.5) * k) * math.sqrt(2.0 / float(n))
    mat[0, :] *= 1.0 / math.sqrt(2.0)
    return mat


def _sep_blur5(x: torch.Tensor, k5_h: torch.Tensor, k5_v: torch.Tensor) -> torch.Tensor:
    xh = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="replicate"), k5_h, groups=3)
    return F.conv2d(F.pad(xh, (0, 0, 2, 2), mode="replicate"), k5_v, groups=3)


def _optics_halation_blur_impl(
    x: torch.Tensor,             # (N, 3, 112, 176)
    hal_gain: torch.Tensor,      # (N, 1, 1, 1)
    ir_mode: torch.Tensor,       # (N,) int64
    r_vec: torch.Tensor,         # (N, 8)
    mv: torch.Tensor,            # (N, 2)
    base_grid: torch.Tensor,     # (1, 112, 176, 2)
    k5_h: torch.Tensor,
    k5_v: torch.Tensor,
    defoc_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-clamp HDR IR halation, 940nm AGC gain, defocus & 3-tap exposure motion blur."""
    h_r, w_r = x.shape[-2], x.shape[-1]
    over = F.relu(x - 0.84) * hal_gain
    b1 = _sep_blur5(over, k5_h, k5_v)
    b2 = F.avg_pool2d(F.pad(b1, (3, 3, 3, 3), mode="replicate"), kernel_size=7, stride=1)
    x = x + 1.45 * b1 + 0.85 * b2

    is_940 = ir_mode == 2
    x = torch.where(is_940[:, None, None, None], x * 1.58, x)

    defocus_amt = (F.relu(r_vec[:, 0] - 0.22) * (1.28 * defoc_scale)).clamp(0.0, 1.0)[:, None, None, None]
    x_defoc = _sep_blur5(x, k5_h, k5_v)
    x = x * (1.0 - defocus_amt) + x_defoc * defocus_amt

    mot_mag = torch.hypot(mv[:, 0], mv[:, 1])
    has_mot = (mot_mag > 0.14)[:, None, None, None]
    dx_n = mv[:, 0, None, None] * (2.0 / float(w_r))
    dy_n = mv[:, 1, None, None] * (2.0 / float(h_r))
    g_neg = torch.stack([base_grid[..., 0] - 0.38 * dx_n, base_grid[..., 1] - 0.38 * dy_n], dim=-1)
    g_pos = torch.stack([base_grid[..., 0] + 0.38 * dx_n, base_grid[..., 1] + 0.38 * dy_n], dim=-1)
    x_neg = F.grid_sample(x, g_neg, mode="bilinear", padding_mode="border", align_corners=False)
    x_pos = F.grid_sample(x, g_pos, mode="bilinear", padding_mode="border", align_corners=False)
    x_mot = 0.50 * x + 0.25 * x_neg + 0.25 * x_pos
    # Smooth continuous 1/15s..1/30s long-shutter smear when |mv| > 2.2 px
    long_sh_w = torch.clamp((mot_mag - 2.2) * 0.22, 0.0, 0.52)[:, None, None, None]
    x_mot = x_mot * (1.0 - long_sh_w) + _sep_blur5(x_mot, k5_h, k5_v) * long_sh_w
    out = torch.where(has_mot, x_mot, x)
    return out, defocus_amt.reshape(-1, 1)


def _rain_droplets_impl(
    x: torch.Tensor,
    rain_active: torch.Tensor,
    sparse_r: torch.Tensor,
    r_streak: torch.Tensor,
    base_grid: torch.Tensor,
    yy_r: torch.Tensor,
    xx_r: torch.Tensor,
    k5_h: torch.Tensor,
    k5_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refract `(112, 176)` ROI through spherical lens water droplets + add falling rain streaks."""
    h_r, w_r = x.shape[-2], x.shape[-1]
    active_f = rain_active[:, None, None, None].float()

    sparse = F.relu(sparse_r - 0.74) * 3.8
    dome = F.interpolate(sparse, size=(h_r, w_r), mode="bicubic", align_corners=False).clamp(0.0, 1.2)
    dome = dome * active_f

    d_pad = F.pad(dome, (1, 1, 1, 1), mode="replicate")
    gx = (d_pad[:, 0, 1:-1, 2:] - d_pad[:, 0, 1:-1, :-2]) * 0.22
    gy = (d_pad[:, 0, 2:, 1:-1] - d_pad[:, 0, :-2, 1:-1]) * 0.32
    refract_grid = torch.stack(
        [
            (base_grid[..., 0] - gx).clamp(-1.0, 1.0),
            (base_grid[..., 1] - gy).clamp(-1.0, 1.0),
        ],
        dim=-1,
    )
    warped = F.grid_sample(x, refract_grid, mode="bilinear", padding_mode="border", align_corners=False)
    blurred = _sep_blur5(warped, k5_h, k5_v)
    in_drop = torch.clamp(dome * 1.4, 0.0, 0.85)
    rim_spec = torch.clamp((gx.abs() + gy.abs()) * 0.85, 0.0, 0.25).unsqueeze(1)
    x_drop = warped * (1.0 - 0.55 * in_drop) + blurred * (0.55 * in_drop) + rim_spec * active_f

    slant = xx_r + (0.18 + 0.14 * r_streak[:, 0:1, None, None]) * yy_r
    phase = r_streak[:, 1:2, None, None] * 20.0
    streaks = F.relu(torch.sin(slant * 68.0 + phase) - 0.92) * 2.2 * active_f
    out = x_drop + streaks

    pen = in_drop.mean(dim=(-2, -1)).clamp(0.0, 0.85)
    return out, pen


def _sensor_noise_usm_impl(
    x: torch.Tensor,             # (N, 3, 112, 176)
    plume_low: torch.Tensor,     # (N, 1, 6, 9)
    r_vec: torch.Tensor,         # (N, 8)
    reg_bt: torch.Tensor,        # (N,) int64
    sides_bt: torch.Tensor,      # (N,) int64
    night_bt: torch.Tensor,      # (N,) bool
    ir_mode: torch.Tensor,       # (N,) int64
    y_phase: torch.Tensor,       # (1, 1, 112, 1)
    noise_rgb: torch.Tensor,     # (N, 3, 112, 176)
    row_fpn_raw: torch.Tensor,   # (N, 1, 112, 1)
    col_fpn_raw: torch.Tensor,   # (N, 1, 1, 176)
    k5_h: torch.Tensor,
    k5_v: torch.Tensor,
    sigma_stage_scale: float,
) -> torch.Tensor:
    """Fused exhaust plume, PWM flicker, full-well clamp, Poisson-Gaussian + FPN noise & USM halo."""
    h_r, w_r = x.shape[-2], x.shape[-1]
    is_mono_ir = ir_mode > 0
    is_940 = ir_mode == 2
    is_rain = reg_bt == 7  # REGIME_TO_ID["heavy_rain_droplets"] == 7

    has_plume = (sides_bt == 1) & (night_bt | is_rain) & (r_vec[:, 1] < 0.18)
    plume_up = F.interpolate(plume_low, size=(h_r, w_r), mode="bilinear", align_corners=False)
    plume_a = has_plume[:, None, None, None].float() * torch.clamp((plume_up - 0.55) * 1.1, 0.0, 0.42)
    x = x * (1.0 - plume_a) + 0.72 * plume_a

    # REGIME_TO_ID["illuminated_night"] == 6, "night_rgb_glare" == 3
    has_pwm = ((reg_bt == 6) | (reg_bt == 3)) & (r_vec[:, 2] < 0.45)
    pwm_freq = ((3.0 + 4.0 * r_vec[:, 3]) * (2.0 * math.pi))[:, None, None, None]
    pwm_phi = (r_vec[:, 4] * (2.0 * math.pi))[:, None, None, None]
    pwm_band = 1.0 + has_pwm[:, None, None, None].float() * 0.14 * torch.sin(pwm_freq * y_phase + pwm_phi)
    x = torch.clamp(x * pwm_band, 0.0, 1.0)

    base_sigma = (sigma_stage_scale * r_vec[:, 5])[:, None, None, None]
    gain_mult = torch.where(
        is_940[:, None, None, None],
        2.66,
        torch.where(night_bt[:, None, None, None], 1.55, 1.0),
    )
    sigma_read = base_sigma * gain_mult
    sigma_shot = sigma_read * 1.35

    n_f = noise_rgb.float()
    noise_mono = n_f[:, 0:1].expand_as(n_f)
    n_unit = torch.where(is_mono_ir[:, None, None, None], noise_mono, n_f)
    shot_read = n_unit * torch.sqrt((sigma_shot * sigma_shot) * x.clamp_min(1e-4) + (sigma_read * sigma_read))

    fpn_amp = torch.where(
        is_940[:, None, None, None],
        0.026,
        torch.where(night_bt[:, None, None, None], 0.010, 0.002),
    )
    row_fpn = row_fpn_raw * fpn_amp
    col_fpn = col_fpn_raw * (0.45 * fpn_amp)
    x = torch.clamp(x + shot_read + row_fpn + col_fpn, 0.0, 1.0)

    usm_amt = torch.where(r_vec[:, 6] < 0.55, 0.35 + 0.85 * r_vec[:, 7], 0.0)[:, None, None, None]
    x_blur = _sep_blur5(x, k5_h, k5_v)
    return torch.clamp(x + usm_amt * (x - x_blur), 0.0, 1.0)


def _yuv420_dct8x8_impl(
    rgb: torch.Tensor,
    q_val: torch.Tensor,
    ir_mode: torch.Tensor,
    dct_m: torch.Tensor,
    dct_mt: torch.Tensor,
    q_luma: torch.Tensor,
    q_chroma: torch.Tensor,
) -> torch.Tensor:
    """Exact `YUV420` (`NV12`) 2x2 chroma subsampling + `8x8` 2D DCT quantization on `(H_roi, W_roi)`."""
    n_bt, _, h_r, w_r = rgb.shape
    bh_y, bw_y = h_r // 8, w_r // 8
    n_y_blk = bh_y * bw_y
    bh_c, bw_c = h_r // 16, w_r // 16
    n_c_blk = bh_c * bw_c

    is_mono_ir = ir_mode > 0
    r255 = rgb[:, 0:1] * 255.0
    g255 = rgb[:, 1:2] * 255.0
    b255 = rgb[:, 2:3] * 255.0

    y_lvl = (0.299 * r255 + 0.587 * g255 + 0.114 * b255) - 128.0
    cb_lvl = -0.168736 * r255 - 0.331264 * g255 + 0.500 * b255
    cr_lvl = 0.500 * r255 - 0.418688 * g255 - 0.081312 * b255

    cb_sub = F.avg_pool2d(cb_lvl, kernel_size=2, stride=2)
    cr_sub = F.avg_pool2d(cr_lvl, kernel_size=2, stride=2)

    q_clamped = q_val.clamp(10.0, 95.0)
    scale = torch.where(q_clamped < 50.0, 5000.0 / q_clamped, 200.0 - 2.0 * q_clamped) / 100.0
    s_view = scale.view(n_bt, 1, 1, 1, 1)

    ql_mat = (q_luma * s_view).clamp(1.0, 255.0).reshape(n_bt, 1, 8, 8)
    qc_mat = (q_chroma * s_view).clamp(1.0, 255.0).reshape(n_bt, 1, 8, 8)

    y_blk = y_lvl.reshape(n_bt, bh_y, 8, bw_y, 8).permute(0, 1, 3, 2, 4).reshape(n_bt, n_y_blk, 8, 8)
    y_dct = torch.matmul(dct_m, torch.matmul(y_blk, dct_mt))
    y_q = torch.round(y_dct / ql_mat) * ql_mat
    y_rec = (
        torch.matmul(dct_mt, torch.matmul(y_q, dct_m))
        .reshape(n_bt, bh_y, bw_y, 8, 8)
        .permute(0, 1, 3, 2, 4)
        .reshape(n_bt, 1, h_r, w_r)
        + 128.0
    )

    cb_blk = cb_sub.reshape(n_bt, bh_c, 8, bw_c, 8).permute(0, 1, 3, 2, 4).reshape(n_bt, n_c_blk, 8, 8)
    cr_blk = cr_sub.reshape(n_bt, bh_c, 8, bw_c, 8).permute(0, 1, 3, 2, 4).reshape(n_bt, n_c_blk, 8, 8)
    c_blk = torch.cat([cb_blk, cr_blk], dim=1)
    c_dct = torch.matmul(dct_m, torch.matmul(c_blk, dct_mt))
    c_q = torch.round(c_dct / qc_mat) * qc_mat
    c_rec_blk = torch.matmul(dct_mt, torch.matmul(c_q, dct_m))
    cb_rec = c_rec_blk[:, :n_c_blk].reshape(n_bt, bh_c, bw_c, 8, 8).permute(0, 1, 3, 2, 4).reshape(n_bt, 1, h_r // 2, w_r // 2)
    cr_rec = c_rec_blk[:, n_c_blk:].reshape(n_bt, bh_c, bw_c, 8, 8).permute(0, 1, 3, 2, 4).reshape(n_bt, 1, h_r // 2, w_r // 2)
    c_rec = torch.cat([cb_rec, cr_rec], dim=0)

    c_up = F.interpolate(c_rec, size=(h_r, w_r), mode="bilinear", align_corners=False)
    ir_damp = torch.where(is_mono_ir[:, None, None, None], 0.15, 1.0)
    cb_up = c_up[:n_bt] * ir_damp
    cr_up = c_up[n_bt:] * ir_damp

    r_out = (y_rec + 1.402 * cr_up) / 255.0
    g_out = (y_rec - 0.344136 * cb_up - 0.714136 * cr_up) / 255.0
    b_out = (y_rec + 1.772 * cb_up) / 255.0
    return torch.clamp(torch.cat([r_out, g_out, b_out], dim=1), 0.0, 1.0)


def _tdnr_and_isp_tail_impl(
    x_seq: torch.Tensor,
    is_night: torch.Tensor,
    r_tdnr: torch.Tensor,
    mv: torch.Tensor,
    defocus_flat: torch.Tensor,
    q_val: torch.Tensor,
    droplet_pen: torch.Tensor,
    enable_tdnr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len = is_night.shape
    if seq_len >= 2:
        tdnr_w = (enable_tdnr * is_night[:, 1:].float() * (0.12 + 0.18 * r_tdnr))[:, :, None, None, None]
        diff = (x_seq[:, 1:] - x_seq[:, :-1]).abs().mean(dim=2, keepdim=True)
        ghost_gate = torch.clamp((0.28 - diff) * 5.0, 0.0, 1.0)
        w_eff = tdnr_w * ghost_gate
        x_seq_t1 = x_seq[:, 1:] * (1.0 - w_eff) + x_seq[:, :-1] * w_eff
        x_seq = torch.cat([x_seq[:, 0:1], x_seq_t1], dim=1)

    mot_mag = torch.hypot(mv[:, 0:1], mv[:, 1:2])
    blur_pen = torch.clamp(1.0 - 0.055 * mot_mag - 0.25 * defocus_flat, 0.30, 1.0)
    q_pen = torch.clamp(0.65 + 0.35 * (q_val[:, None] / 60.0), 0.65, 1.0)
    isp_mult = (
        blur_pen.reshape(bsz, seq_len, 1)
        * q_pen.reshape(bsz, seq_len, 1)
        * (1.0 - 0.45 * droplet_pen)
    ).clamp(0.15, 1.0)
    return torch.clamp(x_seq, 0.0, 1.0), isp_mult


_CODEC_CUDA_CACHE: dict[str, Callable[..., object]] = {}


def _dispatch_codec_cuda(name: str, fn: Callable[..., object], *args: object):
    first = args[0]
    if isinstance(first, torch.Tensor) and first.is_cuda:
        c_fn = _CODEC_CUDA_CACHE.get(name)
        if c_fn is None:
            c_fn = torch.compile(fn, dynamic=True)
            _CODEC_CUDA_CACHE[name] = c_fn
        return c_fn(*args)
    return fn(*args)


@dataclass
class SensorDegradedROIBatch:
    """Output of sensor/optics/ISP/codec pipeline in `(112, 176)` ROI."""

    roi_rgb: torch.Tensor               # (B, T, 3, 112, 176) float32 in [0.0, 1.0]
    isp_readability_mult: torch.Tensor  # (B, T, 1) float32 in [0.0, 1.0] global blur/rain/noise factor
    droplet_occ_half: torch.Tensor      # (B, T, 1) float32 lens droplet distortion penalty


class SensorCodecGPU:
    """Vectorized camera optics, rain droplets, NIR/CMOS sensor noise, YUV420 & 8x8 DCT codec."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self._cache: dict[torch.device, tuple[torch.Tensor, ...]] = {}

    def to(self, device: torch.device | str) -> SensorCodecGPU:
        self.device = torch.device(device)
        return self

    def _get_device_tables(self, dev: torch.device) -> tuple[torch.Tensor, ...]:
        cached = self._cache.get(dev)
        if cached is not None:
            return cached
        dct_m = _build_dct8_matrix(dev)
        dct_mt = dct_m.t().contiguous()
        q_luma = torch.tensor(_JPEG_LUMA_Q50, device=dev, dtype=torch.float32).view(1, 1, 8, 1, 8)
        q_chroma = torch.tensor(_JPEG_CHROMA_Q50, device=dev, dtype=torch.float32).view(1, 1, 8, 1, 8)

        gy = (torch.arange(SENSOR_ROI_H, device=dev, dtype=torch.float32) + 0.5) * (2.0 / float(SENSOR_ROI_H)) - 1.0
        gx = (torch.arange(SENSOR_ROI_W, device=dev, dtype=torch.float32) + 0.5) * (2.0 / float(SENSOR_ROI_W)) - 1.0
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        base_grid = torch.stack([xx, yy], dim=-1)[None]  # (1, 112, 176, 2)

        g5_1d = torch.tensor([0.0625, 0.25, 0.375, 0.25, 0.0625], device=dev, dtype=torch.float32)
        k5_h = g5_1d.view(1, 1, 1, 5).expand(3, 1, 1, 5).contiguous()
        k5_v = g5_1d.view(1, 1, 5, 1).expand(3, 1, 5, 1).contiguous()
        y_phase = torch.linspace(0.0, 1.0, SENSOR_ROI_H, device=dev)[None, None, :, None]
        xx_r = torch.linspace(0.0, 1.0, SENSOR_ROI_W, device=dev)[None, None, None, :]

        out = (dct_m, dct_mt, q_luma, q_chroma, base_grid, k5_h, k5_v, y_phase, xx_r)
        self._cache[dev] = out
        return out

    @torch.no_grad()
    def process_rois(
        self,
        shaded: ShadedROIBatch,
        traj: TrajectoryBatch,
        curriculum_stage: int = 4,
        seed: int = 0,
        jpeg_quality_override: float | None = None,
        halation_boost_override: float | None = None,
    ) -> SensorDegradedROIBatch:
        """Apply pre-clamp HDR halation, blur, rain droplets, sensor noise, USM, YUV420 & 8x8 DCT."""
        hdr = shaded.hdr_roi_rgb
        bsz, seq_len, _, h_r, w_r = hdr.shape
        n_bt = bsz * seq_len
        dev = hdr.device
        stage = int(max(1, min(4, curriculum_stage)))

        dct_m, dct_mt, q_luma, q_chroma, base_grid, k5_h, k5_v, y_phase, xx_r = self._get_device_tables(dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed((seed * 2654435761 + 97531) & 0x7FFFFFFF)
        rng_py = random.Random(seed * 31 + 7)

        x = hdr.reshape(n_bt, 3, h_r, w_r)
        ir_mode = traj.sensor_ir_mode.reshape(n_bt)
        reg_bt = traj.regime_ids[:, None].expand(bsz, seq_len).reshape(n_bt)
        night_bt = traj.is_night.reshape(n_bt)
        mv = traj.motion_vec_px.reshape(n_bt, 2)

        # 1. Fused Pre-clamp HDR IR halation + 940nm AGC + defocus + 3-tap motion blur
        if halation_boost_override is not None:
            hal_gain = torch.full((n_bt, 1, 1, 1), float(halation_boost_override), device=dev)
        else:
            hal_gain = shaded.ir_halation_gain.reshape(n_bt, 1, 1, 1) * (0.45 + 0.18 * stage)

        r_vec = torch.rand((n_bt, 8), generator=gen, device=dev)
        defoc_scale = float(0.12 + 0.16 * stage)
        x, defocus_flat = _dispatch_codec_cuda(
            f"optics_{stage}",
            _optics_halation_blur_impl,
            x,
            hal_gain,
            ir_mode,
            r_vec,
            mv,
            base_grid,
            k5_h,
            k5_v,
            defoc_scale,
        )

        # 2. Sparse CPU-indexed spherical lens water droplet refraction + falling rain streaks
        rain_seq_idx = [
            i
            for i, tag in enumerate(traj.regime_tags)
            if tag == "heavy_rain_droplets"
            or (stage >= 3 and tag not in ("night_ir_850", "night_ir_940") and rng_py.random() < 0.08)
        ]
        droplet_pen = torch.zeros((bsz, seq_len, 1), device=dev, dtype=torch.float32)
        if rain_seq_idx:
            if len(rain_seq_idx) == 1 and dev.type == "cuda" and bsz >= 2:
                rain_run_idx = [rain_seq_idx[0], rain_seq_idx[0]]
            else:
                rain_run_idx = rain_seq_idx
            n_r_seq = len(rain_run_idx)
            n_r_bt = n_r_seq * seq_len
            idx_r = torch.from_numpy(np.asarray(rain_run_idx, dtype=np.int64)).to(dev)
            x_bt = x.reshape(bsz, seq_len, 3, h_r, w_r)
            x_sub = x_bt.index_select(0, idx_r).reshape(n_r_bt, 3, h_r, w_r)
            rain_active = (traj.sensor_ir_mode.index_select(0, idx_r) == 0).reshape(n_r_bt)
            sparse_r = torch.rand((n_r_bt, 1, 9, 14), generator=gen, device=dev)
            r_streak = torch.rand((n_r_bt, 2), generator=gen, device=dev)
            x_sub_out, pen_sub = _dispatch_codec_cuda(
                "rain",
                _rain_droplets_impl,
                x_sub,
                rain_active,
                sparse_r,
                r_streak,
                base_grid,
                y_phase,
                xx_r,
                k5_h,
                k5_v,
            )
            x_bt[idx_r[: len(rain_seq_idx)]] = x_sub_out.reshape(n_r_seq, seq_len, 3, h_r, w_r)[: len(rain_seq_idx)]
            droplet_pen[idx_r[: len(rain_seq_idx)]] = pen_sub.reshape(n_r_seq, seq_len, 1)[: len(rain_seq_idx)]
            x = x_bt.reshape(n_bt, 3, h_r, w_r)

        # 3. Fused exhaust plume + PWM flicker + full-well clamp + NIR/CMOS noise + FPN + USM halo
        plume_low = torch.rand((n_bt, 1, 6, 9), generator=gen, device=dev)
        n_dtype = torch.float16 if dev.type == "cuda" else torch.float32
        noise_rgb = torch.empty((n_bt, 3, h_r, w_r), device=dev, dtype=n_dtype).normal_(generator=gen)
        row_fpn_raw = torch.randn((n_bt, 1, h_r, 1), generator=gen, device=dev)
        col_fpn_raw = torch.randn((n_bt, 1, 1, w_r), generator=gen, device=dev)
        sigma_stage_scale = float(0.006 + 0.008 * stage)

        x = _dispatch_codec_cuda(
            f"noise_{stage}",
            _sensor_noise_usm_impl,
            x,
            plume_low,
            r_vec,
            reg_bt,
            traj.plate_sides.reshape(n_bt),
            night_bt,
            ir_mode,
            y_phase,
            noise_rgb,
            row_fpn_raw,
            col_fpn_raw,
            k5_h,
            k5_v,
            sigma_stage_scale,
        )
        del noise_rgb, row_fpn_raw, col_fpn_raw, plume_low

        # 4. YUV420 (NV12) 2x2 Chroma Subsampling + Exact 8x8 2D DCT Codec
        if jpeg_quality_override is not None:
            q_val = torch.full((n_bt,), float(jpeg_quality_override), device=dev)
        else:
            q_lo = {1: 54.0, 2: 34.0, 3: 22.0, 4: 18.0}[stage]
            q_hi = 78.0
            q_val = q_lo + (q_hi - q_lo) * torch.rand((n_bt,), generator=gen, device=dev)

        x = _dispatch_codec_cuda("dct", _yuv420_dct8x8_impl, x, q_val, ir_mode, dct_m, dct_mt, q_luma, q_chroma)

        # 5. Fused Temporal 3DNR Ghosting + ISP Readability Multiplier
        x_seq = x.reshape(bsz, seq_len, 3, h_r, w_r)
        r_tdnr = torch.rand((bsz, max(1, seq_len - 1)), generator=gen, device=dev)
        enable_tdnr = 1.0 if (seq_len >= 2 and stage >= 2) else 0.0
        roi_out, isp_mult = _dispatch_codec_cuda(
            f"tdnr_{1 if seq_len >= 2 else 0}",
            _tdnr_and_isp_tail_impl,
            x_seq,
            traj.is_night,
            r_tdnr,
            mv,
            defocus_flat,
            q_val,
            droplet_pen,
            enable_tdnr,
        )

        return SensorDegradedROIBatch(
            roi_rgb=roi_out,
            isp_readability_mult=isp_mult,
            droplet_occ_half=droplet_pen,
        )
