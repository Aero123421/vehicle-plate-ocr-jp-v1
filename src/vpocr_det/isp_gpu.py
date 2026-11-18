"""GPU-accelerated Camera ISP & Sensor Degradation Simulator.

Operates directly on `(B, 3, H, W)` single frames or `(B, K, 3, H, W)` / `(K, 3, H, W)` temporal clips
in `[0, 1]` with:
- Zero spatial phase shift (`0.000 px` symmetric kernels, optical-center radial LCA, & pixel-center-aligned resampling),
- 100% isolated RNG state (never touches global `torch` RNG; bit-exact reproducibility via `reseed(seed)`),
- Clip-consistent camera ISP persistence across `K` frames in `forward_clips(...)` while varying CMOS
  shot + read + WDR-shadow-amplified sensor noise independently per frame,
- Honest physical sensor modeling: WDR local shadow-boost + amplified shadow noise, lateral chromatic
  aberration (LCA), and orthonormal `8x8` 2D DCT block quantization perturbation (not a full temporal H.264 codec),
- Deterministic compositional sensor-family holdout splits (`domain_split in {"train", "val_holdout", "cal_holdout"}`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F

VALID_ISP_DOMAIN_SPLITS: tuple[str, ...] = ("train", "val_holdout", "cal_holdout")

LOWRES_SCALES: tuple[float, ...] = (0.50, 0.58, 0.65, 0.72, 0.80, 0.88)

SENSOR_FAMILIES_BY_DOMAIN_SPLIT: dict[str, tuple[str, ...]] = {
    "train": (
        "cmos_srgb_std",
        "cctv_wdr_sharp",
        "substream_lowres_420",
        "outdoor_glare_dct8x8",
    ),
    "val_holdout": (
        "wide_lca_wdr_boost",
        "global_dct8x8_coarse",
        "anamorphic_haze_cmos",
    ),
    "cal_holdout": (
        "shadow_gain_wdr_420",
        "defocus_lca_cctv",
        "droplet_contrast_cmos",
    ),
}


@dataclass
class GPUISPConfig:
    """Probabilities, sensor-family split, and strength ranges for `GPUCameraISPAugmenter`."""

    domain_split: str = "train"             # "train" | "val_holdout" | "cal_holdout" -> disjoint sensor families!
    p_augment: float = 0.94                 # overall probability per sample/clip of applying ISP pipeline
    p_lowres_capture: float | None = None   # 0.36 on "train" (224p/260p/290p/320p/360p/400p on 448p), 0.55 on holdout splits
    p_interlaced: float = 0.08              # half-vertical-res CCTV interlace mode (within low-res capture)
    p_anamorphic: float = 0.16              # horizontal squeeze (within low-res capture)
    p_blur: float = 0.30                    # optical defocus (3x3..7x7) or directional shutter motion blur
    p_chromatic_aberration: float = 0.42    # radial lateral chromatic aberration (R/B magnification dispersion)
    p_oversharpen: float = 0.42             # cheap CCTV ISP unsharp-mask halo ringing (amount up to 1.4)
    p_chroma_sub: float = 0.52              # YCbCr 4:2:0 2x2 chroma subsampling bleed
    p_codec_block: float = 0.48             # 8x8 2D DCT block quantization perturbation (back-compat field name)
    p_block_quant: float | None = None      # optional explicit alias for 8x8 2D DCT block quantization perturbation
    p_pframe_ghost: float = 0.45            # temporal P-frame residual quantization ghosting on multi-frame clips
    p_wdr_shadow_boost: float = 0.46        # WDR local shadow tone-boost + proportional shadow read-noise amplification
    p_photometric: float = 0.85             # metered exposure, gamma, Planckian WB mismatch, contrast
    p_tunnel_sodium: float = 0.15           # highway/mountain tunnel orange sodium-vapor cast (train split)
    p_backlit_shadow_crush: float = 0.22    # backlit S-curve shadow crush + upper-sky veiling glare (train split)
    p_wet_specular_streak: float = 0.18     # wet-road vertical specular tail-lamp/headlight reflections (train split)
    p_fog_haze: float = 0.16                # atmospheric fog / haze (`img * trans + airlight * (1 - trans)`)
    p_night_ir: float = 0.12                # monochrome Night-IR mode (p = 0.12)
    p_glare_blob: float = 0.24              # localized specular headlight / sun glare bloom
    p_lens_droplets: float = 0.18           # localized circular blurry/bright lens water droplets
    p_sensor_noise: float = 0.65            # CMOS Poisson shot + Gaussian read (+ WDR shadow gain) noise

    def __post_init__(self) -> None:
        if self.domain_split not in VALID_ISP_DOMAIN_SPLITS:
            raise ValueError(
                f"GPUISPConfig.domain_split must be one of {VALID_ISP_DOMAIN_SPLITS}, got {self.domain_split!r}"
            )
        if self.p_lowres_capture is None:
            self.p_lowres_capture = 0.36 if self.domain_split == "train" else 0.55
        else:
            self.p_lowres_capture = float(self.p_lowres_capture)
        if self.p_block_quant is not None:
            self.p_codec_block = float(self.p_block_quant)


def _build_psf_bank_7x7() -> Tensor:
    """Build `(15, 3, 1, 7, 7)` symmetric zero-phase bank: `[0..3]`=Gaussian defocus, `[4..14]`=motion blurs."""
    kernels: list[Tensor] = []
    coords = torch.arange(7, dtype=torch.float32) - 3.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r2 = xx * xx + yy * yy

    # Symmetric optical defocus (3x3, 5x5, 7x7 Gaussian; exact centroid at (3, 3))
    for sigma, radius in ((0.75, 1), (1.10, 2), (1.55, 3), (2.15, 3)):
        g = torch.exp(-0.5 * r2 / (sigma * sigma)) * ((xx.abs() <= radius) & (yy.abs() <= radius)).to(torch.float32)
        g = g / g.sum().clamp_min(1e-6)
        kernels.append(g)

    # Symmetric directional motion blur kernels (exact centroid at (3, 3))
    angles_deg = (0.0, 0.0, 90.0, 45.0, -45.0, 18.0, -18.0, 32.0, -32.0, 68.0, -68.0)
    half_lens = (2.0, 3.0, 2.5, 2.8, 2.8, 3.0, 3.0, 2.3, 2.3, 2.4, 2.4)
    for ang_d, hlen in zip(angles_deg, half_lens):
        rad = math.radians(ang_d)
        dx, dy = math.cos(rad), math.sin(rad)
        proj = xx * dx + yy * dy
        perp = (xx * dy - yy * dx).abs()
        k = torch.clamp(1.15 - perp, min=0.0) * (proj.abs() <= hlen + 0.35).to(torch.float32)
        k = k / k.sum().clamp_min(1e-6)
        kernels.append(k)

    stacked = torch.stack(kernels, dim=0)  # (15, 7, 7)
    return stacked[:, None, None, :, :].expand(-1, 3, 1, 7, 7).contiguous()


def _build_psf_bank_11x11() -> Tensor:
    """Build `(24, 3, 1, 11, 11)` symmetric zero-phase bank: `[0..3]`=defocus, `[4..23]`=shutter motion `k=3..11 px`."""
    kernels: list[Tensor] = []
    coords = torch.arange(11, dtype=torch.float32) - 5.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r2 = xx * xx + yy * yy

    for sigma, radius in ((0.75, 1), (1.15, 2), (1.65, 3), (2.25, 4)):
        g = torch.exp(-0.5 * r2 / (sigma * sigma)) * ((xx.abs() <= radius) & (yy.abs() <= radius)).to(torch.float32)
        g = g / g.sum().clamp_min(1e-6)
        kernels.append(g)

    # Directional shutter motion blur kernels spanning k = 3, 5, 7, 9, 11 px (half-length 1.0..5.0 around (5, 5))
    motion_specs: tuple[tuple[float, float], ...] = (
        (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0), (0.0, 5.0),
        (14.0, 1.5), (-14.0, 1.5), (14.0, 3.5), (-14.0, 3.5), (14.0, 5.0), (-14.0, 5.0),
        (28.0, 2.5), (-28.0, 2.5), (28.0, 4.5), (-28.0, 4.5),
        (45.0, 2.0), (-45.0, 4.0), (68.0, 3.0), (-68.0, 3.0), (90.0, 3.5),
    )
    for ang_d, hlen in motion_specs:
        rad = math.radians(ang_d)
        dx, dy = math.cos(rad), math.sin(rad)
        proj = xx * dx + yy * dy
        perp = (xx * dy - yy * dx).abs()
        k = torch.clamp(1.15 - perp, min=0.0) * torch.clamp(hlen + 0.65 - proj.abs(), min=0.0, max=1.0)
        k = k / k.sum().clamp_min(1e-6)
        kernels.append(k)

    stacked = torch.stack(kernels, dim=0)  # (24, 11, 11)
    return stacked[:, None, None, :, :].expand(-1, 3, 1, 11, 11).contiguous()


def _build_dct8_basis() -> tuple[Tensor, Tensor]:
    """Build exact `8x8` orthonormal Type-II DCT matrix `D` and spatial-frequency weight table `F_uv`."""
    ns = torch.arange(8, dtype=torch.float32)
    ks = torch.arange(8, dtype=torch.float32)[:, None]
    mat = torch.cos(math.pi * (ns[None, :] + 0.5) * ks / 8.0)
    mat[0, :] *= math.sqrt(1.0 / 8.0)
    mat[1:, :] *= math.sqrt(2.0 / 8.0)
    uu, vv = torch.meshgrid(torch.arange(8, dtype=torch.float32), torch.arange(8, dtype=torch.float32), indexing="ij")
    # Standard radial spatial-frequency quantization taper (higher AC frequencies quantized more coarsely)
    freq_weight = 1.0 + 0.35 * (uu + vv) + 0.08 * (uu * vv)
    return mat.contiguous(), freq_weight.contiguous()


class GPUCameraISPAugmenter(nn.Module):
    """GPU Camera-ISP & Sensor Degradation Augmenter.

    Preserves exact pixel-center geometry (`0.000 px` spatial phase shift), isolates all RNG draws
    from PyTorch's global generator, supports clip-persistent temporal ISP via `forward_clips`, and
    isolates sensor-family parameter distributions across `domain_split`.
    """

    LOWRES_SCALES: tuple[float, ...] = (0.50, 0.58, 0.65, 0.72, 0.80, 0.88)
    _HOLDOUT_LOWRES_SCALES: tuple[float, ...] = (0.22, 0.35, 0.50, 0.70)
    UPSAMPLE_MODES: tuple[str, ...] = ("bilinear", "bicubic", "nearest")
    NUM_DEFOCUS_KERNELS: int = 4

    psf_bank_fp32: Tensor
    psf_bank_fp16: Tensor
    psf_bank_11x11_fp32: Tensor
    psf_bank_11x11_fp16: Tensor
    dct8_mat_fp32: Tensor
    dct8_freq_w_fp32: Tensor
    rgb_to_y: Tensor

    def __init__(
        self,
        config: GPUISPConfig | None = None,
        seed: int | None = None,
        *,
        p_lowres: float | None = None,
        p_night_ir: float | None = None,
        domain_split: str | None = None,
        **_kwargs: object,
    ) -> None:
        super().__init__()
        cfg = config or GPUISPConfig()
        if domain_split is not None:
            raw_lr = p_lowres if p_lowres is not None else (
                None if (config is None) else cfg.p_lowres_capture
            )
            cfg = replace(cfg, domain_split=str(domain_split), p_lowres_capture=raw_lr)
        elif p_lowres is not None:
            cfg = replace(cfg, p_lowres_capture=float(p_lowres))
        if p_night_ir is not None:
            cfg = replace(cfg, p_night_ir=float(p_night_ir))
        self.config = cfg
        self.last_sensor_families: list[str] = []

        split_salt = {"train": 0, "val_holdout": 0x1F35A7, "cal_holdout": 0x3B9ACA}[self.config.domain_split]
        base_seed = (20261111 ^ split_salt) if seed is None else int(seed)
        self._init_seed = base_seed
        self._py_rng = random.Random(base_seed)
        self._cpu_gen = torch.Generator(device="cpu")
        self._cpu_gen.manual_seed((base_seed ^ 0x5A5A5A5A) & 0x7FFFFFFFFFFFFFFF)

        bank = _build_psf_bank_7x7()
        bank11 = _build_psf_bank_11x11()
        dct_m, dct_fw = _build_dct8_basis()
        self.register_buffer("psf_bank_fp32", bank, persistent=False)
        self.register_buffer("psf_bank_fp16", bank.to(torch.float16), persistent=False)
        self.register_buffer("psf_bank_11x11_fp32", bank11, persistent=False)
        self.register_buffer("psf_bank_11x11_fp16", bank11.to(torch.float16), persistent=False)
        self.register_buffer("dct8_mat_fp32", dct_m, persistent=False)
        self.register_buffer("dct8_freq_w_fp32", dct_fw, persistent=False)
        self.register_buffer(
            "rgb_to_y",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @property
    def available_sensor_families(self) -> tuple[str, ...]:
        """Return the disjoint sensor family pool active for `self.config.domain_split`."""
        return SENSOR_FAMILIES_BY_DOMAIN_SPLIT[self.config.domain_split]

    def reseed(self, seed: int) -> None:
        """Reset internal RNG streams for bit-exact deterministic replay without touching global RNG."""
        s = int(seed)
        self._init_seed = s
        self._py_rng.seed(s)
        self._cpu_gen.manual_seed((s ^ 0x5A5A5A5A) & 0x7FFFFFFFFFFFFFFF)

    def _rand_prob(self, shape: tuple[int, ...], p: float, device: torch.device, dtype: torch.dtype) -> Tensor:
        if p <= 0.0:
            return torch.zeros(shape, device=device, dtype=dtype)
        if p >= 1.0:
            return torch.ones(shape, device=device, dtype=dtype)
        u = torch.rand(shape, generator=self._cpu_gen, dtype=torch.float32)
        return (u < p).to(device=device, dtype=dtype)

    def _rand_bool(self, shape: tuple[int, ...], p: float, device: torch.device) -> Tensor:
        if p <= 0.0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        if p >= 1.0:
            return torch.ones(shape, device=device, dtype=torch.bool)
        u = torch.rand(shape, generator=self._cpu_gen, dtype=torch.float32)
        return (u < p).to(device=device)

    def _rand_uniform(
        self,
        shape: tuple[int, ...],
        low: float,
        high: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        u = torch.rand(shape, generator=self._cpu_gen, dtype=torch.float32)
        out = low + (high - low) * u
        return out.to(device=device, dtype=dtype)

    def _apply_ycbcr_420(self, sub_rgb: Tensor, eff_ir_per_frame: Tensor) -> Tensor:
        """True ITU-R BT.601 full-range `YCbCr 4:2:0` 2x2 chroma subsampling + bilinear chroma upsampling."""
        n_f, c, h, w = sub_rgb.shape
        if h < 4 or w < 4:
            return sub_rgb
        orig_dt = sub_rgb.dtype
        x = sub_rgb.to(dtype=torch.float32)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.500000 * b
        cr = 0.500000 * r - 0.418688 * g - 0.081312 * b
        ch_h, ch_w = max(1, h // 2), max(1, w // 2)
        chroma = torch.cat([cb, cr], dim=1)
        chroma_sub = F.interpolate(chroma, size=(ch_h, ch_w), mode="area")
        chroma_up = F.interpolate(chroma_sub, size=(h, w), mode="bilinear", align_corners=False)
        cb_u, cr_u = chroma_up[:, 0:1], chroma_up[:, 1:2]
        r_out = y + 1.402000 * cr_u
        g_out = y - 0.344136 * cb_u - 0.714136 * cr_u
        b_out = y + 1.772000 * cb_u
        rec = torch.cat([r_out, g_out, b_out], dim=1).clamp_(0.0, 1.0)
        if bool(eff_ir_per_frame.any().item()):
            ir_m = eff_ir_per_frame.view(n_f, 1, 1, 1)
            rec = torch.where(ir_m, y.clamp(0.0, 1.0).expand(n_f, c, h, w), rec)
        return rec.to(dtype=orig_dt)

    def _apply_h264_dct8x8_and_pframe(
        self,
        sub_5d: Tensor,
        q_step_clip: Tensor,
        ghost_alpha_clip: Tensor,
        eff_ir_clip: Tensor,
    ) -> Tensor:
        """Apply `8x8` DCT quantization (`q_step` up to `0.095`) + temporal P-frame residual ghosting across `K` frames."""
        n_c, k_f, c, h, w = sub_5d.shape
        orig_dt = sub_5d.dtype
        dev = sub_5d.device
        if k_f <= 1 or not bool((ghost_alpha_clip > 1e-4).any().item()):
            q_f = q_step_clip.view(n_c, 1).expand(n_c, k_f).reshape(-1)
            ir_f = eff_ir_clip.view(n_c, 1).expand(n_c, k_f).reshape(-1)
            out_flat = self._apply_dct8x8_quantization(sub_5d.reshape(n_c * k_f, c, h, w), q_f, ir_f)
            return out_flat.view(n_c, k_f, c, h, w).to(dtype=orig_dt)

        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        x_all = sub_5d.to(dtype=torch.float32)
        if pad_h > 0 or pad_w > 0:
            x_flat = x_all.reshape(n_c * k_f, c, h, w)
            x_flat = F.pad(x_flat, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), mode="replicate")
            x_all = x_flat.view(n_c, k_f, c, h + pad_h, w + pad_w)
        hp, wp = x_all.shape[-2], x_all.shape[-1]
        bh, bw = hp // 8, wp // 8

        d_mat = self.dct8_mat_fp32.to(device=dev, dtype=torch.float32)
        d_mat_t = d_mat.transpose(0, 1).contiguous()
        f_w = self.dct8_freq_w_fp32.to(device=dev, dtype=torch.float32).view(1, 1, 1, 1, 8, 8)
        ac_mask = torch.ones((1, 1, 1, 1, 8, 8), device=dev, dtype=torch.float32)
        ac_mask[..., 0, 0] = 0.0

        delta_i = q_step_clip.to(device=dev, dtype=torch.float32).view(n_c, 1, 1, 1, 1, 1) * f_w
        delta_p = delta_i * 1.45
        g_alpha = ghost_alpha_clip.to(device=dev, dtype=torch.float32).view(n_c, 1, 1, 1)
        do_p = g_alpha > 1e-4
        ir_m = eff_ir_clip.to(device=dev, dtype=torch.bool).view(n_c, 1, 1, 1)

        def _quant_blocks(frame_4d: Tensor, step_6d: Tensor, deadzone: float = 0.0) -> Tensor:
            blks = frame_4d.view(n_c, c, bh, 8, bw, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
            cf = torch.matmul(d_mat, torch.matmul(blks, d_mat_t))
            if deadzone > 0.0:
                dz = deadzone * step_6d * ac_mask
                cf = torch.where(cf.abs() < dz, torch.zeros_like(cf), cf)
            cf_q = torch.round(cf / step_6d) * step_6d
            rb = torch.matmul(d_mat_t, torch.matmul(cf_q, d_mat))
            return rb.permute(0, 1, 2, 4, 3, 5).reshape(n_c, c, hp, wp)

        rec_frames: list[Tensor] = []
        rec_prev = _quant_blocks(x_all[:, 0], delta_i).clamp_(0.0, 1.0)
        if bool(ir_m.any().item()):
            rec_prev = torch.where(ir_m, rec_prev[:, :1].expand(n_c, c, hp, wp), rec_prev)
        rec_frames.append(rec_prev)

        for t_i in range(1, k_f):
            cur = x_all[:, t_i]
            intra_t = _quant_blocks(cur, delta_i).clamp_(0.0, 1.0)
            res = cur - rec_prev
            res_q = _quant_blocks(res, delta_p, deadzone=0.58)
            inter_t = (rec_prev + (1.0 - g_alpha) * res_q).clamp_(0.0, 1.0)
            rec_prev = torch.where(do_p, inter_t, intra_t)
            if bool(ir_m.any().item()):
                rec_prev = torch.where(ir_m, rec_prev[:, :1].expand(n_c, c, hp, wp), rec_prev)
            rec_frames.append(rec_prev)

        out = torch.stack(rec_frames, dim=1)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :, pad_top : pad_top + h, pad_left : pad_left + w]
        return out.clamp_(0.0, 1.0).to(dtype=orig_dt)

    def _degrade_lowres_slice(
        self,
        sub: Tensor,
        h: int,
        w: int,
        *,
        frames_per_clip: int = 1,
        eff_ir_clip: Tensor | None = None,
    ) -> Tensor:
        is_train = self.config.domain_split == "train"
        scale_pool = self.LOWRES_SCALES if is_train else self._HOLDOUT_LOWRES_SCALES
        scale = self._py_rng.choice(scale_pool)
        sx = scale
        sy = scale
        interlaced = self._py_rng.random() < self.config.p_interlaced
        if interlaced:
            sy *= 0.52
        if self._py_rng.random() < self.config.p_anamorphic:
            sx *= self._py_rng.uniform(0.65, 0.85)

        min_h = 8 if is_train else 12
        min_w = 12 if is_train else 16
        lh = min(h, max(min_h, int(round(h * sy))))
        lw = min(w, max(min_w, int(round(w * sx))))
        if (lh % 2) != (h % 2):
            lh = (lh - 1) if lh >= h else (lh + 1)
        if (lw % 2) != (w % 2):
            lw = (lw - 1) if lw >= w else (lw + 1)
        lh = max(2 if is_train else 1, lh)
        lw = max(2 if is_train else 1, lw)

        work_dtype = torch.float64 if sub.dtype == torch.float64 else torch.float32
        if is_train:
            # Area downsampling preserves exact inter-frame sub-pixel motion phase and 0.000 px centroid alignment!
            low = F.interpolate(sub.to(dtype=work_dtype), size=(lh, lw), mode="area")
            n_tot = int(low.shape[0])
            k_f = max(1, int(frames_per_clip))
            n_c = max(1, n_tot // k_f)
            dev = low.device
            if eff_ir_clip is None:
                ir_c = torch.zeros((n_c,), device=dev, dtype=torch.bool)
            else:
                ir_c = eff_ir_clip.to(device=dev, dtype=torch.bool).reshape(n_c)
            ir_f = ir_c[:, None].expand(n_c, k_f).reshape(n_tot)

            # 1) Per-frame independent Poisson-Gaussian read/shot noise at native low-res sensor raster
            if self.config.p_sensor_noise > 0.0 and self._py_rng.random() < 0.82:
                lin_lr = torch.where(
                    low.float() <= 0.04045,
                    low.float() / 12.92,
                    ((low.float() + 0.055) / 1.055).pow(2.4),
                )
                sig_c = self._rand_uniform((n_c, 1, 1, 1), 0.005, 0.026, dev, torch.float32)
                sig_ir = self._rand_uniform((n_c, 1, 1, 1), 0.014, 0.038, dev, torch.float32)
                sig_use = torch.where(ir_c.view(n_c, 1, 1, 1), sig_ir, sig_c)
                sig_f = sig_use[:, None].expand(n_c, k_f, 1, 1, 1).reshape(n_tot, 1, 1, 1)
                ir_m4 = ir_f.view(n_tot, 1, 1, 1)
                rgb2y_f = self.rgb_to_y.to(device=dev, dtype=torch.float32)
                lin_lr = torch.where(ir_m4, (lin_lr * rgb2y_f).sum(1, keepdim=True).expand_as(lin_lr), lin_lr)
                noisy_lr = self._apply_raw_sensor_noise(lin_lr, sig_f.square() * 3.0, sig_f * 0.58, ir_m4)
                enc_lr = torch.where(
                    noisy_lr <= 0.0031308,
                    12.92 * noisy_lr,
                    1.055 * noisy_lr.clamp_min(1e-8).pow(1.0 / 2.4) - 0.055,
                )
                low = self._highlight_shoulder(enc_lr).clamp(0.0, 1.0).to(dtype=work_dtype)

            # 2) True YCbCr 4:2:0 2x2 chroma subsampling + bilinear chroma upsampling at native substream resolution
            if lh >= 4 and lw >= 4 and self._py_rng.random() < 0.88:
                low = self._apply_ycbcr_420(low.float(), ir_f).to(dtype=work_dtype)

            # 3) H.264 8x8 DCT quantization (dct_step up to 0.095) + temporal P-frame residual ghosting
            if lh >= 8 and lw >= 8 and self._py_rng.random() < 0.85:
                q_lr = self._rand_uniform((n_c,), 0.014, 0.095, dev, torch.float32)
                do_pf = self._rand_prob((n_c,), self.config.p_pframe_ghost if k_f >= 2 else 0.0, dev, torch.float32)
                g_lr = self._rand_uniform((n_c,), 0.07, 0.22, dev, torch.float32) * do_pf
                low_5d = low.float().view(n_c, k_f, int(low.shape[1]), lh, lw)
                low = self._apply_h264_dct8x8_and_pframe(low_5d, q_lr, g_lr, ir_c).reshape_as(low).to(dtype=work_dtype)
        else:
            # Exact legacy holdout path (bit-identical for val_holdout & cal_holdout)
            if h >= 2 * lh and w >= 2 * lw and (h % 2 == 0) and (w % 2 == 0):
                pooled = F.avg_pool2d(sub.to(dtype=work_dtype), kernel_size=2, stride=2, ceil_mode=False)
            else:
                pooled = sub.to(dtype=work_dtype)
            low = F.interpolate(pooled, size=(lh, lw), mode="bilinear", align_corners=False)

        up_mode = self._py_rng.choice(self.UPSAMPLE_MODES)
        if up_mode == "nearest":
            # Exact integer reflection-symmetric pixel-center nearest-neighbor upsampling (`0.000000 px` phase bias!)
            ry2 = 2 * torch.arange(h, device=low.device, dtype=torch.int64) + 1
            rx2 = 2 * torch.arange(w, device=low.device, dtype=torch.int64) + 1
            my = ry2 * int(lh)
            mx = rx2 * int(lw)
            iy = torch.where(ry2 < int(h), (my - 1) // (2 * int(h)), my // (2 * int(h))).clamp_(0, lh - 1)
            ix = torch.where(rx2 < int(w), (mx - 1) // (2 * int(w)), mx // (2 * int(w))).clamp_(0, lw - 1)
            rec = low.index_select(2, iy).index_select(3, ix).to(dtype=sub.dtype).clone()
        elif up_mode == "bicubic":
            rec = F.interpolate(low, size=(h, w), mode="bicubic", align_corners=False).to(dtype=sub.dtype)
        else:
            rec = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False).to(dtype=sub.dtype)

        if interlaced and h >= 8 and w >= 4:
            rec = self._apply_interlace_comb(rec)
        return rec.clamp_(0.0, 1.0)

    @staticmethod
    def _apply_interlace_comb(rec: Tensor) -> Tensor:
        """Symmetric zero-phase horizontal interlace comb filter `[0.175, 0.650, 0.175]` (`0.000 px` shift)."""
        odd = rec[:, :, 1::2, :]
        odd_pad = F.pad(odd, (1, 1, 0, 0), mode="replicate")
        rec[:, :, 1::2, :] = 0.175 * odd_pad[:, :, :, :-2] + 0.650 * odd + 0.175 * odd_pad[:, :, :, 2:]
        return rec

    def _apply_lateral_chromatic_aberration(
        self,
        sub_rgb: Tensor,
        kr_per_frame: Tensor,
        kb_per_frame: Tensor,
    ) -> Tensor:
        """Apply radial optical-center lateral chromatic aberration (`0.000 px` net centroid shift).

        Red and Blue channels undergo wavelength-dependent radial magnification `1 + k * r^2`
        around the pixel-center-aligned optical axis `(0, 0)` while Green stays at reference `1.0`.
        """
        n_f, _, h, w = sub_rgb.shape
        orig_dt = sub_rgb.dtype
        dev = sub_rgb.device
        # Exact `align_corners=False` normalized pixel centers: `2*(i + 0.5)/N - 1`
        gy = ((torch.arange(h, device=dev, dtype=torch.float32) + 0.5) * (2.0 / float(h)) - 1.0).view(1, h, 1)
        gx = ((torch.arange(w, device=dev, dtype=torch.float32) + 0.5) * (2.0 / float(w)) - 1.0).view(1, 1, w)
        r2 = gx * gx + gy * gy  # (1, H, W)

        kr = kr_per_frame.to(dtype=torch.float32).view(n_f, 1, 1)
        kb = kb_per_frame.to(dtype=torch.float32).view(n_f, 1, 1)
        scale_r = 1.0 + kr * r2
        scale_b = 1.0 + kb * r2

        grid_r = torch.stack([gx * scale_r, gy * scale_r], dim=-1)
        grid_b = torch.stack([gx * scale_b, gy * scale_b], dim=-1)
        sub_f32 = sub_rgb.to(dtype=torch.float32)
        ch_r = F.grid_sample(sub_f32[:, 0:1], grid_r, mode="bilinear", padding_mode="border", align_corners=False)
        ch_b = F.grid_sample(sub_f32[:, 2:3], grid_b, mode="bilinear", padding_mode="border", align_corners=False)
        return torch.cat([ch_r, sub_f32[:, 1:2], ch_b], dim=1).clamp_(0.0, 1.0).to(dtype=orig_dt)

    def _apply_dct8x8_quantization(
        self,
        sub_rgb: Tensor,
        q_step_per_frame: Tensor,
        eff_ir_per_frame: Tensor,
    ) -> Tensor:
        """Apply true orthonormal `8x8` 2D DCT block quantization perturbation (not a temporal H.264 codec).

        Transforms non-overlapping `8x8` spatial blocks into the orthonormal DCT-II domain, quantizes
        coefficients with spatial-frequency taper `Q(u, v)`, and reconstructs via `D^T Q(D X D^T) D`.
        Preserves `0.000 px` spatial phase and monochrome `R == G == B` on Night-IR frames.
        """
        n_f, c, h, w = sub_rgb.shape
        orig_dt = sub_rgb.dtype
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        x_f32 = sub_rgb.to(dtype=torch.float32)
        if pad_h > 0 or pad_w > 0:
            x_f32 = F.pad(x_f32, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), mode="replicate")
        hp, wp = x_f32.shape[-2], x_f32.shape[-1]
        bh, bw = hp // 8, wp // 8

        d_mat = self.dct8_mat_fp32.to(device=sub_rgb.device, dtype=torch.float32)
        d_mat_t = d_mat.transpose(0, 1).contiguous()
        f_w = self.dct8_freq_w_fp32.to(device=sub_rgb.device, dtype=torch.float32).view(1, 1, 1, 1, 8, 8)

        blocks = x_f32.view(n_f, c, bh, 8, bw, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        coeff = torch.matmul(d_mat, torch.matmul(blocks, d_mat_t))
        delta = q_step_per_frame.to(dtype=torch.float32).view(n_f, 1, 1, 1, 1, 1) * f_w
        coeff_q = torch.round(coeff / delta) * delta
        rec_blocks = torch.matmul(d_mat_t, torch.matmul(coeff_q, d_mat))
        rec = rec_blocks.permute(0, 1, 2, 4, 3, 5).reshape(n_f, c, hp, wp)
        if pad_h > 0 or pad_w > 0:
            rec = rec[:, :, pad_top : pad_top + h, pad_left : pad_left + w]
        rec = rec.clamp_(0.0, 1.0)
        if bool(eff_ir_per_frame.any().item()):
            ir_m = eff_ir_per_frame.view(n_f, 1, 1, 1)
            mono = rec[:, :1, :, :].expand(n_f, c, h, w)
            rec = torch.where(ir_m, mono, rec)
        return rec.to(dtype=orig_dt)

    def _apply_raw_sensor_noise(
        self, linear_signal: Tensor, shot_scale: Tensor, read_std: Tensor, monochrome: Tensor,
    ) -> Tensor:
        """Sample electrons and independent read noise in linear light using an isolated device RNG.

        Conditional variance is `shot_scale * signal + read_std**2`. This remains Poisson
        at low photon counts rather than substituting a signal-scaled Gaussian texture.
        """
        rng_seed = int(torch.randint(0, 2**63 - 1, (), generator=self._cpu_gen).item())
        generator = torch.Generator(device=linear_signal.device).manual_seed(rng_seed)
        signal = linear_signal.float().clamp_min(0)
        scale = shot_scale.float().clamp_min(1e-8)
        noisy = torch.poisson(signal / scale, generator=generator) * scale
        noisy += torch.randn(signal.shape, device=signal.device, generator=generator) * read_std.float()
        noisy = torch.where(monochrome, noisy[:, :1].expand_as(noisy), noisy)
        return noisy.clamp_min(0).to(linear_signal.dtype)

    @staticmethod
    def _apply_lens_bloom(linear_signal: Tensor, strength: Tensor) -> Tensor:
        """Scatter bright scene radiance through a broad PSF and a central lens ghost.

        Dark scenes without bright sources produce no flare. The ghost follows the
        source reflected about the optical centre, and the PSF preserves IR neutrality.
        """
        source = (linear_signal - 0.55).clamp_min(0)
        h, w = source.shape[-2:]
        low = F.adaptive_avg_pool2d(source, (max(1, h // 8), max(1, w // 8)))
        spread = F.avg_pool2d(F.pad(low, (3, 3, 3, 3), mode="replicate"), 7, stride=1)
        halo = F.interpolate(spread, size=(h, w), mode="bilinear", align_corners=False)
        ghost = F.interpolate(torch.flip(spread, (-2, -1)), size=(h, w), mode="bilinear", align_corners=False)
        return linear_signal + strength * (0.8 * halo + 0.2 * ghost)

    @staticmethod
    def _highlight_shoulder(signal: Tensor, knee: float = 0.80) -> Tensor:
        """Unit-slope rational shoulder; preserve midtones and finite highlight detail."""
        excess = (signal - knee).clamp_min(0.0)
        return torch.where(signal <= knee, signal, knee + (1.0 - knee) * excess / (1.0 - knee + excess))

    def _apply_core(
        self,
        images: Tensor,
        *,
        num_clips: int,
        frames_per_clip: int,
        allow_motion_blur: bool,
        already_ir: Tensor | None,
        force: bool,
    ) -> Tensor:
        """Apply ISP where camera parameters are sampled per clip (`num_clips`) and noise varies per frame."""
        total_b, c, h, w = images.shape
        if total_b == 0 or (not self.training and not force):
            self.last_sensor_families = []
            return images.clamp(0.0, 1.0).clone()

        fam_pool = SENSOR_FAMILIES_BY_DOMAIN_SPLIT[self.config.domain_split]
        self.last_sensor_families = [self._py_rng.choice(fam_pool) for _ in range(num_clips)]

        orig_dtype = images.dtype
        device = images.device
        w_dtype = torch.float16 if images.is_cuda else torch.float32
        # Always `.clone()` so CPU float32 caller tensors are NEVER mutated or aliased!
        x = images.to(dtype=w_dtype).clone().clamp_(0.0, 1.0)

        # Sample clip-level augmentation mask (shuffled/independent per clip, never position-biased!)
        if num_clips <= 2:
            aug_clip_mask = torch.ones((num_clips,), dtype=torch.bool)
        else:
            aug_clip_mask = (torch.rand((num_clips,), generator=self._cpu_gen) < self.config.p_augment)
            if not aug_clip_mask.any():
                aug_clip_mask[self._py_rng.randrange(num_clips)] = True

        x_bk = x.view(num_clips, frames_per_clip, c, h, w)
        aug_idx = torch.nonzero(aug_clip_mask, as_tuple=False).squeeze(1).to(device=device)
        ba = int(aug_idx.numel())
        work_bk = x_bk.index_select(0, aug_idx)  # (ba, K, 3, H, W)
        work = work_bk.reshape(ba * frames_per_clip, c, h, w)

        def _expand_clip_param(p_clip: Tensor) -> Tensor:
            _, cp, hp, wp = p_clip.shape
            return p_clip.unsqueeze(1).expand(ba, frames_per_clip, cp, hp, wp).reshape(ba * frames_per_clip, cp, hp, wp)

        # Sample Night-IR clip flags FIRST so LCA, WB, haze, WDR, and DCT all preserve strict IR monochrome!
        is_ir_c = self._rand_bool((ba, 1, 1, 1), self.config.p_night_ir, device)
        if already_ir is not None:
            air_sel = already_ir.to(device=device, dtype=torch.bool).index_select(0, aug_idx).view(ba, 1, 1, 1)
            eff_ir_c = is_ir_c | air_sel
        else:
            eff_ir_c = is_ir_c
        eff_ir = _expand_clip_param(eff_ir_c.to(w_dtype)) > 0.5

        # Split-dependent sensor hyperparameter envelopes (disjoint optical/ISP regimes across splits)
        is_train_split = self.config.domain_split == "train"
        if self.config.domain_split == "val_holdout":
            lca_mag_lo, lca_mag_hi = 0.0055, 0.0135
            wdr_amp_lo, wdr_amp_hi = 0.55, 1.45
            dct_step_lo, dct_step_hi = 0.018, 0.048
        elif self.config.domain_split == "cal_holdout":
            lca_mag_lo, lca_mag_hi = 0.0035, 0.0095
            wdr_amp_lo, wdr_amp_hi = 0.85, 1.85
            dct_step_lo, dct_step_hi = 0.010, 0.032
        else:
            lca_mag_lo, lca_mag_hi = 0.0020, 0.0085
            wdr_amp_lo, wdr_amp_hi = 0.35, 1.15
            dct_step_lo, dct_step_hi = 0.012, 0.095

        # 1. Camera-side Low-Resolution Capture & Substream Simulation (persistent across clip frames!)
        low_mask_cpu = torch.rand((ba,), generator=self._cpu_gen) < float(self.config.p_lowres_capture or 0.52)
        low_clip_ids = torch.nonzero(low_mask_cpu, as_tuple=False).squeeze(1).tolist()
        if low_clip_ids:
            work_5d = work.view(ba, frames_per_clip, c, h, w)
            if is_train_split:
                for cid in low_clip_ids:
                    idx_t = torch.tensor([cid], device=device, dtype=torch.long)
                    flat_grp = work_5d.index_select(0, idx_t).reshape(frames_per_clip, c, h, w)
                    ir_grp = eff_ir_c.index_select(0, idx_t).view(1)
                    deg_grp = self._degrade_lowres_slice(
                        flat_grp,
                        h,
                        w,
                        frames_per_clip=frames_per_clip,
                        eff_ir_clip=ir_grp,
                    ).view(1, frames_per_clip, c, h, w)
                    work_5d.index_copy_(0, idx_t, deg_grp)
            else:
                half = max(1, len(low_clip_ids) // 2)
                for grp in (low_clip_ids[:half], low_clip_ids[half:]):
                    if not grp:
                        continue
                    idx_t = torch.tensor(grp, device=device, dtype=torch.long)
                    flat_grp = work_5d.index_select(0, idx_t).reshape(len(grp) * frames_per_clip, c, h, w)
                    deg_grp = self._degrade_lowres_slice(flat_grp, h, w).view(len(grp), frames_per_clip, c, h, w)
                    work_5d.index_copy_(0, idx_t, deg_grp)
            work = work_5d.reshape(ba * frames_per_clip, c, h, w)

        # 2. Lateral Chromatic Aberration (radial R/B dispersion around optical center; disabled on narrowband IR!)
        if self.config.p_chromatic_aberration > 0.0 and h >= 16 and w >= 16:
            lca_mask_cpu = torch.rand((ba,), generator=self._cpu_gen) < self.config.p_chromatic_aberration
            eff_ir_cpu = eff_ir_c.view(ba).to(device="cpu")
            lca_ids = torch.nonzero(lca_mask_cpu & (~eff_ir_cpu), as_tuple=False).squeeze(1).tolist()
            if lca_ids:
                work_5d = work.view(ba, frames_per_clip, c, h, w)
                idx_t = torch.tensor(lca_ids, device=device, dtype=torch.long)
                n_lca = len(lca_ids)
                sgn_lca = torch.where(
                    self._rand_bool((n_lca, 1), 0.5, device),
                    torch.ones((n_lca, 1), device=device, dtype=torch.float32),
                    -torch.ones((n_lca, 1), device=device, dtype=torch.float32),
                )
                mag_r = self._rand_uniform((n_lca, 1), lca_mag_lo, lca_mag_hi, device, torch.float32) * sgn_lca
                mag_b = -mag_r * self._rand_uniform((n_lca, 1), 0.75, 1.25, device, torch.float32)
                kr_f = mag_r.expand(n_lca, frames_per_clip).reshape(-1)
                kb_f = mag_b.expand(n_lca, frames_per_clip).reshape(-1)
                sub_lca = work_5d.index_select(0, idx_t).reshape(n_lca * frames_per_clip, c, h, w)
                out_lca = self._apply_lateral_chromatic_aberration(sub_lca, kr_f, kb_f)
                work_5d.index_copy_(0, idx_t, out_lca.view(n_lca, frames_per_clip, c, h, w))
                work = work_5d.reshape(ba * frames_per_clip, c, h, w)

        # 3. Optical Defocus (+ directional shutter motion blur k=3..11 px with per-frame sharpness variation when allow_motion_blur=True)
        blur_mask_cpu = torch.rand((ba,), generator=self._cpu_gen) < self.config.p_blur
        blur_clip_ids = torch.nonzero(blur_mask_cpu, as_tuple=False).squeeze(1).tolist()
        if blur_clip_ids and h >= 8 and w >= 8:
            if is_train_split and h > 540 and w > 960:
                bank11 = self.psf_bank_11x11_fp16 if w_dtype == torch.float16 else self.psf_bank_11x11_fp32
                max_k11 = bank11.shape[0] if allow_motion_blur else self.NUM_DEFOCUS_KERNELS
                work_5d = work.view(ba, frames_per_clip, c, h, w)
                for cid in blur_clip_ids:
                    if allow_motion_blur and self._py_rng.random() < 0.72:
                        k_idx = self._py_rng.randrange(self.NUM_DEFOCUS_KERNELS, max_k11)
                    else:
                        k_idx = self._py_rng.randrange(max_k11)
                    k_w = bank11[k_idx]
                    sub_b = work_5d[cid]  # (K, 3, H, W)
                    padded = F.pad(sub_b, (5, 5, 5, 5), mode="replicate")
                    conv_b = F.conv2d(padded, k_w, groups=3)
                    if allow_motion_blur and frames_per_clip >= 2:
                        # Per-frame shutter sharpness variation (lucky sharper frames vs full motion blur within clip)
                        w_f = self._rand_uniform((frames_per_clip, 1, 1, 1), 0.14, 1.00, device, w_dtype)
                        if self._py_rng.random() < 0.35:
                            sharp_slot = self._py_rng.randrange(frames_per_clip)
                            w_f[sharp_slot] = w_f[sharp_slot] * 0.22
                        work_5d[cid] = (sub_b * (1.0 - w_f) + conv_b * w_f).clamp_(0.0, 1.0)
                    else:
                        work_5d[cid] = conv_b.clamp_(0.0, 1.0)
                work = work_5d.reshape(ba * frames_per_clip, c, h, w)
            else:
                bank = self.psf_bank_fp16 if w_dtype == torch.float16 else self.psf_bank_fp32
                max_k = bank.shape[0] if allow_motion_blur else self.NUM_DEFOCUS_KERNELS
                k_w = bank[self._py_rng.randrange(max_k)]
                work_5d = work.view(ba, frames_per_clip, c, h, w)
                idx_t = torch.tensor(blur_clip_ids, device=device, dtype=torch.long)
                sub_b = work_5d.index_select(0, idx_t).reshape(len(blur_clip_ids) * frames_per_clip, c, h, w)
                padded = F.pad(sub_b, (3, 3, 3, 3), mode="replicate")
                blurred = F.conv2d(padded, k_w, groups=3).view(len(blur_clip_ids), frames_per_clip, c, h, w)
                work_5d.index_copy_(0, idx_t, blurred)
                work = work_5d.reshape(ba * frames_per_clip, c, h, w)

        rgb2y = self.rgb_to_y.to(dtype=w_dtype)

        # Optical scattering precedes photon counting, tone processing and coding.
        if self.config.p_glare_blob > 0.0 or self.config.p_sensor_noise > 0.0:
            linear = torch.where(work.float() <= 0.04045, work.float() / 12.92,
                                 ((work.float() + 0.055) / 1.055).pow(2.4))
            if self.config.p_glare_blob > 0.0:
                glare = self._rand_prob((ba, 1, 1, 1), self.config.p_glare_blob, device, torch.float32)
                linear = self._apply_lens_bloom(linear, _expand_clip_param(glare))
            if self.config.p_sensor_noise > 0.0:
                noise = self._rand_prob((ba, 1, 1, 1), self.config.p_sensor_noise, device, torch.float32)
                sigma_ir = self._rand_uniform((ba, 1, 1, 1), .018, .042, device, torch.float32)
                sigma_rgb = self._rand_uniform((ba, 1, 1, 1), .004, .022, device, torch.float32)
                sigma = _expand_clip_param(torch.where(eff_ir_c, sigma_ir, sigma_rgb))
                # Narrowband IR has one electron/read-noise realization per pixel.
                linear = torch.where(eff_ir, (linear * rgb2y.float()).sum(1, keepdim=True).expand_as(linear), linear)
                noisy = self._apply_raw_sensor_noise(linear, sigma.square() * 3., sigma * .55, eff_ir)
                linear = torch.where(_expand_clip_param(noise) > 0, noisy, linear)
            encoded = torch.where(linear <= .0031308, 12.92 * linear,
                                  1.055 * linear.clamp_min(1e-8).pow(1 / 2.4) - .055)
            work = self._highlight_shoulder(encoded).clamp_min(0).to(w_dtype)

        low2 = F.avg_pool2d(work, kernel_size=2, stride=2, ceil_mode=True)
        low4 = F.avg_pool2d(low2, kernel_size=2, stride=2, ceil_mode=True)

        # 4. WDR tone mapping amplifies the existing photon/read realization in shadows.
        if self.config.p_wdr_shadow_boost > 0.0:
            do_wdr_c = self._rand_prob((ba, 1, 1, 1), self.config.p_wdr_shadow_boost, device, w_dtype)
            wdr_amp_c = self._rand_uniform((ba, 1, 1, 1), wdr_amp_lo, wdr_amp_hi, device, w_dtype) * do_wdr_c
            if bool((wdr_amp_c > 1e-4).any().item()):
                lum_low4 = (low4 * rgb2y).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
                lum_env = F.interpolate(lum_low4, size=(h, w), mode="bilinear", align_corners=False)
                wdr_amp = _expand_clip_param(wdr_amp_c)
                shadow_w = (1.0 - lum_env).clamp(0.0, 1.0).square()
                local_gain = 1.0 + wdr_amp * shadow_w
                # Rational local tone curve `x * G / (1 + (G - 1) * x)` -> maps 0->0, 1->1, boosts shadows cleanly!
                work = (work * local_gain / (1.0 + (local_gain - 1.0) * work).clamp_min(1e-4)).clamp_(0.0, 1.0)

        # 5. Cheap CCTV ISP Unsharp-Mask Oversharpening Ringing
        if self.config.p_oversharpen > 0.0:
            blur_ref = F.interpolate(low2, size=(h, w), mode="bilinear", align_corners=False)
            do_sharp_c = self._rand_prob((ba, 1, 1, 1), self.config.p_oversharpen, device, w_dtype)
            amount_c = self._rand_uniform((ba, 1, 1, 1), 0.35, 1.40, device, w_dtype) * do_sharp_c
            work.addcmul_(work - blur_ref, _expand_clip_param(amount_c)).clamp_(0.0, 1.0)

        # 6. Chroma 4:2:0 Subsampling & Orthonormal 8x8 2D DCT Block Quantization Perturbation (+ P-frame ghosting)
        c420_mask_cpu = torch.rand((ba,), generator=self._cpu_gen) < self.config.p_chroma_sub
        c420_ids = torch.nonzero(c420_mask_cpu, as_tuple=False).squeeze(1).tolist()
        if c420_ids and h >= 8 and w >= 8:
            work_5d = work.view(ba, frames_per_clip, c, h, w)
            idx_t = torch.tensor(c420_ids, device=device, dtype=torch.long)
            sub_c = work_5d.index_select(0, idx_t).reshape(len(c420_ids) * frames_per_clip, c, h, w)
            if is_train_split:
                ir_420_f = eff_ir_c.index_select(0, idx_t).view(len(c420_ids), 1).expand(len(c420_ids), frames_per_clip).reshape(-1)
                out_c = self._apply_ycbcr_420(sub_c, ir_420_f).view(len(c420_ids), frames_per_clip, c, h, w)
            else:
                low4_5d = low4.view(ba, frames_per_clip, c, low4.shape[2], low4.shape[3])
                sub_l4 = low4_5d.index_select(0, idx_t).reshape(len(c420_ids) * frames_per_clip, c, low4.shape[2], low4.shape[3])
                y_full = (sub_c * rgb2y).sum(dim=1, keepdim=True)
                chroma_low = sub_l4 - (sub_l4 * rgb2y).sum(dim=1, keepdim=True)
                chroma_up = F.interpolate(chroma_low, size=(h, w), mode="bilinear", align_corners=False)
                out_c = (y_full + chroma_up).clamp_(0.0, 1.0).view(len(c420_ids), frames_per_clip, c, h, w)
            work_5d.index_copy_(0, idx_t, out_c)
            work = work_5d.reshape(ba * frames_per_clip, c, h, w)

        blk_mask_cpu = torch.rand((ba,), generator=self._cpu_gen) < self.config.p_codec_block
        blk_ids = torch.nonzero(blk_mask_cpu, as_tuple=False).squeeze(1).tolist()
        if blk_ids and h >= 8 and w >= 8:
            work_5d = work.view(ba, frames_per_clip, c, h, w)
            idx_t = torch.tensor(blk_ids, device=device, dtype=torch.long)
            n_blk = len(blk_ids)
            q_step_c = self._rand_uniform((n_blk, 1), dct_step_lo, dct_step_hi, device, torch.float32)
            if is_train_split and frames_per_clip >= 2 and self.config.p_pframe_ghost > 0.0:
                do_pf_b = self._rand_prob((n_blk,), self.config.p_pframe_ghost, device, torch.float32)
                g_alpha_b = self._rand_uniform((n_blk,), 0.06, 0.22, device, torch.float32) * do_pf_b
                ir_blk_c = eff_ir_c.index_select(0, idx_t).view(n_blk)
                sub_q5 = work_5d.index_select(0, idx_t)
                out_q5 = self._apply_h264_dct8x8_and_pframe(sub_q5, q_step_c.view(n_blk), g_alpha_b, ir_blk_c)
                work_5d.index_copy_(0, idx_t, out_q5)
            else:
                q_step_f = q_step_c.expand(n_blk, frames_per_clip).reshape(-1)
                ir_blk_f = eff_ir_c.index_select(0, idx_t).view(n_blk, 1).expand(n_blk, frames_per_clip).reshape(-1)
                sub_q = work_5d.index_select(0, idx_t).reshape(n_blk * frames_per_clip, c, h, w)
                out_q = self._apply_dct8x8_quantization(sub_q, q_step_f, ir_blk_f)
                work_5d.index_copy_(0, idx_t, out_q.view(n_blk, frames_per_clip, c, h, w))
            work = work_5d.reshape(ba * frames_per_clip, c, h, w)

        # 7. Persistent Camera Photometric Calibration (exposure, WB cast, gamma, contrast; WB neutral on IR!)
        if self.config.p_photometric > 0.0:
            do_ph = self._rand_bool((ba, 1, 1, 1), self.config.p_photometric, device)
            gain_c = torch.where(do_ph, self._rand_uniform((ba, 1, 1, 1), 0.30, 1.65, device, w_dtype), torch.ones((ba, 1, 1, 1), device=device, dtype=w_dtype))
            cct_shift = self._rand_uniform((ba, 1, 1, 1), -.11, .11, device, torch.float32)
            gm_tint = self._rand_uniform((ba, 1, 1, 1), -.035, .035, device, torch.float32)
            # First-order warm/cool tangent to the Planckian locus, with small
            # orthogonal tint. Normalizing Y decouples WB from auto-exposure.
            wb_rnd = torch.cat((1.0 + cct_shift, 1.0 + gm_tint, 1.0 - cct_shift), dim=1)
            wb_rnd = (wb_rnd / (wb_rnd * self.rgb_to_y.float()).sum(1, keepdim=True)).to(w_dtype)
            wb_c = torch.where(do_ph & (~eff_ir_c), wb_rnd, torch.ones((ba, 3, 1, 1), device=device, dtype=w_dtype))
            gamma_c = torch.where(do_ph, self._rand_uniform((ba, 1, 1, 1), 0.60, 1.85, device, w_dtype), torch.ones((ba, 1, 1, 1), device=device, dtype=w_dtype))
            contrast_c = torch.where(do_ph, self._rand_uniform((ba, 1, 1, 1), 0.74, 1.34, device, w_dtype), torch.ones((ba, 1, 1, 1), device=device, dtype=w_dtype))
            # Meter the entire clip after WDR/scattering; one controller setting
            # for all K frames avoids brightness pumping or any geometric change.
            y_mean = (work.float() * self.rgb_to_y.float()).sum(1).reshape(ba, frames_per_clip, h, w).mean((1, 2, 3)).view(ba, 1, 1, 1)
            bright = ((y_mean - .40) / .15).clamp(0.0, 1.0)
            meter_gain = torch.minimum(torch.ones_like(y_mean), .55 / y_mean.clamp_min(.01))
            gain_c = (gain_c.float() * torch.where(do_ph, meter_gain, torch.ones_like(meter_gain))).to(w_dtype)
            gamma_c = torch.maximum(gamma_c, (1.0 - .40 * (1.0 - bright)).to(w_dtype))
            gain = _expand_clip_param(gain_c)
            balanced = work * _expand_clip_param(wb_c)
            work = balanced * gain / (1.0 + (gain - 1.0).clamp_min(0.0) * balanced)
            work = work.clamp_min(1e-4).pow(_expand_clip_param(gamma_c))
            work = (work - .5) * _expand_clip_param(contrast_c) + .5
            work = self._highlight_shoulder(work.clamp_min(0.0))

        # 7B. Train-split Highway/Mountain Tunnel Sodium Cast, Backlit Shadow Crush, & Wet-Road Specular Streaks
        if is_train_split:
            vy_norm = torch.linspace(0.0, 1.0, h, device=device, dtype=w_dtype).view(1, 1, h, 1)
            # (a) Tunnel low/high-pressure sodium vapor amber-orange spectral cast (~2100K, disabled on IR)
            if self.config.p_tunnel_sodium > 0.0:
                do_sod_c = self._rand_bool((ba, 1, 1, 1), self.config.p_tunnel_sodium, device) & (~eff_ir_c)
                if bool(do_sod_c.any().item()):
                    sod_mix_c = torch.where(
                        do_sod_c,
                        self._rand_uniform((ba, 1, 1, 1), 0.42, 0.80, device, w_dtype),
                        torch.zeros((ba, 1, 1, 1), device=device, dtype=w_dtype),
                    )
                    sod_mix = _expand_clip_param(sod_mix_c)
                    y_sod = (work * rgb2y).sum(dim=1, keepdim=True)
                    sod_tint = torch.tensor([1.36, 0.95, 0.32], device=device, dtype=w_dtype).view(1, 3, 1, 1)
                    ceil_env = 1.08 - 0.16 * (vy_norm - 0.25).abs()
                    sod_rgb = self._highlight_shoulder((y_sod * sod_tint * ceil_env).clamp_min(0.0))
                    work = (work * (1.0 - sod_mix) + sod_rgb * sod_mix).clamp_(0.0, 1.0)

            # (b) Backlit S-curve shadow crush + upper-horizon sky/sun veiling glare
            if self.config.p_backlit_shadow_crush > 0.0:
                do_bl_c = self._rand_prob((ba, 1, 1, 1), self.config.p_backlit_shadow_crush, device, w_dtype)
                if bool((do_bl_c > 0.5).any().item()):
                    crush_pow_c = 1.0 + do_bl_c * self._rand_uniform((ba, 1, 1, 1), 0.55, 1.35, device, w_dtype)
                    veil_str_c = do_bl_c * self._rand_uniform((ba, 1, 1, 1), 0.08, 0.26, device, w_dtype)
                    crush_pow = _expand_clip_param(crush_pow_c)
                    veil_str = _expand_clip_param(veil_str_c)
                    sky_veil = (1.0 - vy_norm).clamp(0.0, 1.0).square() * veil_str
                    crushed = work.clamp_min(1e-4).pow(crush_pow)
                    # S-curve blend crushing dark shadows while lifting upper-frame backlit glare
                    s_w = (1.0 - work).clamp(0.0, 1.0)
                    work = (crushed * s_w + work * (1.0 - s_w) + sky_veil).clamp_(0.0, 1.0)

            # (c) Wet-road vertical specular tail-lamp & headlight reflection columns on lower pavement
            if self.config.p_wet_specular_streak > 0.0 and h >= 24 and w >= 24:
                do_wet_c = self._rand_prob((ba, 1, 1, 1), self.config.p_wet_specular_streak, device, w_dtype)
                if bool((do_wet_c > 0.5).any().item()):
                    wet_str = _expand_clip_param(
                        do_wet_c * self._rand_uniform((ba, 1, 1, 1), 0.24, 0.58, device, w_dtype)
                    )
                    wf = work.float()
                    red_exc = (wf[:, 0:1] - 0.52 * (wf[:, 1:2] + wf[:, 2:3])).clamp_min(0.0)
                    bright_src = (wf - 0.68).clamp_min(0.0)
                    lamp_src = bright_src + torch.cat(
                        [red_exc * 1.35, red_exc * 0.18, red_exc * 0.12], dim=1
                    ) * (~eff_ir).to(torch.float32)
                    sh4, sw4 = max(6, h // 4), max(6, w // 4)
                    src_q = F.adaptive_avg_pool2d(lamp_src, (sh4, sw4))
                    # Downward causal vertical reflection smear onto wet road below lamps (zero upward bleed!)
                    streak_q = torch.zeros_like(src_q)
                    for dy_s, decay_s in ((1, 0.36), (2, 0.27), (3, 0.19), (4, 0.12), (5, 0.06)):
                        if dy_s < sh4:
                            streak_q[:, :, dy_s:, :] = streak_q[:, :, dy_s:, :] + decay_s * src_q[:, :, :-dy_s, :]
                    streak_full = F.interpolate(streak_q, size=(h, w), mode="bilinear", align_corners=False).to(w_dtype)
                    road_gate = ((vy_norm - 0.44) / 0.22).clamp(0.0, 1.0)
                    ripple = 0.72 + 0.28 * torch.sin(vy_norm * 46.0)
                    work = self._highlight_shoulder(work + streak_full * wet_str * road_gate * ripple).clamp_(0.0, 1.0)

        # 8. Coarse-grid fused Fog/Haze + Specular Glare Blobs + Lens Water Droplets (persistent per clip!)
        gh, gw = max(8, h // 8), max(10, w // 8)
        gy = torch.linspace(0.0, 1.0, gh, device=device, dtype=w_dtype).view(1, 1, gh, 1)
        gx = torch.linspace(0.0, 1.0, gw, device=device, dtype=w_dtype).view(1, 1, 1, gw)
        aspect = float(w) / float(max(1, h))

        do_fog = self._rand_prob((ba, 1, 1, 1), self.config.p_fog_haze, device, w_dtype)
        fog_d = self._rand_uniform((ba, 1, 1, 1), 0.24, 0.62, device, w_dtype) * do_fog
        trans_low = (1.0 - fog_d * (1.0 - 0.45 * gy)).clamp_(0.25, 1.0)
        air_rgb = self._rand_uniform((ba, 3, 1, 1), 0.74, 0.94, device, w_dtype)
        air_mono = air_rgb.mean(dim=1, keepdim=True).expand(ba, 3, 1, 1)
        air_col = torch.where(eff_ir_c, air_mono, air_rgb)
        add_low_c = air_col * (1.0 - trans_low).expand(ba, 3, gh, gw)
        mul_low_c = trans_low.expand(ba, 1, gh, gw).clone()

        tex_coarse = F.interpolate(
            self._rand_uniform((ba, 1, max(3, gh // 4), max(4, gw // 4)), 0.78, 1.22, device, w_dtype),
            size=(gh, gw),
            mode="bilinear",
            align_corners=False,
        )
        tex_fine = self._rand_uniform((ba, 1, gh, gw), 0.90, 1.10, device, w_dtype)
        # Illumination variation is an exposure field, not an additive energy
        # source: roll off positive gain before the convex fog/droplet mixture.
        texture_gain = F.interpolate(_expand_clip_param(tex_coarse * tex_fine), size=(h, w), mode="bilinear", align_corners=False)
        work = work * texture_gain / (1.0 + (texture_gain - 1.0).clamp_min(0.0) * work)

        mul_low = _expand_clip_param(mul_low_c)
        add_low = _expand_clip_param(add_low_c)

        if self.config.p_lens_droplets > 0.0:
            do_drop = self._rand_prob((ba, 1, 1, 1), self.config.p_lens_droplets, device, w_dtype)
            cx_d = self._rand_uniform((ba, 3, 1, 1), 0.10, 0.90, device, w_dtype)
            cy_d = self._rand_uniform((ba, 3, 1, 1), 0.10, 0.90, device, w_dtype)
            rad_d = self._rand_uniform((ba, 3, 1, 1), 0.04, 0.10, device, w_dtype)
            dist_d = torch.sqrt(((gx - cx_d) * aspect).square() + (gy - cy_d).square() + 1e-4)
            drop_m_c = torch.sigmoid((rad_d - dist_d) * 50.0).amax(dim=1, keepdim=True) * do_drop
            drop_m = _expand_clip_param(drop_m_c)
            defoc_low = F.interpolate(low4, size=(gh, gw), mode="bilinear", align_corners=False)
            refract_low = (defoc_low * 1.15 + 0.10).clamp_(0.0, 1.0)
            mul_low = mul_low * (1.0 - 0.80 * drop_m)
            add_low = add_low + refract_low * (0.80 * drop_m)

        field_low = torch.cat([mul_low, add_low], dim=1)
        field_full = F.interpolate(field_low, size=(h, w), mode="bilinear", align_corners=False)
        work.mul_(field_full[:, :1]).add_(field_full[:, 1:]).clamp_(0.0, 1.0)

        # 9. Monochrome Night-IR projection after ISP calibration and nuisance fields.
        if bool(eff_ir.any().item()):
            ir_y = (work * rgb2y).sum(dim=1, keepdim=True).expand(ba * frames_per_clip, c, h, w)
            work = torch.where(eff_ir, ir_y, work)

        # Copy the single narrowband sensor channel after transfer functions too:
        # vectorized pow kernels can differ by an ULP at channel/tail boundaries.
        work = torch.where(eff_ir, work[:, :1].expand_as(work), work)

        x_bk.index_copy_(0, aug_idx, work.view(ba, frames_per_clip, c, h, w))
        return x_bk.reshape(total_b, c, h, w).clamp_(0.0, 1.0).to(dtype=orig_dtype)

    @torch.no_grad()
    def forward(self, images: Tensor, force: bool = False) -> Tensor:
        """Apply camera-ISP & sensor degradations on `(B, 3, H, W)` single frames."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W) tensor, got {tuple(images.shape)}")
        return self._apply_core(
            images,
            num_clips=images.shape[0],
            frames_per_clip=1,
            allow_motion_blur=True,
            already_ir=None,
            force=force,
        )

    @torch.no_grad()
    def forward_clips(
        self,
        clips: Tensor,
        *,
        allow_motion_blur: bool = False,
        already_ir: list[bool] | Tensor | None = None,
        force: bool = False,
    ) -> Tensor:
        """Apply clip-consistent camera ISP across `(B, K, 3, H, W)` or `(K, 3, H, W)` video clips."""
        squeeze_b = False
        if clips.ndim == 4 and clips.shape[1] == 3:
            clips = clips.unsqueeze(0)
            squeeze_b = True
        elif clips.ndim != 5 or clips.shape[2] != 3:
            raise ValueError(f"Expected (B, K, 3, H, W) or (K, 3, H, W) tensor, got {tuple(clips.shape)}")

        b, k, c, h, w = clips.shape
        if b == 0 or k == 0:
            return clips.squeeze(0) if squeeze_b else clips

        air_t: Tensor | None = None
        if already_ir is not None:
            if isinstance(already_ir, Tensor):
                air_t = already_ir.reshape(b).to(dtype=torch.bool)
            else:
                air_t = torch.tensor([bool(v) for v in already_ir], dtype=torch.bool)

        flat = clips.reshape(b * k, c, h, w)
        aug_flat = self._apply_core(
            flat,
            num_clips=b,
            frames_per_clip=k,
            allow_motion_blur=allow_motion_blur,
            already_ir=air_t,
            force=force,
        )
        out = aug_flat.reshape(b, k, c, h, w)
        return out.squeeze(0) if squeeze_b else out


__all__ = [
    "VALID_ISP_DOMAIN_SPLITS",
    "LOWRES_SCALES",
    "SENSOR_FAMILIES_BY_DOMAIN_SPLIT",
    "GPUISPConfig",
    "GPUCameraISPAugmenter",
]
