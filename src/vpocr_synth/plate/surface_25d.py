"""Extended-canonical `(192, 384)` 2.5D physical license plate surface builder.

Constructs batched or single-plate 2.5D material packets (`Albedo_RGB`, `NIR_Albedo`,
`Height_mm`, `Retro_Mask`, `Roughness`, `Metallic`, `Emissive_RGB`, and per-slot
canonical stroke masks/boxes) on CPU or CUDA directly from `GlyphSDFAtlas` and
official MLIT millimeter layouts (`PlateGeometry`).

Canvas coordinate system:
  - Extended canvas: `(H_ext=192, W_ext=384)`
  - Central physical plate (`330x165 mm` Medium / `440x220 mm` Large):
    rows `[16:176]` (`160 px`), cols `[32:352]` (`320 px`), exact `2.0000:1` aspect.
  - Outer `16px / 32px` (`10%`) surround: recessed bumper pocket (`-8.0..-2.5 mm`),
    grille texture, and optional upper bumper overhang hood (`+8..+18 mm`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from vpocr_ocr.vocab import NUM_SLOTS, PlateVocab
from vpocr_synth.plate.geometry import PlateGeometry
from vpocr_synth.plate.glyph_dies import (
    GlyphSDFAtlas,
    get_default_glyph_atlas,
    load_permissive_truetype,
)
from vpocr_synth.plate.models import PlateConfig, PlateSize, PlateText, PlateType

EXT_H: int = 192
EXT_W: int = 384
PLATE_H: int = 160
PLATE_W: int = 320
PAD_Y: int = 16
PAD_X: int = 32

# Pixel-center coordinates of the 4 physical plate corners [TL, TR, BR, BL] in (192, 384)
EXT_PLATE_CORNERS_XY: tuple[tuple[float, float], ...] = (
    (float(PAD_X) - 0.5, float(PAD_Y) - 0.5),                  # TL: (31.5, 15.5)
    (float(PAD_X + PLATE_W) - 0.5, float(PAD_Y) - 0.5),        # TR: (351.5, 15.5)
    (float(PAD_X + PLATE_W) - 0.5, float(PAD_Y + PLATE_H) - 0.5),  # BR: (351.5, 175.5)
    (float(PAD_X) - 0.5, float(PAD_Y + PLATE_H) - 0.5),        # BL: (31.5, 175.5)
)

_BUNDLED_FONTS_DIR = Path(__file__).resolve().parents[3] / "data" / "fonts"


class NonPlateStencilAtlas:
    """Cached `(72, 3, 96, 160)` atlas of Japanese/Latin text & adversarial non-plate hard negatives.

    Renders real Katakana/Kanji/Latin/digit roadside & vehicle-body stencils (`319` blue route shields,
    `スケルトン車` painted door text, `最大積載量 5000kg` decals, single-row `春日 2503` stencils,
    `TOLL GATE` / `HINO 500` badges, glossy white/pink blobs, yellow-black hazard stripes, blank bolted
    metal panels, headlights, and static clutter) exclusively from bundled SIL-OFL-1.1 fonts so
    `target_status == 1` (`UNREADABLE`) defeats every false-positive crop mode.
    """

    ROI_H: int = 96
    ROI_W: int = 160

    def __init__(self) -> None:
        self.templates_rgb: np.ndarray = self._build_templates()  # (72, 3, 96, 160) float32
        self._device_cache: dict[torch.device, torch.Tensor] = {}

    def to_torch(self, device: torch.device | str) -> torch.Tensor:
        dev = torch.device(device)
        cached = self._device_cache.get(dev)
        if cached is None:
            cached = torch.from_numpy(self.templates_rgb).to(dev, non_blocking=True)
            self._device_cache[dev] = cached
        return cached

    @torch.no_grad()
    def sample_crops(
        self,
        num_samples: int,
        out_hw: tuple[int, int],
        device: torch.device | str,
        gen: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample `(N, 3, H, W)` augmented text-bearing non-plate crops on `device`."""
        dev = torch.device(device)
        bank = self.to_torch(dev)
        k_tot = bank.shape[0]
        n = max(1, int(num_samples))
        if gen is not None and gen.device == dev:
            idx = torch.randint(0, k_tot, (n,), generator=gen, device=dev)
            r_u = torch.rand((n, 7), generator=gen, device=dev)
        else:
            idx = torch.randint(0, k_tot, (n,), generator=gen).to(dev)
            r_u = torch.rand((n, 7), generator=gen).to(dev)

        base = bank.index_select(0, idx)  # (N, 3, 96, 160)
        h_out, w_out = int(out_hw[0]), int(out_hw[1])

        # Mild affine scale/translation/shear so stencils appear at realistic Stage-2 crop offsets
        sx = (0.80 + 0.36 * r_u[:, 0]).unsqueeze(1)
        sy = (0.80 + 0.36 * r_u[:, 1]).unsqueeze(1)
        sh = (-0.14 + 0.28 * r_u[:, 2]).unsqueeze(1)
        tx = (-0.18 + 0.36 * r_u[:, 3]).unsqueeze(1)
        ty = (-0.18 + 0.36 * r_u[:, 4]).unsqueeze(1)
        zeros = torch.zeros_like(sx)
        row0 = torch.cat([sx, sh, tx], dim=1)
        row1 = torch.cat([zeros, sy, ty], dim=1)
        theta = torch.stack([row0, row1], dim=1)  # (N, 2, 3)
        grid = F.affine_grid(theta, size=[n, 3, h_out, w_out], align_corners=False)
        warped = F.grid_sample(base, grid, mode="bilinear", padding_mode="border", align_corners=False)

        gain = (0.55 + 0.65 * r_u[:, 5]).view(n, 1, 1, 1)
        is_ir = (r_u[:, 6] < 0.22).view(n, 1, 1, 1)
        lit = (warped * gain).clamp(0.0, 1.0)
        mono = (0.299 * lit[:, 0:1] + 0.587 * lit[:, 1:2] + 0.114 * lit[:, 2:3]).expand_as(lit)
        return torch.where(is_ir, mono, lit)

    @classmethod
    def _build_templates(cls) -> np.ndarray:
        h, w = cls.ROI_H, cls.ROI_W
        font_paths = [
            str(_BUNDLED_FONTS_DIR / "BIZUDPGothic-Bold.ttf"),
            str(_BUNDLED_FONTS_DIR / "MPLUSRounded1c-Bold.ttf"),
            str(_BUNDLED_FONTS_DIR / "ZenMaruGothic-Bold.ttf"),
        ]
        f_cache: dict[tuple[int, int], Any] = {}

        def gf(fi: int, sz: int):
            key = (fi % len(font_paths), sz)
            if key not in f_cache:
                f_cache[key] = load_permissive_truetype(font_paths[key[0]], size=sz, index=0)
            return f_cache[key]

        def draw_centered(draw: ImageDraw.ImageDraw, text: str, fnt: Any, cx: int, cy: int, fill: tuple[int, int, int]) -> None:
            bb = draw.textbbox((0, 0), text, font=fnt)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.text((cx - tw // 2 - bb[0], cy - th // 2 - bb[1]), text, fill=fill, font=fnt)

        templates: list[np.ndarray] = []

        # 1. 12x Blue MLIT Route Shield / Highway Number Signs ("319", "246", "16", "4", etc.)
        route_specs = [
            ("国道", "319"), ("国道", "246"), ("国道", "16"), ("国道", "4"),
            ("国道", "17"), ("国道", "254"), ("国道", "58"), ("国道", "1"),
            ("県道", "43"), ("都道", "318"), ("R", "319"), ("ROUTE", "129"),
        ]
        for idx_r, (hdr, num) in enumerate(route_specs):
            bg_v = 48 + (idx_r * 13) % 70
            img = Image.new("RGB", (w, h), (bg_v, bg_v + 6, bg_v + 12))
            dr = ImageDraw.Draw(img)
            # Rounded blue shield / rectangular highway panel
            pad_x = 18 if len(num) >= 3 else 26
            dr.rounded_rectangle([pad_x, 8, w - pad_x, h - 8], radius=18, fill=(14, 72, 176), outline=(244, 246, 250), width=4)
            draw_centered(dr, hdr, gf(idx_r, 15), w // 2, 24, (242, 245, 250))
            draw_centered(dr, num, gf(idx_r + 1, 42 if len(num) <= 3 else 34), w // 2, 58, (248, 250, 252))
            templates.append(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        # 2. 14x Painted Truck Door / Tailgate Katakana & Kanji Decals ("スケルトン車", "最大積載量", etc.)
        decal_specs = [
            ("スケルトン車", "", (238, 240, 242), (24, 32, 42)),
            ("スケルトン車", "", (28, 86, 52), (242, 244, 240)),
            ("最大積載量 5000kg", "", (232, 234, 236), (22, 24, 26)),
            ("最大積載量 3500kg", "", (36, 58, 108), (245, 245, 242)),
            ("最大積載量", "2000kg", (228, 186, 32), (20, 20, 20)),
            ("産業廃棄物収集運搬車", "", (240, 242, 240), (18, 62, 36)),
            ("産業廃棄物", "第 013-3190 号", (42, 46, 52), (240, 242, 244)),
            ("自家用", "", (235, 236, 238), (26, 28, 30)),
            ("安全第一", "制限速度厳守", (242, 244, 240), (22, 92, 46)),
            ("積載車", "架装番号 319", (196, 202, 208), (24, 26, 32)),
            ("限定車", "SPECIAL", (32, 34, 38), (236, 204, 68)),
            ("高圧ガス", "火気厳禁", (24, 26, 28), (238, 46, 38)),
            ("回送車", "OUT OF SERVICE", (38, 42, 48), (246, 172, 36)),
            ("作業中", "追突注意", (236, 198, 28), (22, 22, 22)),
        ]
        for idx_d, (line1, line2, bg_c, fg_c) in enumerate(decal_specs):
            img = Image.new("RGB", (w, h), bg_c)
            dr = ImageDraw.Draw(img)
            # Subtle truck body panel seam line
            dr.line([(0, 10), (w, 10)], fill=(max(0, bg_c[0] - 28), max(0, bg_c[1] - 28), max(0, bg_c[2] - 28)), width=2)
            if line2:
                sz1 = 24 if len(line1) <= 6 else 18
                draw_centered(dr, line1, gf(idx_d, sz1), w // 2, 34, fg_c)
                draw_centered(dr, line2, gf(idx_d + 1, 20), w // 2, 66, fg_c)
            else:
                sz1 = 28 if len(line1) <= 6 else (20 if len(line1) <= 9 else 14)
                draw_centered(dr, line1, gf(idx_d, sz1), w // 2, h // 2, fg_c)
            templates.append(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        # 3. 10x Single-Row Fleet / Gate Stencils ("春日 2503", "社番 1082", etc. — strictly 1-row!)
        stencil_specs = [
            ("春日 2503", (62, 74, 66), (238, 240, 236)),
            ("春日 3190", (226, 228, 230), (24, 28, 34)),
            ("社番 1082", (38, 64, 112), (244, 244, 240)),
            ("No. 319", (236, 236, 232), (26, 26, 28)),
            ("品川 048", (184, 190, 196), (22, 24, 28)),
            ("B-2503", (44, 48, 54), (242, 206, 48)),
            ("営 5802", (26, 78, 48), (242, 244, 242)),
            ("88-319", (212, 214, 218), (28, 30, 34)),
            ("東 409", (52, 56, 62), (238, 240, 242)),
            ("K-0319", (228, 184, 34), (24, 24, 24)),
        ]
        for idx_s, (txt_s, bg_c, fg_c) in enumerate(stencil_specs):
            img = Image.new("RGB", (w, h), bg_c)
            dr = ImageDraw.Draw(img)
            if idx_s % 2 == 0:
                dr.rectangle([8, 20, w - 8, h - 20], outline=fg_c, width=2)
            draw_centered(dr, txt_s, gf(idx_s, 30), w // 2, h // 2, fg_c)
            templates.append(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        # 4. 8x Speed Limit Rings & Commercial Door Phone Numbers ("50", "40", "TEL 03-...", etc.)
        sign_specs = [
            ("ring", "50"), ("ring", "40"), ("ring", "60"), ("ring", "30"),
            ("phone", "TEL 03-5912-3408"), ("phone", "0120-88-3190"),
            ("phone", "FAX 06-6319-2503"), ("phone", "駐車禁止 NO PARKING"),
        ]
        for idx_p, (kind, txt_p) in enumerate(sign_specs):
            if kind == "ring":
                img = Image.new("RGB", (w, h), (76, 82, 88))
                dr = ImageDraw.Draw(img)
                r_c = 38
                dr.ellipse([w // 2 - r_c, h // 2 - r_c, w // 2 + r_c, h // 2 + r_c], fill=(246, 246, 246), outline=(218, 28, 26), width=9)
                draw_centered(dr, txt_p, gf(idx_p, 36), w // 2, h // 2, (24, 32, 48))
            else:
                bg_c = (240, 242, 244) if idx_p % 2 == 0 else (34, 42, 56)
                fg_c = (22, 48, 118) if idx_p % 2 == 0 else (244, 244, 240)
                img = Image.new("RGB", (w, h), bg_c)
                dr = ImageDraw.Draw(img)
                dr.rectangle([6, 14, w - 6, h - 14], outline=(204, 36, 32) if "駐車" in txt_p else fg_c, width=3)
                draw_centered(dr, txt_p, gf(idx_p, 16), w // 2, h // 2, fg_c)
            templates.append(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        # 5. 4x Chrome Radiator Grilles & Mullion Window Clutter
        for idx_g in range(4):
            arr = np.zeros((3, h, w), dtype=np.float32)
            yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h, dtype=np.float32), np.linspace(0.0, 1.0, w, dtype=np.float32), indexing="ij")
            if idx_g < 2:
                slats = (np.sin(yy * math.pi * (10 + 6 * idx_g)) > 0.25).astype(np.float32)
                posts = (np.sin(xx * math.pi * (8 + 4 * idx_g)) > 0.72).astype(np.float32)
                lum = np.clip(0.10 + 0.62 * slats + 0.22 * posts, 0.0, 1.0)
                arr[:] = lum[None]
            else:
                # Dark tinted cab glass + vertical window mullions + specular sky bar
                mullion = (np.abs(np.sin(xx * math.pi * 5.0)) > 0.92).astype(np.float32)
                sky_ref = np.exp(-((yy - 0.34 + 0.12 * xx) ** 2) / 0.018).astype(np.float32)
                arr[0] = np.clip(0.12 + 0.55 * mullion + 0.28 * sky_ref, 0.0, 1.0)
                arr[1] = np.clip(0.15 + 0.56 * mullion + 0.36 * sky_ref, 0.0, 1.0)
                arr[2] = np.clip(0.20 + 0.58 * mullion + 0.48 * sky_ref, 0.0, 1.0)
            templates.append(arr)

        # 6. 8x Latin / Mixed Commercial Signage & Vehicle Emblems
        latin_specs = [
            ("TOLL GATE", "ETC ONLY", (18, 86, 48), (246, 248, 246)),
            ("EXIT 12B", "出口 500m", (20, 74, 152), (246, 248, 250)),
            ("DIESEL TURBO", "4WD", (32, 34, 38), (222, 226, 232)),
            ("HYBRID", "SYNERGY", (236, 240, 246), (26, 76, 168)),
            ("HINO 500", "", (44, 48, 54), (240, 242, 244)),
            ("ISUZU ELF", "", (232, 234, 238), (192, 28, 30)),
            ("CAUTION 24V", "立入禁止", (236, 192, 22), (20, 20, 20)),
            ("STOP", "止まれ", (196, 26, 24), (248, 248, 248)),
        ]
        for idx_l, (l1, l2, bg_c, fg_c) in enumerate(latin_specs):
            img = Image.new("RGB", (w, h), bg_c)
            dr = ImageDraw.Draw(img)
            dr.rectangle([4, 6, w - 4, h - 6], outline=fg_c, width=2)
            if l2:
                draw_centered(dr, l1, gf(idx_l, 22), w // 2, 34, fg_c)
                draw_centered(dr, l2, gf(idx_l + 1, 18), w // 2, 66, fg_c)
            else:
                draw_centered(dr, l1, gf(idx_l, 28), w // 2, h // 2, fg_c)
            templates.append(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        # 7. 6x Glossy White / Pink / Pastel Specular Blobs
        yy_b, xx_b = np.meshgrid(
            np.linspace(-1.0, 1.0, h, dtype=np.float32),
            np.linspace(-1.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        blob_colors = [
            ((0.96, 0.96, 0.97), (0.28, 0.30, 0.34), -0.15, -0.05),
            ((0.96, 0.62, 0.76), (0.22, 0.18, 0.24), 0.10, 0.05),
            ((0.98, 0.74, 0.84), (0.82, 0.84, 0.88), -0.20, 0.10),
            ((0.95, 0.92, 0.86), (0.16, 0.17, 0.20), 0.18, -0.10),
            ((0.92, 0.48, 0.66), (0.14, 0.12, 0.16), 0.0, 0.0),
            ((0.94, 0.95, 0.98), (0.60, 0.64, 0.70), 0.05, -0.15),
        ]
        for fg_c, bg_c, cx_b, cy_b in blob_colors:
            g_b = np.exp(-(((xx_b - cx_b) / 0.58) ** 2 + ((yy_b - cy_b) / 0.52) ** 2)).astype(np.float32)
            spec_b = (g_b ** 4) * 0.28
            arr = np.zeros((3, h, w), dtype=np.float32)
            for ci in range(3):
                arr[ci] = np.clip(bg_c[ci] * (1.0 - g_b) + fg_c[ci] * g_b + spec_b, 0.0, 1.0)
            templates.append(arr)

        # 8. 4x Yellow-Black & Red-White Diagonal Hazard Warning Stripes
        yy_h, xx_h = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 3.5, w, dtype=np.float32),
            indexing="ij",
        )
        for idx_h in range(4):
            phase_h = 0.25 * idx_h
            stripe = (((xx_h + (1.2 if idx_h % 2 == 0 else -1.2) * yy_h + phase_h) % 1.0) > 0.48).astype(np.float32)
            c1 = (0.94, 0.78, 0.08) if idx_h < 3 else (0.90, 0.14, 0.12)
            c2 = (0.09, 0.09, 0.10) if idx_h < 3 else (0.95, 0.95, 0.95)
            arr = np.zeros((3, h, w), dtype=np.float32)
            for ci in range(3):
                arr[ci] = stripe * c1[ci] + (1.0 - stripe) * c2[ci]
            templates.append(arr)

        # 9. 4x Pure Blank Brushed Metal / Empty Bolted Panels (Plate-shaped, zero characters!)
        blank_bgs = [
            (226, 228, 230),
            (236, 198, 38),
            (32, 92, 54),
            (152, 158, 166),
        ]
        for idx_m, bg_m in enumerate(blank_bgs):
            img = Image.new("RGB", (w, h), (36, 38, 42))
            dr = ImageDraw.Draw(img)
            dr.rounded_rectangle([14, 10, w - 14, h - 10], radius=6, fill=bg_m, outline=(60, 62, 66), width=3)
            # Two mounting bolts with zero text
            for bx_m in (42, w - 42):
                dr.ellipse([bx_m - 5, 22 - 5, bx_m + 5, 22 + 5], fill=(168, 172, 178), outline=(72, 74, 78), width=1)
            arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            # Brushed horizontal metallic grain
            rng_m = np.random.default_rng(700 + idx_m)
            grain = rng_m.normal(0.0, 0.025, size=(1, h, 1)).astype(np.float32)
            templates.append(np.clip(arr + grain, 0.0, 1.0))

        # 10. 2x Dual Headlight Glare & Extreme Static Noise
        lamp2 = (
            np.exp(-(((xx_b + 0.45) / 0.24) ** 2 + (yy_b / 0.32) ** 2))
            + np.exp(-(((xx_b - 0.45) / 0.24) ** 2 + (yy_b / 0.32) ** 2))
        ).astype(np.float32)
        arr_hl = np.zeros((3, h, w), dtype=np.float32)
        arr_hl[0] = np.clip(0.08 + 0.96 * lamp2, 0.0, 1.0)
        arr_hl[1] = np.clip(0.09 + 0.94 * lamp2, 0.0, 1.0)
        arr_hl[2] = np.clip(0.11 + 0.88 * lamp2, 0.0, 1.0)
        templates.append(arr_hl)

        rng_st = np.random.default_rng(799)
        templates.append(rng_st.uniform(0.0, 1.0, size=(3, h, w)).astype(np.float32))

        return np.stack(templates[:72], axis=0).astype(np.float32)


_DEFAULT_STENCIL_ATLAS: NonPlateStencilAtlas | None = None


def get_default_nonplate_stencil_atlas() -> NonPlateStencilAtlas:
    """Return process-cached `NonPlateStencilAtlas` singleton."""
    global _DEFAULT_STENCIL_ATLAS
    if _DEFAULT_STENCIL_ATLAS is None:
        _DEFAULT_STENCIL_ATLAS = NonPlateStencilAtlas()
    return _DEFAULT_STENCIL_ATLAS


@dataclass
class PlateSurface25DPacket:
    """Batched `(B, ...)` 2.5D canonical surface G-buffer & per-slot visibility masks."""

    albedo_rgb: torch.Tensor        # (B, 3, 192, 384) float32 in [0, 1]
    nir_albedo: torch.Tensor        # (B, 1, 192, 384) float32 in [0, 1] (850/940nm reflectance)
    height_mm: torch.Tensor         # (B, 1, 192, 384) float32 in [-10.0, +18.0] mm
    height_u8: torch.Tensor         # (B, 1, 192, 384) uint8 quantized [-10..+18] mm
    retro_mask: torch.Tensor        # (B, 1, 192, 384) float32 in [0, 1]
    roughness: torch.Tensor         # (B, 1, 192, 384) float32 in [0.05, 1.0]
    metallic: torch.Tensor          # (B, 1, 192, 384) float32 in [0, 1]
    emissive_rgb: torch.Tensor      # (B, 3, 192, 384) float32 >= 0 (字光式 stroke glow)
    stroke_mask: torch.Tensor       # (B, 1, 192, 384) float32 in [0, 1]
    slot_boxes_canon: torch.Tensor  # (B, 12, 4) float32 [x0, y0, x1, y1] in (160, 320)
    slot_masks_half: torch.Tensor   # (B, 12, 80, 160) float32 per-slot stroke masks on half-plate grid
    surface_occ_2d: torch.Tensor    # (B, 1, 80, 160) float32 2D surface occlusion (mud/frame/slash)
    target_slots: torch.Tensor      # (B, 12) int64
    target_plate_type: torch.Tensor # (B,) int64 (0..4)
    base_polarity: torch.Tensor     # (B,) float32 (+1.0 bright-bg, -1.0 dark-bg)
    is_illuminated: torch.Tensor    # (B,) bool
    is_temporary: torch.Tensor      # (B,) bool
    has_seal: torch.Tensor          # (B,) bool
    plate_side: torch.Tensor        # (B,) int64 (0=front, 1=rear)


class PlateSurface25DBuilder:
    """Vectorized PyTorch 2.5D plate packet builder (runs identically on CPU and CUDA)."""

    def __init__(
        self,
        vocab: PlateVocab | None = None,
        atlas: GlyphSDFAtlas | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.vocab = vocab or PlateVocab()
        self.atlas = atlas or get_default_glyph_atlas(self.vocab)
        self.device = torch.device(device)
        self._coord_cache: dict[torch.device, tuple[torch.Tensor, ...]] = {}

    def to(self, device: torch.device | str) -> PlateSurface25DBuilder:
        self.device = torch.device(device)
        return self

    def _get_grids(self, dev: torch.device) -> tuple[torch.Tensor, ...]:
        cached = self._coord_cache.get(dev)
        if cached is not None:
            return cached
        # Extended grid in normalized medium-plate millimeters: plate is [0..330] x [0..165] mm
        # Pixel center j in [0..383] -> x_mm = ((j + 0.5) - 32.0) * (330.0 / 320.0)
        y_px = torch.arange(EXT_H, device=dev, dtype=torch.float32) + 0.5
        x_px = torch.arange(EXT_W, device=dev, dtype=torch.float32) + 0.5
        y_mm_1d = (y_px - float(PAD_Y)) * (165.0 / float(PLATE_H))  # [-16.0, 181.0] mm
        x_mm_1d = (x_px - float(PAD_X)) * (330.0 / float(PLATE_W))  # [-32.5, 362.5] mm
        yy_mm, xx_mm = torch.meshgrid(y_mm_1d, x_mm_1d, indexing="ij")  # (192, 384)

        # Plate interior coordinates (160, 320)
        y_p_1d = (torch.arange(PLATE_H, device=dev, dtype=torch.float32) + 0.5)
        x_p_1d = (torch.arange(PLATE_W, device=dev, dtype=torch.float32) + 0.5)

        # Pre-cached material color constants on device (zero runtime H2D syncs!)
        c_yel = torch.tensor([0.97, 0.78, 0.10], device=dev).view(1, 3, 1, 1)
        c_grn = torch.tensor([0.09, 0.36, 0.20], device=dev).view(1, 3, 1, 1)
        c_el_grn = torch.tensor([0.14, 1.00, 0.48], device=dev).view(1, 3, 1, 1)
        c_el_amb = torch.tensor([0.38, 1.00, 0.30], device=dev).view(1, 3, 1, 1)
        c_red_ink = torch.tensor([0.86, 0.09, 0.09], device=dev).view(1, 3, 1, 1)
        c_bolt = torch.tensor([0.74, 0.76, 0.78], device=dev).view(1, 3, 1, 1)
        c_mud = torch.tensor([0.26, 0.21, 0.16], device=dev).view(1, 3, 1, 1)
        c_salt = torch.tensor([0.76, 0.78, 0.77], device=dev).view(1, 3, 1, 1)
        c_rust = torch.tensor([0.44, 0.21, 0.09], device=dev).view(1, 3, 1, 1)
        pat_cols = torch.tensor(
            [
                [0.96, 0.62, 0.74],  # flower
                [0.45, 0.74, 0.88],  # geom
                [0.92, 0.22, 0.24],  # expo red
                [0.18, 0.46, 0.90],  # expo blue
                [0.28, 0.62, 0.88],  # wave
                [0.42, 0.58, 0.82],  # mountain
            ],
            device=dev,
            dtype=torch.float32,
        ).view(6, 1, 3, 1, 1)

        # Precompute static spatial geometry masks on (192, 384) once per device!
        dx_p = torch.abs(xx_mm - 165.0) - (165.0 - 10.0)
        dy_p = torch.abs(yy_mm - 82.5) - (82.5 - 10.0)
        sd_plate_mm = (
            torch.hypot(torch.clamp_min(dx_p, 0.0), torch.clamp_min(dy_p, 0.0))
            + torch.clamp_max(torch.maximum(dx_p, dy_p), 0.0)
            - 10.0
        )[None, None, :, :]
        in_plate = torch.clamp(0.5 - sd_plate_mm * 1.2, 0.0, 1.0)
        grille_stripes = 0.85 + 0.15 * torch.sin(yy_mm[None, None, :, :] * (2.0 * math.pi / 9.0))
        hood_geom = torch.clamp((-1.5 - yy_mm[None, None, :, :]) / 2.5, 0.0, 1.0)
        in_ribbon = (
            torch.clamp((sd_plate_mm + 8.0) * 1.5, 0.0, 1.0)
            * torch.clamp((-0.6 - sd_plate_mm) * 1.5, 0.0, 1.0)
        )
        rim_geom = torch.clamp(1.0 - (torch.abs(sd_plate_mm + 5.25) - 1.25) / 0.8, 0.0, 1.0)
        d_left_bolt = torch.hypot(xx_mm - 60.0, yy_mm - 25.0)[None, None, :, :]
        d_right_bolt = torch.hypot(xx_mm - 270.0, yy_mm - 25.0)[None, None, :, :]
        h_left_bolt = torch.maximum(
            torch.abs(xx_mm - 60.0) * 0.8660254 + torch.abs(yy_mm - 25.0) * 0.5,
            torch.abs(yy_mm - 25.0),
        )[None, None, :, :]
        h_right_bolt = torch.maximum(
            torch.abs(xx_mm - 270.0) * 0.8660254 + torch.abs(yy_mm - 25.0) * 0.5,
            torch.abs(yy_mm - 25.0),
        )[None, None, :, :]
        # Large (440x220mm) JIS D 5705 4-bolt layout: upper (100, 35), (340, 35) and lower (100, 185), (340, 185) mm
        # mapped via 0.75x onto normalized 330x165mm canvas -> (75.0, 26.25), (255.0, 26.25), (75.0, 138.75), (255.0, 138.75)
        d_left_bolt_lg = torch.hypot(xx_mm - 75.0, yy_mm - 26.25)[None, None, :, :]
        d_right_bolt_lg = torch.hypot(xx_mm - 255.0, yy_mm - 26.25)[None, None, :, :]
        h_left_bolt_lg = torch.maximum(
            torch.abs(xx_mm - 75.0) * 0.8660254 + torch.abs(yy_mm - 26.25) * 0.5,
            torch.abs(yy_mm - 26.25),
        )[None, None, :, :]
        h_right_bolt_lg = torch.maximum(
            torch.abs(xx_mm - 255.0) * 0.8660254 + torch.abs(yy_mm - 26.25) * 0.5,
            torch.abs(yy_mm - 26.25),
        )[None, None, :, :]
        d_bot_bolt_lg = torch.minimum(
            torch.hypot(xx_mm - 75.0, yy_mm - 138.75),
            torch.hypot(xx_mm - 255.0, yy_mm - 138.75),
        )[None, None, :, :]
        h_bot_bolt_lg = torch.minimum(
            torch.maximum(
                torch.abs(xx_mm - 75.0) * 0.8660254 + torch.abs(yy_mm - 138.75) * 0.5,
                torch.abs(yy_mm - 138.75),
            ),
            torch.maximum(
                torch.abs(xx_mm - 255.0) * 0.8660254 + torch.abs(yy_mm - 138.75) * 0.5,
                torch.abs(yy_mm - 138.75),
            ),
        )[None, None, :, :]
        ax, ay, bx, by = 292.0, 6.0, 38.0, 159.0
        vx, vy = bx - ax, by - ay
        t_proj = torch.clamp(((xx_mm - ax) * vx + (yy_mm - ay) * vy) / (vx * vx + vy * vy), 0.0, 1.0)
        dist_slash = torch.hypot((xx_mm - ax) - t_proj * vx, (yy_mm - ay) - t_proj * vy)[None, None, :, :]
        slash_geom = torch.clamp((8.5 - dist_slash) * 1.2, 0.0, 1.0) * in_plate
        carbon_weave = (0.12 + 0.05 * torch.sin(xx_mm * 1.8) * torch.cos(yy_mm * 1.8))[None, None, :, :]

        # Gravity-driven vertical bolt rust streaks below Medium (60, 270; y=25) & Large (75, 255; y=26.25) upper bolts
        dy_r_med = (yy_mm - 26.5).clamp_min(0.0)
        dx_r_med = torch.minimum(torch.abs(xx_mm - 60.0), torch.abs(xx_mm - 270.0))
        rust_streak_med = (
            torch.exp(-((dx_r_med / (3.2 - 0.03 * dy_r_med.clamp_max(55.0))) ** 2))
            * torch.exp(-dy_r_med / 22.0)
            * (yy_mm >= 25.5).float()
        )[None, None, :, :] * in_plate
        dy_r_lg = (yy_mm - 27.75).clamp_min(0.0)
        dx_r_lg = torch.minimum(torch.abs(xx_mm - 75.0), torch.abs(xx_mm - 255.0))
        rust_streak_lg = (
            torch.exp(-((dx_r_lg / (3.6 - 0.03 * dy_r_lg.clamp_max(55.0))) ** 2))
            * torch.exp(-dy_r_lg / 24.0)
            * (yy_mm >= 26.75).float()
        )[None, None, :, :] * in_plate

        cached = (
            xx_mm, yy_mm, x_p_1d, y_p_1d,
            c_yel, c_grn, c_el_grn, c_el_amb, c_red_ink, c_bolt, c_mud, c_salt, c_rust, pat_cols,
            sd_plate_mm, in_plate, grille_stripes, hood_geom, in_ribbon, rim_geom,
            d_left_bolt, d_right_bolt, h_left_bolt, h_right_bolt,
            d_left_bolt_lg, d_right_bolt_lg, h_left_bolt_lg, h_right_bolt_lg,
            d_bot_bolt_lg, h_bot_bolt_lg, slash_geom, carbon_weave,
            rust_streak_med, rust_streak_lg,
        )
        self._coord_cache[dev] = cached
        return cached

    def build_single(
        self,
        plate_text: PlateText,
        config: PlateConfig | None = None,
        plate_side: int = 0,
        seed: int | None = None,
    ) -> PlateSurface25DPacket:
        """Build a `B=1` 2.5D surface packet for a single plate."""
        cfg = config or PlateConfig()
        return self.build_batch(
            plate_texts=[plate_text],
            configs=[cfg],
            plate_sides=[plate_side],
            seed=seed if seed is not None else (cfg.seed or 0),
        )

    @torch.no_grad()
    def build_batch(
        self,
        plate_texts: Sequence[PlateText],
        configs: Sequence[PlateConfig],
        plate_sides: Sequence[int] | None = None,
        seed: int = 0,
        layout_jitter: bool = False,
    ) -> PlateSurface25DPacket:
        """Vectorized construction of `B` extended-canonical `(192, 384)` 2.5D plate packets."""
        bsz = len(plate_texts)
        assert len(configs) == bsz
        dev = self.device
        atlas_t = self.atlas.to_torch(dev)  # (2, 8, 254, 64, 64) uint8
        (
            xx_mm, yy_mm, x_p_1d, y_p_1d,
            c_yel, c_grn, c_el_grn, c_el_amb, c_red_ink, c_bolt, c_mud, c_salt, c_rust, pat_cols,
            sd_plate_mm, in_plate, grille_stripes, hood_geom, in_ribbon, rim_geom,
            d_left_bolt, d_right_bolt, h_left_bolt, h_right_bolt,
            d_left_bolt_lg, d_right_bolt_lg, h_left_bolt_lg, h_right_bolt_lg,
            d_bot_bolt_lg, h_bot_bolt_lg, slash_geom, carbon_weave,
            rust_streak_med, rust_streak_lg,
        ) = self._get_grids(dev)

        rng = random.Random(seed)
        gen = torch.Generator(device=dev)
        gen.manual_seed((seed * 2654435761 + 17) & 0x7FFFFFFF)

        if plate_sides is None:
            sides_list = [1 if c.with_seal else rng.randint(0, 1) for c in configs]
        else:
            sides_list = [int(s) for s in plate_sides]

        # ------------------------------------------------------------------
        # 1. Pack per-plate layout & material scalars on CPU, upload once!
        # ------------------------------------------------------------------
        # Up to 7 upper-row glyphs (slots 0..3 region, 4..6 class)
        # Up to 6 lower-row glyphs (slot 7 kana, slots 8..11 serial, slot 12 hyphen)
        upper_boxes = np.zeros((bsz, 7, 4), dtype=np.float32)  # [x0, y0, w, h] in (160, 320)
        lower_boxes = np.zeros((bsz, 6, 4), dtype=np.float32)
        upper_tids = np.zeros((bsz, 7), dtype=np.int64)
        lower_tids = np.zeros((bsz, 6), dtype=np.int64)
        upper_bucket = np.zeros((bsz, 7), dtype=np.int64)
        lower_bucket = np.zeros((bsz, 6), dtype=np.int64)

        slot_boxes_np = np.zeros((bsz, NUM_SLOTS, 4), dtype=np.float32)
        target_slots_np = np.zeros((bsz, NUM_SLOTS), dtype=np.int64)
        ptype_np = np.zeros(bsz, dtype=np.int64)
        style_np = np.zeros(bsz, dtype=np.int64)

        bg_rgb_np = np.zeros((bsz, 3), dtype=np.float32)
        txt_rgb_np = np.zeros((bsz, 3), dtype=np.float32)
        rim_rgb_np = np.zeros((bsz, 3), dtype=np.float32)
        bg_nir_np = np.zeros(bsz, dtype=np.float32)
        txt_nir_np = np.zeros(bsz, dtype=np.float32)
        bg_retro_np = np.zeros(bsz, dtype=np.float32)
        txt_retro_np = np.zeros(bsz, dtype=np.float32)
        polarity_np = np.ones(bsz, dtype=np.float32)

        is_graphic_np = np.zeros(bsz, dtype=bool)
        graphic_mono_np = np.zeros(bsz, dtype=bool)
        graphic_pat_np = np.zeros(bsz, dtype=np.int64)     # 0=none, 1=flower, 2=geom, 3=expo, 4=wave, 5=mountain
        graphic_rim_np = np.zeros(bsz, dtype=np.int64)     # 0=none, 1=yellow, 2=green
        is_illum_np = np.zeros(bsz, dtype=bool)
        is_temp_np = np.zeros(bsz, dtype=bool)
        has_seal_np = np.zeros(bsz, dtype=bool)
        has_border_np = np.zeros(bsz, dtype=bool)
        has_emboss_np = np.zeros(bsz, dtype=bool)
        bolt_cap_r_np = np.zeros(bsz, dtype=np.float32)
        bolt_h_np = np.zeros(bsz, dtype=np.float32)
        seal_r_np = np.zeros(bsz, dtype=np.float32)
        frame_type_np = np.zeros(bsz, dtype=np.int64)      # 0=none, 1=chrome, 2=black_plastic, 3=dealer_thick, 4=carbon
        frame_h_np = np.zeros(bsz, dtype=np.float32)
        frame_in_np = np.zeros(bsz, dtype=np.float32)
        bumper_depth_np = np.zeros(bsz, dtype=np.float32)
        hood_h_np = np.zeros(bsz, dtype=np.float32)
        mud_lvl_np = np.zeros(bsz, dtype=np.float32)
        weather_np = np.zeros(bsz, dtype=np.float32)
        slash_nir_alpha_np = np.zeros(bsz, dtype=np.float32)
        plate_rough_np = np.full(bsz, 0.26, dtype=np.float32)
        frame_top_extra_np = np.zeros(bsz, dtype=np.float32)
        rust_lvl_np = np.zeros(bsz, dtype=np.float32)
        salt_dust_np = np.zeros(bsz, dtype=np.float32)
        sdf_bias_np = np.zeros(bsz, dtype=np.float32)
        emboss_h_np = np.full(bsz, 1.5, dtype=np.float32)
        emboss_bevel_np = np.full(bsz, 0.85, dtype=np.float32)
        chip_lvl_np = np.zeros(bsz, dtype=np.float32)
        seal_h_np = np.full(bsz, 8.0, dtype=np.float32)
        bolt_dx_l_np = np.zeros(bsz, dtype=np.float32)
        bolt_dy_l_np = np.zeros(bsz, dtype=np.float32)
        bolt_dx_r_np = np.zeros(bsz, dtype=np.float32)
        bolt_dy_r_np = np.zeros(bsz, dtype=np.float32)

        is_large_np = np.zeros(bsz, dtype=bool)
        bot_bolt_mode_np = np.zeros(bsz, dtype=np.int64)   # 0=none, 1=raised M10 hex+washer (85%), 2=flush/unbolted hole (15%)
        hyphen_id = self.vocab.token_to_id["-"]
        pat_map = {"flower": 1, "geometric": 2, "rainbow": 3, "expo": 3, "wave": 4, "mountain": 5}
        frame_map = {"chrome": 1, "black_plastic": 2, "dealer_thick": 3, "carbon": 4}

        for b in range(bsz):
            pt = plate_texts[b]
            cfg = configs[b]
            side = sides_list[b]

            encoded_12 = self.vocab.encode(pt)
            target_slots_np[b] = encoded_12
            ptype_np[b] = self.vocab.encode_plate_type(cfg.plate_type)
            style_np[b] = rng.randrange(self.atlas.NUM_STYLES)
            is_large_np[b] = cfg.plate_size == PlateSize.LARGE

            layout = PlateGeometry.calculate_layout(
                pt,
                cfg.plate_size,
                jitter_rng=rng if layout_jitter else None,
            )
            sx = float(PLATE_W) / float(layout.width_mm)
            sy = float(PLATE_H) / float(layout.height_mm)

            # Populate default canonical boxes for inactive <PAD> slots so every slot 0..11 has valid coords
            for s_i in range(4):
                slot_boxes_np[b, s_i] = [88.0 * sx + s_i * 19.0, 12.0 * sy, 107.0 * sx + s_i * 19.0, 56.0 * sy]
            for s_i in range(3):
                slot_boxes_np[b, 4 + s_i] = [(173.0 + s_i * 23.0) * sx, 12.0 * sy, (196.0 + s_i * 23.0) * sx, 56.0 * sy]

            ser_Idx = 0
            for cslot in layout.slots:
                bx0 = cslot.x * sx
                by0 = cslot.y * sy
                bw = cslot.width * sx
                bh = cslot.height * sy
                tid = self.vocab.token_to_id.get(cslot.char, 0)
                asp = cslot.width / max(1.0, cslot.height)
                bkt = 1 if (asp < 0.78 and not cslot.is_dot and cslot.char != "-") else 0

                if cslot.field == "region":
                    u_idx = cslot.index
                    upper_boxes[b, u_idx] = [bx0, by0, bw, bh]
                    upper_tids[b, u_idx] = tid
                    upper_bucket[b, u_idx] = bkt
                    slot_boxes_np[b, u_idx] = [bx0, by0, bx0 + bw, by0 + bh]
                elif cslot.field == "classification":
                    u_idx = 4 + cslot.index
                    upper_boxes[b, u_idx] = [bx0, by0, bw, bh]
                    upper_tids[b, u_idx] = tid
                    upper_bucket[b, u_idx] = bkt
                    slot_boxes_np[b, u_idx] = [bx0, by0, bx0 + bw, by0 + bh]
                elif cslot.field == "hiragana":
                    lower_boxes[b, 0] = [bx0, by0, bw, bh]
                    lower_tids[b, 0] = tid
                    lower_bucket[b, 0] = bkt
                    slot_boxes_np[b, 7] = [bx0, by0, bx0 + bw, by0 + bh]
                elif cslot.field == "serial":
                    if cslot.char == "-":
                        lower_boxes[b, 5] = [bx0, by0, bw, bh]
                        lower_tids[b, 5] = hyphen_id
                        lower_bucket[b, 5] = 0
                    else:
                        l_idx = 1 + ser_Idx
                        lower_boxes[b, l_idx] = [bx0, by0, bw, bh]
                        lower_tids[b, l_idx] = tid
                        lower_bucket[b, l_idx] = 1 if not cslot.is_dot else 0
                        slot_boxes_np[b, 8 + ser_Idx] = [bx0, by0, bx0 + bw, by0 + bh]
                        ser_Idx += 1

            # Propagate active lead box to trailing <PAD> slots for field termination supervision
            reg_len = len(pt.region)
            if reg_len < 4:
                lead_box = slot_boxes_np[b, reg_len - 1]
                for r_pad in range(reg_len, 4):
                    slot_boxes_np[b, r_pad] = lead_box
            cls_len = len(pt.classification)
            if cls_len < 3:
                lead_cbox = slot_boxes_np[b, 4 + cls_len - 1]
                for c_pad in range(cls_len, 3):
                    slot_boxes_np[b, 4 + c_pad] = lead_cbox

            # Measured real-world base color drift per PlateType / temporary / graphic
            weath = float(cfg.weathering) if cfg.weathering > 0 else rng.uniform(0.0, 0.25)
            weather_np[b] = weath

            if cfg.is_temporary_plate:
                is_temp_np[b] = True
                bg_rgb_np[b] = [rng.uniform(0.90, 0.97), rng.uniform(0.90, 0.97), rng.uniform(0.88, 0.95)]
                txt_rgb_np[b] = [rng.uniform(0.07, 0.15), rng.uniform(0.07, 0.15), rng.uniform(0.08, 0.16)]
                rim_rgb_np[b] = txt_rgb_np[b]
                bg_nir_np[b] = rng.uniform(0.86, 0.94)
                txt_nir_np[b] = rng.uniform(0.06, 0.11)  # Carbon black absorbs NIR strongly
                bg_retro_np[b] = rng.uniform(0.82, 0.95)
                txt_retro_np[b] = 0.0
                polarity_np[b] = 1.0
                # Red organic slash ink is 80-92% transparent in 850/940nm NIR -> 0.08..0.20 NIR opacity!
                slash_nir_alpha_np[b] = rng.uniform(0.08, 0.20)
            elif cfg.is_graphic or cfg.plate_type == PlateType.GRAPHIC:
                is_graphic_np[b] = True
                graphic_mono_np[b] = bool(cfg.graphic_monochrome)
                graphic_pat_np[b] = pat_map.get(cfg.graphic_pattern or "flower", rng.randint(1, 5))
                if cfg.graphic_border_type == "yellow":
                    graphic_rim_np[b] = 1
                elif cfg.graphic_border_type == "green":
                    graphic_rim_np[b] = 2
                bg_rgb_np[b] = [rng.uniform(0.92, 0.98), rng.uniform(0.92, 0.98), rng.uniform(0.90, 0.97)]
                txt_rgb_np[b] = [rng.uniform(0.06, 0.15), rng.uniform(0.22, 0.37), rng.uniform(0.11, 0.22)]
                rim_rgb_np[b] = txt_rgb_np[b]
                bg_nir_np[b] = rng.uniform(0.86, 0.95)
                txt_nir_np[b] = rng.uniform(0.12, 0.21)
                bg_retro_np[b] = rng.uniform(0.82, 0.96)
                txt_retro_np[b] = 0.0
                polarity_np[b] = 1.0
            elif cfg.plate_type == PlateType.STANDARD_PRIVATE:
                # Crisp white to dirty/warm aged cream-white
                warm = weath * rng.uniform(0.04, 0.12)
                base_w = rng.uniform(0.86, 0.98) - 0.08 * weath
                bg_rgb_np[b] = [base_w, base_w - 0.25 * warm, base_w - warm]
                txt_rgb_np[b] = [rng.uniform(0.06, 0.16), rng.uniform(0.21, 0.38), rng.uniform(0.10, 0.23)]
                rim_rgb_np[b] = txt_rgb_np[b]
                bg_nir_np[b] = rng.uniform(0.86, 0.96) * (1.0 - 0.15 * weath)
                txt_nir_np[b] = rng.uniform(0.11, 0.21)
                bg_retro_np[b] = rng.uniform(0.84, 0.98) * (1.0 - 0.25 * weath)
                txt_retro_np[b] = 0.0
                polarity_np[b] = 1.0
            elif cfg.plate_type == PlateType.STANDARD_COMMERCIAL:
                # Diversified commercial green albedo across:
                #   (1) new deep emerald green (~42%),
                #   (2) sun-faded olive green (~30%),
                #   (3) quarry / cement-dust chalky slate green (~28%).
                u_grn = rng.random()
                if u_grn < 0.42:
                    bg_rgb_np[b] = [
                        rng.uniform(8.0, 28.0) / 255.0,
                        rng.uniform(42.0, 78.0) / 255.0,
                        rng.uniform(20.0, 46.0) / 255.0,
                    ]
                elif u_grn < 0.72:
                    bg_rgb_np[b] = [
                        rng.uniform(26.0, 58.0) / 255.0,
                        rng.uniform(56.0, 96.0) / 255.0,
                        rng.uniform(24.0, 52.0) / 255.0,
                    ]
                else:
                    bg_rgb_np[b] = [
                        rng.uniform(38.0, 74.0) / 255.0,
                        rng.uniform(58.0, 98.0) / 255.0,
                        rng.uniform(48.0, 88.0) / 255.0,
                    ]
                w_txt = rng.uniform(0.86, 0.98) - 0.08 * weath
                txt_rgb_np[b] = [w_txt, w_txt, w_txt - 0.02]
                rim_rgb_np[b] = txt_rgb_np[b]
                bg_nir_np[b] = rng.uniform(0.09, 0.48)
                txt_nir_np[b] = rng.uniform(0.82, 0.93)
                bg_retro_np[b] = 0.04
                txt_retro_np[b] = rng.uniform(0.72, 0.90)
                polarity_np[b] = -1.0
            elif cfg.plate_type == PlateType.KEI_PRIVATE:
                # Rich amber-yellow to sun-faded pale cream-yellow!
                uv_fade = rng.uniform(0.35, 0.78) if rng.random() < (0.22 + 0.4 * weath) else rng.uniform(0.0, 0.18)
                r_y = rng.uniform(0.93, 0.99) - 0.04 * weath
                g_y = rng.uniform(0.73, 0.83) * (1.0 - uv_fade) + rng.uniform(0.88, 0.94) * uv_fade
                b_y = rng.uniform(0.07, 0.19) * (1.0 - uv_fade) + rng.uniform(0.57, 0.76) * uv_fade
                bg_rgb_np[b] = [r_y, g_y, b_y]
                c_blk = rng.uniform(0.07, 0.17) + 0.05 * weath
                txt_rgb_np[b] = [c_blk, c_blk, c_blk]
                rim_rgb_np[b] = txt_rgb_np[b]
                # Yellow sheeting reflects 850/940nm NIR strongly just like white sheeting!
                bg_nir_np[b] = rng.uniform(0.84, 0.94) * (1.0 - 0.14 * weath)
                txt_nir_np[b] = rng.uniform(0.06, 0.12)
                bg_retro_np[b] = rng.uniform(0.78, 0.95) * (1.0 - 0.25 * weath)
                txt_retro_np[b] = 0.0
                polarity_np[b] = 1.0
            else:  # KEI_COMMERCIAL (charcoal-black bg, amber-yellow text)
                c_bg = rng.uniform(0.07, 0.19)
                bg_rgb_np[b] = [c_bg, c_bg, c_bg + 0.01]
                txt_rgb_np[b] = [rng.uniform(0.91, 0.99), rng.uniform(0.72, 0.86), rng.uniform(0.09, 0.35)]
                rim_rgb_np[b] = txt_rgb_np[b]
                bg_nir_np[b] = rng.uniform(0.06, 0.13)
                txt_nir_np[b] = rng.uniform(0.80, 0.91)
                bg_retro_np[b] = 0.02
                txt_retro_np[b] = rng.uniform(0.70, 0.88)
                polarity_np[b] = -1.0

            is_illum_np[b] = bool(cfg.is_illuminated) and not cfg.is_temporary_plate
            has_border_np[b] = bool(cfg.with_border)
            has_emboss_np[b] = bool(cfg.with_emboss) and not is_illum_np[b]

            # Rear-left aluminum seal (封印): +8.0..10.2 mm height, Ø22..28 mm when rear & with_seal
            legal_seal_type = (
                cfg.plate_type in (PlateType.STANDARD_PRIVATE, PlateType.STANDARD_COMMERCIAL)
                or (is_graphic_np[b] and graphic_rim_np[b] in (0, 2))
            )
            if side == 1 and cfg.with_seal and legal_seal_type and not cfg.is_temporary_plate:
                has_seal_np[b] = True
                if layout_jitter:
                    seal_r_np[b] = rng.uniform(11.0, 14.2)  # heavy rear-seal overlap envelope
                    seal_h_np[b] = rng.uniform(8.0, 10.2)
                else:
                    seal_r_np[b] = rng.uniform(11.0, 12.5)  # Ø22..25 mm
                    seal_h_np[b] = 8.0

            if cfg.bolt_cap_size_mm >= 14.0:
                bolt_cap_r_np[b] = 0.5 * float(cfg.bolt_cap_size_mm)  # r = 9..11 mm
                bolt_h_np[b] = 5.5
            elif cfg.with_bolt_holes:
                if layout_jitter and rng.random() < 0.22:
                    # Oversized aftermarket theft-deterrent bolt caps / wide flange washers overlapping upper slots
                    bolt_cap_r_np[b] = rng.uniform(8.2, 12.5)
                    bolt_h_np[b] = rng.uniform(5.2, 7.6)
                elif is_large_np[b]:
                    bolt_cap_r_np[b] = 6.2  # M10 hex bolt + flange washer mapped via 0.75x (~16.5mm physical Ø)
                    bolt_h_np[b] = 4.6
                else:
                    bolt_cap_r_np[b] = 5.5  # standard M6 hex bolt + washer Ø11 mm
                    bolt_h_np[b] = 4.0
            if layout_jitter and cfg.with_bolt_holes:
                # Realistic slight mounting / overlap jitter pulling bolts/seals closer to Region 0 & Class 6
                bolt_dx_l_np[b] = rng.uniform(-2.5, 4.5)
                bolt_dy_l_np[b] = rng.uniform(-2.5, 3.0)
                bolt_dx_r_np[b] = rng.uniform(-4.5, 2.5)
                bolt_dy_r_np[b] = rng.uniform(-2.5, 3.0)
            if is_large_np[b] and cfg.with_bolt_holes and not is_temp_np[b]:
                # ~85% raised M10 hex bolts + washers, ~15% flush/unbolted lower mounting holes
                bot_bolt_mode_np[b] = 2 if rng.random() < 0.15 else 1

            # Continuous SDF stroke-weight perturbation (±0.40..0.90px), emboss relief & paint chipping!
            if layout_jitter:
                if rng.random() < 0.70:
                    sgn_s = 1.0 if rng.random() < 0.52 else -1.0
                    sdf_bias_np[b] = sgn_s * rng.uniform(0.40, 0.90)
                else:
                    sdf_bias_np[b] = rng.uniform(-0.25, 0.25)
                emboss_h_np[b] = rng.uniform(0.85, 2.35)
                emboss_bevel_np[b] = rng.uniform(0.55, 1.25)
                if (weath > 0.10 or float(cfg.mud_level) > 0.05 or rng.random() < 0.24):
                    chip_lvl_np[b] = rng.uniform(0.12, 0.52)

            f_style = cfg.frame_style or ("black_plastic" if (cfg.plate_frame or is_illum_np[b]) else None)
            if f_style in frame_map:
                ft = frame_map[f_style]
                frame_type_np[b] = ft
                if is_illum_np[b]:
                    frame_h_np[b] = rng.uniform(4.8, 6.2)
                    frame_in_np[b] = rng.uniform(5.0, 7.5)
                elif ft == 3:  # dealer_thick
                    frame_h_np[b] = rng.uniform(4.5, 6.0)
                    frame_in_np[b] = rng.uniform(6.0, 8.8)
                else:
                    frame_h_np[b] = rng.uniform(3.5, 5.2)
                    frame_in_np[b] = rng.uniform(4.0, 6.5)
                if rng.random() < 0.30:
                    frame_top_extra_np[b] = rng.uniform(0.5, 5.0)

            if polarity_np[b] < 0.0:
                plate_rough_np[b] = rng.uniform(0.10, 0.30)
            else:
                plate_rough_np[b] = rng.uniform(0.12, 0.38)

            if bolt_cap_r_np[b] > 0.1 and not is_temp_np[b]:
                p_rust = 0.32 if (polarity_np[b] < 0.0 or weath > 0.10) else 0.14
                if rng.random() < p_rust:
                    rust_lvl_np[b] = rng.uniform(0.28, 0.88)

            bumper_depth_np[b] = rng.uniform(-8.0, -2.5)
            hood_h_np[b] = rng.uniform(8.0, 18.0) if rng.random() < 0.45 else 0.0
            mud_lvl_np[b] = float(cfg.mud_level)
            if mud_lvl_np[b] > 1e-3 and polarity_np[b] < 0.0 and rng.random() < 0.52:
                salt_dust_np[b] = 1.0

        any_graphic = bool(is_graphic_np.any())
        any_illum = bool(is_illum_np.any())
        any_temp = bool(is_temp_np.any())
        any_seal = bool(has_seal_np.any())
        any_frame = bool((frame_type_np > 0).any())
        any_mud = bool((mud_lvl_np > 1e-3).any())

        # Pack all per-plate float32 scalars into ONE contiguous (B, 36) tensor upload!
        float_pack_np = np.column_stack(
            [
                bg_rgb_np,          # 0..2
                txt_rgb_np,         # 3..5
                rim_rgb_np,         # 6..8
                bg_nir_np,          # 9
                txt_nir_np,         # 10
                bg_retro_np,        # 11
                txt_retro_np,       # 12
                polarity_np,        # 13
                bolt_cap_r_np,      # 14
                bolt_h_np,          # 15
                seal_r_np,          # 16
                frame_h_np,         # 17
                frame_in_np,        # 18
                bumper_depth_np,    # 19
                hood_h_np,          # 20
                mud_lvl_np,         # 21
                slash_nir_alpha_np, # 22
                plate_rough_np,     # 23
                frame_top_extra_np, # 24
                rust_lvl_np,        # 25
                salt_dust_np,       # 26
                sdf_bias_np,        # 27
                emboss_h_np,        # 28
                emboss_bevel_np,    # 29
                chip_lvl_np,        # 30
                seal_h_np,          # 31
                bolt_dx_l_np,       # 32
                bolt_dy_l_np,       # 33
                bolt_dx_r_np,       # 34
                bolt_dy_r_np,       # 35
            ]
        ).astype(np.float32, copy=False)
        fp = torch.from_numpy(float_pack_np).to(dev)

        bg_rgb = fp[:, 0:3, None, None]
        txt_rgb = fp[:, 3:6, None, None]
        rim_rgb = fp[:, 6:9, None, None]
        bg_nir = fp[:, 9:10, None, None]
        txt_nir = fp[:, 10:11, None, None]
        bg_retro = fp[:, 11:12, None, None]
        txt_retro = fp[:, 12:13, None, None]
        base_polarity_t = fp[:, 13]
        bolt_cap_r = fp[:, 14:15, None, None]
        bolt_h = fp[:, 15:16, None, None]
        seal_r = fp[:, 16:17, None, None]
        frame_h = fp[:, 17:18, None, None]
        frame_in = fp[:, 18:19, None, None]
        bumper_depth = fp[:, 19:20, None, None]
        hood_h = fp[:, 20:21, None, None]
        mud_lvl = fp[:, 21:22, None, None]
        slash_nir_alpha = fp[:, 22:23, None, None]
        plate_rough = fp[:, 23:24, None, None]
        frame_top_extra = fp[:, 24:25, None, None]
        rust_lvl = fp[:, 25:26, None, None]
        salt_dust = fp[:, 26:27, None, None]
        sdf_bias = fp[:, 27:28, None, None]
        emboss_h = fp[:, 28:29, None, None]
        emboss_bevel_w = fp[:, 29:30, None, None]
        chip_lvl = fp[:, 30:31, None, None]
        seal_h = fp[:, 31:32, None, None]
        bolt_dx_l = fp[:, 32:33, None, None]
        bolt_dy_l = fp[:, 33:34, None, None]
        bolt_dx_r = fp[:, 34:35, None, None]
        bolt_dy_r = fp[:, 35:36, None, None]

        # Pack all per-plate int64 scalars into ONE contiguous (B, 26) tensor upload!
        int_pack_np = np.column_stack(
            [
                target_slots_np,                     # 0..11
                ptype_np,                            # 12
                style_np,                            # 13
                np.asarray(sides_list, np.int64),    # 14
                graphic_pat_np,                      # 15
                graphic_rim_np,                      # 16
                frame_type_np,                       # 17
                is_graphic_np.astype(np.int64),      # 18
                graphic_mono_np.astype(np.int64),    # 19
                is_illum_np.astype(np.int64),        # 20
                is_temp_np.astype(np.int64),         # 21
                has_seal_np.astype(np.int64),        # 22
                has_border_np.astype(np.int64),      # 23
                has_emboss_np.astype(np.int64),      # 24
                bot_bolt_mode_np,                    # 25
            ]
        )
        ip = torch.from_numpy(int_pack_np).to(dev)
        target_slots_t = ip[:, 0:12]
        ptype_t = ip[:, 12]
        styles_t = ip[:, 13]
        sides_t = ip[:, 14]
        graphic_pat = ip[:, 15]
        graphic_rim = ip[:, 16]
        frame_type = ip[:, 17]
        is_graphic = ip[:, 18].bool()
        graphic_mono = ip[:, 19].bool()
        is_illum = ip[:, 20].bool()
        is_temp = ip[:, 21].bool()
        has_seal = ip[:, 22].bool()
        has_border = ip[:, 23].bool()
        has_emboss = ip[:, 24].bool()
        bot_bolt_mode = ip[:, 25]

        # Pack boxes & glyph IDs cleanly
        boxes_all_t = torch.from_numpy(
            np.concatenate([upper_boxes, lower_boxes, slot_boxes_np], axis=1)
        ).to(dev)  # (B, 25, 4)
        u_boxes_t = boxes_all_t[:, 0:7]
        l_boxes_t = boxes_all_t[:, 7:13]
        slot_boxes_t = boxes_all_t[:, 13:25]

        glyph_meta_t = torch.from_numpy(
            np.concatenate([upper_tids, lower_tids, upper_bucket, lower_bucket], axis=1)
        ).to(dev)  # (B, 26)
        u_tids_t = glyph_meta_t[:, 0:7]
        l_tids_t = glyph_meta_t[:, 7:13]
        u_bkt_t = glyph_meta_t[:, 13:20]
        l_bkt_t = glyph_meta_t[:, 20:26]

        # ------------------------------------------------------------------
        # 2. Vectorized Separable SDF Glyph Evaluation on (B, 160, 320)
        # ------------------------------------------------------------------
        plate_sdf_px = torch.full((bsz, PLATE_H, PLATE_W), -8.0, device=dev, dtype=torch.float32)
        slot_masks_half = torch.zeros((bsz, NUM_SLOTS, 80, 160), device=dev, dtype=torch.float32)

        # 2a. Upper Strip: rows [0:64] of (160, 320), 7 mutually disjoint glyph boxes!
        sdf_upper, sm_upper = self._splat_disjoint_strip(
            atlas_t=atlas_t,
            styles_t=styles_t,
            boxes=u_boxes_t,
            tids=u_tids_t,
            buckets=u_bkt_t,
            x_1d=x_p_1d,
            y_1d=y_p_1d[:64],
            num_semantic_slots=7,
            full_h=PLATE_H,
            y_offset=0,
        )
        plate_sdf_px[:, :64, :] = sdf_upper
        slot_masks_half[:, 0:7, :, :] = sm_upper

        # 2b. Lower Strip: rows [64:160] of (160, 320), 6 mutually disjoint glyph boxes!
        sdf_lower, sm_lower = self._splat_disjoint_strip(
            atlas_t=atlas_t,
            styles_t=styles_t,
            boxes=l_boxes_t,
            tids=l_tids_t,
            buckets=l_bkt_t,
            x_1d=x_p_1d,
            y_1d=y_p_1d[64:],
            num_semantic_slots=5,  # first 5 of 6 are semantic slots 7..11 (index 5 is hyphen '-')
            full_h=PLATE_H,
            y_offset=64,
        )
        plate_sdf_px[:, 64:, :] = sdf_lower
        slot_masks_half[:, 7:12, :, :] = sm_lower

        # Embed plate SDF into extended (192, 384) canvas
        ext_sdf_px = torch.full((bsz, 1, EXT_H, EXT_W), -8.0, device=dev, dtype=torch.float32)
        ext_sdf_px[:, 0, PAD_Y : PAD_Y + PLATE_H, PAD_X : PAD_X + PLATE_W] = plate_sdf_px

        # ------------------------------------------------------------------
        # 3. Surround Bumper & Optional Graphic Plate Artwork
        # ------------------------------------------------------------------
        surround_base = torch.rand((bsz, 3, 1, 1), generator=gen, device=dev) * 0.28 + 0.04
        plate_bg_rgb = bg_rgb.expand(bsz, 3, EXT_H, EXT_W).clone()
        plate_bg_nir = bg_nir.expand(bsz, 1, EXT_H, EXT_W).clone()
        retro = bg_retro * in_plate

        if any_graphic:
            g_indices = torch.from_numpy(np.flatnonzero(is_graphic_np)).to(dev)
            n_g_raw = int(g_indices.shape[0])
            g_phase = torch.rand((n_g_raw, 1, 1, 1), generator=gen, device=dev) * (2.0 * math.pi)
            if n_g_raw == 1 and dev.type == "cuda":
                # Avoid PyTorch dynamic=True dim==1 Triton specialization recompile
                g_eval = g_indices.expand(2)
                pat_sub = _run_graphic_patterns(
                    xx_mm,
                    yy_mm,
                    graphic_pat.index_select(0, g_eval),
                    graphic_mono.index_select(0, g_eval),
                    pat_cols,
                    g_phase.expand(2, 1, 1, 1),
                )[:1]
            else:
                pat_sub = _run_graphic_patterns(
                    xx_mm,
                    yy_mm,
                    graphic_pat.index_select(0, g_indices),
                    graphic_mono.index_select(0, g_indices),
                    pat_cols,
                    g_phase,
                )  # (G, 4, 192, 384)
            p_alpha = pat_sub[:, 3:4] * in_plate
            plate_bg_rgb[g_indices] = plate_bg_rgb[g_indices] * (1.0 - p_alpha) + pat_sub[:, 0:3] * p_alpha
            plate_bg_nir[g_indices] = plate_bg_nir[g_indices] * (1.0 - 0.18 * p_alpha) + 0.55 * (0.18 * p_alpha)
            retro[g_indices] = retro[g_indices] * (1.0 - 0.35 * p_alpha)

        warm_el = (
            (torch.rand((bsz, 1, 1, 1), generator=gen, device=dev) < 0.22).float()
            if any_illum
            else torch.zeros((bsz, 1, 1, 1), device=dev)
        )
        mud_low = (
            torch.rand((bsz, 1, 16, 32), generator=gen, device=dev)
            if any_mud
            else torch.zeros((bsz, 1, 16, 32), device=dev)
        )
        chip_low = (
            torch.rand((bsz, 1, 24, 48), generator=gen, device=dev)
            if layout_jitter
            else torch.zeros((bsz, 1, 24, 48), device=dev)
        )

        if layout_jitter:
            is_lg_f = torch.from_numpy(is_large_np).to(dev)[:, None, None, None].float()
            bx_l = 60.0 * (1.0 - is_lg_f) + 75.0 * is_lg_f + bolt_dx_l
            by_l = 25.0 * (1.0 - is_lg_f) + 26.25 * is_lg_f + bolt_dy_l
            bx_r = 270.0 * (1.0 - is_lg_f) + 255.0 * is_lg_f + bolt_dx_r
            by_r = 25.0 * (1.0 - is_lg_f) + 26.25 * is_lg_f + bolt_dy_r
            xx4 = xx_mm[None, None, :, :]
            yy4 = yy_mm[None, None, :, :]
            d_left_eff = torch.hypot(xx4 - bx_l, yy4 - by_l)
            d_right_eff = torch.hypot(xx4 - bx_r, yy4 - by_r)
            h_left_eff = torch.maximum(
                torch.abs(xx4 - bx_l) * 0.8660254 + torch.abs(yy4 - by_l) * 0.5,
                torch.abs(yy4 - by_l),
            )
            h_right_eff = torch.maximum(
                torch.abs(xx4 - bx_r) * 0.8660254 + torch.abs(yy4 - by_r) * 0.5,
                torch.abs(yy4 - by_r),
            )
            rust_streak_eff = torch.where(is_lg_f > 0.5, rust_streak_lg, rust_streak_med)
        elif bool(is_large_np.any()):
            is_lg_t = torch.from_numpy(is_large_np).to(dev)[:, None, None, None]
            d_left_eff = torch.where(is_lg_t, d_left_bolt_lg, d_left_bolt)
            d_right_eff = torch.where(is_lg_t, d_right_bolt_lg, d_right_bolt)
            h_left_eff = torch.where(is_lg_t, h_left_bolt_lg, h_left_bolt)
            h_right_eff = torch.where(is_lg_t, h_right_bolt_lg, h_right_bolt)
            rust_streak_eff = torch.where(is_lg_t, rust_streak_lg, rust_streak_med)
        else:
            d_left_eff = d_left_bolt
            d_right_eff = d_right_bolt
            h_left_eff = h_left_bolt
            h_right_eff = h_right_bolt
            rust_streak_eff = rust_streak_med

        (
            albedo,
            nir,
            height_mm,
            height_u8,
            retro,
            rough,
            metal,
            emissive_rgb,
            s_mask,
            occ_half,
        ) = _run_compose_surface_tail(
            surround_base,
            grille_stripes,
            hood_geom,
            bumper_depth,
            hood_h,
            plate_bg_rgb,
            plate_bg_nir,
            retro,
            ext_sdf_px,
            in_plate,
            in_ribbon,
            rim_geom,
            sd_plate_mm,
            yy_mm,
            d_left_eff,
            d_right_eff,
            h_left_eff,
            h_right_eff,
            d_bot_bolt_lg,
            h_bot_bolt_lg,
            slash_geom,
            carbon_weave,
            rust_streak_eff,
            c_yel,
            c_grn,
            rim_rgb,
            txt_rgb,
            txt_nir,
            txt_retro,
            c_el_grn,
            c_el_amb,
            c_red_ink,
            c_bolt,
            c_mud,
            c_salt,
            c_rust,
            graphic_rim,
            has_border,
            is_graphic,
            has_emboss,
            is_illum,
            is_temp,
            has_seal,
            frame_type,
            bot_bolt_mode,
            bolt_cap_r,
            bolt_h,
            seal_r,
            frame_h,
            frame_in,
            mud_lvl,
            slash_nir_alpha,
            plate_rough,
            frame_top_extra,
            rust_lvl,
            salt_dust,
            warm_el,
            mud_low,
            sdf_bias,
            emboss_h,
            emboss_bevel_w,
            chip_lvl,
            seal_h,
            chip_low,
        )

        return PlateSurface25DPacket(
            albedo_rgb=albedo,
            nir_albedo=nir,
            height_mm=height_mm,
            height_u8=height_u8,
            retro_mask=retro,
            roughness=rough,
            metallic=metal,
            emissive_rgb=emissive_rgb,
            stroke_mask=s_mask,
            slot_boxes_canon=slot_boxes_t,
            slot_masks_half=slot_masks_half,
            surface_occ_2d=occ_half,
            target_slots=target_slots_t,
            target_plate_type=ptype_t,
            base_polarity=base_polarity_t,
            is_illuminated=is_illum,
            is_temporary=is_temp,
            has_seal=has_seal,
            plate_side=sides_t,
        )

    @staticmethod
    def _splat_disjoint_strip(
        atlas_t: torch.Tensor,
        styles_t: torch.Tensor,
        boxes: torch.Tensor,
        tids: torch.Tensor,
        buckets: torch.Tensor,
        x_1d: torch.Tensor,
        y_1d: torch.Tensor,
        num_semantic_slots: int,
        full_h: int,
        y_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized bilinear SDF lookup for `K` mutually-disjoint axis-aligned glyph slots."""
        sdf_px, strip_masks = _run_splat_strip(
            atlas_t, styles_t, boxes, tids, buckets, x_1d, y_1d, num_semantic_slots
        )
        bsz = boxes.shape[0]
        sem_masks = torch.zeros((bsz, num_semantic_slots, full_h // 2, 160), device=atlas_t.device, dtype=torch.float32)
        hy0 = y_offset // 2
        hy1 = hy0 + (y_1d.shape[0] // 2)
        sem_masks[:, :, hy0:hy1, :] = strip_masks
        return sdf_px, sem_masks

def _graphic_patterns_impl(
    xx_mm: torch.Tensor,
    yy_mm: torch.Tensor,
    graphic_pat: torch.Tensor,
    graphic_mono: torch.Tensor,
    pat_cols: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Vectorized procedural Japanese graphic plate background artwork `(G, 4, 192, 384)`."""
    u = (xx_mm / 330.0)[None, None, :, :]
    v = (yy_mm / 165.0)[None, None, :, :]

    dx_f = u - 0.78
    dy_f = v - 0.36
    r_f = torch.hypot(dx_f, dy_f)
    ang_f = torch.atan2(dy_f, dx_f)
    petals = torch.clamp(1.0 - (r_f - (0.16 + 0.06 * torch.cos(5.0 * ang_f + phase))) * 14.0, 0.0, 1.0)

    tri1 = torch.sin((u * 14.0 + v * 9.0) * math.pi + phase)
    tri2 = torch.sin((u * 14.0 - v * 9.0) * math.pi - phase)
    geom_a = torch.clamp(tri1 * tri2, 0.0, 1.0) * 0.48

    b_red = torch.exp(-(((u - 0.14) ** 2 + (v - 0.26) ** 2) / 0.018)) + torch.exp(
        -(((u - 0.88) ** 2 + (v - 0.72) ** 2) / 0.022)
    )
    b_blu = torch.exp(-(((u - 0.84) ** 2 + (v - 0.22) ** 2) / 0.016)) + torch.exp(
        -(((u - 0.20) ** 2 + (v - 0.82) ** 2) / 0.020)
    )
    expo_a = torch.clamp(b_red + b_blu, 0.0, 0.65)
    w_r = b_red / (b_red + b_blu + 1e-4)
    col_expo = w_r * pat_cols[2] + (1.0 - w_r) * pat_cols[3]

    wave_y = 0.68 + 0.07 * torch.sin(u * 12.0 + phase) + 0.03 * torch.cos(u * 23.0)
    wave_a = torch.clamp((v - wave_y) * 9.0, 0.0, 1.0) * 0.50

    ridge_y = 0.82 - 0.46 * torch.exp(-(((u - 0.52) / 0.26) ** 2))
    mtn_a = torch.clamp((v - ridge_y) * 10.0, 0.0, 1.0) * 0.46

    p1 = (graphic_pat == 1)[:, None, None, None].float()
    p2 = (graphic_pat == 2)[:, None, None, None].float()
    p3 = (graphic_pat == 3)[:, None, None, None].float()
    p4 = (graphic_pat == 4)[:, None, None, None].float()
    p5 = (graphic_pat == 5)[:, None, None, None].float()

    rgb = p1 * pat_cols[0] + p2 * pat_cols[1] + p3 * col_expo + p4 * pat_cols[4] + p5 * pat_cols[5]
    alpha = p1 * (petals * 0.52) + p2 * geom_a + p3 * expo_a + p4 * wave_a + p5 * mtn_a

    mono_f = graphic_mono[:, None, None, None].float()
    lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    gray_rgb = (0.52 + 0.35 * lum).expand_as(rgb)
    rgb = rgb * (1.0 - mono_f) + gray_rgb * mono_f

    return torch.cat([rgb, alpha], dim=1)


def _splat_strip_impl(
    atlas_t: torch.Tensor,
    styles_t: torch.Tensor,
    boxes: torch.Tensor,
    tids: torch.Tensor,
    buckets: torch.Tensor,
    x_1d: torch.Tensor,
    y_1d: torch.Tensor,
    num_semantic_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = boxes.shape[0]
    dev = atlas_t.device
    h_strip = y_1d.shape[0]
    w_strip = x_1d.shape[0]

    active = (tids > 0) & (boxes[:, :, 2] > 0.5) & (boxes[:, :, 3] > 0.5)
    w_safe = boxes[:, :, 2:3].clamp_min(1.0)
    h_safe = boxes[:, :, 3:4].clamp_min(1.0)

    u_all = (x_1d[None, None, :] - boxes[:, :, 0:1]) / w_safe
    in_x = active[:, :, None] & (u_all >= 0.0) & (u_all < 1.0)
    any_x = in_x.any(dim=1)
    slot_at_x = in_x.int().argmax(dim=1)

    u_sel = torch.gather(u_all, 1, slot_at_x[:, None, :]).squeeze(1)
    y0_sel = torch.gather(boxes[:, :, 1], 1, slot_at_x)
    ws_sel = torch.gather(w_safe.squeeze(-1), 1, slot_at_x)
    hs_sel = torch.gather(h_safe.squeeze(-1), 1, slot_at_x)
    tid_sel = torch.gather(tids, 1, slot_at_x)
    _ = buckets  # retained for signature compatibility; continuous aspect blend eliminates 0.78 cliff

    v_grid = (y_1d[None, :, None] - y0_sel[:, None, :]) / hs_sel[:, None, :]
    valid_px = any_x[:, None, :] & (v_grid >= 0.0) & (v_grid < 1.0)

    # Continuous aspect blend between Bucket 0 (aspect=1.0) and Bucket 1 (aspect=0.58)
    asp_sel = ws_sel / hs_sel
    alpha_cond = ((0.94 - asp_sel) / (0.94 - 0.58)).clamp(0.0, 1.0)[:, None, :]

    atlas_1d = atlas_t.reshape(-1)
    glyph_base_0 = (styles_t[:, None] * 254 + tid_sel) * 4096
    bucket_stride = 8 * 254 * 4096

    fx = torch.clamp(u_sel[:, None, :].expand(bsz, h_strip, w_strip) * 64.0 - 0.5, 0.0, 63.0)
    fy = torch.clamp(v_grid * 64.0 - 0.5, 0.0, 63.0)

    x0 = fx.long()
    y0 = fy.long()
    x1 = (x0 + 1).clamp_max(63)
    y1 = (y0 + 1).clamp_max(63)
    wx = fx - x0.float()
    wy = fy - y0.float()

    base0_row0 = glyph_base_0[:, None, :] + y0 * 64
    base0_row1 = glyph_base_0[:, None, :] + y1 * 64

    v0_00 = torch.take(atlas_1d, base0_row0 + x0).float()
    v0_01 = torch.take(atlas_1d, base0_row0 + x1).float()
    v0_10 = torch.take(atlas_1d, base0_row1 + x0).float()
    v0_11 = torch.take(atlas_1d, base0_row1 + x1).float()
    sdf_u8_0 = (
        (1.0 - wy) * ((1.0 - wx) * v0_00 + wx * v0_01)
        + wy * ((1.0 - wx) * v0_10 + wx * v0_11)
    )

    v1_00 = torch.take(atlas_1d, base0_row0 + bucket_stride + x0).float()
    v1_01 = torch.take(atlas_1d, base0_row0 + bucket_stride + x1).float()
    v1_10 = torch.take(atlas_1d, base0_row1 + bucket_stride + x0).float()
    v1_11 = torch.take(atlas_1d, base0_row1 + bucket_stride + x1).float()
    sdf_u8_1 = (
        (1.0 - wy) * ((1.0 - wx) * v1_00 + wx * v1_01)
        + wy * ((1.0 - wx) * v1_10 + wx * v1_11)
    )

    sdf_u8_interp = (1.0 - alpha_cond) * sdf_u8_0 + alpha_cond * sdf_u8_1
    sdf_px = (sdf_u8_interp - 128.0) / 16.0
    sdf_px = torch.where(valid_px, sdf_px, torch.full_like(sdf_px, -8.0))

    stroke_bin = torch.clamp(sdf_px + 0.5, 0.0, 1.0)
    stroke_half_strip = F.avg_pool2d(stroke_bin.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
    slot_x_half = slot_at_x[:, 0::2]
    any_x_half = any_x[:, 0::2] | any_x[:, 1::2]
    slot_ids = torch.arange(num_semantic_slots, device=dev, dtype=slot_x_half.dtype)[None, :, None]
    sel_all = ((slot_x_half[:, None, :] == slot_ids) & any_x_half[:, None, :]).float()
    strip_masks = stroke_half_strip[:, None, :, :] * sel_all[:, :, None, :]
    return sdf_px, strip_masks


def _compose_surface_tail_impl(
    surround_base: torch.Tensor,
    grille_stripes: torch.Tensor,
    hood_geom: torch.Tensor,
    bumper_depth: torch.Tensor,
    hood_h: torch.Tensor,
    plate_bg_rgb: torch.Tensor,
    plate_bg_nir: torch.Tensor,
    retro: torch.Tensor,
    ext_sdf_px: torch.Tensor,
    in_plate: torch.Tensor,
    in_ribbon: torch.Tensor,
    rim_geom: torch.Tensor,
    sd_plate_mm: torch.Tensor,
    yy_mm: torch.Tensor,
    d_left_bolt: torch.Tensor,
    d_right_bolt: torch.Tensor,
    h_left_bolt: torch.Tensor,
    h_right_bolt: torch.Tensor,
    d_bot_bolt_lg: torch.Tensor,
    h_bot_bolt_lg: torch.Tensor,
    slash_geom: torch.Tensor,
    carbon_weave: torch.Tensor,
    rust_streak_geom: torch.Tensor,
    c_yel: torch.Tensor,
    c_grn: torch.Tensor,
    rim_rgb: torch.Tensor,
    txt_rgb: torch.Tensor,
    txt_nir: torch.Tensor,
    txt_retro: torch.Tensor,
    c_el_grn: torch.Tensor,
    c_el_amb: torch.Tensor,
    c_red_ink: torch.Tensor,
    c_bolt: torch.Tensor,
    c_mud: torch.Tensor,
    c_salt: torch.Tensor,
    c_rust: torch.Tensor,
    graphic_rim: torch.Tensor,
    has_border: torch.Tensor,
    is_graphic: torch.Tensor,
    has_emboss: torch.Tensor,
    is_illum: torch.Tensor,
    is_temp: torch.Tensor,
    has_seal: torch.Tensor,
    frame_type: torch.Tensor,
    bot_bolt_mode: torch.Tensor,
    bolt_cap_r: torch.Tensor,
    bolt_h: torch.Tensor,
    seal_r: torch.Tensor,
    frame_h: torch.Tensor,
    frame_in: torch.Tensor,
    mud_lvl: torch.Tensor,
    slash_nir_alpha: torch.Tensor,
    plate_rough: torch.Tensor,
    frame_top_extra: torch.Tensor,
    rust_lvl: torch.Tensor,
    salt_dust: torch.Tensor,
    warm_el: torch.Tensor,
    mud_low: torch.Tensor,
    sdf_bias: torch.Tensor,
    emboss_h: torch.Tensor,
    emboss_bevel_w: torch.Tensor,
    chip_lvl: torch.Tensor,
    seal_h: torch.Tensor,
    chip_low: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    bsz = surround_base.shape[0]
    # Continuous SDF stroke-weight dilation/erosion (±0.40..0.90px, gentler 0.65x on dense upper Kanji) + paint chipping
    upper_row_scale = torch.where(yy_mm[None, None, :, :] < 62.0, 0.65, 1.0)
    chip_field = F.interpolate(chip_low, size=(EXT_H, EXT_W), mode="bilinear", align_corners=False)
    chip_erode = F.relu(chip_field - 0.56) * chip_lvl * 1.45
    sdf_eff = ext_sdf_px + sdf_bias * upper_row_scale - chip_erode

    stroke_alpha = torch.clamp(sdf_eff + 0.5, 0.0, 1.0)
    emboss_bevel = torch.clamp((sdf_eff + 0.65) / emboss_bevel_w.clamp_min(0.45), 0.0, 1.0)
    emboss_bevel = emboss_bevel * emboss_bevel * (3.0 - 2.0 * emboss_bevel)

    albedo = surround_base * grille_stripes
    nir = surround_base.mean(dim=1, keepdim=True) * grille_stripes
    hood_mask = hood_geom * (hood_h > 0.0).float()
    height_mm = bumper_depth * (1.0 - in_plate) + hood_h * hood_mask
    rough = 0.70 * (1.0 - in_plate) + plate_rough * in_plate

    yel_mask = (graphic_rim == 1)[:, None, None, None].float() * in_ribbon
    grn_mask = (graphic_rim == 2)[:, None, None, None].float() * in_ribbon
    plate_bg_rgb = plate_bg_rgb * (1.0 - yel_mask) + c_yel * yel_mask
    plate_bg_rgb = plate_bg_rgb * (1.0 - grn_mask) + c_grn * grn_mask
    plate_bg_nir = plate_bg_nir * (1.0 - 0.72 * grn_mask)

    albedo = albedo * (1.0 - in_plate) + plate_bg_rgb * in_plate
    nir = nir * (1.0 - in_plate) + plate_bg_nir * in_plate

    rim_bevel = rim_geom * has_border[:, None, None, None].float()
    paint_rim = rim_bevel * (~is_graphic)[:, None, None, None].float()
    albedo = albedo * (1.0 - paint_rim) + rim_rgb * paint_rim
    nir = nir * (1.0 - paint_rim) + txt_nir * paint_rim
    retro = retro * (1.0 - paint_rim) + txt_retro * paint_rim
    height_mm = height_mm + 1.4 * rim_bevel * in_plate

    s_mask = stroke_alpha * in_plate
    albedo = albedo * (1.0 - s_mask) + txt_rgb * s_mask
    nir = nir * (1.0 - s_mask) + txt_nir * s_mask
    retro = retro * (1.0 - s_mask) + txt_retro * s_mask
    rough = rough * (1.0 - s_mask) + 0.22 * s_mask
    height_mm = height_mm + emboss_h * emboss_bevel * in_plate * has_emboss[:, None, None, None].float()

    # Vertical bolt rust streaks below upper mounting bolts
    rust_alpha = (rust_streak_geom * rust_lvl * 0.78).clamp(0.0, 0.78)
    albedo = albedo * (1.0 - rust_alpha) + c_rust * rust_alpha
    nir = nir * (1.0 - 0.65 * rust_alpha) + 0.22 * (0.65 * rust_alpha)
    retro = retro * (1.0 - 0.85 * rust_alpha)
    rough = rough * (1.0 - rust_alpha) + 0.84 * rust_alpha

    illum_f = is_illum[:, None, None, None].float()
    el_col = (1.0 - warm_el) * c_el_grn + warm_el * c_el_amb
    glow_profile = torch.clamp((ext_sdf_px + 1.2) / 1.4, 0.0, 1.0) * in_plate
    emissive_rgb = el_col * glow_profile * illum_f * 2.1

    slash_mask = slash_geom * is_temp[:, None, None, None].float()
    albedo = albedo * (1.0 - 0.94 * slash_mask) + c_red_ink * (0.94 * slash_mask)
    nir = nir * (1.0 - slash_nir_alpha * slash_mask) + 0.78 * (slash_nir_alpha * slash_mask)
    retro = retro * (1.0 - 0.55 * slash_mask)
    occ_2d_ext = torch.maximum(slash_mask, 0.55 * rust_alpha)

    has_bolts = (bolt_cap_r > 0.1).float()
    no_seal_f = (~has_seal)[:, None, None, None].float()
    r_washer_mask = torch.clamp((bolt_cap_r - d_right_bolt) * 1.4, 0.0, 1.0) * has_bolts
    l_washer_mask = torch.clamp((bolt_cap_r - d_left_bolt) * 1.4, 0.0, 1.0) * has_bolts * no_seal_f
    r_hex_mask = torch.clamp((0.76 * bolt_cap_r - h_right_bolt) * 1.8, 0.0, 1.0) * has_bolts
    l_hex_mask = torch.clamp((0.76 * bolt_cap_r - h_left_bolt) * 1.8, 0.0, 1.0) * has_bolts * no_seal_f

    # JIS D 5705 4-bolt Large plate lower bolts: 85% raised M10 hex head + flange washer, 15% flush/unbolted hole
    has_bot_bolted = (bot_bolt_mode == 1)[:, None, None, None].float() * has_bolts
    has_bot_hole = (bot_bolt_mode == 2)[:, None, None, None].float()
    b_washer_mask = torch.clamp((bolt_cap_r - d_bot_bolt_lg) * 1.4, 0.0, 1.0) * has_bot_bolted
    b_hex_mask = torch.clamp((0.76 * bolt_cap_r - h_bot_bolt_lg) * 1.8, 0.0, 1.0) * has_bot_bolted

    upper_bolt_mask = torch.maximum(r_washer_mask, l_washer_mask)
    bolt_mask = torch.maximum(upper_bolt_mask, b_washer_mask)
    hex_mask = torch.maximum(torch.maximum(r_hex_mask, l_hex_mask), b_hex_mask)
    bolt_shade = (0.86 + 0.18 * hex_mask) * c_bolt
    albedo = albedo * (1.0 - bolt_mask) + bolt_shade * bolt_mask
    nir = nir * (1.0 - bolt_mask) + 0.68 * bolt_mask
    retro = retro * (1.0 - bolt_mask)
    rough = rough * (1.0 - bolt_mask) + (0.26 - 0.06 * hex_mask) * bolt_mask
    metal = (0.78 + 0.10 * hex_mask) * bolt_mask
    bolt_relief_h = bolt_h * (0.38 * bolt_mask + 0.62 * hex_mask)
    height_mm = torch.where(bolt_mask > 0.05, torch.maximum(height_mm, bolt_relief_h), height_mm)
    occ_2d_ext = torch.maximum(occ_2d_ext, upper_bolt_mask)

    bot_hole_mask = torch.clamp((4.125 - d_bot_bolt_lg) * 1.6, 0.0, 1.0) * has_bot_hole
    albedo = albedo * (1.0 - bot_hole_mask) + 0.07 * bot_hole_mask
    nir = nir * (1.0 - bot_hole_mask) + 0.06 * bot_hole_mask
    retro = retro * (1.0 - bot_hole_mask)
    rough = rough * (1.0 - bot_hole_mask) + 0.82 * bot_hole_mask
    height_mm = height_mm * (1.0 - bot_hole_mask) + bumper_depth * bot_hole_mask

    seal_f = has_seal[:, None, None, None].float()
    seal_mask = torch.clamp((seal_r - d_left_bolt) * 1.4, 0.0, 1.0) * seal_f
    seal_plateau = torch.clamp((seal_r + 0.4 - d_left_bolt) / 0.9, 0.0, 1.0) * seal_f
    seal_ring = 0.78 - 0.10 * torch.exp(-((d_left_bolt - 0.65 * seal_r) ** 2) / 1.8)
    albedo = albedo * (1.0 - seal_mask) + seal_ring.expand(bsz, 3, EXT_H, EXT_W) * seal_mask
    nir = nir * (1.0 - seal_mask) + 0.72 * seal_mask
    retro = retro * (1.0 - seal_mask)
    rough = rough * (1.0 - seal_mask) + 0.20 * seal_mask
    metal = torch.maximum(metal, 0.88 * seal_mask)
    height_mm = torch.where(seal_plateau > 0.02, torch.maximum(height_mm, seal_h * seal_plateau), height_mm)
    occ_2d_ext = torch.maximum(occ_2d_ext, seal_mask)

    f_f = (frame_type > 0)[:, None, None, None].float()
    bot_extra = ((frame_type == 3)[:, None, None, None].float()) * 4.5 * (yy_mm[None, None, :, :] > 120.0).float()
    top_extra = frame_top_extra * (yy_mm[None, None, :, :] < 34.0).float()
    in_frame = (
        torch.clamp((4.0 - sd_plate_mm) * 1.3, 0.0, 1.0)
        * torch.clamp((sd_plate_mm + frame_in + bot_extra + top_extra) * 1.3, 0.0, 1.0)
        * f_f
    )
    is_chrome = (frame_type == 1)[:, None, None, None].float()
    is_carbon = (frame_type == 4)[:, None, None, None].float()
    f_col = (is_chrome * 0.82 + is_carbon * carbon_weave + (1.0 - is_chrome - is_carbon) * 0.10).expand(bsz, 3, EXT_H, EXT_W)
    albedo = albedo * (1.0 - in_frame) + f_col * in_frame
    nir = nir * (1.0 - in_frame) + (is_chrome * 0.78 + (1.0 - is_chrome) * 0.12) * in_frame
    retro = retro * (1.0 - in_frame)
    rough = rough * (1.0 - in_frame) + (is_chrome * 0.09 + (1.0 - is_chrome) * 0.58) * in_frame
    metal = torch.maximum(metal, is_chrome * 0.94 * in_frame)
    height_mm = torch.where(in_frame > 0.02, torch.maximum(height_mm, frame_h * in_frame), height_mm)
    occ_2d_ext = torch.maximum(occ_2d_ext, in_frame)

    mud_noise = F.interpolate(mud_low, size=(EXT_H, EXT_W), mode="bilinear", align_corners=False)
    lower_bias = torch.clamp((yy_mm[None, None, :, :] + 20.0) / 185.0, 0.2, 1.0)
    mud_alpha = torch.clamp((mud_noise * lower_bias - (1.0 - 0.85 * mud_lvl)) * 3.2, 0.0, 0.92) * (mud_lvl > 1e-3).float()
    grime_rgb = (1.0 - salt_dust) * c_mud + salt_dust * c_salt
    grime_nir = (1.0 - salt_dust) * 0.24 + salt_dust * 0.68
    albedo = torch.clamp(albedo * (1.0 - mud_alpha) + grime_rgb * mud_alpha, 0.0, 1.0)
    nir = torch.clamp(nir * (1.0 - mud_alpha) + grime_nir * mud_alpha, 0.0, 1.0)
    retro = torch.clamp(retro * (1.0 - 0.95 * mud_alpha), 0.0, 1.0)
    rough = torch.clamp(rough * (1.0 - mud_alpha) + 0.88 * mud_alpha, 0.05, 1.0)
    metal = torch.clamp(metal, 0.0, 1.0)
    occ_2d_ext = torch.maximum(occ_2d_ext, mud_alpha)

    height_mm = torch.clamp(height_mm, -10.0, 18.0)
    height_u8 = torch.clamp(torch.round((height_mm + 10.0) * (255.0 / 28.0)), 0.0, 255.0).to(torch.uint8)
    occ_half = F.avg_pool2d(
        occ_2d_ext[:, :, PAD_Y : PAD_Y + PLATE_H, PAD_X : PAD_X + PLATE_W],
        kernel_size=2,
        stride=2,
    )
    return albedo, nir, height_mm, height_u8, retro, rough, metal, emissive_rgb, s_mask, occ_half


_SPLAT_CUDA_FNS: dict[int, Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {}
_COMPOSE_CUDA_FN: Callable[..., tuple[torch.Tensor, ...]] | None = None
_GRAPHIC_CUDA_FN: Callable[..., torch.Tensor] | None = None


def _run_graphic_patterns(*args: torch.Tensor) -> torch.Tensor:
    global _GRAPHIC_CUDA_FN
    if args[0].is_cuda:
        if _GRAPHIC_CUDA_FN is None:
            _GRAPHIC_CUDA_FN = torch.compile(_graphic_patterns_impl, dynamic=True)
        return _GRAPHIC_CUDA_FN(*args)
    return _graphic_patterns_impl(*args)


def _run_splat_strip(
    atlas_t: torch.Tensor,
    styles_t: torch.Tensor,
    boxes: torch.Tensor,
    tids: torch.Tensor,
    buckets: torch.Tensor,
    x_1d: torch.Tensor,
    y_1d: torch.Tensor,
    num_semantic_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if atlas_t.is_cuda:
        fn = _SPLAT_CUDA_FNS.get(num_semantic_slots)
        if fn is None:
            fn = torch.compile(
                lambda a, st, bx, td, bk, x1, y1: _splat_strip_impl(
                    a, st, bx, td, bk, x1, y1, num_semantic_slots
                ),
                dynamic=True,
            )
            _SPLAT_CUDA_FNS[num_semantic_slots] = fn
        return fn(atlas_t, styles_t, boxes, tids, buckets, x_1d, y_1d)
    return _splat_strip_impl(atlas_t, styles_t, boxes, tids, buckets, x_1d, y_1d, num_semantic_slots)


def _run_compose_surface_tail(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
    global _COMPOSE_CUDA_FN
    if args[0].is_cuda:
        if _COMPOSE_CUDA_FN is None:
            _COMPOSE_CUDA_FN = torch.compile(_compose_surface_tail_impl, dynamic=True)
        return _COMPOSE_CUDA_FN(*args)
    return _compose_surface_tail_impl(*args)
