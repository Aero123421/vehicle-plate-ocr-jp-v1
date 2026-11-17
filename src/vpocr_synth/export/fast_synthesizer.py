"""High-throughput direct `(T, 64, 128, 3)` `CropSequenceSample` synthesizer (>= 80 seq/s on CPU)."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import math
import multiprocessing as mp
import os
import random
from typing import Any, ClassVar
import cv2
import numpy as np
from PIL import Image, ImageDraw

from vpocr_synth.camera.illumination import IlluminationSimulator
from vpocr_synth.camera.models import IlluminationType
from vpocr_synth.crop.extractor import CropExtractor
from vpocr_synth.crop.models import CROP_HEIGHT, CROP_WIDTH, CropSequenceSample
from vpocr_synth.crop.readability import SlotReadabilityMapper
from vpocr_synth.motion.camera import PinholeCamera
from vpocr_synth.motion.generator import compute_rear_plate_pose
from vpocr_synth.motion.models import CameraParams, Pose3D, ScenarioType
from vpocr_synth.motion.trajectory import create_trajectory
from vpocr_synth.plate.geometry import PlateGeometry
from vpocr_synth.plate.glyph_dies import DieStyleParams, MLITVectorDieEngine
from vpocr_synth.plate.models import PlateConfig, PlateSize, PlateText, PlateType
from vpocr_synth.plate.renderer import PlateRenderer
from vpocr_synth.plate.sampler import PlateSampler
from vpocr_synth.plate.stratified_sampler import StratifiedPlateSampler


class FastPlatePatchRenderer:
    """Glyph-cached OpenCV/NumPy canonical plate patch renderer (~0.2ms per plate)."""

    FONT_WEIGHT_PATHS: ClassVar[tuple[str, ...]] = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
    )
    DIE_VARIANTS: ClassVar[tuple[str, ...]] = (
        "neutral",
        "rounded_thin",
        "rounded_thick",
        "jp_stamped",
    )
    _DEFAULT_BOLD_PATH: ClassVar[str] = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    _GLYPH_MASK_CACHE: ClassVar[dict[tuple[str, str, str], np.ndarray]] = {}
    _ELLIPSE_3X3: ClassVar[np.ndarray] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    @classmethod
    def _render_jp_stamped_digit(cls, char: str, die_variant: str = "jp_stamped") -> np.ndarray | None:
        """Render procedural Japanese MLIT stamped vector digits/Latin via `MLITVectorDieEngine`."""
        if not MLITVectorDieEngine.supports(char) or char in ("-", "・"):
            return None
        params = DieStyleParams(
            stroke_jitter=-0.05 if die_variant == "rounded_thin" else (0.06 if die_variant == "rounded_thick" else 0.0),
            closed_four=(die_variant != "rounded_thin"),
            flat_top_three=(die_variant in ("jp_stamped", "rounded_thick")),
            curved_nine=True,
            curled_six=True,
            hooked_seven=(die_variant != "rounded_thin"),
        )
        sdf_u8 = MLITVectorDieEngine.render_sdf_u8(char, size=96, aspect_w_over_h=0.62, params=params)
        # Convert signed distance (128 = contour, 16 levels/px) to smooth anti-aliased alpha mask
        alpha = np.clip((sdf_u8.astype(np.float32) - 128.0) * (255.0 / 24.0) + 128.0, 0.0, 255.0).astype(np.uint8)
        ys, xs = np.nonzero(alpha > 16)
        if ys.size == 0:
            return alpha
        return alpha[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1].copy()

    @classmethod
    def _get_glyph_mask(cls, font_path: str, char: str, die_variant: str = "neutral") -> np.ndarray:
        key = (font_path, die_variant, char)
        cached = cls._GLYPH_MASK_CACHE.get(key)
        if cached is not None:
            return cached

        if die_variant in ("jp_stamped", "rounded_thin", "rounded_thick") and MLITVectorDieEngine.supports(char) and char not in ("-", "・"):
            vec_mask = cls._render_jp_stamped_digit(char, die_variant=die_variant)
            if vec_mask is not None:
                cls._GLYPH_MASK_CACHE[key] = vec_mask
                return vec_mask

        font = PlateRenderer._get_font(font_path, 96, 0)
        bbox = font.getbbox(char)
        gw = max(2, bbox[2] - bbox[0])
        gh = max(2, bbox[3] - bbox[1])
        pad = 8
        img = Image.new("L", (gw + 2 * pad, gh + 2 * pad), 0)
        draw = ImageDraw.Draw(img)
        draw.text((pad - bbox[0], pad - bbox[1]), char, font=font, fill=255)
        raw = np.array(img, dtype=np.uint8)

        if die_variant == "rounded_thin":
            raw = cv2.erode(raw, cls._ELLIPSE_3X3, iterations=1)
            raw = cv2.GaussianBlur(raw, (3, 3), 0.65)
        elif die_variant == "rounded_thick":
            raw = cv2.dilate(raw, cls._ELLIPSE_3X3, iterations=1)
            raw = cv2.GaussianBlur(raw, (3, 3), 0.80)
        elif die_variant == "jp_stamped":
            # Stamped metal die rounding: morphological close + soft rounded stroke caps
            raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cls._ELLIPSE_3X3)
            raw = cv2.GaussianBlur(raw, (5, 5), 0.95)
            raw = np.clip((raw.astype(np.float32) - 96.0) * (255.0 / 96.0), 0.0, 255.0).astype(np.uint8)

        ys, xs = np.nonzero(raw > 12)
        if ys.size > 0:
            mask = raw[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1].copy()
        else:
            mask = raw[pad : pad + gh, pad : pad + gw].copy()

        cls._GLYPH_MASK_CACHE[key] = mask
        return mask

    @classmethod
    def _draw_graphic_background(
        cls,
        canvas: np.ndarray,
        pattern: str,
        seed_val: int,
    ) -> None:
        """Fast OpenCV procedural floral/geometric/mountain/wave/rainbow graphic plate background."""
        h_px, w_px = canvas.shape[:2]
        overlay = canvas.copy()
        p_rng = random.Random(seed_val + 31337)

        if pattern == "flower":
            petal_cols = [(244, 168, 192), (248, 185, 206), (235, 148, 180), (195, 224, 246)]
            centers = (
                (0.17, 0.70, 0.14),
                (0.31, 0.81, 0.12),
                (0.11, 0.84, 0.10),
                (0.47, 0.75, 0.13),
                (0.66, 0.82, 0.11),
                (0.82, 0.74, 0.11),
            )
            for ux, uy, ur in centers:
                bx, by = int(w_px * ux), int(h_px * uy)
                cr = max(3, int(h_px * ur))
                pr = max(2, int(cr * 0.58))
                col = p_rng.choice(petal_cols)
                for a in range(5):
                    ang = a * (2.0 * math.pi / 5.0) + 0.2
                    px = int(round(bx + math.cos(ang) * cr))
                    py = int(round(by + math.sin(ang) * cr))
                    cv2.circle(overlay, (px, py), pr, col, -1, lineType=cv2.LINE_AA)
                cv2.circle(overlay, (bx, by), max(1, int(cr * 0.32)), (250, 218, 92), -1, lineType=cv2.LINE_AA)
        elif pattern == "geometric":
            geo_cols = [(232, 118, 112), (245, 188, 96), (112, 204, 138), (108, 168, 235), (178, 132, 224)]
            band_w = max(6, int(w_px * 0.11))
            slant = int(h_px * 0.34)
            y_top = int(h_px * 0.48)
            for i in range(6):
                x0 = int(i * band_w * 1.15) - slant // 2
                pts = np.array(
                    [[x0, h_px], [x0 + band_w, h_px], [x0 + band_w + slant, y_top], [x0 + slant, y_top]],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(overlay, pts, geo_cols[i % len(geo_cols)], lineType=cv2.LINE_AA)
        elif pattern == "mountain":
            cv2.circle(
                overlay,
                (int(w_px * 0.24), int(h_px * 0.38)),
                max(5, int(h_px * 0.18)),
                (250, 166, 128),
                -1,
                lineType=cv2.LINE_AA,
            )
            pts_far = np.array(
                [[0, h_px], [0, int(h_px * 0.62)], [int(w_px * 0.32), int(h_px * 0.42)],
                 [int(w_px * 0.62), int(h_px * 0.58)], [int(w_px * 0.86), int(h_px * 0.45)], [w_px, int(h_px * 0.60)], [w_px, h_px]],
                dtype=np.int32,
            )
            cv2.fillPoly(overlay, [pts_far], (168, 182, 218), lineType=cv2.LINE_AA)
            pts_near = np.array(
                [[0, h_px], [int(w_px * 0.18), int(h_px * 0.62)], [int(w_px * 0.48), int(h_px * 0.72)],
                 [int(w_px * 0.76), int(h_px * 0.56)], [w_px, int(h_px * 0.70)], [w_px, h_px]],
                dtype=np.int32,
            )
            cv2.fillPoly(overlay, [pts_near], (116, 178, 148), lineType=cv2.LINE_AA)
        elif pattern == "wave":
            xs = np.linspace(0, w_px, 18, dtype=np.float32)
            for idx_w, col in enumerate(((120, 182, 226), (142, 206, 232), (176, 226, 224))):
                ys = h_px * (0.56 + 0.12 * idx_w) + np.sin(xs * 0.05 + idx_w * 1.3) * (h_px * 0.08)
                poly = np.vstack(
                    [np.array([[0.0, float(h_px)]], dtype=np.float32),
                     np.stack([xs, ys], axis=1),
                     np.array([[float(w_px), float(h_px)]], dtype=np.float32)]
                ).astype(np.int32)
                cv2.fillPoly(overlay, [poly], col, lineType=cv2.LINE_AA)
        else:  # rainbow / pastel arcs
            arc_cols = ((238, 128, 128), (246, 182, 112), (246, 224, 120), (132, 208, 146), (126, 178, 238))
            cx, cy = int(-w_px * 0.06), int(h_px * 0.96)
            base_r = int(w_px * 0.82)
            step_r = max(3, int(h_px * 0.07))
            for idx_a, col in enumerate(arc_cols):
                rad = max(6, base_r - idx_a * step_r)
                cv2.ellipse(overlay, (cx, cy), (rad, int(rad * 0.68)), 0, 275, 360, col, step_r, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.48, canvas, 0.52, 0.0, dst=canvas)

    @classmethod
    def render_patch(
        cls,
        plate_text: PlateText,
        config: PlateConfig,
        scale: float = 0.62,
        *,
        temporary_plate: bool = False,
        frame_style: str | None = None,
        font_variant: tuple[str, str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render canonical RGB plate patch `(H, W, 3) uint8` and `slot_occlusion_12 (12,) float32`."""
        layout = PlateGeometry.calculate_layout(plate_text, config.plate_size)
        w_px = max(64, int(round(layout.width_mm * scale)))
        h_px = max(32, int(round(layout.height_mm * scale)))

        if config.seed is not None:
            seed_base = int(config.seed)
        else:
            key_bytes = f"{plate_text.region}|{plate_text.classification}|{plate_text.hiragana}|{plate_text.serial}".encode("utf-8")
            seed_base = int.from_bytes(hashlib.sha256(key_bytes).digest()[:8], "little") & 0x7FFFFFFF
        style_hash = (seed_base * 2654435761) & 0xFFFFFFFF

        if temporary_plate:
            # 仮ナンバー (臨時運行許可番号標): white background, charcoal-black text/border, crimson diagonal stripe
            bg_rgb = (246, 247, 244)
            txt_rgb = (26, 28, 30)
            border_rgb = (32, 34, 36)
            bolt_inner = (208, 212, 214)
            bolt_rim = (128, 132, 136)
        else:
            palette = PlateRenderer.COLOR_PALETTES.get(
                config.plate_type, PlateRenderer.COLOR_PALETTES[PlateType.STANDARD_PRIVATE]
            )
            bg_rgb = palette["bg"]
            txt_rgb = palette["text"]
            border_rgb = palette["border"]
            bolt_inner = palette["bolt_inner"]
            bolt_rim = palette["bolt_rim"]
            if config.plate_type == PlateType.STANDARD_COMMERCIAL and not config.is_graphic:
                g_sel = (style_hash >> 15) % 100
                if g_sel < 42:
                    bg_rgb = (16 + (style_hash % 12), 54 + ((style_hash >> 4) % 30), 26 + ((style_hash >> 8) % 20))
                elif g_sel < 72:
                    bg_rgb = (32 + (style_hash % 24), 64 + ((style_hash >> 4) % 30), 28 + ((style_hash >> 8) % 22))
                else:
                    bg_rgb = (44 + (style_hash % 26), 68 + ((style_hash >> 4) % 28), 54 + ((style_hash >> 8) % 30))
            if config.is_graphic or config.plate_type == PlateType.GRAPHIC:
                if config.plate_type == PlateType.KEI_PRIVATE:
                    border_rgb = (248, 198, 28)
                elif config.plate_type == PlateType.STANDARD_COMMERCIAL:
                    border_rgb = (23, 86, 48)
                else:
                    border_rgb = (196, 202, 198)

        canvas = np.empty((h_px, w_px, 3), dtype=np.uint8)
        canvas[:, :, 0] = bg_rgb[0]
        canvas[:, :, 1] = bg_rgb[1]
        canvas[:, :, 2] = bg_rgb[2]

        # Richer floral/geometric/mountain/wave/rainbow graphic plate background
        if not temporary_plate and (config.is_graphic or config.plate_type == PlateType.GRAPHIC):
            pat = config.graphic_pattern or ("flower", "geometric", "mountain", "wave", "rainbow")[style_hash % 5]
            cls._draw_graphic_background(canvas, pat, int(config.pattern_seed if config.pattern_seed is not None else seed_base))

        # Temporary plate (仮ナンバー) bold crimson-red diagonal stripe from upper-right toward lower-left
        if temporary_plate:
            stripe_w = max(3, int(round(h_px * 0.115)))
            cv2.line(
                canvas,
                (int(round(w_px * 0.78)), 1),
                (int(round(w_px * 0.24)), h_px - 2),
                (214, 28, 32),
                stripe_w,
                lineType=cv2.LINE_AA,
            )

        # Outer pressed border
        inset = max(1, int(round(layout.border_inset_mm * scale)))
        bw = max(1, int(round(layout.border_width_mm * scale)))
        if config.with_border:
            cv2.rectangle(canvas, (inset, inset), (w_px - 1 - inset, h_px - 1 - inset), border_rgb, bw, lineType=cv2.LINE_AA)

        # Multi-font weight & stamped-die stroke morphology selection
        if font_variant is not None:
            font_path, die_variant = font_variant
        else:
            cfg_font = config.font_path or cls._DEFAULT_BOLD_PATH
            if cfg_font == cls._DEFAULT_BOLD_PATH:
                # 50% Bold, 25% Medium, 25% Black
                w_sel = (style_hash >> 3) % 4
                font_path = cls.FONT_WEIGHT_PATHS[0 if w_sel < 2 else (1 if w_sel == 2 else 2)]
            else:
                font_path = cfg_font
            die_variant = cls.DIE_VARIANTS[(style_hash >> 7) % len(cls.DIE_VARIANTS)]

        text_mask = np.zeros((h_px, w_px), dtype=np.float32)

        for slot in layout.slots:
            sx = slot.x * scale
            sy = slot.y * scale
            sw = slot.width * scale
            sh = slot.height * scale

            if slot.is_dot:
                cx = int(round(sx + sw * 0.5))
                cy = int(round(sy + sh * 0.5))
                rad = max(1, int(round(sw * 0.5)))
                cv2.circle(text_mask, (cx, cy), rad, 1.0, -1, lineType=cv2.LINE_AA)
            elif slot.char == "-":
                x0, y0 = int(round(sx)), int(round(sy))
                x1, y1 = max(x0 + 2, int(round(sx + sw))), max(y0 + 1, int(round(sy + sh)))
                text_mask[max(0, y0) : min(h_px, y1), max(0, x0) : min(w_px, x1)] = 1.0
            else:
                g_mask = cls._get_glyph_mask(font_path, slot.char, die_variant)
                gh, gw = g_mask.shape[:2]
                th = max(2, int(round(sh * 0.94)))
                tw = max(2, int(round(sw * 0.94)))
                if slot.field == "serial" and slot.char == "1":
                    aspect = gw / float(max(1, gh))
                    tw = min(tw, max(int(round(sw * 0.5)), int(round(th * aspect * 1.05))))
                resized = cv2.resize(g_mask, (tw, th), interpolation=cv2.INTER_AREA if tw < gw else cv2.INTER_LINEAR)
                ox = int(round(sx + (sw - tw) * 0.5))
                oy = int(round(sy + (sh - th) * 0.5))
                x0, y0 = max(0, ox), max(0, oy)
                x1, y1 = min(w_px, ox + tw), min(h_px, oy + th)
                if x1 > x0 and y1 > y0:
                    patch = resized[y0 - oy : y1 - oy, x0 - ox : x1 - ox].astype(np.float32) / 255.0
                    text_mask[y0:y1, x0:x1] = np.maximum(text_mask[y0:y1, x0:x1], patch)

        canvas_f = canvas.astype(np.float32)
        if config.is_illuminated and not temporary_plate:
            halo = cv2.GaussianBlur(text_mask, (7, 7), 2.0)[:, :, np.newaxis]
            if config.plate_type in (PlateType.KEI_PRIVATE, PlateType.KEI_COMMERCIAL):
                glow_col = np.array([255.0, 232.0, 72.0], dtype=np.float32)
                core_col = np.array([255.0, 250.0, 175.0], dtype=np.float32)
            else:
                glow_col = np.array([60.0, 245.0, 125.0], dtype=np.float32)
                core_col = np.array([165.0, 255.0, 200.0], dtype=np.float32)
            canvas_f = canvas_f * (1.0 - 0.75 * halo) + glow_col * (0.75 * halo)
            tm3 = text_mask[:, :, np.newaxis]
            canvas_f = canvas_f * (1.0 - tm3) + core_col * tm3
        else:
            # Subtle 3D stamped emboss highlight/shadow along stroke edges
            if config.with_emboss and config.emboss_intensity > 0.0:
                eb = float(config.emboss_intensity) * 0.32
                hl = np.zeros_like(text_mask)
                hl[:-1, :-1] = np.maximum(0.0, text_mask[1:, 1:] - text_mask[:-1, :-1])
                canvas_f += (hl * eb * 55.0)[:, :, np.newaxis]
            if config.weathering > 0.0:
                text_mask = text_mask * (1.0 - 0.25 * float(config.weathering))
            tm3 = text_mask[:, :, np.newaxis]
            t_col = np.array(txt_rgb, dtype=np.float32)
            canvas_f = canvas_f * (1.0 - tm3) + t_col * tm3

        rng = np.random.default_rng(seed_base + 5113)

        # Subtle surface dirt / grime patina & bolt rust streaks when weathering > 0
        if config.weathering > 0.0:
            w_lvl = float(np.clip(config.weathering, 0.0, 1.0))
            grime = cv2.resize(rng.uniform(0.0, 1.0, (6, 12)).astype(np.float32), (w_px, h_px))
            patina = np.array([182.0, 174.0, 152.0], dtype=np.float32)
            g_alpha = ((0.05 + 0.20 * w_lvl) * grime)[:, :, np.newaxis]
            canvas_f = canvas_f * (1.0 - g_alpha) + patina * g_alpha
            for bolt in layout.bolts:
                bx = int(round(bolt.x * scale))
                by = int(round(bolt.y * scale))
                s_len = min(h_px - by, max(3, int(round((12.0 + 22.0 * w_lvl) * scale))))
                if s_len > 2 and 1 <= bx < w_px - 1:
                    decay = np.linspace(0.32 * w_lvl, 0.0, s_len, dtype=np.float32)[:, np.newaxis]
                    rust = np.array([128.0, 86.0, 48.0], dtype=np.float32)
                    canvas_f[by : by + s_len, bx - 1 : bx + 2] = (
                        canvas_f[by : by + s_len, bx - 1 : bx + 2] * (1.0 - decay[:, :, np.newaxis])
                        + rust * decay[:, :, np.newaxis]
                    )

        canvas = np.clip(canvas_f, 0.0, 255.0).astype(np.uint8)

        # Bolts (2-bolt medium / 4-bolt JIS D 5705 M10 large) & optional rear aluminum seal (封印)
        if config.with_bolt_holes:
            can_have_seal = (
                not temporary_plate
                and config.with_seal
                and config.plate_type in (PlateType.STANDARD_PRIVATE, PlateType.STANDARD_COMMERCIAL, PlateType.GRAPHIC)
            )
            lower_unbolted = layout.size == PlateSize.LARGE and ((style_hash >> 23) % 100 < 15)
            for b_idx, bolt in enumerate(layout.bolts):
                is_top_left = b_idx == 0
                is_lower = b_idx >= 2
                bx = int(round(bolt.x * scale))
                by = int(round(bolt.y * scale))
                if is_top_left and can_have_seal:
                    sr = max(3, int(round(12.5 * scale)))
                    cv2.circle(canvas, (min(w_px - 1, bx + 1), min(h_px - 1, by + 1)), sr, (24, 30, 26), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (bx, by), sr, (206, 211, 216), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (max(0, bx - 1), max(0, by - 1)), max(2, int(sr * 0.72)), (236, 240, 244), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (bx, by), sr, (132, 138, 145), 1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (bx, by), max(1, int(sr * 0.42)), (152, 158, 165), 1, lineType=cv2.LINE_AA)
                elif is_lower and lower_unbolted:
                    br = max(1, int(round(bolt.radius * scale)))
                    cv2.circle(canvas, (bx, by), br, (26, 28, 30), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (bx, by), br, (78, 82, 84), 1, lineType=cv2.LINE_AA)
                else:
                    w_mult = 0.96 if is_lower else (1.35 if layout.size == PlateSize.LARGE else 1.15)
                    wr = max(2, int(round(bolt.radius * w_mult * scale)))
                    hr = max(1, int(round(wr * 0.78)))
                    cv2.circle(canvas, (min(w_px - 1, bx + 1), min(h_px - 1, by + 1)), wr, (22, 28, 24), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (bx, by), wr, bolt_inner, -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (bx, by), wr, bolt_rim, 1, lineType=cv2.LINE_AA)
                    hex_pts = np.array(
                        [
                            [int(round(bx + hr * math.cos(k * math.pi / 3.0))), int(round(by + hr * math.sin(k * math.pi / 3.0)))]
                            for k in range(6)
                        ],
                        dtype=np.int32,
                    )
                    hex_col = tuple(min(255, int(c) + 22) for c in bolt_inner)
                    cv2.fillConvexPoly(canvas, hex_pts, hex_col, lineType=cv2.LINE_AA)
                    cv2.polylines(canvas, [hex_pts], True, bolt_rim, 1, lineType=cv2.LINE_AA)

        slot_occ_12 = np.zeros(12, dtype=np.float32)

        # Optional chrome / black dealer plate frame (1-3px metallic/dark rim around border)
        if config.plate_frame or frame_style is not None:
            eff_frame = frame_style or ("chrome" if (style_hash >> 11) % 2 == 0 else "black")
            fw = max(1, min(3, int(round(3.6 * scale))))
            if eff_frame == "chrome":
                cv2.rectangle(canvas, (0, 0), (w_px - 1, h_px - 1), (214, 220, 228), fw)
                cv2.line(canvas, (0, 0), (w_px - 1, 0), (248, 250, 255), 1)
                cv2.line(canvas, (0, 0), (0, h_px - 1), (242, 246, 252), 1)
                if fw >= 2:
                    cv2.rectangle(canvas, (fw, fw), (w_px - 1 - fw, h_px - 1 - fw), (108, 114, 122), 1)
            else:
                cv2.rectangle(canvas, (0, 0), (w_px - 1, h_px - 1), (34, 36, 40), fw)
                cv2.line(canvas, (1, 0), (w_px - 2, 0), (76, 80, 86), 1)

        if config.mud_level > 0.0:
            m_lvl = float(np.clip(config.mud_level, 0.0, 1.0))
            yy = np.linspace(0.0, 1.0, h_px, dtype=np.float32)[:, np.newaxis]
            bias = np.clip((yy - (0.82 - 0.42 * m_lvl)) / 0.35, 0.0, 1.0)
            noise = cv2.resize(rng.uniform(0.0, 1.0, (8, 16)).astype(np.float32), (w_px, h_px))
            m_alpha = np.clip(noise * bias * (0.45 + 0.55 * m_lvl), 0.0, 0.90)
            mud_col = np.array([104.0, 88.0, 68.0], dtype=np.float32)
            canvas = np.clip(
                canvas.astype(np.float32) * (1.0 - m_alpha[:, :, np.newaxis]) + mud_col * m_alpha[:, :, np.newaxis],
                0.0,
                255.0,
            ).astype(np.uint8)
            slot_occ_12[7:] = np.maximum(slot_occ_12[7:], float(np.mean(m_alpha[int(h_px * 0.5) :, :])))

        if config.persistent_occluder_slot is not None and 0 <= config.persistent_occluder_slot < 12:
            norm_boxes = SlotReadabilityMapper.get_normalized_slot_boxes(plate_text, config.plate_size)
            u0, v0, u1, v1 = norm_boxes[config.persistent_occluder_slot]
            pad_x = max(2, int(round(3.5 * scale)))
            pad_y = max(2, int(round(3.5 * scale)))
            x0 = max(0, int(math.floor(u0 * w_px)) - pad_x)
            y0 = max(0, int(math.floor(v0 * h_px)) - pad_y)
            x1 = min(w_px, int(math.ceil(u1 * w_px)) + pad_x)
            y1 = min(h_px, int(math.ceil(v1 * h_px)) + pad_y)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (72, 60, 46), -1)
            slot_occ_12[config.persistent_occluder_slot] = 1.0

        return canvas, slot_occ_12


@dataclass
class FastSynthesizerConfig:
    """Distribution parameters for high-throughput `FastCropSequenceSynthesizer`."""

    use_stratified: bool = True
    p_multicam: float = 0.24
    p_complementary_glare: float = 0.11
    p_complementary_cols: float = 0.10
    p_gate_bars_wipe: float = 0.08
    p_steep_overhead_lowres: float = 0.10
    p_bumper_shadow: float = 0.07
    p_unreadable_occluded: float = 0.08
    p_unreadable_side: float = 0.04
    p_negative_signboard: float = 0.04
    p_track_mixed: float = 0.09
    p_garbage_frame: float = 0.07
    p_edge_truncation: float = 0.08
    default_num_frames: int = 8


def _init_synth_worker() -> None:
    cv2.setNumThreads(0)


def _worker_synth_chunk(args: tuple[FastSynthesizerConfig, int, int, int | None, str, list[str] | None]) -> list[CropSequenceSample]:
    cv2.setNumThreads(0)
    cfg, start_seed, count, num_frames, mode, emit_variants = args
    synth = FastCropSequenceSynthesizer(cfg)
    return synth.synthesize_batch(
        num_sequences=count,
        start_seed=start_seed,
        num_frames=num_frames,
        mode=mode,
        emit_variants=emit_variants,
    )


class FastCropSequenceSynthesizer:
    """High-throughput direct `(T, 64, 128, 3)` sequence synthesizer (>= 80 seq/sec/core on CPU)."""

    SIGNBOARD_TEXTS: ClassVar[tuple[str, ...]] = (
        "安全確認",
        "徐行注意",
        "自動ゲート",
        "入口専用",
        "駐車禁止",
        "一時停止",
        "No.1入口",
        "No.2出口",
        "P空車2F",
        "積載5000",
        "月極108",
        "制限10km",
    )

    def __init__(self, config: FastSynthesizerConfig | None = None) -> None:
        self.config = config or FastSynthesizerConfig()
        self.base_sampler = PlateSampler()
        self.stratified_sampler = StratifiedPlateSampler(base_sampler=self.base_sampler)
        self.plate_sampler = (
            self.stratified_sampler if self.config.use_stratified else self.base_sampler
        )

    def synthesize_batch_parallel(
        self,
        num_sequences: int,
        start_seed: int = 0,
        num_frames: int | None = None,
        mode: str = "mixed",
        emit_variants: list[str] | None = None,
        num_workers: int | None = None,
    ) -> list[CropSequenceSample]:
        """Synthesize sequences across multiple CPU cores using fork-safe ProcessPoolExecutor."""
        workers = num_workers if num_workers is not None else min(16, max(1, (os.cpu_count() or 4) - 2))
        if workers <= 1 or num_sequences < 64:
            return self.synthesize_batch(num_sequences, start_seed, num_frames, mode, emit_variants)

        chunk_size = max(32, math.ceil(num_sequences / float(workers * 2)))
        tasks = []
        offset = 0
        while offset < num_sequences:
            cnt = min(chunk_size, num_sequences - offset)
            tasks.append((self.config, start_seed + offset, cnt, num_frames, mode, emit_variants))
            offset += cnt

        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        out: list[CropSequenceSample] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_synth_worker,
        ) as pool:
            for chunk in pool.map(_worker_synth_chunk, tasks):
                out.extend(chunk)
        return out

    @classmethod
    def _render_negative_signboard_sequence(
        cls,
        seed: int,
        t_raw: int,
        variants: list[str],
    ) -> CropSequenceSample:
        """Render non-plate signboard / bumper grille false-positive crops (`UNREADABLE`)."""
        rng = random.Random(seed + 9187)
        np_rng = np.random.default_rng(seed + 9187)
        is_grille = rng.random() < 0.35

        base = np.empty((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
        if is_grille:
            # Dark automotive grille / bumper slats with chrome trim (classic detector false-positive)
            bg_c = rng.choice([(36, 38, 42), (55, 58, 64), (175, 180, 186)])
            base[:] = bg_c
            for y_s in range(8, CROP_HEIGHT - 6, rng.choice([7, 9, 11])):
                cv2.line(base, (4, y_s), (CROP_WIDTH - 4, y_s), (195, 200, 208), rng.choice([1, 2]))
            cv2.rectangle(base, (2, 2), (CROP_WIDTH - 3, CROP_HEIGHT - 3), (95, 100, 108), 2)
        else:
            # Japanese facility/gate rectangular signboard (`安全確認`, `徐行`, `入口専用`, etc.)
            scheme = rng.choice([
                ((245, 245, 240), (20, 45, 130), (180, 35, 35)),
                ((245, 215, 35), (22, 22, 24), (22, 22, 24)),
                ((238, 244, 238), (20, 110, 55), (20, 110, 55)),
            ])
            base[:] = scheme[0]
            cv2.rectangle(base, (3, 3), (CROP_WIDTH - 4, CROP_HEIGHT - 4), scheme[2], 2)
            label = rng.choice(cls.SIGNBOARD_TEXTS)[:6]
            font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
            n_ch = len(label)
            slot_w = (CROP_WIDTH - 20) / max(1, n_ch)
            for idx_c, ch in enumerate(label):
                gm = FastPlatePatchRenderer._get_glyph_mask(font_path, ch)
                tw = max(6, int(round(slot_w * 0.82)))
                th = 30
                rs = cv2.resize(gm, (tw, th), interpolation=cv2.INTER_AREA)
                ox = int(round(10 + idx_c * slot_w + (slot_w - tw) * 0.5))
                oy = 17
                alpha = (rs.astype(np.float32) / 255.0)[:, :, None]
                patch = base[oy : oy + th, ox : ox + tw].astype(np.float32)
                fg = np.array(scheme[1], dtype=np.float32)
                base[oy : oy + th, ox : ox + tw] = np.clip(patch * (1.0 - alpha) + fg * alpha, 0, 255).astype(np.uint8)

        crops_list: list[np.ndarray] = []
        ts_list: list[float] = []
        src_list: list[int] = []
        var_list: list[str] = []
        for k in range(t_raw):
            dx = rng.uniform(-5.0, 5.0)
            dy = rng.uniform(-3.0, 3.0)
            M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
            shifted = cv2.warpAffine(base, M, (CROP_WIDTH, CROP_HEIGHT), borderMode=cv2.BORDER_REPLICATE)
            noisy = np.clip(shifted.astype(np.float32) + np_rng.normal(0.0, 4.5, shifted.shape), 0, 255).astype(np.uint8)
            for v_name in variants:
                vc = CropExtractor.apply_view_variant(noisy, v_name) if v_name != "raw" else noisy
                crops_list.append(vc)
                ts_list.append(float(k) * 0.1)
                src_list.append(int(k))
                var_list.append(v_name)

        n_tot = len(crops_list)
        return CropSequenceSample(
            sequence_id=f"fastseq_neg_{seed:07d}",
            plate_text=None,
            plate_type=PlateType.STANDARD_PRIVATE.value,
            crops_rgb=np.stack(crops_list, axis=0),
            timestamps_s=np.array(ts_list, dtype=np.float32),
            source_frame_ids=np.array(src_list, dtype=np.int32),
            view_variants=var_list,
            camera_ids=np.zeros(n_tot, dtype=np.int32),
            plate_sides=np.full(n_tot, 2, dtype=np.int32),
            frame_valid_mask=np.zeros(n_tot, dtype=np.float32),
            is_foreign_plate=np.zeros(n_tot, dtype=bool),
            slot_readability=np.zeros((n_tot, 12), dtype=np.float32),
            sequence_target_status="UNREADABLE",
            metadata={"synth_mode": "negative_signboard"},
        )

    def synthesize_batch(
        self,
        num_sequences: int,
        start_seed: int = 0,
        num_frames: int | None = None,
        mode: str = "mixed",
        emit_variants: list[str] | None = None,
    ) -> list[CropSequenceSample]:
        """Synthesize a batch of `CropSequenceSample` objects deterministically."""
        return [
            self.synthesize_one(
                seed=start_seed + i,
                num_frames=num_frames,
                mode=mode,
                emit_variants=emit_variants,
            )
            for i in range(num_sequences)
        ]

    def synthesize_one(
        self,
        seed: int,
        num_frames: int | None = None,
        mode: str = "mixed",
        glare_mode: str | None = None,
        emit_variants: list[str] | None = None,
        plate_text: PlateText | None = None,
        plate_config: PlateConfig | None = None,
    ) -> CropSequenceSample:
        """Synthesize a single `(T, 64, 128, 3)` `CropSequenceSample` directly in crop space.

        Supported `mode` values:
          - `"mixed"`: realistic training curriculum across readable, complementary glare,
            multi-camera (front/rear/side), persistent slot occlusion (`UNREADABLE`), and `track_mixed` (`TRACK_MIXED`).
          - `"readable"`: clean/moderate camera trajectory with `sequence_target_status == "READABLE"`.
          - `"complementary_rows"`: sweeping row glare where early frames wash out lower row and late frames wash out upper row.
          - `"unreadable_occluded"`: persistent single-slot occluder across all frames (`UNREADABLE`).
          - `"unreadable_side"`: pure-side camera view >76 deg (`UNREADABLE`).
          - `"track_mixed"`: splices 2..5 frames of a foreign plate (`TRACK_MIXED`).
          - `"multicam"`: 3-camera gate pass (front + oblique/pure_side + rear with seal).
        """
        seed_eff = 0 if seed is None else int(seed)
        rng = random.Random(seed_eff)
        np_rng = np.random.default_rng(seed_eff)
        t_raw = num_frames if num_frames is not None else self.config.default_num_frames
        t_raw = max(4, t_raw)
        variants = emit_variants if emit_variants else ["raw"]

        # 1. Choose scenario sub-mode if mode == "mixed"
        sub_mode = mode
        if mode == "mixed":
            r_draw = rng.random()
            thresholds = [
                ("track_mixed", self.config.p_track_mixed),
                ("unreadable_occluded", self.config.p_unreadable_occluded),
                ("unreadable_side", self.config.p_unreadable_side),
                ("negative_signboard", self.config.p_negative_signboard),
                ("complementary_rows", self.config.p_complementary_glare),
                ("complementary_cols", self.config.p_complementary_cols),
                ("gate_bars_wipe", self.config.p_gate_bars_wipe),
                ("steep_overhead_lowres", self.config.p_steep_overhead_lowres),
                ("bumper_shadow", self.config.p_bumper_shadow),
                ("multicam", self.config.p_multicam),
            ]
            cum = 0.0
            sub_mode = "readable"
            for m_name, prob in thresholds:
                cum += prob
                if r_draw < cum:
                    sub_mode = m_name
                    break

        if sub_mode == "negative_signboard":
            return self._render_negative_signboard_sequence(seed=seed, t_raw=t_raw, variants=variants)

        # 2. Sample PlateText & PlateConfig (use true 1-char-diff hard negative pair 50% of time on track_mixed!)
        paired_foreign: tuple[PlateText, PlateConfig] | None = None
        if (
            sub_mode == "track_mixed"
            and plate_text is None
            and plate_config is None
            and rng.random() < 0.50
        ):
            (pt, pc), (f_pt0, f_pc0) = self.stratified_sampler.sample_hard_negative_pair(seed=seed)
            plate_text, plate_config = pt, pc
            paired_foreign = (f_pt0, f_pc0)
        elif plate_text is None or plate_config is None:
            pt, pc = self.plate_sampler.sample(seed=seed)
            plate_text = plate_text or pt
            plate_config = plate_config or pc

        active_mask = SlotReadabilityMapper.get_active_slot_mask(plate_text)
        active_indices = [int(i) for i, a in enumerate(active_mask) if a]

        front_cfg = replace(
            plate_config,
            with_seal=False,
            mud_level=rng.uniform(0.0, 0.25) if (sub_mode not in ("complementary_rows", "readable") and rng.random() < 0.2) else 0.0,
            plate_frame=(rng.random() < 0.20),
            weathering=rng.uniform(0.0, 0.35) if rng.random() < 0.25 else 0.0,
            is_illuminated=(rng.random() < 0.08),
        )
        if sub_mode == "unreadable_occluded":
            occ_slot = rng.choice(active_indices)
            front_cfg = replace(front_cfg, persistent_occluder_slot=occ_slot)

        is_std = front_cfg.plate_type in (
            PlateType.STANDARD_PRIVATE,
            PlateType.STANDARD_COMMERCIAL,
            PlateType.GRAPHIC,
        )
        rear_cfg = replace(front_cfg, with_seal=is_std)

        patch_scale = 0.62 * (330.0 / 440.0) if front_cfg.plate_size == PlateSize.LARGE else 0.62
        front_patch, front_occ_12 = FastPlatePatchRenderer.render_patch(
            plate_text, front_cfg, scale=patch_scale
        )
        rear_patch, rear_occ_12 = (
            FastPlatePatchRenderer.render_patch(plate_text, rear_cfg, scale=patch_scale)
            if sub_mode == "multicam"
            else (front_patch, front_occ_12)
        )

        # 3. 3D Trajectory & Virtual Camera(s)
        duration_s = rng.uniform(1.6, 3.6)
        scen = rng.choice([
            ScenarioType.STRAIGHT_APPROACH,
            ScenarioType.DECELERATE_AND_STOP,
            ScenarioType.STOP_AND_GO,
            ScenarioType.CURVING_ENTRY,
            ScenarioType.SWITCHBACK_REVERSING,
        ])
        if sub_mode in ("complementary_rows", "complementary_cols", "gate_bars_wipe"):
            scen = ScenarioType.STRAIGHT_APPROACH
        trajectory = create_trajectory(scen, duration_s, seed)
        veh_len = 7.0 if front_cfg.plate_size == PlateSize.LARGE else 4.6
        is_large = front_cfg.plate_size == PlateSize.LARGE
        pw_m, ph_m = (0.440, 0.220) if is_large else (0.330, 0.165)

        mid_pose = trajectory.evaluate(duration_s * 0.5)
        end_pose = trajectory.evaluate(duration_s)
        start_pose = trajectory.evaluate(0.0)
        rear_start = compute_rear_plate_pose(start_pose, veh_len)
        rear_mid = compute_rear_plate_pose(mid_pose, veh_len)
        u_f, _, n_f = PinholeCamera.compute_plate_axes(mid_pose)
        u_r, _, n_r = PinholeCamera.compute_plate_axes(rear_mid)

        cams: list[tuple[int, int, PinholeCamera]] = []  # (cam_id, plate_side, PinholeCamera)
        if sub_mode == "unreadable_side":
            az = math.radians(rng.choice([-1.0, 1.0]) * rng.uniform(84.0, 102.0))
            anchor = end_pose.position - 1.8 * n_f
            c_pos = np.array([
                anchor[0] + 7.0 * (math.cos(az) * n_f[0] + math.sin(az) * u_f[0]),
                anchor[1] + 7.0 * (math.cos(az) * n_f[1] + math.sin(az) * u_f[1]),
                rng.uniform(1.2, 3.6),
            ])
            cp = PinholeCamera.aim_at(c_pos, mid_pose.position, fov_h_deg=38.0, camera_id=0, mount_role="pure_side")
            cams.append((0, 2, PinholeCamera(cp)))
        elif sub_mode == "steep_overhead_lowres":
            # Overhead gantry / 2F office / parking truss mount: steep downward pitch + low-res approach
            cp_ov = PinholeCamera.aim_at(
                np.array([rng.uniform(-1.6, 1.6), end_pose.position[1] - rng.uniform(3.8, 5.6), rng.uniform(3.8, 5.8)]),
                mid_pose.position,
                fov_h_deg=rng.uniform(46.0, 60.0),
                roll_deg=rng.uniform(-5.0, 5.0),
                camera_id=0,
                mount_role="overhead",
            )
            cams.append((0, 0, PinholeCamera(cp_ov)))
        elif sub_mode == "multicam":
            # Cam 0: front; Cam 1: pure_side or high oblique; Cam 2: rear (+seal)
            cp0 = PinholeCamera.aim_at(
                np.array([rng.uniform(-1.5, 1.5), end_pose.position[1] - rng.uniform(4.5, 6.5), rng.uniform(0.9, 3.8)]),
                mid_pose.position,
                fov_h_deg=52.0,
                roll_deg=rng.uniform(-5.0, 5.0),
                camera_id=0,
                mount_role="front",
            )
            cams.append((0, 0, PinholeCamera(cp0)))

            if rng.random() < 0.5:
                az_s = math.radians(rng.choice([-1.0, 1.0]) * 92.0)
                anc = end_pose.position - 1.8 * n_f
                cp1 = PinholeCamera.aim_at(
                    np.array([anc[0] + 6.8 * math.sin(az_s), anc[1] - 6.8 * math.cos(az_s), rng.uniform(1.2, 4.2)]),
                    mid_pose.position,
                    fov_h_deg=38.0,
                    camera_id=1,
                    mount_role="pure_side",
                )
                cams.append((1, 2, PinholeCamera(cp1)))
            else:
                cp1 = PinholeCamera.aim_at(
                    np.array([rng.choice([-1.0, 1.0]) * 4.8, end_pose.position[1] - 6.8, rng.uniform(4.2, 7.6)]),
                    mid_pose.position,
                    fov_h_deg=56.0,
                    roll_deg=rng.uniform(-6.0, 6.0),
                    camera_id=1,
                    mount_role="oblique",
                )
                cams.append((1, 0, PinholeCamera(cp1)))

            cp2 = PinholeCamera.aim_at(
                np.array([
                    rear_start.position[0] + rng.uniform(-1.5, 1.5),
                    rear_start.position[1] + rng.uniform(4.8, 7.0),
                    rng.uniform(1.2, 4.5),
                ]),
                rear_mid.position,
                fov_h_deg=50.0,
                roll_deg=rng.uniform(-4.0, 4.0),
                camera_id=2,
                mount_role="rear",
            )
            cams.append((2, 1, PinholeCamera(cp2)))
        else:
            cp0 = PinholeCamera.aim_at(
                np.array([rng.uniform(-1.8, 1.8), end_pose.position[1] - rng.uniform(4.2, 6.8), rng.uniform(1.0, 4.2)]),
                mid_pose.position,
                fov_h_deg=rng.uniform(42.0, 60.0),
                roll_deg=rng.uniform(-4.0, 4.0),
                camera_id=0,
                mount_role="front",
            )
            cams.append((0, 0, PinholeCamera(cp0)))

        # 4. Illumination & Camera Parameters
        illum = rng.choices(
            [
                IlluminationType.DAY_CLEAR,
                IlluminationType.DAY_CLOUDY,
                IlluminationType.BACKLIGHT,
                IlluminationType.DUSK_DAWN,
                IlluminationType.NIGHT_HEADLIGHTS,
                IlluminationType.NIGHT_IR,
            ],
            weights=[0.32, 0.20, 0.14, 0.10, 0.12, 0.12],
            k=1,
        )[0]
        if sub_mode == "complementary_rows":
            illum = IlluminationType.DAY_CLEAR

        eff_glare_mode = glare_mode if glare_mode is not None else (
            "complementary_rows" if sub_mode == "complementary_rows" else (
                "random_sweep" if (sub_mode == "readable" and rng.random() < 0.18) else "none"
            )
        )
        glare_intensity = 0.98 if eff_glare_mode == "complementary_rows" else (
            rng.uniform(0.35, 0.65) if eff_glare_mode != "none" else 0.0
        )

        shutter_s = 0.0015 if sub_mode == "complementary_rows" else math.exp(rng.uniform(math.log(0.0006), math.log(0.016)))
        gain_db = rng.uniform(12.0, 22.0) if illum in (IlluminationType.NIGHT_IR, IlluminationType.NIGHT_HEADLIGHTS) else rng.uniform(0.0, 10.0)
        noise_sigma = 1.2 * (10.0 ** (gain_db / 20.0))
        substream_scale = (
            rng.uniform(0.60, 0.78)
            if sub_mode == "steep_overhead_lowres"
            else (
                1.0
                if sub_mode in ("complementary_rows", "complementary_cols", "gate_bars_wipe", "readable")
                else rng.choice([1.0, 1.0, 0.72, 0.58])
            )
        )
        jpeg_q = rng.randint(58, 86)

        norm_slot_boxes = SlotReadabilityMapper.get_normalized_slot_boxes(plate_text, front_cfg.plate_size)
        # Prebuild (48, 1, 2) array of unit slot corners for single-call cv2.perspectiveTransform!
        slot_unit_quads = np.empty((48, 1, 2), dtype=np.float32)
        for s in range(12):
            u0, v0, u1, v1 = norm_slot_boxes[s]
            base_i = s * 4
            slot_unit_quads[base_i, 0] = (u0, v0)
            slot_unit_quads[base_i + 1, 0] = (u1, v0)
            slot_unit_quads[base_i + 2, 0] = (u1, v1)
            slot_unit_quads[base_i + 3, 0] = (u0, v1)
        src_unit_4 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

        veh_col = rng.choice([(225, 226, 230), (42, 44, 48), (168, 172, 178), (210, 170, 42), (38, 65, 118)])

        crops_list: list[np.ndarray] = []
        ts_list: list[float] = []
        src_id_list: list[int] = []
        var_list: list[str] = []
        cam_id_list: list[int] = []
        side_list: list[int] = []
        valid_list: list[float] = []
        foreign_list: list[bool] = []
        read_list: list[np.ndarray] = []

        ar_dx, ar_dy, ar_sx, ar_sy = 0.0, 0.0, 0.0, 0.0
        rho = 0.80
        innov = math.sqrt(1.0 - rho * rho)

        for k in range(t_raw):
            prog = k / float(max(1, t_raw - 1))
            t_sec = prog * duration_s
            cam_id, p_side, cam = cams[k % len(cams)]

            raw_pose = trajectory.evaluate(t_sec)
            pose_k = compute_rear_plate_pose(raw_pose, veh_len) if p_side == 1 else raw_pose
            inc_deg, _, _ = cam.compute_incidence_angle(pose_k)

            # Optional garbage crop injection (not on complementary_rows endpoints)
            allow_garbage = (
                sub_mode in ("readable", "multicam")
                and eff_glare_mode != "complementary_rows"
                and 0 < k < t_raw - 1
                and rng.random() < self.config.p_garbage_frame
            )
            if allow_garbage:
                val = 250 if rng.random() < 0.5 else 38
                raw_crop = np.full((CROP_HEIGHT, CROP_WIDTH, 3), val, dtype=np.uint8)
                f_valid = 0.0
                f_read = np.zeros(12, dtype=np.float32)
            elif p_side == 2 or inc_deg > 76.0:
                # Pure side / back-facing camera: sees side body panel, zero plate readability
                raw_crop = np.empty((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
                raw_crop[:, :, 0] = veh_col[0]
                raw_crop[:, :, 1] = veh_col[1]
                raw_crop[:, :, 2] = veh_col[2]
                cv2.line(raw_crop, (0, 32), (CROP_WIDTH, 34), (32, 34, 38), 3)
                if illum == IlluminationType.NIGHT_IR:
                    g_val = int(0.25 * (0.299 * veh_col[0] + 0.587 * veh_col[1] + 0.114 * veh_col[2]))
                    raw_crop[:] = g_val
                f_valid = 0.0
                f_read = np.zeros(12, dtype=np.float32)
            else:
                corners_3d = cam.compute_plate_corners_3d(pose_k, pw_m, ph_m, is_rear=(p_side == 1))
                pts_cam = np.array([cam.project_point(c3)[:2] for c3 in corners_3d], dtype=np.float32)

                # Optical flow vector over shutter_s for exact directional motion blur
                pose_dt = trajectory.evaluate(min(duration_s, t_sec + shutter_s))
                if p_side == 1:
                    pose_dt = compute_rear_plate_pose(pose_dt, veh_len)
                c0_dt = cam.project_point(cam.compute_plate_corners_3d(pose_dt, pw_m, ph_m)[0])[:2]
                du_cam = c0_dt[0] - float(pts_cam[0, 0])
                dv_cam = c0_dt[1] - float(pts_cam[0, 1])
                blur_px = math.hypot(du_cam, dv_cam)

                x0, y0 = float(np.min(pts_cam[:, 0])), float(np.min(pts_cam[:, 1]))
                x1, y1 = float(np.max(pts_cam[:, 0])), float(np.max(pts_cam[:, 1]))
                bw = max(8.0, x1 - x0)
                bh = max(4.0, y1 - y0)

                # AR(1) detector bounding box drift
                if eff_glare_mode == "complementary_rows":
                    mx, my, dx, dy = 0.08, 0.10, 0.0, 0.0
                else:
                    ar_dx = rho * ar_dx + innov * rng.gauss(0.0, 0.045)
                    ar_dy = rho * ar_dy + innov * rng.gauss(0.0, 0.045)
                    ar_sx = rho * ar_sx + innov * rng.gauss(0.0, 0.05)
                    ar_sy = rho * ar_sy + innov * rng.gauss(0.0, 0.05)
                    mx = max(-0.02, 0.08 + ar_sx)
                    my = max(-0.02, 0.10 + ar_sy)
                    dx, dy = ar_dx, ar_dy

                cx0 = x0 - mx * bw + dx * bw
                cy0 = y0 - my * bh + dy * bh
                cx1 = x1 + mx * bw + dx * bw
                cy1 = y1 + my * bh + dy * bh

                if (
                    sub_mode not in ("complementary_rows", "readable")
                    and 0 < k < t_raw - 1
                    and rng.random() < self.config.p_edge_truncation
                ):
                    if rng.random() < 0.5:
                        cx0 = x0 + rng.uniform(0.18, 0.36) * bw
                    else:
                        cx1 = x1 - rng.uniform(0.18, 0.36) * bw

                cw_box = max(4.0, cx1 - cx0)
                ch_box = max(2.0, cy1 - cy0)
                sx_crop = CROP_WIDTH / cw_box
                sy_crop = CROP_HEIGHT / ch_box

                dst_crop = np.empty((4, 2), dtype=np.float32)
                dst_crop[:, 0] = (pts_cam[:, 0] - cx0) * sx_crop
                dst_crop[:, 1] = (pts_cam[:, 1] - cy0) * sy_crop

                # Single-call projection of all 12 slots to compute exact truncation IOA!
                H_unit_to_crop = cv2.getPerspectiveTransform(src_unit_4, dst_crop)
                proj_slots = cv2.perspectiveTransform(slot_unit_quads, H_unit_to_crop).reshape(12, 4, 2)
                sx0_a = np.min(proj_slots[:, :, 0], axis=1)
                sy0_a = np.min(proj_slots[:, :, 1], axis=1)
                sx1_a = np.max(proj_slots[:, :, 0], axis=1)
                sy1_a = np.max(proj_slots[:, :, 1], axis=1)
                s_areas = np.maximum(1e-4, (sx1_a - sx0_a) * (sy1_a - sy0_a))
                ix0_a = np.maximum(0.0, sx0_a)
                iy0_a = np.maximum(0.0, sy0_a)
                ix1_a = np.minimum(float(CROP_WIDTH), sx1_a)
                iy1_a = np.minimum(float(CROP_HEIGHT), sy1_a)
                ioa_12 = np.clip(
                    np.maximum(0.0, ix1_a - ix0_a) * np.maximum(0.0, iy1_a - iy0_a) / s_areas,
                    0.0,
                    1.0,
                )

                # Warp plate patch directly into (64, 128) crop!
                src_patch = rear_patch if p_side == 1 else front_patch
                occ_12 = (rear_occ_12 if p_side == 1 else front_occ_12).copy()
                eff_pw = max(24, int(round(bw * substream_scale)))
                eff_ph = max(12, int(round(bh * substream_scale)))
                if eff_pw < src_patch.shape[1] * 0.85:
                    work_patch = cv2.resize(src_patch, (eff_pw, eff_ph), interpolation=cv2.INTER_AREA)
                else:
                    work_patch = src_patch

                ph_s, pw_s = work_patch.shape[:2]
                src_4 = np.array([[0.0, 0.0], [float(pw_s), 0.0], [float(pw_s), float(ph_s)], [0.0, float(ph_s)]], dtype=np.float32)
                H_p = cv2.getPerspectiveTransform(src_4, dst_crop)

                bg_crop = np.empty((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
                bg_crop[:, :, 0] = int(veh_col[0] * 0.45)
                bg_crop[:, :, 1] = int(veh_col[1] * 0.45)
                bg_crop[:, :, 2] = int(veh_col[2] * 0.45)

                warped = cv2.warpPerspective(
                    work_patch,
                    H_p,
                    (CROP_WIDTH, CROP_HEIGHT),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                p_mask = np.zeros((CROP_HEIGHT, CROP_WIDTH), dtype=np.uint8)
                cv2.fillConvexPoly(p_mask, np.int32(dst_crop), 255, lineType=cv2.LINE_AA)
                raw_crop = np.where(p_mask[:, :, np.newaxis] > 0, warped, bg_crop)

                # Dynamic foreground occlusions & shadows in canonical/crop space:
                if sub_mode == "complementary_cols":
                    # Person/worker legs or traffic cone stepping laterally from left half -> right half!
                    occ_map = np.zeros((64, 128), dtype=np.float32)
                    c_u = 0.24 * (1.0 - prog) + 0.76 * prog
                    half_u = 0.27
                    uu_c = np.linspace(0.0, 1.0, 128, dtype=np.float32)[None, :]
                    occ_band = (np.abs(uu_c - c_u) <= half_u).astype(np.float32)
                    occ_map[:] = occ_band
                    slot_dyn_occ = IlluminationSimulator.compute_slot_glare_washout(occ_map)
                    occ_12 = np.maximum(occ_12, np.array(slot_dyn_occ, dtype=np.float32))

                    fg_col = np.array(
                        rng.choice([(28, 36, 58), (225, 92, 24), (42, 46, 50)]),
                        dtype=np.float32,
                    )
                    w_occ = cv2.warpPerspective(
                        occ_map,
                        cv2.getPerspectiveTransform(
                            np.array([[0.0, 0.0], [128.0, 0.0], [128.0, 64.0], [0.0, 64.0]], dtype=np.float32),
                            dst_crop,
                        ),
                        (CROP_WIDTH, CROP_HEIGHT),
                        flags=cv2.INTER_LINEAR,
                    )[:, :, None]
                    raw_crop = np.clip(
                        raw_crop.astype(np.float32) * (1.0 - w_occ) + fg_col * w_occ, 0.0, 255.0
                    ).astype(np.uint8)

                elif sub_mode == "gate_bars_wipe":
                    # Sliding vertical accordion gate / fence bars wiping across the plate over time
                    occ_map = np.zeros((64, 128), dtype=np.float32)
                    phase = prog * 1.15
                    for bar_i in range(4):
                        bu = (bar_i * 0.28 + phase) % 1.12 - 0.06
                        x_b0 = max(0, int(round((bu - 0.045) * 128)))
                        x_b1 = min(128, int(round((bu + 0.045) * 128)))
                        if x_b1 > x_b0:
                            occ_map[:, x_b0:x_b1] = 1.0
                    slot_dyn_occ = IlluminationSimulator.compute_slot_glare_washout(occ_map)
                    occ_12 = np.maximum(occ_12, np.array(slot_dyn_occ, dtype=np.float32))
                    bar_col = np.array([158.0, 164.0, 170.0], dtype=np.float32)
                    w_occ = cv2.warpPerspective(
                        occ_map,
                        cv2.getPerspectiveTransform(
                            np.array([[0.0, 0.0], [128.0, 0.0], [128.0, 64.0], [0.0, 64.0]], dtype=np.float32),
                            dst_crop,
                        ),
                        (CROP_WIDTH, CROP_HEIGHT),
                        flags=cv2.INTER_LINEAR,
                    )[:, :, None]
                    raw_crop = np.clip(
                        raw_crop.astype(np.float32) * (1.0 - 0.92 * w_occ) + bar_col * (0.92 * w_occ),
                        0.0,
                        255.0,
                    ).astype(np.uint8)

                elif sub_mode == "bumper_shadow":
                    # Deep recessed bumper casting dark horizontal shadow on upper half + wet road reflection below
                    vv_s = np.linspace(0.0, 1.0, CROP_HEIGHT, dtype=np.float32)[:, None, None]
                    shadow_mult = np.where(vv_s < 0.46, 0.34 + 0.30 * (vv_s / 0.46), 1.0 + 0.14 * (vv_s - 0.46))
                    raw_crop = np.clip(raw_crop.astype(np.float32) * shadow_mult, 0.0, 255.0).astype(np.uint8)

                # Directional motion blur in crop space
                du_c = du_cam * sx_crop
                dv_c = dv_cam * sy_crop
                b_len = int(round(math.hypot(du_c, dv_c)))
                if b_len >= 2:
                    k_len = min(19, b_len if b_len % 2 == 1 else b_len + 1)
                    psf = np.zeros((k_len, k_len), dtype=np.float32)
                    c_mid = k_len // 2
                    ang = math.atan2(dv_c, du_c)
                    dx_k = int(round(math.cos(ang) * c_mid))
                    dy_k = int(round(math.sin(ang) * c_mid))
                    cv2.line(psf, (c_mid - dx_k, c_mid - dy_k), (c_mid + dx_k, c_mid + dy_k), 1.0, 1)
                    psf /= max(1e-5, float(np.sum(psf)))
                    raw_crop = cv2.filter2D(raw_crop, -1, psf)

                # Illumination & Night IR
                crop_f = raw_crop.astype(np.float32)
                if illum == IlluminationType.NIGHT_IR:
                    gray_c = 0.299 * crop_f[:, :, 0] + 0.587 * crop_f[:, :, 1] + 0.114 * crop_f[:, :, 2]
                    pm_f = p_mask.astype(np.float32) / 255.0
                    ir_c = gray_c * (0.18 * (1.0 - pm_f) + 0.92 * pm_f)
                    crop_f = np.stack([ir_c, ir_c, ir_c], axis=-1)
                elif illum == IlluminationType.NIGHT_HEADLIGHTS:
                    pm_f = (p_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
                    crop_f = crop_f * (0.20 * (1.0 - pm_f) + 0.88 * pm_f)
                elif illum == IlluminationType.BACKLIGHT:
                    crop_f = crop_f * 0.58 + 22.0

                # Motion-linked sweeping specular glare (including complementary_rows!)
                glare_map = IlluminationSimulator.compute_plate_glare_map(
                    norm_progress=prog,
                    glare_mode=eff_glare_mode,
                    glare_intensity=glare_intensity,
                    seed=seed,
                )
                slot_glare_12 = IlluminationSimulator.compute_slot_glare_washout(glare_map)
                if float(np.max(glare_map)) > 1e-3:
                    warped_g = cv2.warpPerspective(
                        glare_map,
                        cv2.getPerspectiveTransform(
                            np.array([[0.0, 0.0], [128.0, 0.0], [128.0, 64.0], [0.0, 64.0]], dtype=np.float32),
                            dst_crop,
                        ),
                        (CROP_WIDTH, CROP_HEIGHT),
                        flags=cv2.INTER_LINEAR,
                    )[:, :, np.newaxis]
                    crop_f = crop_f * (1.0 - 0.94 * warped_g) + 255.0 * warped_g

                raw_crop = np.clip(crop_f, 0.0, 255.0).astype(np.uint8)
                f_valid = 1.0
                f_read = SlotReadabilityMapper.compute_frame_slot_readability(
                    plate_height_px=bh,
                    plate_width_px=bw,
                    incidence_deg=inc_deg,
                    is_readable_view=True,
                    blur_px=blur_px,
                    defocus_px=0.2,
                    noise_sigma=noise_sigma,
                    jpeg_quality=jpeg_q,
                    substream_scale=substream_scale,
                    slot_occlusion_12=occ_12,
                    slot_glare_12=slot_glare_12,
                    slot_crop_ioa_12=ioa_12,
                )

            for v_name in variants:
                v_crop = CropExtractor.apply_view_variant(raw_crop, v_name) if v_name != "raw" else raw_crop.copy()
                crops_list.append(v_crop)
                ts_list.append(float(t_sec))
                src_id_list.append(int(k))
                var_list.append(v_name)
                cam_id_list.append(int(cam_id))
                side_list.append(int(p_side))
                valid_list.append(float(f_valid))
                foreign_list.append(False)
                read_list.append(f_read.copy())

        crops_arr = np.stack(crops_list, axis=0)

        # Vectorized CMOS sensor noise across the entire (T, 64, 128, 3) stack
        if noise_sigma > 1.5:
            noise = np_rng.normal(0.0, noise_sigma * 0.55, crops_arr.shape).astype(np.float32)
            if illum == IlluminationType.NIGHT_IR:
                n_mono = noise[:, :, :, :1]
                noise = np.repeat(n_mono, 3, axis=3)
            crops_arr = np.clip(crops_arr.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        # Track-Mixed injection (splices 2..5 genuine 3D camera-projected frames of a foreign plate!)
        foreign_str: str | None = None
        foreign_dict: dict[str, Any] | None = None
        if sub_mode == "track_mixed":
            if paired_foreign is not None and paired_foreign[0].ocr_string != plate_text.ocr_string:
                f_pt, f_pc = paired_foreign
            else:
                for off in range(1, 30):
                    f_pt, f_pc = self.base_sampler.sample(seed=seed + off * 4999)
                    if f_pt.ocr_string != plate_text.ocr_string:
                        break
            foreign_str = f_pt.full_text
            foreign_dict = f_pt.to_dict()

            # Synthesize genuine 3D perspective-warped camera sequence for the foreign vehicle
            # so individual frames have ZERO visual artifacts distinguishing them from normal frames!
            f_seq = self.synthesize_one(
                seed=seed + 79191,
                num_frames=t_raw,
                mode="readable",
                glare_mode="none",
                emit_variants=variants,
                plate_text=f_pt,
                plate_config=f_pc,
            )
            n_tot = len(crops_list)
            v_cnt = max(1, len(variants))
            raw_sp = min(max(2, t_raw // 2), rng.randint(2, max(2, min(5, t_raw - 2))))
            sp_raw_start = rng.randint(0, max(0, t_raw - raw_sp))
            idx_start = sp_raw_start * v_cnt
            idx_end = min(n_tot, (sp_raw_start + raw_sp) * v_cnt)
            for idx_sp in range(idx_start, idx_end):
                crops_arr[idx_sp] = f_seq.crops_rgb[idx_sp]
                side_list[idx_sp] = int(f_seq.plate_sides[idx_sp])
                foreign_list[idx_sp] = True
                valid_list[idx_sp] = 0.0
                read_list[idx_sp] = np.zeros(12, dtype=np.float32)

        ts_arr = np.array(ts_list, dtype=np.float32)
        src_arr = np.array(src_id_list, dtype=np.int32)
        cam_arr = np.array(cam_id_list, dtype=np.int32)
        side_arr = np.array(side_list, dtype=np.int32)
        valid_arr = np.array(valid_list, dtype=np.float32)
        foreign_arr = np.array(foreign_list, dtype=bool)
        read_arr = np.stack(read_list, axis=0).astype(np.float32)

        status = SlotReadabilityMapper.compute_sequence_target_status(
            plate_text=plate_text,
            slot_readability=read_arr,
            frame_valid_mask=valid_arr,
            is_foreign_plate=foreign_arr,
        )

        meta: dict[str, Any] = {
            "synth_mode": sub_mode,
            "illumination": illum.value,
            "glare_mode": eff_glare_mode,
            "persistent_occluder_slot": front_cfg.persistent_occluder_slot,
        }
        if foreign_str is not None:
            meta["foreign_plate_text"] = foreign_str
        if foreign_dict is not None:
            meta["foreign_plate_dict"] = foreign_dict

        return CropSequenceSample(
            sequence_id=f"fastseq_{seed:07d}",
            plate_text=plate_text,
            plate_type=front_cfg.plate_type.value,
            crops_rgb=crops_arr,
            timestamps_s=ts_arr,
            source_frame_ids=src_arr,
            view_variants=var_list,
            camera_ids=cam_arr,
            plate_sides=side_arr,
            frame_valid_mask=valid_arr,
            is_foreign_plate=foreign_arr,
            slot_readability=read_arr,
            sequence_target_status=status,
            metadata=meta,
        )
