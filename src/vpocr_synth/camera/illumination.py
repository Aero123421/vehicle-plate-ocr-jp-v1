"""Physically grounded scene radiance, auto-exposure, WDR, night/IR, and motion-linked glare."""

import math
import random
from typing import Any
import cv2
import numpy as np

from vpocr_synth.camera.models import CameraProfile, IlluminationType
from vpocr_synth.motion.models import CameraParams, FrameAnnotation


class IlluminationSimulator:
    """Simulates scene radiance, camera auto-exposure (with gain ceiling & lag), WDR, IR, and glare."""

    def __init__(
        self,
        profile: CameraProfile,
        camera_params: CameraParams | None = None,
        is_illuminated_plate: bool = False,
    ) -> None:
        self.profile = profile
        self.illum_type = profile.illumination
        self.camera_params = camera_params
        self.is_illuminated_plate = is_illuminated_plate
        self.ae_current_gain = 1.0
        self.ae_smoothing_alpha = float(np.clip(profile.ae_lag_alpha, 0.03, 1.0))
        self.rng = random.Random(profile.seed)
        self._coord_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _get_norm_coords(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._coord_cache.get((h, w))
        if cached is not None:
            return cached
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
        self._coord_cache[(h, w)] = (yy, xx)
        return yy, xx

    def _compute_horizon_y(self, h: int) -> int:
        if self.camera_params is not None:
            pitch_rad = math.radians(self.camera_params.pitch_deg)
            hy = int(round(self.camera_params.cy - self.camera_params.fy * math.tan(pitch_rad)))
            return max(0, min(h, hy))
        return int(round(0.32 * h))

    @staticmethod
    def tone_map_zero_anchored(linear_rgb: np.ndarray, wdr_strength: float = 0.0) -> np.ndarray:
        """Zero-anchored C1-smooth filmic tone curve preserving true darks without gray-fog lift."""
        x = np.maximum(0.0, linear_rgb)
        if wdr_strength > 1e-3:
            w = float(np.clip(wdr_strength, 0.0, 1.0))
            # Local/global shadow lift + highlight compression while keeping x=0 -> 0
            shadow_boost = w * 0.48 * np.sqrt(x + 1e-6) * np.exp(-2.0 * x)
            x = (x + shadow_boost) / (1.0 + 0.55 * w * np.maximum(0.0, x - 0.55))

        # C1-continuous linear below 0.78 + exponential highlight shoulder above 0.78 -> [0.0, 1.0]
        knee = 0.78
        rem = 1.0 - knee
        above = np.maximum(0.0, x - knee)
        shoulder = knee + rem * (1.0 - np.exp(-above / rem))
        mapped = np.where(x <= knee, x, shoulder)
        return np.clip(mapped * 255.0, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def compute_plate_glare_map(
        norm_progress: float,
        glare_mode: str,
        glare_intensity: float,
        seed: int = 0,
        out_h: int = 64,
        out_w: int = 128,
    ) -> np.ndarray:
        """Compute canonical plate-space [0..1] specular glare/washout map of shape (out_h, out_w).

        - 'complementary_rows': early frames (p -> 0) saturate lower row (hiragana + serial),
          later frames (p -> 1) saturate upper row (region + classification).
        - 'random_sweep': angled sun streak sweeping across plate as vehicle pose advances.
        - 'full_washout': persistent specular blowout across entire plate.
        """
        if glare_mode == "none" or (glare_intensity <= 1e-3 and glare_mode != "complementary_rows"):
            return np.zeros((out_h, out_w), dtype=np.float32)

        p = float(np.clip(norm_progress, 0.0, 1.0))
        vv = np.linspace(0.0, 1.0, out_h, dtype=np.float32)[:, np.newaxis]
        uu = np.linspace(0.0, 1.0, out_w, dtype=np.float32)[np.newaxis, :]

        if glare_mode == "complementary_rows":
            eff_int = max(0.98, float(glare_intensity))
            # Smooth sweep from lower row (v=0.70) in early frames to upper row (v=0.20) in late frames
            v_center = 0.70 * (1.0 - p) + 0.20 * p
            band_half_h = 0.30 * (1.0 - p) + 0.19 * p
            # Slight natural sun-angle tilt (<= 0.015 across width so rows stay cleanly separated)
            tilt = 0.015 * (uu - 0.5)
            dist_norm = np.abs((vv + tilt) - v_center) / max(0.05, band_half_h)
            glare = eff_int * np.exp(-(dist_norm ** 6))
            return np.clip(glare, 0.0, 1.0).astype(np.float32)

        if glare_mode == "full_washout":
            eff_int = max(0.95, float(glare_intensity))
            dist2 = ((uu - 0.5) / 0.75) ** 2 + ((vv - 0.5) / 0.75) ** 2
            return np.clip(eff_int * np.exp(-0.25 * dist2), 0.0, 1.0).astype(np.float32)

        # "random_sweep": diagonal specular reflection bar sweeping with trajectory progress
        rng = random.Random(seed + 409)
        angle = rng.uniform(-0.55, 0.55)
        start_c = rng.choice([-0.15, 1.15])
        end_c = 1.0 - start_c
        center = start_c + p * (end_c - start_c)
        coord = (uu - 0.5) * math.cos(angle) + (vv - 0.5) * math.sin(angle) + 0.5
        width = rng.uniform(0.14, 0.24)
        glare = float(glare_intensity) * np.exp(-(((coord - center) / width) ** 4))
        return np.clip(glare, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def compute_slot_glare_washout(glare_map_64x128: np.ndarray) -> list[float]:
        """Measure per-slot glare washout [0..1] across the 12 canonical plate slots on (64, 128) map."""
        if float(np.max(glare_map_64x128)) <= 1e-4:
            return [0.0] * 12

        h, w = glare_map_64x128.shape[:2]
        # Normalized [u0, v0, u1, v1] zones for the 12 canonical slots on standard Japanese layout
        norm_zones = [
            # 0..3: region (upper left-center: u=0.22..0.50, v=0.09..0.33)
            (0.22, 0.10, 0.29, 0.33),
            (0.29, 0.10, 0.36, 0.33),
            (0.36, 0.10, 0.43, 0.33),
            (0.43, 0.10, 0.50, 0.33),
            # 4..6: classification (upper right-center: u=0.57..0.79, v=0.09..0.33)
            (0.57, 0.10, 0.64, 0.33),
            (0.64, 0.10, 0.72, 0.33),
            (0.72, 0.10, 0.79, 0.33),
            # 7: hiragana (lower left: u=0.06..0.18, v=0.54..0.79)
            (0.06, 0.54, 0.18, 0.79),
            # 8..11: serial (lower 4 slots: v=0.43..0.90)
            (0.23, 0.44, 0.35, 0.89),
            (0.39, 0.44, 0.51, 0.89),
            (0.65, 0.44, 0.77, 0.89),
            (0.81, 0.44, 0.93, 0.89),
        ]
        washouts: list[float] = []
        for u0, v0, u1, v1 in norm_zones:
            x0 = max(0, min(w - 1, int(round(u0 * w))))
            x1 = max(x0 + 1, min(w, int(round(u1 * w))))
            y0 = max(0, min(h - 1, int(round(v0 * h))))
            y1 = max(y0 + 1, min(h, int(round(v1 * h))))
            washouts.append(float(np.mean(glare_map_64x128[y0:y1, x0:x1])))
        return washouts

    def apply_illumination(
        self,
        bgr_image: np.ndarray,
        frame_annotation: FrameAnnotation,
        frame_index: int,
        total_frames: int = 1,
    ) -> tuple[np.ndarray, float, list[float]]:
        """Apply scene radiance, AE, WDR, and motion-linked glare.

        Returns (processed_bgr, effective_exposure_scale, slot_glare_washout_12).
        """
        h, w = bgr_image.shape[:2]
        base_norm = bgr_image.astype(np.float32) / 255.0
        yy, xx = self._get_norm_coords(h, w)
        horizon_y = self._compute_horizon_y(h)
        horizon_frac = horizon_y / max(1.0, float(h))

        # Build plate polygon mask if visible
        has_plate = frame_annotation.is_visible and len(frame_annotation.corners.points) == 4
        plate_mask_u8 = np.zeros((h, w), dtype=np.uint8)
        if has_plate:
            pts_i32 = np.array(frame_annotation.corners.points, dtype=np.int32)
            cv2.fillConvexPoly(plate_mask_u8, pts_i32, 255, lineType=cv2.LINE_AA)
        plate_mask = (plate_mask_u8.astype(np.float32) / 255.0)[:, :, np.newaxis]

        # 1. Compute Physically Grounded Linear Scene Radiance L (in normalized [0..2.5] units)
        if self.illum_type == IlluminationType.DAY_CLEAR:
            # Crisp high-contrast daylight: warm sun tint, sunlit road & upper highlight, true dark shadows
            sun_grad = (1.10 - 0.14 * yy)[:, :, np.newaxis]
            radiance = np.clip(base_norm * sun_grad * 1.25 + 0.055 * np.sqrt(base_norm + 1e-4), 0.0, 2.2)
            color_wb = np.array([0.95, 1.00, 1.05], dtype=np.float32)  # BGR warm sunlight
            ae_target_mean = 0.45
            ae_gain_bounds = (0.65, 2.10)

        elif self.illum_type == IlluminationType.DAY_CLOUDY:
            # Diffuse cool overcast sky, balanced midtones
            radiance = np.clip(base_norm * 1.15 + 0.05, 0.02, 2.0)
            color_wb = np.array([1.06, 1.00, 0.95], dtype=np.float32)
            ae_target_mean = 0.43
            ae_gain_bounds = (0.68, 2.00)

        elif self.illum_type == IlluminationType.BACKLIGHT:
            # Strong backlit sky / upper sun-glare zone (> 205), foreground & vehicle front in shadow (< 85 unless WDR active)
            radiance = base_norm * 0.34
            sky_band_y = max(horizon_y, int(round(h * 0.18)))
            if sky_band_y > 0:
                radiance[:sky_band_y] = np.maximum(base_norm[:sky_band_y] * 1.55 + 0.36, 0.90)

            # Atmospheric sun glare veil spilling downward across horizon & upper vehicle/plate
            sky_frac = sky_band_y / max(1.0, float(h))
            dist_below_h = np.maximum(0.0, yy - sky_frac)
            sun_cx = 0.52
            flare = 0.38 * np.exp(-dist_below_h / 0.14) * np.exp(-(((xx - sun_cx) / 0.48) ** 2))
            radiance = radiance + flare[:, :, np.newaxis]
            color_wb = np.array([0.94, 0.99, 1.06], dtype=np.float32)
            ae_target_mean = 0.36
            ae_gain_bounds = (0.62, 1.02)  # Bright backlight pulls AE down so foreground stays shadowed without WDR

        elif self.illum_type == IlluminationType.DUSK_DAWN:
            # Golden/indigo twilight: low ambient, warm sodium/sunset cast
            radiance = base_norm * 0.44
            if horizon_y > 0:
                radiance[:horizon_y] = base_norm[:horizon_y] * 0.48
            color_wb = np.array([0.78, 0.90, 1.16], dtype=np.float32)
            ae_target_mean = 0.25
            ae_gain_bounds = (0.80, 1.45)

        elif self.illum_type == IlluminationType.NIGHT_HEADLIGHTS:
            # Dark night scene: sky nearly black, road unlit (< 45), active warm headlights + plate light
            radiance = base_norm * 0.13 + 0.008
            if horizon_y > 0:
                # Night sky has near-zero radiance (eliminates daytime sky glow!)
                radiance[:horizon_y] = base_norm[:horizon_y] * 0.035 + 0.012

            # Vehicle headlight cones & license plate illumination
            if has_plate:
                x0, y0, x1, y1 = frame_annotation.corners.to_bbox()
                pcx = 0.5 * (x0 + x1) / max(1.0, float(w))
                pcy = 0.5 * (y0 + y1) / max(1.0, float(h))
                pw_n = max(0.03, (x1 - x0) / max(1.0, float(w)))

                # Left and right headlight cones illuminating road below/ahead of bumper
                cone_accum = np.zeros((h, w), dtype=np.float32)
                for side_s in (-1.9, 1.9):
                    lx = pcx + side_s * pw_n
                    ly = pcy - 0.02
                    # Headlight orb core
                    orb_d2 = ((xx - lx) / max(0.022, pw_n * 0.45)) ** 2 + ((yy - ly) / max(0.03, pw_n * 0.6)) ** 2
                    orb = 1.45 * np.exp(-orb_d2)
                    # Forward road light pool expanding downward from headlight
                    dy_down = np.maximum(0.0, yy - ly)
                    spread_x = np.maximum(0.05, pw_n * 0.8 + dy_down * 0.65)
                    pool = (
                        0.62
                        * (dy_down / (dy_down + 0.04))
                        * np.exp(-(((xx - lx) / spread_x) ** 2))
                        * np.exp(-dy_down / 0.48)
                    )
                    cone_accum += orb + pool

                warm_beam = np.stack(
                    [cone_accum * 0.78, cone_accum * 0.94, cone_accum * 1.08],
                    axis=-1,
                )
                radiance = radiance + warm_beam * (0.35 + 0.65 * base_norm)

                # Plate illumination (license lamp / oncoming headlight retro-reflection)
                dist_m = max(1.8, frame_annotation.distance_m)
                plate_illum_level = float(np.clip(3.6 / dist_m, 0.58, 1.12))
                plate_rad = base_norm * plate_illum_level
                if self.is_illuminated_plate:
                    # 字光式 glowing green strokes stay bright at night
                    green_dom = np.maximum(0.0, base_norm[:, :, 1] - 0.5 * base_norm[:, :, 0])
                    plate_rad[:, :, 1] += green_dom * 0.75
                radiance = radiance * (1.0 - plate_mask) + plate_rad * plate_mask

            color_wb = np.array([0.88, 0.96, 1.08], dtype=np.float32)
            ae_target_mean = 0.11  # Realistic night AE gain ceiling keeps dark surroundings < 45!
            ae_gain_bounds = (0.75, 1.45)

        else:  # IlluminationType.NIGHT_IR
            # Monochrome 850nm IR: pitch-dark night sky, dim IR-lit ground (< 45), bright retro-reflective plate (> 135)
            gray_base = (
                0.114 * base_norm[:, :, 0]
                + 0.587 * base_norm[:, :, 1]
                + 0.299 * base_norm[:, :, 2]
            )
            dist_m = max(1.8, frame_annotation.distance_m)
            # Central IR LED illuminator beam falloff
            beam_r2 = ((xx - 0.5) / 0.68) ** 2 + ((yy - 0.56) / 0.62) ** 2
            ir_beam = np.exp(-0.55 * beam_r2)
            ir_dist = float(np.clip(5.2 / dist_m, 0.55, 1.45))

            ir_scene = gray_base * (0.16 * ir_beam * min(1.15, ir_dist)) + 0.015
            if horizon_y > 0:
                # Night sky at infinity reflects zero camera IR LED light
                ir_scene[:horizon_y] = 0.025 + 0.015 * gray_base[:horizon_y]

            if has_plate:
                eff_boost = max(1.45, float(self.profile.ir_plate_reflectance_boost))
                p_mask_2d = plate_mask[:, :, 0]
                # 850nm NIR spectral response: retro-reflective sheeting returns bright IR while keeping crisp stroke contrast
                # Works across both light-bg plates (high gray_base on sheeting, dark strokes) and dark-bg plates
                plate_vals = gray_base[plate_mask_u8 > 127]
                p_median = float(np.median(plate_vals)) if plate_vals.size > 0 else 0.75
                if p_median >= 0.45:
                    # Standard private / kei private / graphic: bright retro-reflective sheeting, dark ink strokes
                    nir_plate = (0.06 + 0.54 * (gray_base ** 1.25)) * eff_boost * min(1.22, max(0.85, ir_dist))
                else:
                    # Commercial green/black: NIR penetrates/reflects from substrate + bright painted glyphs
                    nir_plate = (0.34 + 0.42 * gray_base) * eff_boost * min(1.22, max(0.85, ir_dist))

                # Close-range IR over-bloom halation when d < 2.7m and high IR boost
                if dist_m < 2.7 and eff_boost > 1.9:
                    over = np.maximum(0.0, nir_plate - 0.85) * p_mask_2d
                    bloom = cv2.GaussianBlur(over, (11, 11), 3.0)
                    nir_plate = nir_plate + 0.65 * bloom

                ir_scene = ir_scene * (1.0 - p_mask_2d) + nir_plate * p_mask_2d

            radiance = np.stack([ir_scene, ir_scene, ir_scene], axis=-1)
            color_wb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            ae_target_mean = 0.10  # Respect sensor AE max gain ceiling so night background stays dark (< 45)
            ae_gain_bounds = (0.85, 1.28)

        # 2. Auto-Exposure Metering with Gain Ceiling, EV Bias, and Temporal Lag
        scene_mean = float(np.mean(radiance))
        desired_gain = float(np.clip(ae_target_mean / max(0.03, scene_mean), ae_gain_bounds[0], ae_gain_bounds[1]))
        if frame_index == 0:
            # Slight initial AE offset before settling (models gate entry AE transient)
            init_jitter = self.rng.uniform(0.92, 1.08)
            self.ae_current_gain = desired_gain * init_jitter
        else:
            self.ae_current_gain += self.ae_smoothing_alpha * (desired_gain - self.ae_current_gain)

        ev_mult = 2.0 ** float(np.clip(self.profile.ev_bias, -2.0, 2.0))
        effective_exposure = float(self.ae_current_gain * ev_mult)

        # White balance shift (disabled for monochrome IR)
        if self.illum_type != IlluminationType.NIGHT_IR and abs(self.profile.wb_temp_shift) > 1e-3:
            wb_s = float(np.clip(self.profile.wb_temp_shift, -1.0, 1.0))
            wb_mod = np.array([1.0 - 0.18 * wb_s, 1.0, 1.0 + 0.18 * wb_s], dtype=np.float32)
            color_wb = color_wb * wb_mod

        exposed = radiance * effective_exposure * color_wb[np.newaxis, np.newaxis, :]

        # 3. Motion-Linked Specular Glare Band Warped onto 3D Plate Quad
        norm_prog = frame_index / float(max(1, total_frames - 1))
        glare_map = self.compute_plate_glare_map(
            norm_progress=norm_prog,
            glare_mode=self.profile.glare_mode,
            glare_intensity=self.profile.glare_intensity,
            seed=self.profile.seed,
        )
        slot_washout = self.compute_slot_glare_washout(glare_map)

        if has_plate and float(np.max(glare_map)) > 1e-3:
            gh, gw = glare_map.shape[:2]
            src_g = np.array([[0.0, 0.0], [float(gw), 0.0], [float(gw), float(gh)], [0.0, float(gh)]], dtype=np.float32)
            dst_g = np.array(frame_annotation.corners.points, dtype=np.float32)
            H_g = cv2.getPerspectiveTransform(src_g, dst_g)
            warped_glare = cv2.warpPerspective(
                glare_map,
                H_g,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )[:, :, np.newaxis]
            # Specular reflection washes out underlying contrast toward pure saturated specular highlight
            exposed = exposed * (1.0 - 0.92 * warped_glare) + 2.4 * warped_glare
        else:
            slot_washout = [0.0] * 12

        # 4. Zero-Anchored Tone Mapping + Strict Monochrome Enforcement for Night IR
        out_bgr = self.tone_map_zero_anchored(exposed, wdr_strength=self.profile.wdr_strength)
        if self.illum_type == IlluminationType.NIGHT_IR:
            gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
            out_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return out_bgr, effective_exposure, slot_washout
