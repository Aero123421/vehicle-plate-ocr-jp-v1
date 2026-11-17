"""Optical modeling: distance-dependent defocus, vignetting, chromatic aberration, and lens dirt."""

import math
import random
from typing import ClassVar
import cv2
import numpy as np

from vpocr_synth.camera.models import CameraProfile, IlluminationType


class OpticsSimulator:
    """Simulates physical lens optics, depth of field, vignetting, chromatic aberration, and lens dirt."""

    _VIGNETTE_GRID_CACHE: ClassVar[dict[tuple[int, int], np.ndarray]] = {}

    def __init__(self, profile: CameraProfile) -> None:
        self.profile = profile
        self.focus_distance_m = 4.5  # Optimal gate focus plane
        self.rng = random.Random(profile.seed + 101)

        # Normalized persistent lens dirt / raindrop pattern for resolution-independent sequence consistency
        self.has_dirt = self.rng.random() < profile.lens_dirt_prob
        self.dirt_spots_norm: list[tuple[float, float, float, float]] = []  # (nx, ny, nr, opacity)
        if self.has_dirt:
            num_spots = self.rng.randint(3, 9)
            for _ in range(num_spots):
                self.dirt_spots_norm.append((
                    self.rng.uniform(0.06, 0.94),
                    self.rng.uniform(0.08, 0.92),
                    self.rng.uniform(0.012, 0.038),
                    self.rng.uniform(0.14, 0.34),
                ))

    @classmethod
    def _get_radial_grid(cls, h: int, w: int) -> np.ndarray:
        cached = cls._VIGNETTE_GRID_CACHE.get((h, w))
        if cached is not None:
            return cached
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        r_sq = ((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
        max_r_sq = max(1.0, float(cx ** 2 + cy ** 2))
        norm_r_sq = r_sq / max_r_sq
        cls._VIGNETTE_GRID_CACHE[(h, w)] = norm_r_sq
        return norm_r_sq

    def apply_optics(
        self,
        bgr_image: np.ndarray,
        distance_m: float,
        is_visible: bool,
    ) -> tuple[np.ndarray, float]:
        """Apply optical transformations, returning (processed_bgr, defocus_radius_px)."""
        h, w = bgr_image.shape[:2]
        img = bgr_image.copy()

        # 1. Defocus Blur (Circle of Confusion based on depth)
        d = max(0.5, distance_m)
        d_diff = abs(1.0 / d - 1.0 / self.focus_distance_m)
        coc_radius = float(min(8.5, max(0.0, d_diff * 45.0 * self.profile.defocus_scale)))

        if coc_radius >= 0.75:
            ksize = int(math.ceil(coc_radius * 2.0))
            if ksize % 2 == 0:
                ksize += 1
            img = cv2.GaussianBlur(img, (ksize, ksize), coc_radius * 0.52)

        # 2. Chromatic Aberration (Lateral color fringe near frame edges, disabled for monochrome IR)
        if w >= 640 and h >= 360 and self.profile.illumination != IlluminationType.NIGHT_IR:
            b, g, r = cv2.split(img)
            M_r = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, 1.0012)
            M_b = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, 0.9988)
            r_warped = cv2.warpAffine(r, M_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            b_warped = cv2.warpAffine(b, M_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            img = cv2.merge([b_warped, g, r_warped])

        # 3. Vignetting (Radial light falloff toward corners)
        vig_strength = self.profile.vignette_strength
        if vig_strength > 0.02:
            norm_r_sq = self._get_radial_grid(h, w)
            falloff = 1.0 - vig_strength * norm_r_sq
            img = np.clip(img.astype(np.float32) * falloff[:, :, np.newaxis], 0, 255).astype(np.uint8)

        # 4. Lens Dirt / Rain Water Droplets (Time-consistent)
        if self.has_dirt:
            for nx, ny, nr, opacity in self.dirt_spots_norm:
                x = int(round(nx * w))
                y = int(round(ny * h))
                r = max(4, int(round(nr * w)))
                x0 = max(0, x - r)
                x1 = min(w, x + r)
                y0 = max(0, y - r)
                y1 = min(h, y + r)
                if x1 > x0 and y1 > y0:
                    roi = img[y0:y1, x0:x1]
                    blur_roi = cv2.GaussianBlur(roi, (15, 15), 5.0)
                    img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1.0 - opacity, blur_roi, opacity, 0)
                    rim_val = 90 if self.profile.illumination in (IlluminationType.NIGHT_IR, IlluminationType.NIGHT_HEADLIGHTS) else 195
                    cv2.circle(img, (x, y), r, (rim_val, rim_val, rim_val), 1, lineType=cv2.LINE_AA)

        return img, coc_radius
