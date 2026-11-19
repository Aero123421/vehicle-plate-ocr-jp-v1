"""Procedural MLIT stamped-die SDF engine and skeleton-first isotropic GlyphSDFAtlas.

Enforces 100% permissive OSS font licensing (`SIL OFL-1.1`, `Apache-2.0`,
`Bitstream-Vera/PD`) with an explicit blocklist preventing GPL/AGPL/Ubuntu fonts
from ever being loaded. Solves the 4-character region (`尾張小牧`, `伊勢志摩`)
vertical-stroke thinning bug via Skeleton-First Horizontal Pre-Compression
followed by Isotropic Circular-Die Euclidean Distance Transform.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import ClassVar
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from vpocr_ocr.vocab import PAD_TOKEN, PlateVocab

# Explicit legal blocklist: GPL / Ubuntu / URW / Narrow system fonts are strictly forbidden.
FORBIDDEN_FONT_SUBSTRINGS: tuple[str, ...] = ("ubuntu", "narrow", "urw")

_REPO_FONTS_DIR = Path(__file__).resolve().parents[3] / "data" / "fonts"

# Verified 100% permissive OSS fonts installed on Linux host + bundled SIL-OFL-1.1 JP rounded/UD gothic.
ALLOWED_OSS_FONTS: tuple[tuple[str, int, str], ...] = (
    (str(_REPO_FONTS_DIR / "MPLUSRounded1c-Bold.ttf"), 0, "SIL-OFL-1.1"),
    (str(_REPO_FONTS_DIR / "MPLUSRounded1c-Medium.ttf"), 0, "SIL-OFL-1.1"),
    (str(_REPO_FONTS_DIR / "ZenMaruGothic-Bold.ttf"), 0, "SIL-OFL-1.1"),
    (str(_REPO_FONTS_DIR / "ZenMaruGothic-Medium.ttf"), 0, "SIL-OFL-1.1"),
    (str(_REPO_FONTS_DIR / "BIZUDPGothic-Bold.ttf"), 0, "SIL-OFL-1.1"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0, "SIL-OFL-1.1"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 5, "SIL-OFL-1.1"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", 0, "SIL-OFL-1.1"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0, "SIL-OFL-1.1"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 5, "SIL-OFL-1.1"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc", 0, "SIL-OFL-1.1"),
    ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0, "Apache-2.0"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0, "SIL-OFL-1.1"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, "Bitstream-Vera"),
)

ALLOWED_OSS_FONT_PATHS: frozenset[str] = frozenset(
    os.path.realpath(p) for p, _, _ in ALLOWED_OSS_FONTS if os.path.isfile(p)
) | frozenset(p for p, _, _ in ALLOWED_OSS_FONTS)


class ForbiddenFontError(ValueError):
    """Raised when attempting to load a non-permissive or blocklisted font."""


def validate_font_path(font_path: str | Path, strict_whitelist: bool = True) -> str:
    """Validate that `font_path` passes the license blocklist and OSS whitelist."""
    raw = str(font_path)
    low = raw.lower()
    for bad in FORBIDDEN_FONT_SUBSTRINGS:
        if bad in low:
            raise ForbiddenFontError(
                f"Forbidden font substring {bad!r} detected in font path: {raw!r}"
            )
    resolved = os.path.realpath(raw)
    resolved_low = resolved.lower()
    for bad in FORBIDDEN_FONT_SUBSTRINGS:
        if bad in resolved_low:
            raise ForbiddenFontError(
                f"Forbidden font substring {bad!r} detected in resolved path: {resolved!r}"
            )
    if strict_whitelist and raw not in ALLOWED_OSS_FONT_PATHS and resolved not in ALLOWED_OSS_FONT_PATHS:
        raise ForbiddenFontError(
            f"Font path {raw!r} is not in verified permissive OSS whitelist."
        )
    if not os.path.isfile(raw):
        raise FileNotFoundError(f"Whitelisted OSS font file not found: {raw!r}")
    return raw


def load_permissive_truetype(
    font_path: str | Path,
    size: int,
    index: int = 0,
    strict_whitelist: bool = True,
) -> ImageFont.FreeTypeFont:
    """Safely load a FreeTypeFont after enforcing the OSS license policy."""
    checked = validate_font_path(font_path, strict_whitelist=strict_whitelist)
    return ImageFont.truetype(checked, size=size, index=index)


def smin_poly(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth-minimum (`C^1` continuous) for CNC inner-corner die fillets."""
    if k <= 1e-6:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


@dataclass(frozen=True)
class DieStyleParams:
    """Parametric variations for Japanese MLIT stamped metal character dies."""

    stroke_jitter: float = 0.0      # Fractional stroke radius delta in [-0.18, +0.18]
    fillet_k: float = 0.018         # Smooth-minimum inner junction fillet radius
    ink_wear: float = 0.0           # Subtle high-frequency edge micro-wear in [0.0, 0.25]
    closed_four: bool = True        # Closed-top triangle vs open-top '4' die variant
    hooked_seven: bool = True       # Left-hooked/slightly curved vs straight '7'
    tiny_cut_one: bool = False      # Mostly serifless straight '1', occasional tiny top cut
    curved_nine: bool = True        # Authentic MLIT bottom-left J-hook '9' vs open diagonal sweep
    flat_top_three: bool = False    # Commercial/regional flat-top '角頭3' vs round-top '3'
    curled_six: bool = True         # Authentic MLIT top-right curled hook '6' vs open sweep
    seed: int = 0


def _sd_segment(
    px: np.ndarray,
    py: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> np.ndarray:
    """Exact Euclidean distance from grid `(px, py)` to line segment `A -> B`."""
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denom = vx * vx + vy * vy
    if denom < 1e-12:
        return np.hypot(wx, wy)
    t = np.clip((wx * vx + wy * vy) / denom, 0.0, 1.0)
    dx = wx - t * vx
    dy = wy - t * vy
    return np.hypot(dx, dy)


def _sd_polyline(
    px: np.ndarray,
    py: np.ndarray,
    pts: list[tuple[float, float]],
) -> np.ndarray:
    """Exact minimum Euclidean distance from grid `(px, py)` to a smooth polyline."""
    d = np.full_like(px, 1e6, dtype=np.float32)
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        d = np.minimum(d, _sd_segment(px, py, ax, ay, bx, by))
    return d


def _arc_points(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    deg_start: float,
    deg_end: float,
    n_pts: int = 20,
) -> list[tuple[float, float]]:
    """Sample smooth elliptical arc points in normalized `[0, 1] x [0, 1]` box coordinates.

    Angle convention: `0°` = right (`+x`), `90°` = down (`+y`), `180°` = left, `270°` = up (`-y`).
    """
    angles = np.linspace(math.radians(deg_start), math.radians(deg_end), n_pts)
    return [(cx + rx * float(math.cos(a)), cy + ry * float(math.sin(a))) for a in angles]


class MLITVectorDieEngine:
    """Procedural smooth-minimum SDF engine for all 25 MLIT ASCII & symbol tokens.

    Tokens covered (25):
      - Digits: `0..9`
      - Symbols: `-`, `・`
      - Classification Latin: `A, C, F, H, K, L, M, P, X, Y`
      - SOFA extra Latin: `B, E, T`
    """

    SUPPORTED_TOKENS: ClassVar[frozenset[str]] = frozenset(
        list("0123456789")
        + ["-", "・"]
        + ["A", "C", "F", "H", "K", "L", "M", "P", "X", "Y"]
        + ["B", "E", "T"]
    )

    SDF_ZERO_U8: ClassVar[int] = 128
    SDF_SCALE_U8: ClassVar[float] = 16.0  # 16 uint8 levels per pixel of signed distance

    @classmethod
    def supports(cls, token: str) -> bool:
        return token in cls.SUPPORTED_TOKENS

    @classmethod
    def render_sdf_u8(
        cls,
        token: str,
        size: int = 64,
        aspect_w_over_h: float = 0.58,
        params: DieStyleParams | None = None,
    ) -> np.ndarray:
        """Render an isotropic signed distance field (`uint8`, `128` = zero contour).

        Skeleton primitives are defined in normalized `[0, 1]^2` slot coordinates and mapped
        to physical space `(u * aspect, v)` BEFORE Euclidean distance calculation so horizontal,
        vertical, and curved strokes share identical physical millimeter stroke thickness.
        """
        if token not in cls.SUPPORTED_TOKENS:
            raise ValueError(f"Token {token!r} is not in MLITVectorDieEngine.SUPPORTED_TOKENS")
        p = params or DieStyleParams()
        jitter = float(np.clip(p.stroke_jitter, -0.18, 0.18))
        aspect = float(np.clip(aspect_w_over_h, 0.40, 1.60))

        # Normalized UV grid at pixel centers in [0, 1]
        coords = (np.arange(size, dtype=np.float32) + 0.5) / float(size)
        v_grid, u_grid = np.meshgrid(coords, coords, indexing="ij")

        # Special symbols with dedicated physical slot geometry (`-` has 20x12mm, `・` has 12x12mm)
        if token == "・":
            px = (u_grid - 0.5)
            py = (v_grid - 0.5)
            radius = 0.37 * (1.0 + 0.6 * jitter)
            sdf_norm = radius - np.hypot(px, py)
            sdf_px = sdf_norm * float(size)
            return cls._quantize_sdf(sdf_px, p)

        if token == "-":
            # Slot box in PlateGeometry is ~20x12mm (aspect ~1.67)
            eff_asp = 1.67
            px = u_grid * eff_asp
            py = v_grid * 1.0
            d_seg = _sd_segment(px, py, 0.22 * eff_asp, 0.50, 0.78 * eff_asp, 0.50)
            r_bar = 0.27 * (1.0 + jitter)
            sdf_px = (r_bar - d_seg) * float(size)
            return cls._quantize_sdf(sdf_px, p)

        # Character glyphs: evaluate Euclidean skeleton distance in isotropic physical coordinates
        px = u_grid * aspect
        py = v_grid

        def seg(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
            return _sd_segment(px, py, x0 * aspect, y0, x1 * aspect, y1)

        def poly(pts_uv: list[tuple[float, float]]) -> np.ndarray:
            scaled = [(x * aspect, y) for x, y in pts_uv]
            return _sd_polyline(px, py, scaled)

        k = float(max(0.0, p.fillet_k))
        d_skel = cls._skeleton_distance(token, seg, poly, k, p)

        # Base physical stroke half-width (~0.062 of char height = ~5.0mm total stroke on 80mm serial,
        # ~2.8mm total stroke on 44mm upper/kana) so enclosed counters ('4','8','9','6','0') remain crisp under blur.
        base_r = min(0.064, 0.051 + 0.020 * min(aspect, 0.85))
        stroke_r = base_r * (1.0 + jitter)

        # Convert physical distance to 64x64 cell pixel units along conservative axis
        px_per_phys = float(size) / max(0.65, min(1.0, aspect * 1.25))
        sdf_px = (stroke_r - d_skel) * px_per_phys
        return cls._quantize_sdf(sdf_px, p)

    @classmethod
    def _quantize_sdf(cls, sdf_px: np.ndarray, p: DieStyleParams) -> np.ndarray:
        if p.ink_wear > 1e-4:
            rng = np.random.default_rng(p.seed + 104729)
            low_noise = rng.standard_normal((12, 12)).astype(np.float32)
            wear_field = cv2.resize(
                low_noise,
                (sdf_px.shape[1], sdf_px.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            # Subtle edge wear near stroke boundary without erasing strokes
            sdf_px = sdf_px - float(p.ink_wear) * 1.15 * np.abs(wear_field)
        vals = np.round(float(cls.SDF_ZERO_U8) + sdf_px * float(cls.SDF_SCALE_U8))
        return np.clip(vals, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _skeleton_distance(
        token: str,
        seg,
        poly,
        k: float,
        p: DieStyleParams,
    ) -> np.ndarray:
        """Authentic Japanese MLIT stamped-die centerlines in normalized `[0, 1]^2` coordinates."""
        if token == "0":
            # Stadium-oval: straight vertical sides joined by smooth semicircular caps
            left = seg(0.20, 0.30, 0.20, 0.70)
            right = seg(0.80, 0.30, 0.80, 0.70)
            top = poly(_arc_points(0.50, 0.30, 0.30, 0.19, 180.0, 360.0, 18))
            bot = poly(_arc_points(0.50, 0.70, 0.30, 0.19, 0.0, 180.0, 18))
            return np.minimum(np.minimum(left, right), np.minimum(top, bot))

        if token == "1":
            # Straight vertical rounded stem (no bottom serif; occasional tiny top-left cut)
            stem = seg(0.50, 0.10, 0.50, 0.90)
            if p.tiny_cut_one:
                cut = seg(0.36, 0.17, 0.50, 0.10)
                return smin_poly(stem, cut, k)
            return stem

        if token == "2":
            # Wide-aperture top arc + crisp diagonal sweep + flat base bar
            top_arc = poly(_arc_points(0.50, 0.29, 0.31, 0.18, 198.0, 392.0, 20))
            diag = poly([(0.76, 0.39), (0.52, 0.62), (0.18, 0.89)])
            base = seg(0.18, 0.89, 0.84, 0.89)
            return smin_poly(np.minimum(top_arc, diag), base, k)

        if token == "3":
            if p.flat_top_three:
                # Authentic MLIT flat-top '角頭3' (used on commercial and regional embossing dies):
                # Horizontal top bar (no downward left tick like '7') + diagonal to indented waist + deep round lower bowl.
                top_bar = seg(0.20, 0.11, 0.80, 0.11)
                diag = seg(0.80, 0.11, 0.38, 0.45)
                bot_lobe = poly([(0.38, 0.45)] + _arc_points(0.51, 0.68, 0.31, 0.22, -88.0, 156.0, 22))
                return smin_poly(smin_poly(top_bar, diag, k * 0.6), bot_lobe, k * 0.6)
            # Round-top '3': wide-aperture upper and lower lobes (open on left so never confused with '8')
            top_lobe = poly(_arc_points(0.49, 0.29, 0.30, 0.18, 202.0, 450.0, 20))
            bot_lobe = poly(_arc_points(0.50, 0.69, 0.32, 0.20, -90.0, 156.0, 20))
            waist = seg(0.39, 0.48, 0.52, 0.48)
            return smin_poly(np.minimum(top_lobe, bot_lobe), waist, k * 0.7)

        if token == "4":
            # Authentic MLIT '4': closed triangular counter (majority) or genuine open-top 2-stem '4'
            hbar = seg(0.11, 0.65, 0.89, 0.65)
            if p.closed_four:
                diag = seg(0.66, 0.10, 0.12, 0.65)
                stem = seg(0.66, 0.10, 0.66, 0.90)
                return smin_poly(smin_poly(diag, hbar, k * 0.8), stem, k * 0.8)
            diag = seg(0.31, 0.12, 0.12, 0.65)
            stem = seg(0.68, 0.12, 0.68, 0.90)
            return smin_poly(smin_poly(diag, hbar, k * 0.8), stem, k * 0.8)

        if token == "5":
            # Crisp horizontal top bar + straight left stem + arched waist & wide-open lower-left aperture (never closes into '6')
            top_bar = seg(0.22, 0.11, 0.81, 0.11)
            left_stem = seg(0.22, 0.11, 0.20, 0.46)
            bowl = poly([(0.20, 0.46)] + _arc_points(0.51, 0.66, 0.31, 0.23, 240.0, 498.0, 22))
            return smin_poly(smin_poly(top_bar, left_stem, k * 0.8), bowl, k * 0.7)

        if token == "6":
            # Closed lower loop + either authentic curled top-right hook or sweeping open spine
            loop = poly(_arc_points(0.50, 0.69, 0.30, 0.20, 0.0, 360.0, 24))
            if p.curled_six:
                hook = poly(
                    [(0.20, 0.69), (0.20, 0.34)]
                    + _arc_points(0.49, 0.31, 0.29, 0.20, 180.0, 342.0, 18)
                )
                return smin_poly(loop, hook, k * 0.5)
            spine = poly([(0.72, 0.11), (0.44, 0.22), (0.24, 0.42), (0.20, 0.68)])
            return smin_poly(loop, spine, k * 0.6)

        if token == "7":
            top_bar = seg(0.16, 0.11, 0.84, 0.11)
            if p.hooked_seven:
                hook = seg(0.16, 0.11, 0.16, 0.23)
                stem = poly([(0.84, 0.11), (0.61, 0.48), (0.43, 0.90)])
                return smin_poly(smin_poly(top_bar, hook, k), stem, k)
            stem = seg(0.84, 0.11, 0.38, 0.90)
            return smin_poly(top_bar, stem, k)

        if token == "8":
            upper = poly(_arc_points(0.50, 0.29, 0.28, 0.18, 0.0, 360.0, 22))
            lower = poly(_arc_points(0.50, 0.69, 0.32, 0.21, 0.0, 360.0, 24))
            return smin_poly(upper, lower, k * 0.5)

        if token == "9":
            # Authentic MLIT '9': closed upper bowl + right vertical stem curving into a bottom-left J-hook
            # with >= 0.23 vertical gap below the upper bowl so blur never closes '9' into '8' or flattens it into '3'.
            loop = poly(_arc_points(0.50, 0.30, 0.30, 0.19, 0.0, 360.0, 24))
            if p.curved_nine:
                j_hook = poly(
                    [(0.80, 0.30), (0.80, 0.67)]
                    + _arc_points(0.50, 0.69, 0.30, 0.20, 0.0, 164.0, 18)
                )
                return smin_poly(loop, j_hook, k * 0.5)
            spine = poly([(0.80, 0.31), (0.77, 0.60), (0.57, 0.82), (0.26, 0.89)])
            return smin_poly(loop, spine, k * 0.6)

        if token == "A":
            l_leg = seg(0.50, 0.10, 0.15, 0.90)
            r_leg = seg(0.50, 0.10, 0.85, 0.90)
            bar = seg(0.27, 0.61, 0.73, 0.61)
            return smin_poly(smin_poly(l_leg, r_leg, k * 0.6), bar, k)

        if token == "B":
            stem = seg(0.19, 0.11, 0.19, 0.89)
            top_b = poly([(0.19, 0.11), (0.54, 0.11)] + _arc_points(0.54, 0.295, 0.25, 0.185, -90.0, 90.0, 14) + [(0.19, 0.48)])
            bot_b = poly([(0.19, 0.48), (0.56, 0.48)] + _arc_points(0.56, 0.685, 0.27, 0.205, -90.0, 90.0, 14) + [(0.19, 0.89)])
            return smin_poly(stem, np.minimum(top_b, bot_b), k * 0.7)

        if token == "C":
            # Wide-opening 'C' (distinct from '0')
            left = seg(0.20, 0.32, 0.20, 0.68)
            top = poly(_arc_points(0.51, 0.32, 0.31, 0.21, 180.0, 326.0, 16))
            bot = poly(_arc_points(0.51, 0.68, 0.31, 0.21, 34.0, 180.0, 16))
            return np.minimum(left, np.minimum(top, bot))

        if token == "E":
            stem = seg(0.20, 0.11, 0.20, 0.89)
            t_bar = seg(0.20, 0.11, 0.82, 0.11)
            m_bar = seg(0.20, 0.49, 0.74, 0.49)
            b_bar = seg(0.20, 0.89, 0.84, 0.89)
            return smin_poly(stem, np.minimum(t_bar, np.minimum(m_bar, b_bar)), k)

        if token == "F":
            # High-waist 'F'
            stem = seg(0.21, 0.11, 0.21, 0.90)
            t_bar = seg(0.21, 0.11, 0.83, 0.11)
            m_bar = seg(0.21, 0.46, 0.74, 0.46)
            return smin_poly(stem, np.minimum(t_bar, m_bar), k)

        if token == "H":
            l_stem = seg(0.19, 0.10, 0.19, 0.90)
            r_stem = seg(0.81, 0.10, 0.81, 0.90)
            bar = seg(0.19, 0.49, 0.81, 0.49)
            return smin_poly(np.minimum(l_stem, r_stem), bar, k)

        if token == "K":
            stem = seg(0.19, 0.10, 0.19, 0.90)
            arm = seg(0.82, 0.11, 0.23, 0.53)
            leg = seg(0.37, 0.43, 0.84, 0.90)
            return smin_poly(stem, smin_poly(arm, leg, k * 0.7), k * 0.7)

        if token == "L":
            stem = seg(0.21, 0.10, 0.21, 0.89)
            base = seg(0.21, 0.89, 0.84, 0.89)
            return smin_poly(stem, base, k)

        if token == "M":
            # Deep-V 'M'
            l_stem = seg(0.15, 0.10, 0.15, 0.90)
            r_stem = seg(0.85, 0.10, 0.85, 0.90)
            l_diag = seg(0.15, 0.11, 0.50, 0.76)
            r_diag = seg(0.85, 0.11, 0.50, 0.76)
            v_mid = smin_poly(l_diag, r_diag, k * 0.6)
            return smin_poly(np.minimum(l_stem, r_stem), v_mid, k * 0.6)

        if token == "P":
            # High-waist closed-bowl 'P'
            stem = seg(0.20, 0.11, 0.20, 0.90)
            bowl = poly(
                [(0.20, 0.11), (0.55, 0.11)]
                + _arc_points(0.55, 0.30, 0.27, 0.19, -90.0, 90.0, 16)
                + [(0.20, 0.49)]
            )
            return smin_poly(stem, bowl, k * 0.8)

        if token == "T":
            top = seg(0.14, 0.11, 0.86, 0.11)
            stem = seg(0.50, 0.11, 0.50, 0.90)
            return smin_poly(top, stem, k)

        if token == "X":
            d1 = seg(0.16, 0.10, 0.84, 0.90)
            d2 = seg(0.84, 0.10, 0.16, 0.90)
            return smin_poly(d1, d2, k * 0.6)

        if token == "Y":
            l_arm = seg(0.15, 0.10, 0.50, 0.50)
            r_arm = seg(0.85, 0.10, 0.50, 0.50)
            stem = seg(0.50, 0.50, 0.50, 0.90)
            return smin_poly(smin_poly(l_arm, r_arm, k * 0.7), stem, k)

        raise ValueError(f"Unhandled MLIT die token: {token!r}")


def _extract_1d_medial_skeleton(mask_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract gap-free 1D-separable medial ridge skeleton + local half-width from binary glyph."""
    fg = (mask_u8 > 127).astype(np.uint8)
    if not np.any(fg):
        return fg, np.zeros_like(fg, dtype=np.float32)

    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    # Deterministic tiny tie-break prevents 2-px plateaus on even-pixel strokes while keeping 100% connectivity
    h, w = dist.shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32) * 1e-4,
        np.arange(w, dtype=np.float32) * 1.3e-4,
        indexing="ij",
    )
    d_tb = dist + (yy + xx) * (fg > 0)

    d_pad = np.pad(d_tb, 1, mode="constant", constant_values=0.0)
    c = d_pad[1:-1, 1:-1]
    horiz_max = (c >= d_pad[1:-1, :-2]) & (c >= d_pad[1:-1, 2:])
    vert_max = (c >= d_pad[:-2, 1:-1]) & (c >= d_pad[2:, 1:-1])
    skel = ((fg > 0) & (dist >= 0.65) & (horiz_max | vert_max)).astype(np.uint8)
    if not np.any(skel):
        skel = fg
    return skel, dist


class GlyphSDFAtlas:
    """Multi-style, multi-aspect-bucket `(2, 8, 254, 64, 64)` `uint8` SDF glyph atlas.

    Bucket 0 (`aspect = 1.0`): Standard square/wide glyphs (1–2 char regions, kana, wide slots).
    Bucket 1 (`aspect = 0.58`): Skeleton-First Pre-Compressed + Isotropically Dilated glyphs
        for 3–4 char regions (`尾張小牧`, `伊勢志摩`) and condensed slots so vertical strokes
        retain full physical millimeter width instead of thinning by 42%.
    """

    NUM_STYLES: ClassVar[int] = 8
    GLYPH_SIZE: ClassVar[int] = 64
    ASPECT_BUCKETS: ClassVar[tuple[float, float]] = (1.0, 0.58)
    SDF_ZERO_U8: ClassVar[int] = 128
    SDF_SCALE_U8: ClassVar[float] = 16.0

    # 8 pure Japanese JIS permissive OSS style recipes (SIL-OFL-1.1 rounded/UD/road-sign gothic):
    # (jp_font_path, jp_font_index, ascii_mode, die_params, die_radius_scale)
    STYLE_RECIPES: ClassVar[tuple[tuple[str, int, str, DieStyleParams, float], ...]] = (
        (
            str(_REPO_FONTS_DIR / "MPLUSRounded1c-Bold.ttf"),
            0,
            "mlit_die",
            DieStyleParams(
                stroke_jitter=0.04, fillet_k=0.019, closed_four=True, hooked_seven=True,
                curved_nine=True, flat_top_three=False, curled_six=True, seed=1,
            ),
            1.05,
        ),
        (
            str(_REPO_FONTS_DIR / "ZenMaruGothic-Bold.ttf"),
            0,
            "mlit_die",
            DieStyleParams(
                stroke_jitter=0.00, fillet_k=0.018, closed_four=True, hooked_seven=True,
                curved_nine=True, flat_top_three=True, curled_six=True, seed=2,
            ),
            1.01,
        ),
        (
            str(_REPO_FONTS_DIR / "BIZUDPGothic-Bold.ttf"),
            0,
            "mlit_die",
            DieStyleParams(
                stroke_jitter=0.14, fillet_k=0.023, closed_four=True, hooked_seven=False,
                curved_nine=True, flat_top_three=True, curled_six=True, seed=3,
            ),
            1.14,
        ),
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            0,
            "mlit_die",
            DieStyleParams(
                stroke_jitter=0.07, fillet_k=0.020, closed_four=True, hooked_seven=True,
                tiny_cut_one=True, curved_nine=True, flat_top_three=False, curled_six=True, seed=4,
            ),
            1.07,
        ),
        (
            str(_REPO_FONTS_DIR / "MPLUSRounded1c-Medium.ttf"),
            0,
            "mlit_die",
            DieStyleParams(
                stroke_jitter=-0.03, fillet_k=0.016, ink_wear=0.06, closed_four=False, hooked_seven=False,
                curved_nine=True, flat_top_three=True, curled_six=True, seed=5,
            ),
            0.97,
        ),
        (
            str(_REPO_FONTS_DIR / "ZenMaruGothic-Medium.ttf"),
            0,
            "mlit_die",
            DieStyleParams(
                stroke_jitter=0.12, fillet_k=0.024, ink_wear=0.05, closed_four=True, hooked_seven=True,
                curved_nine=False, flat_top_three=False, curled_six=False, seed=6,
            ),
            1.11,
        ),
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
            0,
            "mlit_die",
            DieStyleParams(
                stroke_jitter=0.02, fillet_k=0.018, ink_wear=0.09, closed_four=True, hooked_seven=False,
                tiny_cut_one=True, curved_nine=True, flat_top_three=False, curled_six=True, seed=7,
            ),
            1.02,
        ),
        (
            str(_REPO_FONTS_DIR / "BIZUDPGothic-Bold.ttf"),
            0,
            "mlit_die",
            DieStyleParams(
                stroke_jitter=0.09, fillet_k=0.021, ink_wear=0.09, closed_four=True, hooked_seven=True,
                curved_nine=True, flat_top_three=True, curled_six=True, seed=8,
            ),
            1.09,
        ),
    )

    def __init__(self, vocab: PlateVocab | None = None) -> None:
        self.vocab = vocab or PlateVocab()
        self.vocab_size = self.vocab.size
        assert self.vocab_size == 254, f"Expected 254 vocab tokens, got {self.vocab_size}"

        # Validate every configured OSS font path upfront so blocklist violations fail immediately
        for fpath, _, mode, _, _ in self.STYLE_RECIPES:
            validate_font_path(fpath)
            if mode == "liberation":
                validate_font_path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")
            elif mode == "dejavu":
                validate_font_path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

        self.sdf_all: np.ndarray = self._build_all_buckets()  # (2, 8, 254, 64, 64) uint8
        self._device_cache: dict[torch.device, torch.Tensor] = {}

    @property
    def sdf_standard(self) -> np.ndarray:
        """Standard (`aspect = 1.0`) SDF atlas of shape `(8, 254, 64, 64)` `uint8`."""
        return self.sdf_all[0]

    @property
    def sdf_condensed(self) -> np.ndarray:
        """Condensed (`aspect = 0.58`) skeleton-first isotropic SDF atlas `(8, 254, 64, 64)` `uint8`."""
        return self.sdf_all[1]

    def get_bucket_sdf(self, bucket: int | float = 0) -> np.ndarray:
        """Return `(8, 254, 64, 64)` `uint8` SDF atlas for bucket index `0/1` or aspect float."""
        if isinstance(bucket, int) and bucket in (0, 1):
            return self.sdf_all[bucket]
        idx = 1 if float(bucket) < 0.78 else 0
        return self.sdf_all[idx]

    def to_torch(self, device: torch.device | str) -> torch.Tensor:
        """Return cached `(2, 8, 254, 64, 64)` `uint8` tensor (~16.6 MB) on `device`."""
        dev = torch.device(device)
        cached = self._device_cache.get(dev)
        if cached is None:
            cached = torch.from_numpy(self.sdf_all).to(dev, non_blocking=True)
            self._device_cache[dev] = cached
        return cached

    def _build_all_buckets(self) -> np.ndarray:
        out = np.zeros(
            (2, self.NUM_STYLES, self.vocab_size, self.GLYPH_SIZE, self.GLYPH_SIZE),
            dtype=np.uint8,
        )

        # Cache loaded PIL fonts & rasterized base glyph bitmaps per unique (font_path, font_index)
        font_cache: dict[tuple[str, int, int], ImageFont.FreeTypeFont] = {}

        def get_font(path: str, idx: int, sz: int) -> ImageFont.FreeTypeFont:
            key = (path, idx, sz)
            f = font_cache.get(key)
            if f is None:
                f = load_permissive_truetype(path, size=sz, index=idx)
                font_cache[key] = f
            return f

        # Pre-rasterize unique Japanese font bitmaps + 2x (128x128) skeleton/counter/Voronoi-gap sheets
        skel_face_cache: dict[
            tuple[str, int],
            dict[
                int,
                tuple[
                    np.ndarray,
                    float,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    int,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ],
            ],
        ] = {}
        jp_faces: dict[
            tuple[str, int],
            dict[
                int,
                tuple[
                    np.ndarray,
                    np.ndarray,
                    float,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    int,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ],
            ],
        ] = {}
        jp_tids = [
            tid
            for tid in range(1, self.vocab_size)
            if not MLITVectorDieEngine.supports(self.vocab.id_to_token[tid])
        ]
        k_close19 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        k_ones3 = np.ones((3, 3), dtype=np.uint8)

        for fpath, fidx, _, _, _ in self.STYLE_RECIPES:
            fkey = (fpath, fidx)
            if fkey in jp_faces:
                continue
            if "NotoSansCJK" in fpath:
                skel_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            elif "MPLUSRounded1c-Bold" in fpath:
                skel_path = str(_REPO_FONTS_DIR / "MPLUSRounded1c-Medium.ttf")
            elif "ZenMaruGothic-Bold" in fpath:
                skel_path = str(_REPO_FONTS_DIR / "ZenMaruGothic-Medium.ttf")
            else:
                skel_path = fpath
            skey = (skel_path, fidx)
            if skel_key_data := skel_face_cache.get(skey):
                skel_map = skel_key_data
            else:
                # 2x supersampled (128x128) skeleton & topological Voronoi gap extraction
                skel_font_2x = get_font(skel_path, fidx, 112)
                s2_stride = 136  # 128 + 2*4 pad
                sk_sheet_2x = np.zeros((16 * s2_stride, 16 * s2_stride), dtype=np.uint8)
                for tid in jp_tids:
                    tok = self.vocab.id_to_token[tid]
                    r_r, r_c = divmod(tid, 16)
                    sk_sheet_2x[
                        r_r * s2_stride + 4 : r_r * s2_stride + 132,
                        r_c * s2_stride + 4 : r_c * s2_stride + 132,
                    ] = self._rasterize_char_tight(
                        tok, skel_font_2x, canvas_hw=(128, 128), margin_frac=0.045
                    )
                dist_sh_2x = cv2.distanceTransform(sk_sheet_2x, cv2.DIST_L2, 5)
                sh_h2, sh_w2 = dist_sh_2x.shape
                yy2 = np.arange(sh_h2, dtype=np.float32)[:, None] * 1e-4
                xx2 = np.arange(sh_w2, dtype=np.float32)[None, :] * 1.3e-4
                d_tb2 = dist_sh_2x + (yy2 + xx2) * (sk_sheet_2x > 0)
                d_pad2 = np.pad(d_tb2, 1, mode="constant", constant_values=0.0)
                c2_p = d_pad2[1:-1, 1:-1]
                ridge_sh_2x = (
                    (sk_sheet_2x > 0)
                    & (dist_sh_2x >= 0.85)
                    & (
                        ((c2_p >= d_pad2[1:-1, :-2]) & (c2_p >= d_pad2[1:-1, 2:]))
                        | ((c2_p >= d_pad2[:-2, 1:-1]) & (c2_p >= d_pad2[2:, 1:-1]))
                    )
                ).astype(np.uint8)

                skel_map = {}
                for tid in jp_tids:
                    r_r, r_c = divmod(tid, 16)
                    sk_bmp_2x = sk_sheet_2x[
                        r_r * s2_stride + 4 : r_r * s2_stride + 132,
                        r_c * s2_stride + 4 : r_c * s2_stride + 132,
                    ]
                    sk_cell_2x = ridge_sh_2x[
                        r_r * s2_stride + 4 : r_r * s2_stride + 132,
                        r_c * s2_stride + 4 : r_c * s2_stride + 132,
                    ]
                    if not np.any(sk_cell_2x):
                        sk_cell_2x = sk_bmp_2x
                    # Equivalent 1x skeleton density (sum / (2 * 64 * 64) since 1D curve length scales 2x)
                    dens = float(sk_cell_2x.sum()) / float(2 * 64 * 64)
                    core_2x, voronoi_gap_2x, end_2x, n_counters = (
                        self._extract_counter_voronoi_and_end_features_2x(
                            sk_bmp_2x, sk_cell_2x, k_close19, k_ones3
                        )
                    )
                    # Downproject 2x topological masks to 64x64 for Bucket 0 open-counter / bay preservation
                    sk64 = np.zeros((64, 64), dtype=np.uint8)
                    sy2, sx2 = np.nonzero(sk_cell_2x)
                    if sy2.size > 0:
                        sk64[np.clip(sy2 >> 1, 0, 63), np.clip(sx2 >> 1, 0, 63)] = 1
                    core64 = np.zeros((64, 64), dtype=np.uint8)
                    cy2, cx2 = np.nonzero(core_2x)
                    if cy2.size > 0:
                        core64[np.clip(cy2 >> 1, 0, 63), np.clip(cx2 >> 1, 0, 63)] = 1
                    gap64 = np.zeros((64, 64), dtype=np.uint8)
                    gy2, gx2 = np.nonzero(voronoi_gap_2x)
                    if gy2.size > 0:
                        gap64[np.clip(gy2 >> 1, 0, 63), np.clip(gx2 >> 1, 0, 63)] = 1
                        gap64[sk64 > 0] = 0
                    end64 = np.zeros((64, 64), dtype=np.uint8)
                    ey2, ex2 = np.nonzero(end_2x)
                    if ey2.size > 0:
                        end64[np.clip(ey2 >> 1, 0, 63), np.clip(ex2 >> 1, 0, 63)] = 1

                    skel_map[tid] = (
                        sk64,
                        dens,
                        core64,
                        gap64,
                        end64,
                        n_counters,
                        sk_cell_2x,
                        core_2x,
                        voronoi_gap_2x,
                        end_2x,
                    )
                skel_face_cache[skey] = skel_map

            bold_font = get_font(fpath, fidx, 54)
            face_map: dict[
                int,
                tuple[
                    np.ndarray,
                    np.ndarray,
                    float,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    int,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ],
            ] = {}
            for tid in jp_tids:
                tok = self.vocab.id_to_token[tid]
                bmp64 = self._rasterize_char_tight(tok, bold_font, canvas_hw=(64, 64), margin_frac=0.045)
                (
                    sk64,
                    dens,
                    core64,
                    gap64,
                    end64,
                    n_counters,
                    sk_cell_2x,
                    core_2x,
                    voronoi_gap_2x,
                    end_2x,
                ) = skel_map[tid]
                if np.any(core64) or np.any(gap64):
                    bmp64 = bmp64.copy()
                    # Preserve enclosed MLIT counters (口/田/日/目 & small Hiragana loops) without breaking skeleton
                    bmp64[(core64 > 0) & (sk64 == 0)] = 0
                    if np.any(gap64):
                        d_bold_in = cv2.distanceTransform(bmp64, cv2.DIST_L2, 3)
                        bmp64[(gap64 > 0) & (sk64 == 0) & (d_bold_in <= 1.85)] = 0
                face_map[tid] = (
                    bmp64,
                    sk64,
                    dens,
                    core64,
                    gap64,
                    end64,
                    n_counters,
                    sk_cell_2x,
                    core_2x,
                    voronoi_gap_2x,
                    end_2x,
                )
            jp_faces[fkey] = face_map

        # Tiled sheet helper: 16x16 cells of (80x80) = (1280x1280) -> single cv2.distanceTransform in <2ms!
        cell_pad = 8
        cell_sz = self.GLYPH_SIZE + 2 * cell_pad  # 80

        # 2x supersampled condensed physical sheet (128x76, aspect 76/128 = 0.594 ~ 0.58)
        w_phys_2x = 76
        h_phys_2x = 128
        c_pad_2x = 12
        c_cell_w2 = w_phys_2x + 2 * c_pad_2x   # 100
        c_cell_h2 = h_phys_2x + 2 * c_pad_2x   # 152

        for s_idx, (fpath, fidx, ascii_mode, die_params, r_scale) in enumerate(self.STYLE_RECIPES):
            face_map = jp_faces[(fpath, fidx)]
            ascii_font = None
            if ascii_mode == "liberation":
                ascii_font = get_font("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0, 60)
            elif ascii_mode == "dejavu":
                ascii_font = get_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, 60)

            for b_idx, aspect in enumerate(self.ASPECT_BUCKETS):
                is_condensed = b_idx == 1
                # Tiled sheet for bitmap-to-SDF tokens in Bucket 0
                sheet_fg = np.zeros((16 * cell_sz, 16 * cell_sz), dtype=np.uint8)
                sheet_core = np.zeros((16 * cell_sz, 16 * cell_sz), dtype=np.uint8)
                sheet_gap = np.zeros((16 * cell_sz, 16 * cell_sz), dtype=np.uint8)
                sheet_end = np.zeros((16 * cell_sz, 16 * cell_sz), dtype=np.uint8)
                sheet_tids: list[int] = []
                std_jp_meta: dict[int, tuple[bool, bool, bool, bool]] = {}

                # 2x supersampled condensed physical sheets (128, 76) per cell
                cond_skel_sheet_2x = np.zeros((16 * c_cell_h2, 16 * c_cell_w2), dtype=np.uint8)
                cond_core_sheet_2x = np.zeros((16 * c_cell_h2, 16 * c_cell_w2), dtype=np.uint8)
                cond_gap_sheet_2x = np.zeros((16 * c_cell_h2, 16 * c_cell_w2), dtype=np.uint8)
                cond_end_sheet_2x = np.zeros((16 * c_cell_h2, 16 * c_cell_w2), dtype=np.uint8)
                cond_radii: dict[int, tuple[float, bool, bool, bool, bool]] = {}

                for tid in range(1, self.vocab_size):
                    tok = self.vocab.id_to_token[tid]
                    if MLITVectorDieEngine.supports(tok):
                        if ascii_font is not None and tok not in ("-", "・"):
                            bmp = self._rasterize_char_tight(tok, ascii_font, canvas_hw=(64, 64), margin_frac=0.11)
                            r_row, r_col = divmod(tid, 16)
                            y0 = r_row * cell_sz + cell_pad
                            x0 = r_col * cell_sz + cell_pad
                            sheet_fg[y0 : y0 + 64, x0 : x0 + 64] = bmp
                            sheet_tids.append(tid)
                        else:
                            char_asp = 0.90 if not is_condensed else 0.56
                            out[b_idx, s_idx, tid] = MLITVectorDieEngine.render_sdf_u8(
                                tok,
                                size=self.GLYPH_SIZE,
                                aspect_w_over_h=char_asp,
                                params=die_params,
                            )
                        continue

                    (
                        bmp64,
                        _sk64,
                        density,
                        core64,
                        gap64,
                        end64,
                        n_counters,
                        sk_cell_2x,
                        core_2x,
                        voronoi_gap_2x,
                        end_2x,
                    ) = face_map[tid]
                    is_hiragana = len(tok) == 1 and ("\u3040" <= tok <= "\u309f")
                    r_row, r_col = divmod(tid, 16)

                    if not is_condensed:
                        # Bucket 0 (aspect=1.0): open-counter + Voronoi-gap shaped bitmap SDF
                        y0 = r_row * cell_sz + cell_pad
                        x0 = r_col * cell_sz + cell_pad
                        sheet_fg[y0 : y0 + 64, x0 : x0 + 64] = bmp64
                        sheet_tids.append(tid)
                        hc = bool(np.any(core64))
                        hg = bool(np.any(gap64))
                        he = bool(np.any(end64))
                        if hc:
                            sheet_core[y0 : y0 + 64, x0 : x0 + 64] = core64
                        if hg:
                            sheet_gap[y0 : y0 + 64, x0 : x0 + 64] = gap64
                        if he:
                            sheet_end[y0 : y0 + 64, x0 : x0 + 64] = end64
                        if hc or hg or he:
                            std_jp_meta[tid] = (hc, hg, he, is_hiragana)
                    else:
                        # Bucket 1 (aspect=0.58): 2x Supersampled (76x128) Skeleton + Background Voronoi Ridge!
                        sy2, sx2 = np.nonzero(sk_cell_2x)
                        ty2 = sy2
                        tx2 = np.clip(
                            (sx2.astype(np.float32) * ((w_phys_2x - 1.0) / 127.0)).round().astype(np.int32),
                            0,
                            w_phys_2x - 1,
                        )
                        cy0 = r_row * c_cell_h2 + c_pad_2x
                        cx0 = r_col * c_cell_w2 + c_pad_2x
                        cond_skel_sheet_2x[cy0 + ty2, cx0 + tx2] = 1

                        has_core = bool(np.any(core_2x))
                        if has_core:
                            ky2, kx2 = np.nonzero(core_2x)
                            ktx2 = np.clip(
                                (kx2.astype(np.float32) * ((w_phys_2x - 1.0) / 127.0)).round().astype(np.int32),
                                0,
                                w_phys_2x - 1,
                            )
                            cond_core_sheet_2x[cy0 + ky2, cx0 + ktx2] = 1
                            cond_core_sheet_2x[cy0 + ty2, cx0 + tx2] = 0

                        has_gap = bool(np.any(voronoi_gap_2x))
                        if has_gap:
                            gy2, gx2 = np.nonzero(voronoi_gap_2x)
                            gtx2 = np.clip(
                                (gx2.astype(np.float32) * ((w_phys_2x - 1.0) / 127.0)).round().astype(np.int32),
                                0,
                                w_phys_2x - 1,
                            )
                            cond_gap_sheet_2x[cy0 + gy2, cx0 + gtx2] = 1
                            # Never allow mapped Voronoi background ridge to overwrite active stroke skeleton
                            cond_gap_sheet_2x[cy0 + ty2, cx0 + tx2] = 0
                            has_gap = bool(
                                np.any(
                                    cond_gap_sheet_2x[
                                        cy0 : cy0 + h_phys_2x, cx0 : cx0 + w_phys_2x
                                    ]
                                )
                            )

                        has_end = bool(np.any(end_2x))
                        if has_end:
                            ey2, ex2 = np.nonzero(end_2x)
                            etx2 = np.clip(
                                (ex2.astype(np.float32) * ((w_phys_2x - 1.0) / 127.0)).round().astype(np.int32),
                                0,
                                w_phys_2x - 1,
                            )
                            cond_end_sheet_2x[cy0 + ey2, cx0 + etx2] = 1

                        # Adaptive isotropic physical die radius (in 1x physical pixel units)
                        base_die_r = (
                            float(np.clip(1.94 - 3.8 * density - 0.06 * min(n_counters, 4), 1.08, 1.80))
                            * r_scale
                        )
                        cond_radii[tid] = (base_die_r, has_core, has_gap, has_end, is_hiragana)

                # Batch Euclidean SDF for standard Bucket-0 bitmap glyphs
                if sheet_tids:
                    d_in = np.clip(cv2.distanceTransform(sheet_fg, cv2.DIST_L2, 3), 0.0, 12.0)
                    d_out = np.clip(cv2.distanceTransform(1 - (sheet_fg > 0).astype(np.uint8), cv2.DIST_L2, 3), 0.0, 12.0)
                    d_core_sh = (
                        cv2.distanceTransform(1 - sheet_core, cv2.DIST_L2, 3)
                        if np.any(sheet_core)
                        else None
                    )
                    d_gap_sh = (
                        cv2.distanceTransform(1 - sheet_gap, cv2.DIST_L2, 3)
                        if np.any(sheet_gap)
                        else None
                    )
                    d_end_sh = (
                        cv2.distanceTransform(1 - sheet_end, cv2.DIST_L2, 3)
                        if np.any(sheet_end)
                        else None
                    )
                    sdf_sheet = (d_in - d_out) + (r_scale - 1.0) * 3.5
                    for tid in sheet_tids:
                        r_row, r_col = divmod(tid, 16)
                        y0 = r_row * cell_sz + cell_pad
                        x0 = r_col * cell_sz + cell_pad
                        sdf_cell = sdf_sheet[y0 : y0 + 64, x0 : x0 + 64]
                        if meta := std_jp_meta.get(tid):
                            hc, hg, he, is_hira = meta
                            sdf_cell = sdf_cell.copy()
                            if hc and d_core_sh is not None:
                                d_cc = d_core_sh[y0 : y0 + 64, x0 : x0 + 64]
                                core_floor = -1.25 if is_hira else -0.92
                                core_slope = 1.02 if is_hira else 0.90
                                sdf_cell = np.minimum(sdf_cell, core_floor + core_slope * d_cc)
                            if hg and d_gap_sh is not None:
                                d_gp = d_gap_sh[y0 : y0 + 64, x0 : x0 + 64]
                                sdf_cell = np.minimum(sdf_cell, -0.52 + 1.05 * d_gp)
                            if he and d_end_sh is not None:
                                d_end = d_end_sh[y0 : y0 + 64, x0 : x0 + 64]
                                sdf_cell = sdf_cell - 0.26 * np.exp(-0.45 * d_end * d_end)
                        out[b_idx, s_idx, tid] = np.clip(
                            np.round(float(self.SDF_ZERO_U8) + sdf_cell * float(self.SDF_SCALE_U8)),
                            0.0,
                            255.0,
                        ).astype(np.uint8)

                # Batch Euclidean SDF in 2x supersampled (128, 76) float space -> downsample float SDF to 64x64
                if cond_radii:
                    d_from_skel_2x = np.clip(
                        cv2.distanceTransform(1 - cond_skel_sheet_2x, cv2.DIST_L2, 5), 0.0, 24.0
                    )
                    d_from_core_2x = (
                        cv2.distanceTransform(1 - cond_core_sheet_2x, cv2.DIST_L2, 5)
                        if np.any(cond_core_sheet_2x)
                        else None
                    )
                    d_from_gap_2x = (
                        cv2.distanceTransform(1 - cond_gap_sheet_2x, cv2.DIST_L2, 5)
                        if np.any(cond_gap_sheet_2x)
                        else None
                    )
                    d_from_end_2x = (
                        cv2.distanceTransform(1 - cond_end_sheet_2x, cv2.DIST_L2, 3)
                        if np.any(cond_end_sheet_2x)
                        else None
                    )
                    for tid, (die_r, has_core, has_gap, has_end, is_hira) in cond_radii.items():
                        r_row, r_col = divmod(tid, 16)
                        cy0 = r_row * c_cell_h2 + c_pad_2x
                        cx0 = r_col * c_cell_w2 + c_pad_2x
                        d_cell_1x = 0.5 * d_from_skel_2x[cy0 : cy0 + h_phys_2x, cx0 : cx0 + w_phys_2x]
                        # Isotropic circular dilation in 2x supersampled physical float space (1x SDF units)
                        sdf_phys_2x = die_r - d_cell_1x
                        if has_end and d_from_end_2x is not None:
                            de_1x = 0.5 * d_from_end_2x[cy0 : cy0 + h_phys_2x, cx0 : cx0 + w_phys_2x]
                            sdf_phys_2x = sdf_phys_2x - 0.22 * np.exp(-0.55 * de_1x * de_1x)
                        if has_gap and d_from_gap_2x is not None:
                            dg_2x = d_from_gap_2x[cy0 : cy0 + h_phys_2x, cx0 : cx0 + w_phys_2x]
                            # Protect open inter-stroke corridors via 2D background Voronoi medial ridge
                            sdf_phys_2x = np.minimum(sdf_phys_2x, -0.58 + 0.78 * dg_2x)
                        if has_core and d_from_core_2x is not None:
                            dc_1x = 0.5 * d_from_core_2x[cy0 : cy0 + h_phys_2x, cx0 : cx0 + w_phys_2x]
                            core_floor = -1.25 if is_hira else -0.88
                            core_slope = 1.12 if is_hira else 0.94
                            sdf_phys_2x = np.minimum(sdf_phys_2x, core_floor + core_slope * dc_1x)
                        # Downsample 2x supersampled float SDF (128x76 -> 64x64) BEFORE uint8 quantization!
                        sdf_uv = cv2.resize(
                            sdf_phys_2x,
                            (self.GLYPH_SIZE, self.GLYPH_SIZE),
                            interpolation=cv2.INTER_AREA,
                        )
                        out[b_idx, s_idx, tid] = np.clip(
                            np.round(float(self.SDF_ZERO_U8) + sdf_uv * float(self.SDF_SCALE_U8)),
                            0.0,
                            255.0,
                        ).astype(np.uint8)

        # Enforce <PAD> at token 0 is strictly empty zero
        out[:, :, 0, :, :] = 0
        return out

    @staticmethod
    def _extract_counter_voronoi_and_end_features_2x(
        sk_bmp_2x: np.ndarray,
        sk_cell_2x: np.ndarray,
        k_close19: np.ndarray,
        k_ones3: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Extract 2x enclosed counter cores, 2D background Voronoi inter-stroke ridges, & stroke-ends."""
        bg = (1 - (sk_bmp_2x > 0).astype(np.uint8)).astype(np.uint8)
        d_bg = cv2.distanceTransform(bg, cv2.DIST_L2, 5)
        num_cc, labels = cv2.connectedComponents(bg, connectivity=4)
        core_mask = np.zeros_like(sk_bmp_2x, dtype=np.uint8)
        counter_mask = np.zeros_like(sk_bmp_2x, dtype=np.uint8)
        n_counters = 0
        if num_cc > 2:
            border_lbls = (
                set(np.unique(labels[0, :]).tolist())
                | set(np.unique(labels[-1, :]).tolist())
                | set(np.unique(labels[:, 0]).tolist())
                | set(np.unique(labels[:, -1]).tolist())
            )
            for lbl in range(1, num_cc):
                if lbl in border_lbls:
                    continue
                comp = labels == lbl
                area = int(comp.sum())
                if area < 2:
                    continue
                n_counters += 1
                counter_mask[comp] = 1
                comp_d = d_bg * comp
                peak_d = float(comp_d.max())
                thresh = max(1.15, 0.52 * peak_d)
                c_core = comp & (d_bg >= thresh)
                if not np.any(c_core):
                    py, px = np.unravel_index(int(np.argmax(comp_d)), comp_d.shape)
                    core_mask[py, px] = 1
                else:
                    core_mask[c_core] = 1

        # 2D background Voronoi medial ridge inside morphological stroke corridor hull
        fg_u8 = (sk_bmp_2x > 0).astype(np.uint8)
        closed = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, k_close19)
        corridor = (closed > 0) & (bg > 0) & (counter_mask == 0)
        if np.any(corridor):
            h2, w2 = d_bg.shape
            yy = np.arange(h2, dtype=np.float32)[:, None] * 1e-4
            xx = np.arange(w2, dtype=np.float32)[None, :] * 1.3e-4
            db_tb = d_bg + (yy + xx) * (bg > 0)
            db_pad = np.pad(db_tb, 1, mode="constant", constant_values=0.0)
            c = db_pad[1:-1, 1:-1]
            h_ridge = (c >= db_pad[1:-1, :-2]) & (c >= db_pad[1:-1, 2:])
            v_ridge = (c >= db_pad[:-2, 1:-1]) & (c >= db_pad[2:, 1:-1])
            voronoi_gap = (
                corridor & (d_bg >= 1.15) & (d_bg <= 9.5) & (h_ridge | v_ridge)
            ).astype(np.uint8)
        else:
            voronoi_gap = np.zeros_like(sk_bmp_2x, dtype=np.uint8)

        sk_u8 = (sk_cell_2x > 0).astype(np.uint8)
        nbr = cv2.filter2D(sk_u8, cv2.CV_8U, k_ones3, borderType=cv2.BORDER_CONSTANT)
        end_mask = ((sk_u8 > 0) & (nbr == 2)).astype(np.uint8)
        return core_mask, voronoi_gap, end_mask, n_counters

    @staticmethod
    def _rasterize_char_tight(
        char: str,
        font: ImageFont.FreeTypeFont,
        canvas_hw: tuple[int, int] = (64, 64),
        margin_frac: float = 0.045,
    ) -> np.ndarray:
        """Render single glyph tightly centered into `canvas_hw` binary `{0, 1}` uint8 array."""
        ch_h, ch_w = canvas_hw
        sc_sz = max(128, 2 * max(ch_h, ch_w))
        scratch = Image.new("L", (sc_sz, sc_sz), 0)
        draw = ImageDraw.Draw(scratch)
        draw.text((sc_sz // 8, sc_sz // 10), char, fill=255, font=font)
        arr = np.asarray(scratch, dtype=np.uint8)
        ys, xs = np.nonzero(arr > 96)
        if ys.size == 0:
            return np.zeros(canvas_hw, dtype=np.uint8)

        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = arr[y0:y1, x0:x1]

        # Preserve natural aspect for single-stroke glyphs like '一', '川', 'ノ'-like kana so '一' isn't stretched vertically to full box!
        gh, gw = crop.shape
        avail_h = int(round(ch_h * (1.0 - 2.0 * margin_frac)))
        avail_w = int(round(ch_w * (1.0 - 2.0 * margin_frac)))

        # Keep natural height ratio if flat horizontal character like '一' (gh / gw < 0.35)
        if gh < 0.38 * gw:
            tgt_w = avail_w
            tgt_h = max(4, int(round(avail_w * (gh / max(1.0, float(gw))))))
        elif gw < 0.38 * gh:
            tgt_h = avail_h
            tgt_w = max(4, int(round(avail_h * (gw / max(1.0, float(gh))))))
        else:
            tgt_h = avail_h
            tgt_w = avail_w

        resized = cv2.resize(crop, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        out = np.zeros(canvas_hw, dtype=np.uint8)
        oy = (ch_h - tgt_h) // 2
        ox = (ch_w - tgt_w) // 2
        out[oy : oy + tgt_h, ox : ox + tgt_w] = (resized > 110).astype(np.uint8)
        return out

    def render_physical_mask(
        self,
        token_or_id: str | int,
        phys_hw: tuple[int, int] = (64, 38),
        style_idx: int = 0,
        condensed: bool | None = None,
        sdf_threshold_offset: float = 0.0,
    ) -> np.ndarray:
        """Render binary `{0, 1}` glyph mask at physical slot dimensions `phys_hw`."""
        tid = (
            self.vocab.token_to_id[token_or_id]
            if isinstance(token_or_id, str)
            else int(token_or_id)
        )
        ph, pw = phys_hw
        s_idx = style_idx % self.NUM_STYLES
        if condensed is None:
            asp = float(pw) / max(1.0, float(ph))
            alpha = float(np.clip((0.94 - asp) / (0.94 - 0.58), 0.0, 1.0))
            sdf_f = (
                (1.0 - alpha) * self.sdf_all[0, s_idx, tid].astype(np.float32)
                + alpha * self.sdf_all[1, s_idx, tid].astype(np.float32)
            )
        else:
            b_idx = 1 if condensed else 0
            sdf_f = self.sdf_all[b_idx, s_idx, tid].astype(np.float32)
        sdf_p = cv2.resize(
            sdf_f,
            (pw, ph),
            interpolation=cv2.INTER_LINEAR,
        )
        thresh_u8 = float(self.SDF_ZERO_U8) - float(sdf_threshold_offset) * float(self.SDF_SCALE_U8)
        return (sdf_p >= thresh_u8).astype(np.uint8)

    @classmethod
    def instance(cls, vocab: PlateVocab | None = None) -> "GlyphSDFAtlas":
        """Return process-cached `GlyphSDFAtlas` singleton."""
        return get_default_glyph_atlas(vocab=vocab)

    def get_sdf(
        self,
        token_or_id: str | int,
        bucket: int = 0,
        style_idx: int = 0,
    ) -> np.ndarray:
        """Return zero-centered float32 SDF (`> 0` inside glyph) for `(bucket, style_idx)`."""
        tid = (
            self.vocab.token_to_id[token_or_id]
            if isinstance(token_or_id, str)
            else int(token_or_id)
        )
        b_idx = int(bucket) % len(self.ASPECT_BUCKETS)
        s_idx = int(style_idx) % self.NUM_STYLES
        return self.sdf_all[b_idx, s_idx, tid].astype(np.float32) - float(self.SDF_ZERO_U8)


_DEFAULT_ATLAS: GlyphSDFAtlas | None = None


def get_default_glyph_atlas(vocab: PlateVocab | None = None) -> GlyphSDFAtlas:
    """Return process-cached `GlyphSDFAtlas` singleton."""
    global _DEFAULT_ATLAS
    if _DEFAULT_ATLAS is None:
        _DEFAULT_ATLAS = GlyphSDFAtlas(vocab=vocab)
    return _DEFAULT_ATLAS
