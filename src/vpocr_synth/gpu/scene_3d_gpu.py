"""`Scene3DGPU`: Compositional 3D Scene Grammar & 1-Ray PBR Exposure-Integrated Video Engine (`v0.6.0`).

Renders physically & temporally consistent multi-frame camera clips `(K, 3, H, W)` (or single frames `K=1`)
with 100% permissive procedural 3D ray-casting, true 1-ray 3D solar-cone shadow visibility, and unified GGX/IBL PBR:

1. **Typed Compositional 3D Scene Graph Grammar $p(G, \\xi)$ Across Vehicles & World Structures**:
   - Shared geometric productions (extrusions, 3D CSG recessed mounting cavities, $\\mathbb{Z}^2$ linear lattices,
     general $D_N$ radial aperture/stud repetitions for $N \\in \\{3..12\\}$, multi-octave flexural wave sheets,
     and multi-link articulated rigid crane booms / trusses / portals) across all 9 operational vehicle subtypes
     (including cab-behind articulated `cargo_crane_truck` / `unic` trucks) and site architecture.
   - Substrate-decoupled 1..3-row multi-script $\\alpha$-paint & 3D normal-relief typography directly on curved,
     corrugated, and planar substrates without rectangular borders, alongside bolted/framed non-plate placards.

2. **True 3D 1-Ray Shadow Visibility $V(x, \\omega_i)$ & Unified Microfacet GGX + Clearcoat + Horizon IBL BRDF**:
   - Casts true 3D shadow rays from hit points $x + \\epsilon n_g$ along solar-cone directions $\\omega_i$
     (one solar-disc sample per clip, $\\theta_\\odot \\in [0.25^\\circ, 6.0^\\circ]$, shared across quadrature nodes) against
     3D primitives and solid quads onto ground, vehicle bodies, and recessed plates alike.
   - Continuous camera pose $T_c(t)$ and focal length $K(t)$ evolution within each exposure window and across frames,
     plus deterministic compositional holdout splits (`"train" | "val_holdout" | "cal_holdout"`).

3. **First-Principles Physical Observability & Multi-Task Ground Truth at Exact Mid-Exposure `t_k`**:
   - `CLS_PLATE (0)`: Exact 3D `[TL, TR, BR, BL]` corners, integrated blur vector/magnitude, ray-cast visible support,
     min projected edge, and Poisson-Gaussian photon contrast SNR driving smooth Nyquist/MTF usability & slot readability.
   - `CLS_VEHICLE (1)` & `CLS_SIGNBOARD_HARDNEG (2)`: Exact visible/amodal boxes, privacy cabins, Z=0 footprints, and cutouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import os
import random
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset

from vpocr_det.types import (
    CLS_PLATE,
    CLS_SIGNBOARD_HARDNEG,
    CLS_VEHICLE,
    NUM_GEN4_DET_CLASSES,
    SIDE_FRONT,
    SIDE_PURE_SIDE,
    SIDE_REAR,
    SceneObjectGT,
    SynthSceneSample,
)
from vpocr_synth.export.fast_synthesizer import FastPlatePatchRenderer
from vpocr_synth.plate.models import PlateSize, PlateType
from vpocr_synth.plate.stratified_sampler import StratifiedPlateSampler

# 9 Operational Construction-Site Vehicle Subtypes (aligned with heavy-vehicle final product taxonomy)
SUBTYPE_PASSENGER_KEI: int = 0
SUBTYPE_VAN_KEITRUCK: int = 1
SUBTYPE_FLATBED_CARGO: int = 2
SUBTYPE_BOX_TRUCK: int = 3
SUBTYPE_DUMP_TRUCK: int = 4
SUBTYPE_CONCRETE_MIXER: int = 5
SUBTYPE_TANKER_SPECIAL: int = 6
SUBTYPE_TRACTOR_TRAILER: int = 7
SUBTYPE_CRANE_HEAVY: int = 8
NUM_VEHICLE_SUBTYPES: int = 9

VEHICLE_SUBTYPE_NAMES: tuple[str, ...] = (
    "passenger_kei",
    "van_keitruck",
    "flatbed_cargo",
    "box_truck",
    "dump_truck",
    "concrete_mixer",
    "tanker_special",
    "tractor_trailer",
    "crane_heavy",
)

# Primitive shape types
_SHAPE_OBB: int = 0
_SHAPE_SLOPED_CAB: int = 1
_SHAPE_CYLINDER_X: int = 2      # Tire / wheel / radial D_N hub along local X
_SHAPE_TAPER_DRUM_Y: int = 3    # Concrete mixer / tanker barrel / aggregate mound along local Y
_SHAPE_FLEX_SHEET: int = 4      # Multi-octave flexural wave sheet (vehicle cover / curtain / site screen)

# Semantic part roles for PBR shading & privacy/cutout extraction
_ROLE_BODY: int = 0
_ROLE_CABIN_GLASS: int = 1
_ROLE_DARK_CHASSIS: int = 2
_ROLE_TIRE: int = 3
_ROLE_CHROME_METAL: int = 4
_ROLE_HEADLIGHT: int = 5
_ROLE_TAILLIGHT: int = 6
_ROLE_ENV_SOLID: int = 7
_ROLE_CARGO_MATTE: int = 8

# Gen-6 Compositional Holdout Splits (disjoint grammar combos, font subsets, D_N orders, and camera motion families)
VALID_DOMAIN_SPLITS: tuple[str, ...] = ("train", "val_holdout", "cal_holdout")

FONT_SUBSETS_BY_DOMAIN_SPLIT: dict[str, tuple[tuple[str, str], ...]] = {
    "train": (
        ("noto_sans_bold", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        ("noto_sans_regular", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ("noto_serif_bold", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"),
        ("dejavu_sans_bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
    "val_holdout": (
        ("noto_sans_medium", "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
        ("noto_serif_regular", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        ("dejavu_serif_bold", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    ),
    "cal_holdout": (
        ("noto_sans_black", "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"),
        ("noto_serif_semibold", "/usr/share/fonts/opentype/noto/NotoSerifCJK-SemiBold.ttc"),
        ("dejavu_mono_bold", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    ),
}

RADIAL_DN_ORDERS_BY_DOMAIN_SPLIT: dict[str, tuple[int, ...]] = {
    "train": (4, 6, 8, 12),
    "val_holdout": (5, 10, 11),
    "cal_holdout": (3, 7, 9),
}

# Latin-square compositional holdout: individual structural/marking operators appear across splits,
# while 3-way (veh_prod, world_prod, marking_prod) combinations are 100% pairwise disjoint!
GRAMMAR_COMBOS_BY_DOMAIN_SPLIT: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "train": (
        ("tr_cavity_portal_paint", "csg_recess_cavity", "articulated_truss_portal", "frameless_paint_and_bolted_placard"),
        ("tr_sheet_lattice_relief", "flex_wave_sheet", "linear_lattice_wall", "relief_emboss_and_framed_blank"),
        ("tr_boom_hoard_stencil", "articulated_boom_arm", "flex_wave_screen", "conformal_alpha_stencil"),
    ),
    "val_holdout": (
        ("vh_cavity_screen_stencil", "csg_recess_cavity", "flex_wave_screen", "conformal_alpha_stencil"),
        ("vh_sheet_portal_relief", "flex_wave_sheet", "articulated_truss_portal", "relief_emboss_and_framed_blank"),
        ("vh_boom_lattice_placard", "articulated_boom_arm", "linear_lattice_wall", "frameless_paint_and_bolted_placard"),
    ),
    "cal_holdout": (
        ("ch_cavity_lattice_relief", "csg_recess_cavity", "linear_lattice_wall", "relief_emboss_and_framed_blank"),
        ("ch_sheet_screen_placard", "flex_wave_sheet", "flex_wave_screen", "frameless_paint_and_bolted_placard"),
        ("ch_boom_portal_stencil", "articulated_boom_arm", "articulated_truss_portal", "conformal_alpha_stencil"),
    ),
}

CAMERA_FAMILIES_BY_DOMAIN_SPLIT: dict[str, tuple[str, ...]] = {
    "train": ("ptz_pan_linear", "zoom_push_linear", "pole_harmonic_sway", "stationary_rig_fixed"),
    "val_holdout": ("ptz_orbit_curved", "zoom_pull_coupled", "pole_bimodal_flutter"),
    "cal_holdout": ("dolly_lateral_track", "zoom_breathing_osc", "pole_damped_tremor"),
}


@dataclass(slots=True)
class _Prim3D:
    """Single 3D volumetric primitive in world coordinates at instant `t`."""

    center: tuple[float, float, float]
    rot_w2l: np.ndarray  # (3, 3) float32 world-to-local rotation (R^T)
    half_ext: tuple[float, float, float]  # (sx, sy, sz) meters
    shape: int
    veh_idx: int  # 0 = environment/pillar, 1..V = vehicle index
    role: int
    albedo_rgb: tuple[float, float, float]  # linear RGB [0, 1]
    roughness: float = 0.45
    rib_freq: float = 0.0
    mud_level: float = 0.0
    mud_phase: float = 0.0
    slope_param: float = 0.35
    metallic: float = 0.0
    clearcoat: float = 0.0
    specular_f0: float = 0.04
    retro_gain: float = 0.0
    normal_undul_amp: float = 0.0
    radial_n: int = 0  # D_N radial symmetry order in {3..12} (0 = solid)
    wave_amp: float = 0.0
    wave_freq: float = 0.0
    # Substrate-decoupled conformal RGBA alpha-paint / 3D normal-relief stencil:
    # (decal_idx, proj_face, u_center, v_center, u_half, v_half, relief_strength)
    decal_spec: tuple[int, int, float, float, float, float, float] | None = None


@dataclass(slots=True)
class _PlanarQuad3D:
    """3D textured planar quadrilateral (license plate, bolted/framed placard, or frameless alpha stencil)."""

    center: np.ndarray        # (3,) float32 world center
    u_axis: np.ndarray        # (3,) float32 viewer-right unit vector on plane
    v_axis: np.ndarray        # (3,) float32 viewer-down unit vector on plane
    normal: np.ndarray        # (3,) float32 outward unit normal
    half_w: float             # meters
    half_h: float             # meters
    tex_key: tuple[str, int]  # ("plate_front" | "plate_rear" | "sign" | "sticker" | "alpha_decal", idx)
    cls_id: int               # CLS_PLATE or CLS_SIGNBOARD_HARDNEG
    veh_idx: int              # 0 if scene sign, 1..V if mounted on vehicle
    plate_side: int = SIDE_FRONT
    retro_gain: float = 1.0        # retroreflective return under coaxial headlight/IR
    is_green_plate: bool = False
    is_large_plate: bool = False
    plate_text_str: str = ""
    plate_type_str: str = ""
    roughness: float = 0.36
    metallic: float = 0.0
    clearcoat: float = 0.0
    specular_f0: float = 0.0
    normal_undul_amp: float = 0.0
    has_backing_substrate: bool = True


@dataclass
class Scene3DPlateGT:
    """Exact mid-exposure ground truth for a visible license plate or hard-negative signboard.

    Exposes first-principles physical observability quantities alongside orthogonal 4-axis supervision:
    - `visible_support_fraction`: ray-cast canonical visible support fraction in `[0, 1]`.
    - `min_projected_edge_px`: shortest projected quad edge in pixels (Nyquist limit gauge).
    - `blur_magnitude_px`: integrated shutter + camera trajectory optical blur magnitude in pixels.
    - `local_illuminance`: mean linear sensor irradiance on visible plate support.
    - `local_contrast_snr` & `slot_contrast_snr`: physical Poisson-Gaussian signal-to-noise ratio.
    """

    cls_id: int                     # CLS_PLATE (0) or CLS_SIGNBOARD_HARDNEG (2)
    track_id: int                   # persistent ID across the K-frame clip
    veh_track_id: int               # 0 for standalone sign, >=1 for parent vehicle
    bbox_xyxy: np.ndarray           # (4,) float32 clipped visible bbox
    corners_xy: np.ndarray          # (4, 2) float32 exact 3D mid-exposure [TL, TR, BR, BL]
    blur_vec_xy: np.ndarray         # (2,) float32 optical motion vector across exposure [dx, dy] in px
    visibility: float               # [0, 1] unoccluded pixel fraction
    plate_side: int = SIDE_FRONT
    is_large: bool = False
    overhang_shadow: float = 0.0    # measured 3D ray-cast shadow occlusion fraction (1 - direct_sun_vis)
    plate_text_str: str = ""
    plate_type_str: str = ""
    is_plate: float = 1.0
    plate_vis_mask: np.ndarray = field(default_factory=lambda: np.ones((16, 32), dtype=np.float32))
    geometry_usable: float = 1.0
    slot_readability: np.ndarray = field(default_factory=lambda: np.ones((12,), dtype=np.float32))
    visible_support_fraction: float = 1.0
    min_projected_edge_px: float = 16.0
    blur_magnitude_px: float = 0.0
    local_illuminance: float = 0.25
    local_contrast_snr: float = 10.0
    slot_contrast_snr: np.ndarray = field(default_factory=lambda: np.full((12,), 10.0, dtype=np.float32))


@dataclass
class Scene3DVehicleGT:
    """Exact mid-exposure ground truth for a 3D vehicle (including crisp pixel cutout & heavy-vehicle cues).

    `visible_roi_mask` `(64, 64)` fractional optical coverage (`[0, 1]`) is defined over `bbox_xyxy = [x0, y0, x1, y1]`
    (integer pixel edges `[ix0, iy0, ix1, iy1]`) with nominal source pixel-center mapping:
    `x_src = x0 - 0.5 + (j + 0.5) * (x1 - x0) / 64.0`, `y_src = y0 - 0.5 + (i + 0.5) * (y1 - y0) / 64.0`.
    """

    track_id: int
    vehicle_subtype: int            # 0..8 operational construction vehicle class
    archetype_name: str
    bbox_xyxy: np.ndarray           # (4,) float32 tight visible bbox [x0, y0, x1, y1]
    amodal_bbox_xyxy: np.ndarray    # (4,) float32 unrestricted near-plane-clipped 3D-projected bbox
    cabin_bbox_xyxy: np.ndarray     # (4,) float32 cabin/windshield privacy redaction box
    ground_footprint_xy: np.ndarray # (4, 2) float32 [FL, FR, RR, RL] world Z=0 road-plane enclosing quad
    visible_roi_mask: np.ndarray    # (64, 64) float32 optical cutout inside bbox_xyxy
    visibility: float               # [0, 1] optical visible coverage vs unoccluded in-frame vehicle silhouette
    appearance_signature: np.ndarray # (8,) float32 deterministic physical vehicle attributes
    cabin_valid: bool = True
    footprint_valid: bool = True
    articulated_footprint_xy: np.ndarray | None = None  # (M_art, 2) world Z=0 clipped tractor/trailer polygon(s)


@dataclass
class Scene3DFrameGT:
    """All ground-truth annotations and camera metadata for frame `k` of a clip."""

    timestamp_s: float
    exposure_s: float
    is_night_ir: bool
    has_chainlink_fence: bool
    plates_and_signs: list[Scene3DPlateGT] = field(default_factory=list)
    vehicles: list[Scene3DVehicleGT] = field(default_factory=list)
    ignore_bboxes_xyxy: list[tuple[float, float, float, float]] = field(default_factory=list)
    ignore_bbox_classes: list[int] = field(default_factory=list)
    occluded_plate_track_ids: list[int] = field(default_factory=list)
    occluded_vehicle_track_ids: list[int] = field(default_factory=list)
    cam_pos_w: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    cam_R_c2w: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    cam_intrinsics_fxfycxcy: tuple[float, float, float, float] = (500.0, 500.0, 319.5, 191.5)
    ae_gain_applied: float = 1.0


def _poly_signed_area_64(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _convex_hull_ccw_64(pts: np.ndarray) -> np.ndarray:
    """Exact `float64` Andrew monotone-chain 2D convex hull in CCW winding."""
    p = np.asarray(pts, dtype=np.float64)
    if p.shape[0] <= 1:
        return p
    order = np.lexsort((p[:, 1], p[:, 0]))
    ps = p[order]

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for pt in ps:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], pt) <= 1e-11:
            lower.pop()
        lower.append(pt)
    upper: list[np.ndarray] = []
    for pt in ps[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], pt) <= 1e-11:
            upper.pop()
        upper.append(pt)
    if len(lower) <= 1 or len(upper) <= 1:
        return ps[:1]
    return np.stack(lower[:-1] + upper[:-1], axis=0)


_ULP_SHIFTS_3_8: np.ndarray | None = None


def _get_ulp_shifts_3_8() -> np.ndarray:
    global _ULP_SHIFTS_3_8
    if _ULP_SHIFTS_3_8 is None:
        from itertools import product

        _ULP_SHIFTS_3_8 = np.array(list(product((-1, 0, 1), repeat=8)), dtype=np.int8).reshape(-1, 4, 2)
    return _ULP_SHIFTS_3_8


def _conservative_enclosing_quad_2d(
    poly_uv: np.ndarray,
    cam_pos_w: np.ndarray | None = None,
    cam_R_c2w: np.ndarray | None = None,
    intrinsics_fxfycxcy: tuple[float, float, float, float] | None = None,
    z_near: float = 0.22,
    z_far: float | None = None,
) -> np.ndarray:
    """Return a conservative minimum-area `(4, 2)` convex quadrilateral `Q` enclosing 100% of `poly_uv` (`P <= Q`).

    - `M == 4`: exact `(4, 2)` polygon (`0%` area difference).
    - `M == 3`: exact triangle represented with collinear midpoint `[v0, v1, 0.5*(v1+v2), v2]` (`0%` area difference).
    - `M >= 5`: minimum-area convex IEEE-754 `float32` 4-gon enclosing `poly_uv` over finite road-plane camera depth
      `z_near <= Z_c < +inf` below the horizon on `Z = 0` (or `z_near <= Z_c <= z_far` when `z_far` is set),
      verified in exact rational (`Fraction`) arithmetic with exhaustive `3^8 = 6,561` joint `+-1 ULP` steepest-descent polish.
    """
    from fractions import Fraction
    from itertools import combinations

    pts_f32 = np.asarray(poly_uv, dtype=np.float32)
    pts = pts_f32.astype(np.float64)
    m = pts.shape[0]
    if m == 0:
        return np.zeros((4, 2), dtype=np.float32)
    if m == 4:
        return pts_f32.copy()
    if m == 3:
        return np.array([pts[0], pts[1], 0.5 * (pts[1] + pts[2]), pts[2]], dtype=np.float32)

    orig_ccw = _poly_signed_area_64(pts) >= 0.0
    hull = _convex_hull_ccw_64(pts)
    kh = hull.shape[0]
    if kh < 3:
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        return q if orig_ccw else q[::-1].copy()
    if kh == 3:
        q = np.array([hull[0], hull[1], 0.5 * (hull[1] + hull[2]), hull[2]], dtype=np.float32)
        return q if orig_ccw else q[::-1].copy()
    if kh == 4:
        q = hull.astype(np.float32)
        return q if orig_ccw else q[::-1].copy()

    # Exact rational input coordinates for zero-loss float32 containment & strict near-plane feasibility
    px_rat = [Fraction(float(x)) for x in pts_f32[:, 0]]
    py_rat = [Fraction(float(y)) for y in pts_f32[:, 1]]

    has_cam_check = False
    g_u, g_v, g_c, dw2_min, dw2_max = 0.0, 0.0, 0.0, 0.0, 0.0
    s_u, s_v, s_c = 0.0, 0.0, 1.0
    z_near_pos = max(0.05, float(z_near))
    cam_z_f64 = 0.0
    enforce_strict_znear = False
    fx_q = fy_q = cx_q = cy_q = cam_x_q = cam_y_q = cam_z_q = Fraction(0)
    r00_q = r01_q = r02_q = r10_q = r11_q = r12_q = r20_q = r21_q = r22_q = Fraction(0)
    req_t_q = req_zd_q = Fraction(0)
    z_far_q: Fraction | None = None

    if (
        cam_pos_w is not None
        and cam_R_c2w is not None
        and intrinsics_fxfycxcy is not None
        and float(cam_pos_w[2]) > 0.05
    ):
        fx, fy, cx, cy = [float(x) for x in intrinsics_fxfycxcy]
        if fx > 1.0 and fy > 1.0:
            r00, r01, r02 = float(cam_R_c2w[0, 0]), float(cam_R_c2w[0, 1]), float(cam_R_c2w[0, 2])
            r10, r11, r12 = float(cam_R_c2w[1, 0]), float(cam_R_c2w[1, 1]), float(cam_R_c2w[1, 2])
            r20, r21, r22 = float(cam_R_c2w[2, 0]), float(cam_R_c2w[2, 1]), float(cam_R_c2w[2, 2])
            cam_z_f64 = float(cam_pos_w[2])
            g_u = r20 / fx
            g_v = r21 / fy
            g_c = r22 - g_u * cx - g_v * cy
            s_u = (r00 * r02 + r10 * r12 + r20 * r22) / fx
            s_v = (r01 * r02 + r11 * r12 + r21 * r22) / fy
            s_c = (r02 * r02 + r12 * r12 + r22 * r22) - s_u * cx - s_v * cy
            hull_dw2 = g_u * hull[:, 0] + g_v * hull[:, 1] + g_c
            cam_dw2_near = -cam_z_f64 / z_near_pos
            if bool(np.all(hull_dw2 < -2e-5)) and bool(np.all(hull_dw2 >= cam_dw2_near - 1e-3)):
                has_cam_check = True
                # Strict near-plane feasibility: NEVER add outward slack past `cam_dw2_near` when hull is inside `Z_c >= z_near`!
                dw2_min = min(float(hull_dw2.min()), cam_dw2_near)
                if z_far is None:
                    dw2_max = -1e-11
                else:
                    zf_val = max(max(0.25, z_near_pos * 2.0), float(z_far))
                    cam_dw2_far = -cam_z_f64 / zf_val
                    dw2_max = min(-1e-11, max(float(hull_dw2.max()) + 1e-7, cam_dw2_far))
                    z_far_q = Fraction(zf_val)

                fx_q, fy_q, cx_q, cy_q = Fraction(fx), Fraction(fy), Fraction(cx), Fraction(cy)
                cam_x_q = Fraction(float(cam_pos_w[0]))
                cam_y_q = Fraction(float(cam_pos_w[1]))
                cam_z_q = Fraction(cam_z_f64)
                r00_q, r01_q, r02_q = Fraction(r00), Fraction(r01), Fraction(r02)
                r10_q, r11_q, r12_q = Fraction(r10), Fraction(r11), Fraction(r12)
                r20_q, r21_q, r22_q = Fraction(r20), Fraction(r21), Fraction(r22)

                zn_strict = max(Fraction(z_near_pos), Fraction(str(z_near)))
                in_ts: list[Fraction] = []
                in_zds: list[Fraction] = []
                for ux_q, vy_q in zip(px_rat, py_rat):
                    rx = (ux_q - cx_q) / fx_q
                    ry = (vy_q - cy_q) / fy_q
                    dwx = r00_q * rx + r01_q * ry + r02_q
                    dwy = r10_q * rx + r11_q * ry + r12_q
                    dwz = r20_q * rx + r21_q * ry + r22_q
                    tq = -cam_z_q / dwz
                    zdq = tq * (dwx * r02_q + dwy * r12_q) - cam_z_q * r22_q
                    in_ts.append(tq)
                    in_zds.append(zdq)
                strict_t = all(tv >= zn_strict for tv in in_ts)
                strict_zd = all(zv >= zn_strict for zv in in_zds)
                req_t_q = zn_strict if strict_t else min(in_ts)
                req_zd_q = zn_strict if strict_zd else min(in_zds)
                enforce_strict_znear = strict_t and strict_zd
                if strict_t:
                    dw2_min = cam_dw2_near

    hx = hull[:, 0:1]  # (kh, 1)
    hy = hull[:, 1:2]  # (kh, 1)

    def _eval_angles_batch(ths_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized exact float64 evaluation of `(B, 4)` outward normal angles -> `(areas (B,), quads (B, 4, 2))`."""
        base = ths_b[:, 0:1]
        rel = np.mod(ths_b - base, 2.0 * math.pi)
        order = np.argsort(rel, axis=1)
        ths_s = base + np.take_along_axis(rel, order, axis=1)  # (B, 4)
        diffs = np.diff(np.concatenate([ths_s, ths_s[:, 0:1] + 2.0 * math.pi], axis=1), axis=1)
        valid = np.all((diffs > 1e-8) & (diffs < math.pi - 1e-8), axis=1)

        cs = np.cos(ths_s)  # (B, 4)
        sn = np.sin(ths_s)  # (B, 4)
        hs = np.max(hx[None, :, :] * cs[:, None, :] + hy[None, :, :] * sn[:, None, :], axis=1)  # (B, 4)
        c_next = np.roll(cs, -1, axis=1)
        s_next = np.roll(sn, -1, axis=1)
        h_next = np.roll(hs, -1, axis=1)
        det = cs * s_next - sn * c_next
        valid &= np.all(det > 1e-10, axis=1)
        safe_det = np.where(det > 1e-10, det, 1.0)
        vx = (hs * s_next - h_next * sn) / safe_det
        vy = (-hs * c_next + h_next * cs) / safe_det
        valid &= np.all((np.abs(vx) <= 1e7) & (np.abs(vy) <= 1e7), axis=1)
        if has_cam_check:
            dw = g_u * vx + g_v * vy + g_c
            valid &= np.all((dw >= dw2_min) & (dw <= dw2_max), axis=1)
            if enforce_strict_znear:
                zd_num = cam_z_f64 * (s_u * vx + s_v * vy + s_c)
                valid &= np.all(zd_num >= (-dw) * z_near_pos, axis=1)
        areas = 0.5 * np.abs(np.sum(vx * np.roll(vy, -1, axis=1) - np.roll(vx, -1, axis=1) * vy, axis=1))
        areas = np.where(valid, areas, float("inf"))
        return areas, np.stack([vx, vy], axis=-1)

    edge_th: list[float] = []
    for j in range(kh):
        e = hull[(j + 1) % kh] - hull[j]
        en = float(np.linalg.norm(e))
        if en > 1e-11:
            edge_th.append(math.atan2(-float(e[0]) / en, float(e[1]) / en))
    cand_th: list[float] = list(edge_th)
    if has_cam_check and math.hypot(g_u, g_v) > 1e-12:
        cand_th.append(math.atan2(g_v, g_u))
        cand_th.append(math.atan2(-g_v, -g_u))

    best_area = float("inf")
    best_quad: np.ndarray | None = None
    best_ths: np.ndarray | None = None
    top_seeds: list[tuple[float, np.ndarray]] = []

    def _record_seed(area_val: float, th_vec: np.ndarray) -> None:
        top_seeds.append((area_val, th_vec.copy()))

    def _consider_batch(cands_2d: np.ndarray) -> None:
        nonlocal best_area, best_quad, best_ths
        if cands_2d.shape[0] == 0:
            return
        ars, qs = _eval_angles_batch(cands_2d)
        idx = int(np.argmin(ars))
        if math.isfinite(float(ars[idx])):
            _record_seed(float(ars[idx]), cands_2d[idx])
            if float(ars[idx]) < best_area:
                best_area = float(ars[idx])
                best_quad = qs[idx].copy()
                best_ths = cands_2d[idx].copy()

    # 1. Horizon-level-set OBB + rotating-calipers edge OBBs
    obb_seeds = [[t, t + 0.5 * math.pi, t + math.pi, t + 1.5 * math.pi] for t in cand_th]
    if obb_seeds:
        _consider_batch(np.asarray(obb_seeds, dtype=np.float64))

    # 2. All 4-edge combinations of (hull edges + horizon boundary)
    m_c = len(cand_th)
    if 4 <= m_c <= 15:
        c4 = [[cand_th[i] for i in comb] for comb in combinations(range(m_c), 4)]
        _consider_batch(np.asarray(c4, dtype=np.float64))

    # 2b. Vectorized 2-flush + 2-pivoting global grid for multi-pivoting optima (`m_c <= 10`)
    if 2 <= m_c <= 10:
        g2 = np.linspace(0.05, 0.95, 13)
        ga, gb = np.meshgrid(g2, g2, indexing="ij")
        ga_f, gb_f = ga.ravel(), gb.ravel()
        rows_2p2: list[np.ndarray] = []
        for i0, i1 in combinations(range(m_c), 2):
            t0_c, t1_c = cand_th[i0], cand_th[i1]
            d01 = (t1_c - t0_c) % (2.0 * math.pi)
            d10 = 2.0 * math.pi - d01
            if d01 > 1e-5 and d10 > 1e-5:
                rows_2p2.append(
                    np.column_stack(
                        [
                            np.full_like(ga_f, t0_c),
                            np.full_like(ga_f, t1_c),
                            t0_c + ga_f * d01,
                            t1_c + gb_f * d10,
                        ]
                    )
                )
        if rows_2p2:
            _consider_batch(np.concatenate(rows_2p2, axis=0))

    # 3. All 3-edge combinations + 1D supporting-line search with exact feasibility-boundary bisection
    if 3 <= m_c <= 13:
        s_grid = np.linspace(1e-6, 1.0 - 1e-6, 129)
        for comb in combinations(range(m_c), 3):
            t3 = [cand_th[i] for i in comb]
            base = t3[0]
            rels = sorted([(t - base) % (2.0 * math.pi) for t in t3])
            arcs = (
                (base + rels[0], base + rels[1]),
                (base + rels[1], base + rels[2]),
                (base + rels[2], base + 2.0 * math.pi),
            )
            for a0, a1 in arcs:
                span = a1 - a0
                if span <= 1e-7:
                    continue
                th4_vec = a0 + s_grid * span
                batch_3p1 = np.column_stack(
                    [
                        np.full_like(th4_vec, t3[0]),
                        np.full_like(th4_vec, t3[1]),
                        np.full_like(th4_vec, t3[2]),
                        th4_vec,
                    ]
                )
                ars, qs = _eval_angles_batch(batch_3p1)
                fin = np.isfinite(ars)
                if not np.any(fin):
                    continue
                idx_min = int(np.argmin(ars))
                _record_seed(float(ars[idx_min]), batch_3p1[idx_min])
                if float(ars[idx_min]) < best_area:
                    best_area = float(ars[idx_min])
                    best_quad = qs[idx_min].copy()
                    best_ths = batch_3p1[idx_min].copy()

                # Exact 44-step bisection at every feasibility boundary transition
                for idx_s in range(len(ars) - 1):
                    if bool(fin[idx_s]) != bool(fin[idx_s + 1]):
                        t_f = float(th4_vec[idx_s]) if fin[idx_s] else float(th4_vec[idx_s + 1])
                        t_inf = float(th4_vec[idx_s + 1]) if fin[idx_s] else float(th4_vec[idx_s])
                        for _ in range(44):
                            tm = 0.5 * (t_f + t_inf)
                            row_m = np.array([[t3[0], t3[1], t3[2], tm]], dtype=np.float64)
                            am, qm = _eval_angles_batch(row_m)
                            if math.isfinite(float(am[0])):
                                t_f = tm
                                if float(am[0]) < best_area:
                                    best_area, best_quad, best_ths = float(am[0]), qm[0].copy(), row_m[0].copy()
                            else:
                                t_inf = tm
                        _record_seed(best_area, np.array([t3[0], t3[1], t3[2], t_f], dtype=np.float64))

                # Strictly-feasible interior golden-section refinement around `idx_min`
                li = idx_min
                while li > 0 and fin[li - 1] and li > idx_min - 2:
                    li -= 1
                ri = idx_min
                while ri < len(ars) - 1 and fin[ri + 1] and ri < idx_min + 2:
                    ri += 1
                lo = float(th4_vec[li])
                hi = float(th4_vec[ri])
                if hi > lo + 1e-12:
                    for _ in range(40):
                        m1 = lo + 0.38196601125 * (hi - lo)
                        m2 = lo + 0.61803398875 * (hi - lo)
                        pair = np.array([[t3[0], t3[1], t3[2], m1], [t3[0], t3[1], t3[2], m2]], dtype=np.float64)
                        ap, qp = _eval_angles_batch(pair)
                        if float(ap[0]) < best_area:
                            best_area, best_quad, best_ths = float(ap[0]), qp[0].copy(), pair[0].copy()
                        if float(ap[1]) < best_area:
                            best_area, best_quad, best_ths = float(ap[1]), qp[1].copy(), pair[1].copy()
                        if math.isfinite(float(ap[0])) and (not math.isfinite(float(ap[1])) or float(ap[0]) <= float(ap[1])):
                            hi = m2
                        else:
                            lo = m1

    # 4. Multi-start 4-D coordinate + pairwise diagonal pattern-search polish down to `1e-13` rad
    dirs_4d_list: list[list[float]] = []
    for d_i in range(4):
        for sgn in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[d_i] = sgn
            dirs_4d_list.append(v)
    for i1, i2 in combinations(range(4), 2):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                v = [0.0, 0.0, 0.0, 0.0]
                v[i1] = s1
                v[i2] = s2
                dirs_4d_list.append(v)
    dirs_4d = np.asarray(dirs_4d_list, dtype=np.float64)  # (32, 4)

    polish_starts: list[np.ndarray] = []
    if best_ths is not None:
        polish_starts.append(best_ths.copy())
    for _, st_th in sorted(top_seeds, key=lambda x: x[0])[:6]:
        if all(float(np.max(np.abs(np.sin(0.5 * (st_th - ps))))) > 1e-2 for ps in polish_starts):
            polish_starts.append(st_th.copy())
            if len(polish_starts) >= 4:
                break

    for start_th in polish_starts:
        cur = start_th.copy()
        cur_ar_arr, cur_q_arr = _eval_angles_batch(cur[None, :])
        cur_area = float(cur_ar_arr[0])
        if not math.isfinite(cur_area):
            continue
        if cur_area < best_area:
            best_area, best_quad = cur_area, cur_q_arr[0].copy()
        step = 0.04
        while step > 1e-13:
            trials = cur[None, :] + step * dirs_4d
            ars, qs = _eval_angles_batch(trials)
            idx = int(np.argmin(ars))
            if math.isfinite(float(ars[idx])) and float(ars[idx]) < cur_area - 1e-11:
                cur_area = float(ars[idx])
                cur = trials[idx].copy()
                if cur_area < best_area:
                    best_area = cur_area
                    best_quad = qs[idx].copy()
            else:
                step *= 0.5

    if best_quad is None:
        v_dir = np.array([0.0, 1.0], dtype=np.float64)
        u_dir = np.array([1.0, 0.0], dtype=np.float64)
        if len(edge_th) > 0:
            u_dir = np.array([-math.sin(edge_th[0]), math.cos(edge_th[0])], dtype=np.float64)
            v_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float64)
        pu = hull @ u_dir
        pv = hull @ v_dir
        u0, u1 = float(pu.min()), float(pu.max())
        v0, v1 = float(pv.min()), float(pv.max())
        best_quad = np.stack(
            [u0 * u_dir + v0 * v_dir, u1 * u_dir + v0 * v_dir, u1 * u_dir + v1 * v_dir, u0 * u_dir + v1 * v_dir],
            axis=0,
        )

    # 5. Exact IEEE-754 float32 ULP lattice rounding + exhaustive 3^8=6561 joint +-1-ULP rational steepest descent
    q_ccw64 = _convex_hull_ccw_64(best_quad)
    if q_ccw64.shape[0] != 4:
        out_fb = best_quad.astype(np.float32)
        return out_fb if orig_ccw else out_fb[::-1].copy()

    def _f32_to_ord_i64(val_f32: np.float32) -> int:
        bits = int(np.float32(val_f32).view(np.uint32))
        mag = bits & 0x7FFFFFFF
        if mag == 0:
            return 0
        return mag if (bits >> 31) == 0 else -mag

    def _ulp_step_f32(val_f32: np.float32, k_steps: int) -> np.float32:
        idx = max(-0x7F7FFFFF, min(0x7F7FFFFF, _f32_to_ord_i64(val_f32) + int(k_steps)))
        u32 = np.uint32(idx) if idx >= 0 else np.uint32(0x80000000 | (-idx))
        return u32.view(np.float32)

    def _ulp_steps_vec_f32(val_f32: np.float32, offsets_i64: np.ndarray) -> np.ndarray:
        clamped = np.clip(_f32_to_ord_i64(val_f32) + offsets_i64, -0x7F7FFFFF, 0x7F7FFFFF)
        pos_u32 = np.where(clamped >= 0, clamped, 0).astype(np.uint32)
        neg_u32 = np.uint32(0x80000000) | np.where(clamped < 0, -clamped, 0).astype(np.uint32)
        return np.where(clamped >= 0, pos_u32, neg_u32).view(np.float32)

    def _exact_area2_if_feasible_f32(q_cand: np.ndarray) -> Fraction | None:
        if not bool(np.all(np.isfinite(q_cand))):
            return None
        qx = [Fraction(float(x)) for x in q_cand[:, 0]]
        qy = [Fraction(float(y)) for y in q_cand[:, 1]]
        for j_e in range(4):
            jp = (j_e - 1) % 4
            jn = (j_e + 1) % 4
            ex = qx[jn] - qx[j_e]
            ey = qy[jn] - qy[j_e]
            if ex == 0 and ey == 0:
                return None
            if (qx[j_e] - qx[jp]) * ey - (qy[j_e] - qy[jp]) * ex <= 0:
                return None
            qxj = qx[j_e]
            qyj = qy[j_e]
            for pkx, pky in zip(px_rat, py_rat):
                if ex * (pky - qyj) - ey * (pkx - qxj) < 0:
                    return None
        if has_cam_check:
            q64_chk = q_cand.astype(np.float64)
            for j_c in range(4):
                rx = (qx[j_c] - cx_q) / fx_q
                ry = (qy[j_c] - cy_q) / fy_q
                dwz = r20_q * rx + r21_q * ry + r22_q
                if dwz >= 0:
                    return None
                tq = -cam_z_q / dwz
                if tq < req_t_q:
                    return None
                if z_far_q is not None and tq > z_far_q:
                    return None
                dwx = r00_q * rx + r01_q * ry + r02_q
                dwy = r10_q * rx + r11_q * ry + r12_q
                zdq = tq * (dwx * r02_q + dwy * r12_q) - cam_z_q * r22_q
                if zdq < req_zd_q:
                    return None
                if enforce_strict_znear:
                    dw_f = g_u * float(q64_chk[j_c, 0]) + g_v * float(q64_chk[j_c, 1]) + g_c
                    if dw_f >= 0.0 or (-cam_z_f64 / dw_f) < z_near_pos:
                        return None
                    zd_f = (-cam_z_f64 / dw_f) * (s_u * float(q64_chk[j_c, 0]) + s_v * float(q64_chk[j_c, 1]) + s_c)
                    if zd_f < z_near_pos:
                        return None
        area2 = Fraction(0)
        for j_e in range(4):
            jn = (j_e + 1) % 4
            area2 += qx[j_e] * qy[jn] - qx[jn] * qy[j_e]
        return abs(area2) if area2 != 0 else None

    def _valid_area_fast64(q_cand: np.ndarray) -> float:
        q64 = q_cand.astype(np.float64)
        if has_cam_check:
            dw = g_u * q64[:, 0] + g_v * q64[:, 1] + g_c
            if z_far is None:
                if not bool(np.all((dw >= dw2_min - 1e-12) & (dw < 0.0))):
                    return float("inf")
            else:
                if not bool(np.all((dw >= dw2_min - 1e-12) & (dw <= dw2_max + 1e-12) & (dw < 0.0))):
                    return float("inf")
        for j_e in range(4):
            p0 = q64[j_e]
            p1 = q64[(j_e + 1) % 4]
            ej = p1 - p0
            if ej[0] * ej[0] + ej[1] * ej[1] <= 1e-18:
                return float("inf")
            if (p0[0] - q64[(j_e - 1) % 4, 0]) * ej[1] - (p0[1] - q64[(j_e - 1) % 4, 1]) * ej[0] <= 0.0:
                return float("inf")
            if float(np.min(ej[0] * (pts[:, 1] - p0[1]) - ej[1] * (pts[:, 0] - p0[0]))) < -1e-9:
                return float("inf")
        return 0.5 * abs(float(np.sum(q64[:, 0] * np.roll(q64[:, 1], -1) - np.roll(q64[:, 0], -1) * q64[:, 1])))

    offs_fine_128 = np.arange(-128, 129, dtype=np.int64)
    offs_lp_24 = np.arange(-24, 25, dtype=np.int64)

    def _sweep_corner_grid(
        cur_q: np.ndarray,
        cur_a2: Fraction | None,
        j: int,
        offsets_arr: np.ndarray,
        skip_zero: bool = True,
        same_quad_as_a2: bool = True,
    ) -> tuple[Fraction | None, np.ndarray, bool]:
        jp = (j - 1) % 4
        jn = (j + 1) % 4
        jpp = (j - 2) % 4
        bu, bv = cur_q[j, 0], cur_q[j, 1]
        us = _ulp_steps_vec_f32(bu, offsets_arr)
        vs = _ulp_steps_vec_f32(bv, offsets_arr)
        bu64, bv64 = float(bu), float(bv)
        du64 = us.astype(np.float64) - bu64
        dv64 = vs.astype(np.float64) - bv64
        xp, yp = float(cur_q[jp, 0]), float(cur_q[jp, 1])
        xn, yn = float(cur_q[jn, 0]), float(cur_q[jn, 1])
        xpp, ypp = float(cur_q[jpp, 0]), float(cur_q[jpp, 1])

        # Cancellation-free affine half-planes `a_k * dU + b_k * dV + c_rel_k >= 0` centered at `(bu64, bv64)`
        a_list = [-(yp - ypp), (yn - yp), -(ypp - yn)]
        b_list = [(xp - xpp), (xp - xn), (xpp - xn)]
        c_list = [
            (xp - xpp) * (bv64 - yp) - (yp - ypp) * (bu64 - xp) - 1e-8,
            (bu64 - xp) * (yn - bv64) - (bv64 - yp) * (xn - bu64) - 1e-8,
            (xn - bu64) * (ypp - yn) - (yn - bv64) * (xpp - xn) - 1e-8,
        ]
        for pkx_f, pky_f in zip(pts[:, 0], pts[:, 1]):
            pxf, pyf = float(pkx_f), float(pky_f)
            a_list.append(pyf - yp)
            b_list.append(-(pxf - xp))
            c_list.append((bu64 - xp) * (pyf - yp) - (bv64 - yp) * (pxf - xp) + 1e-8)

            a_list.append(yn - pyf)
            b_list.append(pxf - xn)
            c_list.append((xn - bu64) * (pyf - bv64) - (yn - bv64) * (pxf - bu64) + 1e-8)

        if has_cam_check:
            dw0 = g_u * bu64 + g_v * bv64 + g_c
            a_list.extend([-g_u, g_u])
            b_list.extend([-g_v, g_v])
            c_list.extend([-dw0 + 1e-14, (dw0 - dw2_min) + 1e-12])
            if enforce_strict_znear:
                a_z = cam_z_f64 * s_u + z_near_pos * g_u
                b_z = cam_z_f64 * s_v + z_near_pos * g_v
                c_z0 = cam_z_f64 * (s_u * bu64 + s_v * bv64 + s_c) + z_near_pos * dw0
                a_list.append(a_z)
                b_list.append(b_z)
                c_list.append(c_z0 + 1e-12)
            if z_far is not None:
                a_list.append(-g_u)
                b_list.append(-g_v)
                c_list.append((dw2_max - dw0) + 1e-12)

        a_arr = np.asarray(a_list, dtype=np.float64)[:, None, None]
        b_arr = np.asarray(b_list, dtype=np.float64)[:, None, None]
        c_arr = np.asarray(c_list, dtype=np.float64)[:, None, None]
        feas_grid = np.all((a_arr * du64[None, :, None] + b_arr * dv64[None, None, :] + c_arr) >= 0.0, axis=0)

        d2a_local = (yn - yp) * du64[:, None] + (xp - xn) * dv64[None, :]
        cur_2a_f64 = abs(
            float(
                (xp * bv64 - bu64 * yp)
                + (bu64 * yn - xn * bv64)
                + (xn * ypp - xpp * yn)
                + (xpp * yp - xp * ypp)
            )
        )
        max_span = max(1.0, abs(yn - yp) * float(np.max(np.abs(du64))) + abs(xp - xn) * float(np.max(np.abs(dv64))))
        if cur_a2 is not None and not same_quad_as_a2:
            d2a_grid = (cur_2a_f64 - float(cur_a2)) + d2a_local
            tol_s = max(1e-6, max(max_span, cur_2a_f64) * 1e-11)
        else:
            d2a_grid = d2a_local
            tol_s = max(1e-9, max_span * 1e-12)
        if cur_a2 is not None:
            iu_arr, iv_arr = np.nonzero(feas_grid & (d2a_grid <= tol_s))
        else:
            iu_arr, iv_arr = np.nonzero(feas_grid)
        if iu_arr.size == 0:
            return cur_a2, cur_q, False
        d2a_vals = d2a_grid[iu_arr, iv_arr]
        order = np.argsort(d2a_vals)
        best_j_q: np.ndarray | None = None
        best_j_a2 = cur_a2
        best_d2a_bound = tol_s if cur_a2 is not None else float("inf")
        for oi in order:
            if float(d2a_vals[oi]) > best_d2a_bound:
                break
            iu_k = int(iu_arr[oi])
            iv_k = int(iv_arr[oi])
            if skip_zero and int(offsets_arr[iu_k]) == 0 and int(offsets_arr[iv_k]) == 0:
                continue
            qc = cur_q.copy()
            qc[j, 0] = us[iu_k]
            qc[j, 1] = vs[iv_k]
            a2_c = _exact_area2_if_feasible_f32(qc)
            if a2_c is not None and (best_j_a2 is None or a2_c < best_j_a2):
                best_j_a2 = a2_c
                best_j_q = qc
                best_d2a_bound = (float(a2_c - cur_a2) if cur_a2 is not None else (float(a2_c) - cur_2a_f64)) + tol_s
        if best_j_q is not None:
            return best_j_a2, best_j_q, True
        return cur_a2, cur_q, False

    def _lp_jump_single_corner(cur_q: np.ndarray, cur_a2: Fraction | None, j: int) -> tuple[Fraction | None, np.ndarray, bool]:
        """Exact 2-variable LP vertex jump for corner `j` keeping the other 3 corners fixed in `float32`."""
        jp = (j - 1) % 4
        jn = (j + 1) % 4
        jpp = (j - 2) % 4
        xp, yp = float(cur_q[jp, 0]), float(cur_q[jp, 1])
        xn, yn = float(cur_q[jn, 0]), float(cur_q[jn, 1])
        xpp, ypp = float(cur_q[jpp, 0]), float(cur_q[jpp, 1])
        e_opp1 = np.array([xpp - xn, ypp - yn], dtype=np.float64)
        e_opp2 = np.array([xp - xpp, yp - ypp], dtype=np.float64)
        if e_opp1[0] * e_opp1[0] + e_opp1[1] * e_opp1[1] <= 1e-18 or e_opp2[0] * e_opp2[0] + e_opp2[1] * e_opp2[1] <= 1e-18:
            return cur_a2, cur_q, False
        if (xn - xp) * (ypp - yn) == 0.0 and (yn - yp) * (xpp - xn) == 0.0:
            pass
        if (xpp - xn) * (yp - ypp) - (ypp - yn) * (xp - xpp) <= 0.0:
            return cur_a2, cur_q, False
        if float(np.min(e_opp1[0] * (pts[:, 1] - yn) - e_opp1[1] * (pts[:, 0] - xn))) < -1e-7:
            return cur_a2, cur_q, False
        if float(np.min(e_opp2[0] * (pts[:, 1] - ypp) - e_opp2[1] * (pts[:, 0] - xpp))) < -1e-7:
            return cur_a2, cur_q, False
        if has_cam_check and any(_corner_cam_status(cur_q[m, 0], cur_q[m, 1]) != 0 for m in (jp, jn, jpp)):
            return cur_a2, cur_q, False

        x0 = 0.5 * (xp + xn)
        y0 = 0.5 * (yp + yn)
        a_lp = [-(yp - ypp), (yn - yp), -(ypp - yn)]
        b_lp = [(xp - xpp), (xp - xn), (xpp - xn)]
        c_lp = [
            (xp - xpp) * (y0 - yp) - (yp - ypp) * (x0 - xp),
            (x0 - xp) * (yn - y0) - (y0 - yp) * (xn - x0),
            (xn - x0) * (ypp - yn) - (yn - y0) * (xpp - xn),
        ]
        for pkx_f, pky_f in zip(pts[:, 0], pts[:, 1]):
            pxf, pyf = float(pkx_f), float(pky_f)
            a_lp.append(pyf - yp)
            b_lp.append(-(pxf - xp))
            c_lp.append((x0 - xp) * (pyf - yp) - (y0 - yp) * (pxf - xp))
            a_lp.append(yn - pyf)
            b_lp.append(pxf - xn)
            c_lp.append((xn - x0) * (pyf - y0) - (yn - y0) * (pxf - x0))
        if has_cam_check:
            dw0 = g_u * x0 + g_v * y0 + g_c
            a_lp.extend([-g_u, g_u])
            b_lp.extend([-g_v, g_v])
            c_lp.extend([-dw0, dw0 - dw2_min])
            if enforce_strict_znear:
                a_lp.append(cam_z_f64 * s_u + z_near_pos * g_u)
                b_lp.append(cam_z_f64 * s_v + z_near_pos * g_v)
                c_lp.append(cam_z_f64 * (s_u * x0 + s_v * y0 + s_c) + z_near_pos * dw0)
            if z_far is not None:
                a_lp.append(-g_u)
                b_lp.append(-g_v)
                c_lp.append(dw2_max - dw0)

        aa = np.asarray(a_lp, dtype=np.float64)
        bb = np.asarray(b_lp, dtype=np.float64)
        cc = np.asarray(c_lp, dtype=np.float64)
        nrm = np.hypot(aa, bb)
        keep = nrm > 1e-18
        aa, bb, cc = aa[keep] / nrm[keep], bb[keep] / nrm[keep], cc[keep] / nrm[keep]
        k_m = aa.shape[0]
        if k_m < 2:
            return cur_a2, cur_q, False
        ri, si = np.triu_indices(k_m, k=1)
        det = aa[ri] * bb[si] - aa[si] * bb[ri]
        v_det = np.abs(det) > 1e-13
        if not bool(np.any(v_det)):
            return cur_a2, cur_q, False
        ri, si, det = ri[v_det], si[v_det], det[v_det]
        du_v = (bb[ri] * cc[si] - bb[si] * cc[ri]) / det
        dv_v = (cc[ri] * aa[si] - cc[si] * aa[ri]) / det
        finite_m = np.isfinite(du_v) & np.isfinite(dv_v) & (np.hypot(du_v, dv_v) < 1e9)
        if not bool(np.any(finite_m)):
            return cur_a2, cur_q, False
        du_v, dv_v = du_v[finite_m], dv_v[finite_m]
        rad_v = np.maximum(1.0, np.hypot(du_v, dv_v))
        slacks = aa[:, None] * du_v[None, :] + bb[:, None] * dv_v[None, :] + cc[:, None]
        feas_v = np.all(slacks >= -1e-6 * rad_v[None, :], axis=0)
        if not bool(np.any(feas_v)):
            return cur_a2, cur_q, False
        du_f = du_v[feas_v]
        dv_f = dv_v[feas_v]
        obj_f = (yn - yp) * du_f + (xp - xn) * dv_f
        ord_lp = np.argsort(obj_f)
        best_a2_lp = cur_a2
        best_q_lp = cur_q
        imp_lp = False
        seen_uv: set[tuple[int, int]] = set()
        for idx_lp in ord_lp[:3]:
            u_f32 = np.float32(x0 + float(du_f[idx_lp]))
            v_f32 = np.float32(y0 + float(dv_f[idx_lp]))
            if not (math.isfinite(float(u_f32)) and math.isfinite(float(v_f32))):
                continue
            key_uv = (int(u_f32.view(np.uint32)), int(v_f32.view(np.uint32)))
            if key_uv in seen_uv:
                continue
            seen_uv.add(key_uv)
            q_try_lp = best_q_lp.copy()
            q_try_lp[j, 0] = u_f32
            q_try_lp[j, 1] = v_f32
            a2_n, q_n, m_n = _sweep_corner_grid(
                q_try_lp,
                best_a2_lp,
                j,
                offs_lp_24,
                skip_zero=False,
                same_quad_as_a2=False,
            )
            if m_n and a2_n is not None:
                best_a2_lp, best_q_lp, imp_lp = a2_n, q_n, True
                break
        return best_a2_lp, best_q_lp, imp_lp

    def _polish_single_corner_128_exact(cur_q: np.ndarray, cur_a2: Fraction, rad: int = 128) -> tuple[Fraction, np.ndarray, bool]:
        """Exact 2-D LP vertex jump + multiscale coarse-jump + exact `(2*rad+1)^2` ULP coordinate descent."""
        any_imp = False
        for j in range(4):
            a2_lp, q_lp, m_lp = _lp_jump_single_corner(cur_q, cur_a2, j)
            if m_lp and a2_lp is not None:
                cur_a2, cur_q, any_imp = a2_lp, q_lp, True
        for stride_c in (2048, 256, 32):
            offs_c = np.arange(-8, 9, dtype=np.int64) * stride_c
            for j in range(4):
                a2_c, cur_q, m_c = _sweep_corner_grid(cur_q, cur_a2, j, offs_c)
                if m_c and a2_c is not None:
                    cur_a2 = a2_c
                    any_imp = True
        offs_fine = offs_fine_128 if rad == 128 else np.arange(-rad, rad + 1, dtype=np.int64)
        while True:
            q_before = cur_q.copy()
            moved_js: list[int] = []
            for j in range(4):
                a2_f, cur_q, m_f = _sweep_corner_grid(cur_q, cur_a2, j, offs_fine)
                if m_f and a2_f is not None:
                    cur_a2 = a2_f
                    moved_js.append(j)
                    any_imp = True
            if not moved_js:
                break
            if len(moved_js) >= 2:
                dq_p = cur_q.astype(np.float64) - q_before.astype(np.float64)
                j_lead = moved_js[0]
                j_follow = moved_js[1]

                def _try_valley_alpha(alpha_v: float, ref_a2: Fraction) -> tuple[Fraction | None, np.ndarray | None]:
                    cand_v = (q_before.astype(np.float64) + alpha_v * dq_p).astype(np.float32)
                    if not bool(np.all(np.isfinite(cand_v))):
                        return None, None
                    a_dir = _exact_area2_if_feasible_f32(cand_v)
                    if a_dir is not None and a_dir < ref_a2:
                        return a_dir, cand_v
                    # Project trailing corner onto exact 2-D LP valley floor at `cand_v[j_lead]`
                    a_proj, q_proj, ok_pr = _lp_jump_single_corner(cand_v, ref_a2, j_follow)
                    if ok_pr and a_proj is not None and a_proj < ref_a2:
                        return a_proj, q_proj
                    a_proj2, q_proj2, ok_pr2 = _lp_jump_single_corner(cand_v, ref_a2, j_lead)
                    if ok_pr2 and a_proj2 is not None and a_proj2 < ref_a2:
                        return a_proj2, q_proj2
                    return None, None

                last_ok_a = 1.0
                fail_a: float | None = None
                for exp_k in range(1, 22):
                    al = float(1 << exp_k)
                    a_v, q_v = _try_valley_alpha(al, cur_a2)
                    if a_v is not None and q_v is not None:
                        cur_a2, cur_q = a_v, q_v
                        last_ok_a = al
                    else:
                        fail_a = al
                        break
                if fail_a is not None and fail_a > 2.0:
                    lo_a, hi_a = last_ok_a, fail_a
                    for _ in range(12):
                        mid_a = 0.5 * (lo_a + hi_a)
                        a_v, q_v = _try_valley_alpha(mid_a, cur_a2)
                        if a_v is not None and q_v is not None:
                            cur_a2, cur_q = a_v, q_v
                            lo_a = mid_a
                        else:
                            hi_a = mid_a
        return cur_a2, cur_q, any_imp

    def _polish_joint_ulp_rational(q_start: np.ndarray) -> tuple[Fraction | None, np.ndarray]:
        cur_q = q_start.copy()
        cur_a2 = _exact_area2_if_feasible_f32(cur_q)
        if cur_a2 is None:
            return None, cur_q

        shifts_3_8 = _get_ulp_shifts_3_8()  # (6561, 4, 2)
        idx_lut = shifts_3_8 + 1
        q_outer_anchor = cur_q.copy()
        outer_steps_since_anchor = 0

        while True:
            # 1. Long-range exact +-128 ULP single-corner coordinate descent until fixed point
            cur_a2, cur_q, _ = _polish_single_corner_128_exact(cur_q, cur_a2, rad=128)

            # 2. Exhaustive 3^8 = 6561 joint +-1 ULP step with cancellation-free bilinear `delta_2a`
            q_neg = np.nextafter(cur_q, np.float32(-np.inf))
            q_pos = np.nextafter(cur_q, np.float32(np.inf))
            lut = np.stack([q_neg, cur_q, q_pos], axis=0)  # (3, 4, 2)
            cands = np.empty((6561, 4, 2), dtype=np.float32)
            for ci in range(4):
                for di in range(2):
                    cands[:, ci, di] = lut[idx_lut[:, ci, di], ci, di]

            c64 = cands.astype(np.float64)
            ok = np.ones(6561, dtype=bool)
            if has_cam_check:
                dw = g_u * c64[:, :, 0] + g_v * c64[:, :, 1] + g_c
                ok &= np.all((dw >= dw2_min - 1e-11) & (dw < 0.0), axis=1)
                if enforce_strict_znear:
                    zd_num_j = cam_z_f64 * (s_u * c64[:, :, 0] + s_v * c64[:, :, 1] + s_c)
                    ok &= np.all(zd_num_j >= (-dw) * z_near_pos - 1e-11, axis=1)
                if z_far is not None:
                    ok &= np.all(dw <= dw2_max + 1e-11, axis=1)
            c_nx = np.roll(c64, -1, axis=1)
            c_pv = np.roll(c64, 1, axis=1)
            e = c_nx - c64
            e_pv = c64 - c_pv
            ok &= np.all((e[:, :, 0] * e[:, :, 0] + e[:, :, 1] * e[:, :, 1]) > 1e-18, axis=1)
            ok &= np.all((e_pv[:, :, 0] * e[:, :, 1] - e_pv[:, :, 1] * e[:, :, 0]) > 0.0, axis=1)
            cross_p = (
                e[:, :, None, 0] * (pts[None, None, :, 1] - c64[:, :, None, 1])
                - e[:, :, None, 1] * (pts[None, None, :, 0] - c64[:, :, None, 0])
            )
            ok &= np.all(cross_p >= -1e-7, axis=(1, 2))
            ok[3280] = False  # exclude identity shift (0,0,...,0)

            surv = cands[ok]
            surv_sh = shifts_3_8[ok]
            moved = False
            if surv.shape[0] > 0:
                s64 = surv.astype(np.float64)
                q0_64 = cur_q.astype(np.float64)
                q0_nx = np.roll(q0_64, -1, axis=0)
                dq = s64 - q0_64[None, :, :]
                dq_nx = np.roll(dq, -1, axis=1)
                # Exact cancellation-free `2 * (Area(q0 + dq) - Area(q0))` for CCW quads
                d2a_joint = np.sum(
                    q0_64[None, :, 0] * dq_nx[:, :, 1]
                    - q0_64[None, :, 1] * dq_nx[:, :, 0]
                    + dq[:, :, 0] * q0_nx[None, :, 1]
                    - dq[:, :, 1] * q0_nx[None, :, 0]
                    + dq[:, :, 0] * dq_nx[:, :, 1]
                    - dq_nx[:, :, 0] * dq[:, :, 1],
                    axis=1,
                )
                max_dspan = max(1.0, float(np.max(np.abs(d2a_joint))))
                j_tol = max(1e-9, max_dspan * 1e-11)
                tight_mask = d2a_joint <= j_tol
                surv_t = surv[tight_mask]
                sh_t = surv_sh[tight_mask]
                d2a_t = d2a_joint[tight_mask]
                order = np.argsort(d2a_t)
                best_step_q: np.ndarray | None = None
                best_step_sh: np.ndarray | None = None
                best_step_a2 = cur_a2
                best_j_bound = j_tol
                for oi in order:
                    if float(d2a_t[oi]) > best_j_bound:
                        break
                    a2_c = _exact_area2_if_feasible_f32(surv_t[oi])
                    if a2_c is not None and a2_c < best_step_a2:
                        best_step_a2 = a2_c
                        best_step_q = surv_t[oi].copy()
                        best_step_sh = sh_t[oi].copy()
                        best_j_bound = float(a2_c - cur_a2) + j_tol
                if best_step_q is not None and best_step_sh is not None:
                    base_q_ls = cur_q
                    cur_q = best_step_q
                    cur_a2 = best_step_a2
                    moved = True
                    # Exponential doubling line-search along winning 8-D ULP direction (`O(log N)` instead of `O(N)`!)
                    for mul_ls in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
                        q_ls = base_q_ls.copy()
                        for ci in range(4):
                            for di in range(2):
                                sk = int(best_step_sh[ci, di]) * mul_ls
                                if sk != 0:
                                    q_ls[ci, di] = _ulp_step_f32(base_q_ls[ci, di], sk)
                        a2_ls = _exact_area2_if_feasible_f32(q_ls)
                        if a2_ls is not None and a2_ls < cur_a2:
                            cur_q = q_ls
                            cur_a2 = a2_ls
                        else:
                            break

            if not moved:
                # Coupled 2-corner multi-ULP edge slides (+-2..+-24 ULPs) before declaring global fixed point
                for j_e in range(4):
                    jn = (j_e + 1) % 4
                    for k_st in (-24, -12, -8, -5, -3, -2, 2, 3, 5, 8, 12, 24):
                        for ax_i in (0, 1):
                            cand_e = cur_q.copy()
                            cand_e[j_e, ax_i] = _ulp_step_f32(cur_q[j_e, ax_i], k_st)
                            cand_e[jn, ax_i] = _ulp_step_f32(cur_q[jn, ax_i], k_st)
                            if _valid_area_fast64(cand_e) <= 0.5 * float(cur_a2) + max(1e-6, float(cur_a2) * 1e-11):
                                a2_e = _exact_area2_if_feasible_f32(cand_e)
                                if a2_e is not None and a2_e < cur_a2:
                                    cur_q = cand_e
                                    cur_a2 = a2_e
                                    moved = True
            if not moved:
                break

            outer_steps_since_anchor += 1
            if outer_steps_since_anchor >= 2:
                dq_out = cur_q.astype(np.float64) - q_outer_anchor.astype(np.float64)
                active_cs = [ci for ci in range(4) if float(np.hypot(dq_out[ci, 0], dq_out[ci, 1])) > 0.0]
                if active_cs:
                    def _try_outer_valley(alpha_o: float, ref_a2_o: Fraction) -> tuple[Fraction | None, np.ndarray | None]:
                        cand_o = (q_outer_anchor.astype(np.float64) + alpha_o * dq_out).astype(np.float32)
                        if not bool(np.all(np.isfinite(cand_o))):
                            return None, None
                        a_do = _exact_area2_if_feasible_f32(cand_o)
                        if a_do is not None and a_do < ref_a2_o:
                            return a_do, cand_o
                        for c_pr in active_cs:
                            a_po, q_po, ok_po = _lp_jump_single_corner(cand_o, ref_a2_o, c_pr)
                            if ok_po and a_po is not None and a_po < ref_a2_o:
                                return a_po, q_po
                        return None, None

                    last_ok_o = 1.0
                    fail_o: float | None = None
                    for exp_o in range(1, 22):
                        al_o = float(1 << exp_o)
                        a_vo, q_vo = _try_outer_valley(al_o, cur_a2)
                        if a_vo is not None and q_vo is not None:
                            cur_a2, cur_q = a_vo, q_vo
                            last_ok_o = al_o
                        else:
                            fail_o = al_o
                            break
                    if fail_o is not None and fail_o > 2.0:
                        lo_o, hi_o = last_ok_o, fail_o
                        for _ in range(12):
                            mid_o = 0.5 * (lo_o + hi_o)
                            a_vo, q_vo = _try_outer_valley(mid_o, cur_a2)
                            if a_vo is not None and q_vo is not None:
                                cur_a2, cur_q = a_vo, q_vo
                                lo_o = mid_o
                            else:
                                hi_o = mid_o
                q_outer_anchor = cur_q.copy()
                outer_steps_since_anchor = 0
        return cur_a2, cur_q

    def _corner_cam_status(u_f32: np.float32, v_f32: np.float32) -> int:
        """Return `0` if corner satisfies camera depth, `+1` if past horizon (`dw >= 0` or `> z_far`), `-1` if `< z_near`."""
        if not has_cam_check:
            return 0
        ux_q = Fraction(float(u_f32))
        vy_q = Fraction(float(v_f32))
        rx = (ux_q - cx_q) / fx_q
        ry = (vy_q - cy_q) / fy_q
        dwz = r20_q * rx + r21_q * ry + r22_q
        if dwz >= 0:
            return 1
        tq = -cam_z_q / dwz
        if z_far_q is not None and tq > z_far_q:
            return 1
        if tq < req_t_q:
            return -1
        dwx = r00_q * rx + r01_q * ry + r02_q
        dwy = r10_q * rx + r11_q * ry + r12_q
        zdq = tq * (dwx * r02_q + dwy * r12_q) - cam_z_q * r22_q
        if zdq < req_zd_q:
            return -1
        if enforce_strict_znear:
            dw_f = g_u * float(u_f32) + g_v * float(v_f32) + g_c
            if dw_f >= 0.0:
                return 1
            if (-cam_z_f64 / dw_f) < z_near_pos:
                return -1
            if (-cam_z_f64 / dw_f) * (s_u * float(u_f32) + s_v * float(v_f32) + s_c) < z_near_pos:
                return -1
        return 0

    def _repair_seed_f32(q_raw: np.ndarray, snap_shared: bool) -> np.ndarray:
        q_r = q_raw.copy()
        if snap_shared:
            for j in range(4):
                d2 = np.sum((pts - q_r[j].astype(np.float64)) ** 2, axis=1)
                km = int(np.argmin(d2))
                if float(d2[km]) <= 0.25:
                    q_r[j] = pts_f32[km]
        if has_cam_check and (abs(g_u) > 1e-15 or abs(g_v) > 1e-15):
            ax_c = 0 if abs(g_u) >= abs(g_v) else 1
            g_dom = g_u if ax_c == 0 else g_v
            for j in range(4):
                for _ in range(24):
                    st_c = _corner_cam_status(q_r[j, 0], q_r[j, 1])
                    if st_c == 0:
                        break
                    # st_c == +1 (crossed horizon: need `dw` MORE negative -> step `-sign(g_dom)`)
                    # st_c == -1 (behind z_near: need `dw` LESS negative -> step `+sign(g_dom)`)
                    want_pos = (g_dom < 0.0) if st_c == 1 else (g_dom > 0.0)
                    q_r[j, ax_c] = np.nextafter(q_r[j, ax_c], np.float32(np.inf if want_pos else -np.inf))
        return q_r

    def _unlock_joint_feasibility(q_seed: np.ndarray) -> tuple[Fraction | None, np.ndarray]:
        a0_f = _exact_area2_if_feasible_f32(q_seed)
        if a0_f is not None:
            return a0_f, q_seed
        # Stage A0: Exact 2-D LP single-corner feasibility unlock keeping the other 3 corners pinned in `float32`
        best_lp_u_a2: Fraction | None = None
        best_lp_u_q: np.ndarray | None = None
        for j in range(4):
            a_lp_u, q_lp_u, ok_lp_u = _lp_jump_single_corner(q_seed, best_lp_u_a2, j)
            if ok_lp_u and a_lp_u is not None:
                best_lp_u_a2, best_lp_u_q = a_lp_u, q_lp_u
        if best_lp_u_a2 is not None and best_lp_u_q is not None:
            return best_lp_u_a2, best_lp_u_q
        # Stage A: Fast vectorized 1D+1D affine single-corner feasibility unlock (+-16, +-128, +-384 ULPs)
        for rad_u in (16, 128, 384):
            offs_u = np.arange(-rad_u, rad_u + 1, dtype=np.int64)
            best_u_a2: Fraction | None = None
            best_u_q: np.ndarray | None = None
            for j in range(4):
                jp = (j - 1) % 4
                jn = (j + 1) % 4
                jpp = (j - 2) % 4
                bu, bv = q_seed[j, 0], q_seed[j, 1]
                us = _ulp_steps_vec_f32(bu, offs_u)
                vs = _ulp_steps_vec_f32(bv, offs_u)
                u64 = us.astype(np.float64)
                v64 = vs.astype(np.float64)
                xp, yp = float(q_seed[jp, 0]), float(q_seed[jp, 1])
                xn, yn = float(q_seed[jn, 0]), float(q_seed[jn, 1])
                xpp, ypp = float(q_seed[jpp, 0]), float(q_seed[jpp, 1])
                # Verify the 2 non-incident edges are already valid before sweeping corner `j`
                e_opp1 = np.array([xpp - xn, ypp - yn], dtype=np.float64)
                e_opp2 = np.array([xp - xpp, yp - ypp], dtype=np.float64)
                if float(np.min(e_opp1[0] * (pts[:, 1] - yn) - e_opp1[1] * (pts[:, 0] - xn))) < -1e-7:
                    continue
                if float(np.min(e_opp2[0] * (pts[:, 1] - ypp) - e_opp2[1] * (pts[:, 0] - xpp))) < -1e-7:
                    continue
                if has_cam_check and any(_corner_cam_status(q_seed[m, 0], q_seed[m, 1]) != 0 for m in (jp, jn, jpp)):
                    continue
                a_u = [-(yp - ypp), (yn - yp), -(ypp - yn)]
                b_u = [(xp - xpp), (xp - xn), (xpp - xn)]
                c_u = [
                    (yp - ypp) * xp - (xp - xpp) * yp - 1e-7,
                    yp * xn - xp * yn - 1e-7,
                    xn * (ypp - yn) - yn * (xpp - xn) - 1e-7,
                ]
                for pkx_f, pky_f in zip(pts[:, 0], pts[:, 1]):
                    a_u.append(float(pky_f) - yp)
                    b_u.append(-(float(pkx_f) - xp))
                    c_u.append(yp * (float(pkx_f) - xp) - xp * (float(pky_f) - yp) + 1e-7)
                    a_u.append(yn - float(pky_f))
                    b_u.append(float(pkx_f) - xn)
                    c_u.append(xn * float(pky_f) - yn * float(pkx_f) + 1e-7)
                if has_cam_check:
                    a_u.extend([-g_u, g_u])
                    b_u.extend([-g_v, g_v])
                    c_u.extend([-g_c + 1e-13, g_c - dw2_min + 1e-11])
                    if enforce_strict_znear:
                        a_u.append(cam_z_f64 * s_u + z_near_pos * g_u)
                        b_u.append(cam_z_f64 * s_v + z_near_pos * g_v)
                        c_u.append(cam_z_f64 * s_c + z_near_pos * g_c + 1e-11)
                    if z_far is not None:
                        a_u.append(-g_u)
                        b_u.append(-g_v)
                        c_u.append(dw2_max - g_c + 1e-11)
                a_ua = np.asarray(a_u, dtype=np.float64)[:, None, None]
                b_ua = np.asarray(b_u, dtype=np.float64)[:, None, None]
                c_ua = np.asarray(c_u, dtype=np.float64)[:, None, None]
                fg_u = np.all((a_ua * u64[None, :, None] + b_ua * v64[None, None, :] + c_ua) >= 0.0, axis=0)
                iu_u, iv_u = np.nonzero(fg_u)
                if iu_u.size == 0:
                    continue
                d2_u = (yn - yp) * (u64[iu_u] - float(bu)) + (xp - xn) * (v64[iv_u] - float(bv))
                for oi in np.argsort(d2_u):
                    qc_u = q_seed.copy()
                    qc_u[j, 0] = us[int(iu_u[oi])]
                    qc_u[j, 1] = vs[int(iv_u[oi])]
                    ac_j = _exact_area2_if_feasible_f32(qc_u)
                    if ac_j is not None:
                        if best_u_a2 is None or ac_j < best_u_a2:
                            best_u_a2 = ac_j
                            best_u_q = qc_u
                        break
            if best_u_a2 is not None and best_u_q is not None:
                return best_u_a2, best_u_q

        # Stage B: Joint 4-corner 3^8 unlock when >=2 corners simultaneously rounded into infeasibility
        shifts_3_8 = _get_ulp_shifts_3_8()
        idx_lut = shifts_3_8 + 1
        for step_k in (1, 2, 4, 8, 16):
            q_lo = np.empty_like(q_seed)
            q_hi = np.empty_like(q_seed)
            for ci in range(4):
                for di in range(2):
                    q_lo[ci, di] = _ulp_step_f32(q_seed[ci, di], -step_k)
                    q_hi[ci, di] = _ulp_step_f32(q_seed[ci, di], step_k)
            lut = np.stack([q_lo, q_seed, q_hi], axis=0)
            cands = np.empty((6561, 4, 2), dtype=np.float32)
            for ci in range(4):
                for di in range(2):
                    cands[:, ci, di] = lut[idx_lut[:, ci, di], ci, di]
            c64 = cands.astype(np.float64)
            ok = np.ones(6561, dtype=bool)
            if has_cam_check:
                dw = g_u * c64[:, :, 0] + g_v * c64[:, :, 1] + g_c
                ok &= np.all((dw >= dw2_min - 1e-11) & (dw < 0.0), axis=1)
                if enforce_strict_znear:
                    zd_num_j = cam_z_f64 * (s_u * c64[:, :, 0] + s_v * c64[:, :, 1] + s_c)
                    ok &= np.all(zd_num_j >= (-dw) * z_near_pos - 1e-11, axis=1)
                if z_far is not None:
                    ok &= np.all(dw <= dw2_max + 1e-11, axis=1)
            c_nx = np.roll(c64, -1, axis=1)
            c_pv = np.roll(c64, 1, axis=1)
            e = c_nx - c64
            e_pv = c64 - c_pv
            ok &= np.all((e[:, :, 0] * e[:, :, 0] + e[:, :, 1] * e[:, :, 1]) > 1e-18, axis=1)
            ok &= np.all((e_pv[:, :, 0] * e[:, :, 1] - e_pv[:, :, 1] * e[:, :, 0]) > 0.0, axis=1)
            cross_p = (
                e[:, :, None, 0] * (pts[None, None, :, 1] - c64[:, :, None, 1])
                - e[:, :, None, 1] * (pts[None, None, :, 0] - c64[:, :, None, 0])
            )
            ok &= np.all(cross_p >= -1e-8, axis=(1, 2))
            surv = cands[ok]
            if surv.shape[0] > 0:
                s64 = surv.astype(np.float64)
                s_nx = np.roll(s64, -1, axis=1)
                ars = 0.5 * np.abs(np.sum(s64[:, :, 0] * s_nx[:, :, 1] - s_nx[:, :, 0] * s64[:, :, 1], axis=1))
                for oi in np.argsort(ars):
                    ac = _exact_area2_if_feasible_f32(surv[oi])
                    if ac is not None:
                        return ac, surv[oi].copy()
        return None, q_seed

    def _search_ulp_around(q_init: np.ndarray, rad: int = 128, snap_shared: bool = True) -> tuple[Fraction | None, np.ndarray]:
        del rad
        q_w = _repair_seed_f32(q_init, snap_shared=snap_shared)
        ba2_w, q_w = _unlock_joint_feasibility(q_w)
        if ba2_w is None:
            return None, q_w
        return _polish_joint_ulp_rational(q_w)

    best_f32_a2, q_f32 = _search_ulp_around(q_ccw64.astype(np.float32), snap_shared=False)
    a2_s, q_s = _search_ulp_around(q_ccw64.astype(np.float32), snap_shared=True)
    if a2_s is not None and (best_f32_a2 is None or a2_s < best_f32_a2):
        best_f32_a2, q_f32 = a2_s, q_s

    # Exact 3-shared-hull-vertex combinatorial seeds (handles large exponent-ratio wedges such as `P_v5`)
    hull_f32 = hull.astype(np.float32)
    m_hull = hull_f32.shape[0]
    if 4 <= m_hull <= 8:
        for i0, i1, i2 in combinations(range(m_hull), 3):
            v0, v1, v2 = hull_f32[i0], hull_f32[i1], hull_f32[i2]
            for j_free, q_pin in (
                (0, np.stack([v0, v0, v1, v2], axis=0)),
                (1, np.stack([v0, v1, v1, v2], axis=0)),
                (2, np.stack([v0, v1, v2, v2], axis=0)),
            ):
                a2_p, q_p, ok_p = _lp_jump_single_corner(q_pin, best_f32_a2, j_free)
                if ok_p and a2_p is not None and (best_f32_a2 is None or a2_p < best_f32_a2):
                    a2_pol, q_pol = _polish_joint_ulp_rational(q_p)
                    if a2_pol is not None and (best_f32_a2 is None or a2_pol < best_f32_a2):
                        best_f32_a2, q_f32 = a2_pol, q_pol

    if best_f32_a2 is None and best_ths is not None:
        max_coord = max(1.0, float(np.max(np.abs(hull))))
        base_b = float(best_ths[0])
        ths_sorted = sorted([base_b + ((float(t) - base_b) % (2.0 * math.pi)) for t in best_ths])
        cs_b = np.cos(ths_sorted)
        sn_b = np.sin(ths_sorted)
        hs_base = np.max(hx * cs_b[None, :] + hy * sn_b[None, :], axis=0)
        c_nx = np.roll(cs_b, -1)
        s_nx = np.roll(sn_b, -1)
        det_b = cs_b * s_nx - sn_b * c_nx
        for rel_pad in (2e-7, 8e-7, 3e-6, 1.2e-5, 5e-5, 2e-4):
            hs_p = hs_base + max_coord * rel_pad
            h_nx = np.roll(hs_p, -1)
            q_p64 = np.column_stack([(hs_p * s_nx - h_nx * sn_b) / det_b, (-hs_p * c_nx + h_nx * cs_b) / det_b])
            if has_cam_check and bool(np.any(g_u * q_p64[:, 0] + g_v * q_p64[:, 1] + g_c >= -1e-6)):
                old_dw2_max = dw2_max
                dw2_max = min(old_dw2_max, -max(1e-6, rel_pad * 10.0))
                if len(cand_th) >= 3:
                    for comb in combinations(range(len(cand_th)), 3):
                        t3 = [cand_th[i] for i in comb]
                        b0 = t3[0]
                        rls = sorted([(t - b0) % (2.0 * math.pi) for t in t3])
                        sg = np.linspace(1e-5, 1.0 - 1e-5, 65)
                        for a0, a1 in ((b0 + rls[0], b0 + rls[1]), (b0 + rls[1], b0 + rls[2]), (b0 + rls[2], b0 + 2.0 * math.pi)):
                            if a1 - a0 <= 1e-6:
                                continue
                            tv = a0 + sg * (a1 - a0)
                            ars_p, qs_p = _eval_angles_batch(np.column_stack([np.full_like(tv, t3[0]), np.full_like(tv, t3[1]), np.full_like(tv, t3[2]), tv]))
                            if np.any(np.isfinite(ars_p)):
                                im_p = int(np.argmin(ars_p))
                                q_try = _convex_hull_ccw_64(qs_p[im_p])
                                if q_try.shape[0] == 4:
                                    q_try_p = q_try + (q_try - np.mean(hull, axis=0, keepdims=True)) * rel_pad
                                    a2_try, q_res = _search_ulp_around(q_try_p.astype(np.float32), rad=10)
                                    if a2_try is not None and (best_f32_a2 is None or a2_try < best_f32_a2):
                                        best_f32_a2, q_f32 = a2_try, q_res
                dw2_max = old_dw2_max
            else:
                q_try = _convex_hull_ccw_64(q_p64)
                if q_try.shape[0] == 4:
                    a2_try, q_res = _search_ulp_around(q_try.astype(np.float32), rad=10)
                    if a2_try is not None and (best_f32_a2 is None or a2_try < best_f32_a2):
                        best_f32_a2, q_f32 = a2_try, q_res
            if best_f32_a2 is not None:
                break

    return q_f32 if orig_ccw else q_f32[::-1].copy()


_minimum_area_enclosing_quad_2d = _conservative_enclosing_quad_2d


@dataclass
class Scene3DClipSample:
    """Multi-frame rendered clip `(K, 3, H, W)` float32 in `[0, 1]` + exact per-frame GT."""

    frames_chw: Tensor              # (K, 3, H, W) float32 [0, 1]
    frame_gts: list[Scene3DFrameGT]
    scene_mode: str                 # "construction_gate" | "night_gate_ir" | "street_approach" | "hard_negative"
    camera_height_m: float
    camera_pitch_deg: float
    domain_split: str = "train"
    grammar_combo_id: str = "tr_cavity_portal_paint"
    font_family_id: str = "noto_sans_bold"
    camera_motion_family: str = "stationary_rig_fixed"
    radial_dn_order: int = 6
    solar_radius_deg: float = 0.53

    def to_synth_scene_samples(self, include_signboards: bool = True) -> list[SynthSceneSample]:
        """Convert each frame `k` of this clip into a shared `SynthSceneSample` (`vpocr_det.types`)."""
        out: list[SynthSceneSample] = []
        frames_u8 = (self.frames_chw.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for k_idx, fgt in enumerate(self.frame_gts):
            objs: list[SceneObjectGT] = []
            for vgt in fgt.vehicles:
                bx0, by0, bx1, by1 = [float(x) for x in vgt.bbox_xyxy]
                vc4 = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], dtype=np.float32)
                objs.append(
                    SceneObjectGT(
                        cls_id=CLS_VEHICLE,
                        bbox_xyxy=vgt.bbox_xyxy.copy(),
                        corners_xy=vc4,
                        visibility=float(vgt.visibility),
                        vehicle_archetype=vgt.archetype_name,
                        vehicle_subtype=int(vgt.vehicle_subtype),
                        amodal_bbox_xyxy=vgt.amodal_bbox_xyxy.copy(),
                        cabin_bbox_xyxy=vgt.cabin_bbox_xyxy.copy(),
                        cabin_valid=bool(vgt.cabin_valid),
                        ground_footprint_xy=vgt.ground_footprint_xy.copy(),
                        footprint_valid=bool(vgt.footprint_valid),
                        visible_roi_mask=vgt.visible_roi_mask.copy(),
                    )
                )
            for pgt in fgt.plates_and_signs:
                if (not include_signboards) and pgt.cls_id == CLS_SIGNBOARD_HARDNEG:
                    continue
                objs.append(
                    SceneObjectGT(
                        cls_id=int(pgt.cls_id),
                        bbox_xyxy=pgt.bbox_xyxy.copy(),
                        corners_xy=pgt.corners_xy.copy(),
                        plate_side=int(pgt.plate_side),
                        visibility=float(pgt.visibility),
                        plate_text_str=pgt.plate_text_str,
                        plate_type_str=pgt.plate_type_str,
                    )
                )
            if fgt.ignore_bbox_classes and len(fgt.ignore_bbox_classes) == len(fgt.ignore_bboxes_xyxy):
                ig_list = [
                    b
                    for b, c in zip(fgt.ignore_bboxes_xyxy, fgt.ignore_bbox_classes)
                    if include_signboards or int(c) != CLS_SIGNBOARD_HARDNEG
                ]
            else:
                ig_list = list(fgt.ignore_bboxes_xyxy)
            out.append(
                SynthSceneSample(
                    image_rgb=frames_u8[k_idx],
                    objects=objs,
                    ignore_bboxes_xyxy=ig_list,
                    scene_tag=self.scene_mode,
                    camera_height_m=float(self.camera_height_m),
                )
            )
        return out

    def to_detector_targets(
        self,
        num_classes: int = NUM_GEN4_DET_CLASSES,
        device: torch.device | str = "cpu",
    ) -> list[dict[str, Tensor | list[Tensor]]]:
        """Build per-frame training target dicts consumable by both 2-class and 3-class `PlateDetLoss`.

        Footprint contract:
        - `"vehicle_footprint_polygons"`: `list[Tensor]` preserving each vehicle's exact Sutherland-Hodgman
          clipped `Z = 0` convex polygon `(M_i, 2)` (`M_i in {3, 4, 5}`, `0%` area loss).
        - `"vehicle_footprints"`: `(N_v, 4, 2)` conservative 4-vertex convex enclosure (`P_i <= Q_i`,
          `Area(Q_i) >= Area(P_i)`, zero lost footprint area, remaining inside `Z_c >= 0.22`).

        Orthogonal 4-axis plate supervision contract (`"plate_sign_indices"` into `"labels"`/`"boxes"`):
        - `"plate_is_plate"`: `(N_ps,)` `1.0` for `CLS_PLATE`, `0.0` for `CLS_SIGNBOARD_HARDNEG`.
        - `"plate_vis_masks"`: `(N_ps, 16, 32)` canonical rectified optical transmittance map `V(u, v)`.
        - `"plate_geometry_usable"`: `(N_ps,)` canonical 4-corner rectification usability `p_geom_usable`.
        - `"plate_slot_readability"`: `(N_ps, 12)` per-slot optical readability `p_readable,s`.
        """
        dev = torch.device(device)
        targets: list[dict[str, Tensor | list[Tensor]]] = []
        for fgt in self.frame_gts:
            labels_l: list[int] = []
            boxes_l: list[np.ndarray] = []
            corners_l: list[np.ndarray] = []
            veh_idx_l: list[int] = []
            veh_sub_l: list[int] = []
            veh_am_l: list[np.ndarray] = []
            veh_cab_l: list[np.ndarray] = []
            veh_cab_v_l: list[bool] = []
            veh_fp_l: list[np.ndarray] = []
            veh_fp_poly_l: list[Tensor] = []
            veh_fp_v_l: list[bool] = []
            veh_mask_l: list[np.ndarray] = []

            ps_idx_l: list[int] = []
            ps_is_plate_l: list[float] = []
            ps_vis_mask_l: list[np.ndarray] = []
            ps_geom_usable_l: list[float] = []
            ps_slot_read_l: list[np.ndarray] = []

            for vgt in fgt.vehicles:
                if CLS_VEHICLE >= num_classes:
                    continue
                veh_idx_l.append(len(labels_l))
                labels_l.append(CLS_VEHICLE)
                boxes_l.append(vgt.bbox_xyxy)
                bx0, by0, bx1, by1 = [float(x) for x in vgt.bbox_xyxy]
                corners_l.append(np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], dtype=np.float32))
                veh_sub_l.append(int(vgt.vehicle_subtype))
                veh_am_l.append(vgt.amodal_bbox_xyxy)
                veh_cab_l.append(vgt.cabin_bbox_xyxy)
                veh_cab_v_l.append(bool(vgt.cabin_valid))
                fp_v = bool(vgt.footprint_valid) and vgt.ground_footprint_xy.shape[0] >= 3
                veh_fp_v_l.append(fp_v)
                if not fp_v:
                    veh_fp_l.append(np.zeros((4, 2), dtype=np.float32))
                    veh_fp_poly_l.append(torch.zeros((0, 2), device=dev, dtype=torch.float32))
                else:
                    raw_poly = vgt.ground_footprint_xy.astype(np.float32)
                    veh_fp_poly_l.append(torch.from_numpy(raw_poly.copy()).to(device=dev, dtype=torch.float32))
                    veh_fp_l.append(
                        _conservative_enclosing_quad_2d(
                            raw_poly,
                            cam_pos_w=fgt.cam_pos_w,
                            cam_R_c2w=fgt.cam_R_c2w,
                            intrinsics_fxfycxcy=fgt.cam_intrinsics_fxfycxcy,
                            z_near=0.22,
                        )
                    )
                veh_mask_l.append(vgt.visible_roi_mask)

            for pgt in fgt.plates_and_signs:
                if int(pgt.cls_id) >= num_classes:
                    continue
                ps_idx_l.append(len(labels_l))
                labels_l.append(int(pgt.cls_id))
                boxes_l.append(pgt.bbox_xyxy)
                corners_l.append(pgt.corners_xy)
                ps_is_plate_l.append(float(pgt.is_plate) if int(pgt.cls_id) == CLS_PLATE else 0.0)
                ps_vis_mask_l.append(np.asarray(pgt.plate_vis_mask, dtype=np.float32).reshape(16, 32))
                ps_geom_usable_l.append(float(pgt.geometry_usable) if int(pgt.cls_id) == CLS_PLATE else 0.0)
                ps_slot_read_l.append(np.asarray(pgt.slot_readability, dtype=np.float32).reshape(12))

            if fgt.ignore_bbox_classes and len(fgt.ignore_bbox_classes) == len(fgt.ignore_bboxes_xyxy):
                filt_ig = [
                    b
                    for b, c in zip(fgt.ignore_bboxes_xyxy, fgt.ignore_bbox_classes)
                    if int(c) < int(num_classes)
                ]
            else:
                filt_ig = list(fgt.ignore_bboxes_xyxy)
            ig_arr = (
                np.asarray(filt_ig, dtype=np.float32).reshape(-1, 4)
                if filt_ig
                else np.zeros((0, 4), dtype=np.float32)
            )
            targets.append(
                {
                    "labels": torch.tensor(labels_l, device=dev, dtype=torch.long),
                    "boxes": (
                        torch.from_numpy(np.stack(boxes_l, axis=0)).to(device=dev, dtype=torch.float32)
                        if boxes_l
                        else torch.zeros((0, 4), device=dev, dtype=torch.float32)
                    ),
                    "corners": (
                        torch.from_numpy(np.stack(corners_l, axis=0)).to(device=dev, dtype=torch.float32)
                        if corners_l
                        else torch.zeros((0, 4, 2), device=dev, dtype=torch.float32)
                    ),
                    "ignore_boxes": torch.from_numpy(ig_arr).to(device=dev, dtype=torch.float32),
                    "vehicle_indices": torch.tensor(veh_idx_l, device=dev, dtype=torch.long),
                    "vehicle_subtypes": torch.tensor(veh_sub_l, device=dev, dtype=torch.long),
                    "vehicle_amodal_boxes": (
                        torch.from_numpy(np.stack(veh_am_l, axis=0)).to(device=dev, dtype=torch.float32)
                        if veh_am_l
                        else torch.zeros((0, 4), device=dev, dtype=torch.float32)
                    ),
                    "vehicle_cabin_boxes": (
                        torch.from_numpy(np.stack(veh_cab_l, axis=0)).to(device=dev, dtype=torch.float32)
                        if veh_cab_l
                        else torch.zeros((0, 4), device=dev, dtype=torch.float32)
                    ),
                    "vehicle_cabin_valid": torch.tensor(veh_cab_v_l, device=dev, dtype=torch.bool),
                    "vehicle_footprints": (
                        torch.from_numpy(np.stack(veh_fp_l, axis=0)).to(device=dev, dtype=torch.float32)
                        if veh_fp_l
                        else torch.zeros((0, 4, 2), device=dev, dtype=torch.float32)
                    ),
                    "vehicle_footprint_polygons": veh_fp_poly_l,
                    "vehicle_footprint_valid": torch.tensor(veh_fp_v_l, device=dev, dtype=torch.bool),
                    "vehicle_roi_masks": (
                        torch.from_numpy(np.stack(veh_mask_l, axis=0)).to(device=dev, dtype=torch.float32)
                        if veh_mask_l
                        else torch.zeros((0, 64, 64), device=dev, dtype=torch.float32)
                    ),
                    "plate_sign_indices": torch.tensor(ps_idx_l, device=dev, dtype=torch.long),
                    "plate_is_plate": torch.tensor(ps_is_plate_l, device=dev, dtype=torch.float32),
                    "plate_vis_masks": (
                        torch.from_numpy(np.stack(ps_vis_mask_l, axis=0)).to(device=dev, dtype=torch.float32)
                        if ps_vis_mask_l
                        else torch.zeros((0, 16, 32), device=dev, dtype=torch.float32)
                    ),
                    "plate_geometry_usable": torch.tensor(ps_geom_usable_l, device=dev, dtype=torch.float32),
                    "plate_slot_readability": (
                        torch.from_numpy(np.stack(ps_slot_read_l, axis=0)).to(device=dev, dtype=torch.float32)
                        if ps_slot_read_l
                        else torch.zeros((0, 12), device=dev, dtype=torch.float32)
                    ),
                }
            )
        return targets


@dataclass
class Scene3DClipConfig:
    """Configuration for `Scene3DGPUEngine`."""

    image_hw: tuple[int, int] = (384, 640)
    num_frames: int = 4
    bank_size: int = 320
    bank_seed: int = 20264040
    split: str = "train"               # "train" | "val" | "test" -> disjoint registration identity banks!
    domain_split: str = "train"        # "train" | "val_holdout" | "cal_holdout" -> disjoint grammar/font/motion/ISP holdouts!
    p_negative_clip: float = 0.14
    p_chainlink_fence: float = 0.30    # applied identically to positive AND negative clips!
    p_gate_pillar_occlusion: float = 0.24
    p_night_scene: float = 0.30
    p_night_ir: float = 0.16
    p_rain_wet: float = 0.22
    p_stationary_clip: float = 0.16    # parked / waiting-at-gate vehicles (including exact v == 0)
    max_substeps: int = 5              # odd Gauss-Legendre shutter sub-steps M in {1, 3, 5, 7, ...}
    sensor_noise: bool = True          # disable stochastic sensor sampling for radiometric/quadrature diagnostics

    def __post_init__(self) -> None:
        m = int(self.max_substeps)
        if m < 1 or (m > 1 and m % 2 == 0):
            raise ValueError(
                f"Scene3DClipConfig.max_substeps must be 1 or an odd integer >= 3 so mid-exposure t_c is exact, got {m}"
            )
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Scene3DClipConfig.split must be 'train', 'val', or 'test', got {self.split!r}")
        if self.domain_split not in VALID_DOMAIN_SPLITS:
            raise ValueError(
                f"Scene3DClipConfig.domain_split must be one of {VALID_DOMAIN_SPLITS}, got {self.domain_split!r}"
            )


def _road_value_noise(x: Tensor, y: Tensor, phase: float) -> Tensor:
    """Smooth 2D lattice noise; independent corners prevent coherent planar bands."""
    ix, iy = x.floor(), y.floor()
    tx, ty = x - ix, y - iy
    tx, ty = tx.square() * (3.0 - 2.0 * tx), ty.square() * (3.0 - 2.0 * ty)

    def corner(ox: float, oy: float) -> Tensor:
        return (torch.sin((ix + ox) * 127.1 + (iy + oy) * 311.7 + phase * 17.3) * 43758.5453).frac().abs()

    a = torch.lerp(corner(0., 0.), corner(1., 0.), tx)
    b = torch.lerp(corner(0., 1.), corner(1., 1.), tx)
    return torch.lerp(a, b, ty) - 0.5


def _road_surface_fields(
    wx: Tensor, wy: Tensor, distance: Tensor, footprint: Tensor, phase: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """World-anchored asphalt grain, repairs and water coverage with analytic mip damping.

    Rotated, incommensurate noise octaves have no preferred road-plane direction.
    Unresolved octaves converge to their mean instead of changing world frequency
    with distance (which would make the material swim when the camera moves).
    """
    distance_fade = 1.0 / (1.0 + 0.06 * distance.clamp_min(0.0))
    patch = torch.zeros_like(wx)
    gran = torch.zeros_like(wx)
    for frequency, angle, weight in ((.19, .37, .56), (.47, 1.71, .29), (1.13, 2.63, .15)):
        c, s = math.cos(angle), math.sin(angle)
        x, y = c * wx - s * wy, s * wx + c * wy
        patch = patch + weight * distance_fade * torch.exp(-(footprint * frequency).square()) * _road_value_noise(
            x * frequency, y * frequency, phase + angle,
        )
        grain_freq = frequency * 19.0
        gran = gran + 2.0 * weight * distance_fade * torch.exp(-(footprint * grain_freq).square()) * _road_value_noise(
            x * grain_freq, y * grain_freq, phase - angle,
        )
    # Shallow wheel ruts retain more water; prefilter their narrow lateral profile too.
    rut = torch.exp(-((wx.abs() - 1.05).square()) / (.20 + footprint.square()))
    rut = rut * distance_fade / (1.0 + footprint.square() / .20)
    coverage = ((patch + .12 + .07 * rut) * 2.6).clamp(0.0, 1.0)
    puddle = coverage.square() * (3.0 - 2.0 * coverage)
    repairs = ((patch - .035) * 3.0).clamp(0.0, 1.0) * .22
    return gran, repairs, puddle


def _srgb_u8_to_linear_tensor(img_u8: np.ndarray, device: torch.device) -> Tensor:
    """Convert `(H, W, 3)` uint8 sRGB patch into `(1, 3, H, W)` linear radiance float32 tensor."""
    t = torch.from_numpy(np.ascontiguousarray(img_u8)).to(device=device, dtype=torch.float32)
    t = (t.permute(2, 0, 1).unsqueeze(0) / 255.0).clamp(0.0, 1.0)
    return torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)


def _srgb_rgba_u8_to_linear_tensor(img_rgba_u8: np.ndarray, device: torch.device) -> Tensor:
    """Convert `(H, W, 4)` uint8 sRGBA stencil into `(1, 4, H, W)` linear RGB + linear alpha `[0, 1]` tensor."""
    t = torch.from_numpy(np.ascontiguousarray(img_rgba_u8)).to(device=device, dtype=torch.float32)
    t = (t.permute(2, 0, 1).unsqueeze(0) / 255.0).clamp(0.0, 1.0)
    rgb_lin = torch.where(t[:, :3] <= 0.04045, t[:, :3] / 12.92, ((t[:, :3] + 0.055) / 1.055) ** 2.4)
    return torch.cat([rgb_lin, t[:, 3:4]], dim=1)


def _resolve_torch_device(device: torch.device | str | None) -> torch.device:
    """Resolve unindexed `'cuda'` to explicit `torch.device('cuda', current_device)` for multi-GPU safety."""
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda" and dev.index is None:
        idx = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        dev = torch.device("cuda", idx)
    return dev


class _GPU3DTextureBank:
    """Cached multi-mipmap linear-HDR texture bank with paired front/rear registrations & domain-split isolation."""

    _BANK_VERSION: int = 6
    _CACHE: dict[tuple[int, str, int | None, int, int, str, str, int], _GPU3DTextureBank] = {}

    SIGN_SPECS: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], ...] = (
        ("工事車両出入口", (246, 246, 242), (22, 54, 138), (198, 34, 34)),
        ("徐行 10km/h", (248, 214, 38), (22, 22, 24), (194, 30, 30)),
        ("一旦停止 左右確認", (246, 245, 240), (192, 28, 32), (28, 42, 98)),
        ("P 空車 200円", (244, 246, 248), (18, 62, 158), (204, 38, 38)),
        ("月極駐車場 No.2", (246, 246, 242), (24, 34, 92), (34, 52, 128)),
        ("関係者以外立入禁止", (248, 246, 242), (188, 28, 32), (28, 32, 38)),
        ("ゲート前駐停車禁止", (242, 246, 242), (20, 92, 52), (20, 92, 52)),
        ("誘導員配置中", (36, 78, 52), (244, 246, 242), (238, 206, 48)),
        ("最大積載量 8500kg", (238, 198, 34), (20, 22, 24), (20, 22, 24)),
        ("追突注意 ダンプ", (236, 62, 36), (248, 248, 246), (248, 248, 246)),
    )
    # Index 0 MUST remain `("危", ...)` (`[危]` hazard placard); indices 1..6 rear stickers;
    # indices 7..12 front cab fleet stencils & manufacturer grille badges; indices 13..15 bumper reflectors/vents.
    STICKER_SPECS: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int] | None], ...] = (
        ("危", (22, 22, 24), (242, 196, 28), (242, 196, 28)),
        ("最大積載量 9000kg", (238, 198, 34), (20, 22, 24), (20, 22, 24)),
        ("最大積載量 3500kg", (236, 240, 242), (22, 24, 26), (22, 24, 26)),
        ("春日 2503", (228, 232, 236), (26, 32, 42), None),
        ("関東 1616", (42, 86, 146), (244, 246, 248), None),
        ("丸運 3802", (48, 98, 68), (244, 245, 240), None),
        ("制限速度80km", (242, 206, 38), (20, 22, 24), (188, 32, 32)),
        ("C 164", (226, 186, 38), (28, 36, 52), None),
        ("898", (196, 214, 202), (32, 42, 40), None),
        ("No.24", (238, 242, 246), (22, 48, 118), None),
        ("ISUZU", (34, 38, 44), (226, 232, 238), (168, 174, 182)),
        ("FUSO", (38, 42, 46), (224, 228, 234), (162, 168, 176)),
        ("HINO", (218, 224, 230), (28, 32, 38), (138, 144, 152)),
        ("   ", (238, 176, 18), (238, 176, 18), (168, 98, 12)),
        ("   ", (28, 30, 34), (16, 18, 20), (88, 94, 102)),
        ("108", (44, 116, 176), (242, 245, 248), None),
    )

    # Substrate-decoupled 1..3-row multi-script (Kanji/Kana/Latin/Digits) frameless alpha-paint & relief stencils
    ALPHA_STENCIL_SPECS: tuple[tuple[tuple[str, ...], tuple[int, int, int]], ...] = (
        (("最大積載量 9500kg",), (244, 246, 248)),
        (("生コン 東京第4",), (24, 52, 118)),
        (("UNIC 2.93t",), (238, 198, 34)),
        (("安全第一 構内徐行",), (206, 34, 34)),
        (("東都重機建設", "No.408 横浜"), (244, 246, 248)),
        (("関東ロジスティクス", "TEL 045-812"), (22, 38, 86)),
        (("特殊車両通行許可", "A-7290 せ 84"), (242, 212, 42)),
        (("丸栄生コン(株)", "品質管理 M-18"), (26, 74, 52)),
        (("第一資材センター", "産業廃棄物収集運搬車", "許可 013-09421"), (244, 245, 240)),
        (("北関東建機リース", "クレーン作業半径内立入禁止", "機番 K-3105"), (28, 30, 34)),
        (("HINO PROFIA", "7200kg MAX"), (228, 232, 238)),
        (("資材搬入ゲート B", "一旦停止・誘導員確認", "SPEED LIMIT 10"), (196, 32, 32)),
    )

    def __init__(
        self,
        device: torch.device | str,
        bank_size: int = 320,
        seed: int = 20264040,
        split: str = "train",
        domain_split: str = "train",
    ) -> None:
        cv2.setNumThreads(0)
        self.device = _resolve_torch_device(device)
        self.bank_size = max(80, int(bank_size))
        self.seed = int(seed)
        self.split = str(split)
        self.domain_split = str(domain_split)
        if self.domain_split not in VALID_DOMAIN_SPLITS:
            raise ValueError(f"Unsupported domain_split {self.domain_split!r}")
        raw_fonts = FONT_SUBSETS_BY_DOMAIN_SPLIT[self.domain_split]
        self.font_subset: tuple[tuple[str, str], ...] = tuple(
            (fid, fpath if os.path.exists(fpath) else FastPlatePatchRenderer._DEFAULT_BOLD_PATH)
            for fid, fpath in raw_fonts
        )
        self.cjk_fallback_path: str = self.font_subset[0][1]
        # Each registration entry `reg_idx` holds paired front (no seal) & rear (with seal) exact-divisible-by-8 mips!
        self.front_plate_mips: list[tuple[Tensor, Tensor, Tensor]] = []
        self.rear_plate_mips: list[tuple[Tensor, Tensor, Tensor]] = []
        self.plate_meta: list[dict[str, Any]] = []
        self.reg_buckets: dict[str, list[int]] = {
            "heavy_comm": [],
            "heavy_priv": [],
            "med_comm": [],
            "med_priv": [],
            "kei_priv": [],
            "kei_comm": [],
            "special_8_9": [],
        }
        self.sign_mips: list[tuple[Tensor, Tensor]] = []
        self.sticker_mips: list[tuple[Tensor, Tensor]] = []
        self.alpha_decal_mips: list[tuple[Tensor, Tensor]] = []
        self.decal_font_ids: list[str] = []
        self._build_all(self.seed, self.split, self.domain_split)

    @classmethod
    def get(
        cls,
        device: torch.device | str,
        bank_size: int = 320,
        seed: int = 20264040,
        split: str = "train",
        domain_split: str = "train",
    ) -> _GPU3DTextureBank:
        dev = _resolve_torch_device(device)
        eff_size = max(80, int(bank_size))
        key = (os.getpid(), dev.type, dev.index, eff_size, int(seed), str(split), str(domain_split), cls._BANK_VERSION)
        inst = cls._CACHE.get(key)
        if inst is None:
            inst = cls(device=dev, bank_size=eff_size, seed=seed, split=split, domain_split=domain_split)
            cls._CACHE[key] = inst
        return inst

    @classmethod
    def clear_cache(cls) -> None:
        cls._CACHE.clear()

    @staticmethod
    def _glyph_mask_for_char(ch: str, font_path: str, cjk_fallback_path: str) -> np.ndarray:
        use_path = cjk_fallback_path if (ord(ch) > 127 and "DejaVu" in font_path) else font_path
        return FastPlatePatchRenderer._get_glyph_mask(use_path, ch, "neutral")

    @classmethod
    def _render_text_board(
        cls,
        text: str,
        w: int,
        h: int,
        bg_rgb: tuple[int, int, int],
        fg_rgb: tuple[int, int, int],
        border_rgb: tuple[int, int, int] | None,
        font_path: str | None = None,
        cjk_fallback_path: str | None = None,
    ) -> np.ndarray:
        patch = np.empty((h, w, 3), dtype=np.uint8)
        patch[:] = bg_rgb
        if border_rgb is not None:
            cv2.rectangle(patch, (2, 2), (w - 3, h - 3), border_rgb, max(2, h // 22))
        chars = [c for c in text if c != " "]
        if not chars:
            return patch
        fp = font_path or FastPlatePatchRenderer._DEFAULT_BOLD_PATH
        fb = cjk_fallback_path or FastPlatePatchRenderer._DEFAULT_BOLD_PATH
        mx = max(5, int(w * 0.07))
        slot_w = (w - 2 * mx) / float(len(chars))
        th = max(10, int(h * 0.64))
        oy = (h - th) // 2
        fg_f = np.array(fg_rgb, dtype=np.float32)
        for i, ch in enumerate(chars):
            gm = cls._glyph_mask_for_char(ch, fp, fb)
            tw = max(6, int(round(slot_w * 0.84)))
            rs = cv2.resize(gm, (tw, th), interpolation=cv2.INTER_AREA)
            ox = int(round(mx + i * slot_w + (slot_w - tw) * 0.5))
            x0, x1 = max(0, ox), min(w, ox + tw)
            if x1 > x0:
                a = (rs[:, : x1 - x0].astype(np.float32) / 255.0)[:, :, None]
                roi = patch[oy : oy + th, x0:x1].astype(np.float32)
                patch[oy : oy + th, x0:x1] = np.clip(roi * (1.0 - a) + fg_f * a, 0.0, 255.0).astype(np.uint8)
        for bx in (int(w * 0.08), int(w * 0.92)):
            for by in (int(h * 0.16), int(h * 0.84)):
                cv2.circle(patch, (bx, by), max(2, h // 24), (96, 102, 108), -1, lineType=cv2.LINE_AA)
        return patch

    @classmethod
    def _render_frameless_alpha_stencil(
        cls,
        rows: tuple[str, ...],
        w: int,
        h: int,
        fg_rgb: tuple[int, int, int],
        font_path: str,
        cjk_fallback_path: str,
    ) -> np.ndarray:
        """Render 1..3-row multi-script frameless RGBA stencil (`alpha == 0` outside glyph strokes, zero border/bolts)."""
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = int(fg_rgb[0])
        rgba[:, :, 1] = int(fg_rgb[1])
        rgba[:, :, 2] = int(fg_rgb[2])
        n_rows = max(1, len(rows))
        pad_y = max(4, int(h * 0.08))
        avail_h = max(12, h - 2 * pad_y)
        row_h = avail_h // n_rows
        alpha_f = np.zeros((h, w), dtype=np.float32)
        for r_idx, row_txt in enumerate(rows):
            chars = [c for c in row_txt if c != " "]
            if not chars:
                continue
            mx = max(6, int(w * 0.05))
            slot_w = (w - 2 * mx) / float(len(chars))
            th = max(8, int(row_h * (0.78 if n_rows == 1 else 0.82)))
            oy = pad_y + r_idx * row_h + max(0, (row_h - th) // 2)
            for ci, ch in enumerate(chars):
                gm = cls._glyph_mask_for_char(ch, font_path, cjk_fallback_path)
                tw = max(5, int(round(slot_w * 0.84)))
                rs = cv2.resize(gm, (tw, th), interpolation=cv2.INTER_AREA)
                ox = int(round(mx + ci * slot_w + (slot_w - tw) * 0.5))
                x0, x1 = max(0, ox), min(w, ox + tw)
                y0, y1 = max(0, oy), min(h, oy + th)
                if x1 > x0 and y1 > y0:
                    sub_a = rs[: y1 - y0, : x1 - x0].astype(np.float32)
                    alpha_f[y0:y1, x0:x1] = np.maximum(alpha_f[y0:y1, x0:x1], sub_a)
        rgba[:, :, 3] = np.clip(alpha_f, 0.0, 255.0).astype(np.uint8)
        return rgba

    @staticmethod
    def _render_bolted_nonplate_placard(variant: int, w: int = 192, h: int = 96) -> np.ndarray:
        """Render bolted/framed blank or geometric non-plate rectangular brackets (`is_plate == 0`, zero text)."""
        palettes = (
            ((232, 236, 238), (64, 70, 78)),    # Blank white painted mounting bracket with dark rim & 4 bolts
            ((34, 72, 48), (218, 224, 220)),    # Blank dark-green commercial plate holder with chrome rim & 4 bolts
            ((234, 196, 36), (28, 30, 34)),     # Blank yellow inspection frame with black rim & 4 bolts
            ((156, 162, 170), (82, 88, 96)),    # Brushed aluminum access cover with embossed inner rib & 4 bolts
        )
        bg_c, rim_c = palettes[variant % len(palettes)]
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        bevel = 0.92 + 0.08 * np.cos((yy - 0.3) * math.pi) * (1.0 - 0.15 * np.abs(xx - 0.5))
        img = np.clip(np.array(bg_c, dtype=np.float32)[None, None, :] * bevel[:, :, None], 0.0, 255.0).astype(np.uint8)
        cv2.rectangle(img, (2, 2), (w - 3, h - 3), rim_c, 3)
        cv2.rectangle(img, (7, 7), (w - 8, h - 8), rim_c, 1)
        if variant % 2 == 1:
            # Embossed horizontal stiffener beads (no characters!)
            for ry in (int(h * 0.36), int(h * 0.64)):
                cv2.line(img, (18, ry), (w - 19, ry), rim_c, 2, lineType=cv2.LINE_AA)
        for bx in (int(w * 0.14), int(w * 0.86)):
            for by in (int(h * 0.18), int(h * 0.82)):
                cv2.circle(img, (bx, by), max(3, h // 18), (74, 78, 84), -1, lineType=cv2.LINE_AA)
                cv2.circle(img, (bx, by), max(2, h // 28), (168, 174, 180), 1, lineType=cv2.LINE_AA)
        return img

    @classmethod
    def _render_procedural_hardneg_board(
        cls,
        kind: str,
        variant: int,
        w: int = 208,
        h: int = 96,
        font_subset: tuple[tuple[str, str], ...] | None = None,
        cjk_fallback_path: str | None = None,
    ) -> np.ndarray:
        """Synthesize non-plate architectural, bolted-blank, & multi-font fleet hard-negative textures (divisible by 8)."""
        if kind == "bolted_blank":
            return cls._render_bolted_nonplate_placard(variant, w=w, h=h)

        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        if kind == "ribbed_metal":
            palettes = (
                (166.0, 172.0, 180.0),
                (62.0, 74.0, 88.0),
                (44.0, 68.0, 56.0),
                (198.0, 200.0, 196.0),
            )
            base = np.array(palettes[variant % len(palettes)], dtype=np.float32)
            freq = 11.0 + 4.0 * float(variant % 3)
            axis_coord = xx if (variant % 2 == 0) else yy
            rib_wave = np.sin(axis_coord * freq * 2.0 * math.pi)
            trap = np.clip(rib_wave * 1.65, -1.0, 1.0)
            shade = 0.78 + 0.24 * trap
            seam = (np.abs(xx - 0.50) < 0.014) | (np.abs(yy - 0.52) < 0.018)
            shade = np.where(seam, shade * 0.52, shade)
            shade *= 0.90 + 0.10 * (1.0 - yy)
            img = np.clip(base[None, None, :] * shade[:, :, None], 0.0, 255.0).astype(np.uint8)
            for bx in (14, w // 2 - 8, w // 2 + 8, w - 15):
                for by in (12, h - 13):
                    cv2.circle(img, (bx, by), 3, (78, 82, 88), -1, lineType=cv2.LINE_AA)
            cv2.rectangle(img, (1, 1), (w - 2, h - 2), (86, 90, 96), 2)
            return img

        if kind == "grated_rail":
            bg = np.full((h, w, 3), (26, 30, 34), dtype=np.float32)
            steel = np.array([152.0, 158.0, 164.0] if variant % 2 == 0 else [118.0, 128.0, 122.0], dtype=np.float32)
            v_bars = (np.mod(xx * (18.0 + 4.0 * (variant % 2)), 1.0) < 0.24).astype(np.float32)
            h_rods = (np.mod(yy * 7.0, 1.0) < 0.20).astype(np.float32)
            flange = ((yy < 0.16) | (yy > 0.84) | ((yy > 0.45) & (yy < 0.55))).astype(np.float32)
            mask = np.clip(np.maximum(flange, np.maximum(v_bars * 0.88, h_rods * 0.68)), 0.0, 1.0)
            bevel = 0.86 + 0.14 * np.cos(yy * 14.0 * math.pi)
            out = bg * (1.0 - mask[:, :, None]) + (steel[None, None, :] * bevel[:, :, None]) * mask[:, :, None]
            img = np.clip(out, 0.0, 255.0).astype(np.uint8)
            for bx in (18, w // 2, w - 19):
                cv2.circle(img, (bx, 8), 3, (72, 76, 82), -1)
                cv2.circle(img, (bx, h - 9), 3, (72, 76, 82), -1)
            return img

        if kind == "slatted_fence":
            wood_tones = (
                (118.0, 82.0, 48.0),
                (78.0, 56.0, 38.0),
                (58.0, 62.0, 66.0),
            )
            tone = np.array(wood_tones[variant % len(wood_tones)], dtype=np.float32)
            horiz = (variant % 2 == 0)
            n_slats = 6.0 if horiz else 11.0
            c_main = yy if horiz else xx
            c_grain = xx if horiz else yy
            slat_pos = np.mod(c_main * n_slats, 1.0)
            in_slat = (slat_pos > 0.09) & (slat_pos < 0.91)
            slat_id = np.floor(c_main * n_slats)
            plank_var = 0.88 + 0.18 * (0.5 + 0.5 * np.sin(slat_id * 4.7 + float(variant)))
            grain = 1.0 + 0.07 * np.sin(c_grain * 42.0 + np.sin(c_main * 19.0) * 2.5)
            batten = (np.abs(xx - 0.18) < 0.035) | (np.abs(xx - 0.82) < 0.035)
            col = tone[None, None, :] * (plank_var * grain)[:, :, None]
            gap_col = np.array([20.0, 22.0, 24.0], dtype=np.float32)[None, None, :]
            out = np.where((in_slat | batten)[:, :, None], col, gap_col)
            out = np.where(batten[:, :, None], out * 0.82, out)
            return np.clip(out, 0.0, 255.0).astype(np.uint8)

        if kind == "window_grid":
            wall_cols = ((176, 174, 168), (94, 102, 112), (142, 126, 112))
            wall_c = np.array(wall_cols[variant % len(wall_cols)], dtype=np.float32)
            out = np.broadcast_to(wall_c[None, None, :], (h, w, 3)).copy()
            rows = 2
            cols = 4 if (variant % 2 == 0) else 5
            for r_i in range(rows):
                y0 = int(round((0.12 + r_i * 0.42) * h))
                y1 = int(round((0.44 + r_i * 0.42) * h))
                for c_i in range(cols):
                    x0 = int(round((0.06 + c_i * (0.88 / cols)) * w))
                    x1 = int(round((0.06 + (c_i + 0.78) * (0.88 / cols)) * w))
                    lit = ((r_i * 7 + c_i * 5 + variant) % 4 == 0)
                    pane = np.array([156.0, 168.0, 152.0] if lit else [34.0, 46.0, 58.0], dtype=np.float32)
                    out[y0:y1, x0:x1] = pane
                    out[y0:y1:4, x0:x1] = out[y0:y1:4, x0:x1] * 0.78 + 18.0
                    ym = (y0 + y1) // 2
                    out[ym : ym + 2, x0:x1] = wall_c * 0.72
            return np.clip(out, 0.0, 255.0).astype(np.uint8)

        # `door_decal`: domain-split font 2-row logistics/company decal & shop sign
        decal_specs = (
            ("丸和運輸株式会社", "TEL 03-5812", (242, 244, 246), (22, 48, 112), (22, 48, 112)),
            ("東京建機リース", "第3資材置場", (34, 68, 52), (242, 244, 238), (232, 198, 42)),
            ("北関東重機(有)", "最大積載11.5t", (238, 202, 42), (24, 26, 30), (24, 26, 30)),
            ("安全第一 構内徐行", "誘導員詰所 B2", (246, 245, 240), (184, 30, 32), (32, 44, 96)),
        )
        line1, line2, bg_c, fg1_c, fg2_c = decal_specs[variant % len(decal_specs)]
        top_h = h // 2
        bot_h = h - top_h
        if font_subset and len(font_subset) >= 1:
            bold_font = font_subset[variant % len(font_subset)][1]
            reg_font = font_subset[(variant + 1) % len(font_subset)][1]
            fb = cjk_fallback_path or font_subset[0][1]
        else:
            reg_font = getattr(FastPlatePatchRenderer, "_DEFAULT_REG_PATH", FastPlatePatchRenderer._DEFAULT_BOLD_PATH)
            bold_font = FastPlatePatchRenderer._DEFAULT_BOLD_PATH
            fb = bold_font
        patch = np.empty((h, w, 3), dtype=np.uint8)
        patch[:] = bg_c
        for row_idx, (txt_r, y_off, rh, f_path, fg_c) in enumerate(
            ((line1, 3, top_h - 4, bold_font, fg1_c), (line2, top_h + 1, bot_h - 4, reg_font, fg2_c))
        ):
            chars = [c for c in txt_r if c != " "]
            if not chars:
                continue
            mx = max(6, int(w * 0.06))
            slot_w = (w - 2 * mx) / float(len(chars))
            th = max(10, int(rh * (0.72 if row_idx == 0 else 0.66)))
            oy = y_off + max(0, (rh - th) // 2)
            fg_f = np.array(fg_c, dtype=np.float32)
            for ci, ch in enumerate(chars):
                gm = cls._glyph_mask_for_char(ch, f_path, fb)
                tw = max(5, int(round(slot_w * 0.82)))
                rs = cv2.resize(gm, (tw, th), interpolation=cv2.INTER_AREA)
                ox = int(round(mx + ci * slot_w + (slot_w - tw) * 0.5))
                x0, x1 = max(0, ox), min(w, ox + tw)
                if x1 > x0 and oy + th <= h:
                    a = (rs[:, : x1 - x0].astype(np.float32) / 255.0)[:, :, None]
                    roi = patch[oy : oy + th, x0:x1].astype(np.float32)
                    patch[oy : oy + th, x0:x1] = np.clip(roi * (1.0 - a) + fg_f * a, 0.0, 255.0).astype(np.uint8)
        cv2.line(patch, (8, top_h), (w - 9, top_h), fg2_c, 1, lineType=cv2.LINE_AA)
        cv2.rectangle(patch, (2, 2), (w - 3, h - 3), fg2_c, 2)
        return patch

    @staticmethod
    def _apply_green_spectral_variant(patch_u8: np.ndarray, variant: int) -> np.ndarray:
        pf = patch_u8.astype(np.float32)
        g_dom = (pf[..., 1] > pf[..., 0] * 1.12) & (pf[..., 2] < 165.0)
        if variant == 0:
            pf[g_dom, 0] *= 0.65
            pf[g_dom, 1] *= 0.68
            pf[g_dom, 2] *= 0.72
        elif variant == 1:
            pf[g_dom, 0] = pf[g_dom, 0] * 1.15 + 10.0
            pf[g_dom, 1] = pf[g_dom, 1] * 1.06 + 8.0
            pf[g_dom, 2] = pf[g_dom, 2] * 1.28 + 14.0
        else:
            pf[g_dom, 0] = pf[g_dom, 0] * 1.25 + 12.0
            pf[g_dom, 1] = pf[g_dom, 1] * 0.96 + 6.0
        return np.clip(pf, 0.0, 255.0).astype(np.uint8)

    def _build_plate_mips(self, patch_u8: np.ndarray, is_large: bool) -> tuple[Tensor, Tensor, Tensor]:
        # Exact divisible-by-8 base raster matching physical 2:1 aspect ratio -> 0.000 px mip border shift!
        target_wh = (256, 128) if is_large else (192, 96)
        if (patch_u8.shape[1], patch_u8.shape[0]) != target_wh:
            patch_u8 = cv2.resize(patch_u8, target_wh, interpolation=cv2.INTER_AREA)
        t0 = _srgb_u8_to_linear_tensor(patch_u8, self.device)
        t1 = F.avg_pool2d(t0, kernel_size=2, stride=2, ceil_mode=False)
        t2 = F.avg_pool2d(t1, kernel_size=2, stride=2, ceil_mode=False)
        return (t0, t1, t2)

    def _build_all(self, seed: int, split: str, domain_split: str) -> None:
        n_fs = len(self.font_subset)
        # Signs (208x80 + 208x96 procedural + bolted non-plate blank brackets, divisible by 8) & stickers
        for s_i, (txt, bg_c, fg_c, bd_c) in enumerate(self.SIGN_SPECS):
            _, fp_s = self.font_subset[s_i % n_fs]
            u8 = self._render_text_board(txt, 208, 80, bg_c, fg_c, bd_c, font_path=fp_s, cjk_fallback_path=self.cjk_fallback_path)
            t0 = _srgb_u8_to_linear_tensor(u8, self.device)
            t1 = F.avg_pool2d(t0, kernel_size=2, stride=2, ceil_mode=False)
            self.sign_mips.append((t0, t1))

        # Indices 10..25: procedural architectural, multi-font door, & bolted/framed blank non-plate placards
        for hn_kind, n_vars in (
            ("ribbed_metal", 3),
            ("grated_rail", 2),
            ("slatted_fence", 2),
            ("door_decal", 3),
            ("window_grid", 2),
            ("bolted_blank", 4),
        ):
            for v_i in range(n_vars):
                u8_hn = self._render_procedural_hardneg_board(
                    hn_kind, v_i, 208, 96, font_subset=self.font_subset, cjk_fallback_path=self.cjk_fallback_path
                )
                t0_hn = _srgb_u8_to_linear_tensor(u8_hn, self.device)
                t1_hn = F.avg_pool2d(t0_hn, kernel_size=2, stride=2, ceil_mode=False)
                self.sign_mips.append((t0_hn, t1_hn))

        for st_i, (txt, bg_c, fg_c, bd_c) in enumerate(self.STICKER_SPECS):
            w_s = 80 if len(txt) == 1 else 176
            h_s = 80 if len(txt) == 1 else 48
            _, fp_st = self.font_subset[st_i % n_fs]
            u8 = self._render_text_board(txt, w_s, h_s, bg_c, fg_c, bd_c, font_path=fp_st, cjk_fallback_path=self.cjk_fallback_path)
            t0 = _srgb_u8_to_linear_tensor(u8, self.device)
            t1 = F.avg_pool2d(t0, kernel_size=2, stride=2, ceil_mode=False)
            self.sticker_mips.append((t0, t1))

        # Indices 16..25: multi-font company door decals, ribbed panels, & bolted/framed non-plate brackets on vehicles
        for v_i in range(4):
            u8_dd = self._render_procedural_hardneg_board(
                "door_decal", v_i, 176, 80, font_subset=self.font_subset, cjk_fallback_path=self.cjk_fallback_path
            )
            t0_dd = _srgb_u8_to_linear_tensor(u8_dd, self.device)
            t1_dd = F.avg_pool2d(t0_dd, kernel_size=2, stride=2, ceil_mode=False)
            self.sticker_mips.append((t0_dd, t1_dd))
        for v_i in range(2):
            u8_rb = self._render_procedural_hardneg_board("ribbed_metal", v_i, 176, 64)
            t0_rb = _srgb_u8_to_linear_tensor(u8_rb, self.device)
            t1_rb = F.avg_pool2d(t0_rb, kernel_size=2, stride=2, ceil_mode=False)
            self.sticker_mips.append((t0_rb, t1_rb))
        for v_i in range(4):
            u8_bb = self._render_bolted_nonplate_placard(v_i, 192, 96)
            t0_bb = _srgb_u8_to_linear_tensor(u8_bb, self.device)
            t1_bb = F.avg_pool2d(t0_bb, kernel_size=2, stride=2, ceil_mode=False)
            self.sticker_mips.append((t0_bb, t1_bb))

        # Build frameless 1..3-row RGBA alpha-paint & relief stencils exclusively with this domain_split's font subset!
        for dec_i, (rows_t, fg_col) in enumerate(self.ALPHA_STENCIL_SPECS):
            fid_d, fp_d = self.font_subset[dec_i % n_fs]
            h_dec = 64 if len(rows_t) == 1 else (96 if len(rows_t) == 2 else 128)
            rgba_u8 = self._render_frameless_alpha_stencil(rows_t, 224, h_dec, fg_col, fp_d, self.cjk_fallback_path)
            t0_a = _srgb_rgba_u8_to_linear_tensor(rgba_u8, self.device)
            t1_a = F.avg_pool2d(t0_a, kernel_size=2, stride=2, ceil_mode=False)
            self.alpha_decal_mips.append((t0_a, t1_a))
            self.decal_font_ids.append(fid_d)

        split_id = {"train": 0, "val": 1, "test": 2}[split]
        domain_id = {"train": 0, "val_holdout": 1, "cal_holdout": 2}[domain_split]
        part_id = split_id * 3 + domain_id  # 0..8 -> 9 pairwise-disjoint (split, domain_split) partitions!
        split_seed = seed + part_id * 1_000_003
        sampler = StratifiedPlateSampler(seed=split_seed)
        rng = random.Random(split_seed + 9191)
        strata: list[tuple[str, PlateType, PlateSize, bool]] = (
            [("heavy_comm", PlateType.STANDARD_COMMERCIAL, PlateSize.LARGE, False)] * 6
            + [("heavy_priv", PlateType.STANDARD_PRIVATE, PlateSize.LARGE, False)] * 3
            + [("special_8_9", PlateType.STANDARD_COMMERCIAL, PlateSize.LARGE, False)] * 2
            + [("med_comm", PlateType.STANDARD_COMMERCIAL, PlateSize.MEDIUM, False)] * 3
            + [("med_priv", PlateType.STANDARD_PRIVATE, PlateSize.MEDIUM, False)] * 5
            + [("kei_priv", PlateType.KEI_PRIVATE, PlateSize.MEDIUM, False)] * 3
            + [("kei_comm", PlateType.KEI_COMMERCIAL, PlateSize.MEDIUM, False)] * 2
            + [("med_priv", PlateType.GRAPHIC, PlateSize.MEDIUM, False)] * 1
        )
        type_to_rem = {pt_enum: i for i, pt_enum in enumerate(sampler._plate_types)}
        n_pt = len(sampler._plate_types)
        occ_by_type = {pt_enum: 0 for pt_enum in sampler._plate_types}

        for idx in range(self.bank_size):
            bucket_name, tgt_type, p_size, is_temp = strata[idx % len(strata)]
            if idx % 64 == 63:
                is_temp = True

            occ_k = occ_by_type[tgt_type]
            occ_by_type[tgt_type] = occ_k + 1
            # Strictly disjoint registration seed residue + unique 4-digit serial residue mod 9 per (split, domain_split)!
            s_seed = ((occ_k * 37 + idx * 7) * 9 + part_id) * n_pt + type_to_rem[tgt_type]
            pt, cfg = sampler.sample(seed=s_seed)
            base_q = ((int(pt.serial) * 31 + idx * 17) % 1110)
            serial_val = base_q * 9 + (part_id if part_id > 0 else 9)  # in 1..9999 with `serial_val % 9 == part_id`!
            if serial_val < 1000 and rng.random() < 0.72:
                q4 = 112 + (base_q % 998)
                serial_val = min(9999, q4 * 9 + part_id)
                while serial_val > 9999 or (serial_val % 9) != part_id:
                    serial_val -= 9
            pt = replace(pt, serial=int(serial_val))

            if bucket_name == "special_8_9":
                pref = rng.choice(["8", "9", "0"])
                rest = pt.classification[1:] if len(pt.classification) >= 2 else "00"
                pt = replace(pt, classification=pref + rest)
            elif p_size == PlateSize.LARGE and pt.classification and pt.classification[0] not in ("1", "2", "8", "9", "0"):
                pref = rng.choices(["1", "2", "8"], weights=[0.62, 0.18, 0.20], k=1)[0]
                pt = replace(pt, classification=pref + pt.classification[1:])

            can_seal = tgt_type in (PlateType.STANDARD_PRIVATE, PlateType.STANDARD_COMMERCIAL, PlateType.GRAPHIC)
            has_frame = (not is_temp) and (rng.random() < 0.24)
            frame_style = ("chrome" if rng.random() < 0.5 else "black") if has_frame else None
            weath = rng.uniform(0.06, 0.48) if rng.random() < 0.45 else 0.0
            mud_l = rng.uniform(0.05, 0.34) if rng.random() < 0.28 else 0.0
            is_green = tgt_type == PlateType.STANDARD_COMMERCIAL and not is_temp
            do_green_var = is_green and (rng.random() < 0.65)
            green_var = idx % 3

            rcfg_front = replace(
                cfg,
                plate_type=tgt_type,
                plate_size=p_size,
                with_seal=False,
                plate_frame=has_frame,
                weathering=weath,
                mud_level=mud_l,
                seed=split_seed + idx * 2,
            )
            rcfg_rear = replace(
                rcfg_front,
                with_seal=(can_seal and not is_temp),
                seed=split_seed + idx * 2 + 1,
            )
            front_u8, _ = FastPlatePatchRenderer.render_patch(
                pt, rcfg_front, scale=0.58, temporary_plate=is_temp, frame_style=frame_style
            )
            rear_u8, _ = FastPlatePatchRenderer.render_patch(
                pt, rcfg_rear, scale=0.58, temporary_plate=is_temp, frame_style=frame_style
            )
            if do_green_var:
                front_u8 = self._apply_green_spectral_variant(front_u8, green_var)
                rear_u8 = self._apply_green_spectral_variant(rear_u8, green_var)

            is_large_b = p_size == PlateSize.LARGE
            self.front_plate_mips.append(self._build_plate_mips(front_u8, is_large_b))
            self.rear_plate_mips.append(self._build_plate_mips(rear_u8, is_large_b))
            self.plate_meta.append(
                {
                    "plate_text_str": pt.full_text,
                    "plate_type_str": tgt_type.value,
                    "is_large": is_large_b,
                    "is_green": is_green,
                    "retro_gain": 0.42 if (is_green or tgt_type == PlateType.KEI_COMMERCIAL) else 1.45,
                }
            )
            self.reg_buckets[bucket_name].append(idx)


def _rot_z(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rot_zyx(yaw_rad: float, pitch_rad: float, roll_rad: float = 0.0) -> np.ndarray:
    """World rotation `R` for local vehicle frame (`+X` right, `+Y` forward, `+Z` up)."""
    cz, sz = math.cos(yaw_rad), math.sin(yaw_rad)
    cx, sx = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(roll_rad), math.sin(roll_rad)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    return rz @ rx @ ry


_GL8_NODES, _GL8_WEIGHTS = np.polynomial.legendre.leggauss(8)


@dataclass
class _VehiclePassSpec:
    """Persistent physical specification & kinematic heading-integrated trajectory across a clip."""

    track_id: int
    archetype: str
    subtype: int
    length_m: float
    width_m: float
    cab_h_m: float
    body_h_m: float
    paint_lin: tuple[float, float, float]
    secondary_lin: tuple[float, float, float]
    mud_level: float
    mud_phase: float
    front_plate_idx: int
    rear_plate_idx: int
    sticker_idx: int
    has_sticker: bool
    has_hazard_sign: bool
    plate_offset_x_front: float   # lateral plate mount offset (m)
    plate_offset_x_rear: float
    # Kinematic trajectory parameters (integrated along instantaneous heading -> zero sideways sliding!)
    p0: np.ndarray                # (3,) start position at t=0
    speed0_mps: float             # signed longitudinal speed (m/s; <0 = reversing!)
    accel_mps2: float             # signed longitudinal acceleration (m/s^2)
    motion_mode: str              # "stationary" | "cruise" | "brake_to_stop" | "stop_and_go" | "reverse"
    yaw0_rad: float
    yaw_rate_rad: float
    trailer_yaw_delta_rad: float  # articulated angle for tractor-trailers
    bump_amp_m: float             # suspension heave/pitch over gate speed bump
    bump_freq_hz: float
    bump_phase: float
    # Optional Gen-4 front cab/grille/bumper hard-negative distractors & dump cargo load (defaulted for back-compat)
    front_cab_hn_idx: int = 7
    has_front_cab_hn: bool = False
    front_bumper_hn_idx: int = 13
    has_front_bumper_hn: bool = False
    has_dump_load: bool = False
    dump_load_lin: tuple[float, float, float] = (0.86, 0.88, 0.92)
    # Gen-5 (`v0.5.0`) bumper material diversity, heavy-vehicle subtype realism & door decals
    bumper_style_front: int = 2   # 0 = matte black (38%), 1 = painted (32%), 2 = chrome (30%)
    bumper_style_rear: int = 0    # 0 = matte black (38%), 1 = painted (32%), 2 = chrome (30%)
    is_large_truck: bool = True
    has_reefer_nose: bool = False
    is_smooth_frp_box: bool = False
    has_wind_deflector: bool = False
    has_side_door_decal: bool = False
    side_door_decal_idx: int = 16
    # Gen-6 (`v0.6.0`) compositional 3D grammar, continuous PBR materials & substrate-decoupled markings
    paint_metallic: float = 0.08
    paint_clearcoat: float = 0.62
    radial_dn_n: int = 6
    veh_struct_prod: str = "csg_recess_cavity"
    marking_prod: str = "conformal_alpha_stencil"
    decal_idx: int = 0
    decal_relief: float = 0.0
    has_flex_sheet: bool = False
    flex_sheet_lin: tuple[float, float, float] = (0.28, 0.34, 0.26)
    boom_yaw_rad: float = 0.0
    boom_elev_rad: float = 0.14

    def signed_speed_at(self, t: float) -> float:
        """Signed longitudinal speed `s(t)` along vehicle forward axis `+Y` (continuous piecewise-linear)."""
        if self.motion_mode == "stationary":
            return 0.0
        if self.motion_mode == "stop_and_go":
            if t < 0.18:
                return max(0.0, float(self.speed0_mps) * (1.0 - float(t) / 0.18))
            if t < 0.30:
                return 0.0
            s_cap = max(0.65, float(self.speed0_mps) * 1.15 + 1.5)
            return min(s_cap, 3.6 * (float(t) - 0.30))
        if self.motion_mode == "reverse":
            return min(0.0, float(self.speed0_mps) + float(self.accel_mps2) * float(t))
        s_raw = float(self.speed0_mps) + float(self.accel_mps2) * float(t)
        if self.motion_mode == "brake_to_stop" or (self.speed0_mps > 0.0 and s_raw < 0.0):
            return max(0.0, s_raw)
        if self.speed0_mps < 0.0 and s_raw > 0.0:
            return min(0.0, s_raw)
        return float(s_raw)

    def angular_rate_at(self, t: float) -> float:
        """Instantaneous yaw rate `omega(t)` (continuous piecewise-linear, zero jump at `s=0`)."""
        if self.motion_mode == "stationary" or abs(self.yaw_rate_rad) < 1e-12:
            return 0.0
        spd_ref = max(0.8, abs(float(self.speed0_mps)))
        s_t = abs(self.signed_speed_at(t))
        return float(self.yaw_rate_rad) * min(1.2, s_t / spd_ref)

    def _motion_breakpoints_to(self, t: float) -> list[float]:
        """Exact ordered breakpoints from `0.0` to `t` so `s(u)` and `omega(u)` are strictly linear per piece."""
        tf = float(t)
        if abs(tf) < 1e-15:
            return [0.0]
        lo, hi = (0.0, tf) if tf >= 0.0 else (tf, 0.0)
        spd_ref = max(0.8, abs(float(self.speed0_mps)))
        s_lim = 1.2 * spd_ref

        cands: list[float] = []
        if self.motion_mode == "stop_and_go":
            s_cap = max(0.65, float(self.speed0_mps) * 1.15 + 1.5)
            cands.extend([0.18, 0.30, 0.30 + s_cap / 3.6])
            if abs(self.speed0_mps) > 1e-12:
                cands.append(0.18 * (1.0 - s_lim / float(self.speed0_mps)))
            cands.append(0.30 + s_lim / 3.6)
        elif abs(self.accel_mps2) > 1e-12:
            for s_tgt in (-s_lim, 0.0, s_lim):
                cands.append((s_tgt - float(self.speed0_mps)) / float(self.accel_mps2))

        inner = sorted({b for b in cands if (lo + 1e-12) < b < (hi - 1e-12)}, reverse=(tf < 0.0))
        return [0.0, *inner, tf]

    def pose_at(self, t: float) -> tuple[np.ndarray, float, float, float]:
        """Return `(pos_xyz, yaw_rad, pitch_rad, roll_rad)` at continuous time `t` seconds.

        Integrates exact analytical heading `theta(u) = theta_0 + int_0^u omega(tau) dtau` over exact
        piecewise-linear intervals `[t_i, t_{i+1}]`:
        - `0.00000 m/s` lateral velocity (strictly non-holonomic everywhere),
        - `0.00000 m` drift on stopped intervals (`s == 0`),
        - `C^0/C^1` continuous at `t = 0` (zero suspension jump between `t = 0` and `t = 1e-8`).
        """
        if self.motion_mode == "stationary":
            pos = np.array(
                [float(self.p0[0]), float(self.p0[1]), max(0.0, float(self.p0[2]))],
                dtype=np.float64,
            )
            return pos, float(self.yaw0_rad), 0.0, 0.0

        tf = float(t)
        pts = self._motion_breakpoints_to(tf)
        dx_tot = 0.0
        dy_tot = 0.0
        yaw_cur = float(self.yaw0_rad)

        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            h_seg = b - a
            if abs(h_seg) < 1e-15:
                continue
            s_a = self.signed_speed_at(a)
            s_b = self.signed_speed_at(b)
            w_a = self.angular_rate_at(a)
            w_b = self.angular_rate_at(b)
            if abs(s_a) == 0.0 and abs(s_b) == 0.0:
                # Identically stopped across `[a, b]` -> zero translation and zero yaw change!
                continue
            ds = s_b - s_a
            dw = w_b - w_a
            half_h = 0.5 * h_seg
            seg_dx = 0.0
            seg_dy = 0.0
            for xi_g, wi_g in zip(_GL8_NODES, _GL8_WEIGHTS):
                frac = 0.5 * (1.0 + float(xi_g))  # in (0, 1)
                s_u = s_a + ds * frac
                # Exact analytical integral of linear `omega(u) = w_a + dw * frac` on `[a, a + frac * h_seg]`!
                yaw_u = yaw_cur + h_seg * (w_a * frac + 0.5 * dw * frac * frac)
                seg_dx += float(wi_g) * (-s_u * math.sin(yaw_u))
                seg_dy += float(wi_g) * (s_u * math.cos(yaw_u))
            dx_tot += half_h * seg_dx
            dy_tot += half_h * seg_dy
            yaw_cur += 0.5 * h_seg * (w_a + w_b)

        yaw = float(yaw_cur)

        s_now = abs(self.signed_speed_at(tf))
        if self.bump_amp_m > 0.0 and s_now > 0.0:
            u_g = min(1.0, max(0.0, s_now / 0.6))
            # Cubic Hermite smoothstep (`dg/ds == 0` at both `s == 0` and `s == 0.6` -> C^1 across all breakpoints!)
            move_gate = u_g * u_g * (3.0 - 2.0 * u_g)
        else:
            move_gate = 0.0
        phi = 2.0 * math.pi * self.bump_freq_hz * tf + self.bump_phase
        s_phi = math.sin(phi)
        # Non-negative C^inf heave `0.5 * move_gate * A * sin^2(phi)` (eliminates `abs(sin)` velocity cusps everywhere!)
        heave_z = 0.5 * move_gate * self.bump_amp_m * (s_phi * s_phi)
        pitch = move_gate * (self.bump_amp_m * 0.55) * math.cos(phi)
        roll = move_gate * (self.bump_amp_m * 0.30) * math.sin(2.0 * math.pi * self.bump_freq_hz * 0.7 * tf + 1.1)
        pos = np.array(
            [
                float(self.p0[0]) + dx_tot,
                float(self.p0[1]) + dy_tot,
                max(0.0, float(self.p0[2])) + heave_z,
            ],
            dtype=np.float64,
        )
        return pos, yaw, pitch, roll


_ARTICULATED_ARCHETYPES: tuple[str, ...] = (
    "semi_container_trailer",
    "skeletal_chassis_trailer",
    "heavy_lowboy_trailer",
)

ARCHETYPE_CATALOG: tuple[tuple[str, int, float], ...] = (
    ("dump_10t", SUBTYPE_DUMP_TRUCK, 0.14),
    ("deep_dump", SUBTYPE_DUMP_TRUCK, 0.07),
    ("concrete_mixer", SUBTYPE_CONCRETE_MIXER, 0.10),
    ("flatbed_truck", SUBTYPE_FLATBED_CARGO, 0.09),
    ("box_wing_truck", SUBTYPE_BOX_TRUCK, 0.09),
    ("semi_container_trailer", SUBTYPE_TRACTOR_TRAILER, 0.05),
    ("skeletal_chassis_trailer", SUBTYPE_TRACTOR_TRAILER, 0.05),
    ("heavy_lowboy_trailer", SUBTYPE_TRACTOR_TRAILER, 0.05),
    ("tractor_head", SUBTYPE_TRACTOR_TRAILER, 0.05),
    ("rough_terrain_crane", SUBTYPE_CRANE_HEAVY, 0.03),
    ("cargo_crane_truck", SUBTYPE_CRANE_HEAVY, 0.03),
    ("tanker_lorry", SUBTYPE_TANKER_SPECIAL, 0.05),
    ("kei_truck", SUBTYPE_VAN_KEITRUCK, 0.06),
    ("hiace_van", SUBTYPE_VAN_KEITRUCK, 0.05),
    ("sedan", SUBTYPE_PASSENGER_KEI, 0.03),
    ("kei_tall_wagon", SUBTYPE_PASSENGER_KEI, 0.03),
    ("suv_4wd", SUBTYPE_PASSENGER_KEI, 0.03),
)


@dataclass(slots=True)
class _CameraTrajectorySpec:
    """Continuous within-exposure & cross-frame camera trajectory `T_c(t), K(t)` anchored at `t = 0`."""

    cam_o_base: np.ndarray        # (3,) float32 world camera origin at t = 0
    R_cam_base: np.ndarray        # (3, 3) float32 camera-to-world rotation at t = 0
    fx_base: float
    fy_base: float
    family: str                   # motion family from CAMERA_FAMILIES_BY_DOMAIN_SPLIT[domain_split]
    stationary: bool = False
    vel_xyz_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pan_rate_rad_s: float = 0.0
    tilt_rate_rad_s: float = 0.0
    roll_rate_rad_s: float = 0.0
    zoom_rate_s: float = 0.0
    vib_amp_rad: float = 0.0
    vib_freq_hz: float = 11.0
    vib_phase: float = 0.0

    def eval_at(self, t: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Return exact continuous `(cam_o(t), R_cam(t), fx(t), fy(t))` (`== base` at `t = 0.0`)."""
        tf = float(t)
        if self.stationary or abs(tf) < 1e-15:
            return (
                self.cam_o_base.copy(),
                self.R_cam_base.copy(),
                float(self.fx_base),
                float(self.fy_base),
            )

        vx, vy, vz = self.vel_xyz_mps
        # Smooth tanh saturation keeps multi-second camera excursions bounded within `<= 0.34 m`
        t_pos = 0.45 * math.tanh(tf / 0.45)
        t_ang = 0.55 * math.tanh(tf / 0.55)

        dx = vx * t_pos
        dy = vy * t_pos
        dz = vz * t_pos
        d_pan = self.pan_rate_rad_s * t_ang
        d_tilt = self.tilt_rate_rad_s * t_ang
        d_roll = self.roll_rate_rad_s * t_ang
        z_scale = 1.0 + self.zoom_rate_s * t_ang

        w0 = 2.0 * math.pi * self.vib_freq_hz
        ph = self.vib_phase
        if self.family == "ptz_orbit_curved":
            dx += 0.14 * (1.0 - math.cos(1.6 * t_ang)) * (1.0 if vx >= 0.0 else -1.0)
            d_pan += 0.035 * math.sin(1.6 * t_ang)
        elif self.family == "zoom_pull_coupled":
            dy -= 0.12 * t_pos
            z_scale = float(np.clip(1.0 + self.zoom_rate_s * t_ang, 0.78, 1.32))
        elif self.family == "dolly_lateral_track":
            dx = vx * t_pos
            d_pan = self.pan_rate_rad_s * t_ang
        elif self.family == "zoom_breathing_osc":
            z_scale = 1.0 + self.zoom_rate_s * math.sin(2.0 * math.pi * 2.4 * tf)
        elif self.family == "pole_harmonic_sway":
            d_tilt += self.vib_amp_rad * (math.sin(w0 * tf + ph) - math.sin(ph))
            d_pan += 0.65 * self.vib_amp_rad * (math.cos(0.73 * w0 * tf + ph) - math.cos(ph))
            dz += 0.008 * (math.sin(w0 * tf + ph) - math.sin(ph))
        elif self.family == "pole_bimodal_flutter":
            d_tilt += self.vib_amp_rad * (
                0.65 * (math.sin(w0 * tf + ph) - math.sin(ph))
                + 0.35 * (math.sin(2.37 * w0 * tf + 1.7 * ph) - math.sin(1.7 * ph))
            )
            d_roll += 0.45 * self.vib_amp_rad * (math.sin(1.61 * w0 * tf + ph) - math.sin(ph))
        elif self.family == "pole_damped_tremor":
            damp = math.exp(-1.1 * abs(tf))
            d_tilt += self.vib_amp_rad * (damp * math.sin(w0 * tf + ph) - math.sin(ph))
            d_pan += 0.55 * self.vib_amp_rad * (damp * math.sin(1.42 * w0 * tf + 0.6 * ph) - math.sin(0.6 * ph))
        elif self.vib_amp_rad > 0.0:
            d_tilt += self.vib_amp_rad * (math.sin(w0 * tf + ph) - math.sin(ph))

        cam_o_t = np.array(
            [
                float(self.cam_o_base[0]) + dx,
                float(self.cam_o_base[1]) + dy,
                max(0.55, float(self.cam_o_base[2]) + dz),
            ],
            dtype=np.float32,
        )
        # Local camera perturbation (tilt around cam +X, roll around cam +Z) + world-Z pan
        ct, st = math.cos(d_tilt), math.sin(d_tilt)
        cr, sr = math.cos(d_roll), math.sin(d_roll)
        R_loc = np.array(
            [
                [cr, -sr, 0.0],
                [ct * sr, ct * cr, -st],
                [st * sr, st * cr, ct],
            ],
            dtype=np.float32,
        )
        R_cam_t = (_rot_z(d_pan) @ self.R_cam_base @ R_loc).astype(np.float32)
        z_scale_c = float(np.clip(z_scale, 0.76, 1.36))
        return cam_o_t, R_cam_t, float(self.fx_base) * z_scale_c, float(self.fy_base) * z_scale_c


def _make_planar_quad(
    center_w: np.ndarray,
    outward_normal_w: np.ndarray,
    half_w: float,
    half_h: float,
    tex_key: tuple[str, int],
    cls_id: int,
    veh_idx: int,
    *,
    up_hint_w: np.ndarray | None = None,
    plate_side: int = SIDE_FRONT,
    retro_gain: float = 1.0,
    is_green_plate: bool = False,
    is_large_plate: bool = False,
    plate_text_str: str = "",
    plate_type_str: str = "",
    roughness: float = 0.36,
    metallic: float = 0.0,
    clearcoat: float = 0.0,
    specular_f0: float = 0.0,
    normal_undul_amp: float = 0.0,
    has_backing_substrate: bool = True,
) -> _PlanarQuad3D:
    n = outward_normal_w.astype(np.float32)
    n = n / max(1e-6, float(np.linalg.norm(n)))
    z_up = (
        up_hint_w.astype(np.float32)
        if up_hint_w is not None
        else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    z_up = z_up / max(1e-6, float(np.linalg.norm(z_up)))
    u = np.cross(z_up, n)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-4:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        u = u / u_norm
    v = np.cross(u, n)
    v = v / max(1e-6, float(np.linalg.norm(v)))
    return _PlanarQuad3D(
        center=center_w.astype(np.float32),
        u_axis=u,
        v_axis=v,
        normal=n,
        half_w=float(half_w),
        half_h=float(half_h),
        tex_key=tex_key,
        cls_id=cls_id,
        veh_idx=veh_idx,
        plate_side=plate_side,
        retro_gain=float(retro_gain),
        is_green_plate=is_green_plate,
        is_large_plate=is_large_plate,
        plate_text_str=plate_text_str,
        plate_type_str=plate_type_str,
        roughness=float(roughness),
        metallic=float(metallic),
        clearcoat=float(clearcoat),
        specular_f0=float(specular_f0),
        normal_undul_amp=float(normal_undul_amp),
        has_backing_substrate=bool(has_backing_substrate),
    )


def _prod_csg_recess_cavity(
    *,
    pos_w: np.ndarray,
    R_l2w: np.ndarray,
    center_l: tuple[float, float, float],
    half_w: float,
    half_h: float,
    outward_sign_y: float,
    top_lip_out_m: float,
    cheek_out_m: float,
    inward_depth_m: float,
    top_clear_z_m: float,
    cheek_clear_x_m: float,
    veh_idx: int,
    role: int,
    albedo_rgb: tuple[float, float, float],
    roughness: float = 0.48,
    metallic: float = 0.16,
    clearcoat: float = 0.18,
) -> list[_Prim3D]:
    """Shared 3D CSG recessed cavity production (top overhang lintel + left/right cheeks + bottom sill).

    Used identically across vehicle bumper/tailgate recesses (`veh_idx > 0`) and world wall niches (`veh_idx == 0`)
    so overhang and lateral cheek shadows emerge purely from 3D 1-ray visibility `V(x, omega_i)`.
    """
    RT = R_l2w.T.copy()
    cx, cy, cz = center_l
    sy_dir = 1.0 if outward_sign_y >= 0.0 else -1.0

    # Top overhang lintel
    y_in_top = cy - sy_dir * inward_depth_m
    y_out_top = cy + sy_dir * top_lip_out_m
    ly_top = 0.5 * (y_in_top + y_out_top)
    hy_top = max(0.012, 0.5 * abs(y_out_top - y_in_top))
    hz_top = 0.018
    lz_top = cz + half_h + top_clear_z_m + hz_top
    hx_top = half_w + cheek_clear_x_m + 0.032

    # Left & right side cheek walls
    y_out_chk = cy + sy_dir * cheek_out_m
    ly_chk = 0.5 * (y_in_top + y_out_chk)
    hy_chk = max(0.012, 0.5 * abs(y_out_chk - y_in_top))
    hx_chk = 0.016
    hz_chk = half_h + 0.5 * top_clear_z_m + 0.015
    lz_chk = cz + 0.25 * top_clear_z_m

    # Bottom sill / underrun lip
    hz_bot = 0.016
    lz_bot = cz - half_h - 0.042 - hz_bot

    out: list[_Prim3D] = []
    top_w = pos_w + R_l2w @ np.array([cx, ly_top, lz_top], dtype=np.float32)
    out.append(
        _Prim3D(
            center=(float(top_w[0]), float(top_w[1]), float(top_w[2])),
            rot_w2l=RT,
            half_ext=(hx_top, hy_top, hz_top),
            shape=_SHAPE_OBB,
            veh_idx=veh_idx,
            role=role,
            albedo_rgb=albedo_rgb,
            roughness=roughness,
            metallic=metallic,
            clearcoat=clearcoat,
        )
    )
    if lz_bot - hz_bot > 0.04:
        bot_w = pos_w + R_l2w @ np.array([cx, ly_chk, lz_bot], dtype=np.float32)
        out.append(
            _Prim3D(
                center=(float(bot_w[0]), float(bot_w[1]), float(bot_w[2])),
                rot_w2l=RT,
                half_ext=(hx_top, hy_chk, hz_bot),
                shape=_SHAPE_OBB,
                veh_idx=veh_idx,
                role=role,
                albedo_rgb=albedo_rgb,
                roughness=roughness,
                metallic=metallic,
                clearcoat=clearcoat,
            )
        )
    for sx_s in (-1.0, 1.0):
        lx_c = cx + sx_s * (half_w + cheek_clear_x_m + hx_chk)
        chk_w = pos_w + R_l2w @ np.array([lx_c, ly_chk, lz_chk], dtype=np.float32)
        out.append(
            _Prim3D(
                center=(float(chk_w[0]), float(chk_w[1]), float(chk_w[2])),
                rot_w2l=RT,
                half_ext=(hx_chk, hy_chk, hz_chk),
                shape=_SHAPE_OBB,
                veh_idx=veh_idx,
                role=role,
                albedo_rgb=albedo_rgb,
                roughness=roughness,
                metallic=metallic,
                clearcoat=clearcoat,
            )
        )
    return out


def _prod_linear_lattice_z2(
    *,
    origin_w: np.ndarray,
    R_l2w: np.ndarray,
    half_span_y: float,
    half_span_z: float,
    bar_thick_x: float,
    n_rails: int,
    n_mullions: int,
    veh_idx: int,
    role: int,
    albedo_rgb: tuple[float, float, float],
    roughness: float = 0.42,
    metallic: float = 0.55,
) -> list[_Prim3D]:
    """Shared 1D/2D periodic Z^2 lattice extrusion (vehicle side racks/ladders & world perimeter lattices)."""
    RT = R_l2w.T.copy()
    out: list[_Prim3D] = []
    nr = max(2, int(n_rails))
    nm = max(2, int(n_mullions))
    rail_hz = min(0.032, max(0.014, half_span_z / float(nr * 2.8)))
    mul_hy = min(0.036, max(0.015, half_span_y / float(nm * 3.0)))

    for ri in range(nr):
        fz = -1.0 + 2.0 * (ri / float(max(1, nr - 1)))
        cw = origin_w + R_l2w @ np.array([0.0, 0.0, fz * (half_span_z - rail_hz)], dtype=np.float32)
        out.append(
            _Prim3D(
                center=(float(cw[0]), float(cw[1]), float(cw[2])),
                rot_w2l=RT,
                half_ext=(bar_thick_x, half_span_y, rail_hz),
                shape=_SHAPE_OBB,
                veh_idx=veh_idx,
                role=role,
                albedo_rgb=albedo_rgb,
                roughness=roughness,
                metallic=metallic,
            )
        )
    for mi in range(nm):
        fy = -1.0 + 2.0 * (mi / float(max(1, nm - 1)))
        cw = origin_w + R_l2w @ np.array([0.0, fy * (half_span_y - mul_hy), 0.0], dtype=np.float32)
        out.append(
            _Prim3D(
                center=(float(cw[0]), float(cw[1]), float(cw[2])),
                rot_w2l=RT,
                half_ext=(bar_thick_x * 0.92, mul_hy, half_span_z),
                shape=_SHAPE_OBB,
                veh_idx=veh_idx,
                role=role,
                albedo_rgb=albedo_rgb,
                roughness=roughness,
                metallic=metallic,
            )
        )
    return out


def _prod_articulated_truss_or_boom(
    *,
    base_w: np.ndarray,
    R_base_l2w: np.ndarray,
    mast_h: float,
    boom_len_1: float,
    boom_len_2: float,
    section_r: float,
    slew_yaw_rad: float,
    elev_1_rad: float,
    elev_2_rad: float,
    veh_idx: int,
    role: int,
    albedo_rgb: tuple[float, float, float],
    accent_rgb: tuple[float, float, float],
    roughness: float = 0.34,
    metallic: float = 0.32,
) -> list[_Prim3D]:
    """Shared multi-link rigid articulated arm/truss production (`unic` crane booms & world truss portals)."""
    out: list[_Prim3D] = []
    R_slew = R_base_l2w @ _rot_zyx(slew_yaw_rad, 0.0, 0.0)
    mast_c = base_w + R_slew @ np.array([0.0, 0.0, 0.5 * mast_h], dtype=np.float32)
    out.append(
        _Prim3D(
            center=(float(mast_c[0]), float(mast_c[1]), float(mast_c[2])),
            rot_w2l=R_slew.T.copy(),
            half_ext=(section_r * 1.15, section_r * 1.15, 0.5 * mast_h),
            shape=_SHAPE_OBB,
            veh_idx=veh_idx,
            role=role,
            albedo_rgb=albedo_rgb,
            roughness=roughness,
            metallic=metallic,
            clearcoat=0.35,
        )
    )
    piv1_w = base_w + R_slew @ np.array([0.0, 0.0, mast_h * 0.92], dtype=np.float32)
    # Link 1 extends rearward (`-Y` local) elevated by `elev_1_rad`
    R_b1 = R_slew @ _rot_zyx(0.0, -elev_1_rad, 0.0)
    b1_c = piv1_w + R_b1 @ np.array([0.0, -0.5 * boom_len_1, 0.0], dtype=np.float32)
    out.append(
        _Prim3D(
            center=(float(b1_c[0]), float(b1_c[1]), float(b1_c[2])),
            rot_w2l=R_b1.T.copy(),
            half_ext=(section_r, 0.5 * boom_len_1, section_r * 1.08),
            shape=_SHAPE_OBB,
            veh_idx=veh_idx,
            role=role,
            albedo_rgb=albedo_rgb,
            roughness=roughness,
            rib_freq=6.0,
            metallic=metallic,
            clearcoat=0.40,
        )
    )
    # Hydraulic lift cylinder strut beneath Link 1
    R_hyd = R_slew @ _rot_zyx(0.0, -(elev_1_rad + 0.36), 0.0)
    hyd_len = 0.48 * boom_len_1
    hyd_c = (base_w + R_slew @ np.array([0.0, -0.06, mast_h * 0.38], dtype=np.float32)) + R_hyd @ np.array([0.0, -0.5 * hyd_len, 0.0], dtype=np.float32)
    out.append(
        _Prim3D(
            center=(float(hyd_c[0]), float(hyd_c[1]), float(hyd_c[2])),
            rot_w2l=R_hyd.T.copy(),
            half_ext=(section_r * 0.42, 0.5 * hyd_len, section_r * 0.42),
            shape=_SHAPE_OBB,
            veh_idx=veh_idx,
            role=_ROLE_CHROME_METAL if veh_idx > 0 else role,
            albedo_rgb=(0.68, 0.71, 0.74) if veh_idx > 0 else accent_rgb,
            roughness=0.16,
            metallic=0.88,
        )
    )
    # Telescopic / articulated outer Link 2
    piv2_w = piv1_w + R_b1 @ np.array([0.0, -boom_len_1 * 0.92, 0.0], dtype=np.float32)
    R_b2 = R_b1 @ _rot_zyx(0.0, -elev_2_rad, 0.0)
    b2_c = piv2_w + R_b2 @ np.array([0.0, -0.5 * boom_len_2, 0.0], dtype=np.float32)
    out.append(
        _Prim3D(
            center=(float(b2_c[0]), float(b2_c[1]), float(b2_c[2])),
            rot_w2l=R_b2.T.copy(),
            half_ext=(section_r * 0.76, 0.5 * boom_len_2, section_r * 0.82),
            shape=_SHAPE_OBB,
            veh_idx=veh_idx,
            role=role,
            albedo_rgb=accent_rgb,
            roughness=roughness,
            metallic=metallic,
            clearcoat=0.35,
        )
    )
    return out


def _prod_flex_wave_sheet(
    *,
    center_w: np.ndarray,
    R_l2w: np.ndarray,
    half_ext: tuple[float, float, float],
    veh_idx: int,
    role: int,
    albedo_rgb: tuple[float, float, float],
    roughness: float = 0.52,
    wave_amp: float = 0.045,
    wave_freq: float = 4.6,
    phase: float = 0.7,
    decal_spec: tuple[int, int, float, float, float, float, float] | None = None,
) -> _Prim3D:
    """Shared multi-octave flexural wave sheet production (vehicle covers/curtains & world hoarding screens)."""
    sx, sy, sz = half_ext
    eff_amp = min(float(wave_amp), float(sx) * 0.62)
    return _Prim3D(
        center=(float(center_w[0]), float(center_w[1]), float(center_w[2])),
        rot_w2l=R_l2w.T.copy(),
        half_ext=(float(sx), float(sy), float(sz)),
        shape=_SHAPE_FLEX_SHEET,
        veh_idx=veh_idx,
        role=role,
        albedo_rgb=albedo_rgb,
        roughness=roughness,
        mud_phase=float(phase),
        metallic=0.04,
        clearcoat=0.14,
        normal_undul_amp=0.035,
        wave_amp=eff_amp,
        wave_freq=float(wave_freq),
        decal_spec=decal_spec,
    )


def _assemble_vehicle_3d(
    spec: _VehiclePassSpec,
    t: float,
    veh_idx: int,
    bank: _GPU3DTextureBank,
) -> tuple[
    list[_Prim3D],
    list[_PlanarQuad3D],
    np.ndarray,
    np.ndarray | None,
    list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]],
]:
    """Build exact 3D primitives, 3D plates/stickers, Z=0 road footprint `(4, 3)`, optional articulated
    `(8, 3)` Z=0 footprint, and conservative 3D cabin OBBs `(center_w, R_l2w, half_ext)` at time `t`."""
    pos, yaw, pitch, roll = spec.pose_at(t)
    R = _rot_zyx(yaw, pitch, roll)
    RT = R.T.copy()
    up_body = R[:, 2].copy()

    def _to_world(lx: float, ly: float, lz: float) -> np.ndarray:
        return pos + R @ np.array([lx, ly, lz], dtype=np.float32)

    prims: list[_Prim3D] = []
    quads: list[_PlanarQuad3D] = []
    cabin_obbs_w: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = []
    articulated_fp_w: np.ndarray | None = None

    L = spec.length_m
    W = spec.width_m
    hw = 0.5 * W
    hl = 0.5 * L
    arch = spec.archetype
    is_articulated = arch in _ARTICULATED_ARCHETYPES
    is_heavy = spec.subtype in (
        SUBTYPE_DUMP_TRUCK,
        SUBTYPE_CONCRETE_MIXER,
        SUBTYPE_TANKER_SPECIAL,
        SUBTYPE_TRACTOR_TRAILER,
        SUBTYPE_CRANE_HEAVY,
        SUBTYPE_BOX_TRUCK,
        SUBTYPE_FLATBED_CARGO,
    )

    is_med_truck = (arch in ("box_wing_truck", "flatbed_truck", "cargo_crane_truck")) and (not spec.is_large_truck)
    wheel_r = 0.40 if is_med_truck else (0.52 if is_heavy else (0.29 if "kei" in arch else 0.34))
    wheel_w = 0.22 if is_med_truck else (0.28 if is_heavy else 0.20)
    chassis_z = wheel_r * 0.92
    dn_wheel_n = max(3, int(spec.radial_dn_n))
    n_dec_bank = max(1, len(bank.alpha_decal_mips))
    dec_i = int(spec.decal_idx) % n_dec_bank
    dec_rel = float(spec.decal_relief)

    # 1. Wheels / axles with compositional D_N radial aperture/stud hubs (`N in {3..12}`)
    if arch == "cargo_crane_truck" and not spec.is_large_truck:
        axle_ys = [hl - 1.15, -hl + 1.25]
    elif spec.subtype in (SUBTYPE_DUMP_TRUCK, SUBTYPE_CONCRETE_MIXER, SUBTYPE_TANKER_SPECIAL, SUBTYPE_CRANE_HEAVY):
        axle_ys = [hl - 1.35, -hl + 2.05, -hl + 0.85]
    elif arch == "tractor_head":
        axle_ys = [hl - 1.22, -hl + 1.92, -hl + 0.72]
    elif is_articulated:
        axle_ys = [hl - 1.25, 0.55]
    else:
        axle_ys = [hl * 0.62, -hl * 0.62]

    for ay in axle_ys:
        for side_s in (-1.0, 1.0):
            wc = _to_world(side_s * (hw - wheel_w * 0.55), ay, wheel_r)
            prims.append(
                _Prim3D(
                    center=(float(wc[0]), float(wc[1]), float(wc[2])),
                    rot_w2l=RT,
                    half_ext=(wheel_w * 0.55, wheel_r, wheel_r),
                    shape=_SHAPE_CYLINDER_X,
                    veh_idx=veh_idx,
                    role=_ROLE_TIRE,
                    albedo_rgb=(0.035, 0.038, 0.042),
                    roughness=0.82,
                    mud_level=spec.mud_level * 1.15,
                    mud_phase=spec.mud_phase,
                    metallic=0.14,
                    radial_n=dn_wheel_n,
                )
            )

    # Dark chassis frame rail between wheels (bobtail `tractor_head` uses explicit dual I-beams below)
    if arch != "tractor_head":
        ch_len = hl * 0.88 if not is_articulated else hl * 0.45
        ch_y = 0.0 if not is_articulated else (hl - ch_len)
        ch_c = _to_world(0.0, ch_y, chassis_z + 0.16)
        prims.append(
            _Prim3D(
                center=(float(ch_c[0]), float(ch_c[1]), float(ch_c[2])),
                rot_w2l=RT,
                half_ext=(hw * 0.62, ch_len, 0.22),
                shape=_SHAPE_OBB,
                veh_idx=veh_idx,
                role=_ROLE_DARK_CHASSIS,
                albedo_rgb=(0.045, 0.048, 0.054),
                roughness=0.75,
                mud_level=spec.mud_level,
                mud_phase=spec.mud_phase + 0.7,
                metallic=0.24,
            )
        )

    def _record_cabin_obb(c_w: np.ndarray, r_l2w: np.ndarray, sx: float, sy: float, sz: float) -> None:
        # Conservative +4cm safety enclosure around the exact rendered cabin glass primitive!
        cabin_obbs_w.append((c_w.astype(np.float32), r_l2w.astype(np.float32), (sx + 0.04, sy + 0.04, sz + 0.04)))

    # 2. Cab + Compositional Upper Superstructure per Archetype
    rear_plate_local_x: float | None = None
    rear_plate_local_y = -hl - 0.038
    rear_plate_local_z = chassis_z + 0.12
    rear_plate_tilt_z = -0.06
    taillight_local_z = chassis_z * 0.82 + 0.14
    front_plate_local_y = hl + 0.038
    front_plate_local_z = wheel_r * 0.95

    if arch in ("sedan", "kei_tall_wagon", "suv_4wd", "hiace_van"):
        low_h = (spec.body_h_m - wheel_r * 0.45) * (0.52 if arch != "hiace_van" else 0.56)
        low_c = _to_world(0.0, 0.0, wheel_r * 0.45 + low_h * 0.5)
        van_decal = (dec_i, 0, 0.0, 0.05, 0.42, 0.34, dec_rel) if (arch == "hiace_van" and len(bank.alpha_decal_mips) > 0) else None
        prims.append(
            _Prim3D(
                center=(float(low_c[0]), float(low_c[1]), float(low_c[2])),
                rot_w2l=RT,
                half_ext=(hw, hl, low_h * 0.5),
                shape=_SHAPE_OBB,
                veh_idx=veh_idx,
                role=_ROLE_BODY,
                albedo_rgb=spec.paint_lin,
                roughness=0.18,
                mud_level=spec.mud_level * 0.65,
                mud_phase=spec.mud_phase,
                metallic=spec.paint_metallic,
                clearcoat=max(0.55, spec.paint_clearcoat),
                normal_undul_amp=0.014,
                decal_spec=van_decal,
            )
        )
        gh_len = hl * (0.58 if arch == "sedan" else (0.86 if arch == "hiace_van" else 0.74))
        gh_y = -hl * 0.06 if arch == "sedan" else -hl * 0.04
        gh_h = spec.body_h_m - (wheel_r * 0.45 + low_h)
        gh_z = wheel_r * 0.45 + low_h + gh_h * 0.5
        gh_c = _to_world(0.0, gh_y, gh_z)
        gh_ext = (hw * 0.88, gh_len, gh_h * 0.5)
        prims.append(
            _Prim3D(
                center=(float(gh_c[0]), float(gh_c[1]), float(gh_c[2])),
                rot_w2l=RT,
                half_ext=gh_ext,
                shape=_SHAPE_SLOPED_CAB,
                veh_idx=veh_idx,
                role=_ROLE_CABIN_GLASS,
                albedo_rgb=(0.05, 0.07, 0.09),
                roughness=0.06,
                slope_param=0.42 if arch == "sedan" else 0.24,
                clearcoat=0.92,
                specular_f0=0.045,
            )
        )
        _record_cabin_obb(gh_c, R, gh_ext[0], gh_ext[1], gh_ext[2])
        rear_plate_local_z = wheel_r * 0.95 + (0.22 if arch != "sedan" else 0.16)
        front_plate_local_z = wheel_r * 0.82
    else:
        # Cab-over-engine Japanese truck / medium-duty cab / kei-truck / sleeper tractor / right-offset crane cab
        is_rt_crane = arch == "rough_terrain_crane"
        cab_len = (
            0.88 if arch == "kei_truck"
            else (
                1.62 if arch == "tractor_head"
                else (1.42 if is_rt_crane else (1.08 if is_med_truck else (1.18 if not is_heavy else 1.32)))
            )
        )
        cab_cy = (hl - cab_len * 0.5 - 0.12) if is_rt_crane else (hl - cab_len * 0.5)
        cab_cx = (hw * 0.46) if is_rt_crane else 0.0
        cab_hw = (hw * 0.48) if is_rt_crane else (hw * (0.92 if is_med_truck else 0.96))
        cab_low_h = (spec.cab_h_m - chassis_z) * 0.52
        cab_low_z = chassis_z + cab_low_h * 0.5
        c_low = _to_world(cab_cx, cab_cy, cab_low_z)
        cab_decal = (
            (dec_i, 0, 0.0, 0.02, 0.58, 0.36, dec_rel)
            if (not spec.has_side_door_decal and len(bank.alpha_decal_mips) > 0)
            else None
        )
        prims.append(
            _Prim3D(
                center=(float(c_low[0]), float(c_low[1]), float(c_low[2])),
                rot_w2l=RT,
                half_ext=(cab_hw, cab_len * 0.5, cab_low_h * 0.5),
                shape=_SHAPE_OBB,
                veh_idx=veh_idx,
                role=_ROLE_BODY,
                albedo_rgb=spec.paint_lin,
                roughness=0.24,
                rib_freq=6.0,
                mud_level=spec.mud_level * 0.8,
                mud_phase=spec.mud_phase,
                metallic=spec.paint_metallic,
                clearcoat=spec.paint_clearcoat,
                normal_undul_amp=0.018,
                decal_spec=cab_decal,
            )
        )
        cab_up_h = (spec.cab_h_m - chassis_z) * 0.48
        cab_up_z = chassis_z + cab_low_h + cab_up_h * 0.5
        c_up = _to_world(cab_cx, cab_cy - 0.04, cab_up_z)
        cab_glass_ext = (cab_hw * 0.96, cab_len * 0.48, cab_up_h * 0.5)
        prims.append(
            _Prim3D(
                center=(float(c_up[0]), float(c_up[1]), float(c_up[2])),
                rot_w2l=RT,
                half_ext=cab_glass_ext,
                shape=_SHAPE_SLOPED_CAB,
                veh_idx=veh_idx,
                role=_ROLE_CABIN_GLASS,
                albedo_rgb=(0.05, 0.07, 0.09),
                roughness=0.07,
                slope_param=0.26 if is_med_truck else 0.22,
                clearcoat=0.92,
                specular_f0=0.045,
            )
        )
        _record_cabin_obb(c_up, R, cab_glass_ext[0], cab_glass_ext[1], cab_glass_ext[2])

        # Heavy/medium-duty protruding side mirrors
        mir_off = 0.11 if is_med_truck else 0.14
        mir_hz = 0.18 if is_med_truck else 0.22
        for side_s in (-1.0, 1.0):
            mc = _to_world(side_s * (hw + mir_off), hl - 0.22, cab_up_z)
            prims.append(
                _Prim3D(
                    center=(float(mc[0]), float(mc[1]), float(mc[2])),
                    rot_w2l=RT,
                    half_ext=(0.09 if is_med_truck else 0.10, 0.08, mir_hz),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_DARK_CHASSIS,
                    albedo_rgb=(0.04, 0.04, 0.05),
                    roughness=0.48,
                    metallic=0.20,
                    clearcoat=0.25,
                )
            )

        # Front steel/resin bumper: randomized across matte black (38%), body-painted (32%), and chrome (30%)!
        if is_heavy:
            if int(spec.bumper_style_front) == 0:
                fb_role, fb_alb, fb_rough, fb_met, fb_coat = _ROLE_DARK_CHASSIS, (0.035, 0.038, 0.042), 0.72, 0.18, 0.05
            elif int(spec.bumper_style_front) == 1:
                fb_role, fb_alb, fb_rough, fb_met, fb_coat = _ROLE_BODY, spec.paint_lin, 0.24, spec.paint_metallic, spec.paint_clearcoat
            else:
                fb_role, fb_alb, fb_rough, fb_met, fb_coat = _ROLE_CHROME_METAL, (0.62, 0.65, 0.68), 0.15, 0.90, 0.28
        else:
            fb_role, fb_alb, fb_rough, fb_met, fb_coat = (
                (_ROLE_DARK_CHASSIS, (0.035, 0.038, 0.042), 0.70, 0.15, 0.05)
                if int(spec.bumper_style_front) == 0
                else (_ROLE_BODY, spec.paint_lin, 0.24, spec.paint_metallic, spec.paint_clearcoat)
            )
        fb_hz = 0.18 if is_med_truck else 0.22
        fb_c = _to_world(0.0, hl - 0.08, chassis_z * 0.82)
        prims.append(
            _Prim3D(
                center=(float(fb_c[0]), float(fb_c[1]), float(fb_c[2])),
                rot_w2l=RT,
                half_ext=(hw * 0.99, 0.16, fb_hz),
                shape=_SHAPE_OBB,
                veh_idx=veh_idx,
                role=fb_role,
                albedo_rgb=fb_alb,
                roughness=fb_rough,
                mud_level=spec.mud_level,
                mud_phase=spec.mud_phase + 0.3,
                metallic=fb_met,
                clearcoat=fb_coat,
                normal_undul_amp=0.015,
            )
        )
        front_plate_local_y = hl + 0.105
        front_plate_local_z = chassis_z * 0.80

        def _rear_bumper_style_props() -> tuple[int, tuple[float, float, float], float, float, float]:
            if int(spec.bumper_style_rear) == 1:
                return _ROLE_BODY, spec.paint_lin, 0.28, spec.paint_metallic, spec.paint_clearcoat
            if int(spec.bumper_style_rear) == 2:
                return _ROLE_CHROME_METAL, (0.62, 0.65, 0.68), 0.15, 0.90, 0.28
            return _ROLE_DARK_CHASSIS, (0.035, 0.038, 0.042), 0.72, 0.20, 0.05

        bed_y0 = -hl
        bed_y1 = hl - cab_len - 0.14
        bed_cy = 0.5 * (bed_y0 + bed_y1)
        bed_hl = max(0.6, 0.5 * (bed_y1 - bed_y0))

        if arch in ("dump_10t", "deep_dump"):
            # Ribbed steel dump vessel + cab protector + overhanging rear tailgate -> genuine 3D ray-cast tailgate umbra!
            bed_h = (1.55 if arch == "dump_10t" else 2.05)
            bed_cz = chassis_z + 0.30 + bed_h * 0.5
            bc = _to_world(0.0, bed_cy, bed_cz)
            dump_decal = ((dec_i + 1) % n_dec_bank, 0, 0.05, 0.12, 0.38, 0.26, dec_rel)
            prims.append(
                _Prim3D(
                    center=(float(bc[0]), float(bc[1]), float(bc[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.99, bed_hl, bed_h * 0.5),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=spec.secondary_lin,
                    roughness=0.44,
                    rib_freq=10.0,
                    mud_level=max(0.22, spec.mud_level * 1.35),
                    mud_phase=spec.mud_phase + 1.2,
                    metallic=0.22,
                    clearcoat=0.24,
                    normal_undul_amp=0.026,
                    decal_spec=dump_decal,
                )
            )
            # Cab protector (`プロテクター`) overhanging cab roof
            pc = _to_world(0.0, bed_y1 + 0.25, spec.cab_h_m + 0.14)
            prims.append(
                _Prim3D(
                    center=(float(pc[0]), float(pc[1]), float(pc[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.96, 0.32, 0.10),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=spec.secondary_lin,
                    roughness=0.46,
                    metallic=0.22,
                )
            )
            if spec.has_dump_load:
                # Genuine 3D multi-lobed tapered conical heap (`_SHAPE_TAPER_DRUM_Y`) rising above dump vessel rim
                z_rim = chassis_z + 0.30 + bed_h
                mound_specs = (
                    (0.0, bed_cy + 0.04, z_rim - 0.06, 0.0, 0.0, (hw * 0.90, bed_hl * 0.88, 0.44), 0.38, 0.0),
                    (-hw * 0.13, bed_cy + bed_hl * 0.20, z_rim + 0.14, -0.06, 0.05, (hw * 0.64, bed_hl * 0.48, 0.34), 0.48, 0.7),
                    (hw * 0.12, bed_cy - bed_hl * 0.18, z_rim + 0.10, 0.07, -0.05, (hw * 0.60, bed_hl * 0.44, 0.30), 0.50, 1.9),
                )
                for m_lx, m_ly, m_lz, m_pit, m_rol, m_he, m_tap, m_ph in mound_specs:
                    ld_c = _to_world(m_lx, m_ly, m_lz)
                    R_mound = R @ _rot_zyx(0.0, m_pit, m_rol)
                    prims.append(
                        _Prim3D(
                            center=(float(ld_c[0]), float(ld_c[1]), float(ld_c[2])),
                            rot_w2l=R_mound.T.copy(),
                            half_ext=m_he,
                            shape=_SHAPE_TAPER_DRUM_Y,
                            veh_idx=veh_idx,
                            role=_ROLE_CARGO_MATTE,
                            albedo_rgb=spec.dump_load_lin,
                            roughness=0.84,
                            mud_phase=spec.mud_phase + m_ph,
                            slope_param=m_tap,
                        )
                    )
            elif spec.has_flex_sheet or spec.veh_struct_prod == "flex_wave_sheet":
                # Multi-octave flexural wave cargo sheet rolled across dump rim (`_ROLE_BODY` so `_ROLE_CARGO_MATTE` stays pure mound)
                sh_c = _to_world(0.0, bed_cy, chassis_z + 0.30 + bed_h + 0.05)
                R_sh = R @ _rot_zyx(0.0, 0.0, 0.5 * math.pi)
                prims.append(
                    _prod_flex_wave_sheet(
                        center_w=sh_c,
                        R_l2w=R_sh,
                        half_ext=(0.07, bed_hl * 0.94, hw * 0.94),
                        veh_idx=veh_idx,
                        role=_ROLE_BODY,
                        albedo_rgb=spec.flex_sheet_lin,
                        roughness=0.54,
                        wave_amp=0.042,
                        wave_freq=5.2,
                        phase=spec.mud_phase,
                    )
                )
            # Side Z^2 ladder on left dump vessel flank
            lad_w = _to_world(-(hw * 0.99 + 0.025), bed_y1 - 0.52, bed_cz)
            prims.extend(
                _prod_linear_lattice_z2(
                    origin_w=lad_w,
                    R_l2w=R,
                    half_span_y=0.24,
                    half_span_z=bed_h * 0.44,
                    bar_thick_x=0.018,
                    n_rails=5,
                    n_mullions=2,
                    veh_idx=veh_idx,
                    role=_ROLE_DARK_CHASSIS,
                    albedo_rgb=(0.09, 0.10, 0.11),
                )
            )
            rb_role, rb_alb, rb_rough, rb_met, rb_coat = _rear_bumper_style_props()
            ub_c = _to_world(0.0, -hl + 0.48, chassis_z * 0.68)
            prims.append(
                _Prim3D(
                    center=(float(ub_c[0]), float(ub_c[1]), float(ub_c[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.88, 0.10, 0.12),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=rb_role,
                    albedo_rgb=rb_alb,
                    roughness=rb_rough,
                    metallic=rb_met,
                    clearcoat=rb_coat,
                )
            )
            rear_plate_local_y = -hl + 0.34   # Recessed 34cm underneath overhanging 3D dump tailgate!
            rear_plate_local_z = chassis_z * 0.78

        elif arch == "concrete_mixer":
            # 3D tilted rotating conical-cylindrical concrete mixer drum + conformal frameless curved-drum typography!
            drum_pitch = -0.22
            R_drum = R @ _rot_zyx(0.0, drum_pitch, (t * 1.4) % (2.0 * math.pi))
            dc = _to_world(0.0, bed_cy + 0.16, chassis_z + 1.25)
            drum_decal = ((dec_i + 2) % n_dec_bank, 2, 0.0, 0.0, 0.46, 0.34, dec_rel)
            prims.append(
                _Prim3D(
                    center=(float(dc[0]), float(dc[1]), float(dc[2])),
                    rot_w2l=R_drum.T.copy(),
                    half_ext=(hw * 0.94, bed_hl * 0.88, hw * 0.94),
                    shape=_SHAPE_TAPER_DRUM_Y,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=spec.secondary_lin,
                    roughness=0.30,
                    rib_freq=5.0,
                    mud_level=spec.mud_level,
                    mud_phase=spec.mud_phase,
                    metallic=0.14,
                    clearcoat=0.48,
                    normal_undul_amp=0.018,
                    decal_spec=drum_decal,
                )
            )
            # Rear charge hopper and overhanging splash hood cast shadows through ray visibility.
            hopper_top_z = spec.body_h_m * 0.84
            hc = _to_world(0.0, -hl + 0.34, hopper_top_z - 0.36)
            prims.append(
                _Prim3D(
                    center=(float(hc[0]), float(hc[1]), float(hc[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.38, 0.28, 0.38),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=spec.paint_lin,
                    roughness=0.40,
                    metallic=0.18,
                )
            )
            hood_c = _to_world(0.0, -hl + 0.12, hopper_top_z + 0.04)
            prims.append(
                _Prim3D(
                    center=(float(hood_c[0]), float(hood_c[1]), float(hood_c[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.78, 0.16, 0.06),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=spec.paint_lin,
                    roughness=0.38,
                    metallic=0.18,
                )
            )
            # Hopper-adjacent forward-tilted physical plate bracket.
            side_m = -1.0 if spec.plate_offset_x_rear < 0.0 else 1.0
            rear_plate_local_x = side_m * (hw * 0.54)
            rear_plate_local_y = -hl - 0.045
            rear_plate_local_z = spec.body_h_m * 0.68
            rear_plate_tilt_z = -0.14

        elif arch == "tractor_head":
            # Standalone bobtail fifth-wheel semi-tractor rear (exposed I-beams + fifth-wheel coupler + zebra bar)
            rail_hl = max(0.8, 0.5 * (hl + bed_y1))
            rail_cy = 0.5 * (-hl + bed_y1)
            for side_s in (-1.0, 1.0):
                rc = _to_world(side_s * 0.44, rail_cy, chassis_z + 0.16)
                prims.append(
                    _Prim3D(
                        center=(float(rc[0]), float(rc[1]), float(rc[2])),
                        rot_w2l=RT,
                        half_ext=(0.09, rail_hl, 0.18),
                        shape=_SHAPE_OBB,
                        veh_idx=veh_idx,
                        role=_ROLE_DARK_CHASSIS,
                        albedo_rgb=(0.038, 0.040, 0.046),
                        roughness=0.72,
                        mud_level=spec.mud_level,
                        mud_phase=spec.mud_phase + 0.4,
                        metallic=0.25,
                    )
                )
                qf_c = _to_world(side_s * (hw * 0.80), -hl + 1.32, wheel_r + 0.12)
                prims.append(
                    _Prim3D(
                        center=(float(qf_c[0]), float(qf_c[1]), float(qf_c[2])),
                        rot_w2l=RT,
                        half_ext=(0.30, 1.05, 0.06),
                        shape=_SHAPE_OBB,
                        veh_idx=veh_idx,
                        role=_ROLE_DARK_CHASSIS,
                        albedo_rgb=(0.045, 0.048, 0.052),
                        roughness=0.66,
                        metallic=0.18,
                    )
                )
            fw_c = _to_world(0.0, -hl + 1.45, chassis_z + 0.38)
            prims.append(
                _Prim3D(
                    center=(float(fw_c[0]), float(fw_c[1]), float(fw_c[2])),
                    rot_w2l=RT,
                    half_ext=(0.48, 0.44, 0.08),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_DARK_CHASSIS,
                    albedo_rgb=(0.055, 0.058, 0.065),
                    roughness=0.52,
                    metallic=0.45,
                )
            )
            xm_c = _to_world(0.0, -hl + 0.06, chassis_z * 0.84)
            prims.append(
                _Prim3D(
                    center=(float(xm_c[0]), float(xm_c[1]), float(xm_c[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.86, 0.08, 0.14),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_DARK_CHASSIS,
                    albedo_rgb=(0.040, 0.042, 0.048),
                    roughness=0.72,
                    metallic=0.24,
                )
            )
            zb_c = _to_world(0.0, -hl - 0.01, chassis_z * 0.84 + 0.13)
            prims.append(
                _Prim3D(
                    center=(float(zb_c[0]), float(zb_c[1]), float(zb_c[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.84, 0.02, 0.035),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_CHROME_METAL,
                    albedo_rgb=(0.78, 0.24, 0.18),
                    roughness=0.20,
                    rib_freq=18.0,
                    metallic=0.35,
                    retro_gain=0.90,
                )
            )
            rear_plate_local_x = (-0.42 if spec.plate_offset_x_rear < 0.0 else 0.42)
            rear_plate_local_y = -hl - 0.03
            rear_plate_local_z = chassis_z * 0.78

        elif arch == "tanker_lorry":
            tc = _to_world(0.0, bed_cy, chassis_z + 1.12)
            tank_decal = ((dec_i + 3) % n_dec_bank, 2, 0.0, 0.0, 0.44, 0.36, dec_rel)
            prims.append(
                _Prim3D(
                    center=(float(tc[0]), float(tc[1]), float(tc[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.95, bed_hl * 0.96, 0.86),
                    shape=_SHAPE_TAPER_DRUM_Y,
                    veh_idx=veh_idx,
                    role=_ROLE_CHROME_METAL,
                    albedo_rgb=spec.secondary_lin,
                    roughness=0.16,
                    mud_level=spec.mud_level * 0.6,
                    mud_phase=spec.mud_phase,
                    slope_param=0.05,  # nearly cylindrical tank
                    metallic=0.82,
                    clearcoat=0.32,
                    normal_undul_amp=0.014,
                    decal_spec=tank_decal,
                )
            )
            rear_plate_local_y = -hl + 0.10
            rear_plate_local_z = chassis_z * 0.78

        elif arch == "rough_terrain_crane":
            # Carrier deck + slewing counterweight turret + compositional multi-link telescopic boom!
            deck_c = _to_world(0.0, -0.15, chassis_z + 0.44)
            prims.append(
                _Prim3D(
                    center=(float(deck_c[0]), float(deck_c[1]), float(deck_c[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.98, hl * 0.88, 0.34),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=spec.secondary_lin,
                    roughness=0.36,
                    mud_level=spec.mud_level,
                    mud_phase=spec.mud_phase,
                    metallic=0.18,
                    clearcoat=0.32,
                )
            )
            tur_h = max(0.75, spec.cab_h_m - (chassis_z + 0.65))
            tur_c = _to_world(-hw * 0.14, -hl * 0.34, chassis_z + 0.72 + tur_h * 0.5)
            prims.append(
                _Prim3D(
                    center=(float(tur_c[0]), float(tur_c[1]), float(tur_c[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.76, hl * 0.42, tur_h * 0.5),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=spec.paint_lin,
                    roughness=0.34,
                    metallic=0.18,
                    clearcoat=0.36,
                    decal_spec=((dec_i + 4) % n_dec_bank, 0, 0.0, 0.0, 0.45, 0.35, dec_rel),
                )
            )
            # Multi-link telescopic boom seated alongside right-offset cab (extending forward along `+Y` via yaw=pi)
            boom_base = _to_world(-hw * 0.34, -hl * 0.42, chassis_z + 0.76)
            prims.extend(
                _prod_articulated_truss_or_boom(
                    base_w=boom_base,
                    R_base_l2w=R,
                    mast_h=max(0.55, tur_h * 0.85),
                    boom_len_1=hl * 0.95,
                    boom_len_2=hl * 0.46,
                    section_r=0.28,
                    slew_yaw_rad=math.pi + 0.04 * spec.boom_yaw_rad,
                    elev_1_rad=0.05 + 0.35 * spec.boom_elev_rad,
                    elev_2_rad=-0.02,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=spec.paint_lin,
                    accent_rgb=spec.secondary_lin,
                    roughness=0.34,
                )
            )
            rear_plate_local_y = -hl + 0.06
            rear_plate_local_z = chassis_z * 0.80

        elif arch == "cargo_crane_truck":
            # Cab-behind articulated cargo crane truck (`ユニック車` / `unic`):
            # Outrigger stabilizer jacks + multi-link articulated boom stowed over rear drop-side flatbed!
            crane_y = hl - cab_len - 0.36
            # Transverse outrigger beam + left/right vertical jack cylinders
            outrig_c = _to_world(0.0, crane_y, chassis_z + 0.24)
            prims.append(
                _Prim3D(
                    center=(float(outrig_c[0]), float(outrig_c[1]), float(outrig_c[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.99, 0.14, 0.14),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_DARK_CHASSIS,
                    albedo_rgb=(0.06, 0.06, 0.07),
                    roughness=0.60,
                    metallic=0.35,
                )
            )
            jack_hz = 0.36
            for side_s in (-1.0, 1.0):
                jk_c = _to_world(side_s * (hw * 0.96), crane_y, max(jack_hz + 0.10, chassis_z + 0.16))
                prims.append(
                    _Prim3D(
                        center=(float(jk_c[0]), float(jk_c[1]), float(jk_c[2])),
                        rot_w2l=RT,
                        half_ext=(0.075, 0.09, jack_hz),
                        shape=_SHAPE_OBB,
                        veh_idx=veh_idx,
                        role=_ROLE_BODY,
                        albedo_rgb=(0.66, 0.42, 0.05),
                        roughness=0.38,
                        rib_freq=10.0,
                        metallic=0.25,
                    )
                )
            u_bed_y0 = -hl
            u_bed_y1 = crane_y - 0.30
            u_bed_cy = 0.5 * (u_bed_y0 + u_bed_y1)
            u_bed_hl = max(0.55, 0.5 * (u_bed_y1 - u_bed_y0))
            gate_h = 0.42 if not is_med_truck else 0.36
            bc_u = _to_world(0.0, u_bed_cy, chassis_z + 0.14 + gate_h * 0.5)
            prims.append(
                _Prim3D(
                    center=(float(bc_u[0]), float(bc_u[1]), float(bc_u[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.98, u_bed_hl, gate_h * 0.5),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=spec.secondary_lin,
                    roughness=0.40,
                    rib_freq=6.0,
                    mud_level=spec.mud_level,
                    mud_phase=spec.mud_phase,
                    metallic=0.16,
                    clearcoat=0.30,
                    decal_spec=((dec_i + 5) % n_dec_bank, 0, 0.0, 0.0, 0.48, 0.36, dec_rel),
                )
            )
            # Multi-link articulated crane column + main lift boom + outer telescopic jib stowed rearward over bed
            unic_base = _to_world(0.0, crane_y, chassis_z + 0.34)
            mast_u = max(0.85, (spec.cab_h_m - chassis_z) * 0.72)
            prims.extend(
                _prod_articulated_truss_or_boom(
                    base_w=unic_base,
                    R_base_l2w=R,
                    mast_h=mast_u,
                    boom_len_1=min(2.6, u_bed_hl * 1.15),
                    boom_len_2=min(1.6, u_bed_hl * 0.72),
                    section_r=0.14,
                    slew_yaw_rad=0.12 * spec.boom_yaw_rad,
                    elev_1_rad=max(0.05, spec.boom_elev_rad),
                    elev_2_rad=-0.08,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=(0.68, 0.14, 0.06) if (spec.track_id % 2 == 0) else (0.66, 0.44, 0.05),
                    accent_rgb=(0.12, 0.13, 0.15),
                    roughness=0.30,
                    metallic=0.28,
                )
            )
            rb_role, rb_alb, rb_rough, rb_met, rb_coat = _rear_bumper_style_props()
            rb_c = _to_world(0.0, -hl + 0.30, chassis_z * 0.74)
            prims.append(
                _Prim3D(
                    center=(float(rb_c[0]), float(rb_c[1]), float(rb_c[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.86, 0.06, 0.09),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=rb_role,
                    albedo_rgb=rb_alb,
                    roughness=rb_rough,
                    metallic=rb_met,
                    clearcoat=rb_coat,
                )
            )
            rear_plate_local_y = -hl + 0.16
            rear_plate_local_z = chassis_z * 0.80

        elif is_articulated:
            # Articulated trailer rotating around fifth-wheel kingpin!
            kingpin_w = _to_world(0.0, 0.45, chassis_z + 0.25)
            yaw_trl = yaw + spec.trailer_yaw_delta_rad
            R_trl = _rot_zyx(yaw_trl, pitch * 0.5, roll * 0.5)
            RT_trl = R_trl.T.copy()
            up_trl = R_trl[:, 2].copy()
            trl_hl = hl * 0.72
            trl_hw_eff = hw * (1.02 if arch == "heavy_lowboy_trailer" else 0.99)
            trl_center_w = kingpin_w + R_trl @ np.array([0.0, -trl_hl + 0.55, 0.0], dtype=np.float32)

            if arch == "semi_container_trailer":
                box_c = trl_center_w + np.array([0.0, 0.0, 1.25], dtype=np.float32)
                prims.append(
                    _Prim3D(
                        center=(float(box_c[0]), float(box_c[1]), float(box_c[2])),
                        rot_w2l=RT_trl,
                        half_ext=(hw * 0.99, trl_hl, 1.22),
                        shape=_SHAPE_OBB,
                        veh_idx=veh_idx,
                        role=_ROLE_BODY,
                        albedo_rgb=spec.secondary_lin,
                        roughness=0.40,
                        rib_freq=14.0,
                        mud_level=spec.mud_level * 0.7,
                        mud_phase=spec.mud_phase,
                        metallic=0.24,
                        clearcoat=0.22,
                        normal_undul_amp=0.025,
                        decal_spec=((dec_i + 6) % n_dec_bank, 0, 0.10, 0.15, 0.36, 0.28, dec_rel),
                    )
                )
            elif arch == "skeletal_chassis_trailer":
                for side_s in (-1.0, 1.0):
                    ib_c = trl_center_w + R_trl @ np.array([side_s * 0.50, 0.0, 0.04], dtype=np.float32)
                    prims.append(
                        _Prim3D(
                            center=(float(ib_c[0]), float(ib_c[1]), float(ib_c[2])),
                            rot_w2l=RT_trl,
                            half_ext=(0.09, trl_hl * 0.98, 0.16),
                            shape=_SHAPE_OBB,
                            veh_idx=veh_idx,
                            role=_ROLE_DARK_CHASSIS,
                            albedo_rgb=(0.040, 0.044, 0.050),
                            roughness=0.68,
                            mud_level=spec.mud_level * 0.85,
                            mud_phase=spec.mud_phase,
                            metallic=0.28,
                        )
                    )
                for by_b in (trl_hl - 0.12, 0.0, -trl_hl + 0.12):
                    bol_c = trl_center_w + R_trl @ np.array([0.0, by_b, 0.10], dtype=np.float32)
                    prims.append(
                        _Prim3D(
                            center=(float(bol_c[0]), float(bol_c[1]), float(bol_c[2])),
                            rot_w2l=RT_trl,
                            half_ext=(hw * 0.99, 0.11, 0.10),
                            shape=_SHAPE_OBB,
                            veh_idx=veh_idx,
                            role=_ROLE_DARK_CHASSIS,
                            albedo_rgb=(0.055, 0.060, 0.068),
                            roughness=0.62,
                            metallic=0.28,
                        )
                    )
                for by_t in (trl_hl - 0.12, -trl_hl + 0.12):
                    for side_s in (-1.0, 1.0):
                        tw_c = trl_center_w + R_trl @ np.array([side_s * (hw * 0.95), by_t, 0.24], dtype=np.float32)
                        prims.append(
                            _Prim3D(
                                center=(float(tw_c[0]), float(tw_c[1]), float(tw_c[2])),
                                rot_w2l=RT_trl,
                                half_ext=(0.08, 0.09, 0.07),
                                shape=_SHAPE_OBB,
                                veh_idx=veh_idx,
                                role=_ROLE_CARGO_MATTE,
                                albedo_rgb=(0.56, 0.34, 0.05),
                                roughness=0.46,
                                metallic=0.25,
                            )
                        )
                zb_t = trl_center_w + R_trl @ np.array([0.0, -trl_hl - 0.01, -0.02], dtype=np.float32)
                prims.append(
                    _Prim3D(
                        center=(float(zb_t[0]), float(zb_t[1]), float(zb_t[2])),
                        rot_w2l=RT_trl,
                        half_ext=(hw * 0.92, 0.02, 0.04),
                        shape=_SHAPE_OBB,
                        veh_idx=veh_idx,
                        role=_ROLE_CHROME_METAL,
                        albedo_rgb=(0.80, 0.22, 0.16),
                        roughness=0.20,
                        rib_freq=18.0,
                        metallic=0.35,
                        retro_gain=0.90,
                    )
                )
            else:
                deck_c = trl_center_w + np.array([0.0, 0.0, -0.15], dtype=np.float32)
                prims.append(
                    _Prim3D(
                        center=(float(deck_c[0]), float(deck_c[1]), float(deck_c[2])),
                        rot_w2l=RT_trl,
                        half_ext=(hw * 1.02, trl_hl, 0.18),
                        shape=_SHAPE_OBB,
                        veh_idx=veh_idx,
                        role=_ROLE_BODY,
                        albedo_rgb=spec.secondary_lin,
                        roughness=0.50,
                        metallic=0.22,
                    )
                )
                exc_c = trl_center_w + np.array([0.0, -0.2, 0.78], dtype=np.float32)
                prims.append(
                    _Prim3D(
                        center=(float(exc_c[0]), float(exc_c[1]), float(exc_c[2])),
                        rot_w2l=RT_trl,
                        half_ext=(hw * 0.84, trl_hl * 0.52, 0.74),
                        shape=_SHAPE_OBB,
                        veh_idx=veh_idx,
                        role=_ROLE_CARGO_MATTE,
                        albedo_rgb=(0.58, 0.36, 0.05),
                        roughness=0.42,
                        clearcoat=0.28,
                    )
                )
            # Rear trailer tandem wheels with D_N radial apertures
            for ay_t in (-trl_hl + 1.35, -trl_hl + 0.45):
                for side_s in (-1.0, 1.0):
                    wc_t = trl_center_w + R_trl @ np.array([side_s * (hw - 0.16), ay_t, -0.25], dtype=np.float32)
                    prims.append(
                        _Prim3D(
                            center=(float(wc_t[0]), float(wc_t[1]), max(wheel_r, float(wc_t[2]))),
                            rot_w2l=RT_trl,
                            half_ext=(0.16, wheel_r, wheel_r),
                            shape=_SHAPE_CYLINDER_X,
                            veh_idx=veh_idx,
                            role=_ROLE_TIRE,
                            albedo_rgb=(0.035, 0.038, 0.042),
                            roughness=0.82,
                            metallic=0.14,
                            radial_n=dn_wheel_n,
                        )
                    )
            # Rear plate on articulated trailer back + 3D CSG recessed cavity (zero analytic umbra!)
            rp_meta = bank.plate_meta[spec.rear_plate_idx]
            rp_pw, rp_ph = (0.22, 0.11) if rp_meta["is_large"] else (0.165, 0.0825)
            rp_loc_trl = (float(spec.plate_offset_x_rear), float(-trl_hl - 0.04), -0.18)
            rp_c = trl_center_w + R_trl @ np.array(rp_loc_trl, dtype=np.float32)
            rp_n = R_trl @ np.array([0.0, -1.0, -0.04], dtype=np.float32)
            prims.extend(
                _prod_csg_recess_cavity(
                    pos_w=trl_center_w,
                    R_l2w=R_trl,
                    center_l=rp_loc_trl,
                    half_w=rp_pw,
                    half_h=rp_ph,
                    outward_sign_y=-1.0,
                    top_lip_out_m=0.017,
                    cheek_out_m=0.006,
                    inward_depth_m=0.085,
                    top_clear_z_m=0.102,
                    cheek_clear_x_m=0.038,
                    veh_idx=veh_idx,
                    role=_ROLE_DARK_CHASSIS,
                    albedo_rgb=(0.05, 0.055, 0.06),
                )
            )
            quads.append(
                _make_planar_quad(
                    rp_c,
                    rp_n,
                    rp_pw,
                    rp_ph,
                    ("plate_rear", spec.rear_plate_idx),
                    CLS_PLATE,
                    veh_idx,
                    up_hint_w=up_trl,
                    plate_side=SIDE_REAR,
                    retro_gain=float(rp_meta["retro_gain"]),
                    is_green_plate=bool(rp_meta["is_green"]),
                    is_large_plate=bool(rp_meta["is_large"]),
                    plate_text_str=str(rp_meta["plate_text_str"]),
                    plate_type_str=str(rp_meta["plate_type_str"]),
                    roughness=0.26,
                    clearcoat=0.38,
                    specular_f0=0.04,
                    normal_undul_amp=0.024,
                )
            )
            rear_plate_local_y = 999.0  # already placed on articulated trailer

            # Exact world Z=0 dual-body footprint ([tractor_FL, FR, RR, RL, trailer_FL, FR, RR, RL])!
            R_yaw_tr = _rot_z(yaw)
            R_yaw_tl = _rot_z(yaw_trl)
            p_xy0 = np.array([pos[0], pos[1], 0.0], dtype=np.float32)
            kp_xy0 = p_xy0 + R_yaw_tr @ np.array([0.0, 0.45, 0.0], dtype=np.float32)
            tl_xy0 = kp_xy0 + R_yaw_tl @ np.array([0.0, -trl_hl + 0.55, 0.0], dtype=np.float32)
            trac_y_rear = 0.0
            trac_4 = np.stack(
                [
                    p_xy0 + R_yaw_tr @ np.array([-hw, hl, 0.0], dtype=np.float32),
                    p_xy0 + R_yaw_tr @ np.array([hw, hl, 0.0], dtype=np.float32),
                    p_xy0 + R_yaw_tr @ np.array([hw, trac_y_rear, 0.0], dtype=np.float32),
                    p_xy0 + R_yaw_tr @ np.array([-hw, trac_y_rear, 0.0], dtype=np.float32),
                ],
                axis=0,
            )
            trl_4 = np.stack(
                [
                    tl_xy0 + R_yaw_tl @ np.array([-trl_hw_eff, trl_hl, 0.0], dtype=np.float32),
                    tl_xy0 + R_yaw_tl @ np.array([trl_hw_eff, trl_hl, 0.0], dtype=np.float32),
                    tl_xy0 + R_yaw_tl @ np.array([trl_hw_eff, -trl_hl, 0.0], dtype=np.float32),
                    tl_xy0 + R_yaw_tl @ np.array([-trl_hw_eff, -trl_hl, 0.0], dtype=np.float32),
                ],
                axis=0,
            )
            articulated_fp_w = np.concatenate([trac_4, trl_4], axis=0)  # (8, 3) all with Z == 0.0!

        elif arch == "box_wing_truck":
            box_h = spec.body_h_m - chassis_z - 0.18
            bc = _to_world(0.0, bed_cy, chassis_z + 0.18 + box_h * 0.5)
            box_decal = ((dec_i + 7) % n_dec_bank, 0, -0.05, 0.12, 0.42, 0.28, dec_rel)
            prims.append(
                _Prim3D(
                    center=(float(bc[0]), float(bc[1]), float(bc[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.99, bed_hl, box_h * 0.5),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY if spec.is_smooth_frp_box else _ROLE_CHROME_METAL,
                    albedo_rgb=spec.secondary_lin,
                    roughness=0.18 if spec.is_smooth_frp_box else 0.26,
                    rib_freq=0.0 if spec.is_smooth_frp_box else 16.0,
                    mud_level=spec.mud_level * 0.65,
                    mud_phase=spec.mud_phase,
                    metallic=0.06 if spec.is_smooth_frp_box else 0.72,
                    clearcoat=0.68 if spec.is_smooth_frp_box else 0.30,
                    normal_undul_amp=0.022,
                    decal_spec=box_decal,
                )
            )
            box_top_z = chassis_z + 0.18 + box_h
            if spec.has_wind_deflector:
                defl_h = max(0.24, min(0.56, (box_top_z - spec.cab_h_m) * 0.88))
                defl_y = cab_cy + (0.14 if spec.has_reefer_nose else 0.02)
                defl_hl = cab_len * (0.28 if spec.has_reefer_nose else 0.40)
                df_c = _to_world(0.0, defl_y, spec.cab_h_m + defl_h * 0.5)
                prims.append(
                    _Prim3D(
                        center=(float(df_c[0]), float(df_c[1]), float(df_c[2])),
                        rot_w2l=RT,
                        half_ext=(cab_hw * 0.88, defl_hl, defl_h * 0.5),
                        shape=_SHAPE_SLOPED_CAB,
                        veh_idx=veh_idx,
                        role=_ROLE_BODY,
                        albedo_rgb=spec.paint_lin,
                        roughness=0.22,
                        slope_param=0.52,
                        metallic=spec.paint_metallic,
                        clearcoat=spec.paint_clearcoat,
                    )
                )
            if spec.has_reefer_nose:
                rf_hz = min(0.26, max(0.16, (box_top_z - spec.cab_h_m) * 0.45))
                rf_z = max(spec.cab_h_m + rf_hz + 0.03, box_top_z - rf_hz - 0.06)
                rf_c = _to_world(0.0, bed_y1 + 0.18, rf_z)
                prims.append(
                    _Prim3D(
                        center=(float(rf_c[0]), float(rf_c[1]), float(rf_c[2])),
                        rot_w2l=RT,
                        half_ext=(hw * 0.56, 0.18, rf_hz),
                        shape=_SHAPE_OBB,
                        veh_idx=veh_idx,
                        role=_ROLE_BODY,
                        albedo_rgb=(0.88, 0.90, 0.92),
                        roughness=0.30,
                        rib_freq=14.0,
                        clearcoat=0.45,
                    )
                )
            rb_role, rb_alb, rb_rough, rb_met, rb_coat = _rear_bumper_style_props()
            rb_c = _to_world(0.0, -hl + 0.34, chassis_z * 0.72)
            prims.append(
                _Prim3D(
                    center=(float(rb_c[0]), float(rb_c[1]), float(rb_c[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.88, 0.07, 0.10),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=rb_role,
                    albedo_rgb=rb_alb,
                    roughness=rb_rough,
                    metallic=rb_met,
                    clearcoat=rb_coat,
                )
            )
            rear_plate_local_y = -hl + 0.20
            rear_plate_local_z = chassis_z * 0.76

        else:
            # Flatbed cargo truck & Kei-truck (drop-side bed + compositional flexural wave sheet / lattice rack / boom!)
            gate_h = 0.32 if arch == "kei_truck" else (0.38 if is_med_truck else 0.46)
            bc = _to_world(0.0, bed_cy, chassis_z + 0.12 + gate_h * 0.5)
            prims.append(
                _Prim3D(
                    center=(float(bc[0]), float(bc[1]), float(bc[2])),
                    rot_w2l=RT,
                    half_ext=(hw * 0.98, bed_hl, gate_h * 0.5),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_BODY,
                    albedo_rgb=spec.secondary_lin,
                    roughness=0.40,
                    rib_freq=6.0,
                    mud_level=spec.mud_level,
                    mud_phase=spec.mud_phase,
                    metallic=0.16,
                    clearcoat=0.30,
                    decal_spec=((dec_i + 8) % n_dec_bank, 0, 0.0, 0.0, 0.48, 0.36, dec_rel),
                )
            )
            if arch == "flatbed_truck":
                if spec.has_flex_sheet or spec.veh_struct_prod == "flex_wave_sheet":
                    # Multi-octave flexural wave sheet cover over cargo (`_SHAPE_FLEX_SHEET`)
                    cargo_c = _to_world(0.0, bed_cy + 0.10, chassis_z + 0.12 + gate_h + 0.28)
                    R_sh = R @ _rot_zyx(0.0, 0.0, 0.5 * math.pi)
                    prims.append(
                        _prod_flex_wave_sheet(
                            center_w=cargo_c,
                            R_l2w=R_sh,
                            half_ext=(0.12, bed_hl * 0.80, hw * 0.84),
                            veh_idx=veh_idx,
                            role=_ROLE_CARGO_MATTE,
                            albedo_rgb=spec.flex_sheet_lin,
                            roughness=0.56,
                            wave_amp=0.055,
                            wave_freq=4.4,
                            phase=spec.mud_phase,
                        )
                    )
                elif spec.veh_struct_prod == "articulated_boom_arm":
                    # Compositional light utility jib + Z^2 side stake rack on flatbed
                    jib_base = _to_world(0.0, bed_y1 - 0.28, chassis_z + 0.12 + gate_h)
                    prims.extend(
                        _prod_articulated_truss_or_boom(
                            base_w=jib_base,
                            R_base_l2w=R,
                            mast_h=0.68,
                            boom_len_1=min(1.85, bed_hl * 0.95),
                            boom_len_2=min(1.15, bed_hl * 0.60),
                            section_r=0.11,
                            slew_yaw_rad=0.10 * spec.boom_yaw_rad,
                            elev_1_rad=max(0.06, spec.boom_elev_rad),
                            elev_2_rad=-0.06,
                            veh_idx=veh_idx,
                            role=_ROLE_BODY,
                            albedo_rgb=(0.64, 0.40, 0.06),
                            accent_rgb=(0.14, 0.15, 0.17),
                            roughness=0.34,
                        )
                    )
                else:
                    cargo_c = _to_world(0.0, bed_cy + 0.15, chassis_z + 0.12 + gate_h + 0.34)
                    prims.append(
                        _Prim3D(
                            center=(float(cargo_c[0]), float(cargo_c[1]), float(cargo_c[2])),
                            rot_w2l=RT,
                            half_ext=(hw * 0.84, bed_hl * 0.78, 0.34),
                            shape=_SHAPE_OBB,
                            veh_idx=veh_idx,
                            role=_ROLE_CARGO_MATTE,
                            albedo_rgb=spec.flex_sheet_lin,
                            roughness=0.64,
                            normal_undul_amp=0.032,
                        )
                    )
                # Periodic Z^2 side stake lattice along flatbed rails
                for side_s in (-1.0, 1.0):
                    rk_w = _to_world(side_s * (hw * 0.96), bed_cy, chassis_z + 0.12 + gate_h + 0.20)
                    prims.extend(
                        _prod_linear_lattice_z2(
                            origin_w=rk_w,
                            R_l2w=R,
                            half_span_y=bed_hl * 0.88,
                            half_span_z=0.20,
                            bar_thick_x=0.018,
                            n_rails=2,
                            n_mullions=5,
                            veh_idx=veh_idx,
                            role=_ROLE_DARK_CHASSIS,
                            albedo_rgb=(0.24, 0.26, 0.28),
                        )
                    )
                rb_role, rb_alb, rb_rough, rb_met, rb_coat = _rear_bumper_style_props()
                rb_c = _to_world(0.0, -hl + 0.30, chassis_z * 0.74)
                prims.append(
                    _Prim3D(
                        center=(float(rb_c[0]), float(rb_c[1]), float(rb_c[2])),
                        rot_w2l=RT,
                        half_ext=(hw * 0.86, 0.06, 0.09),
                        shape=_SHAPE_OBB,
                        veh_idx=veh_idx,
                        role=rb_role,
                        albedo_rgb=rb_alb,
                        roughness=rb_rough,
                        metallic=rb_met,
                        clearcoat=rb_coat,
                    )
                )
            elif arch == "kei_truck" and (spec.has_flex_sheet or spec.veh_struct_prod == "flex_wave_sheet"):
                sh_k = _to_world(0.0, bed_cy, chassis_z + 0.12 + gate_h + 0.06)
                R_sh = R @ _rot_zyx(0.0, 0.0, 0.5 * math.pi)
                prims.append(
                    _prod_flex_wave_sheet(
                        center_w=sh_k,
                        R_l2w=R_sh,
                        half_ext=(0.06, bed_hl * 0.86, hw * 0.88),
                        veh_idx=veh_idx,
                        role=_ROLE_CARGO_MATTE,
                        albedo_rgb=spec.flex_sheet_lin,
                        roughness=0.58,
                        wave_amp=0.036,
                        wave_freq=5.6,
                        phase=spec.mud_phase,
                    )
                )
            rear_plate_local_y = -hl + (0.04 if arch == "kei_truck" else 0.16)
            rear_plate_local_z = chassis_z * 0.80

        # Side cab-door company/fleet decal or bolted non-plate bracket (`CLS_SIGNBOARD_HARDNEG`, `SIDE_PURE_SIDE`)
        if spec.has_side_door_decal and not is_rt_crane and len(bank.sticker_mips) > 16:
            dd_idx = int(spec.side_door_decal_idx) % len(bank.sticker_mips)
            dd_pw = min(0.24, cab_len * 0.34)
            dd_ph = min(0.095, cab_low_h * 0.32)
            for side_s in (-1.0, 1.0):
                dd_c = _to_world(side_s * (cab_hw + 0.006), cab_cy, cab_low_z)
                dd_n = R @ np.array([side_s, 0.0, 0.0], dtype=np.float32)
                quads.append(
                    _make_planar_quad(
                        dd_c,
                        dd_n,
                        dd_pw,
                        dd_ph,
                        ("sticker", dd_idx),
                        CLS_SIGNBOARD_HARDNEG,
                        veh_idx,
                        up_hint_w=up_body,
                        plate_side=SIDE_PURE_SIDE,
                        retro_gain=0.65,
                        roughness=0.26,
                        clearcoat=0.45,
                        specular_f0=0.04,
                        normal_undul_amp=0.018,
                    )
                )

    # Headlights & taillights (emissive in Linear HDR!)
    for side_s in (-1.0, 1.0):
        hl_c = _to_world(side_s * hw * 0.74, hl + 0.01, front_plate_local_z + 0.24)
        prims.append(
            _Prim3D(
                center=(float(hl_c[0]), float(hl_c[1]), float(hl_c[2])),
                rot_w2l=RT,
                half_ext=(0.14, 0.05, 0.09),
                shape=_SHAPE_OBB,
                veh_idx=veh_idx,
                role=_ROLE_HEADLIGHT,
                albedo_rgb=(0.95, 0.92, 0.85),
                roughness=0.10,
            )
        )
        if not is_articulated:
            tl_z_eff = taillight_local_z if arch in ("concrete_mixer", "tractor_head") else (rear_plate_local_z + 0.14)
            tl_c = _to_world(side_s * hw * 0.76, -hl - 0.01, tl_z_eff)
            prims.append(
                _Prim3D(
                    center=(float(tl_c[0]), float(tl_c[1]), float(tl_c[2])),
                    rot_w2l=RT,
                    half_ext=(0.15, 0.04, 0.07),
                    shape=_SHAPE_OBB,
                    veh_idx=veh_idx,
                    role=_ROLE_TAILLIGHT,
                    albedo_rgb=(0.88, 0.06, 0.04),
                    roughness=0.15,
                )
            )

    def _try_place_face_patch(
        boxes_xz: list[tuple[float, float, float, float]],
        cand_xs: tuple[float, ...],
        cz: float,
        pw: float,
        ph: float,
        max_x: float,
        margin: float = 0.065,
    ) -> float | None:
        for cx_cand in cand_xs:
            if abs(cx_cand) + pw > max_x:
                continue
            ok_p = True
            for bx0, bx1, bz0, bz1 in boxes_xz:
                if (cx_cand + pw + margin > bx0) and (cx_cand - pw - margin < bx1) and (cz + ph + margin > bz0) and (cz - ph - margin < bz1):
                    ok_p = False
                    break
            if ok_p:
                boxes_xz.append((cx_cand - pw, cx_cand + pw, cz - ph, cz + ph))
                return float(cx_cand)
        return None

    # 3. Front & Rear 3D Physical License Plates + Genuine 3D CSG Recessed Mounting Cavities!
    fp_meta = bank.plate_meta[spec.front_plate_idx]
    fp_pw, fp_ph = (0.22, 0.11) if fp_meta["is_large"] else (0.165, 0.0825)
    fp_lx = float(spec.plate_offset_x_front)
    front_boxes_xz: list[tuple[float, float, float, float]] = [
        (fp_lx - fp_pw, fp_lx + fp_pw, front_plate_local_z - fp_ph, front_plate_local_z + fp_ph)
    ]
    if is_heavy or spec.veh_struct_prod == "csg_recess_cavity":
        prims.extend(
            _prod_csg_recess_cavity(
                pos_w=pos,
                R_l2w=R,
                center_l=(fp_lx, front_plate_local_y, front_plate_local_z),
                half_w=fp_pw,
                half_h=fp_ph,
                outward_sign_y=1.0,
                top_lip_out_m=0.052 if spec.veh_struct_prod == "csg_recess_cavity" else 0.036,
                cheek_out_m=0.040 if spec.veh_struct_prod == "csg_recess_cavity" else 0.026,
                inward_depth_m=0.090,
                top_clear_z_m=0.030,
                cheek_clear_x_m=0.036,
                veh_idx=veh_idx,
                role=_ROLE_DARK_CHASSIS,
                albedo_rgb=(0.045, 0.048, 0.052),
            )
        )
    fp_c = _to_world(fp_lx, front_plate_local_y, front_plate_local_z)
    fp_n = R @ np.array([0.0, 1.0, -0.05], dtype=np.float32)  # slight downward bracket tilt
    quads.append(
        _make_planar_quad(
            fp_c,
            fp_n,
            fp_pw,
            fp_ph,
            ("plate_front", spec.front_plate_idx),
            CLS_PLATE,
            veh_idx,
            up_hint_w=up_body,
            plate_side=SIDE_FRONT,
            retro_gain=float(fp_meta["retro_gain"]),
            is_green_plate=bool(fp_meta["is_green"]),
            is_large_plate=bool(fp_meta["is_large"]),
            plate_text_str=str(fp_meta["plate_text_str"]),
            plate_type_str=str(fp_meta["plate_type_str"]),
            roughness=0.26,
            clearcoat=0.38,
            specular_f0=0.04,
            normal_undul_amp=0.024,
        )
    )

    if rear_plate_local_y < 500.0:
        rp_meta = bank.plate_meta[spec.rear_plate_idx]
        rp_pw, rp_ph = (0.22, 0.11) if rp_meta["is_large"] else (0.165, 0.0825)
        rp_x = float(rear_plate_local_x if rear_plate_local_x is not None else spec.plate_offset_x_rear)
        rear_boxes_xz: list[tuple[float, float, float, float]] = [
            (rp_x - rp_pw, rp_x + rp_pw, rear_plate_local_z - rp_ph, rear_plate_local_z + rp_ph)
        ]
        if is_heavy or spec.veh_struct_prod == "csg_recess_cavity":
            prims.extend(
                _prod_csg_recess_cavity(
                    pos_w=pos,
                    R_l2w=R,
                    center_l=(rp_x, rear_plate_local_y, rear_plate_local_z),
                    half_w=rp_pw,
                    half_h=rp_ph,
                    outward_sign_y=-1.0,
                    top_lip_out_m=0.017,
                    cheek_out_m=0.006,
                    inward_depth_m=0.095,
                    top_clear_z_m=0.102,
                    cheek_clear_x_m=0.038,
                    veh_idx=veh_idx,
                    role=_ROLE_DARK_CHASSIS,
                    albedo_rgb=(0.045, 0.048, 0.052),
                )
            )
        rp_c = _to_world(rp_x, rear_plate_local_y, rear_plate_local_z)
        rp_n = R @ np.array([0.0, -1.0, rear_plate_tilt_z], dtype=np.float32)
        quads.append(
            _make_planar_quad(
                rp_c,
                rp_n,
                rp_pw,
                rp_ph,
                ("plate_rear", spec.rear_plate_idx),
                CLS_PLATE,
                veh_idx,
                up_hint_w=up_body,
                plate_side=SIDE_REAR,
                retro_gain=float(rp_meta["retro_gain"]),
                is_green_plate=bool(rp_meta["is_green"]),
                is_large_plate=bool(rp_meta["is_large"]),
                plate_text_str=str(rp_meta["plate_text_str"]),
                plate_type_str=str(rp_meta["plate_type_str"]),
                roughness=0.26,
                clearcoat=0.38,
                specular_f0=0.04,
                normal_undul_amp=0.024,
            )
        )

        # Vehicle-mounted placards & bolted/framed blank non-plate brackets
        # (strictly disjoint in `(X, Z)` with `>= 6.5 cm` margin AND strictly recessed inward behind real plate planes!)
        if spec.has_sticker and arch != "tractor_head":
            is_pass_car = arch in ("sedan", "kei_tall_wagon", "suv_4wd", "hiace_van")
            st_pw = 0.18 if arch == "kei_truck" else 0.20
            st_ph = 0.065
            if arch == "concrete_mixer":
                st_z = chassis_z + 0.68
                st_pref = (-hw * 0.52) if rp_x >= 0.0 else (hw * 0.52)
                st_cands: tuple[float, ...] = (st_pref, 0.0, -st_pref)
            elif is_pass_car:
                st_z = rear_plate_local_z + 0.24
                st_pref = (-hw * 0.52) if rp_x >= 0.0 else (hw * 0.52)
                st_cands = (st_pref, -st_pref, 0.0)
            else:
                st_z = rear_plate_local_z + 0.01
                st_s0 = -1.0 if rp_x >= 0.0 else 1.0
                st_cands = (st_s0 * hw * 0.60, -st_s0 * hw * 0.60, st_s0 * hw * 0.68, -st_s0 * hw * 0.68)
            st_x_ok = _try_place_face_patch(rear_boxes_xz, st_cands, st_z, st_pw, st_ph, hw - 0.03)
            if st_x_ok is not None:
                st_y = rear_plate_local_y + 0.024
                st_c = _to_world(st_x_ok, st_y, st_z)
                st_n = R @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
                quads.append(
                    _make_planar_quad(
                        st_c,
                        st_n,
                        st_pw,
                        st_ph,
                        ("sticker", int(spec.sticker_idx) % len(bank.sticker_mips)),
                        CLS_SIGNBOARD_HARDNEG,
                        veh_idx,
                        up_hint_w=up_body,
                        plate_side=SIDE_REAR,
                        retro_gain=0.85,
                        roughness=0.32,
                        clearcoat=0.25,
                        specular_f0=0.04,
                        normal_undul_amp=0.022,
                    )
                )
        if spec.has_hazard_sign:
            hz_z = front_plate_local_z + 0.02
            hz_cands = (-hw * 0.54, hw * 0.54, -hw * 0.68, hw * 0.68) if fp_lx >= 0.0 else (hw * 0.54, -hw * 0.54, hw * 0.68, -hw * 0.68)
            hz_x_ok = _try_place_face_patch(front_boxes_xz, hz_cands, hz_z, 0.15, 0.15, hw - 0.03)
            if hz_x_ok is not None:
                hz_c_f = _to_world(hz_x_ok, front_plate_local_y - 0.020, hz_z)
                hz_n_f = R @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
                quads.append(
                    _make_planar_quad(
                        hz_c_f,
                        hz_n_f,
                        0.15,
                        0.15,
                        ("sticker", 0),  # `[危]`
                        CLS_SIGNBOARD_HARDNEG,
                        veh_idx,
                        up_hint_w=up_body,
                        plate_side=SIDE_FRONT,
                        retro_gain=1.10,
                        roughness=0.28,
                        clearcoat=0.35,
                        specular_f0=0.04,
                    )
                )

    # Front cab stencils, grille badges, bolted blank brackets, and bumper intake slots (`CLS_SIGNBOARD_HARDNEG`)
    if arch not in ("sedan", "kei_tall_wagon", "suv_4wd", "hiace_van"):
        n_st_bank = len(bank.sticker_mips)
        if spec.has_front_cab_hn and n_st_bank > 7:
            cab_hn_i = int(spec.front_cab_hn_idx) % n_st_bank
            if cab_hn_i == 0:
                cab_hn_i = 7
            is_center_badge = cab_hn_i in (10, 11, 12)
            cab_pw = 0.23 if is_center_badge else 0.20
            cab_ph = 0.060 if is_center_badge else 0.072
            cab_hn_z = max(
                front_plate_local_z + fp_ph + cab_ph + 0.08,
                chassis_z + (spec.cab_h_m - chassis_z) * (0.24 if is_center_badge else 0.34),
            )
            cab_pref = 0.0 if is_center_badge else ((hw * 0.56) if (spec.track_id % 2 == 0) else (-hw * 0.56))
            cab_cands = (0.0,) if is_center_badge else (cab_pref, -cab_pref, 0.0)
            cab_x_ok = _try_place_face_patch(front_boxes_xz, cab_cands, cab_hn_z, cab_pw, cab_ph, hw - 0.03)
            if cab_x_ok is not None:
                cab_hn_y = min(hl + 0.016, front_plate_local_y - 0.020)
                cab_hn_c = _to_world(cab_x_ok, cab_hn_y, cab_hn_z)
                cab_hn_n = R @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
                quads.append(
                    _make_planar_quad(
                        cab_hn_c,
                        cab_hn_n,
                        cab_pw,
                        cab_ph,
                        ("sticker", cab_hn_i),
                        CLS_SIGNBOARD_HARDNEG,
                        veh_idx,
                        up_hint_w=up_body,
                        plate_side=SIDE_FRONT,
                        retro_gain=0.75 if is_center_badge else 0.65,
                        roughness=0.28,
                        clearcoat=0.35,
                        specular_f0=0.04,
                    )
                )
        if spec.has_front_bumper_hn and n_st_bank > 13:
            bmp_hn_i = int(spec.front_bumper_hn_idx) % n_st_bank
            if bmp_hn_i == 0:
                bmp_hn_i = 13
            bmp_pw, bmp_ph = 0.155, 0.052
            bmp_z = front_plate_local_z - 0.015
            bmp_s0 = -1.0 if fp_lx > 0.0 else 1.0
            bmp_cands = (
                bmp_s0 * hw * 0.64,
                -bmp_s0 * hw * 0.64,
                bmp_s0 * hw * 0.74,
                -bmp_s0 * hw * 0.74,
            )
            bmp_x_ok = _try_place_face_patch(front_boxes_xz, bmp_cands, bmp_z, bmp_pw, bmp_ph, hw - 0.03)
            if bmp_x_ok is not None:
                bmp_hn_c = _to_world(bmp_x_ok, front_plate_local_y - 0.020, bmp_z)
                bmp_hn_n = R @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
                quads.append(
                    _make_planar_quad(
                        bmp_hn_c,
                        bmp_hn_n,
                        bmp_pw,
                        bmp_ph,
                        ("sticker", bmp_hn_i),
                        CLS_SIGNBOARD_HARDNEG,
                        veh_idx,
                        up_hint_w=up_body,
                        plate_side=SIDE_FRONT,
                        retro_gain=1.25 if bmp_hn_i == 13 else 0.45,
                        roughness=0.32,
                        clearcoat=0.28,
                        specular_f0=0.04,
                    )
                )

    # Exact world Z = 0.0 road-plane footprint [FL, FR, RR, RL] (never lifted by suspension pitch/roll!)
    R_yaw = _rot_z(yaw)
    p_ground = np.array([pos[0], pos[1], 0.0], dtype=np.float32)
    if articulated_fp_w is not None:
        # Tight oriented bounding rectangle on Z=0 along tractor yaw enclosing ALL 8 tractor + articulated trailer ground corners!
        rel_pts = (articulated_fp_w - p_ground[None, :]) @ R_yaw  # (8, 3) in tractor planar frame, Z==0
        xmin_f = float(rel_pts[:, 0].min())
        xmax_f = float(rel_pts[:, 0].max())
        ymin_f = float(rel_pts[:, 1].min())
        ymax_f = float(rel_pts[:, 1].max())
        footprint_w = np.stack(
            [
                p_ground + R_yaw @ np.array([xmin_f, ymax_f, 0.0], dtype=np.float32),
                p_ground + R_yaw @ np.array([xmax_f, ymax_f, 0.0], dtype=np.float32),
                p_ground + R_yaw @ np.array([xmax_f, ymin_f, 0.0], dtype=np.float32),
                p_ground + R_yaw @ np.array([xmin_f, ymin_f, 0.0], dtype=np.float32),
            ],
            axis=0,
        )
    else:
        footprint_w = np.stack(
            [
                p_ground + R_yaw @ np.array([-hw, hl, 0.0], dtype=np.float32),
                p_ground + R_yaw @ np.array([hw, hl, 0.0], dtype=np.float32),
                p_ground + R_yaw @ np.array([hw, -hl, 0.0], dtype=np.float32),
                p_ground + R_yaw @ np.array([-hw, -hl, 0.0], dtype=np.float32),
            ],
            axis=0,
        )
    footprint_w[:, 2] = 0.0
    return prims, quads, footprint_w, articulated_fp_w, cabin_obbs_w


_BOX_EDGES_12: tuple[tuple[int, int], ...] = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


class Scene3DGPUEngine:
    """High-throughput 3D multi-frame video clip synthesizer (`cuda` or `cpu`, deterministic by `seed`)."""

    def __init__(
        self,
        config: Scene3DClipConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config or Scene3DClipConfig()
        self.device = _resolve_torch_device(device)
        self.bank = _GPU3DTextureBank.get(
            self.device,
            bank_size=self.config.bank_size,
            seed=self.config.bank_seed,
            split=self.config.split,
            domain_split=self.config.domain_split,
        )

    @staticmethod
    def _project_points(
        pts_w: np.ndarray,
        cam_o: np.ndarray,
        R_cam: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project `(N, 3)` world points to `(N, 2)` pixel-center `[u, v]` + `(N,)` depth `Z_c`."""
        R64 = R_cam.astype(np.float64)
        cam64 = cam_o.astype(np.float64)
        rel = (pts_w.astype(np.float64) - cam64[None, :]) @ R64  # (N, 3) -> [X_c, Y_c, Z_c]
        zc = rel[:, 2]
        safe_z = np.maximum(0.15, zc)
        fx64, fy64 = float(fx), float(fy)
        cx64, cy64 = float(cx), float(cy)
        u = fx64 * (rel[:, 0] / safe_z) + cx64
        v = fy64 * (rel[:, 1] / safe_z) + cy64
        uv_f32 = np.stack([u, v], axis=-1).astype(np.float32)
        zc_f32 = zc.astype(np.float32)

        # Directed 1-ULP inward rounding ONLY on near-plane-clipped Z=0 ground-footprint boundary vertices
        # (`abs(zc - 0.22) <= 5e-4`), so IEEE-754 float32 pixel rounding never places a clipped boundary vertex
        # `0.5-2 ULPs` below `Z_c = 0.22 m` while leaving every interior, OBB, and plate/sign vertex untouched!
        if pts_w.shape[0] > 0 and float(cam64[2]) > 0.05 and fx64 > 1.0 and fy64 > 1.0:
            near_mask = (pts_w[:, 2] == 0.0) & (zc >= 0.2195) & (zc <= 0.2205)
            if bool(np.any(near_mask)):
                from fractions import Fraction

                fx_q, fy_q = Fraction(fx64), Fraction(fy64)
                cx_q, cy_q = Fraction(cx64), Fraction(cy64)
                cam_z_f = float(cam64[2])
                cam_z_q = Fraction(cam_z_f)
                r00_q, r01_q, r02_q = Fraction(float(R64[0, 0])), Fraction(float(R64[0, 1])), Fraction(float(R64[0, 2]))
                r10_q, r11_q, r12_q = Fraction(float(R64[1, 0])), Fraction(float(R64[1, 1])), Fraction(float(R64[1, 2]))
                r20_q, r21_q, r22_q = Fraction(float(R64[2, 0])), Fraction(float(R64[2, 1])), Fraction(float(R64[2, 2]))
                zn_22 = max(Fraction(0.22), Fraction(11, 50))
                gu = float(R64[2, 0]) / fx64
                gv = float(R64[2, 1]) / fy64
                gc = float(R64[2, 2]) - gu * cx64 - gv * cy64
                ax_step = 0 if abs(gu) >= abs(gv) else 1
                dir_val = np.float32(np.inf if (gu if ax_step == 0 else gv) > 0.0 else -np.inf)

                for k_i in np.flatnonzero(near_mask):
                    # Only guard vertices whose 3D input was actually clipped at `z_near >= 0.22`
                    d3_64 = float((pts_w[k_i].astype(np.float64) - cam64) @ R64[:, 2])
                    if d3_64 < 0.22 - 2e-6:
                        continue
                    for _ in range(32):
                        uk_f, vk_f = float(uv_f32[k_i, 0]), float(uv_f32[k_i, 1])
                        uk_q, vk_q = Fraction(uk_f), Fraction(vk_f)
                        rx_q = (uk_q - cx_q) / fx_q
                        ry_q = (vk_q - cy_q) / fy_q
                        dwz_q = r20_q * rx_q + r21_q * ry_q + r22_q
                        dwx_q = r00_q * rx_q + r01_q * ry_q + r02_q
                        dwy_q = r10_q * rx_q + r11_q * ry_q + r12_q
                        dw_f = gu * uk_f + gv * vk_f + gc
                        if dwz_q < 0 and dw_f < 0.0:
                            tq = -cam_z_q / dwz_q
                            zdq = tq * (dwx_q * r02_q + dwy_q * r12_q) - cam_z_q * r22_q
                            tf = -cam_z_f / dw_f
                            rw_f = R64 @ np.array([(uk_f - cx64) / fx64, (vk_f - cy64) / fy64, 1.0], dtype=np.float64)
                            zdf = tf * float(rw_f[0] * R64[0, 2] + rw_f[1] * R64[1, 2]) - cam_z_f * float(R64[2, 2])
                            if tq >= zn_22 and zdq >= zn_22 and tf >= 0.22 and zdf >= 0.22:
                                zc_f32[k_i] = np.float32(max(float(zc_f32[k_i]), 0.22000003))
                                break
                        uv_f32[k_i, ax_step] = np.nextafter(uv_f32[k_i, ax_step], dir_val)
        return uv_f32, zc_f32

    @staticmethod
    def _clip_obb_to_near_plane_uv(
        center_w: np.ndarray,
        R_l2w: np.ndarray,
        half_ext: tuple[float, float, float],
        cam_o: np.ndarray,
        R_cam: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        z_near: float = 0.16,
    ) -> np.ndarray:
        """Exact 3D near-plane (`Z_c >= z_near`) clipping of a convex 3D OBB before pinhole projection!"""
        sx, sy, sz = half_ext
        offs = np.array(
            [
                [-sx, -sy, -sz], [sx, -sy, -sz], [-sx, sy, -sz], [sx, sy, -sz],
                [-sx, -sy, sz], [sx, -sy, sz], [-sx, sy, sz], [sx, sy, sz],
            ],
            dtype=np.float32,
        )
        pts_w = center_w[None, :] + offs @ R_l2w.T
        rel = (pts_w - cam_o[None, :]) @ R_cam  # (8, 3)
        z = rel[:, 2]
        if np.all(z < z_near):
            return np.zeros((0, 2), dtype=np.float32)
        valid_cam: list[np.ndarray] = [rel[i] for i in range(8) if z[i] >= z_near]
        if np.any(z < z_near):
            for i0, i1 in _BOX_EDGES_12:
                z0, z1 = float(z[i0]), float(z[i1])
                if (z0 < z_near) != (z1 < z_near):
                    alpha = (z_near - z0) / (z1 - z0)
                    valid_cam.append(rel[i0] + alpha * (rel[i1] - rel[i0]))
        pts_c = np.stack(valid_cam, axis=0)
        u = fx * (pts_c[:, 0] / pts_c[:, 2]) + cx
        v = fy * (pts_c[:, 1] / pts_c[:, 2]) + cy
        return np.stack([u, v], axis=-1).astype(np.float32)

    @staticmethod
    def _clip_quad_to_near_plane_uv(
        qd: _PlanarQuad3D,
        cam_o: np.ndarray,
        R_cam: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        z_near: float = 0.16,
    ) -> np.ndarray:
        c4 = np.stack(
            [
                qd.center - qd.half_w * qd.u_axis - qd.half_h * qd.v_axis,
                qd.center + qd.half_w * qd.u_axis - qd.half_h * qd.v_axis,
                qd.center + qd.half_w * qd.u_axis + qd.half_h * qd.v_axis,
                qd.center - qd.half_w * qd.u_axis + qd.half_h * qd.v_axis,
            ],
            axis=0,
        )
        rel = (c4 - cam_o[None, :]) @ R_cam
        z = rel[:, 2]
        if np.all(z < z_near):
            return np.zeros((0, 2), dtype=np.float32)
        out_c: list[np.ndarray] = []
        for i in range(4):
            j = (i + 1) % 4
            if z[i] >= z_near:
                out_c.append(rel[i])
            if (z[i] < z_near) != (z[j] < z_near):
                alpha = float((z_near - z[i]) / (z[j] - z[i]))
                out_c.append(rel[i] + alpha * (rel[j] - rel[i]))
        if not out_c:
            return np.zeros((0, 2), dtype=np.float32)
        pts_c = np.stack(out_c, axis=0)
        u = fx * (pts_c[:, 0] / pts_c[:, 2]) + cx
        v = fy * (pts_c[:, 1] / pts_c[:, 2]) + cy
        return np.stack([u, v], axis=-1).astype(np.float32)

    def _sample_vehicle_spec(
        self,
        rng: random.Random,
        track_id: int,
        lane_x: float,
        y_range: tuple[float, float],
        stationary: bool,
        *,
        veh_struct_prod: str | None = None,
        marking_prod: str | None = None,
        radial_dn_n: int | None = None,
    ) -> _VehiclePassSpec:
        names = [a[0] for a in ARCHETYPE_CATALOG]
        subs = [a[1] for a in ARCHETYPE_CATALOG]
        weights = [a[2] for a in ARCHETYPE_CATALOG]
        idx_a = rng.choices(range(len(names)), weights=weights, k=1)[0]
        arch = names[idx_a]
        subtype = subs[idx_a]

        is_lg = True
        if arch in ("dump_10t", "deep_dump"):
            L, W, cab_h, body_h = rng.uniform(7.4, 8.8), rng.uniform(2.44, 2.50), rng.uniform(2.85, 3.25), rng.uniform(3.1, 3.6)
            p_bucket = "heavy_comm" if rng.random() < 0.76 else "heavy_priv"
        elif arch == "concrete_mixer":
            L, W, cab_h, body_h = rng.uniform(7.6, 8.4), 2.48, rng.uniform(2.85, 3.15), 3.65
            p_bucket = "heavy_comm" if rng.random() < 0.65 else ("special_8_9" if rng.random() < 0.5 else "heavy_priv")
        elif arch == "tanker_lorry":
            L, W, cab_h, body_h = rng.uniform(7.8, 9.4), 2.49, 2.92, 3.25
            p_bucket = "heavy_comm" if rng.random() < 0.80 else "special_8_9"
        elif arch == "rough_terrain_crane":
            L, W, cab_h, body_h = rng.uniform(8.2, 10.6), rng.uniform(2.48, 2.62), 2.95, 3.55
            p_bucket = "special_8_9"
        elif arch == "cargo_crane_truck":
            is_lg = rng.random() < 0.62
            L = rng.uniform(7.6, 9.2) if is_lg else rng.uniform(5.6, 6.8)
            W = 2.48 if is_lg else 2.18
            cab_h, body_h = (2.92 if is_lg else 2.52), (3.18 if is_lg else 2.78)
            p_bucket = ("heavy_comm" if rng.random() < 0.72 else "special_8_9") if is_lg else ("med_comm" if rng.random() < 0.68 else "med_priv")
        elif arch == "tractor_head":
            L, W, cab_h = rng.uniform(5.8, 6.8), 2.49, rng.uniform(3.05, 3.42)
            body_h = cab_h
            p_bucket = "heavy_comm" if rng.random() < 0.88 else "heavy_priv"
        elif arch == "skeletal_chassis_trailer":
            L, W, cab_h, body_h = rng.uniform(11.5, 13.8), 2.49, 3.22, 3.22
            p_bucket = "heavy_comm"
        elif arch in ("semi_container_trailer", "heavy_lowboy_trailer"):
            L, W, cab_h, body_h = rng.uniform(11.2, 13.8), 2.49, 3.25, 3.72
            p_bucket = "heavy_comm"
        elif arch == "box_wing_truck":
            is_lg = rng.random() < 0.58
            L = rng.uniform(7.8, 9.6) if is_lg else rng.uniform(5.8, 6.8)
            W = 2.48 if is_lg else 2.18
            cab_h, body_h = (2.95 if is_lg else 2.55), (3.55 if is_lg else 3.05)
            p_bucket = ("heavy_comm" if rng.random() < 0.78 else "heavy_priv") if is_lg else ("med_comm" if rng.random() < 0.70 else "med_priv")
        elif arch == "flatbed_truck":
            is_lg = rng.random() < 0.42
            L = rng.uniform(7.2, 8.6) if is_lg else rng.uniform(4.7, 6.2)
            W = 2.45 if is_lg else 1.88
            cab_h, body_h = (2.75 if is_lg else 2.18), (2.75 if is_lg else 2.18)
            p_bucket = ("heavy_comm" if rng.random() < 0.65 else "heavy_priv") if is_lg else ("med_priv" if rng.random() < 0.55 else "med_comm")
        elif arch == "kei_truck":
            is_lg = False
            L, W, cab_h, body_h = 3.39, 1.48, 1.78, 1.78
            p_bucket = "kei_priv" if rng.random() < 0.72 else "kei_comm"
        elif arch == "hiace_van":
            is_lg = False
            L, W, cab_h, body_h = rng.uniform(4.69, 5.38), rng.uniform(1.69, 1.88), 1.98, 2.08
            p_bucket = "med_priv" if rng.random() < 0.68 else "med_comm"
        elif arch == "kei_tall_wagon":
            is_lg = False
            L, W, cab_h, body_h = 3.39, 1.48, 1.76, 1.78
            p_bucket = "kei_priv" if rng.random() < 0.85 else "kei_comm"
        elif arch == "suv_4wd":
            is_lg = False
            L, W, cab_h, body_h = rng.uniform(4.55, 4.95), rng.uniform(1.82, 1.94), 1.76, 1.80
            p_bucket = "med_priv"
        else:
            is_lg = False
            L, W, cab_h, body_h = rng.uniform(4.45, 4.88), rng.uniform(1.74, 1.84), 1.46, 1.48
            p_bucket = "med_priv"

        palettes: tuple[tuple[float, float, float], ...] = (
            (0.88, 0.89, 0.91),  # Isuzu / Hino commercial bright white
            (0.92, 0.93, 0.95),  # Refrigerated logistics pure white
            (0.82, 0.80, 0.70),  # Japanese fleet cream / ivory
            (0.72, 0.74, 0.76),  # Metallic silver / aluminum wing
            (0.38, 0.41, 0.45),  # Industrial slate gray
            (0.42, 0.52, 0.46),  # Fuso street mixer mint-olive (`w3/08`)
            (0.05, 0.32, 0.26),  # Yamato logistics teal-green (`cg/08`)
            (0.06, 0.22, 0.56),  # Sagawa / municipal cobalt blue (`w3/01`)
            (0.04, 0.14, 0.34),  # Deep navy fleet blue
            (0.05, 0.24, 0.14),  # Construction forest green
            (0.62, 0.38, 0.04),  # Urban mixer / roadwork ochre yellow (`cg/01`)
            (0.04, 0.04, 0.05),  # Snow dump / heavy chassis jet black (`w3/03`)
            (0.54, 0.05, 0.04),  # Emergency / heavy support crimson (`cg/06`)
            (0.36, 0.05, 0.04),  # Maroon / burgundy cab
            (0.12, 0.28, 0.36),  # Container cyan-steel (`w3/06`)
        )
        bright_cargo_palettes: tuple[tuple[float, float, float], ...] = (
            (0.92, 0.93, 0.95),
            (0.88, 0.89, 0.91),
            (0.82, 0.80, 0.70),
            (0.74, 0.76, 0.80),
        )
        paint_lin = rng.choice(palettes)
        if arch in ("box_wing_truck", "concrete_mixer") and rng.random() < 0.52:
            # High-frequency Japanese commercial two-tone: colored cab + bright white/silver box or mixer drum!
            sec_lin = rng.choice(bright_cargo_palettes)
        elif rng.random() < 0.44:
            sec_lin = paint_lin
        else:
            sec_lin = rng.choice(palettes)

        # Paired registration: rigid vehicles (including bobtail `tractor_head`) ALWAYS share exact registration (`fp_idx == rp_idx`)!
        # Articulated tractor-trailers have independent tractor front & semi-trailer rear commercial registrations.
        is_articulated = arch in _ARTICULATED_ARCHETYPES
        reg_list = self.bank.reg_buckets[p_bucket]
        fp_idx = rng.choice(reg_list)
        if is_articulated:
            rp_idx = rng.choice(reg_list)
            if len(reg_list) > 1 and rp_idx == fp_idx:
                rp_idx = reg_list[(reg_list.index(fp_idx) + 1) % len(reg_list)]
        else:
            rp_idx = fp_idx

        approaching = rng.random() < 0.52
        dist_mid = 0.5 * (y_range[0] + y_range[1])
        max_oblique = 36.0 if dist_mid < 13.0 else 18.0
        oblique_deg = rng.uniform(-max_oblique, max_oblique) if rng.random() < 0.55 else rng.uniform(-10.0, 10.0)
        base_yaw = (math.pi if approaching else 0.0) + math.radians(oblique_deg)

        y0 = rng.uniform(y_range[0], y_range[1])
        if stationary:
            motion_mode = "stationary"
            speed = 0.0
            accel_mag = 0.0
            yaw_rate = 0.0
        else:
            motion_mode = rng.choices(
                ["cruise", "gate_creep", "brake_to_stop", "stop_and_go", "reverse"],
                weights=[0.54, 0.08, 0.16, 0.12, 0.10],
                k=1,
            )[0]
            if motion_mode == "gate_creep":
                motion_mode = "cruise"
                speed = rng.uniform(0.12, 0.85)
                accel_mag = 0.0
                yaw_rate = math.radians(rng.uniform(-4.0, 4.0)) if rng.random() < 0.3 else 0.0
            elif motion_mode == "reverse":
                speed = -rng.uniform(0.9, 3.2)  # backing into construction bay!
                accel_mag = rng.uniform(-0.8, 0.4)
                yaw_rate = math.radians(rng.uniform(-10.0, 10.0))
            elif motion_mode == "stop_and_go":
                speed = rng.uniform(2.2, 5.2)
                accel_mag = 0.0
                yaw_rate = math.radians(rng.uniform(-9.0, 9.0))
            elif motion_mode == "brake_to_stop":
                speed = rng.uniform(3.0, 9.5)
                accel_mag = -rng.uniform(2.5, 5.5)
                yaw_rate = math.radians(rng.uniform(-12.0, 12.0)) if rng.random() < 0.4 else 0.0
            else:
                speed = rng.uniform(2.8, 17.5)
                accel_mag = rng.uniform(-2.8, 2.6) if rng.random() < 0.5 else 0.0
                yaw_rate = math.radians(rng.uniform(-16.0, 16.0)) if rng.random() < 0.45 else 0.0

        off_f = rng.choice([-0.34, 0.0, 0.0, 0.34]) if "kei" in arch else (rng.uniform(-0.38, 0.38) if rng.random() < 0.25 else 0.0)
        is_heavy = subtype in (SUBTYPE_DUMP_TRUCK, SUBTYPE_CONCRETE_MIXER, SUBTYPE_CRANE_HEAVY, SUBTYPE_TRACTOR_TRAILER, SUBTYPE_TANKER_SPECIAL)
        off_r = rng.uniform(-0.42, 0.42) if is_heavy and rng.random() < 0.35 else 0.0

        mud_lvl_val = rng.uniform(0.08, 0.52) if is_heavy else (rng.uniform(0.0, 0.24) if rng.random() < 0.4 else 0.0)
        mud_ph_val = rng.uniform(0.0, 10.0)
        sticker_i_val = rng.randrange(1, min(7, len(self.bank.sticker_mips)))
        has_st_val = is_heavy and rng.random() < 0.65
        has_hz_val = arch == "tanker_lorry" or (is_heavy and rng.random() < 0.14)
        tr_yaw_d_val = math.radians(rng.uniform(-22.0, 22.0)) if is_articulated else 0.0
        bump_amp_val = rng.uniform(0.015, 0.075) if motion_mode != "stationary" and rng.random() < 0.55 else 0.0
        bump_freq_val = rng.uniform(1.6, 3.8)
        bump_ph_val = rng.uniform(0.0, 6.28)
        fcab_hn_i_val = rng.choice([7, 8, 9, 10, 11, 12, 15]) if len(self.bank.sticker_mips) >= 16 else 1
        has_fcab_hn_val = (is_heavy or arch in ("box_wing_truck", "flatbed_truck", "kei_truck")) and rng.random() < 0.68
        fbmp_hn_i_val = rng.choice([13, 14]) if len(self.bank.sticker_mips) >= 15 else 1
        has_fbmp_hn_val = (is_heavy or arch in ("box_wing_truck", "flatbed_truck")) and rng.random() < 0.58
        has_dload_val = arch in ("dump_10t", "deep_dump") and rng.random() < 0.62
        dload_lin_val = rng.choice(
            (
                (0.86, 0.88, 0.92),  # Piled snow (`w3/03`)
                (0.58, 0.56, 0.52),  # Crushed gravel
                (0.62, 0.52, 0.36),  # Excavated sand / earth
            )
        )

        # Gen-5/Gen-6 decoupled deterministic stream (preserves legacy `rng` draw sequence 100%)
        v5_mix = (
            (int(track_id) & 0xFFFF) * 0x9E3779B97F4A7C15
            ^ (int(round(float(lane_x) * 10000.0)) & 0xFFFFFFFFFFFFFFFF) * 0xBF58476D1CE4E5B9
            ^ (int(round(float(y0) * 10000.0)) & 0xFFFFFFFFFFFFFFFF) * 0x94D049BB133111EB
            ^ int(idx_a) * 0xD6E8FEB86659FD93
            ^ int(fp_idx) * 0xA24BAED4963EE407
        ) & ((1 << 64) - 1)
        rng_v5 = random.Random(v5_mix)
        bmp_front_style = rng_v5.choices([0, 1, 2], weights=[0.38, 0.32, 0.30], k=1)[0]
        bmp_rear_style = rng_v5.choices([0, 1, 2], weights=[0.38, 0.32, 0.30], k=1)[0]
        _ = rng_v5.uniform(0.08, 0.72)  # retired scalar overhang proxy; shadows now come from 3D ray visibility V(x, omega_i)
        smooth_frp = bool(arch == "box_wing_truck" and rng_v5.random() < 0.52)
        reefer_nose = bool(arch == "box_wing_truck" and rng_v5.random() < 0.48)
        wind_defl = bool(arch == "box_wing_truck" and rng_v5.random() < 0.50)
        has_door_dec = bool(arch not in ("sedan", "kei_tall_wagon", "suv_4wd") and rng_v5.random() < 0.55)
        door_dec_idx = int(rng_v5.randrange(16, len(self.bank.sticker_mips))) if len(self.bank.sticker_mips) > 16 else 1
        # Wider intersection / turning / oblique yard yaw angles (`0°..360°`)
        if dist_mid >= 7.8 and abs(lane_x) <= 2.4 and rng_v5.random() < 0.24:
            base_yaw = rng_v5.uniform(0.0, 2.0 * math.pi)
            if abs(math.sin(base_yaw)) > 0.62 and abs(speed) > 4.2:
                speed = math.copysign(4.2, speed)

        dn_choices = RADIAL_DN_ORDERS_BY_DOMAIN_SPLIT[self.config.domain_split]
        dn_val = int(radial_dn_n) if radial_dn_n is not None else int(rng_v5.choice(dn_choices))
        combo_row = rng_v5.choice(GRAMMAR_COMBOS_BY_DOMAIN_SPLIT[self.config.domain_split])
        v_prod_val = str(veh_struct_prod) if veh_struct_prod is not None else str(combo_row[1])
        m_prod_val = str(marking_prod) if marking_prod is not None else str(combo_row[3])
        is_silver_paint = paint_lin[0] > 0.65 and abs(paint_lin[0] - paint_lin[2]) < 0.08
        p_metal = rng_v5.uniform(0.32, 0.74) if is_silver_paint else rng_v5.uniform(0.02, 0.24)
        p_clear = rng_v5.uniform(0.38, 0.88)
        dec_idx_val = int(rng_v5.randrange(max(1, len(self.bank.alpha_decal_mips))))
        dec_rel_val = rng_v5.uniform(0.34, 0.86) if ("relief" in m_prod_val or rng_v5.random() < 0.40) else 0.0
        has_flex_val = bool(
            (v_prod_val == "flex_wave_sheet" or rng_v5.random() < 0.36)
            and arch in ("dump_10t", "deep_dump", "flatbed_truck", "kei_truck", "box_wing_truck")
        )
        flex_sheet_palettes: tuple[tuple[float, float, float], ...] = (
            (0.14, 0.28, 0.16),
            (0.28, 0.32, 0.36),
            (0.58, 0.48, 0.24),
            (0.09, 0.22, 0.42),
            (0.70, 0.72, 0.75),
        )
        flex_lin_val = rng_v5.choice(flex_sheet_palettes)
        boom_yaw_val = math.radians(rng_v5.uniform(-16.0, 16.0))
        boom_elev_val = rng_v5.uniform(0.08, 0.34)

        return _VehiclePassSpec(
            track_id=track_id,
            archetype=arch,
            subtype=subtype,
            length_m=float(L),
            width_m=float(W),
            cab_h_m=float(cab_h),
            body_h_m=float(body_h),
            paint_lin=paint_lin,
            secondary_lin=sec_lin,
            mud_level=float(mud_lvl_val),
            mud_phase=float(mud_ph_val),
            front_plate_idx=fp_idx,
            rear_plate_idx=rp_idx,
            sticker_idx=int(sticker_i_val),
            has_sticker=bool(has_st_val),
            has_hazard_sign=bool(has_hz_val),
            plate_offset_x_front=float(off_f),
            plate_offset_x_rear=float(off_r),
            p0=np.array([lane_x, y0, 0.0], dtype=np.float32),
            speed0_mps=float(speed),
            accel_mps2=float(accel_mag),
            motion_mode=motion_mode,
            yaw0_rad=float(base_yaw),
            yaw_rate_rad=float(yaw_rate),
            trailer_yaw_delta_rad=float(tr_yaw_d_val),
            bump_amp_m=float(bump_amp_val),
            bump_freq_hz=float(bump_freq_val),
            bump_phase=float(bump_ph_val),
            front_cab_hn_idx=int(fcab_hn_i_val),
            has_front_cab_hn=bool(has_fcab_hn_val),
            front_bumper_hn_idx=int(fbmp_hn_i_val),
            has_front_bumper_hn=bool(has_fbmp_hn_val),
            has_dump_load=bool(has_dload_val),
            dump_load_lin=dload_lin_val,
            bumper_style_front=int(bmp_front_style),
            bumper_style_rear=int(bmp_rear_style),
            is_large_truck=bool(is_lg),
            has_reefer_nose=reefer_nose,
            is_smooth_frp_box=smooth_frp,
            has_wind_deflector=wind_defl,
            has_side_door_decal=has_door_dec,
            side_door_decal_idx=door_dec_idx,
            paint_metallic=float(p_metal),
            paint_clearcoat=float(p_clear),
            radial_dn_n=dn_val,
            veh_struct_prod=v_prod_val,
            marking_prod=m_prod_val,
            decal_idx=dec_idx_val,
            decal_relief=float(dec_rel_val),
            has_flex_sheet=has_flex_val,
            flex_sheet_lin=flex_lin_val,
            boom_yaw_rad=float(boom_yaw_val),
            boom_elev_rad=float(boom_elev_val),
        )

    @staticmethod
    def _obb2d_overlap(c1: np.ndarray, yaw1: float, hw1: float, hl1: float, c2: np.ndarray, yaw2: float, hw2: float, hl2: float) -> bool:
        """Exact 2D Separating-Axis Theorem (SAT) check for rotated planar vehicle footprints."""
        u1 = np.array([math.cos(yaw1), math.sin(yaw1)], dtype=np.float32)
        v1 = np.array([-math.sin(yaw1), math.cos(yaw1)], dtype=np.float32)
        u2 = np.array([math.cos(yaw2), math.sin(yaw2)], dtype=np.float32)
        v2 = np.array([-math.sin(yaw2), math.cos(yaw2)], dtype=np.float32)
        d = (c2[:2] - c1[:2]).astype(np.float32)
        for ax in (u1, v1, u2, v2):
            r1 = hw1 * abs(float(np.dot(u1, ax))) + hl1 * abs(float(np.dot(v1, ax)))
            r2 = hw2 * abs(float(np.dot(u2, ax))) + hl2 * abs(float(np.dot(v2, ax)))
            if abs(float(np.dot(d, ax))) > (r1 + r2):
                return False
        return True

    @staticmethod
    def _vehicle_planar_obbs_at(
        vs: _VehiclePassSpec,
        t: float,
        pad_w: float = 0.0,
        pad_l: float = 0.0,
    ) -> list[tuple[np.ndarray, float, float, float]]:
        """Exact planar `(center_xy, yaw_rad, hw, hl)` OBB(s) enclosing assembled tractor + articulated trailer!"""
        pos, yaw, _, _ = vs.pose_at(t)
        hw = 0.5 * float(vs.width_m)
        hl = 0.5 * float(vs.length_m)
        R_yaw = _rot_z(yaw)
        p_xy0 = np.array([float(pos[0]), float(pos[1]), 0.0], dtype=np.float32)

        if vs.archetype in _ARTICULATED_ARCHETYPES:
            # Tractor spans local `x in [-(hw+0.24), +(hw+0.24)]` (with side mirrors) and `y in [-0.10, hl+0.12]`
            tr_c = (p_xy0 + R_yaw @ np.array([0.0, 0.5 * (hl + 0.02), 0.0], dtype=np.float32))[:2]
            tr_obb = (tr_c, float(yaw), hw + 0.24 + pad_w, 0.5 * (hl + 0.22) + pad_l)

            # Articulated semi-trailer rotating around fifth-wheel kingpin `(0.0, +0.45)`!
            yaw_trl = float(yaw + vs.trailer_yaw_delta_rad)
            R_trl_z = _rot_z(yaw_trl)
            trl_hl = hl * 0.72
            trl_hw = hw * (1.02 if vs.archetype == "heavy_lowboy_trailer" else 0.99)
            kp_xy0 = p_xy0 + R_yaw @ np.array([0.0, 0.45, 0.0], dtype=np.float32)
            tl_c = (kp_xy0 + R_trl_z @ np.array([0.0, -trl_hl + 0.55, 0.0], dtype=np.float32))[:2]
            tl_obb = (tl_c, yaw_trl, trl_hw + 0.08 + pad_w, trl_hl + 0.10 + pad_l)
            return [tr_obb, tl_obb]

        return [(p_xy0[:2], float(yaw), hw + 0.24 + pad_w, hl + 0.20 + pad_l)]

    @staticmethod
    def _max_planar_point_speed_mps(vs: _VehiclePassSpec, t_lo: float, t_hi: float) -> float:
        """Exact analytical upper bound on planar speed `||dp/dt||_2` of every point on `vs` over `[t_lo, t_hi]`."""
        if vs.motion_mode == "stationary":
            return 0.0
        probe_ts = (float(t_lo), 0.0, 0.18, 0.30, float(t_hi))
        s_max = max(abs(vs.signed_speed_at(tp)) for tp in probe_ts)
        if vs.motion_mode == "stop_and_go":
            s_max = max(s_max, max(0.65, float(vs.speed0_mps) * 1.15 + 1.5))
        w_max = 1.2 * abs(float(vs.yaw_rate_rad))
        lon_span = (1.35 if vs.archetype in _ARTICULATED_ARCHETYPES else 0.55) * float(vs.length_m) + 0.50
        r_arm = math.hypot(0.5 * float(vs.width_m) + 0.35, lon_span)
        return float(s_max + w_max * r_arm)

    @classmethod
    def _trajectories_clear(
        cls,
        cand: _VehiclePassSpec,
        existing: list[_VehiclePassSpec],
        duration_s: float = 1.25,
        t_start_s: float = 0.0,
    ) -> bool:
        """Guarantee zero rotated/articulated 3D vehicle interpenetration across continuous `[t_start_s, duration_s]`.

        Uses a Lipschitz swept-margin bound: choosing `dt <= 2 * r_sweep / (V_max(cand) + V_max(ex))` with
        `r_sweep = 0.12 m <= pad_w` guarantees that for every continuous `t in [t_k - dt/2, t_k + dt/2]`, the true
        moving OBB `B(t)` is strictly contained inside the padded OBB `B_pad(t_k)` checked at grid sample `t_k`.
        """
        if not existing:
            return True
        t_lo = min(float(t_start_s), float(duration_s))
        t_hi = max(t_lo + 1e-5, max(float(t_start_s), float(duration_s)))
        span_s = t_hi - t_lo
        v_cand = cls._max_planar_point_speed_mps(cand, t_lo, t_hi)
        r_sweep = 0.12

        for ex in existing:
            v_rel = max(0.5, v_cand + cls._max_planar_point_speed_mps(ex, t_lo, t_hi))
            dt_max = (2.0 * r_sweep) / v_rel
            n_steps = max(16, int(math.ceil(span_s / max(1e-4, dt_max))))
            inv_n = 1.0 / float(n_steps)
            for si in range(n_steps + 1):
                t_chk = t_lo + span_s * (si * inv_n)
                c_obbs = cls._vehicle_planar_obbs_at(cand, t_chk, pad_w=0.28, pad_l=0.38)
                e_obbs = cls._vehicle_planar_obbs_at(ex, t_chk, pad_w=0.28, pad_l=0.38)
                for cc, cyaw, chw, chl in c_obbs:
                    rc = math.hypot(chw, chl)
                    cx_f, cy_f = float(cc[0]), float(cc[1])
                    for ec, eyaw, ehw, ehl in e_obbs:
                        r_sum = rc + math.hypot(ehw, ehl)
                        dx = float(ec[0]) - cx_f
                        dy = float(ec[1]) - cy_f
                        if (dx * dx + dy * dy) > (r_sum * r_sum):
                            continue
                        if cls._obb2d_overlap(cc, cyaw, chw, chl, ec, eyaw, ehw, ehl):
                            return False
        return True

    @staticmethod
    def _sample_mip_trilinear(mips: tuple[Tensor, ...], grid: Tensor, px_span: float) -> Tensor:
        """Exact border-preserving C^0-continuous trilinear mipmap lookup (`align_corners=False`, zero mip-switch pop)."""
        tex_w = float(mips[0].shape[-1])
        max_lod = float(len(mips) - 1)
        lod = max(0.0, min(max_lod, math.log2(max(1e-3, tex_w / max(4.0, px_span)))))
        l0 = int(math.floor(lod))
        frac = lod - float(l0)
        s0 = F.grid_sample(mips[l0], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        if frac <= 1e-7 or l0 >= len(mips) - 1:
            return s0
        s1 = F.grid_sample(mips[l0 + 1], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        return s0 * (1.0 - frac) + s1 * frac

    @torch.no_grad()
    def _count_isolated_vehicle_pixels(
        self,
        h: int,
        w: int,
        cam_o: np.ndarray,
        R_cam: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        veh_prims: list[_Prim3D],
        veh_quads: list[_PlanarQuad3D] | None = None,
        veh_idx: int = 1,
    ) -> int:
        """Count exact unoccluded in-frame silhouette pixels of an isolated vehicle (curved prims + attached quads)."""
        q_list = veh_quads or []
        if not veh_prims and not q_list:
            return 0
        z3 = torch.zeros(3, device=self.device, dtype=torch.float32)
        sun = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
        _, iso_veh_buf, _, _, _ = self._render_substep_hdr(
            h, w, cam_o, R_cam, fx, fy, cx, cy,
            prims=veh_prims,
            quads=q_list,
            sun_dir=sun,
            sky_zenith=z3,
            sky_horizon=z3,
            road_albedo_base=0.05,
            is_night=False,
            is_ir=True,
            is_wet=False,
            fence_cfg=None,
            record_masks=False,
        )
        return int((iso_veh_buf == int(veh_idx)).sum().item())

    # Canonical 12-slot `(r0, r1, c0, c1, u_span, nyquist_ph_px)` ROIs on `(16, 32)` rectified plate plane:
    # Slots 0..3: region kanji | Slots 4..6: classification | Slot 7: kana | Slots 8..11: 4-digit serial
    # Per-slot SNR uses observed sensor radiance and noise on the visible ray support.
    _SLOT_CANON_ROIS_16X32: tuple[tuple[int, int, int, int, float, float], ...] = (
        (1, 6, 4, 8, 0.10, 14.0),
        (1, 6, 7, 11, 0.09, 14.0),
        (1, 6, 10, 14, 0.10, 14.0),
        (1, 6, 13, 17, 0.09, 14.0),
        (1, 6, 17, 21, 0.11, 10.5),
        (1, 6, 20, 24, 0.11, 10.5),
        (1, 6, 24, 28, 0.11, 10.5),
        (7, 13, 1, 7, 0.16, 10.0),
        (6, 14, 7, 13, 0.17, 6.5),
        (6, 14, 12, 18, 0.16, 6.5),
        (6, 14, 19, 25, 0.17, 6.5),
        (6, 14, 25, 31, 0.17, 6.5),
    )

    def _compute_plate_supervision_targets(
        self,
        *,
        qd: _PlanarQuad3D,
        q_cov_map: np.ndarray,
        c4_uv: np.ndarray,
        cam_o: np.ndarray,
        R_cam: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w: int,
        h: int,
        pw_px: float,
        ph_px: float,
        facing_cos: float,
        vis_q: float,
        blur_vec: np.ndarray,
        sensor_lum_map: np.ndarray | None = None,
        sensor_sigma_map: np.ndarray | None = None,
        shadow_vis_map: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, float, np.ndarray, float, float, float, float, float, np.ndarray, float]:
        """Compute first-principles physical observability quantities & smooth C^inf 4-axis supervision."""
        # Sample 2x2 subgrid per `(16, 32)` canonical cell (`32 x 64 = 2048` exact 3D points on `qd`)
        us = (np.arange(64, dtype=np.float32) + 0.5) / 64.0 * 2.0 - 1.0  # (-1, 1)
        vs = (np.arange(32, dtype=np.float32) + 0.5) / 32.0 * 2.0 - 1.0  # (-1, 1)
        vv, uu = np.meshgrid(vs, us, indexing="ij")
        pts_w = (
            qd.center[None, :]
            + (uu.reshape(-1, 1) * float(qd.half_w)) * qd.u_axis[None, :]
            + (vv.reshape(-1, 1) * float(qd.half_h)) * qd.v_axis[None, :]
        )
        uv_s, zc_s = self._project_points(pts_w, cam_o, R_cam, fx, fy, cx, cy)
        ix = np.rint(uv_s[:, 0]).astype(np.int32)
        iy = np.rint(uv_s[:, 1]).astype(np.int32)
        in_b = (zc_s > 0.20) & (ix >= 0) & (ix < int(w)) & (iy >= 0) & (iy < int(h))

        vals = np.zeros((32 * 64,), dtype=np.float32)
        lum_s = np.full((32 * 64,), 0.22, dtype=np.float32)
        sig_s = np.full((32 * 64,), 0.008, dtype=np.float32)
        shv_s = np.ones((32 * 64,), dtype=np.float32)
        if np.any(in_b):
            iy_b, ix_b = iy[in_b], ix[in_b]
            vals[in_b] = q_cov_map[iy_b, ix_b]
            if sensor_lum_map is not None:
                lum_s[in_b] = sensor_lum_map[iy_b, ix_b]
            if sensor_sigma_map is not None:
                sig_s[in_b] = np.maximum(1e-4, sensor_sigma_map[iy_b, ix_b])
            if shadow_vis_map is not None:
                shv_s[in_b] = shadow_vis_map[iy_b, ix_b]

        plate_vis_16x32 = vals.reshape(16, 2, 32, 2).mean(axis=(1, 3)).astype(np.float32)
        np.clip(plate_vis_16x32, 0.0, 1.0, out=plate_vis_16x32)
        def cell_support_mean(samples: np.ndarray) -> np.ndarray:
            numerator = (samples * vals).reshape(16, 2, 32, 2).mean(axis=(1, 3))
            return np.divide(numerator, plate_vis_16x32, out=np.zeros_like(numerator), where=plate_vis_16x32 > 0)

        lum_16x32 = cell_support_mean(lum_s)
        lum2_16x32 = cell_support_mean(lum_s ** 2)
        sig2_16x32 = cell_support_mean(sig_s ** 2)
        shv_16x32 = cell_support_mean(shv_s)

        vis_support_frac = float(plate_vis_16x32.mean())
        edge_lens = [float(np.linalg.norm(c4_uv[(ei + 1) % 4] - c4_uv[ei])) for ei in range(4)]
        min_edge_px = float(min(edge_lens))
        b_px = float(np.linalg.norm(blur_vec))

        def support_moments(r0: int, r1: int, c0: int, c1: int) -> tuple[float, float, float]:
            weights = plate_vis_16x32[r0:r1, c0:c1]
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                return 0., 0., 1e-4
            mean = float((lum_16x32[r0:r1, c0:c1] * weights).sum()) / weight_sum
            mean_square = float((lum2_16x32[r0:r1, c0:c1] * weights).sum()) / weight_sum
            noise_variance = float((sig2_16x32[r0:r1, c0:c1] * weights).sum()) / weight_sum
            return mean, math.sqrt(max(0., mean_square - mean * mean)), math.sqrt(max(1e-8, noise_variance))

        local_illum, local_contrast, mean_sig = support_moments(0, 16, 0, 32)
        support_sum = float(plate_vis_16x32.sum())
        measured_shadow_frac = (float(np.clip(1 - float((shv_16x32 * plate_vis_16x32).sum()) / support_sum, 0., 1.))
                                if support_sum > 0 else 0.)

        cos_pos = max(1e-4, float(facing_cos))
        eff_plate_area_px = max(1.0, float(pw_px) * float(ph_px) * cos_pos * max(0.05, vis_support_frac))
        rel_blur_plate = b_px / max(4.0, float(pw_px))
        mtf_plate = 1.0 / math.sqrt(1.0 + 0.65 * rel_blur_plate * rel_blur_plate)
        # Measured RMS radiance contrast over the same visible support as the sensor variance.
        # No fixed ink/background contrast is presumed for missing or invisible characters.
        local_snr = float(mtf_plate * (local_contrast / mean_sig) * math.sqrt(min(64.0, eff_plate_area_px / 12.0)))

        if int(qd.cls_id) != CLS_PLATE:
            return (
                0.0,
                plate_vis_16x32,
                0.0,
                np.zeros((12,), dtype=np.float32),
                vis_support_frac,
                min_edge_px,
                b_px,
                local_illum,
                local_snr,
                np.zeros((12,), dtype=np.float32),
                measured_shadow_frac,
            )

        # Smooth C^inf physical observability curves (Hill / MTF / Foreshortening Jacobian / SNR — zero step cliffs!)
        r_jac = cos_pos / 0.15
        g_jac = (r_jac ** 3) / (1.0 + r_jac ** 3)
        r_nyq = max(1e-3, min_edge_px) / 5.2
        g_nyq = (r_nyq ** 4) / (1.0 + r_nyq ** 4)
        r_vis = max(1e-3, vis_support_frac) / 0.25
        g_vis = (r_vis ** 4) / (1.0 + r_vis ** 4)
        g_mtf = 1.0 / (1.0 + 0.65 * rel_blur_plate * rel_blur_plate)
        r_snr = max(1e-3, local_snr) / 1.8
        g_snr = (r_snr * r_snr) / (1.0 + r_snr * r_snr)
        geom_usable = float(np.clip(g_jac * g_nyq * g_vis * g_mtf * g_snr, 0.0, 1.0))

        slot_snr = np.zeros((12,), dtype=np.float32)
        slot_read = np.zeros((12,), dtype=np.float32)
        h_jac_slot = ((cos_pos / 0.13) ** 2) / (1.0 + (cos_pos / 0.13) ** 2)
        for s_idx, (r0, r1, c0, c1, u_span, nyq_ph) in enumerate(self._SLOT_CANON_ROIS_16X32):
            vis_s = float(plate_vis_16x32[r0:r1, c0:c1].mean())
            _, contrast_slot, sig_slot = support_moments(r0, r1, c0, c1)
            sw_px = max(0.75, 1.35 * float(pw_px) * u_span)
            mtf_s = 1.0 / math.sqrt(1.0 + (b_px / sw_px) ** 2 + (0.70 / sw_px) ** 2)
            cell_frac = float((r1 - r0) * (c1 - c0)) / float(16 * 32)
            slot_area_px = max(0.20, cell_frac * float(pw_px) * float(ph_px) * cos_pos)
            snr_s = float(vis_s * mtf_s * (contrast_slot / sig_slot) * math.sqrt(slot_area_px))
            slot_snr[s_idx] = snr_s

            nu_s = max(1e-3, float(ph_px)) / (0.60 * nyq_ph)
            h_nyq_s = (nu_s ** 4) / (1.0 + nu_s ** 4)
            h_snr_s = ((snr_s / 2.6) ** 2) / (1.0 + (snr_s / 2.6) ** 2)
            slot_read[s_idx] = float(np.clip(vis_s * mtf_s * h_nyq_s * h_snr_s * h_jac_slot, 0.0, 1.0))

        return (
            1.0,
            plate_vis_16x32,
            geom_usable,
            slot_read,
            vis_support_frac,
            min_edge_px,
            b_px,
            local_illum,
            local_snr,
            slot_snr,
            measured_shadow_frac,
        )

    @staticmethod
    def _flex_wave_eval(yv: Tensor, zv: Tensor, amp: float, freq: float, phase: float) -> Tensor:
        """Multi-octave flexural sheet mid-surface displacement `f(y, z)` in local meters."""
        a1 = freq * yv + phase + 0.35 * torch.sin(2.1 * zv)
        a2 = 2.3 * freq * yv - 1.7 * zv + 1.9 * phase
        a3 = 4.1 * freq * yv + 3.3 * zv
        return amp * (0.62 * torch.sin(a1) + 0.26 * torch.sin(a2) + 0.12 * torch.cos(a3))

    @classmethod
    def _intersect_flex_sheet_local(
        cls,
        ox: Tensor,
        oy: Tensor,
        oz: Tensor,
        dl_x: Tensor | float,
        dl_y: Tensor | float,
        dl_z: Tensor | float,
        sx: float,
        sy: float,
        sz: float,
        amp: float,
        freq: float,
        phase: float,
        t_near: float,
    ) -> tuple[Tensor, Tensor]:
        """Intersect the bounded wave shell to 10 micrometres or float32 ray spacing.

        Conservative steps use an analytic directional Lipschitz bound, so grazing
        rays cannot jump over a fold as they can with a fixed sampling grid. Rays
        originating inside the shell find its exit, including finite box edges.
        """
        # Broadcast each component independently: shadow rays have scalar directions,
        # camera rays have tensors, and mixed broadcastable components are also valid.
        dl_x, dl_y, dl_z = (torch.as_tensor(d, dtype=ox.dtype, device=ox.device) for d in (dl_x, dl_y, dl_z))

        def slab(origin: Tensor, direction: Tensor, half_extent: float) -> tuple[Tensor, Tensor]:
            parallel = direction == 0
            safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
            a, b = (-half_extent - origin) / safe_direction, (half_extent - origin) / safe_direction
            inside = origin.abs() <= half_extent
            low = torch.where(parallel, torch.where(inside, -torch.inf, torch.inf), torch.minimum(a, b))
            high = torch.where(parallel, torch.where(inside, torch.inf, -torch.inf), torch.maximum(a, b))
            return low, high

        tx0, tx1 = slab(ox, dl_x, sx)
        ty0, ty1 = slab(oy, dl_y, sy)
        tz0, tz1 = slab(oz, dl_z, sz)
        t_bmin = torch.maximum(torch.maximum(tx0, ty0), tz0)
        t_bmax = torch.minimum(torch.minimum(tx1, ty1), tz1)
        box_ok = (t_bmax >= t_bmin) & (t_bmax > t_near) & torch.isfinite(t_bmax)
        if not box_ok.any():
            return torch.full_like(t_bmin, 1e6), box_ok

        tau = max(0.014, 0.24 * sx)
        w_f = max(0.5, float(freq))
        amp_f = float(amp)
        ph_f = float(phase)

        t_a = torch.where(box_ok, t_bmin.clamp_min(t_near), torch.zeros_like(t_bmin))
        t_b = torch.where(box_ok, torch.maximum(t_a, t_bmax), t_a)

        def shell_residual(t: Tensor) -> Tensor:
            return ((ox + t * dl_x) - cls._flex_wave_eval(
                oy + t * dl_y, oz + t * dl_z, amp_f, w_f, ph_f)).abs() - tau

        initial = shell_residual(t_a)
        entry = box_ok & (initial <= 0) & (t_bmin > t_near)
        inside = (initial < 0) & ~entry
        direction = torch.where(inside, -1., 1.)
        # |df/dy| <= (0.62 + 0.26*2.3 + 0.12*4.1) A w = 1.71 A w
        # |df/dz| <= (0.62*0.35*2.1 + 0.26*1.7 + 0.12*3.3) A = 1.2937 A
        lipschitz = (dl_x.abs() + abs(amp_f) * (1.71 * w_f * dl_y.abs() + 1.2937 * dl_z.abs())).clamp_min(1e-12)
        t = t_a.clone()
        found = entry.clone()
        active = box_ok & ~found
        while bool(active.any()):
            residual = direction * shell_residual(t)
            at_end = t >= t_b
            contact = active & ((residual <= 1e-5) | (inside & at_end))
            found |= contact
            active &= ~contact & ~at_end
            step = residual.clamp_min(0.) / lipschitz
            next_t = torch.minimum(t + step, t_b)
            # At large ray parameters float32 spacing can exceed the surface
            # tolerance. Accept only arithmetic stagnation, never a fixed step cap.
            stagnant = active & (next_t <= t)
            found |= stagnant
            active &= ~stagnant
            t = torch.where(active, next_t, t)

        t_hit = torch.where(found, t, torch.full_like(t, 1e6))
        return t_hit, found & (t_hit > t_near)

    @torch.no_grad()
    def _render_substep_hdr(
        self,
        h: int,
        w: int,
        cam_o: np.ndarray,
        R_cam: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        prims: list[_Prim3D],
        quads: list[_PlanarQuad3D],
        sun_dir: Tensor,
        sky_zenith: Tensor,
        sky_horizon: Tensor,
        road_albedo_base: float,
        is_night: bool,
        is_ir: bool,
        is_wet: bool,
        fence_cfg: dict[str, float] | None,
        record_masks: bool,
        env_style: dict[str, float] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor]:
        """Render a single instant in Linear HDR `(3, H, W)` via 3-pass G-Buffer + True 3D 1-Ray Shadow + Unified GGX/IBL."""
        dev = self.device
        es = env_style or {"seed_ph": 1.3, "ground_type": 0.0, "skyline_type": 0.0}
        ph_e = float(es["seed_ph"])
        g_type = int(es["ground_type"])
        sky_type = int(es["skyline_type"])

        vv = torch.arange(h, device=dev, dtype=torch.float32)
        uu = torch.arange(w, device=dev, dtype=torch.float32)
        yn = (vv - cy) / fy
        xn = (uu - cx) / fx

        R_t = torch.from_numpy(R_cam).to(device=dev, dtype=torch.float32)
        cam_o_t = torch.from_numpy(cam_o).to(device=dev, dtype=torch.float32)

        dx_w = xn[None, :] * R_t[0, 0] + yn[:, None] * R_t[0, 1] + R_t[0, 2]
        dy_w = xn[None, :] * R_t[1, 0] + yn[:, None] * R_t[1, 1] + R_t[1, 2]
        dz_w = xn[None, :] * R_t[2, 0] + yn[:, None] * R_t[2, 1] + R_t[2, 2]
        d_norm = torch.sqrt(dx_w * dx_w + dy_w * dy_w + dz_w * dz_w).clamp_min(1e-6)
        rd_x = dx_w / d_norm
        rd_y = dy_w / d_norm
        rd_z = dz_w / d_norm
        # Coaxial camera IR/headlight angular beam falloff `cos^3(theta)` around optical axis `z_cam`
        cos_opt = (1.0 / d_norm).clamp(0.2, 1.0)
        ir_ang_gain = cos_opt.pow(3.0)

        sd_raw = sun_dir.to(device=dev, dtype=torch.float32)
        sd_len = float(torch.linalg.norm(sd_raw).item())
        sd_unit = sd_raw / max(1e-6, sd_len)
        sd_x = float(sd_unit[0].item())
        sd_y = float(sd_unit[1].item())
        sd_z = float(sd_unit[2].item())
        env_energy_gate = float(((sky_zenith.abs().sum() + sky_horizon.abs().sum()) * 25.0).clamp(0.0, 1.0).item())

        def _eval_env_radiance(rx: Tensor, ry: Tensor, rz: Tensor) -> Tensor:
            """Evaluate sky, cumulus clouds, horizon skyline, and ground-bounce environment radiance along `(rx, ry, rz)`."""
            if is_ir:
                az_i = torch.atan2(rx, ry.clamp_min(-1.0) + torch.where(ry.abs() < 1e-4, torch.full_like(ry, 1e-4), torch.zeros_like(ry)))
                env_ir = torch.where(rz >= 0.0, torch.full_like(rz, 0.004), torch.full_like(rz, 0.011))
                if env_energy_gate > 0.0:
                    if sky_type == 0:
                        step_az = torch.floor(az_i * 11.0 + ph_e)
                        bldg_h = 0.05 + 0.11 * (0.5 + 0.5 * torch.sin(step_az * 3.7 + ph_e))
                        in_bldg = (rz >= -0.01) & (rz < bldg_h)
                        win_u = ((az_i * 140.0).remainder(1.0) > 0.42).to(torch.float32)
                        win_v = ((rz * 110.0).remainder(1.0) > 0.45).to(torch.float32)
                        cell_h = torch.sin(torch.floor(az_i * 140.0) * 12.9898 + torch.floor(rz * 110.0) * 78.233 + ph_e)
                        win_lit = win_u * win_v * (cell_h > 0.66).to(torch.float32)
                        env_ir = torch.where(in_bldg, 0.012 + 0.14 * win_lit, env_ir)
                    elif sky_type == 1:
                        tree_h = 0.04 + 0.07 * (0.5 + 0.5 * torch.sin(az_i * 9.3 + ph_e) * torch.cos(az_i * 23.1 - ph_e))
                        in_tree = (rz >= -0.01) & (rz < tree_h)
                        leaf = 0.75 + 0.25 * torch.sin(az_i * 61.0 + rz * 83.0)
                        env_ir = torch.where(in_tree, leaf * 0.055, env_ir)
                return env_ir.unsqueeze(0).expand(3, *rz.shape)

            if env_energy_gate <= 1e-6:
                return torch.zeros((3, *rz.shape), device=dev, dtype=torch.float32)

            az_ang = torch.atan2(rx, torch.where(ry.abs() < 1e-4, torch.full_like(ry, 1e-4), ry))
            sky_mix = rz.clamp(0.0, 1.0).pow(0.58).unsqueeze(0)
            zen_v = sky_zenith.view(3, *([1] * rz.ndim))
            hor_v = sky_horizon.view(3, *([1] * rz.ndim))
            up_rad = hor_v * (1.0 - sky_mix) + zen_v * sky_mix
            if not is_night:
                cloud_f = (
                    0.55 * torch.sin(az_ang * 5.2 + ph_e) * torch.cos(rz * 11.0 - ph_e * 0.7)
                    + 0.45 * torch.sin(az_ang * 13.7 - rz * 19.0 + ph_e * 1.9)
                ).clamp(0.0, 1.0) * (rz > 0.02).to(torch.float32)
                c_tint = torch.tensor([0.22, 0.23, 0.25], device=dev, dtype=torch.float32).view(3, *([1] * rz.ndim))
                up_rad = up_rad + cloud_f.unsqueeze(0) * c_tint * env_energy_gate

            gnd_irr = 0.045 if is_night else (0.34 + 0.62 * max(0.08, sd_z))
            gnd_col = torch.tensor(
                [road_albedo_base * 1.02 * gnd_irr, road_albedo_base * 1.00 * gnd_irr, road_albedo_base * 0.96 * gnd_irr],
                device=dev,
                dtype=torch.float32,
            ).view(3, *([1] * rz.ndim))
            gnd_mix = (-rz).clamp(0.0, 1.0).pow(0.45).unsqueeze(0)
            down_rad = hor_v * 0.35 * (1.0 - gnd_mix) + gnd_col * (0.45 + 0.55 * gnd_mix)
            env_rgb = torch.where((rz >= 0.0).unsqueeze(0), up_rad, down_rad)

            if sky_type == 0:
                step_az = torch.floor(az_ang * 11.0 + ph_e)
                bldg_h = 0.05 + 0.11 * (0.5 + 0.5 * torch.sin(step_az * 3.7 + ph_e))
                in_bldg = (rz >= -0.01) & (rz < bldg_h)
                win_u = ((az_ang * 140.0).remainder(1.0) > 0.42).to(torch.float32)
                win_v = ((rz * 110.0).remainder(1.0) > 0.45).to(torch.float32)
                cell_hash = torch.sin(torch.floor(az_ang * 140.0) * 12.9898 + torch.floor(rz * 110.0) * 78.233 + ph_e)
                room_on = (cell_hash > 0.66).to(torch.float32) if is_night else (cell_hash > 0.15).to(torch.float32)
                win_lit = win_u * win_v * room_on
                b_base = 0.015 if is_night else 0.13
                b_win = 0.22 if is_night else 0.05
                b_col = torch.stack(
                    [
                        torch.full_like(rz, b_base) + b_win * win_lit,
                        torch.full_like(rz, b_base * 1.02) + b_win * 0.92 * win_lit,
                        torch.full_like(rz, b_base * 1.08) + b_win * 0.70 * win_lit,
                    ],
                    dim=0,
                )
                env_rgb = torch.where(in_bldg.unsqueeze(0), b_col * env_energy_gate, env_rgb)
            elif sky_type == 1:
                tree_h = 0.04 + 0.07 * (0.5 + 0.5 * torch.sin(az_ang * 9.3 + ph_e) * torch.cos(az_ang * 23.1 - ph_e))
                in_tree = (rz >= -0.01) & (rz < tree_h)
                leaf = 0.75 + 0.25 * torch.sin(az_ang * 61.0 + rz * 83.0)
                t_scale = 0.018 if is_night else 0.095
                t_col = torch.stack([leaf * t_scale * 0.65, leaf * t_scale * 1.15, leaf * t_scale * 0.58], dim=0)
                env_rgb = torch.where(in_tree.unsqueeze(0), t_col * env_energy_gate, env_rgb)
            return env_rgb

        # Background sky / distant skyline radiance
        hdr = _eval_env_radiance(rd_x, rd_y, rd_z)

        # Deferred G-Buffer tensors for Unified PBR + True 3D 1-Ray Shadow Visibility
        t_best = torch.full((h, w), 1e5, device=dev, dtype=torch.float32)
        t_geom = torch.full((h, w), 1e5, device=dev, dtype=torch.float32)
        hit_kind_buf = torch.zeros((h, w), device=dev, dtype=torch.int8)  # 0=sky, 1=ground, 2=prim, 3=quad, 4=emissive
        prim_id_buf = torch.full((h, w), -1, device=dev, dtype=torch.int16)
        veh_buf = torch.zeros((h, w), device=dev, dtype=torch.int16)
        quad_buf = torch.full((h, w), -1, device=dev, dtype=torch.int16)
        role_buf = torch.zeros((h, w), device=dev, dtype=torch.int16) if record_masks else None
        trans_buf = torch.ones((h, w), device=dev, dtype=torch.float32)

        alb_buf = torch.zeros((3, h, w), device=dev, dtype=torch.float32)
        nw_x_buf = torch.zeros((h, w), device=dev, dtype=torch.float32)
        nw_y_buf = torch.zeros((h, w), device=dev, dtype=torch.float32)
        nw_z_buf = torch.ones((h, w), device=dev, dtype=torch.float32)
        ng_x_buf = torch.zeros((h, w), device=dev, dtype=torch.float32)
        ng_y_buf = torch.zeros((h, w), device=dev, dtype=torch.float32)
        ng_z_buf = torch.ones((h, w), device=dev, dtype=torch.float32)
        rough_buf = torch.full((h, w), 0.85, device=dev, dtype=torch.float32)
        metal_buf = torch.zeros((h, w), device=dev, dtype=torch.float32)
        f0_buf = torch.zeros((h, w), device=dev, dtype=torch.float32)
        coat_buf = torch.zeros((h, w), device=dev, dtype=torch.float32)
        retro_buf = torch.zeros((h, w), device=dev, dtype=torch.float32)
        emit_buf = torch.zeros((h, w), device=dev, dtype=torch.float32)
        nir_scale_buf = torch.ones((h, w), device=dev, dtype=torch.float32)
        green_plate_buf = torch.zeros((h, w), device=dev, dtype=torch.bool)

        # Pass 1A: Ground Plane Z = 0 Intersection (Zero analytic ellipse shadow proxy; 3D shadows cast in Pass 2!)
        hit_road = dz_w < -1e-4
        if hit_road.any():
            lam_r = torch.where(hit_road, (-float(cam_o[2])) / dz_w.clamp_max(-1e-4), t_best)
            valid_r = hit_road & (lam_r > 0.16) & (lam_r < 120.0)
            if valid_r.any():
                wx = float(cam_o[0]) + lam_r * dx_w
                wy = float(cam_o[1]) + lam_r * dy_w
                # Ray/plane differentials bound the world-space pixel footprint,
                # including its rapid growth at grazing road angles.
                inv_dz = 1.0 / dz_w.clamp_max(-1e-4)
                du_x = lam_r / fx * (R_t[0, 0] - dx_w * inv_dz * R_t[2, 0])
                du_y = lam_r / fx * (R_t[1, 0] - dy_w * inv_dz * R_t[2, 0])
                dv_x = lam_r / fy * (R_t[0, 1] - dx_w * inv_dz * R_t[2, 1])
                dv_y = lam_r / fy * (R_t[1, 1] - dy_w * inv_dz * R_t[2, 1])
                footprint = torch.maximum(du_x.square() + du_y.square(), dv_x.square() + dv_y.square()).sqrt()
                gran, tar_patch, puddle = _road_surface_fields(wx, wy, lam_r, footprint, ph_e)
                dapple = 0.86 + 0.14 * torch.sin(wx * 0.45 + ph_e) * torch.cos(wy * 0.38 - ph_e)
                if g_type == 1:
                    joint = ((wx.remainder(3.5) < 0.04) | (wy.remainder(4.0) < 0.04)).to(torch.float32)
                    rut = torch.exp(-((wx.abs() - 1.05) ** 2) / 0.18) * 0.18
                    base_l = (road_albedo_base * 1.85 * (1.0 + 0.12 * gran) * (1.0 - 0.35 * joint - rut)).clamp(0.03, 0.52)
                    road_rgb = torch.stack([base_l * 1.02, base_l * 1.00, base_l * 0.95], dim=0)
                elif g_type == 2:
                    rut = torch.exp(-((wx.abs() - 1.10) ** 2) / 0.22) * 0.22
                    base_l = (road_albedo_base * 1.45 * (1.0 + 0.24 * gran) * (1.0 - rut)).clamp(0.025, 0.42)
                    road_rgb = torch.stack([base_l * 1.24, base_l * 0.98, base_l * 0.72], dim=0)
                else:
                    lane_mark = (((torch.abs(wx - 3.1) < 0.07) | (torch.abs(wx + 3.1) < 0.07)) & (ph_e < 4.2)).to(torch.float32)
                    stop_bar = ((wy > 5.8) & (wy < 6.15) & (wx.abs() < 2.9) & (ph_e > 2.5)).to(torch.float32) * 0.75
                    mark = torch.maximum(lane_mark, stop_bar)
                    road_lum = (road_albedo_base * (1.0 + 0.18 * gran) * (1.0 - tar_patch)).clamp(0.02, 0.44)
                    road_rgb = torch.stack(
                        [
                            road_lum * 1.01 + 0.38 * mark,
                            road_lum * 1.00 + 0.38 * mark,
                            road_lum * 0.97 + 0.36 * mark,
                        ],
                        dim=0,
                    )
                road_rgb = road_rgb * dapple[None, :, :]
                if is_wet:
                    road_rgb = road_rgb * (0.86 - 0.30 * puddle)[None, :, :]
                    r_rough = 0.62 * (1.0 - puddle) + 0.065 * puddle
                    r_coat = 0.12 + 0.43 * puddle
                    r_f0 = 0.02 + 0.015 * puddle
                    bump_scale = 0.035 * (1.0 - 0.88 * puddle)
                else:
                    r_rough = torch.full_like(lam_r, 0.86)
                    r_coat = torch.zeros_like(lam_r)
                    r_f0 = torch.full_like(lam_r, 0.032)
                    bump_scale = torch.full_like(lam_r, 0.04)

                rn_x = bump_scale * gran
                rn_y = -bump_scale * gran
                rn_len = torch.sqrt(rn_x * rn_x + rn_y * rn_y + 1.0)
                alb_buf = torch.where(valid_r[None, :, :], road_rgb, alb_buf)
                nw_x_buf = torch.where(valid_r, rn_x / rn_len, nw_x_buf)
                nw_y_buf = torch.where(valid_r, rn_y / rn_len, nw_y_buf)
                nw_z_buf = torch.where(valid_r, 1.0 / rn_len, nw_z_buf)
                ng_x_buf = torch.where(valid_r, torch.zeros_like(ng_x_buf), ng_x_buf)
                ng_y_buf = torch.where(valid_r, torch.zeros_like(ng_y_buf), ng_y_buf)
                ng_z_buf = torch.where(valid_r, torch.ones_like(ng_z_buf), ng_z_buf)
                rough_buf = torch.where(valid_r, r_rough, rough_buf)
                f0_buf = torch.where(valid_r, r_f0, f0_buf)
                coat_buf = torch.where(valid_r, r_coat, coat_buf)
                hit_kind_buf = torch.where(valid_r, torch.ones_like(hit_kind_buf), hit_kind_buf)
                t_best = torch.where(valid_r, lam_r, t_best)
                t_geom = torch.where(valid_r, lam_r, t_geom)

        # Pass 1B: 3D Volumetric, D_N Perforated, & Flexural Wave Sheet Primitives
        for p_idx, pr in enumerate(prims):
            sx, sy, sz = pr.half_ext
            rad_bound = math.sqrt(sx * sx + sy * sy + sz * sz)
            rel_c = (np.array(pr.center, dtype=np.float32) - cam_o) @ R_cam
            if rel_c[2] + rad_bound < 0.20:
                continue
            uv_clip = self._clip_obb_to_near_plane_uv(
                np.array(pr.center, dtype=np.float32), pr.rot_w2l.T, pr.half_ext,
                cam_o, R_cam, fx, fy, cx, cy, z_near=0.16,
            )
            if uv_clip.shape[0] == 0:
                continue
            u0 = max(0, int(math.floor(float(uv_clip[:, 0].min()) - 2.0)))
            u1 = min(w, int(math.ceil(float(uv_clip[:, 0].max()) + 2.0)))
            v0 = max(0, int(math.floor(float(uv_clip[:, 1].min()) - 2.0)))
            v1 = min(h, int(math.ceil(float(uv_clip[:, 1].max()) + 2.0)))
            if u1 <= u0 or v1 <= v0:
                continue
            px_span_pr = max(4.0, float(max(u1 - u0, v1 - v0)))

            sub_dx = dx_w[v0:v1, u0:u1]
            sub_dy = dy_w[v0:v1, u0:u1]
            sub_dz = dz_w[v0:v1, u0:u1]

            Rw2l_t = torch.from_numpy(pr.rot_w2l).to(device=dev, dtype=torch.float32)
            oc_w = cam_o_t - torch.tensor(pr.center, device=dev, dtype=torch.float32)
            oc_l = Rw2l_t @ oc_w
            dl_x = Rw2l_t[0, 0] * sub_dx + Rw2l_t[0, 1] * sub_dy + Rw2l_t[0, 2] * sub_dz
            dl_y = Rw2l_t[1, 0] * sub_dx + Rw2l_t[1, 1] * sub_dy + Rw2l_t[1, 2] * sub_dz
            dl_z = Rw2l_t[2, 0] * sub_dx + Rw2l_t[2, 1] * sub_dy + Rw2l_t[2, 2] * sub_dz

            is_cap = torch.zeros_like(dl_x, dtype=torch.bool)
            if pr.shape in (_SHAPE_OBB, _SHAPE_SLOPED_CAB):
                inv_x = torch.where(dl_x.abs() > 1e-6, 1.0 / dl_x, torch.copysign(torch.full_like(dl_x, 1e6), dl_x))
                inv_y = torch.where(dl_y.abs() > 1e-6, 1.0 / dl_y, torch.copysign(torch.full_like(dl_y, 1e6), dl_y))
                inv_z = torch.where(dl_z.abs() > 1e-6, 1.0 / dl_z, torch.copysign(torch.full_like(dl_z, 1e6), dl_z))

                tx0, tx1 = (-sx - oc_l[0]) * inv_x, (sx - oc_l[0]) * inv_x
                ty0, ty1 = (-sy - oc_l[1]) * inv_y, (sy - oc_l[1]) * inv_y
                tz0, tz1 = (-sz - oc_l[2]) * inv_z, (sz - oc_l[2]) * inv_z

                t_min = torch.maximum(torch.maximum(torch.minimum(tx0, tx1), torch.minimum(ty0, ty1)), torch.minimum(tz0, tz1))
                t_max = torch.minimum(torch.minimum(torch.maximum(tx0, tx1), torch.maximum(ty0, ty1)), torch.maximum(tz0, tz1))

                sloped_hit = torch.zeros_like(t_min, dtype=torch.bool)
                if pr.shape == _SHAPE_SLOPED_CAB:
                    ny_s = float(pr.slope_param)
                    nz_s = math.sqrt(max(0.1, 1.0 - ny_s * ny_s))
                    d_s = ny_s * (sy * 0.72) + nz_s * (sz * 0.78)
                    den_s = ny_s * dl_y + nz_s * dl_z
                    num_s = d_s - (ny_s * oc_l[1] + nz_s * oc_l[2])
                    t_s = num_s / torch.where(den_s.abs() > 1e-6, den_s, torch.full_like(den_s, 1e-6))
                    enter_slope = (den_s < -1e-6) & (t_s >= t_min)
                    sloped_hit = enter_slope
                    t_min = torch.where(den_s < -1e-6, torch.maximum(t_min, t_s), t_min)
                    t_max = torch.where(den_s > 1e-6, torch.minimum(t_max, t_s), t_max)
                    t_max = torch.where((den_s.abs() <= 1e-6) & (num_s < 0.0), torch.full_like(t_max, -1.0), t_max)

                t_hit = t_min
                valid_p = (t_min > 0.16) & (t_max >= t_min) & (t_hit < t_best[v0:v1, u0:u1])
                if not valid_p.any():
                    continue
                hx = oc_l[0] + t_hit * dl_x
                hy = oc_l[1] + t_hit * dl_y
                hz = oc_l[2] + t_hit * dl_z
                ax = (hx / max(1e-4, sx)).abs()
                ay = (hy / max(1e-4, sy)).abs()
                az = (hz / max(1e-4, sz)).abs()
                is_x = (ax >= ay) & (ax >= az)
                is_y = (~is_x) & (ay >= az)
                nl_x = torch.where(is_x, torch.sign(hx), torch.zeros_like(hx))
                nl_y = torch.where(is_y, torch.sign(hy), torch.zeros_like(hy))
                nl_z = torch.where((~is_x) & (~is_y), torch.sign(hz), torch.zeros_like(hz))
                if pr.shape == _SHAPE_SLOPED_CAB:
                    ny_s = float(pr.slope_param)
                    nz_s = math.sqrt(max(0.1, 1.0 - ny_s * ny_s))
                    nl_x = torch.where(sloped_hit, torch.zeros_like(nl_x), nl_x)
                    nl_y = torch.where(sloped_hit, torch.full_like(nl_y, ny_s), nl_y)
                    nl_z = torch.where(sloped_hit, torch.full_like(nl_z, nz_s), nl_z)

            elif pr.shape == _SHAPE_CYLINDER_X:
                oy_n, oz_n = oc_l[1] / sy, oc_l[2] / sz
                dy_n, dz_n = dl_y / sy, dl_z / sz
                qa = (dy_n * dy_n + dz_n * dz_n).clamp_min(1e-6)
                qb = oy_n * dy_n + oz_n * dz_n
                qc = oy_n * oy_n + oz_n * oz_n - 1.0
                disc = qb * qb - qa * qc
                sq = torch.sqrt(disc.clamp_min(0.0))
                tc0 = (-qb - sq) / qa
                tc1 = (-qb + sq) / qa
                inv_x = torch.where(dl_x.abs() > 1e-6, 1.0 / dl_x, torch.copysign(torch.full_like(dl_x, 1e6), dl_x))
                tx0, tx1 = (-sx - oc_l[0]) * inv_x, (sx - oc_l[0]) * inv_x
                t_min = torch.maximum(torch.where(disc >= 0.0, tc0, torch.full_like(tc0, 1e6)), torch.minimum(tx0, tx1))
                t_max = torch.minimum(torch.where(disc >= 0.0, tc1, torch.full_like(tc1, -1e6)), torch.maximum(tx0, tx1))
                t_hit = t_min
                base_valid_c = (disc >= 0.0) & (t_min > 0.16) & (t_max >= t_min)

                # D_N dihedral radial aperture pass-through (N in {3..12}): genuine 3D ray perforation!
                if pr.radial_n >= 3 and base_valid_c.any():
                    n_fold = float(pr.radial_n)
                    sector_w = (2.0 * math.pi) / n_fold
                    half_sec = 0.5 * sector_w
                    r_ring = 0.45
                    r_hole = min(0.115, 0.56 * r_ring * math.sin(half_sec))
                    hx0 = oc_l[0] + t_min * dl_x
                    hy0 = (oc_l[1] + t_min * dl_y) / sy
                    hz0 = (oc_l[2] + t_min * dl_z) / sz
                    rr0 = torch.sqrt(hy0 * hy0 + hz0 * hz0)
                    th0 = torch.remainder(torch.atan2(hz0, hy0) + pr.mud_phase + half_sec, sector_w) - half_sec
                    du0 = rr0 * torch.cos(th0) - r_ring
                    dv0 = rr0 * torch.sin(th0)
                    perf0 = (hx0.abs() >= sx * 0.92) & ((du0 * du0 + dv0 * dv0) < (r_hole * r_hole))
                    if perf0.any():
                        hx1 = oc_l[0] + t_max * dl_x
                        hy1 = (oc_l[1] + t_max * dl_y) / sy
                        hz1 = (oc_l[2] + t_max * dl_z) / sz
                        rr1 = torch.sqrt(hy1 * hy1 + hz1 * hz1)
                        th1 = torch.remainder(torch.atan2(hz1, hy1) + pr.mud_phase + half_sec, sector_w) - half_sec
                        du1 = rr1 * torch.cos(th1) - r_ring
                        dv1 = rr1 * torch.sin(th1)
                        perf1 = (hx1.abs() >= sx * 0.92) & ((du1 * du1 + dv1 * dv1) < (r_hole * r_hole))
                        # Rays piercing aligned D_N apertures exit cleanly into the scene behind; oblique rays hit inner bore at t_max
                        base_valid_c = base_valid_c & (~(perf0 & perf1))
                        t_hit = torch.where(perf0 & (~perf1), t_max, t_hit)

                valid_p = base_valid_c & (t_hit < t_best[v0:v1, u0:u1])
                if not valid_p.any():
                    continue
                hx = oc_l[0] + t_hit * dl_x
                hy = oc_l[1] + t_hit * dl_y
                hz = oc_l[2] + t_hit * dl_z
                is_cap = hx.abs() >= sx * 0.95
                nl_x = torch.where(is_cap, torch.sign(hx), torch.zeros_like(hx))
                nl_y = torch.where(is_cap, torch.zeros_like(hy), hy / (sy * sy))
                nl_z = torch.where(is_cap, torch.zeros_like(hz), hz / (sz * sz))
                # Flip normal toward incoming ray if interior wall of D_N bore was struck at t_max
                dot_in = nl_x * dl_x + nl_y * dl_y + nl_z * dl_z
                flip_s = torch.where(dot_in > 0.0, -torch.ones_like(dot_in), torch.ones_like(dot_in))
                nl_x, nl_y, nl_z = nl_x * flip_s, nl_y * flip_s, nl_z * flip_s
                n_len = torch.sqrt(nl_x * nl_x + nl_y * nl_y + nl_z * nl_z).clamp_min(1e-5)
                nl_x, nl_y, nl_z = nl_x / n_len, nl_y / n_len, nl_z / n_len

            elif pr.shape == _SHAPE_FLEX_SHEET:
                ox_t = oc_l[0].expand_as(dl_x)
                oy_t = oc_l[1].expand_as(dl_y)
                oz_t = oc_l[2].expand_as(dl_z)
                t_hit, hit_sh = self._intersect_flex_sheet_local(
                    ox_t, oy_t, oz_t, dl_x, dl_y, dl_z,
                    sx, sy, sz, pr.wave_amp, pr.wave_freq, pr.mud_phase, t_near=0.16,
                )
                valid_p = hit_sh & (t_hit < t_best[v0:v1, u0:u1])
                if not valid_p.any():
                    continue
                hx = oc_l[0] + t_hit * dl_x
                hy = oc_l[1] + t_hit * dl_y
                hz = oc_l[2] + t_hit * dl_z
                w_f = max(0.5, float(pr.wave_freq))
                amp_f = float(pr.wave_amp)
                ph_f = float(pr.mud_phase)
                a1 = w_f * hy + ph_f + 0.35 * torch.sin(2.1 * hz)
                a2 = 2.3 * w_f * hy - 1.7 * hz + 1.9 * ph_f
                a3 = 4.1 * w_f * hy + 3.3 * hz
                df_dy = amp_f * (0.62 * w_f * torch.cos(a1) + 0.26 * 2.3 * w_f * torch.cos(a2) - 0.12 * 4.1 * w_f * torch.sin(a3))
                df_dz = amp_f * (
                    0.62 * 0.35 * 2.1 * torch.cos(2.1 * hz) * torch.cos(a1)
                    - 0.26 * 1.7 * torch.cos(a2)
                    - 0.12 * 3.3 * torch.sin(a3)
                )
                raw_nx = torch.ones_like(df_dy)
                raw_ny = -df_dy
                raw_nz = -df_dz
                dot_sh = raw_nx * dl_x + raw_ny * dl_y + raw_nz * dl_z
                s_sh = torch.where(dot_sh > 0.0, -torch.ones_like(dot_sh), torch.ones_like(dot_sh))
                nl_x = raw_nx * s_sh
                nl_y = raw_ny * s_sh
                nl_z = raw_nz * s_sh
                n_len = torch.sqrt(nl_x * nl_x + nl_y * nl_y + nl_z * nl_z).clamp_min(1e-5)
                nl_x, nl_y, nl_z = nl_x / n_len, nl_y / n_len, nl_z / n_len

            else:
                # `_SHAPE_TAPER_DRUM_Y`: Exact watertight piecewise truncated double-cone + end-caps!
                taper = float(np.clip(pr.slope_param, 0.0, 0.65))
                ox_n, oy_n, oz_n = oc_l[0] / sx, oc_l[1] / sy, oc_l[2] / sz
                dx_n, dy_n, dz_n = dl_x / sx, dl_y / sy, dl_z / sz
                inf_t = torch.full_like(dx_n, 1e6)
                t_hit = inf_t.clone()
                hit_kind = torch.zeros_like(dx_n, dtype=torch.int8)

                for s_h, kind_id in ((1.0, 1), (-1.0, -1)):
                    k_c = taper * s_h
                    qa = dx_n * dx_n + dz_n * dz_n - (k_c * dy_n) * (k_c * dy_n)
                    qb = ox_n * dx_n + oz_n * dz_n + k_c * dy_n * (1.0 - k_c * oy_n)
                    qc = ox_n * ox_n + oz_n * oz_n - (1.0 - k_c * oy_n) * (1.0 - k_c * oy_n)
                    disc = qb * qb - qa * qc
                    sq = torch.sqrt(disc.clamp_min(0.0))
                    s_qb = torch.where(qb >= 0.0, torch.ones_like(qb), -torch.ones_like(qb))
                    q_cit = -(qb + s_qb * sq)
                    safe_q = torch.where(q_cit.abs() > 1e-12, q_cit, torch.copysign(torch.full_like(q_cit, 1e-12), q_cit))
                    safe_a = torch.where(qa.abs() > 1e-12, qa, torch.copysign(torch.full_like(qa, 1e-12), qa))
                    tr1 = qc / safe_q
                    tr2 = q_cit / safe_a
                    for tr, cond_den in ((tr1, q_cit.abs() > 1e-12), (tr2, qa.abs() > 1e-12)):
                        yn_r = oy_n + tr * dy_n
                        in_half = (yn_r >= -1e-4) & (yn_r <= 1.0) if s_h > 0.0 else (yn_r <= 1e-4) & (yn_r >= -1.0)
                        ok_r = (disc >= 0.0) & cond_den & (tr > 0.16) & in_half & (tr < t_hit)
                        t_hit = torch.where(ok_r, tr, t_hit)
                        hit_kind = torch.where(ok_r, torch.full_like(hit_kind, kind_id), hit_kind)

                inv_y = torch.where(dy_n.abs() > 1e-6, 1.0 / dy_n, torch.copysign(torch.full_like(dy_n, 1e6), dy_n))
                r_cap2 = (1.0 - taper) * (1.0 - taper)
                for s_cap, kind_id in ((1.0, 2), (-1.0, -2)):
                    tc_p = (s_cap - oy_n) * inv_y
                    xc_p = ox_n + tc_p * dx_n
                    zc_p = oz_n + tc_p * dz_n
                    ok_c = (tc_p > 0.16) & ((xc_p * xc_p + zc_p * zc_p) <= r_cap2) & (tc_p < t_hit)
                    t_hit = torch.where(ok_c, tc_p, t_hit)
                    hit_kind = torch.where(ok_c, torch.full_like(hit_kind, kind_id), hit_kind)

                valid_p = (hit_kind != 0) & (t_hit < t_best[v0:v1, u0:u1])
                if not valid_p.any():
                    continue
                hx = oc_l[0] + t_hit * dl_x
                hy = oc_l[1] + t_hit * dl_y
                hz = oc_l[2] + t_hit * dl_z
                is_cap = hit_kind.abs() == 2
                s_cone = torch.where(hit_kind == 1, torch.ones_like(hy), -torch.ones_like(hy))
                k_eff = taper * s_cone
                r_eff = (1.0 - k_eff * (hy / sy)).clamp_min(0.05)
                nl_x = torch.where(is_cap, torch.zeros_like(hx), hx / (sx * sx))
                nl_y = torch.where(is_cap, torch.sign(hit_kind.to(torch.float32)), (k_eff * r_eff) / sy)
                nl_z = torch.where(is_cap, torch.zeros_like(hz), hz / (sz * sz))
                n_len = torch.sqrt(nl_x * nl_x + nl_y * nl_y + nl_z * nl_z).clamp_min(1e-5)
                nl_x, nl_y, nl_z = nl_x / n_len, nl_y / n_len, nl_z / n_len

            # Unperturbed geometric world normal for shadow-ray origin offset (immune to bump/rib terminator acne!)
            ng_x_p = Rw2l_t[0, 0] * nl_x + Rw2l_t[1, 0] * nl_y + Rw2l_t[2, 0] * nl_z
            ng_y_p = Rw2l_t[0, 1] * nl_x + Rw2l_t[1, 1] * nl_y + Rw2l_t[2, 1] * nl_z
            ng_z_p = Rw2l_t[0, 2] * nl_x + Rw2l_t[1, 2] * nl_y + Rw2l_t[2, 2] * nl_z

            if pr.rib_freq > 0.0:
                rib = 0.22 * torch.cos(hy * pr.rib_freq * math.pi)
                nl_y = nl_y + rib * (1.0 - nl_y.abs())

            if pr.normal_undul_amp > 0.0:
                u_amp = float(pr.normal_undul_amp)
                und1 = torch.sin(hx * 6.3 + hy * 4.7 + pr.mud_phase) * torch.cos(hz * 5.9 - hy * 3.8)
                und2 = 0.45 * torch.sin(hx * 14.9 - hz * 13.1 + 1.7 * pr.mud_phase)
                und = u_amp * (und1 + und2)
                nl_x = nl_x + und * (1.0 - nl_x.abs())
                nl_y = nl_y - 0.8 * und * (1.0 - nl_y.abs())

            alb = torch.tensor(pr.albedo_rgb, device=dev, dtype=torch.float32)[:, None, None].expand(3, v1 - v0, u1 - u0).clone()
            p_rough = torch.full((v1 - v0, u1 - u0), float(pr.roughness), device=dev, dtype=torch.float32)
            p_metal = torch.full((v1 - v0, u1 - u0), float(pr.metallic), device=dev, dtype=torch.float32)
            p_f0 = torch.full((v1 - v0, u1 - u0), float(pr.specular_f0), device=dev, dtype=torch.float32)
            p_coat = torch.full((v1 - v0, u1 - u0), float(pr.clearcoat), device=dev, dtype=torch.float32)
            p_retro = torch.full((v1 - v0, u1 - u0), float(pr.retro_gain), device=dev, dtype=torch.float32)
            xn_l = (hx / max(1e-4, sx)).clamp(-1.0, 1.0)
            yn_l = (hy / max(1e-4, sy)).clamp(-1.0, 1.0)
            zn_l = (hz / max(1e-4, sz)).clamp(-1.0, 1.0)

            if pr.veh_idx > 0 and pr.role == _ROLE_BODY:
                in_grille = (nl_y > 0.72) & (xn_l.abs() < 0.74) & (zn_l > -0.68) & (zn_l < 0.42)
                if in_grille.any():
                    slat = (torch.sin(zn_l * 28.0) > 0.42).to(torch.float32)
                    g_col = (0.025 + 0.34 * slat)[None, :, :].expand(3, -1, -1)
                    alb = torch.where(in_grille[None, :, :], g_col, alb)
                    p_rough = torch.where(in_grille, 0.55 - 0.30 * slat, p_rough)
                    p_metal = torch.where(in_grille, 0.65 * slat, p_metal)
                seam = ((xn_l.abs() > 0.88) & (xn_l.abs() < 0.91)) | ((zn_l.abs() > 0.86) & (zn_l.abs() < 0.89))
                alb = torch.where(seam[None, :, :], alb * 0.55, alb)
                in_refl = (nl_y < -0.72) & (zn_l < -0.72) & (xn_l.abs() > 0.25)
                if in_refl.any():
                    stripe_rw = ((hx * 10.0).remainder(1.0) > 0.5).to(torch.float32)
                    refl_col = torch.stack([0.75 + 0.15 * stripe_rw, 0.06 + 0.68 * stripe_rw, 0.06 + 0.68 * stripe_rw], dim=0)
                    alb = torch.where(in_refl[None, :, :], refl_col, alb)
                    p_retro = torch.where(in_refl, torch.full_like(p_retro, 0.85), p_retro)

            elif pr.veh_idx > 0 and pr.role == _ROLE_CABIN_GLASS:
                on_pillar = (xn_l.abs() > 0.84) | (zn_l > 0.80) | (zn_l < -0.82) | ((yn_l.abs() < 0.08) & (nl_x.abs() > 0.6))
                wiper = (nl_y > 0.35) & (zn_l < -0.45) & (((xn_l + 0.32 + 0.35 * (zn_l + 0.8)).abs() < 0.03) | ((xn_l - 0.28 + 0.35 * (zn_l + 0.8)).abs() < 0.03))
                frame_col = torch.tensor([0.09, 0.10, 0.11], device=dev, dtype=torch.float32)[:, None, None].expand_as(alb)
                int_var = (0.04 + 0.035 * torch.cos(xn_l * 4.5) * (0.5 - 0.5 * zn_l))[None, :, :].expand_as(alb)
                frame_m = on_pillar | wiper
                alb = torch.where(frame_m[None, :, :], frame_col, alb + int_var)
                p_rough = torch.where(frame_m, torch.full_like(p_rough, 0.45), p_rough)
                p_coat = torch.where(frame_m, torch.full_like(p_coat, 0.10), p_coat)

            elif pr.shape == _SHAPE_CYLINDER_X and (pr.role == _ROLE_TIRE or pr.radial_n >= 3):
                r_rad = torch.sqrt(yn_l * yn_l + zn_l * zn_l)
                rim_limit = 0.64 if pr.role == _ROLE_TIRE else 0.98
                is_outer_cap = (hx.abs() >= sx * 0.90) & (r_rad < rim_limit)
                if is_outer_cap.any():
                    n_sym = float(pr.radial_n if pr.radial_n >= 3 else 6)
                    ang_w = torch.atan2(zn_l, yn_l) + pr.mud_phase
                    sec_w = (2.0 * math.pi) / n_sym
                    th_s = torch.remainder(ang_w + 0.5 * sec_w, sec_w) - 0.5 * sec_w
                    du_st = r_rad * torch.cos(th_s) - 0.28
                    dv_st = r_rad * torch.sin(th_s)
                    r_st2 = du_st * du_st + dv_st * dv_st
                    on_stud = r_st2 < (0.038 * 0.038)
                    hub = (r_rad < 0.17).to(torch.float32)
                    if pr.role == _ROLE_TIRE:
                        rim_l = 0.38 - 0.22 * on_stud.to(torch.float32) - 0.20 * hub
                        rim_col = torch.stack([rim_l, rim_l * 1.02, rim_l * 1.05], dim=0)
                        alb = torch.where(is_outer_cap[None, :, :], rim_col, alb)
                        p_rough = torch.where(is_outer_cap, torch.full_like(p_rough, 0.28), p_rough)
                        p_metal = torch.where(is_outer_cap, torch.full_like(p_metal, 0.68), p_metal)
                    # 3D normal bevel on D_N stud/bolt ring
                    stud_m = is_outer_cap & on_stud
                    if stud_m.any():
                        bev_y = (du_st * torch.cos(ang_w) - dv_st * torch.sin(ang_w)) * 8.5
                        bev_z = (du_st * torch.sin(ang_w) + dv_st * torch.cos(ang_w)) * 8.5
                        nl_y = torch.where(stud_m, nl_y + bev_y, nl_y)
                        nl_z = torch.where(stud_m, nl_z + bev_z, nl_z)

            elif pr.veh_idx > 0 and pr.role == _ROLE_CHROME_METAL and sy < 0.30:
                in_slot = (nl_y.abs() > 0.7) & (zn_l.abs() < 0.36) & (xn_l.abs() > 0.34) & (xn_l.abs() < 0.76)
                slot_col = torch.tensor([0.035, 0.038, 0.042], device=dev, dtype=torch.float32)[:, None, None].expand_as(alb)
                alb = torch.where(in_slot[None, :, :], slot_col, alb)
                p_rough = torch.where(in_slot, torch.full_like(p_rough, 0.78), p_rough)
                p_metal = torch.where(in_slot, torch.zeros_like(p_metal), p_metal)

            elif pr.veh_idx > 0 and pr.role == _ROLE_CARGO_MATTE and pr.shape == _SHAPE_TAPER_DRUM_Y:
                gran = (
                    torch.sin(hx * 16.7 + pr.mud_phase) * torch.cos(hy * 14.3 - hz * 18.1 + 0.5 * pr.mud_phase)
                    + 0.5 * torch.sin(hx * 31.1 - hy * 27.9 + pr.mud_phase)
                )
                alb = (alb * (1.0 + 0.11 * gran[None, :, :])).clamp(0.02, 0.99)
                nl_x = nl_x + 0.14 * gran
                nl_z = nl_z + 0.10 * gran.abs()

            elif pr.veh_idx == 0 and pr.role == _ROLE_ENV_SOLID and sy > 3.0 and pr.mud_phase > 0.5:
                w_style = int(round(float(pr.mud_phase)))
                if w_style == 1:
                    rib_w = 0.5 + 0.5 * torch.cos(hy * 26.0)
                    seam_y = ((hy.remainder(1.85)).abs() < 0.035) | ((hz.remainder(0.92)).abs() < 0.028)
                    mod = (0.76 + 0.30 * rib_w) * torch.where(seam_y, torch.full_like(rib_w, 0.42), torch.ones_like(rib_w))
                    alb = (alb * mod[None, :, :]).clamp(0.02, 0.96)
                elif w_style == 2:
                    rail_h = (hz.remainder(0.42)).abs() < 0.055
                    grate_v = (hy.remainder(0.16)).abs() < 0.042
                    on_bar = rail_h | grate_v
                    bar_col = (alb * 1.18).clamp(0.05, 0.92)
                    gap_col = (alb * 0.24).clamp(0.015, 0.35)
                    alb = torch.where(on_bar[None, :, :], bar_col, gap_col)
                elif w_style == 3:
                    plank_id = torch.floor((hz + sz) / 0.22)
                    plank_gap = ((hz + sz).remainder(0.22)) < 0.024
                    post_v = (hy.remainder(2.10)).abs() < 0.065
                    wood_var = 0.85 + 0.22 * torch.sin(plank_id * 3.7 + hy * 0.4)
                    wood_base = torch.stack(
                        [
                            (0.36 * wood_var).clamp(0.05, 0.72),
                            (0.23 * wood_var).clamp(0.04, 0.52),
                            (0.12 * wood_var).clamp(0.02, 0.32),
                        ],
                        dim=0,
                    )
                    dark_gap = wood_base * 0.22
                    alb = torch.where(plank_gap[None, :, :] & (~post_v[None, :, :]), dark_gap, wood_base)
                elif w_style == 4:
                    cell_y = (hy + sy).remainder(1.10)
                    cell_z = (hz + sz).remainder(0.62)
                    in_pane = (cell_y > 0.16) & (cell_y < 0.94) & (cell_z > 0.12) & (cell_z < 0.50)
                    pane_col = torch.tensor([0.05, 0.08, 0.12], device=dev, dtype=torch.float32)[:, None, None].expand_as(alb)
                    mull_col = (alb * 1.12).clamp(0.08, 0.88)
                    alb = torch.where(in_pane[None, :, :], pane_col, mull_col)
                    p_rough = torch.where(in_pane, torch.full_like(p_rough, 0.08), p_rough)
                    p_coat = torch.where(in_pane, torch.full_like(p_coat, 0.80), p_coat)

            # Substrate-decoupled conformal RGBA alpha-paint & 3D normal-relief typography (ZERO rectangular border!)
            if pr.decal_spec is not None and len(self.bank.alpha_decal_mips) > 0:
                d_idx, proj_face, u_c, v_c, u_hw, v_hh, rel_str = pr.decal_spec
                mips_dec = self.bank.alpha_decal_mips[int(d_idx) % len(self.bank.alpha_decal_mips)]
                uh_s = max(1e-3, float(u_hw))
                vh_s = max(1e-3, float(v_hh))
                if proj_face == 2:
                    # Cylindrical / conical wrap around drum/tank circumference
                    s_side = torch.where(hx >= 0.0, torch.ones_like(hx), -torch.ones_like(hx))
                    ang_cyl = torch.atan2(zn_l, xn_l.abs().clamp_min(1e-3))
                    u_dec = (s_side * yn_l - float(u_c)) / uh_s
                    v_dec = -((ang_cyl / 0.88) - float(v_c)) / vh_s
                    face_ok = (~is_cap) & (xn_l.abs() > 0.12) & (u_dec.abs() <= 1.0) & (v_dec.abs() <= 1.0)
                    tu_lx, tu_ly, tu_lz = torch.zeros_like(hx), s_side, torch.zeros_like(hx)
                    tv_lx = s_side * torch.sin(ang_cyl)
                    tv_ly = torch.zeros_like(hx)
                    tv_lz = -torch.cos(ang_cyl)
                elif proj_face == 1:
                    s_y = torch.where(hy >= 0.0, -torch.ones_like(hy), torch.ones_like(hy))
                    u_dec = (s_y * xn_l - float(u_c)) / uh_s
                    v_dec = -(zn_l - float(v_c)) / vh_s
                    face_ok = (nl_y.abs() > 0.60) & (u_dec.abs() <= 1.0) & (v_dec.abs() <= 1.0)
                    tu_lx, tu_ly, tu_lz = s_y, torch.zeros_like(hy), torch.zeros_like(hy)
                    tv_lx, tv_ly, tv_lz = torch.zeros_like(hy), torch.zeros_like(hy), -torch.ones_like(hy)
                else:
                    s_x = torch.where(hx >= 0.0, torch.ones_like(hx), -torch.ones_like(hx))
                    u_dec = (s_x * yn_l - float(u_c)) / uh_s
                    v_dec = -(zn_l - float(v_c)) / vh_s
                    face_ok = (nl_x.abs() > 0.58) & (u_dec.abs() <= 1.0) & (v_dec.abs() <= 1.0)
                    tu_lx, tu_ly, tu_lz = torch.zeros_like(hx), s_x, torch.zeros_like(hx)
                    tv_lx, tv_ly, tv_lz = torch.zeros_like(hx), torch.zeros_like(hx), -torch.ones_like(hx)

                if (valid_p & face_ok).any():
                    grid_d = torch.stack([u_dec.clamp(-1.0, 1.0), v_dec.clamp(-1.0, 1.0)], dim=-1).unsqueeze(0)
                    dec_rgba = self._sample_mip_trilinear(mips_dec, grid_d, px_span_pr * min(uh_s, vh_s))
                    alpha_d = torch.where(face_ok, dec_rgba[3].clamp(0.0, 1.0), torch.zeros_like(u_dec))
                    if float(rel_str) != 0.0:
                        mip_g = mips_dec[min(1, len(mips_dec) - 1)][:, 3:4]
                        eps_uv = 0.014
                        g_pu = torch.stack([(u_dec + eps_uv).clamp(-1.0, 1.0), v_dec.clamp(-1.0, 1.0)], dim=-1).unsqueeze(0)
                        g_mu = torch.stack([(u_dec - eps_uv).clamp(-1.0, 1.0), v_dec.clamp(-1.0, 1.0)], dim=-1).unsqueeze(0)
                        g_pv = torch.stack([u_dec.clamp(-1.0, 1.0), (v_dec + eps_uv).clamp(-1.0, 1.0)], dim=-1).unsqueeze(0)
                        g_mv = torch.stack([u_dec.clamp(-1.0, 1.0), (v_dec - eps_uv).clamp(-1.0, 1.0)], dim=-1).unsqueeze(0)
                        da_du = (
                            F.grid_sample(mip_g, g_pu, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
                            - F.grid_sample(mip_g, g_mu, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
                        )
                        da_dv = (
                            F.grid_sample(mip_g, g_pv, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
                            - F.grid_sample(mip_g, g_mv, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
                        )
                        rel_amp = abs(float(rel_str)) * 1.15 * face_ok.to(torch.float32)
                        nl_x = nl_x - rel_amp * (da_du * tu_lx + da_dv * tv_lx)
                        nl_y = nl_y - rel_amp * (da_du * tu_ly + da_dv * tv_ly)
                        nl_z = nl_z - rel_amp * (da_du * tu_lz + da_dv * tv_lz)
                    # Negative `rel_str` denotes unpainted substrate-colored 3D stamped/embossed relief lettering!
                    if float(rel_str) >= 0.0:
                        alb = alb * (1.0 - alpha_d[None]) + dec_rgba[:3] * alpha_d[None]
                        p_rough = p_rough * (1.0 - alpha_d) + 0.30 * alpha_d

            if pr.mud_level > 0.01:
                h_frac = ((hz + sz) / max(0.1, 2.0 * sz)).clamp(0.0, 1.0)
                splat = 0.5 + 0.5 * torch.sin(hx * 11.3 + pr.mud_phase) * torch.cos(hy * 9.7 - hz * 7.1 + pr.mud_phase)
                mud_a = (pr.mud_level * (1.0 - 0.68 * h_frac) * (0.45 + 0.75 * splat)).clamp(0.0, 0.88)
                mud_col = torch.tensor([0.11, 0.085, 0.058], device=dev, dtype=torch.float32)[:, None, None]
                alb = alb * (1.0 - mud_a[None]) + mud_col * mud_a[None]
                p_rough = p_rough * (1.0 - mud_a) + 0.88 * mud_a
                p_coat = p_coat * (1.0 - mud_a)

            nw_len = torch.sqrt(nl_x * nl_x + nl_y * nl_y + nl_z * nl_z).clamp_min(1e-5)
            nl_x, nl_y, nl_z = nl_x / nw_len, nl_y / nw_len, nl_z / nw_len
            nw_x = Rw2l_t[0, 0] * nl_x + Rw2l_t[1, 0] * nl_y + Rw2l_t[2, 0] * nl_z
            nw_y = Rw2l_t[0, 1] * nl_x + Rw2l_t[1, 1] * nl_y + Rw2l_t[2, 1] * nl_z
            nw_z = Rw2l_t[0, 2] * nl_x + Rw2l_t[1, 2] * nl_y + Rw2l_t[2, 2] * nl_z

            is_lamp = pr.role in (_ROLE_HEADLIGHT, _ROLE_TAILLIGHT)
            if pr.role == _ROLE_HEADLIGHT:
                emit_val = 6.4 if is_ir else (5.8 if is_night else 1.4)
            elif pr.role == _ROLE_TAILLIGHT:
                emit_val = 3.4 if is_ir else (2.8 if is_night else 0.9)
            else:
                emit_val = 0.0
            nir_sc = 0.45 if pr.role == _ROLE_CABIN_GLASS else (0.70 if pr.role == _ROLE_TIRE else 1.0)
            hk_val = 4 if is_lamp else 2

            alb_buf[:, v0:v1, u0:u1] = torch.where(valid_p[None, :, :], alb, alb_buf[:, v0:v1, u0:u1])
            nw_x_buf[v0:v1, u0:u1] = torch.where(valid_p, nw_x, nw_x_buf[v0:v1, u0:u1])
            nw_y_buf[v0:v1, u0:u1] = torch.where(valid_p, nw_y, nw_y_buf[v0:v1, u0:u1])
            nw_z_buf[v0:v1, u0:u1] = torch.where(valid_p, nw_z, nw_z_buf[v0:v1, u0:u1])
            ng_x_buf[v0:v1, u0:u1] = torch.where(valid_p, ng_x_p, ng_x_buf[v0:v1, u0:u1])
            ng_y_buf[v0:v1, u0:u1] = torch.where(valid_p, ng_y_p, ng_y_buf[v0:v1, u0:u1])
            ng_z_buf[v0:v1, u0:u1] = torch.where(valid_p, ng_z_p, ng_z_buf[v0:v1, u0:u1])
            rough_buf[v0:v1, u0:u1] = torch.where(valid_p, p_rough, rough_buf[v0:v1, u0:u1])
            metal_buf[v0:v1, u0:u1] = torch.where(valid_p, p_metal, metal_buf[v0:v1, u0:u1])
            f0_buf[v0:v1, u0:u1] = torch.where(valid_p, p_f0, f0_buf[v0:v1, u0:u1])
            coat_buf[v0:v1, u0:u1] = torch.where(valid_p, p_coat, coat_buf[v0:v1, u0:u1])
            retro_buf[v0:v1, u0:u1] = torch.where(valid_p, p_retro, retro_buf[v0:v1, u0:u1])
            emit_buf[v0:v1, u0:u1] = torch.where(valid_p, torch.full_like(emit_buf[v0:v1, u0:u1], emit_val), emit_buf[v0:v1, u0:u1])
            nir_scale_buf[v0:v1, u0:u1] = torch.where(valid_p, torch.full_like(nir_scale_buf[v0:v1, u0:u1], nir_sc), nir_scale_buf[v0:v1, u0:u1])
            green_plate_buf[v0:v1, u0:u1] = torch.where(valid_p, torch.zeros_like(green_plate_buf[v0:v1, u0:u1]), green_plate_buf[v0:v1, u0:u1])
            hit_kind_buf[v0:v1, u0:u1] = torch.where(valid_p, torch.full_like(hit_kind_buf[v0:v1, u0:u1], hk_val), hit_kind_buf[v0:v1, u0:u1])
            prim_id_buf[v0:v1, u0:u1] = torch.where(valid_p, torch.full_like(prim_id_buf[v0:v1, u0:u1], p_idx), prim_id_buf[v0:v1, u0:u1])
            quad_buf[v0:v1, u0:u1] = torch.where(valid_p, torch.full_like(quad_buf[v0:v1, u0:u1], -1), quad_buf[v0:v1, u0:u1])
            t_best[v0:v1, u0:u1] = torch.where(valid_p, t_hit, t_best[v0:v1, u0:u1])
            t_geom[v0:v1, u0:u1] = torch.where(valid_p, t_hit, t_geom[v0:v1, u0:u1])
            veh_buf[v0:v1, u0:u1] = torch.where(valid_p, torch.full_like(veh_buf[v0:v1, u0:u1], pr.veh_idx), veh_buf[v0:v1, u0:u1])
            if record_masks and role_buf is not None:
                role_buf[v0:v1, u0:u1] = torch.where(valid_p, torch.full_like(role_buf[v0:v1, u0:u1], pr.role), role_buf[v0:v1, u0:u1])

        # Pass 1C: Exact 3D Planar Quads & Substrate-Decoupled Frameless Alpha Stencils (Zero analytic overhang_shadow!)
        for q_idx, qd in enumerate(quads):
            to_cam = cam_o - qd.center
            dist_c = float(np.linalg.norm(to_cam))
            if dist_c < 0.20 or float(np.dot(qd.normal, to_cam)) / dist_c <= 0.06:
                continue
            uv_q_clip = self._clip_quad_to_near_plane_uv(qd, cam_o, R_cam, fx, fy, cx, cy, z_near=0.16)
            if uv_q_clip.shape[0] == 0:
                continue
            u0 = max(0, int(math.floor(float(uv_q_clip[:, 0].min()) - 2.0)))
            u1 = min(w, int(math.ceil(float(uv_q_clip[:, 0].max()) + 2.0)))
            v0 = max(0, int(math.floor(float(uv_q_clip[:, 1].min()) - 2.0)))
            v1 = min(h, int(math.ceil(float(uv_q_clip[:, 1].max()) + 2.0)))
            if u1 <= u0 or v1 <= v0:
                continue
            px_span = max(float(uv_q_clip[:, 0].max() - uv_q_clip[:, 0].min()), float(uv_q_clip[:, 1].max() - uv_q_clip[:, 1].min()))

            sub_dx = dx_w[v0:v1, u0:u1]
            sub_dy = dy_w[v0:v1, u0:u1]
            sub_dz = dz_w[v0:v1, u0:u1]
            n_t = torch.from_numpy(qd.normal).to(device=dev, dtype=torch.float32)
            u_t = torch.from_numpy(qd.u_axis).to(device=dev, dtype=torch.float32)
            v_t = torch.from_numpy(qd.v_axis).to(device=dev, dtype=torch.float32)
            c_t = torch.from_numpy(qd.center).to(device=dev, dtype=torch.float32)

            denom = n_t[0] * sub_dx + n_t[1] * sub_dy + n_t[2] * sub_dz
            num = float(torch.dot(n_t, c_t - cam_o_t).item())
            lam_q = num / torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, -1e-6))

            px_w = cam_o_t[0] + lam_q * sub_dx - c_t[0]
            py_w = cam_o_t[1] + lam_q * sub_dy - c_t[1]
            pz_w = cam_o_t[2] + lam_q * sub_dz - c_t[2]

            xi = (px_w * u_t[0] + py_w * u_t[1] + pz_w * u_t[2]) / max(1e-4, qd.half_w)
            eta = (px_w * v_t[0] + py_w * v_t[1] + pz_w * v_t[2]) / max(1e-4, qd.half_h)

            if not qd.has_backing_substrate:
                # Frameless painted alpha stencil applied onto an existing underlying surface (zero rectangular border!)
                in_stencil = (
                    (denom < -1e-5)
                    & (lam_q > 0.16)
                    & (xi.abs() <= 1.0)
                    & (eta.abs() <= 1.0)
                    & (hit_kind_buf[v0:v1, u0:u1] > 0)
                    & ((lam_q - t_best[v0:v1, u0:u1]).abs() <= 0.12)
                )
                if not in_stencil.any():
                    continue
                kind_s, tidx_s = qd.tex_key
                mips_st = (
                    self.bank.alpha_decal_mips[tidx_s % len(self.bank.alpha_decal_mips)]
                    if kind_s == "alpha_decal" and len(self.bank.alpha_decal_mips) > 0
                    else self.bank.sign_mips[tidx_s % len(self.bank.sign_mips)]
                )
                grid_s = torch.stack([xi.clamp(-1.0, 1.0), eta.clamp(-1.0, 1.0)], dim=-1).unsqueeze(0)
                samp_s = self._sample_mip_trilinear(mips_st, grid_s, px_span)
                alpha_s = (samp_s[3].clamp(0.0, 1.0) if samp_s.shape[0] >= 4 else torch.ones_like(xi)) * in_stencil.to(torch.float32)
                if (alpha_s > 1e-4).any():
                    cur_alb = alb_buf[:, v0:v1, u0:u1]
                    alb_buf[:, v0:v1, u0:u1] = cur_alb * (1.0 - alpha_s[None]) + samp_s[:3] * alpha_s[None]
                    rough_buf[v0:v1, u0:u1] = rough_buf[v0:v1, u0:u1] * (1.0 - alpha_s) + float(qd.roughness) * alpha_s
                continue

            # Coplanar +3cm tolerance ONLY against the plate's OWN parent vehicle body (never nearer pillars/signs/trucks!)
            if qd.veh_idx > 0:
                same_owner = veh_buf[v0:v1, u0:u1] == qd.veh_idx
                depth_tol = torch.where(same_owner, torch.full_like(lam_q, 0.03), torch.zeros_like(lam_q))
            else:
                depth_tol = torch.zeros_like(lam_q)

            in_quad = (
                (denom < -1e-5)
                & (lam_q > 0.16)
                & (xi.abs() <= 1.0)
                & (eta.abs() <= 1.0)
                & (lam_q <= t_best[v0:v1, u0:u1] + depth_tol)
            )
            if not in_quad.any():
                continue

            kind, tidx = qd.tex_key
            if kind == "plate_front":
                mips_any: tuple[Tensor, ...] = self.bank.front_plate_mips[tidx]
            elif kind == "plate_rear":
                mips_any = self.bank.rear_plate_mips[tidx]
            elif kind == "sign":
                mips_any = self.bank.sign_mips[tidx]
            elif kind == "alpha_decal":
                mips_any = self.bank.alpha_decal_mips[tidx]
            else:
                mips_any = self.bank.sticker_mips[tidx]

            grid = torch.stack([xi.clamp(-1.0, 1.0), eta.clamp(-1.0, 1.0)], dim=-1).unsqueeze(0)
            sampled = self._sample_mip_trilinear(mips_any, grid, px_span)[:3]

            qn_x = torch.full_like(xi, float(n_t[0]))
            qn_y = torch.full_like(xi, float(n_t[1]))
            qn_z = torch.full_like(xi, float(n_t[2]))
            if qd.normal_undul_amp > 0.0:
                ua = float(qd.normal_undul_amp)
                du_w = ua * (torch.sin(xi * 4.3 + eta * 2.9 + float(q_idx)) + 0.4 * torch.cos(xi * 9.7 - eta * 7.1))
                dv_w = ua * (torch.cos(eta * 5.1 - xi * 3.4 + float(q_idx)) + 0.4 * torch.sin(eta * 11.3 + xi * 6.2))
                qn_x = qn_x + du_w * u_t[0] + dv_w * v_t[0]
                qn_y = qn_y + du_w * u_t[1] + dv_w * v_t[1]
                qn_z = qn_z + du_w * u_t[2] + dv_w * v_t[2]
                qn_len = torch.sqrt(qn_x * qn_x + qn_y * qn_y + qn_z * qn_z).clamp_min(1e-5)
                qn_x, qn_y, qn_z = qn_x / qn_len, qn_y / qn_len, qn_z / qn_len

            alb_buf[:, v0:v1, u0:u1] = torch.where(in_quad[None, :, :], sampled, alb_buf[:, v0:v1, u0:u1])
            nw_x_buf[v0:v1, u0:u1] = torch.where(in_quad, qn_x, nw_x_buf[v0:v1, u0:u1])
            nw_y_buf[v0:v1, u0:u1] = torch.where(in_quad, qn_y, nw_y_buf[v0:v1, u0:u1])
            nw_z_buf[v0:v1, u0:u1] = torch.where(in_quad, qn_z, nw_z_buf[v0:v1, u0:u1])
            ng_x_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(xi, float(n_t[0])), ng_x_buf[v0:v1, u0:u1])
            ng_y_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(xi, float(n_t[1])), ng_y_buf[v0:v1, u0:u1])
            ng_z_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(xi, float(n_t[2])), ng_z_buf[v0:v1, u0:u1])
            rough_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(xi, float(qd.roughness)), rough_buf[v0:v1, u0:u1])
            metal_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(xi, float(qd.metallic)), metal_buf[v0:v1, u0:u1])
            f0_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(xi, float(qd.specular_f0)), f0_buf[v0:v1, u0:u1])
            coat_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(xi, float(qd.clearcoat)), coat_buf[v0:v1, u0:u1])
            retro_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(xi, float(qd.retro_gain)), retro_buf[v0:v1, u0:u1])
            emit_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.zeros_like(xi), emit_buf[v0:v1, u0:u1])
            nir_scale_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.ones_like(xi), nir_scale_buf[v0:v1, u0:u1])
            green_plate_buf[v0:v1, u0:u1] = torch.where(
                in_quad,
                torch.full_like(green_plate_buf[v0:v1, u0:u1], bool(qd.is_green_plate)),
                green_plate_buf[v0:v1, u0:u1],
            )
            hit_kind_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(hit_kind_buf[v0:v1, u0:u1], 3), hit_kind_buf[v0:v1, u0:u1])
            prim_id_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(prim_id_buf[v0:v1, u0:u1], -1), prim_id_buf[v0:v1, u0:u1])
            t_best[v0:v1, u0:u1] = torch.where(in_quad, torch.minimum(lam_q, t_best[v0:v1, u0:u1]), t_best[v0:v1, u0:u1])
            t_geom[v0:v1, u0:u1] = torch.where(in_quad, lam_q, t_geom[v0:v1, u0:u1])
            # ALWAYS update both quad_buf and veh_buf (standalone signs have qd.veh_idx == 0, clearing occluded vehicle pixels!)
            quad_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(quad_buf[v0:v1, u0:u1], q_idx), quad_buf[v0:v1, u0:u1])
            veh_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(veh_buf[v0:v1, u0:u1], qd.veh_idx), veh_buf[v0:v1, u0:u1])
            if record_masks and role_buf is not None:
                q_role = _ROLE_BODY if qd.veh_idx > 0 else _ROLE_ENV_SOLID
                role_buf[v0:v1, u0:u1] = torch.where(in_quad, torch.full_like(role_buf[v0:v1, u0:u1], q_role), role_buf[v0:v1, u0:u1])

        # Pass 2: True 3D 1-Ray Shadow Visibility `V(x, omega_i)` onto Ground, 3D Bodies, AND Planar Quads!
        shadow_vis = torch.ones((h, w), device=dev, dtype=torch.float32)
        surf_mask = (hit_kind_buf > 0) & (hit_kind_buf < 4)
        if (not is_ir) and (sd_z > 0.01) and surf_mask.any():
            ndotg_all = ng_x_buf * sd_x + ng_y_buf * sd_y + ng_z_buf * sd_z
            lit_cand = surf_mask & (ndotg_all > -0.05)
            if lit_cand.any():
                wx_hit = cam_o_t[0] + t_geom * dx_w + 0.006 * ng_x_buf
                wy_hit = cam_o_t[1] + t_geom * dy_w + 0.006 * ng_y_buf
                wz_hit = cam_o_t[2] + t_geom * dz_w + 0.006 * ng_z_buf
                sd_np = np.array([sd_x, sd_y, sd_z], dtype=np.float32)
                I3_np = np.eye(3, dtype=np.float32)
                inv_sd_z = 1.0 / max(0.08, sd_z)

                for p_idx, pr in enumerate(prims):
                    if pr.role in (_ROLE_HEADLIGHT, _ROLE_TAILLIGHT):
                        continue
                    c_np = np.array(pr.center, dtype=np.float32)
                    he_np = np.array(pr.half_ext, dtype=np.float32)
                    aabb_he = np.abs(pr.rot_w2l.T) @ he_np
                    z_top = float(c_np[2] + aabb_he[2])
                    if z_top < 0.01:
                        continue
                    l_throw = min(55.0, max(0.4, (z_top + 0.25) * inv_sd_z))
                    lo_sv = np.minimum(c_np - aabb_he, c_np - aabb_he - l_throw * sd_np) - 0.03
                    hi_sv = np.maximum(c_np + aabb_he, c_np + aabb_he - l_throw * sd_np) + 0.03
                    sv_c = 0.5 * (lo_sv + hi_sv)
                    sv_he = 0.5 * (hi_sv - lo_sv)
                    uv_sv = self._clip_obb_to_near_plane_uv(
                        sv_c, I3_np, (float(sv_he[0]), float(sv_he[1]), float(sv_he[2])),
                        cam_o, R_cam, fx, fy, cx, cy, z_near=0.15,
                    )
                    if uv_sv.shape[0] == 0:
                        continue
                    su0 = max(0, int(math.floor(float(uv_sv[:, 0].min()) - 1.0)))
                    su1 = min(w, int(math.ceil(float(uv_sv[:, 0].max()) + 1.0)))
                    sv0 = max(0, int(math.floor(float(uv_sv[:, 1].min()) - 1.0)))
                    sv1 = min(h, int(math.ceil(float(uv_sv[:, 1].max()) + 1.0)))
                    if su1 <= su0 or sv1 <= sv0:
                        continue

                    sub_vis = shadow_vis[sv0:sv1, su0:su1]
                    sub_cand = lit_cand[sv0:sv1, su0:su1] & (sub_vis > 0.0) & (wz_hit[sv0:sv1, su0:su1] <= z_top + 0.02)
                    if not sub_cand.any():
                        continue

                    Rw = pr.rot_w2l
                    sdl_x = float(Rw[0, 0] * sd_x + Rw[0, 1] * sd_y + Rw[0, 2] * sd_z)
                    sdl_y = float(Rw[1, 0] * sd_x + Rw[1, 1] * sd_y + Rw[1, 2] * sd_z)
                    sdl_z = float(Rw[2, 0] * sd_x + Rw[2, 1] * sd_y + Rw[2, 2] * sd_z)
                    dx_p = wx_hit[sv0:sv1, su0:su1] - float(c_np[0])
                    dy_p = wy_hit[sv0:sv1, su0:su1] - float(c_np[1])
                    dz_p = wz_hit[sv0:sv1, su0:su1] - float(c_np[2])
                    ox_s = float(Rw[0, 0]) * dx_p + float(Rw[0, 1]) * dy_p + float(Rw[0, 2]) * dz_p
                    oy_s = float(Rw[1, 0]) * dx_p + float(Rw[1, 1]) * dy_p + float(Rw[1, 2]) * dz_p
                    oz_s = float(Rw[2, 0]) * dx_p + float(Rw[2, 1]) * dy_p + float(Rw[2, 2]) * dz_p

                    sx, sy, sz = pr.half_ext
                    t_self = torch.where(
                        prim_id_buf[sv0:sv1, su0:su1] == p_idx,
                        torch.full_like(ox_s, 0.040),
                        torch.full_like(ox_s, 0.005),
                    )

                    if pr.shape in (_SHAPE_OBB, _SHAPE_SLOPED_CAB):
                        inv_sx = (1.0 / sdl_x) if abs(sdl_x) > 1e-6 else math.copysign(1e6, sdl_x if sdl_x != 0.0 else 1.0)
                        inv_sy = (1.0 / sdl_y) if abs(sdl_y) > 1e-6 else math.copysign(1e6, sdl_y if sdl_y != 0.0 else 1.0)
                        inv_sz = (1.0 / sdl_z) if abs(sdl_z) > 1e-6 else math.copysign(1e6, sdl_z if sdl_z != 0.0 else 1.0)
                        tx0, tx1 = (-sx - ox_s) * inv_sx, (sx - ox_s) * inv_sx
                        ty0, ty1 = (-sy - oy_s) * inv_sy, (sy - oy_s) * inv_sy
                        tz0, tz1 = (-sz - oz_s) * inv_sz, (sz - oz_s) * inv_sz
                        tmin_s = torch.maximum(torch.maximum(torch.minimum(tx0, tx1), torch.minimum(ty0, ty1)), torch.minimum(tz0, tz1))
                        tmax_s = torch.minimum(torch.minimum(torch.maximum(tx0, tx1), torch.maximum(ty0, ty1)), torch.maximum(tz0, tz1))
                        if pr.shape == _SHAPE_SLOPED_CAB:
                            ny_s = float(pr.slope_param)
                            nz_s = math.sqrt(max(0.1, 1.0 - ny_s * ny_s))
                            d_s = ny_s * (sy * 0.72) + nz_s * (sz * 0.78)
                            den_s = ny_s * sdl_y + nz_s * sdl_z
                            num_s = d_s - (ny_s * oy_s + nz_s * oz_s)
                            if abs(den_s) > 1e-6:
                                t_sp = num_s / den_s
                                if den_s < 0.0:
                                    tmin_s = torch.maximum(tmin_s, t_sp)
                                else:
                                    tmax_s = torch.minimum(tmax_s, t_sp)
                            else:
                                tmax_s = torch.where(num_s < 0.0, torch.full_like(tmax_s, -1.0), tmax_s)
                        occ_p = sub_cand & (tmax_s >= torch.maximum(tmin_s, t_self))

                    elif pr.shape == _SHAPE_CYLINDER_X:
                        oy_n, oz_n = oy_s / sy, oz_s / sz
                        dy_n, dz_n = sdl_y / sy, sdl_z / sz
                        qa = max(1e-6, dy_n * dy_n + dz_n * dz_n)
                        qb = oy_n * dy_n + oz_n * dz_n
                        qc = oy_n * oy_n + oz_n * oz_n - 1.0
                        disc = qb * qb - qa * qc
                        sq = torch.sqrt(disc.clamp_min(0.0))
                        tc0 = (-qb - sq) / qa
                        tc1 = (-qb + sq) / qa
                        inv_sx = (1.0 / sdl_x) if abs(sdl_x) > 1e-6 else math.copysign(1e6, sdl_x if sdl_x != 0.0 else 1.0)
                        tx0, tx1 = (-sx - ox_s) * inv_sx, (sx - ox_s) * inv_sx
                        tmin_s = torch.maximum(torch.where(disc >= 0.0, tc0, torch.full_like(tc0, 1e6)), torch.minimum(tx0, tx1))
                        tmax_s = torch.minimum(torch.where(disc >= 0.0, tc1, torch.full_like(tc1, -1e6)), torch.maximum(tx0, tx1))
                        hit_cyl = (disc >= 0.0) & (tmax_s >= torch.maximum(tmin_s, t_self))
                        if pr.radial_n >= 3 and hit_cyl.any():
                            n_fold = float(pr.radial_n)
                            sector_w = (2.0 * math.pi) / n_fold
                            half_sec = 0.5 * sector_w
                            r_ring = 0.45
                            r_hole = min(0.115, 0.56 * r_ring * math.sin(half_sec))
                            hx0 = ox_s + tmin_s * sdl_x
                            hy0 = (oy_s + tmin_s * sdl_y) / sy
                            hz0 = (oz_s + tmin_s * sdl_z) / sz
                            rr0 = torch.sqrt(hy0 * hy0 + hz0 * hz0)
                            th0 = torch.remainder(torch.atan2(hz0, hy0) + pr.mud_phase + half_sec, sector_w) - half_sec
                            du0 = rr0 * torch.cos(th0) - r_ring
                            dv0 = rr0 * torch.sin(th0)
                            perf0 = (hx0.abs() >= sx * 0.92) & ((du0 * du0 + dv0 * dv0) < (r_hole * r_hole))
                            hx1 = ox_s + tmax_s * sdl_x
                            hy1 = (oy_s + tmax_s * sdl_y) / sy
                            hz1 = (oz_s + tmax_s * sdl_z) / sz
                            rr1 = torch.sqrt(hy1 * hy1 + hz1 * hz1)
                            th1 = torch.remainder(torch.atan2(hz1, hy1) + pr.mud_phase + half_sec, sector_w) - half_sec
                            du1 = rr1 * torch.cos(th1) - r_ring
                            dv1 = rr1 * torch.sin(th1)
                            perf1 = (hx1.abs() >= sx * 0.92) & ((du1 * du1 + dv1 * dv1) < (r_hole * r_hole))
                            hit_cyl = hit_cyl & (~(perf0 & perf1))
                        occ_p = sub_cand & hit_cyl

                    elif pr.shape == _SHAPE_FLEX_SHEET:
                        t_sh_f, ok_sh_f = self._intersect_flex_sheet_local(
                            ox_s, oy_s, oz_s, sdl_x, sdl_y, sdl_z,
                            sx, sy, sz, pr.wave_amp, pr.wave_freq, pr.mud_phase, t_near=0.005,
                        )
                        occ_p = sub_cand & ok_sh_f & (t_sh_f >= t_self)

                    else:
                        # `_SHAPE_TAPER_DRUM_Y` shadow occlusion
                        taper = float(np.clip(pr.slope_param, 0.0, 0.65))
                        ox_n, oy_n, oz_n = ox_s / sx, oy_s / sy, oz_s / sz
                        dx_n, dy_n, dz_n = sdl_x / sx, sdl_y / sy, sdl_z / sz
                        occ_drum = torch.zeros_like(sub_cand)
                        for s_h in (1.0, -1.0):
                            k_c = taper * s_h
                            qa = dx_n * dx_n + dz_n * dz_n - (k_c * dy_n) * (k_c * dy_n)
                            qb = ox_n * dx_n + oz_n * dz_n + k_c * dy_n * (1.0 - k_c * oy_n)
                            qc = ox_n * ox_n + oz_n * oz_n - (1.0 - k_c * oy_n) * (1.0 - k_c * oy_n)
                            disc = qb * qb - qa * qc
                            sq = torch.sqrt(disc.clamp_min(0.0))
                            s_qb = torch.where(qb >= 0.0, torch.ones_like(qb), -torch.ones_like(qb))
                            q_cit = -(qb + s_qb * sq)
                            safe_q = torch.where(q_cit.abs() > 1e-12, q_cit, torch.copysign(torch.full_like(q_cit, 1e-12), q_cit))
                            safe_a = float(qa if abs(qa) > 1e-12 else math.copysign(1e-12, qa if qa != 0.0 else 1.0))
                            tr1 = qc / safe_q
                            tr2 = q_cit / safe_a
                            for tr, cond_ok in ((tr1, q_cit.abs() > 1e-12), (tr2, abs(qa) > 1e-12)):
                                yn_r = oy_n + tr * dy_n
                                in_half = (yn_r >= -1e-4) & (yn_r <= 1.0) if s_h > 0.0 else (yn_r <= 1e-4) & (yn_r >= -1.0)
                                occ_drum = occ_drum | ((disc >= 0.0) & cond_ok & in_half & (tr >= t_self))
                        if abs(dy_n) > 1e-6:
                            r_cap2 = (1.0 - taper) * (1.0 - taper)
                            for s_cap in (1.0, -1.0):
                                tc_p = (s_cap - oy_n) / dy_n
                                xc_p = ox_n + tc_p * dx_n
                                zc_p = oz_n + tc_p * dz_n
                                occ_drum = occ_drum | ((tc_p >= t_self) & ((xc_p * xc_p + zc_p * zc_p) <= r_cap2))
                        occ_p = sub_cand & occ_drum

                    if occ_p.any():
                        shadow_vis[sv0:sv1, su0:su1] = torch.where(occ_p, torch.zeros_like(sub_vis), sub_vis)

                # Standalone solid scene signboards/placards also cast genuine 3D planar-quad shadows!
                for q_idx, qd in enumerate(quads):
                    if qd.veh_idx != 0 or (not qd.has_backing_substrate):
                        continue
                    den_qs = float(qd.normal[0] * sd_x + qd.normal[1] * sd_y + qd.normal[2] * sd_z)
                    if abs(den_qs) <= 1e-5:
                        continue
                    c_q = qd.center.astype(np.float32)
                    q_rad = qd.half_w + qd.half_h + 0.03
                    z_top_q = float(c_q[2]) + q_rad
                    l_th_q = min(45.0, max(0.4, (z_top_q + 0.20) * inv_sd_z))
                    lo_q = np.minimum(c_q - q_rad, c_q - q_rad - l_th_q * sd_np)
                    hi_q = np.maximum(c_q + q_rad, c_q + q_rad - l_th_q * sd_np)
                    uv_sq = self._clip_obb_to_near_plane_uv(
                        0.5 * (lo_q + hi_q), I3_np,
                        (float(0.5 * (hi_q[0] - lo_q[0])), float(0.5 * (hi_q[1] - lo_q[1])), float(0.5 * (hi_q[2] - lo_q[2]))),
                        cam_o, R_cam, fx, fy, cx, cy, z_near=0.15,
                    )
                    if uv_sq.shape[0] == 0:
                        continue
                    su0 = max(0, int(math.floor(float(uv_sq[:, 0].min()) - 1.0)))
                    su1 = min(w, int(math.ceil(float(uv_sq[:, 0].max()) + 1.0)))
                    sv0 = max(0, int(math.floor(float(uv_sq[:, 1].min()) - 1.0)))
                    sv1 = min(h, int(math.ceil(float(uv_sq[:, 1].max()) + 1.0)))
                    if su1 <= su0 or sv1 <= sv0:
                        continue
                    sub_vis = shadow_vis[sv0:sv1, su0:su1]
                    sub_cand = lit_cand[sv0:sv1, su0:su1] & (sub_vis > 0.0) & (quad_buf[sv0:sv1, su0:su1] != q_idx)
                    if not sub_cand.any():
                        continue
                    rx_q = float(c_q[0]) - wx_hit[sv0:sv1, su0:su1]
                    ry_q = float(c_q[1]) - wy_hit[sv0:sv1, su0:su1]
                    rz_q = float(c_q[2]) - wz_hit[sv0:sv1, su0:su1]
                    t_qs = (rx_q * float(qd.normal[0]) + ry_q * float(qd.normal[1]) + rz_q * float(qd.normal[2])) / den_qs
                    px_s = t_qs * sd_x - rx_q
                    py_s = t_qs * sd_y - ry_q
                    pz_s = t_qs * sd_z - rz_q
                    xi_s = (px_s * float(qd.u_axis[0]) + py_s * float(qd.u_axis[1]) + pz_s * float(qd.u_axis[2])) / max(1e-4, qd.half_w)
                    eta_s = (px_s * float(qd.v_axis[0]) + py_s * float(qd.v_axis[1]) + pz_s * float(qd.v_axis[2])) / max(1e-4, qd.half_h)
                    occ_q = sub_cand & (t_qs > 0.008) & (xi_s.abs() <= 1.0) & (eta_s.abs() <= 1.0)
                    if occ_q.any():
                        shadow_vis[sv0:sv1, su0:su1] = torch.where(occ_q, torch.zeros_like(sub_vis), sub_vis)

        if record_masks:
            self._last_mid_shadow_vis = shadow_vis

        # Pass 3: Unified Energy-Consistent Cook-Torrance GGX + Clearcoat + Horizon/Ground Reflection IBL + Coaxial NIR
        any_surf = hit_kind_buf > 0
        if any_surf.any():
            vx = -rd_x
            vy = -rd_y
            vz = -rd_z
            ndotv = (nw_x_buf * vx + nw_y_buf * vy + nw_z_buf * vz).clamp(1e-4, 1.0)
            dist_m = (t_geom * d_norm).clamp(1.2, 200.0)

            if is_ir:
                # Strictly monochrome 850nm/940nm coaxial NIR LED reflectance (bit-exact R == G == B!)
                nir_alb = (0.42 * alb_buf[0] + 0.38 * alb_buf[1] + 0.20 * alb_buf[2]) * nir_scale_buf
                if green_plate_buf.any():
                    whiteness = torch.minimum(alb_buf[0], alb_buf[2])
                    nir_green = 0.22 * (0.48 * alb_buf[0] + 0.24 * alb_buf[1] + 0.28 * alb_buf[2]) + 1.15 * whiteness
                    nir_alb = torch.where(green_plate_buf, nir_green, nir_alb)

                ir_road = 0.015 + 2.4 * ir_ang_gain / (1.0 + 0.028 * dist_m * dist_m)
                ir_prim = (0.012 + 3.2 * ir_ang_gain / (1.0 + 0.025 * dist_m * dist_m)) * (0.20 + 0.80 * ndotv)
                coax_q = (retro_buf * 5.8 * ir_ang_gain / (1.0 + 0.030 * dist_m * dist_m)).clamp(0.06, 3.2)
                spec_ir = ndotv.pow(12.0) * (0.15 + 0.55 * (1.0 - rough_buf)) * 0.42

                lum_ir = torch.where(
                    hit_kind_buf == 1,
                    (0.30 * alb_buf[0] + 0.45 * alb_buf[1] + 0.25 * alb_buf[2]) * ir_road,
                    torch.where(
                        hit_kind_buf == 3,
                        nir_alb * (0.04 + coax_q),
                        torch.where(
                            hit_kind_buf == 4,
                            0.90 * emit_buf,
                            nir_alb * ir_prim + spec_ir,
                        ),
                    ),
                )
                hdr = torch.where(any_surf[None, :, :], lum_ir[None, :, :].expand(3, h, w), hdr)
            else:
                # Halfway vector `h = normalize(v + omega_i)`
                hx_h = vx + sd_x
                hy_h = vy + sd_y
                hz_h = vz + sd_z
                h_len = torch.sqrt(hx_h * hx_h + hy_h * hy_h + hz_h * hz_h).clamp_min(1e-5)
                hx_h, hy_h, hz_h = hx_h / h_len, hy_h / h_len, hz_h / h_len

                ndotl = (nw_x_buf * sd_x + nw_y_buf * sd_y + nw_z_buf * sd_z).clamp(0.0, 1.0)
                ndotg = ng_x_buf * sd_x + ng_y_buf * sd_y + ng_z_buf * sd_z
                geom_term = ((ndotg + 0.04) / 0.08).clamp(0.0, 1.0)
                vis_direct = shadow_vis * ndotl * geom_term

                ndoth = (nw_x_buf * hx_h + nw_y_buf * hy_h + nw_z_buf * hz_h).clamp(0.0, 1.0)
                vdoth = (vx * hx_h + vy * hy_h + vz * hz_h).clamp(0.0, 1.0)
                pow5_vh = (1.0 - vdoth).pow(5.0)
                pow5_nv = (1.0 - ndotv).pow(5.0)

                spec_gate = (metal_buf + (1.0 - metal_buf) * (f0_buf / 0.04)).clamp(0.0, 1.0)
                F0_rgb = f0_buf[None] * (1.0 - metal_buf[None]) + alb_buf * metal_buf[None]
                F_spec = F0_rgb + (1.0 - F0_rgb) * pow5_vh[None] * spec_gate[None]

                # Trowbridge-Reitz GGX NDF + Height-Correlated Schlick-Smith Geometry
                alpha_g = rough_buf.clamp(0.045, 0.99).pow(2.0)
                alpha2 = alpha_g * alpha_g
                den_d = (ndoth * ndoth * (alpha2 - 1.0) + 1.0).clamp_min(1e-4)
                D_ggx = alpha2 / (math.pi * den_d * den_d)
                k_g = ((rough_buf + 1.0) * (rough_buf + 1.0)) * 0.125
                vis_smith = 0.25 / (((ndotv * (1.0 - k_g) + k_g) * (ndotl * (1.0 - k_g) + k_g)).clamp_min(1e-4))
                ggx_lobe = (D_ggx * vis_smith).clamp(0.0, 18.0)

                # Secondary Automotive / Water-Film Clearcoat GGX Lobe (`alpha_c = 0.08^2 = 0.0064`)
                F_coat = (0.04 + 0.96 * pow5_vh) * coat_buf
                F_coat_v = (0.04 + 0.96 * pow5_nv) * coat_buf
                den_dc = (ndoth * ndoth * (0.0064 - 1.0) + 1.0).clamp_min(1e-4)
                D_coat = 0.0064 / (math.pi * den_dc * den_dc)
                k_c = 0.1458
                coat_lobe = F_coat * (D_coat * 0.25 / (((ndotv * (1.0 - k_c) + k_c) * (ndotl * (1.0 - k_c) + k_c)).clamp_min(1e-4))).clamp(0.0, 22.0)

                # Energy-Conserving Diffuse Lobe + Equal-Energy D65 Direct Solar Radiance
                kd = (1.0 - F_spec) * (1.0 - metal_buf[None]) * (1.0 - F_coat_v[None])
                sun_str = 0.22 if is_night else 1.38
                direct_hdr = sun_str * vis_direct[None] * (kd * alb_buf + (1.0 - F_coat_v[None]) * F_spec * ggx_lobe[None] + coat_lobe[None])

                # Reflection-Vector Environment IBL (`r = rd + 2*(n.v)*n`) + Hemisphere Sky/Ground Diffuse
                if env_energy_gate > 1e-6:
                    rx_r = rd_x + 2.0 * ndotv * nw_x_buf
                    ry_r = rd_y + 2.0 * ndotv * nw_y_buf
                    rz_r = rd_z + 2.0 * ndotv * nw_z_buf
                    refl_env_rgb = _eval_env_radiance(rx_r, ry_r, rz_r)
                    rough_env_rgb = (sky_horizon * 0.65 + sky_zenith * 0.35)[:, None, None]
                    gloss = (1.0 - rough_buf).clamp(0.0, 1.0).pow(1.6)
                    ibl_env = refl_env_rgb * gloss[None] + rough_env_rgb * (1.0 - gloss[None])
                    max_gloss = torch.maximum(1.0 - rough_buf[None], F0_rgb)
                    F_ibl = (F0_rgb + (max_gloss - F0_rgb) * pow5_nv[None]) * spec_gate[None]
                    ao_sh = (0.62 + 0.38 * shadow_vis)[None]
                    # The film reflects F_coat_v and transmits only the remainder
                    # to the substrate: Fc + (1-Fc)*Fb <= 1, even at grazing.
                    ibl_hdr = ao_sh * ((1.0 - F_coat_v[None]) * F_ibl * ibl_env + F_coat_v[None] * refl_env_rgb)

                    hemi_up = (0.5 + 0.5 * nw_z_buf).clamp(0.0, 1.0)
                    sky_amb_rgb = sky_zenith[:, None, None] * hemi_up[None] + sky_horizon[:, None, None] * (1.0 - hemi_up)[None] * 0.55
                    amb_base = 0.06 if is_night else 0.24
                    amb_hdr = kd * alb_buf * (amb_base + 0.58 * sky_amb_rgb) * (0.72 + 0.28 * shadow_vis)[None]
                else:
                    ibl_hdr = torch.zeros_like(direct_hdr)
                    amb_hdr = kd * alb_buf * (0.06 if is_night else 0.38)

                # Coaxial Headlight / Gate-Lamp Retroreflection (`retro_buf > 0`)
                if is_night:
                    coax_vis = (retro_buf * 5.2 * ir_ang_gain / (1.0 + 0.035 * dist_m * dist_m)).clamp(0.0, 2.8)
                    retro_hdr = alb_buf * (coax_vis * (0.55 + 0.45 * shadow_vis))[None]
                elif env_energy_gate > 1e-6:
                    retro_hdr = alb_buf * (retro_buf * 0.08 * ir_ang_gain * shadow_vis)[None]
                else:
                    retro_hdr = torch.zeros_like(direct_hdr)

                surf_hdr = torch.where(
                    (hit_kind_buf == 4)[None, :, :],
                    alb_buf * emit_buf[None, :, :],
                    amb_hdr + direct_hdr + ibl_hdr + retro_hdr,
                )
                hdr = torch.where(any_surf[None, :, :], surf_hdr, hdr)

        # Foreground Galvanized Diamond Chain-Link Wire Mesh Fence (`金網`, `cg/06`!)
        # Attenuates `trans_buf` identically for BOTH vehicles and license plates!
        if fence_cfg is not None:
            f_depth = float(fence_cfg["depth_m"])
            pitch_px = float(fence_cfg["pitch_px"])
            wire_r = float(fence_cfg["wire_half_px"])
            top_v = float(fence_cfg["top_v_px"])
            behind_fence = t_best > f_depth
            if behind_fence.any():
                ug = uu[None, :].expand(h, w) + float(fence_cfg["phase_u"])
                vg = vv[:, None].expand(h, w) + float(fence_cfg["phase_v"])
                d1 = torch.remainder(ug + vg * 1.12, pitch_px) - 0.5 * pitch_px
                d2 = torch.remainder(ug - vg * 1.12, pitch_px) - 0.5 * pitch_px
                wire_d = torch.minimum(d1.abs(), d2.abs())
                wire_a = ((wire_r + 0.55 - wire_d).clamp(0.0, 1.0) * float(fence_cfg["opacity"])) * (vv[:, None] >= top_v).to(torch.float32)
                rail_a = ((vv[:, None] - top_v).abs() <= max(1.8, wire_r * 2.4)).to(torch.float32) * 0.92
                tot_a = (torch.maximum(wire_a, rail_a) * behind_fence.to(torch.float32)).clamp(0.0, 1.0)
                wire_lum = float(fence_cfg["wire_hdr_lum"])
                wire_rgb = (
                    torch.full((3, 1, 1), wire_lum, device=dev, dtype=torch.float32)
                    if is_ir
                    else torch.tensor([wire_lum * 0.96, wire_lum * 1.0, wire_lum * 1.04], device=dev, dtype=torch.float32)[:, None, None]
                )
                hdr = hdr * (1.0 - tot_a[None]) + wire_rgb * tot_a[None]
                trans_buf = trans_buf * (1.0 - tot_a)

        return hdr, veh_buf, role_buf, quad_buf, trans_buf

    @staticmethod
    def _eval_sign_at(sd: dict[str, Any], t_inst: float) -> tuple[_PlanarQuad3D, _Prim3D | None]:
        """Evaluate a bolted/framed placard, swaying roadside signboard, or frameless alpha stencil at `t_inst`."""
        tid = int(sd["track_id"])
        if sd.get("sway", False):
            sway_x = 0.085 * math.sin(4.2 * t_inst + float(tid))
            sway_yaw = 0.15 * math.cos(3.6 * t_inst + 0.5 * float(tid))
            sway_roll = 0.055 * math.sin(3.1 * t_inst + 0.3 * float(tid))
        else:
            sway_x = 0.0
            sway_yaw = 0.0
            sway_roll = 0.0
        sc = sd["pos"] + np.array([sway_x, 0.0, 0.0], dtype=np.float32)
        R_s = _rot_zyx(float(sd["yaw"]) + sway_yaw, 0.0, sway_roll)
        sn = R_s[:, 1]
        up_s = R_s[:, 2]
        tex_kind = str(sd.get("tex_kind", "sign"))
        has_backing = bool(sd.get("has_backing_substrate", True))
        qd = _make_planar_quad(
            sc,
            sn,
            float(sd["hw"]),
            float(sd["hh"]),
            (tex_kind, int(sd["sidx"])),
            CLS_SIGNBOARD_HARDNEG,
            0,
            up_hint_w=up_s,
            retro_gain=float(sd.get("retro_gain", 0.95 if has_backing else 0.0)),
            roughness=float(sd.get("roughness", 0.36)),
            metallic=float(sd.get("metallic", 0.08)),
            clearcoat=float(sd.get("clearcoat", 0.22)),
            specular_f0=float(sd.get("specular_f0", 0.04)),
            normal_undul_amp=float(sd.get("normal_undul_amp", 0.020 if has_backing else 0.0)),
            has_backing_substrate=has_backing,
        )
        if not bool(sd.get("has_pole", has_backing)):
            return qd, None
        pole_prim = _Prim3D(
            center=(float(sc[0]), float(sc[1]), float(sc[2]) * 0.5),
            rot_w2l=np.eye(3, dtype=np.float32),
            half_ext=(0.035, 0.035, float(sc[2]) * 0.5),
            shape=_SHAPE_OBB,
            veh_idx=0,
            role=_ROLE_ENV_SOLID,
            albedo_rgb=(0.32, 0.34, 0.36),
            roughness=0.45,
            metallic=0.45,
        )
        return qd, pole_prim

    @staticmethod
    def _clip_z0_rect_to_near_plane(
        rect_w4: np.ndarray,
        cam_o: np.ndarray,
        R_cam: np.ndarray,
        z_near: float = 0.22,
    ) -> tuple[np.ndarray, bool]:
        """Exact Sutherland-Hodgman polygon intersection of a `Z=0` road polygon with `Z_c >= z_near`."""
        from fractions import Fraction

        R64 = R_cam.astype(np.float64)
        z_axis = R64[:, 2]
        cam64 = cam_o.astype(np.float64)
        pts = rect_w4.astype(np.float64, copy=True)
        pts[:, 2] = 0.0
        zn_f = float(z_near)
        d = (pts - cam64[None, :]) @ z_axis - zn_f
        if np.all(d < 0.0):
            return np.zeros((0, 3), dtype=np.float32), False
        if np.all(d >= 0.0):
            out_all = pts.astype(np.float32)
            out_all[:, 2] = 0.0
            return out_all, True
        out_pts: list[np.ndarray] = []
        is_bndry: list[bool] = []
        n_v = pts.shape[0]
        for i in range(n_v):
            j = (i + 1) % n_v
            di, dj = float(d[i]), float(d[j])
            if di >= 0.0:
                out_pts.append(pts[i])
                is_bndry.append(di <= 1e-6)
            if (di < 0.0) != (dj < 0.0):
                alpha = di / (di - dj)
                inter = pts[i] + alpha * (pts[j] - pts[i])
                inter[2] = 0.0
                out_pts.append(inter)
                is_bndry.append(True)
        if len(out_pts) < 3:
            return np.zeros((0, 3), dtype=np.float32), False
        poly = np.stack(out_pts, axis=0)
        area = 0.5 * abs(float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])))
        if area < 1e-4:
            return np.zeros((0, 3), dtype=np.float32), False
        poly[:, 2] = 0.0
        poly_f32 = poly.astype(np.float32)
        poly_f32[:, 2] = 0.0

        # Directed 1-ULP inward rounding on newly clipped boundary vertices so float32 rounding never falls below `z_near`
        R_inv2 = np.linalg.inv(R64)[2, :]
        zn_q = max(Fraction(zn_f), Fraction(str(z_near)))
        cx_q, cy_q, cz_q = Fraction(float(cam_o[0])), Fraction(float(cam_o[1])), Fraction(float(cam_o[2]))
        zx_q, zy_q, zz_q = Fraction(float(R_cam[0, 2])), Fraction(float(R_cam[1, 2])), Fraction(float(R_cam[2, 2]))
        ax_b = 0 if abs(float(z_axis[0])) >= abs(float(z_axis[1])) else 1
        dir_b = np.float32(np.inf if float(z_axis[ax_b]) >= 0.0 else -np.inf)
        for idx_b, b_flag in enumerate(is_bndry):
            if not b_flag:
                continue
            for _ in range(8):
                px_q = Fraction(float(poly_f32[idx_b, 0]))
                py_q = Fraction(float(poly_f32[idx_b, 1]))
                zd_q = (px_q - cx_q) * zx_q + (py_q - cy_q) * zy_q - cz_q * zz_q
                v_diff = poly_f32[idx_b].astype(np.float64) - cam64
                if zd_q >= zn_q and float(v_diff @ z_axis) >= zn_f and float(R_inv2 @ v_diff) >= zn_f:
                    break
                poly_f32[idx_b, ax_b] = np.nextafter(poly_f32[idx_b, ax_b], dir_b)
        return poly_f32, True

    @staticmethod
    def _quad_corners_w(qd: _PlanarQuad3D) -> np.ndarray:
        return np.stack(
            [
                qd.center - qd.half_w * qd.u_axis - qd.half_h * qd.v_axis,  # TL
                qd.center + qd.half_w * qd.u_axis - qd.half_h * qd.v_axis,  # TR
                qd.center + qd.half_w * qd.u_axis + qd.half_h * qd.v_axis,  # BR
                qd.center - qd.half_w * qd.u_axis + qd.half_h * qd.v_axis,  # BL
            ],
            axis=0,
        )

    @torch.no_grad()
    def render_clip(self, seed: int, num_frames: int | None = None, image_hw: tuple[int, int] | None = None) -> Scene3DClipSample:
        """Synthesize one deterministic `K`-frame 3D video clip with Gauss-Legendre shutter & exact GT."""
        rng = random.Random(int(seed))
        K = int(num_frames if num_frames is not None else self.config.num_frames)
        h, w = image_hw if image_hw is not None else self.config.image_hw
        dev = self.device

        is_neg = rng.random() < self.config.p_negative_clip
        is_night = rng.random() < self.config.p_night_scene
        is_ir = is_night and (rng.random() < (self.config.p_night_ir / max(0.05, self.config.p_night_scene)))
        is_wet = rng.random() < self.config.p_rain_wet
        has_fence = rng.random() < self.config.p_chainlink_fence
        # Sample foreground gate pillar identically on positive AND negative clips (zero label shortcut!)
        has_pillar = (K >= 2) and (rng.random() < self.config.p_gate_pillar_occlusion)
        stationary_all = rng.random() < self.config.p_stationary_clip

        # Camera rig: covers close-up giant gate, top-edge approach, mid-gate, and distant wide pole!
        cam_mode = rng.choices(["close_gate", "mid_gate", "high_pole_wide", "top_edge_pitch"], weights=[0.38, 0.34, 0.16, 0.12], k=1)[0]
        if cam_mode == "close_gate":
            cam_x = rng.uniform(-2.9, 2.9)
            cam_z = rng.uniform(1.10, 2.45)
            hfov_deg = rng.uniform(52.0, 82.0)
            y_spawn = (4.8, 9.6)
            look_y, look_z = rng.uniform(5.2, 8.8), rng.uniform(0.85, 1.75)
        elif cam_mode == "top_edge_pitch":
            cam_x = rng.uniform(-2.6, 2.6)
            cam_z = rng.uniform(2.1, 4.1)
            hfov_deg = rng.uniform(46.0, 70.0)
            y_spawn = (7.2, 14.5)
            look_y, look_z = rng.uniform(4.4, 6.6), rng.uniform(0.0, 0.30)
        elif cam_mode == "high_pole_wide":
            cam_x = rng.uniform(-4.0, 4.0)
            cam_z = rng.uniform(3.4, 6.2)
            hfov_deg = rng.uniform(54.0, 86.0)
            y_spawn = (10.5, 25.0)
            look_y, look_z = rng.uniform(12.5, 20.5), rng.uniform(0.6, 1.5)
        else:
            cam_x = rng.uniform(-3.0, 3.0)
            cam_z = rng.uniform(1.5, 3.2)
            hfov_deg = rng.uniform(45.0, 72.0)
            y_spawn = (6.5, 16.5)
            look_y, look_z = rng.uniform(8.5, 14.5), rng.uniform(0.8, 1.5)

        cam_o = np.array([cam_x, 0.0, cam_z], dtype=np.float32)
        look_p = np.array([rng.uniform(-1.0, 1.0), look_y, look_z], dtype=np.float32)
        z_cam = look_p - cam_o
        z_cam = z_cam / max(1e-6, float(np.linalg.norm(z_cam)))
        x_cam0 = np.cross(z_cam, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        x_cam0 = x_cam0 / max(1e-6, float(np.linalg.norm(x_cam0)))
        y_cam0 = np.cross(z_cam, x_cam0)
        roll_rad = math.radians(rng.uniform(-3.5, 3.5))
        x_cam = math.cos(roll_rad) * x_cam0 + math.sin(roll_rad) * y_cam0
        y_cam = -math.sin(roll_rad) * x_cam0 + math.cos(roll_rad) * y_cam0
        R_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)  # columns = camera axes
        pitch_deg = math.degrees(math.asin(float(np.clip(z_cam[2], -1.0, 1.0))))

        fx = (0.5 * float(w)) / math.tan(0.5 * math.radians(hfov_deg))
        fy = fx * rng.uniform(0.96, 1.04)
        cx = 0.5 * float(w - 1)
        cy = 0.5 * float(h - 1)

        # Keep the compositional draw independent of legacy scene/identity RNG consumption.
        rng_g6 = random.Random(int(seed) ^ 0x6C39A7E5)
        combo = rng_g6.choice(GRAMMAR_COMBOS_BY_DOMAIN_SPLIT[self.config.domain_split])
        radial_order = rng_g6.choice(RADIAL_DN_ORDERS_BY_DOMAIN_SPLIT[self.config.domain_split])
        camera_family = rng_g6.choice(CAMERA_FAMILIES_BY_DOMAIN_SPLIT[self.config.domain_split])
        solar_radius_deg = rng_g6.uniform(0.25, 6.0)
        camera_path = _CameraTrajectorySpec(
            cam_o_base=cam_o.copy(), R_cam_base=R_cam.copy(), fx_base=fx, fy_base=fy,
            family=camera_family, stationary=stationary_all or camera_family == "stationary_rig_fixed",
            vel_xyz_mps=(rng_g6.uniform(-0.12, 0.12), 0.0, 0.0),
            pan_rate_rad_s=rng_g6.uniform(-0.025, 0.025),
            tilt_rate_rad_s=rng_g6.uniform(-0.012, 0.012),
            zoom_rate_s=rng_g6.uniform(-0.035, 0.035),
            vib_amp_rad=rng_g6.uniform(0.0002, 0.0012), vib_phase=rng_g6.uniform(0, 2 * math.pi),
        )

        # Sample lighting, frame timestamps, and ALL static/nuisance environment BEFORE vehicles
        # so positive and negative clips share the exact same nuisance & camera distribution!
        env_style = {
            "seed_ph": rng.uniform(0.2, 6.1),
            "ground_type": float(rng.choices([0, 1, 2], weights=[0.50, 0.26, 0.24], k=1)[0]),
            "skyline_type": float(rng.choices([0, 1, 2], weights=[0.45, 0.40, 0.15], k=1)[0]),
        }
        env_prims_base: list[_Prim3D] = []
        static_obs: list[tuple[np.ndarray, float, float]] = []  # (xy_center, hw, hl) for collision clearance
        I3 = np.eye(3, dtype=np.float32)
        world_origin = np.array([rng_g6.choice((-1.0, 1.0)) * 7.8, 14.0, 1.4], dtype=np.float32)
        world_color = (0.22, 0.30, 0.24)
        if combo[2] == "linear_lattice_wall":
            grammar_prims = _prod_linear_lattice_z2(
                origin_w=world_origin, R_l2w=I3, half_span_y=2.0, half_span_z=1.4,
                bar_thick_x=0.07, n_rails=4, n_mullions=6, veh_idx=0,
                role=_ROLE_ENV_SOLID, albedo_rgb=world_color,
            )
        elif combo[2] == "flex_wave_screen":
            grammar_prims = [_prod_flex_wave_sheet(
                center_w=world_origin, R_l2w=I3, half_ext=(0.12, 2.0, 1.4),
                veh_idx=0, role=_ROLE_ENV_SOLID, albedo_rgb=world_color,
                phase=rng_g6.uniform(0, 2 * math.pi), decal_spec=(0, 0, 0., 0., .7, .5, 0.),
            )]
        else:
            grammar_prims = _prod_articulated_truss_or_boom(
                base_w=world_origin - np.array([0., 0., 1.4], dtype=np.float32), R_base_l2w=I3,
                mast_h=3.3, boom_len_1=2.8, boom_len_2=1.8, section_r=0.10,
                slew_yaw_rad=0., elev_1_rad=0.08, elev_2_rad=0.14,
                veh_idx=0, role=_ROLE_ENV_SOLID, albedo_rgb=world_color, accent_rgb=(.38, .40, .42),
            )
        env_prims_base.extend(grammar_prims)
        for gp in grammar_prims:
            xy_extent = np.abs(gp.rot_w2l.T[:2]) @ np.asarray(gp.half_ext)
            static_obs.append((np.array(gp.center[:2], dtype=np.float32), float(xy_extent[0]) + .14, float(xy_extent[1]) + .14))
        rng_v5 = random.Random(((int(seed) & 0xFFFFFFFFFFFFFFFF) * 0x9E3779B97F4A7C15 + 0xBF58476D1CE4E5B9) & ((1 << 64) - 1))
        wall_prob = rng.choice([0.0, 0.45, 0.85])
        for side_s in (-1.0, 1.0):
            if rng.random() < wall_prob:
                wx_h = side_s * rng.uniform(5.3, 8.4)
                wall_col = rng.choice([(0.54, 0.56, 0.58), (0.12, 0.26, 0.18), (0.36, 0.34, 0.31), (0.18, 0.24, 0.34)])
                wall_cz = rng.uniform(1.1, 1.8)
                wall_hz = rng.uniform(1.1, 1.8)
                wall_style_v5 = float(rng_v5.choice([0, 1, 2, 3, 4]))
                env_prims_base.append(
                    _Prim3D(
                        center=(wx_h, 18.0, wall_cz),
                        rot_w2l=I3,
                        half_ext=(0.15, 22.0, wall_hz),
                        shape=_SHAPE_OBB,
                        veh_idx=0,
                        role=_ROLE_ENV_SOLID,
                        albedo_rgb=wall_col,
                        roughness=0.58,
                        rib_freq=rng.choice([0.0, 6.0, 12.0]),
                        mud_phase=wall_style_v5,
                    )
                )
                static_obs.append((np.array([wx_h, 18.0], dtype=np.float32), 0.15 + 0.14, 22.0 + 0.14))

        if rng.random() < 0.55:
            for cy_cone in (6.5, 11.5, 16.5):
                cx_cone = rng.choice([-1.0, 1.0]) * rng.uniform(4.85, 5.45)
                env_prims_base.append(
                    _Prim3D(
                        center=(cx_cone, cy_cone, 0.36),
                        rot_w2l=I3,
                        half_ext=(0.16, 0.16, 0.36),
                        shape=_SHAPE_OBB,
                        veh_idx=0,
                        role=_ROLE_ENV_SOLID,
                        albedo_rgb=(0.78, 0.14, 0.04),
                        roughness=0.45,
                    )
                )
                static_obs.append((np.array([cx_cone, cy_cone], dtype=np.float32), 0.16 + 0.12, 0.16 + 0.12))

        if has_pillar:
            # Foreground gate pillar (`門柱`) placed on inter-lane island or foreground post (`y in [2.1, 3.3]`)
            pil_x = rng.choice([-1.68, 1.68]) + rng.uniform(-0.14, 0.14)
            pil_y = rng.uniform(2.2, 3.6)
            pil_hx = rng.uniform(0.22, 0.34)
            env_prims_base.append(
                _Prim3D(
                    center=(pil_x, pil_y, 1.35),
                    rot_w2l=I3,
                    half_ext=(pil_hx, 0.26, 1.35),
                    shape=_SHAPE_OBB,
                    veh_idx=0,
                    role=_ROLE_ENV_SOLID,
                    albedo_rgb=(0.34, 0.35, 0.36),
                    roughness=0.72,
                )
            )
            static_obs.append((np.array([pil_x, pil_y], dtype=np.float32), pil_hx + 0.18, 0.44))

        # Bolted roadside signs & swaying safety placards (`CLS_SIGNBOARD_HARDNEG`, label-independent distribution!)
        n_signs = rng.choices([0, 1, 2, 3], weights=[0.26, 0.36, 0.24, 0.14], k=1)[0]
        sign_defs: list[dict[str, Any]] = []
        for si in range(n_signs):
            sx_s = rng.choice([-1.0, 1.0]) * rng.uniform(4.75, 5.65)
            sy_s = rng.uniform(4.8, 18.5)
            sz_s = rng.uniform(0.95, 2.35)
            sway = rng.random() < 0.35
            sidx_leg = rng.randrange(len(self.bank.SIGN_SPECS))
            sidx_eff = rng_v5.randrange(len(self.bank.sign_mips)) if (len(self.bank.sign_mips) > len(self.bank.SIGN_SPECS) and rng_v5.random() < 0.62) else sidx_leg
            sign_defs.append(
                {
                    "track_id": 100 + si,
                    "pos": np.array([sx_s, sy_s, sz_s], dtype=np.float32),
                    "yaw": math.pi + rng.uniform(-0.55, 0.55),
                    "hw": rng.uniform(0.24, 0.52),
                    "hh": rng.uniform(0.11, 0.24),
                    "sidx": int(sidx_eff),
                    "sway": sway,
                }
            )
            sign_r = float(sign_defs[-1]["hw"]) + (0.12 if sway else 0.04) + 0.10
            static_obs.append((np.array([sx_s, sy_s], dtype=np.float32), sign_r, sign_r))

        # Gen-5 extra architectural hard-negative panels (ribbed metal, bridge rail, wood fence, shop decal, window grid)
        # Sampled identically across positive & negative clips (zero label shortcut!) and registered in `static_obs`.
        if len(self.bank.sign_mips) > 10:
            n_arch_hn = rng_v5.choices([0, 1, 2], weights=[0.28, 0.44, 0.28], k=1)[0]
            for ahi in range(n_arch_hn):
                sx_a = rng_v5.choice([-1.0, 1.0]) * rng_v5.uniform(5.65, 7.10)
                sy_a = rng_v5.uniform(5.8, 28.0)
                sz_a = rng_v5.uniform(1.05, 2.75)
                hw_a = rng_v5.uniform(0.38, 0.92)
                hh_a = rng_v5.uniform(0.18, 0.46)
                sign_defs.append(
                    {
                        "track_id": 120 + ahi,
                        "pos": np.array([sx_a, sy_a, sz_a], dtype=np.float32),
                        "yaw": math.pi + rng_v5.uniform(-0.62, 0.62),
                        "hw": hw_a,
                        "hh": hh_a,
                        "sidx": int(rng_v5.randrange(10, len(self.bank.sign_mips))),
                        "sway": False,
                    }
                )
                sign_ra = hw_a + 0.14
                static_obs.append((np.array([sx_a, sy_a], dtype=np.float32), sign_ra, sign_ra))

        fence_cfg: dict[str, float] | None = None
        if has_fence:
            fence_cfg = {
                "depth_m": rng.uniform(2.2, 4.2),
                "pitch_px": rng.uniform(11.0, 24.0),
                "wire_half_px": rng.uniform(0.42, 0.95),
                "top_v_px": rng.uniform(-10.0, float(h) * 0.28),
                "phase_u": rng.uniform(0.0, 20.0),
                "phase_v": rng.uniform(0.0, 20.0),
                "opacity": rng.uniform(0.55, 0.90),
                "wire_hdr_lum": rng.uniform(0.06, 0.22) if is_night else rng.uniform(0.32, 0.78),
            }

        sun_az = rng.uniform(0.0, 6.28)
        sun_el = rng.uniform(0.18, 1.15)
        sun_dir = torch.tensor(
            [math.cos(sun_el) * math.cos(sun_az), math.cos(sun_el) * math.sin(sun_az), math.sin(sun_el)],
            device=dev,
            dtype=torch.float32,
        )
        # A fixed solar-disc sample per clip gives common random numbers when the
        # exposure quadrature order changes; resampling per node changes the scene
        # integrand and prevents meaningful shutter-convergence comparisons.
        sun_np = sun_dir.cpu().numpy()
        tangent = np.cross(sun_np, np.array([0., 0., 1.], dtype=np.float32))
        tangent /= max(1e-6, float(np.linalg.norm(tangent)))
        bitangent = np.cross(sun_np, tangent)
        radius = math.tan(math.radians(solar_radius_deg)) * math.sqrt(rng_g6.random())
        azimuth = rng_g6.uniform(0, 2 * math.pi)
        sun_dir = torch.as_tensor(sun_np + radius * (math.cos(azimuth) * tangent + math.sin(azimuth) * bitangent), device=dev)
        if is_night:
            sky_zen = torch.tensor([0.008, 0.011, 0.022], device=dev, dtype=torch.float32)
            sky_hor = torch.tensor([0.025, 0.030, 0.045], device=dev, dtype=torch.float32)
            exp_s = rng.uniform(1.0 / 60.0, 1.0 / 15.0)
        else:
            sky_roll = rng.random()
            if sky_roll < 0.28:
                # High-luminance blown-out overcast sky (`w3/01`, `cg/06`) so white box trucks never bleed into white sky!
                sky_zen = torch.tensor([0.82, 0.85, 0.90], device=dev, dtype=torch.float32)
                sky_hor = torch.tensor([0.88, 0.90, 0.94], device=dev, dtype=torch.float32)
                exp_s = rng.uniform(1.0 / 500.0, 1.0 / 30.0)
            elif sky_roll < 0.54:
                sky_zen = torch.tensor([0.36, 0.44, 0.56], device=dev, dtype=torch.float32)
                sky_hor = torch.tensor([0.52, 0.55, 0.58], device=dev, dtype=torch.float32)
                exp_s = rng.uniform(1.0 / 500.0, 1.0 / 25.0)
            else:
                sky_zen = torch.tensor([0.16, 0.34, 0.72], device=dev, dtype=torch.float32)
                sky_hor = torch.tensor([0.55, 0.62, 0.72], device=dev, dtype=torch.float32)
                exp_s = rng.uniform(1.0 / 1000.0, 1.0 / 60.0)

        road_alb = rng.uniform(0.055, 0.16)
        fps_cands = [f for f in (10.0, 12.0, 15.0, 20.0, 25.0, 30.0) if (1.0 / f) >= exp_s * 1.12] or [10.0]
        fps = rng.choice(fps_cands)
        base_dt = 1.0 / fps
        timestamps: list[float] = [0.0]
        for _ in range(1, K):
            mult = 2.0 if rng.random() < 0.12 else rng.uniform(0.92, 1.08)
            dt_step = max(exp_s * 1.025, base_dt * mult)
            timestamps.append(timestamps[-1] + dt_step)
        clip_span_s = max(0.35, timestamps[-1] + 0.6 * exp_s)

        # Sample collision-free vehicles (clear of camera, static pillars/signs, and each other across `clip_span_s`)
        # Gen-5 (`v0.5.0`) crowded multi-vehicle traffic & gate queues: `2..6` simultaneous vehicles in `>= 55%` of positive clips
        # across staggered lanes & depths `5m..36m`, with exact per-vehicle visibility/bbox/plate-association GT!
        if cam_z < 3.8:
            static_obs.append((cam_o[:2].copy(), 0.55, 0.55))
        veh_specs: list[_VehiclePassSpec] = []
        if not is_neg:
            _ = rng.choices([1, 2, 3], weights=[0.58, 0.32, 0.10], k=1)[0]
            lanes_leg = [-3.3, 0.0, 3.3]
            rng.shuffle(lanes_leg)
            rng_veh = random.Random(rng.getrandbits(64) ^ rng_v5.getrandbits(64))

            n_veh = rng_v5.choices([1, 2, 3, 4, 5, 6], weights=[0.22, 0.24, 0.20, 0.15, 0.11, 0.08], k=1)[0]
            y_base0 = max(5.0, float(y_spawn[0]))
            y_base1 = max(y_base0 + 2.8, min(14.5, float(y_spawn[1])))
            slot_grid: list[tuple[float, float, float]] = [
                (0.0, y_base0, y_base1),
                (-3.35, max(5.2, y_base0 + 1.2), min(17.5, y_base1 + 3.2)),
                (3.35, max(5.2, y_base0 + 1.8), min(18.0, y_base1 + 3.8)),
                (-1.75, max(15.5, y_base0 + 10.5), min(25.5, y_base1 + 12.5)),
                (1.75, max(16.5, y_base0 + 11.5), min(26.5, y_base1 + 13.5)),
                (0.0, max(25.5, y_base0 + 19.5), 34.5),
                (-3.35, max(24.0, y_base0 + 18.0), 35.5),
                (3.35, max(24.5, y_base0 + 18.5), 36.0),
            ]
            lead_slot = slot_grid[0]
            tail_slots = slot_grid[1:]
            rng_veh.shuffle(tail_slots)
            ordered_slots = [lead_slot] + tail_slots if rng_veh.random() < 0.65 else (tail_slots[:2] + [lead_slot] + tail_slots[2:])

            for vi in range(n_veh):
                for attempt in range(28):
                    sl_idx = (vi + attempt) % len(ordered_slots)
                    lane_base, yr_lo, yr_hi = ordered_slots[sl_idx]
                    tier_push = (attempt // len(ordered_slots)) * 3.2
                    lx_cand = lane_base + rng_veh.uniform(-0.24, 0.24)
                    yr_cand = (min(33.5, yr_lo + tier_push), min(36.0, max(yr_lo + tier_push + 2.2, yr_hi + tier_push)))
                    cand_vs = self._sample_vehicle_spec(
                        rng_veh,
                        track_id=vi + 1,
                        lane_x=lx_cand,
                        y_range=yr_cand,
                        stationary=(stationary_all or (vi > 0 and rng_veh.random() < 0.28)),
                        veh_struct_prod=combo[1], marking_prod=combo[3], radial_dn_n=radial_order,
                    )
                    t_chk_lo = -0.6 * exp_s
                    if not self._trajectories_clear(cand_vs, veh_specs, duration_s=clip_span_s, t_start_s=t_chk_lo):
                        continue
                    obs_ok = True
                    if static_obs:
                        v_c_obs = max(0.5, self._max_planar_point_speed_mps(cand_vs, t_chk_lo, clip_span_s))
                        dt_obs = (2.0 * 0.10) / v_c_obs
                        n_obs = max(16, int(math.ceil((clip_span_s - t_chk_lo) / max(1e-4, dt_obs))))
                        for si_chk in range(n_obs + 1):
                            t_chk = t_chk_lo + (clip_span_s - t_chk_lo) * (si_chk / float(n_obs))
                            c_obbs = self._vehicle_planar_obbs_at(cand_vs, t_chk, pad_w=0.18, pad_l=0.22)
                            for cc, cyaw, chw, chl in c_obbs:
                                rc_o = math.hypot(chw, chl)
                                for obs_xy, obs_hw, obs_hl in static_obs:
                                    r_so = rc_o + math.hypot(obs_hw, obs_hl)
                                    dx_o = float(obs_xy[0]) - float(cc[0])
                                    dy_o = float(obs_xy[1]) - float(cc[1])
                                    if (dx_o * dx_o + dy_o * dy_o) > (r_so * r_so):
                                        continue
                                    if self._obb2d_overlap(cc, cyaw, chw, chl, obs_xy, 0.0, obs_hw, obs_hl):
                                        obs_ok = False
                                        break
                                if not obs_ok:
                                    break
                            if not obs_ok:
                                break
                    if obs_ok:
                        veh_specs.append(cand_vs)
                        break

        # Exposure-proportional signal scale & causal 1-frame-delayed Auto-Exposure (AE) initialization
        ref_exp_s = (1.0 / 35.0) if is_night else (1.0 / 150.0)
        exp_charge_scale = float(exp_s / ref_exp_s)
        target_l = 0.17 if is_night else 0.28
        ambient_prior_l = (0.045 if is_ir else (0.065 if is_night else 0.26)) * exp_charge_scale
        ae_gain = float(np.clip(target_l / max(0.012, ambient_prior_l), 0.20, 9.0)) * rng.uniform(0.88, 1.14)

        frames_out: list[Tensor] = []
        gts_out: list[Scene3DFrameGT] = []

        for k_idx, t_c in enumerate(timestamps):
            cam_o, R_cam, fx, fy = camera_path.eval_at(t_c)
            # 1. Evaluate scene endpoints `t_c - T/2` and `t_c + T/2` to measure exact projected corner displacement
            t_s0 = t_c - 0.5 * exp_s
            t_s1 = t_c + 0.5 * exp_s
            cam0, rot0, fx0, fy0 = camera_path.eval_at(t_s0)
            cam1, rot1, fx1, fy1 = camera_path.eval_at(t_s1)
            quads_t0: list[_PlanarQuad3D] = []
            quads_t1: list[_PlanarQuad3D] = []
            max_disp_px = 0.0

            for vi_1, vs in enumerate(veh_specs, start=1):
                vp0, vq0, _, _, _ = _assemble_vehicle_3d(vs, t_s0, vi_1, self.bank)
                vp1, vq1, _, _, _ = _assemble_vehicle_3d(vs, t_s1, vi_1, self.bank)
                quads_t0.extend(vq0)
                quads_t1.extend(vq1)
                if vp0 and vp1:
                    c0_arr = np.array([pr.center for pr in vp0], dtype=np.float32)
                    c1_arr = np.array([pr.center for pr in vp1], dtype=np.float32)
                    uv0, z0 = self._project_points(c0_arr, cam0, rot0, fx0, fy0, cx, cy)
                    uv1, z1 = self._project_points(c1_arr, cam1, rot1, fx1, fy1, cx, cy)
                    vz = (z0 > 0.22) & (z1 > 0.22)
                    if np.any(vz):
                        disp_v = float(np.linalg.norm(uv1[vz] - uv0[vz], axis=-1).max())
                        max_disp_px = max(max_disp_px, disp_v)

            for sd in sign_defs:
                sq0, _ = self._eval_sign_at(sd, t_s0)
                sq1, _ = self._eval_sign_at(sd, t_s1)
                quads_t0.append(sq0)
                quads_t1.append(sq1)

            quad_blur_vecs: list[np.ndarray] = []
            for q0, q1 in zip(quads_t0, quads_t1):
                c4_0 = self._quad_corners_w(q0)
                c4_1 = self._quad_corners_w(q1)
                uv0_q, z0_q = self._project_points(c4_0, cam0, rot0, fx0, fy0, cx, cy)
                uv1_q, z1_q = self._project_points(c4_1, cam1, rot1, fx1, fy1, cx, cy)
                vz_q = (z0_q > 0.20) & (z1_q > 0.20)
                if np.any(vz_q):
                    d_corners = uv1_q[vz_q] - uv0_q[vz_q]
                    max_disp_px = max(max_disp_px, float(np.linalg.norm(d_corners, axis=-1).max()))
                    quad_blur_vecs.append(d_corners.mean(axis=0).astype(np.float32))
                else:
                    quad_blur_vecs.append(np.zeros((2,), dtype=np.float32))

            # 2. Odd Gauss-Legendre quadrature on `[-0.5*exp_s, +0.5*exp_s]` (`E[t^2] == T^2/12` & exact `dt=0` center!)
            if self.config.max_substeps <= 1 or max_disp_px < 0.12:
                n_sub = 1
                sub_offsets = [0.0]
                sub_weights = [1.0]
            else:
                req_odd = max(3, int(math.ceil(max_disp_px / 1.5)) | 1)
                n_sub = min(int(self.config.max_substeps), req_odd)
                if n_sub % 2 == 0:
                    n_sub = max(3, n_sub - 1)
                xi, wi = np.polynomial.legendre.leggauss(n_sub)
                sub_offsets = [float(0.5 * exp_s * x) for x in xi]
                sub_weights = [float(0.5 * w_g) for w_g in wi]
            mid_sub_idx = n_sub // 2
            sub_offsets[mid_sub_idx] = 0.0  # exact `t_inst == t_c` at middle node!

            hdr_accum = torch.zeros((3, h, w), device=dev, dtype=torch.float32)
            bloom_accum = torch.zeros((3, h, w), device=dev, dtype=torch.float32)
            swept_veh_occ: dict[int, Tensor] = {
                vi_1: torch.zeros((h, w), device=dev, dtype=torch.bool) for vi_1 in range(1, len(veh_specs) + 1)
            }
            swept_quad_occ: dict[int, Tensor] = {
                qi: torch.zeros((h, w), device=dev, dtype=torch.bool) for qi in range(len(quads_t0))
            }

            mid_veh_buf: Tensor | None = None
            mid_role_buf: Tensor | None = None
            mid_quad_buf: Tensor | None = None
            mid_trans_buf: Tensor | None = None
            mid_quads: list[_PlanarQuad3D] = []
            mid_footprints: dict[int, np.ndarray] = {}
            mid_art_footprints: dict[int, np.ndarray | None] = {}
            mid_cabins: dict[int, list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]]] = {}
            mid_veh_prims: dict[int, list[_Prim3D]] = {}
            num_veh_quads = 0

            for s_i, (dt_s, w_s) in enumerate(zip(sub_offsets, sub_weights)):
                t_inst = t_c + dt_s
                is_mid = s_i == mid_sub_idx
                prims_inst = list(env_prims_base)
                quads_inst: list[_PlanarQuad3D] = []

                for vi_1, vs in enumerate(veh_specs, start=1):
                    vp, vq, fp_w, art_fp_w, cab_obbs = _assemble_vehicle_3d(vs, t_inst, vi_1, self.bank)
                    prims_inst.extend(vp)
                    quads_inst.extend(vq)
                    if is_mid:
                        mid_footprints[vi_1] = fp_w
                        mid_art_footprints[vi_1] = art_fp_w
                        mid_cabins[vi_1] = cab_obbs
                        mid_veh_prims[vi_1] = vp
                if is_mid:
                    num_veh_quads = len(quads_inst)

                for sd in sign_defs:
                    sq_i, sp_i = self._eval_sign_at(sd, t_inst)
                    quads_inst.append(sq_i)
                    prims_inst.append(sp_i)

                cam_s, rot_s, fx_s, fy_s = camera_path.eval_at(t_inst)
                hdr_s, v_b, role_b, q_b, trans_b = self._render_substep_hdr(
                    h, w, cam_s, rot_s, fx_s, fy_s, cx, cy,
                    prims_inst, quads_inst,
                    sun_dir, sky_zen, sky_hor, road_alb,
                    is_night=is_night, is_ir=is_ir, is_wet=is_wet,
                    fence_cfg=fence_cfg,
                    record_masks=is_mid,
                    env_style=env_style,
                )
                hdr_accum.add_(hdr_s, alpha=w_s)
                # Per-substep optical glare source before temporal averaging (preserves moving specular/IR streaks!)
                glare_thr = 0.85 if is_ir else 1.05
                bloom_accum.add_((hdr_s - glare_thr).clamp_min(0.0), alpha=w_s)

                vis_trans = trans_b > 1e-4
                for vi_1 in swept_veh_occ:
                    swept_veh_occ[vi_1] |= (v_b == vi_1) & vis_trans
                for qi in swept_quad_occ:
                    swept_quad_occ[qi] |= (q_b == qi) & vis_trans

                if is_mid:
                    mid_veh_buf = v_b
                    mid_role_buf = role_b
                    mid_quad_buf = q_b
                    mid_trans_buf = trans_b
                    mid_quads = quads_inst

            # Optical PSF Glare / IR Halation in Linear HDR space
            if float(bloom_accum.max().item()) > 0.015:
                b_low = F.avg_pool2d(bloom_accum.unsqueeze(0), kernel_size=4, stride=4)
                b_mid = F.avg_pool2d(b_low, kernel_size=3, stride=1, padding=1)
                bloom = F.interpolate(b_mid, size=(h, w), mode="bilinear", align_corners=False)[0]
                hdr_accum = hdr_accum + bloom * (0.62 if is_ir else (0.52 if is_night else 0.24))

            # Exposure-duration photon integration + Causal 1-frame-delayed AE gain!
            sensor_lin = hdr_accum * exp_charge_scale
            ae_gain_used = float(ae_gain)
            exp_hdr = sensor_lin * ae_gain_used

            # Update AE controller state AFTER frame `k_idx` exposure for frame `k_idx + 1`
            mean_l = float((0.2126 * sensor_lin[0] + 0.7152 * sensor_lin[1] + 0.0722 * sensor_lin[2]).mean().item())
            desired_gain = float(np.clip(target_l / max(0.012, mean_l), 0.20, 9.0))
            dt_ae = (timestamps[k_idx + 1] - timestamps[k_idx]) if (k_idx + 1 < K) else base_dt
            alpha_ae = 1.0 - math.exp(-max(1e-3, dt_ae) / 0.26)
            ae_gain = (1.0 - alpha_ae) * ae_gain + alpha_ae * desired_gain

            # Linear-space Poisson-Gaussian shot + read CMOS sensor noise BEFORE tone-mapping & sRGB transfer!
            noise_gen = torch.Generator(device=dev)
            noise_gen.manual_seed((int(seed) * 131071 + k_idx * 8191) & 0x7FFFFFFF)
            gain_mult = 1.0 + 0.30 * max(0.0, ae_gain_used - 1.0)
            shot_k = (0.0018 if is_night else 0.0005) * gain_mult
            read_std = (0.0075 if is_night else 0.0022) * gain_mult
            n_ch = 1 if is_ir else 3
            raw_signal = exp_hdr[:n_ch].clamp_min(0.0)
            if self.config.sensor_noise:
                raw_shot = torch.poisson(raw_signal / shot_k, generator=noise_gen) * shot_k
                raw_read = torch.randn(raw_signal.shape, generator=noise_gen, device=dev) * read_std
                noisy_lin = (raw_shot + raw_read).clamp_min(0.0).expand(3, h, w)
            else:
                noisy_lin = exp_hdr.clamp_min(0.0)
            lum_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=dev)[:, None, None]
            sensor_lum_cpu = (exp_hdr * lum_weights).sum(0).cpu().numpy()
            variance = shot_k * exp_hdr.clamp_min(0.0) + read_std ** 2
            sensor_var = variance[0] if is_ir else (variance * lum_weights.square()).sum(0)
            sensor_sigma_cpu = sensor_var.sqrt().cpu().numpy()
            # Save before isolated-vehicle ownership renders can replace the engine's diagnostic buffer.
            shadow_vis_cpu = self._last_mid_shadow_vis.cpu().numpy().copy()

            # Reinhard sensor saturation + sRGB transfer (monochrome preserved for 850nm/940nm Night-IR)
            toned = noisy_lin / (1.0 + 0.78 * noisy_lin)
            srgb = torch.where(toned <= 0.0031308, toned * 12.92, 1.055 * toned.clamp_min(1e-6).pow(1.0 / 2.4) - 0.055)
            srgb = srgb.clamp(0.0, 1.0)
            frames_out.append(srgb)

            # Extract exact mid-exposure (`t == t_c`) Vehicle & Plate/Sign Ground Truth + Exposure Sweep Ignore Regions
            veh_gts: list[Scene3DVehicleGT] = []
            plate_gts: list[Scene3DPlateGT] = []
            ignore_bboxes: list[tuple[float, float, float, float]] = []
            ignore_cls_ids: list[int] = []
            occ_veh_tids: list[int] = []
            occ_plate_tids: list[int] = []
            emitted_veh_indices: set[int] = set()
            emitted_quad_indices: set[int] = set()

            assert (
                mid_veh_buf is not None
                and mid_role_buf is not None
                and mid_quad_buf is not None
                and mid_trans_buf is not None
            )
            veh_buf_cpu = mid_veh_buf.cpu().numpy()
            role_buf_cpu = mid_role_buf.cpu().numpy()
            quad_buf_cpu = mid_quad_buf.cpu().numpy()
            trans_cpu = mid_trans_buf.cpu().numpy()

            for vi_1, vs in enumerate(veh_specs, start=1):
                cov_v = np.clip((veh_buf_cpu == vi_1).astype(np.float32) * trans_cpu, 0.0, 1.0)
                vis_mask = cov_v > 0.05
                vis_px_cnt = int(np.count_nonzero(vis_mask))
                vis_opt_sum = float(cov_v.sum())
                if vis_px_cnt < 16 or vis_opt_sum < 8.0:
                    continue

                # Unrestricted near-plane-clipped 3D amodal bounding box (all primitives + attached quads!)
                am_uv_parts: list[np.ndarray] = []
                for vp in mid_veh_prims.get(vi_1, []):
                    uv_p = self._clip_obb_to_near_plane_uv(
                        np.array(vp.center, dtype=np.float32),
                        vp.rot_w2l.T,
                        vp.half_ext,
                        cam_o, R_cam, fx, fy, cx, cy, z_near=0.16,
                    )
                    if uv_p.shape[0] > 0:
                        am_uv_parts.append(uv_p)
                for qd_v in mid_quads[:num_veh_quads]:
                    if qd_v.veh_idx == vi_1:
                        uv_qv = self._clip_quad_to_near_plane_uv(
                            qd_v, cam_o, R_cam, fx, fy, cx, cy, z_near=0.16,
                        )
                        if uv_qv.shape[0] > 0:
                            am_uv_parts.append(uv_qv)
                if not am_uv_parts:
                    continue
                uv_am_all = np.concatenate(am_uv_parts, axis=0)
                amodal_box = np.array(
                    [
                        float(uv_am_all[:, 0].min()),
                        float(uv_am_all[:, 1].min()),
                        float(uv_am_all[:, 0].max()),
                        float(uv_am_all[:, 1].max()),
                    ],
                    dtype=np.float32,
                )

                ys_v, xs_v = np.nonzero(vis_mask)
                vis_box = np.array(
                    [float(xs_v.min()), float(ys_v.min()), float(xs_v.max() + 1), float(ys_v.max() + 1)],
                    dtype=np.float32,
                )
                if (vis_box[2] - vis_box[0]) < 8.0 or (vis_box[3] - vis_box[1]) < 6.0:
                    continue

                # Fractional optical 64x64 vehicle cutout inside `vis_box` (attenuated by foreground fence wires!)
                ix0, iy0 = int(max(0, math.floor(vis_box[0]))), int(max(0, math.floor(vis_box[1])))
                ix1, iy1 = int(min(w, math.ceil(vis_box[2]))), int(min(h, math.ceil(vis_box[3])))
                crop_cov = cov_v[iy0:iy1, ix0:ix1]
                roi_mask_64 = cv2.resize(crop_cov, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
                np.clip(roi_mask_64, 0.0, 1.0, out=roi_mask_64)

                # Conservative Cabin Privacy Box (`CabinRedactor`): near-plane-clipped cabin OBBs unioned with rendered glass!
                cab_uv_parts: list[np.ndarray] = []
                for c_w, R_c, he_c in mid_cabins.get(vi_1, []):
                    uv_c = self._clip_obb_to_near_plane_uv(
                        c_w, R_c, he_c, cam_o, R_cam, fx, fy, cx, cy, z_near=0.16,
                    )
                    if uv_c.shape[0] > 0:
                        cab_uv_parts.append(uv_c)

                glass_mask = (veh_buf_cpu == vi_1) & (role_buf_cpu == _ROLE_CABIN_GLASS) & (trans_cpu > 0.01)
                has_glass_px = bool(np.any(glass_mask))
                cx0, cy0, cx1, cy1 = float(w), float(h), 0.0, 0.0
                if cab_uv_parts:
                    uv_cab_all = np.concatenate(cab_uv_parts, axis=0)
                    cx0 = float(np.clip(uv_cab_all[:, 0].min(), 0.0, float(w)))
                    cy0 = float(np.clip(uv_cab_all[:, 1].min(), 0.0, float(h)))
                    cx1 = float(np.clip(uv_cab_all[:, 0].max(), 0.0, float(w)))
                    cy1 = float(np.clip(uv_cab_all[:, 1].max(), 0.0, float(h)))
                if has_glass_px:
                    ys_g, xs_g = np.nonzero(glass_mask)
                    cx0 = min(cx0, float(xs_g.min()))
                    cy0 = min(cy0, float(ys_g.min()))
                    cx1 = max(cx1, float(xs_g.max() + 1))
                    cy1 = max(cy1, float(ys_g.max() + 1))
                if (cx1 - cx0) >= 1.0 and (cy1 - cy0) >= 1.0:
                    cabin_valid = True
                    cabin_box = np.array([cx0, cy0, cx1, cy1], dtype=np.float32)
                else:
                    cabin_valid = False
                    cabin_box = np.zeros((4,), dtype=np.float32)

                # World Z = 0.0 road-plane footprint (Sutherland-Hodgman near-plane clipped; NEVER drops cabin/vehicle GT!)
                fp_poly_w, fp_ok = self._clip_z0_rect_to_near_plane(mid_footprints[vi_1], cam_o, R_cam, z_near=0.22)
                if fp_ok:
                    fp_uv, _ = self._project_points(fp_poly_w, cam_o, R_cam, fx, fy, cx, cy)
                else:
                    fp_uv = np.zeros((0, 2), dtype=np.float32)

                art_fp_uv: np.ndarray | None = None
                art_w = mid_art_footprints.get(vi_1)
                if art_w is not None:
                    art_parts_uv: list[np.ndarray] = []
                    tr_c, tr_ok = self._clip_z0_rect_to_near_plane(art_w[:4], cam_o, R_cam, z_near=0.22)
                    tl_c, tl_ok = self._clip_z0_rect_to_near_plane(art_w[4:], cam_o, R_cam, z_near=0.22)
                    if tr_ok:
                        tr_uv, _ = self._project_points(tr_c, cam_o, R_cam, fx, fy, cx, cy)
                        art_parts_uv.append(tr_uv)
                    if tl_ok:
                        tl_uv, _ = self._project_points(tl_c, cam_o, R_cam, fx, fy, cx, cy)
                        art_parts_uv.append(tl_uv)
                    if art_parts_uv:
                        art_fp_uv = np.concatenate(art_parts_uv, axis=0).astype(np.float32)

                veh_quads_mid = [q for q in mid_quads[:num_veh_quads] if q.veh_idx == vi_1]
                iso_px = self._count_isolated_vehicle_pixels(
                    h, w, cam_o, R_cam, fx, fy, cx, cy,
                    mid_veh_prims.get(vi_1, []),
                    veh_quads=veh_quads_mid,
                    veh_idx=vi_1,
                )
                vis_frac = float(np.clip(vis_opt_sum / float(max(1, iso_px)), 0.0, 1.0))

                app_sig = np.array(
                    [
                        vs.paint_lin[0], vs.paint_lin[1], vs.paint_lin[2],
                        vs.secondary_lin[0], vs.secondary_lin[1], vs.secondary_lin[2],
                        vs.length_m / 14.0, float(vs.subtype) / 8.0,
                    ],
                    dtype=np.float32,
                )
                emitted_veh_indices.add(vi_1)
                veh_gts.append(
                    Scene3DVehicleGT(
                        track_id=vs.track_id,
                        vehicle_subtype=vs.subtype,
                        archetype_name=vs.archetype,
                        bbox_xyxy=vis_box,
                        amodal_bbox_xyxy=amodal_box,
                        cabin_bbox_xyxy=cabin_box,
                        ground_footprint_xy=fp_uv,
                        visible_roi_mask=roi_mask_64,
                        visibility=vis_frac,
                        appearance_signature=app_sig,
                        cabin_valid=cabin_valid,
                        footprint_valid=bool(fp_ok),
                        articulated_footprint_xy=art_fp_uv,
                    )
                )

            # Extract exact mid-exposure 4-corner plates & hard-negative signs (unified `facing_cos > 0.06`!)
            quad_tids: list[int] = []
            veh_hn_ordinal: dict[int, int] = {}
            for q_idx, qd in enumerate(mid_quads):
                if qd.veh_idx > 0:
                    vs_p = veh_specs[qd.veh_idx - 1]
                    v_tid = vs_p.track_id
                    if qd.cls_id == CLS_PLATE:
                        p_tid = vs_p.track_id * 10 + (1 if qd.plate_side == SIDE_REAR else 0)
                    else:
                        ord_k = veh_hn_ordinal.get(qd.veh_idx, 0)
                        veh_hn_ordinal[qd.veh_idx] = ord_k + 1
                        p_tid = vs_p.track_id * 10 + 2 + min(7, ord_k)
                else:
                    v_tid = 0
                    si_idx = max(0, q_idx - num_veh_quads)
                    p_tid = int(sign_defs[si_idx]["track_id"]) if si_idx < len(sign_defs) else (500 + q_idx)
                quad_tids.append(p_tid)

                to_cam = cam_o - qd.center
                dist_q = float(np.linalg.norm(to_cam))
                if dist_q < 0.25 or float(np.dot(qd.normal, to_cam)) / dist_q <= 0.06:
                    continue

                c4_w = self._quad_corners_w(qd)
                c4_uv, c4_z = self._project_points(c4_w, cam_o, R_cam, fx, fy, cx, cy)
                if np.any(c4_z <= 0.20):
                    continue

                pw_px = 0.5 * (float(np.linalg.norm(c4_uv[1] - c4_uv[0])) + float(np.linalg.norm(c4_uv[2] - c4_uv[3])))
                ph_px = 0.5 * (float(np.linalg.norm(c4_uv[3] - c4_uv[0])) + float(np.linalg.norm(c4_uv[2] - c4_uv[1])))
                if pw_px < 5.5 or ph_px < 2.8:
                    continue

                bx0 = float(np.clip(c4_uv[:, 0].min(), 0.0, w - 1.0))
                by0 = float(np.clip(c4_uv[:, 1].min(), 0.0, h - 1.0))
                bx1 = float(np.clip(c4_uv[:, 0].max(), 0.0, w))
                by1 = float(np.clip(c4_uv[:, 1].max(), 0.0, h))
                if (bx1 - bx0) < 4.5 or (by1 - by0) < 2.5:
                    continue

                # Common-transmittance optical visibility (attenuated by foreground chain-link fence wires!)
                q_cov_map = (quad_buf_cpu == q_idx).astype(np.float32) * trans_cpu
                q_opt_px = float(q_cov_map.sum())
                poly_area = max(
                    1.0,
                    0.5 * abs(float(np.sum(c4_uv[:, 0] * np.roll(c4_uv[:, 1], -1) - np.roll(c4_uv[:, 0], -1) * c4_uv[:, 1]))),
                )
                vis_q = float(np.clip(q_opt_px / poly_area, 0.0, 1.0))
                min_vis_q = 0.20 if qd.cls_id == CLS_PLATE else 0.35
                if vis_q < min_vis_q:
                    continue

                blur_vec = quad_blur_vecs[q_idx] if q_idx < len(quad_blur_vecs) else np.zeros((2,), dtype=np.float32)
                facing_cos = float(np.dot(qd.normal, to_cam)) / dist_q
                (is_plate_f, p_vis_16x32, geom_usable_f, slot_read_12,
                 support_f, min_edge_f, blur_f, illum_f, snr_f, slot_snr_12,
                 shadow_f) = self._compute_plate_supervision_targets(
                    qd=qd,
                    q_cov_map=q_cov_map,
                    c4_uv=c4_uv,
                    cam_o=cam_o,
                    R_cam=R_cam,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    pw_px=pw_px,
                    ph_px=ph_px,
                    facing_cos=facing_cos,
                    vis_q=vis_q,
                    blur_vec=blur_vec,
                    sensor_lum_map=sensor_lum_cpu,
                    sensor_sigma_map=sensor_sigma_cpu,
                    shadow_vis_map=shadow_vis_cpu,
                )
                emitted_quad_indices.add(q_idx)
                plate_gts.append(
                    Scene3DPlateGT(
                        cls_id=qd.cls_id,
                        track_id=p_tid,
                        veh_track_id=v_tid,
                        bbox_xyxy=np.array([bx0, by0, bx1, by1], dtype=np.float32),
                        corners_xy=c4_uv,
                        blur_vec_xy=blur_vec,
                        visibility=vis_q,
                        plate_side=qd.plate_side,
                        is_large=qd.is_large_plate,
                        overhang_shadow=shadow_f,
                        plate_text_str=qd.plate_text_str,
                        plate_type_str=qd.plate_type_str,
                        is_plate=is_plate_f,
                        plate_vis_mask=p_vis_16x32,
                        geometry_usable=geom_usable_f,
                        slot_readability=slot_read_12,
                        visible_support_fraction=support_f,
                        min_projected_edge_px=min_edge_f,
                        blur_magnitude_px=blur_f,
                        local_illuminance=illum_f,
                        local_contrast_snr=snr_f,
                        slot_contrast_snr=slot_snr_12,
                    )
                )

            # Populate `occluded_*_track_ids` and `ignore_bboxes_xyxy` for ANY rendered/swept positive excluded at `t_c`
            # (covers excluded `CLS_VEHICLE`, `CLS_PLATE`, AND Gen-4 `CLS_SIGNBOARD_HARDNEG` down to >= 1 pixel!)
            for vi_1, vs in enumerate(veh_specs, start=1):
                if vi_1 not in emitted_veh_indices:
                    occ_veh_tids.append(int(vs.track_id))
                    sw_v = swept_veh_occ[vi_1].cpu().numpy()
                    if int(np.count_nonzero(sw_v)) >= 1:
                        ys_i, xs_i = np.nonzero(sw_v)
                        ignore_bboxes.append(
                            (
                                float(max(0, int(xs_i.min()) - 1)),
                                float(max(0, int(ys_i.min()) - 1)),
                                float(min(w, int(xs_i.max()) + 2)),
                                float(min(h, int(ys_i.max()) + 2)),
                            )
                        )
                        ignore_cls_ids.append(CLS_VEHICLE)

            for q_idx, qd in enumerate(mid_quads):
                if q_idx not in emitted_quad_indices:
                    if qd.cls_id == CLS_PLATE:
                        occ_plate_tids.append(int(quad_tids[q_idx]))
                    if qd.cls_id in (CLS_PLATE, CLS_SIGNBOARD_HARDNEG):
                        sw_q = swept_quad_occ[q_idx].cpu().numpy()
                        if int(np.count_nonzero(sw_q)) >= 1:
                            ys_i, xs_i = np.nonzero(sw_q)
                            ignore_bboxes.append(
                                (
                                    float(max(0, int(xs_i.min()) - 2)),
                                    float(max(0, int(ys_i.min()) - 2)),
                                    float(min(w, int(xs_i.max()) + 3)),
                                    float(min(h, int(ys_i.max()) + 3)),
                                )
                            )
                            ignore_cls_ids.append(int(qd.cls_id))

            gts_out.append(
                Scene3DFrameGT(
                    timestamp_s=float(t_c),
                    exposure_s=float(exp_s),
                    is_night_ir=is_ir,
                    has_chainlink_fence=has_fence,
                    plates_and_signs=plate_gts,
                    vehicles=veh_gts,
                    ignore_bboxes_xyxy=ignore_bboxes,
                    ignore_bbox_classes=ignore_cls_ids,
                    occluded_plate_track_ids=occ_plate_tids,
                    occluded_vehicle_track_ids=occ_veh_tids,
                    cam_pos_w=cam_o.copy(),
                    cam_R_c2w=R_cam.copy(),
                    cam_intrinsics_fxfycxcy=(float(fx), float(fy), float(cx), float(cy)),
                    ae_gain_applied=ae_gain_used,
                )
            )

        mode_str = "hard_negative" if is_neg else ("night_gate_ir" if is_ir else ("night_gate" if is_night else cam_mode))
        return Scene3DClipSample(
            frames_chw=torch.stack(frames_out, dim=0),
            frame_gts=gts_out,
            scene_mode=mode_str,
            camera_height_m=float(cam_z),
            camera_pitch_deg=float(pitch_deg),
            domain_split=self.config.domain_split,
            grammar_combo_id=combo[0], radial_dn_order=radial_order,
            font_family_id=self.bank.font_subset[0][0],
            camera_motion_family=camera_family, solar_radius_deg=solar_radius_deg,
        )


class Scene3DClipDataset(Dataset):
    """Deterministic PyTorch `Dataset` yielding `Scene3DClipSample` video clips for Phase 3 & Phase 4 training."""

    def __init__(
        self,
        num_clips: int,
        config: Scene3DClipConfig | None = None,
        start_seed: int = 20264000,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_clips = int(num_clips)
        self.config = config or Scene3DClipConfig()
        self.start_seed = int(start_seed)
        self.device = _resolve_torch_device(device)
        self.device_str = str(self.device)
        self._engine: Scene3DGPUEngine | None = None

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> Scene3DClipSample:
        if self._engine is None:
            self._engine = Scene3DGPUEngine(config=self.config, device=self.device)
        return self._engine.render_clip(seed=self.start_seed + int(idx))


__all__ = [
    "CLS_SIGNBOARD_HARDNEG",
    "NUM_GEN4_DET_CLASSES",
    "SUBTYPE_PASSENGER_KEI",
    "SUBTYPE_VAN_KEITRUCK",
    "SUBTYPE_FLATBED_CARGO",
    "SUBTYPE_BOX_TRUCK",
    "SUBTYPE_DUMP_TRUCK",
    "SUBTYPE_CONCRETE_MIXER",
    "SUBTYPE_TANKER_SPECIAL",
    "SUBTYPE_TRACTOR_TRAILER",
    "SUBTYPE_CRANE_HEAVY",
    "NUM_VEHICLE_SUBTYPES",
    "VEHICLE_SUBTYPE_NAMES",
    "Scene3DPlateGT",
    "Scene3DVehicleGT",
    "Scene3DFrameGT",
    "Scene3DClipSample",
    "Scene3DClipConfig",
    "Scene3DGPUEngine",
    "Scene3DClipDataset",
]
