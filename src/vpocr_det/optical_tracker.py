# SPDX-License-Identifier: Apache-2.0
"""Sub-pixel Coarse-to-Fine ZNCC + 4-DOF Inverse-Compositional Lucas-Kanade Optical Plate Tracker.

Provides deterministic, zero-allocation-light NumPy sub-pixel plate corner lock (`< 0.3 ms` CPU)
for both `VehiclePlateVideoPipeline` and `MultiRateEdgeCascadePipeline`, matching `web/optical_tracker.mjs`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

OPT_TW: int = 40
OPT_TH: int = 20
OPT_PAD: float = 0.08
OPT_SPAN: float = 1.0 + 2.0 * OPT_PAD  # 1.16

_CW: int = 20
_CH: int = 10

_XI_FINE = OPT_SPAN * ((np.arange(OPT_TW, dtype=np.float32) + 0.5) / float(OPT_TW) - 0.5)[None, :]  # (1, 40)
_ETA_FINE = OPT_SPAN * ((np.arange(OPT_TH, dtype=np.float32) + 0.5) / float(OPT_TH) - 0.5)[:, None]  # (20, 1)
_XI_COARSE = OPT_SPAN * ((np.arange(_CW, dtype=np.float32) + 0.5) / float(_CW) - 0.5)[None, :]       # (1, 20)
_ETA_COARSE = OPT_SPAN * ((np.arange(_CH, dtype=np.float32) + 0.5) / float(_CH) - 0.5)[:, None]      # (10, 1)


@dataclass(slots=True)
class QuadBasis:
    mx: float
    my: float
    aux: float
    auy: float
    avx: float
    avy: float
    auvx: float
    auvy: float
    width: float
    height: float


@dataclass(slots=True)
class OpticalPlateRef:
    corners: np.ndarray       # (4, 2) float32
    basis: QuadBasis
    tpl: np.ndarray           # (800,) float32 zero-mean unit-std
    coarse: np.ndarray        # (200,) float32 zero-mean unit-norm
    jac: np.ndarray           # (800, 4) float32 steepest-descent Jacobian
    inv_h: np.ndarray         # (4, 4) float64 regularized inverse Hessian
    sharpness: float


def quad_basis(corners_xy: np.ndarray) -> QuadBasis:
    c = np.asarray(corners_xy, dtype=np.float32).reshape(4, 2)
    mx = float(0.25 * (c[0, 0] + c[1, 0] + c[2, 0] + c[3, 0]))
    my = float(0.25 * (c[0, 1] + c[1, 1] + c[2, 1] + c[3, 1]))
    aux = float(0.5 * (c[1, 0] - c[0, 0] + c[2, 0] - c[3, 0]))
    auy = float(0.5 * (c[1, 1] - c[0, 1] + c[2, 1] - c[3, 1]))
    avx = float(0.5 * (c[3, 0] - c[0, 0] + c[2, 0] - c[1, 0]))
    avy = float(0.5 * (c[3, 1] - c[0, 1] + c[2, 1] - c[1, 1]))
    auvx = float(c[0, 0] - c[1, 0] + c[2, 0] - c[3, 0])
    auvy = float(c[0, 1] - c[1, 1] + c[2, 1] - c[3, 1])
    width = max(2.0, math.hypot(aux, auy))
    height = max(2.0, math.hypot(avx, avy))
    return QuadBasis(mx, my, aux, auy, avx, avy, auvx, auvy, width, height)


def warp_to_corners(basis: QuadBasis, tu: float, tv: float, su: float, sv: float) -> np.ndarray:
    u0 = OPT_SPAN * tu - 0.5 * su
    u1 = OPT_SPAN * tu + 0.5 * su
    v0 = OPT_SPAN * tv - 0.5 * sv
    v1 = OPT_SPAN * tv + 0.5 * sv
    uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float32)
    u = uv[:, 0]
    v = uv[:, 1]
    px = basis.mx + u * basis.aux + v * basis.avx + u * v * basis.auvx
    py = basis.my + u * basis.auy + v * basis.avy + u * v * basis.auvy
    return np.stack([px, py], axis=-1).astype(np.float32)


def corners_to_warp(basis: QuadBasis, corners_xy: np.ndarray) -> tuple[float, float, float, float]:
    ib = quad_basis(corners_xy)
    dx = ib.mx - basis.mx
    dy = ib.my - basis.my
    det = basis.aux * basis.avy - basis.auy * basis.avx
    if abs(det) > 1e-6:
        tu = (dx * basis.avy - dy * basis.avx) / (det * OPT_SPAN)
        tv = (basis.aux * dy - basis.auy * dx) / (det * OPT_SPAN)
    else:
        tu, tv = 0.0, 0.0
    su = float(np.clip(ib.width / basis.width, 0.65, 1.55))
    sv = float(np.clip(ib.height / basis.height, 0.65, 1.55))
    return float(tu), float(tv), su, sv


def extract_roi_gray(
    frame_rgb: np.ndarray,
    cx: float,
    cy: float,
    half_w: float,
    half_h: float,
) -> tuple[np.ndarray, int, int] | None:
    """Extract a tight grayscale float32 sub-window around `(cx, cy)` to avoid full-frame RGB->Gray cost."""
    h, w = frame_rgb.shape[:2]
    x0 = max(0, int(math.floor(cx - half_w - 4.0)))
    y0 = max(0, int(math.floor(cy - half_h - 4.0)))
    x1 = min(w, int(math.ceil(cx + half_w + 4.0)))
    y1 = min(h, int(math.ceil(cy + half_h + 4.0)))
    if x1 - x0 < 6 or y1 - y0 < 4:
        return None
    sub = frame_rgb[y0:y1, x0:x1]
    gray = (
        0.299 * sub[:, :, 0].astype(np.float32)
        + 0.587 * sub[:, :, 1].astype(np.float32)
        + 0.114 * sub[:, :, 2].astype(np.float32)
    )
    return gray, x0, y0


def _sample_bilinear_grid(gray: np.ndarray, off_x: int, off_y: int, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    gh, gw = gray.shape
    lx = np.clip(px - float(off_x), 0.0, float(gw) - 1.001)
    ly = np.clip(py - float(off_y), 0.0, float(gh) - 1.001)
    x0 = lx.astype(np.int32)
    y0 = ly.astype(np.int32)
    fx = lx - x0
    fy = ly - y0
    v00 = gray[y0, x0]
    v01 = gray[y0, x0 + 1]
    v10 = gray[y0 + 1, x0]
    v11 = gray[y0 + 1, x0 + 1]
    return (v00 * (1.0 - fx) + v01 * fx) * (1.0 - fy) + (v10 * (1.0 - fx) + v11 * fx) * fy


def create_optical_ref(frame_rgb: np.ndarray, corners_xy: np.ndarray) -> OpticalPlateRef | None:
    """Create a 4-DOF IC-LK + ZNCC optical reference template, rejecting low-stroke horizontal stripes."""
    c = np.asarray(corners_xy, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(c)):
        return None
    basis = quad_basis(c)
    if basis.width < 10.0 or basis.height < 4.8:
        return None

    roi = extract_roi_gray(frame_rgb, basis.mx, basis.my, 0.85 * basis.width, 0.85 * basis.height)
    if roi is None:
        return None
    gray, ox, oy = roi
    u = _XI_FINE
    v = _ETA_FINE
    px = basis.mx + u * basis.aux + v * basis.avx + u * v * basis.auvx
    py = basis.my + u * basis.auy + v * basis.avy + u * v * basis.auvy
    patch = _sample_bilinear_grid(gray, ox, oy, px, py)  # (20, 40)

    std = float(np.std(patch))
    if std < 2.8:
        return None
    tpl_2d = ((patch - float(np.mean(patch))) / std).astype(np.float32)

    coarse_2d = 0.25 * (
        tpl_2d[0::2, 0::2] + tpl_2d[0::2, 1::2] + tpl_2d[1::2, 0::2] + tpl_2d[1::2, 1::2]
    )
    coarse_flat = (coarse_2d - float(np.mean(coarse_2d))).reshape(-1)
    c_norm = float(np.linalg.norm(coarse_flat))
    if c_norm < 1e-5:
        return None
    coarse = (coarse_flat / c_norm).astype(np.float32)

    # Central-difference spatial gradients with border taper
    gx = np.empty_like(tpl_2d)
    gy = np.empty_like(tpl_2d)
    gx[:, 1:-1] = 0.5 * (tpl_2d[:, 2:] - tpl_2d[:, :-2])
    gx[:, 0] = 0.5 * (tpl_2d[:, 1] - tpl_2d[:, 0])
    gx[:, -1] = 0.5 * (tpl_2d[:, -1] - tpl_2d[:, -2])
    gy[1:-1, :] = 0.5 * (tpl_2d[2:, :] - tpl_2d[:-2, :])
    gy[0, :] = 0.5 * (tpl_2d[1, :] - tpl_2d[0, :])
    gy[-1, :] = 0.5 * (tpl_2d[-1, :] - tpl_2d[-2, :])

    wx = np.ones((1, OPT_TW), dtype=np.float32)
    wy = np.ones((OPT_TH, 1), dtype=np.float32)
    wx[0, 0] = wx[0, -1] = 0.35
    wy[0, 0] = wy[-1, 0] = 0.35
    w_mask = wx * wy
    gx *= w_mask
    gy *= w_mask

    xc = (np.arange(OPT_TW, dtype=np.float32) + 0.5 - 0.5 * float(OPT_TW))[None, :]
    yc = (np.arange(OPT_TH, dtype=np.float32) + 0.5 - 0.5 * float(OPT_TH))[:, None]
    jac = np.stack(
        [
            gx.reshape(-1),
            gy.reshape(-1),
            (gx * xc).reshape(-1),
            (gy * yc).reshape(-1),
        ],
        axis=-1,
    ).astype(np.float32)

    h_mat = (jac.T.astype(np.float64)) @ jac.astype(np.float64)
    # Reject pure horizontal stripes (bumper grilles/hoods lacking vertical character strokes)
    min_h0 = 5.5 if basis.width < 18.0 else 14.0
    if float(h_mat[0, 0]) < min_h0 or float(h_mat[0, 0]) < 0.09 * float(h_mat[1, 1]):
        return None
    sharpness = float(h_mat[0, 0] + 0.5 * h_mat[1, 1])

    h_mat[0, 0] += 1.5
    h_mat[1, 1] += 1.5
    h_mat[2, 2] += 120.0
    h_mat[3, 3] += 80.0
    try:
        inv_h = np.linalg.inv(h_mat)
    except np.linalg.LinAlgError:
        return None

    return OpticalPlateRef(
        corners=c.copy(),
        basis=basis,
        tpl=tpl_2d.reshape(-1),
        coarse=coarse,
        jac=jac,
        inv_h=inv_h,
        sharpness=sharpness,
    )


def track_plate_optical(
    ref: OpticalPlateRef | None,
    frame_rgb: np.ndarray,
    init_corners_xy: np.ndarray | None = None,
    *,
    search_radius_px: float = 28.0,
    search_radius_y: float | None = None,
    allow_scale_search: bool = False,
    min_zncc: float = 0.58,
    max_iter: int = 6,
) -> tuple[bool, float, np.ndarray | None]:
    """Track `ref` onto `frame_rgb` starting from `init_corners_xy`, returning `(ok, zncc, corners_4x2)`."""
    if ref is None:
        return False, -1.0, None
    h, w = frame_rgb.shape[:2]
    basis = ref.basis
    init_c = ref.corners if init_corners_xy is None else np.asarray(init_corners_xy, dtype=np.float32).reshape(4, 2)
    init_tu, init_tv, su, sv = corners_to_warp(basis, init_c)
    tu, tv = init_tu, init_tv

    eff_rad_x = min(float(search_radius_px), max(12.0, 1.35 * basis.width * su))
    eff_rad_y = min(
        float(search_radius_px * 0.62 if search_radius_y is None else search_radius_y),
        max(8.0, 1.05 * basis.height * sv),
    )

    pred_cx = basis.mx + OPT_SPAN * tu * basis.aux + OPT_SPAN * tv * basis.avx
    pred_cy = basis.my + OPT_SPAN * tu * basis.auy + OPT_SPAN * tv * basis.avy
    roi = extract_roi_gray(
        frame_rgb,
        pred_cx,
        pred_cy,
        eff_rad_x + 0.90 * basis.width * su,
        eff_rad_y + 0.90 * basis.height * sv,
    )
    if roi is None:
        return False, -1.0, None
    gray, ox, oy = roi

    def _eval_coarse(c_tu: float, c_tv: float, c_su: float, c_sv: float) -> float:
        cx_c = basis.mx + OPT_SPAN * c_tu * basis.aux + OPT_SPAN * c_tv * basis.avx
        cy_c = basis.my + OPT_SPAN * c_tu * basis.auy + OPT_SPAN * c_tv * basis.avy
        if cx_c < 2.0 or cx_c > float(w - 3) or cy_c < 2.0 or cy_c > float(h - 3):
            return -1.0
        u = OPT_SPAN * c_tu + c_su * _XI_COARSE
        v = OPT_SPAN * c_tv + c_sv * _ETA_COARSE
        px = basis.mx + u * basis.aux + v * basis.avx + u * v * basis.auvx
        py = basis.my + u * basis.auy + v * basis.avy + u * v * basis.auvy
        patch_c = _sample_bilinear_grid(gray, ox, oy, px, py).reshape(-1)
        d = patch_c - float(np.mean(patch_c))
        sq = float(np.dot(d, d))
        if sq < 6.25 * float(_CW * _CH):
            return -1.0
        return float(np.dot(d, ref.coarse) / math.sqrt(sq))

    if eff_rad_x > 1.5:
        rad_u = float(np.clip(eff_rad_x / max(8.0, basis.width * OPT_SPAN), 0.02, 1.6))
        rad_v = float(np.clip(eff_rad_y / max(6.0, basis.height * OPT_SPAN), 0.02, 1.3))
        steps_u = min(10, max(2, int(round(eff_rad_x / max(1.1, 0.14 * basis.width)))))
        steps_v = min(8, max(2, int(round(eff_rad_y / max(0.9, 0.18 * basis.height)))))
        step_u = rad_u / float(steps_u)
        step_v = rad_v / float(steps_v)

        best_tu, best_tv, best_su, best_sv = tu, tv, su, sv
        best_raw = _eval_coarse(tu, tv, su, sv)
        best_pen = best_raw

        for iv in range(-steps_v, steps_v + 1):
            cand_tv = tv + iv * step_v
            vn2 = (float(iv) / float(steps_v)) ** 2
            for iu in range(-steps_u, steps_u + 1):
                if iu == 0 and iv == 0:
                    continue
                cand_tu = tu + iu * step_u
                raw_sc = _eval_coarse(cand_tu, cand_tv, su, sv)
                pen_sc = raw_sc - 0.055 * ((float(iu) / float(steps_u)) ** 2 + vn2)
                if pen_sc > best_pen:
                    best_pen = pen_sc
                    best_raw = raw_sc
                    best_tu, best_tv = cand_tu, cand_tv

        fine_u = 0.45 * step_u
        fine_v = 0.45 * step_v
        base_tu, base_tv = best_tu, best_tv
        for iv in (-1, 0, 1):
            for iu in (-1, 0, 1):
                if iu == 0 and iv == 0:
                    continue
                cand_tu = base_tu + iu * fine_u
                cand_tv = base_tv + iv * fine_v
                raw_sc = _eval_coarse(cand_tu, cand_tv, best_su, best_sv)
                pen_sc = raw_sc - 0.055 * (((cand_tu - init_tu) / rad_u) ** 2 + ((cand_tv - init_tv) / rad_v) ** 2)
                if pen_sc > best_pen:
                    best_pen = pen_sc
                    best_raw = raw_sc
                    best_tu, best_tv = cand_tu, cand_tv

        if allow_scale_search and best_raw > 0.30:
            for factor in (0.93, 1.07, 1.14):
                c_su = float(np.clip(su * factor, 0.68, 1.52))
                c_sv = float(np.clip(sv * factor, 0.68, 1.52))
                raw_sc = _eval_coarse(best_tu, best_tv, c_su, c_sv)
                if raw_sc > best_raw + 0.012:
                    best_raw = raw_sc
                    best_su, best_sv = c_su, c_sv

        if best_raw < 0.28:
            return False, best_raw, None
        tu, tv, su, sv = best_tu, best_tv, best_su, best_sv

    # Stage 2: Sub-pixel 4-DOF Inverse-Compositional Gauss-Newton refinement on 40x20 grid
    n_fine = float(OPT_TW * OPT_TH)
    for _ in range(max(1, int(max_iter))):
        u = OPT_SPAN * tu + su * _XI_FINE
        v = OPT_SPAN * tv + sv * _ETA_FINE
        px = basis.mx + u * basis.aux + v * basis.avx + u * v * basis.auvx
        py = basis.my + u * basis.auy + v * basis.avy + u * v * basis.auvy
        samp = _sample_bilinear_grid(gray, ox, oy, px, py).reshape(-1)
        d = samp - float(np.mean(samp))
        std = math.sqrt(float(np.dot(d, d)) / n_fine)
        if std < 2.4:
            return False, -1.0, None
        norm_i = d / std
        err = (norm_i - ref.tpl).astype(np.float64)
        b_vec = ref.jac.T.astype(np.float64) @ err
        dp = ref.inv_h @ b_vec

        dx_g = float(np.clip(dp[0], -3.5, 3.5))
        dy_g = float(np.clip(dp[1], -2.5, 2.5))
        d_su = float(np.clip(dp[2], -0.08, 0.08))
        d_sv = float(np.clip(dp[3], -0.08, 0.08))

        inv_du = 1.0 / (1.0 + d_su)
        inv_dv = 1.0 / (1.0 + d_sv)
        tu -= su * inv_du * (dx_g / float(OPT_TW))
        tv -= sv * inv_dv * (dy_g / float(OPT_TH))
        su = float(np.clip(su * inv_du, 0.65, 1.58))
        sv = float(np.clip(sv * inv_dv, 0.65, 1.58))
        ratio = su / sv
        if ratio > 1.18:
            su = sv * 1.18
        elif ratio < 0.84:
            sv = su / 0.84

        if abs(dx_g) < 0.025 and abs(dy_g) < 0.025 and abs(d_su) < 0.002 and abs(d_sv) < 0.002:
            break

    u = OPT_SPAN * tu + su * _XI_FINE
    v = OPT_SPAN * tv + sv * _ETA_FINE
    px = basis.mx + u * basis.aux + v * basis.avx + u * v * basis.auvx
    py = basis.my + u * basis.auy + v * basis.avy + u * v * basis.auvy
    samp = _sample_bilinear_grid(gray, ox, oy, px, py).reshape(-1)
    d = samp - float(np.mean(samp))
    sq = float(np.dot(d, d))
    final_zncc = float(np.dot(d, ref.tpl) / math.sqrt(sq * n_fine)) if sq > 5.76 * n_fine else -1.0

    if final_zncc < float(min_zncc):
        return False, final_zncc, None

    shift_x = abs((tu - init_tu) * OPT_SPAN * basis.width)
    shift_y = abs((tv - init_tv) * OPT_SPAN * basis.height)
    if shift_x > eff_rad_x * 1.25 + 4.0 or shift_y > eff_rad_y * 1.25 + 4.0:
        return False, final_zncc, None

    corners = warp_to_corners(basis, tu, tv, su, sv)
    if not np.all(np.isfinite(corners)):
        return False, final_zncc, None
    if np.any(corners[:, 0] < -4.0) or np.any(corners[:, 0] > float(w + 4)) or np.any(corners[:, 1] < -4.0) or np.any(corners[:, 1] > float(h + 4)):
        return False, final_zncc, None
    edges = np.roll(corners, -1, axis=0) - corners
    next_edges = np.roll(edges, -1, axis=0)
    if np.any(edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0] <= 0.0):
        return False, final_zncc, None

    return True, final_zncc, corners


VEH_TPL_W: int = 16
VEH_TPL_H: int = 16
_VEH_CW: int = 8
_VEH_CH: int = 8

_VEH_U16 = (0.10 + 0.80 * (np.arange(VEH_TPL_W, dtype=np.float32) + 0.5) / float(VEH_TPL_W))[None, :]  # (1, 16)
_VEH_V16 = (0.10 + 0.80 * (np.arange(VEH_TPL_H, dtype=np.float32) + 0.5) / float(VEH_TPL_H))[:, None]  # (16, 1)
_VEH_U8 = (0.10 + 0.80 * (np.arange(_VEH_CW, dtype=np.float32) + 0.5) / float(_VEH_CW))[None, :]       # (1, 8)
_VEH_V8 = (0.10 + 0.80 * (np.arange(_VEH_CH, dtype=np.float32) + 0.5) / float(_VEH_CH))[:, None]       # (8, 1)


@dataclass(slots=True)
class VehicleAppearanceRef:
    """Compact 16x16 grayscale ZNCC patch + 24D RGB color histogram + 32D appearance vector."""

    tpl: np.ndarray         # (256,) float32 zero-mean unit-L2-norm
    coarse: np.ndarray      # (64,) float32 zero-mean unit-L2-norm
    tpl_std: float
    color_hist: np.ndarray  # (24,) float32 L2-normalized sqrt-prob 3x8 RGB histogram
    app32: np.ndarray       # (32,) float32 unit-L2-norm [24D RGB hist + 8D 4x2 luma grid]


def extract_vehicle_appearance(
    frame_rgb: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
) -> VehicleAppearanceRef | None:
    """Extract interior-80% 16x16 ZNCC template, 24D RGB histogram, and 32D appearance vector."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        return None
    h, w = frame_rgb.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    bw = x1 - x0
    bh = y1 - y0
    if not (math.isfinite(bw) and math.isfinite(bh)) or bw < 10.0 or bh < 10.0:
        return None

    # Bilinearly sample interior 80% on 16x16 grid directly from tight RGB slice
    ix0 = max(0, int(math.floor(x0 + 0.08 * bw)) - 1)
    iy0 = max(0, int(math.floor(y0 + 0.08 * bh)) - 1)
    ix1 = min(w, int(math.ceil(x1 - 0.08 * bw)) + 2)
    iy1 = min(h, int(math.ceil(y1 - 0.08 * bh)) + 2)
    if ix1 - ix0 < 3 or iy1 - iy0 < 3:
        return None

    sub = frame_rgb[iy0:iy1, ix0:ix1, :3].astype(np.float32)
    px = x0 + bw * _VEH_U16  # (1, 16)
    py = y0 + bh * _VEH_V16  # (16, 1)
    lx = np.clip(px - float(ix0), 0.0, float(ix1 - ix0) - 1.001)
    ly = np.clip(py - float(iy0), 0.0, float(iy1 - iy0) - 1.001)
    x_idx = lx.astype(np.int32)
    y_idx = ly.astype(np.int32)
    fx = (lx - x_idx)[:, :, None]
    fy = (ly - y_idx)[:, :, None]
    v00 = sub[y_idx, x_idx]
    v01 = sub[y_idx, x_idx + 1]
    v10 = sub[y_idx + 1, x_idx]
    v11 = sub[y_idx + 1, x_idx + 1]
    rgb16 = (v00 * (1.0 - fx) + v01 * fx) * (1.0 - fy) + (v10 * (1.0 - fx) + v11 * fx) * fy  # (16, 16, 3)

    luma16 = 0.299 * rgb16[:, :, 0] + 0.587 * rgb16[:, :, 1] + 0.114 * rgb16[:, :, 2]
    d16 = luma16 - float(np.mean(luma16))
    sq16 = float(np.sum(d16 * d16))
    tpl_std = math.sqrt(sq16 / float(VEH_TPL_W * VEH_TPL_H))
    if tpl_std < 2.2:
        return None
    tpl = (d16.reshape(-1) / math.sqrt(sq16)).astype(np.float32)

    c8_px = x0 + bw * _VEH_U8  # (1, 8)
    c8_py = y0 + bh * _VEH_V8  # (8, 1)
    c8_lx = np.clip(c8_px - float(ix0), 0.0, float(ix1 - ix0) - 1.001)
    c8_ly = np.clip(c8_py - float(iy0), 0.0, float(iy1 - iy0) - 1.001)
    c8_xi = c8_lx.astype(np.int32)
    c8_yi = c8_ly.astype(np.int32)
    c8_fx = (c8_lx - c8_xi)[:, :, None]
    c8_fy = (c8_ly - c8_yi)[:, :, None]
    c8_rgb = (
        (sub[c8_yi, c8_xi] * (1.0 - c8_fx) + sub[c8_yi, c8_xi + 1] * c8_fx) * (1.0 - c8_fy)
        + (sub[c8_yi + 1, c8_xi] * (1.0 - c8_fx) + sub[c8_yi + 1, c8_xi + 1] * c8_fx) * c8_fy
    )
    c8 = 0.299 * c8_rgb[:, :, 0] + 0.587 * c8_rgb[:, :, 1] + 0.114 * c8_rgb[:, :, 2]
    dc8 = c8 - float(np.mean(c8))
    c_sq = float(np.sum(dc8 * dc8))
    if c_sq < 1e-4:
        return None
    coarse = (dc8.reshape(-1) / math.sqrt(c_sq)).astype(np.float32)

    # 24D RGB histogram (3 x 8 bins) with Bhattacharyya sqrt normalization
    bins = np.clip((rgb16 * (8.0 / 256.0)).astype(np.int32), 0, 7)
    hist = np.zeros((24,), dtype=np.float32)
    for ch in range(3):
        hist[ch * 8 : (ch + 1) * 8] = np.bincount(bins[:, :, ch].reshape(-1), minlength=8).astype(np.float32)
    hist = np.sqrt(hist / float(VEH_TPL_W * VEH_TPL_H))
    h_norm = float(np.linalg.norm(hist))
    if h_norm > 1e-6:
        hist /= h_norm

    # 8D 4x2 spatial luma profile normalized to match RGB hist energy
    grid4x2 = luma16.reshape(2, 8, 4, 4).mean(axis=(1, 3)).reshape(-1) / 255.0
    g_d = grid4x2 - float(np.mean(grid4x2))
    g_norm = float(np.linalg.norm(g_d))
    if g_norm > 1e-6:
        g_d = g_d / g_norm

    app32 = np.concatenate([hist * 0.80, g_d.astype(np.float32) * 0.60], axis=0)
    a_norm = float(np.linalg.norm(app32))
    if a_norm > 1e-6:
        app32 /= a_norm

    return VehicleAppearanceRef(
        tpl=tpl,
        coarse=coarse,
        tpl_std=tpl_std,
        color_hist=hist,
        app32=app32,
    )


def track_vehicle_patch_ncc(
    ref: VehicleAppearanceRef | None,
    frame_rgb: np.ndarray,
    pred_bbox_xyxy: tuple[float, float, float, float],
    *,
    max_shift_px: float = 24.0,
    min_ncc: float = 0.42,
) -> tuple[bool, float, tuple[float, float, float, float], float, float]:
    """Coarse-to-fine 2D ZNCC vehicle template shift search (`±max_shift_px`) with flat-asphalt guard."""
    if ref is None:
        return False, -1.0, pred_bbox_xyxy, 0.0, 0.0
    h, w = frame_rgb.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in pred_bbox_xyxy]
    bw = x1 - x0
    bh = y1 - y0
    if bw < 10.0 or bh < 10.0:
        return False, -1.0, pred_bbox_xyxy, 0.0, 0.0

    rad = max(4, int(round(min(float(max_shift_px), max(12.0, 0.45 * max(bw, bh))))))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    roi = extract_roi_gray(frame_rgb, cx, cy, 0.50 * bw + float(rad) + 2.0, 0.50 * bh + float(rad) + 2.0)
    if roi is None:
        return False, -1.0, pred_bbox_xyxy, 0.0, 0.0
    gray, ox, oy = roi
    gh, gw = gray.shape
    min_std = max(2.2, 0.14 * ref.tpl_std)

    base_x8 = (x0 + bw * _VEH_U8) - float(ox)   # (1, 8)
    base_y8 = (y0 + bh * _VEH_V8) - float(oy)   # (8, 1)
    base_x16 = (x0 + bw * _VEH_U16) - float(ox) # (1, 16)
    base_y16 = (y0 + bh * _VEH_V16) - float(oy) # (16, 1)

    def _eval_coarse(dx: float, dy: float) -> float:
        nx0 = x0 + dx
        ny0 = y0 + dy
        if nx0 + 0.5 * bw < -4.0 or nx0 + 0.5 * bw > float(w + 4) or ny0 + 0.5 * bh < -4.0 or ny0 + 0.5 * bh > float(h + 4):
            return -1.0
        lx = np.clip(base_x8 + dx, 0.0, float(gw) - 1.001)
        ly = np.clip(base_y8 + dy, 0.0, float(gh) - 1.001)
        xi = lx.astype(np.int32)
        yi = ly.astype(np.int32)
        fx = lx - xi
        fy = ly - yi
        patch = (
            (gray[yi, xi] * (1.0 - fx) + gray[yi, xi + 1] * fx) * (1.0 - fy)
            + (gray[yi + 1, xi] * (1.0 - fx) + gray[yi + 1, xi + 1] * fx) * fy
        ).reshape(-1)
        d = patch - float(np.mean(patch))
        sq = float(np.dot(d, d))
        if sq < (min_std * min_std) * float(_VEH_CW * _VEH_CH) * 0.45:
            return -1.0
        return float(np.dot(d, ref.coarse) / math.sqrt(sq))

    def _eval_fine(dx: float, dy: float) -> float:
        nx0 = x0 + dx
        ny0 = y0 + dy
        if nx0 + 0.5 * bw < -4.0 or nx0 + 0.5 * bw > float(w + 4) or ny0 + 0.5 * bh < -4.0 or ny0 + 0.5 * bh > float(h + 4):
            return -1.0
        lx = np.clip(base_x16 + dx, 0.0, float(gw) - 1.001)
        ly = np.clip(base_y16 + dy, 0.0, float(gh) - 1.001)
        xi = lx.astype(np.int32)
        yi = ly.astype(np.int32)
        fx = lx - xi
        fy = ly - yi
        patch = (
            (gray[yi, xi] * (1.0 - fx) + gray[yi, xi + 1] * fx) * (1.0 - fy)
            + (gray[yi + 1, xi] * (1.0 - fx) + gray[yi + 1, xi + 1] * fx) * fy
        ).reshape(-1)
        d = patch - float(np.mean(patch))
        sq = float(np.dot(d, d))
        if sq < (min_std * min_std) * float(VEH_TPL_W * VEH_TPL_H):
            return -1.0
        return float(np.dot(d, ref.tpl) / math.sqrt(sq))

    inv_r2 = 1.0 / float(max(16, rad * rad))
    best_dx, best_dy = 0.0, 0.0
    best_pen = _eval_coarse(0.0, 0.0)

    # Stage 1: Coarse 8px grid over ±rad
    step1 = max(4, rad // 3)
    for dy in range(-rad, rad + 1, step1):
        for dx in range(-rad, rad + 1, step1):
            if dx == 0 and dy == 0:
                continue
            sc = _eval_coarse(float(dx), float(dy))
            pen = sc - 0.015 * float(dx * dx + dy * dy) * inv_r2
            if pen > best_pen:
                best_pen = pen
                best_dx, best_dy = float(dx), float(dy)

    # Stage 2: Medium 3px refinement on 8x8
    base_dx, base_dy = best_dx, best_dy
    step2 = max(2, int(round(step1 / 2.5)))
    for oy_s in (-2 * step2, -step2, 0, step2, 2 * step2):
        for ox_s in (-2 * step2, -step2, 0, step2, 2 * step2):
            if ox_s == 0 and oy_s == 0:
                continue
            dx = float(np.clip(base_dx + ox_s, -rad, rad))
            dy = float(np.clip(base_dy + oy_s, -rad, rad))
            sc = _eval_coarse(dx, dy)
            pen = sc - 0.015 * (dx * dx + dy * dy) * inv_r2
            if pen > best_pen:
                best_pen = pen
                best_dx, best_dy = dx, dy

    # Stage 3: Exact 16x16 fine refinement (4px -> 2px -> 1px)
    best_ncc = _eval_fine(best_dx, best_dy)
    best_fine_pen = best_ncc - 0.010 * (best_dx * best_dx + best_dy * best_dy) * inv_r2
    for step3 in (4, 2, 1):
        b_dx, b_dy = best_dx, best_dy
        for oy_s in (-step3, 0, step3):
            for ox_s in (-step3, 0, step3):
                if ox_s == 0 and oy_s == 0:
                    continue
                dx = float(np.clip(b_dx + ox_s, -rad, rad))
                dy = float(np.clip(b_dy + oy_s, -rad, rad))
                sc = _eval_fine(dx, dy)
                pen = sc - 0.010 * (dx * dx + dy * dy) * inv_r2
                if pen > best_fine_pen:
                    best_fine_pen = pen
                    best_ncc = sc
                    best_dx, best_dy = dx, dy

    nx0 = float(np.clip(x0 + best_dx, 0.0, float(w - 2)))
    ny0 = float(np.clip(y0 + best_dy, 0.0, float(h - 2)))
    nx1 = float(np.clip(x1 + best_dx, nx0 + 2.0, float(w)))
    ny1 = float(np.clip(y1 + best_dy, ny0 + 2.0, float(h)))
    return best_ncc >= float(min_ncc), float(best_ncc), (nx0, ny0, nx1, ny1), float(best_dx), float(best_dy)
