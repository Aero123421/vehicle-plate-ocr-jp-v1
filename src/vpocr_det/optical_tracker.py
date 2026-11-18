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
