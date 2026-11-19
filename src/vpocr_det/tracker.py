"""Gen-6 bounded PMBM-style tracker with uncertain camera motion and attachment.

Geometry uses a Hadamard corner filter for plates and continuous-time CV for
vehicle components. Bernoulli existence, persistent nonplate marks, and global
assignment alternatives are distinct from geometry. Resource defaults below are
explicit engineering priors, not a claim of fitted sequence calibration. Supply
held-out policy calibration in TrackerCalibration for deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import heapq
import itertools
import math
from typing import Any
import numpy as np

from vpocr_det.optical_tracker import (
    VehicleAppearanceRef,
    extract_vehicle_appearance,
    track_vehicle_patch_ncc,
)
from vpocr_det.quad_cropper import (
    is_valid_plate_quad,
    select_containment_crop_window,
)
from vpocr_det.types import (PlateDetection, VehicleDetection, ObservationActionMeta,
                              SearchRequest, assert_not_observation)


# Exact Hadamard corner-to-mode matrix B (`q = B @ m`, `m = 0.25 * B.T @ q`) for [TL, TR, BR, BL]:
# Mode 0: centroid `c`; Mode 1: horizontal span `u`; Mode 2: vertical span `v`; Mode 3: warp `w`.
_B_HADAMARD: np.ndarray = np.array(
    [
        [1.0, -1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
    ],
    dtype=np.float64,
)
_B_INV_HADAMARD: np.ndarray = 0.25 * _B_HADAMARD.T
_B_INV_KRON_I2: np.ndarray = np.kron(_B_INV_HADAMARD, np.eye(2, dtype=np.float64))  # (8, 8)
_E_POS_2X6: np.ndarray = np.array(
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
    dtype=np.float64,
)
_H_24: np.ndarray = np.kron(_B_HADAMARD, _E_POS_2X6)  # (8, 24)
_I_24: np.ndarray = np.eye(24, dtype=np.float64)
_I_8: np.ndarray = np.eye(8, dtype=np.float64)

# Indices inside 24-D state corresponding to velocities and accelerations across all 4 modes
_VEL_ACC_INDICES: np.ndarray = np.array(
    [k * 6 + d for k in range(4) for d in (2, 3, 4, 5)],
    dtype=np.int64,
)
_POS_INDICES: np.ndarray = np.array(
    [k * 6 + d for k in range(4) for d in (0, 1)],
    dtype=np.int64,
)


def _norm_ppf_tail(p_upper_tail: float) -> float:
    """Rational Acklam inverse standard normal `Phi^{-1}(1 - p_upper_tail)` for `0 < p < 0.5`."""
    p = max(1e-15, min(0.499999, float(p_upper_tail)))
    # Abramowitz-Stegun / Acklam high-accuracy tail rational approximation
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q) + c[5]
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
        return -num / den
    q = (1.0 - p) - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r) + 1.0
    return num / den


def _logsumexp(vals: list[float]) -> float:
    if not vals:
        return -math.inf
    m = max(vals)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _cholesky_spd_8x8(S: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Factor symmetric `8x8` `S` with bounded roundoff jitter; return `(L, logdet)` or `(None, nan)`."""
    S_sym = 0.5 * (S + S.T)
    if not np.all(np.isfinite(S_sym)):
        return None, math.nan
    tr_scale = max(1e-6, float(np.trace(S_sym)) / 8.0)
    for mult in (0.0, 1e-11, 1e-9, 1e-7):
        S_try = S_sym if mult == 0.0 else (S_sym + (mult * tr_scale) * _I_8)
        try:
            L = np.linalg.cholesky(S_try)
            diag_L = np.diag(L)
            if np.any(diag_L <= 0.0):
                continue
            logdet = 2.0 * float(np.sum(np.log(diag_L)))
            return L, logdet
        except np.linalg.LinAlgError:
            continue
    return None, math.nan


def _cho_solve_spd(L: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve `(L @ L.T) X = rhs` via forward/backward triangular solves (zero matrix inverse)."""
    y = np.linalg.solve(L, rhs)
    return np.linalg.solve(L.T, y)


def _build_F24_and_Q24(h: float, ref_plate_w: float) -> tuple[np.ndarray, np.ndarray]:
    """Build exact 24-D transition `F(h)` and continuous-white-jerk `Q_smooth(h)` (`W(h) (x) J_k`)."""
    hf = float(h)
    h2 = hf * hf
    h3 = h2 * hf
    h4 = h3 * hf
    h5 = h4 * hf

    T_3 = np.array(
        [
            [1.0, hf, 0.5 * h2],
            [0.0, 1.0, hf],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    F_6 = np.kron(T_3, np.eye(2, dtype=np.float64))
    F_24 = np.kron(np.eye(4, dtype=np.float64), F_6)

    W_3 = np.array(
        [
            [h5 / 20.0, h4 / 8.0, h3 / 6.0],
            [h4 / 8.0, h3 / 3.0, 0.5 * h2],
            [h3 / 6.0, 0.5 * h2, hf],
        ],
        dtype=np.float64,
    )
    # Spectral white-jerk intensity `J_k` (px^2 / s^5) scaled smoothly by plate width squared:
    # Centroid mode (k=0) carries full translational agility; u/v (k=1,2) carry moderate scale/perspective
    # agility; residual warp w (k=3) is tightly regularized to preserve quad planarity.
    w_sc = max(14.0, float(ref_plate_w))
    j_base = (22.0 * w_sc) ** 2  # ~ (22 plate-widths / s^{2.5})^2
    mode_scales = (
        (1.00, 1.15),   # Mode 0: centroid (x, y — slightly higher vertical heave agility)
        (0.06, 0.04),   # Mode 1: horizontal span & keystone u
        (0.04, 0.06),   # Mode 2: vertical span & keystone v
        (0.008, 0.008), # Mode 3: projective residual warp w
    )
    Q_24 = np.zeros((24, 24), dtype=np.float64)
    for k, (sx_j, sy_j) in enumerate(mode_scales):
        J_k = np.diag([j_base * sx_j, j_base * sy_j])
        Q_24[k * 6 : (k + 1) * 6, k * 6 : (k + 1) * 6] = np.kron(W_3, J_k)
    return F_24, Q_24


def build_observation_cov_8x8(
    det: PlateDetection,
    ref_plate_w: float,
    alpha_blur_phase: float = 0.0,
    sigma_floor_px: float = 0.25,
) -> np.ndarray:
    """Build `8x8` image-space corner observation covariance `R` (`Astra` Section 3).

    `R = blockdiag(Sigma_loc,0..3) + alpha_blur_phase * (1_4 (x) b)(1_4 (x) b)^T + sigma_floor^2 I_8`.
    """
    R = np.zeros((8, 8), dtype=np.float64)
    if det.corner_cov_xy is not None and det.corner_cov_xy.shape == (4, 2, 2) and np.all(np.isfinite(det.corner_cov_xy)):
        for c_i in range(4):
            blk = det.corner_cov_xy[c_i].astype(np.float64)
            blk = 0.5 * (blk + blk.T)
            R[c_i * 2 : c_i * 2 + 2, c_i * 2 : c_i * 2 + 2] = blk
    else:
        # Calibrated isotropic fallback for legacy single-scale `PlateQuadNet` without Cholesky head
        qual = max(0.15, min(1.0, float(det.quad_confidence)))
        sig_fb = max(0.45, 0.018 * max(16.0, float(ref_plate_w)) / math.sqrt(qual))
        R += (sig_fb * sig_fb) * _I_8

    if alpha_blur_phase > 0.0 and det.blur_vec_xy is not None and np.all(np.isfinite(det.blur_vec_xy)):
        b_xy = np.asarray(det.blur_vec_xy, dtype=np.float64).reshape(2)
        b_all = np.tile(b_xy, 4)  # (1_4 (x) b) in R^8
        R += float(alpha_blur_phase) * np.outer(b_all, b_all)

    floor_var = max(1e-4, float(sigma_floor_px) ** 2)
    R += floor_var * _I_8
    return 0.5 * (R + R.T)


@dataclass(frozen=True)
class PredictedROIProposal:
    """Kalman-predicted next-frame concentric ROI proposal with union-bound expansion `gamma`."""

    track_id: int
    predicted_corners_xy: np.ndarray              # (4, 2) float32 predicted mid-exposure quad
    predicted_bbox_xyxy: tuple[float, float, float, float]
    corner_std_xy: np.ndarray                     # (4, 2) float32 marginal corner std (px)
    gamma: float                                  # >= 1.0 concentric ROI expansion factor
    needs_detector_reacquire: bool                # True when uncertainty exceeds local ROI acuity budget
    is_dormant: bool = False


class _KalmanPlateTrack:
    """Single-plate 24-D Hadamard-Mode Kalman state + shock detector + unnormalized Re-ID accumulator."""

    def __init__(
        self,
        track_id: int,
        det: PlateDetection,
        timestamp_s: float,
        alpha_blur_phase: float = 0.0,
        sigma_floor_px: float = 0.25,
        is_tentative_relink_candidate: bool = False,
        init_vel_xy: tuple[float, float] | None = None,
    ) -> None:
        self.track_id: int = int(track_id)
        q0 = np.asarray(det.corners_xy, dtype=np.float64).reshape(4, 2)
        m0 = _B_INV_HADAMARD @ q0  # (4, 2) = [c, u, v, w]

        self.x: np.ndarray = np.zeros((24,), dtype=np.float64)
        for k in range(4):
            self.x[k * 6 + 0] = m0[k, 0]
            self.x[k * 6 + 1] = m0[k, 1]
        if init_vel_xy is not None and all(math.isfinite(v) for v in init_vel_xy):
            self.x[2] = float(init_vel_xy[0])
            self.x[3] = float(init_vel_xy[1])

        w0, h0 = self._quad_span_wh(q0)
        self.ref_w: float = max(6.0, w0)
        self.ref_h: float = max(3.0, h0)
        self.last_obs_bbox_xyxy: tuple[float, float, float, float] = (
            float(q0[:, 0].min()),
            float(q0[:, 1].min()),
            float(q0[:, 0].max()),
            float(q0[:, 1].max()),
        )

        # Initialize 24x24 covariance P from initial observation covariance + broad velocity/accel prior
        R0 = build_observation_cov_8x8(det, self.ref_w, alpha_blur_phase=alpha_blur_phase, sigma_floor_px=sigma_floor_px)
        P_modes_pos = _B_INV_KRON_I2 @ R0 @ _B_INV_KRON_I2.T  # (8, 8) exact mode-position covariance!
        self.P: np.ndarray = np.zeros((24, 24), dtype=np.float64)
        for k in range(4):
            for l in range(4):
                self.P[k * 6 : k * 6 + 2, l * 6 : l * 6 + 2] = P_modes_pos[k * 2 : k * 2 + 2, l * 2 : l * 2 + 2]

        vel_stds = (8.5 * self.ref_w, 2.2 * self.ref_w, 2.2 * self.ref_w, 0.55 * self.ref_w)
        acc_stds = (18.0 * self.ref_w, 4.5 * self.ref_w, 4.5 * self.ref_w, 1.2 * self.ref_w)
        for k in range(4):
            sv2 = vel_stds[k] ** 2
            sa2 = acc_stds[k] ** 2
            self.P[k * 6 + 2, k * 6 + 2] = sv2
            self.P[k * 6 + 3, k * 6 + 3] = sv2
            self.P[k * 6 + 4, k * 6 + 4] = sa2
            self.P[k * 6 + 5, k * 6 + 5] = sa2
        self.P = 0.5 * (self.P + self.P.T)

        self.last_state_ts: float = float(timestamp_s)
        self.first_obs_ts: float = float(timestamp_s)
        self.last_obs_ts: float = float(timestamp_s)
        self.last_valid_modes_uvw: np.ndarray = m0[1:4].copy()  # (3, 2) fallback shape modes
        self.last_blur_vec_xy: np.ndarray = (
            np.asarray(det.blur_vec_xy, dtype=np.float64).reshape(2).copy()
            if (det.blur_vec_xy is not None and np.all(np.isfinite(det.blur_vec_xy)))
            else np.zeros((2,), dtype=np.float64)
        )

        self.score_ema: float = float(det.score)
        self.plate_side: int = int(det.plate_side)
        self.side_votes: np.ndarray = np.zeros((3,), dtype=np.float64)
        if 0 <= int(det.plate_side) < 3:
            self.side_votes[int(det.plate_side)] += float(max(0.2, det.quad_confidence))
        self.vehicle_bbox_xyxy: tuple[float, float, float, float] | None = det.vehicle_bbox_xyxy
        self.parent_vehicle_track_id: int | None = det.parent_vehicle_track_id
        self.is_large_ema: float = float(det.is_large_prob)

        self.hits: int = 1
        self.age: int = 1
        self.missed: int = 0
        self.consecutive_shocks: int = 0
        self.shock_prob: float = 0.02
        self.last_shock_triggered: bool = False
        self.last_nis_par: float = 0.0
        self.last_nis_perp: float = 0.0

        # Unnormalized EMA resultant for local Re-ID (`Astra` Section 4: frozen during coast/dormancy/tentative!)
        self.is_tentative_relink: bool = bool(is_tentative_relink_candidate)
        self.tentative_buffer: list[tuple[float, PlateDetection]] = []
        if self.is_tentative_relink:
            self.tentative_buffer.append((float(timestamp_s), det))

        self.reid_accum: np.ndarray | None = None
        if (
            not self.is_tentative_relink
            and det.reid_emb is not None
            and not det.quad_edge_touch
            and det.quad_confidence >= 0.30
        ):
            e0 = np.asarray(det.reid_emb, dtype=np.float64).reshape(-1)
            nrm = float(np.linalg.norm(e0))
            if nrm > 1e-6 and np.all(np.isfinite(e0)):
                self.reid_accum = (e0 / nrm).copy()

    @staticmethod
    def _quad_span_wh(q: np.ndarray) -> tuple[float, float]:
        w_top = float(np.linalg.norm(q[1] - q[0]))
        w_bot = float(np.linalg.norm(q[2] - q[3]))
        h_left = float(np.linalg.norm(q[3] - q[0]))
        h_right = float(np.linalg.norm(q[2] - q[1]))
        return max(4.0, 0.5 * (w_top + w_bot)), max(2.5, 0.5 * (h_left + h_right))

    @property
    def corners_xy(self) -> np.ndarray:
        q = (_H_24 @ self.x).reshape(4, 2)
        q32 = q.astype(np.float32)
        if is_valid_plate_quad(q32):
            return q32
        # Never repair by sorting corners (preserves semantic [TL, TR, BR, BL]); fall back to last valid u,v,w shape!
        m_safe = np.vstack([self.x[0:2].reshape(1, 2), self.last_valid_modes_uvw])
        return (_B_HADAMARD @ m_safe).astype(np.float32)

    @property
    def cx(self) -> float:
        return float(self.x[0])

    @property
    def cy(self) -> float:
        return float(self.x[1])

    @property
    def vx(self) -> float:
        return float(self.x[2])

    @property
    def vy(self) -> float:
        return float(self.x[3])

    @property
    def w(self) -> float:
        q = self.corners_xy
        return max(4.0, float(q[:, 0].max() - q[:, 0].min()))

    @property
    def h(self) -> float:
        q = self.corners_xy
        return max(2.5, float(q[:, 1].max() - q[:, 1].min()))

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        q = self.corners_xy
        return (
            float(q[:, 0].min()),
            float(q[:, 1].min()),
            float(q[:, 0].max()),
            float(q[:, 1].max()),
        )

    @property
    def reid_unit(self) -> np.ndarray | None:
        if self.reid_accum is None:
            return None
        nrm = float(np.linalg.norm(self.reid_accum))
        if nrm < 0.15 or not math.isfinite(nrm):
            return None
        return self.reid_accum / nrm

    def peek_predict(self, target_ts: float) -> tuple[np.ndarray, np.ndarray]:
        """Non-destructively propagate `(x, P)` to `target_ts` (`h <= 0` returns current `(x, P)`)."""
        h_dt = float(target_ts) - self.last_state_ts
        if h_dt <= 1e-9:
            return self.x.copy(), self.P.copy()
        F_24, Q_24 = _build_F24_and_Q24(h_dt, self.ref_w)
        x_pred = F_24 @ self.x
        P_pred = F_24 @ self.P @ F_24.T + Q_24
        P_pred = 0.5 * (P_pred + P_pred.T)
        return x_pred, P_pred

    def predict_to(self, timestamp_s: float, count_miss: bool = True) -> bool:
        """Advance state to `timestamp_s`. Returns `False` if `h < -1e-9` (stale timestamp)."""
        ts = float(timestamp_s)
        h_dt = ts - self.last_state_ts
        if h_dt < -1e-9:
            return False
        if h_dt > 1e-9:
            F_24, Q_24 = _build_F24_and_Q24(h_dt, self.ref_w)
            self.x = F_24 @ self.x
            self.P = F_24 @ self.P @ F_24.T + Q_24
            self.P = 0.5 * (self.P + self.P.T)
            # Mild physical acceleration decay during prolonged coasting so unobserved acceleration cannot blow up
            if self.missed >= 1:
                acc_decay = math.exp(-3.5 * h_dt)
                for k in range(4):
                    self.x[k * 6 + 4] *= acc_decay
                    self.x[k * 6 + 5] *= acc_decay
            self.last_state_ts = ts
        if count_miss:
            self.age += 1
            self.missed += 1
        return True

    def corner_covariance_8x8(self, P_mat: np.ndarray | None = None) -> np.ndarray:
        P_use = self.P if P_mat is None else P_mat
        C = _H_24 @ P_use @ _H_24.T
        return 0.5 * (C + C.T)

    def update(
        self,
        det: PlateDetection,
        timestamp_s: float,
        alpha_blur_phase: float = 0.0,
        sigma_floor_px: float = 0.25,
    ) -> bool:
        """Assimilate `det` at `timestamp_s` via 8x8 Cholesky Joseph update + subspace heave shock handling."""
        ts = float(timestamp_s)
        # Strict duplicate-timestamp & stale-timestamp guard (`Astra` Section 1):
        if abs(ts - self.last_obs_ts) < 1e-9 or ts < self.last_state_ts - 1e-9:
            return False

        q_obs = np.asarray(det.corners_xy, dtype=np.float64).reshape(4, 2)
        if not is_valid_plate_quad(q_obs.astype(np.float32)):
            return False
        z = q_obs.reshape(8)

        x_minus = self.x
        P_minus = self.P
        R = build_observation_cov_8x8(det, self.ref_w, alpha_blur_phase=alpha_blur_phase, sigma_floor_px=sigma_floor_px)

        z_pred = _H_24 @ x_minus
        r = z - z_pred
        U0 = P_minus @ _H_24.T
        S0 = _H_24 @ U0 + R
        L0, logdet_S0 = _cholesky_spd_8x8(S0)
        if L0 is None:
            return False

        S0_inv_r = _cho_solve_spd(L0, r)
        N0 = float(r @ S0_inv_r)

        # --- Subspace Coherent Heave/Scale NIS Shock Detector (`Astra` Section 2) ---
        q_pred = z_pred.reshape(4, 2)
        c_pred = q_pred.mean(axis=0)
        s_off = q_pred - c_pred  # (4, 2)
        D_full = np.zeros((8, 2), dtype=np.float64)
        for i_c in range(4):
            D_full[i_c * 2 + 0, 1] = s_off[i_c, 0]
            D_full[i_c * 2 + 1, 0] = 1.0
            D_full[i_c * 2 + 1, 1] = s_off[i_c, 1]

        scale_norm = float(np.linalg.norm(s_off))
        use_2d = scale_norm >= 3.0
        D_sub = D_full if use_2d else D_full[:, 0:1]
        S0_inv_D = _cho_solve_spd(L0, D_sub)
        G = D_sub.T @ S0_inv_D
        G = 0.5 * (G + G.T)
        g_vec = D_sub.T @ S0_inv_r

        d_rank = D_sub.shape[1]
        if d_rank == 2:
            det_G = float(G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0])
            if det_G <= 1e-9 * max(1.0, float(G[0, 0] * G[1, 1])):
                d_rank = 1
                D_sub = D_full[:, 0:1]
                G = D_sub.T @ _cho_solve_spd(L0, D_sub)
                g_vec = D_sub.T @ S0_inv_r

        if d_rank == 2:
            G_inv_g = np.linalg.solve(G, g_vec)
            N_par = max(0.0, float(g_vec @ G_inv_g))
            chi2_trig = 13.8155  # chi^2_{2, 0.999}
            chi2_shape = 18.5476  # chi^2_{6, 0.995}
        else:
            N_par = max(0.0, float((g_vec[0] * g_vec[0]) / max(1e-12, float(G[0, 0]))))
            chi2_trig = 10.8276  # chi^2_{1, 0.999}
            chi2_shape = 20.2777  # chi^2_{7, 0.995}
        N_perp = max(0.0, N0 - N_par)
        self.last_nis_par = N_par
        self.last_nis_perp = N_perp

        topo_ok = float(getattr(det, "topology_prob", 1.0)) >= 0.40
        shock_cand = (
            self.hits >= 2
            and N_par > chi2_trig
            and N_perp <= chi2_shape
            and topo_ok
        )

        # Standard smooth posterior `(x0_plus, P0_plus)`
        K0 = _cho_solve_spd(L0, U0.T).T  # (24, 8)
        x0_plus = x_minus + K0 @ r
        IKH0 = _I_24 - K0 @ _H_24
        P0_plus = IKH0 @ P_minus @ IKH0.T + K0 @ R @ K0.T
        P0_plus = 0.5 * (P0_plus + P0_plus.T)

        used_shock = False
        if shock_cand:
            # Construct physically bounded heave/scale jump + velocity impulse covariance `Q_shock`
            h_dt_eff = max(1.0 / 60.0, min(0.5, ts - self.last_obs_ts))
            V_modes = _B_INV_KRON_I2 @ D_full  # (8, 2)
            sig_heave_p = max(6.0, 1.35 * self.ref_h)
            sig_scale_p = 0.22
            Omega_p = np.diag([sig_heave_p * sig_heave_p, sig_scale_p * sig_scale_p])

            sig_heave_v = max(18.0, 4.0 * self.ref_h)
            sig_scale_v = 0.75
            Omega_v = np.diag([sig_heave_v * sig_heave_v, sig_scale_v * sig_scale_v])

            M_jump = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
            M_imp = np.array(
                [
                    [(h_dt_eff ** 2) / 3.0, 0.5 * h_dt_eff, 0.0],
                    [0.5 * h_dt_eff, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
            Q_shock = np.zeros((24, 24), dtype=np.float64)
            for k in range(4):
                V_k = V_modes[k * 2 : k * 2 + 2, :]
                for l in range(4):
                    V_l = V_modes[l * 2 : l * 2 + 2, :]
                    Q_kl = np.kron(M_jump, V_k @ Omega_p @ V_l.T) + np.kron(M_imp, V_k @ Omega_v @ V_l.T)
                    Q_shock[k * 6 : (k + 1) * 6, l * 6 : (l + 1) * 6] = Q_kl

            P1_minus = 0.5 * ((P_minus + Q_shock) + (P_minus + Q_shock).T)
            U1 = P1_minus @ _H_24.T
            S1 = _H_24 @ U1 + R
            L1, logdet_S1 = _cholesky_spd_8x8(S1)
            if L1 is not None:
                S1_inv_r = _cho_solve_spd(L1, r)
                N1 = float(r @ S1_inv_r)
                # Innovation log-Bayes factor including determinant penalty (`Astra` Eq. 214)
                prior_logit = math.log(max(1e-4, self.shock_prob) / max(1e-4, 1.0 - self.shock_prob))
                post_logit = prior_logit + 0.5 * ((N0 - N1) + (logdet_S0 - logdet_S1))
                pi_plus = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, post_logit))))

                if pi_plus > 0.5:
                    K1 = _cho_solve_spd(L1, U1.T).T
                    if self.consecutive_shocks == 0:
                        # STRICT DERIVATIVE PROTECTION (`Astra` Eq. 236):
                        # Unconfirmed 1st shock updates ONLY positions/modes; velocity & acceleration rows
                        # of `K_tilde` are identically ZERO in both mean and Joseph covariance update!
                        K_tilde = K1.copy()
                        K_tilde[_VEL_ACC_INDICES, :] = 0.0
                        x_new = x_minus + K_tilde @ r
                        IKH_t = _I_24 - K_tilde @ _H_24
                        P_new = IKH_t @ P1_minus @ IKH_t.T + K_tilde @ R @ K_tilde.T
                        P_new = 0.5 * (P_new + P_new.T)
                    else:
                        # Temporally confirmed coherent shock -> exact 2-Gaussian moment-matched posterior!
                        x1_plus = x_minus + K1 @ r
                        IKH1 = _I_24 - K1 @ _H_24
                        P1_plus = IKH1 @ P1_minus @ IKH1.T + K1 @ R @ K1.T
                        P1_plus = 0.5 * (P1_plus + P1_plus.T)
                        w0_m = 1.0 - pi_plus
                        w1_m = pi_plus
                        x_new = w0_m * x0_plus + w1_m * x1_plus
                        d0 = (x0_plus - x_new).reshape(24, 1)
                        d1 = (x1_plus - x_new).reshape(24, 1)
                        P_new = w0_m * (P0_plus + d0 @ d0.T) + w1_m * (P1_plus + d1 @ d1.T)
                        P_new = 0.5 * (P_new + P_new.T)

                    q_chk = (_H_24 @ x_new).reshape(4, 2).astype(np.float32)
                    if is_valid_plate_quad(q_chk):
                        self.x = x_new
                        self.P = P_new
                        self.consecutive_shocks += 1
                        self.shock_prob = min(0.65, max(0.08, pi_plus * 0.5))
                        used_shock = True

        if not used_shock:
            # On first-to-second hit initialization, seed constant velocity directly from exact timestamp delta
            # so newly born fast-moving tracks lock velocity immediately without waiting 4 frames!
            if self.hits == 1:
                dt_init = ts - self.last_obs_ts
                if 1e-4 < dt_init <= 1.0:
                    m_obs = _B_INV_HADAMARD @ q_obs
                    m_prev = np.vstack([self.x[0:2], self.last_valid_modes_uvw])
                    v_init = (m_obs - m_prev) / dt_init
                    max_v = 35.0 * self.ref_w
                    for k in range(4):
                        vx_k = float(np.clip(v_init[k, 0], -max_v, max_v))
                        vy_k = float(np.clip(v_init[k, 1], -max_v, max_v))
                        x0_plus[k * 6 + 2] = 0.75 * vx_k + 0.25 * x0_plus[k * 6 + 2]
                        x0_plus[k * 6 + 3] = 0.75 * vy_k + 0.25 * x0_plus[k * 6 + 3]

            q_post = (_H_24 @ x0_plus).reshape(4, 2).astype(np.float32)
            if not is_valid_plate_quad(q_post):
                return False
            self.x = x0_plus
            self.P = P0_plus
            self.consecutive_shocks = 0
            self.shock_prob = max(0.02, self.shock_prob * 0.5)

        self.last_shock_triggered = used_shock
        q_committed = (_H_24 @ self.x).reshape(4, 2)
        m_committed = _B_INV_HADAMARD @ q_committed
        self.last_valid_modes_uvw = m_committed[1:4].copy()
        nw, nh = self._quad_span_wh(q_committed)
        self.ref_w, self.ref_h = nw, nh
        self.last_obs_bbox_xyxy = (
            float(q_committed[:, 0].min()),
            float(q_committed[:, 1].min()),
            float(q_committed[:, 0].max()),
            float(q_committed[:, 1].max()),
        )

        if det.blur_vec_xy is not None and np.all(np.isfinite(det.blur_vec_xy)):
            self.last_blur_vec_xy = np.asarray(det.blur_vec_xy, dtype=np.float64).reshape(2).copy()
        else:
            self.last_blur_vec_xy *= 0.5

        self.score_ema = 0.55 * float(det.score) + 0.45 * self.score_ema
        if 0 <= int(det.plate_side) < 3:
            self.side_votes[int(det.plate_side)] += float(max(0.2, det.quad_confidence))
            self.plate_side = int(np.argmax(self.side_votes))
        if det.vehicle_bbox_xyxy is not None:
            self.vehicle_bbox_xyxy = det.vehicle_bbox_xyxy
        if det.parent_vehicle_track_id is not None:
            self.parent_vehicle_track_id = int(det.parent_vehicle_track_id)
        self.is_large_ema = 0.40 * float(det.is_large_prob) + 0.60 * self.is_large_ema

        self.last_obs_ts = ts
        self.last_state_ts = ts
        self.hits += 1
        self.missed = 0

        if self.is_tentative_relink:
            self.tentative_buffer.append((ts, det))
        elif det.reid_emb is not None and not det.quad_edge_touch and det.quad_confidence >= 0.30:
            e_t = np.asarray(det.reid_emb, dtype=np.float64).reshape(-1)
            nrm = float(np.linalg.norm(e_t))
            if nrm > 1e-6 and np.all(np.isfinite(e_t)):
                e_u = e_t / nrm
                if self.reid_accum is None:
                    self.reid_accum = e_u.copy()
                else:
                    self.reid_accum = 0.75 * self.reid_accum + 0.25 * e_u
        return True

    def build_roi_proposal(self, next_timestamp_s: float, dt_jitter_s: float = 0.005,
                           eps_containment: float = 1e-3, max_gamma: float = 4.0,
                           reacquire_gamma: float = 3.2,
                           is_dormant: bool = False) -> PredictedROIProposal:
        # Legacy adapter: gamma is descriptive, never a clipped observation repair.
        x, P = self.peek_predict(max(self.last_state_ts, next_timestamp_s + dt_jitter_s))
        q = (_H_24 @ x).reshape(4, 2)
        C = _H_24 @ P @ _H_24.T
        cov = np.stack([C[2*k:2*k+2, 2*k:2*k+2] for k in range(4)])
        cov += np.eye(2)[None] * 1e-9
        plan = select_containment_crop_window(q, cov, (1, 1), eps_containment)
        bounds = plan.required_window_xyxy
        width = max(float(np.ptp(q[:, 0])), 1e-9)
        height = max(float(np.ptp(q[:, 1])), 1e-9)
        gamma = max(1.0, (bounds[2]-bounds[0])/width, (bounds[3]-bounds[1])/height)
        return PredictedROIProposal(self.track_id, q.astype(np.float32), bounds,
                                   np.sqrt(np.diagonal(cov, axis1=1, axis2=2)),
                                   gamma, gamma > reacquire_gamma, is_dormant)


@dataclass(frozen=True)
class TrackerCalibration:
    """Conditional likelihood/cost resource. Defaults require sequence validation.

    Densities use pixel coordinates (8-D corners / 4-D vehicle box state).
    Policy odds must be fitted for the executed sampling policy, particularly
    targeted searches. No detector/layout/topology marginals are multiplied.
    """
    detection_probability: float = 0.9
    survival_tau_s: float = 120.0
    birth_odds: float = 0.05
    persistent_birth_ratio: float = 0.25
    plate_clutter_density: float = 1e-20
    vehicle_clutter_density: float = 1e-12
    appearance_kappa: float = 12.0
    innovation_tau_s: float = 0.5
    c_false: float = 19.0
    c_wait: float = 1.0
    c_retire: float = 0.01
    c_retain: float = 1.0
    dormant_after_s: float = 0.3
    acceleration_spectral_density: float = 100.0
    max_hypotheses: int = 32
    assignment_node_budget: int = 20000
    policy_birth_odds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.detection_probability < 1:
            raise ValueError('detection_probability must be in [0,1)')
        if min(self.survival_tau_s, self.birth_odds, self.persistent_birth_ratio,
               self.plate_clutter_density, self.vehicle_clutter_density,
               self.c_false, self.c_wait, self.c_retire, self.c_retain,
               self.innovation_tau_s) <= 0:
            raise ValueError('Time constants, densities and costs must be positive')
        if self.max_hypotheses < 1 or self.assignment_node_budget < 1:
            raise ValueError('Assignment budgets must be positive')


def _prob(p: float) -> float:
    return min(1.0-1e-12, max(1e-12, float(p)))


def _sigmoid(x: float) -> float:
    return _prob(math.exp(-float(np.logaddexp(0.0, -x))))


def _logit(p: float) -> float:
    p = _prob(p)
    return math.log(p)-math.log1p(-p)


def missed_existence_probability(r_pred: float, p_D: float) -> float:
    """Exact Bernoulli null update; an unexecuted action is the identity."""
    if p_D == 0:
        return r_pred
    return _prob(r_pred * (1-p_D) / (1-r_pred*p_D))


def detection_availability(bbox_xyxy: tuple[float, float, float, float],
                           action: ObservationActionMeta, p_detect: float = 0.9) -> float:
    if action.detector_ran_full:
        return p_detect
    # Conservative sufficient visibility: all target support must have been searched.
    x0, y0, x1, y1 = bbox_xyxy
    return p_detect if any(a <= x0 and b <= y0 and c >= x1 and d >= y1
                           for a, b, c, d in action.searched_rois_xyxy) else 0.0


@dataclass(frozen=True)
class CameraMotionEstimate:
    affine_2x3: np.ndarray
    parameter_cov_6x6: np.ndarray
    residual_cov_2x2: np.ndarray
    n_correspondences: int = 0

    @property
    def g_t(self) -> np.ndarray:
        return self.affine_2x3

    @property
    def Sigma_g(self) -> np.ndarray:
        return self.parameter_cov_6x6


class BackgroundCameraMotionEstimator:
    """Sparse background KLT with forward/backward checks and robust affine IRLS.

    Residual scatter is retained separately from parameter uncertainty: parallax
    cannot be made artificially certain by increasing correspondence count.
    """
    def __init__(self, max_corners: int = 240, residual_floor_px: float = 0.25,
                 unavailable_std_px: float = 8.0) -> None:
        self.max_corners = max_corners
        self.residual_floor_px = residual_floor_px
        self.unavailable_std_px = unavailable_std_px
        self._previous: np.ndarray | None = None
        self._previous_boxes: list[tuple[float, float, float, float]] = []

    def reset(self) -> None:
        self._previous = None
        self._previous_boxes = []

    def _unavailable(self) -> CameraMotionEstimate:
        cov = np.zeros((6, 6))
        cov[2, 2] = cov[5, 5] = self.unavailable_std_px**2
        return CameraMotionEstimate(np.eye(2, 3), cov,
                                    np.eye(2)*self.unavailable_std_px**2)

    def fit_correspondences(self, source_xy: np.ndarray, target_xy: np.ndarray) -> CameraMotionEstimate:
        src, dst = np.asarray(source_xy, float), np.asarray(target_xy, float)
        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
            raise ValueError('Expected matching (N,2) background correspondences')
        ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
        src, dst = src[ok], dst[ok]
        X = np.column_stack([src, np.ones(len(src))])
        if len(src) < 6 or np.linalg.matrix_rank(X) < 3:
            return self._unavailable()
        weights = np.ones(len(src))
        for _ in range(8):
            root = np.sqrt(weights)
            beta = np.linalg.lstsq(X*root[:, None], dst*root[:, None], rcond=None)[0]
            residual = dst-X@beta
            radius = np.linalg.norm(residual, axis=1)
            scale = max(self.residual_floor_px, float(np.median(radius))/1.1774)
            weights = np.minimum(1.0, 1.345*scale/np.maximum(radius, 1e-12))
        residual_cov = (residual.T*weights) @ residual/max(1.0, weights.sum()-3)
        residual_cov += np.eye(2)*self.residual_floor_px**2
        information_inv = np.linalg.pinv((X.T*weights) @ X)
        param_cov = np.kron(residual_cov, information_inv)
        return CameraMotionEstimate(beta.T, param_cov, residual_cov, len(src))

    def estimate(self, frame_rgb: np.ndarray,
                 active_vehicle_boxes: list[tuple[float, float, float, float]] = (),
                 active_vehicle_masks: list[np.ndarray] = ()) -> CameraMotionEstimate:
        import cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY) if frame_rgb.ndim == 3 else frame_rgb.copy()
        previous, previous_boxes = self._previous, self._previous_boxes
        self._previous = gray.copy()
        self._previous_boxes = list(active_vehicle_boxes)
        if previous is None or previous.shape != gray.shape:
            return self._unavailable()
        mask = np.full(gray.shape, 255, np.uint8)
        for box in [*previous_boxes, *active_vehicle_boxes]:
            a, b, c, d = np.asarray(box, int)
            mask[max(0,b):max(0,d), max(0,a):max(0,c)] = 0
        for silhouette in active_vehicle_masks:
            if silhouette.shape == gray.shape:
                mask[silhouette > 0] = 0
        p = cv2.goodFeaturesToTrack(previous, self.max_corners, 0.01, 8, mask=mask)
        if p is None or len(p) < 6:
            return self._unavailable()
        q, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, p, None)
        if q is None:
            return self._unavailable()
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, previous, q, None)
        if back is None:
            return self._unavailable()
        good = status.ravel().astype(bool) & status_back.ravel().astype(bool)
        good &= np.linalg.norm((back-p).reshape(-1,2), axis=1) < 1.5
        # Exclude current foreground endpoints too, including objects entering the image.
        for a,b,c,d in active_vehicle_boxes:
            xy = q.reshape(-1,2)
            good &= ~((xy[:,0]>=a)&(xy[:,0]<=c)&(xy[:,1]>=b)&(xy[:,1]<=d))
        return self.fit_correspondences(p.reshape(-1,2)[good], q.reshape(-1,2)[good])

    update = estimate


def _warp_points(points: np.ndarray, motion: CameraMotionEstimate) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, float)
    X = np.column_stack([points, np.ones(len(points))])
    J = np.zeros((len(points)*2, 6))
    J[0::2, :3] = X
    J[1::2, 3:] = X
    cov = J @ motion.parameter_cov_6x6 @ J.T
    cov += np.kron(np.ones((len(points),len(points))), motion.residual_cov_2x2)
    return X @ motion.affine_2x3.T, cov


def _compensate_plate(t: _KalmanPlateTrack, motion: CameraMotionEstimate) -> None:
    A, b = motion.affine_2x3[:,:2], motion.affine_2x3[:,2]
    old = t.x.copy()
    Jx = np.kron(np.eye(12), A)
    Jg = np.zeros((24,6))
    for k in range(0,24,2):
        Jg[k,:2] = old[k:k+2]
        Jg[k+1,3:5] = old[k:k+2]
    Jg[0,2] = Jg[1,5] = 1.0
    Jq = np.zeros((24,8))
    Jq[_POS_INDICES,:] = _B_INV_KRON_I2
    residual = np.kron(np.ones((4,4)),motion.residual_cov_2x2)
    t.x = Jx@old
    t.x[:2] += b
    t.P = Jx@t.P@Jx.T+Jg@motion.parameter_cov_6x6@Jg.T+Jq@residual@Jq.T
    t.P = (t.P+t.P.T)*.5
    t.last_valid_modes_uvw = np.stack([t.x[6:8],t.x[12:14],t.x[18:20]])
    t.ref_w,t.ref_h = t._quad_span_wh((_H_24@t.x).reshape(4,2))


def _gaussian_logpdf(residual: np.ndarray, covariance: np.ndarray) -> float:
    try:
        L = np.linalg.cholesky((covariance+covariance.T)*0.5)
    except np.linalg.LinAlgError:
        return -math.inf
    u = np.linalg.solve(L, residual)
    return -0.5*(len(u)*math.log(2*math.pi)+2*np.log(np.diag(L)).sum()+u@u)


def _unit(embedding: np.ndarray | None) -> np.ndarray | None:
    if embedding is None:
        return None
    e = np.asarray(embedding, float).reshape(-1)
    norm = np.linalg.norm(e)
    return e/norm if len(e) and np.isfinite(e).all() and norm > 1e-12 else None


def _vmf_log_ratio(a: np.ndarray | None, b: np.ndarray | None, kappa: float) -> float:
    """Normalized vMF innovation density relative to the uniform sphere.

    Uses the exact convergent hypergeometric series for the normalization
    E_uniform exp(kappa*cos(theta)), avoiding a hand-set cosine/nats slope.
    """
    a, b = _unit(a), _unit(b)
    if a is None or b is None or a.shape != b.shape or kappa <= 0:
        return 0.0
    term, terms = 1.0, [1.0]
    for k in range(1, 1000):
        term *= (kappa*kappa/4)/(k*(len(a)/2+k-1))
        terms.append(term)
        if term < sum(terms)*1e-14:
            break
    return float(kappa*np.clip(a@b, -1, 1)-math.log(sum(terms)))


@dataclass(frozen=True)
class AssignmentHypothesis:
    pairs: tuple[tuple[int, int], ...]
    log_weight: float
    probability: float


@dataclass(frozen=True)
class AssignmentPosterior:
    hypotheses: tuple[AssignmentHypothesis, ...]
    discarded_mass_upper_bound: float


def feasible_assignment_posterior(pair_log_weights: np.ndarray, missed_log_weights: np.ndarray,
                                  birth_clutter_log_weights: np.ndarray,
                                  max_hypotheses: int = 32,
                                  node_budget: int = 20000) -> AssignmentPosterior:
    """Best-first complete one-to-one assignments, including all nulls/births.

    An admissible suffix bound permits bounded search. Dropped subtrees carry an
    upper bound on *total* probability mass, so truncation cannot manufacture a
    confident merge. Returned probabilities have that conservative denominator.
    """
    pair = np.asarray(pair_log_weights, float)
    n, m = pair.shape
    birth = np.asarray(birth_clutter_log_weights, float)
    missed = np.asarray(missed_log_weights, float)
    gain = pair-birth[None,:]
    maxima = np.maximum(missed, np.max(gain, axis=1) if m else -math.inf)
    suffix = np.r_[np.cumsum(maxima[::-1])[::-1], 0.0]
    base = float(birth.sum())
    counter = itertools.count()
    queue = [(-(base+suffix[0]), next(counter), 0, frozenset(), (), base)]
    complete: list[tuple[float, tuple[tuple[int,int],...]]] = []
    nodes = 0
    while queue and len(complete) < max_hypotheses and nodes < node_budget:
        _, _, row, used, pairs, score = heapq.heappop(queue)
        nodes += 1
        if row == n:
            complete.append((score, pairs))
            continue
        choices = [(None, float(missed[row]))]
        choices += [(j, float(gain[row,j])) for j in range(m) if j not in used and math.isfinite(gain[row,j])]
        for j, weight in choices:
            score_next = score+weight
            new_used = used if j is None else used | {j}
            new_pairs = pairs if j is None else pairs+((row,j),)
            heapq.heappush(queue, (-(score_next+suffix[row+1]), next(counter),
                                   row+1, new_used, new_pairs, score_next))
    retained = _logsumexp([v for v,_ in complete])
    def _subtree_log_mass_upper_bound(start_row: int, used_cols: frozenset[int],
                                      prefix_score: float) -> float:
        total = float(prefix_score)
        for k in range(start_row, n):
            terms = [float(missed[k])]
            terms.extend(float(gain[k, j]) for j in range(m)
                         if j not in used_cols and math.isfinite(gain[k, j]))
            total += _logsumexp(terms)
        return total

    # Queue subtrees are disjoint. Their completions are a subset of the
    # unconstrained product of row partitions over columns not already used.
    dropped = _logsumexp([_subtree_log_mass_upper_bound(item[2], item[3], item[5])
                          for item in queue])
    denominator = float(np.logaddexp(retained, dropped))
    mass = math.exp(dropped-denominator) if math.isfinite(dropped) else 0.0
    hypotheses = tuple(AssignmentHypothesis(p, w, math.exp(w-denominator)) for w,p in complete)
    return AssignmentPosterior(hypotheses, mass)


@dataclass
class AttachmentState:
    probability: float
    offset_xy: np.ndarray
    covariance_xy: np.ndarray


class _ExistenceState:
    """Bernoulli target existence competing with a shared persistent-null mark."""
    def __init__(self, observation: PlateDetection | VehicleDetection,
                 calibration: TrackerCalibration, timestamp_s: float) -> None:
        self.calibration = calibration
        llr = self._mark(observation)
        odds = calibration.policy_birth_odds.get(str(observation.provenance), calibration.birth_odds)
        # e/(e+kappa), with a persistent-clutter intensity sharing the mark.
        log_e = math.log(odds)+llr
        log_kappa = float(np.logaddexp(0.0, math.log(calibration.persistent_birth_ratio)+log_e))
        self.r = _sigmoid(log_e-log_kappa)
        self.class_log_odds = llr if isinstance(observation, PlateDetection) else 30.0
        self.persistent_nonplate_probability = 1-_sigmoid(self.class_log_odds)
        self.r = min(self.r, _sigmoid(self.class_log_odds))
        self.confirmed = False
        self.last_mark_ts = timestamp_s
        self.last_frame_key = getattr(observation, 'frame_key', None)
        self.last_embedding = _unit(self._embedding(observation))
        self.last_mark = llr
        self.last_observation = observation
        self.attachments: dict[int | None, AttachmentState] = {
            None: AttachmentState(1.0, np.zeros(2), np.eye(2))}
        self.evidence_keys: set[str] = set()
        if self.last_frame_key is not None:
            self.evidence_keys.add(self.last_frame_key)
        self.publish_if_cost_effective()

    @property
    def existence_probability(self) -> float:
        return self.r

    @staticmethod
    def _embedding(observation: PlateDetection | VehicleDetection) -> np.ndarray | None:
        return observation.reid_emb if isinstance(observation, PlateDetection) else observation.appearance_emb

    @staticmethod
    def _mark(observation: PlateDetection | VehicleDetection) -> float:
        value = observation.plate_null_llr if isinstance(observation, PlateDetection) else observation.veh_log_odds
        # Legacy score is a single prior fallback; no product of verifier marginals.
        return float(value) if value is not None and math.isfinite(value) else _logit(observation.score)

    def innovation(self, observation: PlateDetection | VehicleDetection, ts: float) -> float:
        key = getattr(observation, 'frame_key', None)
        if key is not None and key in self.evidence_keys:
            return 0.0
        current = _unit(self._embedding(observation))
        if current is None or self.last_embedding is None or current.shape != self.last_embedding.shape:
            # Missing appearance cannot establish independent class evidence.
            return 0.0
        novelty = max(0.0, 1-float(np.clip(current@self.last_embedding, -1, 1)))
        return min(1.0, novelty, max(0.0, ts-self.last_mark_ts)/self.calibration.innovation_tau_s)

    def detected(self, observation: PlateDetection | VehicleDetection, ts: float,
                 association_probability: float, p_D: float = 0.0) -> None:
        factor = self.innovation(observation, ts)
        if isinstance(observation, PlateDetection):
            self.class_log_odds += factor*self._mark(observation)
            self.persistent_nonplate_probability = 1-_sigmoid(self.class_log_odds)
        class_p = _sigmoid(self.class_log_odds)
        # Geometry supports persistent occupancy, never plate-vs-decal class odds.
        r_detected = class_p
        r_null = missed_existence_probability(self.r,p_D)
        self.r = _prob(association_probability*r_detected+(1-association_probability)*r_null)
        self.last_mark_ts = ts
        self.last_frame_key = getattr(observation, 'frame_key', None)
        if self.last_frame_key is not None:
            self.evidence_keys.add(self.last_frame_key)
        self.last_embedding = _unit(self._embedding(observation))
        self.last_observation = observation
        self.last_mark = self._mark(observation)
        self.publish_if_cost_effective()

    def publish_if_cost_effective(self) -> None:
        c = self.calibration
        if c.c_false*(1-self.r) < c.c_wait*self.r:
            self.confirmed = True

    def predict_existence(self, dt: float) -> None:
        self.r = _prob(self.r*math.exp(-max(0.0,dt)/self.calibration.survival_tau_s))

    def missed_detection(self, p_D: float) -> None:
        self.r = missed_existence_probability(self.r, p_D)

    @property
    def retain(self) -> bool:
        c = self.calibration
        return c.c_retain*self.r > c.c_retire*(1-self.r)


class _PMBMPlateTrack(_KalmanPlateTrack, _ExistenceState):
    def __init__(self, entity_id: int, det: PlateDetection, ts: float,
                 calibration: TrackerCalibration, **kwargs: Any) -> None:
        _KalmanPlateTrack.__init__(self, entity_id, det, ts, **kwargs)
        _ExistenceState.__init__(self, det, calibration, ts)


_BOX_TO_STATE = np.array([[.5,0,.5,0], [0,.5,0,.5], [-1,0,1,0], [0,-1,0,1]])


def _vehicle_measurement(veh: VehicleDetection) -> tuple[np.ndarray, np.ndarray]:
    z = _BOX_TO_STATE @ np.asarray(veh.bbox_xyxy, float)
    if veh.bbox_cov_4x4 is not None:
        R = _BOX_TO_STATE @ np.asarray(veh.bbox_cov_4x4, float) @ _BOX_TO_STATE.T
    else:
        # Legacy observation uncertainty, with no amodal-height assertion.
        R = np.eye(4)*4.0
    return z, (R+R.T)*.5 + np.eye(4)*1e-6


class _KalmanVehicleTrack(_ExistenceState):
    """Continuous-white-acceleration CV state [cx,cy,w,h,vx,vy,vw,vh]."""
    def __init__(self, entity_id: int, veh: VehicleDetection, ts: float,
                 calibration: TrackerCalibration) -> None:
        super().__init__(veh, calibration, ts)
        self.vehicle_track_id = entity_id
        z, R = _vehicle_measurement(veh)
        self.x = np.r_[z, np.zeros(4)]
        self.P = np.zeros((8,8))
        self.P[:4,:4] = R
        self.P[4:,4:] = np.eye(4)*max(z[2], z[3])**2
        self.last_state_ts = self.last_obs_ts = ts
        self.hits, self.missed = 1, 0
        self.stage1_missed = 0
        self.opt_missed = 0
        self.missed_since_ts: float | None = None
        self.veh_ref: VehicleAppearanceRef | None = None
        self.app32: np.ndarray | None = None
        self.patch_ncc: float = 1.0
        self.opt_ok: bool = True
        self.bound_plate_track_ids: list[int] = []
        self.subtype_log_post = np.zeros(9, dtype=np.float64)
        self._update_subtype(veh)

    def update_appearance(self, ref: VehicleAppearanceRef | None) -> None:
        self.stage1_missed = 0
        self.opt_missed = 0
        self.missed_since_ts = None
        self.patch_ncc = 1.0
        self.opt_ok = True
        if ref is None:
            return
        self.veh_ref = ref
        if self.app32 is None or self.app32.shape != ref.app32.shape:
            self.app32 = ref.app32.copy()
        else:
            blended = 0.72 * self.app32 + 0.28 * ref.app32
            nrm = float(np.linalg.norm(blended))
            self.app32 = (blended / nrm).astype(np.float32) if nrm > 1e-6 else ref.app32.copy()

    def _update_subtype(self, veh: VehicleDetection) -> None:
        k = int(getattr(veh, "vehicle_subtype", 0))
        if 0 <= k < len(self.subtype_log_post):
            conf = float(np.clip(getattr(veh, "vehicle_subtype_conf", 0.55),
                                 1.0 / len(self.subtype_log_post), 0.98))
            p = np.full(len(self.subtype_log_post), (1.0 - conf) / (len(self.subtype_log_post) - 1),
                        dtype=np.float64)
            p[k] = conf
            self.subtype_log_post = 0.92 * self.subtype_log_post + np.log(p)
            self.subtype_log_post -= np.logaddexp.reduce(self.subtype_log_post)

    def peek_predict(self, ts: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(0.0, ts-self.last_state_ts)
        F = np.eye(8)
        F[:4,4:] = np.eye(4)*dt
        q = self.calibration.acceleration_spectral_density
        Q = np.block([[np.eye(4)*dt**3/3, np.eye(4)*dt**2/2],
                      [np.eye(4)*dt**2/2, np.eye(4)*dt]])*q
        return F@self.x, F@self.P@F.T+Q

    @staticmethod
    def _bbox(x: np.ndarray) -> tuple[float,float,float,float]:
        cx,cy,w,h = x[:4]
        w,h = max(1e-6,w), max(1e-6,h)
        return float(cx-w/2),float(cy-h/2),float(cx+w/2),float(cy+h/2)

    @property
    def bbox_xyxy(self) -> tuple[float,float,float,float]:
        return self._bbox(self.x)

    def peek_bbox_at(self, ts: float) -> tuple[float,float,float,float]:
        return self._bbox(self.peek_predict(ts)[0])

    def predict_to(self, ts: float) -> None:
        self.x,self.P = self.peek_predict(ts)
        self.last_state_ts = ts
        self.missed += 1

    def compensate(self, motion: CameraMotionEstimate) -> None:
        # Transform centre and box extents; map full state and camera covariance.
        A, b = motion.affine_2x3[:,:2], motion.affine_2x3[:,2]
        old = self.x.copy()
        J = np.zeros((8,8))
        J[:2,:2] = J[4:6,4:6] = A
        J[2:4,2:4] = J[6:8,6:8] = np.abs(A)
        Jg = np.zeros((8,6))
        Jg[0,:3] = [old[0],old[1],1]
        Jg[1,3:] = [old[0],old[1],1]
        Jg[2,[0,1]] = np.sign(A[0])*old[2:4]
        Jg[3,[3,4]] = np.sign(A[1])*old[2:4]
        Jg[4,[0,1]] = old[4:6]
        Jg[5,[3,4]] = old[4:6]
        Jg[6,[0,1]] = np.sign(A[0])*old[6:8]
        Jg[7,[3,4]] = np.sign(A[1])*old[6:8]
        self.x = J@old
        self.x[:2] += b
        self.P = J@self.P@J.T+Jg@motion.parameter_cov_6x6@Jg.T
        self.P[:2,:2] += motion.residual_cov_2x2
        self.P = (self.P+self.P.T)*.5

    def update(self, veh: VehicleDetection, ts: float) -> bool:
        if ts <= self.last_obs_ts:
            return False
        z,R = _vehicle_measurement(veh)
        S = self.P[:4,:4]+R
        K = np.linalg.solve(S,self.P[:,:4].T).T
        self.x += K@(z-self.x[:4])
        J = np.eye(8)
        J[:,:4] -= K
        self.P = J@self.P@J.T+K@R@K.T
        self.P = (self.P+self.P.T)*.5
        self.last_obs_ts = self.last_state_ts = ts
        self.hits += 1
        self.missed = 0
        self._update_subtype(veh)
        return True

    def predicted_mount_box_at(self, bbox: tuple[float,float,float,float]) -> tuple[float,float,float,float] | None:
        raw = self.last_observation
        if raw.plate_mount_box_xyxy is None:
            return None
        source = np.asarray(raw.bbox_xyxy)
        dest = np.asarray(bbox)
        size = source[2:]-source[:2]
        rel = (np.asarray(raw.plate_mount_box_xyxy).reshape(2,2)-source[:2])/np.maximum(size,1e-9)
        return tuple((dest[:2]+rel*(dest[2:]-dest[:2])).ravel())


class _PMBMTrackerBase:
    plate_mode = False

    def _initialize(self, calibration: TrackerCalibration | None, default_dt_s: float) -> None:
        self.calibration = calibration or TrackerCalibration()
        self.default_dt_s = default_dt_s
        self._tracks: list[Any] = []
        self._dormant_tracks: list[Any] = []
        self._current_ts: float | None = None
        self._next_id = 1
        self.last_expired_track_ids: list[int] = []
        self.last_relinked_pairs: list[tuple[int,int]] = []
        self.last_stale_rejected = False
        self.last_assignment_posterior = AssignmentPosterior((),0.0)
        self.camera_motion_estimator = BackgroundCameraMotionEstimator()
        self._seen_measurements: set[tuple[Any,...]] = set()
        self._step_obs_app32: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self._initialize(self.calibration,self.default_dt_s)

    def flush_for_scene_cut(self) -> list[int]:
        """Flush all active/dormant states on a hard scene cut without resetting monotonic IDs."""
        expired = [self._id(t) for t in (*self._tracks, *self._dormant_tracks) if t.confirmed]
        self._tracks.clear()
        self._dormant_tracks.clear()
        self._seen_measurements.clear()
        self._step_obs_app32.clear()
        self.last_expired_track_ids = expired
        self.last_relinked_pairs = []
        self.last_stale_rejected = False
        self.last_assignment_posterior = AssignmentPosterior((), 0.0)
        self.camera_motion_estimator.reset()
        return expired

    def _id(self, t: Any) -> int:
        return t.track_id if self.plate_mode else t.vehicle_track_id

    def _pair_likelihood(self, t: Any, obs: Any, ts: float) -> float:
        if self.plate_mode:
            residual = np.asarray(obs.corners_xy).reshape(8)-_H_24@t.x
            S = t.corner_covariance_8x8()+build_observation_cov_8x8(obs,t.ref_w)
            density = self.calibration.plate_clutter_density
            # Use the filter's predictive heave/scale Gaussian mixture. Coherent
            # motion innovations remain feasible before the filter selects a mode.
            q = (_H_24@t.x).reshape(4,2)
            offsets = q-q.mean(0)
            D = np.zeros((8,2))
            D[0::2,1] = offsets[:,0]
            D[1::2,0] = 1.0
            D[1::2,1] = offsets[:,1]
            h = max(1/60,min(.5,ts-t.last_obs_ts))
            jump = np.diag([max(6.0,1.35*t.ref_h)**2,.22**2])
            impulse = np.diag([max(18.0,4*t.ref_h)**2,.75**2])
            shock_cov = S+D@(jump+h*h/3*impulse)@D.T
            pi = _prob(t.shock_prob)
            geometry = float(np.logaddexp(math.log1p(-pi)+_gaussian_logpdf(residual,S),
                                         math.log(pi)+_gaussian_logpdf(residual,shock_cov)))
        else:
            z,R = _vehicle_measurement(obs)
            residual = z-t.x[:4]
            S = t.P[:4,:4]+R
            density = self.calibration.vehicle_clutter_density
            geometry = _gaussian_logpdf(residual,S)
        # Pair evidence uses a normalized conditional appearance density once.
        key = getattr(obs,'frame_key',None)
        innovation = 0.0 if key is not None and key in t.evidence_keys else min(
            1.0,max(0.0,ts-t.last_mark_ts)/self.calibration.innovation_tau_s)
        app = _vmf_log_ratio(t.last_embedding,t._embedding(obs),self.calibration.appearance_kappa)
        if not self.plate_mode:
            obs_app32 = self._step_obs_app32.get(id(obs))
            t_app32 = getattr(t, "app32", None)
            if obs_app32 is not None and t_app32 is not None:
                a_c = t_app32 - float(np.mean(t_app32))
                b_c = obs_app32 - float(np.mean(obs_app32))
                app32_ll = _vmf_log_ratio(a_c, b_c, min(6.0, 0.50 * self.calibration.appearance_kappa))
                app = (0.75 * app + 0.25 * app32_ll) if (t.last_embedding is not None and t._embedding(obs) is not None) else app32_ll
        return geometry-math.log(density)+innovation*app

    def _birth_weight(self, obs: Any) -> float:
        odds = self.calibration.policy_birth_odds.get(str(obs.provenance),self.calibration.birth_odds)
        return float(np.logaddexp(0.0,math.log(1+self.calibration.persistent_birth_ratio)+
                                  math.log(odds)+_ExistenceState._mark(obs)))

    def _deduplicate(self, observations: list[Any], action: ObservationActionMeta) -> list[Any]:
        unique = []
        for obs in observations:
            expected = PlateDetection if self.plate_mode else VehicleDetection
            if not isinstance(obs,expected):
                raise TypeError('Tracker input must contain fresh observations, never SearchRequest')
            if getattr(obs,'is_coasted',False):
                continue
            key = getattr(obs,'frame_key',None) or action.frame_key
            # Same pixels under different crop policies remain one physical measurement.
            measurement = (key,tuple(np.asarray(obs.corners_xy if self.plate_mode else obs.bbox_xyxy).ravel()))
            if key is not None and measurement in self._seen_measurements:
                continue
            if key is not None:
                self._seen_measurements.add(measurement)
                if self.plate_mode and obs.frame_key is None:
                    obs = replace(obs,frame_key=key)
            elif any(np.array_equal(obs.corners_xy if self.plate_mode else obs.bbox_xyxy,
                                     other.corners_xy if self.plate_mode else other.bbox_xyxy) for other in unique):
                continue
            if self.plate_mode and not is_valid_plate_quad(np.asarray(obs.corners_xy,np.float32)):
                continue
            unique.append(obs)
        return unique

    def _advance(self, observations: list[Any], timestamp_s: float | None,
                 action_meta: ObservationActionMeta | None,
                 camera_motion: CameraMotionEstimate | None,
                 frame_rgb: np.ndarray | None,
                 precomputed_veh_ncc: dict[int, tuple[bool, float, tuple[float, float, float, float], float, float]] | None = None) -> dict[int,Any]:
        ts = (0.0 if self._current_ts is None else self._current_ts+self.default_dt_s) if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(ts):
            raise ValueError('timestamp_s must be finite')
        self.last_expired_track_ids = []
        self.last_relinked_pairs = []
        self.last_stale_rejected = self._current_ts is not None and ts <= self._current_ts+1e-9
        if self.last_stale_rejected:
            return {}
        # Legacy entrypoint means a full detector invocation; pure API should pass action.
        action = action_meta if action_meta is not None else ObservationActionMeta(detector_ran_full=True)
        obs = self._deduplicate(observations,action)
        if camera_motion is None and frame_rgb is not None:
            boxes = [t.bbox_xyxy for t in self._tracks] if not self.plate_mode else []
            camera_motion = self.camera_motion_estimator.estimate(frame_rgb,boxes)
        dt = 0.0 if self._current_ts is None else ts-self._current_ts
        self._current_ts = ts
        for t in [*self._tracks,*self._dormant_tracks]:
            t.predict_existence(dt)
            t.predict_to(ts)
            if camera_motion is not None and dt > 0:
                if self.plate_mode:
                    _compensate_plate(t,camera_motion)
                else:
                    t.compensate(camera_motion)

        veh_textured = (
            not self.plate_mode
            and frame_rgb is not None
            and frame_rgb.ndim == 3
            and frame_rgb.size > 0
            and float(frame_rgb[::8, ::8].std()) > 2.0
        )
        obs_refs: dict[int, VehicleAppearanceRef | None] = {}
        self._step_obs_app32.clear()
        if veh_textured and obs:
            for j, z in enumerate(obs):
                ref = extract_vehicle_appearance(frame_rgb, z.bbox_xyxy)
                obs_refs[j] = ref
                if ref is not None:
                    self._step_obs_app32[id(z)] = ref.app32

        try:
            c = self.calibration
            pD = np.array([detection_availability(t.bbox_xyxy,action,c.detection_probability) for t in self._tracks])
            obs_pD = np.array([detection_availability(z.bbox_xyxy,action,c.detection_probability) for z in obs])
            pair = np.full((len(self._tracks),len(obs)),-math.inf)
            missed = np.array([math.log1p(-t.r*p) for t,p in zip(self._tracks,pD)])
            for i,t in enumerate(self._tracks):
                for j,z in enumerate(obs):
                    p_ij = float(pD[i]) if pD[i] > 0 else float(obs_pD[j])
                    if p_ij > 0:
                        pair[i,j] = math.log(t.r*p_ij)+self._pair_likelihood(t,z,ts)
            birth = np.array([self._birth_weight(z) for z in obs])
            posterior = feasible_assignment_posterior(pair,missed,birth,c.max_hypotheses,c.assignment_node_budget)
            self.last_assignment_posterior = posterior
            marginals = np.zeros_like(pair)
            for hypothesis in posterior.hypotheses:
                for i,j in hypothesis.pairs:
                    marginals[i,j] += hypothesis.probability
            assigned_obs: set[int] = set()
            updates: dict[int,Any] = {}
            best_pairs = posterior.hypotheses[0].pairs if posterior.hypotheses else ()
            assigned_tracks = set()
            for i,j in best_pairs:
                # Ambiguous geometry remains separate; pruning mass reduces certainty.
                if marginals[i,j] <= .5:
                    continue
                t = self._tracks[i]
                if not t.update(obs[j],ts):
                    continue
                pD_pair = float(pD[i]) if pD[i] > 0 else float(obs_pD[j])
                t.detected(obs[j],ts,float(marginals[i,j]),pD_pair)
                if veh_textured:
                    t.update_appearance(obs_refs.get(j))
                assigned_obs.add(j)
                assigned_tracks.add(i)
                updates[self._id(t)] = obs[j]
            for i,t in enumerate(self._tracks):
                if i not in assigned_tracks:
                    # If assignment is uncertain, null evidence only carries its posterior mass.
                    r0 = missed_existence_probability(t.r,float(pD[i]))
                    assigned_mass = float(marginals[i].sum())
                    t.r = _prob((1-assigned_mass)*r0+assigned_mass*t.r)
                    if veh_textured:
                        probe = precomputed_veh_ncc.get(t.vehicle_track_id) if precomputed_veh_ncc is not None else None
                        if probe is None and t.veh_ref is not None:
                            probe = track_vehicle_patch_ncc(
                                t.veh_ref, frame_rgb, t.bbox_xyxy, max_shift_px=24.0, min_ncc=0.42
                            )
                        if probe is not None:
                            ok_opt, ncc, mbox, mdx, mdy = probe
                            t.patch_ncc = float(ncc)
                            t.opt_ok = bool(ok_opt)
                        else:
                            ok_opt, ncc, mbox, mdx, mdy = False, 0.0, t.bbox_xyxy, 0.0, 0.0
                            t.patch_ncc = 0.0
                            t.opt_ok = False

                        if action.detector_ran_full:
                            t.stage1_missed += 1
                            t.x[4:8] *= 0.45
                            if t.missed_since_ts is None:
                                t.missed_since_ts = ts
                            if ok_opt:
                                t.opt_missed = 0
                                mcx = 0.5 * (mbox[0] + mbox[2])
                                mcy = 0.5 * (mbox[1] + mbox[3])
                                t.x[0] = 0.45 * t.x[0] + 0.55 * mcx
                                t.x[1] = 0.45 * t.x[1] + 0.55 * mcy
                            else:
                                t.opt_missed += 1
                            if (
                                t.stage1_missed >= 3
                                or (ts - t.missed_since_ts) > 0.30
                                or (t.veh_ref is not None and not ok_opt and t.stage1_missed >= 2)
                            ):
                                t.r = 1e-6
                        else:
                            if ok_opt:
                                t.opt_missed = 0
                                t.missed_since_ts = None
                                mcx = 0.5 * (mbox[0] + mbox[2])
                                mcy = 0.5 * (mbox[1] + mbox[3])
                                prev_cx, prev_cy = float(t.x[0]), float(t.x[1])
                                t.x[0] = 0.45 * prev_cx + 0.55 * mcx
                                t.x[1] = 0.45 * prev_cy + 0.55 * mcy
                                if dt > 1e-3:
                                    t.x[4] = 0.70 * t.x[4] + 0.30 * float(np.clip((mcx - prev_cx) / dt, -900.0, 900.0))
                                    t.x[5] = 0.70 * t.x[5] + 0.30 * float(np.clip((mcy - prev_cy) / dt, -900.0, 900.0))
                            else:
                                t.opt_missed += 1
                                if ncc < 0.30:
                                    t.x[4:8] *= 0.65
                                if t.missed_since_ts is None:
                                    t.missed_since_ts = ts
                                # Only expire on non-full frames if completely lost on asphalt (`ncc < 0.28`) for >= 3 frames
                                if ncc < 0.28 and (t.opt_missed >= 3 or (ts - t.missed_since_ts) > 0.35):
                                    t.r = 1e-6
            for t in self._dormant_tracks:
                t.missed_detection(detection_availability(t.bbox_xyxy,action,c.detection_probability))
            for j,z in enumerate(obs):
                if j in assigned_obs:
                    continue
                entity = self._new_track(self._next_id,z,ts)
                if self.plate_mode:
                    # Inhomogeneous Poisson birth intensity: observations already
                    # explained by this frame's associated plate measurements leave
                    # only their uncovered spatial fraction available for a birth.
                    # Use fresh measurements, not a predicted/coasted track box.
                    box = np.asarray(z.bbox_xyxy, dtype=float)
                    area = float(np.prod(np.maximum(0., box[2:]-box[:2])))
                    overlap = 0.0
                    for k in assigned_obs:
                        other = np.asarray(obs[k].bbox_xyxy, dtype=float)
                        intersection = float(np.prod(np.maximum(0., np.minimum(box[2:],other[2:])-np.maximum(box[:2],other[:2]))))
                        overlap = max(overlap, intersection / max(area, np.finfo(float).tiny))
                    scale = max(0.0, 1.0-overlap)
                    entity.r = _sigmoid(_logit(entity.r)+math.log(scale)) if scale > 0 else 0.0
                elif veh_textured:
                    entity.update_appearance(obs_refs.get(j))
                self._next_id += 1
                self._tracks.append(entity)
                updates[self._id(entity)] = z
            if self.plate_mode:
                self._resolve_disjoint_merges(ts,updates)
            retained = []
            for t in self._tracks:
                if not t.retain:
                    if t.confirmed:
                        self.last_expired_track_ids.append(self._id(t))
                elif self.plate_mode and t.confirmed and ts-t.last_obs_ts >= c.dormant_after_s:
                    self._dormant_tracks.append(t)
                else:
                    retained.append(t)
            self._tracks = retained
            dormant = []
            for t in self._dormant_tracks:
                if t.retain:
                    dormant.append(t)
                elif t.confirmed:
                    self.last_expired_track_ids.append(self._id(t))
            self._dormant_tracks = dormant
            return updates
        finally:
            self._step_obs_app32.clear()


class PlateTrackStabilizer(_PMBMTrackerBase):
    """Plate PMBM adapter. Public IDs exist only after expected-cost confirmation.

    Legacy score/frame-count constructor options are accepted for source
    compatibility but do not determine birth, identity, or expiration.
    """
    plate_mode = True

    def __init__(self, high_thresh: float = .36, low_thresh: float = .16,
                 min_affinity: float = .28, max_missed: int | None = None,
                 min_hits_to_coast: int = 2, default_dt_s: float = .05,
                 alpha_blur_phase: float = 0.0, sigma_floor_px: float = .25,
                 enable_dormant_reid: bool = True, dormant_max_age_s: float = 8.0,
                 relink_min_hits: int = 2, relink_eps: float = .05,
                 calibration: TrackerCalibration | None = None) -> None:
        self._initialize(calibration,default_dt_s)
        self.alpha_blur_phase,self.sigma_floor_px = alpha_blur_phase,sigma_floor_px
        self.enable_dormant_reid = enable_dormant_reid
        self.relink_min_hits = relink_min_hits
        self.relink_eps = relink_eps
        # Compatibility attributes only. No missed-frame decision uses them.
        self.max_missed = max_missed
        self.high_thresh,self.low_thresh = high_thresh,low_thresh
        self.min_hits_to_coast = min_hits_to_coast

    def _new_track(self, entity: int, obs: PlateDetection, ts: float) -> _PMBMPlateTrack:
        return _PMBMPlateTrack(entity,obs,ts,self.calibration,
                               alpha_blur_phase=self.alpha_blur_phase,
                               sigma_floor_px=self.sigma_floor_px)

    @property
    def active_track_ids(self) -> list[int]:
        return [t.track_id for t in self._tracks if t.confirmed]

    @property
    def dormant_track_ids(self) -> list[int]:
        return [t.track_id for t in self._dormant_tracks if t.confirmed]

    @property
    def tentative_track_ids(self) -> set[int]:
        return {t.track_id for t in self._tracks if not t.confirmed}

    def _dormant_pair_log_odds(self, dorm: _PMBMPlateTrack,
                              tent: _PMBMPlateTrack, now_ts: float) -> float:
        # Numeric vehicle IDs, plate-side labels, and character outputs never enter.
        z = tent.last_observation
        return math.log(dorm.r)+self._pair_likelihood(dorm,z,now_ts)-self._birth_weight(z)

    def _resolve_disjoint_merges(self, ts: float, updates: dict[int,PlateDetection]) -> None:
        if not self.enable_dormant_reid or not self._dormant_tracks:
            return
        candidates = [t for t in self._tracks if t.hits >= self.relink_min_hits and
                      t.track_id in updates]
        if not candidates:
            return
        pair = np.full((len(self._dormant_tracks), len(candidates)), -1e12)
        has_valid_pair = False
        for i, d in enumerate(self._dormant_tracks):
            for j, t in enumerate(candidates):
                if ((d.evidence_keys & t.evidence_keys) or
                        float(getattr(t, "first_obs_ts", t.last_obs_ts)) <= float(d.last_obs_ts)):
                    continue
                pair[i, j] = self._dormant_pair_log_odds(d, t, ts)
                has_valid_pair = True
        if not has_valid_pair:
            return
        posterior = feasible_assignment_posterior(pair,np.zeros(len(pair)),np.zeros(len(candidates)),
                                                  self.calibration.max_hypotheses,
                                                  self.calibration.assignment_node_budget)
        marginals = np.zeros_like(pair)
        for h in posterior.hypotheses:
            for i,j in h.pairs:
                marginals[i,j] += h.probability
        for i,j in (posterior.hypotheses[0].pairs if posterior.hypotheses else ()):
            if marginals[i,j] <= 1-self.relink_eps:
                continue
            d,t = self._dormant_tracks[i],candidates[j]
            # Evidence support intervals must be strictly disjoint for notifications.
            if ((d.evidence_keys & t.evidence_keys) or
                    float(getattr(t, "first_obs_ts", t.last_obs_ts)) <= float(d.last_obs_ts)):
                continue
            old_id,new_id = d.track_id,t.track_id
            keys = d.evidence_keys | t.evidence_keys
            # Keep both posterior snapshots for audit; do not erase merge evidence.
            history = getattr(d,'merge_evidence',[])+[(old_id,new_id,d.x.copy(),d.P.copy(),d.r,t.r)]
            d.x,d.P = t.x.copy(),t.P.copy()
            d.first_obs_ts = min(float(getattr(d, "first_obs_ts", d.last_obs_ts)),
                                 float(getattr(t, "first_obs_ts", t.last_obs_ts)))
            d.last_state_ts,d.last_obs_ts = t.last_state_ts,t.last_obs_ts
            d.last_observation = t.last_observation
            d.r = max(d.r,t.r)
            d.hits += t.hits
            d.missed = 0
            d.evidence_keys = keys
            d.merge_evidence = history
            d.last_valid_modes_uvw = t.last_valid_modes_uvw.copy()
            d.last_embedding = t.last_embedding
            d.last_mark_ts = t.last_mark_ts
            d.last_frame_key = t.last_frame_key
            d.ref_w,d.ref_h = t.ref_w,t.ref_h
            d.last_blur_vec_xy = t.last_blur_vec_xy.copy()
            d.parent_vehicle_track_id = t.parent_vehicle_track_id
            d.vehicle_bbox_xyxy = t.vehicle_bbox_xyxy
            self._tracks.remove(t)
            self._tracks.append(d)
            updates[old_id] = updates.pop(new_id)
            self.last_relinked_pairs.append((new_id,old_id))
        restored = {old for _,old in self.last_relinked_pairs}
        self._dormant_tracks = [d for d in self._dormant_tracks if d.track_id not in restored]

    def step(self, detections: list[PlateDetection], timestamp_s: float | None = None,
             *, action_meta: ObservationActionMeta | None = None,
             camera_motion: CameraMotionEstimate | None = None,
             frame_rgb: np.ndarray | None = None,
             vehicle_velocities: dict[int,tuple[float,float]] | None = None,
             **legacy_unused: Any) -> list[PlateDetection]:
        # Deprecated caller kwargs are discarded without inspecting their contents.
        updates = self._advance(detections,timestamp_s,action_meta,camera_motion,frame_rgb)
        result = []
        for t in self._tracks:
            raw = updates.get(t.track_id)
            if raw is None:
                continue
            C = t.corner_covariance_8x8()
            cov = np.stack([C[2*k:2*k+2,2*k:2*k+2] for k in range(4)])
            result.append(replace(raw,track_id=t.track_id if t.confirmed else None,
                                  bbox_xyxy=t.bbox_xyxy,corners_xy=t.corners_xy.copy(),
                                  corner_cov_xy=cov,is_coasted=False))
        # Predictions are emitted exclusively as SearchRequest, never detections.
        return result

    def get_predicted_roi_proposals(self, next_timestamp_s: float | None = None,
                                    dt_jitter_s: float = .005, eps_containment: float = 1e-3,
                                    max_gamma: float = 4.0, reacquire_gamma: float = 3.2,
                                    include_dormant: bool = True) -> list[PredictedROIProposal]:
        ts = next_timestamp_s if next_timestamp_s is not None else (self._current_ts or 0)+self.default_dt_s
        return [t.build_roi_proposal(ts,dt_jitter_s,eps_containment,max_gamma,reacquire_gamma,
                                      t in self._dormant_tracks)
                for t in [*self._tracks,*(self._dormant_tracks if include_dormant else [])]]

    def get_search_requests(self, next_timestamp_s: float | None = None,
                            img_hw: tuple[int,int] | None = None,
                            include_dormant: bool = True,
                            alpha_crop: float = 1e-3) -> list[SearchRequest]:
        ts = next_timestamp_s if next_timestamp_s is not None else (self._current_ts or 0)+self.default_dt_s
        requests = []
        for t in [*self._tracks,*(self._dormant_tracks if include_dormant else [])]:
            x,P = t.peek_predict(ts)
            q = (_H_24@x).reshape(4,2)
            C = _H_24@P@_H_24.T
            cov = np.stack([C[2*k:2*k+2,2*k:2*k+2] for k in range(4)])+np.eye(2)[None]*1e-9
            plan = select_containment_crop_window(q,cov,img_hw or (1,1),alpha_crop)
            roi = plan.window_xyxy if img_hw is not None else plan.required_window_xyxy
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                continue
            request = SearchRequest(f'plate:{t.track_id}:{ts}',roi,
                                    t.track_id if t.confirmed else None,
                                    'dormant_plate' if t in self._dormant_tracks else 'plate_containment',
                                    (float(P[0,0]),float(P[1,1])),t.r*self.calibration.c_false)
            assert_not_observation(request)
            requests.append(request)
        return requests

    plan_searches = get_search_requests


class MultiVehicleKalmanTracker(_PMBMTrackerBase):
    def __init__(self, max_missed: int | None = None, min_affinity: float | None = None,
                 birth_thresh: float | None = None, default_dt_s: float = .05,
                 calibration: TrackerCalibration | None = None) -> None:
        self._initialize(calibration,default_dt_s)
        self.max_missed = max_missed  # compatibility only
        self.attachment_posteriors: dict[int,dict[int | None,AttachmentState]] = {}
        self._anonymous_attachment_posteriors: list[dict[int | None,AttachmentState]] = []

    def reset(self) -> None:
        super().reset()
        self.attachment_posteriors = {}
        self._anonymous_attachment_posteriors = []

    def flush_for_scene_cut(self) -> list[int]:
        expired = super().flush_for_scene_cut()
        self.attachment_posteriors.clear()
        self._anonymous_attachment_posteriors.clear()
        return expired

    def probe_vehicle_ncc(
        self,
        frame_rgb: np.ndarray | None,
        timestamp_s: float,
    ) -> tuple[int, int, dict[int, tuple[bool, float, tuple[float, float, float, float], float, float]]]:
        """Pre-detection 16x16 Patch-NCC probe across active vehicle tracks for instant scene-cut consensus."""
        if frame_rgb is None or frame_rgb.ndim != 3 or frame_rgb.size == 0:
            return 0, 0, {}
        if float(frame_rgb[::8, ::8].std()) <= 2.0:
            return 0, 0, {}
        probed = 0
        low_ncc = 0
        results: dict[int, tuple[bool, float, tuple[float, float, float, float], float, float]] = {}
        for t in self._tracks:
            if not t.confirmed or t.veh_ref is None:
                continue
            dt_p = max(0.0, min(0.30, float(timestamp_s) - t.last_state_ts))
            cx = float(t.x[0]) + float(t.x[4]) * dt_p
            cy = float(t.x[1]) + float(t.x[5]) * dt_p
            bw = max(10.0, float(t.x[2]))
            bh = max(10.0, float(t.x[3]))
            pred_box = (cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh)
            res = track_vehicle_patch_ncc(t.veh_ref, frame_rgb, pred_box, max_shift_px=24.0, min_ncc=0.42)
            results[t.vehicle_track_id] = res
            probed += 1
            if res[1] < 0.32:
                low_ncc += 1
        return probed, low_ncc, results

    @property
    def active_vehicle_track_ids(self) -> list[int]:
        return [t.vehicle_track_id for t in self._tracks if t.confirmed]

    @property
    def tentative_vehicle_track_ids(self) -> set[int]:
        return {t.vehicle_track_id for t in self._tracks if not t.confirmed}

    def _new_track(self, entity: int, obs: VehicleDetection, ts: float) -> _KalmanVehicleTrack:
        return _KalmanVehicleTrack(entity,obs,ts,self.calibration)

    def step(self, vehicles: list[VehicleDetection], timestamp_s: float | None = None,
             img_hw: tuple[int,int] | None = None, *,
             action_meta: ObservationActionMeta | None = None,
             camera_motion: CameraMotionEstimate | None = None,
             frame_rgb: np.ndarray | None = None,
             precomputed_veh_ncc: dict[int, tuple[bool, float, tuple[float, float, float, float], float, float]] | None = None) -> list[VehicleDetection]:
        updates = self._advance(vehicles,timestamp_s,action_meta,camera_motion,frame_rgb,precomputed_veh_ncc=precomputed_veh_ncc)
        return [replace(updates[t.vehicle_track_id],bbox_xyxy=t.bbox_xyxy,
                        vehicle_track_id=t.vehicle_track_id if t.confirmed else None,
                        vehicle_subtype=int(np.argmax(t.subtype_log_post)),
                        plate_track_ids=list(t.bound_plate_track_ids))
                for t in self._tracks if t.vehicle_track_id in updates]

    def predict_vehicles_at(self, timestamp_s: float,
                            img_hw: tuple[int,int] | None = None) -> list[VehicleDetection]:
        """Deprecated observation adapter; use get_search_requests for predictions.

        A frame without a measurement has no VehicleDetection to emit. The
        retained component states remain available in _tracks for display adapters.
        """
        return []

    def get_vehicle_velocities(self) -> dict[int,tuple[float,float]]:
        return {t.vehicle_track_id:(float(t.x[4]),float(t.x[5]))
                for t in self._tracks if t.confirmed}

    def bind_plates_and_sync(self, vehicles: list[VehicleDetection], plates: list[PlateDetection],
                             update_vehicle_plate_lists: bool = True,
                             img_hw: tuple[int,int] | None = None,
                             plate_tracker: PlateTrackStabilizer | None = None) -> dict[int,tuple[float,float]]:
        """Infer B over unbound and uncertain offsets, with no side-slot constraint.

        Spatial support is a Gaussian moment approximation to the component box;
        learned mount support may be used instead. Repeated relative offsets and
        velocity innovations provide co-motion likelihood. Existing offsets are
        marginalized independently of their numeric parent keys, allowing component
        IDs to change without a discontinuity in plate identity evidence.
        """
        valid = [v for v in vehicles if v.vehicle_track_id is not None]
        tracks = {t.vehicle_track_id:t for t in self._tracks}
        plate_states = {t.track_id:t for t in [*plate_tracker._tracks,*plate_tracker._dormant_tracks]} if plate_tracker else {}
        if update_vehicle_plate_lists:
            for v in vehicles:
                v.plate_track_ids = []
            for t in self._tracks:
                t.bound_plate_track_ids = []
        self._anonymous_attachment_posteriors = []
        area = float(img_hw[0]*img_hw[1]) if img_hw else max(
            1.0,sum(max(1.0,(v.bbox_xyxy[2]-v.bbox_xyxy[0])*(v.bbox_xyxy[3]-v.bbox_xyxy[1])) for v in valid))
        for p in plates:
            pc = np.asarray(p.corners_xy,float).mean(0)
            pt = plate_states.get(p.track_id)
            previous = self.attachment_posteriors.get(p.track_id,{}) if p.track_id is not None else {}
            entries: dict[int | None,AttachmentState] = {None:AttachmentState(1.0,np.zeros(2),np.eye(2))}
            weights: dict[int | None,float] = {None:0.0}
            for v in valid:
                vid = int(v.vehicle_track_id)
                box = np.asarray(v.bbox_xyxy,float)
                center = (box[:2]+box[2:])/2
                extent = np.maximum(box[2:]-box[:2],1.0)
                offset = (pc-center)/extent
                support = np.asarray(v.plate_mount_box_xyxy if v.plate_mount_box_xyxy is not None else box,float)
                support_center = (support[:2]+support[2:])/2
                support_extent = np.maximum(support[2:]-support[:2],1.0)
                vt = tracks.get(vid)
                C = np.diag(support_extent**2/12)
                if vt is not None:
                    C += vt.P[:2,:2]
                if pt is not None:
                    C += pt.P[:2,:2]
                spatial = _gaussian_logpdf(pc-support_center,C)+math.log(area)
                previous_offsets = [state for key,state in previous.items() if key is not None]
                offset_cov = C/np.outer(extent,extent)
                motion = 0.0
                if previous_offsets:
                    # Relative-motion likelihood, including uncertainty in previous offset.
                    total = sum(state.probability for state in previous_offsets)
                    if total > 0.0 and math.isfinite(total):
                        motion = _logsumexp([math.log(max(state.probability/total,1e-12))+
                                              _gaussian_logpdf(offset-state.offset_xy,
                                                               offset_cov+state.covariance_xy)
                                              for state in previous_offsets])
                if pt is not None and vt is not None:
                    V = pt.P[2:4,2:4]+vt.P[4:6,4:6]
                    # Compare co-motion density with broad relative-velocity null.
                    relative = pt.x[2:4]-vt.x[4:6]
                    broad = V+np.diag(extent**2)
                    motion += _gaussian_logpdf(relative,V)-_gaussian_logpdf(relative,broad)
                weights[vid] = spatial+motion-math.log(max(1,len(valid)))
                entries[vid] = AttachmentState(0.0,offset,offset_cov)
            normalizer = _logsumexp(list(weights.values()))
            for key,state in entries.items():
                state.probability = math.exp(weights[key]-normalizer)
            if p.track_id is not None:
                self.attachment_posteriors[p.track_id] = entries
                if pt is not None:
                    pt.attachments = entries
            else:
                self._anonymous_attachment_posteriors.append(entries)
            best = max(entries,key=lambda key:entries[key].probability)
            p.parent_vehicle_track_id = None
            p.vehicle_bbox_xyxy = None
            if best is not None and self.calibration.c_false*(1-entries[best].probability) < self.calibration.c_wait*entries[best].probability:
                p.parent_vehicle_track_id = best
                parent = next(v for v in valid if v.vehicle_track_id == best)
                p.vehicle_bbox_xyxy = parent.bbox_xyxy
                if pt is not None:
                    pt.parent_vehicle_track_id = best
                if update_vehicle_plate_lists and p.track_id is not None:
                    parent.plate_track_ids.append(p.track_id)
                    if best in tracks:
                        tracks[best].bound_plate_track_ids.append(p.track_id)
        return self.get_vehicle_velocities()

    def build_unplated_bumper_proposals(self, vehicles: list[VehicleDetection],
                                        covered_vehicle_ids: set[int],
                                        img_hw: tuple[int,int]) -> list[SearchRequest]:
        """Compatibility name for score-free component/mount search actions."""
        requests = []
        known = {t.vehicle_track_id:t for t in self._tracks if t.confirmed}
        for v in vehicles:
            vid = v.vehicle_track_id
            if vid is None or vid not in known or vid in covered_vehicle_ids:
                continue
            t = known[vid]
            roi = v.plate_mount_box_xyxy or v.bbox_xyxy
            lo = np.maximum(roi[:2],0)
            hi = np.minimum(roi[2:],[img_hw[1],img_hw[0]])
            if np.any(hi <= lo):
                continue
            request = SearchRequest(f'mount:{vid}:{self._current_ts}',tuple(np.r_[lo,hi]),vid,
                                    'vehicle_mount' if v.plate_mount_box_xyxy else 'component_search',
                                    (float(t.P[0,0]),float(t.P[1,1])),t.r*self.calibration.c_false)
            assert_not_observation(request)
            requests.append(request)
        return requests

    def get_search_requests(self, next_timestamp_s: float | None = None,
                            img_hw: tuple[int,int] = (1080,1920),
                            covered_vehicle_ids: set[int] | None = None) -> list[SearchRequest]:
        ts = next_timestamp_s if next_timestamp_s is not None else (self._current_ts or 0)+self.default_dt_s
        requests = []
        for t in self._tracks:
            if not t.confirmed or t.vehicle_track_id in (covered_vehicle_ids or set()):
                continue
            x,P = t.peek_predict(ts)
            box = t._bbox(x)
            mount_roi = t.predicted_mount_box_at(box)
            roi = mount_roi or box
            # Include predictive position uncertainty, without fabricating a measurement.
            k = _norm_ppf_tail(1e-3/4)
            radius = k*np.sqrt(np.maximum(np.diag(P)[:2],0))
            lo = np.maximum(np.asarray(roi[:2])-radius,0)
            hi = np.minimum(np.asarray(roi[2:])+radius,[img_hw[1],img_hw[0]])
            if np.any(hi<=lo):
                continue
            request = SearchRequest(f'component:{t.vehicle_track_id}:{ts}',tuple(np.r_[lo,hi]),
                                    t.vehicle_track_id,
                                    'vehicle_mount' if mount_roi is not None else 'component_search',
                                    (float(P[0,0]),float(P[1,1])),t.r*self.calibration.c_false)
            assert_not_observation(request)
            requests.append(request)
        return requests

    plan_searches = get_search_requests


class CoupledPMBMTracker:
    """Pure Gen-6 facade sharing one CMC estimate across both entity filters."""
    def __init__(self, calibration: TrackerCalibration | None = None) -> None:
        self.plates = PlateTrackStabilizer(calibration=calibration)
        self.vehicles = MultiVehicleKalmanTracker(calibration=calibration)
        self.camera_motion_estimator = BackgroundCameraMotionEstimator()
        self.last_camera_motion: CameraMotionEstimate | None = None

    def reset(self) -> None:
        self.plates.reset()
        self.vehicles.reset()
        self.camera_motion_estimator.reset()
        self.last_camera_motion = None

    def step(self, plates: list[PlateDetection], vehicles: list[VehicleDetection],
             timestamp_s: float, action_meta: ObservationActionMeta,
             frame_rgb: np.ndarray | None = None,
             camera_motion: CameraMotionEstimate | None = None) -> tuple[list[PlateDetection],list[VehicleDetection]]:
        if camera_motion is None and frame_rgb is not None:
            boxes = [t.bbox_xyxy for t in self.vehicles._tracks]+[v.bbox_xyxy for v in vehicles]
            camera_motion = self.camera_motion_estimator.estimate(frame_rgb,boxes)
        self.last_camera_motion = camera_motion
        v = self.vehicles.step(vehicles,timestamp_s,action_meta=action_meta,camera_motion=camera_motion)
        p = self.plates.step(plates,timestamp_s,action_meta=action_meta,camera_motion=camera_motion)
        self.vehicles.bind_plates_and_sync(v,p,img_hw=frame_rgb.shape[:2] if frame_rgb is not None else None,
                                          plate_tracker=self.plates)
        return p,v

    def get_search_requests(self, next_timestamp_s: float,
                            img_hw: tuple[int,int]) -> list[SearchRequest]:
        covered = {t.parent_vehicle_track_id for t in self.plates._tracks if t.confirmed and t.parent_vehicle_track_id is not None}
        return self.plates.get_search_requests(next_timestamp_s,img_hw)+self.vehicles.get_search_requests(next_timestamp_s,img_hw,covered)

    plan_searches = get_search_requests


__all__ = ['PlateTrackStabilizer','MultiVehicleKalmanTracker','CoupledPMBMTracker',
           'BackgroundCameraMotionEstimator','CameraMotionEstimate','TrackerCalibration',
           'AssignmentHypothesis','AssignmentPosterior','AttachmentState',
           'feasible_assignment_posterior','detection_availability',
           'missed_existence_probability','PredictedROIProposal','build_observation_cov_8x8']
