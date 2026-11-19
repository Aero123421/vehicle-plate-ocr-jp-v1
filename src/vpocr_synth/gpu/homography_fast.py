"""Continuous 3D SE(3) vehicle/camera trajectories & closed-form `O(1)` homographies.

Generates realistic multi-frame 3D trajectories across all 14 operational regimes
(including `±72°` steep yaw, `55°` overhead pitch, `turn_in_sweep` nose-dive,
`multicam_rgb_ir`, `track_mixed`, `temporary_plate`, and `negative_garbage`),
projects into native-pixel Camera Sensor Local ROI `(96, 160)` with CMOS rolling
shutter skew, and rectifies noisy 4-corner quads to `(80, 160)` via exact
closed-form `O(1)` projective homography without `torch.linalg.solve`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import os

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import numpy as np
import torch
import torch.nn.functional as F

SENSOR_ROI_H: int = 96
SENSOR_ROI_W: int = 160

REGIME_NAMES: tuple[str, ...] = (
    "day_clear",
    "day_harsh_shadow",
    "dusk_wdr",
    "night_rgb_glare",
    "night_ir_850",
    "night_ir_940",
    "illuminated_night",
    "heavy_rain_droplets",
    "steep_oblique",
    "turn_in_sweep",
    "multicam_rgb_ir",
    "track_mixed",
    "temporary_plate",
    "negative_garbage",
)
REGIME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(REGIME_NAMES)}

# Progressive 4-stage curriculum sampling weights over REGIME_NAMES (length 14)
STAGE_REGIME_WEIGHTS: dict[int, tuple[float, ...]] = {
    # Stage 1: Clean/mild geometry + color/glyph/IR basics + 6% negative_garbage
    1: (0.32, 0.08, 0.14, 0.05, 0.12, 0.06, 0.07, 0.02, 0.03, 0.02, 0.02, 0.00, 0.01, 0.06),
    # Stage 2: Relief parallax, harsh shadows, 850/940nm IR, turn sweep, 10% negative_garbage
    2: (0.13, 0.09, 0.09, 0.08, 0.10, 0.08, 0.07, 0.05, 0.06, 0.05, 0.04, 0.03, 0.03, 0.10),
    # Stage 3: Heavy rain droplets, steep oblique, multicam, glare, track_mixed, 13% negative_garbage
    3: (0.06, 0.08, 0.06, 0.08, 0.09, 0.08, 0.07, 0.08, 0.08, 0.06, 0.05, 0.04, 0.04, 0.13),
    # Stage 4: Full hard real-world OOD envelope + 14% negative_garbage
    4: (0.05, 0.07, 0.06, 0.08, 0.08, 0.07, 0.06, 0.08, 0.09, 0.07, 0.06, 0.05, 0.04, 0.14),
}


@dataclass
class TrajectoryBatch:
    """Batched `(B, T, ...)` 3D SE(3) camera-to-plate pose & sensor ROI projection state."""

    R_c2p: torch.Tensor                  # (B, T, 3, 3) float32 camera->plate rotation (R^T)
    C_local_mm: torch.Tensor             # (B, T, 3) float32 camera optical center in plate mm (Cz < 0 in front)
    view_dir_local: torch.Tensor         # (B, T, 3) float32 unit vector plate->camera (vz = cos(theta_i) > 0)
    parallax_tan_xy: torch.Tensor        # (B, T, 2) float32 (dx/dh, dy/dh) tangent parallax for +h_mm relief
    intrinsics: torch.Tensor             # (B, T, 4) float32 (fx, fy, cx, cy)
    roi_origin_xy: torch.Tensor          # (B, T, 2) float32 (rx0, ry0) of (112, 176) window on full sensor
    corners_full_gt: torch.Tensor        # (B, T, 4, 2) float32 [TL, TR, BR, BL] in full sensor px
    corners_roi_gt: torch.Tensor         # (B, T, 4, 2) float32 [TL, TR, BR, BL] in (112, 176) ROI px
    motion_vec_px: torch.Tensor          # (B, T, 2) float32 exposure motion blur vector (du, dv) in ROI px
    rolling_shutter_px: torch.Tensor     # (B, T, 2) float32 CMOS top-to-bottom row shear (sx, sy) in ROI px
    timestamps_s: torch.Tensor           # (B, T) float32
    camera_ids: torch.Tensor             # (B, T) int64
    plate_sides: torch.Tensor            # (B, T) int64 (0=front, 1=rear)
    yaw_deg: torch.Tensor                # (B, T) float32
    pitch_deg: torch.Tensor              # (B, T) float32
    roll_deg: torch.Tensor               # (B, T) float32
    sensor_ir_mode: torch.Tensor         # (B, T) int64 (0=RGB, 1=850nm IR, 2=940nm IR)
    is_night: torch.Tensor               # (B, T) bool
    splice_second_plate: torch.Tensor    # (B, T) bool (True on spliced vehicle frames of track_mixed)
    multicam_front_rear: torch.Tensor    # (B,) bool (True when multicam switches front->rear)
    regime_ids: torch.Tensor             # (B,) int64
    regime_tags: list[str]               # length B
    primary_sides_cpu: list[int]         # length B (0=front, 1=rear on final/primary frame, zero GPU sync!)
    is_approach: torch.Tensor | None = None   # (B,) bool (True on 13..22px -> 78..138px approach trajectories)
    is_lowres_sr: torch.Tensor | None = None  # (B,) bool (True on persistent 12..28px multi-frame SR trajectories)
    w_target_px: torch.Tensor | None = None   # (B, T) float32 projected sensor plate width in pixels


_HOMOG_DEV_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_homog_dev_consts(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cached = _HOMOG_DEV_CACHE.get(dev)
    if cached is not None:
        return cached
    plate_corners_3d = torch.tensor(
        [
            [-165.0, -82.5, 0.0],
            [165.0, -82.5, 0.0],
            [165.0, 82.5, 0.0],
            [-165.0, 82.5, 0.0],
        ],
        device=dev,
        dtype=torch.float32,
    )
    u_1d = torch.arange(SENSOR_ROI_W, device=dev, dtype=torch.float32)
    v_1d = torch.arange(SENSOR_ROI_H, device=dev, dtype=torch.float32)
    out = (plate_corners_3d, u_1d, v_1d)
    _HOMOG_DEV_CACHE[dev] = out
    return out


def euler_zyx_to_rotmat(
    yaw_rad: torch.Tensor,
    pitch_rad: torch.Tensor,
    roll_rad: torch.Tensor,
) -> torch.Tensor:
    """Build right-handed `3x3` rotation `R = R_z(roll) @ R_x(pitch) @ R_y(yaw)`.

    Plate local frame: `+X` right, `+Y` down, `+Z` into vehicle (away from camera),
    so outward 2.5D relief height `+h_mm` sits at `Z_p = -h_mm` closer to the camera.
    Negative `yaw` places the camera on the left (`C_local.x < 0`), looking in `+X`
    across the rear-left `+8mm` `封印` cap `(x=60mm)` onto Region Slot 0 `(x>=75mm)`.
    """
    cy, sy = torch.cos(yaw_rad), torch.sin(yaw_rad)
    cp, sp = torch.cos(pitch_rad), torch.sin(pitch_rad)
    cr, sr = torch.cos(roll_rad), torch.sin(roll_rad)

    # R_x(pitch) @ R_y(yaw):
    # [ cy,       0,      sy     ]
    # [ sp*sy,    cp,    -sp*cy  ]
    # [-cp*sy,    sp,     cp*cy  ]
    m00 = cy
    m01 = torch.zeros_like(cy)
    m02 = sy
    m10 = sp * sy
    m11 = cp
    m12 = -sp * cy
    m20 = -cp * sy
    m21 = sp
    m22 = cp * cy

    # Left-multiply by R_z(roll) = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]
    r00 = cr * m00 - sr * m10
    r01 = cr * m01 - sr * m11
    r02 = cr * m02 - sr * m12
    r10 = sr * m00 + cr * m10
    r11 = sr * m01 + cr * m11
    r12 = sr * m02 + cr * m12

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def sample_se3_trajectories(
    batch_size: int,
    seq_len: int,
    curriculum_stage: int = 4,
    regime_override: str | Sequence[str] | None = None,
    primary_sides: torch.Tensor | Sequence[int] | int | None = None,
    device: torch.device | str = "cpu",
    seed: int = 0,
    approach_prob: float | None = None,
    lowres_sr_prob: float | None = None,
) -> TrajectoryBatch:
    """Sample vectorized continuous 3D SE(3) trajectories across `T` frames on `device`."""
    dev = torch.device(device)
    stage = int(max(1, min(4, curriculum_stage)))
    rng = np.random.default_rng(seed)

    # 1. Select per-sequence regime tags
    if regime_override is None:
        weights = np.asarray(STAGE_REGIME_WEIGHTS[stage], dtype=np.float64)
        weights = weights / weights.sum()
        reg_ids_np = rng.choice(len(REGIME_NAMES), size=batch_size, p=weights).astype(np.int64)
        tags = [REGIME_NAMES[int(i)] for i in reg_ids_np]
    elif isinstance(regime_override, str):
        rid = REGIME_TO_ID[regime_override]
        reg_ids_np = np.full(batch_size, rid, dtype=np.int64)
        tags = [regime_override] * batch_size
    else:
        tags = [str(regime_override[i % len(regime_override)]) for i in range(batch_size)]
        reg_ids_np = np.asarray([REGIME_TO_ID[t] for t in tags], dtype=np.int64)

    gen = torch.Generator(device="cpu")
    gen.manual_seed((seed * 1664525 + 1013904223) & 0x7FFFFFFF)

    cpu_dev = torch.device("cpu")
    reg_ids_cpu = torch.from_numpy(reg_ids_np)  # (B,)
    t_norm = (
        torch.linspace(0.0, 1.0, seq_len, device=cpu_dev, dtype=torch.float32)[None, :]
        if seq_len > 1
        else torch.zeros((1, 1), device=cpu_dev, dtype=torch.float32)
    )  # (1, T)

    # Stage-dependent maximum angle envelopes (standard oblique extended to ±70° in stage 4)
    max_yaw_std = {1: 26.0, 2: 50.0, 3: 64.0, 4: 70.0}[stage]
    max_pitch_std = {1: 16.0, 2: 32.0, 3: 44.0, 4: 50.0}[stage]
    max_roll_std = {1: 4.5, 2: 8.5, 3: 11.5, 4: 13.5}[stage]

    # Sample start & end angles per sequence on CPU (B*T ~ 384 scalars -> <0.25ms, zero GPU launch overhead!)
    u_rand = torch.rand((batch_size, 16), generator=gen, device=cpu_dev)

    yaw_0 = (u_rand[:, 0] * 2.0 - 1.0) * max_yaw_std
    yaw_delta = (u_rand[:, 1] * 2.0 - 1.0) * (8.0 + 4.0 * stage)
    pitch_0 = (u_rand[:, 2] * 1.35 - 0.35) * max_pitch_std
    pitch_delta = (u_rand[:, 3] * 2.0 - 1.0) * (5.0 + 2.0 * stage)
    roll_0 = (u_rand[:, 4] * 2.0 - 1.0) * max_roll_std
    roll_delta = (u_rand[:, 5] * 2.0 - 1.0) * 3.0

    # Override angle trajectories for specialized geometric regimes!
    is_steep = reg_ids_cpu == REGIME_TO_ID["steep_oblique"]
    is_turn = reg_ids_cpu == REGIME_TO_ID["turn_in_sweep"]

    # Steep oblique: yaw up to ±72° (or overhead pitch up to 55° + roll ±14°)
    steep_sign = torch.where(u_rand[:, 6] < 0.55, -1.0, 1.0)
    steep_yaw = steep_sign * (52.0 + u_rand[:, 7] * 20.0)
    steep_pitch = 22.0 + u_rand[:, 8] * 33.0
    steep_roll = (u_rand[:, 9] * 2.0 - 1.0) * 14.0
    yaw_0 = torch.where(is_steep, steep_yaw, yaw_0)
    yaw_delta = torch.where(is_steep, -steep_sign * (6.0 + u_rand[:, 1] * 8.0), yaw_delta)
    pitch_0 = torch.where(is_steep, steep_pitch, pitch_0)
    roll_0 = torch.where(is_steep, steep_roll, roll_0)

    # Turn-in sweep: Δyaw = 25°..50° across T frames + brake nose-dive pitch pulse!
    turn_sign = torch.where(u_rand[:, 6] < 0.5, -1.0, 1.0)
    turn_start = turn_sign * (12.0 + u_rand[:, 7] * 18.0)
    turn_dyaw = turn_sign * (25.0 + u_rand[:, 8] * 25.0)
    yaw_0 = torch.where(is_turn, turn_start, yaw_0)
    yaw_delta = torch.where(is_turn, turn_dyaw, yaw_delta)

    yaw_deg = (yaw_0[:, None] + yaw_delta[:, None] * t_norm).clamp(-72.0, 72.0)
    brake_dive = is_turn[:, None].float() * (6.5 * torch.sin(math.pi * t_norm))
    pitch_deg = (pitch_0[:, None] + pitch_delta[:, None] * t_norm + brake_dive).clamp(-28.0, 55.0)
    roll_deg = (roll_0[:, None] + roll_delta[:, None] * t_norm).clamp(-14.0, 14.0)

    stay_tilt = (u_rand[:, 10] < 0.20).float() * (5.0 + u_rand[:, 11] * 6.0)
    pitch_eff_deg = (pitch_deg - stay_tilt[:, None]).clamp(-32.0, 55.0)

    # 2. Target sensor plate width scale:
    #    - 38% persistent `distant_lowres_sr` (LogUniform(12.0, 26.0) -> [12.0, 28.0] across ALL T frames
    #      with physically consistent smooth sub-pixel shift 0.15..0.85 source-px/frame sweeping [0, 1)^2!)
    #    - 24% `approach_far_to_near` (13..22 px -> 78..138 px)
    #    - 38% `standard_multiscale` (28..130 px)
    min_w_px = {1: 56.0, 2: 38.0, 3: 28.0, 4: 28.0}[stage]
    max_w_px = 130.0
    w_start = min_w_px + u_rand[:, 12] * (max_w_px - min_w_px)
    approaching = torch.where(u_rand[:, 13] < 0.65, 1.0, -1.0)
    w_end = (w_start * (1.0 + approaching * (0.14 + 0.24 * u_rand[:, 14]))).clamp(min_w_px, max_w_px)
    w_target_px = w_start[:, None] + (w_end - w_start)[:, None] * t_norm

    if lowres_sr_prob is not None:
        eff_lr_p = float(lowres_sr_prob) if seq_len >= 2 else 0.0
    elif approach_prob is not None and float(approach_prob) <= 0.0:
        eff_lr_p = 0.0
    elif stage >= 3 and regime_override is None and seq_len >= 2:
        eff_lr_p = 0.38
    elif stage == 2 and regime_override is None and seq_len >= 2:
        eff_lr_p = 0.24
    else:
        eff_lr_p = 0.0

    if approach_prob is None:
        eff_app_p = (0.24 if stage >= 3 else 0.22) if (stage >= 2 and regime_override is None and seq_len >= 4) else 0.0
    else:
        eff_app_p = float(approach_prob) if seq_len >= 3 else 0.0
    eff_app_p = min(eff_app_p, max(0.0, 1.0 - eff_lr_p))

    u_app = torch.rand((batch_size, 4), generator=gen, device=cpu_dev)
    u_lr = torch.rand((batch_size, 8), generator=gen, device=cpu_dev)
    can_scale_spec = (
        (reg_ids_cpu != REGIME_TO_ID["negative_garbage"])
        & (reg_ids_cpu != REGIME_TO_ID["multicam_rgb_ir"])
        & (reg_ids_cpu != REGIME_TO_ID["track_mixed"])
    )
    can_lr_sr = reg_ids_cpu != REGIME_TO_ID["negative_garbage"]
    is_lowres_sr_cpu = can_lr_sr & (u_app[:, 0] < eff_lr_p)
    is_approach_cpu = can_scale_spec & (~is_lowres_sr_cpu) & (u_app[:, 0] < (eff_lr_p + eff_app_p))

    # Persistent 12..26px low-res super-resolution trajectory
    w_lr_start = 12.0 * torch.pow(26.0 / 12.0, u_lr[:, 0:1])
    w_lr_end = (w_lr_start * (0.94 + 0.16 * u_lr[:, 1:2])).clamp(12.0, 28.0)
    w_lr_px = w_lr_start + (w_lr_end - w_lr_start) * t_norm

    # Far-to-near approach trajectory (13..22px -> 78..138px)
    w_far0 = (13.0 + 6.0 * u_app[:, 1])[:, None]
    w_far1 = (w_far0 + (1.5 + 2.5 * u_app[:, 2])[:, None]).clamp_max(22.0)
    cos_end = torch.cos(yaw_deg[:, -1:] * (math.pi / 180.0)).clamp_min(0.58)
    w_near = ((78.0 + 52.0 * u_app[:, 3])[:, None] / cos_end).clamp(78.0, 138.0)
    use_smooth_ramp = (u_app[:, 2:3] < 0.72)
    s_smooth = torch.pow(t_norm.clamp(0.0, 1.0), 1.48)
    w_app_smooth = w_far0 * torch.pow((w_near / w_far0).clamp_min(1.2), s_smooth)
    tau_far = (t_norm / 0.70).clamp(0.0, 1.0)
    tau_near = ((t_norm - 0.70) / 0.30).clamp(0.0, 1.0)
    r_near = tau_near * (0.35 + 0.65 * tau_near)
    w_app_two_stage = torch.where(
        t_norm < 0.70, w_far0 + (w_far1 - w_far0) * tau_far, w_far1 + (w_near - w_far1) * r_near
    )
    w_app_px = torch.where(use_smooth_ramp, w_app_smooth, w_app_two_stage)
    w_target_px = torch.where(
        is_lowres_sr_cpu[:, None],
        w_lr_px,
        torch.where(is_approach_cpu[:, None], w_app_px, w_target_px),
    )

    # Keep persistent 12..26px low-res SR trajectories within physically recoverable viewing angles
    # (|yaw| <= 42 deg, pitch in [-20, 36] deg) so multi-frame sub-pixel SR has non-zero stroke SNR,
    # while standard & steep_oblique trajectories retain full +/-70..72 deg oblique envelope!
    yaw_deg = torch.where(is_lowres_sr_cpu[:, None], yaw_deg.clamp(-42.0, 42.0), yaw_deg)
    pitch_eff_deg = torch.where(is_lowres_sr_cpu[:, None], pitch_eff_deg.clamp(-20.0, 36.0), pitch_eff_deg)

    fx = torch.full((batch_size, seq_len), 1400.0, device=cpu_dev, dtype=torch.float32)
    fy = fx
    cx_cam = torch.full_like(fx, 960.0)
    cy_cam = torch.full_like(fx, 540.0)
    depth_mm = (fx * 330.0 / w_target_px.clamp_min(11.5)).clamp(2400.0, 42000.0)

    tx0_mm = (u_rand[:, 14] * 2.0 - 1.0) * 650.0
    ty0_mm = (u_rand[:, 15] * 2.0 - 1.0) * 320.0
    vx_mm = (torch.rand((batch_size,), generator=gen, device=cpu_dev) * 2.0 - 1.0) * (180.0 + 120.0 * stage)
    vy_mm = (torch.rand((batch_size,), generator=gen, device=cpu_dev) * 2.0 - 1.0) * (70.0 + 40.0 * stage)
    tx_std_mm = tx0_mm[:, None] + vx_mm[:, None] * t_norm
    ty_std_mm = ty0_mm[:, None] + vy_mm[:, None] * t_norm

    # Physically consistent sub-pixel inter-frame trajectory (0.15..0.85 source-px/frame across [0, 1)^2)
    # Smooth constant velocity + gentle quadratic lane-curve drift + <= 0.045px camera vibration so
    # multi-frame super-resolution across T=16..24 frames is genuinely solvable from sub-pixel phase diversity!
    spd_pf = 0.15 + 0.70 * u_lr[:, 2:3]
    ang_pf = (0.15 + 0.70 * u_lr[:, 3:4]) * (0.5 * math.pi)
    sgn_x = torch.where(u_lr[:, 4:5] < 0.5, -1.0, 1.0)
    sgn_y = torch.where(u_lr[:, 5:6] < 0.5, -1.0, 1.0)
    t_steps = torch.arange(seq_len, device=cpu_dev, dtype=torch.float32)[None, :]
    t_quad = (t_norm - 0.5) ** 2
    phi_sub = u_lr[:, 0:1] * (2.0 * math.pi)
    curv_x = (u_lr[:, 6:7] * 2.0 - 1.0) * 0.28
    curv_y = (u_lr[:, 7:8] * 2.0 - 1.0) * 0.20
    du_sub_px = (
        u_lr[:, 4:5]
        + sgn_x * spd_pf * torch.cos(ang_pf) * t_steps
        + curv_x * t_quad
        + 0.045 * torch.sin(t_steps * 0.85 + phi_sub)
    )
    dv_sub_px = (
        u_lr[:, 5:6]
        + sgn_y * spd_pf * torch.sin(ang_pf) * t_steps
        + curv_y * t_quad
        + 0.040 * torch.cos(t_steps * 1.15 + phi_sub)
    )
    mm_per_px = depth_mm / fx
    tx_lr_mm = tx0_mm[:, None] * 0.30 + du_sub_px * mm_per_px
    ty_lr_mm = ty0_mm[:, None] * 0.30 + dv_sub_px * mm_per_px
    tx_mm = torch.where(is_lowres_sr_cpu[:, None], tx_lr_mm, tx_std_mm)
    ty_mm = torch.where(is_lowres_sr_cpu[:, None], ty_lr_mm, ty_std_mm)

    # 3. Multi-camera & Track-mixed frame scheduling
    fps = 10.0 + torch.rand((batch_size, 1), generator=gen, device=cpu_dev) * 20.0
    dt_frame = 1.0 / fps
    timestamps_s = torch.arange(seq_len, device=cpu_dev, dtype=torch.float32)[None, :] * dt_frame

    camera_ids = torch.zeros((batch_size, seq_len), device=cpu_dev, dtype=torch.int64)
    if primary_sides is None:
        sides_np = (rng.random(batch_size) < 0.48).astype(np.int64)
    elif isinstance(primary_sides, int):
        sides_np = np.full(batch_size, int(primary_sides), dtype=np.int64)
    elif isinstance(primary_sides, torch.Tensor):
        sides_np = primary_sides.detach().cpu().numpy().astype(np.int64)
    else:
        sides_np = np.asarray([int(s) for s in primary_sides], dtype=np.int64)
    plate_sides = torch.from_numpy(sides_np)[:, None].expand(batch_size, seq_len).clone()

    is_multicam_np = reg_ids_np == REGIME_TO_ID["multicam_rgb_ir"]
    mc_fr_np = is_multicam_np & (rng.random(batch_size) < 0.45)
    is_multicam = reg_ids_cpu == REGIME_TO_ID["multicam_rgb_ir"]
    mc_front_rear_cpu = torch.from_numpy(mc_fr_np)
    half_t = max(1, seq_len // 2)
    t_arange = torch.arange(seq_len, device=cpu_dev)[None, :]
    if seq_len >= 2:
        sides_np = np.where(mc_fr_np, 1, sides_np)
        cam1_mask = is_multicam[:, None] & (t_arange >= half_t)
        camera_ids = torch.where(cam1_mask, torch.ones_like(camera_ids), camera_ids)
        fr_cam0 = mc_front_rear_cpu[:, None] & (~cam1_mask)
        fr_cam1 = mc_front_rear_cpu[:, None] & cam1_mask
        plate_sides = torch.where(fr_cam0, torch.zeros_like(plate_sides), plate_sides)
        plate_sides = torch.where(fr_cam1, torch.ones_like(plate_sides), plate_sides)
        yaw_deg = torch.where(cam1_mask, -yaw_deg * 0.75 + 18.0, yaw_deg)
        timestamps_s = torch.where(fr_cam1, timestamps_s + 1.8, timestamps_s)
    primary_sides_cpu = [int(x) for x in sides_np]

    ir_mode = torch.zeros((batch_size, seq_len), device=cpu_dev, dtype=torch.int64)
    ir_mode = torch.where((reg_ids_cpu == REGIME_TO_ID["night_ir_850"])[:, None], torch.ones_like(ir_mode), ir_mode)
    ir_mode = torch.where((reg_ids_cpu == REGIME_TO_ID["night_ir_940"])[:, None], torch.full_like(ir_mode, 2), ir_mode)
    if seq_len >= 2:
        mc_ir_first = is_multicam[:, None] & (t_arange < half_t)
        mc_ir_type = torch.where(
            (torch.rand((batch_size, 1), generator=gen, device=cpu_dev) < 0.65),
            torch.ones_like(ir_mode),
            torch.full_like(ir_mode, 2),
        )
        ir_mode = torch.where(mc_ir_first, mc_ir_type, ir_mode)

    night_regimes = (
        (reg_ids_cpu == REGIME_TO_ID["night_rgb_glare"])
        | (reg_ids_cpu == REGIME_TO_ID["night_ir_850"])
        | (reg_ids_cpu == REGIME_TO_ID["night_ir_940"])
        | (reg_ids_cpu == REGIME_TO_ID["illuminated_night"])
        | is_multicam
    )
    is_night = night_regimes[:, None].expand(batch_size, seq_len)

    splice_mask = torch.zeros((batch_size, seq_len), device=cpu_dev, dtype=torch.bool)
    is_mixed = reg_ids_cpu == REGIME_TO_ID["track_mixed"]
    if seq_len >= 3:
        n_splice = min(max(2, seq_len // 3), 4, seq_len - 1)
        splice_mask = is_mixed[:, None] & (t_arange >= (seq_len - n_splice))

    # 4. Build rotation matrices R (B, T, 3, 3) & project 4 plate corners [TL, TR, BR, BL]
    R_p2c = euler_zyx_to_rotmat(
        yaw_deg * (math.pi / 180.0),
        pitch_eff_deg * (math.pi / 180.0),
        roll_deg * (math.pi / 180.0),
    )
    R_c2p = R_p2c.transpose(-1, -2).contiguous()

    t_vec = torch.stack([tx_mm, ty_mm, depth_mm], dim=-1)
    C_local = -torch.matmul(R_c2p, t_vec.unsqueeze(-1)).squeeze(-1)
    cz_height = (-C_local[..., 2]).clamp_min(200.0)
    parallax_tan = torch.stack([C_local[..., 0] / cz_height, C_local[..., 1] / cz_height], dim=-1)

    c_for_dir = torch.stack([C_local[..., 0], C_local[..., 1], cz_height], dim=-1)
    view_dir_local = F.normalize(c_for_dir, dim=-1)

    plate_corners_3d, _, _ = _get_homog_dev_consts(cpu_dev)
    p_cam = torch.matmul(plate_corners_3d[None, None, :, :], R_p2c.transpose(-1, -2)) + t_vec[:, :, None, :]
    z_c = p_cam[..., 2].clamp_min(100.0)
    u_full = fx[:, :, None] * (p_cam[..., 0] / z_c) + cx_cam[:, :, None]
    v_full = fy[:, :, None] * (p_cam[..., 1] / z_c) + cy_cam[:, :, None]
    corners_full_gt = torch.stack([u_full, v_full], dim=-1)

    quad_center = 0.5 * (corners_full_gt.amin(dim=-2) + corners_full_gt.amax(dim=-2))
    rx0 = torch.round(quad_center[..., 0] - 0.5 * float(SENSOR_ROI_W))
    ry0 = torch.round(quad_center[..., 1] - 0.5 * float(SENSOR_ROI_H))
    roi_origin = torch.stack([rx0, ry0], dim=-1)

    corners_roi_raw = corners_full_gt - roi_origin[:, :, None, :]

    is_ir = ir_mode > 0
    shutter_frac = torch.where(
        is_ir,
        torch.full((batch_size, seq_len), 0.04, device=cpu_dev),
        torch.where(
            is_night
            | (reg_ids_cpu == REGIME_TO_ID["heavy_rain_droplets"])[:, None]
            | (reg_ids_cpu == REGIME_TO_ID["dusk_wdr"])[:, None],
            0.22 + 0.35 * torch.rand((batch_size, seq_len), generator=gen, device=cpu_dev),
            0.06 + 0.14 * torch.rand((batch_size, seq_len), generator=gen, device=cpu_dev),
        ),
    )
    if seq_len >= 2:
        vel_px = torch.zeros((batch_size, seq_len, 2), device=cpu_dev, dtype=torch.float32)
        vel_px[:, 1:, :] = quad_center[:, 1:, :] - quad_center[:, :-1, :]
        vel_px[:, 0, :] = vel_px[:, 1, :]
    else:
        vel_px = (torch.rand((batch_size, 1, 2), generator=gen, device=cpu_dev) * 2.0 - 1.0) * 6.0

    motion_vec_px = (vel_px * shutter_frac.unsqueeze(-1)).clamp(-9.0, 9.0)
    if stage >= 2:
        # Simulate 1/15s..1/30s long-shutter directional smear on ~24% of RGB sequences & far approach phase
        # (Keep persistent `distant_lowres_sr` on its genuine sub-pixel exposure blur so phase shifts survive!)
        u_shut = torch.rand((batch_size, 3), generator=gen, device=cpu_dev)
        ang_sh = u_shut[:, 1] * (2.0 * math.pi)
        dir_sh = torch.stack([torch.cos(ang_sh), 0.45 * torch.sin(ang_sh)], dim=-1)[:, None, :]
        has_long_sh = (~is_ir) & (~is_lowres_sr_cpu[:, None]) & (
            (is_approach_cpu[:, None] & (t_norm < 0.70))
            | ((~is_approach_cpu[:, None]) & (u_shut[:, 0:1] < 0.24))
        )
        mag_sh = (1.8 + 3.4 * torch.rand((batch_size, seq_len, 1), generator=gen, device=cpu_dev))
        near_sharp = is_approach_cpu[:, None] & (t_norm >= 0.74)
        motion_vec_px = torch.where(
            is_lowres_sr_cpu[:, None, None],
            motion_vec_px.clamp(-1.0, 1.0),
            torch.where(
                near_sharp[:, :, None],
                motion_vec_px * 0.28,
                torch.where(has_long_sh[:, :, None], (motion_vec_px * 0.45 + dir_sh * mag_sh).clamp(-9.0, 9.0), motion_vec_px),
            ),
        )
    rs_gain = 0.06 + 0.12 * torch.rand((batch_size, seq_len, 1), generator=gen, device=cpu_dev)
    rolling_shutter_px = (vel_px * rs_gain).clamp(-4.5, 4.5)

    v_rel_corner = corners_roi_raw[..., 1] / float(SENSOR_ROI_H) - 0.5
    corners_roi_gt = torch.stack(
        [
            corners_roi_raw[..., 0] + rolling_shutter_px[..., 0:1] * v_rel_corner,
            corners_roi_raw[..., 1] + rolling_shutter_px[..., 1:2] * v_rel_corner,
        ],
        dim=-1,
    )
    corners_full_sheared = corners_roi_gt + roi_origin[:, :, None, :]
    intrinsics = torch.stack([fx, fy, cx_cam, cy_cam], dim=-1)

    if dev.type != "cpu":
        # Transfer all trajectory tensors to CUDA in 2 tiny contiguous copies (~72 KB total)!
        f_pack = torch.cat(
            [
                R_c2p.reshape(batch_size, seq_len, 9),
                C_local,
                view_dir_local,
                parallax_tan,
                intrinsics,
                roi_origin,
                corners_full_sheared.reshape(batch_size, seq_len, 8),
                corners_roi_gt.reshape(batch_size, seq_len, 8),
                motion_vec_px,
                rolling_shutter_px,
                timestamps_s.unsqueeze(-1),
                yaw_deg.unsqueeze(-1),
                pitch_eff_deg.unsqueeze(-1),
                roll_deg.unsqueeze(-1),
                w_target_px.unsqueeze(-1),
            ],
            dim=-1,
        ).to(dev)
        R_c2p = f_pack[..., 0:9].reshape(batch_size, seq_len, 3, 3)
        C_local = f_pack[..., 9:12]
        view_dir_local = f_pack[..., 12:15]
        parallax_tan = f_pack[..., 15:17]
        intrinsics = f_pack[..., 17:21]
        roi_origin = f_pack[..., 21:23]
        corners_full_sheared = f_pack[..., 23:31].reshape(batch_size, seq_len, 4, 2)
        corners_roi_gt = f_pack[..., 31:39].reshape(batch_size, seq_len, 4, 2)
        motion_vec_px = f_pack[..., 39:41]
        rolling_shutter_px = f_pack[..., 41:43]
        timestamps_s = f_pack[..., 43]
        yaw_deg = f_pack[..., 44]
        pitch_eff_deg = f_pack[..., 45]
        roll_deg = f_pack[..., 46]
        w_target_out = f_pack[..., 47]

        i_pack = torch.stack(
            [camera_ids, plate_sides, ir_mode, is_night.long(), splice_mask.long()],
            dim=-1,
        ).to(dev)
        camera_ids = i_pack[..., 0]
        plate_sides = i_pack[..., 1]
        ir_mode = i_pack[..., 2]
        is_night = i_pack[..., 3].bool()
        splice_mask = i_pack[..., 4].bool()

        b_pack = torch.stack(
            [reg_ids_cpu, mc_front_rear_cpu.long(), is_approach_cpu.long(), is_lowres_sr_cpu.long()],
            dim=-1,
        ).to(dev)
        reg_ids = b_pack[:, 0]
        mc_front_rear = b_pack[:, 1].bool()
        is_approach = b_pack[:, 2].bool()
        is_lowres_sr = b_pack[:, 3].bool()
    else:
        reg_ids = reg_ids_cpu
        mc_front_rear = mc_front_rear_cpu
        is_approach = is_approach_cpu
        is_lowres_sr = is_lowres_sr_cpu
        w_target_out = w_target_px

    return TrajectoryBatch(
        R_c2p=R_c2p,
        C_local_mm=C_local,
        view_dir_local=view_dir_local,
        parallax_tan_xy=parallax_tan,
        intrinsics=intrinsics,
        roi_origin_xy=roi_origin,
        corners_full_gt=corners_full_sheared,
        corners_roi_gt=corners_roi_gt,
        motion_vec_px=motion_vec_px,
        rolling_shutter_px=rolling_shutter_px,
        timestamps_s=timestamps_s,
        camera_ids=camera_ids,
        plate_sides=plate_sides,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_eff_deg,
        roll_deg=roll_deg,
        sensor_ir_mode=ir_mode,
        is_night=is_night,
        splice_second_plate=splice_mask,
        multicam_front_rear=mc_front_rear,
        regime_ids=reg_ids,
        regime_tags=tags,
        primary_sides_cpu=primary_sides_cpu,
        is_approach=is_approach,
        is_lowres_sr=is_lowres_sr,
        w_target_px=w_target_out,
    )


def _project_ray_grid_impl(
    R_c2p: torch.Tensor,
    C_loc: torch.Tensor,
    intr: torch.Tensor,
    rx0_ry0: torch.Tensor,
    rs_xy: torch.Tensor,
    u_1d: torch.Tensor,
    v_1d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_fx = 1.0 / intr[:, 0]
    inv_fy = 1.0 / intr[:, 1]
    cx = intr[:, 2]
    cy = intr[:, 3]

    u_base = (rx0_ry0[:, 0] + 0.5 * rs_xy[:, 0] - cx) * inv_fx
    v_base = (rx0_ry0[:, 1] + 0.5 * rs_xy[:, 1] - cy) * inv_fy

    du_cam = torch.stack([inv_fx, torch.zeros_like(inv_fx), torch.zeros_like(inv_fx)], dim=-1)
    dv_cam = torch.stack(
        [
            -(rs_xy[:, 0] / float(SENSOR_ROI_H)) * inv_fx,
            (1.0 - rs_xy[:, 1] / float(SENSOR_ROI_H)) * inv_fy,
            torch.zeros_like(inv_fx),
        ],
        dim=-1,
    )
    d0_cam = torch.stack([u_base, v_base, torch.ones_like(inv_fx)], dim=-1)

    d0 = torch.bmm(R_c2p, d0_cam.unsqueeze(-1)).squeeze(-1)
    du = torch.bmm(R_c2p, du_cam.unsqueeze(-1)).squeeze(-1)
    dv = torch.bmm(R_c2p, dv_cam.unsqueeze(-1)).squeeze(-1)

    dx = d0[:, 0, None, None] + du[:, 0, None, None] * u_1d[None, None, :] + dv[:, 0, None, None] * v_1d[None, :, None]
    dy = d0[:, 1, None, None] + du[:, 1, None, None] * u_1d[None, None, :] + dv[:, 1, None, None] * v_1d[None, :, None]
    dz = d0[:, 2, None, None] + du[:, 2, None, None] * u_1d[None, None, :] + dv[:, 2, None, None] * v_1d[None, :, None]

    s = -C_loc[:, 2, None, None] / dz.clamp_min(1e-4)
    x0_mm = C_loc[:, 0, None, None] + s * dx
    y0_mm = C_loc[:, 1, None, None] + s * dy
    return x0_mm, y0_mm


_PROJ_RAY_CUDA_FN: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None
_WARP_GRID_CUDA_FN: Callable[..., torch.Tensor] | None = None


def project_plate_ray_grid_roi(traj: TrajectoryBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute base-plane `(X0_mm, Y0_mm)` `(B*T, 112, 176)` in plate mm (`[-165..165] x [-82.5..82.5]`)."""
    global _PROJ_RAY_CUDA_FN
    bsz, seq_len = traj.yaw_deg.shape
    n_bt = bsz * seq_len
    dev = traj.yaw_deg.device
    _, u_1d, v_1d = _get_homog_dev_consts(dev)
    args = (
        traj.R_c2p.reshape(n_bt, 3, 3),
        traj.C_local_mm.reshape(n_bt, 3),
        traj.intrinsics.reshape(n_bt, 4),
        traj.roi_origin_xy.reshape(n_bt, 2),
        traj.rolling_shutter_px.reshape(n_bt, 2),
        u_1d,
        v_1d,
    )
    if dev.type == "cuda":
        if _PROJ_RAY_CUDA_FN is None:
            _PROJ_RAY_CUDA_FN = torch.compile(_project_ray_grid_impl, dynamic=True)
        return _PROJ_RAY_CUDA_FN(*args)
    return _project_ray_grid_impl(*args)


def closed_form_rect_to_quad_coeffs(
    quad_xy: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Exact `O(1)` closed-form projective mapping from unit square `[0, 1]^2` to `quad_xy (..., 4, 2)`.

    Maps `(0,0)->TL`, `(1,0)->TR`, `(1,1)->BR`, `(0,1)->BL` without `torch.linalg.solve`:
      `x_src = (a*u + b*v + c) / (g*u + h*v + 1)`
      `y_src = (d*u + e*v + f) / (g*u + h*v + 1)`
    """
    x0 = quad_xy[..., 0, 0]
    y0 = quad_xy[..., 0, 1]
    x1 = quad_xy[..., 1, 0]
    y1 = quad_xy[..., 1, 1]
    x2 = quad_xy[..., 2, 0]
    y2 = quad_xy[..., 2, 1]
    x3 = quad_xy[..., 3, 0]
    y3 = quad_xy[..., 3, 1]

    dx1 = x1 - x2
    dy1 = y1 - y2
    dx2 = x3 - x2
    dy2 = y3 - y2
    sx = x0 - x1 + x2 - x3
    sy = y0 - y1 + y2 - y3

    det = dx1 * dy2 - dx2 * dy1
    safe_det = torch.where(det.abs() < 1e-6, torch.where(det >= 0, 1e-6, -1e-6), det)

    g = (sx * dy2 - sy * dx2) / safe_det
    h = (dx1 * sy - dy1 * sx) / safe_det
    a = x1 - x0 + g * x1
    b = x3 - x0 + h * x3
    c = x0
    d = y1 - y0 + g * y1
    e = y3 - y0 + h * y3
    f = y0
    return a, b, c, d, e, f, g, h


def _warp_grid_from_quad_impl(
    quad_xy: torch.Tensor,
    mr: torch.Tensor,
    xd_1d: torch.Tensor,
    yd_1d: torch.Tensor,
    h_src: int,
    w_src: int,
) -> torch.Tensor:
    out_h = yd_1d.shape[1]
    out_w = xd_1d.shape[2]
    mx = mr * float(out_w)
    my = mr * float(out_h)
    x0_d = mx - 0.5
    x1_d = float(out_w) - 0.5 - mx
    y0_d = my - 0.5
    y1_d = float(out_h) - 0.5 - my

    u = (xd_1d - x0_d) / (x1_d - x0_d).clamp_min(1e-4)
    v = (yd_1d - y0_d) / (y1_d - y0_d).clamp_min(1e-4)

    a, b, c, d, e, f, g, h = closed_form_rect_to_quad_coeffs(quad_xy.float())
    a = a[:, None, None]
    b = b[:, None, None]
    c = c[:, None, None]
    d = d[:, None, None]
    e = e[:, None, None]
    f = f[:, None, None]
    g = g[:, None, None]
    h = h[:, None, None]

    denom = g * u + h * v + 1.0
    safe_den = torch.where(denom.abs() < 1e-5, torch.full_like(denom, 1e-5), denom)
    xs = (a * u + b * v + c) / safe_den
    ys = (d * u + e * v + f) / safe_den

    gn_x = (xs + 0.5) * (2.0 / float(w_src)) - 1.0
    gn_y = (ys + 0.5) * (2.0 / float(h_src)) - 1.0
    return torch.stack([gn_x, gn_y], dim=-1)


_OUT_ARANGE_CACHE: dict[tuple[torch.device, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_WARP_GRID_CUDA_FNS: dict[tuple[int, int, int, int], Callable[..., torch.Tensor]] = {}


def warp_quad_to_canonical_gpu(
    src_imgs: torch.Tensor,
    quad_xy: torch.Tensor,
    out_hw: tuple[int, int] = (80, 160),
    margin_ratio: float | torch.Tensor = 0.05,
) -> torch.Tensor:
    """Perspective-rectify `src_imgs` `(N, C, H_s, W_s)` from `quad_xy` `(N, 4, 2)` to `out_hw`."""
    n_imgs, _, h_src, w_src = src_imgs.shape
    out_h, out_w = out_hw
    dev = src_imgs.device

    if isinstance(margin_ratio, torch.Tensor):
        mr = margin_ratio.reshape(n_imgs, 1, 1).to(device=dev, dtype=torch.float32)
    else:
        mr = torch.full((n_imgs, 1, 1), float(margin_ratio), device=dev, dtype=torch.float32)

    key = (dev, out_h, out_w)
    xy_1d = _OUT_ARANGE_CACHE.get(key)
    if xy_1d is None:
        xd_1d = torch.arange(out_w, device=dev, dtype=torch.float32)[None, None, :]
        yd_1d = torch.arange(out_h, device=dev, dtype=torch.float32)[None, :, None]
        xy_1d = (xd_1d, yd_1d)
        _OUT_ARANGE_CACHE[key] = xy_1d
    xd_1d, yd_1d = xy_1d

    if dev.type == "cuda":
        fn_key = (h_src, w_src, out_h, out_w)
        fn = _WARP_GRID_CUDA_FNS.get(fn_key)
        if fn is None:
            fn = torch.compile(_warp_grid_from_quad_impl, dynamic=True)
            _WARP_GRID_CUDA_FNS[fn_key] = fn
        grid = fn(quad_xy, mr, xd_1d, yd_1d, h_src, w_src)
    else:
        grid = _warp_grid_from_quad_impl(quad_xy, mr, xd_1d, yd_1d, h_src, w_src)

    return F.grid_sample(
        src_imgs,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
