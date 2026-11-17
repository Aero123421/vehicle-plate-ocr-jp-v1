"""Unit tests for Part 2 Motion Generator and Arbitrary 360-Degree Multi-Camera Rig."""

import math
import pytest
import numpy as np

from vpocr_synth.motion.camera import PinholeCamera
from vpocr_synth.motion.generator import SequenceGenerator, compute_rear_plate_pose
from vpocr_synth.motion.models import (
    CameraParams,
    Pose3D,
    Quad2D,
    ScenarioType,
)
from vpocr_synth.motion.renderer import MotionRenderer
from vpocr_synth.motion.trajectory import (
    StraightApproachTrajectory,
    create_trajectory,
)
from vpocr_synth.plate.models import PlateConfig, PlateSize, PlateText, PlateType
from vpocr_synth.plate.renderer import PlateRenderer


def test_trajectory_continuous_evaluation():
    """Verify trajectory can be evaluated at arbitrary continuous time t and sub-frame delta."""
    traj = StraightApproachTrajectory(duration_s=3.0, seed=42)

    t0 = 1.0
    dt = 0.005
    p0 = traj.evaluate(t0)
    p1 = traj.evaluate(t0 + dt)

    assert isinstance(p0, Pose3D)
    assert p0.position.shape == (3,)
    assert p1.position.shape == (3,)
    assert p1.position[1] < p0.position[1]
    assert p0.speed_kmh > 0.0


def test_trajectory_all_scenarios():
    """Verify all 6 scenarios instantiate and evaluate smoothly."""
    duration = 3.0
    for scenario in ScenarioType:
        traj = create_trajectory(scenario, duration_s=duration, seed=123)
        p_start = traj.evaluate(0.0)
        p_mid = traj.evaluate(duration / 2.0)
        p_end = traj.evaluate(duration)

        assert p_start.position[2] > 0.2
        assert p_mid.position[2] > 0.2
        assert p_end.position[2] > 0.2


def test_geometric_consistency_approaching_vehicle():
    """Verify that as vehicle approaches, distance decreases and projected plate width increases."""
    cam_params = CameraParams(
        width=1920,
        height=1080,
        position=np.array([1.0, 0.0, 2.5]),
        pitch_deg=18.0,
        yaw_deg=4.0,
    )
    cam = PinholeCamera(cam_params)

    pose_far = Pose3D(position=np.array([0.0, 16.0, 0.5]), yaw_deg=0.0)
    pose_near = Pose3D(position=np.array([0.0, 4.0, 0.5]), yaw_deg=0.0)

    quad_far = cam.project_plate_corners(pose_far, width_m=0.33, height_m=0.165)
    quad_near = cam.project_plate_corners(pose_near, width_m=0.33, height_m=0.165)

    dist_far = float(np.linalg.norm(pose_far.position - cam_params.position))
    dist_near = float(np.linalg.norm(pose_near.position - cam_params.position))

    assert dist_near < dist_far
    assert quad_near.width > quad_far.width * 2.5


def test_camera_visibility_calculation():
    """Verify visibility ratio calculation for on-screen, partially clipped, and off-screen quads."""
    cam_params = CameraParams(width=1920, height=1080)
    cam = PinholeCamera(cam_params)

    quad_inside = Quad2D(points=[(400.0, 300.0), (800.0, 300.0), (800.0, 500.0), (400.0, 500.0)])
    assert cam.compute_visibility(quad_inside) == 1.0

    quad_offscreen = Quad2D(points=[(-800.0, -500.0), (-400.0, -500.0), (-400.0, -300.0), (-800.0, -300.0)])
    assert cam.compute_visibility(quad_offscreen) == 0.0

    quad_partial = Quad2D(points=[(1820.0, 400.0), (2020.0, 400.0), (2020.0, 500.0), (1820.0, 500.0)])
    vis = cam.compute_visibility(quad_partial)
    assert 0.4 < vis < 0.6


def test_motion_sequence_determinism():
    """Verify identical seed produces bit-exact identical sequence metadata and frame count."""
    seq_gen = SequenceGenerator()

    seq1 = seq_gen.generate_sequence(seed=777, duration_s=0.8, fps=10.0)
    seq2 = seq_gen.generate_sequence(seed=777, duration_s=0.8, fps=10.0)

    assert seq1.num_frames == seq2.num_frames
    assert seq1.scenario == seq2.scenario
    assert seq1.plate_text.full_text == seq2.plate_text.full_text

    for f1, f2 in zip(seq1.frames, seq2.frames):
        assert f1.timestamp == f2.timestamp
        assert math.isclose(f1.plate_width_px, f2.plate_width_px, rel_tol=1e-5)
        assert math.isclose(f1.distance_m, f2.distance_m, rel_tol=1e-5)


def test_character_quads_inside_plate():
    """Verify that projected character quads are contained within the projected plate bounding box."""
    seq_gen = SequenceGenerator()
    seq = seq_gen.generate_sequence(seed=12, scenario=ScenarioType.STRAIGHT_APPROACH, duration_s=1.2, fps=10.0)

    visible_frames = [f for f in seq.frames if f.is_visible and f.plate_width_px > 40]
    assert len(visible_frames) > 0

    frame = visible_frames[len(visible_frames) // 2]
    p_box = frame.corners.to_bbox()

    for cq in frame.character_quads:
        c_box = cq.quad.to_bbox()
        assert c_box[0] >= p_box[0] - 5.0
        assert c_box[1] >= p_box[1] - 5.0
        assert c_box[2] <= p_box[2] + 5.0
        assert c_box[3] <= p_box[3] + 5.0


def test_incidence_angle_pure_side_and_backfacing_unreadable():
    """Verify incidence angle marks >76 deg extreme side and >=88 deg back-facing views unreadable."""
    pt = PlateText(region="品川", classification="500", hiragana="あ", serial=1234)
    canon = PlateRenderer().render(pt, PlateConfig(scale=1.0, seed=1))
    pose_front = Pose3D(position=np.array([0.0, 6.0, 0.55]), yaw_deg=0.0)  # Normal points [0, -1, 0]

    # 1. Pure side camera at X=7.0, Y=5.8 (~88.4 deg -> extreme side / back-facing)
    cp_side = PinholeCamera.aim_at(np.array([7.0, 5.5, 1.8]), pose_front.position, fov_h_deg=50.0, mount_role="pure_side")
    cam_side = PinholeCamera(cp_side)
    inc_side, _, _ = cam_side.compute_incidence_angle(pose_front)
    assert inc_side > 76.0
    _, ann_side = MotionRenderer(cam_side).render_frame(canon, pose_front, 0.0, 0, plate_side=2)
    assert ann_side.is_readable_view is False

    # 2. Back-facing camera behind vehicle at Y=16.5 looking toward vehicle center
    pose_rear = compute_rear_plate_pose(pose_front, vehicle_length_m=4.6)
    veh_center = 0.5 * (pose_front.position + pose_rear.position)
    cp_back = PinholeCamera.aim_at(np.array([0.0, 16.5, 2.2]), veh_center, fov_h_deg=50.0)
    cam_back = PinholeCamera(cp_back)
    inc_back, _, _ = cam_back.compute_incidence_angle(pose_front)
    assert inc_back >= 88.0
    _, ann_back = MotionRenderer(cam_back).render_frame(canon, pose_front, 0.0, 0)
    assert ann_back.is_visible is False
    assert ann_back.is_readable_view is False

    # 3. Rear plate of the same vehicle IS readable from that rear camera (and left-to-right non-mirrored!)
    inc_rear, _, _ = cam_back.compute_incidence_angle(pose_rear)
    assert inc_rear < 35.0
    _, ann_rear = MotionRenderer(cam_back).render_frame(canon, pose_rear, 0.0, 0, is_rear=True, plate_side=1)
    assert ann_rear.is_visible is True
    assert ann_rear.is_readable_view is True
    # Top-left x must be strictly left of Top-right x (not mirror-flipped!)
    assert ann_rear.corners.points[0][0] < ann_rear.corners.points[1][0]


def test_multicam_360_rig_generation():
    """Verify 4-camera 360-degree rig (front, oblique, pure_side, rear + seal) on a shared vehicle pass."""
    seq_gen = SequenceGenerator()
    pt = PlateText(region="横浜", classification="300", hiragana="さ", serial=7788)
    pc = PlateConfig(plate_type=PlateType.STANDARD_PRIVATE, plate_size=PlateSize.MEDIUM, with_seal=False, seed=3)

    mc_pass = seq_gen.generate_multicam_pass(
        seed=88,
        num_cameras=4,
        rig_preset="construction_4cam_360",
        duration_s=0.8,
        fps_override=8.0,
        resolution=(640, 360),
        apply_degradation=False,
        plate_text=pt,
        plate_config=pc,
    )

    assert len(mc_pass.streams) == 4
    assert mc_pass.front_plate_config.with_seal is False
    assert mc_pass.rear_plate_config.with_seal is True

    by_role = {s.mount_role: s for s in mc_pass.streams}
    assert set(by_role.keys()) == {"front", "oblique", "pure_side", "rear"}

    # Front and rear cameras have readable views
    assert any(f.is_readable_view for f in by_role["front"].clean_sequence.frames)
    assert any(f.is_readable_view for f in by_role["rear"].clean_sequence.frames)
    assert by_role["rear"].plate_side == 1

    # Pure side camera must NEVER mark plate readable
    assert by_role["pure_side"].plate_side == 2
    assert all(not f.is_readable_view for f in by_role["pure_side"].clean_sequence.frames)
