"""High-level single-camera and 360-degree multi-camera sequence generator."""

from dataclasses import replace
import math
from pathlib import Path
import random
from typing import Any
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vpocr_synth.motion.camera import PinholeCamera
from vpocr_synth.motion.models import (
    CameraParams,
    CameraViewStream,
    FrameAnnotation,
    MultiCameraPassResult,
    Pose3D,
    ScenarioType,
    SequenceResult,
)
from vpocr_synth.motion.renderer import MotionRenderer
from vpocr_synth.motion.trajectory import BaseTrajectory, create_trajectory
from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.models import PlateConfig, PlateSize, PlateText, PlateType
from vpocr_synth.plate.renderer import PlateRenderer
from vpocr_synth.plate.sampler import PlateSampler


def compute_rear_plate_pose(front_pose: Pose3D, vehicle_length_m: float = 4.6) -> Pose3D:
    """Compute 3D pose of the vehicle's opposite-end (rear) plate given its front plate pose."""
    _, _, n_front = PinholeCamera.compute_plate_axes(front_pose)
    # Vehicle body extends along -n_front by vehicle_length_m
    rear_pos = front_pose.position - vehicle_length_m * n_front
    return Pose3D(
        position=rear_pos,
        yaw_deg=front_pose.yaw_deg + 180.0,
        pitch_deg=-front_pose.pitch_deg,
        roll_deg=-front_pose.roll_deg,
        velocity=front_pose.velocity.copy(),
    )


class SequenceGenerator:
    """Generates continuous-pose vehicle motion sequences and 360-degree multi-camera gate passes."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config = self._load_config(config_path)
        self.plate_sampler = PlateSampler()
        self.plate_renderer = PlateRenderer()

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        if config_path is None:
            config_path = PlateDataLoader.get_data_dir() / "motion_config.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def generate_sequence(
        self,
        seed: int,
        scenario: ScenarioType | None = None,
        duration_s: float | None = None,
        fps: float | None = None,
        plate_text: PlateText | None = None,
        plate_config: PlateConfig | None = None,
    ) -> SequenceResult:
        """Generate a complete single-camera multi-frame vehicle approach/exit sequence."""
        rng = random.Random(seed)
        seq_id = f"seq_{seed:06d}"

        # 1. Sample Plate Text and Config if not provided
        if plate_text is None or plate_config is None:
            pt, pc = self.plate_sampler.sample(seed=seed)
            plate_text = plate_text or pt
            plate_config = plate_config or pc

        # 2. Sample Scenario
        if scenario is None:
            scenarios = [
                ScenarioType.STRAIGHT_APPROACH,
                ScenarioType.DECELERATE_AND_STOP,
                ScenarioType.STOP_AND_GO,
                ScenarioType.CURVING_ENTRY,
                ScenarioType.SWITCHBACK_REVERSING,
                ScenarioType.PASS_THROUGH_REAR,
            ]
            weights = [
                self.config.get("scenario_weights", {}).get("straight_approach", 0.35),
                self.config.get("scenario_weights", {}).get("decelerate_stop", 0.25),
                self.config.get("scenario_weights", {}).get("stop_and_go", 0.15),
                self.config.get("scenario_weights", {}).get("curving_entry", 0.15),
                self.config.get("scenario_weights", {}).get("switchback_reversing", 0.05),
                self.config.get("scenario_weights", {}).get("pass_through_rear", 0.05),
            ]
            scenario = rng.choices(scenarios, weights=weights, k=1)[0]

        is_rear = (scenario == ScenarioType.PASS_THROUGH_REAR)
        plate_side = 1 if is_rear else 0

        # Ensure rear standard plates carry seal and front plates do not when auto-sampled
        if is_rear and plate_config.plate_type in (PlateType.STANDARD_PRIVATE, PlateType.STANDARD_COMMERCIAL):
            render_cfg = replace(plate_config, with_seal=True)
        else:
            render_cfg = plate_config

        canonical_result = self.plate_renderer.render(plate_text, render_cfg)

        # 3. Sample Camera Extrinsics and Intrinsics
        cam_params = self._sample_camera_params(rng, is_rear)
        camera = PinholeCamera(cam_params)
        motion_renderer = MotionRenderer(camera)

        # 4. Timing Parameters
        if fps is None:
            fps_choices = self.config.get("timing", {}).get("fps_choices", [10, 15, 20, 25, 29.97, 30])
            fps_weights = self.config.get("timing", {}).get("fps_weights", [0.08, 0.15, 0.20, 0.22, 0.15, 0.20])
            fps = float(rng.choices(fps_choices, weights=fps_weights[:len(fps_choices)], k=1)[0])

        if duration_s is None:
            dur_cfg = self.config.get("timing", {}).get("duration_s", {"min": 1.0, "max": 8.0})
            duration_s = rng.uniform(dur_cfg.get("min", 1.0), dur_cfg.get("max", 8.0))

        jitter_std = self.config.get("timing", {}).get("jitter_std_s", 0.001)
        p_drop = self.config.get("timing", {}).get("p_frame_drop", 0.02)

        # 5. Instantiate 3D Trajectory Model
        trajectory = create_trajectory(scenario, duration_s, seed)

        # 6. Sample Vehicle Body Color & Length
        veh_colors = self.config.get("vehicle_colors", [
            [240, 240, 242], [35, 36, 38], [180, 182, 186], [220, 180, 40]
        ])
        raw_col = rng.choice(veh_colors)
        v_col: tuple[int, int, int] = (int(raw_col[0]), int(raw_col[1]), int(raw_col[2]))
        veh_len = 7.2 if render_cfg.plate_size == PlateSize.LARGE else (
            3.4 if render_cfg.plate_type in (PlateType.KEI_PRIVATE, PlateType.KEI_COMMERCIAL) else 4.6
        )

        # 7. Generate Sequence Frames
        total_nominal_frames = int(math.ceil(duration_s * fps))
        frames: list[FrameAnnotation] = []
        images: list[Image.Image] = []

        for k in range(total_nominal_frames):
            t_nom = k / fps
            if t_nom > duration_s:
                break

            if rng.random() < p_drop and 0 < k < total_nominal_frames - 1:
                continue

            t_actual = max(0.0, t_nom + rng.gauss(0.0, jitter_std))
            pose_k = trajectory.evaluate(t_actual)

            img_k, annot_k = motion_renderer.render_frame(
                canonical_result=canonical_result,
                pose=pose_k,
                timestamp=t_actual,
                frame_index=len(frames),
                vehicle_color=v_col,
                is_rear=is_rear,
                plate_side=plate_side,
                vehicle_length_m=veh_len,
                wind_seed=seed,
            )

            frames.append(annot_k)
            images.append(img_k)

        return SequenceResult(
            sequence_id=seq_id,
            scenario=scenario,
            plate_text=plate_text,
            plate_config=render_cfg,
            camera_params=cam_params,
            fps=fps,
            num_frames=len(frames),
            frames=frames,
            images=images,
            seed=seed,
            vehicle_color=v_col,
            camera_id=0,
            plate_side=plate_side,
            vehicle_length_m=veh_len,
            trajectory_seed=seed,
            trajectory_duration_s=duration_s,
            is_rear_offset_pose=False,
        )

    def generate_multicam_pass(
        self,
        seed: int,
        num_cameras: int = 3,
        rig_preset: str | None = None,
        scenario: ScenarioType | None = None,
        duration_s: float | None = None,
        fps_override: float | None = None,
        resolution: tuple[int, int] | None = None,
        apply_degradation: bool = True,
        plate_text: PlateText | None = None,
        plate_config: PlateConfig | None = None,
    ) -> MultiCameraPassResult:
        """Simulate 1 vehicle trajectory observed simultaneously by 1..4 cameras at arbitrary 360-deg mounts."""
        if not (1 <= num_cameras <= 4):
            raise ValueError(f"num_cameras must be 1..4, got {num_cameras}")

        rng = random.Random(seed)
        pass_id = f"mcpass_{seed:06d}"

        # 1. Sample shared vehicle PlateText & front/rear PlateConfigs
        if plate_text is None or plate_config is None:
            pt, pc = self.plate_sampler.sample(seed=seed)
            plate_text = plate_text or pt
            plate_config = plate_config or pc

        is_standard = plate_config.plate_type in (
            PlateType.STANDARD_PRIVATE,
            PlateType.STANDARD_COMMERCIAL,
            PlateType.GRAPHIC,
        )
        front_cfg = replace(plate_config, with_seal=False)
        rear_cfg = replace(plate_config, with_seal=is_standard)

        front_canonical = self.plate_renderer.render(plate_text, front_cfg)
        rear_canonical = self.plate_renderer.render(plate_text, rear_cfg)

        # 2. Sample shared vehicle trajectory
        if scenario is None:
            entry_scenarios = [
                ScenarioType.STRAIGHT_APPROACH,
                ScenarioType.DECELERATE_AND_STOP,
                ScenarioType.STOP_AND_GO,
                ScenarioType.CURVING_ENTRY,
                ScenarioType.SWITCHBACK_REVERSING,
            ]
            scenario = rng.choices(entry_scenarios, weights=[0.38, 0.28, 0.14, 0.14, 0.06], k=1)[0]

        if duration_s is None:
            duration_s = rng.uniform(1.8, 3.6)

        trajectory = create_trajectory(scenario, duration_s, seed)

        veh_colors = self.config.get("vehicle_colors", [
            [240, 240, 242], [35, 36, 38], [180, 182, 186], [220, 180, 40]
        ])
        raw_col = rng.choice(veh_colors)
        v_col: tuple[int, int, int] = (int(raw_col[0]), int(raw_col[1]), int(raw_col[2]))
        veh_len = 7.2 if front_cfg.plate_size == PlateSize.LARGE else (
            3.4 if front_cfg.plate_type in (PlateType.KEI_PRIVATE, PlateType.KEI_COMMERCIAL) else 4.6
        )

        # 3. Determine camera mount roles around 360-deg gate rig
        mc_cfg = self.config.get("multicam", {})
        presets = mc_cfg.get("rig_presets", {
            "gate_standard_3cam": ["front", "oblique", "rear"],
            "construction_4cam_360": ["front", "oblique", "pure_side", "rear"],
            "side_and_front_2cam": ["front", "pure_side"],
            "front_rear_2cam": ["front", "rear"],
        })

        if rig_preset and rig_preset in presets:
            base_roles = list(presets[rig_preset])
            chosen_preset = rig_preset
        else:
            chosen_preset = rig_preset or "random_360"
            role_pool = ["front", "oblique", "pure_side", "oblique_rear", "rear"]
            if num_cameras == 1:
                base_roles = [rng.choice(role_pool)]
            elif num_cameras == 2:
                base_roles = ["front", rng.choice(["oblique", "pure_side", "rear"])]
            elif num_cameras == 3:
                base_roles = ["front", rng.choice(["oblique", "pure_side"]), rng.choice(["rear", "oblique_rear"])]
            else:
                base_roles = ["front", "oblique", "pure_side", "rear"]

        while len(base_roles) < num_cameras:
            base_roles.append(rng.choice(["front", "oblique", "pure_side", "oblique_rear", "rear"]))
        roles = base_roles[:num_cameras]

        # Sample trajectory bounds to position cameras accurately around the pass
        sample_ts = [duration_s * (i / 8.0) for i in range(9)]
        base_poses = [trajectory.evaluate(ts) for ts in sample_ts]
        scenario_is_rear_traj = (scenario == ScenarioType.PASS_THROUGH_REAR)

        if scenario_is_rear_traj:
            rear_poses = base_poses
            front_poses = [compute_rear_plate_pose(p, veh_len) for p in base_poses]
        else:
            front_poses = base_poses
            rear_poses = [compute_rear_plate_pose(p, veh_len) for p in base_poses]

        cam_sim = None
        if apply_degradation:
            from vpocr_synth.camera.simulator import CameraSimulator
            cam_sim = CameraSimulator()

        fps_choices = self.config.get("timing", {}).get("fps_choices", [10, 15, 20, 25, 30])
        jitter_std = self.config.get("timing", {}).get("jitter_std_s", 0.001)
        p_drop = self.config.get("timing", {}).get("p_frame_drop", 0.02)

        streams: list[CameraViewStream] = []
        for cam_id, role in enumerate(roles):
            cam_rng = random.Random(seed * 997 + cam_id * 131 + 17)

            if role in ("rear", "oblique_rear"):
                plate_side = 1
            elif role == "pure_side":
                plate_side = 2
            else:
                plate_side = 0

            cam_params = self._sample_multicam_camera_params(
                rng=cam_rng,
                cam_id=cam_id,
                role=role,
                front_poses=front_poses,
                rear_poses=rear_poses,
                resolution=resolution,
            )
            cam_fps = float(fps_override) if fps_override is not None else float(cam_rng.choice(fps_choices))
            clock_offset = cam_params.clock_offset_s

            camera = PinholeCamera(cam_params)
            motion_renderer = MotionRenderer(camera)

            use_rear_plate = (plate_side == 1)
            canonical = rear_canonical if use_rear_plate else front_canonical
            is_offset_pose = (use_rear_plate and not scenario_is_rear_traj) or (
                (not use_rear_plate) and scenario_is_rear_traj
            )

            total_frames = max(2, int(math.ceil(duration_s * cam_fps)))
            frames: list[FrameAnnotation] = []
            images: list[Image.Image] = []

            for k in range(total_frames):
                t_local = k / cam_fps
                if t_local > duration_s:
                    break
                if cam_rng.random() < p_drop and 0 < k < total_frames - 1:
                    continue

                t_world = float(np.clip(t_local + clock_offset + cam_rng.gauss(0.0, jitter_std), 0.0, duration_s))
                raw_pose = trajectory.evaluate(t_world)
                pose_k = compute_rear_plate_pose(raw_pose, veh_len) if is_offset_pose else raw_pose

                img_k, annot_k = motion_renderer.render_frame(
                    canonical_result=canonical,
                    pose=pose_k,
                    timestamp=t_world,
                    frame_index=len(frames),
                    vehicle_color=v_col,
                    is_rear=use_rear_plate,
                    plate_side=plate_side,
                    vehicle_length_m=veh_len,
                    wind_seed=seed + cam_id * 31,
                )
                frames.append(annot_k)
                images.append(img_k)

            clean_seq = SequenceResult(
                sequence_id=f"{pass_id}_cam{cam_id}_{role}",
                scenario=scenario,
                plate_text=plate_text,
                plate_config=rear_cfg if use_rear_plate else front_cfg,
                camera_params=cam_params,
                fps=cam_fps,
                num_frames=len(frames),
                frames=frames,
                images=images,
                seed=seed + cam_id,
                vehicle_color=v_col,
                camera_id=cam_id,
                plate_side=plate_side,
                vehicle_length_m=veh_len,
                trajectory_seed=seed,
                trajectory_duration_s=duration_s,
                is_rear_offset_pose=is_offset_pose,
            )

            deg_seq = None
            cam_prof = None
            if cam_sim is not None:
                cam_prof = cam_sim.sample_random_profile(seed=seed * 313 + cam_id * 47, difficulty="mixed")
                deg_seq = cam_sim.degrade_sequence(clean_seq, cam_prof)

            streams.append(
                CameraViewStream(
                    camera_id=cam_id,
                    mount_role=role,
                    plate_side=plate_side,
                    clock_offset_s=clock_offset,
                    clean_sequence=clean_seq,
                    degraded_sequence=deg_seq,
                    camera_profile=cam_prof,
                )
            )

        return MultiCameraPassResult(
            pass_id=pass_id,
            scenario=scenario,
            plate_text=plate_text,
            front_plate_config=front_cfg,
            rear_plate_config=rear_cfg,
            vehicle_color=v_col,
            vehicle_length_m=veh_len,
            duration_s=duration_s,
            rig_preset=chosen_preset,
            streams=streams,
            seed=seed,
        )

    def _sample_multicam_camera_params(
        self,
        rng: random.Random,
        cam_id: int,
        role: str,
        front_poses: list[Pose3D],
        rear_poses: list[Pose3D],
        resolution: tuple[int, int] | None = None,
    ) -> CameraParams:
        """Sample arbitrary 360-deg construction gate camera mount for a given role."""
        mc_cfg = self.config.get("multicam", {})
        if resolution is not None:
            width, height = resolution
        else:
            res_list = self.config.get("intrinsics", {}).get("resolutions", [[1920, 1080], [1280, 720]])
            res = rng.choices(res_list[:2], weights=[0.7, 0.3], k=1)[0]
            width, height = int(res[0]), int(res[1])

        mid_idx = len(front_poses) // 2
        f_mid = front_poses[mid_idx]
        r_mid = rear_poses[mid_idx]
        u_f, _, n_f = PinholeCamera.compute_plate_axes(f_mid)
        u_r, _, n_r = PinholeCamera.compute_plate_axes(r_mid)

        side_sign = rng.choice([-1.0, 1.0])
        roll_deg = rng.uniform(
            float(mc_cfg.get("roll_deg", {}).get("min", -8.0)),
            float(mc_cfg.get("roll_deg", {}).get("max", 8.0)),
        )
        wind_shake = rng.uniform(
            float(mc_cfg.get("wind_shake_deg", {}).get("min", 0.02)),
            float(mc_cfg.get("wind_shake_deg", {}).get("max", 0.30)),
        )
        clock_off = rng.uniform(
            float(mc_cfg.get("clock_offset_s", {}).get("min", -0.15)),
            float(mc_cfg.get("clock_offset_s", {}).get("max", 0.15)),
        )

        if role == "front":
            # Heights from low 0.8m pipe up to 4.2m gantry, azimuth 0..25 deg ahead of front bumper
            cam_h = rng.uniform(0.8, 4.2)
            az_deg = side_sign * rng.uniform(0.0, 22.0)
            az_rad = math.radians(az_deg)
            # Place ahead of closest front pose so vehicle stays in front of camera
            f_near = min(front_poses, key=lambda p: float(np.dot(p.position, -n_f)))
            dist = rng.uniform(4.2, 7.5)
            horiz_dir = math.cos(az_rad) * n_f + math.sin(az_rad) * u_f
            cam_pos = np.array([
                f_near.position[0] + dist * horiz_dir[0],
                f_near.position[1] + dist * horiz_dir[1],
                cam_h,
            ], dtype=np.float64)
            target = f_mid.position.copy()
            fov_h = rng.uniform(42.0, 62.0)

        elif role == "oblique":
            # Includes high 2F site office prefab look-down up to 8.0m!
            cam_h = rng.uniform(1.6, 8.0) if rng.random() < 0.65 else rng.uniform(0.9, 3.5)
            az_deg = side_sign * rng.uniform(28.0, 58.0)
            az_rad = math.radians(az_deg)
            min_r = max(4.8, (cam_h - 0.5) * 1.15)
            dist = rng.uniform(min_r, min_r + 3.2)
            f_ref = front_poses[int(len(front_poses) * 0.65)]
            horiz_dir = math.cos(az_rad) * n_f + math.sin(az_rad) * u_f
            cam_pos = np.array([
                f_ref.position[0] + dist * horiz_dir[0],
                f_ref.position[1] + dist * horiz_dir[1],
                cam_h,
            ], dtype=np.float64)
            target = f_mid.position.copy()
            fov_h = rng.uniform(48.0, 72.0)

        elif role == "pure_side":
            # Pure side 70..110 deg azimuth: camera looks sideways across cabin; front plate is >76 deg / back-facing
            cam_h = rng.uniform(0.9, 4.5)
            az_deg = side_sign * rng.uniform(82.0, 106.0)
            az_rad = math.radians(az_deg)
            dist = rng.uniform(5.8, 9.0)
            # Reference near closest front point shifted into cabin so all trajectory poses sit >= 78 deg to front normal
            f_closest = front_poses[-1]
            cabin_anchor = f_closest.position - 1.8 * n_f
            horiz_dir = math.cos(az_rad) * n_f + math.sin(az_rad) * u_f
            cam_pos = np.array([
                cabin_anchor[0] + dist * horiz_dir[0],
                cabin_anchor[1] + dist * horiz_dir[1],
                cam_h,
            ], dtype=np.float64)
            target = (0.5 * (f_mid.position + r_mid.position)).copy()
            fov_h = rng.uniform(34.0, 48.0)

        elif role == "oblique_rear":
            # Oblique rear 115..155 deg (i.e. 25..65 deg off rear normal n_r)
            cam_h = rng.uniform(1.2, 6.8)
            off_rear_deg = side_sign * rng.uniform(26.0, 56.0)
            az_rad = math.radians(off_rear_deg)
            r_farthest = rear_poses[0]
            min_r = max(4.8, (cam_h - 0.5) * 1.15)
            dist = rng.uniform(min_r, min_r + 3.0)
            horiz_dir = math.cos(az_rad) * n_r + math.sin(az_rad) * u_r
            cam_pos = np.array([
                r_farthest.position[0] + dist * horiz_dir[0],
                r_farthest.position[1] + dist * horiz_dir[1],
                cam_h,
            ], dtype=np.float64)
            target = r_mid.position.copy()
            fov_h = rng.uniform(46.0, 68.0)

        else:  # "rear" (155..180 deg from front == 0..25 deg off rear normal n_r)
            cam_h = rng.uniform(0.9, 5.2)
            off_rear_deg = side_sign * rng.uniform(0.0, 22.0)
            az_rad = math.radians(off_rear_deg)
            r_farthest = rear_poses[0]
            min_r = max(4.2, (cam_h - 0.5) * 1.05)
            dist = rng.uniform(min_r, min_r + 3.2)
            horiz_dir = math.cos(az_rad) * n_r + math.sin(az_rad) * u_r
            cam_pos = np.array([
                r_farthest.position[0] + dist * horiz_dir[0],
                r_farthest.position[1] + dist * horiz_dir[1],
                cam_h,
            ], dtype=np.float64)
            target = r_mid.position.copy()
            fov_h = rng.uniform(40.0, 60.0)

        k1 = rng.uniform(-0.12, -0.02) if fov_h > 62.0 else rng.uniform(-0.03, 0.01)
        return PinholeCamera.aim_at(
            position=cam_pos,
            target=target,
            width=width,
            height=height,
            fov_h_deg=fov_h,
            roll_deg=roll_deg,
            k1=k1,
            k2=0.0,
            camera_id=cam_id,
            mount_role=role,
            wind_shake_deg=wind_shake,
            clock_offset_s=clock_off,
            pan_jitter_deg=rng.uniform(-1.5, 1.5),
            tilt_jitter_deg=rng.uniform(-1.2, 1.2),
        )

    def _sample_camera_params(self, rng: random.Random, is_rear: bool) -> CameraParams:
        """Sample realistic security/gate camera extrinsics and intrinsics."""
        cam_cfg = self.config.get("camera", {})
        int_cfg = self.config.get("intrinsics", {})

        res_list = int_cfg.get("resolutions", [[1920, 1080], [1280, 720]])
        res_wts = int_cfg.get("resolution_weights", [0.8, 0.2])
        res = rng.choices(res_list, weights=res_wts[:len(res_list)], k=1)[0]
        width, height = res[0], res[1]

        fov_min = float(int_cfg.get("fov_h_deg", {}).get("min", 30.0))
        fov_max = float(int_cfg.get("fov_h_deg", {}).get("max", 110.0))
        if rng.random() < 0.70:
            fov_h = rng.uniform(35.0, 65.0)
        else:
            fov_h = rng.uniform(fov_min, fov_max)

        h_min = cam_cfg.get("height_m", {}).get("min", 1.8)
        h_max = cam_cfg.get("height_m", {}).get("max", 4.2)
        cam_h = rng.uniform(h_min, h_max)

        lat_min = cam_cfg.get("lateral_offset_m", {}).get("min", -2.0)
        lat_max = cam_cfg.get("lateral_offset_m", {}).get("max", 2.0)
        cam_x = rng.uniform(lat_min, lat_max)

        gate_dist = float(cam_cfg.get("gate_distance_m", 6.2))
        roll = rng.uniform(-1.5, 1.5)

        if fov_h > 80.0:
            k1 = rng.uniform(-0.25, -0.08)
            k2 = rng.uniform(0.01, 0.08)
        elif fov_h > 60.0:
            k1 = rng.uniform(-0.10, -0.02)
            k2 = rng.uniform(0.0, 0.03)
        else:
            k1 = rng.uniform(-0.03, 0.01)
            k2 = 0.0

        aim_y = max(5.8, gate_dist + (2.0 if fov_h < 42.0 else 1.2))
        return PinholeCamera.aim_at(
            position=np.array([cam_x, 0.0, cam_h], dtype=np.float64),
            target=np.array([0.0, aim_y, 0.55], dtype=np.float64),
            width=width,
            height=height,
            fov_h_deg=fov_h,
            roll_deg=roll,
            k1=k1,
            k2=k2,
            mount_role="rear" if is_rear else "front",
            pan_jitter_deg=rng.uniform(-2.8, 2.8),
            tilt_jitter_deg=rng.uniform(-2.2, 2.2),
        )

    @staticmethod
    def create_filmstrip(
        sequence: SequenceResult,
        max_samples: int = 8,
        thumb_width: int = 320,
    ) -> Image.Image:
        """Create a horizontal filmstrip contact sheet previewing key sequence frames."""
        total_frames = len(sequence.images)
        if total_frames <= max_samples:
            indices = list(range(total_frames))
        else:
            indices = [int(round(i * (total_frames - 1) / (max_samples - 1))) for i in range(max_samples)]

        sample_imgs = [sequence.images[i] for i in indices]
        aspect = sample_imgs[0].height / sample_imgs[0].width
        thumb_h = int(round(thumb_width * aspect))

        margin = 12
        header_h = 44
        strip_w = len(sample_imgs) * (thumb_width + margin) + margin
        strip_h = thumb_h + margin * 2 + header_h

        strip = Image.new("RGB", (strip_w, strip_h), (28, 30, 34))
        draw = ImageDraw.Draw(strip)

        font_title = PlateRenderer._get_font("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 19, 0)
        font_label = PlateRenderer._get_font("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 13, 0)

        title = (
            f"{sequence.sequence_id} | {sequence.scenario.value} | "
            f"FPS: {sequence.fps:.2f} | Text: {sequence.plate_text.full_text} ({sequence.num_frames} frames)"
        )
        draw.text((margin, 12), title, fill=(240, 242, 245), font=font_title)

        for col_idx, (frame_idx, img) in enumerate(zip(indices, sample_imgs)):
            thumb = img.resize((thumb_width, thumb_h), Image.Resampling.BILINEAR)
            x = margin + col_idx * (thumb_width + margin)
            y = header_h + margin
            strip.paste(thumb, (x, y))

            annot = sequence.frames[frame_idx]
            label = f"#{frame_idx} t={annot.timestamp:.2f}s W={annot.plate_width_px:.0f}px d={annot.distance_m:.1f}m"
            draw.text((x + 6, y + thumb_h - 22), label, fill=(255, 235, 80), font=font_label)

        return strip

    @staticmethod
    def create_multicam_filmstrip(
        multicam_pass: MultiCameraPassResult,
        max_samples: int = 6,
        thumb_width: int = 300,
    ) -> Image.Image:
        """Create a multi-row grid filmstrip showing all cameras in a MultiCameraPassResult."""
        num_cams = len(multicam_pass.streams)
        thumb_h = int(round(thumb_width * (9.0 / 16.0)))
        margin = 10
        header_h = 46
        row_header_h = 22
        strip_w = max_samples * (thumb_width + margin) + margin
        strip_h = header_h + num_cams * (thumb_h + row_header_h + margin) + margin

        strip = Image.new("RGB", (strip_w, strip_h), (22, 24, 28))
        draw = ImageDraw.Draw(strip)
        font_title = PlateRenderer._get_font("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 17, 0)
        font_sub = PlateRenderer._get_font("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 12, 0)

        header = (
            f"360° MultiCam Rig: {multicam_pass.pass_id} | {multicam_pass.rig_preset} | "
            f"Plate: {multicam_pass.plate_text.full_text} (RearSeal={multicam_pass.rear_plate_config.with_seal})"
        )
        draw.text((margin, 12), header, fill=(242, 245, 248), font=font_title)

        for r_idx, stream in enumerate(multicam_pass.streams):
            seq_imgs = stream.degraded_sequence.images if stream.degraded_sequence is not None else stream.clean_sequence.images
            annots = stream.clean_sequence.frames
            n_f = len(seq_imgs)
            if n_f <= max_samples:
                idxs = list(range(n_f))
            else:
                idxs = [int(round(i * (n_f - 1) / (max_samples - 1))) for i in range(max_samples)]

            y_row = header_h + r_idx * (thumb_h + row_header_h + margin)
            cam_p = stream.clean_sequence.camera_params
            illum_lbl = stream.camera_profile.illumination.value if stream.camera_profile else "clean"
            side_lbl = {0: "FRONT", 1: "REAR(+封印)" if multicam_pass.rear_plate_config.with_seal else "REAR", 2: "SIDE(unreadable)"}[stream.plate_side]
            row_lbl = (
                f"Cam#{stream.camera_id} [{stream.mount_role} | {side_lbl}] "
                f"H={cam_p.position[2]:.1f}m roll={cam_p.roll_deg:+.1f}° FPS={stream.clean_sequence.fps:.0f} illum={illum_lbl}"
            )
            draw.text((margin, y_row), row_lbl, fill=(130, 210, 255), font=font_sub)

            for c_idx, f_idx in enumerate(idxs):
                x = margin + c_idx * (thumb_width + margin)
                y_img = y_row + row_header_h
                thumb = seq_imgs[f_idx].resize((thumb_width, thumb_h), Image.Resampling.BILINEAR)
                strip.paste(thumb, (x, y_img))
                an = annots[f_idx]
                tag = f"t={an.timestamp:.2f}s inc={an.incidence_deg:.0f}° {'READ' if an.is_readable_view else 'UNREAD'}"
                tag_col = (120, 255, 140) if an.is_readable_view else (255, 120, 110)
                draw.text((x + 5, y_img + thumb_h - 18), tag, fill=tag_col, font=font_sub)

        return strip

    @staticmethod
    def export_animated_preview(
        sequence: SequenceResult,
        out_path: Path,
        max_width: int = 640,
    ) -> Path:
        """Export lightweight animated GIF preview of the sequence."""
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sample_w, sample_h = sequence.images[0].size
        ratio = max_width / float(sample_w)
        target_w = max_width
        target_h = int(round(sample_h * ratio))

        downscaled: list[Image.Image] = []
        for img in sequence.images:
            downscaled.append(img.resize((target_w, target_h), Image.Resampling.BILINEAR))

        frame_duration_ms = int(round(1000.0 / sequence.fps))

        gif_path = out_path.with_suffix(".gif")
        downscaled[0].save(
            gif_path,
            save_all=True,
            append_images=downscaled[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=True,
        )

        return gif_path
