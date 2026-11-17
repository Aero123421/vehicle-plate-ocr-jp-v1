"""Main Camera Simulator orchestrating motion blur, optics, sensor, compression, and counterfactuals."""

from dataclasses import replace
import math
from pathlib import Path
import random
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

from vpocr_synth.camera.compression import CompressionSimulator
from vpocr_synth.camera.illumination import IlluminationSimulator
from vpocr_synth.camera.models import (
    CameraProfile,
    DegradedSequenceResult,
    FrameDegradationLog,
    IlluminationType,
)
from vpocr_synth.camera.optics import OpticsSimulator
from vpocr_synth.camera.sensor import SensorSimulator
from vpocr_synth.motion.camera import PinholeCamera
from vpocr_synth.motion.generator import compute_rear_plate_pose
from vpocr_synth.motion.models import SequenceResult
from vpocr_synth.motion.renderer import MotionRenderer
from vpocr_synth.motion.trajectory import create_trajectory
from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.renderer import PlateRenderer


class CameraSimulator:
    """Full camera imaging pipeline simulator with multi-axis sampling and counterfactual pairs."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config = self._load_config(config_path)
        self.plate_renderer = PlateRenderer()

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        if config_path is None:
            config_path = PlateDataLoader.get_data_dir() / "camera_config.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def get_available_profiles(self) -> list[str]:
        """List profile IDs defined in camera_config.yaml."""
        return [p["id"] for p in self.config.get("profiles", [])]

    def create_profile_by_id(self, profile_id: str, seed: int = 0) -> CameraProfile:
        """Instantiate CameraProfile by ID with deterministic random values."""
        rng = random.Random(seed)
        for p in self.config.get("profiles", []):
            if p["id"] == profile_id:
                shutter_range = p.get("shutter_speed_s", [0.001, 0.004])
                shutter = rng.uniform(shutter_range[0], shutter_range[1])

                gain_range = p.get("gain_db", [0.0, 10.0])
                gain = rng.uniform(gain_range[0], gain_range[1])

                sharp_range = p.get("sharpening_amount", [0.4, 0.7])
                sharp = rng.uniform(sharp_range[0], sharp_range[1])

                q_i_range = p.get("jpeg_quality_i", [75, 85])
                q_i = rng.randint(q_i_range[0], q_i_range[1])

                q_p_range = p.get("jpeg_quality_p", [50, 65])
                q_p = rng.randint(q_p_range[0], q_p_range[1])

                illum_str = p.get("illumination", "day_clear")
                illum = IlluminationType(illum_str)

                return CameraProfile(
                    id=p["id"],
                    name=p.get("name", p["id"]),
                    illumination=illum,
                    shutter_speed_s=shutter,
                    gain_db=gain,
                    subframe_samples=p.get("subframe_samples", 5),
                    enable_sharpening=p.get("enable_sharpening", True),
                    sharpening_amount=sharp,
                    jpeg_quality_i=q_i,
                    jpeg_quality_p=q_p,
                    gop_size=p.get("gop_size", 15),
                    vignette_strength=p.get("vignette_strength", 0.2),
                    defocus_scale=p.get("defocus_scale", 0.5),
                    ir_plate_reflectance_boost=p.get("ir_plate_reflectance_boost", 1.0),
                    lens_dirt_prob=p.get("lens_dirt_prob", 0.0),
                    ev_bias=float(p.get("ev_bias", 0.0)),
                    wdr_strength=float(p.get("wdr_strength", 0.0)),
                    temporal_nr_strength=float(p.get("temporal_nr_strength", 0.0)),
                    substream_scale=float(p.get("substream_scale", 1.0)),
                    glare_mode=p.get("glare_mode", "none"),
                    glare_intensity=float(p.get("glare_intensity", 0.0)),
                    seed=seed,
                )

        return CameraProfile(
            id=profile_id,
            name="デフォルトCCTV",
            illumination=IlluminationType.DAY_CLEAR,
            seed=seed,
        )

    def sample_random_profile(self, seed: int, difficulty: str = "mixed") -> CameraProfile:
        """Sample a combinatorial CameraProfile across independent physical axes.

        Args:
            seed: Deterministic random seed.
            difficulty: One of {"easy", "medium", "hard", "extreme", "mixed"}.
        """
        rng = random.Random(seed)
        if difficulty == "mixed":
            tier = rng.choices(["easy", "medium", "hard", "extreme"], weights=[0.25, 0.35, 0.28, 0.12], k=1)[0]
        else:
            tier = difficulty if difficulty in ("easy", "medium", "hard", "extreme") else "medium"

        illum = rng.choices(
            [
                IlluminationType.DAY_CLEAR,
                IlluminationType.DAY_CLOUDY,
                IlluminationType.BACKLIGHT,
                IlluminationType.DUSK_DAWN,
                IlluminationType.NIGHT_HEADLIGHTS,
                IlluminationType.NIGHT_IR,
            ],
            weights=[0.28, 0.20, 0.16, 0.12, 0.12, 0.12],
            k=1,
        )[0]

        # Independent shutter speed (log-uniform from 1/2000s = 0.0005s to 1/15s = 0.0667s, decoupled from FPS!)
        if tier == "easy":
            log_s0, log_s1 = math.log(0.0005), math.log(0.003)
            gain_db = rng.uniform(0.0, 6.0)
            ev_bias = rng.uniform(-0.35, 0.35)
            defocus = rng.uniform(0.1, 0.4)
            tnr = rng.uniform(0.0, 0.12)
            q_i, q_p = rng.randint(80, 92), rng.randint(62, 78)
            sub_scale = 1.0
            dirt_p = 0.05
            glare_mode = "none"
            glare_int = 0.0
        elif tier == "medium":
            log_s0, log_s1 = math.log(0.001), math.log(0.012)
            gain_db = rng.uniform(2.0, 14.0)
            ev_bias = rng.uniform(-0.75, 0.75)
            defocus = rng.uniform(0.3, 0.75)
            tnr = rng.uniform(0.0, 0.35)
            q_i, q_p = rng.randint(68, 84), rng.randint(46, 64)
            sub_scale = rng.choice([1.0, 1.0, 0.75])
            dirt_p = 0.20
            glare_mode = rng.choices(["none", "random_sweep"], weights=[0.75, 0.25], k=1)[0]
            glare_int = rng.uniform(0.35, 0.65) if glare_mode != "none" else 0.0
        elif tier == "hard":
            log_s0, log_s1 = math.log(0.003), math.log(0.035)
            gain_db = rng.uniform(8.0, 24.0)
            ev_bias = rng.uniform(-1.25, 1.25)
            defocus = rng.uniform(0.5, 1.15)
            tnr = rng.uniform(0.15, 0.60)
            q_i, q_p = rng.randint(54, 74), rng.randint(32, 50)
            sub_scale = rng.choice([1.0, 0.75, 0.5])
            dirt_p = 0.45
            glare_mode = rng.choices(["none", "random_sweep", "complementary_rows"], weights=[0.45, 0.30, 0.25], k=1)[0]
            glare_int = rng.uniform(0.65, 0.98) if glare_mode != "none" else 0.0
        else:  # "extreme"
            log_s0, log_s1 = math.log(0.008), math.log(1.0 / 15.0)
            gain_db = rng.uniform(14.0, 28.0)
            ev_bias = rng.uniform(-1.5, 1.5)
            defocus = rng.uniform(0.8, 1.45)
            tnr = rng.uniform(0.30, 0.78)
            q_i, q_p = rng.randint(44, 64), rng.randint(24, 40)
            sub_scale = rng.choice([0.75, 0.5, 0.4])
            dirt_p = 0.65
            glare_mode = rng.choices(["random_sweep", "complementary_rows", "full_washout"], weights=[0.40, 0.45, 0.15], k=1)[0]
            glare_int = rng.uniform(0.85, 1.0)

        shutter_s = math.exp(rng.uniform(log_s0, log_s1))
        if illum in (IlluminationType.NIGHT_IR, IlluminationType.NIGHT_HEADLIGHTS):
            gain_db = max(gain_db, rng.uniform(12.0, 24.0))

        ae_lag = rng.uniform(0.06, 0.48)
        wb_shift = 0.0 if illum == IlluminationType.NIGHT_IR else rng.uniform(-0.85, 0.85)
        wdr = rng.uniform(0.0, 1.0)
        enable_sharp = rng.random() < 0.85
        sharp_amt = rng.uniform(0.15, 1.15) if enable_sharp else 0.0
        gop = rng.choice([8, 12, 15, 20, 30])
        vig = rng.uniform(0.08, 0.38)
        ir_boost = rng.uniform(1.5, 2.6) if illum == IlluminationType.NIGHT_IR else 1.0
        sub_samples = 5 if shutter_s > 0.015 else 3

        return CameraProfile(
            id=f"rand_{tier}_{seed:05d}",
            name=f"Combinatorial ({tier}/{illum.value})",
            illumination=illum,
            shutter_speed_s=shutter_s,
            gain_db=gain_db,
            subframe_samples=sub_samples,
            enable_sharpening=enable_sharp,
            sharpening_amount=sharp_amt,
            jpeg_quality_i=q_i,
            jpeg_quality_p=q_p,
            gop_size=gop,
            vignette_strength=vig,
            defocus_scale=defocus,
            ir_plate_reflectance_boost=ir_boost,
            lens_dirt_prob=dirt_p,
            seed=seed,
            ev_bias=ev_bias,
            ae_lag_alpha=ae_lag,
            wb_temp_shift=wb_shift,
            wdr_strength=wdr,
            temporal_nr_strength=tnr,
            substream_scale=sub_scale,
            glare_mode=glare_mode,
            glare_intensity=glare_int,
        )

    def generate_counterfactual_pair(
        self,
        clean_seq: SequenceResult,
        factor_name: str,
        low_val: Any,
        high_val: Any,
        seed: int = 0,
        base_profile: CameraProfile | None = None,
    ) -> tuple[DegradedSequenceResult, DegradedSequenceResult]:
        """Generate a controlled counterfactual ablation pair varying ONLY `factor_name` on identical motion."""
        baseline = (
            replace(base_profile, seed=seed)
            if base_profile is not None
            else CameraProfile(
                id="cf_base",
                name="Counterfactual Baseline",
                illumination=IlluminationType.DAY_CLEAR,
                shutter_speed_s=0.001,
                gain_db=2.0,
                subframe_samples=4,
                enable_sharpening=True,
                sharpening_amount=0.35,
                jpeg_quality_i=86,
                jpeg_quality_p=72,
                gop_size=15,
                vignette_strength=0.12,
                defocus_scale=0.25,
                lens_dirt_prob=0.0,
                seed=seed,
            )
        )
        if not hasattr(baseline, factor_name):
            raise ValueError(f"Unknown CameraProfile factor '{factor_name}'")

        prof_low = replace(
            baseline,
            id=f"cf_{factor_name}_low",
            name=f"CF {factor_name}={low_val}",
            **{factor_name: low_val},
        )
        prof_high = replace(
            baseline,
            id=f"cf_{factor_name}_high",
            name=f"CF {factor_name}={high_val}",
            **{factor_name: high_val},
        )
        deg_low = self.degrade_sequence(clean_seq, prof_low)
        deg_high = self.degrade_sequence(clean_seq, prof_high)
        return deg_low, deg_high

    def degrade_sequence(
        self,
        clean_sequence: SequenceResult,
        profile: CameraProfile,
    ) -> DegradedSequenceResult:
        """Apply complete physical camera degradation pipeline to a clean multi-frame sequence."""
        illum_sim = IlluminationSimulator(
            profile=profile,
            camera_params=clean_sequence.camera_params,
            is_illuminated_plate=clean_sequence.plate_config.is_illuminated,
        )
        optics_sim = OpticsSimulator(profile)
        sensor_sim = SensorSimulator(profile)
        comp_sim = CompressionSimulator(profile)

        traj_dur = (
            clean_sequence.trajectory_duration_s
            if clean_sequence.trajectory_duration_s is not None
            else (clean_sequence.frames[-1].timestamp if clean_sequence.frames else 3.0)
        )
        traj_seed = (
            clean_sequence.trajectory_seed
            if clean_sequence.trajectory_seed is not None
            else clean_sequence.seed
        )
        trajectory = create_trajectory(clean_sequence.scenario, traj_dur, traj_seed)
        camera = PinholeCamera(clean_sequence.camera_params)
        motion_renderer = MotionRenderer(camera)
        canonical_result = self.plate_renderer.render(
            clean_sequence.plate_text,
            clean_sequence.plate_config,
        )

        is_rear = (clean_sequence.plate_side == 1) or (clean_sequence.scenario.value == "pass_through_rear")
        veh_color = clean_sequence.vehicle_color
        veh_len = clean_sequence.vehicle_length_m
        is_offset_pose = clean_sequence.is_rear_offset_pose

        degraded_images: list[Image.Image] = []
        degraded_frame_dicts: list[dict[str, Any]] = []

        shutter_s = profile.shutter_speed_s
        M = max(1, profile.subframe_samples)
        total_frames = len(clean_sequence.frames)

        for k, clean_annot in enumerate(clean_sequence.frames):
            t_k = clean_annot.timestamp
            frame_seed = profile.seed * 10007 + k * 37 + 13

            # --- Stage 1: Physical Motion Blur via Sub-frame Integration ---
            if M > 1 and shutter_s > 0.0005:
                sub_imgs_bgr: list[np.ndarray] = []
                corner_starts: list[tuple[float, float]] = []
                corner_ends: list[tuple[float, float]] = []

                for m in range(M):
                    frac = (m + 0.5) / float(M)
                    t_sub = t_k - (shutter_s / 2.0) + frac * shutter_s
                    raw_pose = trajectory.evaluate(t_sub)
                    pose_sub = compute_rear_plate_pose(raw_pose, veh_len) if is_offset_pose else raw_pose

                    img_sub, annot_sub = motion_renderer.render_frame(
                        canonical_result=canonical_result,
                        pose=pose_sub,
                        timestamp=t_sub,
                        frame_index=k,
                        vehicle_color=veh_color,
                        is_rear=is_rear,
                        plate_side=clean_sequence.plate_side,
                        vehicle_length_m=veh_len,
                        wind_seed=clean_sequence.seed,
                    )
                    bgr = cv2.cvtColor(np.array(img_sub), cv2.COLOR_RGB2BGR)
                    sub_imgs_bgr.append(bgr)

                    if m == 0 and annot_sub.is_visible:
                        corner_starts = annot_sub.corners.points
                    if m == M - 1 and annot_sub.is_visible:
                        corner_ends = annot_sub.corners.points

                accum = np.zeros_like(sub_imgs_bgr[0], dtype=np.float32)
                for bgr in sub_imgs_bgr:
                    accum += bgr.astype(np.float32)
                bgr_motion_blur = (accum / float(M)).astype(np.uint8)

                if corner_starts and corner_ends:
                    dx = corner_ends[0][0] - corner_starts[0][0]
                    dy = corner_ends[0][1] - corner_starts[0][1]
                    blur_px = float(math.hypot(dx, dy))
                else:
                    blur_px = 0.0
            else:
                bgr_motion_blur = cv2.cvtColor(np.array(clean_sequence.images[k]), cv2.COLOR_RGB2BGR)
                blur_px = 0.0

            # --- Stage 2: Illumination, Auto-Exposure, WDR & Motion-Linked Glare ---
            bgr_illum, exp_scale, slot_washout = illum_sim.apply_illumination(
                bgr_motion_blur,
                clean_annot,
                k,
                total_frames=total_frames,
            )

            # --- Stage 3: Optics (Defocus, Vignetting, Chromatic Aberration, Lens Dirt) ---
            bgr_optics, coc_px = optics_sim.apply_optics(
                bgr_illum,
                clean_annot.distance_m,
                clean_annot.is_visible,
            )

            # --- Stage 4: CMOS Sensor (Poisson-Gaussian Noise, 3DNR Ghosting, Sharpening) ---
            bgr_sensor, noise_sigma = sensor_sim.apply_sensor(
                bgr_optics,
                frame_seed=frame_seed,
            )

            # --- Stage 5: Compression (RTSP Substream + H.264 / GOP Blockiness) ---
            bgr_final, jpeg_q, frame_type = comp_sim.compress_frame(
                bgr_sensor,
                k,
            )

            # Ensure strict monochrome for Night IR across all channels
            if profile.illumination == IlluminationType.NIGHT_IR:
                gray_final = cv2.cvtColor(bgr_final, cv2.COLOR_BGR2GRAY)
                bgr_final = cv2.cvtColor(gray_final, cv2.COLOR_GRAY2BGR)

            final_pil = Image.fromarray(cv2.cvtColor(bgr_final, cv2.COLOR_BGR2RGB))
            degraded_images.append(final_pil)

            deg_log = FrameDegradationLog(
                frame_index=k,
                timestamp=t_k,
                shutter_s=shutter_s,
                blur_px=blur_px,
                exposure_scale=exp_scale,
                gain_db=profile.gain_db,
                noise_sigma=noise_sigma,
                defocus_radius_px=coc_px,
                jpeg_quality=jpeg_q,
                frame_type=frame_type,
                illumination=profile.illumination.value,
                slot_glare_washout=slot_washout,
                substream_scale=profile.substream_scale,
                temporal_nr_strength=profile.temporal_nr_strength,
                wdr_strength=profile.wdr_strength,
            )

            frame_dict = clean_annot.to_dict()
            frame_dict["degradation"] = deg_log.to_dict()
            degraded_frame_dicts.append(frame_dict)

        derived_id = f"{clean_sequence.sequence_id}_{profile.id}"
        return DegradedSequenceResult(
            sequence_id=derived_id,
            parent_sequence_id=clean_sequence.sequence_id,
            profile=profile,
            ground_truth_text=clean_sequence.plate_text.full_text,
            ocr_string=clean_sequence.plate_text.ocr_string,
            fps=clean_sequence.fps,
            num_frames=len(degraded_images),
            frames=degraded_frame_dicts,
            images=degraded_images,
            seed=profile.seed,
        )

    @staticmethod
    def create_comparison_filmstrip(
        clean_seq: SequenceResult,
        degraded_seq: DegradedSequenceResult,
        max_samples: int = 6,
        thumb_width: int = 340,
    ) -> Image.Image:
        """Create side-by-side (clean vs degraded) comparison filmstrip."""
        total_frames = len(clean_seq.images)
        if total_frames <= max_samples:
            indices = list(range(total_frames))
        else:
            indices = [int(round(i * (total_frames - 1) / (max_samples - 1))) for i in range(max_samples)]

        aspect = clean_seq.images[0].height / float(clean_seq.images[0].width)
        thumb_h = int(round(thumb_width * aspect))

        margin = 12
        header_h = 48
        row_gap = 28
        strip_w = len(indices) * (thumb_width + margin) + margin
        strip_h = header_h + 2 * thumb_h + row_gap + margin * 3

        comp_strip = Image.new("RGB", (strip_w, strip_h), (24, 26, 30))
        draw = ImageDraw.Draw(comp_strip)

        font_title = PlateRenderer._get_font("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 18, 0)
        font_sub = PlateRenderer._get_font("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 13, 0)
        font_tag = PlateRenderer._get_font("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 12, 0)

        title = (
            f"比較検証: {clean_seq.sequence_id} (Clean) vs {degraded_seq.sequence_id} "
            f"[{degraded_seq.profile.name} - {degraded_seq.profile.illumination.value}] | "
            f"Text: {clean_seq.plate_text.full_text}"
        )
        draw.text((margin, 12), title, fill=(245, 246, 248), font=font_title)

        y_clean_top = header_h + margin
        y_deg_top = y_clean_top + thumb_h + row_gap

        draw.text((margin, y_clean_top - 18), "【上段: Clean Part 2 (幾何運動・劣化なし)】", fill=(140, 210, 255), font=font_sub)
        draw.text((margin, y_deg_top - 18), "【下段: Degraded Part 3 (Camera Simulator: モーションブラー/露出/ノイズ/光学/圧縮)】", fill=(255, 190, 80), font=font_sub)

        for col_idx, f_idx in enumerate(indices):
            x = margin + col_idx * (thumb_width + margin)

            img_c = clean_seq.images[f_idx].resize((thumb_width, thumb_h), Image.Resampling.BILINEAR)
            comp_strip.paste(img_c, (x, y_clean_top))
            annot_c = clean_seq.frames[f_idx]
            lbl_c = f"#{f_idx} t={annot_c.timestamp:.2f}s W={annot_c.plate_width_px:.0f}px"
            draw.text((x + 6, y_clean_top + thumb_h - 20), lbl_c, fill=(240, 245, 255), font=font_tag)

            img_d = degraded_seq.images[f_idx].resize((thumb_width, thumb_h), Image.Resampling.BILINEAR)
            comp_strip.paste(img_d, (x, y_deg_top))
            deg_log = degraded_seq.frames[f_idx]["degradation"]
            lbl_d = f"blur={deg_log['blur_px']:.1f}px G={deg_log['gain_db']:.0f}dB Q={deg_log['jpeg_quality']}"
            draw.text((x + 6, y_deg_top + thumb_h - 20), lbl_d, fill=(255, 235, 80), font=font_tag)

        return comp_strip
