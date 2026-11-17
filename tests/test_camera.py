"""Unit tests for Part 3 Camera Simulator, Illumination Physics, and Multi-Axis Degradations."""

from dataclasses import replace
import math
import pytest
import cv2
import numpy as np

from vpocr_synth.camera.models import CameraProfile, IlluminationType
from vpocr_synth.camera.simulator import CameraSimulator
from vpocr_synth.crop.extractor import CropExtractor
from vpocr_synth.motion.camera import PinholeCamera
from vpocr_synth.motion.generator import SequenceGenerator
from vpocr_synth.motion.models import CameraParams, Pose3D, ScenarioType
from vpocr_synth.plate.models import PlateConfig, PlateSize, PlateText, PlateType
from vpocr_synth.plate.renderer import PlateRenderer


@pytest.fixture(scope="session")
def cam_sim():
    return CameraSimulator()


@pytest.fixture(scope="session")
def seq_gen():
    return SequenceGenerator()


def test_camera_determinism(cam_sim: CameraSimulator, seq_gen: SequenceGenerator):
    """Verify camera simulator degradation is 100% deterministic given the same seed."""
    clean_seq = seq_gen.generate_sequence(seed=101, duration_s=0.5, fps=10.0)
    profile = cam_sim.create_profile_by_id("day_standard_cctv", seed=42)
    profile.subframe_samples = 3

    deg1 = cam_sim.degrade_sequence(clean_seq, profile)
    deg2 = cam_sim.degrade_sequence(clean_seq, profile)

    assert deg1.num_frames == deg2.num_frames
    assert deg1.sequence_id == deg2.sequence_id
    assert deg1.parent_sequence_id == clean_seq.sequence_id

    for f1, f2 in zip(deg1.frames, deg2.frames):
        d1 = f1["degradation"]
        d2 = f2["degradation"]
        assert math.isclose(d1["blur_px"], d2["blur_px"], rel_tol=1e-5)
        assert math.isclose(d1["noise_sigma"], d2["noise_sigma"], rel_tol=1e-5)
        assert d1["jpeg_quality"] == d2["jpeg_quality"]


def test_motion_blur_proportional_to_shutter(cam_sim: CameraSimulator, seq_gen: SequenceGenerator):
    """Verify that longer shutter speed produces larger physical motion blur on moving vehicle."""
    clean_seq = seq_gen.generate_sequence(seed=0, duration_s=1.0, fps=10.0)

    prof_fast = CameraProfile(id="fast", name="Fast", shutter_speed_s=0.0005, subframe_samples=3, seed=1)
    prof_slow = CameraProfile(id="slow", name="Slow", shutter_speed_s=0.0333, subframe_samples=5, seed=1)

    deg_fast = cam_sim.degrade_sequence(clean_seq, prof_fast)
    deg_slow = cam_sim.degrade_sequence(clean_seq, prof_slow)

    max_blur_fast = max(f["degradation"]["blur_px"] for f in deg_fast.frames)
    max_blur_slow = max(f["degradation"]["blur_px"] for f in deg_slow.frames)

    assert max_blur_slow > 10.0, f"Expected slow shutter blur > 10px, got {max_blur_slow}"
    assert max_blur_fast <= 1.0, f"Expected fast shutter blur <= 1px, got {max_blur_fast}"


def test_night_noise_amplification(cam_sim: CameraSimulator, seq_gen: SequenceGenerator):
    """Verify that night high-gain profile produces higher sensor noise than daytime low-gain."""
    clean_seq = seq_gen.generate_sequence(seed=303, duration_s=0.5, fps=10.0)

    prof_day = cam_sim.create_profile_by_id("day_standard_cctv", seed=10)
    prof_day.subframe_samples = 2
    prof_night = cam_sim.create_profile_by_id("night_ir_monochrome", seed=10)
    prof_night.subframe_samples = 2

    deg_day = cam_sim.degrade_sequence(clean_seq, prof_day)
    deg_night = cam_sim.degrade_sequence(clean_seq, prof_night)

    sigma_day = deg_day.frames[0]["degradation"]["noise_sigma"]
    sigma_night = deg_night.frames[0]["degradation"]["noise_sigma"]

    assert sigma_night > sigma_day * 2.0, f"Night noise ({sigma_night}) should be > day ({sigma_day})"


def test_night_ir_monochrome_conversion(cam_sim: CameraSimulator, seq_gen: SequenceGenerator):
    """Verify that night IR mode outputs monochrome images (R == G == B)."""
    clean_seq = seq_gen.generate_sequence(seed=404, duration_s=0.5, fps=10.0)
    prof_ir = cam_sim.create_profile_by_id("night_ir_monochrome", seed=5)
    prof_ir.subframe_samples = 2

    deg_ir = cam_sim.degrade_sequence(clean_seq, prof_ir)
    img_arr = np.array(deg_ir.images[0])

    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    diff_rg = np.max(np.abs(r.astype(int) - g.astype(int)))
    diff_gb = np.max(np.abs(g.astype(int) - b.astype(int)))

    assert diff_rg == 0, "Night IR must be strictly grayscale"
    assert diff_gb == 0, "Night IR must be strictly grayscale"


def test_gop_compression_structure(cam_sim: CameraSimulator, seq_gen: SequenceGenerator):
    """Verify GOP structure produces I-frames at frame 0 and multiples of gop_size."""
    clean_seq = seq_gen.generate_sequence(seed=505, duration_s=1.2, fps=15.0)
    prof = cam_sim.create_profile_by_id("day_standard_cctv", seed=1)
    prof.subframe_samples = 2

    deg = cam_sim.degrade_sequence(clean_seq, prof)

    for f in deg.frames:
        idx = f["frame_index"]
        ftype = f["degradation"]["frame_type"]
        if idx % prof.gop_size == 0:
            assert ftype == "I", f"Frame {idx} should be I-frame"
            assert f["degradation"]["jpeg_quality"] == prof.jpeg_quality_i
        else:
            assert ftype == "P", f"Frame {idx} should be P-frame"


def test_illumination_quantitative_brightness_physics(cam_sim: CameraSimulator, seq_gen: SequenceGenerator):
    """Verify physically grounded brightness levels across DAY_CLEAR, BACKLIGHT, NIGHT_HEADLIGHTS, and NIGHT_IR."""
    pt = PlateText(region="品川", classification="599", hiragana="さ", serial=1234)
    pc = PlateConfig(plate_type=PlateType.STANDARD_PRIVATE, plate_size=PlateSize.MEDIUM, seed=7)
    clean_seq = seq_gen.generate_sequence(
        seed=10,
        scenario=ScenarioType.DECELERATE_AND_STOP,
        duration_s=0.8,
        fps=8.0,
        plate_text=pt,
        plate_config=pc,
    )

    # Pick a frame with clearly visible plate
    vis_indices = [i for i, f in enumerate(clean_seq.frames) if f.is_visible and f.plate_width_px > 50]
    assert len(vis_indices) > 0
    f_idx = vis_indices[-1]
    annot = clean_seq.frames[f_idx]

    # Build plate polygon mask
    w, h = clean_seq.images[f_idx].size
    plate_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(plate_mask, np.array(annot.corners.points, dtype=np.int32), 255)

    # 1. DAY_CLEAR
    prof_day = CameraProfile(id="day", name="Day", illumination=IlluminationType.DAY_CLEAR, shutter_speed_s=0.0005, subframe_samples=1, seed=11)
    deg_day = cam_sim.degrade_sequence(clean_seq, prof_day)
    arr_day = np.array(deg_day.images[f_idx], dtype=np.float32)
    day_mean = float(np.mean(arr_day))
    assert 85.0 <= day_mean <= 155.0, f"DAY_CLEAR mean out of natural range: {day_mean:.1f}"

    # 2. NIGHT_IR
    prof_ir = CameraProfile(
        id="ir",
        name="Night IR",
        illumination=IlluminationType.NIGHT_IR,
        shutter_speed_s=0.0005,
        subframe_samples=1,
        gain_db=14.0,
        ir_plate_reflectance_boost=1.8,
        seed=11,
    )
    deg_ir = cam_sim.degrade_sequence(clean_seq, prof_ir)
    arr_ir = np.array(deg_ir.images[f_idx], dtype=np.float32)
    ir_bg_mean = float(np.mean(arr_ir[plate_mask == 0]))
    ir_plate_mean = float(np.mean(arr_ir[plate_mask > 0]))
    assert ir_bg_mean < 45.0, f"NIGHT_IR non-plate background too bright (gray-fog!): {ir_bg_mean:.1f}"
    assert ir_plate_mean > 135.0, f"NIGHT_IR retro-reflective plate too dim: {ir_plate_mean:.1f}"
    assert ir_plate_mean > 2.0 * ir_bg_mean, f"NIGHT_IR plate ({ir_plate_mean:.1f}) must be > 2.0x bg ({ir_bg_mean:.1f})"

    # 3. NIGHT_HEADLIGHTS
    prof_hl = CameraProfile(
        id="hl",
        name="Night HL",
        illumination=IlluminationType.NIGHT_HEADLIGHTS,
        shutter_speed_s=0.0005,
        subframe_samples=1,
        gain_db=12.0,
        seed=11,
    )
    deg_hl = cam_sim.degrade_sequence(clean_seq, prof_hl)
    arr_hl = np.array(deg_hl.images[f_idx], dtype=np.float32)
    hl_mean = float(np.mean(arr_hl))
    # Upper dark sky/background away from headlights
    hl_upper_bg = float(np.mean(arr_hl[: int(h * 0.25), :]))
    assert hl_upper_bg < 45.0, f"NIGHT_HEADLIGHTS upper background too bright: {hl_upper_bg:.1f}"
    assert day_mean > hl_mean + 45.0, f"DAY_CLEAR ({day_mean:.1f}) must be much brighter than NIGHT_HEADLIGHTS ({hl_mean:.1f})"

    # 4. BACKLIGHT (sky > 195, shadowed foreground < 90 without WDR; WDR lifts shadows)
    prof_bl = CameraProfile(
        id="bl",
        name="Backlight",
        illumination=IlluminationType.BACKLIGHT,
        shutter_speed_s=0.0005,
        subframe_samples=1,
        vignette_strength=0.05,
        wdr_strength=0.0,
        seed=11,
    )
    deg_bl = cam_sim.degrade_sequence(clean_seq, prof_bl)
    arr_bl = np.array(deg_bl.images[f_idx], dtype=np.float32)
    sky_mean = float(np.mean(arr_bl[: max(10, int(h * 0.14)), :]))
    fg_shadow_mean = float(np.mean(arr_bl[int(h * 0.65) :, :]))
    assert sky_mean > 195.0, f"BACKLIGHT sky should be > 195, got {sky_mean:.1f}"
    assert fg_shadow_mean < 90.0, f"BACKLIGHT shadowed foreground should be < 90, got {fg_shadow_mean:.1f}"

    prof_bl_wdr = replace(prof_bl, id="bl_wdr", wdr_strength=0.9)
    deg_bl_wdr = cam_sim.degrade_sequence(clean_seq, prof_bl_wdr)
    arr_bl_wdr = np.array(deg_bl_wdr.images[f_idx], dtype=np.float32)
    fg_wdr_mean = float(np.mean(arr_bl_wdr[int(h * 0.65) :, :]))
    assert fg_wdr_mean > fg_shadow_mean + 8.0, "WDR must lift shadowed foreground brightness"


def test_vehicle_color_preserved_in_degrade_sequence(cam_sim: CameraSimulator, seq_gen: SequenceGenerator):
    """Verify Bug 2 fix: CameraSimulator preserves SequenceResult.vehicle_color during sub-frame rendering."""
    seq_dark = seq_gen.generate_sequence(seed=15, duration_s=0.4, fps=8.0)
    seq_dark.vehicle_color = (25, 25, 28)
    seq_bright = replace(seq_dark, vehicle_color=(245, 245, 245))

    prof = CameraProfile(id="blur_col", name="BlurCol", shutter_speed_s=0.015, subframe_samples=3, seed=2)
    deg_dark = cam_sim.degrade_sequence(seq_dark, prof)
    deg_bright = cam_sim.degrade_sequence(seq_bright, prof)

    diff = np.mean(np.abs(np.array(deg_bright.images[-1], dtype=np.float32) - np.array(deg_dark.images[-1], dtype=np.float32)))
    assert diff > 2.5, "Vehicle body color difference was lost during CameraSimulator sub-frame integration"


def test_combinatorial_profile_and_counterfactual_pair(cam_sim: CameraSimulator, seq_gen: SequenceGenerator):
    """Verify sample_random_profile covers multi-axis space and generate_counterfactual_pair isolates 1 factor."""
    profs = [cam_sim.sample_random_profile(seed=i, difficulty="mixed") for i in range(25)]
    illums = {p.illumination for p in profs}
    assert len(illums) >= 4
    assert any(p.substream_scale < 1.0 for p in profs)
    assert any(p.temporal_nr_strength > 0.1 for p in profs)

    clean_seq = seq_gen.generate_sequence(seed=77, duration_s=0.5, fps=10.0)
    deg_low, deg_high = cam_sim.generate_counterfactual_pair(
        clean_seq,
        factor_name="ev_bias",
        low_val=-1.0,
        high_val=1.0,
        seed=99,
    )
    m_low = float(np.mean(np.array(deg_low.images[0], dtype=np.float32)))
    m_high = float(np.mean(np.array(deg_high.images[0], dtype=np.float32)))
    assert m_high > m_low * 1.4


def test_plate_surface_artifacts_and_complementary_glare(cam_sim: CameraSimulator, seq_gen: SequenceGenerator):
    """Verify persistent plate surface artifacts and motion-linked complementary_rows glare."""
    renderer = PlateRenderer()
    pt = PlateText(region="世田谷", classification="30A", hiragana="わ", serial=8888)
    cfg = PlateConfig(
        plate_type=PlateType.STANDARD_PRIVATE,
        mud_level=0.35,
        plate_frame=True,
        weathering=0.4,
        is_illuminated=True,
        persistent_occluder_slot=11,
        seed=42,
    )
    res = renderer.render(pt, cfg)
    assert res.occlusion_mask is not None
    assert float(np.max(res.occlusion_mask)) == 1.0

    # Complementary rows glare on clean trajectory
    clean_seq = seq_gen.generate_sequence(
        seed=12,
        scenario=ScenarioType.STRAIGHT_APPROACH,
        duration_s=1.0,
        fps=8.0,
        plate_text=pt,
        plate_config=PlateConfig(seed=12),
    )
    prof_glare = CameraProfile(
        id="comp_glare",
        name="Complementary Glare",
        illumination=IlluminationType.DAY_CLEAR,
        shutter_speed_s=0.0005,
        subframe_samples=1,
        glare_mode="complementary_rows",
        glare_intensity=0.98,
        seed=5,
    )
    deg_glare = cam_sim.degrade_sequence(clean_seq, prof_glare)
    extractor = CropExtractor()
    crop_sample = extractor.extract_from_sequence(clean_seq, deg_glare, crop_mode="oracle", seed=5)

    # Frame 0: lower row (7..11) washed out, upper row (0..6) clear
    r_early = crop_sample.slot_readability[0]
    # Last frame: upper row (0..6) washed out, lower row (7..11) clear
    r_late = crop_sample.slot_readability[-1]
    assert float(np.mean(r_early[:6])) > 0.45 and float(np.mean(r_early[7:])) < 0.15
    assert float(np.mean(r_late[:6])) < 0.15 and float(np.mean(r_late[7:])) > 0.55
    assert crop_sample.sequence_target_status == "READABLE"
