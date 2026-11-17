"""Data models for camera sensor, optics, illumination, and compression simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from PIL import Image

from vpocr_synth.motion.models import FrameAnnotation, SequenceResult
from vpocr_synth.plate.models import PlateConfig, PlateText


class IlluminationType(str, Enum):
    """Environmental illumination scenarios."""

    DAY_CLEAR = "day_clear"                # 日中晴天（直射日光・高コントラスト）
    DAY_CLOUDY = "day_cloudy"              # 曇天（拡散光・均一照明）
    BACKLIGHT = "backlight"                # 逆光（背景白飛び・車体影・フレア）
    DUSK_DAWN = "dusk_dawn"                # 夕方・薄暮（暖色低照度）
    NIGHT_HEADLIGHTS = "night_headlights"  # 夜間（車体ライト・街灯照明）
    NIGHT_IR = "night_ir"                  # 夜間赤外線（モノクロ・ナンバー再帰反射強調）


@dataclass
class CameraProfile:
    """Configurable multi-axis degradation profile for a camera and scene environment."""

    id: str
    name: str
    illumination: IlluminationType = IlluminationType.DAY_CLEAR
    shutter_speed_s: float = 0.001           # Exposure duration (e.g. 1/2000s to 1/15s, independent of FPS)
    gain_db: float = 0.0                     # Sensor analog/digital gain (0 to 30 dB)
    subframe_samples: int = 5                # Integration steps for physical motion blur
    enable_sharpening: bool = True
    sharpening_amount: float = 0.5           # Edge enhancement halo strength
    jpeg_quality_i: int = 80                 # Keyframe (I-frame) compression quality
    jpeg_quality_p: int = 55                 # Predicted frame (P-frame) compression quality
    gop_size: int = 15                       # GOP length (frames per I-frame)
    vignette_strength: float = 0.20          # Corner light falloff
    defocus_scale: float = 0.5               # Out-of-focus circle of confusion scaling
    ir_plate_reflectance_boost: float = 1.0  # Retro-reflective plate brightness boost for IR
    lens_dirt_prob: float = 0.0
    seed: int = 0
    # Extended combinatorial camera axes
    ev_bias: float = 0.0                     # Exposure compensation in EV stops (-1.5 to +1.5)
    ae_lag_alpha: float = 0.22               # Auto-exposure temporal convergence rate (0.05 slow .. 0.60 fast)
    wb_temp_shift: float = 0.0               # White balance shift (-1.0 cool blue .. +1.0 warm sodium)
    wdr_strength: float = 0.0                # Wide Dynamic Range shadow lift & highlight roll-off (0.0 .. 1.0)
    temporal_nr_strength: float = 0.0        # 3DNR temporal noise reduction ghosting trail (0.0 .. 0.85)
    substream_scale: float = 1.0             # Low-res RTSP substream downscale factor (e.g. 0.5 -> upscale)
    glare_mode: str = "none"                 # "none", "complementary_rows", "random_sweep", "full_washout"
    glare_intensity: float = 0.0             # Specular glare / shadow intensity (0.0 .. 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "illumination": self.illumination.value,
            "shutter_speed_s": round(self.shutter_speed_s, 6),
            "gain_db": round(self.gain_db, 2),
            "subframe_samples": self.subframe_samples,
            "enable_sharpening": self.enable_sharpening,
            "sharpening_amount": round(self.sharpening_amount, 2),
            "jpeg_quality_i": self.jpeg_quality_i,
            "jpeg_quality_p": self.jpeg_quality_p,
            "gop_size": self.gop_size,
            "vignette_strength": round(self.vignette_strength, 2),
            "defocus_scale": round(self.defocus_scale, 2),
            "ir_plate_reflectance_boost": round(self.ir_plate_reflectance_boost, 2),
            "lens_dirt_prob": round(self.lens_dirt_prob, 2),
            "ev_bias": round(self.ev_bias, 3),
            "ae_lag_alpha": round(self.ae_lag_alpha, 3),
            "wb_temp_shift": round(self.wb_temp_shift, 3),
            "wdr_strength": round(self.wdr_strength, 3),
            "temporal_nr_strength": round(self.temporal_nr_strength, 3),
            "substream_scale": round(self.substream_scale, 3),
            "glare_mode": self.glare_mode,
            "glare_intensity": round(self.glare_intensity, 3),
            "seed": self.seed,
        }


@dataclass
class FrameDegradationLog:
    """Per-frame physical degradation telemetry."""

    frame_index: int
    timestamp: float
    shutter_s: float
    blur_px: float                 # Estimated plate motion blur magnitude in pixels
    exposure_scale: float          # Dynamic auto-exposure multiplier
    gain_db: float                 # Sensor gain
    noise_sigma: float             # Read + shot noise standard deviation
    defocus_radius_px: float       # Circle of confusion radius
    jpeg_quality: int              # Applied compression quality factor
    frame_type: str                # "I" or "P"
    illumination: str
    slot_glare_washout: list[float] = field(default_factory=lambda: [0.0] * 12)
    substream_scale: float = 1.0
    temporal_nr_strength: float = 0.0
    wdr_strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "shutter_s": round(self.shutter_s, 6),
            "blur_px": round(self.blur_px, 2),
            "exposure_scale": round(self.exposure_scale, 3),
            "gain_db": round(self.gain_db, 2),
            "noise_sigma": round(self.noise_sigma, 3),
            "defocus_radius_px": round(self.defocus_radius_px, 2),
            "jpeg_quality": self.jpeg_quality,
            "frame_type": self.frame_type,
            "illumination": self.illumination,
            "slot_glare_washout": [round(float(v), 3) for v in self.slot_glare_washout],
            "substream_scale": round(self.substream_scale, 3),
            "temporal_nr_strength": round(self.temporal_nr_strength, 3),
            "wdr_strength": round(self.wdr_strength, 3),
        }


@dataclass
class DegradedSequenceResult:
    """Degraded multi-frame sequence derived from a clean motion trajectory."""

    sequence_id: str
    parent_sequence_id: str                  # Links to original clean trajectory
    profile: CameraProfile
    ground_truth_text: str
    ocr_string: str
    fps: float
    num_frames: int
    frames: list[dict[str, Any]]             # Combined clean ground truth + degradation logs
    images: list[Image.Image]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "parent_sequence_id": self.parent_sequence_id,
            "is_derived_sequence": True,     # Flags that this shares trajectory with parent
            "profile": self.profile.to_dict(),
            "ground_truth_text": self.ground_truth_text,
            "ocr_string": self.ocr_string,
            "fps": round(self.fps, 2),
            "num_frames": self.num_frames,
            "seed": self.seed,
            "frames": self.frames,
        }
