"""Data models for vehicle motion and camera sequence simulation."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any
import numpy as np
from PIL import Image

from vpocr_synth.plate.models import PlateConfig, PlateText


class ScenarioType(str, Enum):
    """Vehicle motion scenarios for construction site entry/exit."""

    STRAIGHT_APPROACH = "straight_approach"        # 直進接近
    DECELERATE_AND_STOP = "decelerate_stop"        # 減速して停止
    STOP_AND_GO = "stop_and_go"                    # 停止状態からの再発進
    CURVING_ENTRY = "curving_entry"                # カーブしながらの進入
    SWITCHBACK_REVERSING = "switchback_reversing"  # 切り返し・後退
    PASS_THROUGH_REAR = "pass_through_rear"        # 通過・退出（リア側プレート）


@dataclass
class Pose3D:
    """3D state of the license plate at continuous time t."""

    position: np.ndarray    # [x, y, z] in world meters
    yaw_deg: float          # Heading angle in degrees
    pitch_deg: float = 0.0  # Vertical tilt due to road surface
    roll_deg: float = 0.0   # Lateral tilt due to cornering / suspension
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [vx, vy, vz] in m/s

    @property
    def speed_ms(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def speed_kmh(self) -> float:
        return self.speed_ms * 3.6

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": [round(float(p), 4) for p in self.position],
            "yaw_deg": round(float(self.yaw_deg), 2),
            "pitch_deg": round(float(self.pitch_deg), 2),
            "roll_deg": round(float(self.roll_deg), 2),
            "velocity": [round(float(v), 3) for v in self.velocity],
            "speed_kmh": round(self.speed_kmh, 2),
        }


@dataclass
class CameraParams:
    """Pinhole camera configuration with extrinsics and intrinsics."""

    width: int = 1920
    height: int = 1080
    fov_h_deg: float = 55.0
    position: np.ndarray = field(default_factory=lambda: np.array([1.2, 0.0, 2.8]))  # [x, y, z] in meters
    pitch_deg: float = 20.0  # Downward tilt (俯角)
    yaw_deg: float = 5.0     # Horizontal pan angle
    roll_deg: float = 0.0
    k1: float = 0.0          # Radial distortion coefficient
    k2: float = 0.0
    c_offset_x: float = 0.0  # Principal point offset
    c_offset_y: float = 0.0
    camera_id: int = 0
    mount_role: str = "front"
    wind_shake_deg: float = 0.0
    clock_offset_s: float = 0.0

    @property
    def fx(self) -> float:
        """Focal length in horizontal pixels."""
        return (self.width / 2.0) / math.tan(math.radians(self.fov_h_deg / 2.0))

    @property
    def fy(self) -> float:
        """Focal length in vertical pixels (square pixels)."""
        return self.fx

    @property
    def cx(self) -> float:
        return (self.width / 2.0) + self.c_offset_x

    @property
    def cy(self) -> float:
        return (self.height / 2.0) + self.c_offset_y

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """3x3 Camera Intrinsics matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "mount_role": self.mount_role,
            "resolution": [self.width, self.height],
            "fov_h_deg": round(self.fov_h_deg, 2),
            "position": [round(float(p), 3) for p in self.position],
            "pitch_deg": round(self.pitch_deg, 2),
            "yaw_deg": round(self.yaw_deg, 2),
            "roll_deg": round(self.roll_deg, 2),
            "wind_shake_deg": round(self.wind_shake_deg, 3),
            "clock_offset_s": round(self.clock_offset_s, 4),
            "k1": round(self.k1, 5),
            "k2": round(self.k2, 5),
        }


@dataclass
class Quad2D:
    """Projected 2D quadrilateral in camera image pixel coordinates."""

    points: list[tuple[float, float]]  # [tl, tr, br, bl]

    @property
    def width(self) -> float:
        """Average width of top and bottom edges."""
        p = self.points
        top_w = math.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1])
        bot_w = math.hypot(p[2][0] - p[3][0], p[2][1] - p[3][1])
        return (top_w + bot_w) / 2.0

    @property
    def height(self) -> float:
        """Average height of left and right edges."""
        p = self.points
        left_h = math.hypot(p[3][0] - p[0][0], p[3][1] - p[0][1])
        right_h = math.hypot(p[2][0] - p[1][0], p[2][1] - p[1][1])
        return (left_h + right_h) / 2.0

    def to_bbox(self) -> tuple[float, float, float, float]:
        """AABB bounding box [x_min, y_min, x_max, y_max]."""
        xs = [pt[0] for pt in self.points]
        ys = [pt[1] for pt in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def to_dict(self) -> list[list[float]]:
        return [[round(x, 2), round(y, 2)] for x, y in self.points]


@dataclass
class CharacterQuad:
    """Projected quadrilateral for an individual character."""

    char: str
    field: str
    index: int
    quad: Quad2D

    def to_dict(self) -> dict[str, Any]:
        return {
            "char": self.char,
            "field": self.field,
            "index": self.index,
            "quad": self.quad.to_dict(),
            "bbox": [round(v, 2) for v in self.quad.to_bbox()],
        }


@dataclass
class FrameAnnotation:
    """Full ground-truth labels for a single frame within a sequence."""

    frame_index: int
    timestamp: float
    corners: Quad2D
    plate_width_px: float
    plate_height_px: float
    distance_m: float
    angles_deg: dict[str, float]  # {"yaw": float, "pitch": float, "roll": float}
    visibility: float            # 0.0 to 1.0 (portion inside image canvas)
    is_visible: bool
    pose: Pose3D
    character_quads: list[CharacterQuad]
    field_quads: dict[str, Quad2D]
    incidence_deg: float = 0.0
    is_readable_view: bool = True
    camera_id: int = 0
    plate_side: int = 0  # 0=front, 1=rear, 2=unknown/side

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "camera_id": self.camera_id,
            "plate_side": self.plate_side,
            "corners": self.corners.to_dict(),
            "plate_width_px": round(self.plate_width_px, 2),
            "plate_height_px": round(self.plate_height_px, 2),
            "distance_m": round(self.distance_m, 3),
            "incidence_deg": round(self.incidence_deg, 2),
            "angles_deg": {k: round(v, 2) for k, v in self.angles_deg.items()},
            "visibility": round(self.visibility, 3),
            "is_visible": self.is_visible,
            "is_readable_view": self.is_readable_view,
            "pose": self.pose.to_dict(),
            "character_quads": [cq.to_dict() for cq in self.character_quads],
            "field_quads": {k: q.to_dict() for k, q in self.field_quads.items()},
        }


@dataclass
class SequenceResult:
    """Complete multi-frame synthetic sequence output bundle."""

    sequence_id: str
    scenario: ScenarioType
    plate_text: PlateText
    plate_config: PlateConfig
    camera_params: CameraParams
    fps: float
    num_frames: int
    frames: list[FrameAnnotation]
    images: list[Image.Image]
    seed: int
    vehicle_color: tuple[int, int, int] = (180, 182, 186)
    camera_id: int = 0
    plate_side: int = 0  # 0=front, 1=rear, 2=unknown/side
    vehicle_length_m: float = 4.6
    trajectory_seed: int | None = None
    trajectory_duration_s: float | None = None
    is_rear_offset_pose: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "scenario": self.scenario.value,
            "ground_truth_text": self.plate_text.full_text,
            "ocr_string": self.plate_text.ocr_string,
            "fields": self.plate_text.to_dict(),
            "plate_type": self.plate_config.plate_type.value,
            "plate_size": self.plate_config.plate_size.value,
            "is_graphic": self.plate_config.is_graphic,
            "vehicle_color": list(self.vehicle_color),
            "camera_id": self.camera_id,
            "plate_side": self.plate_side,
            "fps": round(self.fps, 2),
            "num_frames": self.num_frames,
            "camera": self.camera_params.to_dict(),
            "seed": self.seed,
            "frames": [f.to_dict() for f in self.frames],
        }


@dataclass
class CameraViewStream:
    """Single camera observation stream within a multi-camera gate pass."""

    camera_id: int
    mount_role: str            # "front", "oblique", "pure_side", "oblique_rear", "rear"
    plate_side: int            # 0=front, 1=rear, 2=side
    clock_offset_s: float
    clean_sequence: SequenceResult
    degraded_sequence: Any = None  # DegradedSequenceResult | None
    camera_profile: Any = None     # CameraProfile | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "mount_role": self.mount_role,
            "plate_side": self.plate_side,
            "clock_offset_s": round(self.clock_offset_s, 4),
            "fps": round(self.clean_sequence.fps, 2),
            "num_frames": self.clean_sequence.num_frames,
            "camera": self.clean_sequence.camera_params.to_dict(),
            "profile": self.camera_profile.to_dict() if self.camera_profile is not None else None,
            "readable_frames": sum(1 for f in self.clean_sequence.frames if f.is_readable_view),
        }


@dataclass
class MultiCameraPassResult:
    """Simultaneous 1..4 camera observation bundle of a single vehicle pass."""

    pass_id: str
    scenario: ScenarioType
    plate_text: PlateText
    front_plate_config: PlateConfig
    rear_plate_config: PlateConfig
    vehicle_color: tuple[int, int, int]
    vehicle_length_m: float
    duration_s: float
    rig_preset: str
    streams: list[CameraViewStream]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "scenario": self.scenario.value,
            "ground_truth_text": self.plate_text.full_text,
            "ocr_string": self.plate_text.ocr_string,
            "fields": self.plate_text.to_dict(),
            "plate_type": self.front_plate_config.plate_type.value,
            "plate_size": self.front_plate_config.plate_size.value,
            "rear_has_seal": self.rear_plate_config.with_seal,
            "vehicle_color": list(self.vehicle_color),
            "vehicle_length_m": round(self.vehicle_length_m, 2),
            "duration_s": round(self.duration_s, 3),
            "rig_preset": self.rig_preset,
            "num_cameras": len(self.streams),
            "seed": self.seed,
            "streams": [s.to_dict() for s in self.streams],
        }
