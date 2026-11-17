"""Shared data contracts for vehicle-plate-ocr-jp-v1 full-frame detection, tight 4-corner cropping, and tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeAlias
import numpy as np


# Detection class IDs
CLS_PLATE: int = 0
CLS_VEHICLE: int = 1
CLS_SIGNBOARD_HARDNEG: int = 2
NUM_DET_CLASSES: int = 2
NUM_GEN4_DET_CLASSES: int = 3
NUM_VEHICLE_SUBTYPES: int = 8

# Plate side orientation IDs
SIDE_FRONT: int = 0
SIDE_REAR: int = 1
SIDE_PURE_SIDE: int = 2


class ObservationProvenance(StrEnum):
    """Sampling policy that produced a fresh image measurement, never a prediction."""

    DETECTOR_FULL = "detector_full"
    DETECTOR_FAR_ZOOM = "detector_far_zoom"
    SEARCH_MEASUREMENT = "search_measurement"
    STANDALONE_PRECROP = "standalone_precrop"


@dataclass(frozen=True, slots=True)
class ObservationActionMeta:
    """Executed image actions used to condition detection availability p_D(x, a_t).

    An empty action means no detector or search ran, not a negative observation.
    ROI coordinates are in full-frame pixels; frame keys identify camera/frame
    measurements and must remain shared across repeated crops of those pixels.
    """

    action_id: str = ""
    detector_ran_full: bool = False
    detector_ran_far_zoom: bool = False
    searched_rois_xyxy: tuple[tuple[float, float, float, float], ...] = ()
    frame_key: str | None = None


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Score-free action intent; cannot update existence or OCR as a measurement.

    Only a fresh measurement acquired by executing this request may become an
    observation, with SEARCH_MEASUREMENT provenance and appropriate calibration.
    ROI coordinates and the diagonal expected position covariance are in full-frame
    pixels and squared pixels, respectively.
    """

    request_id: str
    roi_xyxy: tuple[float, float, float, float]
    target_entity_id: int | None
    relation_hint: str
    expected_cov_xy: tuple[float, float]
    priority_cost_gain: float


@dataclass
class SceneObjectGT:
    """Ground-truth annotation for a single plate, vehicle, or hard-negative sign in a synthesized full frame."""

    cls_id: int  # 0 = plate, 1 = vehicle, 2 = signboard_hardneg
    bbox_xyxy: np.ndarray  # (4,) float32 in pixel coords [x0, y0, x1, y1]
    corners_xy: np.ndarray  # (4, 2) float32 [TL, TR, BR, BL] in pixel coords (for vehicle: bbox corners)
    plate_side: int = SIDE_FRONT  # 0=front, 1=rear, 2=side
    visibility: float = 1.0  # [0, 1] visible unoccluded fraction
    vehicle_archetype: str = "sedan"
    vehicle_subtype: int = 0
    plate_text_str: str = ""
    plate_type_str: str = ""
    amodal_bbox_xyxy: np.ndarray | None = None
    cabin_bbox_xyxy: np.ndarray | None = None
    cabin_valid: bool = False
    ground_footprint_xy: np.ndarray | None = None
    footprint_valid: bool = True
    visible_roi_mask: np.ndarray | None = None  # (64, 64) float32 inside bbox_xyxy


@dataclass
class SynthSceneSample:
    """A full-frame multi-scale synthesized camera scene before GPU camera-ISP degradation."""

    image_rgb: np.ndarray  # (H, W, 3) uint8 RGB
    objects: list[SceneObjectGT] = field(default_factory=list)
    ignore_bboxes_xyxy: list[tuple[float, float, float, float]] = field(default_factory=list)
    scene_tag: str = "street"
    camera_height_m: float = 2.2


@dataclass
class QuadCropTrainingSample:
    """Training sample for `PlateQuadNet` (expanded noisy ROI -> exact 4-corner tight cutout + verifier)."""

    roi_rgb: np.ndarray  # (96, 160, 3) uint8 RGB
    is_plate: float  # 1.0 for real plate ROI, 0.0 for hard-negative distractor (grille/sign/headlight/road)
    corners_norm: np.ndarray  # (4, 2) float32 normalized in [0, 1] within (W=160, H=96): [TL, TR, BR, BL]
    plate_side: int  # 0=front, 1=rear, 2=side
    tight_quality: float  # [0, 1] readability/visibility score


@dataclass
class PlateDetection:
    """A single detected plate (optionally paired with its parent vehicle & refined 4-corner quad)."""

    bbox_xyxy: tuple[float, float, float, float]
    score: float
    corners_xy: np.ndarray  # (4, 2) float32 [TL, TR, BR, BL] in full-frame pixel coordinates
    quad_refined: bool = False
    quad_confidence: float = 0.0
    quad_is_plate_prob: float = 1.0
    plate_side: int = SIDE_FRONT
    tight_crop_rgb: np.ndarray | None = None  # (64, 128, 3) uint8 exact tight plate-only cutout
    ocr_crop_rgb: np.ndarray | None = None  # (80, 160, 3) or (64, 128, 3) uint8 OCR-margin rectified crop
    track_id: int | None = None
    vehicle_bbox_xyxy: tuple[float, float, float, float] | None = None
    raw_corners_xy: np.ndarray | None = None  # Unlagged PlateQuadNet 4 corners paired with ocr_crop_rgb
    geom_meta: np.ndarray | None = None  # Exact 10-D geometry + photometric vector paired with ocr_crop_rgb
    is_coasted: bool = False  # True when predicted on a missed frame (never feed coasted crops to OCR)
    raw_det_score: float = 0.0  # Unmodified Stage-1 PlateDetNet probability prior to Quad fusion
    quad_edge_touch: bool = False  # True when a predicted corner touches an interior ROI cut edge
    corner_cov_xy: np.ndarray | None = None  # (4, 2, 2) float32 SPD corner covariance in full-frame px^2
    blur_vec_xy: np.ndarray | None = None  # (2,) float32 optical motion blur vector in full-frame px
    reid_emb: np.ndarray | None = None  # (64,) float32 L2-normalized surround Re-ID embedding
    topology_prob: float = 1.0  # [0, 1] valid clockwise convex 4-corner quad topology score
    is_large_prob: float = 0.0  # [0, 1] 4-bolt 440x220mm large plate probability
    roi_gamma: float = 1.0  # Dynamic motion/uncertainty ROI expansion factor used for cropping
    parent_vehicle_track_id: int | None = None  # Persistent parent vehicle track ID from MultiVehicleKalmanTracker
    mount_zone_score: float = 1.0  # [0, 1] anatomical vehicle-mount spatial prior score
    layout_prob: float = 1.0  # [0, 1] patent-clean 4x8 MLIT stroke-occupancy layout verification score
    plate_null_llr: float | None = None  # Joint plate/null log likelihood ratio, conditional on sampling policy
    corner_visibility_4: np.ndarray | None = None  # (4,) visible support for [TL, TR, BR, BL]
    homography_3x3: np.ndarray | None = None  # (3, 3) canonical plate to undistorted image projective map
    support_nonconformity: float = 0.0  # Held-out support statistic; not an independent detection score
    provenance: ObservationProvenance = ObservationProvenance.DETECTOR_FULL
    frame_key: str | None = None  # Stable camera/frame key shared by measurements from the same pixels


@dataclass
class VehicleDetection:
    """Vehicle observation with optional Gen-6 evidence and legacy heavy-vehicle outputs."""

    bbox_xyxy: tuple[float, float, float, float]
    score: float
    vehicle_subtype: int = 0
    subtype_probs: np.ndarray | None = None          # (9,) float32 heavy-vehicle operational subtype probabilities
    cutout_mask_64: np.ndarray | None = None         # (64, 64) float32 fractional optical silhouette in [0, 1]
    cutout_rgba: np.ndarray | None = None            # Optional native-res (H_v, W_v, 4) uint8 RGBA cutout
    amodal_bbox_xyxy: tuple[float, float, float, float] | None = None
    cabin_bbox_xyxy: tuple[float, float, float, float] | None = None
    cabin_valid: bool = False
    ground_footprint_xy: np.ndarray | None = None    # (4, 2) float32 Z=0 road-plane enclosing 4-gon in px
    footprint_valid: bool = False
    vehicle_track_id: int | None = None              # Persistent multi-vehicle Kalman track ID
    plate_track_ids: list[int] = field(default_factory=list)  # Bound front/rear plate track IDs
    primary_gate_subject: bool = True                # True for primary gate lane/foreground vehicle subject
    plate_mount_box_xyxy: tuple[float, float, float, float] | None = None  # Predicted plate mount zone [x0, y0, x1, y1]
    plate_mount_prob: float = 0.0                    # Predicted plate mount zone visibility probability
    veh_log_odds: float | None = None  # Calibrated vehicle evidence; None means unavailable
    bbox_cov_4x4: np.ndarray | None = None  # (4, 4) covariance of full-frame [x0, y0, x1, y1]
    appearance_emb: np.ndarray | None = None  # Class-free appearance; dimension is a model resource choice
    subtype_unknown_prob: float = 0.0  # Explicit unknown subtype outcome, never an identity veto
    provenance: ObservationProvenance = ObservationProvenance.DETECTOR_FULL


Observation: TypeAlias = PlateDetection | VehicleDetection


def assert_not_observation(request: object) -> None:
    """Validate the score-free search boundary; raise TypeError on observations.

    Static consumers should accept SearchRequest separately from Observation.
    This runtime guard also rejects scored subclasses and duck-typed detections
    at dynamically typed boundaries; it never converts a prediction to evidence.
    """

    if not isinstance(request, SearchRequest):
        raise TypeError("Expected a score-free SearchRequest, not an Observation")
    evidence_fields = ("score", "raw_det_score", "veh_log_odds", "plate_null_llr")
    if any(hasattr(request, name) for name in evidence_fields):
        raise TypeError("A SearchRequest must not carry detection scores or observation evidence")


@dataclass
class FrameDetectionResult:
    """Full-frame detection + tight-crop result."""

    plates: list[PlateDetection] = field(default_factory=list)
    vehicle_bboxes_xyxy: list[tuple[float, float, float, float, float]] = field(default_factory=list)  # (x0,y0,x1,y1,score)
    vehicles: list[VehicleDetection] = field(default_factory=list)
    latency_ms: float = 0.0
    ocr_decisions: dict[int, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CLS_PLATE",
    "CLS_VEHICLE",
    "CLS_SIGNBOARD_HARDNEG",
    "NUM_DET_CLASSES",
    "NUM_GEN4_DET_CLASSES",
    "NUM_VEHICLE_SUBTYPES",
    "SIDE_FRONT",
    "SIDE_REAR",
    "SIDE_PURE_SIDE",
    "ObservationProvenance",
    "ObservationActionMeta",
    "SearchRequest",
    "Observation",
    "assert_not_observation",
    "SceneObjectGT",
    "SynthSceneSample",
    "QuadCropTrainingSample",
    "PlateDetection",
    "VehicleDetection",
    "FrameDetectionResult",
]
