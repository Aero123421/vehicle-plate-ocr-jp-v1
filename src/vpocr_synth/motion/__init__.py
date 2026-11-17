"""Motion Generator module for continuous 3D vehicle dynamics and 360-degree camera projection."""

from vpocr_synth.motion.camera import PinholeCamera
from vpocr_synth.motion.generator import SequenceGenerator, compute_rear_plate_pose
from vpocr_synth.motion.models import (
    CameraParams,
    CameraViewStream,
    CharacterQuad,
    FrameAnnotation,
    MultiCameraPassResult,
    Pose3D,
    Quad2D,
    ScenarioType,
    SequenceResult,
)
from vpocr_synth.motion.renderer import MotionRenderer
from vpocr_synth.motion.trajectory import (
    BaseTrajectory,
    CurvingEntryTrajectory,
    DecelerateAndStopTrajectory,
    PassThroughRearTrajectory,
    StopAndGoTrajectory,
    StraightApproachTrajectory,
    SwitchbackReversingTrajectory,
    create_trajectory,
)

__all__ = [
    "BaseTrajectory",
    "CameraParams",
    "CameraViewStream",
    "CharacterQuad",
    "CurvingEntryTrajectory",
    "DecelerateAndStopTrajectory",
    "FrameAnnotation",
    "MotionRenderer",
    "MultiCameraPassResult",
    "PassThroughRearTrajectory",
    "PinholeCamera",
    "Pose3D",
    "Quad2D",
    "ScenarioType",
    "SequenceGenerator",
    "SequenceResult",
    "StopAndGoTrajectory",
    "StraightApproachTrajectory",
    "SwitchbackReversingTrajectory",
    "compute_rear_plate_pose",
    "create_trajectory",
]
