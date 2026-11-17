"""Camera Simulator module for physical sensor, optics, illumination, and compression."""

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
from vpocr_synth.camera.simulator import CameraSimulator

__all__ = [
    "CameraProfile",
    "CameraSimulator",
    "CompressionSimulator",
    "DegradedSequenceResult",
    "FrameDegradationLog",
    "IlluminationSimulator",
    "IlluminationType",
    "OpticsSimulator",
    "SensorSimulator",
]
