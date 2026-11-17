"""Part 4 Crop Simulator: detector bounding-box emulation, 12-slot readability, and multi-view crops."""

from typing import Any, Protocol
import numpy as np

from vpocr_synth.crop.extractor import CropExtractor, CropExtractorConfig
from vpocr_synth.crop.models import CROP_HEIGHT, CROP_WIDTH, CropSequenceSample
from vpocr_synth.crop.readability import DOT_TOKEN, PAD_TOKEN, SlotReadabilityMapper


class CropSimulatorProtocol(Protocol):
    """Protocol for simulating detector bounding-box crops."""

    def crop(
        self,
        frame: np.ndarray,
        plate_box: tuple[float, float, float, float],
        jitter_params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Extract a simulated detection crop around the license plate."""
        ...


__all__ = [
    "CROP_HEIGHT",
    "CROP_WIDTH",
    "CropExtractor",
    "CropExtractorConfig",
    "CropSequenceSample",
    "CropSimulatorProtocol",
    "DOT_TOKEN",
    "PAD_TOKEN",
    "SlotReadabilityMapper",
]
