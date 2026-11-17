"""Japanese License Plate Generator module."""

from vpocr_synth.plate.data_loader import PlateDataLoader, RegionInfo
from vpocr_synth.plate.geometry import CharSlot, PlateGeometry, PlateLayout
from vpocr_synth.plate.models import (
    BoundingBox,
    CharacterBox,
    PlateConfig,
    PlateSize,
    PlateText,
    PlateType,
    RenderResult,
)
from vpocr_synth.plate.renderer import PlateRenderer
from vpocr_synth.plate.sampler import PlateSampler, SamplerDistribution
from vpocr_synth.plate.validator import PlateValidator, ValidationError

__all__ = [
    "BoundingBox",
    "CharSlot",
    "CharacterBox",
    "PlateConfig",
    "PlateDataLoader",
    "PlateGeometry",
    "PlateLayout",
    "PlateRenderer",
    "PlateSampler",
    "PlateSize",
    "PlateText",
    "PlateType",
    "PlateValidator",
    "RegionInfo",
    "RenderResult",
    "SamplerDistribution",
    "ValidationError",
]
