"""Data models for Part 4 Crop Simulator matching the vehicle-plate-ocr-jp-v1 Shared Tensor Contract."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from vpocr_synth.plate.models import PlateText

CROP_HEIGHT: int = 64
CROP_WIDTH: int = 128


@dataclass
class CropSequenceSample:
    """Canonical multi-frame OCR training/evaluation sequence sample.

    Matches every field and dtype in `shared_ocr_contract.md`.
    """

    sequence_id: str
    plate_text: PlateText | None
    plate_type: str                     # standard_private, standard_commercial, kei_private, kei_commercial, graphic
    crops_rgb: np.ndarray               # (T, 64, 128, 3), dtype=uint8
    timestamps_s: np.ndarray            # (T,), dtype=float32
    source_frame_ids: np.ndarray        # (T,), dtype=int32
    view_variants: list[str]            # length T: "raw", "gamma_low", "gamma_high", "clahe"
    camera_ids: np.ndarray              # (T,), dtype=int32 (0..3)
    plate_sides: np.ndarray             # (T,), dtype=int32 (0=front, 1=rear, 2=unknown/side)
    frame_valid_mask: np.ndarray        # (T,), dtype=float32 (1.0 or 0.0)
    is_foreign_plate: np.ndarray        # (T,), dtype=bool
    slot_readability: np.ndarray        # (T, 12), dtype=float32 in [0.0, 1.0]
    sequence_target_status: str         # "READABLE", "UNREADABLE", "CONFLICT", "TRACK_MIXED"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.crops_rgb = np.asarray(self.crops_rgb, dtype=np.uint8)
        self.timestamps_s = np.asarray(self.timestamps_s, dtype=np.float32)
        self.source_frame_ids = np.asarray(self.source_frame_ids, dtype=np.int32)
        self.camera_ids = np.asarray(self.camera_ids, dtype=np.int32)
        self.plate_sides = np.asarray(self.plate_sides, dtype=np.int32)
        self.frame_valid_mask = np.asarray(self.frame_valid_mask, dtype=np.float32)
        self.is_foreign_plate = np.asarray(self.is_foreign_plate, dtype=bool)
        self.slot_readability = np.asarray(self.slot_readability, dtype=np.float32)

    @property
    def num_frames(self) -> int:
        return int(self.crops_rgb.shape[0])

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize non-image sequence metadata for JSONL shard index."""
        from vpocr_synth.crop.readability import SlotReadabilityMapper

        slots_12 = (
            SlotReadabilityMapper.encode_plate_slots(self.plate_text)
            if self.plate_text is not None
            else None
        )
        return {
            "sequence_id": self.sequence_id,
            "plate_text": self.plate_text.to_dict() if self.plate_text is not None else None,
            "slots_12": slots_12,
            "plate_type": self.plate_type,
            "num_frames": self.num_frames,
            "view_variants": list(self.view_variants),
            "sequence_target_status": self.sequence_target_status,
            "metadata": self.metadata,
        }
