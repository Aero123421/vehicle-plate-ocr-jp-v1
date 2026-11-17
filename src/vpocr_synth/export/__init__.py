"""Part 5 Sequence & Label Exporter: high-throughput crop sequence synthesis and NPZ/JSONL sharding."""

from pathlib import Path
from typing import Any, Protocol

from vpocr_synth.export.fast_synthesizer import (
    FastCropSequenceSynthesizer,
    FastPlatePatchRenderer,
    FastSynthesizerConfig,
)
from vpocr_synth.export.shard_io import ShardExporter, ShardReader


class SequenceExporterProtocol(Protocol):
    """Protocol for serializing multi-frame synthetic dataset sequences."""

    def export(
        self,
        output_dir: Path,
        sequences: list[Any],
        split: str = "train",
    ) -> Path:
        """Export serialized sequence files and metadata."""
        ...


__all__ = [
    "FastCropSequenceSynthesizer",
    "FastPlatePatchRenderer",
    "FastSynthesizerConfig",
    "SequenceExporterProtocol",
    "ShardExporter",
    "ShardReader",
]
