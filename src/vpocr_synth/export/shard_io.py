"""Part 5 Dataset Shard Exporter & Reader (.npz + .jsonl)."""

import json
from pathlib import Path
from typing import Any, Iterator
import numpy as np

from vpocr_synth.crop.models import CropSequenceSample
from vpocr_synth.plate.models import PlateText


class ShardExporter:
    """Packs `CropSequenceSample` batches into `.npz` tensor shards + `.jsonl` metadata shards."""

    def __init__(self, shard_size: int = 256, compressed: bool = False) -> None:
        self.shard_size = max(1, shard_size)
        self.compressed = compressed

    def export(
        self,
        output_dir: Path | str,
        sequences: list[CropSequenceSample],
        split: str = "train",
    ) -> Path:
        """Export sequences into `output_dir / split` (or `output_dir` if `split == ''`) and return split directory."""
        base_dir = Path(output_dir)
        target_dir = base_dir / split if split else base_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        shard_records: list[dict[str, Any]] = []
        status_counts: dict[str, int] = {}
        num_shards = (len(sequences) + self.shard_size - 1) // self.shard_size if sequences else 0

        for shard_idx in range(num_shards):
            chunk = sequences[shard_idx * self.shard_size : (shard_idx + 1) * self.shard_size]
            shard_stem = f"shard_{shard_idx:05d}"
            npz_path = target_dir / f"{shard_stem}.npz"
            jsonl_path = target_dir / f"{shard_stem}.jsonl"

            offsets = [0]
            for s in chunk:
                offsets.append(offsets[-1] + s.num_frames)
                status_counts[s.sequence_target_status] = status_counts.get(s.sequence_target_status, 0) + 1

            seq_offsets = np.array(offsets, dtype=np.int64)
            crops_cat = np.concatenate([s.crops_rgb for s in chunk], axis=0).astype(np.uint8)
            ts_cat = np.concatenate([s.timestamps_s for s in chunk], axis=0).astype(np.float32)
            src_cat = np.concatenate([s.source_frame_ids for s in chunk], axis=0).astype(np.int32)
            cam_cat = np.concatenate([s.camera_ids for s in chunk], axis=0).astype(np.int32)
            side_cat = np.concatenate([s.plate_sides for s in chunk], axis=0).astype(np.int32)
            valid_cat = np.concatenate([s.frame_valid_mask for s in chunk], axis=0).astype(np.float32)
            foreign_cat = np.concatenate([s.is_foreign_plate for s in chunk], axis=0).astype(bool)
            read_cat = np.concatenate([s.slot_readability for s in chunk], axis=0).astype(np.float32)

            arrays = {
                "seq_offsets": seq_offsets,
                "crops_rgb": crops_cat,
                "timestamps_s": ts_cat,
                "source_frame_ids": src_cat,
                "camera_ids": cam_cat,
                "plate_sides": side_cat,
                "frame_valid_mask": valid_cat,
                "is_foreign_plate": foreign_cat,
                "slot_readability": read_cat,
            }
            if self.compressed:
                np.savez_compressed(npz_path, **arrays)
            else:
                np.savez(npz_path, **arrays)

            with open(jsonl_path, "w", encoding="utf-8") as f_j:
                for s in chunk:
                    f_j.write(json.dumps(s.to_metadata_dict(), ensure_ascii=False) + "\n")

            shard_records.append({
                "shard_index": shard_idx,
                "npz": npz_path.name,
                "jsonl": jsonl_path.name,
                "num_sequences": len(chunk),
                "total_frames": int(offsets[-1]),
            })

        manifest = {
            "split": split,
            "total_sequences": len(sequences),
            "num_shards": num_shards,
            "shard_size": self.shard_size,
            "status_counts": status_counts,
            "shards": shard_records,
        }
        with open(target_dir / "manifest.json", "w", encoding="utf-8") as f_m:
            json.dump(manifest, f_m, ensure_ascii=False, indent=2)

        return target_dir


class ShardReader:
    """Random-access and streaming reader for `.npz` + `.jsonl` `CropSequenceSample` shards."""

    def __init__(self, shard_dir: Path | str, split: str | None = None) -> None:
        base = Path(shard_dir)
        if split and (base / split / "manifest.json").exists():
            self.shard_dir = base / split
        elif (base / "manifest.json").exists():
            self.shard_dir = base
        elif (base / "train" / "manifest.json").exists():
            self.shard_dir = base / "train"
        else:
            self.shard_dir = base

        manifest_path = self.shard_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest: dict[str, Any] = json.load(f)
            shard_Entries = self.manifest.get("shards", [])
            self._shard_pairs = [
                (self.shard_dir / entry["npz"], self.shard_dir / entry["jsonl"])
                for entry in shard_Entries
            ]
        else:
            npzs = sorted(self.shard_dir.glob("shard_*.npz"))
            self._shard_pairs = [(p, p.with_suffix(".jsonl")) for p in npzs]
            self.manifest = {"shards": []}

        self._loaded_shards: dict[int, tuple[dict[str, np.ndarray], list[dict[str, Any]]]] = {}
        self._index_map: list[tuple[int, int]] = []
        for s_idx, (_, jsonl_p) in enumerate(self._shard_pairs):
            with open(jsonl_p, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
            for local_i in range(count):
                self._index_map.append((s_idx, local_i))

    def _get_shard(self, shard_idx: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        cached = self._loaded_shards.get(shard_idx)
        if cached is not None:
            return cached

        npz_p, jsonl_p = self._shard_pairs[shard_idx]
        with np.load(npz_p, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        with open(jsonl_p, "r", encoding="utf-8") as f:
            metas = [json.loads(line) for line in f if line.strip()]
        self._loaded_shards[shard_idx] = (arrays, metas)
        return arrays, metas

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, index: int) -> CropSequenceSample:
        if index < 0:
            index += len(self._index_map)
        if not (0 <= index < len(self._index_map)):
            raise IndexError(f"Sequence index {index} out of range [0, {len(self._index_map)})")

        shard_idx, local_idx = self._index_map[index]
        arrays, metas = self._get_shard(shard_idx)
        meta = metas[local_idx]
        s0 = int(arrays["seq_offsets"][local_idx])
        s1 = int(arrays["seq_offsets"][local_idx + 1])

        pt_dict = meta.get("plate_text")
        plate_text = (
            PlateText(
                region=pt_dict["region"],
                classification=pt_dict["classification"],
                hiragana=pt_dict["hiragana"],
                serial=int(pt_dict["serial"]),
            )
            if pt_dict is not None
            else None
        )

        return CropSequenceSample(
            sequence_id=meta["sequence_id"],
            plate_text=plate_text,
            plate_type=meta["plate_type"],
            crops_rgb=arrays["crops_rgb"][s0:s1].copy(),
            timestamps_s=arrays["timestamps_s"][s0:s1].copy(),
            source_frame_ids=arrays["source_frame_ids"][s0:s1].copy(),
            view_variants=list(meta["view_variants"]),
            camera_ids=arrays["camera_ids"][s0:s1].copy(),
            plate_sides=arrays["plate_sides"][s0:s1].copy(),
            frame_valid_mask=arrays["frame_valid_mask"][s0:s1].copy(),
            is_foreign_plate=arrays["is_foreign_plate"][s0:s1].copy(),
            slot_readability=arrays["slot_readability"][s0:s1].copy(),
            sequence_target_status=meta["sequence_target_status"],
            metadata=dict(meta.get("metadata", {})),
        )

    def __iter__(self) -> Iterator[CropSequenceSample]:
        for i in range(len(self)):
            yield self[i]

    def load_all(self) -> list[CropSequenceSample]:
        return [self[i] for i in range(len(self))]
