"""Dataset adapters for CropSequenceSample/.npz shards and standalone MicroPlateSequenceDataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from vpocr_ocr.vocab import NUM_SLOTS, PlateVocab, STATUS_TO_ID
from vpocr_synth.plate.models import BoundingBox, PlateConfig, PlateSize, PlateText, PlateType, RenderResult
from vpocr_synth.plate.renderer import PlateRenderer
from vpocr_synth.plate.sampler import PlateSampler


def _parse_plate_text_any(val: Any) -> PlateText | None:
    """Convert PlateText, dict, JSON string, or 4-tuple into PlateText (or None)."""
    if val is None:
        return None
    if isinstance(val, PlateText):
        return val
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return _parse_plate_text_any(val.item())
        if val.size == 0:
            return None
    if isinstance(val, (bytes, bytearray)):
        val = val.decode("utf-8")
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == "none" or s.lower() == "null":
            return None
        if s.startswith("{"):
            d = json.loads(s)
            return _parse_plate_text_any(d)
    if isinstance(val, Mapping):
        if not val.get("region"):
            return None
        return PlateText(
            region=str(val["region"]),
            classification=str(val["classification"]),
            hiragana=str(val["hiragana"]),
            serial=int(val["serial"]),
        )
    if isinstance(val, (tuple, list)) and len(val) == 4:
        return PlateText(
            region=str(val[0]),
            classification=str(val[1]),
            hiragana=str(val[2]),
            serial=int(val[3]),
        )
    raise ValueError(f"Cannot parse plate_text from {val!r}")


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def adapt_sequence_sample(sample: Any, vocab: PlateVocab) -> dict[str, Any]:
    """Convert any CropSequenceSample-compatible object or dict into normalized tensor-ready dict."""
    crops_raw = np.asarray(_get_field(sample, "crops_rgb"))
    if crops_raw.ndim != 4:
        raise ValueError(f"Expected 4D crops_rgb, got shape {crops_raw.shape}")

    if crops_raw.shape[-1] == 3:
        # (T, 64, 128, 3) -> (T, 3, 64, 128)
        crops_chw = np.transpose(crops_raw, (0, 3, 1, 2))
    elif crops_raw.shape[1] == 3:
        crops_chw = crops_raw
    else:
        raise ValueError(f"Unrecognized crops_rgb channel axis: {crops_raw.shape}")

    if crops_chw.dtype == np.uint8:
        crops_f = crops_chw.astype(np.float32) / 255.0
    else:
        crops_f = crops_chw.astype(np.float32)
        if crops_f.max() > 1.5:
            crops_f = crops_f / 255.0
    crops_f = np.clip(crops_f, 0.0, 1.0)

    seq_len = crops_f.shape[0]
    seq_id = str(_get_field(sample, "sequence_id", "seq_0"))

    plate_text = _parse_plate_text_any(_get_field(sample, "plate_text", None))
    if plate_text is None and _get_field(sample, "region", None):
        plate_text = PlateText(
            region=str(_get_field(sample, "region")),
            classification=str(_get_field(sample, "classification")),
            hiragana=str(_get_field(sample, "hiragana")),
            serial=int(_get_field(sample, "serial")),
        )

    raw_ptype = _get_field(sample, "plate_type", PlateType.STANDARD_PRIVATE.value)
    if isinstance(raw_ptype, np.ndarray) and raw_ptype.ndim == 0:
        raw_ptype = raw_ptype.item()
    ptype_str = raw_ptype.value if isinstance(raw_ptype, PlateType) else str(raw_ptype)
    ptype_id = vocab.encode_plate_type(ptype_str)

    timestamps_s = np.asarray(
        _get_field(sample, "timestamps_s", np.arange(seq_len, dtype=np.float32) * 0.1),
        dtype=np.float32,
    ).reshape(seq_len)
    source_frame_ids = np.asarray(
        _get_field(sample, "source_frame_ids", np.arange(seq_len, dtype=np.int64)),
        dtype=np.int64,
    ).reshape(seq_len)
    camera_ids = np.asarray(
        _get_field(sample, "camera_ids", np.zeros(seq_len, dtype=np.int64)),
        dtype=np.int64,
    ).reshape(seq_len)
    plate_sides = np.asarray(
        _get_field(sample, "plate_sides", np.zeros(seq_len, dtype=np.int64)),
        dtype=np.int64,
    ).reshape(seq_len)
    frame_valid_mask = np.asarray(
        _get_field(sample, "frame_valid_mask", np.ones(seq_len, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(seq_len)
    is_foreign_plate = np.asarray(
        _get_field(sample, "is_foreign_plate", np.zeros(seq_len, dtype=bool)),
        dtype=bool,
    ).reshape(seq_len)
    slot_readability = np.asarray(
        _get_field(sample, "slot_readability", np.ones((seq_len, NUM_SLOTS), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(seq_len, NUM_SLOTS)

    raw_status = _get_field(sample, "sequence_target_status", "READABLE")
    if isinstance(raw_status, np.ndarray) and raw_status.ndim == 0:
        raw_status = raw_status.item()
    status_str = str(raw_status).upper()
    status_id = STATUS_TO_ID.get(status_str, 0)

    if plate_text is not None:
        target_slots = np.asarray(vocab.encode(plate_text), dtype=np.int64)
        has_target = True
    else:
        target_slots = np.zeros(NUM_SLOTS, dtype=np.int64)
        has_target = False

    out: dict[str, Any] = {
        "sequence_id": seq_id,
        "crops": crops_f,  # (T, 3, 64, 128)
        "timestamps_s": timestamps_s,
        "source_frame_ids": source_frame_ids,
        "camera_ids": camera_ids,
        "plate_sides": plate_sides,
        "frame_valid_mask": frame_valid_mask,
        "is_foreign_plate": is_foreign_plate,
        "slot_readability": slot_readability,
        "target_slots": target_slots,
        "target_plate_type": ptype_id,
        "target_status": status_id,
        "has_target_plate": has_target,
        "plate_text": plate_text,
        "plate_type": ptype_str,
        "sequence_target_status": status_str,
    }

    frame_slot_targets = _get_field(sample, "frame_slot_targets", None)
    if frame_slot_targets is not None:
        out["frame_slot_targets"] = np.asarray(frame_slot_targets, dtype=np.int64).reshape(
            seq_len, NUM_SLOTS
        )
    elif bool(is_foreign_plate.any()) and has_target:
        meta = _get_field(sample, "metadata", None)
        if isinstance(meta, Mapping) and meta.get("foreign_plate_dict"):
            f_pt = _parse_plate_text_any(meta["foreign_plate_dict"])
            if f_pt is not None:
                f_slots = np.asarray(vocab.encode(f_pt), dtype=np.int64)
                ft = np.full((seq_len, NUM_SLOTS), -1, dtype=np.int64)
                for t_i in range(seq_len):
                    if is_foreign_plate[t_i]:
                        ft[t_i] = f_slots
                    elif frame_valid_mask[t_i] > 0.5:
                        ft[t_i] = target_slots
                out["frame_slot_targets"] = ft
    return out


def collate_plate_sequences(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length multi-frame plate sequence samples into a GPU-ready batch."""
    bsz = len(items)
    max_t = max(int(it["crops"].shape[0]) for it in items)

    crops = torch.zeros((bsz, max_t, 3, 64, 128), dtype=torch.float32)
    timestamps_s = torch.zeros((bsz, max_t), dtype=torch.float32)
    source_frame_ids = torch.full((bsz, max_t), -1, dtype=torch.long)
    camera_ids = torch.zeros((bsz, max_t), dtype=torch.long)
    plate_sides = torch.zeros((bsz, max_t), dtype=torch.long)
    frame_valid_mask = torch.zeros((bsz, max_t), dtype=torch.float32)
    is_foreign_plate = torch.zeros((bsz, max_t), dtype=torch.bool)
    slot_readability = torch.zeros((bsz, max_t, NUM_SLOTS), dtype=torch.float32)
    frame_pad_mask = torch.ones((bsz, max_t), dtype=torch.bool)

    target_slots = torch.zeros((bsz, NUM_SLOTS), dtype=torch.long)
    target_plate_type = torch.zeros((bsz,), dtype=torch.long)
    target_status = torch.zeros((bsz,), dtype=torch.long)
    has_target_plate = torch.zeros((bsz,), dtype=torch.bool)

    has_any_explicit_ft = any("frame_slot_targets" in it for it in items)
    frame_slot_targets = (
        torch.full((bsz, max_t, NUM_SLOTS), -1, dtype=torch.long)
        if has_any_explicit_ft
        else None
    )

    raw_plate_texts: list[PlateText | None] = []
    raw_plate_types: list[str] = []
    sequence_ids: list[str] = []

    for b, it in enumerate(items):
        t_len = int(it["crops"].shape[0])
        crops[b, :t_len] = torch.from_numpy(it["crops"])
        timestamps_s[b, :t_len] = torch.from_numpy(it["timestamps_s"])
        source_frame_ids[b, :t_len] = torch.from_numpy(it["source_frame_ids"])
        camera_ids[b, :t_len] = torch.from_numpy(it["camera_ids"])
        plate_sides[b, :t_len] = torch.from_numpy(it["plate_sides"])
        frame_valid_mask[b, :t_len] = torch.from_numpy(it["frame_valid_mask"])
        is_foreign_plate[b, :t_len] = torch.from_numpy(it["is_foreign_plate"])
        slot_readability[b, :t_len] = torch.from_numpy(it["slot_readability"])
        frame_pad_mask[b, :t_len] = False

        t_slots = torch.from_numpy(it["target_slots"])
        target_slots[b] = t_slots
        target_plate_type[b] = int(it["target_plate_type"])
        target_status[b] = int(it["target_status"])
        has_target_plate[b] = bool(it["has_target_plate"])

        if frame_slot_targets is not None:
            if "frame_slot_targets" in it:
                frame_slot_targets[b, :t_len] = torch.from_numpy(it["frame_slot_targets"])
            elif bool(it["has_target_plate"]):
                valid_f = (~is_foreign_plate[b, :t_len]) & (frame_valid_mask[b, :t_len] > 0.5)
                for t in range(t_len):
                    if valid_f[t]:
                        frame_slot_targets[b, t] = t_slots

        raw_plate_texts.append(it["plate_text"])
        raw_plate_types.append(it["plate_type"])
        sequence_ids.append(it["sequence_id"])

    batch: dict[str, Any] = {
        "crops": crops,
        "timestamps_s": timestamps_s,
        "source_frame_ids": source_frame_ids,
        "camera_ids": camera_ids,
        "plate_sides": plate_sides,
        "frame_valid_mask": frame_valid_mask,
        "is_foreign_plate": is_foreign_plate,
        "slot_readability": slot_readability,
        "frame_pad_mask": frame_pad_mask,
        "target_slots": target_slots,
        "target_plate_type": target_plate_type,
        "target_status": target_status,
        "has_target_plate": has_target_plate,
        "raw_plate_texts": raw_plate_texts,
        "raw_plate_types": raw_plate_types,
        "sequence_ids": sequence_ids,
    }
    if frame_slot_targets is not None:
        batch["frame_slot_targets"] = frame_slot_targets
    return batch


class CropSequenceDataset(Dataset[dict[str, Any]]):
    """Duck-type PyTorch Dataset wrapping in-memory CropSequenceSample objects, dicts, or .npz shards."""

    def __init__(
        self,
        source: Sequence[Any] | str | Path,
        vocab: PlateVocab | None = None,
    ) -> None:
        self.vocab = vocab or PlateVocab()
        self._samples: list[Any] = []
        self._npz_entries: list[tuple[Path, int | None]] = []
        self._packed_shard_cache: dict[Path, tuple[dict[str, np.ndarray], list[dict[str, Any]]]] = {}

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                npz_files = sorted(path.glob("*.npz"))
                if not npz_files and (path / "train").is_dir():
                    npz_files = sorted((path / "train").glob("*.npz"))
            elif path.is_file() and path.suffix == ".npz":
                npz_files = [path]
            else:
                raise FileNotFoundError(f"No .npz shard found at {path}")

            for fpath in npz_files:
                with np.load(fpath, allow_pickle=True) as data:
                    if "seq_offsets" in data and fpath.with_suffix(".jsonl").is_file():
                        n_seq = int(len(data["seq_offsets"]) - 1)
                        for i in range(n_seq):
                            self._npz_entries.append((fpath, i))
                    elif "crops_rgb" in data and data["crops_rgb"].ndim == 5:
                        n_seq = int(data["crops_rgb"].shape[0])
                        for i in range(n_seq):
                            self._npz_entries.append((fpath, i))
                    elif "num_sequences" in data:
                        n_seq = int(np.asarray(data["num_sequences"]).item())
                        for i in range(n_seq):
                            self._npz_entries.append((fpath, i))
                    else:
                        self._npz_entries.append((fpath, None))
        else:
            self._samples = list(source)

    def __len__(self) -> int:
        return len(self._npz_entries) if self._npz_entries else len(self._samples)

    def _get_packed_shard(
        self, fpath: Path
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        if fpath not in self._packed_shard_cache:
            with np.load(fpath, allow_pickle=True) as data:
                arrays = {k: data[k] for k in data.files}
            with open(fpath.with_suffix(".jsonl"), "r", encoding="utf-8") as f:
                metas = [json.loads(line) for line in f if line.strip()]
            self._packed_shard_cache[fpath] = (arrays, metas)
        return self._packed_shard_cache[fpath]

    def _load_npz_entry(self, fpath: Path, sub_idx: int | None) -> dict[str, Any]:
        if sub_idx is not None and fpath.with_suffix(".jsonl").is_file():
            arrays, metas = self._get_packed_shard(fpath)
            if "seq_offsets" in arrays:
                meta = metas[sub_idx]
                s0 = int(arrays["seq_offsets"][sub_idx])
                s1 = int(arrays["seq_offsets"][sub_idx + 1])
                raw_item: dict[str, Any] = {
                    "sequence_id": meta.get("sequence_id", f"seq_{sub_idx}"),
                    "plate_text": meta.get("plate_text"),
                    "plate_type": meta.get("plate_type", "standard_private"),
                    "sequence_target_status": meta.get("sequence_target_status", "READABLE"),
                    "metadata": meta.get("metadata", {}),
                    "crops_rgb": arrays["crops_rgb"][s0:s1],
                    "timestamps_s": arrays["timestamps_s"][s0:s1],
                    "source_frame_ids": arrays["source_frame_ids"][s0:s1],
                    "camera_ids": arrays["camera_ids"][s0:s1],
                    "plate_sides": arrays["plate_sides"][s0:s1],
                    "frame_valid_mask": arrays["frame_valid_mask"][s0:s1],
                    "is_foreign_plate": arrays["is_foreign_plate"][s0:s1],
                    "slot_readability": arrays["slot_readability"][s0:s1],
                }
                return adapt_sequence_sample(raw_item, self.vocab)

        with np.load(fpath, allow_pickle=True) as data:
            if sub_idx is None:
                raw_dict = {k: data[k] for k in data.files}
                if "metadata_json" in raw_dict and "plate_text" not in raw_dict:
                    meta = json.loads(str(np.asarray(raw_dict["metadata_json"]).item()))
                    raw_dict.update(meta)
                return adapt_sequence_sample(raw_dict, self.vocab)

            prefix = f"seq_{sub_idx}_"
            if any(k.startswith(prefix) for k in data.files):
                sub = {k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)}
                if "metadata_json" in sub and "plate_text" not in sub:
                    meta = json.loads(str(np.asarray(sub["metadata_json"]).item()))
                    sub.update(meta)
                return adapt_sequence_sample(sub, self.vocab)

            # Batched 5D shard fallback
            sub = {}
            for k in data.files:
                arr = data[k]
                if arr.ndim >= 1 and arr.shape[0] > sub_idx:
                    sub[k] = arr[sub_idx]
            return adapt_sequence_sample(sub, self.vocab)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._npz_entries:
            fpath, sub_idx = self._npz_entries[idx]
            return self._load_npz_entry(fpath, sub_idx)
        return adapt_sequence_sample(self._samples[idx], self.vocab)


@dataclass
class _CachedBasePlate:
    """Cached canonical rendered plate image and 12-slot bounding boxes."""

    image_rgb: np.ndarray                 # (H_p, W_p, 3) uint8
    plate_text: PlateText
    plate_type: PlateType
    slot_boxes: list[BoundingBox | None]  # length 12
    slot_ids: np.ndarray                  # (12,) int64


def _extract_12_slot_boxes(res: RenderResult) -> list[BoundingBox | None]:
    """Map RenderResult.character_boxes onto exact physical 12-slot indices."""
    boxes: list[BoundingBox | None] = [None] * NUM_SLOTS
    serial_len = len(str(res.plate_text.serial))
    for cb in res.character_boxes:
        if cb.field == "region" and 0 <= cb.index < 4:
            boxes[cb.index] = cb.bbox
        elif cb.field == "classification" and 0 <= cb.index < 3:
            boxes[4 + cb.index] = cb.bbox
        elif cb.field == "hiragana":
            boxes[7] = cb.bbox
        elif cb.field == "serial":
            if serial_len == 4:
                if cb.index == 0:
                    boxes[8] = cb.bbox
                elif cb.index == 1:
                    boxes[9] = cb.bbox
                elif cb.index == 3:
                    boxes[10] = cb.bbox
                elif cb.index == 4:
                    boxes[11] = cb.bbox
            else:
                if 0 <= cb.index < 4:
                    boxes[8 + cb.index] = cb.bbox
    return boxes


class MicroPlateSequenceDataset(Dataset[dict[str, Any]]):
    """Fast self-contained multi-frame Japanese plate dataset for standalone training & benchmarks.

    Modes:
      - 'mixed': balanced training curriculum across all phenomena
      - 'clean': clean multi-frame detector crops with mild perspective/jitter
      - 'degraded': realistic approach trajectory (distance/blur/moving partial glare)
      - 'complementary_glare': first half lower-row glared out, second half upper-row glared out
      - 'persistent_occlusion': >=1 active slot occluded across ALL frames -> UNREADABLE
      - 'track_mixed': foreign vehicle plate spliced into sequence -> TRACK_MIXED
      - 'garbage_tail': clean frames followed by appended garbage frames
      - 'same_frame_multiview': multi-exposure variants sharing (camera_id, source_frame_id)
    """

    def __init__(
        self,
        num_sequences: int = 512,
        seq_len: int = 8,
        mode: str = "mixed",
        plate_pool_size: int = 256,
        fixed_plates: Sequence[tuple[PlateText, PlateConfig]] | None = None,
        vocab: PlateVocab | None = None,
        render_scale: float = 0.65,
        seed: int = 42,
        stochastic: bool = False,
    ) -> None:
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.mode = mode
        self.vocab = vocab or PlateVocab()
        self.render_scale = render_scale
        self.seed = seed
        self.stochastic = stochastic

        self.renderer = PlateRenderer()
        self.sampler = PlateSampler()

        if fixed_plates is not None:
            self._specs: list[tuple[PlateText, PlateConfig]] = list(fixed_plates)
        else:
            self._specs = []
            for i in range(max(2, plate_pool_size)):
                pt, cfg = self.sampler.sample(seed=seed * 10007 + i)
                cfg.scale = render_scale
                cfg.plate_size = PlateSize.MEDIUM
                self._specs.append((pt, cfg))

        self._cache: dict[int, _CachedBasePlate] = {}
        self._seq_cache: dict[int, dict[str, Any]] = {}
        self._call_counter: int = 0

    def __len__(self) -> int:
        return self.num_sequences

    def _get_base_plate(self, spec_idx: int) -> _CachedBasePlate:
        idx = spec_idx % len(self._specs)
        if idx not in self._cache:
            pt, cfg = self._specs[idx]
            cfg_copy = PlateConfig(
                plate_type=cfg.plate_type,
                plate_size=PlateSize.MEDIUM,
                scale=self.render_scale,
                font_path=cfg.font_path,
                font_index=cfg.font_index,
                with_bolt_holes=cfg.with_bolt_holes,
                with_seal=cfg.with_seal,
                with_border=cfg.with_border,
                with_emboss=cfg.with_emboss,
                emboss_intensity=cfg.emboss_intensity,
                is_graphic=cfg.is_graphic,
                graphic_pattern=cfg.graphic_pattern,
                pattern_seed=cfg.pattern_seed,
                seed=cfg.seed,
            )
            res = self.renderer.render(pt, cfg_copy)
            img_rgb = np.asarray(res.image.convert("RGB"), dtype=np.uint8)
            boxes = _extract_12_slot_boxes(res)
            slot_ids = np.asarray(self.vocab.encode(pt), dtype=np.int64)
            self._cache[idx] = _CachedBasePlate(
                image_rgb=img_rgb,
                plate_text=pt,
                plate_type=cfg_copy.plate_type,
                slot_boxes=boxes,
                slot_ids=slot_ids,
            )
        return self._cache[idx]

    @staticmethod
    def make_garbage_crop(rng: np.random.Generator, kind: int | None = None) -> np.ndarray:
        """Synthesize a realistic (64, 128, 3) uint8 non-plate garbage crop."""
        k = int(rng.integers(0, 3)) if kind is None else (kind % 3)
        if k == 0:
            # Specular whiteout / overexposed sky-glare crop
            base = rng.uniform(236.0, 255.0, size=(8, 16, 3)).astype(np.float32)
            img = cv2.resize(base, (128, 64), interpolation=cv2.INTER_LINEAR)
            img += rng.normal(0.0, 4.0, size=img.shape).astype(np.float32)
        elif k == 1:
            # Dark asphalt / bumper sensor noise crop
            base = rng.uniform(8.0, 52.0, size=(8, 16, 3)).astype(np.float32)
            img = cv2.resize(base, (128, 64), interpolation=cv2.INTER_LINEAR)
            img += rng.normal(0.0, 14.0, size=img.shape).astype(np.float32)
        else:
            # Extreme >80-deg side motion smear (horizontal streak with zero glyph structure)
            col = rng.uniform(35.0, 210.0, size=(64, 3, 3)).astype(np.float32)
            img = cv2.resize(col, (128, 64), interpolation=cv2.INTER_LINEAR)
            img = cv2.GaussianBlur(img, (31, 3), sigmaX=12.0, sigmaY=0.8)
            img += rng.normal(0.0, 6.0, size=img.shape).astype(np.float32)
        return np.clip(img, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _warp_plate_crop(
        base: _CachedBasePlate,
        rng: np.random.Generator,
        jitter_scale: float = 0.04,
        glare_region: str | None = None,
        occluded_slots: Sequence[int] | None = None,
        moving_glare_box: tuple[float, float, float, float] | None = None,
        blur_ksize: int = 1,
        downscale_factor: float = 1.0,
        exposure_gamma: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render a single (64, 128, 3) uint8 detector crop + exact (12,) slot_readability."""
        hp, wp = base.image_rgb.shape[:2]
        canvas = base.image_rgb.astype(np.float32).copy()
        vis_map = np.ones((hp, wp), dtype=np.float32)

        # Row boundary y = 0.38 * H separates upper row (y=0.09..0.33) and lower row (y=0.42..0.91)
        row_split_y = int(round(0.38 * hp))

        if glare_region == "lower":
            canvas[row_split_y:, :] = rng.uniform(246.0, 255.0)
            bloom_top = max(0, int(round(0.35 * hp)))
            if row_split_y > bloom_top:
                alpha = np.linspace(0.0, 1.0, row_split_y - bloom_top, endpoint=False, dtype=np.float32)[:, None, None]
                canvas[bloom_top:row_split_y] = (1.0 - alpha) * canvas[bloom_top:row_split_y] + alpha * 252.0
            vis_map[row_split_y:, :] = 0.0
        elif glare_region == "upper":
            canvas[:row_split_y, :] = rng.uniform(246.0, 255.0)
            bloom_bot = min(hp, int(round(0.41 * hp)))
            if bloom_bot > row_split_y:
                alpha = np.linspace(1.0, 0.0, bloom_bot - row_split_y, endpoint=False, dtype=np.float32)[:, None, None]
                canvas[row_split_y:bloom_bot] = (1.0 - alpha) * canvas[row_split_y:bloom_bot] + alpha * 252.0
            vis_map[:row_split_y, :] = 0.0

        if moving_glare_box is not None:
            gx0 = int(np.clip(moving_glare_box[0] * wp, 0, wp))
            gy0 = int(np.clip(moving_glare_box[1] * hp, 0, hp))
            gx1 = int(np.clip(moving_glare_box[2] * wp, 0, wp))
            gy1 = int(np.clip(moving_glare_box[3] * hp, 0, hp))
            if gx1 > gx0 and gy1 > gy0:
                canvas[gy0:gy1, gx0:gx1] = rng.uniform(244.0, 255.0)
                vis_map[gy0:gy1, gx0:gx1] = 0.0

        if occluded_slots:
            for s_idx in occluded_slots:
                bbox = base.slot_boxes[s_idx]
                if bbox is not None:
                    pad_x = max(2, int(round(0.08 * bbox.width)))
                    pad_y = max(2, int(round(0.08 * bbox.height)))
                    x0 = max(0, int(round(bbox.x_min)) - pad_x)
                    y0 = max(0, int(round(bbox.y_min)) - pad_y)
                    x1 = min(wp, int(round(bbox.x_max)) + pad_x)
                    y1 = min(hp, int(round(bbox.y_max)) + pad_y)
                    mud_color = rng.uniform(28.0, 62.0, size=(1, 1, 3)).astype(np.float32)
                    canvas[y0:y1, x0:x1] = mud_color
                    vis_map[y0:y1, x0:x1] = 0.0

        # Detector bbox perspective / margin warp -> (64, 128)
        src_pts = np.array(
            [[0.0, 0.0], [float(wp), 0.0], [float(wp), float(hp)], [0.0, float(hp)]],
            dtype=np.float32,
        )
        margin_x = rng.uniform(1.5, 5.5)
        margin_y = rng.uniform(1.0, 3.5)
        dst_base = np.array(
            [
                [margin_x, margin_y],
                [128.0 - margin_x, margin_y],
                [128.0 - margin_x, 64.0 - margin_y],
                [margin_x, 64.0 - margin_y],
            ],
            dtype=np.float32,
        )
        jitter = rng.normal(0.0, jitter_scale * 64.0, size=(4, 2)).astype(np.float32)
        dst_pts = dst_base + jitter

        homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bg_val = float(rng.uniform(35.0, 95.0))
        warped = cv2.warpPerspective(
            canvas,
            homography,
            (128, 64),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(bg_val, bg_val, bg_val),
        )

        # Optional low-resolution distance simulation + motion blur
        quality_factor = 1.0
        if downscale_factor < 0.99:
            sw = max(24, int(round(128 * downscale_factor)))
            sh = max(12, int(round(64 * downscale_factor)))
            small = cv2.resize(warped, (sw, sh), interpolation=cv2.INTER_AREA)
            warped = cv2.resize(small, (128, 64), interpolation=cv2.INTER_LINEAR)
            quality_factor *= float(np.clip(downscale_factor * 1.15, 0.25, 1.0))

        if blur_ksize > 1:
            k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            warped = cv2.GaussianBlur(warped, (k, k), sigmaX=0.0)
            quality_factor *= float(np.clip(1.0 - 0.09 * (k - 1), 0.30, 1.0))

        # Exposure gamma & mild sensor noise
        if abs(exposure_gamma - 1.0) > 1e-3:
            norm_img = np.clip(warped / 255.0, 0.0, 1.0) ** exposure_gamma
            warped = norm_img * 255.0

        noise_std = float(rng.uniform(1.5, 5.5))
        warped = np.clip(
            warped + rng.normal(0.0, noise_std, size=warped.shape).astype(np.float32),
            0.0,
            255.0,
        ).astype(np.uint8)

        # Compute exact per-slot readability in [0, 1]
        readability = np.zeros(NUM_SLOTS, dtype=np.float32)
        for s_idx, bbox in enumerate(base.slot_boxes):
            if bbox is None:
                continue
            x0 = max(0, int(round(bbox.x_min)))
            y0 = max(0, int(round(bbox.y_min)))
            x1 = min(wp, max(x0 + 1, int(round(bbox.x_max))))
            y1 = min(hp, max(y0 + 1, int(round(bbox.y_max))))
            vis_frac = float(vis_map[y0:y1, x0:x1].mean())

            # Check if projected slot center stays inside crop bounds
            cx = 0.5 * (bbox.x_min + bbox.x_max)
            cy = 0.5 * (bbox.y_min + bbox.y_max)
            pt_h = homography @ np.array([cx, cy, 1.0], dtype=np.float64)
            px = pt_h[0] / max(abs(pt_h[2]), 1e-6)
            py = pt_h[1] / max(abs(pt_h[2]), 1e-6)
            in_bounds = 1.0 if (2.0 <= px <= 126.0 and 2.0 <= py <= 62.0) else 0.0

            readability[s_idx] = float(np.clip(vis_frac * quality_factor * in_bounds, 0.0, 1.0))

        return warped, readability

    def _choose_scenario(self, rng: np.random.Generator) -> str:
        if self.mode != "mixed":
            return self.mode
        scenarios = [
            "clean",
            "degraded",
            "complementary_glare",
            "persistent_occlusion",
            "track_mixed",
            "garbage_tail",
            "same_frame_multiview",
        ]
        probs = [0.22, 0.28, 0.20, 0.12, 0.08, 0.05, 0.05]
        return str(rng.choice(scenarios, p=probs))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.stochastic and idx in self._seq_cache:
            return self._seq_cache[idx]

        if self.stochastic:
            self._call_counter += 1
            seed_val = self.seed * 100003 + idx * 97 + self._call_counter * 7919
        else:
            seed_val = self.seed * 100003 + idx * 97
        rng = np.random.default_rng(seed_val)

        scenario = self._choose_scenario(rng)
        base = self._get_base_plate(idx)
        t_len = self.seq_len

        crops_list: list[np.ndarray] = []
        read_list: list[np.ndarray] = []
        valid_mask = np.ones(t_len, dtype=np.float32)
        foreign_mask = np.zeros(t_len, dtype=bool)
        timestamps = np.arange(t_len, dtype=np.float32) * 0.12
        source_frame_ids = np.arange(t_len, dtype=np.int64)
        camera_ids = np.zeros(t_len, dtype=np.int64)
        plate_sides = np.zeros(t_len, dtype=np.int64)
        frame_slot_targets = np.full((t_len, NUM_SLOTS), -1, dtype=np.int64)

        if scenario == "clean":
            for t in range(t_len):
                crop, r = self._warp_plate_crop(base, rng, jitter_scale=0.03)
                crops_list.append(crop)
                read_list.append(r)
                frame_slot_targets[t] = base.slot_ids
            status_str = "READABLE"

        elif scenario == "degraded":
            # Approach trajectory: early frames distant/blurry + moving specular glare patch
            for t in range(t_len):
                prog = float(t) / max(1.0, float(t_len - 1))
                # Early frames harder, later frames sharper (monotonic multi-frame evidence gain)
                downscale = 0.42 + 0.56 * prog
                blur_k = 5 if prog < 0.25 else (3 if prog < 0.60 else 1)
                # Moving glare patch sweeps across plate so individual frames miss some characters
                gx_center = (0.15 + 0.75 * ((t * 0.37) % 1.0))
                gy_center = 0.22 if (t % 2 == 0) else 0.68
                glare_box = (gx_center - 0.16, gy_center - 0.20, gx_center + 0.16, gy_center + 0.20)
                crop, r = self._warp_plate_crop(
                    base,
                    rng,
                    jitter_scale=0.055,
                    moving_glare_box=glare_box if t < t_len - 1 or t_len > 3 else None,
                    blur_ksize=blur_k,
                    downscale_factor=downscale,
                    exposure_gamma=float(rng.uniform(0.82, 1.22)),
                )
                crops_list.append(crop)
                read_list.append(r)
                frame_slot_targets[t] = base.slot_ids

            all_r = np.stack(read_list, axis=0)
            active_slots = [s for s in range(NUM_SLOTS) if base.slot_boxes[s] is not None]
            min_max_r = min(float(all_r[:, s].max()) for s in active_slots)
            status_str = "READABLE" if min_max_r >= 0.30 else "UNREADABLE"

        elif scenario == "complementary_glare":
            # Early frames: lower row completely glared out (upper row clean)
            # Later frames: upper row completely glared out (lower row clean)
            half = max(1, t_len // 2)
            for t in range(t_len):
                glare_part = "lower" if t < half else "upper"
                crop, r = self._warp_plate_crop(
                    base,
                    rng,
                    jitter_scale=0.045,
                    glare_region=glare_part,
                )
                crops_list.append(crop)
                read_list.append(r)
                frame_slot_targets[t] = base.slot_ids
            status_str = "READABLE"

        elif scenario == "persistent_occlusion":
            active_slots = [s for s in range(NUM_SLOTS) if base.slot_boxes[s] is not None]
            occ_slot = int(rng.choice(active_slots))
            for t in range(t_len):
                crop, r = self._warp_plate_crop(
                    base,
                    rng,
                    jitter_scale=0.035,
                    occluded_slots=[occ_slot],
                )
                crops_list.append(crop)
                read_list.append(r)
                frame_slot_targets[t] = base.slot_ids
            status_str = "UNREADABLE"

        elif scenario == "track_mixed":
            # Pick a guaranteed distinct foreign plate from the pool
            other_idx = (idx + 1 + int(rng.integers(0, max(1, len(self._specs) - 1)))) % len(
                self._specs
            )
            foreign = self._get_base_plate(other_idx)
            if foreign.plate_text == base.plate_text and len(self._specs) > 1:
                foreign = self._get_base_plate((idx + 1) % len(self._specs))

            half = max(1, t_len // 2)
            for t in range(t_len):
                if t < half:
                    crop, r = self._warp_plate_crop(base, rng, jitter_scale=0.03)
                    crops_list.append(crop)
                    read_list.append(r)
                    frame_slot_targets[t] = base.slot_ids
                else:
                    crop, r = self._warp_plate_crop(foreign, rng, jitter_scale=0.03)
                    crops_list.append(crop)
                    read_list.append(r)
                    valid_mask[t] = 0.0
                    foreign_mask[t] = True
                    frame_slot_targets[t] = foreign.slot_ids
            status_str = "TRACK_MIXED"

        elif scenario == "garbage_tail":
            n_clean = max(2, t_len // 2)
            for t in range(t_len):
                if t < n_clean:
                    crop, r = self._warp_plate_crop(base, rng, jitter_scale=0.03)
                    crops_list.append(crop)
                    read_list.append(r)
                    frame_slot_targets[t] = base.slot_ids
                else:
                    crops_list.append(self.make_garbage_crop(rng, kind=t))
                    read_list.append(np.zeros(NUM_SLOTS, dtype=np.float32))
                    valid_mask[t] = 0.0
                    plate_sides[t] = 2
            status_str = "READABLE"

        elif scenario == "same_frame_multiview":
            gammas = [1.0, 0.65, 1.45, 0.90]
            phys_id = 0
            for t in range(t_len):
                if t > 0 and t % 2 == 0:
                    phys_id += 1
                source_frame_ids[t] = phys_id
                timestamps[t] = float(phys_id) * 0.15
                crop, r = self._warp_plate_crop(
                    base,
                    rng,
                    jitter_scale=0.025,
                    exposure_gamma=gammas[t % len(gammas)],
                )
                crops_list.append(crop)
                read_list.append(r)
                frame_slot_targets[t] = base.slot_ids
            status_str = "READABLE"

        else:
            raise ValueError(f"Unknown MicroPlateSequenceDataset mode: {scenario!r}")

        crops_rgb = np.stack(crops_list, axis=0)  # (T, 64, 128, 3) uint8
        slot_readability = np.stack(read_list, axis=0)  # (T, 12) float32

        sample_dict = {
            "sequence_id": f"{scenario}_{idx:05d}",
            "plate_text": base.plate_text,
            "plate_type": base.plate_type.value,
            "crops_rgb": crops_rgb,
            "timestamps_s": timestamps,
            "source_frame_ids": source_frame_ids,
            "camera_ids": camera_ids,
            "plate_sides": plate_sides,
            "frame_valid_mask": valid_mask,
            "is_foreign_plate": foreign_mask,
            "slot_readability": slot_readability,
            "sequence_target_status": status_str,
            "frame_slot_targets": frame_slot_targets,
        }
        adapted = adapt_sequence_sample(sample_dict, self.vocab)
        if not self.stochastic:
            self._seq_cache[idx] = adapted
        return adapted


class StreamingFastSynthDataset(Dataset[dict[str, Any]]):
    """Zero-RAM-overhead multi-worker PyTorch Dataset backed by FastCropSequenceSynthesizer."""

    def __init__(
        self,
        virtual_size: int = 200_000,
        seq_len: int = 8,
        mode: str = "mixed",
        base_seed: int = 100_000,
        variable_length: bool = True,
        vocab: PlateVocab | None = None,
    ) -> None:
        self.virtual_size = virtual_size
        self.seq_len = seq_len
        self.mode = mode
        self.base_seed = base_seed
        self.variable_length = variable_length
        self.vocab = vocab or PlateVocab()
        self._synth: Any = None
        self._call_counter: int = 0

    def __len__(self) -> int:
        return self.virtual_size

    def _ensure_synth(self) -> Any:
        if self._synth is None:
            cv2.setNumThreads(0)
            from vpocr_synth.export.fast_synthesizer import FastCropSequenceSynthesizer

            self._synth = FastCropSequenceSynthesizer()
        return self._synth

    def __getitem__(self, idx: int) -> dict[str, Any]:
        synth = self._ensure_synth()
        # Mix deterministic index with per-worker epoch counter so repeated epochs see fresh plates
        w_info = torch.utils.data.get_worker_info()
        w_off = (w_info.id * 1_000_003) if w_info is not None else 0
        self._call_counter += 1
        seed = self.base_seed + int(idx) + (self._call_counter // max(1, self.virtual_size)) * 10_000_019 + w_off

        if self.variable_length:
            # Sample variable frame lengths 5..12 deterministically from seed
            t_len = 5 + (seed % 8)
        else:
            t_len = self.seq_len

        sample = synth.synthesize_one(seed=seed, num_frames=t_len, mode=self.mode)
        return adapt_sequence_sample(sample, self.vocab)

