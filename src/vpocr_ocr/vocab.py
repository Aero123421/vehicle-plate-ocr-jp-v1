"""12-slot bijective Japanese license plate vocabulary, masks, and MLIT grammar."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import torch

from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.models import PlateText, PlateType
from vpocr_synth.plate.validator import PlateValidator

PAD_TOKEN = "<PAD>"
DOT_TOKEN = "・"
NUM_SLOTS = 12

PLATE_TYPES: tuple[PlateType, ...] = (
    PlateType.STANDARD_PRIVATE,
    PlateType.STANDARD_COMMERCIAL,
    PlateType.KEI_PRIVATE,
    PlateType.KEI_COMMERCIAL,
    PlateType.GRAPHIC,
)

STATUS_NAMES: tuple[str, ...] = ("READABLE", "UNREADABLE", "TRACK_MIXED")
STATUS_TO_ID: dict[str, int] = {
    "READABLE": 0,
    "UNREADABLE": 1,
    "TRACK_MIXED": 2,
    "CONFLICT": 2,  # Alias mapped to TRACK_MIXED class index for training/eval
}


def canonicalize_region(region: str) -> str:
    """Canonicalize regional office aliases (e.g. '袖ヶ浦' -> '袖ケ浦')."""
    if region == "袖ヶ浦":
        return "袖ケ浦"
    return region


_COMPACT_PLATE_RE = re.compile(
    r"^([^\d\sA-Za-z・\-]{1,4})\s*([0-9][0-9A-Z]{0,2})\s*([^\d\s・\-])\s*([0-9・\-]{1,5})$"
)
_FULLWIDTH_ASCII_TRANS = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ−ー―–—　",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ----- ",
)


def parse_plate_string(
    text: str | PlateText,
    vocab: PlateVocab | None = None,
) -> PlateText | None:
    """Parse structured `PlateText` or formatted Japanese plate string into validated `PlateText`."""
    voc = vocab or PlateVocab()
    if isinstance(text, PlateText):
        pt = PlateText(
            region=canonicalize_region(text.region.strip()),
            classification=text.classification.strip().upper(),
            hiragana=text.hiragana.strip(),
            serial=int(text.serial),
        )
    elif isinstance(text, str):
        norm = text.translate(_FULLWIDTH_ASCII_TRANS).strip()
        if not norm:
            return None
        parts = norm.split()
        if len(parts) == 4:
            reg_s, cls_s, kana_s, ser_s = parts
        else:
            m = _COMPACT_PLATE_RE.match(norm)
            if m is None:
                return None
            reg_s, cls_s, kana_s, ser_s = m.groups()
        if any(ch not in "0123456789・-" for ch in ser_s):
            return None
        digits = "".join(ch for ch in ser_s if ch.isdigit())
        if not (1 <= len(digits) <= 4):
            return None
        serial_int = int(digits)
        if not (1 <= serial_int <= 9999):
            return None
        pt = PlateText(
            region=canonicalize_region(reg_s),
            classification=cls_s.upper(),
            hiragana=kana_s,
            serial=serial_int,
        )
    else:
        return None

    try:
        _ = voc.encode(pt)
    except (KeyError, ValueError):
        return None
    if not PlateValidator.is_valid(pt):
        return None
    return pt


class PlateVocab:
    """Bijective 12-slot codec and per-slot MLIT token masks for Japanese plates."""

    def __init__(self, charset_path: str | Path | None = None) -> None:
        if charset_path is not None:
            with open(charset_path, "r", encoding="utf-8") as f:
                raw_chars = [line.strip("\r\n") for line in f if line.strip("\r\n")]
        else:
            raw_chars = list(PlateDataLoader.load_charset())

        tokens: list[str] = [PAD_TOKEN]
        seen: set[str] = {PAD_TOKEN}
        for ch in raw_chars:
            if ch not in seen:
                tokens.append(ch)
                seen.add(ch)
        if DOT_TOKEN not in seen:
            tokens.append(DOT_TOKEN)
            seen.add(DOT_TOKEN)

        self.tokens: tuple[str, ...] = tuple(tokens)
        self.token_to_id: dict[str, int] = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.id_to_token: dict[int, str] = {idx: tok for idx, tok in enumerate(self.tokens)}

        self.pad_id: int = 0
        self.dot_id: int = self.token_to_id[DOT_TOKEN]

        self.plate_types: tuple[PlateType, ...] = PLATE_TYPES
        self.plate_type_to_id: dict[str, int] = {
            pt.value: idx for idx, pt in enumerate(self.plate_types)
        }

        self.regions: tuple[str, ...] = tuple(
            canonicalize_region(r.name) for r in PlateDataLoader.load_regions()
        )
        self.region_set: frozenset[str] = frozenset(self.regions)

        # Precompute (12, vocab_size) boolean mask of MLIT-allowed tokens per slot
        self.slot_allowed_mask_np: np.ndarray = self._build_slot_allowed_mask()
        self.slot_allowed_mask: torch.Tensor = torch.from_numpy(self.slot_allowed_mask_np)

        # Precompute (num_regions, 4) integer slot table for exact region projection
        self.region_slot_ids_np: np.ndarray = np.array(
            [self._encode_region_slots(r) for r in self.regions], dtype=np.int64
        )
        self.region_slot_ids: torch.Tensor = torch.from_numpy(self.region_slot_ids_np)

        # Precompute ASCII SOFA Latin kana token IDs (slot 7; excludes Japanese hiragana 'よ')
        # and classification Latin letter token IDs (slots 5..6) for road log-prior calibration
        self.sofa_latin_slot7_ids_np: np.ndarray = np.array(
            sorted(
                self.token_to_id[ch]
                for ch in PlateValidator.SOFA_CHARS
                if ch.isascii() and ch.isalpha() and ch in self.token_to_id
            ),
            dtype=np.int64,
        )
        self.latin_cls_ids_np: np.ndarray = np.array(
            sorted(
                self.token_to_id[ch]
                for ch in PlateValidator.ALLOWED_CLASSIFICATION_LETTERS
                if ch in self.token_to_id
            ),
            dtype=np.int64,
        )

    @property
    def size(self) -> int:
        """Total vocabulary size including <PAD> at index 0."""
        return len(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def _encode_region_slots(self, region: str) -> list[int]:
        reg = canonicalize_region(region)
        if not (1 <= len(reg) <= 4):
            raise ValueError(f"Region length must be 1..4, got {region!r}")
        ids = [self.token_to_id[c] for c in reg]
        return ids + [self.pad_id] * (4 - len(ids))

    def _build_slot_allowed_mask(self) -> np.ndarray:
        mask = np.zeros((NUM_SLOTS, self.size), dtype=bool)

        # Slots 0..3: valid characters appearing at each position across all 138 regions
        for reg in self.regions:
            for pos in range(4):
                if pos < len(reg):
                    ch = reg[pos]
                    if ch in self.token_to_id:
                        mask[pos, self.token_to_id[ch]] = True
                else:
                    mask[pos, self.pad_id] = True
        # Ensure <PAD> is strictly disallowed at slot 0
        mask[0, self.pad_id] = False

        # Slot 4: classification 1st char -> digits '1'..'9' only
        for d in "123456789":
            mask[4, self.token_to_id[d]] = True

        # Slots 5..6: digits '0'..'9' + 10 allowed MLIT letters + <PAD>
        cls_tail_chars = set("0123456789") | PlateValidator.ALLOWED_CLASSIFICATION_LETTERS
        for s in (5, 6):
            mask[s, self.pad_id] = True
            for ch in cls_tail_chars:
                if ch in self.token_to_id:
                    mask[s, self.token_to_id[ch]] = True

        # Slot 7: all valid kana across categories + SOFA letters (no <PAD>, no お/し/へ/ん)
        valid_slot7: set[str] = set()
        for pt in PLATE_TYPES:
            valid_slot7 |= PlateValidator.get_valid_hiragana_for_type(pt)
        valid_slot7 |= PlateValidator.SOFA_CHARS
        valid_slot7 -= PlateValidator.EXCLUDED_HIRAGANA
        for ch in valid_slot7:
            if ch in self.token_to_id:
                mask[7, self.token_to_id[ch]] = True

        # Slots 8..10: serial leading/middle slots -> digits '0'..'9' + '・'
        for s in (8, 9, 10):
            mask[s, self.dot_id] = True
            for d in "0123456789":
                mask[s, self.token_to_id[d]] = True

        # Slot 11: final serial slot -> digits '0'..'9' only (serial >= 1 always has last digit)
        for d in "0123456789":
            mask[11, self.token_to_id[d]] = True

        return mask

    def encode_fields(
        self,
        region: str,
        classification: str,
        hiragana: str,
        serial: int | str,
    ) -> list[int]:
        """Encode structured Japanese plate fields into 12 integer slot IDs."""
        reg = canonicalize_region(region)
        if not (1 <= len(reg) <= 4):
            raise ValueError(f"Invalid region length: {region!r}")
        if not (1 <= len(classification) <= 3):
            raise ValueError(f"Invalid classification length: {classification!r}")
        if len(hiragana) != 1:
            raise ValueError(f"Invalid hiragana length: {hiragana!r}")

        if isinstance(serial, str):
            digits = "".join(ch for ch in serial if ch.isdigit())
            if not digits:
                raise ValueError(f"Serial has no digits: {serial!r}")
            serial_int = int(digits)
        else:
            serial_int = int(serial)

        if not (1 <= serial_int <= 9999):
            raise ValueError(f"Serial out of range 1..9999: {serial_int}")

        reg_ids = [self.token_to_id[c] for c in reg] + [self.pad_id] * (4 - len(reg))
        cls_ids = [self.token_to_id[c] for c in classification] + [self.pad_id] * (
            3 - len(classification)
        )
        kana_ids = [self.token_to_id[hiragana]]

        serial_str = str(serial_int)
        serial_tokens = [DOT_TOKEN] * (4 - len(serial_str)) + list(serial_str)
        serial_ids = [self.token_to_id[c] for c in serial_tokens]

        slots = reg_ids + cls_ids + kana_ids + serial_ids
        assert len(slots) == NUM_SLOTS
        return slots

    def encode(self, plate: PlateText | tuple[str, str, str, int | str]) -> list[int]:
        """Bijectively encode PlateText (or 4-tuple) into 12 slot token IDs."""
        if isinstance(plate, PlateText):
            return self.encode_fields(
                plate.region,
                plate.classification,
                plate.hiragana,
                plate.serial,
            )
        reg, cls_num, kana, serial = plate
        return self.encode_fields(reg, cls_num, kana, serial)

    def decode_raw_fields(
        self, slot_ids: Sequence[int] | np.ndarray | torch.Tensor
    ) -> dict[str, str]:
        """Decode 12 slot IDs into raw field strings without raising on invalid combinations."""
        if isinstance(slot_ids, torch.Tensor):
            ids = slot_ids.detach().cpu().tolist()
        elif isinstance(slot_ids, np.ndarray):
            ids = slot_ids.tolist()
        else:
            ids = [int(x) for x in slot_ids]

        if len(ids) != NUM_SLOTS:
            raise ValueError(f"Expected {NUM_SLOTS} slot IDs, got {len(ids)}")

        toks = [self.id_to_token.get(int(i), PAD_TOKEN) for i in ids]

        reg_raw = "".join(t for t in toks[0:4] if t != PAD_TOKEN)
        reg_raw = canonicalize_region(reg_raw)
        cls_raw = "".join(t for t in toks[4:7] if t != PAD_TOKEN)
        kana_raw = "" if toks[7] == PAD_TOKEN else toks[7]
        serial_digits = "".join(
            t for t in toks[8:12] if t not in (PAD_TOKEN, DOT_TOKEN)
        )
        serial_slots_str = "".join("_" if t == PAD_TOKEN else t for t in toks[8:12])
        return {
            "region": reg_raw,
            "classification": cls_raw,
            "hiragana": kana_raw,
            "serial_digits": serial_digits,
            "serial_slots": serial_slots_str,
        }

    def decode(
        self,
        slot_ids: Sequence[int] | np.ndarray | torch.Tensor,
        strict: bool = True,
    ) -> PlateText | None:
        """Bijectively decode 12 slot IDs back into a PlateText object.

        Args:
            slot_ids: Sequence of 12 token indices.
            strict: If True, raises ValueError when slots do not form a parseable PlateText.
                If False, returns None on unparseable slots.
        """
        raw = self.decode_raw_fields(slot_ids)
        region = raw["region"]
        classification = raw["classification"]
        hiragana = raw["hiragana"]
        serial_digits = raw["serial_digits"]

        if not region or not classification or len(hiragana) != 1 or not serial_digits.isdigit():
            if strict:
                raise ValueError(f"Cannot decode slots into valid PlateText: {raw}")
            return None

        serial_int = int(serial_digits)
        if not (1 <= serial_int <= 9999):
            if strict:
                raise ValueError(f"Decoded serial {serial_int} outside 1..9999: {raw}")
            return None

        return PlateText(
            region=region,
            classification=classification,
            hiragana=hiragana,
            serial=serial_int,
        )

    def try_decode(
        self, slot_ids: Sequence[int] | np.ndarray | torch.Tensor
    ) -> PlateText | None:
        """Non-throwing wrapper around decode(..., strict=False)."""
        return self.decode(slot_ids, strict=False)

    def encode_plate_type(self, plate_type: PlateType | str) -> int:
        """Encode PlateType enum or value string to 0..4 class index."""
        key = plate_type.value if isinstance(plate_type, PlateType) else str(plate_type)
        if key not in self.plate_type_to_id:
            raise ValueError(f"Unknown plate_type: {plate_type!r}")
        return self.plate_type_to_id[key]

    def decode_plate_type(self, type_id: int) -> PlateType:
        """Decode 0..4 class index to PlateType enum."""
        if not (0 <= int(type_id) < len(self.plate_types)):
            raise ValueError(f"plate_type index out of bounds: {type_id}")
        return self.plate_types[int(type_id)]

    def mask_logits(
        self, slot_logits: torch.Tensor, neg_inf: float = -1e9
    ) -> torch.Tensor:
        """Apply per-slot MLIT allowed-token mask to logits of shape (..., 12, vocab_size)."""
        allowed = self.slot_allowed_mask.to(device=slot_logits.device)
        return slot_logits.masked_fill(~allowed, neg_inf)


@dataclass
class TypeMarginalizedDecodeResult:
    """Output of Type-Marginalized Joint MLIT Grammar Decoding (`MF-JPPlateOCR v3`)."""

    slot_ids: list[int]
    plate_text: PlateText | None
    best_plate_type: PlateType
    plate_type_probs: dict[str, float]
    plate_type_candidates: list[str]
    joint_log_score: float
    gamma_chroma: float
    top2_margin: float = 10.0
    field_margins: dict[str, float] | None = None


KEI_ALLOWED_FIRST_DIGITS: frozenset[str] = frozenset({"4", "5", "6", "7", "8"})


class JPPlateGrammar:
    """MLIT structural grammar validator and exact log-probability constrained decoder.

    Important: Constrained decoding operates on log-probabilities to select valid MLIT strings
    while callers retain raw unmasked softmax probabilities for honest uncertainty calibration.
    """

    def __init__(self, vocab: PlateVocab | None = None) -> None:
        self.vocab = vocab or PlateVocab()
        self._zero_id = self.vocab.token_to_id["0"]
        self._nine_id = self.vocab.token_to_id["9"]
        self._nonzero_digit_ids = [self.vocab.token_to_id[d] for d in "123456789"]
        self._all_digit_ids = [self.vocab.token_to_id[d] for d in "0123456789"]
        self._latin_cls_ids = [
            self.vocab.token_to_id[ch]
            for ch in sorted(PlateValidator.ALLOWED_CLASSIFICATION_LETTERS)
        ]

        # Mild region length log-prior applied ONCE per region (2/3-char = 0.0, 1/4-char = -0.15)
        self._region_log_prior: np.ndarray = np.array(
            [0.0 if len(r) in (2, 3) else -0.15 for r in self.vocab.regions],
            dtype=np.float64,
        )

        # Precompute (5, 10, K) MLIT compatibility log-potentials for Type-Marginalized Decoding
        # Index d_idx=0..8 corresponds to d1 in '1'..'9'; d_idx=9 corresponds to construction '0'-series!
        self._valid_slot7_ids: np.ndarray = np.flatnonzero(self.vocab.slot_allowed_mask_np[7]).astype(np.int64)
        num_types = len(self.vocab.plate_types)
        num_k = int(self._valid_slot7_ids.shape[0])
        psi = np.full((num_types, 10, num_k), -1e12, dtype=np.float64)

        std_priv_kana = PlateValidator.get_valid_hiragana_for_type(PlateType.STANDARD_PRIVATE)
        std_comm_kana = PlateValidator.get_valid_hiragana_for_type(PlateType.STANDARD_COMMERCIAL)
        d1_chars = "1234567890"
        for c_idx, pt in enumerate(self.vocab.plate_types):
            valid_kana_set = PlateValidator.get_valid_hiragana_for_type(pt)
            is_kei_cat = pt in (PlateType.KEI_PRIVATE, PlateType.KEI_COMMERCIAL)
            for d_idx, d_ch in enumerate(d1_chars):
                if d_ch == "0":
                    # 0-series large special construction machinery (`大型特殊建設機械`: 0, 00..09, 00x, 09x)
                    # exists only on standard private/commercial plates (never kei, never graphic).
                    if pt not in (PlateType.STANDARD_PRIVATE, PlateType.STANDARD_COMMERCIAL):
                        continue
                elif is_kei_cat and d_ch not in KEI_ALLOWED_FIRST_DIGITS:
                    continue
                for k_idx, tok_id in enumerate(self._valid_slot7_ids):
                    k_ch = self.vocab.id_to_token[int(tok_id)]
                    if k_ch not in valid_kana_set:
                        continue
                    # Graphic plates exist for standard private (no/mono rim), standard commercial
                    # (green rim, e.g. 沖縄 251 き 84-51), and kei private (yellow rim).
                    # Only kei-exclusive kana (outside std_priv & std_comm) require kei first digits.
                    if (
                        pt == PlateType.GRAPHIC
                        and k_ch not in std_priv_kana
                        and k_ch not in std_comm_kana
                        and d_ch not in KEI_ALLOWED_FIRST_DIGITS
                    ):
                        continue
                    bonus = 0.0
                    if d_ch == "0":
                        # Rare construction machinery fleet log-prior (-2.50 nats)
                        bonus = -2.50
                    elif pt == PlateType.STANDARD_COMMERCIAL and d_ch in ("5", "7"):
                        bonus = -0.45
                    elif pt == PlateType.KEI_COMMERCIAL and k_ch in ("ろ", "わ"):
                        bonus = -0.45
                    elif pt == PlateType.GRAPHIC and k_ch in std_comm_kana and d_ch in ("3",):
                        bonus = -0.15
                    psi[c_idx, d_idx, k_idx] = bonus

        self._psi_mlit_log_potentials: np.ndarray = psi
        # Mild uninformative fleet base log-prior active only as color_precision -> 0 (monochrome IR)
        self._base_type_log_prior: np.ndarray = np.array(
            [0.0, -1.25, -0.15, -1.35, -1.80], dtype=np.float64
        )
        # Expected luminance polarity (+1: bright bg / dark char, -1: dark bg / bright char)
        self._type_expected_polarity: np.ndarray = np.array(
            [1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64
        )
        self._fleet_cache: dict[
            tuple[str, ...], tuple[np.ndarray, list[PlateText], np.ndarray]
        ] = {}

    def encode_registered_fleet(
        self,
        registered_plates: Sequence[str | PlateText],
    ) -> tuple[np.ndarray, list[PlateText], np.ndarray]:
        """Encode validated `registered_plates` into `((R, 12) slot_ids, plates, (R, 5) type_mask)`."""
        cache_key = tuple(
            p.full_text if isinstance(p, PlateText) else str(p) for p in registered_plates
        )
        cached = self._fleet_cache.get(cache_key)
        if cached is not None:
            return cached

        slot_rows: list[list[int]] = []
        plates: list[PlateText] = []
        type_rows: list[list[bool]] = []
        seen_full: set[str] = set()
        for item in registered_plates:
            pt = parse_plate_string(item, vocab=self.vocab)
            if pt is None or pt.full_text in seen_full:
                continue
            compat = [PlateValidator.is_valid(pt, ptype) for ptype in self.vocab.plate_types]
            if not any(compat):
                continue
            seen_full.add(pt.full_text)
            slot_rows.append(self.vocab.encode(pt))
            plates.append(pt)
            type_rows.append(compat)

        if not plates:
            res = (
                np.zeros((0, NUM_SLOTS), dtype=np.int64),
                [],
                np.zeros((0, len(self.vocab.plate_types)), dtype=bool),
            )
        else:
            res = (
                np.asarray(slot_rows, dtype=np.int64),
                plates,
                np.asarray(type_rows, dtype=bool),
            )
        if len(self._fleet_cache) > 32:
            self._fleet_cache.clear()
        self._fleet_cache[cache_key] = res
        return res

    def is_valid(
        self,
        plate_text: PlateText | None,
        plate_type: PlateType | str | None = None,
    ) -> bool:
        """Return True iff plate_text satisfies official MLIT validation rules."""
        if plate_text is None:
            return False
        pt_enum: PlateType | None = None
        if isinstance(plate_type, PlateType):
            pt_enum = plate_type
        elif isinstance(plate_type, str) and plate_type in self.vocab.plate_type_to_id:
            pt_enum = self.vocab.decode_plate_type(self.vocab.plate_type_to_id[plate_type])
        return PlateValidator.is_valid(plate_text, pt_enum)

    def __call__(
        self,
        plate_text: PlateText | None,
        plate_type: PlateType | str | None = None,
    ) -> bool:
        return self.is_valid(plate_text, plate_type)

    def _best_and_runnerup_tails(
        self, log_probs: np.ndarray
    ) -> tuple[float, list[int], float, float, list[int], float]:
        """Return `(best_std_score, best_std_slots, std_gap, best_zero_score, best_zero_slots, zero_gap)`."""
        pad = self.vocab.pad_id
        # --- Standard d1 in '1'..'9' tails ---
        sc_1d = float(log_probs[5, pad] + log_probs[6, pad]) - 0.30
        d5_scores = log_probs[5, self._all_digit_ids]
        d5_order = np.argsort(-d5_scores)
        best_c5_num = int(self._all_digit_ids[int(d5_order[0])])
        sc_2d = float(d5_scores[int(d5_order[0])] + log_probs[6, pad]) - 0.15
        sc_2d_alt = (
            float(d5_scores[int(d5_order[1])] + log_probs[6, pad]) - 0.15
            if len(d5_order) > 1
            else -1e12
        )

        allowed_3d = np.asarray(self._all_digit_ids + self._latin_cls_ids, dtype=np.int64)
        n_num = len(self._all_digit_ids)
        is_lat = np.arange(len(allowed_3d)) >= n_num
        lat_pen = np.where(is_lat[:, None] | is_lat[None, :], -0.20, 0.0)
        mat_3d = (
            log_probs[5, allowed_3d][:, None]
            + log_probs[6, allowed_3d][None, :]
            + lat_pen
        )
        flat_3d = mat_3d.reshape(-1)
        top2_3d_idx = np.argpartition(-flat_3d, 1)[:2]
        if flat_3d[top2_3d_idx[1]] > flat_3d[top2_3d_idx[0]]:
            top2_3d_idx = top2_3d_idx[::-1]
        i3_best = int(top2_3d_idx[0])
        r3, c3 = divmod(i3_best, len(allowed_3d))
        sc_3d = float(flat_3d[i3_best])
        sc_3d_alt = float(flat_3d[int(top2_3d_idx[1])])
        slots_3d = [int(allowed_3d[r3]), int(allowed_3d[c3])]

        std_cands = [
            (sc_3d, slots_3d),
            (sc_2d, [best_c5_num, pad]),
            (sc_1d, [pad, pad]),
        ]
        std_cands.sort(key=lambda x: x[0], reverse=True)
        best_std_sc, best_std_slots = std_cands[0]
        runner_std_sc = max(std_cands[1][0], sc_3d_alt if best_std_slots == slots_3d else -1e12, sc_2d_alt if best_std_slots == [best_c5_num, pad] else -1e12)
        std_gap = float(max(0.0, best_std_sc - runner_std_sc))

        # --- Construction '0'-series tails ('0', '00'..'09', '00x', '09x') ---
        z_c5_allowed = np.asarray([self._zero_id, self._nine_id], dtype=np.int64)
        z_lat_pen = np.where(is_lat[None, :], -0.20, 0.0)
        mat_z3 = (
            log_probs[5, z_c5_allowed][:, None]
            + log_probs[6, allowed_3d][None, :]
            + z_lat_pen
        )
        flat_z3 = mat_z3.reshape(-1)
        top2_z3 = np.argpartition(-flat_z3, 1)[:2]
        if flat_z3[top2_z3[1]] > flat_z3[top2_z3[0]]:
            top2_z3 = top2_z3[::-1]
        iz_best = int(top2_z3[0])
        rz, cz = divmod(iz_best, len(allowed_3d))
        sc_z3 = float(flat_z3[iz_best])
        sc_z3_alt = float(flat_z3[int(top2_z3[1])])
        slots_z3 = [int(z_c5_allowed[rz]), int(allowed_3d[cz])]

        zero_cands = [
            (sc_z3, slots_z3),
            (sc_2d, [best_c5_num, pad]),
            (sc_1d, [pad, pad]),
        ]
        zero_cands.sort(key=lambda x: x[0], reverse=True)
        best_zero_sc, best_zero_slots = zero_cands[0]
        runner_zero_sc = max(
            zero_cands[1][0],
            sc_z3_alt if best_zero_slots == slots_z3 else -1e12,
            sc_2d_alt if best_zero_slots == [best_c5_num, pad] else -1e12,
        )
        zero_gap = float(max(0.0, best_zero_sc - runner_zero_sc))
        return best_std_sc, best_std_slots, std_gap, best_zero_sc, best_zero_slots, zero_gap

    def _best_and_runnerup_serial(
        self, log_probs: np.ndarray
    ) -> tuple[float, list[int], float]:
        """Exact MLIT serial max over 4/3/2/1-digit patterns with whole-field prior and runner-up gap."""
        dot = self.vocab.dot_id
        nz_ids = np.asarray(self._nonzero_digit_ids, dtype=np.int64)
        any_ids = np.asarray(self._all_digit_ids, dtype=np.int64)

        best_nz_at: list[int] = []
        gap_nz_at: list[float] = []
        best_any_at: list[int] = []
        gap_any_at: list[float] = []
        for s in range(8, 12):
            nz_sc = log_probs[s, nz_ids]
            nz_ord = np.argsort(-nz_sc)
            best_nz_at.append(int(nz_ids[int(nz_ord[0])]))
            gap_nz_at.append(float(nz_sc[int(nz_ord[0])] - nz_sc[int(nz_ord[1])]))

            any_sc = log_probs[s, any_ids]
            any_ord = np.argsort(-any_sc)
            best_any_at.append(int(any_ids[int(any_ord[0])]))
            gap_any_at.append(float(any_sc[int(any_ord[0])] - any_sc[int(any_ord[1])]))

        ser_4 = [best_nz_at[0], best_any_at[1], best_any_at[2], best_any_at[3]]
        sc_4 = sum(float(log_probs[8 + i, ser_4[i]]) for i in range(4)) + 0.00
        in_gap_4 = min(gap_nz_at[0], gap_any_at[1], gap_any_at[2], gap_any_at[3])

        ser_3 = [dot, best_nz_at[1], best_any_at[2], best_any_at[3]]
        sc_3 = sum(float(log_probs[8 + i, ser_3[i]]) for i in range(4)) - 0.08
        in_gap_3 = min(gap_nz_at[1], gap_any_at[2], gap_any_at[3])

        ser_2 = [dot, dot, best_nz_at[2], best_any_at[3]]
        sc_2 = sum(float(log_probs[8 + i, ser_2[i]]) for i in range(4)) - 0.16
        in_gap_2 = min(gap_nz_at[2], gap_any_at[3])

        ser_1 = [dot, dot, dot, best_nz_at[3]]
        sc_1 = sum(float(log_probs[8 + i, ser_1[i]]) for i in range(4)) - 0.35
        in_gap_1 = gap_nz_at[3]

        pats = [
            (sc_4, ser_4, in_gap_4),
            (sc_3, ser_3, in_gap_3),
            (sc_2, ser_2, in_gap_2),
            (sc_1, ser_1, in_gap_1),
        ]
        pats.sort(key=lambda x: x[0], reverse=True)
        best_sc, best_slots, within_gap = pats[0]
        pat_gap = float(best_sc - pats[1][0])
        return best_sc, best_slots, float(max(0.0, min(pat_gap, within_gap)))

    def constrain_slot_ids(
        self,
        slot_logits: np.ndarray | torch.Tensor,
        plate_type: PlateType | str | None = None,
    ) -> list[int]:
        """Find the maximum-likelihood MLIT-legal 12-slot sequence for single sample (12, V)."""
        if isinstance(slot_logits, torch.Tensor):
            logits_np = slot_logits.detach().double().cpu().numpy()
        else:
            logits_np = np.asarray(slot_logits, dtype=np.float64)

        if logits_np.shape != (NUM_SLOTS, self.vocab.size):
            raise ValueError(f"Expected shape (12, {self.vocab.size}), got {logits_np.shape}")

        # Log-softmax for numerically calibrated joint scoring across slots
        shifted = logits_np - logits_np.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True) + 1e-15)

        # 1. Region (slots 0..3): exact max over all 138 official regions
        reg_table = self.vocab.region_slot_ids_np  # (138, 4)
        reg_scores = (
            log_probs[0, reg_table[:, 0]]
            + log_probs[1, reg_table[:, 1]]
            + log_probs[2, reg_table[:, 2]]
            + log_probs[3, reg_table[:, 3]]
            + self._region_log_prior
        )
        best_reg_slots = reg_table[int(np.argmax(reg_scores))].tolist()

        # 2. Classification (slots 4..6): compare 1..9 standard & 0-series construction MLIT codes
        pt_enum: PlateType | None = None
        if isinstance(plate_type, PlateType):
            pt_enum = plate_type
        elif isinstance(plate_type, str) and plate_type in self.vocab.plate_type_to_id:
            pt_enum = self.vocab.decode_plate_type(self.vocab.plate_type_to_id[plate_type])

        best_std_sc, best_std_tail, _, best_zero_sc, best_zero_tail, _ = (
            self._best_and_runnerup_tails(log_probs)
        )
        if pt_enum in (PlateType.KEI_PRIVATE, PlateType.KEI_COMMERCIAL):
            allowed_d1_ids = [self.vocab.token_to_id[d] for d in sorted(KEI_ALLOWED_FIRST_DIGITS)]
        else:
            allowed_d1_ids = self._nonzero_digit_ids

        best_c4 = allowed_d1_ids[int(np.argmax(log_probs[4, allowed_d1_ids]))]
        std_total = float(log_probs[4, best_c4]) + best_std_sc
        best_cls_slots = [best_c4] + best_std_tail

        if pt_enum in (None, PlateType.STANDARD_PRIVATE, PlateType.STANDARD_COMMERCIAL):
            zero_total = float(log_probs[4, self._zero_id]) + best_zero_sc - 2.50
            if zero_total > std_total:
                best_cls_slots = [self._zero_id] + best_zero_tail

        # 3. Hiragana (slot 7): valid kana (optionally narrowed by plate_type)
        if pt_enum is not None:
            valid_kana = PlateValidator.get_valid_hiragana_for_type(pt_enum)
            kana_ids = [
                self.vocab.token_to_id[c]
                for c in valid_kana
                if c in self.vocab.token_to_id
            ]
        else:
            kana_ids = np.flatnonzero(self.vocab.slot_allowed_mask_np[7]).tolist()

        best_kana = kana_ids[int(np.argmax(log_probs[7, kana_ids]))]

        # 4. Serial (slots 8..11): exact max over 4/3/2/1-digit patterns
        _, best_ser_slots, _ = self._best_and_runnerup_serial(log_probs)

        return best_reg_slots + best_cls_slots + [best_kana] + best_ser_slots

    def decode_type_marginalized(
        self,
        slot_logits: np.ndarray | torch.Tensor,
        plate_type_logits: np.ndarray | torch.Tensor,
        color_precision_mean: float,
        polarity_logit: float = 0.0,
        polarity_precision: float = 0.0,
        plate_side: int = 0,
        chroma_tau: float = 0.35,
        candidate_mass_threshold: float = 0.90,
    ) -> TypeMarginalizedDecodeResult:
        """Exact LogSumExp marginalized MLIT joint decoder over all 5 `PlateType`s (`< 0.03 ms`).

        Eliminates hard `argmax(plate_type)` kana masking failures under 850nm/940nm monochrome IR,
        faded yellow kei plates, and nighttime `字光式` inverted polarity.
        """
        del plate_side  # Reserved for optional rear fūin metadata priors
        if isinstance(slot_logits, torch.Tensor):
            logits_np = slot_logits.detach().double().cpu().numpy()
        else:
            logits_np = np.asarray(slot_logits, dtype=np.float64)

        if isinstance(plate_type_logits, torch.Tensor):
            type_np = plate_type_logits.detach().double().cpu().numpy()
        else:
            type_np = np.asarray(plate_type_logits, dtype=np.float64)

        if logits_np.shape != (NUM_SLOTS, self.vocab.size):
            raise ValueError(f"Expected slot_logits shape (12, {self.vocab.size}), got {logits_np.shape}")
        if type_np.shape != (len(self.vocab.plate_types),):
            raise ValueError(
                f"Expected plate_type_logits shape ({len(self.vocab.plate_types)},), got {type_np.shape}"
            )

        # Calibrated per-slot log-probabilities
        shifted = logits_np - np.max(logits_np, axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + 1e-15)

        # 1. Exact MLIT Region max over all 138 official regions (slots 0..3)
        reg_table = self.vocab.region_slot_ids_np
        reg_scores = (
            log_probs[0, reg_table[:, 0]]
            + log_probs[1, reg_table[:, 1]]
            + log_probs[2, reg_table[:, 2]]
            + log_probs[3, reg_table[:, 3]]
            + self._region_log_prior
        )
        reg_top2 = np.argpartition(-reg_scores, 1)[:2]
        if reg_scores[reg_top2[1]] > reg_scores[reg_top2[0]]:
            reg_top2 = reg_top2[::-1]
        best_reg_idx = int(reg_top2[0])
        best_reg_score = float(reg_scores[best_reg_idx])
        reg_margin = float(max(0.0, best_reg_score - float(reg_scores[int(reg_top2[1])])))
        best_reg_slots = reg_table[best_reg_idx].tolist()

        # 2. Exact MLIT Serial max over 4/3/2/1-digit patterns (slots 8..11)
        best_ser_score, best_ser_slots, ser_margin = self._best_and_runnerup_serial(log_probs)

        # 3. Classification tail (slots 5..6) + per-first-digit scores for d1 in '1'..'9' + '0'
        best_std_tail_sc, best_std_tail_slots, std_tail_gap, best_zero_tail_sc, best_zero_tail_slots, zero_tail_gap = (
            self._best_and_runnerup_tails(log_probs)
        )

        cls_d1_std = log_probs[4, self._nonzero_digit_ids] + best_std_tail_sc  # (9,)
        cls_d1_zero = np.array([log_probs[4, self._zero_id] + best_zero_tail_sc], dtype=np.float64)
        cls_d1_scores = np.concatenate([cls_d1_std, cls_d1_zero], axis=0)  # (10,)
        kana_scores = log_probs[7, self._valid_slot7_ids]  # (K,)

        # 4. Chroma-precision temperature scaling + bounded luminance-polarity prior (±0.55 = 1.1 nats)
        tau_c = max(1e-4, float(chroma_tau))
        gamma_c = float(np.tanh(max(0.0, float(color_precision_mean)) / tau_c))

        vis_shift = type_np - np.max(type_np)
        vis_log_probs = vis_shift - np.log(np.sum(np.exp(vis_shift)) + 1e-15)

        pol_conf = float(np.tanh(float(polarity_logit)) * np.tanh(max(0.0, float(polarity_precision)) / 0.5))
        pol_prior = (0.85 - 0.20 * gamma_c) * pol_conf * self._type_expected_polarity
        # MLIT physical rule: 図柄入り (PlateType.GRAPHIC, index 4) exists strictly as paint-type
        # (bright/pastel background + dark strokes), never 字光式 inverted polarity!
        if pol_conf < 0.0 and pol_prior.shape[0] >= 5:
            pol_prior = pol_prior.copy()
            pol_prior[4] -= 2.40 * (-pol_conf)

        raw_type_prior = (
            gamma_c * vis_log_probs
            + (1.0 - gamma_c) * self._base_type_log_prior
            + pol_prior
        )
        rp_max = float(np.max(raw_type_prior))
        type_log_prior = raw_type_prior - (rp_max + float(np.log(np.sum(np.exp(raw_type_prior - rp_max)) + 1e-15)))

        # 5. Joint (5, 10, K) evaluation & exact LogSumExp marginalization over PlateType c
        joint = (
            type_log_prior[:, None, None]
            + cls_d1_scores[None, :, None]
            + kana_scores[None, None, :]
            + self._psi_mlit_log_potentials
        )  # (5, 10, K)

        j_max = np.max(joint, axis=0)  # (10, K)
        marginal = j_max + np.log(np.sum(np.exp(joint - j_max[None, :, :]), axis=0) + 1e-15)

        best_flat = int(np.argmax(marginal))
        best_d1_idx, best_k_idx = divmod(best_flat, marginal.shape[1])
        best_dk_score = float(marginal[best_d1_idx, best_k_idx])

        # Exact runner-up margins for classification (d1 or tail) and hiragana (slot 7)
        d1_col_max = np.max(marginal, axis=1)  # (10,)
        other_d1_max = float(
            np.max(np.delete(d1_col_max, best_d1_idx)) if marginal.shape[0] > 1 else -1e12
        )
        d1_gap = float(max(0.0, best_dk_score - other_d1_max))

        k_row_max = np.max(marginal, axis=0)  # (K,)
        other_k_max = float(
            np.max(np.delete(k_row_max, best_k_idx)) if marginal.shape[1] > 1 else -1e12
        )
        kana_margin = float(max(0.0, best_dk_score - other_k_max))

        if best_d1_idx == 9:
            best_d1_tok = int(self._zero_id)
            best_tail_slots = best_zero_tail_slots
            cls_margin = float(min(d1_gap, zero_tail_gap))
        else:
            best_d1_tok = int(self._nonzero_digit_ids[best_d1_idx])
            best_tail_slots = best_std_tail_slots
            cls_margin = float(min(d1_gap, std_tail_gap))

        best_kana_tok = int(self._valid_slot7_ids[best_k_idx])
        best_cls_slots = [best_d1_tok] + best_tail_slots
        slot_ids = best_reg_slots + best_cls_slots + [best_kana_tok] + best_ser_slots

        # Calibrated posterior P(c | y*, visual, polarity)
        post_logits = joint[:, best_d1_idx, best_k_idx]
        p_max = float(np.max(post_logits))
        post_exps = np.exp(post_logits - p_max)
        post_probs = post_exps / (float(np.sum(post_exps)) + 1e-15)

        best_type_idx = int(np.argmax(post_probs))
        best_plate_type = self.vocab.plate_types[best_type_idx]

        plate_type_probs = {
            pt.value: float(post_probs[i]) for i, pt in enumerate(self.vocab.plate_types)
        }

        # Cumulative >= 90% posterior mass candidate set
        order = np.argsort(-post_probs)
        candidates: list[str] = []
        cum_mass = 0.0
        for idx in order:
            p_val = float(post_probs[int(idx)])
            if p_val <= 1e-4 and candidates:
                break
            candidates.append(self.vocab.plate_types[int(idx)].value)
            cum_mass += p_val
            if cum_mass >= candidate_mass_threshold:
                break

        joint_log_score = best_reg_score + best_ser_score + best_dk_score
        top2_margin = float(min(reg_margin, cls_margin, kana_margin, ser_margin))
        field_margins = {
            "region": reg_margin,
            "classification": cls_margin,
            "hiragana": kana_margin,
            "serial": ser_margin,
        }
        plate_text = self.vocab.try_decode(slot_ids)
        return TypeMarginalizedDecodeResult(
            slot_ids=slot_ids,
            plate_text=plate_text,
            best_plate_type=best_plate_type,
            plate_type_probs=plate_type_probs,
            plate_type_candidates=candidates,
            joint_log_score=joint_log_score,
            gamma_chroma=gamma_c,
            top2_margin=top2_margin,
            field_margins=field_margins,
        )
