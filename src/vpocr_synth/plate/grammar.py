"""JPPlateGrammar: non-destructive post-OCR consistency verifier and rescorer."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.models import PlateText, PlateType
from vpocr_synth.plate.validator import PlateValidator

# Machine-readable violation codes
INVALID_REGION = "INVALID_REGION"
INVALID_CLASSIFICATION = "INVALID_CLASSIFICATION"
INVALID_CLASS_LETTER = "INVALID_CLASS_LETTER"
FORBIDDEN_HIRAGANA = "FORBIDDEN_HIRAGANA"
INVALID_HIRAGANA = "INVALID_HIRAGANA"
PLATE_TYPE_KANA_MISMATCH = "PLATE_TYPE_KANA_MISMATCH"
INVALID_SERIAL = "INVALID_SERIAL"
SERIAL_DOT_HYPHEN_MISMATCH = "SERIAL_DOT_HYPHEN_MISMATCH"
INVALID_GRAPHIC_COMBINATION = "INVALID_GRAPHIC_COMBINATION"
INVALID_FORMAT = "INVALID_FORMAT"


def _normalize_region(region: str) -> str:
    if region in ("袖ヶ浦", "袖ケ浦"):
        # canonical is 袖ケ浦 (large ケ)
        return "袖ケ浦"
    # also handle small ヶ anywhere? Only known alias
    return region.replace("ヶ", "ケ") if region == "袖ヶ浦" else region


def _check_serial_format(serial_str: str) -> tuple[int | None, bool]:
    """Return (serial_int, format_ok). Accepts plain digits, dd-dd, ・ddd etc."""
    s = serial_str.strip()
    if not s:
        return None, False
    digits = re.sub(r"[^0-9]", "", s)
    if not digits:
        return None, False
    try:
        val = int(digits)
    except ValueError:
        return None, False
    if not (1 <= val <= 9999):
        return val, False
    # If string contains only digits, it's acceptable (OCR may omit dots/hyphen)
    if re.fullmatch(r"[0-9]+", s):
        return val, True
    # Expected formatted forms
    dlen = len(digits)
    # Note: digits may have leading? serial int string length is dlen unless leading zeros
    # Use length of digit string stripped of leading zeros? Keep dlen as len(str(val)) if no leading zero issue
    effective_len = len(str(val))
    if effective_len == 4:
        expected = f"{str(val)[:2]}-{str(val)[2:]}"
    elif effective_len == 3:
        expected = f"・{val}"
    elif effective_len == 2:
        expected = f"・{chr(0x30FB)}{val}" if False else f"・・{val}"
    else:
        expected = f"・・・{val}"
    # Compare: s should equal expected
    return val, (s == expected)


@dataclass(frozen=True)
class GrammarCheckResult:
    is_valid: bool
    consistency_score: float
    violations: list[str] = field(default_factory=list)
    inferred_plate_types: list[PlateType] = field(default_factory=list)
    normalized_text: PlateText | None = None

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "consistency_score": self.consistency_score,
            "violations": list(self.violations),
            "inferred_plate_types": [t.value for t in self.inferred_plate_types],
            "normalized_text": self.normalized_text.to_dict() if self.normalized_text else None,
        }


class JPPlateGrammar:
    """MLIT-based grammar checker. Never silently overwrites; always flags."""

    VIOLATION_CODES = [
        INVALID_REGION,
        INVALID_CLASSIFICATION,
        INVALID_CLASS_LETTER,
        FORBIDDEN_HIRAGANA,
        INVALID_HIRAGANA,
        PLATE_TYPE_KANA_MISMATCH,
        INVALID_SERIAL,
        SERIAL_DOT_HYPHEN_MISMATCH,
        INVALID_GRAPHIC_COMBINATION,
        INVALID_FORMAT,
    ]

    def __init__(self) -> None:
        self._region_map = PlateDataLoader.get_region_map()

    # ---------- core check on fields ----------
    def check_fields(
        self,
        region: str,
        classification: str,
        hiragana: str,
        serial: int | str,
        plate_type: PlateType | None = None,
        is_graphic: bool = False,
        serial_str: str | None = None,
    ) -> GrammarCheckResult:
        violations: list[str] = []

        norm_region = _normalize_region(region) if region else region

        # Region
        if not region or norm_region not in self._region_map:
            if INVALID_REGION not in violations:
                violations.append(INVALID_REGION)

        # Classification
        if not classification or not (1 <= len(classification) <= 3):
            violations.append(INVALID_CLASSIFICATION)
        else:
            # letter check first for specific code
            has_invalid_letter = False
            for pos, ch in enumerate(classification):
                if ch.isalpha():
                    # Latin letter position
                    if ch not in PlateValidator.ALLOWED_CLASSIFICATION_LETTERS:
                        has_invalid_letter = True
                    # 2-digit classification cannot have letter at pos1
                    if len(classification) == 2 and pos == 1:
                        has_invalid_letter = True
            if has_invalid_letter:
                violations.append(INVALID_CLASS_LETTER)
            # generic classification validation via validator
            cls_errors = PlateValidator.validate_classification(classification)
            if cls_errors:
                if has_invalid_letter:
                    # Only add INVALID_CLASSIFICATION if non-letter prefix/structure is also invalid
                    probe = "".join("A" if c.isalpha() else c for c in classification)
                    if PlateValidator.validate_classification(probe):
                        if INVALID_CLASSIFICATION not in violations:
                            violations.append(INVALID_CLASSIFICATION)
                else:
                    if INVALID_CLASSIFICATION not in violations:
                        violations.append(INVALID_CLASSIFICATION)

        # Hiragana
        if not hiragana or len(hiragana) != 1:
            violations.append(INVALID_HIRAGANA)
        elif hiragana in PlateValidator.EXCLUDED_HIRAGANA:
            violations.append(FORBIDDEN_HIRAGANA)
        # Note: SOFA chars A,B etc are single char latin, not in excluded; they are handled below

        # Serial
        serial_int: int | None = None
        raw_serial_str = serial_str if serial_str is not None else (serial if isinstance(serial, str) else None)
        if isinstance(serial, int):
            serial_int = serial
            if not (1 <= serial <= 9999):
                violations.append(INVALID_SERIAL)
        elif isinstance(serial, str):
            parsed, fmt_ok = _check_serial_format(serial)
            serial_int = parsed
            if parsed is None or not (1 <= parsed <= 9999):
                violations.append(INVALID_SERIAL)
            elif not fmt_ok:
                violations.append(SERIAL_DOT_HYPHEN_MISMATCH)
        else:
            # try int conversion
            try:
                serial_int = int(serial)  # type: ignore[arg-type]
                if not (1 <= serial_int <= 9999):
                    violations.append(INVALID_SERIAL)
            except Exception:
                violations.append(INVALID_SERIAL)

        if raw_serial_str is not None and isinstance(serial, int):
            # explicit serial_str format check
            parsed2, fmt_ok2 = _check_serial_format(str(raw_serial_str))
            if parsed2 is not None and parsed2 == serial and not fmt_ok2:
                if SERIAL_DOT_HYPHEN_MISMATCH not in violations:
                    violations.append(SERIAL_DOT_HYPHEN_MISMATCH)

        # Inferred plate types: compatible with (classification, hiragana, is_graphic)
        inferred: list[PlateType] = []
        if hiragana and len(hiragana) == 1 and hiragana not in PlateValidator.EXCLUDED_HIRAGANA:
            for pt in PlateType:
                valid_set = PlateValidator.get_valid_hiragana_for_type(pt)
                if hiragana not in valid_set:
                    continue
                if is_graphic:
                    if pt == PlateType.KEI_COMMERCIAL:
                        continue
                    if pt == PlateType.KEI_PRIVATE and hiragana == "わ":
                        continue
                inferred.append(pt)
            # If is_graphic, GRAPHIC type already considered; also graphic is a rendering variant
            # Keep as computed.

        # Plate type kana mismatch
        if plate_type is not None and hiragana and len(hiragana) == 1:
            if hiragana in PlateValidator.EXCLUDED_HIRAGANA:
                # forbidden already; mismatch also if not in type set
                valid_set = PlateValidator.get_valid_hiragana_for_type(plate_type)
                if hiragana not in valid_set and PLATE_TYPE_KANA_MISMATCH not in violations:
                    violations.append(PLATE_TYPE_KANA_MISMATCH)
            else:
                valid_set = PlateValidator.get_valid_hiragana_for_type(plate_type)
                if hiragana not in valid_set:
                    violations.append(PLATE_TYPE_KANA_MISMATCH)

        # Graphic restrictions
        if is_graphic or plate_type == PlateType.GRAPHIC:
            # validator logic: kei commercial graphic not allowed, kei private わ graphic not allowed
            if plate_type == PlateType.KEI_COMMERCIAL:
                if INVALID_GRAPHIC_COMBINATION not in violations:
                    violations.append(INVALID_GRAPHIC_COMBINATION)
            if plate_type == PlateType.KEI_PRIVATE and hiragana == "わ":
                if INVALID_GRAPHIC_COMBINATION not in violations:
                    violations.append(INVALID_GRAPHIC_COMBINATION)

        # Deduplicate preserving order
        seen = set()
        uniq: list[str] = []
        for v in violations:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        violations = uniq

        is_valid = len(violations) == 0
        consistency_score = 1.0 if is_valid else max(0.0, 1.0 - 0.25 * len(violations))

        normalized_text: PlateText | None = None
        if is_valid and serial_int is not None:
            try:
                normalized_text = PlateText(region=norm_region, classification=classification, hiragana=hiragana, serial=serial_int)
                # double-check with validator (without plate_type constraint if inferred etc)
                # If plate_type given, already validated; if not, ensure validator passes for at least inferred?
            except Exception:
                normalized_text = None

        return GrammarCheckResult(
            is_valid=is_valid,
            consistency_score=float(consistency_score),
            violations=violations,
            inferred_plate_types=inferred,
            normalized_text=normalized_text,
        )

    # ---------- public check APIs (aliases) ----------
    def check(
        self,
        plate_text: PlateText | str,
        plate_type: PlateType | None = None,
        is_graphic: bool = False,
    ) -> GrammarCheckResult:
        if isinstance(plate_text, str):
            return self._check_string(plate_text, plate_type, is_graphic)
        return self.check_fields(
            region=plate_text.region,
            classification=plate_text.classification,
            hiragana=plate_text.hiragana,
            serial=plate_text.serial,
            plate_type=plate_type,
            is_graphic=is_graphic,
        )

    # aliases for flexibility
    validate = check
    verify = check
    evaluate = check
    check_plate = check
    grammar_check = check

    def _check_string(self, text: str, plate_type: PlateType | None, is_graphic: bool) -> GrammarCheckResult:
        cleaned = text.strip()
        m = re.match(r"^([^\d\s]+)\s*([0-9][0-9A-Za-z]{0,2})\s*([^\d\s\.\・\-])\s*([・\.\d\-]+)$", cleaned)
        if not m:
            return GrammarCheckResult(is_valid=False, consistency_score=0.0, violations=[INVALID_FORMAT], inferred_plate_types=[], normalized_text=None)
        region, classification, hiragana, raw_serial = m.groups()
        parsed, fmt_ok = _check_serial_format(raw_serial)
        # check_fields with serial string will add mismatch
        result = self.check_fields(region=region, classification=classification, hiragana=hiragana, serial=raw_serial, plate_type=plate_type, is_graphic=is_graphic)
        return result

    @staticmethod
    def _candidate_prior_bonus(region: str, cls: str, serial_int: int | None) -> float:
        """Mild whole-candidate fleet log-prior applied ONCE per candidate."""
        norm_r = _normalize_region(region)
        r_len = len(norm_r)
        reg_prior = 0.0 if r_len in (2, 3) else -0.15

        if cls.startswith("0"):
            cls_prior = -2.50
        elif len(cls) == 3:
            cls_prior = -0.20 if any(c.isalpha() for c in cls) else 0.0
        elif len(cls) == 2:
            cls_prior = -0.15
        else:
            cls_prior = -0.30

        if serial_int is None or serial_int <= 0:
            ser_prior = -0.50
        elif serial_int >= 1000:
            ser_prior = 0.0
        elif serial_int >= 100:
            ser_prior = -0.08
        elif serial_int >= 10:
            ser_prior = -0.16
        else:
            ser_prior = -0.35

        return float(reg_prior + cls_prior + ser_prior)

    # ---------- rescore ----------
    def rescore_candidates(
        self,
        region_cands: list[tuple[str, float]],
        class_cands: list[tuple[str, float]],
        kana_cands: list[tuple[str, float]],
        serial_cands: list[tuple[str, float]],
        predicted_plate_type: PlateType | None = None,
        penalty_invalid: float = 4.0,
    ) -> list[dict]:
        if not region_cands or not class_cands or not kana_cands or not serial_cands:
            return []
        results: list[dict] = []
        for (region, r_lp), (cls, c_lp), (kana, k_lp), (serial_s, s_lp) in itertools.product(region_cands, class_cands, kana_cands, serial_cands):
            raw = float(r_lp + c_lp + k_lp + s_lp)
            serial_int, fmt_ok = _check_serial_format(serial_s)
            # Use check_fields with serial string to capture format mismatch
            gr = self.check_fields(
                region=region,
                classification=cls,
                hiragana=kana,
                serial=serial_s,
                plate_type=predicted_plate_type,
                is_graphic=(predicted_plate_type == PlateType.GRAPHIC) if predicted_plate_type else False,
            )
            prior_bonus = self._candidate_prior_bonus(region, cls, serial_int)
            adjusted = raw + prior_bonus - penalty_invalid * len(gr.violations)
            entry = {
                "region": gr.normalized_text.region if gr.normalized_text else _normalize_region(region),
                "classification": cls,
                "hiragana": kana,
                "kana": kana,
                "serial": serial_int if serial_int is not None else serial_s,
                "serial_text": serial_s,
                "formatted_serial": gr.normalized_text.formatted_serial if gr.normalized_text else serial_s,
                "raw_log_prob": raw,
                "grammar_adjusted_score": float(adjusted),
                "adjusted_score": float(adjusted),
                "score": float(adjusted),
                "grammar_result": gr,
                "grammar_check_result": gr,
                "check_result": gr,
                "result": gr,
                "is_valid": gr.is_valid,
                "violations": list(gr.violations),
                "consistency_score": gr.consistency_score,
                "inferred_plate_types": list(gr.inferred_plate_types),
                "normalized_text": gr.normalized_text,
                "plate_text": gr.normalized_text,
                "text": gr.normalized_text.full_text if gr.normalized_text else f"{region} {cls} {kana} {serial_s}",
                "full_text": gr.normalized_text.full_text if gr.normalized_text else f"{region} {cls} {kana} {serial_s}",
            }
            results.append(entry)
        results.sort(key=lambda d: (d["grammar_adjusted_score"], d["raw_log_prob"]), reverse=True)
        return results

    # aliases
    rescore = rescore_candidates
    rank_candidates = rescore_candidates
