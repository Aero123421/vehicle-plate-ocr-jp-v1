"""Validation rules for Japanese license plates."""

import re
from typing import ClassVar
from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.models import PlateText, PlateType


class ValidationError(ValueError):
    """Exception raised when a license plate violates legal formatting rules."""
    pass


class PlateValidator:
    """Validator implementing official MLIT rules for Japanese license plates.

    Sources:
    - 道路運送車両法施行規則 第11条（自動車登録番号標等の様式）
    - 自動車登録規則 第13条（分類番号及びひらがな等の指定）
    - 国土交通省 告示（2017年1月1日施行 分類番号へのローマ字導入告示）
    """

    # MLIT 2017 permitted alphabet letters in classification numbers (10 characters):
    # Distinct from digits 0, 1, 8, etc. (B, D, I, O, Q excluded)
    ALLOWED_CLASSIFICATION_LETTERS: ClassVar[set[str]] = {
        "A", "C", "F", "H", "K", "L", "M", "P", "X", "Y"
    }

    # Universally excluded hiragana:
    # お (confusable with あ/を), し (associates with 死), へ (associates with 屁), ん (pronunciation difficulty)
    EXCLUDED_HIRAGANA: ClassVar[set[str]] = {"お", "し", "へ", "ん", "ゐ", "ゑ"}

    # Allowed hiragana by vehicle category:
    # 登録自動車 (普通車) - 自家用
    STANDARD_PRIVATE_HIRAGANA: ClassVar[set[str]] = {
        "さ", "す", "せ", "そ",
        "た", "ち", "つ", "て", "と",
        "な", "に", "ぬ", "ね", "の",
        "は", "ひ", "ふ", "ほ",
        "ま", "み", "む", "め", "も",
        "や", "ゆ",
        "ら", "り", "る", "ろ",
    }
    # 登録自動車 (普通車) - 事業用 (緑ナンバー)
    STANDARD_COMMERCIAL_HIRAGANA: ClassVar[set[str]] = {
        "あ", "い", "う", "え",
        "か", "き", "く", "け", "こ",
        "を",
    }
    # 軽自動車 - 自家用 (黄ナンバー)
    KEI_PRIVATE_HIRAGANA: ClassVar[set[str]] = {
        "あ", "い", "う", "え",
        "か", "き", "く", "け", "こ",
        "さ", "す", "せ", "そ",
        "た", "ち", "つ", "て", "と",
        "な", "に", "ぬ", "ね", "の",
        "は", "ひ", "ふ", "ほ",
        "ま", "み", "む", "め", "も",
        "や", "ゆ", "よ",
        "ら", "る", "ろ",
        "を",
    }
    # 軽自動車 - 事業用 (黒ナンバー: 一般営業用 り・れ + 希望番号枠溢れ/貸渡等 ろ・わ)
    KEI_COMMERCIAL_HIRAGANA: ClassVar[set[str]] = {"り", "れ", "ろ", "わ"}

    # レンタカー (貸渡用)
    RENTAL_HIRAGANA_STANDARD: ClassVar[set[str]] = {"わ", "れ"}
    RENTAL_HIRAGANA_KEI: ClassVar[set[str]] = {"わ"}

    # 駐留軍 (SOFA vehicles)
    SOFA_CHARS: ClassVar[set[str]] = {"よ", "A", "B", "E", "H", "K", "M", "T", "Y"}

    @classmethod
    def get_valid_hiragana_for_type(cls, plate_type: PlateType) -> set[str]:
        """Get set of all valid hiragana/symbols for a specific plate type."""
        match plate_type:
            case PlateType.STANDARD_PRIVATE:
                return cls.STANDARD_PRIVATE_HIRAGANA | cls.RENTAL_HIRAGANA_STANDARD | cls.SOFA_CHARS
            case PlateType.STANDARD_COMMERCIAL:
                return cls.STANDARD_COMMERCIAL_HIRAGANA
            case PlateType.KEI_PRIVATE:
                return cls.KEI_PRIVATE_HIRAGANA | cls.RENTAL_HIRAGANA_KEI | cls.SOFA_CHARS
            case PlateType.KEI_COMMERCIAL:
                return cls.KEI_COMMERCIAL_HIRAGANA
            case PlateType.GRAPHIC:
                # Graphic plates exist for standard private and kei private
                return cls.STANDARD_PRIVATE_HIRAGANA | cls.KEI_PRIVATE_HIRAGANA | cls.RENTAL_HIRAGANA_STANDARD

    @classmethod
    def validate_region(cls, region: str) -> list[str]:
        """Validate regional transport office name."""
        errors: list[str] = []
        if not region:
            return ["地域名が空です。"]

        # Support common colloquial alias (袖ヶ浦 <-> 袖ケ浦)
        norm_reg = "袖ケ浦" if region == "袖ヶ浦" else region

        valid_regions = PlateDataLoader.get_region_map()
        if norm_reg not in valid_regions:
            errors.append(f"未登録の地域名です: '{region}' (現行の138地域に含まれていません)")

        return errors

    @classmethod
    def is_valid_construction_zero_class(cls, classification: str) -> bool:
        """Return True for legal 0-series 大型特殊自動車（建設機械） classification codes (`0`, `00..09`, `00x`, `09x`)."""
        if not classification or classification[0] != "0":
            return False
        if len(classification) == 1:
            return True
        if len(classification) == 2:
            return classification[1].isdigit()
        if len(classification) == 3:
            c2, c3 = classification[1], classification[2]
            return c2 in ("0", "9") and (c3.isdigit() or c3 in cls.ALLOWED_CLASSIFICATION_LETTERS)
        return False

    @classmethod
    def validate_classification(cls, classification: str) -> list[str]:
        """Validate vehicle classification number (分類番号: 1..3 chars, including 0-series 大型特殊建設機械)."""
        errors: list[str] = []
        if not classification:
            return ["分類番号が空です。"]

        if not (1 <= len(classification) <= 3):
            errors.append(f"分類番号の桁数は1〜3文字である必要があります: '{classification}' (長さ: {len(classification)})")
            return errors

        # 1st character must be digit 1..9, or legal 0-series 大型特殊自動車（建設機械） (`0`, `00..09`, `00x`, `09x`)
        first_char = classification[0]
        if first_char == "0":
            if not cls.is_valid_construction_zero_class(classification):
                errors.append(
                    f"分類番号の先頭は1〜9の数字（大型特殊建設機械の0・00x・09xを除く）である必要があります: '{first_char}'"
                )
        elif not first_char.isdigit():
            errors.append(f"分類番号の先頭は1〜9の数字である必要があります: '{first_char}'")

        # 2nd character (if present)
        if len(classification) >= 2:
            c2 = classification[1]
            if len(classification) == 2:
                # 2-digit classification (historical) must be numeric
                if not c2.isdigit():
                    errors.append(f"2桁分類番号の2文字目は数字である必要があります: '{c2}'")
            else:
                # 3-digit classification: digit or allowed letter
                if not (c2.isdigit() or c2 in cls.ALLOWED_CLASSIFICATION_LETTERS):
                    errors.append(
                        f"分類番号の2文字目に使用できない文字が含まれています: '{c2}' "
                        f"(許可文字: 数字または {sorted(cls.ALLOWED_CLASSIFICATION_LETTERS)})"
                    )

        # 3rd character (if present)
        if len(classification) == 3:
            c3 = classification[2]
            if not (c3.isdigit() or c3 in cls.ALLOWED_CLASSIFICATION_LETTERS):
                errors.append(
                    f"分類番号の3文字目に使用できない文字が含まれています: '{c3}' "
                    f"(許可文字: 数字または {sorted(cls.ALLOWED_CLASSIFICATION_LETTERS)})"
                )

        return errors

    @classmethod
    def validate_hiragana(cls, hiragana: str, plate_type: PlateType | None = None) -> list[str]:
        """Validate kana symbol."""
        errors: list[str] = []
        if not hiragana:
            return ["ひらがなが空です。"]

        if len(hiragana) != 1:
            return [f"ひらがなは1文字である必要があります: '{hiragana}'"]

        if hiragana in cls.EXCLUDED_HIRAGANA:
            errors.append(f"使用禁止のひらがなです: '{hiragana}' (「お・し・へ・ん」は欠番)")

        if plate_type is not None:
            valid_for_type = cls.get_valid_hiragana_for_type(plate_type)
            if hiragana not in valid_for_type:
                errors.append(
                    f"用途区分 '{plate_type.value}' では使用されないひらがなです: '{hiragana}'"
                )

        return errors

    @classmethod
    def validate_serial(cls, serial: int) -> list[str]:
        """Validate serial number (一連指定番号: 1..9999)."""
        errors: list[str] = []
        if not (1 <= serial <= 9999):
            errors.append(f"一連指定番号は1〜9999の範囲である必要があります: {serial}")
        return errors

    @classmethod
    def validate_plate_type(
        cls,
        plate_type: PlateType | None,
        is_graphic: bool = False,
        hiragana: str = "",
    ) -> list[str]:
        """Validate plate category restrictions per MLIT regulations."""
        errors: list[str] = []
        if plate_type is not None:
            if is_graphic or plate_type == PlateType.GRAPHIC:
                if plate_type == PlateType.KEI_COMMERCIAL:
                    errors.append("軽自動車の事業用（黒ナンバー）は法令上、図柄入りナンバープレートの交付対象外です。")
                if plate_type == PlateType.KEI_PRIVATE and hiragana == "わ":
                    errors.append("軽自動車の貸渡用（レンタカー「わ」）は図柄入りナンバープレートの交付対象外です。")
        return errors

    @classmethod
    def validate(
        cls,
        plate_text: PlateText,
        plate_type: PlateType | None = None,
        is_graphic: bool = False,
    ) -> list[str]:
        """Validate a complete PlateText structure."""
        errors: list[str] = []
        errors.extend(cls.validate_region(plate_text.region))
        errors.extend(cls.validate_classification(plate_text.classification))
        errors.extend(cls.validate_hiragana(plate_text.hiragana, plate_type))
        errors.extend(cls.validate_serial(plate_text.serial))
        errors.extend(cls.validate_plate_type(plate_type, is_graphic, plate_text.hiragana))
        if (
            plate_type in (PlateType.KEI_PRIVATE, PlateType.KEI_COMMERCIAL, PlateType.GRAPHIC)
            and plate_text.classification.startswith("0")
        ):
            errors.append("0ナンバー（大型特殊建設機械）は軽自動車・図柄入りナンバーには交付されません。")
        return errors

    @classmethod
    def is_valid(
        cls,
        plate_text: PlateText,
        plate_type: PlateType | None = None,
        is_graphic: bool = False,
    ) -> bool:
        """Check if plate text satisfies all formatting rules."""
        return len(cls.validate(plate_text, plate_type, is_graphic)) == 0

    @classmethod
    def parse_string(cls, text: str) -> PlateText:
        """Parse raw string into PlateText.

        Accepts formats like:
        - "品川 599 あ 12-34"
        - "品川599あ12-34"
        - "世田谷 30A わ ・123"
        - "足立 500 さ ・・・1"
        """
        cleaned = text.strip()
        # Regex to capture: Region + Classification + Kana + Serial
        # Serial may have dots or hyphens
        m = re.match(
            r"^([^\d\s]+)\s*([0-9][0-9A-Z]{0,2})\s*([^\d\s\.\・\-])\s*([・\.\d\-]+)$",
            cleaned,
        )
        if not m:
            raise ValidationError(f"ナンバープレート文字列の形式が無効です: '{text}'")

        region, classification, hiragana, raw_serial = m.groups()

        # Parse serial digits out of dots and hyphens
        digits_only = re.sub(r"[^0-9]", "", raw_serial)
        if not digits_only:
            raise ValidationError(f"一連指定番号に数字が含まれていません: '{raw_serial}'")

        serial_val = int(digits_only)
        plate_text = PlateText(
            region=region,
            classification=classification,
            hiragana=hiragana,
            serial=serial_val,
        )

        errors = cls.validate(plate_text)
        if errors:
            raise ValidationError("; ".join(errors))

        return plate_text
