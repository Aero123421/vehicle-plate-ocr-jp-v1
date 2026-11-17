"""Unit tests for plate validator rules."""

import pytest
from vpocr_synth.plate.models import PlateText, PlateType
from vpocr_synth.plate.validator import PlateValidator, ValidationError


def test_validator_valid_plates():
    valid_samples = [
        ("品川", "599", "さ", 1234, PlateType.STANDARD_PRIVATE),
        ("世田谷", "30A", "す", 123, PlateType.STANDARD_PRIVATE),
        ("尾張小牧", "300", "た", 12, PlateType.STANDARD_PRIVATE),
        ("柏", "500", "な", 1, PlateType.STANDARD_PRIVATE),
        ("堺", "580", "あ", 8888, PlateType.KEI_PRIVATE),
        ("足立", "200", "か", 5555, PlateType.STANDARD_COMMERCIAL),
        ("なにわ", "480", "り", 1122, PlateType.KEI_COMMERCIAL),
        ("横浜", "300", "わ", 9999, PlateType.STANDARD_PRIVATE),  # rental
        ("袖ヶ浦", "501", "よ", 100, PlateType.STANDARD_PRIVATE),  # sofa
    ]

    for reg, cls_num, kana, ser, ptype in valid_samples:
        pt = PlateText(region=reg, classification=cls_num, hiragana=kana, serial=ser)
        errs = PlateValidator.validate(pt, ptype)
        assert len(errs) == 0, f"Expected valid, got errors: {errs} for {pt}"
        assert PlateValidator.is_valid(pt, ptype)


def test_validator_invalid_region():
    # Non-existent region
    pt = PlateText(region="東京", classification="500", hiragana="あ", serial=1234)
    errs = PlateValidator.validate(pt)
    assert any("未登録の地域名" in e for e in errs)

    # Empty region
    pt_empty = PlateText(region="", classification="500", hiragana="あ", serial=1234)
    errs = PlateValidator.validate(pt_empty)
    assert any("地域名が空" in e for e in errs)


def test_validator_classification_rules():
    # Starts with 0 -> invalid
    pt = PlateText(region="品川", classification="050", hiragana="あ", serial=1234)
    assert any("先頭は1〜9" in e for e in PlateValidator.validate(pt))

    # Length > 3 -> invalid
    pt = PlateText(region="品川", classification="5000", hiragana="あ", serial=1234)
    assert any("1〜3文字" in e for e in PlateValidator.validate(pt))

    # Permitted Latin letters: A, C, F, H, K, L, M, P, X, Y
    for valid_letter in ["A", "C", "F", "H", "K", "L", "M", "P", "X", "Y"]:
        pt = PlateText(region="品川", classification=f"30{valid_letter}", hiragana="あ", serial=1234)
        assert PlateValidator.is_valid(pt), f"Expected {valid_letter} to be valid"

    # Excluded Latin letters: B, D, I, O, Q, Z, etc.
    for invalid_letter in ["B", "D", "I", "O", "Q", "Z"]:
        pt = PlateText(region="品川", classification=f"30{invalid_letter}", hiragana="あ", serial=1234)
        errs = PlateValidator.validate(pt)
        assert any("使用できない文字" in e for e in errs), f"Expected {invalid_letter} to be rejected"

    # 2-digit classification with letter -> invalid
    pt = PlateText(region="品川", classification="3A", hiragana="あ", serial=1234)
    assert any("2桁分類番号の2文字目は数字" in e for e in PlateValidator.validate(pt))


def test_validator_excluded_hiragana():
    # お, し, へ, ん must be rejected
    for excluded in ["お", "し", "へ", "ん"]:
        pt = PlateText(region="品川", classification="500", hiragana=excluded, serial=1234)
        errs = PlateValidator.validate(pt)
        assert any("使用禁止のひらがな" in e for e in errs), f"Expected {excluded} to be rejected"


def test_validator_type_hiragana_matching():
    # Kei commercial only allows 'り', 'れ'
    pt_kei_comm = PlateText(region="品川", classification="480", hiragana="あ", serial=1234)
    assert not PlateValidator.is_valid(pt_kei_comm, PlateType.KEI_COMMERCIAL)

    pt_kei_comm_valid = PlateText(region="品川", classification="480", hiragana="り", serial=1234)
    assert PlateValidator.is_valid(pt_kei_comm_valid, PlateType.KEI_COMMERCIAL)

    # Standard commercial uses あいうえかきくけこを
    pt_std_comm = PlateText(region="品川", classification="100", hiragana="さ", serial=1234)
    assert not PlateValidator.is_valid(pt_std_comm, PlateType.STANDARD_COMMERCIAL)


def test_parse_string():
    # Full standard format
    pt1 = PlateValidator.parse_string("品川 599 あ 12-34")
    assert pt1.region == "品川"
    assert pt1.classification == "599"
    assert pt1.hiragana == "あ"
    assert pt1.serial == 1234

    # Format with dots
    pt2 = PlateValidator.parse_string("世田谷 30A さ ・123")
    assert pt2.region == "世田谷"
    assert pt2.classification == "30A"
    assert pt2.serial == 123

    # Format without spaces
    pt3 = PlateValidator.parse_string("練馬330な・・・1")
    assert pt3.region == "練馬"
    assert pt3.serial == 1

    # Invalid string raises ValidationError
    with pytest.raises(ValidationError):
        PlateValidator.parse_string("invalid plate string")


def test_validator_graphic_plate_restrictions():
    """Verify MLIT restrictions for graphic commemorative plates."""
    # Kei Commercial (black plate) CANNOT have graphic plates
    pt_kei_comm = PlateText(region="品川", classification="480", hiragana="り", serial=1234)
    errs = PlateValidator.validate(pt_kei_comm, PlateType.KEI_COMMERCIAL, is_graphic=True)
    assert any("軽自動車の事業用" in e for e in errs)

    # Kei Rental (わ) CANNOT have graphic plates
    pt_kei_rental = PlateText(region="品川", classification="580", hiragana="わ", serial=1234)
    errs_rental = PlateValidator.validate(pt_kei_rental, PlateType.KEI_PRIVATE, is_graphic=True)
    assert any("レンタカー「わ」" in e for e in errs_rental)

    # Standard Private and Commercial CAN have graphic plates
    pt_std = PlateText(region="品川", classification="599", hiragana="さ", serial=1234)
    assert PlateValidator.is_valid(pt_std, PlateType.STANDARD_PRIVATE, is_graphic=True)
