"""Unit tests for plate data models."""

import pytest
from vpocr_synth.plate.models import BoundingBox, CharacterBox, PlateConfig, PlateSize, PlateText, PlateType


def test_plate_text_formatting():
    # 4 digits: hyphenated
    pt4 = PlateText(region="品川", classification="500", hiragana="あ", serial=1234)
    assert pt4.formatted_serial == "12-34"
    assert pt4.full_text == "品川 500 あ 12-34"
    assert pt4.ocr_string == "品川500あ12-34"

    # 3 digits: 1 dot
    pt3 = PlateText(region="世田谷", classification="300", hiragana="さ", serial=123)
    assert pt3.formatted_serial == "・123"
    assert pt3.full_text == "世田谷 300 さ ・123"

    # 2 digits: 2 dots
    pt2 = PlateText(region="横浜", classification="501", hiragana="た", serial=12)
    assert pt2.formatted_serial == "・・12"
    assert pt2.full_text == "横浜 501 た ・・12"

    # 1 digit: 3 dots
    pt1 = PlateText(region="練馬", classification="330", hiragana="な", serial=1)
    assert pt1.formatted_serial == "・・・1"
    assert pt1.full_text == "練馬 330 な ・・・1"


def test_plate_text_invalid_serial():
    with pytest.raises(ValueError):
        PlateText(region="品川", classification="500", hiragana="あ", serial=0)

    with pytest.raises(ValueError):
        PlateText(region="品川", classification="500", hiragana="あ", serial=10000)


def test_bounding_box():
    bbox = BoundingBox(x_min=10.5, y_min=20.0, x_max=60.5, y_max=80.0)
    assert bbox.width == 50.0
    assert bbox.height == 60.0
    assert bbox.as_tuple() == (10.5, 20.0, 60.5, 80.0)
    assert bbox.as_int_tuple() == (10, 20, 60, 80)

    d = bbox.to_dict()
    assert d["width"] == 50.0
    assert d["height"] == 60.0


def test_plate_config_defaults():
    cfg = PlateConfig()
    assert cfg.plate_type == PlateType.STANDARD_PRIVATE
    assert cfg.plate_size == PlateSize.MEDIUM
    assert cfg.scale == 2.0
    assert cfg.with_bolt_holes is True
    assert cfg.with_border is True
