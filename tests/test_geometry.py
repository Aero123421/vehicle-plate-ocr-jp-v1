"""Unit tests for plate geometry and millimeter layout."""

import pytest
from vpocr_synth.plate.geometry import PlateGeometry
from vpocr_synth.plate.models import PlateSize, PlateText


def test_geometry_medium_plate():
    pt = PlateText(region="品川", classification="599", hiragana="あ", serial=1234)
    layout = PlateGeometry.calculate_layout(pt, PlateSize.MEDIUM)

    assert layout.width_mm == 330.0
    assert layout.height_mm == 165.0
    assert layout.left_bolt.x == 60.0
    assert layout.right_bolt.x == 270.0

    # Check 4-digit serial has hyphen
    chars = [s.char for s in layout.slots]
    assert "-" in chars

    # Verify all slots are strictly inside plate mm boundaries
    for slot in layout.slots:
        assert 0.0 <= slot.x < layout.width_mm
        assert 0.0 <= slot.y < layout.height_mm
        assert slot.x + slot.width <= layout.width_mm
        assert slot.y + slot.height <= layout.height_mm


def test_geometry_large_plate():
    pt = PlateText(region="足立", classification="100", hiragana="か", serial=9999)
    layout = PlateGeometry.calculate_layout(pt, PlateSize.LARGE)

    assert layout.width_mm == 440.0
    assert layout.height_mm == 220.0
    # Official symmetric 240mm x 150mm 4-bolt JIS D 5705 layout on 440x220mm Large plate
    assert layout.left_bolt.x == 100.0
    assert layout.right_bolt.x == 340.0
    assert len(layout.bolts) == 4
    assert [(b.x, b.y, b.radius) for b in layout.bolts] == [
        (100.0, 35.0, 5.5),
        (340.0, 35.0, 5.5),
        (100.0, 185.0, 5.5),
        (340.0, 185.0, 5.5),
    ]

    # Upper row sits cleanly inside 100mm..340mm bolts; lower row sits in y >= 88mm strip
    for s in layout.slots:
        if s.field in ("region", "classification"):
            assert s.x >= layout.left_bolt.x + 12.0
            assert s.x + s.width <= layout.right_bolt.x - 12.0
            assert s.y + s.height <= 85.0
        else:
            assert s.y >= 88.0


def test_geometry_dot_padding_counts():
    # 3 digits: exactly 1 dot, NO hyphen
    pt3 = PlateText(region="世田谷", classification="300", hiragana="さ", serial=123)
    l3 = PlateGeometry.calculate_layout(pt3, PlateSize.MEDIUM)
    dots3 = [s for s in l3.slots if s.is_dot]
    hyphens3 = [s for s in l3.slots if s.char == "-"]
    assert len(dots3) == 1
    assert len(hyphens3) == 0

    # 2 digits: exactly 2 dots, NO hyphen
    pt2 = PlateText(region="横浜", classification="500", hiragana="た", serial=12)
    l2 = PlateGeometry.calculate_layout(pt2, PlateSize.MEDIUM)
    dots2 = [s for s in l2.slots if s.is_dot]
    hyphens2 = [s for s in l2.slots if s.char == "-"]
    assert len(dots2) == 2
    assert len(hyphens2) == 0

    # 1 digit: exactly 3 dots, NO hyphen
    pt1 = PlateText(region="練馬", classification="330", hiragana="な", serial=1)
    l1 = PlateGeometry.calculate_layout(pt1, PlateSize.MEDIUM)
    dots1 = [s for s in l1.slots if s.is_dot]
    hyphens1 = [s for s in l1.slots if s.char == "-"]
    assert len(dots1) == 3
    assert len(hyphens1) == 0


def test_geometry_single_char_region_centered():
    """Verify 1-char regions like 柏 or 堺 sit balanced next to 3-digit classification."""
    pt = PlateText(region="柏", classification="500", hiragana="さ", serial=1234)
    layout = PlateGeometry.calculate_layout(pt, PlateSize.MEDIUM)
    reg_slot = [s for s in layout.slots if s.field == "region"][0]
    cls_slots = [s for s in layout.slots if s.field == "classification"]
    assert reg_slot.x == 108.0
    assert reg_slot.width == 40.0
    assert reg_slot.x + reg_slot.width / 2.0 == 128.0
    assert cls_slots[0].x == 173.0


def test_geometry_medium_and_large_normalized_consistency_and_jitter():
    """Verify 2-char Medium & Large plates align in normalized [0,1]^2 and jitter respects clearances."""
    import random

    pt = PlateText(region="福岡", classification="200", hiragana="か", serial=684)
    med = PlateGeometry.calculate_layout(pt, PlateSize.MEDIUM)
    lrg = PlateGeometry.calculate_layout(pt, PlateSize.LARGE)

    med_fb = med.get_field_bboxes(1.0 / med.width_mm)
    lrg_fb = lrg.get_field_bboxes(1.0 / lrg.width_mm)
    assert abs(med_fb["region"].x_min - lrg_fb["region"].x_min) <= 0.035
    assert abs(med_fb["classification"].x_max - lrg_fb["classification"].x_max) <= 0.045

    rng = random.Random(123)
    for size in (PlateSize.MEDIUM, PlateSize.LARGE):
        for reg in ("堺", "長崎", "名古屋", "尾張小牧"):
            p = PlateText(region=reg, classification="581", hiragana="さ", serial=7821)
            for _ in range(15):
                lay = PlateGeometry.calculate_layout(p, size, jitter_rng=rng)
                fboxes = lay.get_field_bboxes(1.0)
                assert fboxes["region"].x_min >= lay.left_bolt.x + 12.5
                assert fboxes["classification"].x_max <= lay.right_bolt.x - 12.5
                assert fboxes["region"].x_max + 5.0 <= fboxes["classification"].x_min
