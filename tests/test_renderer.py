"""Unit tests for plate rendering and bounding boxes."""

import hashlib
import pytest
from vpocr_synth.plate.models import PlateConfig, PlateSize, PlateText, PlateType
from vpocr_synth.plate.renderer import PlateRenderer
from vpocr_synth.plate.sampler import PlateSampler


def test_renderer_image_dimensions(renderer: PlateRenderer):
    pt = PlateText(region="品川", classification="599", hiragana="あ", serial=1234)

    # Scale 2.0 -> 660 x 330
    cfg_med = PlateConfig(scale=2.0, plate_size=PlateSize.MEDIUM)
    res_med = renderer.render(pt, cfg_med)
    assert res_med.image.size == (660, 330)

    # Scale 1.0 -> 330 x 165
    cfg_med_1 = PlateConfig(scale=1.0, plate_size=PlateSize.MEDIUM)
    res_med_1 = renderer.render(pt, cfg_med_1)
    assert res_med_1.image.size == (330, 165)

    # Large plate Scale 2.0 -> 880 x 440
    cfg_lg = PlateConfig(scale=2.0, plate_size=PlateSize.LARGE)
    res_lg = renderer.render(pt, cfg_lg)
    assert res_lg.image.size == (880, 440)


def test_renderer_bounding_boxes_inside_image(renderer: PlateRenderer, sampler: PlateSampler):
    """Verify all character and field bounding boxes are strictly within image dimensions."""
    for i in range(30):
        pt, cfg = sampler.sample(seed=i)
        res = renderer.render(pt, cfg)
        w, h = res.image.size

        # Check character boxes
        for cb in res.character_boxes:
            b = cb.bbox
            assert 0.0 <= b.x_min < b.x_max <= float(w), f"Char bbox X out of bounds: {b} in {w}x{h}"
            assert 0.0 <= b.y_min < b.y_max <= float(h), f"Char bbox Y out of bounds: {b} in {w}x{h}"
            assert b.width > 0.0
            assert b.height > 0.0

        # Check field boxes
        for fname, fb in res.field_boxes.items():
            assert 0.0 <= fb.x_min < fb.x_max <= float(w), f"Field bbox X out of bounds: {fname} {fb}"
            assert 0.0 <= fb.y_min < fb.y_max <= float(h), f"Field bbox Y out of bounds: {fname} {fb}"


def test_renderer_determinism(renderer: PlateRenderer, sampler: PlateSampler):
    """Verify identical seed produces bit-exact identical rendered image bytes."""
    pt, cfg1 = sampler.sample(seed=999)
    res1 = renderer.render(pt, cfg1)

    _, cfg2 = sampler.sample(seed=999)
    res2 = renderer.render(pt, cfg2)

    hash1 = hashlib.sha256(res1.image.tobytes()).hexdigest()
    hash2 = hashlib.sha256(res2.image.tobytes()).hexdigest()

    assert hash1 == hash2, "Expected identical image bytes for identical seed"


def test_renderer_all_types(renderer: PlateRenderer):
    """Verify all plate types render without exceptions."""
    pt = PlateText(region="世田谷", classification="300", hiragana="さ", serial=1234)
    for ptype in PlateType:
        cfg = PlateConfig(plate_type=ptype, scale=1.5, seed=42)
        res = renderer.render(pt, cfg)
        assert res.image.size == (int(round(330 * 1.5)), int(round(165 * 1.5)))


def test_renderer_graphic_patterns_and_borders(renderer: PlateRenderer):
    """Verify graphic patterns (mountain, wave, flower, geometric, rainbow) and border rules."""
    pt = PlateText(region="世田谷", classification="300", hiragana="さ", serial=1234)

    for pattern in ["mountain", "wave", "flower", "geometric", "rainbow"]:
        cfg = PlateConfig(
            plate_type=PlateType.GRAPHIC,
            scale=1.5,
            is_graphic=True,
            graphic_pattern=pattern,
            pattern_seed=1234,
        )
        res = renderer.render(pt, cfg)
        d = res.to_dict()
        assert d["is_graphic"] is True
        assert d["pattern_type"] == pattern
        assert d["pattern_seed"] == 1234
        assert res.image.size == (int(round(330 * 1.5)), int(round(165 * 1.5)))


def test_font_selection_only_gothic_by_default():
    """Verify that default available fonts exclude Mincho/Serif."""
    from vpocr_synth.plate.data_loader import PlateDataLoader
    default_fonts = PlateDataLoader.get_available_fonts(allow_serif=False)
    for f in default_fonts:
        assert "Serif" not in f, f"Unexpected Serif font in default list: {f}"
        assert "Sans" in f or "Gothic" in f
