"""Test that every single one of the 138 registered regions renders cleanly."""

import pytest
from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.models import PlateConfig, PlateText, PlateType
from vpocr_synth.plate.renderer import PlateRenderer


def test_all_138_regions_render_successfully(renderer: PlateRenderer):
    """Verify that every region name in Japan renders without font glyph errors or bounding box issues."""
    regions = PlateDataLoader.load_regions()
    assert len(regions) == 138

    cfg = PlateConfig(scale=1.0)
    for r in regions:
        pt = PlateText(region=r.name, classification="300", hiragana="さ", serial=1234)
        res = renderer.render(pt, cfg)
        assert res.image.size == (330, 165)

        # Ensure all region char bboxes are valid
        reg_cbs = [cb for cb in res.character_boxes if cb.field == "region"]
        assert len(reg_cbs) == len(r.name)
        for cb in reg_cbs:
            assert cb.bbox.width > 0
            assert cb.bbox.height > 0
            assert 0 <= cb.bbox.x_min < cb.bbox.x_max <= 330
            assert 0 <= cb.bbox.y_min < cb.bbox.y_max <= 165
