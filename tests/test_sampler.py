"""Unit tests for plate random sampler."""

import pytest
from vpocr_synth.plate.models import PlateSize, PlateType
from vpocr_synth.plate.sampler import PlateSampler
from vpocr_synth.plate.validator import PlateValidator


def test_sampler_validity(sampler: PlateSampler):
    """Verify 100 randomly sampled plates are all legally valid."""
    for i in range(100):
        pt, cfg = sampler.sample(seed=i)
        assert PlateValidator.is_valid(pt, cfg.plate_type)


def test_sampler_determinism(sampler: PlateSampler):
    """Verify sampling is 100% deterministic given the same seed."""
    pt1, cfg1 = sampler.sample(seed=12345)
    pt2, cfg2 = sampler.sample(seed=12345)

    assert pt1.full_text == pt2.full_text
    assert pt1.serial == pt2.serial
    assert cfg1.plate_type == cfg2.plate_type
    assert cfg1.plate_size == cfg2.plate_size
    assert cfg1.with_seal == cfg2.with_seal


def test_sampler_variety(sampler: PlateSampler):
    """Verify different seeds generate diverse plate types and digit counts."""
    types_seen = set()
    sizes_seen = set()
    digit_lengths_seen = set()
    latin_seen = False

    for i in range(250):
        pt, cfg = sampler.sample(seed=1000 + i)
        types_seen.add(cfg.plate_type)
        sizes_seen.add(cfg.plate_size)
        digit_lengths_seen.add(len(str(pt.serial)))
        if any(c.isalpha() for c in pt.classification):
            latin_seen = True

    # Expect all 5 plate types
    assert len(types_seen) == 5, f"Expected all 5 plate types, saw: {types_seen}"
    # Expect both medium and large
    assert sizes_seen == {PlateSize.MEDIUM, PlateSize.LARGE}
    # Expect 1, 2, 3, 4 digit serials
    assert digit_lengths_seen == {1, 2, 3, 4}
    # Expect Latin letters in classification numbers
    assert latin_seen is True
