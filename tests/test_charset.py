"""Unit tests for OCR character set coverage."""

import pytest
from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.validator import PlateValidator


def test_charset_contains_all_regions(regions_list, charset_list):
    """Verify every character from all 138 regions is included in the charset."""
    charset_set = set(charset_list)
    for reg in regions_list:
        for c in reg.name:
            assert c in charset_set, f"Character '{c}' from region '{reg.name}' missing in charset"


def test_charset_contains_digits_and_symbols(charset_list):
    charset_set = set(charset_list)
    for d in "0123456789":
        assert d in charset_set, f"Digit '{d}' missing in charset"

    assert "・" in charset_set
    assert "-" in charset_set


def test_charset_contains_allowed_latin(charset_list):
    charset_set = set(charset_list)
    for l in PlateValidator.ALLOWED_CLASSIFICATION_LETTERS:
        assert l in charset_set, f"MLIT allowed Latin letter '{l}' missing in charset"


def test_charset_contains_all_valid_hiragana(charset_list):
    charset_set = set(charset_list)
    # Check all plate categories
    all_valid_hiragana = (
        PlateValidator.STANDARD_PRIVATE_HIRAGANA
        | PlateValidator.STANDARD_COMMERCIAL_HIRAGANA
        | PlateValidator.KEI_PRIVATE_HIRAGANA
        | PlateValidator.KEI_COMMERCIAL_HIRAGANA
        | PlateValidator.RENTAL_HIRAGANA_STANDARD
    )
    for h in all_valid_hiragana:
        assert h in charset_set, f"Hiragana '{h}' missing in charset"


def test_charset_excludes_forbidden_hiragana(charset_list):
    charset_set = set(charset_list)
    for forbidden in ["お", "し", "へ", "ん", "ゐ", "ゑ"]:
        assert forbidden not in charset_set, f"Forbidden hiragana '{forbidden}' found in charset"
