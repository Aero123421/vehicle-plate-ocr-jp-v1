"""Pytest configuration and shared fixtures."""

import pytest
from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.models import PlateConfig, PlateSize, PlateText, PlateType
from vpocr_synth.plate.renderer import PlateRenderer
from vpocr_synth.plate.sampler import PlateSampler


@pytest.fixture(scope="session")
def regions_list():
    return PlateDataLoader.load_regions()


@pytest.fixture(scope="session")
def charset_list():
    return PlateDataLoader.load_charset()


@pytest.fixture(scope="session")
def renderer():
    return PlateRenderer()


@pytest.fixture(scope="session")
def sampler():
    return PlateSampler()
