"""Data loader for regions, charset, and fonts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import yaml


@dataclass(frozen=True)
class RegionInfo:
    """Metadata for a registered Japanese license plate region."""

    name: str
    char_count: int
    prefecture: str
    is_gotochi: bool
    office: str = ""


class PlateDataLoader:
    """Loader for regional metadata, character sets, and font assets."""

    _cached_regions: list[RegionInfo] | None = None
    _cached_region_map: dict[str, RegionInfo] | None = None
    _cached_charset: list[str] | None = None

    @staticmethod
    def get_data_dir() -> Path:
        """Find the data directory containing regions.yaml and charset.txt."""
        # 1. Project root data dir
        cwd_data = Path.cwd() / "data"
        if cwd_data.is_dir() and (cwd_data / "regions.yaml").exists():
            return cwd_data

        # 2. Package relative data dir
        pkg_data = Path(__file__).resolve().parent.parent.parent.parent / "data"
        if pkg_data.is_dir() and (pkg_data / "regions.yaml").exists():
            return pkg_data

        # 3. Inside src fallback
        in_src = Path(__file__).resolve().parent / "data"
        if in_src.is_dir() and (in_src / "regions.yaml").exists():
            return in_src

        # Fallback to cwd_data
        return cwd_data

    @classmethod
    def load_regions(cls) -> list[RegionInfo]:
        """Load all valid Japanese license plate regions."""
        if cls._cached_regions is not None:
            return cls._cached_regions

        data_file = cls.get_data_dir() / "regions.yaml"
        if not data_file.exists():
            raise FileNotFoundError(f"Regions data file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        regions = [
            RegionInfo(
                name=r["name"],
                char_count=r.get("char_count", len(r["name"])),
                prefecture=r.get("prefecture", ""),
                is_gotochi=r.get("is_gotochi", False),
                office=r.get("office", ""),
            )
            for r in raw.get("regions", [])
        ]
        cls._cached_regions = regions
        cls._cached_region_map = {r.name: r for r in regions}
        return regions

    @classmethod
    def get_region_map(cls) -> dict[str, RegionInfo]:
        """Get mapping of region name -> RegionInfo."""
        if cls._cached_region_map is None:
            cls.load_regions()
        assert cls._cached_region_map is not None
        return cls._cached_region_map

    @classmethod
    def load_charset(cls) -> list[str]:
        """Load full OCR character set."""
        if cls._cached_charset is not None:
            return cls._cached_charset

        charset_file = cls.get_data_dir() / "charset.txt"
        if not charset_file.exists():
            raise FileNotFoundError(f"Charset file not found: {charset_file}")

        with open(charset_file, "r", encoding="utf-8") as f:
            chars = [line.strip("\r\n") for line in f if line.strip("\r\n")]

        cls._cached_charset = chars
        return chars

    @classmethod
    def get_available_fonts(cls, allow_serif: bool = False) -> list[str]:
        """List verified installed or bundled fonts suitable for Japanese license plates.

        By default, only Sans/Gothic fonts are returned to match real embossed plates.
        Serif (Mincho) fonts are excluded unless allow_serif is explicitly True.
        """
        fonts_dir = cls.get_data_dir() / "fonts"
        bundled = [
            str(fonts_dir / name)
            for name in (
                "BIZUDPGothic-Bold.ttf",
                "MPLUSRounded1c-Bold.ttf",
                "MPLUSRounded1c-Medium.ttf",
                "ZenMaruGothic-Bold.ttf",
                "ZenMaruGothic-Medium.ttf",
            )
        ]
        candidates = [
            *bundled,
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        if allow_serif:
            candidates.extend([
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            ])
        return [f for f in candidates if os.path.isfile(f)]
