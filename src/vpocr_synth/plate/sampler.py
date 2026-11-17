"""Configurable random sampler for Japanese license plates."""

from dataclasses import dataclass, field
import random
from typing import Any
import numpy as np

from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.models import PlateConfig, PlateSize, PlateText, PlateType
from vpocr_synth.plate.validator import PlateValidator


@dataclass
class SamplerDistribution:
    """Configurable probability distributions for plate generation."""

    # Plate type probabilities
    p_standard_private: float = 0.60
    p_kei_private: float = 0.25
    p_standard_commercial: float = 0.08
    p_kei_commercial: float = 0.04
    p_graphic: float = 0.03

    # Plate size (Large is valid only for standard vehicles)
    p_large_plate: float = 0.06

    # Classification number structure
    p_class_3digit_standard: float = 0.88
    p_class_3digit_latin: float = 0.08   # e.g. 30A, 5FH (MLIT 2017)
    p_class_2digit: float = 0.03         # e.g. 33, 55 (historic/classic)
    p_class_1digit: float = 0.01         # e.g. 5, 3 (vintage)

    # Hiragana usage categories
    p_rental: float = 0.03               # 'わ', 'れ'
    p_sofa: float = 0.005                # 'よ', 'A', 'Y' (US military)

    # Serial number digit count distribution
    p_serial_4digit: float = 0.86        # 1000..9999 (12-34)
    p_serial_3digit: float = 0.05        # 100..999 (・123)
    p_serial_2digit: float = 0.05        # 10..99 (・・12)
    p_serial_1digit: float = 0.04        # 1..9 (・・・1)

    # Popular / 希望ナンバー boost probability
    p_popular_number: float = 0.12

    # Visual rendering options
    p_seal: float = 0.45                 # Aluminum seal on rear plate
    p_emboss: float = 0.95
    p_font_black: float = 0.20           # NotoSansCJK-Black weight
    p_font_serif: float = 0.0            # Excluded by default per real Gothic plate specs


class PlateSampler:
    """Generates random, realistic, and legally valid Japanese license plate specifications."""

    # Popular numbers in Japan's 希望ナンバー制度
    POPULAR_NUMBERS: list[int] = [
        1, 3, 5, 7, 8, 9, 11, 24, 33, 55, 77, 88, 111, 333, 555, 777, 888, 999,
        1000, 1001, 1122, 1188, 2525, 3939, 4649, 7777, 8008, 8888, 9999
    ]

    def __init__(self, dist: SamplerDistribution | None = None, allow_serif: bool = False) -> None:
        self.dist = dist or SamplerDistribution()
        self.regions = PlateDataLoader.load_regions()
        self.region_names = [r.name for r in self.regions]
        self.fonts = PlateDataLoader.get_available_fonts(allow_serif=allow_serif)

    def sample(self, seed: int | None = None) -> tuple[PlateText, PlateConfig]:
        """Sample a valid PlateText and corresponding PlateConfig deterministically."""
        rng = random.Random(seed)

        # 1. Plate Type
        plate_type = rng.choices(
            [
                PlateType.STANDARD_PRIVATE,
                PlateType.KEI_PRIVATE,
                PlateType.STANDARD_COMMERCIAL,
                PlateType.KEI_COMMERCIAL,
                PlateType.GRAPHIC,
            ],
            weights=[
                self.dist.p_standard_private,
                self.dist.p_kei_private,
                self.dist.p_standard_commercial,
                self.dist.p_kei_commercial,
                self.dist.p_graphic,
            ],
            k=1,
        )[0]

        # 2. Plate Size
        if plate_type in (PlateType.KEI_PRIVATE, PlateType.KEI_COMMERCIAL):
            plate_size = PlateSize.MEDIUM
        else:
            plate_size = rng.choices(
                [PlateSize.MEDIUM, PlateSize.LARGE],
                weights=[1.0 - self.dist.p_large_plate, self.dist.p_large_plate],
                k=1,
            )[0]

        # 3. Region
        region = rng.choice(self.region_names)

        # 4. Classification Number
        classification = self._sample_classification(plate_type, plate_size, rng)

        # 5. Hiragana
        hiragana = self._sample_hiragana(plate_type, rng)

        # 6. Serial Number
        serial = self._sample_serial(rng)

        plate_text = PlateText(
            region=region,
            classification=classification,
            hiragana=hiragana,
            serial=serial,
        )

        # Validate generated text
        errors = PlateValidator.validate(plate_text, plate_type)
        if errors:
            raise RuntimeError(f"Sampler generated invalid plate: {plate_text} - {errors}")

        # 7. Render Configuration
        config = self._sample_config(plate_type, plate_size, rng, seed)

        return plate_text, config

    def _sample_classification(
        self,
        plate_type: PlateType,
        plate_size: PlateSize,
        rng: random.Random,
    ) -> str:
        """Sample realistic vehicle classification code."""
        # First digit represents vehicle category
        if plate_size == PlateSize.LARGE:
            # Large plates are typically 1 (truck) or 2 (bus), occasionally 8 (special)
            first_digit = rng.choice(["1", "2", "8"])
        elif plate_type in (PlateType.KEI_PRIVATE, PlateType.GRAPHIC):
            first_digit = rng.choices(["5", "7", "4"], weights=[0.8, 0.15, 0.05], k=1)[0]
        elif plate_type == PlateType.KEI_COMMERCIAL:
            first_digit = rng.choices(["4", "6"], weights=[0.9, 0.1], k=1)[0]
        elif plate_type == PlateType.STANDARD_COMMERCIAL:
            first_digit = rng.choices(["1", "2", "4", "8", "3"], weights=[0.4, 0.25, 0.2, 0.1, 0.05], k=1)[0]
        else:
            # Standard private: 3 (regular passenger), 5/7 (small passenger), 1/4 (truck/van), 8 (special)
            first_digit = rng.choices(["3", "5", "7", "1", "4", "8"], weights=[0.55, 0.30, 0.05, 0.04, 0.04, 0.02], k=1)[0]

        style = rng.choices(
            ["3digit_std", "3digit_latin", "2digit", "1digit"],
            weights=[
                self.dist.p_class_3digit_standard,
                self.dist.p_class_3digit_latin,
                self.dist.p_class_2digit,
                self.dist.p_class_1digit,
            ],
            k=1,
        )[0]

        allowed_latin = sorted(PlateValidator.ALLOWED_CLASSIFICATION_LETTERS)

        if style == "3digit_std":
            d2 = str(rng.randint(0, 9))
            d3 = str(rng.randint(0, 9))
            return f"{first_digit}{d2}{d3}"
        elif style == "3digit_latin":
            # MLIT 2017: Latin letter typically appears in position 3, or both 2 and 3
            if rng.random() < 0.85:
                d2 = str(rng.randint(0, 9))
                l3 = rng.choice(allowed_latin)
                return f"{first_digit}{d2}{l3}"
            else:
                l2 = rng.choice(allowed_latin)
                l3 = rng.choice(allowed_latin)
                return f"{first_digit}{l2}{l3}"
        elif style == "2digit":
            d2 = str(rng.randint(0, 9))
            return f"{first_digit}{d2}"
        else:
            return first_digit

    def _sample_hiragana(self, plate_type: PlateType, rng: random.Random) -> str:
        """Sample legal kana symbol for the given plate category."""
        # Rental and SOFA only apply to private categories (standard private or kei private)
        if plate_type in (PlateType.STANDARD_PRIVATE, PlateType.KEI_PRIVATE, PlateType.GRAPHIC):
            # Check rental
            if rng.random() < self.dist.p_rental:
                if plate_type == PlateType.KEI_PRIVATE:
                    return "わ"
                else:
                    return rng.choice(list(PlateValidator.RENTAL_HIRAGANA_STANDARD))

            # Check SOFA (US military SOFA plates are never municipal graphic plates)
            if plate_type != PlateType.GRAPHIC and rng.random() < self.dist.p_sofa:
                return rng.choice(["よ", "A", "Y", "E"])

        # Category standard
        match plate_type:
            case PlateType.STANDARD_PRIVATE:
                return rng.choice(list(PlateValidator.STANDARD_PRIVATE_HIRAGANA))
            case PlateType.STANDARD_COMMERCIAL:
                return rng.choice(list(PlateValidator.STANDARD_COMMERCIAL_HIRAGANA))
            case PlateType.KEI_PRIVATE:
                return rng.choice(list(PlateValidator.KEI_PRIVATE_HIRAGANA))
            case PlateType.KEI_COMMERCIAL:
                return rng.choice(list(PlateValidator.KEI_COMMERCIAL_HIRAGANA))
            case PlateType.GRAPHIC:
                return rng.choice(list(PlateValidator.STANDARD_PRIVATE_HIRAGANA))

    def _sample_serial(self, rng: random.Random) -> int:
        """Sample serial number (1..9999) with realistic digit length and popular numbers."""
        if rng.random() < self.dist.p_popular_number:
            return rng.choice(self.POPULAR_NUMBERS)

        digit_len = rng.choices(
            [4, 3, 2, 1],
            weights=[
                self.dist.p_serial_4digit,
                self.dist.p_serial_3digit,
                self.dist.p_serial_2digit,
                self.dist.p_serial_1digit,
            ],
            k=1,
        )[0]

        if digit_len == 4:
            return rng.randint(1000, 9999)
        elif digit_len == 3:
            return rng.randint(100, 999)
        elif digit_len == 2:
            return rng.randint(10, 99)
        else:
            return rng.randint(1, 9)

    def _sample_config(
        self,
        plate_type: PlateType,
        plate_size: PlateSize,
        rng: random.Random,
        seed: int | None,
    ) -> PlateConfig:
        """Sample rendering options and aesthetic parameters."""
        # Font selection
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if self.fonts:
            r = rng.random()
            if r < self.dist.p_font_serif:
                serifs = [f for f in self.fonts if "Serif" in f]
                if serifs:
                    font_path = rng.choice(serifs)
            elif r < self.dist.p_font_serif + self.dist.p_font_black:
                blacks = [f for f in self.fonts if "Black" in f]
                if blacks:
                    font_path = rng.choice(blacks)
            else:
                bolds = [f for f in self.fonts if "Bold" in f and "Sans" in f]
                if bolds:
                    font_path = rng.choice(bolds)

        with_seal = (
            rng.random() < self.dist.p_seal
            and plate_type in (PlateType.STANDARD_PRIVATE, PlateType.STANDARD_COMMERCIAL)
        )
        with_emboss = rng.random() < self.dist.p_emboss
        is_graphic = (plate_type == PlateType.GRAPHIC)
        pattern_type = None
        pattern_seed = None
        if is_graphic:
            pattern_type = rng.choice(["mountain", "wave", "flower", "geometric", "rainbow"])
            pattern_seed = rng.randint(0, 1_000_000)

        return PlateConfig(
            plate_type=plate_type,
            plate_size=plate_size,
            scale=2.0,
            font_path=font_path,
            font_index=0,
            with_bolt_holes=True,
            with_seal=with_seal,
            with_border=True,
            with_emboss=with_emboss,
            emboss_intensity=rng.uniform(0.35, 0.65),
            is_graphic=is_graphic,
            graphic_pattern=pattern_type,
            pattern_seed=pattern_seed,
            seed=seed,
        )
