"""Data models for Japanese license plate generator."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from PIL import Image


class PlateType(str, Enum):
    """Japanese license plate categories and color schemes."""

    STANDARD_PRIVATE = "standard_private"      # 白地に緑文字 (登録車・自家用)
    STANDARD_COMMERCIAL = "standard_commercial"  # 緑地に白文字 (登録車・事業用)
    KEI_PRIVATE = "kei_private"                # 黄地に黒文字 (軽自動車・自家用)
    KEI_COMMERCIAL = "kei_commercial"          # 黒地に黄文字 (軽自動車・事業用)
    GRAPHIC = "graphic"                        # 図柄入り・記念 (白地/淡色地に絵柄、緑文字)


class PlateSize(str, Enum):
    """Physical plate dimensions."""

    MEDIUM = "medium"  # 中型番号標: 330mm x 165mm (普通・小型乗用車、軽自動車等)
    LARGE = "large"    # 大型番号標: 440mm x 220mm (車両総重量8t以上、最大積載量5t以上の大型等)


@dataclass(frozen=True)
class PlateText:
    """Structured text content of a Japanese license plate."""

    region: str          # 地域名 (例: "品川", "世田谷", "尾張小牧", "柏")
    classification: str  # 分類番号 (例: "500", "30A", "52", "5")
    hiragana: str        # ひらがな (例: "あ", "わ", "よ")
    serial: int          # 一連指定番号 (1..9999)

    def __post_init__(self) -> None:
        if not (1 <= self.serial <= 9999):
            raise ValueError(f"Serial must be 1..9999, got {self.serial}")

    @property
    def formatted_serial(self) -> str:
        """Formatted 4-slot serial string per MLIT specification.

        4 digits: "12-34" (hyphenated)
        3 digits: "・123" (1 dot, no hyphen)
        2 digits: "・・12" (2 dots, no hyphen)
        1 digit:  "・・・1" (3 dots, no hyphen)
        """
        s = str(self.serial)
        if len(s) == 4:
            return f"{s[:2]}-{s[2:]}"
        elif len(s) == 3:
            return f"・{s}"
        elif len(s) == 2:
            return f"・・{s}"
        else:
            return f"・・・{s}"

    @property
    def full_text(self) -> str:
        """Full human-readable representation, e.g. '品川 599 あ 12-34'."""
        return f"{self.region} {self.classification} {self.hiragana} {self.formatted_serial}"

    @property
    def ocr_string(self) -> str:
        """Normalized string for OCR model recognition without spaces, e.g. '品川599あ12-34'."""
        return f"{self.region}{self.classification}{self.hiragana}{self.formatted_serial}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "region": self.region,
            "classification": self.classification,
            "hiragana": self.hiragana,
            "serial": self.serial,
            "formatted_serial": self.formatted_serial,
            "full_text": self.full_text,
            "ocr_string": self.ocr_string,
        }


@dataclass(frozen=True)
class BoundingBox:
    """2D axis-aligned bounding box in pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x_min)),
            int(round(self.y_min)),
            int(round(self.x_max)),
            int(round(self.y_max)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": round(self.x_min, 2),
            "y_min": round(self.y_min, 2),
            "x_max": round(self.x_max, 2),
            "y_max": round(self.y_max, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }


@dataclass(frozen=True)
class CharacterBox:
    """Bounding box for an individual rendered character."""

    char: str
    field: str  # "region", "classification", "hiragana", "serial"
    index: int  # 0-based character index within the field
    bbox: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "char": self.char,
            "field": self.field,
            "index": self.index,
            "bbox": self.bbox.to_dict(),
        }


@dataclass
class PlateConfig:
    """Parametric rendering configuration for a Japanese license plate."""

    plate_type: PlateType = PlateType.STANDARD_PRIVATE
    plate_size: PlateSize = PlateSize.MEDIUM
    scale: float = 2.0  # Pixels per millimeter (e.g. 2.0 -> 660x330 for medium)
    font_path: str = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    font_index: int = 0
    with_bolt_holes: bool = True
    with_seal: bool = False  # Aluminum seal (封印) on rear upper-left bolt (standard cars only)
    with_border: bool = True  # Outer pressed ridge
    with_emboss: bool = True  # 3D stamped emboss shading
    emboss_intensity: float = 0.5
    is_graphic: bool = False
    graphic_pattern: str | None = None  # e.g. "mountain", "wave", "flower", "geometric", "rainbow"
    pattern_seed: int | None = None
    color_scheme: dict[str, tuple[int, int, int]] | None = None
    seed: int | None = None
    mud_level: float = 0.0
    plate_frame: bool = False
    weathering: float = 0.0
    is_illuminated: bool = False
    persistent_occluder_slot: int | None = None
    graphic_border_type: str | None = None  # "none" (standard), "yellow" (kei), "green" (commercial)
    graphic_monochrome: bool = False
    frame_style: str | None = None  # None, "chrome", "black_plastic", "dealer_thick", "carbon"
    bolt_cap_size_mm: float = 0.0
    is_temporary_plate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plate_type": self.plate_type.value,
            "plate_size": self.plate_size.value,
            "scale": self.scale,
            "font_path": self.font_path,
            "font_index": self.font_index,
            "with_bolt_holes": self.with_bolt_holes,
            "with_seal": self.with_seal,
            "with_border": self.with_border,
            "with_emboss": self.with_emboss,
            "emboss_intensity": self.emboss_intensity,
            "is_graphic": self.is_graphic,
            "pattern_type": self.graphic_pattern,
            "pattern_seed": self.pattern_seed,
            "seed": self.seed,
            "mud_level": round(self.mud_level, 3),
            "plate_frame": self.plate_frame,
            "weathering": round(self.weathering, 3),
            "is_illuminated": self.is_illuminated,
            "persistent_occluder_slot": self.persistent_occluder_slot,
            "graphic_border_type": self.graphic_border_type,
            "graphic_monochrome": self.graphic_monochrome,
            "frame_style": self.frame_style,
            "bolt_cap_size_mm": round(self.bolt_cap_size_mm, 2),
            "is_temporary_plate": self.is_temporary_plate,
        }


@dataclass
class RenderResult:
    """Output bundle of the plate generator for a single sample."""

    image: Image.Image
    plate_text: PlateText
    config: PlateConfig
    corners: list[tuple[float, float]]  # [top-left, top-right, bottom-right, bottom-left]
    character_boxes: list[CharacterBox]
    field_boxes: dict[str, BoundingBox]
    occlusion_mask: Any = None  # Optional (H, W) float32 [0.0..1.0] persistent surface occlusion opacity

    def to_dict(self) -> dict[str, Any]:
        """Convert complete sample metadata and ground-truth labels to JSONL dictionary."""
        return {
            "text": self.plate_text.full_text,
            "ocr_string": self.plate_text.ocr_string,
            "fields": self.plate_text.to_dict(),
            "plate_type": self.config.plate_type.value,
            "plate_size": self.config.plate_size.value,
            "is_graphic": self.config.is_graphic,
            "pattern_type": self.config.graphic_pattern if self.config.is_graphic else None,
            "pattern_seed": self.config.pattern_seed if self.config.is_graphic else None,
            "image_size": {"width": self.image.width, "height": self.image.height},
            "corners": [[round(x, 2), round(y, 2)] for x, y in self.corners],
            "character_boxes": [cb.to_dict() for cb in self.character_boxes],
            "field_boxes": {k: v.to_dict() for k, v in self.field_boxes.items()},
            "render_params": self.config.to_dict(),
        }
