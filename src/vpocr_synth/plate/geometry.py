"""Physical millimeter geometry and layout calculations for Japanese license plates.

Based on official specifications:
- 道路運送車両法施行規則 第11条 別表 第一号様式 (昭和26年運輸省令第74号)
- 自動車登録番号標等の様式（告示）
"""

from dataclasses import dataclass
import random
from typing import ClassVar
from vpocr_synth.plate.models import BoundingBox, PlateSize, PlateText


@dataclass(frozen=True)
class CharSlot:
    """A layout slot for a single character or symbol in millimeters."""

    char: str
    field: str
    index: int
    x: float
    y: float
    width: float
    height: float
    is_dot: bool = False

    def to_bbox(self, scale: float) -> BoundingBox:
        return BoundingBox(
            x_min=self.x * scale,
            y_min=self.y * scale,
            x_max=(self.x + self.width) * scale,
            y_max=(self.y + self.height) * scale,
        )


@dataclass(frozen=True)
class BoltHole:
    """Mounting bolt hole location in millimeters."""

    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class PlateLayout:
    """Computed millimeter geometry for a specific license plate."""

    size: PlateSize
    width_mm: float
    height_mm: float
    corner_radius_mm: float
    border_inset_mm: float
    border_width_mm: float
    left_bolt: BoltHole
    right_bolt: BoltHole
    slots: list[CharSlot]
    bolts: tuple[BoltHole, ...] = ()

    def __post_init__(self) -> None:
        if not self.bolts:
            object.__setattr__(self, "bolts", (self.left_bolt, self.right_bolt))

    def get_char_bboxes(self, scale: float) -> list[tuple[str, str, int, BoundingBox]]:
        """Get scaled character bounding boxes as (char, field, index, bbox)."""
        return [(s.char, s.field, s.index, s.to_bbox(scale)) for s in self.slots]

    def get_field_bboxes(self, scale: float) -> dict[str, BoundingBox]:
        """Get aggregate bounding box for each field ('region', 'classification', 'hiragana', 'serial')."""
        fields: dict[str, list[CharSlot]] = {}
        for s in self.slots:
            fields.setdefault(s.field, []).append(s)

        res: dict[str, BoundingBox] = {}
        for fname, fslots in fields.items():
            x_min = min(s.x for s in fslots) * scale
            y_min = min(s.y for s in fslots) * scale
            x_max = max(s.x + s.width for s in fslots) * scale
            y_max = max(s.y + s.height for s in fslots) * scale
            res[fname] = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        return res


class PlateGeometry:
    """Calculates official mm-based layouts for Medium and Large plates."""

    # Medium plate: 330 x 165 mm
    MEDIUM_WIDTH: ClassVar[float] = 330.0
    MEDIUM_HEIGHT: ClassVar[float] = 165.0
    # Large plate: 440 x 220 mm
    LARGE_WIDTH: ClassVar[float] = 440.0
    LARGE_HEIGHT: ClassVar[float] = 220.0

    @classmethod
    def calculate_layout(
        cls,
        plate_text: PlateText,
        plate_size: PlateSize,
        jitter_rng: random.Random | None = None,
    ) -> PlateLayout:
        """Compute millimeter layout according to post-1998 3-digit MLIT 第一号様式.

        When `jitter_rng` is provided (training mode), applies clearance-safe continuous
        layout micro-perturbations (group x-shift, inter-field gap spanning compact and
        regional stamp dies, and mild vertical/scale jitter) directly to the returned
        `PlateLayout` so rendering, canonical slot boxes, and visibility masks all share
        one exact self-consistent geometry object.
        """
        if plate_size == PlateSize.MEDIUM:
            return cls._calculate_medium_layout(plate_text, jitter_rng=jitter_rng)
        else:
            return cls._calculate_large_layout(plate_text, jitter_rng=jitter_rng)

    @classmethod
    def _calculate_medium_layout(
        cls,
        pt: PlateText,
        jitter_rng: random.Random | None = None,
    ) -> PlateLayout:
        """Medium plate (中板: 330 x 165 mm) calibrated MLIT layout calculation."""
        width_mm = cls.MEDIUM_WIDTH
        height_mm = cls.MEDIUM_HEIGHT

        left_bolt = BoltHole(x=60.0, y=25.0, radius=4.5)
        right_bolt = BoltHole(x=270.0, y=25.0, radius=4.5)

        slots: list[CharSlot] = []

        # ==========================================================
        # 1 & 2. Upper Row - Authentic MLIT Fixed-Field Layout
        #        2/3/4-char regions share the same left-anchored region zone
        #        [76..169 mm] so outer bounding boxes never leak region length!
        # ==========================================================
        reg = pt.region
        reg_len = len(reg)
        cls_num = pt.classification
        cls_len = len(cls_num)

        up_y = 12.0
        up_h = 44.0

        reg_cws: list[float] = []
        reg_chs: list[float] = []
        inter_reg_gap = 0.0
        if reg_len == 1:
            # Deterministic: reg_x=108.0, cw=40.0, cls_x0=173.0 for 3-digit class
            reg_cws = [40.0]
            reg_chs = [up_h]
            inter_reg_gap = 0.0
            base_reg_x0 = 108.0
            base_gap = 25.0
            base_cls_cw = 23.0
        elif reg_len == 2:
            # Authentic square 2-Kanji die (~39x44mm, aspect 0.89) + 2.5mm inter-Kanji spacing:
            # reg=[82..162.5 mm], cls=[175..250 mm] — aligned with 3-char & 4-char region boxes!
            reg_cws = [39.0, 39.0]
            reg_chs = [up_h, up_h]
            inter_reg_gap = 2.5
            base_reg_x0 = 82.0
            base_gap = 12.5
            base_cls_cw = 25.0
        elif reg_len == 3:
            for i, c in enumerate(reg):
                if i == 1 and c in ("ケ", "ヶ"):
                    reg_cws.append(25.0)
                    reg_chs.append(35.0)
                else:
                    reg_cws.append(29.0)
                    reg_chs.append(up_h)
            inter_reg_gap = 2.0
            base_reg_x0 = 78.0
            base_gap = 9.0
            base_cls_cw = 24.5
        elif reg_len == 4:
            cw4 = 22.5 if cls_len == 3 else 25.0
            reg_cws = [cw4] * 4
            reg_chs = [up_h] * 4
            inter_reg_gap = 1.0
            base_reg_x0 = 76.0
            base_gap = 8.5
            base_cls_cw = 24.0
        else:
            cw_fb = 88.0 / max(1, reg_len)
            reg_cws = [cw_fb] * reg_len
            reg_chs = [up_h] * reg_len
            inter_reg_gap = 0.0
            base_reg_x0 = 78.0
            base_gap = 9.0
            base_cls_cw = 24.0

        if cls_len == 3:
            cls_cw = base_cls_cw
        elif cls_len == 2:
            cls_cw = 29.0
            base_gap += 9.0
        else:
            cls_cw = 30.0
            base_gap += 18.0

        if jitter_rng is not None:
            up_scale_h = jitter_rng.uniform(0.94, 1.05)
            up_scale_w = jitter_rng.uniform(0.95, 1.04)
            up_h_eff = min(47.0, up_h * up_scale_h)
            up_y = min(61.5 - up_h_eff, max(8.5, up_y + jitter_rng.uniform(-2.0, 2.0)))
            reg_cws = [w * up_scale_w for w in reg_cws]
            reg_chs = [min(up_h_eff, h * up_scale_h) for h in reg_chs]
            if reg_len == 2:
                inter_reg_gap = max(1.0, inter_reg_gap + jitter_rng.uniform(-1.5, 3.0))
            elif reg_len >= 3:
                inter_reg_gap = max(0.0, inter_reg_gap + jitter_rng.uniform(-1.0, 1.5))
            cls_cw = cls_cw * jitter_rng.uniform(0.95, 1.04)
            if cls_len == 3:
                gap_lo, gap_hi = -2.5, 8.0
            elif cls_len == 2:
                gap_lo, gap_hi = -7.0, 16.0
            else:
                gap_lo, gap_hi = -12.0, 24.0
            gap = max(6.5, base_gap + jitter_rng.uniform(gap_lo, gap_hi))
            x_shift = jitter_rng.uniform(-8.0, 4.5) if reg_len == 1 else jitter_rng.uniform(-4.0, 5.0)
            reg_x0 = base_reg_x0 + x_shift
        else:
            up_h_eff = up_h
            gap = base_gap
            reg_x0 = base_reg_x0

        w_reg = sum(reg_cws) + inter_reg_gap * max(0, reg_len - 1)
        w_cls = cls_cw * cls_len
        total_up_w = w_reg + gap + w_cls

        # Enforce bolt/washer clearance [73.0 .. 256.5 mm]
        min_x = left_bolt.x + 13.0
        max_x = right_bolt.x - 13.0
        if total_up_w > (max_x - min_x):
            # Compress inter_reg_gap and gap smoothly if needed so clearance always holds
            excess = total_up_w - (max_x - min_x)
            gap_trim = min(excess, max(0.0, gap - 6.0))
            gap -= gap_trim
            excess -= gap_trim
            if excess > 0.0 and reg_len > 1:
                ig_trim = min(excess / (reg_len - 1), inter_reg_gap)
                inter_reg_gap -= ig_trim
            w_reg = sum(reg_cws) + inter_reg_gap * max(0, reg_len - 1)
            total_up_w = w_reg + gap + w_cls
            if total_up_w > (max_x - min_x):
                scale_fit = ((max_x - min_x) - gap - inter_reg_gap * max(0, reg_len - 1)) / max(1.0, sum(reg_cws) + w_cls)
                reg_cws = [w * scale_fit for w in reg_cws]
                cls_cw *= scale_fit
                w_reg = sum(reg_cws) + inter_reg_gap * max(0, reg_len - 1)
                w_cls = cls_cw * cls_len
                total_up_w = w_reg + gap + w_cls
        if reg_x0 < min_x:
            reg_x0 = min_x
        if reg_x0 + total_up_w > max_x:
            reg_x0 = max(min_x, max_x - total_up_w)
        cls_x0 = reg_x0 + w_reg + gap

        cur_x = reg_x0
        for i, c in enumerate(reg):
            cw = reg_cws[i]
            ch = reg_chs[i]
            cy = up_y + 0.5 * (up_h_eff - ch)
            slots.append(CharSlot(char=c, field="region", index=i, x=cur_x, y=cy, width=cw, height=ch))
            cur_x += cw + (inter_reg_gap if i < reg_len - 1 else 0.0)

        for i, c in enumerate(cls_num):
            slots.append(
                CharSlot(
                    char=c,
                    field="classification",
                    index=i,
                    x=cls_x0 + i * cls_cw,
                    y=up_y,
                    width=cls_cw,
                    height=up_h_eff,
                )
            )

        # ==========================================
        # 3. Lower Row - Kana (y = 88..134 mm)
        # ==========================================
        kana_x = 19.0
        kana_y = 88.0
        kana_w = 42.0
        kana_h = 46.0
        if jitter_rng is not None:
            ks = jitter_rng.uniform(0.92, 1.07)
            kana_w *= ks
            kana_h *= ks
            kana_x = max(14.0, min(68.0 - kana_w, kana_x + jitter_rng.uniform(-2.5, 2.5)))
            kana_y = max(72.0, min(148.0 - kana_h, kana_y + jitter_rng.uniform(-3.0, 3.0)))
        slots.append(
            CharSlot(char=pt.hiragana, field="hiragana", index=0, x=kana_x, y=kana_y, width=kana_w, height=kana_h)
        )

        # ==========================================
        # 4. Lower Row - Serial Number (y = 70..150 mm)
        # ==========================================
        serial_digits = str(pt.serial)
        num_digits = len(serial_digits)
        digit_h = 80.0
        digit_w = 40.0
        digit_y = 70.0
        dot_diam = 12.0
        dot_r = dot_diam / 2.0
        center_y = 110.0

        # Slot centers: 95, 150, 235, 290
        slot_centers = [95.0, 150.0, 235.0, 290.0]
        slot_x_starts = [75.0, 130.0, 215.0, 270.0]

        if num_digits == 4:
            # 4 digits: d1 d2 - d3 d4
            # d1
            slots.append(CharSlot(char=serial_digits[0], field="serial", index=0, x=slot_x_starts[0], y=digit_y, width=digit_w, height=digit_h))
            # d2
            slots.append(CharSlot(char=serial_digits[1], field="serial", index=1, x=slot_x_starts[1], y=digit_y, width=digit_w, height=digit_h))
            # hyphen: width 20mm, height 12mm, centered between d2 and d3 (center 192.5, 110)
            slots.append(CharSlot(char="-", field="serial", index=2, x=182.5, y=104.0, width=20.0, height=12.0))
            # d3
            slots.append(CharSlot(char=serial_digits[2], field="serial", index=3, x=slot_x_starts[2], y=digit_y, width=digit_w, height=digit_h))
            # d4
            slots.append(CharSlot(char=serial_digits[3], field="serial", index=4, x=slot_x_starts[3], y=digit_y, width=digit_w, height=digit_h))
        elif num_digits == 3:
            # 3 digits: ・ d1 d2 d3 (NO hyphen)
            # dot 1
            slots.append(CharSlot(char="・", field="serial", index=0, x=slot_centers[0] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            # d1 in slot 2
            slots.append(CharSlot(char=serial_digits[0], field="serial", index=1, x=slot_x_starts[1], y=digit_y, width=digit_w, height=digit_h))
            # d2 in slot 3
            slots.append(CharSlot(char=serial_digits[1], field="serial", index=2, x=slot_x_starts[2], y=digit_y, width=digit_w, height=digit_h))
            # d3 in slot 4
            slots.append(CharSlot(char=serial_digits[2], field="serial", index=3, x=slot_x_starts[3], y=digit_y, width=digit_w, height=digit_h))
        elif num_digits == 2:
            # 2 digits: ・ ・ d1 d2 (NO hyphen)
            # dot 1
            slots.append(CharSlot(char="・", field="serial", index=0, x=slot_centers[0] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            # dot 2
            slots.append(CharSlot(char="・", field="serial", index=1, x=slot_centers[1] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            # d1 in slot 3
            slots.append(CharSlot(char=serial_digits[0], field="serial", index=2, x=slot_x_starts[2], y=digit_y, width=digit_w, height=digit_h))
            # d2 in slot 4
            slots.append(CharSlot(char=serial_digits[1], field="serial", index=3, x=slot_x_starts[3], y=digit_y, width=digit_w, height=digit_h))
        else:
            # 1 digit: ・ ・ ・ d1 (NO hyphen)
            slots.append(CharSlot(char="・", field="serial", index=0, x=slot_centers[0] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            slots.append(CharSlot(char="・", field="serial", index=1, x=slot_centers[1] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            slots.append(CharSlot(char="・", field="serial", index=2, x=slot_centers[2] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            slots.append(CharSlot(char=serial_digits[0], field="serial", index=3, x=slot_x_starts[3], y=digit_y, width=digit_w, height=digit_h))

        return PlateLayout(
            size=PlateSize.MEDIUM,
            width_mm=width_mm,
            height_mm=height_mm,
            corner_radius_mm=10.0,
            border_inset_mm=4.0,
            border_width_mm=2.5,
            left_bolt=left_bolt,
            right_bolt=right_bolt,
            slots=slots,
            bolts=(left_bolt, right_bolt),
        )

    @classmethod
    def _calculate_large_layout(
        cls,
        pt: PlateText,
        jitter_rng: random.Random | None = None,
    ) -> PlateLayout:
        """Large plate (大板: 440 x 220 mm) calibrated JIS D 5705 4-bolt MLIT layout calculation."""
        width_mm = cls.LARGE_WIDTH
        height_mm = cls.LARGE_HEIGHT

        left_bolt = BoltHole(x=100.0, y=35.0, radius=5.5)
        right_bolt = BoltHole(x=340.0, y=35.0, radius=5.5)
        bottom_left_bolt = BoltHole(x=100.0, y=185.0, radius=5.5)
        bottom_right_bolt = BoltHole(x=340.0, y=185.0, radius=5.5)

        slots: list[CharSlot] = []

        # ===================================================================
        # 1 & 2. Upper Row - Authentic MLIT Fixed-Field Layout inside 100..340mm bolts
        # ===================================================================
        reg = pt.region
        reg_len = len(reg)
        cls_num = pt.classification
        cls_len = len(cls_num)

        up_y = 24.0
        up_h = 48.0

        reg_cws: list[float] = []
        reg_chs: list[float] = []
        inter_reg_gap = 0.0
        if reg_len == 1:
            reg_cws = [48.0]
            reg_chs = [up_h]
            inter_reg_gap = 0.0
            base_reg_x0 = 148.0
            base_gap = 26.0
            base_cls_cw = 29.0
        elif reg_len == 2:
            reg_cws = [43.0, 43.0]
            reg_chs = [up_h, up_h]
            inter_reg_gap = 6.0
            base_reg_x0 = 115.0
            base_gap = 15.0
            base_cls_cw = 32.5
        elif reg_len == 3:
            for i, c in enumerate(reg):
                if i == 1 and c in ("ケ", "ヶ"):
                    reg_cws.append(28.0)
                    reg_chs.append(39.0)
                else:
                    reg_cws.append(32.0)
                    reg_chs.append(up_h)
            inter_reg_gap = 2.0
            base_reg_x0 = 114.0
            base_gap = 12.0
            base_cls_cw = 31.5
        elif reg_len == 4:
            cw4 = 24.5 if cls_len == 3 else 27.5
            reg_cws = [cw4] * 4
            reg_chs = [up_h] * 4
            inter_reg_gap = 1.0
            base_reg_x0 = 114.0
            base_gap = 10.5
            base_cls_cw = 31.0
        else:
            cw_fb = 96.0 / max(1, reg_len)
            reg_cws = [cw_fb] * reg_len
            reg_chs = [up_h] * reg_len
            inter_reg_gap = 0.0
            base_reg_x0 = 115.0
            base_gap = 11.0
            base_cls_cw = 31.0

        if cls_len == 3:
            cls_cw = base_cls_cw
        elif cls_len == 2:
            cls_cw = 35.0
            base_gap += 11.5
        else:
            cls_cw = 37.0
            base_gap += 22.0

        if jitter_rng is not None:
            up_scale_h = jitter_rng.uniform(0.94, 1.05)
            up_scale_w = jitter_rng.uniform(0.95, 1.04)
            up_h_eff = min(52.0, up_h * up_scale_h)
            up_y = min(82.0 - up_h_eff, max(16.0, up_y + jitter_rng.uniform(-2.8, 2.8)))
            reg_cws = [w * up_scale_w for w in reg_cws]
            reg_chs = [min(up_h_eff, h * up_scale_h) for h in reg_chs]
            if reg_len == 2:
                inter_reg_gap = max(1.0, inter_reg_gap + jitter_rng.uniform(-1.5, 3.0))
            elif reg_len >= 3:
                inter_reg_gap = max(0.0, inter_reg_gap + jitter_rng.uniform(-1.0, 1.5))
            cls_cw = cls_cw * jitter_rng.uniform(0.95, 1.04)
            if cls_len == 3:
                gap_lo, gap_hi = -3.0, 8.0
            elif cls_len == 2:
                gap_lo, gap_hi = -8.5, 20.0
            else:
                gap_lo, gap_hi = -15.0, 30.0
            gap = max(8.0, base_gap + jitter_rng.uniform(gap_lo, gap_hi))
            x_shift = jitter_rng.uniform(-9.0, 5.0) if reg_len == 1 else jitter_rng.uniform(-3.0, 6.0)
            reg_x0 = base_reg_x0 + x_shift
        else:
            up_h_eff = up_h
            gap = base_gap
            reg_x0 = base_reg_x0

        w_reg = sum(reg_cws) + inter_reg_gap * max(0, reg_len - 1)
        w_cls = cls_cw * cls_len
        total_up_w = w_reg + gap + w_cls

        min_x = left_bolt.x + 14.0
        max_x = right_bolt.x - 14.0
        if total_up_w > (max_x - min_x):
            excess = total_up_w - (max_x - min_x)
            gap_trim = min(excess, max(0.0, gap - 7.0))
            gap -= gap_trim
            excess -= gap_trim
            if excess > 0.0 and reg_len > 1:
                ig_trim = min(excess / (reg_len - 1), inter_reg_gap)
                inter_reg_gap -= ig_trim
            w_reg = sum(reg_cws) + inter_reg_gap * max(0, reg_len - 1)
            total_up_w = w_reg + gap + w_cls
            if total_up_w > (max_x - min_x):
                scale_fit = ((max_x - min_x) - gap - inter_reg_gap * max(0, reg_len - 1)) / max(1.0, sum(reg_cws) + w_cls)
                reg_cws = [w * scale_fit for w in reg_cws]
                cls_cw *= scale_fit
                w_reg = sum(reg_cws) + inter_reg_gap * max(0, reg_len - 1)
                w_cls = cls_cw * cls_len
                total_up_w = w_reg + gap + w_cls
        if reg_x0 < min_x:
            reg_x0 = min_x
        if reg_x0 + total_up_w > max_x:
            reg_x0 = max(min_x, max_x - total_up_w)
        cls_x0 = reg_x0 + w_reg + gap

        cur_x = reg_x0
        for i, c in enumerate(reg):
            cw = reg_cws[i]
            ch = reg_chs[i]
            cy = up_y + 0.5 * (up_h_eff - ch)
            slots.append(CharSlot(char=c, field="region", index=i, x=cur_x, y=cy, width=cw, height=ch))
            cur_x += cw + (inter_reg_gap if i < reg_len - 1 else 0.0)

        for i, c in enumerate(cls_num):
            slots.append(
                CharSlot(
                    char=c,
                    field="classification",
                    index=i,
                    x=cls_x0 + i * cls_cw,
                    y=up_y,
                    width=cls_cw,
                    height=up_h_eff,
                )
            )

        # ==========================================
        # 3. Lower Row - Kana (y = 104..168 mm, h = 64 mm)
        # ==========================================
        kana_x = 24.0
        kana_y = 104.0
        kana_w = 52.0
        kana_h = 64.0
        if jitter_rng is not None:
            ks = jitter_rng.uniform(0.92, 1.06)
            kana_w *= ks
            kana_h *= ks
            kana_x = max(16.0, min(80.0 - kana_w, kana_x + jitter_rng.uniform(-3.0, 3.0)))
            kana_y = max(90.0, min(179.0 - kana_h, kana_y + jitter_rng.uniform(-3.5, 3.5)))
        slots.append(
            CharSlot(char=pt.hiragana, field="hiragana", index=0, x=kana_x, y=kana_y, width=kana_w, height=kana_h)
        )

        # ==========================================
        # 4. Lower Row - Serial Number (JIS D 5705 90mm tall digits above y=185mm lower bolts)
        # ==========================================
        serial_digits = str(pt.serial)
        num_digits = len(serial_digits)
        digit_h = 90.0
        digit_w = 56.0
        digit_y = 90.0
        dot_diam = 18.0
        dot_r = dot_diam / 2.0
        center_y = 135.0

        slot_x_starts = [86.0, 164.0, 276.0, 355.0]
        slot_centers = [s + digit_w / 2.0 for s in slot_x_starts]

        if num_digits == 4:
            slots.append(CharSlot(char=serial_digits[0], field="serial", index=0, x=slot_x_starts[0], y=digit_y, width=digit_w, height=digit_h))
            slots.append(CharSlot(char=serial_digits[1], field="serial", index=1, x=slot_x_starts[1], y=digit_y, width=digit_w, height=digit_h))
            # hyphen: width 22.5mm, height 18mm
            slots.append(CharSlot(char="-", field="serial", index=2, x=237.5, y=center_y - 9.0, width=22.5, height=18.0))
            slots.append(CharSlot(char=serial_digits[2], field="serial", index=3, x=slot_x_starts[2], y=digit_y, width=digit_w, height=digit_h))
            slots.append(CharSlot(char=serial_digits[3], field="serial", index=4, x=slot_x_starts[3], y=digit_y, width=digit_w, height=digit_h))
        elif num_digits == 3:
            slots.append(CharSlot(char="・", field="serial", index=0, x=slot_centers[0] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            slots.append(CharSlot(char=serial_digits[0], field="serial", index=1, x=slot_x_starts[1], y=digit_y, width=digit_w, height=digit_h))
            slots.append(CharSlot(char=serial_digits[1], field="serial", index=2, x=slot_x_starts[2], y=digit_y, width=digit_w, height=digit_h))
            slots.append(CharSlot(char=serial_digits[2], field="serial", index=3, x=slot_x_starts[3], y=digit_y, width=digit_w, height=digit_h))
        elif num_digits == 2:
            slots.append(CharSlot(char="・", field="serial", index=0, x=slot_centers[0] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            slots.append(CharSlot(char="・", field="serial", index=1, x=slot_centers[1] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            slots.append(CharSlot(char=serial_digits[0], field="serial", index=2, x=slot_x_starts[2], y=digit_y, width=digit_w, height=digit_h))
            slots.append(CharSlot(char=serial_digits[1], field="serial", index=3, x=slot_x_starts[3], y=digit_y, width=digit_w, height=digit_h))
        else:
            slots.append(CharSlot(char="・", field="serial", index=0, x=slot_centers[0] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            slots.append(CharSlot(char="・", field="serial", index=1, x=slot_centers[1] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            slots.append(CharSlot(char="・", field="serial", index=2, x=slot_centers[2] - dot_r, y=center_y - dot_r, width=dot_diam, height=dot_diam, is_dot=True))
            slots.append(CharSlot(char=serial_digits[0], field="serial", index=3, x=slot_x_starts[3], y=digit_y, width=digit_w, height=digit_h))

        return PlateLayout(
            size=PlateSize.LARGE,
            width_mm=width_mm,
            height_mm=height_mm,
            corner_radius_mm=12.0,
            border_inset_mm=5.0,
            border_width_mm=3.0,
            left_bolt=left_bolt,
            right_bolt=right_bolt,
            slots=slots,
            bolts=(left_bolt, right_bolt, bottom_left_bolt, bottom_right_bolt),
        )
