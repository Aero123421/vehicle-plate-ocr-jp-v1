"""High-fidelity 2D parametric renderer for Japanese license plates."""

import math
import random
from typing import ClassVar
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.geometry import CharSlot, PlateGeometry, PlateLayout
from vpocr_synth.plate.models import (
    BoundingBox,
    CharacterBox,
    PlateConfig,
    PlateSize,
    PlateText,
    PlateType,
    RenderResult,
)


class PlateRenderer:
    """Renderer for frontal canonical Japanese license plates.

    Features:
    - Official millimeter layout scaling to arbitrary pixel resolutions
    - High-quality glyph rendering with per-character width scaling
    - Authentic color palettes for standard private, commercial, kei, and graphic plates
    - Stamped metal outer pressed border (外枠) and mounting bolt holes
    - Subtle stamped 3D emboss / specular highlight shading
    - Optional rear aluminum prefecture seal (封印)
    - Procedural background pattern generation for graphic / commemorative plates
    - Pixel-accurate character and field bounding boxes
    - 100% deterministic given a random seed
    """

    # Authentic RGB Color Palettes
    COLOR_PALETTES: ClassVar[dict[PlateType, dict[str, tuple[int, int, int]]]] = {
        PlateType.STANDARD_PRIVATE: {
            "bg": (246, 247, 243),
            "text": (23, 86, 48),       # Stamped deep emerald green
            "border": (23, 86, 48),
            "bolt_inner": (210, 214, 210),
            "bolt_rim": (130, 136, 130),
            "emboss_light": (255, 255, 255),
            "emboss_dark": (15, 60, 32),
        },
        PlateType.STANDARD_COMMERCIAL: {
            "bg": (23, 86, 48),         # Deep emerald green
            "text": (246, 247, 243),     # Crisp white
            "border": (246, 247, 243),
            "bolt_inner": (35, 65, 45),
            "bolt_rim": (180, 200, 180),
            "emboss_light": (255, 255, 255),
            "emboss_dark": (12, 45, 24),
        },
        PlateType.KEI_PRIVATE: {
            "bg": (248, 198, 28),        # Amber / warm yellow
            "text": (28, 28, 28),        # Stamped charcoal black
            "border": (28, 28, 28),
            "bolt_inner": (215, 175, 30),
            "bolt_rim": (90, 80, 20),
            "emboss_light": (255, 220, 80),
            "emboss_dark": (10, 10, 10),
        },
        PlateType.KEI_COMMERCIAL: {
            "bg": (28, 28, 28),         # Stamped charcoal black
            "text": (248, 198, 28),     # Amber / warm yellow
            "border": (248, 198, 28),
            "bolt_inner": (45, 45, 45),
            "bolt_rim": (150, 130, 30),
            "emboss_light": (255, 220, 80),
            "emboss_dark": (10, 10, 10),
        },
        PlateType.GRAPHIC: {
            "bg": (248, 247, 244),
            "text": (23, 86, 48),
            "border": (23, 86, 48),
            "bolt_inner": (210, 214, 210),
            "bolt_rim": (130, 136, 130),
            "emboss_light": (255, 255, 255),
            "emboss_dark": (15, 60, 32),
        },
    }

    # Prefecture single-kanji abbreviations for aluminum seal (封印)
    PREFECTURE_SEAL_CHARS: ClassVar[dict[str, str]] = {
        "東京都": "東", "神奈川県": "神", "大阪府": "大", "愛知県": "愛",
        "北海道": "札", "埼玉県": "埼", "千葉県": "千", "兵庫県": "兵",
        "福岡県": "福", "静岡県": "静", "広島県": "広", "京都府": "京",
        "宮城県": "宮", "新潟県": "新", "長野県": "長", "岐阜県": "岐",
        "栃木県": "栃", "群馬県": "群", "茨城県": "茨", "三重県": "三",
        "福島県": "福", "熊本県": "熊", "岡山県": "岡", "鹿児島県": "鹿",
    }

    _FONT_CACHE: ClassVar[dict[tuple[str, int, int], ImageFont.FreeTypeFont]] = {}

    @classmethod
    def _get_font(cls, font_path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
        key = (font_path, size, index)
        cached = cls._FONT_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            font = ImageFont.truetype(font_path, size, index=index)
        except Exception:
            available = PlateDataLoader.get_available_fonts()
            fallback = available[0] if available else "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
            font = ImageFont.truetype(fallback, size, index=0)
        cls._FONT_CACHE[key] = font
        return font

    def __init__(self, default_font_path: str | None = None) -> None:
        if default_font_path is None:
            available = PlateDataLoader.get_available_fonts()
            self.default_font_path = (
                available[0] if available else "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
            )
        else:
            self.default_font_path = default_font_path

    def render(
        self,
        plate_text: PlateText,
        config: PlateConfig | None = None,
    ) -> RenderResult:
        """Render a frontal license plate with bounding boxes and metadata."""
        if config is None:
            config = PlateConfig()

        rng = random.Random(config.seed)
        np_rng = np.random.default_rng(config.seed)

        scale = config.scale
        layout = PlateGeometry.calculate_layout(plate_text, config.plate_size)

        width_px = int(round(layout.width_mm * scale))
        height_px = int(round(layout.height_mm * scale))

        # 1. Colors
        palette = self.COLOR_PALETTES.get(config.plate_type, self.COLOR_PALETTES[PlateType.STANDARD_PRIVATE]).copy()
        if (
            config.plate_type == PlateType.STANDARD_COMMERCIAL
            and not config.is_graphic
            and not config.color_scheme
            and config.seed is not None
        ):
            # Diversify commercial green across deep emerald, sun-faded olive, and quarry/slate green
            u_grn = rng.random()
            if u_grn < 0.42:
                palette["bg"] = (rng.randint(12, 28), rng.randint(52, 86), rng.randint(24, 48))
            elif u_grn < 0.72:
                palette["bg"] = (rng.randint(30, 56), rng.randint(64, 96), rng.randint(28, 52))
            else:
                palette["bg"] = (rng.randint(42, 72), rng.randint(66, 98), rng.randint(52, 86))
        if config.color_scheme:
            palette.update(config.color_scheme)

        # 2. Base Canvas
        canvas = Image.new("RGBA", (width_px, height_px), palette["bg"] + (255,))

        # 3. Graphic / Commemorative background pattern
        if config.is_graphic or config.plate_type == PlateType.GRAPHIC:
            if not config.graphic_pattern:
                config.graphic_pattern = rng.choice(["mountain", "wave", "flower", "geometric", "rainbow"])
            if config.pattern_seed is None:
                config.pattern_seed = rng.randint(0, 1_000_000)
            p_rng = random.Random(config.pattern_seed)
            self._render_graphic_pattern(canvas, scale, config.graphic_pattern, p_rng)

        # 4. Outer Border (pressed edge rim)
        if config.with_border:
            self._render_outer_border(canvas, layout, scale, palette, config)

        # 5. Text & Symbols Layer
        text_layer = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
        char_boxes: list[CharacterBox] = []

        font_path = config.font_path or self.default_font_path
        font_idx = config.font_index

        for slot in layout.slots:
            char_bbox = self._render_slot(
                text_layer=text_layer,
                slot=slot,
                scale=scale,
                text_color=palette["text"],
                font_path=font_path,
                font_index=font_idx,
            )
            char_boxes.append(
                CharacterBox(
                    char=slot.char,
                    field=slot.field,
                    index=slot.index,
                    bbox=char_bbox,
                )
            )

        # 5.5 Illuminated Plate (字光式) Emissive Stroke Glow
        if config.is_illuminated:
            text_layer = self._apply_illuminated_glow(canvas, text_layer, config)

        # 6. Emboss / Stamped Relief Shading
        if config.with_emboss and config.emboss_intensity > 0.0 and not config.is_illuminated:
            self._apply_emboss(canvas, text_layer, config.emboss_intensity, palette)
        else:
            canvas.alpha_composite(text_layer)

        # 7. Bolt Holes and Seal
        if config.with_bolt_holes:
            self._render_bolt_holes(canvas, layout, scale, palette, config, plate_text, rng)

        # 7.5 Optional Persistent Surface Artifacts (weathering, frame, mud, occluder)
        occlusion_mask = np.zeros((height_px, width_px), dtype=np.float32)
        has_surface_artifacts = (
            config.weathering > 0.0
            or config.plate_frame
            or config.mud_level > 0.0
            or config.persistent_occluder_slot is not None
        )
        if has_surface_artifacts:
            surf_seed = (config.seed if config.seed is not None else 0) + 7919
            surf_rng = random.Random(surf_seed)
            surf_np_rng = np.random.default_rng(surf_seed)

            if config.weathering > 0.0:
                self._apply_weathering(canvas, text_layer, layout, scale, config.weathering, surf_rng, surf_np_rng)

            if config.plate_frame:
                self._apply_plate_frame(canvas, layout, scale, occlusion_mask, surf_rng)

            if config.mud_level > 0.0:
                self._apply_mud_splatter(canvas, config.mud_level, occlusion_mask, surf_rng, surf_np_rng)

            if config.persistent_occluder_slot is not None:
                slot_boxes_12 = self.get_canonical_slot_boxes(plate_text, layout, scale, char_boxes)
                if 0 <= config.persistent_occluder_slot < 12:
                    target_box = slot_boxes_12[config.persistent_occluder_slot]
                    self._apply_persistent_occluder(canvas, target_box, scale, occlusion_mask, surf_rng)

        # 8. Compute field-level bounding boxes
        field_boxes = self._compute_field_boxes(char_boxes)

        # 9. 4 corner coordinates (top-left, top-right, bottom-right, bottom-left)
        corners = [
            (0.0, 0.0),
            (float(width_px), 0.0),
            (float(width_px), float(height_px)),
            (0.0, float(height_px)),
        ]

        rgb_image = canvas.convert("RGB")
        return RenderResult(
            image=rgb_image,
            plate_text=plate_text,
            config=config,
            corners=corners,
            character_boxes=char_boxes,
            field_boxes=field_boxes,
            occlusion_mask=occlusion_mask,
        )

    def _render_slot(
        self,
        text_layer: Image.Image,
        slot: CharSlot,
        scale: float,
        text_color: tuple[int, int, int],
        font_path: str,
        font_index: int,
    ) -> BoundingBox:
        """Render a character, dot, or hyphen into text_layer and return its pixel bounding box."""
        slot_w_px = slot.width * scale
        slot_h_px = slot.height * scale
        slot_x_px = slot.x * scale
        slot_y_px = slot.y * scale

        if slot.is_dot:
            # Filled circle for dot '・'
            draw = ImageDraw.Draw(text_layer)
            dot_bbox = [slot_x_px, slot_y_px, slot_x_px + slot_w_px, slot_y_px + slot_h_px]
            draw.ellipse(dot_bbox, fill=text_color + (255,))
            return BoundingBox(
                x_min=float(slot_x_px),
                y_min=float(slot_y_px),
                x_max=float(slot_x_px + slot_w_px),
                y_max=float(slot_y_px + slot_h_px),
            )

        if slot.char == "-":
            # Hyphen bar
            draw = ImageDraw.Draw(text_layer)
            hyphen_bbox = [slot_x_px, slot_y_px, slot_x_px + slot_w_px, slot_y_px + slot_h_px]
            draw.rectangle(hyphen_bbox, fill=text_color + (255,))
            return BoundingBox(
                x_min=float(slot_x_px),
                y_min=float(slot_y_px),
                x_max=float(slot_x_px + slot_w_px),
                y_max=float(slot_y_px + slot_h_px),
            )

        # Character glyph rendering
        # Render high-res glyph mask
        render_size = 200
        font = self._get_font(font_path, render_size, font_index)

        # Get glyph bbox
        glyph_bbox = font.getbbox(slot.char)
        gw = glyph_bbox[2] - glyph_bbox[0]
        gh = glyph_bbox[3] - glyph_bbox[1]
        if gw <= 0 or gh <= 0:
            gw, gh = render_size // 2, render_size

        pad = 20
        tmp_img = Image.new("RGBA", (gw + 2 * pad, gh + 2 * pad), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp_img)
        tmp_draw.text((pad - glyph_bbox[0], pad - glyph_bbox[1]), slot.char, font=font, fill=text_color + (255,))
        cropped = tmp_img.crop((pad, pad, pad + gw, pad + gh))

        # Aspect ratio and scaling
        target_h = int(round(slot_h_px * 0.94))  # slight vertical inset
        target_w = int(round(slot_w_px * 0.94))  # slight horizontal inset

        if slot.field == "serial" and slot.char == "1":
            # Authentic plate '1': keep natural aspect ratio instead of over-stretching
            aspect = gw / max(1, gh)
            calc_w = int(round(target_h * aspect * 1.05))
            target_w = min(target_w, max(int(round(slot_w_px * 0.5)), calc_w))

        target_w = max(2, target_w)
        target_h = max(2, target_h)

        resized = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Center inside slot box
        off_x = int(round(slot_x_px + (slot_w_px - target_w) / 2.0))
        off_y = int(round(slot_y_px + (slot_h_px - target_h) / 2.0))

        text_layer.paste(resized, (off_x, off_y), resized)

        return BoundingBox(
            x_min=float(off_x),
            y_min=float(off_y),
            x_max=float(off_x + target_w),
            y_max=float(off_y + target_h),
        )

    def _render_outer_border(
        self,
        canvas: Image.Image,
        layout: PlateLayout,
        scale: float,
        palette: dict[str, tuple[int, int, int]],
        config: PlateConfig,
    ) -> None:
        """Render pressed perimeter ridge and rounded corners per MLIT specifications."""
        draw = ImageDraw.Draw(canvas)
        inset_px = layout.border_inset_mm * scale
        border_w_px = max(1, int(round(layout.border_width_mm * scale)))
        radius_px = layout.corner_radius_mm * scale

        width_px = canvas.width
        height_px = canvas.height

        # MLIT Graphic Plate Border Color Rules:
        # - Kei Private Graphic: Bright Yellow Border
        # - Standard Commercial Graphic: Green Border
        # - Standard Private Graphic: Border matches subtle plate body
        border_color = palette["border"]
        if config.is_graphic or config.plate_type == PlateType.GRAPHIC:
            if config.plate_type == PlateType.KEI_PRIVATE:
                border_color = (248, 198, 28)  # Official yellow rim
            elif config.plate_type == PlateType.STANDARD_COMMERCIAL:
                border_color = (23, 86, 48)   # Official green rim
            elif config.plate_type == PlateType.STANDARD_PRIVATE or config.plate_type == PlateType.GRAPHIC:
                border_color = (205, 208, 205)  # Subtle rim on white background

        rect = [
            inset_px,
            inset_px,
            width_px - inset_px,
            height_px - inset_px,
        ]
        draw.rounded_rectangle(
            rect,
            radius=radius_px,
            outline=border_color + (255,),
            width=border_w_px,
        )

    def _apply_emboss(
        self,
        canvas: Image.Image,
        text_layer: Image.Image,
        intensity: float,
        palette: dict[str, tuple[int, int, int]],
    ) -> None:
        """Apply subtle stamped 3D emboss highlight and shadow relief."""
        alpha = text_layer.split()[3]
        if alpha.getextrema() == (0, 0):
            canvas.alpha_composite(text_layer)
            return

        # Top-left highlight
        highlight_color = palette.get("emboss_light", (255, 255, 255))
        shadow_color = palette.get("emboss_dark", (0, 0, 0))

        # Shift masks for relief
        # Highlight shifted (-1, -1)
        w, h = canvas.size
        hl_img = Image.new("RGBA", (w, h), highlight_color + (0,))
        hl_alpha = ImageChops.offset(alpha, -1, -1)
        # diff to get border
        hl_mask = ImageChops.subtract(hl_alpha, alpha)
        hl_mask = hl_mask.point(lambda p: int(p * 0.45 * intensity))
        hl_img.putalpha(hl_mask)

        # Shadow shifted (+1, +1)
        sh_img = Image.new("RGBA", (w, h), shadow_color + (0,))
        sh_alpha = ImageChops.offset(alpha, 1, 1)
        sh_mask = ImageChops.subtract(sh_alpha, alpha)
        sh_mask = sh_mask.point(lambda p: int(p * 0.50 * intensity))
        sh_img.putalpha(sh_mask)

        # Composite layers: canvas -> shadow -> highlight -> text
        canvas.alpha_composite(sh_img)
        canvas.alpha_composite(hl_img)
        canvas.alpha_composite(text_layer)

    def _render_bolt_holes(
        self,
        canvas: Image.Image,
        layout: PlateLayout,
        scale: float,
        palette: dict[str, tuple[int, int, int]],
        config: PlateConfig,
        plate_text: PlateText,
        rng: random.Random,
    ) -> None:
        """Render JIS mounting bolt holes (2-bolt medium / 4-bolt M10 large) and optional seal (封印)."""
        draw = ImageDraw.Draw(canvas)
        lower_unbolted = layout.size == PlateSize.LARGE and rng.random() < 0.15

        for bolt_idx, bolt in enumerate(layout.bolts):
            is_top_left = bolt_idx == 0
            is_lower = bolt_idx >= 2
            bx = bolt.x * scale
            by = bolt.y * scale
            br = bolt.radius * scale

            if is_top_left and config.with_seal and config.plate_type in (
                PlateType.STANDARD_PRIVATE,
                PlateType.STANDARD_COMMERCIAL,
            ):
                # Rear plate aluminum seal (封印: ~25mm diameter cap, leaves clearance for region text)
                seal_r = 12.5 * scale
                sh_off = max(1.0, 1.5 * scale)
                draw.ellipse(
                    [bx - seal_r + sh_off, by - seal_r + sh_off, bx + seal_r + sh_off, by + seal_r + sh_off],
                    fill=(18, 22, 20, 110),
                )
                # Aluminum metallic body
                draw.ellipse(
                    [bx - seal_r, by - seal_r, bx + seal_r, by + seal_r],
                    fill=(210, 214, 218, 255),
                    outline=(150, 155, 160, 255),
                    width=max(1, int(round(1.2 * scale))),
                )
                # Inner ring
                inner_r = seal_r * 0.72
                draw.ellipse(
                    [bx - inner_r, by - inner_r, bx + inner_r, by + inner_r],
                    outline=(180, 185, 190, 255),
                    width=1,
                )
                # Stamped prefecture kanji
                pref_info = PlateDataLoader.get_region_map().get(plate_text.region)
                pref_str = pref_info.prefecture if pref_info else "東"
                seal_char = self.PREFECTURE_SEAL_CHARS.get(pref_str, pref_str[:1] if pref_str else "東")

                try:
                    seal_font_size = int(round(inner_r * 1.1))
                    seal_font = self._get_font(config.font_path, seal_font_size, config.font_index)
                    s_bbox = seal_font.getbbox(seal_char)
                    sw = s_bbox[2] - s_bbox[0]
                    sh = s_bbox[3] - s_bbox[1]
                    draw.text(
                        (bx - sw / 2.0 - s_bbox[0], by - sh / 2.0 - s_bbox[1]),
                        seal_char,
                        font=seal_font,
                        fill=(110, 115, 120, 255),
                    )
                except Exception:
                    pass
            elif is_lower and lower_unbolted:
                # ~15% flush/unbolted lower JIS D 5705 mounting hole cavity
                draw.ellipse(
                    [bx - br, by - br, bx + br, by + br],
                    fill=(24, 26, 28, 255),
                    outline=(72, 76, 78, 230),
                    width=max(1, int(round(0.9 * scale))),
                )
            else:
                # Flange washer + M10/M6 6-sided hex bolt head with directional 2.5D shadow & specular bevel.
                # Lower JIS D 5705 holes at y=185mm use a compact collar (`br * 1.00 = 5.5mm`, top >= 179.0mm)
                # so lower hardware NEVER touches serial digit strokes (`y <= 177.2mm`)!
                if is_lower:
                    washer_r = br * 1.00
                    rim_w = max(1, int(round(0.75 * scale)))
                else:
                    washer_r = br * (1.38 if layout.size == PlateSize.LARGE else 1.18)
                    rim_w = max(1, int(round(1.0 * scale)))
                hex_r = washer_r * 0.78
                sh_off = max(1.0, 1.3 * scale)
                draw.ellipse(
                    [bx - washer_r + sh_off, by - washer_r + sh_off, bx + washer_r + sh_off, by + washer_r + sh_off],
                    fill=(15, 20, 18, 105),
                )
                draw.ellipse(
                    [bx - washer_r, by - washer_r, bx + washer_r, by + washer_r],
                    fill=palette["bolt_inner"] + (255,),
                    outline=palette["bolt_rim"] + (255,),
                    width=rim_w,
                )
                hex_pts = [
                    (bx + hex_r * math.cos(k * math.pi / 3.0), by + hex_r * math.sin(k * math.pi / 3.0))
                    for k in range(6)
                ]
                hex_fill = tuple(min(255, c + 18) for c in palette["bolt_inner"]) + (255,)
                draw.polygon(hex_pts, fill=hex_fill, outline=palette["bolt_rim"] + (255,))
                # Top-left specular facet glint
                draw.line([hex_pts[3], hex_pts[4], hex_pts[5]], fill=(248, 250, 252, 215), width=max(1, int(round(0.9 * scale))))

    def _render_graphic_pattern(
        self,
        canvas: Image.Image,
        scale: float,
        pattern_type: str,
        rng: random.Random,
    ) -> None:
        """Render rich procedural graphic illustration for local/commemorative plates."""
        w, h = canvas.size
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if pattern_type == "mountain":
            # Sun or warm sky glow
            sun_x = int(round(rng.uniform(0.15, 0.35) * w))
            sun_y = int(round(rng.uniform(0.3, 0.45) * h))
            sun_r = int(round(rng.uniform(0.15, 0.22) * h))
            sun_col = rng.choice([
                (255, 120, 80, 160),
                (255, 170, 70, 150),
                (245, 90, 70, 170),
            ])
            draw.ellipse(
                [sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r],
                fill=sun_col,
            )

            # Distant layered mountain peaks
            pts_dist = [(0, h)]
            num_peaks = rng.randint(3, 5)
            step = w / num_peaks
            for p in range(num_peaks + 1):
                px = p * step
                py = h * (0.42 + 0.15 * math.sin(p * 2.1 + 0.5))
                pts_dist.append((px, py))
            pts_dist.append((w, h))
            draw.polygon(pts_dist, fill=(120, 135, 185, 175))

            # Near foreground green/navy mountain ridges
            pts_near = [(0, h)]
            for p in range(num_peaks + 2):
                px = p * (w / (num_peaks + 1))
                py = h * (0.55 + 0.18 * math.sin(p * 1.8 + 1.2))
                pts_near.append((px, py))
            pts_near.append((w, h))
            near_col = rng.choice([
                (45, 135, 85, 195),
                (35, 115, 110, 190),
                (60, 120, 160, 190),
            ])
            draw.polygon(pts_near, fill=near_col)

        elif pattern_type == "wave":
            # Layered ocean / river curves
            wave_palettes = [
                [(0, 120, 190, 165), (0, 165, 225, 175), (45, 205, 225, 185)],
                [(25, 95, 165, 175), (40, 140, 200, 175), (90, 210, 200, 180)],
            ]
            colors = rng.choice(wave_palettes)
            for i, col in enumerate(colors):
                pts = [(0, h)]
                phase = rng.uniform(0, math.pi)
                freq = rng.uniform(0.015, 0.025)
                amp = h * rng.uniform(0.08, 0.13)
                base_y = h * (0.55 + i * 0.12)
                for x in range(0, w + 15, 15):
                    y = base_y + math.sin(x * freq + phase) * amp
                    pts.append((x, y))
                pts.append((w, h))
                draw.polygon(pts, fill=col)

        elif pattern_type == "flower":
            # Floral / cherry blossom clusters on bottom-left and lower edge
            petal_colors = [
                (255, 145, 175, 185),
                (255, 165, 195, 180),
                (245, 125, 165, 190),
            ]
            # Centers
            cluster_centers = [
                (int(w * 0.18), int(h * 0.70), int(h * 0.14)),
                (int(w * 0.32), int(h * 0.80), int(h * 0.12)),
                (int(w * 0.12), int(h * 0.84), int(h * 0.10)),
                (int(w * 0.45), int(h * 0.74), int(h * 0.13)),
                (int(w * 0.62), int(h * 0.82), int(h * 0.11)),
            ]
            for bx, by, cr in cluster_centers:
                for a in range(5):
                    angle = a * (2 * math.pi / 5)
                    px = bx + math.cos(angle) * cr
                    py = by + math.sin(angle) * cr
                    pr = int(cr * 0.55)
                    p_col = rng.choice(petal_colors)
                    draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=p_col)
                # Golden blossom pistil
                center_r = int(cr * 0.3)
                draw.ellipse([bx - center_r, by - center_r, bx + center_r, by + center_r], fill=(255, 215, 75, 220))

        elif pattern_type == "geometric":
            # Modern diagonal blocks / event ribbon chevrons
            geo_colors = [
                (230, 65, 65, 175),
                (245, 165, 35, 175),
                (55, 180, 85, 175),
                (45, 135, 225, 175),
                (145, 75, 210, 175),
            ]
            num_bands = rng.randint(4, 6)
            band_w = int(w * 0.12)
            slant = int(h * 0.35)
            for i in range(num_bands):
                col = geo_colors[i % len(geo_colors)]
                x0 = int(i * band_w * 0.85)
                pts = [
                    (x0, h),
                    (x0 + band_w, h),
                    (x0 + band_w + slant, int(h * 0.52)),
                    (x0 + slant, int(h * 0.52)),
                ]
                draw.polygon(pts, fill=col)

        else:  # "rainbow"
            # Multi-colored arc across lower left to right
            rb_colors = [
                (240, 65, 65),
                (250, 140, 45),
                (250, 215, 55),
                (65, 190, 85),
                (55, 140, 240),
                (145, 65, 205),
            ]
            arc_thickness = max(4, int(round(5.0 * scale)))
            base_r = int(h * 1.15)
            arc_cx = int(-w * 0.1)
            arc_cy = int(h * 0.15)
            for i, col in enumerate(rb_colors):
                r = base_r - i * arc_thickness
                bbox = [arc_cx - r, arc_cy - r, arc_cx + r, arc_cy + r]
                draw.arc(bbox, start=270, end=380, fill=col + (185,), width=arc_thickness)

        # Apply soft blur to emulate offset print integration
        blur_r = max(1, int(round(1.5 * scale)))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_r))
        canvas.alpha_composite(overlay)

    def _compute_field_boxes(self, char_boxes: list[CharacterBox]) -> dict[str, BoundingBox]:
        """Compute aggregate bounding box for each field."""
        fields: dict[str, list[BoundingBox]] = {}
        for cb in char_boxes:
            fields.setdefault(cb.field, []).append(cb.bbox)

        res: dict[str, BoundingBox] = {}
        for fname, bboxes in fields.items():
            x_min = min(b.x_min for b in bboxes)
            y_min = min(b.y_min for b in bboxes)
            x_max = max(b.x_max for b in bboxes)
            y_max = max(b.y_max for b in bboxes)
            res[fname] = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        return res

    @staticmethod
    def get_canonical_slot_boxes(
        plate_text: PlateText,
        layout: PlateLayout,
        scale: float,
        char_boxes: list[CharacterBox],
    ) -> list[BoundingBox]:
        """Map the 12 canonical OCR slots (0..3 region, 4..6 class, 7 kana, 8..11 serial) to pixel bboxes."""
        by_field_idx: dict[tuple[str, int], CharacterBox] = {
            (cb.field, cb.index): cb for cb in char_boxes
        }
        is_large = layout.size == PlateSize.LARGE
        reg_y = (22.5 if is_large else 15.0) * scale
        reg_h = 40.0 * scale
        cls_y = reg_y
        cls_h = reg_h

        boxes: list[BoundingBox] = []

        # 0..3: region
        reg_len = len(plate_text.region)
        reg_default_x0 = (100.0 if is_large else 75.0) * scale
        reg_default_x1 = (225.0 if is_large else 165.0) * scale
        for i in range(4):
            cb = by_field_idx.get(("region", i))
            if cb is not None:
                boxes.append(cb.bbox)
            else:
                slot_w = (reg_default_x1 - reg_default_x0) / 4.0
                x0 = reg_default_x0 + i * slot_w
                boxes.append(BoundingBox(x_min=x0, y_min=reg_y, x_max=x0 + slot_w, y_max=reg_y + reg_h))

        # 4..6: classification
        cls_default_x0 = (245.0 if is_large else 191.0) * scale
        cls_default_x1 = (335.0 if is_large else 260.0) * scale
        for i in range(3):
            cb = by_field_idx.get(("classification", i))
            if cb is not None:
                boxes.append(cb.bbox)
            else:
                slot_w = (cls_default_x1 - cls_default_x0) / 3.0
                x0 = cls_default_x0 + i * slot_w
                boxes.append(BoundingBox(x_min=x0, y_min=cls_y, x_max=x0 + slot_w, y_max=cls_y + cls_h))

        # 7: hiragana
        kana_cb = by_field_idx.get(("hiragana", 0))
        if kana_cb is not None:
            boxes.append(kana_cb.bbox)
        else:
            kx = (25.0 if is_large else 20.0) * scale
            ky = (117.5 if is_large else 90.0) * scale
            boxes.append(BoundingBox(x_min=kx, y_min=ky, x_max=kx + 40.0 * scale, y_max=ky + 40.0 * scale))

        # 8..11: serial (4 physical digit/dot slots; skip hyphen at index=2 when 4 digits)
        num_digits = len(str(plate_text.serial))
        digit_y = (77.5 if is_large else 70.0) * scale
        digit_h = (120.0 if is_large else 80.0) * scale
        digit_w = (60.0 if is_large else 40.0) * scale
        slot_x_starts = [82.5, 162.5, 275.0, 355.0] if is_large else [75.0, 130.0, 215.0, 270.0]

        for s_i in range(4):
            cb_idx = s_i if num_digits < 4 else (s_i if s_i < 2 else s_i + 1)
            cb = by_field_idx.get(("serial", cb_idx))
            sx0 = slot_x_starts[s_i] * scale
            full_slot_box = BoundingBox(
                x_min=sx0,
                y_min=digit_y,
                x_max=sx0 + digit_w,
                y_max=digit_y + digit_h,
            )
            if cb is not None and cb.char != "・":
                boxes.append(cb.bbox)
            else:
                # For '・' dot slots, use the full serial slot zone so occluders and readability operate on the slot
                boxes.append(full_slot_box)

        return boxes

    def _apply_illuminated_glow(
        self,
        canvas: Image.Image,
        text_layer: Image.Image,
        config: PlateConfig,
    ) -> Image.Image:
        """Render electroluminescent / LED glowing character strokes (字光式ナンバー)."""
        w, h = canvas.size
        alpha = text_layer.split()[3]
        if config.plate_type in (PlateType.KEI_PRIVATE, PlateType.KEI_COMMERCIAL):
            glow_outer = (255, 235, 70)
            glow_core = (255, 250, 165)
        else:
            glow_outer = (35, 235, 115)
            glow_core = (155, 255, 195)

        halo_r = max(2, int(round(3.5 * config.scale)))
        halo_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=halo_r))
        halo_alpha = halo_alpha.point(lambda p: min(255, int(p * 1.45)))
        halo_img = Image.new("RGBA", (w, h), glow_outer + (0,))
        halo_img.putalpha(halo_alpha)
        canvas.alpha_composite(halo_img)

        core_img = Image.new("RGBA", (w, h), glow_core + (0,))
        core_img.putalpha(alpha)
        return core_img

    def _apply_weathering(
        self,
        canvas: Image.Image,
        text_layer: Image.Image,
        layout: PlateLayout,
        scale: float,
        weathering: float,
        rng: random.Random,
        np_rng: np.random.Generator,
    ) -> None:
        """Apply sun-faded strokes, grime patina, and subtle bolt rust streaks."""
        w_lvl = float(np.clip(weathering, 0.0, 1.0))
        arr = np.array(canvas, dtype=np.float32)
        h, w = arr.shape[:2]

        # Subtle yellowish-gray road grime & UV patina noise
        low_h = max(8, h // 6)
        low_w = max(16, w // 6)
        grime_coarse = np_rng.normal(0.0, 1.0, (low_h, low_w)).astype(np.float32)
        grime_img = Image.fromarray(np.clip((grime_coarse * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8))
        grime_full = np.array(grime_img.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

        patina_tint = np.array([185.0, 178.0, 155.0], dtype=np.float32)
        blend = (0.06 + 0.22 * w_lvl) * grime_full[:, :, np.newaxis]
        arr[:, :, :3] = arr[:, :, :3] * (1.0 - blend) + patina_tint * blend

        # Slight paint fade over character strokes
        t_alpha = np.array(text_layer.split()[3], dtype=np.float32) / 255.0
        fade_strength = 0.28 * w_lvl
        fade_target = np.array([215.0, 215.0, 210.0], dtype=np.float32)
        f_mask = (t_alpha * fade_strength)[:, :, np.newaxis]
        arr[:, :, :3] = arr[:, :, :3] * (1.0 - f_mask) + fade_target * f_mask

        # Rust streaks descending from bolt holes
        for bolt in layout.bolts:
            bx = int(round(bolt.x * scale))
            by = int(round(bolt.y * scale))
            streak_len = int(round((14.0 + 28.0 * w_lvl) * scale))
            streak_w = max(2, int(round(3.2 * scale)))
            y0 = max(0, by)
            y1 = min(h, by + streak_len)
            x0 = max(0, bx - streak_w)
            x1 = min(w, bx + streak_w + 1)
            if y1 > y0 and x1 > x0:
                yy = np.linspace(0.0, 1.0, y1 - y0, dtype=np.float32)[:, np.newaxis]
                xx = np.abs(np.linspace(-1.0, 1.0, x1 - x0, dtype=np.float32))[np.newaxis, :]
                r_alpha = (1.0 - yy) * np.clip(1.0 - xx, 0.0, 1.0) * (0.35 * w_lvl)
                rust_col = np.array([138.0, 82.0, 42.0], dtype=np.float32)
                arr[y0:y1, x0:x1, :3] = (
                    arr[y0:y1, x0:x1, :3] * (1.0 - r_alpha[:, :, np.newaxis])
                    + rust_col * r_alpha[:, :, np.newaxis]
                )

        updated = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")
        canvas.paste(updated, (0, 0))

    def _apply_plate_frame(
        self,
        canvas: Image.Image,
        layout: PlateLayout,
        scale: float,
        occlusion_mask: np.ndarray,
        rng: random.Random,
    ) -> None:
        """Render aftermarket chrome or dark resin outer license plate frame."""
        w, h = canvas.size
        frame_w_px = max(2, int(round(rng.uniform(4.5, 6.8) * scale)))
        is_chrome = rng.random() < 0.55
        if is_chrome:
            base_col = (202, 208, 214, 255)
            inner_col = (125, 132, 140, 255)
        else:
            base_col = (34, 36, 40, 255)
            inner_col = (65, 68, 74, 255)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([0, 0, w - 1, h - 1], outline=base_col, width=frame_w_px)
        draw.rectangle(
            [frame_w_px, frame_w_px, w - 1 - frame_w_px, h - 1 - frame_w_px],
            outline=inner_col,
            width=max(1, int(round(1.0 * scale))),
        )
        canvas.alpha_composite(overlay)

        occlusion_mask[:frame_w_px, :] = np.maximum(occlusion_mask[:frame_w_px, :], 0.85)
        occlusion_mask[h - frame_w_px :, :] = np.maximum(occlusion_mask[h - frame_w_px :, :], 0.85)
        occlusion_mask[:, :frame_w_px] = np.maximum(occlusion_mask[:, :frame_w_px], 0.85)
        occlusion_mask[:, w - frame_w_px :] = np.maximum(occlusion_mask[:, w - frame_w_px :], 0.85)

    def _apply_mud_splatter(
        self,
        canvas: Image.Image,
        mud_level: float,
        occlusion_mask: np.ndarray,
        rng: random.Random,
        np_rng: np.random.Generator,
    ) -> None:
        """Render realistic construction-site mud crust and splatter droplets."""
        m_lvl = float(np.clip(mud_level, 0.0, 1.0))
        w, h = canvas.size
        arr = np.array(canvas, dtype=np.float32)

        # Lower-edge mud band + Perlin-like multi-scale blob field
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
        bottom_bias = np.clip((yy - (0.82 - 0.45 * m_lvl)) / 0.35, 0.0, 1.0)

        coarse = np_rng.uniform(0.0, 1.0, (max(6, h // 10), max(12, w // 10))).astype(np.float32)
        coarse_up = np.array(
            Image.fromarray((coarse * 255.0).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        mud_alpha = np.clip((coarse_up * 0.85 + 0.25) * bottom_bias * (0.45 + 0.55 * m_lvl), 0.0, 0.95)

        # Discrete splatter droplets across plate
        splat_mask = Image.new("L", (w, h), 0)
        s_draw = ImageDraw.Draw(splat_mask)
        num_blobs = int(round(12 + 55 * m_lvl))
        for _ in range(num_blobs):
            bx = rng.uniform(0, w)
            by = rng.uniform(h * 0.20, h)
            rx = rng.uniform(2.0, 9.0 + 14.0 * m_lvl) * (w / 660.0)
            ry = rx * rng.uniform(0.6, 1.4)
            op = int(round(rng.uniform(110, 235) * min(1.0, 0.5 + m_lvl)))
            s_draw.ellipse([bx - rx, by - ry, bx + rx, by + ry], fill=op)

        splat_arr = np.array(splat_mask, dtype=np.float32) / 255.0
        total_mud = np.clip(np.maximum(mud_alpha, splat_arr), 0.0, 0.96)

        mud_rgb = np.array([104.0, 88.0, 68.0], dtype=np.float32)
        arr[:, :, :3] = arr[:, :, :3] * (1.0 - total_mud[:, :, np.newaxis]) + mud_rgb * total_mud[:, :, np.newaxis]

        occlusion_mask[:] = np.maximum(occlusion_mask, total_mud)
        canvas.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA"), (0, 0))

    def _apply_persistent_occluder(
        self,
        canvas: Image.Image,
        target_box: BoundingBox,
        scale: float,
        occlusion_mask: np.ndarray,
        rng: random.Random,
    ) -> None:
        """Apply an opaque physical occluder (mud clod / tow hitch / cap) over a target character slot."""
        w, h = canvas.size
        pad_x = max(3.0, 3.5 * scale)
        pad_y = max(3.0, 3.5 * scale)
        x0 = max(0, int(math.floor(target_box.x_min - pad_x)))
        y0 = max(0, int(math.floor(target_box.y_min - pad_y)))
        x1 = min(w, int(math.ceil(target_box.x_max + pad_x)))
        y1 = min(h, int(math.ceil(target_box.y_max + pad_y)))
        if x1 <= x0 or y1 <= y0:
            return

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Irregular polygon + solid core so 100% of character strokes in target_box are obscured
        occ_style = rng.choice(["mud_clod", "bumper_hitch", "tape_patch"])
        if occ_style == "mud_clod":
            col = (82, 68, 50, 255)
            rim = (62, 50, 36, 255)
        elif occ_style == "bumper_hitch":
            col = (42, 44, 48, 255)
            rim = (24, 26, 30, 255)
        else:
            col = (145, 142, 136, 255)
            rim = (110, 108, 102, 255)

        draw.rounded_rectangle([x0, y0, x1, y1], radius=max(2, int(round(4.0 * scale))), fill=col, outline=rim, width=2)
        canvas.alpha_composite(overlay)
        occlusion_mask[y0:y1, x0:x1] = 1.0
