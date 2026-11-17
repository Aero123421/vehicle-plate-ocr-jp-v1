"""Bijective 12-slot Japanese plate mapping and physical per-slot readability calculator."""

import math
import numpy as np

from vpocr_synth.plate.geometry import PlateGeometry
from vpocr_synth.plate.models import BoundingBox, PlateSize, PlateText, RenderResult
from vpocr_synth.plate.renderer import PlateRenderer

PAD_TOKEN: str = "<PAD>"
DOT_TOKEN: str = "・"


class SlotReadabilityMapper:
    """Maps `PlateText` <-> 12 canonical OCR slots and computes `(T, 12)` `slot_readability`."""

    @staticmethod
    def canonicalize_region(region: str) -> str:
        """Canonicalize colloquial region variants (e.g. '袖ヶ浦' -> '袖ケ浦')."""
        if region == "袖ヶ浦":
            return "袖ケ浦"
        return region.replace("ヶ", "ケ")

    @classmethod
    def encode_plate_slots(cls, plate_text: PlateText) -> list[str]:
        """Encode PlateText bijectively into 12 canonical character slots.

        Slots:
          - 0..3: region (4 slots, left-aligned, padded with '<PAD>')
          - 4..6: classification (3 slots, left-aligned, padded with '<PAD>')
          - 7:    hiragana (1 slot)
          - 8..11: serial (4 slots, leading dots '・' for <4 digits)
        """
        reg = cls.canonicalize_region(plate_text.region)
        reg_slots = list(reg[:4]) + [PAD_TOKEN] * max(0, 4 - len(reg))

        cls_str = plate_text.classification
        cls_slots = list(cls_str[:3]) + [PAD_TOKEN] * max(0, 3 - len(cls_str))

        kana_slot = [plate_text.hiragana]

        s_digits = str(plate_text.serial)
        s_padded = (DOT_TOKEN * max(0, 4 - len(s_digits))) + s_digits
        serial_slots = list(s_padded[:4])

        slots = reg_slots + cls_slots + kana_slot + serial_slots
        assert len(slots) == 12
        return slots

    @staticmethod
    def decode_plate_slots(slots: list[str]) -> PlateText:
        """Decode 12 canonical slots back into a validated `PlateText`."""
        if len(slots) != 12:
            raise ValueError(f"Expected 12 slots, got {len(slots)}")

        region = "".join(t for t in slots[0:4] if t != PAD_TOKEN)
        classification = "".join(t for t in slots[4:7] if t != PAD_TOKEN)
        hiragana = slots[7]
        serial_digits = "".join(t for t in slots[8:12] if t not in (PAD_TOKEN, DOT_TOKEN, "-", ""))
        if not serial_digits or not serial_digits.isdigit():
            raise ValueError(f"Invalid serial digits in slots[8:12]: {slots[8:12]}")

        return PlateText(
            region=region,
            classification=classification,
            hiragana=hiragana,
            serial=int(serial_digits),
        )

    @classmethod
    def get_active_slot_mask(cls, plate_text: PlateText) -> np.ndarray:
        """Return boolean mask of shape (12,) where True indicates a non-<PAD> slot."""
        slots = cls.encode_plate_slots(plate_text)
        return np.array([tok != PAD_TOKEN for tok in slots], dtype=bool)

    @staticmethod
    def get_normalized_slot_boxes(
        plate_text: PlateText,
        plate_size: PlateSize = PlateSize.MEDIUM,
    ) -> np.ndarray:
        """Compute exact normalized [u0, v0, u1, v1] in [0, 1] for all 12 slots on the plate."""
        layout = PlateGeometry.calculate_layout(plate_text, plate_size)
        # Convert layout slots into CharacterBox-like bboxes at scale=1.0
        from vpocr_synth.plate.models import CharacterBox

        dummy_cbs = [
            CharacterBox(char=s.char, field=s.field, index=s.index, bbox=s.to_bbox(1.0))
            for s in layout.slots
        ]
        boxes_mm = PlateRenderer.get_canonical_slot_boxes(plate_text, layout, 1.0, dummy_cbs)
        w_mm = layout.width_mm
        h_mm = layout.height_mm
        norm = np.zeros((12, 4), dtype=np.float32)
        for idx, b in enumerate(boxes_mm):
            norm[idx, 0] = float(np.clip(b.x_min / w_mm, 0.0, 1.0))
            norm[idx, 1] = float(np.clip(b.y_min / h_mm, 0.0, 1.0))
            norm[idx, 2] = float(np.clip(b.x_max / w_mm, 0.0, 1.0))
            norm[idx, 3] = float(np.clip(b.y_max / h_mm, 0.0, 1.0))
        return norm

    @staticmethod
    def extract_slot_occlusion_from_render(render_result: RenderResult) -> np.ndarray:
        """Measure persistent per-slot surface occlusion [0..1] from RenderResult.occlusion_mask."""
        occ = render_result.occlusion_mask
        if occ is None or float(np.max(occ)) <= 1e-4:
            return np.zeros(12, dtype=np.float32)

        h, w = occ.shape[:2]
        layout = PlateGeometry.calculate_layout(render_result.plate_text, render_result.config.plate_size)
        slot_boxes = PlateRenderer.get_canonical_slot_boxes(
            render_result.plate_text,
            layout,
            render_result.config.scale,
            render_result.character_boxes,
        )
        ratios = np.zeros(12, dtype=np.float32)
        for s_idx, b in enumerate(slot_boxes):
            x0 = max(0, min(w - 1, int(round(b.x_min))))
            x1 = max(x0 + 1, min(w, int(round(b.x_max))))
            y0 = max(0, min(h - 1, int(round(b.y_min))))
            y1 = max(y0 + 1, min(h, int(round(b.y_max))))
            ratios[s_idx] = float(np.mean(occ[y0:y1, x0:x1]))
        return ratios

    @staticmethod
    def compute_slot_crop_ioa(
        norm_slot_boxes: np.ndarray,
        plate_corners_2d: list[tuple[float, float]] | np.ndarray,
        crop_bbox_xyxy: tuple[float, float, float, float],
        canvas_wh: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Compute fraction [0..1] of each projected character slot inside the detector crop window."""
        import cv2

        pts_dst = np.asarray(plate_corners_2d, dtype=np.float32)
        if pts_dst.shape != (4, 2):
            return np.zeros(12, dtype=np.float32)

        src_unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_unit, pts_dst)

        cx0, cy0, cx1, cy1 = crop_bbox_xyxy
        if canvas_wh is not None:
            cw, ch = canvas_wh
            cx0 = max(0.0, cx0)
            cy0 = max(0.0, cy0)
            cx1 = min(float(cw), cx1)
            cy1 = min(float(ch), cy1)

        if cx1 <= cx0 + 1e-3 or cy1 <= cy0 + 1e-3:
            return np.zeros(12, dtype=np.float32)

        ioas = np.zeros(12, dtype=np.float32)
        for s in range(12):
            u0, v0, u1, v1 = norm_slot_boxes[s]
            q_unit = np.array([[[u0, v0]], [[u1, v0]], [[u1, v1]], [[u0, v1]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(q_unit, H).reshape(4, 2)
            sx0, sy0 = float(np.min(proj[:, 0])), float(np.min(proj[:, 1]))
            sx1, sy1 = float(np.max(proj[:, 0])), float(np.max(proj[:, 1]))
            s_area = max(1e-4, (sx1 - sx0) * (sy1 - sy0))

            ix0 = max(sx0, cx0)
            iy0 = max(sy0, cy0)
            ix1 = min(sx1, cx1)
            iy1 = min(sy1, cy1)
            if ix1 > ix0 and iy1 > iy0:
                ioas[s] = float(np.clip(((ix1 - ix0) * (iy1 - iy0)) / s_area, 0.0, 1.0))
        return ioas

    @classmethod
    def compute_frame_slot_readability(
        cls,
        plate_height_px: float,
        plate_width_px: float,
        incidence_deg: float,
        is_readable_view: bool = True,
        blur_px: float = 0.0,
        defocus_px: float = 0.0,
        noise_sigma: float = 0.0,
        jpeg_quality: int = 80,
        substream_scale: float = 1.0,
        slot_occlusion_12: np.ndarray | list[float] | None = None,
        slot_glare_12: np.ndarray | list[float] | None = None,
        slot_crop_ioa_12: np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        """Compute physical per-character readability `(12,)` float32 in `[0.0, 1.0]`."""
        if (not is_readable_view) or incidence_deg > 76.0 or plate_height_px < 6.0:
            return np.zeros(12, dtype=np.float32)

        eff_h = float(plate_height_px) * float(np.clip(substream_scale, 0.2, 1.0))
        eff_w = float(plate_width_px) * float(np.clip(substream_scale, 0.2, 1.0))

        # 1. Viewing foreshortening factor
        cos_inc = math.cos(math.radians(min(89.0, max(0.0, incidence_deg))))
        cos_76 = math.cos(math.radians(76.0))
        cos_22 = math.cos(math.radians(22.0))
        r_angle = float(np.clip((cos_inc - cos_76) / max(1e-4, cos_22 - cos_76), 0.0, 1.0)) ** 0.75

        # 2. Sensor SNR & codec factor
        r_noise = float(np.clip(1.0 - max(0.0, noise_sigma - 18.0) / 55.0, 0.25, 1.0))
        r_codec = float(np.clip((jpeg_quality - 14.0) / 36.0, 0.30, 1.0))
        r_isp = r_noise * r_codec

        occ = np.zeros(12, dtype=np.float32) if slot_occlusion_12 is None else np.asarray(slot_occlusion_12, dtype=np.float32)
        glare = np.zeros(12, dtype=np.float32) if slot_glare_12 is None else np.asarray(slot_glare_12, dtype=np.float32)
        ioa = np.ones(12, dtype=np.float32) if slot_crop_ioa_12 is None else np.asarray(slot_crop_ioa_12, dtype=np.float32)

        out = np.zeros(12, dtype=np.float32)
        for s in range(12):
            # Resolution factor by physical character row height
            if s < 4:
                # Region kanji (complex strokes, 24.2% of plate height)
                char_w = 0.095 * eff_w
                r_res = float(np.clip((eff_h - 7.0) / 16.0, 0.0, 1.0))
            elif s < 7:
                # Classification digits/letters (24.2% of plate height)
                char_w = 0.082 * eff_w
                r_res = float(np.clip((eff_h - 6.5) / 14.0, 0.0, 1.0))
            elif s == 7:
                # Hiragana (24.2% of plate height)
                char_w = 0.115 * eff_w
                r_res = float(np.clip((eff_h - 6.8) / 15.0, 0.0, 1.0))
            else:
                # Large 4-slot serial digits/dots (48.5% of plate height)
                char_w = 0.120 * eff_w
                r_res = float(np.clip((eff_h - 5.5) / 11.5, 0.0, 1.0))

            # Motion blur & optical defocus relative to character width
            eff_blur = (0.55 * blur_px + 0.38 * defocus_px) * float(np.clip(substream_scale, 0.4, 1.0))
            rel_blur = eff_blur / max(2.8, char_w)
            r_blur = float(math.exp(-0.95 * (rel_blur ** 1.35)))

            # Surface occlusion (mud / persistent occluder -> 0.0 when occ >= 0.85)
            occ_s = float(np.clip(occ[s], 0.0, 1.0))
            r_occ = 0.0 if occ_s >= 0.82 else float(np.clip(1.0 - (occ_s / 0.82) ** 1.2, 0.0, 1.0))

            # Specular glare / blowout washout
            glare_s = float(np.clip(glare[s], 0.0, 1.0))
            r_glare = 0.02 if glare_s >= 0.90 else float(np.clip(1.0 - (glare_s / 0.92) ** 1.6, 0.02, 1.0))

            # Detector crop edge truncation
            ioa_s = float(np.clip(ioa[s], 0.0, 1.0))
            if ioa_s < 0.35:
                r_crop = 0.0
            elif ioa_s >= 0.90:
                r_crop = 1.0
            else:
                r_crop = (ioa_s - 0.35) / 0.55

            out[s] = float(np.clip(r_res * r_angle * r_blur * r_isp * r_occ * r_glare * r_crop, 0.0, 1.0))

        return out

    @classmethod
    def compute_sequence_target_status(
        cls,
        plate_text: PlateText | None,
        slot_readability: np.ndarray,
        frame_valid_mask: np.ndarray,
        is_foreign_plate: np.ndarray,
        prefer_conflict_label: bool = False,
    ) -> str:
        """Compute canonical sequence target status in {'READABLE', 'UNREADABLE', 'CONFLICT', 'TRACK_MIXED'}."""
        if plate_text is None:
            return "UNREADABLE"

        foreign = np.asarray(is_foreign_plate, dtype=bool)
        if np.any(foreign):
            return "CONFLICT" if prefer_conflict_label else "TRACK_MIXED"

        valid = (np.asarray(frame_valid_mask, dtype=np.float32) > 0.5) & (~foreign)
        if not np.any(valid):
            return "UNREADABLE"

        active_mask = cls.get_active_slot_mask(plate_text)
        valid_scores = np.asarray(slot_readability, dtype=np.float32)[valid][:, active_mask]
        max_per_slot = np.max(valid_scores, axis=0)

        if np.any(max_per_slot < 0.25):
            return "UNREADABLE"
        if np.all(max_per_slot >= 0.35):
            return "READABLE"
        return "READABLE" if float(np.min(max_per_slot)) >= 0.30 else "UNREADABLE"
