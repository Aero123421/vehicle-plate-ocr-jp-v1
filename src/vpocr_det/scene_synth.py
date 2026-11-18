"""High-throughput full-frame scene & tight 4-corner ROI data engine for vehicle-plate-ocr-jp-v1.

Provides:
- `FullSceneDetSynthesizer`: < 4 ms/scene multi-scale full-frame camera scene generator with
  cached 1,000+ diverse Japanese plate bank, 6 vehicle/truck archetypes, >= 45% distant small
  plates (9-28 px), 15% pure hard-negative scenes, and exact 3D perspective 4-corner GT.
- `QuadCropSynthesizer`: < 0.8 ms/sample (96, 160, 3) loose/noisy ROI generator (75% positive
  plates with exact physical-plate-only [TL, TR, BR, BL] corners, 25% hard-negative traps).
- `SceneDetDataset` + `collate_det_scenes` and `QuadCropDataset` + `collate_quad_crops` with
  `cv2.setNumThreads(0)` worker initialization for linear multi-worker scaling.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import math
import os
import random
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from vpocr_det.types import (
    CLS_PLATE,
    CLS_VEHICLE,
    QuadCropTrainingSample,
     SIDE_FRONT,
    SIDE_PURE_SIDE,
    SIDE_REAR,
    SceneObjectGT,
    SynthSceneSample,
)
from vpocr_synth.export.fast_synthesizer import FastPlatePatchRenderer
from vpocr_synth.plate.models import PlateSize, PlateType
from vpocr_synth.plate.stratified_sampler import StratifiedPlateSampler

QUAD_ROI_H: int = 96
QUAD_ROI_W: int = 160
DUAL_CANVAS_H: int = 320
DUAL_CANVAS_W: int = 480


def _build_dct8_matrix_np() -> np.ndarray:
    mat = np.empty((8, 8), dtype=np.float32)
    for k in range(8):
        scale = math.sqrt(1.0 / 8.0) if k == 0 else math.sqrt(2.0 / 8.0)
        for n in range(8):
            mat[k, n] = scale * math.cos(math.pi * (2 * n + 1) * k / 16.0)
    return mat


_DCT8_NP: np.ndarray = _build_dct8_matrix_np()
_DCT8_NP_T: np.ndarray = np.ascontiguousarray(_DCT8_NP.T)
_DCT8_FREQ_WEIGHT_NP: np.ndarray = (
    1.0 + 0.22 * (np.arange(8, dtype=np.float32)[:, None] + np.arange(8, dtype=np.float32)[None, :])
)


def det_worker_init_fn(worker_id: int) -> None:
    """PyTorch DataLoader worker initializer disabling OpenCV thread contention."""
    cv2.setNumThreads(0)
    base = (torch.initial_seed() + worker_id * 104729) & 0x7FFFFFFF
    random.seed(base)
    np.random.seed(base % (2**31 - 1))


@dataclass(slots=True)
class _CachedPlateEntry:
    """Multi-scale mipmapped plate patch entry in the per-worker plate bank."""

    patch_full: np.ndarray     # (~102, ~205, 3) or (~136, ~273, 3) uint8
    patch_half: np.ndarray     # INTER_AREA 0.5x
    patch_quarter: np.ndarray  # INTER_AREA 0.25x
    patch_tiny: np.ndarray     # INTER_AREA ~14x28 for alias-free distant plates
    plate_text_str: str
    plate_type_str: str
    is_large: bool
    with_seal: bool
    is_temporary: bool
    has_frame: bool


class _SceneAssetBank:
    """Process-local cached bank of 1,000+ diverse plate patches, road markings, signs, and truck stickers."""

    _INSTANCE_BY_PID: dict[int, _SceneAssetBank] = {}

    ROAD_TEXTS: tuple[str, ...] = ("止まれ", "50", "40", "徐行", "専用")
    SHOP_SIGNS: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], ...] = (
        ("P 空車 200円", (244, 246, 248), (18, 62, 158), (204, 38, 38)),
        ("月極駐車場", (246, 246, 242), (24, 34, 92), (34, 52, 128)),
        ("No.1入口", (242, 246, 242), (20, 96, 52), (20, 96, 52)),
        ("制限 20km", (248, 218, 42), (24, 24, 26), (192, 32, 32)),
        ("P 満車", (245, 245, 245), (196, 30, 34), (32, 68, 152)),
        ("駐車禁止", (246, 244, 240), (188, 28, 30), (188, 28, 30)),
        ("24h 営業", (28, 56, 132), (248, 232, 68), (238, 242, 248)),
        ("出口専用", (244, 245, 240), (24, 82, 48), (24, 82, 48)),
    )
    TRUCK_STICKERS: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int]], ...] = (
        ("最大積載量 5000kg", (238, 196, 32), (22, 22, 24)),
        ("最大積載量 2000kg", (242, 244, 242), (22, 24, 26)),
        ("最大積載量 8500kg", (238, 196, 32), (22, 22, 24)),
        ("最大積載量 3500kg", (222, 228, 234), (20, 22, 24)),
        ("追突注意", (238, 64, 38), (248, 248, 246)),
        ("制限速度80km", (242, 206, 38), (20, 22, 24)),
    )
    TRUCK_BED_NUMBERS: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int]], ...] = (
        ("春日 2503", (228, 232, 236), (26, 32, 42)),
        ("関東 1616", (44, 88, 148), (242, 244, 246)),
        ("丸運 3802", (58, 108, 78), (244, 244, 240)),
        ("東都 616", (166, 172, 178), (22, 26, 32)),
        ("第一 5120", (234, 236, 238), (168, 32, 36)),
    )
    FENCE_BANNERS: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], ...] = (
        ("駐車場内での事故・盗難は責任を負いません", (246, 246, 244), (24, 32, 48), (192, 36, 36)),
        ("契約者以外無断駐車禁止 罰金3万円", (248, 245, 238), (188, 28, 32), (28, 42, 96)),
        ("SANTA MONICA Crepes OPEN", (64, 168, 142), (248, 244, 236), (228, 92, 118)),
        ("EXPRESS FOR AIRPORT / 空港リムジン", (240, 244, 248), (24, 68, 148), (38, 92, 168)),
    )

    def __init__(self, bank_size: int = 1080, seed: int = 20261111) -> None:
        cv2.setNumThreads(0)
        self.bank_size = max(1000, int(bank_size))
        self.entries: list[_CachedPlateEntry] = []
        self.buckets: dict[tuple[str, int], list[int]] = {
            ("standard", SIDE_FRONT): [],
            ("standard", SIDE_REAR): [],
            ("standard_white", SIDE_FRONT): [],
            ("standard_white", SIDE_REAR): [],
            ("kei", SIDE_FRONT): [],
            ("kei", SIDE_REAR): [],
            ("kei_black", SIDE_FRONT): [],
            ("kei_black", SIDE_REAR): [],
            ("commercial_med", SIDE_FRONT): [],
            ("commercial_med", SIDE_REAR): [],
            ("commercial_large", SIDE_FRONT): [],
            ("commercial_large", SIDE_REAR): [],
            ("commercial_green_large", SIDE_FRONT): [],
            ("commercial_green_large", SIDE_REAR): [],
            ("temporary", SIDE_FRONT): [],
            ("temporary", SIDE_REAR): [],
        }
        self.road_masks: list[np.ndarray] = []
        self.sign_patches: list[np.ndarray] = []
        self.sticker_patches: list[np.ndarray] = []
        self.bed_num_patches: list[np.ndarray] = []
        self.banner_patches: list[np.ndarray] = []
        self.asphalt_deltas: list[np.ndarray] = []
        self.facade_deltas: list[np.ndarray] = []
        self.blue_route_patches: list[np.ndarray] = []
        self.frameless_decal_rgba: list[np.ndarray] = []
        self.bordered_decal_patches: list[np.ndarray] = []
        self.multiline_parking_patches: list[np.ndarray] = []
        self.stratified_plate_buckets: dict[tuple[str, int, bool], list[tuple[_CachedPlateEntry, int]]] = defaultdict(list)
        self.temporary_plate_pairs: dict[int, list[tuple[_CachedPlateEntry, int]]] = {SIDE_FRONT: [], SIDE_REAR: []}

        self._build_text_assets()
        self._build_surface_textures(seed)
        self._build_plate_bank(seed)
        self._build_stage2_extended_assets(seed)

    @classmethod
    def get(cls, bank_size: int = 1080) -> _SceneAssetBank:
        pid = os.getpid()
        inst = cls._INSTANCE_BY_PID.get(pid)
        if inst is None or inst.bank_size < bank_size:
            inst = cls(bank_size=bank_size)
            cls._INSTANCE_BY_PID[pid] = inst
        return inst

    @staticmethod
    def _render_label_patch(
        text: str,
        width: int,
        height: int,
        bg_rgb: tuple[int, int, int],
        fg_rgb: tuple[int, int, int],
        border_rgb: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        patch = np.empty((height, width, 3), dtype=np.uint8)
        patch[:] = bg_rgb
        if border_rgb is not None:
            cv2.rectangle(patch, (1, 1), (width - 2, height - 2), border_rgb, 2)
        chars = [c for c in text if c != " "]
        if not chars:
            return patch
        font_path = FastPlatePatchRenderer._DEFAULT_BOLD_PATH
        margin_x = max(4, int(width * 0.06))
        slot_w = (width - 2 * margin_x) / float(len(chars))
        th = max(8, int(height * 0.62))
        oy = (height - th) // 2
        fg_f = np.array(fg_rgb, dtype=np.float32)
        for i, ch in enumerate(chars):
            gm = FastPlatePatchRenderer._get_glyph_mask(font_path, ch, "neutral")
            tw = max(4, int(round(slot_w * 0.86)))
            rs = cv2.resize(gm, (tw, th), interpolation=cv2.INTER_AREA)
            ox = int(round(margin_x + i * slot_w + (slot_w - tw) * 0.5))
            x0, x1 = max(0, ox), min(width, ox + tw)
            if x1 > x0:
                alpha = (rs[:, : x1 - x0].astype(np.float32) / 255.0)[:, :, None]
                roi = patch[oy : oy + th, x0:x1].astype(np.float32)
                patch[oy : oy + th, x0:x1] = np.clip(roi * (1.0 - alpha) + fg_f * alpha, 0.0, 255.0).astype(np.uint8)
        return patch

    def _build_text_assets(self) -> None:
        font_path = FastPlatePatchRenderer._DEFAULT_BOLD_PATH
        for txt in self.ROAD_TEXTS:
            w, h = 140, 56
            mask = np.zeros((h, w), dtype=np.uint8)
            slot_w = (w - 12) / float(len(txt))
            for i, ch in enumerate(txt):
                gm = FastPlatePatchRenderer._get_glyph_mask(font_path, ch, "neutral")
                tw = max(8, int(round(slot_w * 0.84)))
                th = 42
                rs = cv2.resize(gm, (tw, th), interpolation=cv2.INTER_AREA)
                ox = int(round(6 + i * slot_w + (slot_w - tw) * 0.5))
                mask[7 : 7 + th, ox : ox + tw] = np.maximum(mask[7 : 7 + th, ox : ox + tw], rs)
            self.road_masks.append(mask)

        for label, bg_c, fg_c, bd_c in self.SHOP_SIGNS:
            self.sign_patches.append(self._render_label_patch(label, 148, 52, bg_c, fg_c, bd_c))

        for label, bg_c, fg_c in self.TRUCK_STICKERS:
            self.sticker_patches.append(self._render_label_patch(label, 132, 30, bg_c, fg_c, fg_c))

        for label, bg_c, fg_c in self.TRUCK_BED_NUMBERS:
            self.bed_num_patches.append(self._render_label_patch(label, 168, 42, bg_c, fg_c, None))

        for label, bg_c, fg_c, bd_c in self.FENCE_BANNERS:
            self.banner_patches.append(self._render_label_patch(label, 220, 44, bg_c, fg_c, bd_c))

    def _build_surface_textures(self, seed: int) -> None:
        """Precompute fast tileable int16 high-frequency road asphalt & building facade textures."""
        np_rng = np.random.default_rng(seed + 4242)
        for i in range(4):
            # Granular gravel + tar cracks + oil patches road delta in [-28, +28]
            fine = np_rng.normal(0.0, 9.5 + 2.5 * i, size=(256, 256)).astype(np.float32)
            med = cv2.resize(np_rng.normal(0.0, 11.0, size=(64, 64)).astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
            tex = np.clip(fine + med, -32.0, 32.0)
            crack_canvas = np.zeros((256, 256), dtype=np.float32)
            for _ in range(10):
                pts = np.cumsum(np_rng.integers(-18, 19, size=(8, 2)), axis=0) + np_rng.integers(20, 236, size=(1, 2))
                pts = np.clip(pts, 0, 255).astype(np.int32)
                cv2.polylines(crack_canvas, [pts], isClosed=False, color=-22.0, thickness=int(np_rng.integers(1, 3)))
            self.asphalt_deltas.append(np.round(tex + crack_canvas).astype(np.int16))

            # Brick / tile / corrugated shutter facade delta in [-24, +24]
            fac = np_rng.normal(0.0, 5.5, size=(256, 256)).astype(np.float32)
            step_y = int([8, 12, 6, 16][i])
            step_x = int([18, 24, 32, 16][i])
            fac[::step_y, :] -= 14.0
            fac[:, ::step_x] -= 10.0
            self.facade_deltas.append(np.round(np.clip(fac, -26.0, 26.0)).astype(np.int16))

    def _build_plate_bank(self, seed: int) -> None:
        sampler = StratifiedPlateSampler(seed=seed)
        rng = random.Random(seed + 7717)
        type_to_rem = {pt_enum: i for i, pt_enum in enumerate(sampler._plate_types)}
        n_pt = len(sampler._plate_types)
        occ_by_type = {pt_enum: 0 for pt_enum in sampler._plate_types}
        occ_by_stratum: dict[tuple[PlateType, bool, bool], int] = defaultdict(int)

        # Explicit decoupled strata: (target_type, force_large, is_temp) across 25-slot cycle (zero modulo collision!)
        strata_cycle: list[tuple[PlateType, bool, bool]] = (
            [(PlateType.STANDARD_PRIVATE, False, False)] * 7
            + [(PlateType.STANDARD_PRIVATE, True, False)] * 2
            + [(PlateType.STANDARD_COMMERCIAL, True, False)] * 5
            + [(PlateType.STANDARD_COMMERCIAL, False, False)] * 3
            + [(PlateType.KEI_PRIVATE, False, False)] * 3
            + [(PlateType.KEI_COMMERCIAL, False, False)] * 2
            + [(PlateType.GRAPHIC, False, False)] * 2
            + [(PlateType.STANDARD_PRIVATE, False, True)] * 1
        )

        for idx in range(self.bank_size):
            tgt_type, want_large, is_temp = strata_cycle[idx % len(strata_cycle)]
            if idx % 50 == 49:
                tgt_type, want_large, is_temp = (PlateType.STANDARD_COMMERCIAL, False, True)
            # Alternate front/rear independently within each (type, size, temp) stratum
            occ_k = occ_by_type[tgt_type]
            occ_by_type[tgt_type] = occ_k + 1
            stratum_key = (tgt_type, want_large, is_temp)
            occ_s = occ_by_stratum[stratum_key]
            occ_by_stratum[stratum_key] = occ_s + 1
            is_rear = (occ_s % 2) == 1

            s_seed = (occ_k * 17 + idx * 3) * n_pt + type_to_rem[tgt_type]
            pt, cfg = sampler.sample(seed=s_seed)
            p_size = PlateSize.LARGE if want_large else PlateSize.MEDIUM

            # Ensure legal large-vehicle classification prefix (1xx/2xx/8xx/9xx) on forced LARGE plates
            if p_size == PlateSize.LARGE and pt.classification and pt.classification[0] not in ("1", "2", "8", "9", "0"):
                new_first = rng.choices(["1", "2", "8", "9"], weights=[0.50, 0.24, 0.20, 0.06], k=1)[0]
                pt = replace(pt, classification=new_first + pt.classification[1:])

            can_seal = p_size in (PlateSize.MEDIUM, PlateSize.LARGE) and cfg.plate_type in (
                PlateType.STANDARD_PRIVATE,
                PlateType.STANDARD_COMMERCIAL,
                PlateType.GRAPHIC,
            )
            has_frame = (not is_temp) and (rng.random() < 0.28)
            frame_style = ("chrome" if rng.random() < 0.55 else "black") if has_frame else None

            rcfg = replace(
                cfg,
                plate_size=p_size,
                with_seal=(is_rear and can_seal and not is_temp),
                plate_frame=has_frame,
                weathering=rng.uniform(0.08, 0.58) if rng.random() < 0.42 else 0.0,
                mud_level=rng.uniform(0.06, 0.38) if rng.random() < 0.20 else 0.0,
                is_illuminated=(not is_temp and rng.random() < 0.05),
                seed=idx + 1000,
            )

            patch_full, _ = FastPlatePatchRenderer.render_patch(
                pt,
                rcfg,
                scale=0.60,
                temporary_plate=is_temp,
                frame_style=frame_style,
            )

            # Photometric chalk-dust / faded-paint / shadowed dark-olive diversity on green & black work plates
            if (not is_temp) and rcfg.plate_type in (PlateType.STANDARD_COMMERCIAL, PlateType.KEI_COMMERCIAL) and rng.random() < 0.34:
                pf = patch_full.astype(np.float32)
                mode_ph = rng.random()
                if mode_ph < 0.42:
                    # Construction cement/quarry dust veil lifting dark green/black levels & softening contrast
                    veil = rng.uniform(0.10, 0.26)
                    dust_rgb = np.array([rng.uniform(148, 182), rng.uniform(146, 178), rng.uniform(138, 168)], dtype=np.float32)
                    pf = pf * (1.0 - veil) + dust_rgb * veil
                elif mode_ph < 0.76:
                    # Deep under-bed / chassis shadow making green plate appear dark olive-charcoal
                    gain_g = rng.uniform(0.46, 0.72)
                    pf = pf * np.array([gain_g * 0.94, gain_g * 1.02, gain_g * 0.92], dtype=np.float32)
                else:
                    # Sun-faded desaturated commercial paint + lifted black floor
                    gray_p = pf.mean(axis=-1, keepdims=True)
                    desat = rng.uniform(0.20, 0.42)
                    pf = (pf * (1.0 - desat) + gray_p * desat) * rng.uniform(0.86, 1.02) + rng.uniform(6.0, 18.0)
                patch_full = np.clip(pf, 0.0, 255.0).astype(np.uint8)

            fh, fw = patch_full.shape[:2]
            patch_half = cv2.resize(patch_full, (max(32, fw // 2), max(16, fh // 2)), interpolation=cv2.INTER_AREA)
            patch_quarter = cv2.resize(patch_full, (max(18, fw // 4), max(9, fh // 4)), interpolation=cv2.INTER_AREA)
            patch_tiny = cv2.resize(patch_full, (28, 14), interpolation=cv2.INTER_AREA)

            entry = _CachedPlateEntry(
                patch_full=patch_full,
                patch_half=patch_half,
                patch_quarter=patch_quarter,
                patch_tiny=patch_tiny,
                plate_text_str=pt.full_text,
                plate_type_str="temporary_kari" if is_temp else rcfg.plate_type.value,
                is_large=(p_size == PlateSize.LARGE),
                with_seal=rcfg.with_seal,
                is_temporary=is_temp,
                has_frame=has_frame,
            )
            e_idx = len(self.entries)
            self.entries.append(entry)

            side_key = SIDE_REAR if is_rear else SIDE_FRONT
            if is_temp:
                self.buckets[("temporary", side_key)].append(e_idx)
            elif rcfg.plate_type in (PlateType.KEI_PRIVATE, PlateType.KEI_COMMERCIAL):
                self.buckets[("kei", side_key)].append(e_idx)
                if rcfg.plate_type == PlateType.KEI_COMMERCIAL:
                    self.buckets[("kei_black", side_key)].append(e_idx)
            elif p_size == PlateSize.LARGE:
                self.buckets[("commercial_large", side_key)].append(e_idx)
                if rcfg.plate_type == PlateType.STANDARD_COMMERCIAL:
                    self.buckets[("commercial_green_large", side_key)].append(e_idx)
            elif rcfg.plate_type == PlateType.STANDARD_COMMERCIAL:
                self.buckets[("commercial_med", side_key)].append(e_idx)
            else:
                self.buckets[("standard", side_key)].append(e_idx)
                if rcfg.plate_type == PlateType.STANDARD_PRIVATE:
                    self.buckets[("standard_white", side_key)].append(e_idx)
                elif rcfg.plate_type == PlateType.GRAPHIC:
                    # Keep graphic share modest in kei bucket (~12%) without crowding out kei_commercial
                    if (occ_k % 3) == 0:
                        self.buckets[("kei", side_key)].append(e_idx)

    def pick_plate(self, category: str, plate_side: int, rng: random.Random, p_temporary: float = 0.04) -> _CachedPlateEntry:
        side = SIDE_REAR if plate_side == SIDE_REAR else SIDE_FRONT
        if p_temporary > 0.0 and category != "kei" and rng.random() < p_temporary:
            temp_list = self.buckets[("temporary", side)]
            if temp_list:
                return self.entries[rng.choice(temp_list)]
        lst = self.buckets.get((category, side))
        if not lst:
            lst = self.buckets[("standard", side)]
        return self.entries[rng.choice(lst)]

    @staticmethod
    def _render_frameless_rgba_stencil(
        text: str,
        width: int,
        height: int,
        fg_rgb: tuple[int, int, int],
    ) -> np.ndarray:
        """Render frameless painted body text as RGBA `(H, W, 4)` with zero backing substrate (`alpha == 0` outside strokes)."""
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        chars = [c for c in text if c != " "]
        if not chars:
            return rgba
        font_path = FastPlatePatchRenderer._DEFAULT_BOLD_PATH
        margin_x = max(3, int(width * 0.04))
        slot_w = (width - 2 * margin_x) / float(len(chars))
        th = max(10, int(height * 0.72))
        oy = (height - th) // 2
        alpha_acc = np.zeros((height, width), dtype=np.uint8)
        for i, ch in enumerate(chars):
            gm = FastPlatePatchRenderer._get_glyph_mask(font_path, ch, "neutral")
            tw = max(4, int(round(slot_w * 0.88)))
            rs = cv2.resize(gm, (tw, th), interpolation=cv2.INTER_AREA)
            ox = int(round(margin_x + i * slot_w + (slot_w - tw) * 0.5))
            x0, x1 = max(0, ox), min(width, ox + tw)
            if x1 > x0:
                alpha_acc[oy : oy + th, x0:x1] = np.maximum(alpha_acc[oy : oy + th, x0:x1], rs[:, : x1 - x0])
        rgba[:, :, 0] = fg_rgb[0]
        rgba[:, :, 1] = fg_rgb[1]
        rgba[:, :, 2] = fg_rgb[2]
        rgba[:, :, 3] = alpha_acc
        return rgba

    @staticmethod
    def _render_blue_route_sign(
        route_str: str,
        width: int,
        height: int,
        shield_style: bool,
    ) -> np.ndarray:
        """Render MLIT national/prefectural blue highway route sign `RGB=(14, 68, 158)` with white border & bold digits."""
        blue_c = (14, 68, 158)
        white_c = (246, 248, 250)
        if shield_style:
            patch = np.full((height, width, 3), 132, dtype=np.uint8)
            pts_outer = np.array(
                [
                    [int(width * 0.08), int(height * 0.10)],
                    [int(width * 0.92), int(height * 0.10)],
                    [int(width * 0.86), int(height * 0.62)],
                    [width // 2, int(height * 0.94)],
                    [int(width * 0.14), int(height * 0.62)],
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(patch, pts_outer, white_c, lineType=cv2.LINE_AA)
            pts_inner = np.array(
                [
                    [int(width * 0.13), int(height * 0.16)],
                    [int(width * 0.87), int(height * 0.16)],
                    [int(width * 0.81), int(height * 0.60)],
                    [width // 2, int(height * 0.87)],
                    [int(width * 0.19), int(height * 0.60)],
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(patch, pts_inner, blue_c, lineType=cv2.LINE_AA)
        else:
            patch = np.empty((height, width, 3), dtype=np.uint8)
            patch[:] = blue_c
            cv2.rectangle(patch, (2, 2), (width - 3, height - 3), white_c, 3)

        font_path = FastPlatePatchRenderer._DEFAULT_BOLD_PATH
        chars = list(route_str)
        span_w = int(width * (0.60 if shield_style else 0.74))
        x_start = (width - span_w) // 2
        slot_w = span_w / float(max(1, len(chars)))
        th = int(height * (0.46 if shield_style else 0.62))
        oy = int(height * (0.24 if shield_style else 0.19))
        fg_f = np.array(white_c, dtype=np.float32)
        for i, ch in enumerate(chars):
            gm = FastPlatePatchRenderer._get_glyph_mask(font_path, ch, "neutral")
            tw = max(6, int(round(slot_w * 0.88)))
            rs = cv2.resize(gm, (tw, th), interpolation=cv2.INTER_AREA)
            ox = int(round(x_start + i * slot_w + (slot_w - tw) * 0.5))
            x0, x1 = max(0, ox), min(width, ox + tw)
            if x1 > x0 and oy + th <= height:
                a = (rs[:, : x1 - x0].astype(np.float32) / 255.0)[:, :, None]
                roi = patch[oy : oy + th, x0:x1].astype(np.float32)
                patch[oy : oy + th, x0:x1] = np.clip(roi * (1.0 - a) + fg_f * a, 0.0, 255.0).astype(np.uint8)
        return patch

    @classmethod
    def _render_multiline_sign_patch(
        cls,
        top_text: str,
        bot_text: str,
        top_bg: tuple[int, int, int],
        top_fg: tuple[int, int, int],
        bot_bg: tuple[int, int, int],
        bot_fg: tuple[int, int, int],
        border_rgb: tuple[int, int, int],
    ) -> np.ndarray:
        w, h = 168, 76
        top_h = 36
        p_top = cls._render_label_patch(top_text, w, top_h, top_bg, top_fg, None)
        p_bot = cls._render_label_patch(bot_text, w, h - top_h, bot_bg, bot_fg, None)
        out = np.vstack([p_top, p_bot])
        cv2.rectangle(out, (1, 1), (w - 2, h - 2), border_rgb, 2)
        cv2.line(out, (2, top_h), (w - 3, top_h), border_rgb, 1)
        return out

    def _build_stage2_extended_assets(self, seed: int) -> None:
        """Build Stage-2 structured hard-negative assets & stratified 4-color x class-length plate buckets."""
        ext_rng = random.Random(int(seed) + 99199)

        # 1. Blue highway / national-route signs (`"319"`, `"246"`, `"16"`, `"R17"`, `"1"`, `"58"`, `"20"`, `"4"`)
        for r_txt in ("319", "246", "16", "R17", "1", "58", "20", "4"):
            self.blue_route_patches.append(self._render_blue_route_sign(r_txt, 144, 76, shield_style=False))
            if not r_txt.startswith("R"):
                self.blue_route_patches.append(self._render_blue_route_sign(r_txt, 112, 92, shield_style=True))

        # 2. Frameless painted truck-body text stencils (RGBA with zero rectangular backing or border!)
        frameless_texts = (
            "スケルトン車",
            "最大積載量 2000kg",
            "産廃収集運搬車",
            "自家用",
            "最大積載量 5000kg",
            "一般廃棄物",
            "限定積載 3000kg",
        )
        ink_colors = (
            (244, 246, 248),
            (22, 24, 28),
            (242, 204, 38),
            (24, 58, 132),
            (26, 92, 54),
        )
        for txt in frameless_texts:
            for ink in ink_colors:
                self.frameless_decal_rgba.append(self._render_frameless_rgba_stencil(txt, 184, 42, ink))

        # 3. Bordered payload, speed-limit, hazard, and bolted-blank decals
        self.bordered_decal_patches.extend(self.sticker_patches)
        for label, bg_c, fg_c, bd_c in (
            ("最大積載量 2000kg", (242, 204, 36), (20, 22, 24), (20, 22, 24)),
            ("最大積載量 4000kg", (244, 246, 244), (20, 22, 26), (20, 22, 26)),
            ("制限速度 80km/h", (244, 210, 40), (22, 22, 24), (192, 32, 32)),
            ("危", (22, 24, 26), (242, 198, 32), (242, 198, 32)),
            ("産廃収集運搬車", (242, 244, 246), (22, 56, 128), (22, 56, 128)),
        ):
            w_d = 92 if label == "危" else 152
            h_d = 54 if label == "危" else 36
            self.bordered_decal_patches.append(self._render_label_patch(label, w_d, h_d, bg_c, fg_c, bd_c))
        # Bolted blank chassis / dealer metal plate (zero glyphs)
        for blank_c in ((214, 218, 224), (36, 40, 46), (28, 72, 48)):
            bp = np.empty((44, 92, 3), dtype=np.uint8)
            bp[:] = blank_c
            cv2.rectangle(bp, (1, 1), (90, 42), (148, 154, 162), 2)
            cv2.circle(bp, (16, 10), 3, (176, 180, 186), -1)
            cv2.circle(bp, (76, 10), 3, (176, 180, 186), -1)
            self.bordered_decal_patches.append(bp)

        # 4. Multiline shop & coin-parking tariff signs (`"24h 最大800円"`, `"P 空車"`)
        for t1, t2, bg1, fg1, bg2, fg2, bd in (
            ("P 空車", "24h 最大800円", (18, 62, 158), (246, 248, 250), (246, 246, 242), (204, 34, 36), (18, 62, 158)),
            ("24h 最大800円", "60分 200円", (204, 34, 36), (248, 248, 246), (246, 246, 244), (22, 28, 42), (204, 34, 36)),
            ("P 空車", "月極 契約者専用", (22, 98, 56), (246, 248, 246), (244, 246, 242), (24, 34, 88), (22, 98, 56)),
            ("COIN PARK 24h", "最大 1200円", (246, 208, 38), (20, 22, 26), (24, 48, 122), (248, 246, 240), (22, 24, 28)),
            ("P 満車", "24h 最大800円", (196, 30, 34), (248, 248, 248), (244, 244, 240), (26, 36, 96), (32, 52, 124)),
        ):
            self.multiline_parking_patches.append(self._render_multiline_sign_patch(t1, t2, bg1, fg1, bg2, fg2, bd))

        # 5. Index existing entries + build paired front/rear supplemental registrations for exact Stage-2 strata
        rear_idx_set: set[int] = set()
        for (_, s_k), idx_list in self.buckets.items():
            if s_k == SIDE_REAR:
                rear_idx_set.update(idx_list)

        for idx, ent in enumerate(self.entries):
            side_k = SIDE_REAR if idx in rear_idx_set else SIDE_FRONT
            pid = idx + 1
            if ent.is_temporary:
                self.temporary_plate_pairs[side_k].append((ent, pid))
                continue
            parts = ent.plate_text_str.split()
            cls_str = parts[1] if len(parts) >= 2 else "500"
            is_short = len(cls_str) in (1, 2)
            ptype = ent.plate_type_str
            if ptype == PlateType.STANDARD_COMMERCIAL.value:
                ckey = "green_comm"
            elif ptype == PlateType.KEI_COMMERCIAL.value:
                ckey = "black_kei"
            elif ptype == PlateType.KEI_PRIVATE.value:
                ckey = "yellow_kei"
            else:
                ckey = "white_priv"
            self.stratified_plate_buckets[(ckey, side_k, is_short)].append((ent, pid))

        sampler = StratifiedPlateSampler(seed=int(seed) + 55111)
        type_to_rem = {pt_enum: i for i, pt_enum in enumerate(sampler._plate_types)}
        n_pt = len(sampler._plate_types)
        next_pid = len(self.entries) + 1
        color_specs = (
            ("green_comm", PlateType.STANDARD_COMMERCIAL, ("1", "2", "8", "11", "22", "88", "14", "83")),
            ("black_kei", PlateType.KEI_COMMERCIAL, ("40", "41", "48", "4", "6", "47")),
            ("yellow_kei", PlateType.KEI_PRIVATE, ("50", "58", "40", "5", "7", "55")),
            ("white_priv", PlateType.STANDARD_PRIVATE, ("3", "5", "33", "55", "77", "88", "11", "34")),
        )
        sup_counter = 0
        for ckey, pt_enum, short_pool in color_specs:
            for is_short in (True, False):
                need = max(
                    18,
                    36 - min(
                        len(self.stratified_plate_buckets[(ckey, SIDE_FRONT, is_short)]),
                        len(self.stratified_plate_buckets[(ckey, SIDE_REAR, is_short)]),
                    ),
                )
                for _ in range(need):
                    sup_counter += 1
                    s_seed = (sup_counter * 29 + 7) * n_pt + type_to_rem[pt_enum]
                    pt, cfg = sampler.sample(seed=s_seed)
                    want_large = (pt_enum == PlateType.STANDARD_COMMERCIAL and ext_rng.random() < 0.55) or (
                        pt_enum == PlateType.STANDARD_PRIVATE and ext_rng.random() < 0.16
                    )
                    p_size = PlateSize.LARGE if want_large else PlateSize.MEDIUM
                    if is_short:
                        cls_tok = ext_rng.choice(short_pool)
                        if p_size == PlateSize.LARGE and cls_tok[0] not in ("1", "2", "8", "9"):
                            cls_tok = ext_rng.choice(("1", "2", "8", "11", "22", "88"))
                        pt = replace(pt, classification=cls_tok)
                    elif len(pt.classification) != 3:
                        prefix = ext_rng.choice(("1", "2", "8")) if p_size == PlateSize.LARGE else ext_rng.choice(("3", "5", "4", "8"))
                        pt = replace(pt, classification=prefix + f"{ext_rng.randint(0, 99):02d}")
                    elif p_size == PlateSize.LARGE and pt.classification[0] not in ("1", "2", "8", "9", "0"):
                        pt = replace(pt, classification=ext_rng.choice(("1", "2", "8")) + pt.classification[1:])

                    has_frame = ext_rng.random() < 0.26
                    frame_style = ("chrome" if ext_rng.random() < 0.55 else "black") if has_frame else None
                    weath = ext_rng.uniform(0.08, 0.55) if ext_rng.random() < 0.40 else 0.0
                    mud = ext_rng.uniform(0.06, 0.36) if ext_rng.random() < 0.24 else 0.0
                    can_seal = pt_enum in (PlateType.STANDARD_PRIVATE, PlateType.STANDARD_COMMERCIAL, PlateType.GRAPHIC)
                    reg_pid = next_pid
                    next_pid += 1

                    for is_rear_side, side_k in ((False, SIDE_FRONT), (True, SIDE_REAR)):
                        rcfg = replace(
                            cfg,
                            plate_type=pt_enum,
                            plate_size=p_size,
                            with_seal=(is_rear_side and can_seal),
                            plate_frame=has_frame,
                            weathering=weath,
                            mud_level=mud,
                            is_illuminated=False,
                            seed=90000 + sup_counter * 2 + int(is_rear_side),
                        )
                        p_full, _ = FastPlatePatchRenderer.render_patch(
                            pt,
                            rcfg,
                            scale=0.60,
                            temporary_plate=False,
                            frame_style=frame_style,
                        )
                        fh, fw = p_full.shape[:2]
                        ent_s2 = _CachedPlateEntry(
                            patch_full=p_full,
                            patch_half=cv2.resize(p_full, (max(32, fw // 2), max(16, fh // 2)), interpolation=cv2.INTER_AREA),
                            patch_quarter=cv2.resize(p_full, (max(18, fw // 4), max(9, fh // 4)), interpolation=cv2.INTER_AREA),
                            patch_tiny=cv2.resize(p_full, (28, 14), interpolation=cv2.INTER_AREA),
                            plate_text_str=pt.full_text,
                            plate_type_str=pt_enum.value,
                            is_large=(p_size == PlateSize.LARGE),
                            with_seal=rcfg.with_seal,
                            is_temporary=False,
                            has_frame=has_frame,
                        )
                        self.stratified_plate_buckets[(ckey, side_k, is_short)].append((ent_s2, reg_pid))

    def pick_stratified_plate(
        self,
        plate_side: int,
        rng: random.Random,
        p_temporary: float = 0.03,
    ) -> tuple[_CachedPlateEntry, int]:
        """Sample positive plate with exact 28% green / 18% black kei / 18% yellow kei / 36% white & 30% 1/2-digit class."""
        side = SIDE_REAR if plate_side == SIDE_REAR else SIDE_FRONT
        if p_temporary > 0.0 and rng.random() < p_temporary:
            t_lst = self.temporary_plate_pairs.get(side)
            if t_lst:
                return rng.choice(t_lst)
        u_c = rng.random()
        if u_c < 0.28:
            ckey = "green_comm"
        elif u_c < 0.46:
            ckey = "black_kei"
        elif u_c < 0.64:
            ckey = "yellow_kei"
        else:
            ckey = "white_priv"
        is_short = rng.random() < 0.30
        lst = self.stratified_plate_buckets.get((ckey, side, is_short))
        if not lst:
            lst = self.stratified_plate_buckets.get((ckey, side, not is_short))
        if not lst:
            ent_fb = self.pick_plate("standard", side, rng, p_temporary=0.0)
            return ent_fb, 1
        return rng.choice(lst)


@dataclass
class SceneSynthConfig:
    """Configuration for `FullSceneDetSynthesizer`."""

    image_size: tuple[int, int] = (384, 640)  # (H, W)
    plate_bank_size: int = 1080
    p_hard_negative_scene: float = 0.15       # 15% 0-vehicle scenes with crosswalks/windows/gates/signs
    p_small_plate: float = 0.61               # >= 45% distant small plates (9-28 px in 640w)
    p_mid_plate: float = 0.26                 # ~35% mid-range plates (28-75 px)
    p_large_plate: float = 0.13               # ~20% close-up plates (75-210 px)
    p_temporary_plate: float = 0.048          # ~4% 仮ナンバー red-diagonal temporary plates
    p_pure_side_vehicle: float = 0.11         # pure side-view vehicles (CLS_VEHICLE only, no CLS_PLATE)
    min_plate_visibility: float = 0.58
    min_plate_side_px: float = 3.5


# Precomputed UV grid (32 points including near-corner 4%/6% perimeter) for pixel-exact occlusion visibility checking
_UU_8 = np.linspace(0.04, 0.96, 8, dtype=np.float32)
_VV_4 = np.linspace(0.06, 0.94, 4, dtype=np.float32)
_PLATE_UV_GRID_32 = np.stack(np.meshgrid(_UU_8, _VV_4), axis=-1).reshape(32, 1, 2)
_UNIT_QUAD_4 = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]], [[0.0, 1.0]]], dtype=np.float32)

_VEH_U_6 = np.linspace(0.10, 0.90, 6, dtype=np.float32)
_VEH_V_6 = np.linspace(0.10, 0.90, 6, dtype=np.float32)
_VEH_GRID_U, _VEH_GRID_V = np.meshgrid(_VEH_U_6, _VEH_V_6)
_VEH_GRID_U = _VEH_GRID_U.ravel()
_VEH_GRID_V = _VEH_GRID_V.ravel()


def _warp_patch_into_canvas(
    canvas: np.ndarray,
    patch: np.ndarray,
    dst_corners_xy: np.ndarray,
) -> None:
    """Fast ROI-local perspective warp of `patch` onto `canvas` at exact `dst_corners_xy` (4, 2) [TL,TR,BR,BL]."""
    h_img, w_img = canvas.shape[:2]
    x0 = max(0, int(math.floor(float(np.min(dst_corners_xy[:, 0])))) - 1)
    y0 = max(0, int(math.floor(float(np.min(dst_corners_xy[:, 1])))) - 1)
    x1 = min(w_img, int(math.ceil(float(np.max(dst_corners_xy[:, 0])))) + 2)
    y1 = min(h_img, int(math.ceil(float(np.max(dst_corners_xy[:, 1])))) + 2)
    rw = x1 - x0
    rh = y1 - y0
    if rw < 2 or rh < 2:
        return

    src_img = patch
    ph, pw = src_img.shape[:2]
    # Pre-minify with true INTER_AREA before warpPerspective (OpenCV warpPerspective silently falls back to INTER_LINEAR!)
    if rw < pw * 0.72 or rh < ph * 0.72:
        pre_w = max(8, min(pw, int(round(rw * 1.35))))
        pre_h = max(4, min(ph, int(round(rh * 1.35))))
        src_img = cv2.resize(src_img, (pre_w, pre_h), interpolation=cv2.INTER_AREA)
        ph, pw = src_img.shape[:2]

    src_pts = np.array(
        [[0.0, 0.0], [float(pw), 0.0], [float(pw), float(ph)], [0.0, float(ph)]],
        dtype=np.float32,
    )
    local_dst = dst_corners_xy.astype(np.float32).copy()
    local_dst[:, 0] -= float(x0)
    local_dst[:, 1] -= float(y0)

    H = cv2.getPerspectiveTransform(src_pts, local_dst)
    warped = cv2.warpPerspective(src_img, H, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(local_dst).astype(np.int32), 255, lineType=cv2.LINE_AA)

    roi = canvas[y0:y1, x0:x1]
    a = (mask.astype(np.uint16))[:, :, None]
    canvas[y0:y1, x0:x1] = ((warped.astype(np.uint16) * a + roi.astype(np.uint16) * (255 - a)) >> 8).astype(np.uint8)


def _warp_rgba_patch_into_canvas(
    canvas: np.ndarray,
    rgba_patch: np.ndarray,
    dst_corners_xy: np.ndarray,
    opacity: float = 0.94,
) -> None:
    """Perspective-warp frameless RGBA stencil `(H, W, 4)` onto `canvas` with zero rectangular backing or border."""
    h_img, w_img = canvas.shape[:2]
    x0 = max(0, int(math.floor(float(np.min(dst_corners_xy[:, 0])))) - 1)
    y0 = max(0, int(math.floor(float(np.min(dst_corners_xy[:, 1])))) - 1)
    x1 = min(w_img, int(math.ceil(float(np.max(dst_corners_xy[:, 0])))) + 2)
    y1 = min(h_img, int(math.ceil(float(np.max(dst_corners_xy[:, 1])))) + 2)
    rw = x1 - x0
    rh = y1 - y0
    if rw < 2 or rh < 2:
        return

    src_img = rgba_patch
    ph, pw = src_img.shape[:2]
    if rw < pw * 0.72 or rh < ph * 0.72:
        pre_w = max(8, min(pw, int(round(rw * 1.35))))
        pre_h = max(4, min(ph, int(round(rh * 1.35))))
        src_img = cv2.resize(src_img, (pre_w, pre_h), interpolation=cv2.INTER_AREA)
        ph, pw = src_img.shape[:2]

    src_pts = np.array(
        [[0.0, 0.0], [float(pw), 0.0], [float(pw), float(ph)], [0.0, float(ph)]],
        dtype=np.float32,
    )
    local_dst = dst_corners_xy.astype(np.float32).copy()
    local_dst[:, 0] -= float(x0)
    local_dst[:, 1] -= float(y0)

    H = cv2.getPerspectiveTransform(src_pts, local_dst)
    warped = cv2.warpPerspective(
        src_img,
        H,
        (rw, rh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    a = (warped[:, :, 3:4].astype(np.float32) * (float(opacity) / 255.0))
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(roi * (1.0 - a) + warped[:, :, :3].astype(np.float32) * a, 0.0, 255.0).astype(np.uint8)


def _apply_coupled_lowres_dct_420_np(
    canvas: np.ndarray,
    rng: random.Random,
    curr_obj_px: float,
    target_px_range: tuple[float, float] = (12.0, 30.0),
) -> tuple[float, tuple[float, float]]:
    """Coupled pre-resize `12..30 px` + YCbCr 4:2:0 + H.264 8x8 DCT degradation preserving zero phase shift."""
    h_img, w_img = canvas.shape[:2]
    target_px = rng.uniform(float(target_px_range[0]), float(target_px_range[1]))
    s_lr = float(np.clip(target_px / max(8.0, float(curr_obj_px)), 0.14, 0.82))
    lw = max(20, int(round(w_img * s_lr)))
    lh = max(14, int(round(h_img * s_lr)))

    small = cv2.resize(canvas, (lw, lh), interpolation=cv2.INTER_AREA).astype(np.float32) * (1.0 / 255.0)

    bvx_c, bvy_c = 0.0, 0.0
    r_blur = rng.random()
    if r_blur < 0.36:
        k_len = rng.choice((3, 5))
        ang = rng.uniform(0.0, math.pi)
        kern = np.zeros((k_len, k_len), dtype=np.float32)
        c_k = 0.5 * float(k_len - 1)
        ca, sa = math.cos(ang), math.sin(ang)
        for s_i in np.linspace(-c_k, c_k, k_len * 2 + 1):
            kx = int(round(c_k + s_i * ca))
            ky = int(round(c_k + s_i * sa))
            if 0 <= kx < k_len and 0 <= ky < k_len:
                kern[ky, kx] += 1.0
        kern /= max(1e-6, float(kern.sum()))
        small = cv2.filter2D(small, -1, kern, borderType=cv2.BORDER_REPLICATE)
        bvx_c = float(k_len * ca) / max(0.10, s_lr)
        bvy_c = float(k_len * sa) / max(0.10, s_lr)
    elif r_blur < 0.72:
        small = cv2.GaussianBlur(small, (3, 3), rng.uniform(0.50, 1.05))

    # Sensor Poisson-Gaussian noise at low-res capture scale
    if rng.random() < 0.75:
        np_rng = np.random.default_rng(rng.getrandbits(32))
        sigma = rng.uniform(0.006, 0.026)
        small = np.clip(small + np_rng.normal(0.0, sigma, size=small.shape).astype(np.float32), 0.0, 1.0)

    # ITU-R BT.601 YCbCr 4:2:0 2x2 area chroma subsampling + bilinear upsampling
    r_ch, g_ch, b_ch = small[:, :, 0], small[:, :, 1], small[:, :, 2]
    y_luma = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    cb = -0.168736 * r_ch - 0.331264 * g_ch + 0.500 * b_ch
    cr = 0.500 * r_ch - 0.418688 * g_ch - 0.081312 * b_ch
    cw = max(6, lw // 2)
    ch = max(4, lh // 2)
    cb_sub = cv2.resize(cb, (cw, ch), interpolation=cv2.INTER_AREA)
    cr_sub = cv2.resize(cr, (cw, ch), interpolation=cv2.INTER_AREA)
    cb_up = cv2.resize(cb_sub, (lw, lh), interpolation=cv2.INTER_LINEAR)
    cr_up = cv2.resize(cr_sub, (lw, lh), interpolation=cv2.INTER_LINEAR)

    ycbcr = np.stack([y_luma, cb_up, cr_up], axis=-1)

    # H.264 8x8 DCT quantization in YCbCr domain (`dct_step` up to `0.095`)
    dct_step = rng.uniform(0.020, 0.095)
    pad_h = (8 - (lh % 8)) % 8
    pad_w = (8 - (lw % 8)) % 8
    if pad_h > 0 or pad_w > 0:
        ycbcr_pad = np.pad(ycbcr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    else:
        ycbcr_pad = ycbcr
    ph8, pw8 = ycbcr_pad.shape[:2]
    bh, bw = ph8 // 8, pw8 // 8
    blocks = ycbcr_pad.reshape(bh, 8, bw, 8, 3).transpose(0, 2, 4, 1, 3)  # (bh, bw, 3, 8, 8)
    coeffs = (_DCT8_NP @ blocks) @ _DCT8_NP_T
    ch_scale = np.array([1.0, 1.35, 1.35], dtype=np.float32).reshape(1, 1, 3, 1, 1)
    q_step = dct_step * ch_scale * _DCT8_FREQ_WEIGHT_NP[None, None, None, :, :]
    coeffs_q = np.round(coeffs / q_step) * q_step
    rec_blocks = (_DCT8_NP_T @ coeffs_q) @ _DCT8_NP
    rec_ycbcr = rec_blocks.transpose(0, 3, 1, 4, 2).reshape(ph8, pw8, 3)[:lh, :lw, :]

    yr, cbr, crr = rec_ycbcr[:, :, 0], rec_ycbcr[:, :, 1], rec_ycbcr[:, :, 2]
    rgb_lr = np.stack(
        [
            yr + 1.402 * crr,
            yr - 0.344136 * cbr - 0.714136 * crr,
            yr + 1.772 * cbr,
        ],
        axis=-1,
    )
    rgb_u8 = np.clip(np.round(rgb_lr * 255.0), 0.0, 255.0).astype(np.uint8)

    u_up = rng.random()
    if u_up < 0.48:
        canvas[:] = cv2.resize(rgb_u8, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    elif u_up < 0.82:
        canvas[:] = cv2.resize(rgb_u8, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    else:
        # Pixel-center-exact nearest-neighbor upsampling (zero +0.5*(s-1) phase shift!)
        iy = np.clip(np.floor((np.arange(h_img, dtype=np.float32) + 0.5) * (float(lh) / float(h_img))).astype(np.int32), 0, lh - 1)
        ix = np.clip(np.floor((np.arange(w_img, dtype=np.float32) + 0.5) * (float(lw) / float(w_img))).astype(np.int32), 0, lw - 1)
        canvas[:] = rgb_u8[np.ix_(iy, ix)]

    qual = float(np.clip((target_px / 32.0) * (1.0 - 2.2 * dct_step), 0.24, 0.84))
    return qual, (bvx_c, bvy_c)


def _compute_3d_plate_homography(
    pw_m: float,
    ph_m: float,
    center_3d: tuple[float, float, float],
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Exact 3x3 pinhole homography mapping unit square [0,1]^2 -> pixel coords [TL, TR, BR, BL]."""
    yw = math.radians(yaw_deg)
    pt = math.radians(pitch_deg)
    rl = math.radians(roll_deg)

    cyw, syw = math.cos(yw), math.sin(yw)
    cpt, spt = math.cos(pt), math.sin(pt)
    crl, srl = math.cos(rl), math.sin(rl)

    # R = Rz(roll) @ Rx(pitch) @ Ry(yaw)
    # Column 0 (r1: right axis) and Column 1 (r2: down axis) in camera frame (X right, Y down, Z forward)
    r00 = crl * cyw + srl * spt * syw
    r10 = srl * cyw - crl * spt * syw
    r20 = -cpt * syw

    r01 = -srl * cpt
    r11 = crl * cpt
    r21 = -spt

    tx, ty, tz = center_3d
    c0x, c0y, c0z = pw_m * r00, pw_m * r10, pw_m * r20
    c1x, c1y, c1z = ph_m * r01, ph_m * r11, ph_m * r21
    ox = tx - 0.5 * c0x - 0.5 * c1x
    oy = ty - 0.5 * c0y - 0.5 * c1y
    oz = max(0.6, tz - 0.5 * c0z - 0.5 * c1z)

    M = np.array(
        [
            [c0x, c1x, ox],
            [c0y, c1y, oy],
            [c0z, c1z, oz],
        ],
        dtype=np.float32,
    )
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K @ M


class FullSceneDetSynthesizer:
    """High-throughput (< 4 ms/scene) full-frame multi-vehicle & hard-negative camera scene generator."""

    SCENE_FAMILIES: tuple[str, ...] = (
        "urban_road",
        "parking_lot",
        "highway",
        "logistics_gate",
        "street_facade",
    )
    WEATHER_MODES: tuple[str, ...] = (
        "clear",
        "overcast",
        "sunset",
        "night",
        "wet_road",
    )
    VEHICLE_ARCHETYPES: tuple[str, ...] = (
        "sedan",
        "hatchback",
        "kei_car",
        "kei_van",
        "minivan",
        "suv",
        "cargo_truck",
        "box_truck",
        "dump_truck",
        "bus",
        "commercial_van",
    )
    VEHICLE_WEIGHTS: tuple[float, ...] = (
        0.16,  # sedan
        0.10,  # hatchback
        0.13,  # kei_car
        0.09,  # kei_van
        0.11,  # minivan
        0.08,  # suv
        0.09,  # cargo_truck
        0.09,  # box_truck
        0.05,  # dump_truck
        0.04,  # bus
        0.06,  # commercial_van
    )

    BODY_COLORS: tuple[tuple[int, int, int], ...] = (
        (234, 236, 240),  # pearl white
        (218, 222, 226),  # silver
        (158, 164, 172),  # metallic gray
        (36, 38, 44),     # obsidian black
        (52, 56, 64),     # charcoal
        (34, 68, 128),    # deep navy blue
        (168, 36, 42),    # crimson red
        (194, 178, 148),  # champagne beige
        (58, 98, 72),     # olive green
        (218, 178, 46),   # yellow / taxi / kei
        (72, 124, 182),   # sky blue
    )

    def __init__(
        self,
        config: SceneSynthConfig | None = None,
        *,
        image_size: tuple[int, int] | None = None,
        image_hw: tuple[int, int] | None = None,
    ) -> None:
        cfg = config or SceneSynthConfig()
        eff_hw = image_hw or image_size
        if eff_hw is not None:
            cfg = replace(cfg, image_size=(int(eff_hw[0]), int(eff_hw[1])))
        self.config = cfg
        self._bank: _SceneAssetBank | None = None

    @property
    def bank(self) -> _SceneAssetBank:
        if self._bank is None:
            self._bank = _SceneAssetBank.get(self.config.plate_bank_size)
        return self._bank

    def _draw_background(
        self,
        canvas: np.ndarray,
        scene_family: str,
        weather: str,
        horizon_y: int,
        vp_x: int,
        rng: random.Random,
        extra_clutter: bool = False,
    ) -> None:
        """Render rich urban/parking/highway/gate/facade backgrounds with crosswalks, windows & Japanese signs."""
        h, w = canvas.shape[:2]
        hy = max(28, min(h - 60, horizon_y))

        # 1. Sky + Ground vertical gradients via fast row broadcasting
        if weather == "night":
            sky_top, sky_bot = (14, 18, 28), (32, 38, 54)
            rd_top, rd_bot = (28, 30, 35), (20, 22, 26)
        elif weather == "sunset":
            sky_top, sky_bot = (74, 96, 152), (236, 148, 88)
            rd_top, rd_bot = (74, 70, 74), (52, 50, 56)
        elif weather == "overcast":
            sky_top, sky_bot = (168, 176, 184), (198, 202, 206)
            rd_top, rd_bot = (88, 90, 94), (66, 68, 72)
        elif weather == "wet_road":
            sky_top, sky_bot = (132, 144, 158), (172, 180, 188)
            rd_top, rd_bot = (62, 66, 72), (40, 44, 50)
        else:  # clear
            sky_top, sky_bot = (96, 154, 224), (186, 212, 238)
            rd_top, rd_bot = (94, 96, 100), (70, 72, 76)

        t_sky = np.linspace(0.0, 1.0, hy, dtype=np.float32)[:, None]
        sky_grad = (np.array(sky_top, dtype=np.float32) * (1.0 - t_sky) + np.array(sky_bot, dtype=np.float32) * t_sky).astype(np.uint8)
        canvas[:hy] = sky_grad[:, None, :]

        rem_h = h - hy
        t_rd = np.linspace(0.0, 1.0, rem_h, dtype=np.float32)[:, None]
        rd_grad = (np.array(rd_top, dtype=np.float32) * (1.0 - t_rd) + np.array(rd_bot, dtype=np.float32) * t_rd).astype(np.int16)
        # Tile granular/cracked asphalt texture onto road surface so road is never a flat gradient
        asp_tile = self.bank.asphalt_deltas[rng.randrange(len(self.bank.asphalt_deltas))]
        ox_a, oy_a = rng.randrange(64), rng.randrange(64)
        asp_tiled = np.tile(np.roll(asp_tile, (oy_a, ox_a), axis=(0, 1)), (rem_h // 256 + 2, w // 256 + 2))[:rem_h, :w]
        canvas[hy:] = np.clip(rd_grad[:, None, :] + asp_tiled[:, :, None], 0, 255).astype(np.uint8)

        dim = 0.42 if weather == "night" else (0.82 if weather in ("sunset", "wet_road") else 1.0)

        def _sc(col: tuple[int, int, int]) -> tuple[int, int, int]:
            return (int(col[0] * dim), int(col[1] * dim), int(col[2] * dim))

        # 2. Building facades + grids of rectangular windows + shop/parking signs along horizon
        if scene_family in ("urban_road", "street_facade", "parking_lot", "logistics_gate") or extra_clutter:
            n_bldg = rng.randint(5, 9) if (scene_family == "street_facade" or extra_clutter) else rng.randint(3, 6)
            bx_cur = 0
            for _ in range(n_bldg):
                bw = rng.randint(max(48, w // 9), max(80, w // 4))
                bh = rng.randint(max(36, int(hy * 0.45)), max(52, hy + 12))
                by0 = max(0, hy - bh + rng.randint(0, 10))
                by1 = min(h, hy + rng.randint(0, 8))
                bx1 = min(w, bx_cur + bw)
                if bx1 - bx_cur < 24 or by1 - by0 < 12:
                    break
                b_col = _sc(rng.choice([(188, 186, 180), (156, 160, 166), (206, 198, 186), (134, 138, 146), (172, 158, 148)]))
                cv2.rectangle(canvas, (bx_cur, by0), (bx1, by1), b_col, -1)
                fac_tile = self.bank.facade_deltas[rng.randrange(len(self.bank.facade_deltas))]
                fh_b, fw_b = by1 - by0, bx1 - bx_cur
                fac_sub = np.tile(fac_tile, (fh_b // 256 + 2, fw_b // 256 + 2))[:fh_b, :fw_b, None]
                canvas[by0:by1, bx_cur:bx1] = np.clip(canvas[by0:by1, bx_cur:bx1].astype(np.int16) + fac_sub, 0, 255).astype(np.uint8)
                cv2.rectangle(canvas, (bx_cur, by0), (bx1, by1), _sc((104, 108, 114)), 1)

                # Grid of rectangular windows (hard-negative plate-aspect rectangles!)
                win_w = rng.randint(10, 22)
                win_h = max(6, int(win_w * rng.uniform(0.48, 0.78)))
                step_x = win_w + rng.randint(6, 12)
                step_y = win_h + rng.randint(7, 13)
                w_dark = (212, 196, 128) if (weather == "night" and rng.random() < 0.35) else _sc((44, 54, 66))
                w_frame = _sc((214, 218, 222))
                for wy in range(by0 + 8, by1 - win_h - 6, step_y):
                    for wx in range(bx_cur + 6, bx1 - win_w - 4, step_x):
                        cv2.rectangle(canvas, (wx, wy), (wx + win_w, wy + win_h), w_dark, -1)
                        cv2.rectangle(canvas, (wx, wy), (wx + win_w, wy + win_h), w_frame, 1)

                # Shop / parking signboard on building facade
                # Shop / parking signboard or striped storefront awning on building facade
                if rng.random() < (0.82 if extra_clutter else 0.56):
                    sp = rng.choice(self.bank.banner_patches if rng.random() < 0.42 else self.bank.sign_patches)
                    sw = min(bx1 - bx_cur - 6, rng.randint(42, 108))
                    sh = max(12, int(sw * rng.uniform(0.24, 0.46)))
                    sx = rng.randint(bx_cur + 2, max(bx_cur + 2, bx1 - sw - 2))
                    sy = min(h - sh - 2, max(4, by1 - sh - rng.randint(2, 24)))
                    if sw >= 18 and sh >= 9:
                        rs_sp = cv2.resize(sp, (sw, sh), interpolation=cv2.INTER_AREA)
                        if dim < 0.99:
                            rs_sp = (rs_sp.astype(np.float32) * max(0.55, dim)).astype(np.uint8)
                        canvas[sy : sy + sh, sx : sx + sw] = rs_sp
                        # Optional red/white striped shop awning right below sign (Harajuku storefront trap!)
                        if rng.random() < 0.45 and sy + sh + 10 < h:
                            aw_h = min(10, h - (sy + sh) - 2)
                            for ax in range(sx, sx + sw - 3, 8):
                                cv2.rectangle(canvas, (ax, sy + sh), (ax + 4, sy + sh + aw_h), _sc((212, 68, 72)), -1)
                                cv2.rectangle(canvas, (ax + 4, sy + sh), (min(sx + sw, ax + 8), sy + sh + aw_h), _sc((238, 236, 232)), -1)

                bx_cur = bx1 + rng.randint(-4, 14)
                if bx_cur >= w - 20:
                    break

        # Roadside trees / foliage canopies
        if rng.random() < 0.65:
            for _ in range(rng.randint(2, 6)):
                tx = rng.randint(15, w - 15)
                ty = hy - rng.randint(4, 26)
                tr = rng.randint(14, 34)
                f_col = _sc(rng.choice([(46, 88, 48), (58, 104, 54), (38, 74, 44)]))
                cv2.circle(canvas, (tx, ty), tr, f_col, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (tx + tr // 2, ty + 4), int(tr * 0.75), _sc((66, 114, 60)), -1, lineType=cv2.LINE_AA)

        # 3. Road / Highway / Parking / Gate Geometry
        lane_white = _sc((226, 228, 232))
        lane_yellow = _sc((228, 182, 42))

        if scene_family == "highway":
            # Acoustic sound walls + metallic W-beam guardrails + overhead green gantry sign
            wall_h = rng.randint(18, 36)
            cv2.rectangle(canvas, (0, hy - wall_h), (w, hy + 4), _sc((152, 158, 164)), -1)
            for gx in range(0, w, 28):
                cv2.line(canvas, (gx, hy - wall_h), (gx, hy + 4), _sc((112, 118, 124)), 1)
            # Overhead gantry truss & green highway boards
            gy = max(8, hy - rng.randint(36, 68))
            cv2.rectangle(canvas, (int(w * 0.08), gy), (int(w * 0.92), gy + 5), _sc((168, 174, 180)), -1)
            for bx0_f, bx1_f in ((0.16, 0.44), (0.54, 0.82)):
                gx0, gx1 = int(w * bx0_f), int(w * bx1_f)
                gh = rng.randint(20, 34)
                cv2.rectangle(canvas, (gx0, max(2, gy - gh // 2)), (gx1, gy + gh // 2), _sc((24, 108, 62)), -1)
                cv2.rectangle(canvas, (gx0, max(2, gy - gh // 2)), (gx1, gy + gh // 2), lane_white, 1)
            # Side metallic guardrails converging from bottom edges to horizon
            cv2.line(canvas, (0, int(h * 0.78)), (max(0, vp_x - 70), hy + 4), _sc((192, 198, 204)), 4, lineType=cv2.LINE_AA)
            cv2.line(canvas, (w - 1, int(h * 0.78)), (min(w - 1, vp_x + 70), hy + 4), _sc((192, 198, 204)), 4, lineType=cv2.LINE_AA)

        elif scene_family == "parking_lot":
            # Parking bay lines + concrete wheel stops (車止め) + freestanding parking tariff sign
            n_bays = rng.randint(5, 9)
            by_top = hy + int(rem_h * 0.14)
            by_bot = h - int(rem_h * 0.08)
            for bi in range(n_bays + 1):
                u = bi / float(n_bays)
                x_top = int(w * (0.08 + 0.84 * u))
                x_bot = int((x_top - vp_x) * 1.55 + vp_x)
                cv2.line(canvas, (x_top, by_top), (x_bot, by_bot), lane_white, 2, lineType=cv2.LINE_AA)
                if bi < n_bays:
                    ws_x = int(0.5 * (x_top + x_top + int(w * 0.84 / n_bays)))
                    ws_y = by_top + int((by_bot - by_top) * 0.22)
                    cv2.rectangle(canvas, (ws_x - 14, ws_y), (ws_x + 14, ws_y + 5), _sc((186, 188, 190)), -1)
            # Freestanding parking signboard
            sp = self.bank.sign_patches[rng.randrange(len(self.bank.sign_patches))]
            sx = rng.choice([int(w * 0.06), int(w * 0.76)])
            sy = max(12, hy - 26)
            cv2.line(canvas, (sx + 32, sy + 28), (sx + 32, hy + 28), _sc((155, 160, 166)), 3)
            rs_sp = cv2.resize(sp, (68, 28), interpolation=cv2.INTER_AREA)
            canvas[sy : sy + 28, sx : min(w, sx + 68)] = rs_sp[:, : min(w, sx + 68) - sx]

        elif scene_family == "logistics_gate":
            # Accordion metal gate bars + perimeter fence + safety cones (カラーコーン) + gate sign
            gate_y0 = max(16, hy - 8)
            gate_y1 = min(h - 16, hy + int(rem_h * 0.34))
            gx0 = int(w * rng.uniform(0.02, 0.12))
            gx1 = int(w * rng.uniform(0.42, 0.88 if extra_clutter else 0.58))
            bar_col = _sc((182, 188, 196))
            cv2.rectangle(canvas, (gx0, gate_y0), (gx1, gate_y0 + 3), bar_col, -1)
            cv2.rectangle(canvas, (gx0, gate_y1 - 3), (gx1, gate_y1), bar_col, -1)
            step_b = rng.randint(12, 20)
            for bx in range(gx0, gx1, step_b):
                cv2.line(canvas, (bx, gate_y0), (bx, gate_y1), bar_col, 2)
                if bx + step_b <= gx1:
                    cv2.line(canvas, (bx, gate_y0), (bx + step_b, gate_y1), _sc((148, 154, 162)), 1, lineType=cv2.LINE_AA)
                    cv2.line(canvas, (bx + step_b, gate_y0), (bx, gate_y1), _sc((148, 154, 162)), 1, lineType=cv2.LINE_AA)
            # Orange cones with white reflective collar
            for _ in range(rng.randint(2, 5)):
                cx_c = rng.randint(24, w - 24)
                cy_c = rng.randint(hy + 24, h - 18)
                ch_c = int(12 + 22 * ((cy_c - hy) / float(max(1, rem_h))))
                cw_c = max(5, ch_c // 2)
                pts_cone = np.array([[cx_c, cy_c - ch_c], [cx_c - cw_c, cy_c], [cx_c + cw_c, cy_c]], dtype=np.int32)
                cv2.fillConvexPoly(canvas, pts_cone, _sc((236, 78, 28)), lineType=cv2.LINE_AA)
                cv2.line(canvas, (cx_c - cw_c // 2, cy_c - ch_c // 2), (cx_c + cw_c // 2, cy_c - ch_c // 2), lane_white, 2)

        # Multi-lane solid & dashed white/yellow road lines (for road/highway/facade/gate)
        if scene_family != "parking_lot":
            for u_bot, is_dashed, col in (
                (-0.42, False, lane_white),
                (-0.14, True, lane_white),
                (0.02, False, lane_yellow if rng.random() < 0.55 else lane_white),
                (0.22, True, lane_white),
                (0.46, False, lane_white),
            ):
                xb = int(vp_x + u_bot * w * 1.45)
                xt = int(vp_x + u_bot * w * 0.10)
                if not is_dashed:
                    cv2.line(canvas, (xt, hy + 4), (xb, h - 1), col, 2, lineType=cv2.LINE_AA)
                else:
                    for seg in range(6):
                        t0 = (seg / 6.0) ** 1.6
                        t1 = ((seg + 0.52) / 6.0) ** 1.6
                        sx0 = int(xt + (xb - xt) * t0)
                        sy0 = int(hy + 4 + (h - hy - 5) * t0)
                        sx1 = int(xt + (xb - xt) * t1)
                        sy1 = int(hy + 4 + (h - hy - 5) * t1)
                        cv2.line(canvas, (sx0, sy0), (sx1, sy1), col, max(1, int(1 + 2 * t1)), lineType=cv2.LINE_AA)

        # 4. Crosswalk zebra stripes (urban_road / street_facade / hard negatives)
        if (scene_family in ("urban_road", "street_facade") and rng.random() < 0.62) or extra_clutter:
            cw_t = rng.uniform(0.32, 0.72)
            cy0 = int(hy + rem_h * cw_t)
            ch_z = max(14, int(rem_h * 0.13 * (0.5 + cw_t)))
            cy1 = min(h - 4, cy0 + ch_z)
            n_stripes = rng.randint(6, 11)
            span_top = w * (0.36 + 0.45 * cw_t)
            span_bot = span_top * 1.22
            for si in range(n_stripes):
                u0 = (si + 0.15) / n_stripes - 0.5
                u1 = (si + 0.68) / n_stripes - 0.5
                quad_z = np.array(
                    [
                        [vp_x + u0 * span_top, cy0],
                        [vp_x + u1 * span_top, cy0],
                        [vp_x + u1 * span_bot, cy1],
                        [vp_x + u0 * span_bot, cy1],
                    ],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(canvas, quad_z, _sc((218, 220, 222)), lineType=cv2.LINE_AA)

        # 5. Perspective-warped asphalt road painted text (`止まれ`, `50`, `徐行`)
        if rng.random() < (0.85 if extra_clutter else 0.52):
            rm = rng.choice(self.bank.road_masks)
            rt = rng.uniform(0.38, 0.80)
            ry0 = int(hy + rem_h * rt)
            ry1 = min(h - 3, ry0 + max(12, int(26 * rt)))
            rx_c = int(vp_x + rng.uniform(-0.22, 0.22) * w)
            rw_top = max(24, int(52 * rt))
            rw_bot = int(rw_top * 1.24)
            dst_q = np.array(
                [
                    [rx_c - rw_top, ry0],
                    [rx_c + rw_top, ry0],
                    [rx_c + rw_bot, ry1],
                    [rx_c - rw_bot, ry1],
                ],
                dtype=np.float32,
            )
            txt_rgb_patch = np.empty((rm.shape[0], rm.shape[1], 3), dtype=np.uint8)
            p_col = _sc((224, 182, 48) if rng.random() < 0.35 else (216, 218, 220))
            bg_asphalt = canvas[min(h - 1, (ry0 + ry1) // 2), min(w - 1, max(0, rx_c))].tolist()
            alpha_m = (rm.astype(np.float32) / 255.0)[:, :, None] * 0.82
            txt_rgb_patch[:] = (
                np.array(bg_asphalt, dtype=np.float32) * (1.0 - alpha_m) + np.array(p_col, dtype=np.float32) * alpha_m
            ).astype(np.uint8)
            _warp_patch_into_canvas(canvas, txt_rgb_patch, dst_q)

        # 6. Night streetlamp pools or wet-road vertical specular streaks
        if weather == "night":
            for _ in range(rng.randint(2, 4)):
                lx = rng.randint(40, w - 40)
                ly = rng.randint(max(6, hy - 50), hy + 10)
                cv2.circle(canvas, (lx, ly), rng.randint(4, 8), (252, 238, 192), -1, lineType=cv2.LINE_AA)
        elif weather == "wet_road":
            # Wet asphalt reflection streaks mirroring sky/building brightness above horizon
            reflect_h = min(hy, rem_h)
            if reflect_h > 12:
                flipped = canvas[hy - reflect_h : hy, :][::-1]
                blurred_ref = cv2.blur(flipped, (3, 13))
                canvas[hy : hy + reflect_h] = cv2.addWeighted(canvas[hy : hy + reflect_h], 0.72, blurred_ref, 0.28, 0.0)

    def _draw_vehicle_and_compute_plate(
        self,
        canvas: np.ndarray,
        archetype: str,
        plate_side: int,
        vx0: float,
        vy0: float,
        vx1: float,
        vy1: float,
        weather: str,
        rng: random.Random,
    ) -> tuple[float, float] | None:
        """Draw detailed Japanese car/kei/SUV/cab-over-truck/bus silhouette; return plate anchor `(px_center, py_center)`."""
        h_img, w_img = canvas.shape[:2]
        vw = max(10.0, vx1 - vx0)
        vh = max(10.0, vy1 - vy0)
        ix0, iy0, ix1, iy1 = int(round(vx0)), int(round(vy0)), int(round(vx1)), int(round(vy1))

        dim = 0.48 if weather == "night" else (0.86 if weather == "sunset" else 1.0)
        raw_col = rng.choice(self.BODY_COLORS)
        if archetype in ("cargo_truck", "box_truck", "dump_truck"):
            raw_col = rng.choice([(232, 235, 238), (206, 212, 218), (44, 92, 158), (58, 112, 82), (168, 174, 180)])
        elif archetype == "bus":
            raw_col = rng.choice([(236, 238, 240), (198, 224, 242), (238, 216, 172)])
        body_col = (int(raw_col[0] * dim), int(raw_col[1] * dim), int(raw_col[2] * dim))
        dark_body = (max(12, int(body_col[0] * 0.62)), max(12, int(body_col[1] * 0.62)), max(14, int(body_col[2] * 0.62)))
        glass_col = (int(42 * dim + 12), int(54 * dim + 14), int(68 * dim + 18))
        tire_col = (20, 22, 25)
        chrome_col = (int(210 * dim), int(215 * dim), int(222 * dim))

        # Ground contact shadow beneath vehicle
        sh_y0 = int(round(vy1 - vh * 0.06))
        sh_y1 = int(round(vy1 + vh * 0.05))
        cv2.ellipse(
            canvas,
            (int(round(0.5 * (vx0 + vx1))), int(round(vy1))),
            (max(4, int(vw * 0.48)), max(2, int((sh_y1 - sh_y0) * 0.6))),
            0,
            0,
            360,
            (18, 20, 24),
            -1,
            lineType=cv2.LINE_AA,
        )

        # Pure side-view vehicle (car or truck crossing/parked sideways -> NO visible plate!)
        if plate_side == SIDE_PURE_SIDE:
            is_truck_side = archetype in ("cargo_truck", "box_truck", "dump_truck")
            if is_truck_side:
                cab_w = int(vw * 0.28)
                bed_x0 = ix0 + cab_w + max(1, int(vw * 0.02))
                # Chassis + underrun bar
                cv2.rectangle(canvas, (ix0 + int(vw * 0.06), int(vy0 + vh * 0.68)), (ix1 - int(vw * 0.04), int(vy0 + vh * 0.84)), (32, 34, 38), -1)
                # Cab + Cargo box
                cv2.rectangle(canvas, (ix0, int(vy0 + vh * 0.22)), (ix0 + cab_w, int(vy0 + vh * 0.82)), body_col, -1)
                cv2.rectangle(canvas, (ix0 + 2, int(vy0 + vh * 0.28)), (ix0 + int(cab_w * 0.65), int(vy0 + vh * 0.54)), glass_col, -1)
                box_col = (int(224 * dim), int(228 * dim), int(232 * dim))
                cv2.rectangle(canvas, (bed_x0, iy0), (ix1, int(vy0 + vh * 0.72)), box_col, -1)
                cv2.rectangle(canvas, (bed_x0, iy0), (ix1, int(vy0 + vh * 0.72)), dark_body, 1)
                # Large painted fleet number on truck side bed (`春日 2503` — NOT a license plate!)
                if vw >= 48 and rng.random() < 0.75:
                    bn = rng.choice(self.bank.bed_num_patches)
                    clamped_bed_x0 = max(0, bed_x0)
                    clamped_ix1 = min(w_img, ix1)
                    avail_w = clamped_ix1 - clamped_bed_x0 - 6
                    if avail_w >= 20:
                        bw_n = max(16, min(avail_w, w_img - 4, int(avail_w * 0.48)))
                        bh_n = max(5, min(h_img - 4, int(bw_n * 0.25)))
                        bx_n = max(0, min(w_img - bw_n, clamped_ix1 - bw_n - max(2, int(avail_w * 0.06))))
                        by_n = max(0, min(h_img - bh_n, int(vy0 + vh * 0.46)))
                        canvas[by_n : by_n + bh_n, bx_n : bx_n + bw_n] = cv2.resize(bn, (bw_n, bh_n), interpolation=cv2.INTER_AREA)
            else:
                low_y0 = int(round(vy0 + vh * 0.44))
                low_y1 = int(round(vy0 + vh * 0.86))
                cv2.rectangle(canvas, (ix0, low_y0), (ix1, low_y1), body_col, -1)
                cab_pts = np.array(
                    [
                        [vx0 + vw * 0.18, low_y0],
                        [vx0 + vw * 0.30, vy0 + vh * 0.08],
                        [vx0 + vw * 0.78, vy0 + vh * 0.08],
                        [vx0 + vw * 0.90, low_y0],
                    ],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(canvas, cab_pts, body_col, lineType=cv2.LINE_AA)
                win_pts = np.array(
                    [
                        [vx0 + vw * 0.23, low_y0 - 1],
                        [vx0 + vw * 0.32, vy0 + vh * 0.14],
                        [vx0 + vw * 0.75, vy0 + vh * 0.14],
                        [vx0 + vw * 0.85, low_y0 - 1],
                    ],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(canvas, win_pts, glass_col, lineType=cv2.LINE_AA)
            # Side wheels (dual tandem rear wheels with lug-nut rings on trucks!)
            wr = max(3, int(vh * 0.15))
            wy = int(round(vy1 - wr))
            wheel_xs = (0.20, 0.66, 0.84) if is_truck_side else (0.22, 0.78)
            for wx_f in wheel_xs:
                wx = int(round(vx0 + vw * wx_f))
                cv2.circle(canvas, (wx, wy), wr, tire_col, -1, lineType=cv2.LINE_AA)
                rim_r = max(1, int(wr * 0.56))
                cv2.circle(canvas, (wx, wy), rim_r, chrome_col, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (wx, wy), max(1, int(rim_r * 0.38)), (38, 40, 44), -1, lineType=cv2.LINE_AA)
                if wr >= 8:
                    for ang_i in range(6):
                        ang = ang_i * (math.pi / 3.0)
                        lx = int(round(wx + rim_r * 0.68 * math.cos(ang)))
                        ly = int(round(wy + rim_r * 0.68 * math.sin(ang)))
                        cv2.circle(canvas, (lx, ly), max(1, wr // 9), (28, 30, 34), -1)
            return None

        # Left & right tires for front/rear views
        tw = max(2, int(vw * 0.15))
        th_tire = max(3, int(vh * 0.16))
        cv2.rectangle(canvas, (ix0 + int(vw * 0.04), iy1 - th_tire), (ix0 + int(vw * 0.04) + tw, iy1), tire_col, -1)
        cv2.rectangle(canvas, (ix1 - int(vw * 0.04) - tw, iy1 - th_tire), (ix1 - int(vw * 0.04), iy1), tire_col, -1)

        is_front = plate_side == SIDE_FRONT
        headlamp_col = (252, 248, 224) if weather == "night" else (232, 238, 246)
        taillamp_col = (242, 38, 42) if weather == "night" else (196, 32, 38)

        # --- ARCHETYPE 4: CARGO / BOX / DUMP TRUCK ---
        if archetype in ("cargo_truck", "box_truck", "dump_truck"):
            if is_front:
                # Cab-over flat front with tall windshield, wide slotted grille, corner headlamps, steel bumper
                cab_y0 = int(round(vy0 + (vh * 0.12 if archetype == "box_truck" else vh * 0.04)))
                if archetype == "box_truck":
                    # Upper box overhang above cab
                    cv2.rectangle(canvas, (ix0, iy0), (ix1, cab_y0 + 4), (int(228 * dim), int(230 * dim), int(234 * dim)), -1)
                bmp_y0 = int(round(vy0 + vh * 0.72))
                bmp_y1 = int(round(vy0 + vh * 0.90))
                cv2.rectangle(canvas, (ix0 + int(vw * 0.03), cab_y0), (ix1 - int(vw * 0.03), bmp_y0), body_col, -1)
                # Tall windshield + dual heavy wipers (hard negative bars inside windshield!)
                ws_y0 = int(round(cab_y0 + (bmp_y0 - cab_y0) * 0.10))
                ws_y1 = int(round(cab_y0 + (bmp_y0 - cab_y0) * 0.54))
                cv2.rectangle(canvas, (ix0 + int(vw * 0.08), ws_y0), (ix1 - int(vw * 0.08), ws_y1), glass_col, -1)
                if vw >= 28:
                    cv2.line(canvas, (ix0 + int(vw * 0.16), ws_y1 - 2), (ix0 + int(vw * 0.44), ws_y1 - max(3, int(vh * 0.07))), (22, 24, 28), max(1, int(vw * 0.015)), lineType=cv2.LINE_AA)
                    cv2.line(canvas, (ix0 + int(vw * 0.52), ws_y1 - 2), (ix0 + int(vw * 0.80), ws_y1 - max(3, int(vh * 0.07))), (22, 24, 28), max(1, int(vw * 0.015)), lineType=cv2.LINE_AA)
                # Wide radiator grille slats (hard-negative horizontal bars)
                gr_x0, gr_x1 = ix0 + int(vw * 0.18), ix1 - int(vw * 0.18)
                gr_y0, gr_y1 = int(round(ws_y1 + vh * 0.05)), bmp_y0 - 2
                if gr_y1 > gr_y0 + 2 and gr_x1 > gr_x0 + 4:
                    cv2.rectangle(canvas, (gr_x0, gr_y0), (gr_x1, gr_y1), (34, 36, 40), -1)
                    for gy in range(gr_y0 + 2, gr_y1, max(2, (gr_y1 - gr_y0) // 3)):
                        cv2.line(canvas, (gr_x0 + 1, gy), (gr_x1 - 1, gy), chrome_col, 1)
                # Corner headlamps + amber turn signals
                hl_w = max(3, int(vw * 0.13))
                hl_y0 = max(ws_y1 + 2, bmp_y0 - max(4, int(vh * 0.12)))
                cv2.rectangle(canvas, (ix0 + int(vw * 0.04), hl_y0), (ix0 + int(vw * 0.04) + hl_w, bmp_y0 - 1), headlamp_col, -1)
                cv2.rectangle(canvas, (ix1 - int(vw * 0.04) - hl_w, hl_y0), (ix1 - int(vw * 0.04), bmp_y0 - 1), headlamp_col, -1)
                # Heavy steel bumper (or snowplow / loader lower blade mount with high radiator-grille plate!)
                high_grille_mount = (archetype != "box_truck") and (rng.random() < 0.28)
                bmp_col = (218, 168, 36) if (high_grille_mount and rng.random() < 0.45) else (chrome_col if rng.random() < 0.35 else (48, 52, 58))
                cv2.rectangle(canvas, (ix0 + 1, bmp_y0), (ix1 - 1, bmp_y1), bmp_col, -1)
                px_c = 0.5 * (vx0 + vx1) + rng.uniform(-0.06, 0.06) * vw
                if high_grille_mount and gr_y1 > gr_y0 + 2:
                    py_c = 0.5 * (gr_y0 + gr_y1)
                else:
                    py_c = 0.5 * (bmp_y0 + bmp_y1)
                return (px_c, py_c)
            else:
                rear_variant = rng.random()
                if archetype != "box_truck" and rear_variant < 0.34:
                    # Exposed semi-tractor / concrete-mixer / crane rear chassis (NO flat rectangular box wall above!)
                    is_mixer = rear_variant < 0.18
                    cab_top_w = int(vw * (0.64 if is_mixer else 0.72))
                    cx_i = (ix0 + ix1) // 2
                    if is_mixer:
                        # Curved ready-mix concrete drum + spiral stripe + chute bracket
                        drum_cy = int(round(vy0 + vh * 0.38))
                        cv2.ellipse(
                            canvas,
                            (cx_i, drum_cy),
                            (max(6, int(vw * 0.40)), max(6, int(vh * 0.32))),
                            0,
                            0,
                            360,
                            body_col,
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                        stripe_c = (int(42 * dim), int(124 * dim), int(186 * dim))
                        cv2.line(canvas, (cx_i - int(vw * 0.32), drum_cy - int(vh * 0.10)), (cx_i + int(vw * 0.30), drum_cy + int(vh * 0.12)), stripe_c, max(2, int(vh * 0.04)), lineType=cv2.LINE_AA)
                    else:
                        # Tractor rear cab window + fifth-wheel open chassis frame rails
                        cv2.rectangle(canvas, (cx_i - cab_top_w // 2, iy0), (cx_i + cab_top_w // 2, int(round(vy0 + vh * 0.54))), body_col, -1)
                        cv2.rectangle(canvas, (cx_i - cab_top_w // 3, int(round(vy0 + vh * 0.08))), (cx_i + cab_top_w // 3, int(round(vy0 + vh * 0.28))), glass_col, -1)
                    ch_y0 = int(round(vy0 + vh * 0.56))
                    ch_y1 = int(round(vy0 + vh * 0.88))
                    cv2.rectangle(canvas, (ix0 + int(vw * 0.10), ch_y0), (ix1 - int(vw * 0.10), ch_y1), (28, 30, 35), -1)
                    # Optional red/white or yellow/black crane/mixer chevron underrun bumper
                    bar_y = int(round(vy0 + vh * 0.82))
                    bar_h = max(3, int(vh * 0.05))
                    cv2.rectangle(canvas, (ix0 + int(vw * 0.08), bar_y), (ix1 - int(vw * 0.08), bar_y + bar_h), (228, 224, 216), -1)
                    if vw >= 28 and rng.random() < 0.55:
                        c_chv = (208, 42, 42) if rng.random() < 0.5 else (24, 26, 30)
                        step_c = max(6, int(vw * 0.12))
                        for sx_c in range(ix0 + int(vw * 0.08), ix1 - int(vw * 0.08), step_c):
                            cv2.line(canvas, (sx_c, bar_y + bar_h), (min(ix1, sx_c + step_c // 2), bar_y), c_chv, max(1, step_c // 3))
                    tl_w = max(4, int(vw * 0.15))
                    tl_h = max(3, int(vh * 0.06))
                    tl_y = ch_y0 + max(2, int(vh * 0.04))
                    for tl_x0 in (ix0 + int(vw * 0.08), ix1 - int(vw * 0.08) - tl_w):
                        cv2.rectangle(canvas, (tl_x0, tl_y), (tl_x0 + tl_w, tl_y + tl_h), taillamp_col, -1)
                    # Tractor/mixer rear plates are often left-offset or centered on dark frame bracket
                    off_frac = rng.choice([-0.18, -0.12, 0.0, 0.06])
                    px_c = 0.5 * (vx0 + vx1) + off_frac * vw
                    py_c = ch_y0 + 0.54 * (bar_y - ch_y0)
                    return (px_c, py_c)

                # CRITICAL TRUCK REAR: high cargo bed/box, dark recessed under-bed area, tail-lamp blocks,
                # underrun bar, mudflaps, and rectangular `最大積載量 5000kg` sticker near plate (NOT CLS_PLATE!)
                bed_y1 = int(round(vy0 + vh * 0.68))
                under_y1 = int(round(vy0 + vh * 0.90))
                bed_col = (int(226 * dim), int(229 * dim), int(233 * dim)) if archetype == "box_truck" else body_col
                cv2.rectangle(canvas, (ix0, iy0), (ix1, bed_y1), bed_col, -1)
                cv2.rectangle(canvas, (ix0, iy0), (ix1, bed_y1), dark_body, max(1, int(vw * 0.015)))

                if archetype == "box_truck" and vw >= 28:
                    # Vertical locking rods + reflective conspicuity stripe
                    for rx_f in (0.36, 0.64):
                        rx = int(round(vx0 + vw * rx_f))
                        cv2.line(canvas, (rx, iy0 + 2), (rx, bed_y1 - 2), chrome_col, 1)
                    cv2.line(canvas, (ix0 + 2, bed_y1 - 2), (ix1 - 2, bed_y1 - 2), (206, 48, 42), max(1, int(vh * 0.015)))
                elif vw >= 24:
                    # Dump/cargo tailgate ribs
                    for ry_f in (0.28, 0.48):
                        ry = int(round(vy0 + vh * ry_f))
                        cv2.line(canvas, (ix0 + 2, ry), (ix1 - 2, ry), dark_body, 1)

                # Dark recessed under-bed chassis cavity
                cv2.rectangle(canvas, (ix0 + int(vw * 0.08), bed_y1), (ix1 - int(vw * 0.08), under_y1), (24, 26, 30), -1)
                # Mudflaps
                mf_w = max(3, int(vw * 0.14))
                cv2.rectangle(canvas, (ix0 + int(vw * 0.06), bed_y1 + 2), (ix0 + int(vw * 0.06) + mf_w, iy1 - 1), (18, 20, 22), -1)
                cv2.rectangle(canvas, (ix1 - int(vw * 0.06) - mf_w, bed_y1 + 2), (ix1 - int(vw * 0.06), iy1 - 1), (18, 20, 22), -1)
                # Underrun protection bar (突入防止装置)
                bar_y = int(round(vy0 + vh * 0.85))
                cv2.rectangle(canvas, (ix0 + int(vw * 0.12), bar_y), (ix1 - int(vw * 0.12), bar_y + max(2, int(vh * 0.03))), (68, 72, 78), -1)
                # Red/amber rectangular tail-lamp blocks
                tl_w = max(4, int(vw * 0.16))
                tl_h = max(3, int(vh * 0.065))
                tl_y = bed_y1 + max(1, int(vh * 0.02))
                for tl_x0 in (ix0 + int(vw * 0.09), ix1 - int(vw * 0.09) - tl_w):
                    cv2.rectangle(canvas, (tl_x0, tl_y), (tl_x0 + tl_w, tl_y + tl_h), taillamp_col, -1)
                    cv2.rectangle(canvas, (tl_x0, tl_y), (tl_x0 + tl_w // 3, tl_y + tl_h), (232, 148, 28), -1)

                # Rectangular `最大積載量 5000kg` sticker right near plate (negative background ONLY!)
                if vw >= 22:
                    st_patch = rng.choice(self.bank.sticker_patches)
                    st_w = max(10, min(w_img - 4, int(vw * 0.28)))
                    st_h = max(4, int(st_w * 0.24))
                    st_x = int(round(vx1 - vw * 0.36))
                    st_y = max(iy0 + 2, bed_y1 - st_h - max(2, int(vh * 0.03)))
                    sx0, sy0 = max(0, st_x), max(0, st_y)
                    sx1, sy1 = min(w_img, st_x + st_w), min(h_img, st_y + st_h)
                    if sx1 - sx0 >= 6 and sy1 - sy0 >= 3:
                        rs_st = cv2.resize(st_patch, (sx1 - sx0, sy1 - sy0), interpolation=cv2.INTER_AREA)
                        if dim < 0.99:
                            rs_st = (rs_st.astype(np.float32) * max(0.5, dim)).astype(np.uint8)
                        canvas[sy0:sy1, sx0:sx1] = rs_st

                px_c = 0.5 * (vx0 + vx1) + rng.uniform(-0.12, 0.06) * vw
                py_c = bed_y1 + 0.52 * (bar_y - bed_y1)
                return (px_c, py_c)

        # --- ARCHETYPE 5: BUS / COMMERCIAL VAN ---
        if archetype == "bus":
            cv2.rectangle(canvas, (ix0, iy0), (ix1, int(round(vy0 + vh * 0.88))), body_col, -1)
            stripe_col = (int(38 * dim), int(108 * dim), int(178 * dim))
            cv2.rectangle(canvas, (ix0, int(round(vy0 + vh * 0.52))), (ix1, int(round(vy0 + vh * 0.62))), stripe_col, -1)
            cv2.rectangle(canvas, (ix0 + int(vw * 0.06), int(round(vy0 + vh * 0.14))), (ix1 - int(vw * 0.06), int(round(vy0 + vh * 0.48))), glass_col, -1)
            # Amber destination LED sign
            if is_front and vw >= 24:
                cv2.rectangle(canvas, (ix0 + int(vw * 0.24), int(round(vy0 + vh * 0.04))), (ix1 - int(vw * 0.24), int(round(vy0 + vh * 0.11))), (32, 30, 28), -1)
                cv2.line(canvas, (ix0 + int(vw * 0.28), int(round(vy0 + vh * 0.075))), (ix1 - int(vw * 0.28), int(round(vy0 + vh * 0.075))), (242, 168, 42), max(1, int(vh * 0.02)))
            lamp_c = headlamp_col if is_front else taillamp_col
            ly = int(round(vy0 + vh * 0.72))
            lw = max(3, int(vw * 0.13))
            cv2.rectangle(canvas, (ix0 + int(vw * 0.06), ly), (ix0 + int(vw * 0.06) + lw, ly + max(2, int(vh * 0.06))), lamp_c, -1)
            cv2.rectangle(canvas, (ix1 - int(vw * 0.06) - lw, ly), (ix1 - int(vw * 0.06), ly + max(2, int(vh * 0.06))), lamp_c, -1)
            return (0.5 * (vx0 + vx1), vy0 + vh * 0.79)

        # --- ARCHETYPES 1, 2, 3: SEDAN / HATCHBACK, KEI CAR / KEI VAN, MINIVAN / SUV, COMMERCIAL VAN ---
        is_boxy = archetype in ("kei_car", "kei_van", "minivan", "commercial_van")
        belt_frac = 0.42 if is_boxy else 0.48
        top_inset = 0.07 if is_boxy else 0.15
        belt_y = vy0 + vh * belt_frac
        bmp_bot_y = vy0 + vh * 0.88

        # Upper cabin trapezoid + glass windshield/rear window
        cab_poly = np.array(
            [
                [vx0 + vw * 0.03, belt_y],
                [vx0 + vw * top_inset, vy0],
                [vx1 - vw * top_inset, vy0],
                [vx1 - vw * 0.03, belt_y],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, cab_poly, body_col, lineType=cv2.LINE_AA)
        win_poly = np.array(
            [
                [vx0 + vw * 0.09, belt_y - 1],
                [vx0 + vw * (top_inset + 0.04), vy0 + vh * 0.06],
                [vx1 - vw * (top_inset + 0.04), vy0 + vh * 0.06],
                [vx1 - vw * 0.09, belt_y - 1],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, win_poly, glass_col, lineType=cv2.LINE_AA)

        # Lower body & bumper
        cv2.rectangle(canvas, (ix0, int(round(belt_y))), (ix1, int(round(bmp_bot_y))), body_col, -1)
        cv2.line(canvas, (ix0, int(round(vy0 + vh * 0.66))), (ix1, int(round(vy0 + vh * 0.66))), dark_body, 1)

        # Headlights (front) or Taillights (rear)
        lamp_y0 = int(round(belt_y + vh * 0.04))
        lamp_y1 = int(round(belt_y + vh * 0.14))
        lamp_w = max(3, int(vw * 0.18))
        l_col = headlamp_col if is_front else taillamp_col
        cv2.rectangle(canvas, (ix0 + int(vw * 0.04), lamp_y0), (ix0 + int(vw * 0.04) + lamp_w, lamp_y1), l_col, -1)
        cv2.rectangle(canvas, (ix1 - int(vw * 0.04) - lamp_w, lamp_y0), (ix1 - int(vw * 0.04), lamp_y1), l_col, -1)

        if is_front:
            if archetype in ("kei_car", "kei_van"):
                # CRITICAL KEI FRONT: OFF-CENTER plate shifted left or right next to black radiator grille!
                shift_side = -1.0 if rng.random() < 0.55 else 1.0
                px_c = 0.5 * (vx0 + vx1) + shift_side * rng.uniform(0.18, 0.28) * vw
                py_c = vy0 + vh * 0.76
                # Opposite-side black radiator intake grille (hard-negative rectangular slot!)
                gr_cx = 0.5 * (vx0 + vx1) - shift_side * 0.18 * vw
                gr_hw = max(3, int(vw * 0.19))
                gr_y0 = int(round(vy0 + vh * 0.69))
                gr_y1 = int(round(vy0 + vh * 0.84))
                cv2.rectangle(canvas, (int(gr_cx - gr_hw), gr_y0), (int(gr_cx + gr_hw), gr_y1), (26, 28, 32), -1)
                for gy in range(gr_y0 + 2, gr_y1, max(2, (gr_y1 - gr_y0) // 3)):
                    cv2.line(canvas, (int(gr_cx - gr_hw + 1), gy), (int(gr_cx + gr_hw - 1), gy), (72, 76, 82), 1)
                return (px_c, py_c)
            else:
                # Centered radiator grille slats (chrome grille on minivan/SUV, dark slats on sedan)
                gr_x0 = ix0 + int(vw * 0.24)
                gr_x1 = ix1 - int(vw * 0.24)
                gr_y0 = lamp_y0
                gr_y1 = int(round(vy0 + vh * (0.72 if archetype in ("minivan", "suv") else 0.64)))
                if gr_x1 > gr_x0 + 4 and gr_y1 > gr_y0 + 2:
                    cv2.rectangle(canvas, (gr_x0, gr_y0), (gr_x1, gr_y1), (32, 34, 38), -1)
                    slat_c = chrome_col if archetype in ("minivan", "suv") else (96, 102, 110)
                    for gy in range(gr_y0 + 2, gr_y1, max(2, (gr_y1 - gr_y0) // 4)):
                        cv2.line(canvas, (gr_x0 + 1, gy), (gr_x1 - 1, gy), slat_c, 1)
                return (0.5 * (vx0 + vx1), vy0 + vh * 0.76)
        else:
            # Rear view: tailgate garnish bar above recessed plate pocket (prominent chrome on minivan/SUV)
            pocket_w = max(6, int(vw * 0.34))
            pcx = int(round(0.5 * (vx0 + vx1)))
            py_top = int(round(vy0 + vh * (0.58 if archetype in ("minivan", "suv", "commercial_van") else 0.65)))
            if archetype in ("minivan", "suv"):
                cv2.rectangle(canvas, (pcx - pocket_w // 2 - 2, py_top - max(2, int(vh * 0.04))), (pcx + pocket_w // 2 + 2, py_top), chrome_col, -1)
            cv2.rectangle(canvas, (pcx - pocket_w // 2, py_top), (pcx + pocket_w // 2, py_top + max(3, int(vh * 0.14))), dark_body, -1)
            py_c = vy0 + vh * (0.67 if archetype in ("minivan", "suv", "commercial_van") else 0.73)
            return (0.5 * (vx0 + vx1), py_c)

    def synthesize_scene(self, seed: int | None = None) -> SynthSceneSample:
        """Synthesize a single full-frame camera scene `< 4 ms` on 1 CPU thread."""
        rng = random.Random(0 if seed is None else int(seed))
        h_img, w_img = self.config.image_size
        s_img = w_img / 640.0

        scene_family = rng.choice(self.SCENE_FAMILIES)
        weather = rng.choices(self.WEATHER_MODES, weights=[0.36, 0.22, 0.12, 0.14, 0.16], k=1)[0]
        cam_height_m = rng.choices(
            [rng.uniform(1.2, 2.2), rng.uniform(2.2, 4.8), rng.uniform(4.8, 9.0)],
            weights=[0.32, 0.40, 0.28],
            k=1,
        )[0]

        # Higher pole/bridge camera -> higher horizon / steeper downward viewing pitch
        h_norm = (cam_height_m - 1.2) / 7.8
        horizon_y = int(round(h_img * (0.38 - 0.20 * h_norm + rng.uniform(-0.04, 0.04))))
        horizon_y = max(28, min(int(h_img * 0.50), horizon_y))
        vp_x = int(round(w_img * rng.uniform(0.36, 0.64)))

        canvas = np.empty((h_img, w_img, 3), dtype=np.uint8)
        is_hard_neg = rng.random() < self.config.p_hard_negative_scene
        self._draw_background(
            canvas,
            scene_family=scene_family,
            weather=weather,
            horizon_y=horizon_y,
            vp_x=vp_x,
            rng=rng,
            extra_clutter=is_hard_neg,
        )

        if is_hard_neg:
            return SynthSceneSample(
                image_rgb=canvas,
                objects=[],
                scene_tag=f"{scene_family}_{weather}_hardneg",
                camera_height_m=round(cam_height_m, 2),
            )

        # Sample 1 to 8 vehicles stratified across distant (9-28px), mid (28-75px), and close-up (75-210px)
        n_vehicles = rng.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[0.14, 0.20, 0.20, 0.16, 0.12, 0.08, 0.06, 0.04], k=1)[0]
        fx = fy = 620.0 * s_img
        cx_cam, cy_cam = w_img * 0.5, float(horizon_y)

        veh_specs: list[dict[str, Any]] = []
        occupied_u: list[tuple[float, float, float]] = []

        for v_i in range(n_vehicles):
            arch = rng.choices(self.VEHICLE_ARCHETYPES, weights=self.VEHICLE_WEIGHTS, k=1)[0]
            r_side = rng.random()
            if r_side < self.config.p_pure_side_vehicle:
                p_side = SIDE_PURE_SIDE
            elif r_side < 0.55:
                p_side = SIDE_FRONT
            else:
                p_side = SIDE_REAR

            # Physical dimensions (meters) & plate category
            is_large_plate = False
            if arch in ("kei_car", "kei_van"):
                vw_m, vh_m, vlen_m = 1.48, 1.66, 3.39
                p_cat = "kei"
            elif arch == "bus":
                vw_m, vh_m, vlen_m = 2.46, 3.10, 9.20
                p_cat = "commercial_large"
                is_large_plate = True
            elif arch in ("cargo_truck", "box_truck", "dump_truck"):
                is_large_plate = rng.random() < 0.65
                vw_m = 2.38 if is_large_plate else 1.88
                vh_m = 3.05 if arch == "box_truck" else 2.55
                vlen_m = 7.40 if is_large_plate else 4.95
                p_cat = "commercial_large" if is_large_plate else "commercial_med"
            elif arch == "commercial_van":
                vw_m, vh_m, vlen_m = 1.72, 1.98, 4.69
                p_cat = "commercial_med" if rng.random() < 0.60 else "standard"
            elif arch in ("minivan", "suv"):
                vw_m, vh_m, vlen_m = 1.84, 1.84, 4.75
                p_cat = "standard"
            else:
                vw_m, vh_m, vlen_m = 1.76, 1.44, 4.55
                p_cat = "standard"

            pw_m, ph_m = (0.440, 0.220) if is_large_plate else (0.330, 0.165)

            # Scale stratification: >= 45% small/distant (9-28 px), ~35% mid (28-75 px), ~20% close (75-210 px)
            r_scale = rng.random()
            if r_scale < self.config.p_small_plate:
                target_pw_px = rng.uniform(9.4, 26.2) * s_img
                scale_tier = "small"
            elif r_scale < self.config.p_small_plate + self.config.p_mid_plate:
                target_pw_px = rng.uniform(28.2, 73.5) * s_img
                scale_tier = "mid"
            else:
                max_close = 132.0 if n_vehicles >= 5 else (156.0 if is_large_plate else 192.0)
                target_pw_px = rng.uniform(75.5, max_close) * s_img
                scale_tier = "large"

            # 3D perspective angles: yaw (-65..+65), pitch (-35..+15), roll (-8..+8)
            max_yaw = 36.0 if scale_tier == "small" else 65.0
            yaw_deg = rng.uniform(-max_yaw, max_yaw) if rng.random() < 0.45 else rng.uniform(-22.0, 22.0)
            pitch_deg = rng.uniform(-35.0, -8.0) * h_norm + rng.uniform(-12.0, 15.0) * (1.0 - 0.6 * h_norm)
            if scale_tier == "small":
                pitch_deg = max(-22.0, min(12.0, pitch_deg))
            else:
                pitch_deg = max(-35.0, min(15.0, pitch_deg))
            roll_deg = rng.uniform(-5.0, 5.0) if scale_tier == "small" else rng.uniform(-8.0, 8.0)

            cos_yaw = max(0.40, math.cos(math.radians(yaw_deg)))
            # Allow close-up construction-gate regime (115..205 px plates, partial cabs) on ~35% of `large` tier
            if scale_tier == "large" and n_vehicles <= 3 and rng.random() < 0.35:
                min_z_for_frame = (fx * pw_m) / max(32.0, 0.36 * float(w_img))
            else:
                min_z_for_frame = (fx * vw_m * 0.95) / (0.76 * float(w_img))
            z_m = max(1.02, min_z_for_frame, (fx * pw_m * cos_yaw) / max(6.0, target_pw_px))

            span_w_m = vlen_m if p_side == SIDE_PURE_SIDE else (vw_m * max(0.75, cos_yaw) + 0.28 * vlen_m * abs(math.sin(math.radians(yaw_deg))))
            veh_w_px = fx * span_w_m / z_m
            veh_h_px = fy * vh_m / z_m

            # Screen placement consistent with depth Z (distant near horizon, close-up in foreground)
            depth_t = np.clip((38.0 - min(38.0, z_m)) / 35.8, 0.0, 1.0)
            vy1 = horizon_y + (h_img - horizon_y) * (0.14 + 0.84 * (depth_t ** 1.35)) + rng.uniform(-6.0, 8.0) * s_img
            vy1 = min(float(h_img) + veh_h_px * 0.14, max(float(horizon_y + 10), vy1))
            vy0 = vy1 - veh_h_px
            if vy0 < -veh_h_px * 0.35:
                shift_y = -veh_h_px * 0.15 - vy0
                vy0 += shift_y
                vy1 += shift_y

            # Horizontal lane placement avoiding complete 100% overlap of similar-depth vehicles while allowing partial occlusion
            best_vcx = w_img * rng.uniform(0.12, 0.88)
            for _ in range(5):
                cand_x = w_img * rng.uniform(0.10, 0.90)
                too_close = False
                for ox, ow, oz in occupied_u:
                    if abs(oz - z_m) < 4.5 and abs(cand_x - ox) < 0.65 * (veh_w_px + ow) * 0.5:
                        too_close = True
                        break
                if not too_close:
                    best_vcx = cand_x
                    break
            occupied_u.append((best_vcx, veh_w_px, z_m))

            vx0 = best_vcx - 0.5 * veh_w_px
            vx1 = best_vcx + 0.5 * veh_w_px

            veh_specs.append(
                {
                    "arch": arch,
                    "p_side": p_side,
                    "p_cat": p_cat,
                    "pw_m": pw_m,
                    "ph_m": ph_m,
                    "z_m": z_m,
                    "yaw": yaw_deg,
                    "pitch": pitch_deg,
                    "roll": roll_deg,
                    "vx0": vx0,
                    "vy0": vy0,
                    "vx1": vx1,
                    "vy1": vy1,
                }
            )

        # Painter's algorithm: sort vehicles farthest-first (descending Z_m) -> nearest-last
        veh_specs.sort(key=lambda d: d["z_m"], reverse=True)

        rendered_records: list[dict[str, Any]] = []
        inst_map = np.zeros((h_img, w_img), dtype=np.int16)

        for v_order, spec in enumerate(veh_specs):
            inst_id = v_order + 1
            ix0_v = max(0, int(math.floor(spec["vx0"])))
            iy0_v = max(0, int(math.floor(spec["vy0"])))
            ix1_v = min(w_img, int(math.ceil(spec["vx1"])))
            iy1_v = min(h_img, int(math.ceil(spec["vy1"])))
            prev_sub = canvas[iy0_v:iy1_v, ix0_v:ix1_v].copy() if (ix1_v > ix0_v and iy1_v > iy0_v) else None

            p_anchor = self._draw_vehicle_and_compute_plate(
                canvas,
                archetype=spec["arch"],
                plate_side=spec["p_side"],
                vx0=spec["vx0"],
                vy0=spec["vy0"],
                vx1=spec["vx1"],
                vy1=spec["vy1"],
                weather=weather,
                rng=rng,
            )

            # Add subtle sheet-metal panel-seam / rib texture on wide close-up gate cabs so they never look like flat walls
            if (spec["vx1"] - spec["vx0"]) > 0.58 * float(w_img) and ix1_v - ix0_v >= 32 and iy1_v - iy0_v >= 32:
                fac = self.bank.facade_deltas[inst_id % len(self.bank.facade_deltas)]
                sub_c = canvas[iy0_v:iy1_v, ix0_v:ix1_v]
                ty_m = np.arange(iy1_v - iy0_v) % 256
                tx_m = np.arange(ix1_v - ix0_v) % 256
                rib_delta = (fac[np.ix_(ty_m, tx_m)] // 3)[:, :, None]
                canvas[iy0_v:iy1_v, ix0_v:ix1_v] = np.clip(sub_c.astype(np.int16) + rib_delta, 0, 255).astype(np.uint8)

            plate_corners: np.ndarray | None = None
            plate_entry: _CachedPlateEntry | None = None

            if p_anchor is not None and spec["p_side"] != SIDE_PURE_SIDE:
                px_c, py_c = p_anchor
                vw_sp = max(4.0, spec["vx1"] - spec["vx0"])
                p_est_half = 0.55 * (fx * spec["pw_m"] / max(1.2, spec["z_m"]))
                lo_x = max(p_est_half + 4.0, spec["vx0"] + 0.12 * vw_sp)
                hi_x = min(float(w_img) - p_est_half - 4.0, spec["vx1"] - 0.12 * vw_sp)
                if hi_x > lo_x:
                    px_c = float(np.clip(px_c, lo_x, hi_x))
                else:
                    px_c = float(np.clip(px_c, spec["vx0"] + 0.14 * vw_sp, spec["vx1"] - 0.14 * vw_sp))
                z_p = spec["z_m"]
                tx = (px_c - cx_cam) * z_p / fx
                ty = (py_c - cy_cam) * z_p / fy
                H_3d = _compute_3d_plate_homography(
                    pw_m=spec["pw_m"],
                    ph_m=spec["ph_m"],
                    center_3d=(tx, ty, z_p),
                    yaw_deg=spec["yaw"],
                    pitch_deg=spec["pitch"],
                    roll_deg=spec["roll"],
                    fx=fx,
                    fy=fy,
                    cx=cx_cam,
                    cy=cy_cam,
                )
                plate_corners = cv2.perspectiveTransform(_UNIT_QUAD_4, H_3d).reshape(4, 2).astype(np.float32)

                plate_entry = self.bank.pick_plate(
                    spec["p_cat"],
                    spec["p_side"],
                    rng,
                    p_temporary=self.config.p_temporary_plate,
                )
                w_top_p = float(np.linalg.norm(plate_corners[1] - plate_corners[0]))
                w_bot_p = float(np.linalg.norm(plate_corners[2] - plate_corners[3]))
                pw_proj = max(float(np.max(plate_corners[:, 0]) - np.min(plate_corners[:, 0])), 0.5 * (w_top_p + w_bot_p))
                if pw_proj < 24.0:
                    src_patch = plate_entry.patch_tiny
                elif pw_proj < 48.0:
                    src_patch = plate_entry.patch_quarter
                elif pw_proj < 96.0:
                    src_patch = plate_entry.patch_half
                else:
                    src_patch = plate_entry.patch_full

                # Adjust brightness for night / truck recessed under-bed shadow
                if weather == "night" or (spec["arch"] in ("cargo_truck", "box_truck", "dump_truck") and spec["p_side"] == SIDE_REAR):
                    shade = 0.62 if (spec["arch"] in ("cargo_truck", "box_truck", "dump_truck") and spec["p_side"] == SIDE_REAR) else 0.78
                    src_patch = (src_patch.astype(np.uint16) * int(shade * 256) >> 8).astype(np.uint8)

                _warp_patch_into_canvas(canvas, src_patch, plate_corners)
            else:
                H_3d = None

            if prev_sub is not None:
                painted = np.any(canvas[iy0_v:iy1_v, ix0_v:ix1_v] != prev_sub, axis=-1)
                inst_map[iy0_v:iy1_v, ix0_v:ix1_v][painted] = inst_id
            if plate_corners is not None:
                cv2.fillConvexPoly(inst_map, np.round(plate_corners).astype(np.int32), inst_id)

            rendered_records.append(
                {
                    "spec": spec,
                    "inst_id": inst_id,
                    "H_3d": H_3d,
                    "plate_corners": plate_corners,
                    "plate_entry": plate_entry,
                }
            )

        # Wet-road reflection streaks below foreground vehicles
        if weather == "wet_road":
            for rec in rendered_records:
                sp = rec["spec"]
                rx0 = max(0, int(round(sp["vx0"])))
                rx1 = min(w_img, int(round(sp["vx1"])))
                ry1 = int(round(sp["vy1"]))
                rh_src = min(24, max(4, int((sp["vy1"] - sp["vy0"]) * 0.30)))
                if rx1 - rx0 >= 8 and 0 <= ry1 - rh_src and ry1 + rh_src <= h_img:
                    strip = canvas[ry1 - rh_src : ry1, rx0:rx1][::-1]
                    canvas[ry1 : ry1 + rh_src, rx0:rx1] = cv2.addWeighted(
                        canvas[ry1 : ry1 + rh_src, rx0:rx1], 0.74, cv2.blur(strip, (3, 7)), 0.26, 0.0
                    )

        # Exact Ground Truth with Pixel-Exact Instance Occlusion (`inst_map`) & Frame Clipping
        gt_objects: list[SceneObjectGT] = []

        for rec in rendered_records:
            sp = rec["spec"]
            inst_id = int(rec["inst_id"])
            vx0, vy0, vx1, vy1 = sp["vx0"], sp["vy0"], sp["vx1"], sp["vy1"]

            # 1. Vehicle GT
            gx = vx0 + (vx1 - vx0) * _VEH_GRID_U
            gy = vy0 + (vy1 - vy0) * _VEH_GRID_V
            in_bounds = (gx >= 0.0) & (gx < float(w_img)) & (gy >= 0.0) & (gy < float(h_img))
            vis_mask = in_bounds.copy()
            if np.any(in_bounds):
                gxi = np.clip(gx[in_bounds].astype(np.int32), 0, w_img - 1)
                gyi = np.clip(gy[in_bounds].astype(np.int32), 0, h_img - 1)
                vis_mask[in_bounds] = inst_map[gyi, gxi] <= inst_id
            veh_vis = float(np.mean(vis_mask))

            cvx0 = float(np.clip(vx0, 0.0, float(w_img)))
            cvy0 = float(np.clip(vy0, 0.0, float(h_img)))
            cvx1 = float(np.clip(vx1, 0.0, float(w_img)))
            cvy1 = float(np.clip(vy1, 0.0, float(h_img)))

            if veh_vis >= 0.18 and min(cvx1 - cvx0, cvy1 - cvy0) >= 8.0:
                v_bbox = np.array([cvx0, cvy0, cvx1, cvy1], dtype=np.float32)
                v_corners = np.array(
                    [[cvx0, cvy0], [cvx1, cvy0], [cvx1, cvy1], [cvx0, cvy1]],
                    dtype=np.float32,
                )
                gt_objects.append(
                    SceneObjectGT(
                        cls_id=CLS_VEHICLE,
                        bbox_xyxy=v_bbox,
                        corners_xy=v_corners,
                        plate_side=sp["p_side"],
                        visibility=round(veh_vis, 4),
                        vehicle_archetype=sp["arch"],
                    )
                )

            # 2. Plate GT (emit ONLY if pixel-exact visible fraction >= 0.35 and min(w, h) >= 3.5 px)
            p_corners = rec["plate_corners"]
            H_3d = rec["H_3d"]
            p_entry = rec["plate_entry"]
            if p_corners is not None and H_3d is not None and p_entry is not None:
                pts_32 = cv2.perspectiveTransform(_PLATE_UV_GRID_32, H_3d).reshape(32, 2)
                px_g = pts_32[:, 0]
                py_g = pts_32[:, 1]
                p_in = (px_g >= 0.0) & (px_g < float(w_img)) & (py_g >= 0.0) & (py_g < float(h_img))
                p_vis_mask = p_in.copy()
                if np.any(p_in):
                    pxi = np.clip(px_g[p_in].astype(np.int32), 0, w_img - 1)
                    pyi = np.clip(py_g[p_in].astype(np.int32), 0, h_img - 1)
                    p_vis_mask[p_in] = inst_map[pyi, pxi] <= inst_id
                p_vis = float(np.mean(p_vis_mask))

                pbx0 = float(np.clip(np.min(p_corners[:, 0]), 0.0, float(w_img)))
                pby0 = float(np.clip(np.min(p_corners[:, 1]), 0.0, float(h_img)))
                pbx1 = float(np.clip(np.max(p_corners[:, 0]), 0.0, float(w_img)))
                pby1 = float(np.clip(np.max(p_corners[:, 1]), 0.0, float(h_img)))
                pbw = pbx1 - pbx0
                pbh = pby1 - pby0

                if p_vis >= self.config.min_plate_visibility and min(pbw, pbh) >= self.config.min_plate_side_px:
                    gt_objects.append(
                        SceneObjectGT(
                            cls_id=CLS_PLATE,
                            bbox_xyxy=np.array([pbx0, pby0, pbx1, pby1], dtype=np.float32),
                            corners_xy=p_corners.copy(),
                            plate_side=sp["p_side"],
                            visibility=round(p_vis, 4),
                            vehicle_archetype=sp["arch"],
                            plate_text_str=p_entry.plate_text_str,
                            plate_type_str=p_entry.plate_type_str,
                        )
                    )

        return SynthSceneSample(
            image_rgb=canvas,
            objects=gt_objects,
            scene_tag=f"{scene_family}_{weather}",
            camera_height_m=round(cam_height_m, 2),
        )


@dataclass
class QuadCropSynthConfig:
    """Configuration for `QuadCropSynthesizer`."""

    p_positive: float = 0.48          # Balanced 48% positive plate ROIs, 52% diverse hard-negative traps
    p_lowres_distant: float = 0.42    # distant 12x6..30x15 low-res degraded plates upsampled to (96, 160)
    p_temporary_plate: float = 0.04
    plate_bank_size: int = 1080
    domain_split: str = "train"


class QuadCropSynthesizer:
    """Ultra-fast (< 0.8 ms/sample) `(96, 160, 3)` noisy ROI -> tight 4-corner + verifier sample generator."""

    STRUCTURED_HARD_NEG_MODES: tuple[str, ...] = (
        "blue_highway_route_sign",
        "frameless_truck_body_text",
        "bordered_payload_and_speed_decals",
        "shop_parking_multiline_sign",
        "grille_emblem_bumper_slats",
        "building_window_grid_and_reflectors",
    )

    HARD_NEG_MODES: tuple[str, ...] = (
        "radiator_grille",
        "headlight_taillight",
        "truck_sticker",
        "truck_bed_fleet_number",
        "shop_parking_sign",
        "fence_safety_banner",
        "storefront_awning_crowd",
        "dual_truck_wheels",
        "cracked_gravel_asphalt",
        "vehicle_windshield_cabin",
        "windshield_wipers_dual",
        "mixer_drum_stripe",
        "orange_safety_fence",
        "steel_ramp_plates",
        "bollard_cone_row",
        "natural_1f_clutter_crop",
        "crosswalk_stripes",
        "building_windows",
        "bumper_vent_bar",
        "dump_truck_reflector_latch",
        "gate_barrier_arm",
        "taxi_roof_placard",
        "road_text",
    )

    def __init__(self, config: QuadCropSynthConfig | None = None) -> None:
        self.config = config or QuadCropSynthConfig()
        if self.config.domain_split not in {"train", "val_holdout", "cal_holdout"}:
            raise ValueError("Unknown synthetic domain split")
        self._bank: _SceneAssetBank | None = None
        self._scene_helper: FullSceneDetSynthesizer | None = None

    @property
    def bank(self) -> _SceneAssetBank:
        if self._bank is None:
            self._bank = _SceneAssetBank.get(self.config.plate_bank_size)
        return self._bank

    @property
    def _bg_helper(self) -> FullSceneDetSynthesizer:
        if self._scene_helper is None:
            self._scene_helper = FullSceneDetSynthesizer(
                SceneSynthConfig(image_size=(DUAL_CANVAS_H, DUAL_CANVAS_W), plate_bank_size=self.config.plate_bank_size)
            )
        return self._scene_helper

    def _apply_roi_degradation(
        self,
        roi: np.ndarray,
        rng: random.Random,
        force_lowres: bool = False,
        plate_occ_scale: float = 1.0,
    ) -> float:
        """Apply camera low-res capture + blur + H.264 DCT/4:2:0 or JPEG artifacts + glare/night."""
        qual = 1.0
        h, w = roi.shape[:2]

        if self.config.domain_split == "train":
            # 45% coupled pre-resize 12..30 px + YCbCr 4:2:0 + H.264 8x8 DCT degradation
            if force_lowres or rng.random() < 0.45:
                curr_px = max(16.0, 84.0 * float(plate_occ_scale))
                q_lr, _ = _apply_coupled_lowres_dct_420_np(roi, rng, curr_obj_px=curr_px, target_px_range=(12.0, 30.0))
                qual *= q_lr
            elif rng.random() < 0.28:
                k_sz = rng.choice([3, 5])
                roi[:] = cv2.GaussianBlur(roi, (k_sz, k_sz), rng.uniform(0.6, 1.3))
                qual *= 0.86
        else:
            # Holdout-invariant legacy path (`val_holdout` & `cal_holdout`)
            if force_lowres or rng.random() < self.config.p_lowres_distant:
                min_lw = max(24, min(58, int(round(24.0 / max(0.45, min(1.0, plate_occ_scale))))))
                lw = rng.randint(min_lw, max(min_lw + 4, 66))
                lh = max(14, int(round(lw * (h / float(w)) * rng.uniform(0.88, 1.12))))
                small = cv2.resize(roi, (lw, lh), interpolation=cv2.INTER_AREA)
                if rng.random() < 0.65:
                    small = cv2.GaussianBlur(small, (3, 3), rng.uniform(0.55, 1.15))
                if rng.random() < 0.55:
                    jq = rng.randint(28, 66)
                    ok, enc = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), jq])
                    if ok:
                        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                        if dec is not None:
                            small = dec
                up_mode = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST])
                roi[:] = cv2.resize(small, (w, h), interpolation=up_mode)
                qual *= max(0.32, min(0.85, lw / 68.0))
            elif rng.random() < 0.30:
                k_sz = rng.choice([3, 5])
                roi[:] = cv2.GaussianBlur(roi, (k_sz, k_sz), rng.uniform(0.6, 1.3))
                qual *= 0.86

        # Photometric / Night / Glare
        r_light = rng.random()
        if r_light < 0.14:
            # Monochrome Night-IR
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            roi[:, :, 0] = gray
            roi[:, :, 1] = gray
            roi[:, :, 2] = gray
            qual *= 0.90
        elif r_light < 0.28:
            # Dark night / underexposed shadow
            gain = rng.uniform(0.38, 0.68)
            roi[:] = (roi.astype(np.uint16) * int(gain * 256) >> 8).astype(np.uint8)
            qual *= 0.85

        # Specular headlight / sun glare spot
        if rng.random() < 0.20:
            gx = rng.randint(20, w - 20)
            gy = rng.randint(15, h - 15)
            gr = rng.randint(14, 34)
            glare = np.zeros((h, w), dtype=np.float32)
            cv2.circle(glare, (gx, gy), gr, rng.uniform(0.45, 0.82), -1, lineType=cv2.LINE_AA)
            glare = cv2.GaussianBlur(glare, (15, 15), 6.0)[:, :, None]
            roi[:] = np.clip(roi.astype(np.float32) * (1.0 - 0.75 * glare) + 250.0 * glare, 0.0, 255.0).astype(np.uint8)
            qual *= 0.82

        # Interlace field-combing / vertical double-ghosting (~12%, symmetric across positives & negatives)
        r_comb = rng.random()
        if r_comb < 0.06:
            dx_c = rng.randint(1, 3)
            roi[1::2, dx_c:] = roi[1::2, :-dx_c]
            qual *= 0.80
        elif r_comb < 0.12:
            dy_g = rng.randint(2, 4)
            alpha_g = rng.uniform(0.24, 0.38)
            shifted = np.empty_like(roi)
            shifted[:dy_g] = roi[:dy_g]
            shifted[dy_g:] = roi[:-dy_g]
            roi[:] = np.clip(roi.astype(np.float32) * (1.0 - alpha_g) + shifted.astype(np.float32) * alpha_g, 0.0, 255.0).astype(np.uint8)
            qual *= 0.78

        return qual

    def _synthesize_hard_negative(self, rng: random.Random) -> QuadCropTrainingSample:
        """Synthesize realistic non-plate false-positive trap ROI (`is_plate = 0.0`)."""
        roi = np.empty((QUAD_ROI_H, QUAD_ROI_W, 3), dtype=np.uint8)
        if self.config.domain_split == "train" and rng.random() < 0.76:
            mode = rng.choice(self.STRUCTURED_HARD_NEG_MODES)
        else:
            mode = rng.choice(self.HARD_NEG_MODES)
        bg_body = rng.choice(FullSceneDetSynthesizer.BODY_COLORS)
        roi[:] = (int(bg_body[0] * 0.75), int(bg_body[1] * 0.75), int(bg_body[2] * 0.75))

        if mode == "blue_highway_route_sign":
            # Sky / highway gantry truss + MLIT blue route sign ("319", "246", "16", "R17")
            roi[:] = rng.choice([(148, 178, 214), (168, 176, 186), (26, 32, 46)])
            px_pole = rng.randint(68, 92)
            cv2.rectangle(roi, (px_pole - 4, 0), (px_pole + 4, QUAD_ROI_H), (142, 148, 156), -1)
            bp = rng.choice(self.bank.blue_route_patches)
            dst_q = np.array(
                [
                    [rng.uniform(22, 44), rng.uniform(10, 22)],
                    [rng.uniform(116, 138), rng.uniform(10, 22)],
                    [rng.uniform(116, 140), rng.uniform(72, 86)],
                    [rng.uniform(20, 44), rng.uniform(72, 86)],
                ],
                dtype=np.float32,
            )
            _warp_patch_into_canvas(roi, bp, dst_q)

        elif mode == "frameless_truck_body_text":
            # Colored/ribbed truck metal body + frameless painted text ("スケルトン車", "最大積載量 2000kg", "産廃収集運搬車", "自家用")
            metal_c = rng.choice([(226, 230, 236), (196, 202, 210), (42, 84, 142), (48, 96, 68), (224, 182, 40), (46, 50, 58)])
            roi[:] = metal_c
            rib_dark = (max(12, metal_c[0] - 28), max(12, metal_c[1] - 28), max(14, metal_c[2] - 28))
            rib_hi = (min(252, metal_c[0] + 22), min(252, metal_c[1] + 22), min(252, metal_c[2] + 22))
            for ry_rib in range(rng.randint(8, 16), QUAD_ROI_H - 6, rng.randint(14, 22)):
                cv2.line(roi, (0, ry_rib), (QUAD_ROI_W, ry_rib), rib_dark, 2)
                cv2.line(roi, (0, ry_rib + 2), (QUAD_ROI_W, ry_rib + 2), rib_hi, 1)
            lum_m = 0.299 * metal_c[0] + 0.587 * metal_c[1] + 0.114 * metal_c[2]
            cand_rgba = [
                st for st in self.bank.frameless_decal_rgba
                if abs((0.299 * float(st[0, 0, 0]) + 0.587 * float(st[0, 0, 1]) + 0.114 * float(st[0, 0, 2])) - lum_m) > 75.0
            ]
            st_rgba = rng.choice(cand_rgba if cand_rgba else self.bank.frameless_decal_rgba)
            dst_q = np.array(
                [
                    [rng.uniform(10, 28), rng.uniform(24, 36)],
                    [rng.uniform(132, 150), rng.uniform(22, 36)],
                    [rng.uniform(132, 150), rng.uniform(58, 72)],
                    [rng.uniform(10, 28), rng.uniform(58, 72)],
                ],
                dtype=np.float32,
            )
            _warp_rgba_patch_into_canvas(roi, st_rgba, dst_q, opacity=rng.uniform(0.84, 0.98))

        elif mode == "bordered_payload_and_speed_decals":
            roi[:] = rng.choice([(208, 214, 220), (46, 50, 58), (52, 88, 134), (56, 96, 72)])
            cv2.line(roi, (0, rng.randint(68, 84)), (QUAD_ROI_W, rng.randint(68, 84)), (204, 44, 40), 3)
            dc = rng.choice(self.bank.bordered_decal_patches)
            dst_q = np.array(
                [
                    [rng.uniform(14, 36), rng.uniform(20, 34)],
                    [rng.uniform(122, 146), rng.uniform(20, 34)],
                    [rng.uniform(122, 146), rng.uniform(60, 76)],
                    [rng.uniform(14, 36), rng.uniform(60, 76)],
                ],
                dtype=np.float32,
            )
            _warp_patch_into_canvas(roi, dc, dst_q)

        elif mode == "shop_parking_multiline_sign":
            roi[:] = rng.choice([(164, 168, 174), (124, 130, 138), (192, 186, 176), (34, 38, 48)])
            mp = rng.choice(self.bank.multiline_parking_patches)
            dst_q = np.array(
                [
                    [rng.uniform(12, 32), rng.uniform(10, 22)],
                    [rng.uniform(128, 148), rng.uniform(10, 22)],
                    [rng.uniform(128, 148), rng.uniform(72, 86)],
                    [rng.uniform(12, 32), rng.uniform(72, 86)],
                ],
                dtype=np.float32,
            )
            _warp_patch_into_canvas(roi, mp, dst_q)

        elif mode == "grille_emblem_bumper_slats":
            roi[:] = bg_body
            gx0, gy0 = rng.randint(6, 22), rng.randint(8, 22)
            gx1, gy1 = QUAD_ROI_W - rng.randint(6, 22), QUAD_ROI_H - rng.randint(8, 22)
            cv2.rectangle(roi, (gx0, gy0), (gx1, gy1), (24, 26, 30), -1)
            cv2.rectangle(roi, (gx0, gy0), (gx1, gy1), (198, 204, 212), 2)
            step = rng.choice([6, 8, 10])
            for y_s in range(gy0 + step, gy1 - 2, step):
                cv2.line(roi, (gx0 + 2, y_s), (gx1 - 2, y_s), (168, 174, 184), rng.choice([1, 2]))
            cx_e, cy_e = (gx0 + gx1) // 2, (gy0 + gy1) // 2
            cv2.ellipse(roi, (cx_e, cy_e), (rng.randint(12, 20), rng.randint(8, 13)), 0, 0, 360, (218, 224, 232), 2, lineType=cv2.LINE_AA)

        elif mode == "building_window_grid_and_reflectors":
            if rng.random() < 0.58:
                is_night_w = rng.random() < 0.55
                roi[:] = (36, 40, 50) if is_night_w else rng.choice([(176, 172, 166), (144, 148, 156)])
                ww, wh = rng.randint(22, 38), rng.randint(14, 24)
                for y_w in range(8, QUAD_ROI_H - wh, wh + 8):
                    for x_w in range(10, QUAD_ROI_W - ww, ww + 10):
                        lit = is_night_w and (rng.random() < 0.62)
                        pane_c = rng.choice([(242, 224, 158), (218, 234, 248)]) if lit else (42, 52, 64)
                        cv2.rectangle(roi, (x_w, y_w), (x_w + ww, y_w + wh), pane_c, -1)
                        cv2.rectangle(roi, (x_w, y_w), (x_w + ww, y_w + wh), (206, 210, 216), 1)
            else:
                roi[:] = rng.choice([(46, 52, 60), (74, 78, 84), (168, 174, 180)])
                ry_g = rng.randint(32, 52)
                cv2.rectangle(roi, (0, ry_g), (QUAD_ROI_W, ry_g + 16), (218, 222, 228), -1)
                for rx_r in range(18, QUAD_ROI_W - 22, rng.randint(34, 52)):
                    ref_c = (238, 122, 26) if rng.random() < 0.55 else (214, 36, 40)
                    cv2.rectangle(roi, (rx_r, ry_g + 2), (rx_r + 22, ry_g + 14), ref_c, -1)
                    cv2.rectangle(roi, (rx_r, ry_g + 2), (rx_r + 22, ry_g + 14), (244, 246, 248), 1)

        elif mode == "radiator_grille":
            gx0, gy0 = rng.randint(8, 28), rng.randint(10, 24)
            gx1, gy1 = QUAD_ROI_W - rng.randint(8, 28), QUAD_ROI_H - rng.randint(10, 24)
            cv2.rectangle(roi, (gx0, gy0), (gx1, gy1), (26, 28, 32), -1)
            cv2.rectangle(roi, (gx0, gy0), (gx1, gy1), (196, 202, 210), rng.randint(2, 4))
            step = rng.choice([6, 8, 10, 12])
            for y in range(gy0 + step, gy1 - 2, step):
                cv2.line(roi, (gx0 + 3, y), (gx1 - 3, y), (168, 174, 182), rng.choice([1, 2]))
            if rng.random() < 0.5:
                for x in range(gx0 + 12, gx1 - 6, rng.choice([14, 20])):
                    cv2.line(roi, (x, gy0 + 2), (x, gy1 - 2), (68, 72, 78), 1)

        elif mode == "headlight_taillight":
            is_tail = rng.random() < 0.5
            lx0, ly0 = rng.randint(14, 34), rng.randint(18, 30)
            lx1, ly1 = QUAD_ROI_W - rng.randint(14, 34), QUAD_ROI_H - rng.randint(18, 30)
            outer_c = (198, 34, 38) if is_tail else (218, 228, 242)
            cv2.rectangle(roi, (lx0, ly0), (lx1, ly1), outer_c, -1)
            cv2.rectangle(roi, (lx0, ly0), (lx1, ly1), (160, 166, 174), 2)
            if is_tail:
                mid = lx0 + (lx1 - lx0) // 3
                cv2.rectangle(roi, (lx0 + 2, ly0 + 2), (mid, ly1 - 2), (238, 152, 28), -1)
                cv2.rectangle(roi, (mid + 2, ly0 + (ly1 - ly0) // 2), (lx1 - 2, ly1 - 2), (238, 240, 242), -1)
            else:
                for cx_f in (0.32, 0.68):
                    cx_l = int(lx0 + (lx1 - lx0) * cx_f)
                    cy_l = (ly0 + ly1) // 2
                    cv2.circle(roi, (cx_l, cy_l), max(6, (ly1 - ly0) // 3), (250, 252, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(roi, (cx_l, cy_l), max(6, (ly1 - ly0) // 3), (120, 132, 148), 2, lineType=cv2.LINE_AA)

        elif mode == "truck_sticker":
            roi[:] = rng.choice([(196, 202, 208), (44, 48, 54), (58, 98, 142)])
            st = rng.choice(self.bank.sticker_patches)
            dst_q = np.array(
                [
                    [rng.uniform(14, 36), rng.uniform(22, 36)],
                    [rng.uniform(122, 146), rng.uniform(20, 36)],
                    [rng.uniform(122, 148), rng.uniform(58, 76)],
                    [rng.uniform(12, 36), rng.uniform(58, 76)],
                ],
                dtype=np.float32,
            )
            _warp_patch_into_canvas(roi, st, dst_q)

        elif mode == "shop_parking_sign":
            roi[:] = rng.choice([(168, 172, 178), (128, 134, 142), (192, 186, 176)])
            sp = rng.choice(self.bank.sign_patches)
            dst_q = np.array(
                [
                    [rng.uniform(10, 36), rng.uniform(12, 28)],
                    [rng.uniform(124, 150), rng.uniform(12, 28)],
                    [rng.uniform(124, 150), rng.uniform(66, 84)],
                    [rng.uniform(10, 36), rng.uniform(66, 84)],
                ],
                dtype=np.float32,
            )
            _warp_patch_into_canvas(roi, sp, dst_q)

        elif mode == "crosswalk_stripes":
            roi[:] = rng.choice([(68, 70, 74), (86, 88, 92)])
            slant = rng.randint(-18, 18)
            for x in range(8, QUAD_ROI_W - 8, rng.randint(22, 32)):
                pts = np.array(
                    [[x, 12], [x + 13, 12], [x + 13 + slant, QUAD_ROI_H - 12], [x + slant, QUAD_ROI_H - 12]],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(roi, pts, (218, 220, 224), lineType=cv2.LINE_AA)

        elif mode == "building_windows":
            roi[:] = rng.choice([(182, 178, 172), (148, 152, 158)])
            ww, wh = rng.randint(24, 42), rng.randint(16, 28)
            for y in range(10, QUAD_ROI_H - wh, wh + 10):
                for x in range(12, QUAD_ROI_W - ww, ww + 12):
                    cv2.rectangle(roi, (x, y), (x + ww, y + wh), (42, 52, 64), -1)
                    cv2.rectangle(roi, (x, y), (x + ww, y + wh), (218, 222, 226), 2)

        elif mode == "bumper_vent_bar":
            # Truck steel underrun bar + dark chassis vent
            roi[:] = (28, 30, 34)
            by = rng.randint(32, 52)
            cv2.rectangle(roi, (8, by), (QUAD_ROI_W - 8, by + rng.randint(12, 20)), (158, 164, 172), -1)
            cv2.rectangle(roi, (18, 8), (QUAD_ROI_W - 18, by - 6), (48, 52, 58), 2)

        elif mode == "truck_bed_fleet_number":
            # Large painted `春日 2503` / `東都 616` fleet number on dump/cargo truck bed (NOT a plate!)
            roi[:] = rng.choice([(44, 68, 118), (186, 192, 198), (58, 96, 76), (62, 66, 72)])
            cv2.line(roi, (0, rng.randint(12, 24)), (QUAD_ROI_W, rng.randint(12, 24)), (32, 36, 40), 3)
            cv2.line(roi, (0, rng.randint(72, 86)), (QUAD_ROI_W, rng.randint(72, 86)), (188, 172, 92), 3)
            bn = rng.choice(self.bank.bed_num_patches)
            dst_q = np.array(
                [
                    [rng.uniform(8, 32), rng.uniform(24, 36)],
                    [rng.uniform(128, 152), rng.uniform(24, 36)],
                    [rng.uniform(128, 152), rng.uniform(58, 72)],
                    [rng.uniform(8, 32), rng.uniform(58, 72)],
                ],
                dtype=np.float32,
            )
            _warp_patch_into_canvas(roi, bn, dst_q)

        elif mode == "fence_safety_banner":
            # Parking fence slats + wide rectangular notice banner (`駐車場内での事故...`)
            roi[:] = rng.choice([(132, 136, 142), (96, 102, 108)])
            for x in range(4, QUAD_ROI_W, rng.randint(10, 16)):
                cv2.line(roi, (x, 0), (x, QUAD_ROI_H), (198, 202, 208), 2)
            bp = rng.choice(self.bank.banner_patches)
            dst_q = np.array(
                [
                    [rng.uniform(8, 26), rng.uniform(22, 34)],
                    [rng.uniform(134, 154), rng.uniform(24, 36)],
                    [rng.uniform(134, 154), rng.uniform(60, 74)],
                    [rng.uniform(8, 26), rng.uniform(58, 72)],
                ],
                dtype=np.float32,
            )
            _warp_patch_into_canvas(roi, bp, dst_q)

        elif mode == "storefront_awning_crowd":
            # Striped shop awning + storefront sign + vertical mullions + pedestrian heads/shoulders
            roi[:] = (168, 156, 142)
            bp = rng.choice(self.bank.banner_patches)
            roi[4:24, 10:150] = cv2.resize(bp, (140, 20), interpolation=cv2.INTER_AREA)
            for ax in range(6, 154, 10):
                cv2.rectangle(roi, (ax, 24), (ax + 5, 44), (204, 68, 74), -1)
                cv2.rectangle(roi, (ax + 5, 24), (ax + 10, 44), (240, 236, 230), -1)
            cv2.rectangle(roi, (12, 44), (148, 92), (188, 138, 86), -1)
            for _ in range(rng.randint(3, 6)):
                px = rng.randint(18, 142)
                py = rng.randint(58, 74)
                cv2.circle(roi, (px, py), rng.randint(6, 10), (32, 28, 26), -1)
                cv2.ellipse(roi, (px, py + 18), (rng.randint(10, 16), 14), 0, 0, 360, rng.choice([(42, 46, 54), (218, 214, 208), (156, 48, 52)]), -1)

        elif mode == "dual_truck_wheels":
            # Dark wheel arch + dual tandem truck tires + metallic rim + lug nut ring
            roi[:] = (58, 62, 68)
            cv2.rectangle(roi, (0, 0), (QUAD_ROI_W, 24), (34, 38, 44), -1)
            for wx in (48, 112):
                wy = rng.randint(48, 58)
                wr = rng.randint(24, 32)
                cv2.circle(roi, (wx, wy), wr, (22, 24, 28), -1, lineType=cv2.LINE_AA)
                rr = int(wr * 0.56)
                cv2.circle(roi, (wx, wy), rr, (192, 198, 204), -1, lineType=cv2.LINE_AA)
                cv2.circle(roi, (wx, wy), int(rr * 0.38), (36, 38, 42), -1, lineType=cv2.LINE_AA)
                for k_a in range(6):
                    ang = k_a * (math.pi / 3.0)
                    cv2.circle(roi, (int(wx + rr * 0.68 * math.cos(ang)), int(wy + rr * 0.68 * math.sin(ang))), 2, (28, 30, 34), -1)

        elif mode == "cracked_gravel_asphalt":
            # High-contrast granular asphalt + tar cracks + partial lane stripe / manhole
            base_g = rng.randint(56, 108)
            roi[:] = (base_g, base_g + 2, base_g + 5)
            asp = self.bank.asphalt_deltas[rng.randrange(len(self.bank.asphalt_deltas))]
            ox, oy = rng.randrange(96), rng.randrange(160)
            roi[:] = np.clip(roi.astype(np.int16) + (asp[oy : oy + QUAD_ROI_H, ox : ox + QUAD_ROI_W, None] * 1.4).astype(np.int16), 0, 255).astype(np.uint8)
            if rng.random() < 0.6:
                y_l = rng.randint(24, 72)
                cv2.line(roi, (rng.randint(0, 30), y_l), (rng.randint(120, 160), y_l + rng.randint(-14, 14)), (214, 216, 218), rng.randint(4, 9))

        elif mode == "vehicle_windshield_cabin":
            # Sloped dark windshield + sky reflection band + A-pillars + wiper
            roi[:] = rng.choice(FullSceneDetSynthesizer.BODY_COLORS)
            pts = np.array([[22, 14], [138, 14], [152, 82], [8, 82]], dtype=np.int32)
            cv2.fillConvexPoly(roi, pts, (38, 48, 60), lineType=cv2.LINE_AA)
            ref_pts = np.array([[34, 18], [92, 18], [76, 78], [20, 78]], dtype=np.int32)
            cv2.fillConvexPoly(roi, ref_pts, (148, 168, 188), lineType=cv2.LINE_AA)
            cv2.line(roi, (24, 78), (118, 66), (22, 24, 28), 3, lineType=cv2.LINE_AA)

        elif mode == "bollard_cone_row":
            # Night/day row of reflective road bollards / traffic cones
            is_night = rng.random() < 0.65
            roi[:] = (22, 26, 34) if is_night else (78, 82, 88)
            for bx in range(18, QUAD_ROI_W - 14, rng.randint(18, 28)):
                bh_p = rng.randint(32, 54)
                by0 = rng.randint(24, 42)
                cv2.rectangle(roi, (bx, by0), (bx + 8, min(QUAD_ROI_H - 4, by0 + bh_p)), (226, 78, 32), -1)
                cv2.rectangle(roi, (bx, by0 + 8), (bx + 8, by0 + 15), (244, 246, 248), -1)
                cv2.rectangle(roi, (bx, by0 + 22), (bx + 8, by0 + 28), (244, 246, 248), -1)

        elif mode == "natural_1f_clutter_crop":
            # Arbitrary high-contrast overlapping rectangles, circles, and granular clutter (prevents texture shortcut!)
            asp = self.bank.asphalt_deltas[rng.randrange(len(self.bank.asphalt_deltas))]
            ox, oy = rng.randrange(96), rng.randrange(160)
            roi[:] = np.clip(roi.astype(np.int16) + asp[oy : oy + QUAD_ROI_H, ox : ox + QUAD_ROI_W, None], 0, 255).astype(np.uint8)
            for _ in range(rng.randint(3, 7)):
                c_rand = (rng.randint(20, 240), rng.randint(20, 240), rng.randint(20, 240))
                if rng.random() < 0.5:
                    x0_r, y0_r = rng.randint(0, 120), rng.randint(0, 65)
                    cv2.rectangle(roi, (x0_r, y0_r), (x0_r + rng.randint(14, 60), y0_r + rng.randint(10, 40)), c_rand, rng.choice([-1, 2]))
                else:
                    cv2.circle(roi, (rng.randint(15, 145), rng.randint(15, 80)), rng.randint(8, 26), c_rand, rng.choice([-1, 2]))

        elif mode == "dump_truck_reflector_latch":
            # Dump/cargo truck steel tailgate + rectangular orange/red reflector + stainless latch/hinge
            roi[:] = rng.choice([(46, 74, 124), (58, 98, 78), (178, 184, 192), (52, 56, 62)])
            rx0_r, ry0_r = rng.randint(24, 48), rng.randint(24, 38)
            rx1_r, ry1_r = rx0_r + rng.randint(52, 86), ry0_r + rng.randint(24, 38)
            ref_col = (238, 118, 24) if rng.random() < 0.6 else (212, 34, 38)
            cv2.rectangle(roi, (rx0_r, ry0_r), (rx1_r, ry1_r), ref_col, -1)
            cv2.rectangle(roi, (rx0_r, ry0_r), (rx1_r, ry1_r), (222, 226, 232), 2)
            # Vertical metallic tailgate latch bar
            lx_b = rng.choice([max(8, rx0_r - 12), min(QUAD_ROI_W - 14, rx1_r + 6)])
            cv2.rectangle(roi, (lx_b, 8), (lx_b + 8, QUAD_ROI_H - 8), (196, 202, 210), -1)

        elif mode == "gate_barrier_arm":
            # Construction-site / toll-gate yellow-black or red-white striped boom barrier arm
            roi[:] = rng.choice([(92, 98, 106), (52, 58, 66), (168, 174, 182)])
            ay0 = rng.randint(28, 46)
            ah = rng.randint(14, 24)
            c_hi = (238, 196, 28) if rng.random() < 0.55 else (238, 240, 244)
            c_lo = (26, 28, 32) if c_hi[1] < 220 else (214, 42, 46)
            cv2.rectangle(roi, (4, ay0), (QUAD_ROI_W - 4, ay0 + ah), c_hi, -1)
            for sx_b in range(0, QUAD_ROI_W, 24):
                pts_b = np.array(
                    [[sx_b, ay0 + ah], [sx_b + 12, ay0 + ah], [sx_b + 20, ay0], [sx_b + 8, ay0]],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(roi, pts_b, c_lo, lineType=cv2.LINE_AA)

        elif mode == "taxi_roof_placard":
            # Glowing taxi roof lantern / truck speed-indicator lamp housing (NOT a license plate!)
            is_night_p = rng.random() < 0.65
            roi[:] = (20, 24, 32) if is_night_p else (138, 148, 162)
            cv2.rectangle(roi, (14, 62), (QUAD_ROI_W - 14, QUAD_ROI_H), rng.choice(FullSceneDetSynthesizer.BODY_COLORS), -1)
            px0_p, py0_p = rng.randint(36, 52), rng.randint(24, 36)
            px1_p, py1_p = QUAD_ROI_W - px0_p, py0_p + rng.randint(22, 30)
            glow_c = rng.choice([(248, 244, 212), (242, 186, 56), (92, 236, 128)])
            cv2.rectangle(roi, (px0_p, py0_p), (px1_p, py1_p), glow_c, -1)
            cv2.rectangle(roi, (px0_p, py0_p), (px1_p, py1_p), (38, 42, 48), 2)

        elif mode == "windshield_wipers_dual":
            # Wet/dusty truck cab windshield base + dual black wiper arms/blades + cowl vents (NOT a plate!)
            roi[:] = rng.choice(FullSceneDetSynthesizer.BODY_COLORS)
            cv2.rectangle(roi, (0, 0), (QUAD_ROI_W, 66), (44, 56, 68), -1)
            cv2.rectangle(roi, (12, 8), (94, 56), (126, 148, 168), -1)
            for wx0 in (18, 86):
                ang_y = rng.randint(-12, 8)
                cv2.line(roi, (wx0, 58), (wx0 + rng.randint(46, 64), 52 + ang_y), (18, 20, 24), rng.randint(3, 6), lineType=cv2.LINE_AA)
                cv2.line(roi, (wx0 + 12, 68), (wx0 + 28, 56), (32, 36, 40), 3, lineType=cv2.LINE_AA)

        elif mode == "mixer_drum_stripe":
            # Curved ready-mix concrete drum with diagonal blue/green/yellow livery band + specular ridge
            roi[:] = rng.choice([(218, 222, 226), (192, 198, 204), (232, 230, 222)])
            stripe_c = rng.choice([(36, 118, 188), (44, 148, 88), (218, 158, 32), (188, 44, 48)])
            y_l = rng.randint(18, 52)
            y_r = y_l + rng.randint(-24, 26)
            bh_m = rng.randint(16, 30)
            pts_m = np.array([[0, y_l], [QUAD_ROI_W, y_r], [QUAD_ROI_W, y_r + bh_m], [0, y_l + bh_m]], dtype=np.int32)
            cv2.fillConvexPoly(roi, pts_m, stripe_c, lineType=cv2.LINE_AA)
            cv2.line(roi, (0, y_l - 4), (QUAD_ROI_W, y_r - 4), (248, 250, 252), 2, lineType=cv2.LINE_AA)

        elif mode == "orange_safety_fence":
            # Construction-site orange plastic safety mesh fence + steel scaffolding pipes
            roi[:] = rng.choice([(98, 102, 106), (72, 76, 82), (138, 142, 146)])
            fy0, fy1 = rng.randint(12, 26), QUAD_ROI_H - rng.randint(10, 24)
            cv2.rectangle(roi, (6, fy0), (QUAD_ROI_W - 6, fy1), (228, 92, 26), 2)
            for gx_f in range(10, QUAD_ROI_W - 8, rng.randint(8, 13)):
                cv2.line(roi, (gx_f, fy0), (gx_f, fy1), (234, 104, 32), 1)
            for gy_f in range(fy0 + 4, fy1, rng.randint(7, 11)):
                cv2.line(roi, (6, gy_f), (QUAD_ROI_W - 6, gy_f), (234, 104, 32), 1)

        elif mode == "steel_ramp_plates":
            # Demolition / quarry steel road deck plates, H-beams, and tire mud tracks
            roi[:] = (76, 72, 68)
            for ry_s in range(14, QUAD_ROI_H - 10, rng.randint(14, 22)):
                cv2.line(roi, (0, ry_s), (QUAD_ROI_W, ry_s + rng.randint(-5, 5)), (44, 42, 40), rng.randint(2, 4))
            cv2.rectangle(roi, (rng.randint(18, 42), rng.randint(24, 40)), (rng.randint(112, 144), rng.randint(58, 76)), (106, 98, 90), 2)

        else:  # road_text
            roi[:] = (74, 76, 80)
            rm = rng.choice(self.bank.road_masks)
            patch = np.full((rm.shape[0], rm.shape[1], 3), 74, dtype=np.uint8)
            col = (224, 182, 46) if rng.random() < 0.4 else (218, 220, 222)
            a = (rm.astype(np.float32) / 255.0)[:, :, None] * 0.85
            patch[:] = (74.0 * (1.0 - a) + np.array(col, dtype=np.float32) * a).astype(np.uint8)
            dst_q = np.array([[22.0, 20.0], [138.0, 18.0], [146.0, 78.0], [14.0, 76.0]], dtype=np.float32)
            _warp_patch_into_canvas(roi, patch, dst_q)

        # Match the empirical positive `occ_scale` marginal so low-res blur (`lw`) has ZERO class-conditional shortcut!
        neg_occ = rng.uniform(0.48, 0.76) if rng.random() < 0.46 else rng.uniform(0.76, 1.05)
        self._apply_roi_degradation(roi, rng, force_lowres=False, plate_occ_scale=neg_occ)
        # Apply symmetric constant-border framing (30% of negatives, matching positives & extract_expanded_plate_roi!)
        # Crucial invariant: 4-sided `expand_scale=1.85` shrunk inner patches with `(36, 40, 46)` or perimeter-clamped
        # borders appear at identical probability on negatives and positives so constant borders carry zero shortcut!
        if rng.random() < 0.30:
            if rng.random() < 0.62:
                ar_neg = rng.uniform(0.75, 2.50)
                exp_s = rng.uniform(1.55, 2.05)
                target_ar = float(QUAD_ROI_W) / float(QUAD_ROI_H)
                rw_u = max(ar_neg * exp_s, 1.0 * exp_s * target_ar)
                rh_u = rw_u / target_ar
                iw_n = max(36, min(QUAD_ROI_W - 20, int(round(float(QUAD_ROI_W) * ar_neg / rw_u))))
                ih_n = max(24, min(QUAD_ROI_H - 14, int(round(float(QUAD_ROI_H) * 1.0 / rh_u))))
                bx0_n = max(2, min(QUAD_ROI_W - iw_n - 2, (QUAD_ROI_W - iw_n) // 2 + rng.randint(-6, 6)))
                by0_n = max(2, min(QUAD_ROI_H - ih_n - 2, (QUAD_ROI_H - ih_n) // 2 + rng.randint(-4, 4)))
                inner_neg = cv2.resize(roi, (iw_n, ih_n), interpolation=cv2.INTER_AREA)
                c_pad = self._sample_constant_surround_rgb(inner_neg, rng)
                roi[:] = c_pad
                roi[by0_n : by0_n + ih_n, bx0_n : bx0_n + iw_n] = inner_neg
            else:
                c_pad = self._sample_constant_surround_rgb(roi, rng)
                for side_k in range(4):
                    if rng.random() < 0.50:
                        if side_k == 0:
                            roi[: rng.randint(6, 26), :] = c_pad
                        elif side_k == 1:
                            roi[QUAD_ROI_H - rng.randint(6, 26) :, :] = c_pad
                        elif side_k == 2:
                            roi[:, : rng.randint(8, 36)] = c_pad
                        else:
                            roi[:, QUAD_ROI_W - rng.randint(8, 36) :] = c_pad
        return QuadCropTrainingSample(
            roi_rgb=roi,
            is_plate=0.0,
            corners_norm=np.zeros((4, 2), dtype=np.float32),
            plate_side=SIDE_PURE_SIDE,
            tight_quality=0.0,
        )

    @staticmethod
    def _sample_constant_surround_rgb(inner_rgb: np.ndarray, rng: random.Random) -> tuple[int, int, int]:
        """Match `_estimate_roi_surround_rgb`: 50% tight-cutout dark matte `(36, 40, 46)±6`, 50% perimeter median `[24..136]`."""
        if rng.random() < 0.50 or inner_rgb.size == 0:
            return (
                rng.randint(30, 44),
                rng.randint(34, 48),
                rng.randint(40, 54),
            )
        h_p, w_p = inner_rgb.shape[:2]
        strips: list[np.ndarray] = []
        if h_p >= 4:
            strips.append(inner_rgb[:2, :].reshape(-1, 3))
            strips.append(inner_rgb[-2:, :].reshape(-1, 3))
        if w_p >= 4:
            strips.append(inner_rgb[:, :2].reshape(-1, 3))
            strips.append(inner_rgb[:, -2:].reshape(-1, 3))
        med = np.median(np.concatenate(strips, axis=0) if strips else inner_rgb.reshape(-1, 3), axis=0)
        j = rng.randint(-6, 6)
        return (
            int(np.clip(round(float(med[0])) + j, 24, 136)),
            int(np.clip(round(float(med[1])) + j, 24, 136)),
            int(np.clip(round(float(med[2])) + j, 24, 136)),
        )

    def synthesize_sample(self, seed: int | None = None) -> QuadCropTrainingSample:
        """Synthesize one `(96, 160, 3)` `QuadCropTrainingSample` in `< 0.8 ms`."""
        split_seed = {"train": 0, "val_holdout": 1 << 48, "cal_holdout": 2 << 48}[self.config.domain_split]
        rng = random.Random(split_seed + (0 if seed is None else int(seed)))
        is_truncated_neg = False
        if rng.random() >= self.config.p_positive:
            if rng.random() < 0.14:
                # ~7.3% of dataset: severely truncated partial-plate ROI (< 58% visible) labeled `is_plate = 0.0`
                # so PlateQuadNet rejects half-plate detector proposals instead of shrinking corners onto a fragment!
                is_truncated_neg = True
            else:
                return self._synthesize_hard_negative(rng)

        plate_side = SIDE_FRONT if rng.random() < 0.50 else SIDE_REAR
        if self.config.domain_split == "train":
            entry, _ = self.bank.pick_stratified_plate(plate_side, rng, p_temporary=self.config.p_temporary_plate)
            if entry.plate_type_str in (PlateType.KEI_COMMERCIAL.value, PlateType.KEI_PRIVATE.value):
                ctx_style = "kei_offset" if (plate_side == SIDE_FRONT and rng.random() < 0.72) else rng.choice(("sedan_bumper", "minivan_garnish"))
            elif entry.plate_type_str == PlateType.STANDARD_COMMERCIAL.value:
                ctx_style = rng.choices(
                    ["truck_recessed", "tractor_exposed_chassis", "mixer_chevron_rear", "bus_flat_panel", "high_grille_work_truck"],
                    weights=[0.30, 0.22, 0.18, 0.18, 0.12],
                    k=1,
                )[0]
            else:
                ctx_style = rng.choices(
                    ["sedan_bumper", "minivan_garnish", "truck_recessed", "bus_flat_panel"],
                    weights=[0.44, 0.34, 0.12, 0.10],
                    k=1,
                )[0]
        else:
            ctx_style = rng.choices(
                [
                    "sedan_bumper",
                    "kei_offset",
                    "truck_recessed",
                    "tractor_exposed_chassis",
                    "mixer_chevron_rear",
                    "bus_flat_panel",
                    "high_grille_work_truck",
                    "minivan_garnish",
                ],
                weights=[0.16, 0.18, 0.14, 0.12, 0.11, 0.11, 0.08, 0.10],
                k=1,
            )[0]

            if ctx_style == "kei_offset":
                p_cat = "kei_black" if rng.random() < 0.38 else "kei"
            elif ctx_style == "bus_flat_panel":
                p_cat = "commercial_green_large" if rng.random() < 0.80 else "commercial_large"
            elif ctx_style in ("truck_recessed", "tractor_exposed_chassis", "mixer_chevron_rear", "high_grille_work_truck"):
                r_c = rng.random()
                if r_c < 0.54:
                    p_cat = "commercial_green_large"
                elif r_c < 0.78:
                    p_cat = "commercial_med"
                else:
                    p_cat = "commercial_large"
            else:
                p_cat = "standard_white" if rng.random() < 0.55 else "standard"

            entry = self.bank.pick_plate(p_cat, plate_side, rng, p_temporary=self.config.p_temporary_plate)
        pw_m, ph_m = (0.440, 0.220) if entry.is_large else (0.330, 0.165)

        # 3D perspective yaw (-68..+68), context-conditional bracket pitch, roll (-18..+18), and distance keystone z ~ logU(2.1, 26.0)
        r_yaw = rng.random()
        if r_yaw < 0.24:
            yaw_deg = rng.choice([-1.0, 1.0]) * rng.uniform(44.0, 67.0)
        elif r_yaw < 0.62:
            yaw_deg = rng.uniform(-44.0, 44.0)
        else:
            yaw_deg = rng.uniform(-22.0, 22.0)
        if ctx_style in ("tractor_exposed_chassis", "mixer_chevron_rear") and rng.random() < 0.60:
            pitch_deg = rng.uniform(-26.0, -6.0)
        else:
            pitch_deg = rng.uniform(-38.0, 25.0)
        roll_deg = rng.uniform(-18.0, 18.0)

        z_cam = math.exp(rng.uniform(math.log(2.1), math.log(26.0)))
        fx_cam = 620.0 * (z_cam / 3.2)
        H_canon = _compute_3d_plate_homography(
            pw_m=pw_m,
            ph_m=ph_m,
            center_3d=(0.0, 0.0, z_cam),
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            fx=fx_cam,
            fy=fx_cam,
            cx=320.0,
            cy=192.0,
        )
        q_cam = cv2.perspectiveTransform(_UNIT_QUAD_4, H_canon).reshape(4, 2)
        qx0, qy0 = float(np.min(q_cam[:, 0])), float(np.min(q_cam[:, 1]))
        qx1, qy1 = float(np.max(q_cam[:, 0])), float(np.max(q_cam[:, 1]))
        qw = max(4.0, qx1 - qx0)
        qh = max(2.0, qy1 - qy0)
        qcx = 0.5 * (qx0 + qx1)
        qcy = 0.5 * (qy0 + qy1)

        # 100% Isotropic Aspect-Locked ROI Extraction matching `extract_expanded_plate_roi(..., expand_scale=1.85)`:
        # Model noisy detector bbox BEFORE extraction, then enforce `rw / rh == QUAD_ROI_W / QUAD_ROI_H` (`sx == sy`!)
        target_ar = float(QUAD_ROI_W) / float(QUAD_ROI_H)
        if is_truncated_neg:
            det_w = qw * rng.uniform(0.75, 1.05)
            det_h = qh * rng.uniform(0.80, 1.15)
            exp_s = rng.uniform(1.45, 1.85)
            rw = max(det_w * exp_s, det_h * exp_s * target_ar, 16.0)
            rh = rw / target_ar
            cut_dir = rng.choice([-1.0, 1.0])
            # Shift ROI center so 44%..66% of the physical plate width falls outside the left or right ROI edge!
            vis_w = rng.uniform(0.34, 0.56) * qw
            if cut_dir < 0.0:
                rx0 = qx1 - vis_w
            else:
                rx0 = qx0 + vis_w - rw
            ry0 = qcy - 0.5 * rh + rng.uniform(-0.15, 0.15) * qh
            rx1 = rx0 + rw
            ry1 = ry0 + rh
        else:
            if rng.random() < 0.85:
                det_w = qw * rng.uniform(0.86, 1.30)
                det_h = qh * rng.uniform(0.86, 1.30)
                det_cx = qcx + rng.uniform(-0.22, 0.22) * qw
                det_cy = qcy + rng.uniform(-0.22, 0.22) * qh
                exp_s = rng.uniform(1.66, 2.04)
            else:
                det_w = qw * rng.uniform(1.08, 1.62)
                det_h = qh * rng.uniform(1.08, 1.62)
                det_cx = qcx + rng.uniform(-0.32, 0.32) * qw
                det_cy = qcy + rng.uniform(-0.32, 0.32) * qh
                exp_s = rng.uniform(1.52, 2.24)

            rw = max(det_w * exp_s, det_h * exp_s * target_ar, 16.0)
            rh = rw / target_ar
            rx0 = det_cx - 0.5 * rw
            ry0 = det_cy - 0.5 * rh
            rx1 = rx0 + rw
            ry1 = ry0 + rh

            # Keep true physical corners cleanly inside [1, W-1] x [1, H-1] AND strictly re-enforce isotropic `rw/rh = target_ar`!
            pad_x = 0.018 * rw
            pad_y = 0.018 * rh
            if qx0 - pad_x < rx0 or qx1 + pad_x > rx1 or qy0 - pad_y < ry0 or qy1 + pad_y > ry1:
                rx0 = min(rx0, qx0 - pad_x)
                rx1 = max(rx1, qx1 + pad_x)
                ry0 = min(ry0, qy0 - pad_y)
                ry1 = max(ry1, qy1 + pad_y)
                rcx_f = 0.5 * (rx0 + rx1)
                rcy_f = 0.5 * (ry0 + ry1)
                rw = max(rx1 - rx0, (ry1 - ry0) * target_ar, 16.0)
                rh = rw / target_ar
                rx0 = rcx_f - 0.5 * rw
                rx1 = rcx_f + 0.5 * rw
                ry0 = rcy_f - 0.5 * rh
                ry1 = rcy_f + 0.5 * rh

        scale_iso = float(QUAD_ROI_W) / (rx1 - rx0)
        dst_pts = np.empty((4, 2), dtype=np.float32)
        dst_pts[:, 0] = (q_cam[:, 0] - rx0) * scale_iso
        dst_pts[:, 1] = (q_cam[:, 1] - ry0) * scale_iso

        # Build surrounding vehicle bumper / grille / bus / exposed chassis / mixer background
        roi = np.empty((QUAD_ROI_H, QUAD_ROI_W, 3), dtype=np.uint8)
        body_c = rng.choice(FullSceneDetSynthesizer.BODY_COLORS)
        px0_i, py0_i = int(np.min(dst_pts[:, 0])), int(np.min(dst_pts[:, 1]))
        px1_i, py1_i = int(np.max(dst_pts[:, 0])), int(np.max(dst_pts[:, 1]))

        if ctx_style == "truck_recessed":
            # Dark recessed truck under-bed + overhanging cargo bed above + underrun bar + optional sticker
            roi[:] = (26, 28, 32)
            bed_lip = max(4, py0_i - rng.randint(2, 12))
            bed_c = rng.choice([(212, 216, 222), (48, 84, 126), (56, 96, 74), (62, 66, 72)])
            cv2.rectangle(roi, (0, 0), (QUAD_ROI_W, bed_lip), bed_c, -1)
            bar_y = min(QUAD_ROI_H - 4, py1_i + rng.randint(4, 14))
            cv2.rectangle(roi, (0, bar_y), (QUAD_ROI_W, min(QUAD_ROI_H, bar_y + 9)), (68, 72, 78), -1)
            if rng.random() < 0.55 and px1_i < QUAD_ROI_W - 26:
                st = rng.choice(self.bank.sticker_patches)
                sw_s = min(44, QUAD_ROI_W - px1_i - 4)
                sh_s = max(8, sw_s // 3)
                sy_s = max(2, min(QUAD_ROI_H - sh_s - 2, py0_i))
                roi[sy_s : sy_s + sh_s, px1_i + 3 : px1_i + 3 + sw_s] = cv2.resize(st, (sw_s, sh_s), interpolation=cv2.INTER_AREA)
        elif ctx_style == "tractor_exposed_chassis":
            # Open semi-tractor / Unic chassis frame WITHOUT white overhanging cargo bed lip!
            roi[:] = rng.choice([(24, 26, 30), (34, 38, 44), (54, 58, 64)])
            # Asphalt road visible behind lower open chassis
            rd_y = min(QUAD_ROI_H, py1_i + rng.randint(2, 16))
            if rd_y < QUAD_ROI_H:
                asp = self.bank.asphalt_deltas[rng.randrange(len(self.bank.asphalt_deltas))]
                ox_p, oy_p = rng.randrange(96), rng.randrange(160)
                roi[rd_y:] = np.clip(64 + asp[oy_p : oy_p + (QUAD_ROI_H - rd_y), ox_p : ox_p + QUAD_ROI_W, None], 0, 255).astype(np.uint8)
            # Horizontal chassis cross-participant & round/rect tail-lamp nearby
            cm_y = max(4, py0_i - rng.randint(2, 10))
            cv2.rectangle(roi, (0, max(0, cm_y - 6)), (QUAD_ROI_W, cm_y), (42, 46, 52), -1)
            if px1_i < QUAD_ROI_W - 22:
                cv2.rectangle(roi, (px1_i + 6, max(6, py0_i)), (min(QUAD_ROI_W - 4, px1_i + 30), min(QUAD_ROI_H - 6, py0_i + 18)), (198, 34, 38), -1)
        elif ctx_style == "mixer_chevron_rear":
            # Concrete mixer / rough-terrain crane rear: chute shadow, chevron underrun bar, optional orange fence clutter!
            roi[:] = rng.choice([(36, 40, 46), (168, 174, 182), (214, 172, 42)])
            if rng.random() < 0.35:
                # Orange safety mesh visible in upper corner so orange mesh is NEVER an anti-plate shortcut!
                for gx_f in range(4, 42, 9):
                    cv2.line(roi, (gx_f, 0), (gx_f, 34), (228, 94, 28), 1)
            chv_y = min(QUAD_ROI_H - 6, py1_i + rng.randint(3, 12))
            cv2.rectangle(roi, (4, chv_y), (QUAD_ROI_W - 4, min(QUAD_ROI_H, chv_y + 11)), (234, 232, 226), -1)
            c_chv = (206, 40, 42) if rng.random() < 0.55 else (24, 26, 30)
            for sx_c in range(6, QUAD_ROI_W - 8, 18):
                cv2.line(roi, (sx_c, min(QUAD_ROI_H - 1, chv_y + 10)), (sx_c + 9, chv_y), c_chv, 4)
        elif ctx_style == "bus_flat_panel":
            # Flush highway/route bus panel with blue/green/red or low-contrast dark-green/teal livery stripe!
            bus_c = rng.choice([(232, 236, 240), (198, 224, 238), (218, 228, 214), (44, 88, 148), (34, 78, 56), (28, 62, 54)])
            roi[:] = bus_c
            st_y = max(2, py0_i - rng.randint(8, 22))
            cv2.rectangle(roi, (0, max(0, st_y - 10)), (QUAD_ROI_W, st_y), rng.choice([(32, 98, 178), (38, 138, 86), (188, 42, 48), (26, 68, 48)]), -1)
            bmp_edge = min(QUAD_ROI_H, py1_i + rng.randint(10, 24))
            if bmp_edge < QUAD_ROI_H:
                roi[bmp_edge:] = (58, 60, 64)
            if px0_i > 24 and rng.random() < 0.55:
                for ly_l in range(max(8, py0_i), min(QUAD_ROI_H - 8, py1_i), 5):
                    cv2.line(roi, (6, ly_l), (px0_i - 6, ly_l), (58, 62, 68), 1)
        elif ctx_style == "high_grille_work_truck":
            # High-mounted front plate directly over dark horizontal radiator grille slats (snowplows / wheel-loaders / heavy dumps)
            roi[:] = body_c
            cv2.rectangle(roi, (8, 8), (QUAD_ROI_W - 8, QUAD_ROI_H - 8), (30, 32, 36), -1)
            for gy_s in range(12, QUAD_ROI_H - 10, rng.choice([6, 8, 10])):
                cv2.line(roi, (10, gy_s), (QUAD_ROI_W - 10, gy_s), (136, 142, 150), 1)
        else:
            roi[:] = body_c
            # Road asphalt strip below lower bumper edge (with granular road texture)
            bmp_edge = min(QUAD_ROI_H, py1_i + rng.randint(6, 24))
            if bmp_edge < QUAD_ROI_H:
                asp = self.bank.asphalt_deltas[rng.randrange(len(self.bank.asphalt_deltas))]
                ox_p, oy_p = rng.randrange(96), rng.randrange(160)
                rh_a = QUAD_ROI_H - bmp_edge
                roi[bmp_edge:] = np.clip(66 + asp[oy_p : oy_p + rh_a, ox_p : ox_p + QUAD_ROI_W, None], 0, 255).astype(np.uint8)
            # Recessed bumper pocket around plate
            cv2.rectangle(
                roi,
                (max(0, px0_i - 6), max(0, py0_i - 5)),
                (min(QUAD_ROI_W - 1, px1_i + 6), min(QUAD_ROI_H - 1, py1_i + 4)),
                (max(14, int(body_c[0] * 0.55)), max(14, int(body_c[1] * 0.55)), max(16, int(body_c[2] * 0.55))),
                -1,
            )
            if ctx_style == "kei_offset":
                # Adjacent black radiator grille slats immediately left or right of offset kei plate!
                if px0_i > QUAD_ROI_W - px1_i and px0_i > 16:
                    gx0, gx1 = 4, px0_i - 5
                else:
                    gx0, gx1 = min(QUAD_ROI_W - 6, px1_i + 5), QUAD_ROI_W - 4
                if gx1 - gx0 >= 12:
                    cv2.rectangle(roi, (gx0, max(4, py0_i - 2)), (gx1, min(QUAD_ROI_H - 4, py1_i + 2)), (24, 26, 30), -1)
                    for gy in range(max(6, py0_i + 2), min(QUAD_ROI_H - 6, py1_i), 6):
                        cv2.line(roi, (gx0 + 2, gy), (gx1 - 2, gy), (82, 86, 92), 1)
            elif ctx_style == "minivan_garnish":
                # Chrome garnish bar directly above plate
                gy_bar = max(2, py0_i - rng.randint(6, 12))
                cv2.rectangle(roi, (max(0, px0_i - 14), gy_bar), (min(QUAD_ROI_W, px1_i + 14), max(gy_bar + 2, py0_i - 1)), (214, 220, 228), -1)

        # Optional thick outer chrome/black dealer bracket OUTSIDE the physical plate [0,1]^2 boundary!
        # `corners_norm` strictly excludes this outer dealer bracket!
        if rng.random() < 0.28:
            H_u2roi = cv2.getPerspectiveTransform(_UNIT_QUAD_4.reshape(4, 2), dst_pts)
            outer_unit = np.array([[[-0.045, -0.06]], [[1.045, -0.06]], [[1.045, 1.11]], [[-0.045, 1.11]]], dtype=np.float32)
            outer_pts = cv2.perspectiveTransform(outer_unit, H_u2roi).reshape(4, 2)
            bezel_col = (212, 218, 226) if rng.random() < 0.5 else (28, 30, 34)
            cv2.fillConvexPoly(roi, np.round(outer_pts).astype(np.int32), bezel_col, lineType=cv2.LINE_AA)

        # Warp physical plate patch into exact `dst_pts` using projected surface edge length for mip selection
        w_edge_mean = 0.5 * (float(np.linalg.norm(dst_pts[1] - dst_pts[0])) + float(np.linalg.norm(dst_pts[2] - dst_pts[3])))
        pw_roi = max(float(px1_i - px0_i), w_edge_mean)
        src_patch = entry.patch_half if pw_roi < 76.0 else entry.patch_full
        _warp_patch_into_canvas(roi, src_patch, dst_pts)

        # True amodal opaque top-lip bumper/tailgate overhang (22% of positives):
        lip_pen = 1.0
        if rng.random() < 0.22:
            H_u2lip = cv2.getPerspectiveTransform(_UNIT_QUAD_4.reshape(4, 2), dst_pts)
            lip_depth = rng.uniform(0.035, 0.115)
            lip_pen = 1.0 - 1.6 * lip_depth
            lip_unit = np.array(
                [[[-0.08, -0.35]], [[1.08, -0.35]], [[1.08, lip_depth]], [[-0.08, lip_depth]]],
                dtype=np.float32,
            )
            lip_pts = cv2.perspectiveTransform(lip_unit, H_u2lip).reshape(4, 2)
            lip_col = (
                (rng.randint(24, 48), rng.randint(28, 52), rng.randint(32, 58))
                if ctx_style in ("truck_recessed", "tractor_exposed_chassis", "mixer_chevron_rear")
                else (max(16, int(body_c[0] * 0.72)), max(16, int(body_c[1] * 0.72)), max(18, int(body_c[2] * 0.72)))
            )
            cv2.fillConvexPoly(roi, np.round(lip_pts).astype(np.int32), lip_col, lineType=cv2.LINE_AA)

        # Heavy upper-row bumper/garnish overhang shadow (35% deep 0.06x-0.25x in train; 32% legacy in holdout)
        sh_pen = 1.0
        p_overhang_sh = 0.35 if self.config.domain_split == "train" else 0.32
        if rng.random() < p_overhang_sh:
            if self.config.domain_split == "train":
                sh_frac = rng.uniform(0.28, 0.56)
                sh_dark = rng.uniform(0.06, 0.25)
            else:
                sh_frac = rng.uniform(0.22, 0.48)
                sh_dark = rng.uniform(0.18, 0.44)
            sh_pen = 0.82 + 0.30 * sh_dark
            sh_tilt = rng.uniform(-5.0, 5.0)
            yy_s, xx_s = np.mgrid[0:QUAD_ROI_H, 0:QUAD_ROI_W].astype(np.float32)
            y_cut = float(py0_i) + sh_frac * max(6.0, float(py1_i - py0_i)) + sh_tilt * ((xx_s - 0.5 * QUAD_ROI_W) / QUAD_ROI_W)
            sh_alpha = np.clip((y_cut - yy_s) / rng.uniform(2.0, 5.5), 0.0, 1.0)
            sh_mult = (1.0 - sh_alpha * (1.0 - sh_dark))[:, :, None]
            roi[:] = np.clip(roi.astype(np.float32) * sh_mult, 0.0, 255.0).astype(np.uint8)

        occ_scale = float(np.clip(w_edge_mean / 84.0, 0.45, 1.15))
        qual_deg = self._apply_roi_degradation(roi, rng, force_lowres=False, plate_occ_scale=occ_scale)

        # Out-of-frame / pre-cropped constant-border surround (30% of positives, matching negatives & extract_expanded_plate_roi!)
        if rng.random() < 0.30:
            if rng.random() < 0.62:
                bx0_c = max(0, px0_i - rng.randint(0, 5))
                by0_c = max(0, py0_i - rng.randint(0, 4))
                bx1_c = min(QUAD_ROI_W, px1_i + rng.randint(1, 6))
                by1_c = min(QUAD_ROI_H, py1_i + rng.randint(1, 5))
                c_pad_p = self._sample_constant_surround_rgb(roi[by0_c:by1_c, bx0_c:bx1_c], rng)
                if by0_c > 0:
                    roi[:by0_c, :] = c_pad_p
                if by1_c < QUAD_ROI_H:
                    roi[by1_c:, :] = c_pad_p
                if bx0_c > 0:
                    roi[:, :bx0_c] = c_pad_p
                if bx1_c < QUAD_ROI_W:
                    roi[:, bx1_c:] = c_pad_p
            else:
                c_pad_p = self._sample_constant_surround_rgb(roi, rng)
                if py0_i > 2 and rng.random() < 0.5:
                    roi[: max(1, py0_i - rng.randint(0, 4)), :] = c_pad_p
                if py1_i < QUAD_ROI_H - 2 and rng.random() < 0.5:
                    roi[min(QUAD_ROI_H - 1, py1_i + rng.randint(1, 5)) :, :] = c_pad_p
                if px0_i > 2 and rng.random() < 0.5:
                    roi[:, : max(1, px0_i - rng.randint(0, 5))] = c_pad_p
                if px1_i < QUAD_ROI_W - 2 and rng.random() < 0.5:
                    roi[:, min(QUAD_ROI_W - 1, px1_i + rng.randint(1, 6)) :] = c_pad_p
        ang_factor = max(0.35, math.cos(math.radians(abs(yaw_deg) * 0.75)) * math.cos(math.radians(abs(pitch_deg) * 0.6)))
        tight_quality = float(np.clip(qual_deg * ang_factor * lip_pen * sh_pen, 0.15, 1.0))

        if is_truncated_neg:
            return QuadCropTrainingSample(
                roi_rgb=roi,
                is_plate=0.0,
                corners_norm=np.zeros((4, 2), dtype=np.float32),
                plate_side=SIDE_PURE_SIDE,
                tight_quality=0.0,
            )

        corners_norm = np.empty((4, 2), dtype=np.float32)
        corners_norm[:, 0] = np.clip(dst_pts[:, 0] / float(QUAD_ROI_W), 0.0, 1.0)
        corners_norm[:, 1] = np.clip(dst_pts[:, 1] / float(QUAD_ROI_H), 0.0, 1.0)

        return QuadCropTrainingSample(
            roi_rgb=roi,
            is_plate=1.0,
            corners_norm=corners_norm,
            plate_side=plate_side,
            tight_quality=round(tight_quality, 4),
        )

    def _render_structured_hardneg_on_canvas(
        self,
        canvas: np.ndarray,
        mode: str,
        weather: str,
        horizon_y: int,
        rng: random.Random,
    ) -> tuple[float, float, float, float, float]:
        """Render one of the 6 structured hard-negative traps onto `(320, 480, 3)` `canvas`; return `(x0, y0, x1, y1, obj_w)`."""
        h_c, w_c = canvas.shape[:2]
        dim = 0.50 if weather == "night" else (0.85 if weather in ("sunset", "wet_road") else 1.0)
        cx_t = rng.uniform(0.30, 0.70) * float(w_c)
        cy_t = rng.uniform(0.32, 0.68) * float(h_c)
        tw = rng.uniform(44.0, 118.0)

        if mode == "blue_highway_route_sign":
            cy_t = rng.uniform(max(44.0, horizon_y - 42.0), min(float(h_c) - 60.0, horizon_y + 38.0))
            th = tw * rng.uniform(0.56, 0.88)
            pole_x = int(round(cx_t))
            cv2.rectangle(canvas, (pole_x - 4, int(round(cy_t))), (pole_x + 4, min(h_c - 4, horizon_y + 72)), (146, 152, 160), -1)
            bp = rng.choice(self.bank.blue_route_patches)
            if dim < 0.99:
                bp = np.clip(bp.astype(np.float32) * max(0.55, dim), 0.0, 255.0).astype(np.uint8)
            dx_top = rng.uniform(-5.0, 5.0)
            dy_r = rng.uniform(-4.0, 4.0)
            dst_q = np.array(
                [
                    [cx_t - 0.5 * tw + dx_top, cy_t - 0.5 * th],
                    [cx_t + 0.5 * tw + dx_top, cy_t - 0.5 * th + dy_r],
                    [cx_t + 0.5 * tw, cy_t + 0.5 * th + dy_r],
                    [cx_t - 0.5 * tw, cy_t + 0.5 * th],
                ],
                dtype=np.float32,
            )
            _warp_patch_into_canvas(canvas, bp, dst_q)
            return (float(dst_q[:, 0].min()), float(dst_q[:, 1].min()), float(dst_q[:, 0].max()), float(dst_q[:, 1].max()), tw)

        if mode == "frameless_truck_body_text":
            # Full side/rear truck panel with ribs, chassis & wheels in surround + frameless painted text (ZERO border!)
            vx0 = max(12, int(round(cx_t - tw * rng.uniform(1.3, 2.0))))
            vx1 = min(w_c - 12, int(round(cx_t + tw * rng.uniform(1.3, 2.0))))
            vy0 = max(16, int(round(cy_t - tw * rng.uniform(0.65, 1.05))))
            vy1 = min(h_c - 22, int(round(cy_t + tw * rng.uniform(0.65, 1.05))))
            raw_m = rng.choice([(228, 232, 238), (196, 202, 210), (42, 86, 144), (48, 98, 68), (226, 184, 42), (46, 50, 58)])
            metal_c = (int(raw_m[0] * dim), int(raw_m[1] * dim), int(raw_m[2] * dim))
            cv2.rectangle(canvas, (vx0, vy0), (vx1, vy1), metal_c, -1)
            cv2.rectangle(canvas, (vx0 + 8, vy1), (vx1 - 8, min(h_c - 6, vy1 + 26)), (28, 30, 36), -1)
            rib_d = (max(10, metal_c[0] - 26), max(10, metal_c[1] - 26), max(12, metal_c[2] - 26))
            for ry_r in range(vy0 + 12, vy1 - 8, rng.randint(14, 24)):
                cv2.line(canvas, (vx0 + 2, ry_r), (vx1 - 2, ry_r), rib_d, 2)
            for wx_t in (vx0 + int((vx1 - vx0) * 0.24), vx0 + int((vx1 - vx0) * 0.78)):
                wy_t = min(h_c - 14, vy1 + 18)
                cv2.circle(canvas, (wx_t, wy_t), 18, (20, 22, 26), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (wx_t, wy_t), 10, (176, 182, 190), -1, lineType=cv2.LINE_AA)
            lum_m = 0.299 * metal_c[0] + 0.587 * metal_c[1] + 0.114 * metal_c[2]
            cand_rgba = [
                st for st in self.bank.frameless_decal_rgba
                if abs((0.299 * float(st[0, 0, 0]) + 0.587 * float(st[0, 0, 1]) + 0.114 * float(st[0, 0, 2])) * dim - lum_m) > 62.0
            ]
            st_rgba = rng.choice(cand_rgba if cand_rgba else self.bank.frameless_decal_rgba)
            if dim < 0.99:
                st_rgba = st_rgba.copy()
                st_rgba[:, :, :3] = np.clip(st_rgba[:, :, :3].astype(np.float32) * max(0.55, dim), 0.0, 255.0).astype(np.uint8)
            th = tw * rng.uniform(0.22, 0.34)
            skew_y = rng.uniform(-5.0, 5.0)
            dst_q = np.array(
                [
                    [cx_t - 0.5 * tw, cy_t - 0.5 * th],
                    [cx_t + 0.5 * tw, cy_t - 0.5 * th + skew_y],
                    [cx_t + 0.5 * tw, cy_t + 0.5 * th + skew_y],
                    [cx_t - 0.5 * tw, cy_t + 0.5 * th],
                ],
                dtype=np.float32,
            )
            _warp_rgba_patch_into_canvas(canvas, st_rgba, dst_q, opacity=rng.uniform(0.85, 0.98))
            return (float(dst_q[:, 0].min()), float(dst_q[:, 1].min()), float(dst_q[:, 0].max()), float(dst_q[:, 1].max()), tw)

        if mode == "bordered_payload_and_speed_decals":
            vx0 = max(16, int(round(cx_t - tw * 1.6)))
            vx1 = min(w_c - 16, int(round(cx_t + tw * 1.6)))
            vy0 = max(16, int(round(cy_t - tw * 1.0)))
            vy1 = min(h_c - 20, int(round(cy_t + tw * 0.9)))
            raw_b = rng.choice([(218, 222, 228), (52, 88, 136), (56, 98, 74), (58, 62, 70)])
            bed_c = (int(raw_b[0] * dim), int(raw_b[1] * dim), int(raw_b[2] * dim))
            cv2.rectangle(canvas, (vx0, vy0), (vx1, vy1), bed_c, -1)
            cv2.rectangle(canvas, (vx0 + 12, vy1), (vx1 - 12, min(h_c - 6, vy1 + 24)), (26, 28, 34), -1)
            cv2.line(canvas, (vx0 + 4, vy1 - 3), (vx1 - 4, vy1 - 3), (206, 44, 40), 3)
            dc = rng.choice(self.bank.bordered_decal_patches)
            if dim < 0.99:
                dc = np.clip(dc.astype(np.float32) * max(0.55, dim), 0.0, 255.0).astype(np.uint8)
            th = tw * rng.uniform(0.28, 0.52)
            dst_q = np.array(
                [
                    [cx_t - 0.5 * tw, cy_t - 0.5 * th],
                    [cx_t + 0.5 * tw, cy_t - 0.5 * th + rng.uniform(-3.0, 3.0)],
                    [cx_t + 0.5 * tw, cy_t + 0.5 * th + rng.uniform(-3.0, 3.0)],
                    [cx_t - 0.5 * tw, cy_t + 0.5 * th],
                ],
                dtype=np.float32,
            )
            _warp_patch_into_canvas(canvas, dc, dst_q)
            return (float(dst_q[:, 0].min()), float(dst_q[:, 1].min()), float(dst_q[:, 0].max()), float(dst_q[:, 1].max()), tw)

        if mode == "shop_parking_multiline_sign":
            cy_t = rng.uniform(max(44.0, horizon_y - 26.0), min(float(h_c) - 56.0, horizon_y + 48.0))
            th = tw * rng.uniform(0.42, 0.58)
            px_l = int(round(cx_t - 0.36 * tw))
            px_r = int(round(cx_t + 0.36 * tw))
            for px_p in (px_l, px_r):
                cv2.rectangle(canvas, (px_p - 2, int(round(cy_t)),), (px_p + 2, min(h_c - 6, int(round(cy_t + th + 48)))), (150, 156, 164), -1)
            mp = rng.choice(self.bank.multiline_parking_patches)
            if dim < 0.99:
                mp = np.clip(mp.astype(np.float32) * max(0.60, dim), 0.0, 255.0).astype(np.uint8)
            dst_q = np.array(
                [
                    [cx_t - 0.5 * tw, cy_t - 0.5 * th],
                    [cx_t + 0.5 * tw, cy_t - 0.5 * th + rng.uniform(-4.0, 4.0)],
                    [cx_t + 0.5 * tw, cy_t + 0.5 * th + rng.uniform(-4.0, 4.0)],
                    [cx_t - 0.5 * tw, cy_t + 0.5 * th],
                ],
                dtype=np.float32,
            )
            _warp_patch_into_canvas(canvas, mp, dst_q)
            return (float(dst_q[:, 0].min()), float(dst_q[:, 1].min()), float(dst_q[:, 0].max()), float(dst_q[:, 1].max()), tw)

        if mode == "grille_emblem_bumper_slats":
            # Full front fascia WITHOUT any plate in the grille/intake zone
            cy_t = rng.uniform(max(float(horizon_y) + 28.0, 128.0), float(h_c) - 62.0)
            vw_g = tw * rng.uniform(2.1, 2.8)
            vh_g = vw_g * rng.uniform(0.68, 0.86)
            vx0 = int(round(cx_t - 0.5 * vw_g))
            vx1 = int(round(cx_t + 0.5 * vw_g))
            vy1 = min(h_c - 10, int(round(cy_t + 0.36 * vh_g)))
            vy0 = max(12, int(round(vy1 - vh_g)))
            raw_c = rng.choice(FullSceneDetSynthesizer.BODY_COLORS)
            body_c = (int(raw_c[0] * dim), int(raw_c[1] * dim), int(raw_c[2] * dim))
            cv2.rectangle(canvas, (max(0, vx0), max(0, int(round(vy0 + 0.42 * vh_g)))), (min(w_c, vx1), vy1), body_c, -1)
            cab_pts = np.array(
                [[vx0 + 0.08 * vw_g, vy0 + 0.42 * vh_g], [vx0 + 0.18 * vw_g, vy0], [vx1 - 0.18 * vw_g, vy0], [vx1 - 0.08 * vw_g, vy0 + 0.42 * vh_g]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(canvas, cab_pts, body_c, lineType=cv2.LINE_AA)
            gx0 = max(6, int(round(cx_t - 0.5 * tw)))
            gx1 = min(w_c - 6, int(round(cx_t + 0.5 * tw)))
            th = tw * rng.uniform(0.38, 0.56)
            gy0 = max(8, int(round(cy_t - 0.5 * th)))
            gy1 = min(h_c - 12, int(round(cy_t + 0.5 * th)))
            cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (24, 26, 32), -1)
            cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (196, 202, 210), 2)
            for ys in range(gy0 + 6, gy1 - 2, rng.choice((6, 8, 10))):
                cv2.line(canvas, (gx0 + 2, ys), (gx1 - 2, ys), (156, 162, 172), 1)
            cv2.ellipse(canvas, ((gx0 + gx1) // 2, (gy0 + gy1) // 2), (max(6, int(tw * 0.12)), max(4, int(th * 0.18))), 0, 0, 360, (216, 222, 230), 2, lineType=cv2.LINE_AA)
            hl_w = max(10, int(vw_g * 0.16))
            cv2.rectangle(canvas, (max(2, vx0 + 8), gy0 - 2), (min(w_c - 2, vx0 + 8 + hl_w), gy0 + max(8, int(th * 0.55))), (238, 242, 248), -1)
            cv2.rectangle(canvas, (max(2, vx1 - 8 - hl_w), gy0 - 2), (min(w_c - 2, vx1 - 8), gy0 + max(8, int(th * 0.55))), (238, 242, 248), -1)
            return (float(gx0), float(gy0), float(gx1), float(gy1), float(gx1 - gx0))

        # building_window_grid_and_reflectors
        if rng.random() < 0.58:
            cy_t = rng.uniform(44.0, max(78.0, float(horizon_y) + 18.0))
            th = tw * rng.uniform(0.45, 0.65)
            bx0 = max(8, int(round(cx_t - tw * 1.1)))
            bx1 = min(w_c - 8, int(round(cx_t + tw * 1.1)))
            by0 = max(4, int(round(cy_t - th * 1.1)))
            by1 = min(h_c - 16, int(round(cy_t + th * 1.1)))
            is_night_w = weather == "night" or rng.random() < 0.42
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (36, 40, 50) if is_night_w else (166, 162, 156), -1)
            ww = max(14, int(tw * 0.30))
            wh = max(9, int(ww * 0.54))
            for yw in range(by0 + 8, by1 - wh - 4, wh + 8):
                for xw in range(bx0 + 8, bx1 - ww - 4, ww + 9):
                    lit = is_night_w and rng.random() < 0.65
                    pc = rng.choice([(244, 226, 158), (218, 234, 248)]) if lit else (42, 52, 64)
                    cv2.rectangle(canvas, (xw, yw), (xw + ww, yw + wh), pc, -1)
                    cv2.rectangle(canvas, (xw, yw), (xw + ww, yw + wh), (206, 210, 216), 1)
            return (cx_t - 0.5 * tw, cy_t - 0.5 * th, cx_t + 0.5 * tw, cy_t + 0.5 * th, tw)

        cy_t = rng.uniform(float(horizon_y) + 16.0, float(h_c) - 44.0)
        th = tw * rng.uniform(0.32, 0.48)
        gx0 = max(12, int(round(cx_t - tw * 1.2)))
        gx1 = min(w_c - 12, int(round(cx_t + tw * 1.2)))
        gy0 = int(round(cy_t - 0.32 * th))
        gy1 = int(round(cy_t + 0.32 * th))
        cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (216, 220, 226), -1)
        for xp in range(gx0 + 16, gx1 - 10, 42):
            cv2.rectangle(canvas, (xp - 3, gy1), (xp + 3, min(h_c - 6, gy1 + 26)), (142, 148, 156), -1)
            ref_c = (238, 122, 24) if rng.random() < 0.55 else (214, 36, 40)
            cv2.rectangle(canvas, (xp - 10, gy0 + 2), (xp + 10, max(gy0 + 5, gy1 - 2)), ref_c, -1)
        return (cx_t - 0.5 * tw, cy_t - 0.5 * th, cx_t + 0.5 * tw, cy_t + 0.5 * th, tw)

    def synthesize_dual_sample(
        self,
        seed: int | None = None,
        sampling_policy: int = 0,
        *,
        fovea_scale: float | None = None,
        surround_scale: float | None = None,
    ) -> dict[str, Any]:
        """Synthesize concentric `fovea_rgb (96, 160, 3)` + `surround_rgb (64, 96, 3)` on a rendered `(320, 480, 3)` canvas.

        Returns dict with:
          - `fovea_rgb`: `(96, 160, 3)` uint8
          - `surround_rgb`: `(64, 96, 3)` uint8
          - `is_plate`: float (`1.0` or `0.0`)
          - `corners_norm`: `(4, 2)` float32 signed unclamped coords in `[-0.35, 1.35]` relative to `fovea` window
          - `corner_vis`: `(4,)` float32
          - `plate_id`: int (`>= 1` per registration for positives, `0` for negatives)
          - `sampling_policy`: int (`0..3`)
        """
        from vpocr_det.quad_cropper import (
            FOVEA_BASE_SCALE,
            SURROUND_BASE_SCALE,
            concentric_roi_windows,
            extract_dual_concentric_plate_rois,
        )

        fov_s = float(FOVEA_BASE_SCALE if fovea_scale is None else fovea_scale)
        sur_s = float(SURROUND_BASE_SCALE if surround_scale is None else surround_scale)

        pol = int(sampling_policy) % 4
        split_seed = {"train": 0, "val_holdout": 1 << 48, "cal_holdout": 2 << 48}[self.config.domain_split]
        eff_seed = int(random.getrandbits(31)) if seed is None else int(seed)
        rng = random.Random(split_seed + eff_seed + 0x51A6E2)
        prop_rng = random.Random(split_seed + eff_seed * 131 + pol * 9973 + 0x2719)

        h_c, w_c = DUAL_CANVAS_H, DUAL_CANVAS_W
        canvas = np.empty((h_c, w_c, 3), dtype=np.uint8)
        scene_family = rng.choice(FullSceneDetSynthesizer.SCENE_FAMILIES)
        weather = rng.choices(FullSceneDetSynthesizer.WEATHER_MODES, weights=[0.34, 0.22, 0.12, 0.16, 0.16], k=1)[0]
        horizon_y = rng.randint(int(h_c * 0.24), int(h_c * 0.44))
        vp_x = rng.randint(int(w_c * 0.34), int(w_c * 0.66))

        is_pos = rng.random() < self.config.p_positive
        self._bg_helper._draw_background(
            canvas,
            scene_family=scene_family,
            weather=weather,
            horizon_y=horizon_y,
            vp_x=vp_x,
            rng=rng,
            extra_clutter=(not is_pos),
        )

        if not is_pos:
            mode = rng.choice(self.STRUCTURED_HARD_NEG_MODES)
            tx0, ty0, tx1, ty1, obj_w = self._render_structured_hardneg_on_canvas(
                canvas, mode=mode, weather=weather, horizon_y=horizon_y, rng=rng
            )
            # Symmetric 45% coupled pre-resize 12..30 px + 8x8 DCT + 4:2:0 degradation on (320, 480, 3)
            if rng.random() < 0.45:
                _apply_coupled_lowres_dct_420_np(canvas, rng, curr_obj_px=obj_w, target_px_range=(12.0, 30.0))
            elif rng.random() < 0.28:
                canvas[:] = cv2.GaussianBlur(canvas, (3, 3), rng.uniform(0.55, 1.15))

            if rng.random() < 0.14:
                gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
                canvas[:, :, 0] = gray
                canvas[:, :, 1] = gray
                canvas[:, :, 2] = gray

            bw_n = max(8.0, tx1 - tx0)
            bh_n = max(6.0, ty1 - ty0)
            jx = prop_rng.uniform(-0.20, 0.20) * bw_n
            jy = prop_rng.uniform(-0.20, 0.20) * bh_n
            sc_n = prop_rng.uniform(0.84, 1.24)
            ncx = 0.5 * (tx0 + tx1) + jx
            ncy = 0.5 * (ty0 + ty1) + jy
            prop_neg = (
                ncx - 0.5 * bw_n * sc_n,
                ncy - 0.5 * bh_n * sc_n,
                ncx + 0.5 * bw_n * sc_n,
                ncy + 0.5 * bh_n * sc_n,
            )
            fov_u8, sur_u8, _, _ = extract_dual_concentric_plate_rois(
                canvas, prop_neg, gamma=prop_rng.uniform(1.0, 1.35), fovea_scale=fov_s, surround_scale=sur_s
            )
            return {
                "fovea_rgb": fov_u8,
                "surround_rgb": sur_u8,
                "is_plate": 0.0,
                "corners_norm": np.zeros((4, 2), dtype=np.float32),
                "corner_vis": np.zeros((4,), dtype=np.float32),
                "plate_id": 0,
                "sampling_policy": pol,
                "plate_side": SIDE_PURE_SIDE,
                "tight_quality": 0.0,
                "visible_support_fraction": 0.0,
                "is_large": 0.0,
                "blur_vec_norm": np.zeros((2,), dtype=np.float32),
                "min_projected_edge_px": 0.0,
                "blur_magnitude_px": 0.0,
                "local_contrast_snr": 0.0,
                "hard_neg_mode": mode,
            }

        # --- Positive sample (`is_plate = 1.0`) on `(320, 480, 3)` canvas ---
        plate_side = SIDE_FRONT if rng.random() < 0.48 else SIDE_REAR
        entry, plate_id = self.bank.pick_stratified_plate(plate_side, rng, p_temporary=self.config.p_temporary_plate)
        ptype = entry.plate_type_str
        if ptype in (PlateType.KEI_COMMERCIAL.value, PlateType.KEI_PRIVATE.value):
            arch = rng.choice(("kei_car", "kei_van"))
        elif ptype == PlateType.STANDARD_COMMERCIAL.value:
            arch = rng.choices(
                ("box_truck", "cargo_truck", "dump_truck", "bus", "commercial_van"),
                weights=[0.28, 0.24, 0.18, 0.14, 0.16],
                k=1,
            )[0]
        else:
            arch = rng.choices(
                ("sedan", "hatchback", "minivan", "suv", "commercial_van"),
                weights=[0.28, 0.18, 0.26, 0.18, 0.10],
                k=1,
            )[0]

        pw_m, ph_m = (0.440, 0.220) if entry.is_large else (0.330, 0.165)
        if arch in ("kei_car", "kei_van"):
            vw_m, vh_m = 1.48, 1.68
        elif arch == "bus":
            vw_m, vh_m = 2.46, 3.05
        elif arch in ("box_truck", "cargo_truck", "dump_truck"):
            vw_m, vh_m = (2.38, 2.95) if entry.is_large else (1.90, 2.45)
        elif arch in ("minivan", "suv", "commercial_van"):
            vw_m, vh_m = 1.82, 1.88
        else:
            vw_m, vh_m = 1.76, 1.46

        # Render crisp base canvas width `42..118 px`; 45% coupled low-res pass downsamples to 12..30 px!
        target_canvas_pw = rng.uniform(44.0, 116.0)
        r_yaw = rng.random()
        if r_yaw < 0.22:
            yaw_deg = rng.choice((-1.0, 1.0)) * rng.uniform(42.0, 64.0)
        elif r_yaw < 0.62:
            yaw_deg = rng.uniform(-42.0, 42.0)
        else:
            yaw_deg = rng.uniform(-20.0, 20.0)
        pitch_deg = rng.uniform(-30.0, 18.0)
        roll_deg = rng.uniform(-14.0, 14.0)

        fx = 465.0
        cos_yaw = max(0.42, math.cos(math.radians(yaw_deg)))
        z_m = max(1.65, (fx * pw_m * cos_yaw) / target_canvas_pw)
        veh_w_px = min(float(w_c) * 0.86, fx * (vw_m * max(0.78, cos_yaw) + 0.85 * abs(math.sin(math.radians(yaw_deg)))) / z_m)
        veh_h_px = min(float(h_c) * 0.82, fx * vh_m / z_m)

        vcx = rng.uniform(max(125.0, 0.5 * veh_w_px + 18.0), min(float(w_c) - 125.0, float(w_c) - 0.5 * veh_w_px - 18.0))
        vy1 = rng.uniform(max(float(horizon_y) + 64.0, veh_h_px + 16.0), float(h_c) - 18.0)
        vy0 = vy1 - veh_h_px
        vx0 = vcx - 0.5 * veh_w_px
        vx1 = vcx + 0.5 * veh_w_px

        p_anchor = self._bg_helper._draw_vehicle_and_compute_plate(
            canvas,
            archetype=arch,
            plate_side=plate_side,
            vx0=vx0,
            vy0=vy0,
            vx1=vx1,
            vy1=vy1,
            weather=weather,
            rng=rng,
        )
        px_c, py_c = p_anchor if p_anchor is not None else (vcx, vy0 + 0.74 * veh_h_px)
        px_c = float(np.clip(px_c, 74.0, float(w_c) - 74.0))
        py_c = float(np.clip(py_c, 52.0, float(h_c) - 48.0))

        cx_cam, cy_cam = 0.5 * float(w_c), float(horizon_y)
        tx_3d = (px_c - cx_cam) * z_m / fx
        ty_3d = (py_c - cy_cam) * z_m / fx
        H_3d = _compute_3d_plate_homography(
            pw_m=pw_m,
            ph_m=ph_m,
            center_3d=(tx_3d, ty_3d, z_m),
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            fx=fx,
            fy=fx,
            cx=cx_cam,
            cy=cy_cam,
        )
        q_canvas = cv2.perspectiveTransform(_UNIT_QUAD_4, H_3d).reshape(4, 2).astype(np.float32)

        # Keep true corners inside [20, W_c-20] x [16, H_c-16] so sensor never clips true plate
        dx_clamp = 0.0
        dy_clamp = 0.0
        if float(q_canvas[:, 0].min()) < 20.0:
            dx_clamp = 20.0 - float(q_canvas[:, 0].min())
        elif float(q_canvas[:, 0].max()) > float(w_c) - 20.0:
            dx_clamp = (float(w_c) - 20.0) - float(q_canvas[:, 0].max())
        if float(q_canvas[:, 1].min()) < 16.0:
            dy_clamp = 16.0 - float(q_canvas[:, 1].min())
        elif float(q_canvas[:, 1].max()) > float(h_c) - 16.0:
            dy_clamp = (float(h_c) - 16.0) - float(q_canvas[:, 1].max())
        if dx_clamp != 0.0 or dy_clamp != 0.0:
            q_canvas[:, 0] += dx_clamp
            q_canvas[:, 1] += dy_clamp

        # Optional outer chrome/black dealer bracket outside physical [0, 1]^2 boundary
        if rng.random() < 0.26:
            H_u2c = cv2.getPerspectiveTransform(_UNIT_QUAD_4.reshape(4, 2), q_canvas)
            outer_u = np.array([[[-0.045, -0.06]], [[1.045, -0.06]], [[1.045, 1.10]], [[-0.045, 1.10]]], dtype=np.float32)
            outer_p = cv2.perspectiveTransform(outer_u, H_u2c).reshape(4, 2)
            cv2.fillConvexPoly(canvas, np.round(outer_p).astype(np.int32), (210, 216, 224) if rng.random() < 0.5 else (28, 30, 34), lineType=cv2.LINE_AA)

        w_edge_mean = 0.5 * (float(np.linalg.norm(q_canvas[1] - q_canvas[0])) + float(np.linalg.norm(q_canvas[2] - q_canvas[3])))
        src_patch = entry.patch_half if w_edge_mean < 64.0 else entry.patch_full
        _warp_patch_into_canvas(canvas, src_patch, q_canvas)

        # Optional opaque top-lip bumper/tailgate overhang (~18% of positives)
        corner_unoccluded = np.ones((4,), dtype=bool)
        lip_pen = 1.0
        vis_support = 1.0
        if rng.random() < 0.18:
            H_u2lip = cv2.getPerspectiveTransform(_UNIT_QUAD_4.reshape(4, 2), q_canvas)
            lip_depth = rng.uniform(0.04, 0.11)
            lip_pen = 1.0 - 1.5 * lip_depth
            vis_support = 1.0 - lip_depth
            if lip_depth >= 0.055:
                corner_unoccluded[0] = False
                corner_unoccluded[1] = False
            lip_u = np.array([[[-0.08, -0.28]], [[1.08, -0.28]], [[1.08, lip_depth]], [[-0.08, lip_depth]]], dtype=np.float32)
            lip_p = cv2.perspectiveTransform(lip_u, H_u2lip).reshape(4, 2)
            cv2.fillConvexPoly(canvas, np.round(lip_p).astype(np.int32), (32, 36, 42), lineType=cv2.LINE_AA)

        # 35% deep upper-half overhang shadow (`0.06x-0.25x` luminance multiplier) across recessed pocket & upper half
        sh_pen = 1.0
        qx0_c, qy0_c = float(q_canvas[:, 0].min()), float(q_canvas[:, 1].min())
        qx1_c, qy1_c = float(q_canvas[:, 0].max()), float(q_canvas[:, 1].max())
        qw_c = max(6.0, qx1_c - qx0_c)
        qh_c = max(4.0, qy1_c - qy0_c)
        if rng.random() < 0.35:
            sh_frac = rng.uniform(0.30, 0.56)
            sh_dark = rng.uniform(0.06, 0.25)
            sh_pen = 0.78 + 0.35 * sh_dark
            sx0 = max(0, int(math.floor(qx0_c - 0.35 * qw_c)))
            sx1 = min(w_c, int(math.ceil(qx1_c + 0.35 * qw_c)))
            sy0 = max(0, int(math.floor(qy0_c - 0.45 * qh_c)))
            sy1 = min(h_c, int(math.ceil(qy1_c + 0.15 * qh_c)))
            if sx1 > sx0 + 2 and sy1 > sy0 + 2:
                yy_p, xx_p = np.mgrid[sy0:sy1, sx0:sx1].astype(np.float32)
                sh_tilt = rng.uniform(-4.5, 4.5)
                y_cut = qy0_c + sh_frac * qh_c + sh_tilt * ((xx_p - 0.5 * (qx0_c + qx1_c)) / max(8.0, qw_c))
                y_top = qy0_c - 0.30 * qh_c
                alpha_y = np.clip((y_cut - yy_p) / rng.uniform(1.8, 4.2), 0.0, 1.0) * np.clip((yy_p - y_top) / max(2.0, 0.18 * qh_c), 0.0, 1.0)
                x_pad = 0.22 * qw_c
                alpha_x = np.clip((xx_p - (qx0_c - x_pad)) / max(2.0, 0.15 * qw_c), 0.0, 1.0) * np.clip(((qx1_c + x_pad) - xx_p) / max(2.0, 0.15 * qw_c), 0.0, 1.0)
                sh_m = (1.0 - (alpha_y * alpha_x) * (1.0 - sh_dark))[:, :, None]
                canvas[sy0:sy1, sx0:sx1] = np.clip(canvas[sy0:sy1, sx0:sx1].astype(np.float32) * sh_m, 0.0, 255.0).astype(np.uint8)

        # 45% coupled pre-resize `12..30 px` + H.264 8x8 DCT + 4:2:0 low-res degradation on `(320, 480, 3)`
        qual_deg = 1.0
        bvx_c, bvy_c = 0.0, 0.0
        if rng.random() < 0.45:
            qual_deg, (bvx_c, bvy_c) = _apply_coupled_lowres_dct_420_np(
                canvas, rng, curr_obj_px=w_edge_mean, target_px_range=(12.0, 30.0)
            )
        elif rng.random() < 0.26:
            canvas[:] = cv2.GaussianBlur(canvas, (3, 3), rng.uniform(0.55, 1.15))
            qual_deg *= 0.88

        r_ph = rng.random()
        if r_ph < 0.14:
            gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
            canvas[:, :, 0] = gray
            canvas[:, :, 1] = gray
            canvas[:, :, 2] = gray
            qual_deg *= 0.92
        elif r_ph < 0.26:
            gain_n = rng.uniform(0.42, 0.72)
            canvas[:] = np.clip(canvas.astype(np.float32) * gain_n, 0.0, 255.0).astype(np.uint8)
            qual_deg *= 0.86

        # Sample detector/tracker proposal bbox according to `sampling_policy` (`0..3`),
        # keeping unclamped signed `corners_norm` strictly in `[-0.35, 1.35]` (never clamped to `[0, 1]`!).
        qcx_c = 0.5 * (qx0_c + qx1_c)
        qcy_c = 0.5 * (qy0_c + qy1_c)
        want_overhang = prop_rng.random() < (0.64 if pol in (1, 2) else (0.26 if pol == 0 else 0.16))

        chosen_prop = (qx0_c, qy0_c, qx1_c, qy1_c)
        for _ in range(96):
            if want_overhang:
                ang_p = prop_rng.uniform(0.0, 2.0 * math.pi)
                rad_p = prop_rng.uniform(0.46, 0.88) * math.hypot(qw_c, qh_c)
                pcx = qcx_c + rad_p * math.cos(ang_p)
                pcy = qcy_c + rad_p * math.sin(ang_p) * 0.82
                sc_p = prop_rng.uniform(0.68, 1.04)
                ar_p = math.exp(prop_rng.uniform(-0.22, 0.22))
            else:
                pcx = qcx_c + prop_rng.uniform(-0.20, 0.20) * qw_c
                pcy = qcy_c + prop_rng.uniform(-0.20, 0.20) * qh_c
                sc_p = prop_rng.uniform(0.88, 1.30)
                ar_p = math.exp(prop_rng.uniform(-0.14, 0.14))
            pw_p = max(6.0, qw_c * sc_p * ar_p)
            ph_p = max(4.0, qh_c * sc_p / ar_p)
            cand_prop = (pcx - 0.5 * pw_p, pcy - 0.5 * ph_p, pcx + 0.5 * pw_p, pcy + 0.5 * ph_p)
            win_f_cand, _ = concentric_roi_windows(cand_prop, gamma=1.0, fovea_scale=fov_s, surround_scale=sur_s)
            rx_f, ry_f, rwf, rhf = win_f_cand
            cn_x = (q_canvas[:, 0] - rx_f) / max(1e-3, rwf)
            cn_y = (q_canvas[:, 1] - ry_f) / max(1e-3, rhf)
            if (
                float(cn_x.min()) >= -0.345
                and float(cn_x.max()) <= 1.345
                and float(cn_y.min()) >= -0.345
                and float(cn_y.max()) <= 1.345
            ):
                is_outside_unit = bool(
                    float(cn_x.min()) < 0.0
                    or float(cn_x.max()) > 1.0
                    or float(cn_y.min()) < 0.0
                    or float(cn_y.max()) > 1.0
                )
                if (not want_overhang) or is_outside_unit:
                    chosen_prop = cand_prop
                    break

        fov_u8, sur_u8, win_f, _ = extract_dual_concentric_plate_rois(
            canvas, chosen_prop, gamma=1.0, fovea_scale=fov_s, surround_scale=sur_s
        )
        rx0_f, ry0_f, rw_f, rh_f = win_f
        corners_norm = np.empty((4, 2), dtype=np.float32)
        corners_norm[:, 0] = (q_canvas[:, 0] - rx0_f) / max(1e-3, rw_f)
        corners_norm[:, 1] = (q_canvas[:, 1] - ry0_f) / max(1e-3, rh_f)

        in_sensor = (
            (q_canvas[:, 0] >= 0.0)
            & (q_canvas[:, 0] < float(w_c))
            & (q_canvas[:, 1] >= 0.0)
            & (q_canvas[:, 1] < float(h_c))
        )
        in_fovea = (
            (corners_norm[:, 0] >= 0.0)
            & (corners_norm[:, 0] <= 1.0)
            & (corners_norm[:, 1] >= 0.0)
            & (corners_norm[:, 1] <= 1.0)
        )
        corner_vis = (corner_unoccluded & in_sensor & in_fovea).astype(np.float32)

        ix0_f = max(0.0, float(corners_norm[:, 0].min()))
        iy0_f = max(0.0, float(corners_norm[:, 1].min()))
        ix1_f = min(1.0, float(corners_norm[:, 0].max()))
        iy1_f = min(1.0, float(corners_norm[:, 1].max()))
        span_x = max(1e-4, float(corners_norm[:, 0].max() - corners_norm[:, 0].min()))
        span_y = max(1e-4, float(corners_norm[:, 1].max() - corners_norm[:, 1].min()))
        fov_frac = max(0.0, ix1_f - ix0_f) * max(0.0, iy1_f - iy0_f) / (span_x * span_y)
        vis_frac_total = float(np.clip(vis_support * (0.35 + 0.65 * fov_frac), 0.15, 1.0))

        edge_lens = [float(np.linalg.norm(q_canvas[(ei + 1) % 4] - q_canvas[ei])) for ei in range(4)]
        min_edge_px = float(min(edge_lens)) * min(1.0, qual_deg * 1.15)
        blur_mag_px = float(math.hypot(bvx_c, bvy_c))
        snr_est = float(np.clip(14.0 * qual_deg * sh_pen, 1.5, 22.0))
        ang_factor = max(0.35, math.cos(math.radians(abs(yaw_deg) * 0.75)) * math.cos(math.radians(abs(pitch_deg) * 0.6)))
        tight_quality = float(np.clip(qual_deg * ang_factor * lip_pen * sh_pen, 0.15, 1.0))
        blur_vec_norm = np.array([bvx_c / max(1e-3, rw_f), bvy_c / max(1e-3, rh_f)], dtype=np.float32)

        return {
            "fovea_rgb": fov_u8,
            "surround_rgb": sur_u8,
            "is_plate": 1.0,
            "corners_norm": corners_norm,
            "corner_vis": corner_vis,
            "plate_id": int(plate_id),
            "sampling_policy": pol,
            "plate_side": int(plate_side),
            "tight_quality": round(tight_quality, 4),
            "visible_support_fraction": round(vis_frac_total, 4),
            "is_large": 1.0 if entry.is_large else 0.0,
            "blur_vec_norm": blur_vec_norm,
            "min_projected_edge_px": round(min_edge_px, 3),
            "blur_magnitude_px": round(blur_mag_px, 3),
            "local_contrast_snr": round(snr_est, 3),
        }


class QuadCropBatch(NamedTuple):
    """Collated QuadCrop batch supporting both 5-tuple unpacking and dict-style string lookup."""

    rois: Tensor           # FloatTensor (B, 3, 96, 160) in [0, 1]
    is_plate: Tensor       # FloatTensor (B,)
    corners_norm: Tensor   # FloatTensor (B, 4, 2)
    plate_side: Tensor     # LongTensor (B,)
    tight_quality: Tensor  # FloatTensor (B,)

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, str):
            return getattr(self, key)
        return tuple.__getitem__(self, key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key in self._fields
        return tuple.__contains__(self, key)

    def keys(self) -> tuple[str, ...]:
        return self._fields

    def items(self) -> list[tuple[str, Tensor]]:
        return [(k, getattr(self, k)) for k in self._fields]


def _scene_sample_to_tensors(sample: SynthSceneSample) -> tuple[Tensor, dict[str, Tensor]]:
    """Convert `SynthSceneSample` to `(FloatTensor(3, H, W), target_dict)`."""
    img_t = torch.from_numpy(sample.image_rgb).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
    objs = sample.objects
    n = len(objs)
    if n == 0:
        target = {
            "labels": torch.zeros((0,), dtype=torch.long),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "corners": torch.zeros((0, 4, 2), dtype=torch.float32),
            "sides": torch.zeros((0,), dtype=torch.long),
        }
    else:
        labels = torch.tensor([o.cls_id for o in objs], dtype=torch.long)
        boxes = torch.from_numpy(np.stack([o.bbox_xyxy for o in objs], axis=0).astype(np.float32))
        corners = torch.from_numpy(np.stack([o.corners_xy for o in objs], axis=0).astype(np.float32))
        sides = torch.tensor([o.plate_side for o in objs], dtype=torch.long)
        target = {
            "labels": labels,
            "boxes": boxes,
            "corners": corners,
            "sides": sides,
        }
    return img_t, target


class SceneDetDataset(Dataset):
    """PyTorch `Dataset` generating full-frame detection scenes on the fly across CPU workers."""

    def __init__(
        self,
        num_samples: int = 50_000,
        image_size: tuple[int, int] = (384, 640),
        base_seed: int = 0,
        config: SceneSynthConfig | None = None,
        synthesizer: FullSceneDetSynthesizer | None = None,
        *,
        image_hw: tuple[int, int] | None = None,
        start_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.num_samples = int(num_samples)
        self.base_seed = int(start_seed if start_seed is not None else base_seed)
        self.epoch = 0
        eff_hw = image_hw if image_hw is not None else image_size
        cfg = replace(config, image_size=eff_hw) if (config is not None and image_hw is not None) else (config or SceneSynthConfig(image_size=eff_hw))
        self.synthesizer = synthesizer or FullSceneDetSynthesizer(cfg)
        self._worker_initialized = False

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        if not self._worker_initialized:
            cv2.setNumThreads(0)
            self._worker_initialized = True
        seed = self.base_seed + self.epoch * max(1, self.num_samples) + int(idx)
        sample = self.synthesizer.synthesize_scene(seed=seed)
        return _scene_sample_to_tensors(sample)


def collate_det_scenes(
    batch: list[tuple[Tensor, dict[str, Tensor]] | SynthSceneSample],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Collate `SceneDetDataset` items (or raw `SynthSceneSample`s) into `(images, gt_dicts)`."""
    if not batch:
        return torch.zeros((0, 3, 384, 640), dtype=torch.float32), []
    imgs: list[Tensor] = []
    targets: list[dict[str, Tensor]] = []
    for item in batch:
        if isinstance(item, SynthSceneSample):
            im, tgt = _scene_sample_to_tensors(item)
        else:
            im, tgt = item
        imgs.append(im)
        targets.append(tgt)
    return torch.stack(imgs, dim=0), targets


def _quad_sample_to_tuple(sample: QuadCropTrainingSample) -> tuple[Tensor, float, np.ndarray, int, float]:
    roi_t = torch.from_numpy(sample.roi_rgb).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
    return (roi_t, float(sample.is_plate), sample.corners_norm, int(sample.plate_side), float(sample.tight_quality))


class QuadCropDataset(Dataset):
    """PyTorch `Dataset` generating `(96, 160, 3)` `PlateQuadNet` training samples on the fly."""

    def __init__(
        self,
        num_samples: int = 100_000,
        base_seed: int = 1_000_000,
        config: QuadCropSynthConfig | None = None,
        synthesizer: QuadCropSynthesizer | None = None,
        *,
        start_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.num_samples = int(num_samples)
        self.base_seed = int(start_seed if start_seed is not None else base_seed)
        self.epoch = 0
        self.synthesizer = synthesizer or QuadCropSynthesizer(config)
        self._worker_initialized = False

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self._worker_initialized:
            cv2.setNumThreads(0)
            self._worker_initialized = True
        seed = self.base_seed + self.epoch * max(1, self.num_samples) + int(idx)
        s = self.synthesizer.synthesize_sample(seed=seed)
        roi_t = torch.from_numpy(s.roi_rgb).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
        return (
            roi_t,
            torch.tensor(s.is_plate, dtype=torch.float32),
            torch.from_numpy(s.corners_norm.astype(np.float32)),
            torch.tensor(s.plate_side, dtype=torch.long),
            torch.tensor(s.tight_quality, dtype=torch.float32),
        )


def collate_quad_crops(
    batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | QuadCropTrainingSample],
) -> QuadCropBatch:
    """Collate `QuadCropDataset` items (or `QuadCropTrainingSample`s) into `(rois, is_plate, corners_norm, plate_side, tight_quality)`."""
    if not batch:
        return QuadCropBatch(
            rois=torch.zeros((0, 3, QUAD_ROI_H, QUAD_ROI_W), dtype=torch.float32),
            is_plate=torch.zeros((0,), dtype=torch.float32),
            corners_norm=torch.zeros((0, 4, 2), dtype=torch.float32),
            plate_side=torch.zeros((0,), dtype=torch.long),
            tight_quality=torch.zeros((0,), dtype=torch.float32),
        )
    if isinstance(batch[0], QuadCropTrainingSample):
        rois = torch.stack(
            [torch.from_numpy(s.roi_rgb).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0) for s in batch],  # type: ignore[union-attr]
            dim=0,
        )
        is_plate = torch.tensor([s.is_plate for s in batch], dtype=torch.float32)  # type: ignore[union-attr]
        corners_norm = torch.from_numpy(np.stack([s.corners_norm for s in batch], axis=0).astype(np.float32))  # type: ignore[union-attr]
        plate_side = torch.tensor([s.plate_side for s in batch], dtype=torch.long)  # type: ignore[union-attr]
        tight_quality = torch.tensor([s.tight_quality for s in batch], dtype=torch.float32)  # type: ignore[union-attr]
        return QuadCropBatch(rois, is_plate, corners_norm, plate_side, tight_quality)

    rois = torch.stack([it[0] for it in batch], dim=0)  # type: ignore[index]
    is_plate = torch.stack([it[1] for it in batch], dim=0)  # type: ignore[index]
    corners_norm = torch.stack([it[2] for it in batch], dim=0)  # type: ignore[index]
    plate_side = torch.stack([it[3] for it in batch], dim=0)  # type: ignore[index]
    tight_quality = torch.stack([it[4] for it in batch], dim=0)  # type: ignore[index]
    return QuadCropBatch(rois, is_plate, corners_norm, plate_side, tight_quality)


__all__ = [
    "QUAD_ROI_H",
    "QUAD_ROI_W",
    "SceneSynthConfig",
    "FullSceneDetSynthesizer",
    "QuadCropSynthConfig",
    "QuadCropSynthesizer",
    "QuadCropBatch",
    "SceneDetDataset",
    "QuadCropDataset",
    "collate_det_scenes",
    "collate_quad_crops",
    "det_worker_init_fn",
]
