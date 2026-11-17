"""Part 4 Crop Extractor: oracle, rectified, and realistic detector bounding-box simulation."""

from dataclasses import dataclass, field
import math
import random
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageDraw

from vpocr_synth.camera.models import DegradedSequenceResult
from vpocr_synth.crop.models import CROP_HEIGHT, CROP_WIDTH, CropSequenceSample
from vpocr_synth.crop.readability import SlotReadabilityMapper
from vpocr_synth.motion.models import FrameAnnotation, MultiCameraPassResult, SequenceResult
from vpocr_synth.plate.models import PlateConfig, PlateText
from vpocr_synth.plate.renderer import PlateRenderer
from vpocr_synth.plate.sampler import PlateSampler


@dataclass
class CropExtractorConfig:
    """Configuration for detector crop extraction and realistic tracker perturbations."""

    crop_mode: str = "realistic_detector"   # "oracle", "realistic_detector", "rectified"
    base_margin_x: float = 0.08             # Fractional horizontal margin around plate bbox
    base_margin_y: float = 0.10             # Fractional vertical margin around plate bbox
    ar1_rho: float = 0.82                   # Temporal autocorrelation of detector bbox drift
    drift_sigma_xy: float = 0.055           # Stddev of normalized center drift
    scale_jitter_sigma: float = 0.06        # Stddev of scale/aspect jitter
    truncate_prob: float = 0.10             # Per-frame probability of partial edge truncation
    drop_prob: float = 0.04                 # Per-frame detector false-negative miss probability
    garbage_prob: float = 0.06              # Per-frame garbage/whiteout/empty crop injection prob
    inject_track_mixed: bool = False        # Explicitly splice 2..5 frames of a foreign plate
    track_mixed_prob: float = 0.0           # Random probability of track_mixed per sequence
    emit_variants: list[str] = field(default_factory=lambda: ["raw"])


class CropExtractor:
    """Extracts `(T, 64, 128, 3)` RGB crop sequences and `(T, 12)` `slot_readability` from sequences."""

    _CLAHE: Any = None

    def __init__(self, config: CropExtractorConfig | None = None) -> None:
        self.config = config or CropExtractorConfig()
        self.plate_sampler = PlateSampler()
        self.plate_renderer = PlateRenderer()

    @classmethod
    def _get_clahe(cls) -> Any:
        if cls._CLAHE is None:
            cls._CLAHE = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(4, 4))
        return cls._CLAHE

    @classmethod
    def apply_view_variant(cls, crop_rgb: np.ndarray, variant: str) -> np.ndarray:
        """Produce photometric preprocessing variant ('raw', 'gamma_low', 'gamma_high', 'clahe')."""
        if variant == "raw":
            return crop_rgb.copy()
        if variant == "gamma_low":
            # Shadow recovery (gamma < 1 lifts dark underexposed regions)
            norm = crop_rgb.astype(np.float32) / 255.0
            out = np.power(np.clip(norm, 0.0, 1.0), 0.55) * 255.0
            return np.clip(out, 0.0, 255.0).astype(np.uint8)
        if variant == "gamma_high":
            # Highlight compression / recovery (gamma > 1 darkens washed-out highlights)
            norm = crop_rgb.astype(np.float32) / 255.0
            out = np.power(np.clip(norm, 0.0, 1.0), 1.80) * 255.0
            return np.clip(out, 0.0, 255.0).astype(np.uint8)
        if variant == "clahe":
            lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_eq = cls._get_clahe().apply(l_ch)
            merged = cv2.merge([l_eq, a_ch, b_ch])
            return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        raise ValueError(f"Unsupported view_variant '{variant}'")

    def crop(
        self,
        frame: np.ndarray,
        plate_box: tuple[float, float, float, float],
        jitter_params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Extract a single (64, 128, 3) crop around plate_box (satisfies CropSimulatorProtocol)."""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = plate_box
        bw = max(4.0, x1 - x0)
        bh = max(4.0, y1 - y0)

        jp = jitter_params or {}
        mx = float(jp.get("margin_x", self.config.base_margin_x))
        my = float(jp.get("margin_y", self.config.base_margin_y))
        dx = float(jp.get("dx", 0.0)) * bw
        dy = float(jp.get("dy", 0.0)) * bh

        cx0 = x0 - mx * bw + dx
        cy0 = y0 - my * bh + dy
        cx1 = x1 + mx * bw + dx
        cy1 = y1 + my * bh + dy

        crop_rgb = self._crop_and_resize_xyxy(frame, (cx0, cy0, cx1, cy1))
        return crop_rgb, (cx0, cy0, cx1, cy1)

    @staticmethod
    def _crop_and_resize_xyxy(
        frame_rgb: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Sub-pixel affine warp from arbitrary bbox [x0, y0, x1, y1] directly to (64, 128, 3) RGB."""
        x0, y0, x1, y1 = bbox_xyxy
        bw = max(2.0, x1 - x0)
        bh = max(2.0, y1 - y0)
        src_tri = np.array([[x0, y0], [x1, y0], [x0, y1]], dtype=np.float32)
        dst_tri = np.array(
            [[0.0, 0.0], [float(CROP_WIDTH), 0.0], [0.0, float(CROP_HEIGHT)]],
            dtype=np.float32,
        )
        M = cv2.getAffineTransform(src_tri, dst_tri)
        interp = cv2.INTER_AREA if bw > CROP_WIDTH * 1.3 else cv2.INTER_LINEAR
        return cv2.warpAffine(
            frame_rgb,
            M,
            (CROP_WIDTH, CROP_HEIGHT),
            flags=interp,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def _rectify_quad_to_crop(
        frame_rgb: np.ndarray,
        corners_2d: list[tuple[float, float]],
        rng: random.Random,
        keypoint_jitter_px: float = 0.0,
    ) -> np.ndarray:
        """4-point homography rectification to canonical (64, 128, 3) RGB."""
        pts = np.array(corners_2d, dtype=np.float32)
        if keypoint_jitter_px > 0.0:
            for i in range(4):
                pts[i, 0] += rng.gauss(0.0, keypoint_jitter_px)
                pts[i, 1] += rng.gauss(0.0, keypoint_jitter_px)
        dst = np.array(
            [
                [0.0, 0.0],
                [float(CROP_WIDTH), 0.0],
                [float(CROP_WIDTH), float(CROP_HEIGHT)],
                [0.0, float(CROP_HEIGHT)],
            ],
            dtype=np.float32,
        )
        H = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(
            frame_rgb,
            H,
            (CROP_WIDTH, CROP_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _sample_distinct_foreign_plate(
        self,
        primary_plate: PlateText,
        seed: int,
    ) -> tuple[PlateText, PlateConfig]:
        for offset in range(1, 50):
            f_text, f_cfg = self.plate_sampler.sample(seed=seed + offset * 7919)
            if f_text.ocr_string != primary_plate.ocr_string:
                f_cfg.scale = 1.0
                return f_text, f_cfg
        fallback = PlateText(region="横浜", classification="330", hiragana="ら", serial=9876)
        return fallback, PlateConfig(scale=1.0, seed=seed + 1)

    def extract_from_sequence(
        self,
        clean_sequence: SequenceResult,
        degraded_sequence: DegradedSequenceResult | None = None,
        crop_mode: str | None = None,
        seed: int = 0,
        emit_variants: list[str] | None = None,
        inject_track_mixed: bool | None = None,
        garbage_prob: float | None = None,
        truncate_prob: float | None = None,
        drop_prob: float | None = None,
    ) -> CropSequenceSample:
        """Extract a `CropSequenceSample` from a single-camera sequence."""
        stream_items: list[tuple[np.ndarray, FrameAnnotation, dict[str, Any] | None, int, int, PlateConfig]] = []
        src_images = degraded_sequence.images if degraded_sequence is not None else clean_sequence.images
        deg_frames = degraded_sequence.frames if degraded_sequence is not None else None

        for idx, annot in enumerate(clean_sequence.frames):
            rgb = np.array(src_images[idx], dtype=np.uint8)
            deg_log = deg_frames[idx].get("degradation") if deg_frames is not None else None
            stream_items.append(
                (
                    rgb,
                    annot,
                    deg_log,
                    clean_sequence.camera_id,
                    clean_sequence.plate_side,
                    clean_sequence.plate_config,
                )
            )

        seq_id = degraded_sequence.sequence_id if degraded_sequence is not None else clean_sequence.sequence_id
        return self._extract_from_items(
            sequence_id=seq_id,
            plate_text=clean_sequence.plate_text,
            primary_plate_config=clean_sequence.plate_config,
            items=stream_items,
            crop_mode=crop_mode,
            seed=seed,
            emit_variants=emit_variants,
            inject_track_mixed=inject_track_mixed,
            garbage_prob=garbage_prob,
            truncate_prob=truncate_prob,
            drop_prob=drop_prob,
        )

    def extract_from_multicam_pass(
        self,
        multicam_pass: MultiCameraPassResult,
        crop_mode: str | None = None,
        seed: int = 0,
        emit_variants: list[str] | None = None,
        inject_track_mixed: bool | None = None,
    ) -> CropSequenceSample:
        """Extract a unified multi-camera `CropSequenceSample` ordered by timestamp across 1..4 cameras."""
        items: list[tuple[np.ndarray, FrameAnnotation, dict[str, Any] | None, int, int, PlateConfig]] = []
        for stream in multicam_pass.streams:
            c_seq = stream.clean_sequence
            d_seq = stream.degraded_sequence
            imgs = d_seq.images if d_seq is not None else c_seq.images
            deg_frames = d_seq.frames if d_seq is not None else None
            for idx, annot in enumerate(c_seq.frames):
                rgb = np.array(imgs[idx], dtype=np.uint8)
                deg_log = deg_frames[idx].get("degradation") if deg_frames is not None else None
                items.append(
                    (
                        rgb,
                        annot,
                        deg_log,
                        stream.camera_id,
                        stream.plate_side,
                        c_seq.plate_config,
                    )
                )

        # Sort merged multi-camera observations chronologically
        items.sort(key=lambda it: (it[1].timestamp, it[3]))

        return self._extract_from_items(
            sequence_id=multicam_pass.pass_id,
            plate_text=multicam_pass.plate_text,
            primary_plate_config=multicam_pass.front_plate_config,
            items=items,
            crop_mode=crop_mode,
            seed=seed,
            emit_variants=emit_variants,
            inject_track_mixed=inject_track_mixed,
        )

    def _extract_from_items(
        self,
        sequence_id: str,
        plate_text: PlateText,
        primary_plate_config: PlateConfig,
        items: list[tuple[np.ndarray, FrameAnnotation, dict[str, Any] | None, int, int, PlateConfig]],
        crop_mode: str | None = None,
        seed: int = 0,
        emit_variants: list[str] | None = None,
        inject_track_mixed: bool | None = None,
        garbage_prob: float | None = None,
        truncate_prob: float | None = None,
        drop_prob: float | None = None,
    ) -> CropSequenceSample:
        mode = crop_mode or self.config.crop_mode
        if mode not in ("oracle", "realistic_detector", "rectified"):
            raise ValueError(f"Unsupported crop_mode '{mode}'")

        variants = emit_variants if emit_variants is not None else list(self.config.emit_variants)
        if not variants:
            variants = ["raw"]

        rng = random.Random(seed)
        do_track_mixed = (
            inject_track_mixed
            if inject_track_mixed is not None
            else (self.config.inject_track_mixed or (rng.random() < self.config.track_mixed_prob))
        )
        p_garb = self.config.garbage_prob if garbage_prob is None else garbage_prob
        p_trunc = self.config.truncate_prob if truncate_prob is None else truncate_prob
        p_drop = self.config.drop_prob if drop_prob is None else drop_prob

        # Precompute persistent surface occlusion per camera plate_config (front vs rear)
        occ_cache: dict[int, np.ndarray] = {}
        norm_slot_boxes = SlotReadabilityMapper.get_normalized_slot_boxes(
            plate_text,
            primary_plate_config.plate_size,
        )

        # AR(1) state per camera_id for temporally correlated detector drift
        ar_state: dict[int, tuple[float, float, float, float]] = {}
        rho = self.config.ar1_rho
        innov_scale = math.sqrt(max(0.0, 1.0 - rho * rho))

        crops_list: list[np.ndarray] = []
        ts_list: list[float] = []
        src_id_list: list[int] = []
        var_list: list[str] = []
        cam_id_list: list[int] = []
        side_list: list[int] = []
        valid_list: list[float] = []
        foreign_list: list[bool] = []
        read_list: list[np.ndarray] = []

        total_raw = len(items)
        for raw_idx, (frame_rgb, annot, deg_log, cam_id, p_side, p_cfg) in enumerate(items):
            # Optional detector miss (never drop all frames; keep >= 2 raw frames)
            if mode == "realistic_detector" and total_raw > 3 and 0 < raw_idx < total_raw - 1:
                if rng.random() < p_drop:
                    continue

            cfg_key = id(p_cfg)
            if cfg_key not in occ_cache:
                canon = self.plate_renderer.render(plate_text, replace_scale(p_cfg, 1.0))
                occ_cache[cfg_key] = SlotReadabilityMapper.extract_slot_occlusion_from_render(canon)
            slot_occ_12 = occ_cache[cfg_key]

            h_f, w_f = frame_rgb.shape[:2]

            # Garbage / whiteout / non-plate false-positive injection
            inject_garbage_now = (
                mode == "realistic_detector"
                and p_garb > 0.0
                and rng.random() < p_garb
            )

            if inject_garbage_now:
                g_kind = rng.choice(["whiteout", "asphalt_patch", "dark_empty"])
                if g_kind == "whiteout":
                    val = rng.randint(246, 255)
                    raw_crop = np.full((CROP_HEIGHT, CROP_WIDTH, 3), val, dtype=np.uint8)
                elif g_kind == "dark_empty":
                    val = rng.randint(2, 16)
                    raw_crop = np.full((CROP_HEIGHT, CROP_WIDTH, 3), val, dtype=np.uint8)
                else:
                    # False-positive background patch away from plate
                    gx0 = rng.uniform(0.0, max(1.0, w_f - 160.0))
                    gy0 = rng.uniform(max(0.0, h_f * 0.65), max(1.0, h_f - 80.0))
                    raw_crop = self._crop_and_resize_xyxy(frame_rgb, (gx0, gy0, gx0 + 140.0, gy0 + 70.0))
                f_valid = 0.0
                f_read = np.zeros(12, dtype=np.float32)

            elif (not annot.is_visible) or p_side == 2 or (not annot.is_readable_view) or annot.incidence_deg > 76.0:
                # Pure side / back-face / unreadable extreme angle: crop vehicle side/bumper vicinity, valid=0.0
                if annot.is_visible and annot.plate_width_px >= 6.0:
                    x0, y0, x1, y1 = annot.corners.to_bbox()
                    bw, bh = max(24.0, x1 - x0), max(14.0, y1 - y0)
                    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                    raw_crop = self._crop_and_resize_xyxy(
                        frame_rgb,
                        (cx - bw * 0.6, cy - bh * 0.6, cx + bw * 0.6, cy + bh * 0.6),
                    )
                else:
                    # Vehicle body center projection fallback for pure side / back-facing cameras
                    cx, cy = w_f * 0.5, h_f * 0.56
                    raw_crop = self._crop_and_resize_xyxy(frame_rgb, (cx - 120.0, cy - 60.0, cx + 120.0, cy + 60.0))
                f_valid = 0.0
                f_read = np.zeros(12, dtype=np.float32)

            else:
                # Visible & geometrically readable plate view
                x0, y0, x1, y1 = annot.corners.to_bbox()
                bw = max(8.0, x1 - x0)
                bh = max(4.0, y1 - y0)

                if mode == "rectified":
                    raw_crop = self._rectify_quad_to_crop(frame_rgb, annot.corners.points, rng, keypoint_jitter_px=0.8)
                    ioa_12 = np.ones(12, dtype=np.float32)
                elif mode == "oracle":
                    mx, my = self.config.base_margin_x, self.config.base_margin_y
                    crop_box = (x0 - mx * bw, y0 - my * bh, x1 + mx * bw, y1 + my * bh)
                    raw_crop = self._crop_and_resize_xyxy(frame_rgb, crop_box)
                    ioa_12 = SlotReadabilityMapper.compute_slot_crop_ioa(
                        norm_slot_boxes, annot.corners.points, crop_box, (w_f, h_f)
                    )
                else:  # "realistic_detector" with AR(1) walk + optional partial edge truncation
                    prev_dx, prev_dy, prev_sx, prev_sy = ar_state.get(cam_id, (0.0, 0.0, 0.0, 0.0))
                    dx = rho * prev_dx + innov_scale * rng.gauss(0.0, self.config.drift_sigma_xy)
                    dy = rho * prev_dy + innov_scale * rng.gauss(0.0, self.config.drift_sigma_xy)
                    sx = rho * prev_sx + innov_scale * rng.gauss(0.0, self.config.scale_jitter_sigma)
                    sy = rho * prev_sy + innov_scale * rng.gauss(0.0, self.config.scale_jitter_sigma)
                    ar_state[cam_id] = (dx, dy, sx, sy)

                    mx = max(-0.03, self.config.base_margin_x + sx)
                    my = max(-0.03, self.config.base_margin_y + sy)
                    cx0 = x0 - mx * bw + dx * bw
                    cy0 = y0 - my * bh + dy * bh
                    cx1 = x1 + mx * bw + dx * bw
                    cy1 = y1 + my * bh + dy * bh

                    # Occasional partial edge truncation (cuts 18..42% off one edge)
                    if p_trunc > 0.0 and rng.random() < p_trunc:
                        edge = rng.choice(["left", "right", "top", "bottom"])
                        cut = rng.uniform(0.18, 0.42)
                        if edge == "left":
                            cx0 = x0 + cut * bw
                        elif edge == "right":
                            cx1 = x1 - cut * bw
                        elif edge == "top":
                            cy0 = y0 + cut * bh
                        else:
                            cy1 = y1 - cut * bh

                    crop_box = (cx0, cy0, cx1, cy1)
                    raw_crop = self._crop_and_resize_xyxy(frame_rgb, crop_box)
                    ioa_12 = SlotReadabilityMapper.compute_slot_crop_ioa(
                        norm_slot_boxes, annot.corners.points, crop_box, (w_f, h_f)
                    )

                blur_px = float(deg_log.get("blur_px", 0.0)) if deg_log else 0.0
                coc_px = float(deg_log.get("defocus_radius_px", 0.0)) if deg_log else 0.0
                noise_sigma = float(deg_log.get("noise_sigma", 0.0)) if deg_log else 0.0
                jpeg_q = int(deg_log.get("jpeg_quality", 85)) if deg_log else 85
                sub_s = float(deg_log.get("substream_scale", 1.0)) if deg_log else 1.0
                glare_12 = deg_log.get("slot_glare_washout", [0.0] * 12) if deg_log else [0.0] * 12

                # Detect full whiteout crop empirically as additional safeguard
                if float(np.mean(raw_crop)) > 247.0 and float(np.std(raw_crop)) < 5.0:
                    f_valid = 0.0
                    f_read = np.zeros(12, dtype=np.float32)
                else:
                    f_valid = 1.0
                    f_read = SlotReadabilityMapper.compute_frame_slot_readability(
                        plate_height_px=annot.plate_height_px,
                        plate_width_px=annot.plate_width_px,
                        incidence_deg=annot.incidence_deg,
                        is_readable_view=annot.is_readable_view,
                        blur_px=blur_px,
                        defocus_px=coc_px,
                        noise_sigma=noise_sigma,
                        jpeg_quality=jpeg_q,
                        substream_scale=sub_s,
                        slot_occlusion_12=slot_occ_12,
                        slot_glare_12=glare_12,
                        slot_crop_ioa_12=ioa_12,
                    )

            # Emit requested preprocessing variants sharing the exact same source_frame_id!
            for v_name in variants:
                v_crop = self.apply_view_variant(raw_crop, v_name)
                crops_list.append(v_crop)
                ts_list.append(float(annot.timestamp))
                src_id_list.append(int(raw_idx))
                var_list.append(v_name)
                cam_id_list.append(int(cam_id))
                side_list.append(int(p_side))
                valid_list.append(float(f_valid))
                foreign_list.append(False)
                read_list.append(f_read.copy())

        # Track-Mixed Failure Mode Injection: splice 2..5 crisp frames of a foreign vehicle's plate
        foreign_plate_str: str | None = None
        if do_track_mixed and len(crops_list) >= 3:
            f_text, f_cfg = self._sample_distinct_foreign_plate(plate_text, seed)
            foreign_plate_str = f_text.full_text
            f_render = self.plate_renderer.render(f_text, f_cfg)
            f_rgb_src = np.array(f_render.image, dtype=np.uint8)

            n_splice = min(max(2, len(crops_list) // 2), rng.randint(2, 5))
            start_pos = rng.randint(0, max(0, len(crops_list) - n_splice))
            for k_sp in range(n_splice):
                pos = start_pos + k_sp
                fh, fw = f_rgb_src.shape[:2]
                # Realistic slight perspective warp of foreign plate into (64, 128)
                jx = rng.uniform(-3.0, 3.0)
                jy = rng.uniform(-2.0, 2.0)
                src_p = np.array([[0.0, 0.0], [float(fw), 0.0], [float(fw), float(fh)], [0.0, float(fh)]], dtype=np.float32)
                dst_p = np.array(
                    [
                        [5.0 + jx, 4.0 + jy],
                        [float(CROP_WIDTH) - 5.0 + jx, 5.0 - jy],
                        [float(CROP_WIDTH) - 4.0 - jx, float(CROP_HEIGHT) - 4.0 - jy],
                        [4.0 - jx, float(CROP_HEIGHT) - 5.0 + jy],
                    ],
                    dtype=np.float32,
                )
                H_f = cv2.getPerspectiveTransform(src_p, dst_p)
                f_crop = cv2.warpPerspective(
                    f_rgb_src,
                    H_f,
                    (CROP_WIDTH, CROP_HEIGHT),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(52, 54, 58),
                )
                crops_list[pos] = self.apply_view_variant(f_crop, var_list[pos])
                foreign_list[pos] = True
                valid_list[pos] = 0.0
                read_list[pos] = np.zeros(12, dtype=np.float32)

        if not crops_list:
            crops_arr = np.zeros((1, CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
            ts_arr = np.zeros(1, dtype=np.float32)
            src_arr = np.zeros(1, dtype=np.int32)
            var_list = ["raw"]
            cam_arr = np.zeros(1, dtype=np.int32)
            side_arr = np.zeros(1, dtype=np.int32)
            valid_arr = np.zeros(1, dtype=np.float32)
            foreign_arr = np.zeros(1, dtype=bool)
            read_arr = np.zeros((1, 12), dtype=np.float32)
        else:
            crops_arr = np.stack(crops_list, axis=0).astype(np.uint8)
            ts_arr = np.array(ts_list, dtype=np.float32)
            src_arr = np.array(src_id_list, dtype=np.int32)
            cam_arr = np.array(cam_id_list, dtype=np.int32)
            side_arr = np.array(side_list, dtype=np.int32)
            valid_arr = np.array(valid_list, dtype=np.float32)
            foreign_arr = np.array(foreign_list, dtype=bool)
            read_arr = np.stack(read_list, axis=0).astype(np.float32)

        status = SlotReadabilityMapper.compute_sequence_target_status(
            plate_text=plate_text,
            slot_readability=read_arr,
            frame_valid_mask=valid_arr,
            is_foreign_plate=foreign_arr,
        )

        meta: dict[str, Any] = {"crop_mode": mode}
        if foreign_plate_str is not None:
            meta["foreign_plate_text"] = foreign_plate_str

        return CropSequenceSample(
            sequence_id=sequence_id,
            plate_text=plate_text,
            plate_type=primary_plate_config.plate_type.value,
            crops_rgb=crops_arr,
            timestamps_s=ts_arr,
            source_frame_ids=src_arr,
            view_variants=var_list,
            camera_ids=cam_arr,
            plate_sides=side_arr,
            frame_valid_mask=valid_arr,
            is_foreign_plate=foreign_arr,
            slot_readability=read_arr,
            sequence_target_status=status,
            metadata=meta,
        )

    @staticmethod
    def create_crop_sequence_strip(
        sample: CropSequenceSample,
        max_cols: int = 8,
        scale_up: int = 2,
    ) -> Image.Image:
        """Render a visual inspection contact sheet of a `(T, 64, 128, 3)` `CropSequenceSample`."""
        t_len = sample.num_frames
        if t_len <= max_cols:
            idxs = list(range(t_len))
        else:
            idxs = [int(round(i * (t_len - 1) / (max_cols - 1))) for i in range(max_cols)]

        cw = CROP_WIDTH * scale_up
        ch = CROP_HEIGHT * scale_up
        margin = 8
        header_h = 38
        footer_h = 26
        strip_w = len(idxs) * (cw + margin) + margin
        strip_h = header_h + ch + footer_h + margin * 2

        strip = Image.new("RGB", (strip_w, strip_h), (26, 28, 32))
        draw = ImageDraw.Draw(strip)
        font_h = PlateRenderer._get_font("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 15, 0)
        font_s = PlateRenderer._get_font("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 11, 0)

        pt_str = sample.plate_text.full_text if sample.plate_text else "NONE"
        hdr = f"{sample.sequence_id} | GT: {pt_str} | Status: {sample.sequence_target_status} (T={t_len})"
        draw.text((margin, 9), hdr, fill=(240, 244, 248), font=font_h)

        for col, t_i in enumerate(idxs):
            x = margin + col * (cw + margin)
            y = header_h
            pil_c = Image.fromarray(sample.crops_rgb[t_i]).resize((cw, ch), Image.Resampling.NEAREST)
            strip.paste(pil_c, (x, y))

            is_for = bool(sample.is_foreign_plate[t_i])
            val = float(sample.frame_valid_mask[t_i])
            border_col = (255, 70, 70) if is_for else ((90, 235, 120) if val > 0.5 else (160, 160, 165))
            draw.rectangle([x, y, x + cw - 1, y + ch - 1], outline=border_col, width=2)

            r_up = float(np.mean(sample.slot_readability[t_i, :7]))
            r_lo = float(np.mean(sample.slot_readability[t_i, 7:]))
            lbl = (
                f"f{sample.source_frame_ids[t_i]}:{sample.view_variants[t_i][:4]} "
                f"c{sample.camera_ids[t_i]} U={r_up:.2f} L={r_lo:.2f}"
            )
            if is_for:
                lbl = f"FOREIGN! c{sample.camera_ids[t_i]}"
            draw.text((x + 2, y + ch + 4), lbl, fill=border_col, font=font_s)

        return strip


def replace_scale(cfg: PlateConfig, new_scale: float) -> PlateConfig:
    from dataclasses import replace
    return replace(cfg, scale=new_scale)
