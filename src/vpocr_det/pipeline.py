"""`VehiclePlateVideoPipeline`: Unified end-to-end camera frame & video stream pipeline.

Connects:
1. `PlateDetNet` (`P2` stride-4 + `P3` + `P4` + optional native-pixel Far-Vehicle Smart-Zoom)
2. `PlateQuadNet` (sub-pixel 4-corner localization + non-plate rejection + exact tight plate-only cutout)
3. `PlateTrackStabilizer` (PMBM existence, camera-motion uncertainty and typed search actions)
4. Optional `CabinRedactor` & per-track `MFPlateOCRSession` (`MF-JPPlateOCR`)
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import time
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vpocr_det.detector import (
    PlateDetNet,
    TemporalPlateDetNet,
    UnifiedTemporalDetORT,
    decode_plate_det_predictions,
    decode_temporal_det_predictions,
)
from vpocr_det.optical_tracker import (
    OpticalPlateRef,
    create_optical_ref,
    quad_basis,
    track_plate_optical,
)
from vpocr_det.quad_cropper import (
    PlateQuadNet,
    PlateQuadTrackNet,
    extract_quad_geometry_meta,
    is_valid_plate_quad,
    refine_and_crop_detections,
    warp_plate_canonical,
    warp_plate_quad,
)
from vpocr_det.tracker import BackgroundCameraMotionEstimator, MultiVehicleKalmanTracker, PlateTrackStabilizer
from vpocr_det.types import (FrameDetectionResult, PlateDetection, VehicleDetection,
                              SearchRequest, ObservationActionMeta, ObservationProvenance,
                              assert_not_observation)


def _bbox_iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    ba = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1.0, aa + ba - inter)


def _bbox_iomin_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    ba = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1.0, min(aa, ba))


def _vehicle_containment_nms(vehicles: list[VehicleDetection]) -> list[VehicleDetection]:
    """Suppress duplicate, multi-vehicle wrapper, and cab-inside-truck sub-box detections."""
    if len(vehicles) <= 1:
        return list(vehicles)
    ordered = sorted(vehicles, key=lambda v: -float(v.score))
    kept: list[VehicleDetection] = []
    for cand in ordered:
        cx0, cy0, cx1, cy1 = [float(x) for x in cand.bbox_xyxy]
        cw = max(1.0, cx1 - cx0)
        ch = max(1.0, cy1 - cy0)
        c_area = cw * ch
        suppress = False
        contained_keepers: list[VehicleDetection] = []
        for keeper in kept:
            kx0, ky0, kx1, ky1 = [float(x) for x in keeper.bbox_xyxy]
            kw = max(1.0, kx1 - kx0)
            kh = max(1.0, ky1 - ky0)
            k_area = kw * kh
            iou = _bbox_iou_xyxy(cand.bbox_xyxy, keeper.bbox_xyxy)
            iomin = _bbox_iomin_xyxy(cand.bbox_xyxy, keeper.bbox_xyxy)
            if iou >= 0.55:
                suppress = True
                break
            # Cab/sub-box inside whole vehicle suppression
            if (
                iomin >= 0.82
                and float(cand.score) < float(keeper.score) * 0.90
                and c_area < 0.55 * k_area
            ):
                tol = 0.14 * max(kw, kh)
                shares_bottom = abs(cy1 - ky1) <= tol
                shares_left = abs(cx0 - kx0) <= tol
                shares_right = abs(cx1 - kx1) <= tol
                interior_upper = (cy0 >= ky0 - 0.04 * kh) and (cy1 <= ky0 + 0.82 * kh)
                if shares_bottom or shares_left or shares_right or interior_upper:
                    suppress = True
                    break
            # Track keepers enclosed inside a larger candidate wrapper
            if c_area > k_area * 1.15 and iomin >= 0.80:
                contained_keepers.append(keeper)
        if not suppress and len(contained_keepers) >= 2:
            # Drop multi-object wrapper box covering 2+ separate vehicles
            sum_k = sum(
                max(1.0, (k.bbox_xyxy[2] - k.bbox_xyxy[0]) * (k.bbox_xyxy[3] - k.bbox_xyxy[1]))
                for k in contained_keepers
            )
            if sum_k >= 0.45 * c_area:
                suppress = True
        if not suppress:
            kept.append(cand)
    return kept


def _compute_inner_scene_features(small_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute inner-70% luminance patch and normalized 3x8 RGB Bhattacharyya histogram."""
    gh, gw = small_rgb.shape[:2]
    x0 = int(round(0.15 * gw))
    x1 = max(x0 + 2, int(round(0.85 * gw)))
    y0 = int(round(0.15 * gh))
    y1 = max(y0 + 2, int(round(0.85 * gh)))
    inner = small_rgb[y0:y1, x0:x1, :3].astype(np.float32)
    inner_y = 0.299 * inner[:, :, 0] + 0.587 * inner[:, :, 1] + 0.114 * inner[:, :, 2]
    bins = np.clip((inner * (8.0 / 256.0)).astype(np.int32), 0, 7)
    n_pix = float(max(1, inner.shape[0] * inner.shape[1]))
    hist = np.zeros((24,), dtype=np.float32)
    for ch in range(3):
        hist[ch * 8 : (ch + 1) * 8] = np.bincount(bins[:, :, ch].reshape(-1), minlength=8).astype(np.float32)
    hist = np.sqrt(hist / n_pix)
    h_nrm = float(np.linalg.norm(hist))
    if h_nrm > 1e-6:
        hist /= h_nrm
    return inner_y, hist


def _is_inner_scene_cut(inner_sad: float, inner_hist_corr: float) -> bool:
    return bool(
        (inner_sad >= 34.0 and inner_hist_corr < 0.86)
        or (inner_sad >= 22.0 and inner_hist_corr < 0.72)
        or (inner_hist_corr < 0.58 and inner_sad >= 14.0)
    )


class _InnerSceneCutMonitor:
    """Framed-TV-immune inner-70% scene-cut monitor (`< 0.3 ms` CPU)."""

    def __init__(self, grid_hw: tuple[int, int] = (80, 144)) -> None:
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self._prev_y: np.ndarray | None = None
        self._prev_hist: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_y = None
        self._prev_hist = None

    def step(self, frame_rgb: np.ndarray) -> tuple[bool, float, float]:
        if frame_rgb.ndim != 3 or frame_rgb.size == 0:
            return False, 0.0, 1.0
        gh, gw = self.grid_hw
        small = cv2.resize(frame_rgb, (gw, gh), interpolation=cv2.INTER_AREA)
        cur_y, cur_hist = _compute_inner_scene_features(small)
        if self._prev_y is None or self._prev_hist is None or self._prev_y.shape != cur_y.shape:
            self._prev_y = cur_y
            self._prev_hist = cur_hist
            return False, 0.0, 1.0
        sad = float(np.mean(np.abs(cur_y - self._prev_y)))
        corr = float(np.clip(np.dot(cur_hist, self._prev_hist), 0.0, 1.0))
        self._prev_y = cur_y
        self._prev_hist = cur_hist
        return _is_inner_scene_cut(sad, corr), sad, corr


def _handle_scene_cut(owner: Any, retired_this_frame: dict[int, Any]) -> None:
    """Flush vehicle/plate tracks & recurrent hidden state on a hard scene cut, sealing mature OCR episodes."""
    for tid, sess in list(owner._ocr_sessions.items()):
        dec = _finish_cached(owner, tid, sess)
        owner._retired_decisions[tid] = dec
        owner._terminal_decisions.pop(tid, None)
        retired_this_frame[tid] = dec
        while len(owner._retired_decisions) > owner.max_retired_history:
            owner._retired_decisions.pop(next(iter(owner._retired_decisions)))
    owner._ocr_sessions.clear()
    owner.tracker.flush_for_scene_cut()
    owner.vehicle_tracker.flush_for_scene_cut()
    owner._det_hidden_state = None
    owner._camera_motion_estimator.reset()
    owner._prev_corners.clear()
    owner._opt_refs.clear()
    owner._sr_states.clear()
    if hasattr(owner, "_track_corner_stats"):
        owner._track_corner_stats.clear()
    if hasattr(owner, "_veh_zoom_cooldown"):
        owner._veh_zoom_cooldown.clear()
    if hasattr(owner, "_dormant_reappear_cooldown"):
        owner._dormant_reappear_cooldown.clear()
    if hasattr(owner, "_cached_vehicles"):
        owner._cached_vehicles.clear()
    if hasattr(owner, "_cached_vehicle_bboxes"):
        owner._cached_vehicle_bboxes.clear()


def _session_has_confirmed_high_prec_plate(owner: Any, plate_track_id: int) -> bool:
    sess = owner._ocr_sessions.get(plate_track_id)
    if sess is None:
        return False
    if not _collecting(sess):
        dec = owner._terminal_decisions.get(plate_track_id)
        return dec is not None and getattr(dec, "status", None) == "ACCEPTED"
    try:
        mem_p = sess._state_arrays()[1]
        mean_p = float(np.maximum(mem_p, 0.0).mean())
        ep_prob = float(sess._qual_acc.episode_plate_prob)
        return mean_p >= 0.16 and ep_prob >= 0.72
    except Exception:
        return False


def _run_full_face_vehicle_zoom(
    owner: Any,
    frame_rgb: np.ndarray,
    tracked_vehicles: list[VehicleDetection],
    existing_plates: list[PlateDetection],
    frame_key: str,
) -> list[PlateDetection]:
    """Full-face multi-scale vehicle Smart-Zoom (10% padded up to 4x mag) with extreme oblique aspect [0.48, 6.2]."""
    if (
        not owner.enable_far_zoom
        or not tracked_vehicles
        or not hasattr(owner.det_model, "forward_step")
    ):
        return []
    h_orig, w_orig = frame_rgb.shape[:2]
    if h_orig < 24 or w_orig < 24:
        return []

    # Tick per-vehicle zoom cooldowns
    cooldowns: dict[int, int] = owner._veh_zoom_cooldown
    for vid in list(cooldowns.keys()):
        rem = cooldowns[vid] - 1
        if rem <= 0:
            cooldowns.pop(vid, None)
        else:
            cooldowns[vid] = rem

    # Vehicles that already own a confirmed high-precision OCR plate track need zero extra zoom
    confirmed_vids: set[int] = set()
    for pt in getattr(owner.tracker, "_tracks", ()):
        if getattr(pt, "confirmed", False) and getattr(pt, "parent_vehicle_track_id", None) is not None:
            if _session_has_confirmed_high_prec_plate(owner, pt.track_id):
                confirmed_vids.add(int(pt.parent_vehicle_track_id))

    zoom_tile_hw = (256, 384)
    max_zoom_mag = 4.0
    max_tiles = 2
    in_h, in_w = owner.input_hw
    det_scale = min(float(in_w) / float(w_orig), float(in_h) / float(h_orig))
    tw_lim = max(32, int(math.floor(384.0 / max(1e-6, 1.22 * det_scale))))
    th_lim = max(24, int(math.floor(256.0 / max(1e-6, 1.22 * det_scale))))
    candidates_v: list[VehicleDetection] = []
    for vd in sorted(tracked_vehicles, key=lambda v: -float(v.score)):
        vid = vd.vehicle_track_id
        if vid is not None and (vid in confirmed_vids or cooldowns.get(vid, 0) > 0):
            continue
        if vid is None and float(vd.score) < 0.55:
            continue
        vx0, vy0, vx1, vy1 = [float(x) for x in vd.bbox_xyxy]
        vw = vx1 - vx0
        vh = vy1 - vy0
        if vw < 14.0 or vh < 14.0:
            continue
        # Skip if Pass-1/Pass-2 already detected a plate anywhere on the full vehicle face
        has_plate = False
        for p in existing_plates:
            px0, py0, px1, py1 = p.bbox_xyxy
            pcx = 0.5 * (px0 + px1)
            pcy = 0.5 * (py0 + py1)
            if (
                (vx0 - 0.06 * vw) <= pcx <= (vx1 + 0.06 * vw)
                and (vy0 - 0.04 * vh) <= pcy <= (vy1 + 0.14 * vh)
                and float(p.score) >= 0.35
            ):
                has_plate = True
                break
        if has_plate:
            continue
        # Avoid re-running a duplicate Pass-2 tile on small vehicles that decode_temporal_det_predictions
        # already full-face zoomed on this exact frame; prioritize wide/large vehicles skipped or bumper-clipped
        # by Pass-2 (mid-door vans, heavy-machinery side/hood mounts).
        pass2_already_full_face = (
            vw * det_scale < 240.0
            and math.ceil(1.22 * vw) <= tw_lim
            and math.ceil(1.22 * vh) <= th_lim
        )
        if pass2_already_full_face:
            continue
        candidates_v.append(vd)
        if len(candidates_v) >= max_tiles:
            break

    if not candidates_v:
        return []

    crops: list[np.ndarray] = []
    metas: list[tuple[VehicleDetection, float, int, int, int, int, int, int]] = []
    ht, wt = zoom_tile_hw
    min_w = min(w_orig, max(24, int(math.ceil(float(wt) / max_zoom_mag))))
    min_h = min(h_orig, max(24, int(math.ceil(float(ht) / max_zoom_mag))))
    for vd in candidates_v:
        vx0, vy0, vx1, vy1 = [float(x) for x in vd.bbox_xyxy]
        vw = vx1 - vx0
        vh = vy1 - vy0
        rx0 = int(math.floor(vx0 - 0.10 * vw))
        ry0 = int(math.floor(vy0 - 0.10 * vh))
        rx1 = int(math.ceil(vx1 + 0.10 * vw))
        ry1 = int(math.ceil(vy1 + 0.10 * vh))
        if rx1 - rx0 < min_w:
            cx_w = 0.5 * (rx0 + rx1)
            rx0 = int(round(cx_w - 0.5 * min_w))
            rx1 = rx0 + min_w
        if ry1 - ry0 < min_h:
            cy_h = 0.5 * (ry0 + ry1)
            ry0 = int(round(cy_h - 0.5 * min_h))
            ry1 = ry0 + min_h
        win_w = min(w_orig, max(24, rx1 - rx0))
        win_h = min(h_orig, max(24, ry1 - ry0))
        rx0 = max(0, min(w_orig - win_w, rx0))
        ry0 = max(0, min(h_orig - win_h, ry0))
        rx1 = min(w_orig, rx0 + win_w)
        ry1 = min(h_orig, ry0 + win_h)
        if rx1 - rx0 < 24 or ry1 - ry0 < 24:
            continue
        sub = frame_rgb[ry0:ry1, rx0:rx1]
        sh, sw = sub.shape[:2]
        z_sc = min(float(wt) / float(sw), float(ht) / float(sh), max_zoom_mag)
        if det_scale >= 0.90 and z_sc <= det_scale * 1.15:
            continue
        nw = max(1, min(wt, int(round(sw * z_sc))))
        nh = max(1, min(ht, int(round(sh * z_sc))))
        interp = cv2.INTER_AREA if z_sc < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(sub, (nw, nh), interpolation=interp)
        canvas = np.full((ht, wt, 3), 114, dtype=np.uint8)
        z_px = (wt - nw) // 2
        z_py = (ht - nh) // 2
        canvas[z_py : z_py + nh, z_px : z_px + nw] = resized
        crops.append(canvas)
        metas.append((vd, z_sc, z_px, z_py, rx0, ry0, rx1, ry1))

    if not crops:
        return []

    z_batch = (
        torch.from_numpy(np.stack(crops, axis=0))
        .to(owner.dev, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        / 255.0
    )
    graph_hw = tuple(getattr(owner.det_model, "input_hw", zoom_tile_hw))
    if graph_hw != zoom_tile_hw:
        z_batch = F.interpolate(z_batch, size=graph_hw, mode="bilinear", align_corners=False)
    z_out = owner.det_model.forward_step(z_batch, h_prev=None)
    z_all_probs = torch.sigmoid(z_out["cls_logits"]).cpu().numpy()
    z_probs = z_all_probs[:, :, 0]
    z_vprobs = z_all_probs[:, :, 1]
    z_sprobs = z_all_probs[:, :, 2] if z_all_probs.shape[-1] > 2 else np.zeros_like(z_probs)
    z_boxes_np = z_out["pred_boxes"].cpu().numpy().copy()
    z_corners_np = z_out["pred_corners"].cpu().numpy().copy()
    if graph_hw != zoom_tile_hw:
        tile_sc = np.array([float(wt) / float(graph_hw[1]), float(ht) / float(graph_hw[0])], dtype=np.float32)
        z_boxes_np *= np.tile(tile_sc, 2)
        z_corners_np *= tile_sc

    score_thr = max(float(owner.plate_thresh), 0.24)
    extra_plates: list[PlateDetection] = []
    for k_i, (vd, z_sc, z_px, z_py, rx0, ry0, rx1, ry1) in enumerate(metas):
        mask = (z_probs[k_i] >= score_thr) & (z_probs[k_i] > z_vprobs[k_i]) & (z_probs[k_i] > z_sprobs[k_i])
        found_any = False
        if np.any(mask):
            zb = z_boxes_np[k_i, mask].copy()
            zc = z_corners_np[k_i, mask].copy()
            zs = z_probs[k_i, mask]
            inv_z = 1.0 / max(1e-6, z_sc)
            zb[:, [0, 2]] = np.clip((zb[:, [0, 2]] - z_px) * inv_z + rx0, 0.0, float(w_orig))
            zb[:, [1, 3]] = np.clip((zb[:, [1, 3]] - z_py) * inv_z + ry0, 0.0, float(h_orig))
            zc[:, :, 0] = np.clip((zc[:, :, 0] - z_px) * inv_z + rx0, 0.0, float(w_orig))
            zc[:, :, 1] = np.clip((zc[:, :, 1] - z_py) * inv_z + ry0, 0.0, float(h_orig))
            zbw = zb[:, 2] - zb[:, 0]
            zbh = zb[:, 3] - zb[:, 1]
            aspect = zbw / np.maximum(1e-3, zbh)
            zqw = zc[:, :, 0].max(axis=-1) - zc[:, :, 0].min(axis=-1)
            zqh = zc[:, :, 1].max(axis=-1) - zc[:, :, 1].min(axis=-1)
            valid = (
                (zbw >= 5.0)
                & (zbh >= 2.8)
                & (aspect >= 0.48)
                & (aspect <= 6.20)
                & (zqw >= 0.34 * zbw)
                & (zqw <= 1.95 * zbw)
                & (zqh >= 0.34 * zbh)
                & (zqh <= 1.95 * zbh)
            )
            if np.any(valid):
                vb, vc, vs = zb[valid], zc[valid], zs[valid]
                order = np.argsort(-vs)
                for idx in order[:4]:
                    box_t = (float(vb[idx, 0]), float(vb[idx, 1]), float(vb[idx, 2]), float(vb[idx, 3]))
                    if any(_bbox_iou_xyxy(box_t, ep.bbox_xyxy) >= 0.45 for ep in (*existing_plates, *extra_plates)):
                        continue
                    extra_plates.append(
                        PlateDetection(
                            bbox_xyxy=box_t,
                            score=float(vs[idx]),
                            corners_xy=vc[idx].astype(np.float32),
                            raw_det_score=float(vs[idx]),
                            vehicle_bbox_xyxy=vd.bbox_xyxy,
                            parent_vehicle_track_id=vd.vehicle_track_id,
                            provenance=ObservationProvenance.DETECTOR_FAR_ZOOM,
                            frame_key=frame_key,
                        )
                    )
                    found_any = True
        if not found_any and vd.vehicle_track_id is not None:
            cooldowns[int(vd.vehicle_track_id)] = 6
    return extra_plates


def _suppress_stationary_corner_logos(
    owner: Any,
    plates: list[PlateDetection],
    img_hw: tuple[int, int],
) -> list[PlateDetection]:
    """Suppress unowned static screen-corner TV logos without affecting real parked/moving vehicles."""
    h_f, w_f = float(img_hw[0]), float(img_hw[1])
    if h_f <= 0.0 or w_f <= 0.0:
        return plates
    stats_map: dict[int, dict[str, Any]] = owner._track_corner_stats
    kept: list[PlateDetection] = []
    discarded_ids: set[int] = set()
    for p in plates:
        tid = p.track_id
        if tid is None or p.is_coasted:
            kept.append(p)
            continue
        cx = 0.5 * (float(p.bbox_xyxy[0]) + float(p.bbox_xyxy[2]))
        cy = 0.5 * (float(p.bbox_xyxy[1]) + float(p.bbox_xyxy[3]))
        if p.plate_null_llr is not None and math.isfinite(float(p.plate_null_llr)):
            llr = float(p.plate_null_llr)
        else:
            qp = min(1.0 - 1e-6, max(1e-6, float(p.quad_is_plate_prob)))
            llr = math.log(qp) - math.log1p(-qp)
        owned = p.parent_vehicle_track_id is not None
        st = stats_map.get(tid)
        if st is None:
            st = {
                "first_cx": cx,
                "first_cy": cy,
                "max_disp": 0.0,
                "frames": 1,
                "max_llr": llr,
                "ever_owned": owned,
            }
            stats_map[tid] = st
        else:
            disp = math.hypot(cx - float(st["first_cx"]), cy - float(st["first_cy"]))
            if disp > float(st["max_disp"]):
                st["max_disp"] = disp
            st["frames"] = int(st["frames"]) + 1
            if llr > float(st["max_llr"]):
                st["max_llr"] = llr
            if owned:
                st["ever_owned"] = True
        in_corner = (cx < 0.16 * w_f or cx > 0.84 * w_f) and (cy < 0.16 * h_f or cy > 0.84 * h_f)
        if (
            not bool(st["ever_owned"])
            and in_corner
            and int(st["frames"]) >= 6
            and float(st["max_disp"]) < 1.5
            and float(st["max_llr"]) < 2.0
        ):
            discarded_ids.add(tid)
            continue
        kept.append(p)

    if discarded_ids:
        owner.tracker._tracks = [t for t in owner.tracker._tracks if t.track_id not in discarded_ids]
        owner.tracker._dormant_tracks = [t for t in owner.tracker._dormant_tracks if t.track_id not in discarded_ids]
        for tid in discarded_ids:
            stats_map.pop(tid, None)
            owner._prev_corners.pop(tid, None)
            owner._opt_refs.pop(tid, None)
            owner._sr_states.pop(tid, None)
            sess = owner._ocr_sessions.get(tid)
            if sess is not None and _collecting(sess):
                owner._ocr_sessions.pop(tid, None)
    return kept


def _enforce_patent_clean_ocr_session_config(cfg: Any | None) -> Any | None:
    """Enforce `JP7029732B2` patent-clean invariant (`0` intermediate OCR readouts during collection)."""
    if cfg is None:
        return None
    if bool(getattr(cfg, "decode_intermediate", False)) or bool(
        getattr(cfg, "auto_emit_on_saturation", False)
    ):
        raise ValueError(
            "JP7029732B2 patent-clean pipeline forbids intermediate OCR readout options "
            "(decode_intermediate / auto_emit_on_saturation must be False)"
        )
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        overrides = {
            k: False
            for k in ("decode_intermediate", "auto_emit_on_saturation")
            if hasattr(cfg, k)
        }
        return dataclasses.replace(cfg, **overrides)
    return cfg


def _resolve_patent_clean_v2_session_config(cfg: Any | None, session_cfg_cls: type[Any]) -> Any:
    checked = _enforce_patent_clean_ocr_session_config(cfg)
    if isinstance(checked, session_cfg_cls):
        return checked
    return session_cfg_cls(decode_intermediate=False, auto_emit_on_saturation=False)


class _Stage2ORTAdapter(torch.nn.Module):
    """Expose an exported ORT session through the eager Stage-2 tensor contract."""

    def __init__(self, session: Any) -> None:
        torch.nn.Module.__init__(self)
        self.session = session
        self.is_dual_track_net = len(session.get_inputs()) > 1

    def forward(self, fovea: torch.Tensor, surround: torch.Tensor | None = None,
                policy: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feed = {meta.name: value.detach().cpu().numpy()
                for meta, value in zip(self.session.get_inputs(), (fovea, surround, policy))
                if value is not None}
        out = {meta.name: torch.from_numpy(value)
               for meta, value in zip(self.session.get_outputs(), self.session.run(None, feed))}
        if "plate_null_llr" in out:
            out.setdefault("is_plate_logit", out["plate_null_llr"])
        # Older Gen-6 exports omit auxiliary side/quality heads. Missing heads
        # are uninformative and never replace the exported plate/null evidence.
        batch = fovea.shape[0]
        out.setdefault("side_logits", torch.zeros(batch, 3))
        out.setdefault("quality_logit", torch.full((batch,), -torch.inf))
        return out


class _Stage2DualORTAdapter(_Stage2ORTAdapter, PlateQuadTrackNet):
    """Dispatch the exported sampling-policy input through the dual cropper."""


def _inference_model(model: Any, device: torch.device, *, stage2: bool = False) -> Any:
    if stage2 and hasattr(model, "run") and hasattr(model, "get_inputs"):
        adapter = _Stage2DualORTAdapter if len(model.get_inputs()) > 1 else _Stage2ORTAdapter
        model = adapter(model)
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
    return model.eval()


def _collecting(session: Any) -> bool:
    from vpocr_ocr.session_v2 import SessionStateV2
    return getattr(session, "state", SessionStateV2.COLLECTING) == SessionStateV2.COLLECTING


def _finish_cached(owner: Any, key: int, session: Any) -> Any:
    if key not in owner._terminal_decisions:
        owner._terminal_decisions[key] = session.finish()
    return owner._terminal_decisions[key]


def _relink_sessions(owner: Any, pairs: list[tuple[int, int]]) -> None:
    """Transfer one episode, merging disjoint factors with exactly one prior."""
    for old_id, restored_id in pairs:
        source = owner._ocr_sessions.get(old_id)
        target = owner._ocr_sessions.get(restored_id)
        if source is not None:
            if target is None:
                owner._ocr_sessions[restored_id] = source
                if old_id in owner._terminal_decisions:
                    owner._terminal_decisions[restored_id] = owner._terminal_decisions[old_id]
            elif _collecting(target) and _collecting(source):
                merge = getattr(target, "merge_disjoint_evidence", None)
                if callable(merge):
                    merge(source)
            elif _collecting(target):
                owner._ocr_sessions[restored_id] = source
                owner._terminal_decisions[restored_id] = _finish_cached(owner, old_id, source)
        owner._ocr_sessions.pop(old_id, None)
        owner._terminal_decisions.pop(old_id, None)
        if old_id in owner._prev_corners:
            owner._prev_corners[restored_id] = owner._prev_corners.pop(old_id)
        if hasattr(owner, "_opt_refs") and old_id in owner._opt_refs:
            owner._opt_refs[restored_id] = owner._opt_refs.pop(old_id)
        if hasattr(owner, "_sr_states") and old_id in owner._sr_states:
            owner._sr_states.setdefault(restored_id, owner._sr_states.pop(old_id))
            owner._sr_states.pop(old_id, None)
        if hasattr(owner, "_track_corner_stats") and old_id in owner._track_corner_stats:
            owner._track_corner_stats.setdefault(restored_id, owner._track_corner_stats.pop(old_id))
            owner._track_corner_stats.pop(old_id, None)


def _search_inputs(
    requests: list[SearchRequest],
    frame_key: str,
    *,
    frame_rgb: np.ndarray | None = None,
    opt_refs: dict[int, OpticalPlateRef] | None = None,
    prev_corners: dict[int, np.ndarray] | None = None,
) -> list[PlateDetection]:
    """Zero-prior geometry carriers for Stage-2 execution, never PMBM updates.

    The current cropper accepts geometry carriers, so adaptation of score-free
    requests is confined to this boundary. Only neural outputs are observations.
    When an optical template exists for `request.target_track_id`, pre-aligns
    the search ROI onto the sub-pixel IC-LK + ZNCC optical plate position.
    """
    carriers = []
    provenance = getattr(ObservationProvenance, "STAGE2_SEARCH", ObservationProvenance.SEARCH_MEASUREMENT)
    for request in requests:
        assert_not_observation(request)
        x0, y0, x1, y1 = request.roi_xyxy
        c_init = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        tid = getattr(request, "target_track_id", None)
        if (
            frame_rgb is not None
            and opt_refs is not None
            and tid is not None
            and tid in opt_refs
        ):
            init_q = prev_corners.get(tid) if prev_corners is not None else None
            if init_q is not None:
                # Shift previous 4-corner quad to the predicted search box center
                q_cx = float(init_q[:, 0].mean())
                q_cy = float(init_q[:, 1].mean())
                r_cx = 0.5 * (x0 + x1)
                r_cy = 0.5 * (y0 + y1)
                init_q = init_q + np.array([[r_cx - q_cx, r_cy - q_cy]], dtype=np.float32)
            rad_x = max(18.0, 0.45 * (x1 - x0))
            rad_y = max(12.0, 0.45 * (y1 - y0))
            ok_opt, _, opt_q = track_plate_optical(
                opt_refs[tid],
                frame_rgb,
                init_q,
                search_radius_px=rad_x,
                search_radius_y=rad_y,
                allow_scale_search=True,
                min_zncc=0.56,
            )
            if ok_opt and opt_q is not None:
                h_f, w_f = frame_rgb.shape[:2]
                ox0 = float(np.clip(opt_q[:, 0].min(), 0.0, float(w_f)))
                oy0 = float(np.clip(opt_q[:, 1].min(), 0.0, float(h_f)))
                ox1 = float(np.clip(opt_q[:, 0].max(), 0.0, float(w_f)))
                oy1 = float(np.clip(opt_q[:, 1].max(), 0.0, float(h_f)))
                if ox1 - ox0 >= 8.0 and oy1 - oy0 >= 4.0:
                    x0, y0, x1, y1 = ox0, oy0, ox1, oy1
                    c_init = opt_q
        carriers.append(PlateDetection((x0, y0, x1, y1), 0.0,
            c_init,
            raw_det_score=0.0, provenance=provenance, frame_key=frame_key))
    return carriers


def _apply_optical_refinement_and_template_update(
    frame_rgb: np.ndarray,
    plates: list[PlateDetection],
    opt_refs: dict[int, OpticalPlateRef],
    prev_corners: dict[int, np.ndarray],
) -> None:
    """Fuse sub-pixel IC-LK optical corners on observed tracks and maintain crisp optical templates."""
    h_f, w_f = frame_rgb.shape[:2]
    frame_textured: bool | None = None
    for p in plates:
        tid = p.track_id
        if tid is None:
            continue
        ref = opt_refs.get(tid)
        if p.is_coasted:
            # Update visual coasted overlay geometry with sub-pixel lock while keeping ocr_crop_rgb=None
            if ref is not None:
                init_q = prev_corners.get(tid, p.corners_xy)
                c_min_z = 0.50 if ref.basis.width < 36.0 else 0.62
                ok_c, _, opt_q = track_plate_optical(
                    ref, frame_rgb, init_q, search_radius_px=28.0, allow_scale_search=True, min_zncc=c_min_z
                )
                if ok_c and opt_q is not None:
                    p.corners_xy = opt_q
                    p.bbox_xyxy = (
                        float(np.clip(opt_q[:, 0].min(), 0.0, float(w_f))),
                        float(np.clip(opt_q[:, 1].min(), 0.0, float(h_f))),
                        float(np.clip(opt_q[:, 0].max(), 0.0, float(w_f))),
                        float(np.clip(opt_q[:, 1].max(), 0.0, float(h_f))),
                    )
                    prev_corners[tid] = opt_q.copy()
            continue

        q_cnn = p.raw_corners_xy if p.raw_corners_xy is not None else p.corners_xy
        if ref is not None and not p.quad_edge_touch:
            low_res_ref = ref.basis.width < 36.0
            q_init = p.corners_xy if low_res_ref else q_cnn
            ok_f, zncc_f, opt_q = track_plate_optical(
                ref,
                frame_rgb,
                q_init,
                search_radius_px=9.0 if low_res_ref else 12.0,
                search_radius_y=6.0 if low_res_ref else 8.0,
                min_zncc=0.48 if low_res_ref else 0.72,
                max_iter=5,
            )
            if ok_f and opt_q is not None:
                b_base = quad_basis(q_init)
                corner_diff = float(np.linalg.norm(opt_q - q_init, axis=-1).mean())
                if corner_diff <= max(2.5, 0.24 * b_base.width):
                    w_opt = (
                        min(0.55, max(0.22, (zncc_f - 0.45) * 1.10))
                        if low_res_ref
                        else min(0.35, max(0.15, (zncc_f - 0.70) * 1.15))
                    )
                    q_anchor = (0.65 * p.corners_xy + 0.35 * q_cnn).astype(np.float32) if low_res_ref else q_cnn
                    q_fused = ((1.0 - w_opt) * q_anchor + w_opt * opt_q).astype(np.float32)
                    if is_valid_plate_quad(q_fused):
                        p.corners_xy = q_fused
                        p.raw_corners_xy = q_fused.copy()
                        p.bbox_xyxy = (
                            float(np.clip(q_fused[:, 0].min(), 0.0, float(w_f))),
                            float(np.clip(q_fused[:, 1].min(), 0.0, float(h_f))),
                            float(np.clip(q_fused[:, 0].max(), 0.0, float(w_f))),
                            float(np.clip(q_fused[:, 1].max(), 0.0, float(h_f))),
                        )
                        p.ocr_crop_rgb = None
                        p.tight_crop_rgb = None
                        q_cnn = q_fused
            elif low_res_ref:
                if frame_textured is None:
                    frame_textured = float(frame_rgb[::4, ::4].std()) > 2.0
                if frame_textured and is_valid_plate_quad(p.corners_xy):
                    q_kal = np.asarray(p.corners_xy, dtype=np.float32)
                    p.raw_corners_xy = q_kal.copy()
                    p.bbox_xyxy = (
                        float(np.clip(q_kal[:, 0].min(), 0.0, float(w_f))),
                        float(np.clip(q_kal[:, 1].min(), 0.0, float(h_f))),
                        float(np.clip(q_kal[:, 0].max(), 0.0, float(w_f))),
                        float(np.clip(q_kal[:, 1].max(), 0.0, float(h_f))),
                    )
                    p.ocr_crop_rgb = None
                    p.tight_crop_rgb = None
                    q_cnn = q_kal

        if not p.quad_edge_touch and float(p.quad_is_plate_prob) >= 0.58:
            cand_ref = create_optical_ref(frame_rgb, q_cnn)
            if cand_ref is not None:
                if ref is None or cand_ref.sharpness >= 0.92 * ref.sharpness or cand_ref.basis.width >= 1.12 * ref.basis.width:
                    opt_refs[tid] = cand_ref


class _CanonicalDrizzleState:
    """Rolling canonical 80x160 RGB Drizzle super-resolution state (pure ISP pixel alignment, zero character ops)."""

    __slots__ = ("sr_num", "sr_den", "sr_ref_luma", "sr_eff_views", "sr_width_ema", "_tpl", "_jac", "_inv_h")

    def __init__(self) -> None:
        self.sr_num = np.zeros((3, 80, 160), dtype=np.float32)
        self.sr_den = np.zeros((80, 160), dtype=np.float32)
        self.sr_ref_luma = np.zeros((40, 80), dtype=np.float32)
        self.sr_eff_views: float = 0.0
        self.sr_width_ema: float = 0.0
        self._tpl: np.ndarray | None = None
        self._jac: np.ndarray | None = None
        self._inv_h: np.ndarray | None = None

    def _update_ic_tables(self, luma_40x80: np.ndarray) -> None:
        self.sr_ref_luma = luma_40x80.astype(np.float32, copy=False)
        std = float(np.std(self.sr_ref_luma))
        if std < 0.012:
            self._tpl = None
            self._jac = None
            self._inv_h = None
            return
        tpl = ((self.sr_ref_luma - float(np.mean(self.sr_ref_luma))) / std).astype(np.float32)
        gx = np.empty_like(tpl)
        gy = np.empty_like(tpl)
        gx[:, 1:-1] = 0.5 * (tpl[:, 2:] - tpl[:, :-2])
        gx[:, 0] = tpl[:, 1] - tpl[:, 0]
        gx[:, -1] = tpl[:, -1] - tpl[:, -2]
        gy[1:-1, :] = 0.5 * (tpl[2:, :] - tpl[:-2, :])
        gy[0, :] = tpl[1, :] - tpl[0, :]
        gy[-1, :] = tpl[-1, :] - tpl[-2, :]
        xc = ((np.arange(80, dtype=np.float32) + 0.5) / 80.0 - 0.5)[None, :]
        yc = ((np.arange(40, dtype=np.float32) + 0.5) / 40.0 - 0.5)[:, None]
        jac = np.stack([gx.reshape(-1), gy.reshape(-1), (gx * xc).reshape(-1), (gy * yc).reshape(-1)], axis=-1)
        h_mat = jac.T.astype(np.float64) @ jac.astype(np.float64)
        h_mat[0, 0] += 2.0
        h_mat[1, 1] += 2.0
        h_mat[2, 2] += 80.0
        h_mat[3, 3] += 60.0
        try:
            self._inv_h = np.linalg.inv(h_mat)
            self._tpl = tpl.reshape(-1)
            self._jac = jac
        except np.linalg.LinAlgError:
            self._tpl = None
            self._jac = None
            self._inv_h = None

    def _align_corners_iclk(self, frame_rgb: np.ndarray, corners_xy: np.ndarray, margin_ratio: float) -> tuple[float, np.ndarray]:
        if self._tpl is None or self._jac is None or self._inv_h is None:
            return -1.0, corners_xy
        basis = quad_basis(corners_xy)
        mx = float(margin_ratio) * 80.0
        my = float(margin_ratio) * 40.0
        dst_40 = np.array(
            [[mx - 0.5, my - 0.5], [80.0 - 0.5 - mx, my - 0.5], [80.0 - 0.5 - mx, 40.0 - 0.5 - my], [mx - 0.5, 40.0 - 0.5 - my]],
            dtype=np.float32,
        )
        tu, tv, su, sv = 0.0, 0.0, 1.0, 1.0
        n_pix = 3200.0

        def _warp_luma(c_tu: float, c_tv: float, c_su: float, c_sv: float) -> tuple[np.ndarray, np.ndarray]:
            u0, u1 = c_tu - 0.5 * c_su, c_tu + 0.5 * c_su
            v0, v1 = c_tv - 0.5 * c_sv, c_tv + 0.5 * c_sv
            uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float32)
            u, v = uv[:, 0], uv[:, 1]
            q = np.stack(
                [
                    basis.mx + u * basis.aux + v * basis.avx + u * v * basis.auvx,
                    basis.my + u * basis.auy + v * basis.avy + u * v * basis.auvy,
                ],
                axis=-1,
            ).astype(np.float32)
            H_40 = cv2.getPerspectiveTransform(q, dst_40)
            patch_rgb = cv2.warpPerspective(frame_rgb, H_40, (80, 40), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            lum = (
                0.299 * patch_rgb[:, :, 0].astype(np.float32)
                + 0.587 * patch_rgb[:, :, 1].astype(np.float32)
                + 0.114 * patch_rgb[:, :, 2].astype(np.float32)
            ) / 255.0
            return lum.reshape(-1), q

        best_tu, best_tv, best_zncc = 0.0, 0.0, -1.0
        step_u = min(0.08, 1.2 / max(10.0, basis.width))
        step_v = min(0.10, 1.0 / max(5.0, basis.height))
        for iv in (-1, 0, 1):
            for iu in (-1, 0, 1):
                lum, _ = _warp_luma(iu * step_u, iv * step_v, 1.0, 1.0)
                d = lum - float(np.mean(lum))
                sq = float(np.dot(d, d))
                z = float(np.dot(d, self._tpl) / math.sqrt(sq * n_pix)) if sq > 1e-4 * n_pix else -1.0
                if z > best_zncc:
                    best_zncc = z
                    best_tu, best_tv = iu * step_u, iv * step_v
        if best_zncc < 0.32:
            return best_zncc, corners_xy
        tu, tv = best_tu, best_tv
        eff_span_u = 1.0 - 2.0 * float(margin_ratio)
        eff_span_v = 1.0 - 2.0 * float(margin_ratio)
        for _ in range(4):
            lum, _ = _warp_luma(tu, tv, su, sv)
            d = lum - float(np.mean(lum))
            std = math.sqrt(float(np.dot(d, d)) / n_pix)
            if std < 0.01:
                return -1.0, corners_xy
            err = (d / std - self._tpl).astype(np.float64)
            dp = self._inv_h @ (self._jac.T.astype(np.float64) @ err)
            dx_g = float(np.clip(dp[0], -2.5, 2.5))
            dy_g = float(np.clip(dp[1], -2.0, 2.0))
            d_su = float(np.clip(dp[2], -0.06, 0.06))
            d_sv = float(np.clip(dp[3], -0.06, 0.06))
            tu -= su * (dx_g / (80.0 * eff_span_u))
            tv -= sv * (dy_g / (40.0 * eff_span_v))
            su = float(np.clip(su / (1.0 + d_su / 80.0), 0.85, 1.18))
            sv = float(np.clip(sv / (1.0 + d_sv / 40.0), 0.85, 1.18))
            if abs(dx_g) < 0.03 and abs(dy_g) < 0.03:
                break
        lum, q_final = _warp_luma(tu, tv, su, sv)
        d = lum - float(np.mean(lum))
        sq = float(np.dot(d, d))
        final_z = float(np.dot(d, self._tpl) / math.sqrt(sq * n_pix)) if sq > 1e-4 * n_pix else -1.0
        return final_z, q_final

    def step(
        self,
        frame_rgb: np.ndarray,
        corners_xy: np.ndarray,
        crop_u8: np.ndarray,
        margin_ratio: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_use = np.asarray(corners_xy, dtype=np.float32).reshape(4, 2)
        w_t = 0.5 * (float(np.linalg.norm(q_use[1] - q_use[0])) + float(np.linalg.norm(q_use[2] - q_use[3])))
        if w_t >= 56.0 or crop_u8.shape[:2] != (80, 160):
            return q_use, crop_u8

        re_warped = False
        if self.sr_eff_views >= 1.0:
            zncc, q_ic = self._align_corners_iclk(frame_rgb, q_use, margin_ratio)
            if zncc < 0.56:
                return q_use, crop_u8
            shift = float(np.linalg.norm(q_ic - q_use, axis=-1).mean())
            if shift <= max(2.5, 0.18 * w_t) and is_valid_plate_quad(q_ic):
                q_use = q_ic
                crop_u8 = warp_plate_canonical(frame_rgb, q_use, out_hw=(80, 160), margin_ratio=margin_ratio)
                re_warped = True
                w_t = 0.5 * (float(np.linalg.norm(q_use[1] - q_use[0])) + float(np.linalg.norm(q_use[2] - q_use[3])))

        i_bilin = crop_u8.astype(np.float32).transpose(2, 0, 1) / 255.0
        if self.sr_eff_views <= 0.0:
            gamma_t = 0.0
            self.sr_width_ema = w_t
        else:
            gamma_t = (
                float(np.clip((self.sr_width_ema / max(1e-3, w_t)) ** 3, 0.25, 0.85))
                if w_t > 1.08 * self.sr_width_ema
                else 0.94
            )
            self.sr_num *= gamma_t
            self.sr_den *= gamma_t
            self.sr_width_ema = 0.75 * self.sr_width_ema + 0.25 * w_t
        self.sr_eff_views = gamma_t * self.sr_eff_views + 1.0

        h_f, w_f = frame_rgb.shape[:2]
        x0 = max(0, int(math.floor(float(q_use[:, 0].min()) - 2.0)))
        y0 = max(0, int(math.floor(float(q_use[:, 1].min()) - 2.0)))
        x1 = min(w_f - 1, int(math.ceil(float(q_use[:, 0].max()) + 2.0)))
        y1 = min(h_f - 1, int(math.ceil(float(q_use[:, 1].max()) + 2.0)))
        if x1 > x0 and y1 > y0:
            mx = float(margin_ratio) * 160.0
            my = float(margin_ratio) * 80.0
            dst_160 = np.array(
                [[mx - 0.5, my - 0.5], [160.0 - 0.5 - mx, my - 0.5], [160.0 - 0.5 - mx, 80.0 - 0.5 - my], [mx - 0.5, 80.0 - 0.5 - my]],
                dtype=np.float32,
            )
            H_s2c = cv2.getPerspectiveTransform(q_use, dst_160)
            ys_grid, xs_grid = np.meshgrid(
                np.arange(y0, y1 + 1, dtype=np.float32),
                np.arange(x0, x1 + 1, dtype=np.float32),
                indexing="ij",
            )
            xs_flat = xs_grid.reshape(-1)
            ys_flat = ys_grid.reshape(-1)
            denom = H_s2c[2, 0] * xs_flat + H_s2c[2, 1] * ys_flat + H_s2c[2, 2]
            valid_w = np.abs(denom) > 1e-6
            if np.any(valid_w):
                inv_d = np.where(valid_w, 1.0 / np.where(valid_w, denom, 1.0), 0.0)
                xc = (H_s2c[0, 0] * xs_flat + H_s2c[0, 1] * ys_flat + H_s2c[0, 2]) * inv_d
                yc = (H_s2c[1, 0] * xs_flat + H_s2c[1, 1] * ys_flat + H_s2c[1, 2]) * inv_d
                in_bounds = valid_w & (xc >= 0.0) & (xc <= 159.0) & (yc >= 0.0) & (yc <= 79.0)
                if np.any(in_bounds):
                    xc = xc[in_bounds]
                    yc = yc[in_bounds]
                    rgb_s = frame_rgb[y0 : y1 + 1, x0 : x1 + 1].reshape(-1, 3)[in_bounds].astype(np.float32) / 255.0
                    ix0 = np.clip(np.rint(xc).astype(np.int32), 0, 159)
                    iy0 = np.clip(np.rint(yc).astype(np.int32), 0, 79)
                    den0 = self.sr_den[iy0, ix0]
                    ref_rgb = np.where(
                        (den0 > 0.20)[:, None],
                        (self.sr_num[:, iy0, ix0] / np.maximum(den0[None, :], 1e-6)).T,
                        i_bilin[:, iy0, ix0].T,
                    )
                    res = np.max(np.abs(rgb_s - ref_rgb), axis=-1)
                    w_huber = np.where(res <= 0.18, 1.0, 0.18 / np.maximum(res, 1e-6)).astype(np.float32)
                    sigma_d = float(np.clip(0.38 * (144.0 / max(12.0, w_t)), 0.60, 1.35))
                    inv_2s2 = 0.5 / (sigma_d * sigma_d)
                    for oy in (-1, 0, 1):
                        ny = iy0 + oy
                        vy = (ny >= 0) & (ny < 80)
                        if not np.any(vy):
                            continue
                        for ox in (-1, 0, 1):
                            nx = ix0 + ox
                            v = vy & (nx >= 0) & (nx < 160)
                            if not np.any(v):
                                continue
                            ny_v, nx_v = ny[v], nx[v]
                            kw = w_huber[v] * np.exp(-((xc[v] - nx_v) ** 2 + (yc[v] - ny_v) ** 2) * inv_2s2).astype(np.float32)
                            np.add.at(self.sr_den, (ny_v, nx_v), kw)
                            np.add.at(self.sr_num, (slice(None), ny_v, nx_v), (rgb_s[v] * kw[:, None]).T)

        alpha_sr = float(
            np.clip((56.0 - w_t) / 34.0, 0.0, 0.86) * np.clip((self.sr_eff_views - 1.0) / 3.5, 0.0, 1.0)
        )
        if alpha_sr <= 1e-4:
            lum_160 = 0.299 * i_bilin[0] + 0.587 * i_bilin[1] + 0.114 * i_bilin[2]
            lum_40 = 0.25 * (lum_160[0::2, 0::2] + lum_160[0::2, 1::2] + lum_160[1::2, 0::2] + lum_160[1::2, 1::2])
            self._update_ic_tables(lum_40 if self.sr_eff_views <= 1.01 else (0.75 * self.sr_ref_luma + 0.25 * lum_40))
            return q_use, crop_u8

        eff_alpha = np.where(self.sr_den > 1e-4, alpha_sr * (self.sr_den / (self.sr_den + 0.25)), 0.0).astype(np.float32)
        i_sr = self.sr_num / np.maximum(self.sr_den[None, :, :], 1e-6)
        i_out = np.clip((1.0 - eff_alpha[None, :, :]) * i_bilin + eff_alpha[None, :, :] * i_sr, 0.0, 1.0)
        lum_160 = 0.299 * i_out[0] + 0.587 * i_out[1] + 0.114 * i_out[2]
        lum_40 = 0.25 * (lum_160[0::2, 0::2] + lum_160[0::2, 1::2] + lum_160[1::2, 0::2] + lum_160[1::2, 1::2])
        self._update_ic_tables(0.70 * self.sr_ref_luma + 0.30 * lum_40)
        out_u8 = np.round(i_out.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return q_use, out_u8 if (re_warped or alpha_sr > 1e-4) else crop_u8


def _search_requests(owner: Any, candidates: list[PlateDetection], ts: float,
                     img_hw: tuple[int, int]) -> list[SearchRequest]:
    covered = {t.parent_vehicle_track_id for t in getattr(owner.tracker, "_tracks", ())
               if getattr(t, "confirmed", False)
               and getattr(t, "parent_vehicle_track_id", None) is not None}
    c_wait = float(getattr(getattr(owner.tracker, "calibration", None), "c_wait", 1.0))
    requests = list(owner.tracker.get_search_requests(ts, img_hw=img_hw, include_dormant=False))
    vehicle_search = owner.vehicle_tracker.get_search_requests
    parameters = inspect.signature(vehicle_search).parameters
    accepts_covered = ("covered_vehicle_ids" in parameters
                       and parameters["covered_vehicle_ids"].kind != inspect.Parameter.POSITIONAL_ONLY)
    if accepts_covered or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        requests += vehicle_search(ts, img_hw=img_hw, covered_vehicle_ids=covered)
    else:
        requests += vehicle_search(ts, img_hw=img_hw)
    # Stage-2 refines local plate/mount geometry; Stage-1 handles whole vehicles.
    requests = [r for r in requests
                if getattr(r, "reason", getattr(r, "relation_hint", "")) != "component_search"
                and float(r.priority_cost_gain) >= c_wait]
    # A detector/search pair on the same plate is one physical measurement.
    selected = []
    for request in sorted(requests, key=lambda r: -float(r.priority_cost_gain)):
        if any(_bbox_iou_xyxy(request.roi_xyxy, d.bbox_xyxy) > 0 for d in candidates):
            continue
        if any(_bbox_iou_xyxy(request.roi_xyxy, r.roi_xyxy) >= 0.65 for r in selected):
            continue
        selected.append(request)
    return selected[:max(8, int(getattr(owner, "overload_track_threshold", 4)) * 3)]


def _ocr_entry(session: Any, plate: PlateDetection, ts: float, frame_id: int) -> dict[str, Any]:
    return dict(session=session, crop_rgb=plate.ocr_crop_rgb, timestamp_s=ts,
        geom_meta=plate.geom_meta, camera_id=0, source_frame_id=frame_id,
        frame_key=(0, frame_id), plate_side=plate.plate_side,
        quad_prob=float(plate.quad_is_plate_prob), quad_edge_touch=bool(plate.quad_edge_touch),
        support_nonconformity=float(plate.support_nonconformity),
        corner_visibility_4=plate.corner_visibility_4, plate_null_llr=plate.plate_null_llr)


def _push_v2_ocr_batch_or_loop(sess_cls: type[Any], entries: list[dict[str, Any]]) -> list[Any]:
    """Forward supported metadata; never retry a partially consumed batch."""
    if not entries:
        return []
    if any(not _collecting(e["session"]) for e in entries):
        raise RuntimeError("Cannot push a sealed OCR episode")
    batch_fn = getattr(sess_cls, "push_batch_parallel", None)
    singular = ("timestamp_s", "geom_meta", "camera_id", "source_frame_id", "plate_side",
                "quad_prob", "quad_edge_touch", "frame_key", "support_nonconformity",
                "corner_visibility_4", "plate_null_llr")
    plural = ("timestamps_s", "geom_metas", "camera_ids", "source_frame_ids", "plate_sides",
              "quad_probs", "quad_edge_touches", "frame_keys", "support_nonconformities",
              "corner_visibilities_4", "plate_null_llrs")
    if len(entries) > 1 and callable(batch_fn):
        names = inspect.signature(batch_fn).parameters
        variadic = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in names.values())
        kwargs = {p: [e.get(k) for e in entries] for k,p in zip(singular,plural) if p in names or variadic}
        return batch_fn([e["session"] for e in entries], [e["crop_rgb"] for e in entries], **kwargs)
    results = []
    for entry in entries:
        names = inspect.signature(entry["session"].push).parameters
        variadic = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in names.values())
        kwargs = {k: entry[k] for k in singular if (k in names or variadic) and k in entry}
        results.append(entry["session"].push(entry["crop_rgb"], **kwargs))
    return results


class VehiclePlateVideoPipeline:
    """End-to-end single-frame and streaming video plate detector, tight cropper, and tracker."""

    def __init__(
        self,
        det_model: PlateDetNet | TemporalPlateDetNet,
        quad_model: PlateQuadNet | PlateQuadTrackNet,
        device: torch.device | str = "cpu",
        input_hw: tuple[int, int] = (448, 768),
        plate_thresh: float = 0.18,
        min_is_plate_prob: float = 0.34,
        enable_far_zoom: bool = True,
        use_tracker: bool = True,
        ocr_model: Any | None = None,
        ocr_vocab: Any | None = None,
        ocr_session_config: Any | None = None,
        prefer_v2_canonical_crop: bool = True,
        max_retired_history: int = 1024,
        enable_tracker_roi_reacquire: bool = True,
    ) -> None:
        self.dev = torch.device(device)
        self.det_model = _inference_model(det_model, self.dev)
        self.quad_model = _inference_model(quad_model, self.dev, stage2=True)
        self.input_hw = tuple(getattr(self.det_model, "input_hw", input_hw))
        self.plate_thresh = plate_thresh
        self.min_is_plate_prob = min_is_plate_prob
        self.enable_far_zoom = enable_far_zoom
        self.use_tracker = use_tracker
        self.prefer_v2_canonical_crop = bool(prefer_v2_canonical_crop)
        self.max_retired_history = max(1, int(max_retired_history))
        self.enable_tracker_roi_reacquire = bool(enable_tracker_roi_reacquire)
        self.tracker = PlateTrackStabilizer()
        self.vehicle_tracker = MultiVehicleKalmanTracker()
        if ocr_model is not None and isinstance(ocr_model, torch.nn.Module):
            ocr_model = ocr_model.to(self.dev).eval()
        self.ocr_model = ocr_model
        self.ocr_vocab = ocr_vocab
        self._ocr_session_config: Any | None = None
        self._caller_ocr_session_config: Any | None = None
        self.ocr_session_config = ocr_session_config
        self._ocr_sessions: dict[int, Any] = {}
        self._terminal_decisions: dict[int, Any] = {}
        self._camera_motion_estimator = BackgroundCameraMotionEstimator()
        self._retired_decisions: dict[int, Any] = {}
        self._prev_corners: dict[int, np.ndarray] = {}
        self._opt_refs: dict[int, OpticalPlateRef] = {}
        self._sr_states: dict[int, _CanonicalDrizzleState] = {}
        self._track_corner_stats: dict[int, dict[str, Any]] = {}
        self._veh_zoom_cooldown: dict[int, int] = {}
        self._scene_cut_monitor = _InnerSceneCutMonitor()
        self._det_hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self._prev_frame_luma: float | None = None
        self._frame_idx: int = 0

    @property
    def ocr_session_config(self) -> Any | None:
        if self._caller_ocr_session_config is not None:
            _enforce_patent_clean_ocr_session_config(self._caller_ocr_session_config)
        return self._ocr_session_config

    @ocr_session_config.setter
    def ocr_session_config(self, value: Any | None) -> None:
        self._ocr_session_config = _enforce_patent_clean_ocr_session_config(value)
        self._caller_ocr_session_config = value

    def reset(self) -> None:
        self.tracker.reset()
        self.vehicle_tracker.reset()
        self._ocr_sessions.clear()
        self._terminal_decisions.clear()
        self._camera_motion_estimator.reset()
        self._retired_decisions.clear()
        self._prev_corners.clear()
        self._opt_refs.clear()
        self._sr_states.clear()
        self._track_corner_stats.clear()
        self._veh_zoom_cooldown.clear()
        self._scene_cut_monitor.reset()
        self._det_hidden_state = None
        self._prev_frame_luma = None
        self._frame_idx = 0

    @torch.no_grad()
    def finalize_tracks(self) -> dict[int, Any]:
        """Finalize all active per-track OCR sessions plus previously retired expired tracks."""
        out = dict(self._retired_decisions)
        for key, sess in list(self._ocr_sessions.items()):
            out[key] = _finish_cached(self, key, sess)
        return out

    @torch.no_grad()
    def process_frame(self, frame_rgb: np.ndarray, timestamp_s: float | None = None) -> FrameDetectionResult:
        """Process a single RGB camera frame (stateless if `use_tracker=False`, tracked if `use_tracker=True`)."""
        t0 = time.perf_counter()
        ts = float(timestamp_s if timestamp_s is not None else self._frame_idx * 0.05)
        frame_key = f"0:{self._frame_idx}"
        retired_this_frame: dict[int, Any] = {}
        scene_cut = False
        pre_veh_ncc = None
        if self.use_tracker:
            inner_cut, inner_sad, inner_hist_corr = self._scene_cut_monitor.step(frame_rgb)
            probed_v, low_ncc_v, pre_veh_ncc = self.vehicle_tracker.probe_vehicle_ncc(frame_rgb, ts)
            veh_cut = (
                (
                    probed_v >= 2
                    and (low_ncc_v / float(probed_v)) >= 0.60
                    and (inner_sad >= 10.0 or inner_hist_corr < 0.92)
                )
                or (probed_v == 1 and low_ncc_v == 1 and (inner_sad >= 15.0 or inner_hist_corr < 0.85))
            )
            scene_cut = bool(inner_cut or veh_cut)
            if scene_cut:
                _handle_scene_cut(self, retired_this_frame)
                pre_veh_ncc = None
        camera_motion = self._camera_motion_estimator.estimate(
            frame_rgb, [t.bbox_xyxy for t in self.vehicle_tracker._tracks])
        is_dual_quad = bool(getattr(self.quad_model, "is_dual_track_net", False))
        is_temp_det = bool(getattr(self.det_model, "is_temporal_det_net", False) or isinstance(self.det_model, UnifiedTemporalDetORT))
        if self.ocr_model is not None:
            is_v2_ocr = bool(getattr(self.ocr_model, "is_v2", False))
            use_v2_crop = is_v2_ocr
        else:
            is_v2_ocr = False
            use_v2_crop = self.prefer_v2_canonical_crop

        if use_v2_crop:
            cfg_ocr = getattr(self.ocr_model, "config", None) if self.ocr_model is not None else None
            ocr_hw: tuple[int, int] = tuple(getattr(cfg_ocr, "input_hw", (80, 160)))
            ocr_margin: float = float(getattr(cfg_ocr, "canonical_margin", 0.05))
        else:
            ocr_hw = (64, 128)
            ocr_margin = 0.06

        cur_luma = float(frame_rgb[::4, ::4].mean()) if frame_rgb.size > 0 else 0.0
        luma_blackout = (
            self.use_tracker
            and not scene_cut
            and self._prev_frame_luma is not None
            and self._prev_frame_luma >= 24.0
            and cur_luma < 0.25 * self._prev_frame_luma
        )

        if is_temp_det:
            raw_res, h_next = decode_temporal_det_predictions(
                self.det_model,  # type: ignore[arg-type]
                frame_rgb,
                device=self.dev,
                input_hw=self.input_hw,
                plate_thresh=self.plate_thresh,
                enable_far_zoom=self.enable_far_zoom,
                hidden_state=self._det_hidden_state if self.use_tracker else None,
                frame_key=frame_key,
            )
            if self.use_tracker:
                if not luma_blackout:
                    self._det_hidden_state = h_next
                self._prev_frame_luma = cur_luma
        else:
            raw_res = decode_plate_det_predictions(
                self.det_model,
                frame_rgb,
                device=self.dev,
                input_hw=self.input_hw,
                plate_thresh=self.plate_thresh,
                enable_far_zoom=self.enable_far_zoom,
            )
            if self.use_tracker:
                self._prev_frame_luma = cur_luma

        filtered_vehicles = _vehicle_containment_nms(list(raw_res.vehicles))
        if self.use_tracker:
            tracked_vehicles = self.vehicle_tracker.step(
                filtered_vehicles,
                timestamp_s=ts,
                img_hw=frame_rgb.shape[:2],
                action_meta=ObservationActionMeta(detector_ran_full=True, frame_key=frame_key),
                camera_motion=camera_motion, frame_rgb=frame_rgb,
                precomputed_veh_ncc=pre_veh_ncc,
            )
        else:
            tracked_vehicles = filtered_vehicles

        det_candidates = list(raw_res.plates)
        if is_temp_det and self.enable_far_zoom and tracked_vehicles:
            det_candidates.extend(
                _run_full_face_vehicle_zoom(self, frame_rgb, tracked_vehicles, det_candidates, frame_key)
            )
        for candidate in det_candidates:
            candidate.frame_key = frame_key
        search_requests = (_search_requests(self, det_candidates, ts, frame_rgb.shape[:2])
                           if self.use_tracker and self.enable_tracker_roi_reacquire and is_dual_quad else [])
        det_candidates.extend(
            _search_inputs(
                search_requests,
                frame_key,
                frame_rgb=frame_rgb,
                opt_refs=self._opt_refs,
                prev_corners=self._prev_corners,
            )
        )
        action_meta = ObservationActionMeta(action_id=frame_key, detector_ran_full=True,
            searched_rois_xyxy=tuple(r.roi_xyxy for r in search_requests), frame_key=frame_key)

        refine_extra_kw: dict[str, Any] = {"vehicles": tracked_vehicles} if tracked_vehicles else {}
        refined = refine_and_crop_detections(
            self.quad_model,
            frame_rgb,
            det_candidates,
            device=self.dev,
            min_is_plate_prob=self.min_is_plate_prob,
            ocr_margin_ratio=ocr_margin,
            ocr_out_hw=ocr_hw,
            pixel_center_aligned=use_v2_crop,
            **refine_extra_kw,
        )

        relinked_this_frame: list[tuple[int, int]] = []
        if self.use_tracker:
            veh_vels = (
                self.vehicle_tracker.bind_plates_and_sync(
                    tracked_vehicles, refined, update_vehicle_plate_lists=False,
                    img_hw=frame_rgb.shape[:2],
                )
                if tracked_vehicles
                else None
            )
            plates = self.tracker.step(
                refined,
                timestamp_s=ts,
                action_meta=action_meta, camera_motion=camera_motion, frame_rgb=frame_rgb,
                vehicle_velocities=veh_vels,
            )
            if tracked_vehicles:
                self.vehicle_tracker.bind_plates_and_sync(
                    tracked_vehicles, plates, update_vehicle_plate_lists=True,
                    img_hw=frame_rgb.shape[:2], plate_tracker=self.tracker,
                )
            plates = _suppress_stationary_corner_logos(self, plates, frame_rgb.shape[:2])
            relinked_this_frame = list(self.tracker.last_relinked_pairs)

            _relink_sessions(self, relinked_this_frame)

            for exp_tid in self.tracker.last_expired_track_ids:
                self._prev_corners.pop(exp_tid, None)
                self._opt_refs.pop(exp_tid, None)
                self._sr_states.pop(exp_tid, None)
                self._track_corner_stats.pop(exp_tid, None)
                exp_sess = self._ocr_sessions.pop(exp_tid, None)
                if exp_sess is not None:
                    exp_dec = _finish_cached(self, exp_tid, exp_sess)
                    self._retired_decisions[exp_tid] = exp_dec
                    self._terminal_decisions.pop(exp_tid, None)
                    retired_this_frame[exp_tid] = exp_dec
                    while len(self._retired_decisions) > self.max_retired_history:
                        self._retired_decisions.pop(next(iter(self._retired_decisions)))

            _apply_optical_refinement_and_template_update(
                frame_rgb, plates, self._opt_refs, self._prev_corners
            )

            # Strict Crop Ownership Invariant:
            # - On observed frames (`is_coasted == False`), preserve exact `PlateQuadNet` `ocr_crop_rgb` &
            #   `tight_crop_rgb` without overwriting with lagged tracker corners (only backfill if caller/mock
            #   passed detections with `None` crops).
            # - On coasted frames (`is_coasted == True`), keep `ocr_crop_rgb = None` so background is never fed to OCR!
            for p in plates:
                if p.is_coasted or p.quad_edge_touch or not np.isfinite(p.support_nonconformity):
                    p.ocr_crop_rgb = None
                    p.tight_crop_rgb = None
                    p.geom_meta = None
                    continue
                q_exact = p.raw_corners_xy if p.raw_corners_xy is not None else p.corners_xy
                if p.tight_crop_rgb is None:
                    p.tight_crop_rgb = warp_plate_quad(frame_rgb, q_exact, out_hw=(64, 128), margin_ratio=0.0)
                if p.ocr_crop_rgb is None:
                    if use_v2_crop:
                        p.ocr_crop_rgb = warp_plate_canonical(
                            frame_rgb, q_exact, out_hw=ocr_hw, margin_ratio=ocr_margin
                        )
                    else:
                        p.ocr_crop_rgb = warp_plate_quad(
                            frame_rgb, q_exact, out_hw=ocr_hw, margin_ratio=ocr_margin
                        )
                if (
                    use_v2_crop
                    and p.track_id is not None
                    and ocr_hw == (80, 160)
                    and float(np.linalg.norm(q_exact[1] - q_exact[0]) + np.linalg.norm(q_exact[2] - q_exact[3])) * 0.5 < 56.0
                    and float(frame_rgb[::4, ::4].std()) > 2.0
                ):
                    sr_st = self._sr_states.setdefault(p.track_id, _CanonicalDrizzleState())
                    q_exact, p.ocr_crop_rgb = sr_st.step(frame_rgb, q_exact, p.ocr_crop_rgb, margin_ratio=ocr_margin)
                prev_q = self._prev_corners.get(p.track_id) if p.track_id is not None else None
                p.geom_meta = extract_quad_geometry_meta(q_exact, p.ocr_crop_rgb, prev_corners_xy=prev_q)
                if p.track_id is not None:
                    self._prev_corners[p.track_id] = np.asarray(q_exact, dtype=np.float32).copy()
        else:
            plates = refined
            for p in plates:
                if p.ocr_crop_rgb is not None:
                    q_exact = p.raw_corners_xy if p.raw_corners_xy is not None else p.corners_xy
                    p.geom_meta = extract_quad_geometry_meta(q_exact, p.ocr_crop_rgb, prev_corners_xy=None)

        ocr_map: dict[int, str] = {}
        ocr_verified_map: dict[int, str] = {}
        ocr_tiers: dict[int, str] = {}
        ocr_verified_fields: dict[int, list[str]] = {}
        ocr_decisions: dict[int, Any] = {}
        if self.ocr_model is not None and self.ocr_vocab is not None:
            if is_v2_ocr:
                from vpocr_ocr.session_v2 import MFPlateOCRSessionV2, SessionConfigV2

                sess_cfg_v2 = _resolve_patent_clean_v2_session_config(
                    self.ocr_session_config, SessionConfigV2
                )
                tracked_v2_entries: list[dict[str, Any]] = []
                tracked_v2_keys: list[int] = []
                for idx_p, p in enumerate(plates):
                    if p.is_coasted or p.ocr_crop_rgb is None:
                        continue
                    gm = p.geom_meta if p.geom_meta is not None else extract_quad_geometry_meta(p.raw_corners_xy, p.ocr_crop_rgb)
                    if self.use_tracker and p.track_id is None:
                        continue
                    if not self.use_tracker:
                        # Stateless one-shot OCR session per detection (zero cross-image memory leakage!)
                        ephemeral_sess = MFPlateOCRSessionV2(
                            self.ocr_model,
                            vocab=self.ocr_vocab,
                            config=sess_cfg_v2,
                            device=self.dev,
                        )
                        _push_v2_ocr_batch_or_loop(MFPlateOCRSessionV2,
                            [_ocr_entry(ephemeral_sess, p, ts, self._frame_idx)])
                        dec = ephemeral_sess.finish()
                        key = idx_p
                        ocr_decisions[key] = dec
                        if dec.plate_text is not None and dec.status == "ACCEPTED":
                            ocr_tiers[key] = dec.confidence_tier
                            ocr_verified_fields[key] = list(dec.verified_fields)
                            v_txt = dec.format_verified_text()
                            if v_txt is not None:
                                ocr_verified_map[key] = v_txt
                            if dec.confidence_tier == "FULL_4FIELD":
                                ocr_map[key] = dec.plate_text.full_text
                    else:
                        key = p.track_id
                        if key not in self._ocr_sessions:
                            self._ocr_sessions[key] = MFPlateOCRSessionV2(
                                self.ocr_model,
                                vocab=self.ocr_vocab,
                                config=sess_cfg_v2,
                                device=self.dev,
                            )
                        else:
                            _enforce_patent_clean_ocr_session_config(
                                getattr(self._ocr_sessions[key], "config", None)
                            )
                        if not _collecting(self._ocr_sessions[key]):
                            ocr_decisions[key] = _finish_cached(self, key, self._ocr_sessions[key])
                            continue
                        tracked_v2_keys.append(key)
                        tracked_v2_entries.append(_ocr_entry(self._ocr_sessions[key], p, ts, self._frame_idx))

                if tracked_v2_entries:
                    batch_decs = _push_v2_ocr_batch_or_loop(MFPlateOCRSessionV2, tracked_v2_entries)
                    for key, dec in zip(tracked_v2_keys, batch_decs):
                        ocr_decisions[key] = dec
                        if not _collecting(self._ocr_sessions[key]):
                            self._terminal_decisions[key] = dec
                        if dec.plate_text is not None and dec.status == "ACCEPTED":
                            ocr_tiers[key] = dec.confidence_tier
                            ocr_verified_fields[key] = list(dec.verified_fields)
                            v_txt = dec.format_verified_text()
                            if v_txt is not None:
                                ocr_verified_map[key] = v_txt
                            if dec.confidence_tier == "FULL_4FIELD":
                                ocr_map[key] = dec.plate_text.full_text
            else:
                from vpocr_ocr.session import MFPlateOCRSession, SessionConfig

                sess_cfg_v1 = (
                    self.ocr_session_config
                    if isinstance(self.ocr_session_config, SessionConfig)
                    else SessionConfig()
                )
                for idx_p, p in enumerate(plates):
                    if p.is_coasted or p.ocr_crop_rgb is None:
                        continue
                    if not self.use_tracker or p.track_id is None:
                        ephemeral_v1 = MFPlateOCRSession(
                            self.ocr_model,
                            vocab=self.ocr_vocab,
                            config=sess_cfg_v1,
                            device=self.dev,
                        )
                        ephemeral_v1.push(
                            p.ocr_crop_rgb,
                            timestamp_s=ts,
                            camera_id=0,
                            source_frame_id=0,
                            plate_side=p.plate_side,
                        )
                        dec = ephemeral_v1.finish()
                        key = idx_p
                    else:
                        key = p.track_id
                        if key not in self._ocr_sessions:
                            self._ocr_sessions[key] = MFPlateOCRSession(
                                self.ocr_model,
                                vocab=self.ocr_vocab,
                                config=sess_cfg_v1,
                                device=self.dev,
                            )
                        dec = self._ocr_sessions[key].push(
                            p.ocr_crop_rgb,
                            timestamp_s=ts,
                            camera_id=0,
                            source_frame_id=self._frame_idx,
                            plate_side=p.plate_side,
                        )
                    ocr_decisions[key] = dec
                    if dec.plate_text is not None and dec.status == "ACCEPTED":
                        ocr_map[key] = dec.plate_text.full_text
                        ocr_verified_map[key] = dec.plate_text.full_text

        out_veh_bboxes = (
            [(*v.bbox_xyxy, float(v.score)) for v in tracked_vehicles]
            if tracked_vehicles
            else ([(*v.bbox_xyxy, float(v.score)) for v in filtered_vehicles] if filtered_vehicles else [])
        )
        self._frame_idx += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return FrameDetectionResult(
            plates=plates,
            vehicle_bboxes_xyxy=out_veh_bboxes,
            vehicles=tracked_vehicles,
            latency_ms=elapsed_ms,
            ocr_decisions=ocr_decisions,
            metadata={
                "raw_candidates": len(raw_res.plates),
                "scene_cut": bool(scene_cut),
                "luma_blackout_frozen": bool(luma_blackout),
                "ocr_accepted": ocr_map,
                "ocr_accepted_verified": ocr_verified_map,
                "ocr_tiers": ocr_tiers,
                "ocr_verified_fields": ocr_verified_fields,
                "ocr_decisions": ocr_decisions,
                "retired_track_decisions": retired_this_frame,
                "relinked_track_pairs": relinked_this_frame,
                "dormant_track_ids": list(self.tracker.dormant_track_ids) if self.use_tracker else [],
            },
        )


class FastLuminanceScout:
    """Ultra-lightweight (`< 1.0 ms` CPU) camera-shake-compensated luminance & border-entry motion scout."""

    def __init__(
        self,
        grid_hw: tuple[int, int] = (80, 144),
        max_jitter_cells: int = 5,
        diff_thresh: float = 16.0,
        min_blob_cells: int = 6,
        min_border_entry_cells: int = 20,
    ) -> None:
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.max_jitter_cells = int(max_jitter_cells)
        self.diff_thresh = float(diff_thresh)
        self.min_blob_cells = int(min_blob_cells)
        self.min_border_entry_cells = int(min_border_entry_cells)
        self._prev_y: np.ndarray | None = None
        self._prev_inner_y: np.ndarray | None = None
        self._prev_inner_hist: np.ndarray | None = None
        self._noise_ema: float = 4.0

    def reset(self) -> None:
        self._prev_y = None
        self._prev_inner_y = None
        self._prev_inner_hist = None
        self._noise_ema = 4.0

    def step(
        self,
        frame_rgb: np.ndarray,
        monitored_boxes_xyxy: list[tuple[float, float, float, float]] | None = None,
        excluded_holes_xyxy: list[tuple[float, float, float, float]] | None = None,
    ) -> dict[str, Any]:
        h_orig, w_orig = frame_rgb.shape[:2]
        gh, gw = self.grid_hw
        small_rgb = cv2.resize(frame_rgb, (gw, gh), interpolation=cv2.INTER_AREA)
        y_cur = (
            0.299 * small_rgb[:, :, 0].astype(np.float32)
            + 0.587 * small_rgb[:, :, 1].astype(np.float32)
            + 0.114 * small_rgb[:, :, 2].astype(np.float32)
        )
        inner_y, inner_hist = _compute_inner_scene_features(small_rgb)

        if self._prev_y is None:
            self._prev_y = y_cur
            self._prev_inner_y = inner_y
            self._prev_inner_hist = inner_hist
            return {
                "motion_rois_xyxy": [],
                "unmonitored_rois_xyxy": [],
                "has_border_entry": False,
                "jitter_xy": (0, 0),
                "noise_floor": self._noise_ema,
                "scene_cut": False,
                "inner_sad": 0.0,
                "inner_hist_corr": 1.0,
            }

        inner_sad = (
            float(np.mean(np.abs(inner_y - self._prev_inner_y)))
            if (self._prev_inner_y is not None and self._prev_inner_y.shape == inner_y.shape)
            else 0.0
        )
        inner_hist_corr = (
            float(np.clip(np.dot(inner_hist, self._prev_inner_hist), 0.0, 1.0))
            if self._prev_inner_hist is not None
            else 1.0
        )
        scene_cut = _is_inner_scene_cut(inner_sad, inner_hist_corr)

        prev = self._prev_y
        mj = min(self.max_jitter_cells, min(gh, gw) // 4)
        best_dx, best_dy = 0, 0
        best_l1 = float("inf")
        ref_patch = y_cur[mj : gh - mj, mj : gw - mj]
        for dy in range(-mj, mj + 1):
            for dx in range(-mj, mj + 1):
                cand = prev[mj + dy : gh - mj + dy, mj + dx : gw - mj + dx]
                score = float(np.mean(np.abs(ref_patch - cand))) + 0.30 * float(abs(dx) + abs(dy))
                if score < best_l1:
                    best_l1 = score
                    best_dx, best_dy = dx, dy

        aligned_prev = np.empty_like(prev)
        y_src0 = max(0, best_dy)
        y_src1 = min(gh, gh + best_dy)
        x_src0 = max(0, best_dx)
        x_src1 = min(gw, gw + best_dx)
        aligned_prev[:] = y_cur
        aligned_prev[y_src0 - best_dy : y_src1 - best_dy, x_src0 - best_dx : x_src1 - best_dx] = prev[
            y_src0:y_src1, x_src0:x_src1
        ]

        # Spatial-gradient jitter tolerance (`0.28 * |grad Y|`) suppresses sub-cell wind/foliage edge flicker
        gx = np.zeros_like(aligned_prev)
        gy = np.zeros_like(aligned_prev)
        gx[:, 1:-1] = 0.5 * np.abs(aligned_prev[:, 2:] - aligned_prev[:, :-2])
        gy[1:-1, :] = 0.5 * np.abs(aligned_prev[2:, :] - aligned_prev[:-2, :])
        grad_tol = 0.28 * (gx + gy)

        diff = np.maximum(0.0, np.abs(y_cur - aligned_prev) - grad_tol)
        med_diff = float(np.percentile(diff, 65.0))
        self._noise_ema = 0.85 * self._noise_ema + 0.15 * max(2.0, min(18.0, med_diff))
        thr = max(self.diff_thresh, 3.2 * self._noise_ema)

        mask_u8 = (diff >= thr).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        sx = float(w_orig) / float(gw)
        sy = float(h_orig) / float(gh)
        motion_rois: list[tuple[int, int, int, int]] = []
        unmonitored_rois: list[tuple[int, int, int, int]] = []
        has_border_entry = False

        mon_boxes = monitored_boxes_xyxy or []
        excl_holes = excluded_holes_xyxy or []
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < self.min_blob_cells:
                continue
            x0_g = int(stats[lbl, cv2.CC_STAT_LEFT])
            y0_g = int(stats[lbl, cv2.CC_STAT_TOP])
            bw_g = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh_g = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            x1_g = x0_g + bw_g
            y1_g = y0_g + bh_g

            rx0 = max(0, int(round((x0_g - 2) * sx)))
            ry0 = max(0, int(round((y0_g - 2) * sy)))
            rx1 = min(w_orig, int(round((x1_g + 2) * sx)))
            ry1 = min(h_orig, int(round((y1_g + 2) * sy)))
            roi_f = (float(rx0), float(ry0), float(rx1), float(ry1))
            motion_rois.append((rx0, ry0, rx1, ry1))

            in_hole = any(
                _bbox_iomin_xyxy(roi_f, hole) >= 0.15 or _bbox_iou_xyxy(roi_f, hole) >= 0.05
                for hole in excl_holes
            )
            covered = False
            if not in_hole:
                for mb in mon_boxes:
                    ix = max(0.0, min(roi_f[2], mb[2]) - max(roi_f[0], mb[0]))
                    iy = max(0.0, min(roi_f[3], mb[3]) - max(roi_f[1], mb[1]))
                    if (ix * iy) / max(1.0, (roi_f[2] - roi_f[0]) * (roi_f[3] - roi_f[1])) >= 0.45:
                        covered = True
                        break
            if not covered:
                unmonitored_rois.append((rx0, ry0, rx1, ry1))
                if (
                    area >= self.min_border_entry_cells
                    and (x0_g <= 8 or x1_g >= gw - 8 or y0_g <= 6 or y1_g >= gh - 6)
                ):
                    has_border_entry = True

        self._prev_y = y_cur
        self._prev_inner_y = inner_y
        self._prev_inner_hist = inner_hist
        return {
            "motion_rois_xyxy": motion_rois,
            "unmonitored_rois_xyxy": unmonitored_rois,
            "has_border_entry": has_border_entry,
            "jitter_xy": (int(best_dx), int(best_dy)),
            "noise_floor": round(self._noise_ema, 3),
            "scene_cut": bool(scene_cut),
            "inner_sad": round(inner_sad, 3),
            "inner_hist_corr": round(inner_hist_corr, 4),
        }


class MultiRateEdgeCascadePipeline:
    """Gen-4 Two-Rate Edge CPU Cascade (`SEARCH / ACQUIRE / TRACK / OCCLUDED / RECOVER / EXIT`).

    Achieves `3-5x` single-thread CPU speedup by running `FastLuminanceScout` + direct ROI
    `PlateQuadTrackNet` tracking on intermediate `TRACK` frames while firing full-frame
    `TemporalPlateDetNet` only on periodic heartbeats (`K = 6..10`), unmonitored border entries,
    or occlusion/re-acquisition events. Preserves 100% patent-clean `0` intermediate OCR readouts.
    """

    VALID_STATES: tuple[str, ...] = ("SEARCH", "ACQUIRE", "TRACK", "OCCLUDED", "RECOVER", "EXIT")

    def __init__(
        self,
        det_model: TemporalPlateDetNet | PlateDetNet,
        quad_model: PlateQuadTrackNet | PlateQuadNet,
        device: torch.device | str = "cpu",
        input_hw: tuple[int, int] = (448, 768),
        plate_thresh: float = 0.22,
        min_is_plate_prob: float = 0.34,
        enable_far_zoom: bool = True,
        heartbeat_interval: int = 2,
        max_heartbeat_interval: int = 3,
        overload_track_threshold: int = 4,
        ocr_model: Any | None = None,
        ocr_vocab: Any | None = None,
        ocr_session_config: Any | None = None,
        prefer_v2_canonical_crop: bool = True,
        max_retired_history: int = 1024,
    ) -> None:
        self.dev = torch.device(device)
        self.det_model = _inference_model(det_model, self.dev)
        self.quad_model = _inference_model(quad_model, self.dev, stage2=True)
        self.input_hw = tuple(getattr(self.det_model, "input_hw", input_hw))
        self.plate_thresh = float(plate_thresh)
        self.min_is_plate_prob = float(min_is_plate_prob)
        self.enable_far_zoom = bool(enable_far_zoom)
        self.heartbeat_interval = max(1, int(heartbeat_interval))
        self.max_heartbeat_interval = max(self.heartbeat_interval, int(max_heartbeat_interval))
        self.overload_track_threshold = max(1, int(overload_track_threshold))
        self.prefer_v2_canonical_crop = bool(prefer_v2_canonical_crop)
        self.max_retired_history = max(1, int(max_retired_history))

        self.scout = FastLuminanceScout()
        self.tracker = PlateTrackStabilizer()
        self.vehicle_tracker = MultiVehicleKalmanTracker()
        if ocr_model is not None and isinstance(ocr_model, torch.nn.Module):
            ocr_model = ocr_model.to(self.dev).eval()
        self.ocr_model = ocr_model
        self.ocr_vocab = ocr_vocab
        self._ocr_session_config: Any | None = None
        self._caller_ocr_session_config: Any | None = None
        self.ocr_session_config = ocr_session_config

        self.state: str = "SEARCH"
        self._det_hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self._prev_frame_luma: float | None = None
        self._dormant_reappear_cooldown: dict[int, int] = {}
        self._empty_scene_cooldown: int = 0
        self._ocr_sessions: dict[int, Any] = {}
        self._terminal_decisions: dict[int, Any] = {}
        self._camera_motion_estimator = BackgroundCameraMotionEstimator()
        self._retired_decisions: dict[int, Any] = {}
        self._prev_corners: dict[int, np.ndarray] = {}
        self._opt_refs: dict[int, OpticalPlateRef] = {}
        self._sr_states: dict[int, _CanonicalDrizzleState] = {}
        self._track_corner_stats: dict[int, dict[str, Any]] = {}
        self._veh_zoom_cooldown: dict[int, int] = {}
        self._cached_vehicles: list[VehicleDetection] = []
        self._cached_vehicle_bboxes: list[tuple[float, float, float, float, float]] = []
        self._frame_idx: int = 0
        self._frames_since_det: int = 999
        self._full_det_calls: int = 0

    @property
    def ocr_session_config(self) -> Any | None:
        if self._caller_ocr_session_config is not None:
            _enforce_patent_clean_ocr_session_config(self._caller_ocr_session_config)
        return self._ocr_session_config

    @ocr_session_config.setter
    def ocr_session_config(self, value: Any | None) -> None:
        self._ocr_session_config = _enforce_patent_clean_ocr_session_config(value)
        self._caller_ocr_session_config = value

    def reset(self) -> None:
        self.scout.reset()
        self.tracker.reset()
        self.vehicle_tracker.reset()
        self.state = "SEARCH"
        self._det_hidden_state = None
        self._prev_frame_luma = None
        self._dormant_reappear_cooldown.clear()
        self._empty_scene_cooldown = 0
        self._ocr_sessions.clear()
        self._terminal_decisions.clear()
        self._camera_motion_estimator.reset()
        self._retired_decisions.clear()
        self._prev_corners.clear()
        self._opt_refs.clear()
        self._sr_states.clear()
        self._track_corner_stats.clear()
        self._veh_zoom_cooldown.clear()
        self._cached_vehicles.clear()
        self._cached_vehicle_bboxes.clear()
        self._frame_idx = 0
        self._frames_since_det = 999
        self._full_det_calls = 0

    @torch.no_grad()
    def finalize_tracks(self) -> dict[int, Any]:
        out = dict(self._retired_decisions)
        for key, sess in list(self._ocr_sessions.items()):
            out[key] = _finish_cached(self, key, sess)
        return out

    @torch.no_grad()
    def process_frame(self, frame_rgb: np.ndarray, timestamp_s: float | None = None) -> FrameDetectionResult:
        t0 = time.perf_counter()
        ts = float(timestamp_s if timestamp_s is not None else self._frame_idx * 0.05)
        frame_key = f"0:{self._frame_idx}"
        retired_this_frame: dict[int, Any] = {}
        is_dual_quad = bool(getattr(self.quad_model, "is_dual_track_net", False))
        is_temp_det = bool(getattr(self.det_model, "is_temporal_det_net", False) or isinstance(self.det_model, UnifiedTemporalDetORT))
        is_v2_ocr = bool(getattr(self.ocr_model, "is_v2", False)) if self.ocr_model is not None else False
        use_v2_crop = is_v2_ocr if self.ocr_model is not None else self.prefer_v2_canonical_crop

        if use_v2_crop:
            cfg_ocr = getattr(self.ocr_model, "config", None) if self.ocr_model is not None else None
            ocr_hw: tuple[int, int] = tuple(getattr(cfg_ocr, "input_hw", (80, 160)))
            ocr_margin: float = float(getattr(cfg_ocr, "canonical_margin", 0.05))
        else:
            ocr_hw = (64, 128)
            ocr_margin = 0.06

        if self.ocr_model is not None and self.ocr_vocab is not None:
            _enforce_patent_clean_ocr_session_config(self.ocr_session_config)

        for d_tid in list(self._dormant_reappear_cooldown.keys()):
            rem = self._dormant_reappear_cooldown[d_tid] - 1
            if rem <= 0:
                self._dormant_reappear_cooldown.pop(d_tid, None)
            else:
                self._dormant_reappear_cooldown[d_tid] = rem
        if self._empty_scene_cooldown > 0:
            self._empty_scene_cooldown -= 1

        cur_luma = float(frame_rgb[::4, ::4].mean()) if frame_rgb.size > 0 else 0.0
        luma_blackout = (
            self._prev_frame_luma is not None
            and self._prev_frame_luma >= 24.0
            and cur_luma < 0.25 * self._prev_frame_luma
        )
        self._prev_frame_luma = cur_luma

        proposals = self.tracker.get_predicted_roi_proposals(next_timestamp_s=ts, include_dormant=True)
        active_props = [pr for pr in proposals if not pr.is_dormant]
        dormant_props = [pr for pr in proposals if pr.is_dormant]

        monitored_boxes: list[tuple[float, float, float, float]] = [
            pr.predicted_bbox_xyxy for pr in active_props
        ]
        confirmed_vtracks = [
            vt for vt in getattr(self.vehicle_tracker, "_tracks", ())
            if getattr(vt, "confirmed", False) and hasattr(vt, "peek_predict") and hasattr(vt, "_bbox")
        ]
        if confirmed_vtracks:
            for vt in confirmed_vtracks:
                monitored_boxes.append(vt._bbox(vt.peek_predict(ts)[0]))
        else:
            for v in self._cached_vehicle_bboxes:
                monitored_boxes.append((v[0], v[1], v[2], v[3]))
        excluded_holes = [dp.predicted_bbox_xyxy for dp in dormant_props]
        scout_info = self.scout.step(
            frame_rgb,
            monitored_boxes_xyxy=monitored_boxes,
            excluded_holes_xyxy=excluded_holes,
        )

        probed_v, low_ncc_v, pre_veh_ncc = self.vehicle_tracker.probe_vehicle_ncc(frame_rgb, ts)
        scout_cut = bool(scout_info.get("scene_cut", False))
        inner_sad = float(scout_info.get("inner_sad", 0.0))
        inner_hist_corr = float(scout_info.get("inner_hist_corr", 1.0))
        veh_cut = (
            (
                probed_v >= 2
                and (low_ncc_v / float(probed_v)) >= 0.60
                and (inner_sad >= 10.0 or inner_hist_corr < 0.92)
            )
            or (probed_v == 1 and low_ncc_v == 1 and (inner_sad >= 15.0 or inner_hist_corr < 0.85))
        )
        scene_cut = bool(scout_cut or veh_cut)

        if scene_cut:
            _handle_scene_cut(self, retired_this_frame)
            luma_blackout = False
            proposals = []
            active_props = []
            dormant_props = []
            confirmed_vtracks = []
            pre_veh_ncc = None
            camera_motion = self._camera_motion_estimator.estimate(frame_rgb, [])
        else:
            camera_motion = self._camera_motion_estimator.estimate(
                frame_rgb, [t.bbox_xyxy for t in self.vehicle_tracker._tracks]
            )

        any_reacquire = any(pr.needs_detector_reacquire for pr in active_props)
        all_reacquire = bool(active_props) and all(pr.needs_detector_reacquire for pr in active_props)
        min_reacq_gap = 3 if (self._ocr_sessions or self._retired_decisions) and self._frame_idx >= 10 else 1
        needs_reacquire = (all_reacquire and self._frames_since_det >= min_reacq_gap) or (any_reacquire and self._frames_since_det >= max(2, min_reacq_gap))

        # Explicit high-load scheduling policy (never silently drops tracks!)
        overload_active = len(active_props) >= self.overload_track_threshold
        eff_hb = (
            min(self.max_heartbeat_interval, self.heartbeat_interval + 2)
            if overload_active
            else self.heartbeat_interval
        )

        unmon_rois = scout_info["unmonitored_rois_xyxy"]
        dormant_reappear = False
        if dormant_props and scout_info["motion_rois_xyxy"]:
            for dp in dormant_props:
                if self._dormant_reappear_cooldown.get(dp.track_id, 0) > 0:
                    continue
                for mx0, my0, mx1, my1 in scout_info["motion_rois_xyxy"]:
                    mbox = (float(mx0), float(my0), float(mx1), float(my1))
                    if (
                        _bbox_iou_xyxy(dp.predicted_bbox_xyxy, mbox) >= 0.05
                        or _bbox_iomin_xyxy(dp.predicted_bbox_xyxy, mbox) >= 0.15
                    ):
                        dormant_reappear = True
                        self._dormant_reappear_cooldown[dp.track_id] = 3
                        break
                if dormant_reappear:
                    break

        min_trig_gap = 3 if (self._ocr_sessions or self._retired_decisions) and self._frame_idx >= 10 else 1
        interior_empty_trigger = (
            len(active_props) == 0
            and len(unmon_rois) > 0
            and self._empty_scene_cooldown == 0
            and self._frames_since_det >= min_trig_gap
        )
        border_trigger = bool(scout_info["has_border_entry"]) and (
            (len(active_props) == 0 and self._empty_scene_cooldown == 0 and self._frames_since_det >= min_trig_gap)
            or (len(active_props) > 0 and self._frames_since_det >= 5)
        )
        any_veh_ncc_drop = (
            bool(pre_veh_ncc)
            and (
                (len(active_props) == 0 and any(res[1] < 0.40 for res in pre_veh_ncc.values()) and self._frames_since_det >= 2)
                or (all(res[1] < 0.40 for res in pre_veh_ncc.values()) and self._frames_since_det >= 3)
            )
        )
        veh_starved_trigger = (
            len(confirmed_vtracks) == 0
            and len(active_props) == 0
            and self._empty_scene_cooldown == 0
            and self._frames_since_det >= 2
        )

        run_full_det = (
            self._frame_idx == 0
            or scene_cut
            or self._frames_since_det >= eff_hb
            or (not is_dual_quad)
            or interior_empty_trigger
            or border_trigger
            or needs_reacquire
            or dormant_reappear
            or any_veh_ncc_drop
            or veh_starved_trigger
        )

        if run_full_det:
            if dormant_reappear or (needs_reacquire and len(dormant_props) > 0):
                pre_state = "RECOVER"
            elif len(active_props) == 0 and len(unmon_rois) > 0:
                pre_state = "ACQUIRE"
            else:
                pre_state = self.state

            if is_temp_det:
                raw_res, h_next = decode_temporal_det_predictions(
                    self.det_model,  # type: ignore[arg-type]
                    frame_rgb,
                    device=self.dev,
                    input_hw=self.input_hw,
                    plate_thresh=self.plate_thresh,
                    enable_far_zoom=self.enable_far_zoom,
                    hidden_state=self._det_hidden_state,
                    frame_key=frame_key,
                    scout_rois_xyxy=unmon_rois[:4],
                )
                if not luma_blackout:
                    self._det_hidden_state = h_next
            else:
                raw_res = decode_plate_det_predictions(
                    self.det_model,
                    frame_rgb,
                    device=self.dev,
                    input_hw=self.input_hw,
                    plate_thresh=self.plate_thresh,
                    enable_far_zoom=self.enable_far_zoom,
                )

            filtered_vehicles = _vehicle_containment_nms(list(raw_res.vehicles))
            if filtered_vehicles or self.vehicle_tracker._tracks:
                tracked_vehicles = self.vehicle_tracker.step(
                    filtered_vehicles,
                    timestamp_s=ts,
                    img_hw=frame_rgb.shape[:2],
                    action_meta=ObservationActionMeta(detector_ran_full=True, frame_key=frame_key),
                    camera_motion=camera_motion, frame_rgb=frame_rgb,
                    precomputed_veh_ncc=pre_veh_ncc,
                )
            else:
                tracked_vehicles = []

            self._cached_vehicles = list(tracked_vehicles)
            self._cached_vehicle_bboxes = (
                [(*v.bbox_xyxy, float(v.score)) for v in tracked_vehicles]
                if tracked_vehicles
                else ([(*v.bbox_xyxy, float(v.score)) for v in filtered_vehicles] if filtered_vehicles else [])
            )
            if (
                (interior_empty_trigger or (len(active_props) == 0 and border_trigger))
                and not raw_res.plates
                and not filtered_vehicles
            ):
                self._empty_scene_cooldown = 3
            self._frames_since_det = 0
            self._full_det_calls += 1

            det_candidates = list(raw_res.plates)
            if is_temp_det and self.enable_far_zoom and tracked_vehicles:
                det_candidates.extend(
                    _run_full_face_vehicle_zoom(self, frame_rgb, tracked_vehicles, det_candidates, frame_key)
                )
        else:
            pre_state = "TRACK" if active_props else ("OCCLUDED" if dormant_props else "SEARCH")
            self._frames_since_det += 1
            tracked_vehicles = self.vehicle_tracker.step([], timestamp_s=ts,
                img_hw=frame_rgb.shape[:2], action_meta=ObservationActionMeta(frame_key=frame_key),
                camera_motion=camera_motion, frame_rgb=frame_rgb,
                precomputed_veh_ncc=pre_veh_ncc)
            live_confirmed = [
                vt for vt in getattr(self.vehicle_tracker, "_tracks", ())
                if getattr(vt, "confirmed", False)
                and (getattr(vt, "veh_ref", None) is None or float(getattr(vt, "patch_ncc", 1.0)) >= 0.40)
            ]
            if live_confirmed:
                self._cached_vehicle_bboxes = [(*vt.bbox_xyxy, float(vt.r)) for vt in live_confirmed]
            else:
                self._cached_vehicle_bboxes = []
                self._cached_vehicles = []
            det_candidates = []

        for candidate in det_candidates:
            candidate.frame_key = frame_key
        search_requests = (_search_requests(self, det_candidates, ts, frame_rgb.shape[:2])
                           if is_dual_quad else [])
        det_candidates.extend(
            _search_inputs(
                search_requests,
                frame_key,
                frame_rgb=frame_rgb,
                opt_refs=self._opt_refs,
                prev_corners=self._prev_corners,
            )
        )
        action_meta = ObservationActionMeta(action_id=frame_key, detector_ran_full=run_full_det,
            searched_rois_xyxy=tuple(r.roi_xyxy for r in search_requests), frame_key=frame_key)

        refine_extra_kw: dict[str, Any] = {"vehicles": tracked_vehicles} if tracked_vehicles else {}
        refined = refine_and_crop_detections(
            self.quad_model,
            frame_rgb,
            det_candidates,
            device=self.dev,
            min_is_plate_prob=self.min_is_plate_prob,
            ocr_margin_ratio=ocr_margin,
            ocr_out_hw=ocr_hw,
            pixel_center_aligned=use_v2_crop,
            **refine_extra_kw,
        )

        veh_vels = (
            self.vehicle_tracker.bind_plates_and_sync(
                tracked_vehicles, refined, update_vehicle_plate_lists=False,
                img_hw=frame_rgb.shape[:2],
            )
            if tracked_vehicles
            else None
        )

        plates = self.tracker.step(
            refined,
            timestamp_s=ts,
            action_meta=action_meta, camera_motion=camera_motion, frame_rgb=frame_rgb,
            vehicle_velocities=veh_vels,
        )
        if tracked_vehicles:
            self.vehicle_tracker.bind_plates_and_sync(
                tracked_vehicles, plates, update_vehicle_plate_lists=True,
                img_hw=frame_rgb.shape[:2], plate_tracker=self.tracker,
            )
            self._cached_vehicles = list(tracked_vehicles)
        plates = _suppress_stationary_corner_logos(self, plates, frame_rgb.shape[:2])
        relinked_this_frame = list(self.tracker.last_relinked_pairs)

        _relink_sessions(self, relinked_this_frame)

        for exp_tid in self.tracker.last_expired_track_ids:
            self._prev_corners.pop(exp_tid, None)
            self._opt_refs.pop(exp_tid, None)
            self._sr_states.pop(exp_tid, None)
            self._track_corner_stats.pop(exp_tid, None)
            self._dormant_reappear_cooldown.pop(exp_tid, None)
            exp_sess = self._ocr_sessions.pop(exp_tid, None)
            if exp_sess is not None:
                exp_dec = _finish_cached(self, exp_tid, exp_sess)
                self._retired_decisions[exp_tid] = exp_dec
                self._terminal_decisions.pop(exp_tid, None)
                retired_this_frame[exp_tid] = exp_dec
                while len(self._retired_decisions) > self.max_retired_history:
                    self._retired_decisions.pop(next(iter(self._retired_decisions)))

        _apply_optical_refinement_and_template_update(
            frame_rgb, plates, self._opt_refs, self._prev_corners
        )

        for p in plates:
            if p.is_coasted or p.quad_edge_touch or not np.isfinite(p.support_nonconformity):
                p.ocr_crop_rgb = None
                p.tight_crop_rgb = None
                p.geom_meta = None
                continue
            q_exact = p.raw_corners_xy if p.raw_corners_xy is not None else p.corners_xy
            if p.tight_crop_rgb is None:
                p.tight_crop_rgb = warp_plate_quad(frame_rgb, q_exact, out_hw=(64, 128), margin_ratio=0.0)
            if p.ocr_crop_rgb is None:
                if use_v2_crop:
                    p.ocr_crop_rgb = warp_plate_canonical(
                        frame_rgb, q_exact, out_hw=ocr_hw, margin_ratio=ocr_margin
                    )
                else:
                    p.ocr_crop_rgb = warp_plate_quad(
                        frame_rgb, q_exact, out_hw=ocr_hw, margin_ratio=ocr_margin
                    )
            if (
                use_v2_crop
                and p.track_id is not None
                and ocr_hw == (80, 160)
                and float(np.linalg.norm(q_exact[1] - q_exact[0]) + np.linalg.norm(q_exact[2] - q_exact[3])) * 0.5 < 56.0
                and float(frame_rgb[::4, ::4].std()) > 2.0
            ):
                sr_st = self._sr_states.setdefault(p.track_id, _CanonicalDrizzleState())
                q_exact, p.ocr_crop_rgb = sr_st.step(frame_rgb, q_exact, p.ocr_crop_rgb, margin_ratio=ocr_margin)
            prev_q = self._prev_corners.get(p.track_id) if p.track_id is not None else None
            p.geom_meta = extract_quad_geometry_meta(q_exact, p.ocr_crop_rgb, prev_corners_xy=prev_q)
            if p.track_id is not None:
                self._prev_corners[p.track_id] = np.asarray(q_exact, dtype=np.float32).copy()

        ocr_map: dict[int, str] = {}
        ocr_verified_map: dict[int, str] = {}
        ocr_tiers: dict[int, str] = {}
        ocr_verified_fields: dict[int, list[str]] = {}
        ocr_decisions: dict[int, Any] = {}
        ocr_push_count = 0

        if self.ocr_model is not None and self.ocr_vocab is not None and is_v2_ocr:
            from vpocr_ocr.session_v2 import MFPlateOCRSessionV2, SessionConfigV2

            sess_cfg_v2 = _resolve_patent_clean_v2_session_config(
                self.ocr_session_config, SessionConfigV2
            )
            tracked_v2_entries: list[dict[str, Any]] = []
            tracked_v2_keys: list[int] = []
            for p in plates:
                if p.is_coasted or p.ocr_crop_rgb is None or p.track_id is None:
                    continue
                key = p.track_id
                if key not in self._ocr_sessions:
                    self._ocr_sessions[key] = MFPlateOCRSessionV2(
                        self.ocr_model,
                        vocab=self.ocr_vocab,
                        config=sess_cfg_v2,
                        device=self.dev,
                    )
                else:
                    _enforce_patent_clean_ocr_session_config(
                        getattr(self._ocr_sessions[key], "config", None)
                    )
                if not _collecting(self._ocr_sessions[key]):
                    ocr_decisions[key] = _finish_cached(self, key, self._ocr_sessions[key])
                    continue
                tracked_v2_keys.append(key)
                tracked_v2_entries.append(_ocr_entry(self._ocr_sessions[key], p, ts, self._frame_idx))

            if tracked_v2_entries:
                batch_decs = _push_v2_ocr_batch_or_loop(MFPlateOCRSessionV2, tracked_v2_entries)
                ocr_push_count = len(batch_decs)
                for key, dec in zip(tracked_v2_keys, batch_decs):
                    ocr_decisions[key] = dec
                    if not _collecting(self._ocr_sessions[key]):
                        self._terminal_decisions[key] = dec
                    if dec.plate_text is not None and dec.status == "ACCEPTED":
                        ocr_tiers[key] = dec.confidence_tier
                        ocr_verified_fields[key] = list(dec.verified_fields)
                        v_txt = dec.format_verified_text()
                        if v_txt is not None:
                            ocr_verified_map[key] = v_txt
                        if dec.confidence_tier == "FULL_4FIELD":
                            ocr_map[key] = dec.plate_text.full_text

        # Update stream state machine cleanly
        observed_now = [p for p in plates if not p.is_coasted]
        coasted_now = [p for p in plates if p.is_coasted]
        if relinked_this_frame or (pre_state == "RECOVER" and observed_now):
            self.state = "RECOVER"
        elif retired_this_frame and not observed_now and not coasted_now:
            self.state = "EXIT"
        elif observed_now:
            self.state = "ACQUIRE" if pre_state == "ACQUIRE" else "TRACK"
        elif coasted_now or self.tracker.dormant_track_ids:
            self.state = "OCCLUDED"
        else:
            self.state = "SEARCH"

        self._frame_idx += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        duty_cycle = float(self._full_det_calls) / float(max(1, self._frame_idx))
        return FrameDetectionResult(
            plates=plates,
            vehicle_bboxes_xyxy=list(self._cached_vehicle_bboxes),
            vehicles=list(tracked_vehicles),
            latency_ms=elapsed_ms,
            ocr_decisions=ocr_decisions,
            metadata={
                "cascade_state": self.state,
                "ran_full_detector": run_full_det,
                "scene_cut": bool(scene_cut),
                "dormant_reappear_triggered": bool(dormant_reappear),
                "luma_blackout_frozen": bool(luma_blackout),
                "overload_active": overload_active,
                "effective_heartbeat_interval": eff_hb,
                "detector_duty_cycle": round(duty_cycle, 4),
                "scout_jitter_xy": scout_info["jitter_xy"],
                "scout_unmonitored_rois": len(unmon_rois),
                "ocr_push_count": ocr_push_count,
                "ocr_accepted": ocr_map,
                "ocr_accepted_verified": ocr_verified_map,
                "ocr_tiers": ocr_tiers,
                "ocr_verified_fields": ocr_verified_fields,
                "ocr_decisions": ocr_decisions,
                "retired_track_decisions": retired_this_frame,
                "relinked_track_pairs": relinked_this_frame,
                "dormant_track_ids": list(self.tracker.dormant_track_ids),
            },
        )


def draw_detections_overlay(frame_rgb: np.ndarray, res: FrameDetectionResult) -> np.ndarray:
    """Render clean diagnostic visualization showing heavy-vehicle vehicle silhouettes, road footprints, & 4-corner plates."""
    vis = frame_rgb.copy()
    h, w = vis.shape[:2]

    # Draw Gen-4 crisp `64x64` vehicle silhouette cutouts + Z=0 road-plane footprints when present
    if res.vehicles:
        for v in res.vehicles:
            vx0, vy0, vx1, vy1 = v.bbox_xyxy
            ix0 = max(0, min(w - 1, int(round(vx0))))
            iy0 = max(0, min(h - 1, int(round(vy0))))
            ix1 = max(ix0 + 1, min(w, int(round(vx1))))
            iy1 = max(iy0 + 1, min(h, int(round(vy1))))
            if v.cutout_mask_64 is not None and (ix1 - ix0) >= 4 and (iy1 - iy0) >= 4:
                m_up = cv2.resize(v.cutout_mask_64, (ix1 - ix0, iy1 - iy0), interpolation=cv2.INTER_LINEAR)
                m_bin = (m_up >= 0.48).astype(np.uint8)
                roi = vis[iy0:iy1, ix0:ix1]
                tint = np.array([35, 210, 245], dtype=np.float32)
                alpha = (m_up[:, :, None] * 0.22).astype(np.float32)
                vis[iy0:iy1, ix0:ix1] = np.clip(roi.astype(np.float32) * (1.0 - alpha) + tint * alpha, 0, 255).astype(
                    np.uint8
                )
                contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(vis[iy0:iy1, ix0:ix1], contours, -1, (40, 230, 255), 1, lineType=cv2.LINE_AA)
            else:
                cv2.rectangle(vis, (ix0, iy0), (ix1, iy1), (60, 195, 235), 1, lineType=cv2.LINE_AA)

            if v.footprint_valid and v.ground_footprint_xy is not None:
                fp_pts = np.round(v.ground_footprint_xy).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [fp_pts], isClosed=True, color=(255, 190, 50), thickness=1, lineType=cv2.LINE_AA)
    else:
        for vx0, vy0, vx1, vy1, _ in res.vehicle_bboxes_xyxy:
            cv2.rectangle(
                vis,
                (int(round(vx0)), int(round(vy0))),
                (int(round(vx1)), int(round(vy1))),
                (60, 195, 235),
                1,
                lineType=cv2.LINE_AA,
            )

    # Draw high-contrast lime-green 4-corner quads + corner dots
    for p in res.plates:
        pts = np.round(p.corners_xy).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(40, 255, 90), thickness=2, lineType=cv2.LINE_AA)
        for cx, cy in p.corners_xy:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 2, (255, 70, 70), -1, lineType=cv2.LINE_AA)
        lx = max(2, int(round(p.bbox_xyxy[0])))
        ly = max(14, int(round(p.bbox_xyxy[1])) - 4)
        label = f"#{p.track_id} {p.score:.2f}" if p.track_id is not None else f"{p.score:.2f}"
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 25, 15), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 120), 1, cv2.LINE_AA)

    # Paste top-4 tight plate-only cutouts in top-right corner strip
    inset_x = w - 134
    inset_y = 6
    for p in res.plates[:4]:
        if p.tight_crop_rgb is None or inset_x < 4 or inset_y + 68 > h:
            break
        cv2.rectangle(vis, (inset_x - 2, inset_y - 2), (inset_x + 129, inset_y + 65), (40, 255, 90), 2)
        vis[inset_y : inset_y + 64, inset_x : inset_x + 128] = p.tight_crop_rgb
        inset_y += 72

    return vis
