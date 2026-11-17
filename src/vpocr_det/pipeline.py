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
import time
from typing import Any
import cv2
import numpy as np
import torch

from vpocr_det.detector import (
    PlateDetNet,
    TemporalPlateDetNet,
    UnifiedTemporalDetORT,
    decode_plate_det_predictions,
    decode_temporal_det_predictions,
)
from vpocr_det.quad_cropper import (
    PlateQuadNet,
    PlateQuadTrackNet,
    extract_quad_geometry_meta,
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


def _search_inputs(requests: list[SearchRequest], frame_key: str) -> list[PlateDetection]:
    """Zero-prior geometry carriers for Stage-2 execution, never PMBM updates.

    The current cropper accepts geometry carriers, so adaptation of score-free
    requests is confined to this boundary. Only neural outputs are observations.
    """
    carriers = []
    provenance = getattr(ObservationProvenance, "STAGE2_SEARCH", ObservationProvenance.SEARCH_MEASUREMENT)
    for request in requests:
        assert_not_observation(request)
        x0, y0, x1, y1 = request.roi_xyxy
        carriers.append(PlateDetection(request.roi_xyxy, 0.0,
            np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32),
            raw_det_score=0.0, provenance=provenance, frame_key=frame_key))
    return carriers


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

        if self.use_tracker:
            tracked_vehicles = self.vehicle_tracker.step(
                list(raw_res.vehicles),
                timestamp_s=ts,
                img_hw=frame_rgb.shape[:2],
                action_meta=ObservationActionMeta(detector_ran_full=True, frame_key=frame_key),
                camera_motion=camera_motion, frame_rgb=frame_rgb,
            )
        else:
            tracked_vehicles = list(raw_res.vehicles)

        det_candidates = list(raw_res.plates)
        for candidate in det_candidates:
            candidate.frame_key = frame_key
        search_requests = (_search_requests(self, det_candidates, ts, frame_rgb.shape[:2])
                           if self.use_tracker and self.enable_tracker_roi_reacquire and is_dual_quad else [])
        det_candidates.extend(_search_inputs(search_requests, frame_key))
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

        retired_this_frame: dict[int, Any] = {}
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
            relinked_this_frame = list(self.tracker.last_relinked_pairs)

            _relink_sessions(self, relinked_this_frame)

            for exp_tid in self.tracker.last_expired_track_ids:
                self._prev_corners.pop(exp_tid, None)
                exp_sess = self._ocr_sessions.pop(exp_tid, None)
                if exp_sess is not None:
                    exp_dec = _finish_cached(self, exp_tid, exp_sess)
                    self._retired_decisions[exp_tid] = exp_dec
                    self._terminal_decisions.pop(exp_tid, None)
                    retired_this_frame[exp_tid] = exp_dec
                    while len(self._retired_decisions) > self.max_retired_history:
                        self._retired_decisions.pop(next(iter(self._retired_decisions)))

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
            else list(raw_res.vehicle_bboxes_xyxy)
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
        self._noise_ema: float = 4.0

    def reset(self) -> None:
        self._prev_y = None
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

        if self._prev_y is None:
            self._prev_y = y_cur
            return {
                "motion_rois_xyxy": [],
                "unmonitored_rois_xyxy": [],
                "has_border_entry": False,
                "jitter_xy": (0, 0),
                "noise_floor": self._noise_ema,
            }

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
        return {
            "motion_rois_xyxy": motion_rois,
            "unmonitored_rois_xyxy": unmonitored_rois,
            "has_border_entry": has_border_entry,
            "jitter_xy": (int(best_dx), int(best_dy)),
            "noise_floor": round(self._noise_ema, 3),
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
        heartbeat_interval: int = 6,
        max_heartbeat_interval: int = 10,
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
        camera_motion = self._camera_motion_estimator.estimate(
            frame_rgb, [t.bbox_xyxy for t in self.vehicle_tracker._tracks])
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
        any_reacquire = any(pr.needs_detector_reacquire for pr in active_props)
        all_reacquire = bool(active_props) and all(pr.needs_detector_reacquire for pr in active_props)
        needs_reacquire = all_reacquire or (any_reacquire and self._frames_since_det >= 2)

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

        interior_empty_trigger = (
            len(active_props) == 0
            and len(unmon_rois) > 0
            and self._empty_scene_cooldown == 0
        )
        border_trigger = bool(scout_info["has_border_entry"]) and (
            (len(active_props) == 0 and self._empty_scene_cooldown == 0)
            or (len(active_props) > 0 and self._frames_since_det >= 3)
        )

        run_full_det = (
            self._frame_idx == 0
            or self._frames_since_det >= eff_hb
            or (not is_dual_quad)
            or interior_empty_trigger
            or border_trigger
            or needs_reacquire
            or dormant_reappear
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

            if raw_res.vehicles or self.vehicle_tracker._tracks:
                tracked_vehicles = self.vehicle_tracker.step(
                    list(raw_res.vehicles),
                    timestamp_s=ts,
                    img_hw=frame_rgb.shape[:2],
                    action_meta=ObservationActionMeta(detector_ran_full=True, frame_key=frame_key),
                    camera_motion=camera_motion, frame_rgb=frame_rgb,
                )
            else:
                tracked_vehicles = []

            self._cached_vehicles = list(tracked_vehicles)
            self._cached_vehicle_bboxes = (
                [(*v.bbox_xyxy, float(v.score)) for v in tracked_vehicles]
                if tracked_vehicles
                else list(raw_res.vehicle_bboxes_xyxy)
            )
            if (
                (interior_empty_trigger or (len(active_props) == 0 and border_trigger))
                and not raw_res.plates
                and not raw_res.vehicles
                and not raw_res.vehicle_bboxes_xyxy
            ):
                self._empty_scene_cooldown = 3
            self._frames_since_det = 0
            self._full_det_calls += 1

            det_candidates = list(raw_res.plates)
        else:
            pre_state = "TRACK" if active_props else ("OCCLUDED" if dormant_props else "SEARCH")
            self._frames_since_det += 1
            tracked_vehicles = self.vehicle_tracker.step([], timestamp_s=ts,
                img_hw=frame_rgb.shape[:2], action_meta=ObservationActionMeta(frame_key=frame_key),
                camera_motion=camera_motion, frame_rgb=frame_rgb)
            live_confirmed = [
                vt for vt in getattr(self.vehicle_tracker, "_tracks", ())
                if getattr(vt, "confirmed", False)
            ]
            if live_confirmed:
                self._cached_vehicle_bboxes = [(*vt.bbox_xyxy, float(vt.r)) for vt in live_confirmed]
            det_candidates = []

        for candidate in det_candidates:
            candidate.frame_key = frame_key
        search_requests = (_search_requests(self, det_candidates, ts, frame_rgb.shape[:2])
                           if is_dual_quad else [])
        det_candidates.extend(_search_inputs(search_requests, frame_key))
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
        relinked_this_frame = list(self.tracker.last_relinked_pairs)

        _relink_sessions(self, relinked_this_frame)

        retired_this_frame: dict[int, Any] = {}
        for exp_tid in self.tracker.last_expired_track_ids:
            self._prev_corners.pop(exp_tid, None)
            self._dormant_reappear_cooldown.pop(exp_tid, None)
            exp_sess = self._ocr_sessions.pop(exp_tid, None)
            if exp_sess is not None:
                exp_dec = _finish_cached(self, exp_tid, exp_sess)
                self._retired_decisions[exp_tid] = exp_dec
                self._terminal_decisions.pop(exp_tid, None)
                retired_this_frame[exp_tid] = exp_dec
                while len(self._retired_decisions) > self.max_retired_history:
                    self._retired_decisions.pop(next(iter(self._retired_decisions)))

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
