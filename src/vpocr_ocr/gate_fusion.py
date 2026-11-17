"""Multi-camera uncalibrated gate consolidation (`GatePassConsolidator`) and privacy `CabinRedactor`.

Handles arbitrary 1..N site camera topologies (e.g. single front camera, front + rear,
front + pure-side + rear, or simultaneous opposing vehicles) in latent EvidenceMemory space
without requiring calibrated 3D extrinsics or patent-encumbered string-frequency voting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import cv2
import numpy as np

from vpocr_ocr.session import OCRDecision, SessionConfig, build_decision_from_arrays
from vpocr_ocr.vocab import JPPlateGrammar, NUM_SLOTS, PlateVocab


class CabinRedactor:
    """Irreversible edge-side privacy redactor for vehicle cabin/windshield & pedestrian regions.

    Ensures raw driver/passenger faces are never persisted to EventStore or synced externally.
    """

    def __init__(self, mosaic_block_px: int = 16, fallback_upper_ratio: float = 0.48) -> None:
        self.mosaic_block_px = max(4, int(mosaic_block_px))
        self.fallback_upper_ratio = float(np.clip(fallback_upper_ratio, 0.20, 0.75))

    def _pixelate_roi(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h <= 1 or w <= 1:
            return roi
        sh = max(1, h // self.mosaic_block_px)
        sw = max(1, w // self.mosaic_block_px)
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
        pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.GaussianBlur(pix, (5, 5), sigmaX=1.5)

    def redact(
        self,
        frame_rgb: np.ndarray,
        vehicle_bbox_xyxy: tuple[float, float, float, float] | None = None,
        cabin_polygon_xy: np.ndarray | Sequence[tuple[float, float]] | None = None,
        extra_person_bboxes: Sequence[tuple[float, float, float, float]] | None = None,
    ) -> np.ndarray:
        """Return a copy of `frame_rgb` with cabin windshield and person regions irreversibly redacted."""
        out = np.asarray(frame_rgb, dtype=np.uint8).copy()
        h, w = out.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if cabin_polygon_xy is not None:
            pts = np.asarray(cabin_polygon_xy, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
        elif vehicle_bbox_xyxy is not None:
            # Conservative fallback: redact upper cabin portion of detected vehicle BBox
            x0, y0, x1, y1 = vehicle_bbox_xyxy
            ix0 = int(np.clip(np.floor(x0), 0, w))
            iy0 = int(np.clip(np.floor(y0), 0, h))
            ix1 = int(np.clip(np.ceil(x1), 0, w))
            iy1 = int(np.clip(np.ceil(y0 + (y1 - y0) * self.fallback_upper_ratio), 0, h))
            if ix1 > ix0 and iy1 > iy0:
                mask[iy0:iy1, ix0:ix1] = 255

        if extra_person_bboxes:
            for px0, py0, px1, py1 in extra_person_bboxes:
                ix0 = int(np.clip(np.floor(px0), 0, w))
                iy0 = int(np.clip(np.floor(py0), 0, h))
                ix1 = int(np.clip(np.ceil(px1), 0, w))
                iy1 = int(np.clip(np.ceil(py1), 0, h))
                if ix1 > ix0 and iy1 > iy0:
                    mask[iy0:iy1, ix0:ix1] = 255

        ys, xs = np.nonzero(mask > 0)
        if ys.size == 0:
            return out

        ry0, ry1 = int(ys.min()), int(ys.max()) + 1
        rx0, rx1 = int(xs.min()), int(xs.max()) + 1
        blurred_sub = self._pixelate_roi(out[ry0:ry1, rx0:rx1])
        sub_m = mask[ry0:ry1, rx0:rx1, None] > 0
        out[ry0:ry1, rx0:rx1] = np.where(sub_m, blurred_sub, out[ry0:ry1, rx0:rx1])
        return out


@dataclass
class CameraTrackObservation:
    """Latent EvidenceMemory summary produced by one camera's local vehicle track."""

    camera_id: int
    local_track_id: str
    mount_role: str                  # e.g. "front", "rear", "pure_side", "oblique", "overhead"
    mem_feat: np.ndarray             # (12, D) float32
    mem_mass: np.ndarray             # (12,) float32
    mem_conflict: np.ndarray         # (12,) float32
    num_frames_seen: int
    t_start_s: float = 0.0
    t_end_s: float = 0.0


@dataclass
class MergedLatentEvidence:
    """Result of convex latent fusion across two or more camera track observations."""

    mem_feat: np.ndarray             # (12, D) float32
    mem_mass: np.ndarray             # (12,) float32
    mem_conflict: np.ndarray         # (12,) float32
    max_cross_conflict: float
    total_frames_seen: int
    camera_ids: list[int] = field(default_factory=list)
    local_track_ids: list[str] = field(default_factory=list)


@dataclass
class GatePassEvent:
    """Finalized multi-camera gate vehicle pass event ready for EventStore."""

    event_id: str
    decision: OCRDecision
    camera_ids: list[int]
    local_track_ids: list[str]
    t_start_s: float
    t_end_s: float
    redacted_snapshot_rgb: np.ndarray | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize privacy-safe structured event metadata (never contains raw unredacted pixels or unverified fields)."""
        is_acc = self.decision.status == "ACCEPTED"
        ver_txt = (
            self.decision.format_verified_text()
            if hasattr(self.decision, "format_verified_text")
            else (self.decision.formatted_text if is_acc else None)
        )
        vset = set(getattr(self.decision, "verified_fields", ()))
        tier = getattr(self.decision, "confidence_tier", "NONE")
        if is_acc and self.decision.plate_text is not None:
            raw_d = self.decision.plate_text.to_dict()
            if tier == "FULL_4FIELD" or not hasattr(self.decision, "verified_fields"):
                pt_dict: dict[str, Any] | None = raw_d
            else:
                pt_dict = {
                    "region": raw_d["region"] if "region" in vset else None,
                    "classification": raw_d["classification"] if "classification" in vset else None,
                    "hiragana": raw_d["hiragana"] if "hiragana" in vset else None,
                    "serial": raw_d["serial"] if "serial" in vset else None,
                    "formatted_serial": raw_d["formatted_serial"] if "serial" in vset else None,
                }
        else:
            pt_dict = None
        return {
            "event_id": self.event_id,
            "status": self.decision.status,
            "confidence_tier": tier,
            "verified_fields": list(getattr(self.decision, "verified_fields", ())) if is_acc else [],
            "plate_validity_score": round(float(getattr(self.decision, "plate_validity_score", 0.0)), 4),
            "formatted_text": ver_txt,
            "plate_text": pt_dict,
            "plate_type": self.decision.plate_type,
            "field_confidences": self.decision.field_confidences,
            "conflict_score": round(float(self.decision.conflict_score), 4),
            "camera_ids": self.camera_ids,
            "local_track_ids": self.local_track_ids,
            "num_frames_seen": self.decision.num_frames_seen,
            "t_start_s": round(float(self.t_start_s), 3),
            "t_end_s": round(float(self.t_end_s), 3),
            "has_redacted_snapshot": self.redacted_snapshot_rgb is not None,
        }


def merge_latent_observations(
    obs_a: MergedLatentEvidence | CameraTrackObservation,
    obs_b: CameraTrackObservation,
) -> tuple[MergedLatentEvidence, float]:
    """Convexly merge two camera tracks' latent EvidenceMemory tensors in R^(12 x D).

    Returns `(merged_evidence, pairwise_max_cross_conflict)`.
    Pure-side or occluded slots (`mass ~ 0`) naturally receive zero weight, while opposing
    distinct vehicles produce a large `pairwise_max_cross_conflict` on overlapping slots.
    """
    fa = np.asarray(obs_a.mem_feat, dtype=np.float32).reshape(NUM_SLOTS, -1)
    ma = np.asarray(obs_a.mem_mass, dtype=np.float32).reshape(NUM_SLOTS)
    ca = np.asarray(obs_a.mem_conflict, dtype=np.float32).reshape(NUM_SLOTS)

    fb = np.asarray(obs_b.mem_feat, dtype=np.float32).reshape(NUM_SLOTS, -1)
    mb = np.asarray(obs_b.mem_mass, dtype=np.float32).reshape(NUM_SLOTS)
    cb = np.asarray(obs_b.mem_conflict, dtype=np.float32).reshape(NUM_SLOTS)

    na = fa / (np.linalg.norm(fa, axis=-1, keepdims=True) + 1e-6)
    nb = fb / (np.linalg.norm(fb, axis=-1, keepdims=True) + 1e-6)
    cos_sim = np.clip(np.sum(na * nb, axis=-1), -1.0, 1.0)

    both_active = (
        np.clip((ma - 0.18) / 0.22, 0.0, 1.0)
        * np.clip((mb - 0.18) / 0.22, 0.0, 1.0)
    )
    cross_conf = (
        both_active
        * (1.0 - cos_sim)
        * np.minimum(np.clip(ma, 0.0, 2.0), np.clip(mb, 0.0, 2.0))
    )
    max_cross = float(np.max(cross_conf))

    new_mass = ma + mb
    denom = np.maximum(new_mass, 1e-6)
    w_b = (mb / denom)[:, None]
    new_feat = np.where((new_mass > 1e-6)[:, None], fa + w_b * (fb - fa), 0.0).astype(np.float32)
    new_conf = (ca + cb + cross_conf).astype(np.float32)

    if isinstance(obs_a, MergedLatentEvidence):
        cids = list(obs_a.camera_ids) + [int(obs_b.camera_id)]
        tids = list(obs_a.local_track_ids) + [str(obs_b.local_track_id)]
        prev_frames = obs_a.total_frames_seen
        prev_max_cross = obs_a.max_cross_conflict
    else:
        cids = [int(obs_a.camera_id), int(obs_b.camera_id)]
        tids = [str(obs_a.local_track_id), str(obs_b.local_track_id)]
        prev_frames = int(obs_a.num_frames_seen)
        prev_max_cross = 0.0

    merged = MergedLatentEvidence(
        mem_feat=new_feat,
        mem_mass=new_mass.astype(np.float32),
        mem_conflict=new_conf,
        max_cross_conflict=max(prev_max_cross, max_cross),
        total_frames_seen=prev_frames + int(obs_b.num_frames_seen),
        camera_ids=cids,
        local_track_ids=tids,
    )
    return merged, max_cross


class GatePassConsolidator:
    """Consolidates 1..N uncalibrated camera tracks into privacy-redacted `GatePassEvent`s.

    - Complementary cameras viewing the same vehicle (e.g. front + oblique + rear) fuse
      convexly in latent `EvidenceMemory` before a single final readout.
    - Pure-side cameras (`mem_mass == 0`) contribute zero drift.
    - Simultaneous opposing vehicles (e.g. one entering while another exits) are automatically
      kept separate when latent cross-camera slot conflict exceeds `split_conflict_threshold`.
    """

    def __init__(
        self,
        readout_fn: Any,
        vocab: PlateVocab | None = None,
        session_config: SessionConfig | None = None,
        redactor: CabinRedactor | None = None,
        split_conflict_threshold: float = 0.52,
        max_time_gap_s: float = 8.0,
    ) -> None:
        """Initialize consolidator.

        `readout_fn(mem_feat, mem_mass, mem_conflict)` maps `(1, 12, D), (1, 12), (1, 12)`
        NumPy arrays to `(slot_logits, plate_type_logits, slot_readability_pred, status_logits)`.
        Use `GatePassConsolidator.from_pytorch_model(model)` or `.from_onnx_dir(onnx_dir)`.
        """
        self.readout_fn = readout_fn
        self.vocab = vocab or PlateVocab()
        self.session_config = session_config or SessionConfig()
        self.grammar = JPPlateGrammar(self.vocab)
        self.redactor = redactor or CabinRedactor()
        self.split_conflict_threshold = float(split_conflict_threshold)
        self.max_time_gap_s = float(max_time_gap_s)
        self._event_seq: int = 0

    @classmethod
    def from_pytorch_model(
        cls,
        model: Any,
        vocab: PlateVocab | None = None,
        session_config: SessionConfig | None = None,
        split_conflict_threshold: float = 0.52,
    ) -> GatePassConsolidator:
        import torch

        model.eval()
        dev = next(model.parameters()).device

        def _pt_readout(
            mem_feat: np.ndarray, mem_mass: np.ndarray, mem_conflict: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            with torch.no_grad():
                ro = model.decode_memory(
                    torch.from_numpy(mem_feat).to(dev),
                    torch.from_numpy(mem_mass).to(dev),
                    torch.from_numpy(mem_conflict).to(dev),
                )
            return (
                ro.slot_logits[0].float().cpu().numpy(),
                ro.plate_type_logits[0].float().cpu().numpy(),
                ro.slot_readability_pred[0].float().cpu().numpy(),
                ro.status_logits[0].float().cpu().numpy(),
            )

        return cls(
            readout_fn=_pt_readout,
            vocab=vocab,
            session_config=session_config,
            split_conflict_threshold=split_conflict_threshold,
        )

    @classmethod
    def from_onnx_dir(
        cls,
        onnx_dir: str | Any,
        vocab: PlateVocab | None = None,
        session_config: SessionConfig | None = None,
        use_int8: bool = False,
        intra_op_threads: int = 1,
        split_conflict_threshold: float = 0.52,
    ) -> GatePassConsolidator:
        """Create a PyTorch-free ONNX Runtime consolidator reusing `memory_step_decode(.int8).onnx`."""
        from pathlib import Path
        import onnxruntime as ort

        base = Path(onnx_dir)
        mem_path = base / ("memory_step_decode.int8.onnx" if use_int8 else "memory_step_decode.onnx")
        if not mem_path.is_file():
            raise FileNotFoundError(
                f"Expected ONNX graph {mem_path} (use_int8={use_int8}) does not exist in {base}"
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(intra_op_threads))
        opts.inter_op_num_threads = 1
        mem_sess = ort.InferenceSession(str(mem_path), sess_options=opts, providers=["CPUExecutionProvider"])

        def _ort_readout(
            mem_feat: np.ndarray, mem_mass: np.ndarray, mem_conflict: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            d_dim = int(mem_feat.shape[-1])
            # Passing frame_valid=0.0 performs bit-exact zero-drift readout on merged EvidenceMemory!
            outs = mem_sess.run(
                None,
                {
                    "mem_feat": mem_feat.astype(np.float32),
                    "mem_mass": mem_mass.astype(np.float32),
                    "mem_conflict": mem_conflict.astype(np.float32),
                    "last_timestamp_s": np.array([0.0], dtype=np.float32),
                    "z": np.zeros((1, NUM_SLOTS, d_dim), dtype=np.float32),
                    "slot_quality": np.zeros((1, NUM_SLOTS), dtype=np.float32),
                    "frame_valid": np.zeros((1, 1), dtype=np.float32),
                    "timestamp_s": np.array([0.0], dtype=np.float32),
                },
            )
            return outs[4][0], outs[5][0], outs[6][0], outs[7][0]

        return cls(
            readout_fn=_ort_readout,
            vocab=vocab,
            session_config=session_config,
            split_conflict_threshold=split_conflict_threshold,
        )

    def consolidate_window(
        self,
        observations: Sequence[CameraTrackObservation],
        keyframe_rgb: np.ndarray | None = None,
        vehicle_bbox_xyxy: tuple[float, float, float, float] | None = None,
        cabin_polygon_xy: np.ndarray | Sequence[tuple[float, float]] | None = None,
    ) -> list[GatePassEvent]:
        """Cluster and fuse camera track observations within a gate time window."""
        if not observations:
            return []

        # Order tracks so informative cameras (high mass) anchor clusters before pure-side tracks
        ordered = sorted(
            observations,
            key=lambda o: (-float(np.sum(o.mem_mass)), float(o.t_start_s)),
        )

        clusters: list[tuple[MergedLatentEvidence, float, float]] = []
        for obs in ordered:
            # Pure-side / empty track with zero evidence attaches harmlessly to temporally closest cluster if any
            is_empty = float(np.max(obs.mem_mass)) < 1e-4
            placed = False
            for idx_c, (cl_ev, cl_t0, cl_t1) in enumerate(clusters):
                time_gap = max(0.0, max(obs.t_start_s - cl_t1, cl_t0 - obs.t_end_s))
                if time_gap > self.max_time_gap_s:
                    continue
                cand_ev, pair_conflict = merge_latent_observations(cl_ev, obs)
                if is_empty or pair_conflict < self.split_conflict_threshold:
                    clusters[idx_c] = (
                        cand_ev,
                        min(cl_t0, obs.t_start_s),
                        max(cl_t1, obs.t_end_s),
                    )
                    placed = True
                    break

            if not placed:
                init_ev = MergedLatentEvidence(
                    mem_feat=np.asarray(obs.mem_feat, dtype=np.float32).copy(),
                    mem_mass=np.asarray(obs.mem_mass, dtype=np.float32).copy(),
                    mem_conflict=np.asarray(obs.mem_conflict, dtype=np.float32).copy(),
                    max_cross_conflict=0.0,
                    total_frames_seen=int(obs.num_frames_seen),
                    camera_ids=[int(obs.camera_id)],
                    local_track_ids=[str(obs.local_track_id)],
                )
                clusters.append((init_ev, float(obs.t_start_s), float(obs.t_end_s)))

        redacted_img = (
            self.redactor.redact(
                keyframe_rgb,
                vehicle_bbox_xyxy=vehicle_bbox_xyxy,
                cabin_polygon_xy=cabin_polygon_xy,
            )
            if keyframe_rgb is not None
            else None
        )

        events: list[GatePassEvent] = []
        for cl_ev, t0, t1 in clusters:
            slot_logits, type_logits, read_pred, status_logits = self.readout_fn(
                cl_ev.mem_feat[None, ...],
                cl_ev.mem_mass[None, ...],
                cl_ev.mem_conflict[None, ...],
            )
            dec = build_decision_from_arrays(
                slot_logits=slot_logits,
                plate_type_logits=type_logits,
                slot_readability_pred=read_pred,
                status_logits=status_logits,
                mem_mass=cl_ev.mem_mass,
                mem_conflict=cl_ev.mem_conflict,
                num_frames_seen=cl_ev.total_frames_seen,
                vocab=self.vocab,
                config=self.session_config,
                grammar=self.grammar,
                is_final=True,
            )
            self._event_seq += 1
            events.append(
                GatePassEvent(
                    event_id=f"gate_evt_{self._event_seq:06d}",
                    decision=dec,
                    camera_ids=sorted(set(cl_ev.camera_ids)),
                    local_track_ids=cl_ev.local_track_ids,
                    t_start_s=t0,
                    t_end_s=t1,
                    redacted_snapshot_rgb=redacted_img if len(events) == 0 else None,
                )
            )
        return events


# ==============================================================================
# Gen-2 `MultiCamFusion v2`: Natural-Parameter Hungarian Association & Readout
# ==============================================================================


def linear_sum_assignment_min_cost(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact O(N^3) Kuhn-Munkres / Jonker-Volgenant Hungarian solver in pure NumPy.

    Matches `scipy.optimize.linear_sum_assignment` on any finite `(R, C)` cost matrix without
    requiring `scipy`.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"Expected 2D cost matrix, got shape {cost.shape}")
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    transposed = False
    if n_rows > n_cols:
        cost = cost.T
        n_rows, n_cols = cost.shape
        transposed = True

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            row_costs = cost[i0 - 1]
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = float(row_costs[j - 1] - u[i0] - v[j])
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = float(minv[j])
                        j1 = j
            for j in range(0, n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    row_to_col = np.zeros(n_rows, dtype=np.int64)
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            row_to_col[p[j] - 1] = j - 1

    rows = np.arange(n_rows, dtype=np.int64)
    if transposed:
        order = np.argsort(row_to_col)
        return row_to_col[order], rows[order]
    return rows, row_to_col


def linear_sum_assignment_max_score(
    score_matrix: np.ndarray,
    min_valid_score: float = 0.0,
) -> list[tuple[int, int]]:
    """Globally maximize total bipartite affinity over valid edges `score > min_valid_score`.

    Augments each row with a private slack column of value `min_valid_score` so conflicting or
    out-of-window tracks (`score == -inf` or `<= min_valid_score`) are left cleanly unmatched.
    """
    scores = np.asarray(score_matrix, dtype=np.float64)
    r, c = scores.shape
    if r == 0 or c == 0:
        return []

    invalid_floor = float(min_valid_score) - 1e6
    aug = np.full((r, c + r), invalid_floor, dtype=np.float64)
    finite_scores = np.where(scores > min_valid_score, np.clip(scores, invalid_floor, 1e6), invalid_floor)
    aug[:, :c] = finite_scores
    for i in range(r):
        aug[i, c + i] = float(min_valid_score)

    max_s = float(np.max(aug))
    cost = max_s - aug  # Strictly non-negative finite matrix
    row_ind, col_ind = linear_sum_assignment_min_cost(cost)

    matches: list[tuple[int, int]] = []
    for ri, ci in zip(row_ind.tolist(), col_ind.tolist()):
        if ci < c and float(scores[ri, ci]) > float(min_valid_score):
            matches.append((int(ri), int(ci)))
    return matches


@dataclass
class CameraTrackObservationV2:
    """Patent-clean continuous natural-parameter summary of a single-camera vehicle track.

    Contains ZERO decoded strings and ZERO character vocabulary logits.
    """

    camera_id: int
    local_track_id: str
    mount_role: str                  # e.g. "front", "rear", "front_rgb", "front_ir", "oblique_left", "telephoto"
    mem_h: np.ndarray                # (12, D) float32 natural-parameter info vector
    mem_P: np.ndarray                # (12, D) float32 diagonal precision
    mem_conflict: np.ndarray         # (12,) float32 intra-track conflict
    color_h: np.ndarray              # (Dc,) float32 decoupled color info vector
    color_P: np.ndarray              # (Dc,) float32 decoupled color precision (== 0 on monochrome IR!)
    pol_h: float = 0.0
    pol_P: float = 0.0
    num_frames_seen: int = 1
    t_start_s: float = 0.0
    t_end_s: float = 0.0
    scale_slope: float = 0.0         # d(log2_width)/dt (>0 approaching camera, <0 receding)
    plate_side: int = 0              # 0=front, 1=rear, 2=pure_side/unknown
    mean_sharpness: float = 0.20     # 2x-pooled interior stroke gradient RMS in (0, 1]
    quad_prob: float = 1.0           # Bounded unique-frame PlateQuadNet objectness evidence
    upper_optical_ok: bool = True    # True when >=1 frame in track passed upper-row optical gate
    prefix_crop_complete: bool = True  # True when >=1 frame in track was free of interior ROI truncation
    max_g0: float = 0.0              # Peak normalized plate width g0 = w_px / 320 across track


@dataclass
class MergedNaturalEvidenceV2:
    """Exact Bayesian natural-parameter fusion across 1..N associated camera tracks."""

    mem_h: np.ndarray
    mem_P: np.ndarray
    mem_conflict: np.ndarray
    color_h: np.ndarray
    color_P: np.ndarray
    pol_h: float
    pol_P: float
    max_cross_mahal2: float
    total_frames_seen: int
    t_start_s: float
    t_end_s: float
    lane_motion_sign: float
    plate_side: int
    mean_sharpness: float = 0.20
    quad_prob: float = 1.0
    upper_optical_ok: bool = True
    prefix_crop_complete: bool = True
    max_g0: float = 0.0
    camera_ids: list[int] = field(default_factory=list)
    local_track_ids: list[str] = field(default_factory=list)
    last_cam_center_ts: dict[int, float] = field(default_factory=dict)
    participants: list[CameraTrackObservationV2] = field(default_factory=list, repr=False)


def _canonical_lane_motion_sign(mount_role: str, scale_slope: float) -> float:
    """Convert camera-local `scale_slope` into lane-canonical travel sign (`>0` forward, `<0` opposing).

    A forward-moving vehicle approaches a `front`/`entry` camera (`scale_slope > 0`) and recedes
    from a `rear`/`exit` camera (`scale_slope < 0`), yielding `+` in both; an opposing-lane vehicle
    recedes from `front` and approaches `rear`, yielding `-` in both!
    """
    role_low = mount_role.lower()
    role_pol = -1.0 if ("rear" in role_low or "exit" in role_low) else 1.0
    return float(scale_slope) * role_pol


def compute_pairwise_natural_mahalanobis(
    mem_h_a: np.ndarray,
    mem_P_a: np.ndarray,
    color_h_a: np.ndarray,
    color_P_a: np.ndarray,
    mem_h_b: np.ndarray,
    mem_P_b: np.ndarray,
    color_h_b: np.ndarray,
    color_P_b: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Compute RMS-normalized harmonic-precision overlap distance & cosine agreement in natural-parameter space.

    Returns `(slot_mahal2 (12,), max_slot_mahal2, color_mahal2, char_agreement_gain)`.
    """
    pa = np.maximum(np.asarray(mem_P_a, dtype=np.float32), 0.0)
    pb = np.maximum(np.asarray(mem_P_b, dtype=np.float32), 0.0)
    ha = np.asarray(mem_h_a, dtype=np.float32)
    hb = np.asarray(mem_h_b, dtype=np.float32)

    mu_a = ha / (pa + 1e-5)
    mu_b = hb / (pb + 1e-5)

    # Per-slot RMS normalization makes overlap distance invariant to LayerNorm(D) vs unit-L2 scaling:
    rms_a = np.sqrt(np.mean(np.square(mu_a), axis=-1, keepdims=True) + 1e-6)
    rms_b = np.sqrt(np.mean(np.square(mu_b), axis=-1, keepdims=True) + 1e-6)
    mu_hat_a = mu_a / rms_a
    mu_hat_b = mu_b / rms_b

    w_harm = (pa * pb) / (pa + pb + 1e-5)  # (12, D)
    w_capped = np.minimum(w_harm, 1.0)
    slot_mahal2 = 0.5 * np.mean(w_capped * np.square(mu_hat_a - mu_hat_b), axis=-1)  # (12,)

    # Cosine agreement weighted by harmonic slot overlap; penalize confident orthogonal/anti-correlated slots
    na = mu_a / (np.linalg.norm(mu_a, axis=-1, keepdims=True) + 1e-6)
    nb = mu_b / (np.linalg.norm(mu_b, axis=-1, keepdims=True) + 1e-6)
    cos_s = np.clip(np.sum(na * nb, axis=-1), -1.0, 1.0)
    ov_s = np.mean(w_harm, axis=-1)  # (12,)
    confident_orth_boost = np.where((ov_s >= 0.35) & (cos_s <= 0.15), 1.30 * (0.22 - cos_s), 0.0).astype(np.float32)
    slot_mahal2 = (slot_mahal2 + confident_orth_boost).astype(np.float32)
    max_slot_m2 = float(np.max(slot_mahal2))
    agree_gain = float(np.sum(np.minimum(ov_s, 1.5) * (cos_s - 0.15)))

    cpa = np.maximum(np.asarray(color_P_a, dtype=np.float32), 0.0)
    cpb = np.maximum(np.asarray(color_P_b, dtype=np.float32), 0.0)
    cha = np.asarray(color_h_a, dtype=np.float32)
    chb = np.asarray(color_h_b, dtype=np.float32)
    cmu_a = cha / (cpa + 1e-5)
    cmu_b = chb / (cpb + 1e-5)
    c_rms_a = np.sqrt(np.mean(np.square(cmu_a), axis=-1, keepdims=True) + 1e-6)
    c_rms_b = np.sqrt(np.mean(np.square(cmu_b), axis=-1, keepdims=True) + 1e-6)
    cmu_hat_a = cmu_a / c_rms_a
    cmu_hat_b = cmu_b / c_rms_b
    wc_harm = np.minimum((cpa * cpb) / (cpa + cpb + 1e-5), 1.0)
    color_m2 = float(0.5 * np.mean(wc_harm * np.square(cmu_hat_a - cmu_hat_b)))

    return slot_mahal2.astype(np.float32), max_slot_m2, color_m2, agree_gain


def fuse_natural_observation_group_v2(
    participants: Sequence[CameraTrackObservationV2],
) -> MergedNaturalEvidenceV2:
    """Simultaneous n-ary, permutation-invariant Optical-MTF natural-parameter fusion across `M >= 1` cameras."""
    if not participants:
        raise ValueError("fuse_natural_observation_group_v2 requires at least 1 observation")
    ordered = sorted(
        participants,
        key=lambda o: (int(o.camera_id), float(o.t_start_s), str(o.local_track_id)),
    )
    if len(ordered) == 1:
        o0 = ordered[0]
        t_mid0 = 0.5 * (float(o0.t_start_s) + float(o0.t_end_s))
        return MergedNaturalEvidenceV2(
            mem_h=np.asarray(o0.mem_h, dtype=np.float32).copy(),
            mem_P=np.maximum(np.asarray(o0.mem_P, dtype=np.float32), 0.0).copy(),
            mem_conflict=np.clip(np.asarray(o0.mem_conflict, dtype=np.float32), 0.0, 6.0).copy(),
            color_h=np.asarray(o0.color_h, dtype=np.float32).copy(),
            color_P=np.maximum(np.asarray(o0.color_P, dtype=np.float32), 0.0).copy(),
            pol_h=float(o0.pol_h),
            pol_P=max(0.0, float(o0.pol_P)),
            max_cross_mahal2=0.0,
            total_frames_seen=int(o0.num_frames_seen),
            t_start_s=float(o0.t_start_s),
            t_end_s=float(o0.t_end_s),
            lane_motion_sign=_canonical_lane_motion_sign(o0.mount_role, o0.scale_slope),
            plate_side=int(o0.plate_side),
            mean_sharpness=max(1e-3, float(o0.mean_sharpness)),
            quad_prob=float(o0.quad_prob),
            upper_optical_ok=bool(o0.upper_optical_ok),
            prefix_crop_complete=bool(o0.prefix_crop_complete),
            max_g0=float(o0.max_g0),
            camera_ids=[int(o0.camera_id)],
            local_track_ids=[str(o0.local_track_id)],
            last_cam_center_ts={int(o0.camera_id): t_mid0},
            participants=list(ordered),
        )

    s_vals = [max(1e-3, float(o.mean_sharpness)) for o in ordered]
    s_peak = max(s_vals)
    weights = [float(np.clip((s_k / s_peak) ** 2, 0.10, 1.0)) for s_k in s_vals]
    w_sum = max(1e-6, float(sum(weights)))

    h_acc = np.zeros_like(np.asarray(ordered[0].mem_h, dtype=np.float64))
    p_acc = np.zeros_like(np.asarray(ordered[0].mem_P, dtype=np.float64))
    conf_acc = np.zeros_like(np.asarray(ordered[0].mem_conflict, dtype=np.float64))
    ch_acc = np.zeros_like(np.asarray(ordered[0].color_h, dtype=np.float64))
    cp_acc = np.zeros_like(np.asarray(ordered[0].color_P, dtype=np.float64))
    pol_h_acc = 0.0
    pol_p_acc = 0.0
    qp_acc = 0.0
    best_sign = 0.0
    chosen_side = 2
    cam_ts: dict[int, float] = {}

    for w_k, o_k in zip(weights, ordered):
        h_acc += w_k * np.asarray(o_k.mem_h, dtype=np.float64)
        p_acc += w_k * np.maximum(np.asarray(o_k.mem_P, dtype=np.float64), 0.0)
        conf_acc += np.maximum(np.asarray(o_k.mem_conflict, dtype=np.float64), 0.0)
        ch_acc += np.asarray(o_k.color_h, dtype=np.float64)
        cp_acc += np.maximum(np.asarray(o_k.color_P, dtype=np.float64), 0.0)
        pol_h_acc += float(o_k.pol_h)
        pol_p_acc += max(0.0, float(o_k.pol_P))
        qp_acc += w_k * float(o_k.quad_prob)
        sg = _canonical_lane_motion_sign(o_k.mount_role, o_k.scale_slope)
        if abs(sg) > abs(best_sign):
            best_sign = sg
        if chosen_side not in (0, 1) and int(o_k.plate_side) in (0, 1):
            chosen_side = int(o_k.plate_side)
        cam_ts[int(o_k.camera_id)] = 0.5 * (float(o_k.t_start_s) + float(o_k.t_end_s))

    max_cross_m2 = 0.0
    max_pair_pen = np.zeros_like(conf_acc, dtype=np.float64)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            slot_m2, pair_m2, _, _ = compute_pairwise_natural_mahalanobis(
                ordered[i].mem_h,
                ordered[i].mem_P,
                ordered[i].color_h,
                ordered[i].color_P,
                ordered[j].mem_h,
                ordered[j].mem_P,
                ordered[j].color_h,
                ordered[j].color_P,
            )
            max_cross_m2 = max(max_cross_m2, pair_m2)
            max_pair_pen = np.maximum(max_pair_pen, np.maximum(0.0, slot_m2.astype(np.float64) - 0.22))

    conf_final = np.clip(conf_acc + max_pair_pen, 0.0, 6.0).astype(np.float32)
    return MergedNaturalEvidenceV2(
        mem_h=h_acc.astype(np.float32),
        mem_P=p_acc.astype(np.float32),
        mem_conflict=conf_final,
        color_h=ch_acc.astype(np.float32),
        color_P=cp_acc.astype(np.float32),
        pol_h=float(pol_h_acc),
        pol_P=float(pol_p_acc),
        max_cross_mahal2=float(max_cross_m2),
        total_frames_seen=sum(int(o.num_frames_seen) for o in ordered),
        t_start_s=min(float(o.t_start_s) for o in ordered),
        t_end_s=max(float(o.t_end_s) for o in ordered),
        lane_motion_sign=float(best_sign),
        plate_side=int(chosen_side if chosen_side in (0, 1) else ordered[0].plate_side),
        mean_sharpness=float(s_peak),
        quad_prob=float(qp_acc / w_sum),
        upper_optical_ok=any(bool(o.upper_optical_ok) for o in ordered),
        prefix_crop_complete=any(bool(o.prefix_crop_complete) for o in ordered),
        max_g0=max(float(o.max_g0) for o in ordered),
        camera_ids=[int(o.camera_id) for o in ordered],
        local_track_ids=[str(o.local_track_id) for o in ordered],
        last_cam_center_ts=cam_ts,
        participants=list(ordered),
    )


def merge_natural_observations_v2(
    cluster: MergedNaturalEvidenceV2,
    obs: CameraTrackObservationV2,
) -> MergedNaturalEvidenceV2:
    """Simultaneous n-ary Optical-MTF natural-parameter fusion when `cluster.participants` is populated."""
    if cluster.participants:
        return fuse_natural_observation_group_v2(list(cluster.participants) + [obs])

    slot_m2, max_m2, _, _ = compute_pairwise_natural_mahalanobis(
        cluster.mem_h,
        cluster.mem_P,
        cluster.color_h,
        cluster.color_P,
        obs.mem_h,
        obs.mem_P,
        obs.color_h,
        obs.color_P,
    )
    obs_sign = _canonical_lane_motion_sign(obs.mount_role, obs.scale_slope)
    merged_sign = (
        cluster.lane_motion_sign
        if abs(cluster.lane_motion_sign) >= abs(obs_sign)
        else obs_sign
    )

    new_cam_ts = dict(cluster.last_cam_center_ts)
    new_cam_ts[int(obs.camera_id)] = 0.5 * (float(obs.t_start_s) + float(obs.t_end_s))

    s_cl = max(1e-3, float(cluster.mean_sharpness))
    s_ob = max(1e-3, float(obs.mean_sharpness))
    s_peak = max(s_cl, s_ob)
    w_cl = np.float32(np.clip((s_cl / s_peak) ** 2, 0.10, 1.0))
    w_ob = np.float32(np.clip((s_ob / s_peak) ** 2, 0.10, 1.0))

    cross_pen = np.maximum(0.0, slot_m2 - 0.22).astype(np.float32)
    return MergedNaturalEvidenceV2(
        mem_h=(w_cl * cluster.mem_h + w_ob * np.asarray(obs.mem_h, dtype=np.float32)).astype(np.float32),
        mem_P=(w_cl * cluster.mem_P + w_ob * np.maximum(np.asarray(obs.mem_P, dtype=np.float32), 0.0)).astype(
            np.float32
        ),
        mem_conflict=np.clip(
            cluster.mem_conflict + np.asarray(obs.mem_conflict, dtype=np.float32) + cross_pen,
            0.0,
            6.0,
        ).astype(np.float32),
        color_h=(cluster.color_h + np.asarray(obs.color_h, dtype=np.float32)).astype(np.float32),
        color_P=(cluster.color_P + np.maximum(np.asarray(obs.color_P, dtype=np.float32), 0.0)).astype(np.float32),
        pol_h=float(cluster.pol_h + float(obs.pol_h)),
        pol_P=float(cluster.pol_P + max(0.0, float(obs.pol_P))),
        max_cross_mahal2=max(cluster.max_cross_mahal2, max_m2),
        total_frames_seen=int(cluster.total_frames_seen) + int(obs.num_frames_seen),
        t_start_s=min(cluster.t_start_s, float(obs.t_start_s)),
        t_end_s=max(cluster.t_end_s, float(obs.t_end_s)),
        lane_motion_sign=merged_sign,
        plate_side=cluster.plate_side if cluster.plate_side in (0, 1) else int(obs.plate_side),
        mean_sharpness=float(s_peak),
        quad_prob=float((float(w_cl) * float(cluster.quad_prob) + float(w_ob) * float(obs.quad_prob)) / max(1e-6, float(w_cl + w_ob))),
        upper_optical_ok=bool(cluster.upper_optical_ok or obs.upper_optical_ok),
        prefix_crop_complete=bool(cluster.prefix_crop_complete or obs.prefix_crop_complete),
        max_g0=max(float(cluster.max_g0), float(obs.max_g0)),
        camera_ids=list(cluster.camera_ids) + [int(obs.camera_id)],
        local_track_ids=list(cluster.local_track_ids) + [str(obs.local_track_id)],
        last_cam_center_ts=new_cam_ts,
    )


class MultiCamTrackAssociator:
    """Uncalibrated heterogeneous multi-camera associator using topology + Mahalanobis + Hungarian.

    Enforces:
    1. `scale_slope` canonical travel-direction veto (separates opposing-lane vehicles above `slope_deadzone`).
    2. Single-slot harmonic Mahalanobis veto `max_s d_Mahal^2 < tau_split` (separates `12-34` vs
       `12-38` close-following fleet vehicles!).
    3. Online self-calibrating inter-camera travel-time prior `N(mu_AB, sigma_AB^2)`.
    4. Global bipartite maximum-weight Hungarian matching across cameras.
    """

    def __init__(
        self,
        tau_split: float = 0.72,
        tau_color_split: float = 1.15,
        slope_deadzone: float = 0.05,
        max_time_gap_s: float = 10.0,
        min_merge_score: float = 0.05,
        min_identity_mass: float = 0.08,
        min_agree_gain: float = 0.08,
    ) -> None:
        self.tau_split = float(tau_split)
        self.tau_color_split = float(tau_color_split)
        self.slope_deadzone = float(slope_deadzone)
        self.max_time_gap_s = float(max_time_gap_s)
        self.min_merge_score = float(min_merge_score)
        self.min_identity_mass = float(min_identity_mass)
        self.min_agree_gain = float(min_agree_gain)
        # Online inter-camera travel-time stats: (cam_i, cam_j) -> [mean_dt, var_dt, count]
        self._travel_time_priors: dict[tuple[int, int], list[float]] = {}

    def get_travel_time_prior(self, cam_a: int, cam_b: int) -> tuple[float, float]:
        """Return `(mu_ab, sigma_ab)` for `t_center(cam_b) - t_center(cam_a)`."""
        key = (int(cam_a), int(cam_b))
        if key in self._travel_time_priors:
            mu, var, _ = self._travel_time_priors[key]
            return mu, max(0.35, float(np.sqrt(max(var, 0.12))))
        return 0.0, 4.0

    def _update_travel_time_prior(self, cam_a: int, cam_b: int, dt: float) -> None:
        key = (int(cam_a), int(cam_b))
        rev_key = (int(cam_b), int(cam_a))
        if key not in self._travel_time_priors:
            self._travel_time_priors[key] = [float(dt), 2.25, 1.0]
            self._travel_time_priors[rev_key] = [-float(dt), 2.25, 1.0]
            return
        mu, var, cnt = self._travel_time_priors[key]
        alpha = max(0.12, 1.0 / (cnt + 2.0))
        diff = float(dt) - mu
        new_mu = mu + alpha * diff
        new_var = max(0.12, (1.0 - alpha) * var + alpha * (diff * diff))
        self._travel_time_priors[key] = [new_mu, new_var, cnt + 1.0]
        self._travel_time_priors[rev_key] = [-new_mu, new_var, cnt + 1.0]

    @staticmethod
    def _init_cluster(obs: CameraTrackObservationV2) -> MergedNaturalEvidenceV2:
        return fuse_natural_observation_group_v2([obs])

    def score_pair(
        self,
        cluster: MergedNaturalEvidenceV2,
        obs: CameraTrackObservationV2,
    ) -> float:
        """Compute pairwise log-affinity (`-inf` if vetoed by direction, time, or Mahalanobis)."""
        if int(obs.camera_id) in cluster.camera_ids:
            return -np.inf

        time_gap = max(0.0, max(float(obs.t_start_s) - cluster.t_end_s, cluster.t_start_s - float(obs.t_end_s)))
        if time_gap > self.max_time_gap_s:
            return -np.inf

        # 1. Travel-direction topology hard veto (separates opposing-lane vehicles above slope_deadzone)
        obs_sign = _canonical_lane_motion_sign(obs.mount_role, obs.scale_slope)
        if (
            abs(cluster.lane_motion_sign) >= self.slope_deadzone
            and abs(obs_sign) >= self.slope_deadzone
            and (cluster.lane_motion_sign * obs_sign < 0.0)
        ):
            return -np.inf

        # Require non-trivial shared identity precision & positive feature agreement (never merge on time_bonus alone!)
        if min(float(np.mean(cluster.mem_P)), float(np.mean(obs.mem_P))) < self.min_identity_mass:
            return -np.inf

        # 2. Single-slot & color overlap distance hard veto (separates 1-char-diff fleet cars!)
        slot_m2, max_m2, color_m2, agree_gain = compute_pairwise_natural_mahalanobis(
            cluster.mem_h,
            cluster.mem_P,
            cluster.color_h,
            cluster.color_P,
            obs.mem_h,
            obs.mem_P,
            obs.color_h,
            obs.color_P,
        )
        if agree_gain < self.min_agree_gain:
            return -np.inf
        if max_m2 >= self.tau_split:
            return -np.inf
        if color_m2 >= self.tau_color_split:
            return -np.inf

        # 3. Online inter-camera travel-time Gaussian log-likelihood
        obs_t_mid = 0.5 * (float(obs.t_start_s) + float(obs.t_end_s))
        time_terms = []
        for prev_cam, prev_t in cluster.last_cam_center_ts.items():
            mu_dt, sig_dt = self.get_travel_time_prior(prev_cam, int(obs.camera_id))
            z_dt = ((obs_t_mid - prev_t) - mu_dt) / sig_dt
            time_terms.append(0.75 - 0.35 * (z_dt * z_dt))
        time_bonus = float(np.mean(time_terms)) if time_terms else 0.5

        mean_m2 = float(np.mean(slot_m2))
        return time_bonus + agree_gain - 1.6 * max_m2 - 0.6 * mean_m2 - 0.5 * color_m2

    def associate(
        self,
        observations: Sequence[CameraTrackObservationV2],
    ) -> list[MergedNaturalEvidenceV2]:
        """Globally associate multi-camera tracks into consolidated natural-parameter vehicles."""
        if not observations:
            return []

        by_cam: dict[int, list[CameraTrackObservationV2]] = {}
        for obs in observations:
            by_cam.setdefault(int(obs.camera_id), []).append(obs)

        # Sort cameras so informative character/color cameras anchor bipartite matching first
        cam_order = sorted(
            by_cam.keys(),
            key=lambda cid: (
                -sum(float(np.sum(o.mem_P)) + float(np.sum(o.color_P)) for o in by_cam[cid]),
                cid,
            ),
        )

        clusters: list[MergedNaturalEvidenceV2] = [
            self._init_cluster(o)
            for o in sorted(
                by_cam[cam_order[0]],
                key=lambda x: (float(x.t_start_s), float(x.t_end_s), str(x.local_track_id)),
            )
        ]

        for cid in cam_order[1:]:
            cam_tracks = sorted(
                by_cam[cid],
                key=lambda x: (float(x.t_start_s), float(x.t_end_s), str(x.local_track_id)),
            )
            score_mat = np.full((len(clusters), len(cam_tracks)), -np.inf, dtype=np.float64)
            for i, cl in enumerate(clusters):
                for j, trk in enumerate(cam_tracks):
                    score_mat[i, j] = self.score_pair(cl, trk)

            matched_pairs = linear_sum_assignment_max_score(
                score_mat, min_valid_score=self.min_merge_score
            )
            matched_cols: set[int] = set()
            for r_idx, c_idx in matched_pairs:
                obs_match = cam_tracks[c_idx]
                prev_cl = clusters[r_idx]
                obs_t_mid = 0.5 * (float(obs_match.t_start_s) + float(obs_match.t_end_s))

                merged_cl = merge_natural_observations_v2(prev_cl, obs_match)

                # Update online travel-time prior on high-confidence character matches
                if merged_cl.max_cross_mahal2 < 0.35 and float(np.sum(obs_match.mem_P)) > 1.0:
                    for p_cam, p_ts in prev_cl.last_cam_center_ts.items():
                        if float(np.sum(prev_cl.mem_P)) > 1.0:
                            self._update_travel_time_prior(p_cam, int(obs_match.camera_id), obs_t_mid - p_ts)

                clusters[r_idx] = merged_cl
                matched_cols.add(c_idx)

            for j, trk in enumerate(cam_tracks):
                if j not in matched_cols:
                    clusters.append(self._init_cluster(trk))

        clusters.sort(key=lambda cl: (cl.t_start_s, cl.camera_ids[0]))
        return clusters


@dataclass
class GatePassEventV2:
    """Finalized Gen-2 multi-camera gate vehicle pass event with calibrated type candidates."""

    event_id: str
    decision: Any  # OCRDecisionV2
    camera_ids: list[int]
    local_track_ids: list[str]
    t_start_s: float
    t_end_s: float
    redacted_snapshot_rgb: np.ndarray | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        is_acc = self.decision.status == "ACCEPTED"
        ver_txt = (
            self.decision.format_verified_text()
            if hasattr(self.decision, "format_verified_text")
            else (self.decision.formatted_text if is_acc else None)
        )
        vset = set(getattr(self.decision, "verified_fields", ()))
        tier = getattr(self.decision, "confidence_tier", "NONE")
        if is_acc and self.decision.plate_text is not None:
            raw_d = self.decision.plate_text.to_dict()
            if tier == "FULL_4FIELD":
                pt_dict: dict[str, Any] | None = raw_d
            else:
                pt_dict = {
                    "region": raw_d["region"] if "region" in vset else None,
                    "classification": raw_d["classification"] if "classification" in vset else None,
                    "hiragana": raw_d["hiragana"] if "hiragana" in vset else None,
                    "serial": raw_d["serial"] if "serial" in vset else None,
                    "formatted_serial": raw_d["formatted_serial"] if "serial" in vset else None,
                }
        else:
            pt_dict = None
        return {
            "event_id": self.event_id,
            "status": self.decision.status,
            "confidence_tier": tier,
            "verified_fields": list(getattr(self.decision, "verified_fields", ())) if is_acc else [],
            "plate_validity_score": round(float(getattr(self.decision, "plate_validity_score", 0.0)), 4),
            "formatted_text": ver_txt,
            "verified_formatted_text": ver_txt,
            "plate_text": pt_dict,
            "plate_type": self.decision.plate_type,
            "plate_type_probs": self.decision.plate_type_probs,
            "plate_type_candidates": self.decision.plate_type_candidates,
            "field_confidences": self.decision.field_confidences,
            "conflict_score": round(float(self.decision.conflict_score), 4),
            "camera_ids": self.camera_ids,
            "local_track_ids": self.local_track_ids,
            "num_frames_seen": self.decision.num_frames_seen,
            "t_start_s": round(float(self.t_start_s), 3),
            "t_end_s": round(float(self.t_end_s), 3),
            "has_redacted_snapshot": self.redacted_snapshot_rgb is not None,
        }


class GatePassConsolidatorV2:
    """Patent-clean Gen-2 multi-camera consolidator (`1` single readout per consolidated vehicle!).

    Supports:
    - Dual RGB + 850/940nm IR (sharp IR `mem_h, mem_P` + blurry RGB `color_h, color_P` -> joint
      character + plate color recovery without camera-specific conditional rules).
    - Left/Right Oblique, Asynchronous Front Entry + Rear Exit, and Overview + Telephoto rigs.
    """

    def __init__(
        self,
        readout_fn: Any,
        vocab: PlateVocab | None = None,
        session_config: Any | None = None,
        associator: MultiCamTrackAssociator | None = None,
        redactor: CabinRedactor | None = None,
        registered_plates: Sequence[str | Any] | None = None,
    ) -> None:
        import dataclasses
        from vpocr_ocr.session_v2 import SessionConfigV2

        self.readout_fn = readout_fn
        self.vocab = vocab or PlateVocab()
        base_cfg = session_config or SessionConfigV2()
        if registered_plates is not None:
            base_cfg = dataclasses.replace(base_cfg, registered_plates=tuple(registered_plates))
        self.session_config = base_cfg
        self.grammar = JPPlateGrammar(self.vocab)
        self.associator = associator or MultiCamTrackAssociator()
        self.redactor = redactor or CabinRedactor()
        self._event_seq: int = 0
        self.num_readout_calls: int = 0

    @classmethod
    def from_pytorch_model(
        cls,
        model: Any,
        vocab: PlateVocab | None = None,
        session_config: Any | None = None,
        associator: MultiCamTrackAssociator | None = None,
        registered_plates: Sequence[str | Any] | None = None,
    ) -> GatePassConsolidatorV2:
        import torch

        model.eval()
        dev = next(model.parameters()).device

        def _pt_readout_v2(
            mem_h: np.ndarray,
            mem_P: np.ndarray,
            mem_conflict: np.ndarray,
            color_h: np.ndarray,
            color_P: np.ndarray,
            pol_h: np.ndarray,
            pol_P: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
            with torch.no_grad():
                ro = model.decode_memory(
                    torch.from_numpy(mem_h).to(dev),
                    torch.from_numpy(mem_P).to(dev),
                    torch.from_numpy(mem_conflict).to(dev),
                    torch.from_numpy(color_h).to(dev),
                    torch.from_numpy(color_P).to(dev),
                    torch.from_numpy(pol_h).to(dev),
                    torch.from_numpy(pol_P).to(dev),
                )
            return (
                ro.slot_logits[0].float().cpu().numpy(),
                ro.plate_type_logits[0].float().cpu().numpy(),
                float(ro.polarity_logit[0].item()),
                ro.slot_readability_pred[0].float().cpu().numpy(),
                ro.status_logits[0].float().cpu().numpy(),
            )

        return cls(
            readout_fn=_pt_readout_v2,
            vocab=vocab,
            session_config=session_config,
            associator=associator,
            registered_plates=registered_plates,
        )

    @classmethod
    def from_onnx_dir(
        cls,
        onnx_dir: str | Any,
        vocab: PlateVocab | None = None,
        session_config: Any | None = None,
        associator: MultiCamTrackAssociator | None = None,
        use_int8: bool = False,
        intra_op_threads: int = 1,
        registered_plates: Sequence[str | Any] | None = None,
    ) -> GatePassConsolidatorV2:
        """Create PyTorch-free consolidator running ONLY `memory_readout(.int8).onnx` once per pass."""
        from pathlib import Path
        import onnxruntime as ort

        base = Path(onnx_dir)
        read_path = base / ("memory_readout.int8.onnx" if use_int8 else "memory_readout.onnx")
        if not read_path.is_file():
            raise FileNotFoundError(
                f"Expected ONNX graph {read_path} (use_int8={use_int8}) does not exist in {base}"
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(intra_op_threads))
        opts.inter_op_num_threads = 1
        read_sess = ort.InferenceSession(str(read_path), sess_options=opts, providers=["CPUExecutionProvider"])

        def _ort_readout_v2(
            mem_h: np.ndarray,
            mem_P: np.ndarray,
            mem_conflict: np.ndarray,
            color_h: np.ndarray,
            color_P: np.ndarray,
            pol_h: np.ndarray,
            pol_P: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
            s_log, pt_log, pol_log, sr_pred, st_log = read_sess.run(
                None,
                {
                    "mem_h": mem_h.astype(np.float32),
                    "mem_P": mem_P.astype(np.float32),
                    "mem_conflict": mem_conflict.astype(np.float32),
                    "color_h": color_h.astype(np.float32),
                    "color_P": color_P.astype(np.float32),
                    "pol_h": pol_h.astype(np.float32),
                    "pol_P": pol_P.astype(np.float32),
                },
            )
            return s_log[0], pt_log[0], float(pol_log[0]), sr_pred[0], st_log[0]

        return cls(
            readout_fn=_ort_readout_v2,
            vocab=vocab,
            session_config=session_config,
            associator=associator,
            registered_plates=registered_plates,
        )

    def consolidate_window(
        self,
        observations: Sequence[CameraTrackObservationV2],
        keyframe_rgb: np.ndarray | None = None,
        vehicle_bbox_xyxy: tuple[float, float, float, float] | None = None,
        cabin_polygon_xy: np.ndarray | Sequence[tuple[float, float]] | None = None,
    ) -> list[GatePassEventV2]:
        """Associate in natural-parameter space and execute ONE final readout per vehicle."""
        from vpocr_ocr.session_v2 import build_decision_v2_from_arrays

        clusters = self.associator.associate(observations)
        if not clusters:
            return []

        redacted_img = (
            self.redactor.redact(
                keyframe_rgb,
                vehicle_bbox_xyxy=vehicle_bbox_xyxy,
                cabin_polygon_xy=cabin_polygon_xy,
            )
            if keyframe_rgb is not None
            else None
        )

        events: list[GatePassEventV2] = []
        for cl in clusters:
            self.num_readout_calls += 1
            slot_logits, type_logits, pol_logit, read_pred, status_logits = self.readout_fn(
                cl.mem_h[None, ...],
                cl.mem_P[None, ...],
                cl.mem_conflict[None, ...],
                cl.color_h[None, ...],
                cl.color_P[None, ...],
                np.array([cl.pol_h], dtype=np.float32),
                np.array([cl.pol_P], dtype=np.float32),
            )
            dec = build_decision_v2_from_arrays(
                slot_logits=slot_logits,
                plate_type_logits=type_logits,
                polarity_logit=pol_logit,
                slot_readability_pred=read_pred,
                status_logits=status_logits,
                mem_P=cl.mem_P,
                mem_conflict=cl.mem_conflict,
                color_P=cl.color_P,
                pol_P=cl.pol_P,
                num_frames_seen=cl.total_frames_seen,
                vocab=self.vocab,
                config=self.session_config,
                grammar=self.grammar,
                plate_side=cl.plate_side,
                is_final=True,
                quad_prob=float(cl.quad_prob),
                upper_optical_ok=bool(cl.upper_optical_ok),
                prefix_crop_complete=bool(cl.prefix_crop_complete),
                max_g0=float(cl.max_g0),
            )
            self._event_seq += 1
            events.append(
                GatePassEventV2(
                    event_id=f"gate_v2_evt_{self._event_seq:06d}",
                    decision=dec,
                    camera_ids=sorted(set(cl.camera_ids)),
                    local_track_ids=cl.local_track_ids,
                    t_start_s=cl.t_start_s,
                    t_end_s=cl.t_end_s,
                    redacted_snapshot_rgb=redacted_img if len(events) == 0 else None,
                )
            )
        return events

    consolidate = consolidate_window
