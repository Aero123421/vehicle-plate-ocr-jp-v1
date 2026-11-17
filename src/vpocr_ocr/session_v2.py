"""Vocabulary-free collection, irreversible sealing, and one terminal Bayes decision.

The Gaussian observation model is approximate and requires held-out calibration.
The runtime invariant targets the specified design-around, not a legal opinion.
"""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, field, InitVar
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import torch
from vpocr_ocr.model_v2 import MemoryStateV2
from vpocr_ocr.session import DecisionStatus, _invoke_grammar_checker
from vpocr_ocr.vocab import JPPlateGrammar, NUM_SLOTS, PlateVocab
from vpocr_synth.plate.models import PlateText


@dataclass
class OCRDecisionV2:
    """Structured output decision of `MFPlateOCRSessionV2` and `GatePassConsolidatorV2`."""

    status: DecisionStatus
    plate_text: PlateText | None
    formatted_text: str | None
    plate_type: str
    field_confidences: dict[str, float]
    slot_confidences: list[float]
    slot_masses: list[float]
    conflict_score: float
    is_saturated: bool
    num_frames_seen: int
    plate_type_probs: dict[str, float] = field(default_factory=dict)
    plate_type_candidates: list[str] = field(default_factory=list)
    color_precision_mean: float = 0.0
    polarity_logit: float = 0.0
    polarity_precision: float = 0.0
    confidence_tier: str = "NONE"
    verified_fields: tuple[str, ...] = ()
    plate_validity_score: float = 0.0
    top2_margin: float = 0.0
    field_margins: dict[str, float] = field(default_factory=dict)
    joint_correctness_probability: float = 0.0
    readable_probability: float = 0.0

    def format_verified_text(self) -> str | None:
        """Return full plate text when `FULL_4FIELD`, or mask unverified fields with fixed-width `'?'`."""
        if self.plate_text is None or self.status != "ACCEPTED":
            return None
        if self.confidence_tier == "FULL_4FIELD":
            return self.plate_text.full_text
        vset = set(self.verified_fields)
        reg = self.plate_text.region if "region" in vset else "??"
        cls = self.plate_text.classification if "classification" in vset else "???"
        hira = self.plate_text.hiragana if "hiragana" in vset else "?"
        ser = self.plate_text.formatted_serial if "serial" in vset else "??-??"
        return f"{reg} {cls} {hira} {ser}"


class SessionStateV2(str, Enum):
    COLLECTING = "COLLECTING"
    SEALED = "SEALED"
    FINALIZED = "FINALIZED"


@dataclass
class TerminalCalibrationV2:
    """Loadable discriminative joint/null calibration; fit on separate tracks.

    Defaults are identity transforms, not a claim of empirical calibration.
    No fleet, class-length or kana-frequency offsets affect acceptance.
    """
    temperature: float = 1.0
    null_temperature: float = 1.0
    null_bias: float = 0.0
    readability_weight: float = 0.0
    support_weight: float = 0.0
    identity_probability: float = 1.0
    conflict_weight: float = 0.0
    mixed_bias: float = 0.0
    mixed_logit_scale: float = 1.0
    readout_precision_ceiling: float = 0.0

    def __post_init__(self):
        values = vars(self)
        if not all(math.isfinite(float(v)) for v in values.values()):
            raise ValueError("Calibration values must be finite")
        if self.temperature <= 0 or self.null_temperature <= 0:
            raise ValueError("Calibration temperatures must be positive")
        if not 0 <= self.identity_probability <= 1:
            raise ValueError("Identity probability must be in [0,1]")
        if self.mixed_logit_scale < 0:
            raise ValueError("Mixed logit scale must be nonnegative")
        if self.readout_precision_ceiling < 0:
            raise ValueError("Readout precision ceiling must be nonnegative")

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text()))


def _apply_sealed_precision_ceiling(active_state, ceiling: float):
    """Bound terminal precision while preserving posterior means and conflict."""
    if ceiling > 0:
        if isinstance(active_state[1], torch.Tensor):
            mp_mean = float(torch.clamp(active_state[1].float(), min=0.0).mean().item())
            if mp_mean > ceiling:
                sc = float(ceiling / mp_mean)
                values = (active_state[0] * sc, active_state[1] * sc, *active_state[2:])
                return type(active_state)(*values) if hasattr(active_state, "_fields") else tuple(values)
        elif isinstance(active_state[1], np.ndarray):
            mp_mean = float(np.maximum(active_state[1], 0.0).mean())
            if mp_mean > ceiling:
                sc = np.float32(ceiling / mp_mean)
                values = (active_state[0] * sc, active_state[1] * sc, *active_state[2:])
                return type(active_state)(*values) if hasattr(active_state, "_fields") else tuple(values)
    return active_state


@dataclass
class SessionConfigV2:
    """Explicit decision costs and provenance budget, independent of characters.

    Legacy false flags are constructor-only compatibility for pipeline.py; an
    enabled intermediate/probe path is rejected, never silently executed.
    Early stopping is disabled until the complete policy is calibrated.
    """
    decode_intermediate: InitVar[bool] = False
    auto_emit_on_saturation: InitVar[bool] = False
    C_abstain: float = 0.05
    C_wrong: float = 1.0
    C_delay: float = 0.0
    C_compute: float = 0.0
    field_wrong_costs: dict[str, float] = field(default_factory=dict)
    field_abstain_costs: dict[str, float] = field(default_factory=dict)
    C_wrong_full: float = 1.0
    C_abstain_full: float = 0.05
    calibration: TerminalCalibrationV2 = field(default_factory=TerminalCalibrationV2)
    risk_calibrated: bool = False
    risk_upper_margin: float = 0.0
    expected_interval_s: float = 0.1
    max_evidence_bytes: int = 8 * 1024 * 1024
    constrained_decoding: bool = True
    registered_plates: Sequence[str | PlateText] | None = None  # API compatibility; no fleet boost

    def __post_init__(self, decode_intermediate, auto_emit_on_saturation):
        if decode_intermediate or auto_emit_on_saturation:
            raise ValueError("Intermediate and saturation readouts are forbidden; use seal()/finish()")
        costs = [self.C_abstain, self.C_wrong, self.C_delay, self.C_compute,
                 self.C_wrong_full, self.C_abstain_full, *self.field_wrong_costs.values(),
                 *self.field_abstain_costs.values()]
        if any(not math.isfinite(x) or x < 0 for x in costs):
            raise ValueError("Decision costs must be finite and nonnegative")
        if self.max_evidence_bytes <= 0 or self.expected_interval_s < 0 or not 0 <= self.risk_upper_margin <= 1:
            raise ValueError("Invalid evidence budget or stopping parameters")


def build_collecting_placeholder_v2(
    mem_P: np.ndarray,
    mem_conflict: np.ndarray,
    color_P: np.ndarray,
    pol_h: float,
    pol_P: float,
    num_frames_seen: int,
    vocab: PlateVocab,
) -> OCRDecisionV2:
    """Construct zero-decode `COLLECTING` status purely from continuous natural-parameter precision."""
    slot_masses = mem_P.mean(axis=-1).astype(np.float64).tolist()
    conflict_score = float(np.max(mem_conflict))
    color_prec = float(np.mean(color_P))
    pol_logit = float(pol_h / (max(0.0, pol_P) + 1e-5))
    default_type = vocab.plate_types[0].value
    return OCRDecisionV2(
        status="COLLECTING",
        plate_text=None,
        formatted_text=None,
        plate_type=default_type,
        field_confidences={"region": 0.0, "classification": 0.0, "hiragana": 0.0, "serial": 0.0},
        slot_confidences=[0.0] * NUM_SLOTS,
        slot_masses=[float(m) for m in slot_masses],
        conflict_score=conflict_score,
        is_saturated=False,
        num_frames_seen=num_frames_seen,
        plate_type_probs={},
        plate_type_candidates=[],
        color_precision_mean=color_prec,
        polarity_logit=pol_logit,
        polarity_precision=float(pol_P),
    )


def _logsumexp(x, axis=None):
    return np.logaddexp.reduce(np.asarray(x, dtype=np.float64), axis=axis)


def _terminal_tables(grammar):
    """Exact finite grammar factors: region, class/kana/type, and serial.

    Grammar contributes only validity, without the old uncalibrated bonuses.
    Factorization avoids enumerating billions of complete strings.
    """
    if hasattr(grammar, "_terminal_tables_v2"):
        return grammar._terminal_tables_v2
    v = grammar.vocab
    pad = v.pad_id
    digits = grammar._all_digit_ids
    letters = grammar._latin_cls_ids
    tail3 = digits + letters
    rows, d_indices = [], []
    for di, first in enumerate(grammar._nonzero_digit_ids + [grammar._zero_id]):
        tails = [(pad, pad)] + [(a, pad) for a in digits]
        tails += [(a, b) for a in (tail3 if di < 9 else [grammar._zero_id, grammar._nine_id]) for b in tail3]
        rows.extend((first, a, b) for a, b in tails)
        d_indices.extend([di] * len(tails))
    classes = np.asarray(rows, dtype=np.int64)
    serials = np.asarray([[v.dot_id]*(4-len(str(i))) + [v.token_to_id[c] for c in str(i)]
                          for i in range(1, 10000)], dtype=np.int64)
    # Only support of MLIT potentials is retained, never fleet/length bonuses.
    valid = grammar._psi_mlit_log_potentials[:, np.asarray(d_indices), :] > -1e10
    result = (v.region_slot_ids_np, classes, grammar._valid_slot7_ids, serials, valid)
    grammar._terminal_tables_v2 = result
    return result


def build_decision_v2_from_arrays(slot_logits, plate_type_logits, polarity_logit,
        slot_readability_pred, status_logits, mem_P, mem_conflict, color_P, pol_P,
        num_frames_seen, vocab, config, grammar, plate_side=0, grammar_checker=None,
        is_final=True, quad_prob=1.0, upper_optical_ok=True, prefix_crop_complete=True,
        max_g0=0.0):
    """Terminal calibrated joint posterior + minimum Bayes risk action.

    A complete-string score and its joint partition determine full correctness.
    Field probabilities are sums of that same joint posterior. Null probability
    remains explicit, even if conditioning on grammar yields one clear string.
    Legacy optical arguments are accepted for caller compatibility, never gates.
    """
    del plate_side, upper_optical_ok, prefix_crop_complete, max_g0
    if not is_final:
        raise RuntimeError("Character decisions require a sealed terminal readout")
    cal = config.calibration
    logits = np.nan_to_num(np.asarray(slot_logits, dtype=np.float64), nan=0, posinf=50, neginf=-50)
    prec = np.maximum(np.nan_to_num(mem_P, nan=0, posinf=0, neginf=0), 0)
    # An unsupported latent channel contributes no optical character evidence.
    logits = np.where(prec.mean(-1)[:, None] > 0, logits, 0) / cal.temperature
    lp = logits - _logsumexp(logits, axis=-1)[:, None]
    regions, classes, kana, serials, valid = _terminal_tables(grammar)
    reg = lp[np.arange(4)[None, :], regions].sum(-1)
    ser = lp[np.arange(8, 12)[None, :], serials].sum(-1)
    cls = lp[np.arange(4, 7)[None, :], classes].sum(-1)
    type_logits = np.nan_to_num(np.asarray(plate_type_logits, dtype=np.float64), nan=0, posinf=50, neginf=-50)
    if float(np.mean(color_P)) <= 0:
        type_logits = np.zeros_like(type_logits)
    type_lp = type_logits/cal.temperature - _logsumexp(type_logits/cal.temperature)
    joint_ck = np.where(valid, type_lp[:, None, None] + cls[None, :, None] + lp[7, kana][None, None, :], -np.inf)
    ck = _logsumexp(joint_ck, axis=0)
    zr, zs, zck = _logsumexp(reg), _logsumexp(ser), _logsumexp(ck.ravel())
    pad = vocab.pad_id
    zr_aug = float(np.logaddexp(zr, lp[0:4, pad].sum()))
    zs_aug = float(np.logaddexp(zs, lp[8:12, pad].sum()))
    zck_aug = float(np.logaddexp(zck, lp[4:8, pad].sum()))
    log_partition = zr_aug + zs_aug + zck_aug
    ir, iz = int(reg.argmax()), int(ser.argmax())
    ic, ik = np.unravel_index(int(ck.argmax()), ck.shape)
    ids = regions[ir].tolist() + classes[ic].tolist() + [int(kana[ik])] + serials[iz].tolist()
    plate = vocab.try_decode(ids)
    st = np.nan_to_num(np.asarray(status_logits, dtype=np.float64), nan=0, posinf=50, neginf=-50)
    # The readout already consumes innovations. An additional conflict slope
    # must therefore be fitted jointly on cal_holdout, not counted twice by
    # default. These are categorical log odds, never a conflict threshold.
    conflict = float(np.max(np.maximum(np.nan_to_num(mem_conflict, nan=0, posinf=0, neginf=0), 0)))
    st[2] = cal.mixed_logit_scale * st[2] + cal.mixed_bias + cal.conflict_weight * conflict
    p_mixed = float(np.exp(st[2] - _logsumexp(st)))
    null_odds = st[0] - st[1]  # READABLE conditional on a single identity
    support = float(np.clip(quad_prob, 0, 1))
    visibility = float(np.clip(np.nan_to_num(slot_readability_pred), 0, 1).mean())
    null_odds = (null_odds + zs_aug + cal.null_bias + cal.readability_weight*visibility + cal.support_weight*support) / cal.null_temperature
    # Exact product rule: Stage-2 plate support times conditional single-plate
    # readability. Calibration can change the conditional odds, never remove
    # the bound P(IsPlate & SingleReadablePlate) <= P(IsPlate).
    p_event = support * (1-p_mixed) * float(np.exp(-np.logaddexp(0, -null_odds))) * cal.identity_probability
    if num_frames_seen == 0 or not np.any(prec > 0):
        p_event = 0.0
    if plate is None or (grammar_checker is not None and not _invoke_grammar_checker(grammar_checker, plate, vocab.plate_types[int(joint_ck[:, ic, ik].argmax())])):
        p_event = 0.0
    # Full-string 0/1 loss is minimized by the joint MAP hypothesis. Marginalize
    # the same posterior to evaluate correctness of each proposed field value.
    class_mass, kana_mass = _logsumexp(ck, axis=1), _logsumexp(ck, axis=0)
    confs = {"region": p_event*float(np.exp(reg[ir]-zr_aug)),
             "classification": p_event*float(np.exp(class_mass[ic]-zck_aug)),
             "hiragana": p_event*float(np.exp(kana_mass[ik]-zck_aug)),
             "serial": p_event*float(np.exp(ser[iz]-zs_aug))}
    # Grammar support cannot replace optical evidence in any field slot.
    for field, sl in (("region", slice(0, 4)), ("classification", slice(4, 7)),
                      ("hiragana", slice(7, 8)), ("serial", slice(8, 12))):
        if not np.all(prec[sl].mean(-1) > 0):
            confs[field] = 0.0
    joint_p = p_event*float(np.exp(reg[ir]+ser[iz]+ck[ic, ik]-log_partition))
    if not np.all(prec[0:12].mean(-1) > 0):
        joint_p = 0.0
    fields = tuple(f for f, p in confs.items() if config.field_wrong_costs.get(f, config.C_wrong)*(1-p) < config.field_abstain_costs.get(f, config.C_abstain))
    full = config.C_wrong_full*(1-joint_p) < config.C_abstain_full
    accepted = plate is not None and ("serial" in fields or full)
    tp = np.exp(joint_ck[:, ic, ik] - _logsumexp(joint_ck[:, ic, ik]))
    type_index = int(tp.argmax())
    type_probs = {p.value: float(tp[i]) for i, p in enumerate(vocab.plate_types)}
    p_unreadable = 1 - p_mixed - p_event
    status = "ACCEPTED" if accepted else "TRACK_MIXED" if p_mixed > p_unreadable else "UNREADABLE"
    return OCRDecisionV2(status=status,
        plate_text=plate if accepted else None,
        formatted_text=plate.full_text if accepted else None,
        plate_type=vocab.plate_types[type_index].value,
        field_confidences=confs, slot_confidences=np.exp(lp[np.arange(12), ids]).tolist(),
        slot_masses=prec.mean(-1).tolist(), conflict_score=float(np.max(mem_conflict)),
        is_saturated=False, num_frames_seen=num_frames_seen, plate_type_probs=type_probs,
        plate_type_candidates=[p.value for i,p in enumerate(vocab.plate_types) if tp[i] > 0],
        color_precision_mean=float(np.mean(color_P)), polarity_logit=float(polarity_logit),
        polarity_precision=float(pol_P), confidence_tier="FULL_4FIELD" if full and accepted else "CORE_VERIFIED" if accepted else "NONE",
        verified_fields=fields if accepted else (), plate_validity_score=p_event,
        joint_correctness_probability=joint_p, readable_probability=p_event)


def normalize_crops_and_geom_numpy(
    crop_rgb: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor],
    geom_meta: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize crop(s) to `(V, 3, H, W)` float32 `[0, 1]` and `geom_meta` to `(V, 10)` float32."""
    if isinstance(crop_rgb, (list, tuple)):
        c_list, g_list = [], []
        for idx, c in enumerate(crop_rgb):
            gm_i = geom_meta[idx] if isinstance(geom_meta, (list, tuple)) else None
            ci, gi = normalize_crops_and_geom_numpy(c, gm_i)
            c_list.append(ci)
            g_list.append(gi)
        c_cat = np.concatenate(c_list, axis=0)
        if geom_meta is not None and not isinstance(geom_meta, (list, tuple)):
            gm_arr = (
                geom_meta.detach().cpu().numpy()
                if isinstance(geom_meta, torch.Tensor)
                else np.asarray(geom_meta, dtype=np.float32)
            ).reshape(-1, 10)
            return c_cat, np.ascontiguousarray(gm_arr, dtype=np.float32)
        return c_cat, np.concatenate(g_list, axis=0)

    arr = crop_rgb.detach().cpu().numpy() if isinstance(crop_rgb, torch.Tensor) else np.asarray(crop_rgb)
    if arr.ndim == 3:
        if arr.shape[-1] == 3 and arr.shape[0] != 3:
            arr_chw = np.transpose(arr, (2, 0, 1))[None, ...]
        else:
            arr_chw = arr[None, ...]
    elif arr.ndim == 4:
        if arr.shape[-1] == 3 and arr.shape[1] != 3:
            arr_chw = np.transpose(arr, (0, 3, 1, 2))
        else:
            arr_chw = arr
    else:
        raise ValueError(f"Unsupported crop shape: {arr.shape}")

    if arr_chw.dtype == np.uint8:
        crops_f = arr_chw.astype(np.float32) / 255.0
    else:
        crops_f = np.nan_to_num(arr_chw.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if crops_f.max() > 1.5:
            crops_f = crops_f / 255.0
    crops_f = np.ascontiguousarray(
        np.clip(np.nan_to_num(crops_f, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0),
        dtype=np.float32,
    )

    v = crops_f.shape[0]
    if geom_meta is not None:
        gm = (
            geom_meta.detach().cpu().numpy()
            if isinstance(geom_meta, torch.Tensor)
            else np.asarray(geom_meta, dtype=np.float32)
        )
        if gm.ndim == 1:
            gm = np.repeat(gm.reshape(1, 10), v, axis=0)
        else:
            gm = gm.reshape(v, 10)
        gm = np.nan_to_num(gm, nan=0.0, posinf=5.0, neginf=-5.0)
        geom_f = np.ascontiguousarray(gm, dtype=np.float32)
    else:
        try:
            from vpocr_det.quad_cropper import extract_quad_geometry_meta as _ext_geom
        except ImportError:
            _ext_geom = None
        geoms = []
        for i in range(v):
            if _ext_geom is not None:
                hwc_u8 = np.round(np.transpose(crops_f[i], (1, 2, 0)) * 255.0).astype(np.uint8)
                geoms.append(_ext_geom(None, hwc_u8))
            else:
                import cv2

                cf = crops_f[i]
                h_c, w_c = cf.shape[1], cf.shape[2]
                y_lum = 0.299 * cf[0] + 0.587 * cf[1] + 0.114 * cf[2]
                mu_y = float(np.clip(np.mean(y_lum), 0.0, 1.0))
                r_clip = float(np.mean(y_lum > 0.96))
                chroma = (np.max(cf, axis=0) - np.min(cf, axis=0)).reshape(-1)
                top_k = max(1, (h_c * w_c * 15) // 100)
                s_chroma = float(np.clip(np.mean(np.partition(chroma, -top_k)[-top_k:]), 0.0, 1.0))
                lap = cv2.Laplacian(y_lum.astype(np.float32), cv2.CV_32F, ksize=3)
                sharp = float(np.clip(0.25 * math.log1p(100.0 * float(np.var(lap))), 0.0, 1.0))
                g0 = float(min(1.0, max(-1.0, 0.5 * math.log2(max(1e-3, w_c * 0.9 / 64.0)))))
                g1 = float(min(1.0, max(-1.0, 0.5 * math.log2(max(1e-3, h_c * 0.9 / 32.0)))))
                geoms.append(
                    np.array([g0, g1, 0.0, 0.0, 0.0, 0.0, sharp, mu_y, s_chroma, r_clip], dtype=np.float32)
                )
        geom_f = np.ascontiguousarray(np.stack(geoms, axis=0), dtype=np.float32)

    return crops_f, geom_f


def fuse_views_v2_numpy(z_views, lam_views, zc_views, lamc_views,
                        pol_views, polp_views, geom_views):
    """Choose one complete alternative warp; never add same-frame factors."""
    i = int(np.argmax(np.maximum(lam_views, 0).mean(axis=(1, 2))))
    return (z_views[i], lam_views[i], zc_views[i], lamc_views[i],
            pol_views[i:i+1], polp_views[i:i+1], geom_views[i])


def compute_bandpass_stroke_sharpness(crops_chw: np.ndarray) -> float:
    """Measure noise-immune 2x-pooled interior stroke gradient RMS on `(V, 3, H, W)` crops."""
    y = 0.299 * crops_chw[:, 0] + 0.587 * crops_chw[:, 1] + 0.114 * crops_chw[:, 2]
    h2, w2 = (y.shape[1] // 2) * 2, (y.shape[2] // 2) * 2
    if h2 >= 4 and w2 >= 4:
        y2 = 0.25 * (
            y[:, 0:h2:2, 0:w2:2]
            + y[:, 0:h2:2, 1:w2:2]
            + y[:, 1:h2:2, 0:w2:2]
            + y[:, 1:h2:2, 1:w2:2]
        )
    else:
        y2 = y
    yi = y2[:, 4:-4, 4:-4] if (y2.shape[1] > 10 and y2.shape[2] > 10) else y2
    gx = yi[:, :, 1:] - yi[:, :, :-1]
    gy = yi[:, 1:, :] - yi[:, :-1, :]
    energy = float(np.mean(gx * gx) + np.mean(gy * gy))
    return float(np.clip(math.sqrt(max(0.0, energy)), 1e-3, 1.0))


class _TrackQualityAccumulator:
    """Unweighted stopping quality and temporally discounted episode plate support."""
    def __init__(self): self.reset()
    def reset(self):
        self.n = 0
        self.support_sum = 0.0
        self.plate_log_odds = 0.0
        self._first_raw_support = 0.0
        self._last_support_ts: float | None = None
        self.stage2_sum = np.zeros(3, dtype=np.float64)
        self.geom_sum = np.zeros(6, dtype=np.float64)
        self.geom_sq = np.zeros(6, dtype=np.float64)
    def update(self, support, geometry, stage2, *, raw_support: float | None = None,
               timestamp_s: float | None = None):
        self.n += 1
        self.support_sum += float(np.clip(support, 0, 1))
        q_raw = float(np.clip(support if raw_support is None else raw_support, 0.0, 1.0))
        q_c = min(1.0 - 1e-6, max(1e-6, q_raw))
        llr_step = math.log(q_c) - math.log1p(-q_c)
        if self.n == 1:
            self._first_raw_support = q_raw
            self.plate_log_odds = llr_step
        else:
            dt = (0.5 if (timestamp_s is None or self._last_support_ts is None)
                  else max(0.0, float(timestamp_s) - self._last_support_ts))
            vis_mean = float(np.clip(stage2[1], 0.0, 1.0))
            w_innov = min(1.0, max(0.05, dt / 0.5)) * (0.05 + 0.95 * vis_mean)
            self.plate_log_odds = float(np.clip(self.plate_log_odds + w_innov * llr_step, -30.0, 30.0))
        if timestamp_s is not None and math.isfinite(float(timestamp_s)):
            self._last_support_ts = float(timestamp_s)
        self.stage2_sum += stage2
        self.geom_sum += geometry[:6]
        self.geom_sq += geometry[:6]**2
    @property
    def effective_quad_prob(self): return self.support_sum/max(1, self.n)
    @property
    def episode_plate_prob(self) -> float:
        if self.n == 0:
            return 0.0
        if self.n == 1:
            return float(self._first_raw_support)
        return float(math.exp(-np.logaddexp(0.0, -self.plate_log_odds)))
    @property
    def geometry_variance(self):
        return float(np.maximum(0, self.geom_sq/max(1,self.n)-(self.geom_sum/max(1,self.n))**2).mean())
    @property
    def stage2_means(self):
        """Nonconformity, mean corner visibility, and plate/null LLR."""
        return self.stage2_sum / max(1, self.n)


class _ScaleSlopeEstimator:
    """Exact O(1) online linear regression estimator for `scale_slope = d(g_0)/dt` and optical stroke MTF."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n: int = 0
        self.sum_t: float = 0.0
        self.sum_g: float = 0.0
        self.sum_tt: float = 0.0
        self.sum_tg: float = 0.0
        self.sum_sharp: float = 0.0
        self.t_origin: float | None = None
        self.t_start: float | None = None
        self.t_end: float = 0.0

    def update(self, t_s: float, g0: float, stroke_sharpness: float = 0.20) -> None:
        t = float(t_s) if math.isfinite(float(t_s)) else 0.0
        g = float(g0) if math.isfinite(float(g0)) else 0.0
        if self.t_origin is None:
            self.t_origin = t
        if self.t_start is None or t < self.t_start:
            self.t_start = t
        if t > self.t_end or self.n == 0:
            self.t_end = t
        dt = t - self.t_origin
        self.n += 1
        self.sum_t += dt
        self.sum_g += g
        self.sum_tt += dt * dt
        self.sum_tg += dt * g
        self.sum_sharp += float(np.clip(stroke_sharpness, 1e-3, 1.0))

    @property
    def mean_sharpness(self) -> float:
        return float(self.sum_sharp / float(self.n)) if self.n > 0 else 0.20

    @property
    def slope(self) -> float:
        if self.n < 2:
            return 0.0
        denom = float(self.n) * self.sum_tt - self.sum_t * self.sum_t
        if denom <= 1e-5:
            return 0.0
        num = float(self.n) * self.sum_tg - self.sum_t * self.sum_g
        return float(np.clip(num / denom, -5.0, 5.0))


class _SealedSessionV2:
    """Shared lifecycle and factor provenance for Torch and ONNX runtimes."""

    def reset(self):
        # A public reset cannot reopen an episode, even by erasing its counters.
        if getattr(self, 'state', SessionStateV2.COLLECTING) != SessionStateV2.COLLECTING:
            raise RuntimeError('A sealed OCR episode cannot be reset or reopened')
        self._state = SessionStateV2.COLLECTING
        self._active_state = self._empty_state()
        self._factors = {}
        self._evidence_bytes = 0
        self._auto_frame_counter = -1
        self._last_plate_side = 0
        self._cached_decision = None
        self._readout_consumed = False
        self.num_frames_seen = self.num_readout_calls = 0
        self._slope_est = _ScaleSlopeEstimator()
        self._qual_acc = _TrackQualityAccumulator()

    @property
    def state(self): return self._state

    @property
    def frame_keys(self):
        return frozenset(self._factors)

    def _require_collecting(self):
        if self.state != SessionStateV2.COLLECTING:
            raise RuntimeError('Observations are forbidden after sealing an OCR episode')

    def _placeholder(self):
        _, p, c, _, cp, ph, pp, _, _ = self._state_arrays()
        return build_collecting_placeholder_v2(p[0], c[0], cp[0], float(ph[0]),
            float(pp[0]), self.num_frames_seen, self.vocab)

    def _quality(self):
        _, p, c, _, cp, _, pp, _, _ = self._state_arrays()
        masses = p.mean(-1)
        return np.asarray([[np.log1p(masses.mean()), np.log1p(masses.min()),
            np.log1p(masses.max()), (masses > 0).mean(), np.log1p(cp.mean()),
            np.log1p(pp[0]), c.mean(), c.max(), self._qual_acc.effective_quad_prob,
            self._qual_acc.geometry_variance]], dtype=np.float32)

    def stopping_action(self, deadline=False):
        """One-step Bellman approximation with learned continuation risk.

        Calibration of the entire quantized policy is an explicit prerequisite.
        Until then early stopping is disabled, and deadlines use terminal MBR.
        """
        if not np.any(self._state_arrays()[1] > 0):
            return 'abstain' if deadline else 'collect'
        read, next_risk = self._predict_risk(self._quality())
        if not self.config.risk_calibrated:
            return 'seal_and_read_once' if deadline else 'collect'
        cfg = self.config
        read_cost = cfg.C_wrong * min(1.0, float(read) + cfg.risk_upper_margin)
        if deadline:
            return 'abstain' if cfg.C_abstain <= read_cost else 'seal_and_read_once'
        future = min(cfg.C_abstain, cfg.C_wrong*min(1.0, float(next_risk)+cfg.risk_upper_margin))
        collect_cost = cfg.C_delay*cfg.expected_interval_s + cfg.C_compute + future
        # Prefer collection on ties; sealing is irreversible.
        if collect_cost <= min(read_cost, cfg.C_abstain):
            return 'collect'
        return 'abstain' if cfg.C_abstain <= read_cost else 'seal_and_read_once'

    def _append_factor(self, key, timestamp, observation, side, support, sharpness, stage2,
                       *, raw_support: float | None = None):
        size = sum(a.nbytes for a in observation) + 2048 + len(repr(key).encode("utf-8"))
        if self._evidence_bytes + size > self.config.max_evidence_bytes:
            raise ValueError('Evidence provenance budget exceeded; seal this episode')
        self._apply_observation(observation, timestamp)
        self._factors[key] = (timestamp, tuple(a.copy() for a in observation), side, support,
                              sharpness, stage2.copy(), raw_support)
        self._evidence_bytes += size
        self._last_plate_side = side
        self.num_frames_seen += 1
        self._qual_acc.update(support, observation[-1], stage2,
                              raw_support=raw_support, timestamp_s=timestamp)
        self._slope_est.update(timestamp, float(observation[-1][0]), sharpness)

    def _prepare_observation(self, crop_rgb, timestamp_s, geom_meta, camera_id,
                             source_frame_id, plate_side, quad_prob, frame_key=None,
                             support_nonconformity=None, corner_visibility_4=None,
                             plate_null_llr=None):
        self._require_collecting()
        if frame_key is None:
            if source_frame_id is None:
                frame_key = (int(camera_id), self._auto_frame_counter)
                self._auto_frame_counter -= 1
            else:
                frame_key = (int(camera_id), int(source_frame_id))
        hash(frame_key)
        if frame_key in self._factors:
            return None
        if not math.isfinite(float(timestamp_s)):
            raise ValueError('Observation timestamp must be finite')
        if self._factors and timestamp_s < max(f[0] for f in self._factors.values()):
            raise ValueError('Out-of-order observations require chronological disjoint merge')
        crops, geom = normalize_crops_and_geom_numpy(crop_rgb, geom_meta)
        sides = np.full(len(crops), int(plate_side), dtype=np.int64) if np.isscalar(plate_side) else np.asarray(plate_side, dtype=np.int64)
        if sides.shape != (len(crops),):
            raise ValueError('plate_side must match the number of views')
        support = 1.0 if quad_prob is None else float(quad_prob)
        if not math.isfinite(support) or not 0 <= support <= 1:
            raise ValueError('Observation support must be a probability')
        nc = 0.0 if support_nonconformity is None else float(support_nonconformity)
        llr = 0.0 if plate_null_llr is None else float(plate_null_llr)
        vis = np.ones(4) if corner_visibility_4 is None else np.asarray(corner_visibility_4, dtype=float)
        if (not math.isfinite(nc) or nc < 0 or not math.isfinite(llr)
                or vis.shape != (4,) or not np.all(np.isfinite(vis))
                or np.any((vis < 0) | (vis > 1))):
            raise ValueError('Invalid Stage-2 uncertainty evidence')
        # Smooth, strictly positive damping: weak containment, occlusion, and
        # null evidence reduce precision without vetoing a character or string.
        damping = math.exp(-min(30.0, nc + max(0.0, -llr))) * (0.05 + 0.95 * float(vis.mean()))
        stage2 = np.asarray([nc, vis.mean(), llr], dtype=np.float64)
        return frame_key, float(timestamp_s), crops, geom, sides, support, damping, stage2

    def _consume_encoding(self, prepared, encoded):
        key, ts, crops, geom, sides, support, damping, stage2 = prepared
        obs = list(fuse_views_v2_numpy(*encoded, geom))
        for i in (1, 3, 5):
            obs[i] = obs[i] * np.float32(damping)
        self._append_factor(key, ts, tuple(obs), int(sides[0]), support * damping,
                            compute_bandpass_stroke_sharpness(crops), stage2, raw_support=support)
        action = self.stopping_action()
        if action == 'abstain':
            return self.abstain()
        if action == 'seal_and_read_once':
            self.seal()
            return self.finish()
        return self._placeholder()

    def push(self, crop_rgb, timestamp_s, geom_meta=None, camera_id=0,
             source_frame_id=None, view_variant='raw', plate_side=0,
             quad_prob=None, quad_edge_touch=False, frame_key=None,
             support_nonconformity: float | None = None,
             corner_visibility_4: Sequence[float] | np.ndarray | None = None,
             plate_null_llr: float | None = None):
        del view_variant, quad_edge_touch
        prepared = self._prepare_observation(crop_rgb, timestamp_s, geom_meta,
            camera_id, source_frame_id, plate_side, quad_prob, frame_key,
            support_nonconformity, corner_visibility_4, plate_null_llr)
        if prepared is None:
            return self._placeholder()  # Duplicate: no encoder, risk probe, or state change.
        crops, geom, sides = prepared[2:5]
        return self._consume_encoding(prepared, self._encode(crops, sides, geom))

    @classmethod
    def push_batch_parallel(cls, sessions, crops_rgb, timestamps_s, geom_metas=None,
                            camera_ids=0, source_frame_ids=None, plate_sides=0,
                            quad_probs=None, quad_edge_touches=False, frame_keys=None,
                            support_nonconformities: Sequence[float | None] | None = None,
                            corner_visibilities_4: Sequence[Sequence[float] | np.ndarray | None] | np.ndarray | None = None,
                            plate_null_llrs: Sequence[float | None] | None = None):
        """Same irreversible contract for every participant; validate before mutation."""
        n = len(sessions)
        def spread(v, default=None):
            if v is None: return [default]*n
            if np.isscalar(v): return [v]*n
            if len(v) != n: raise ValueError('Mismatched batch metadata lengths')
            return list(v)
        crops = spread(crops_rgb)
        ts, gm = spread(timestamps_s), spread(geom_metas)
        cam, sf = spread(camera_ids), spread(source_frame_ids)
        sides, qp, edge = spread(plate_sides), spread(quad_probs), spread(quad_edge_touches)
        keys, nc = spread(frame_keys), spread(support_nonconformities)
        vis, llr = spread(corner_visibilities_4), spread(plate_null_llrs)
        if len(set(map(id, sessions))) != n:
            raise ValueError('A batch must contain distinct sessions')
        for session in sessions: session._require_collecting()
        if n and any(not sessions[0]._same_filter(s) for s in sessions):
            raise ValueError('Parallel sessions must share an encoder/filter')
        prepared = [s._prepare_observation(crops[i], ts[i], gm[i], cam[i], sf[i], sides[i], qp[i],
                                          keys[i], nc[i], vis[i], llr[i])
                    for i,s in enumerate(sessions)]
        active = [i for i,p in enumerate(prepared) if p is not None]
        encodings = {}
        if active:
            # One vocabulary-free encoder call across all active tracks/views.
            encoded = sessions[active[0]]._encode(
                np.concatenate([prepared[i][2] for i in active]),
                np.concatenate([prepared[i][4] for i in active]),
                np.concatenate([prepared[i][3] for i in active]))
            offset = 0
            for i in active:
                count = len(prepared[i][2])
                encodings[i] = tuple(a[offset:offset+count] for a in encoded)
                offset += count
        return [s._placeholder() if prepared[i] is None else s._consume_encoding(prepared[i], encodings[i])
                for i,s in enumerate(sessions)]

    def merge_disjoint_evidence(self, other):
        """Replay only disjoint Gaussian observation factors with exactly one prior.

        Summing posterior precisions is invalid with correlated nuisances. Replay
        is deterministic and bounded by max_evidence_bytes; no images/strings are
        retained. Unsafe overlap or incompatible filters is rejected atomically.
        """
        self._require_collecting()
        other._require_collecting()
        if self is other or self.frame_keys & other.frame_keys:
            raise ValueError('Relinking requires disjoint frame keys')
        if not self._same_filter(other):
            raise ValueError('Relinking requires identical encoder/filter parameters')
        if self._evidence_bytes + other._evidence_bytes > self.config.max_evidence_bytes:
            raise ValueError('Merged evidence exceeds provenance budget')
        factors = list(self._factors.items()) + list(other._factors.items())
        # Stable tie-break ensures merge ordering does not depend on track order.
        factors.sort(key=lambda kv: (kv[1][0], repr(kv[0])))
        self.reset()
        for key, (ts, obs, side, support, sharp, stage2, raw_support) in factors:
            self._append_factor(key, ts, obs, side, support, sharp, stage2, raw_support=raw_support)
        return self

    def seal(self):
        if self.state == SessionStateV2.COLLECTING:
            self._state = SessionStateV2.SEALED
        return self

    def abstain(self):
        if self._cached_decision is not None: return self._cached_decision
        self.seal()
        result = self._placeholder()
        result.status = 'UNREADABLE'
        self._cached_decision = result
        self._state = SessionStateV2.FINALIZED
        return result

    def finish(self):
        if self._cached_decision is not None: return self._cached_decision
        if self.state in (SessionStateV2.COLLECTING, SessionStateV2.SEALED):
            action = self.stopping_action(deadline=True)
            self.seal()
            if action == 'abstain': return self.abstain()
        if self.num_readout_calls:
            raise RuntimeError('Terminal readout already attempted; it cannot be repeated')
        self.seal()
        self.num_readout_calls += 1
        # A failed decoder also consumes the one terminal capability.
        try:
            self._active_state = _apply_sealed_precision_ceiling(
                self._active_state, float(self.config.calibration.readout_precision_ceiling))
            ro = self._readout()
            _, p, c, _, cp, _, pp, _, _ = self._state_arrays()
            self._cached_decision = build_decision_v2_from_arrays(*ro[:2],
                polarity_logit=float(ro[2]), slot_readability_pred=ro[3], status_logits=ro[4],
                mem_P=p[0], mem_conflict=c[0], color_P=cp[0], pol_P=float(pp[0]),
                num_frames_seen=self.num_frames_seen, vocab=self.vocab, config=self.config,
                grammar=self.grammar, grammar_checker=self.grammar_checker,
                quad_prob=self._qual_acc.episode_plate_prob)
        finally:
            self._state = SessionStateV2.FINALIZED
        return self._cached_decision

    def to_camera_observation(self, camera_id=0, local_track_id='trk_0', mount_role='front'):
        from vpocr_ocr.gate_fusion import CameraTrackObservationV2
        mh, mp, mc, ch, cp, ph, pp, _, _ = self._state_arrays()
        return CameraTrackObservationV2(camera_id=int(camera_id), local_track_id=str(local_track_id),
            mount_role=str(mount_role), mem_h=mh[0].copy(), mem_P=mp[0].copy(),
            mem_conflict=mc[0].copy(), color_h=ch[0].copy(), color_P=cp[0].copy(),
            pol_h=float(ph[0]), pol_P=float(pp[0]), num_frames_seen=self.num_frames_seen,
            t_start_s=float(self._slope_est.t_start or 0), t_end_s=float(self._slope_est.t_end),
            scale_slope=self._slope_est.slope, plate_side=self._last_plate_side,
            mean_sharpness=self._slope_est.mean_sharpness, quad_prob=self._qual_acc.episode_plate_prob)


class MFPlateOCRSessionV2(_SealedSessionV2):
    """Torch collector. Vocabulary projection is reachable only through finish()."""
    def __init__(self, model, vocab=None, config=None, grammar_checker=None, device=None):
        self.model = model.eval()
        self.vocab, self.config = vocab or PlateVocab(), config or SessionConfigV2()
        self.grammar, self.grammar_checker = JPPlateGrammar(self.vocab), grammar_checker
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.reset()

    @property
    def memory_state(self): return self._active_state
    def _empty_state(self): return self.model.init_memory(1, device=self.device)
    def _state_arrays(self): return tuple(t.detach().float().cpu().numpy() for t in self._active_state)
    def _same_filter(self, other): return isinstance(other, MFPlateOCRSessionV2) and self.model is other.model and self.device == other.device
    @torch.no_grad()
    def _encode(self, crops, sides, geom):
        enc = self.model.encode_frames(torch.from_numpy(crops).to(self.device),
            torch.from_numpy(sides).to(self.device), torch.from_numpy(geom).to(self.device))
        return tuple(t.float().cpu().numpy() for t in enc[:6])
    @torch.no_grad()
    def _apply_observation(self, obs, ts):
        z, lam, zc, lamc, pol, polp, geom = obs
        tensors = [torch.from_numpy(x).to(self.device) for x in (z[None],lam[None],zc[None],lamc[None],pol,polp,geom[None])]
        self._active_state = self.model.step_memory(*self._active_state, *tensors,
            torch.tensor([ts], dtype=torch.float32, device=self.device))
    @torch.no_grad()
    def _predict_risk(self, quality):
        return self.model.stopping_risk_head(torch.from_numpy(quality).to(self.device))[0].cpu().tolist()
    @torch.no_grad()
    def _readout(self):
        if self.state != SessionStateV2.SEALED or self.num_readout_calls != 1 or self._readout_consumed:
            raise RuntimeError('Terminal readout capability is sealed')
        self._readout_consumed = True
        ro = self.model.decode_memory(*self._active_state[:7])
        return tuple(t[0].float().cpu().numpy() for t in ro)
