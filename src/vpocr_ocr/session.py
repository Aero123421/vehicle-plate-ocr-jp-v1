"""Stateful streaming multi-frame OCR session and decision logic."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import inspect
from typing import Literal
import numpy as np
import torch

from vpocr_ocr.model import MFJPPlateOCR, MemoryState
from vpocr_ocr.vocab import JPPlateGrammar, NUM_SLOTS, PlateVocab
from vpocr_synth.plate.models import PlateText

DecisionStatus = Literal[
    "COLLECTING",
    "TENTATIVE",
    "ACCEPTED",
    "UNREADABLE",
    "CONFLICT",
    "TRACK_MIXED",
]


@dataclass
class OCRDecision:
    """Structured output decision of a streaming or finalized MF-JPPlateOCR track."""

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


@dataclass
class SessionConfig:
    """Thresholds controlling streaming state transitions and abstention."""

    constrained_decoding: bool = True
    collecting_min_mass: float = 0.25
    streaming_accept_mass: float = 1.10
    final_accept_mass: float = 0.35
    saturation_mass: float = 1.65
    accept_confidence: float = 0.62
    tentative_confidence: float = 0.35
    accept_readability: float = 0.48
    unreadable_readability_floor: float = 0.38
    conflict_warn_threshold: float = 0.45
    track_mixed_conflict_threshold: float = 0.75
    track_mixed_prob_threshold: float = 0.50
    unreadable_prob_threshold: float = 0.52


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / (np.sum(exps, axis=axis, keepdims=True) + 1e-12)


def _invoke_grammar_checker(
    checker: Callable[..., bool],
    plate_text: PlateText | None,
    plate_type: str,
) -> bool:
    if plate_text is None:
        return False
    try:
        sig = inspect.signature(checker)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
        if has_varargs or len(params) >= 2:
            return bool(checker(plate_text, plate_type))
        return bool(checker(plate_text))
    except (TypeError, ValueError):
        return bool(checker(plate_text))


def build_decision_from_arrays(
    slot_logits: np.ndarray,
    plate_type_logits: np.ndarray,
    slot_readability_pred: np.ndarray,
    status_logits: np.ndarray,
    mem_mass: np.ndarray,
    mem_conflict: np.ndarray,
    num_frames_seen: int,
    vocab: PlateVocab,
    config: SessionConfig,
    grammar: JPPlateGrammar,
    grammar_checker: Callable[..., bool] | None = None,
    is_final: bool = False,
) -> OCRDecision:
    """Shared numpy decision logic used identically by PyTorch and ONNX Runtime sessions."""
    raw_slot_probs = _softmax_np(slot_logits, axis=-1)  # (12, V)
    slot_confs = raw_slot_probs.max(axis=-1).astype(np.float64).tolist()
    slot_masses = mem_mass.astype(np.float64).tolist()
    conflict_score = float(np.max(mem_conflict))

    type_probs = _softmax_np(plate_type_logits, axis=-1)
    pred_type_enum = vocab.decode_plate_type(int(np.argmax(type_probs)))
    pred_type_str = pred_type_enum.value

    if num_frames_seen == 0 or float(np.max(mem_mass)) < 1e-4:
        empty_fields = {
            "region": 0.0,
            "classification": 0.0,
            "hiragana": 0.0,
            "serial": 0.0,
        }
        return OCRDecision(
            status="UNREADABLE" if is_final else "COLLECTING",
            plate_text=None,
            formatted_text=None,
            plate_type=pred_type_str,
            field_confidences=empty_fields,
            slot_confidences=[float(c) for c in slot_confs],
            slot_masses=[float(m) for m in slot_masses],
            conflict_score=conflict_score,
            is_saturated=False,
            num_frames_seen=num_frames_seen,
        )

    status_probs = _softmax_np(status_logits, axis=-1)  # [READABLE, UNREADABLE, TRACK_MIXED]
    p_readable, p_unreadable, p_track_mixed = (
        float(status_probs[0]),
        float(status_probs[1]),
        float(status_probs[2]),
    )

    if config.constrained_decoding:
        slot_ids = grammar.constrain_slot_ids(slot_logits, plate_type=pred_type_enum)
    else:
        slot_ids = slot_logits.argmax(axis=-1).tolist()

    candidate_plate = vocab.try_decode(slot_ids)

    active_checker = grammar_checker if grammar_checker is not None else grammar
    grammar_ok = _invoke_grammar_checker(active_checker, candidate_plate, pred_type_str)

    # Compute per-field confidences from raw unmasked slot probabilities
    reg_active = [slot_confs[i] for i in range(0, 4) if slot_ids[i] != vocab.pad_id] or [
        slot_confs[0]
    ]
    cls_active = [slot_confs[i] for i in range(4, 7) if slot_ids[i] != vocab.pad_id] or [
        slot_confs[4]
    ]
    field_confidences = {
        "region": float(np.mean(reg_active)),
        "classification": float(np.mean(cls_active)),
        "hiragana": float(slot_confs[7]),
        "serial": float(np.mean(slot_confs[8:12])),
    }

    active_indices = [i for i in range(NUM_SLOTS) if slot_ids[i] != vocab.pad_id]
    if not active_indices:
        active_indices = [0, 4, 7, 8, 9, 10, 11]

    min_active_mass = min(float(mem_mass[i]) for i in active_indices)
    min_active_read = min(float(slot_readability_pred[i]) for i in active_indices)
    min_active_conf = min(float(slot_confs[i]) for i in active_indices)

    # 1. Highest priority safety check: foreign plate / cross-frame feature conflict
    if (
        p_track_mixed >= config.track_mixed_prob_threshold
        or conflict_score >= config.track_mixed_conflict_threshold
    ):
        return OCRDecision(
            status="TRACK_MIXED",
            plate_text=None,
            formatted_text=None,
            plate_type=pred_type_str,
            field_confidences=field_confidences,
            slot_confidences=[float(c) for c in slot_confs],
            slot_masses=[float(m) for m in slot_masses],
            conflict_score=conflict_score,
            is_saturated=False,
            num_frames_seen=num_frames_seen,
        )

    if conflict_score >= config.conflict_warn_threshold:
        return OCRDecision(
            status="CONFLICT",
            plate_text=None,
            formatted_text=None,
            plate_type=pred_type_str,
            field_confidences=field_confidences,
            slot_confidences=[float(c) for c in slot_confs],
            slot_masses=[float(m) for m in slot_masses],
            conflict_score=conflict_score,
            is_saturated=False,
            num_frames_seen=num_frames_seen,
        )

    is_saturated = bool(
        grammar_ok
        and candidate_plate is not None
        and min_active_mass >= config.saturation_mass
        and min_active_conf >= config.accept_confidence
        and min_active_read >= config.accept_readability
        and p_unreadable < config.unreadable_prob_threshold
    )

    # 2. Streaming vs Finalized state resolution
    if not is_final:
        if (
            min_active_mass < config.collecting_min_mass
            or min_active_read < config.unreadable_readability_floor
            or candidate_plate is None
        ):
            return OCRDecision(
                status="COLLECTING",
                plate_text=None,
                formatted_text=None,
                plate_type=pred_type_str,
                field_confidences=field_confidences,
                slot_confidences=[float(c) for c in slot_confs],
                slot_masses=[float(m) for m in slot_masses],
                conflict_score=conflict_score,
                is_saturated=False,
                num_frames_seen=num_frames_seen,
            )

        if (
            grammar_ok
            and min_active_mass >= config.streaming_accept_mass
            and min_active_conf >= config.accept_confidence
            and min_active_read >= config.accept_readability
            and p_readable >= p_unreadable
        ):
            status: DecisionStatus = "ACCEPTED"
        else:
            status = "TENTATIVE"

        return OCRDecision(
            status=status,
            plate_text=candidate_plate,
            formatted_text=candidate_plate.full_text,
            plate_type=pred_type_str,
            field_confidences=field_confidences,
            slot_confidences=[float(c) for c in slot_confs],
            slot_masses=[float(m) for m in slot_masses],
            conflict_score=conflict_score,
            is_saturated=is_saturated,
            num_frames_seen=num_frames_seen,
        )

    # Finalized track (`finish()`): resolve into ACCEPTED, TENTATIVE, or UNREADABLE
    if (
        candidate_plate is None
        or not grammar_ok
        or min_active_mass < config.collecting_min_mass
        or min_active_read < config.unreadable_readability_floor
        or p_unreadable >= config.unreadable_prob_threshold
    ):
        return OCRDecision(
            status="UNREADABLE",
            plate_text=None,
            formatted_text=None,
            plate_type=pred_type_str,
            field_confidences=field_confidences,
            slot_confidences=[float(c) for c in slot_confs],
            slot_masses=[float(m) for m in slot_masses],
            conflict_score=conflict_score,
            is_saturated=False,
            num_frames_seen=num_frames_seen,
        )

    if (
        min_active_mass >= config.final_accept_mass
        and min_active_conf >= config.accept_confidence
        and min_active_read >= config.accept_readability
        and p_readable >= p_unreadable
    ):
        final_status: DecisionStatus = "ACCEPTED"
    elif min_active_conf >= config.tentative_confidence:
        final_status = "TENTATIVE"
    else:
        return OCRDecision(
            status="UNREADABLE",
            plate_text=None,
            formatted_text=None,
            plate_type=pred_type_str,
            field_confidences=field_confidences,
            slot_confidences=[float(c) for c in slot_confs],
            slot_masses=[float(m) for m in slot_masses],
            conflict_score=conflict_score,
            is_saturated=False,
            num_frames_seen=num_frames_seen,
        )

    return OCRDecision(
        status=final_status,
        plate_text=candidate_plate,
        formatted_text=candidate_plate.full_text,
        plate_type=pred_type_str,
        field_confidences=field_confidences,
        slot_confidences=[float(c) for c in slot_confs],
        slot_masses=[float(m) for m in slot_masses],
        conflict_score=conflict_score,
        is_saturated=is_saturated,
        num_frames_seen=num_frames_seen,
    )


def _normalize_crop_views(
    crop_rgb: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Convert single crop or sequence of same-frame views to (V, 3, 64, 128) float32 [0, 1]."""
    if isinstance(crop_rgb, (list, tuple)):
        views = [_normalize_crop_views(c, device) for c in crop_rgb]
        return torch.cat(views, dim=0)

    if isinstance(crop_rgb, np.ndarray):
        t = torch.from_numpy(crop_rgb)
    else:
        t = crop_rgb

    if t.ndim == 3:
        if t.shape[-1] == 3 and t.shape[0] != 3:
            t = t.permute(2, 0, 1)  # (3, 64, 128)
        t = t.unsqueeze(0)
    elif t.ndim == 4:
        if t.shape[-1] == 3 and t.shape[1] != 3:
            t = t.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported crop shape: {tuple(t.shape)}")

    if t.dtype == torch.uint8:
        out = t.to(device=device, dtype=torch.float32) / 255.0
    else:
        out = t.to(device=device, dtype=torch.float32)
        if out.max().item() > 1.5:
            out = out / 255.0
    return torch.clamp(out, 0.0, 1.0).contiguous()


class MFPlateOCRSession:
    """Stateful streaming session for a single tracked vehicle using PyTorch MFJPPlateOCR."""

    def __init__(
        self,
        model: MFJPPlateOCR,
        vocab: PlateVocab | None = None,
        config: SessionConfig | None = None,
        grammar_checker: Callable[..., bool] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.vocab = vocab or PlateVocab()
        self.config = config or SessionConfig()
        self.grammar = JPPlateGrammar(self.vocab)
        self.grammar_checker = grammar_checker
        self.device = (
            torch.device(device)
            if device is not None
            else next(self.model.parameters()).device
        )
        self.reset()

    def reset(self) -> None:
        """Reset track EvidenceMemory to empty initial state."""
        self._committed_state: MemoryState = self.model.init_memory(1, device=self.device)
        self._active_state: MemoryState = self._committed_state
        self._current_frame_key: tuple[int, int] | None = None
        self._current_frame_ts: float = 0.0
        self._current_z_views: list[torch.Tensor] = []
        self._current_q_views: list[torch.Tensor] = []
        self._current_v_views: list[torch.Tensor] = []
        self._auto_frame_counter: int = -1
        self.num_frames_seen: int = 0

    @property
    def memory_state(self) -> MemoryState:
        return self._active_state

    @torch.no_grad()
    def push(
        self,
        crop_rgb: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor],
        timestamp_s: float,
        camera_id: int = 0,
        source_frame_id: int | None = None,
        view_variant: str | Sequence[str] = "raw",
        plate_side: int | Sequence[int] = 0,
    ) -> OCRDecision:
        """Push a crop (or list of same-frame crop variants) and return updated streaming state."""
        del view_variant  # Metadata tag; fusion uses learned per-view quality weights
        crops_t = _normalize_crop_views(crop_rgb, self.device)  # (V, 3, 64, 128)
        num_views = crops_t.shape[0]

        if isinstance(plate_side, Sequence) and not isinstance(plate_side, (str, bytes)):
            sides_t = torch.tensor(list(plate_side), dtype=torch.long, device=self.device)
        else:
            sides_t = torch.full((num_views,), int(plate_side), dtype=torch.long, device=self.device)

        enc = self.model.encode_frames(crops_t, sides_t)

        if source_frame_id is None:
            frame_key = (int(camera_id), self._auto_frame_counter)
            self._auto_frame_counter -= 1
        else:
            frame_key = (int(camera_id), int(source_frame_id))

        if self._current_frame_key == frame_key:
            # Same physical (camera_id, source_frame_id): fuse with earlier view(s) on top of
            # _committed_state so multi-view variants NEVER double-vote!
            self._current_z_views.append(enc.z)
            self._current_q_views.append(enc.slot_quality)
            self._current_v_views.append(enc.frame_valid)
        else:
            self._committed_state = self._active_state
            self._current_frame_key = frame_key
            self._current_frame_ts = float(timestamp_s)
            self._current_z_views = [enc.z]
            self._current_q_views = [enc.slot_quality]
            self._current_v_views = [enc.frame_valid]
            self.num_frames_seen += 1

        all_z = torch.cat(self._current_z_views, dim=0)  # (V_tot, 12, D)
        all_q = torch.cat(self._current_q_views, dim=0)  # (V_tot, 12)
        all_v = torch.cat(self._current_v_views, dim=0)  # (V_tot, 1)

        if all_z.shape[0] == 1:
            z_step = all_z
            q_step = all_q
            v_step = all_v
        else:
            z_s, q_s, v_s = self.model.fuse_views(all_z, all_q, all_v)
            z_step = z_s.unsqueeze(0)
            q_step = q_s.unsqueeze(0)
            v_step = v_s.unsqueeze(0)

        ts_tensor = torch.tensor([self._current_frame_ts], dtype=torch.float32, device=self.device)
        self._active_state = self.model.step_memory(
            self._committed_state.mem_feat,
            self._committed_state.mem_mass,
            self._committed_state.mem_conflict,
            self._committed_state.last_timestamp_s,
            z_step,
            q_step,
            v_step,
            ts_tensor,
        )
        return self.state()

    @torch.no_grad()
    def _build_decision(self, is_final: bool) -> OCRDecision:
        st = self._active_state
        ro = self.model.decode_memory(st.mem_feat, st.mem_mass, st.mem_conflict)
        return build_decision_from_arrays(
            slot_logits=ro.slot_logits[0].float().cpu().numpy(),
            plate_type_logits=ro.plate_type_logits[0].float().cpu().numpy(),
            slot_readability_pred=ro.slot_readability_pred[0].float().cpu().numpy(),
            status_logits=ro.status_logits[0].float().cpu().numpy(),
            mem_mass=st.mem_mass[0].float().cpu().numpy(),
            mem_conflict=st.mem_conflict[0].float().cpu().numpy(),
            num_frames_seen=self.num_frames_seen,
            vocab=self.vocab,
            config=self.config,
            grammar=self.grammar,
            grammar_checker=self.grammar_checker,
            is_final=is_final,
        )

    def state(self) -> OCRDecision:
        """Inspect current streaming decision without finalizing the track."""
        return self._build_decision(is_final=False)

    def finish(self) -> OCRDecision:
        """Finalize track and resolve any COLLECTING state into ACCEPTED/TENTATIVE/UNREADABLE."""
        return self._build_decision(is_final=True)
