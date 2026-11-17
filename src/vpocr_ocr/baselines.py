"""Single-frame and pixel-space baseline evaluators vs MF-JPPlateOCR."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from vpocr_ocr.model import MFJPPlateOCR, MemoryReadout, MemoryState
from vpocr_ocr.vocab import JPPlateGrammar, PlateVocab, STATUS_NAMES
from vpocr_synth.plate.models import PlateText


@dataclass
class EvalPrediction:
    """Standardized evaluation result for baselines and MF-JPPlateOCR."""

    slot_ids: list[int]
    plate_text: PlateText | None
    plate_type: str
    status: str
    slot_confidences: list[float]
    slot_readability: list[float]
    conflict_score: float
    selected_frame_idx: int | None = None


def _crops_to_tensor(crops: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert (T, 64, 128, 3) uint8/float or (T, 3, 64, 128) into (1, T, 3, 64, 128) float32 [0,1]."""
    if isinstance(crops, np.ndarray):
        arr = torch.from_numpy(crops)
    else:
        arr = crops

    if arr.ndim == 4:
        arr = arr.unsqueeze(0)  # (1, T, ...)
    if arr.shape[-1] == 3 and arr.shape[2] != 3:
        # (B, T, H, W, 3) -> (B, T, 3, H, W)
        arr = arr.permute(0, 1, 4, 2, 3)

    if arr.dtype == torch.uint8:
        out = arr.to(device=device, dtype=torch.float32) / 255.0
    else:
        out = arr.to(device=device, dtype=torch.float32)
        if out.max().item() > 1.5:
            out = out / 255.0
    return torch.clamp(out, 0.0, 1.0).contiguous()


def _readout_to_predictions(
    readout: MemoryReadout,
    memory: MemoryState,
    vocab: PlateVocab,
    grammar: JPPlateGrammar | None,
    constrained: bool,
    selected_indices: list[int | None] | None = None,
) -> list[EvalPrediction]:
    """Decode batched MemoryReadout into EvalPrediction objects while preserving raw confidences."""
    raw_probs = F.softmax(readout.slot_logits.float(), dim=-1)  # (B, 12, V)
    raw_max_probs = raw_probs.max(dim=-1).values.detach().cpu().tolist()
    type_ids = readout.plate_type_logits.argmax(dim=-1).detach().cpu().tolist()
    status_ids = readout.status_logits.argmax(dim=-1).detach().cpu().tolist()
    read_preds = readout.slot_readability_pred.detach().cpu().tolist()
    conflicts = memory.mem_conflict.max(dim=-1).values.detach().cpu().tolist()
    logits_cpu = readout.slot_logits.detach().float().cpu().numpy()

    eff_grammar = grammar or (JPPlateGrammar(vocab) if constrained else None)

    results: list[EvalPrediction] = []
    bsz = len(type_ids)
    for b in range(bsz):
        pt_enum = vocab.decode_plate_type(type_ids[b])
        if constrained and eff_grammar is not None:
            s_ids = eff_grammar.constrain_slot_ids(logits_cpu[b], plate_type=pt_enum)
        else:
            s_ids = logits_cpu[b].argmax(axis=-1).tolist()

        plate_text = vocab.try_decode(s_ids)
        status_str = STATUS_NAMES[int(status_ids[b])]
        sel_idx = selected_indices[b] if selected_indices is not None else None
        results.append(
            EvalPrediction(
                slot_ids=s_ids,
                plate_text=plate_text,
                plate_type=pt_enum.value,
                status=status_str,
                slot_confidences=[float(x) for x in raw_max_probs[b]],
                slot_readability=[float(x) for x in read_preds[b]],
                conflict_score=float(conflicts[b]),
                selected_frame_idx=sel_idx,
            )
        )
    return results


class _BaseEvaluator:
    """Shared base class for GPU/CPU sequence evaluators."""

    def __init__(
        self,
        model: MFJPPlateOCR,
        vocab: PlateVocab,
        grammar: JPPlateGrammar | None = None,
        constrained: bool = True,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.grammar = grammar or JPPlateGrammar(vocab)
        self.constrained = constrained

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def evaluate_sequence(
        self,
        crops_rgb: np.ndarray | torch.Tensor,
        timestamps_s: np.ndarray | torch.Tensor | None = None,
        plate_sides: np.ndarray | torch.Tensor | None = None,
        camera_ids: np.ndarray | torch.Tensor | None = None,
        source_frame_ids: np.ndarray | torch.Tensor | None = None,
    ) -> EvalPrediction:
        dev = self.device
        crops_t = _crops_to_tensor(crops_rgb, dev)  # (1, T, 3, 64, 128)
        seq_len = crops_t.shape[1]

        if timestamps_s is None:
            ts_t = torch.arange(seq_len, device=dev, dtype=torch.float32).unsqueeze(0) * 0.1
        else:
            ts_t = torch.as_tensor(timestamps_s, device=dev, dtype=torch.float32).reshape(1, seq_len)

        sides_t = (
            torch.as_tensor(plate_sides, device=dev, dtype=torch.long).reshape(1, seq_len)
            if plate_sides is not None
            else None
        )
        cam_t = (
            torch.as_tensor(camera_ids, device=dev, dtype=torch.long).reshape(1, seq_len)
            if camera_ids is not None
            else None
        )
        src_t = (
            torch.as_tensor(source_frame_ids, device=dev, dtype=torch.long).reshape(1, seq_len)
            if source_frame_ids is not None
            else None
        )

        return self.evaluate_batch(
            crops=crops_t,
            timestamps_s=ts_t,
            plate_sides=sides_t,
            camera_ids=cam_t,
            source_frame_ids=src_t,
        )[0]

    def evaluate_batch(
        self,
        crops: torch.Tensor,
        timestamps_s: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        frame_pad_mask: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
        source_frame_ids: torch.Tensor | None = None,
    ) -> list[EvalPrediction]:
        raise NotImplementedError


class SingleFrameLastEvaluator(_BaseEvaluator):
    """Baseline decoding solely from the final observed frame in the track."""

    @torch.no_grad()
    def evaluate_batch(
        self,
        crops: torch.Tensor,
        timestamps_s: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        frame_pad_mask: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
        source_frame_ids: torch.Tensor | None = None,
    ) -> list[EvalPrediction]:
        self.model.eval()
        bsz, seq_len = crops.shape[:2]
        dev = crops.device

        if frame_pad_mask is not None:
            lengths = (~frame_pad_mask).long().sum(dim=1).clamp_min(1)
            last_idx = lengths - 1
        else:
            last_idx = torch.full((bsz,), seq_len - 1, dtype=torch.long, device=dev)

        batch_arange = torch.arange(bsz, device=dev)
        chosen_crops = crops[batch_arange, last_idx]  # (B, 3, 64, 128)
        chosen_sides = (
            plate_sides[batch_arange, last_idx]
            if plate_sides is not None
            else torch.zeros(bsz, dtype=torch.long, device=dev)
        )
        chosen_ts = timestamps_s[batch_arange, last_idx]

        enc = self.model.encode_frames(chosen_crops, chosen_sides)
        state = self.model.init_memory(batch_size=bsz, device=dev)
        state = self.model.step_memory(
            state.mem_feat,
            state.mem_mass,
            state.mem_conflict,
            state.last_timestamp_s,
            enc.z,
            enc.slot_quality,
            enc.frame_valid,
            chosen_ts,
        )
        readout = self.model.decode_memory(state.mem_feat, state.mem_mass, state.mem_conflict)
        return _readout_to_predictions(
            readout,
            state,
            self.vocab,
            self.grammar,
            self.constrained,
            selected_indices=[int(i) for i in last_idx.cpu().tolist()],
        )


class SingleFrameBestQualityEvaluator(_BaseEvaluator):
    """Strong single-frame baseline selecting t* = argmax(frame_valid * mean(slot_quality))."""

    @torch.no_grad()
    def evaluate_batch(
        self,
        crops: torch.Tensor,
        timestamps_s: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        frame_pad_mask: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
        source_frame_ids: torch.Tensor | None = None,
    ) -> list[EvalPrediction]:
        self.model.eval()
        bsz, seq_len, ch, h, w = crops.shape
        dev = crops.device

        flat_crops = crops.reshape(bsz * seq_len, ch, h, w)
        flat_sides = (
            plate_sides.reshape(bsz * seq_len)
            if plate_sides is not None
            else torch.zeros(bsz * seq_len, dtype=torch.long, device=dev)
        )
        enc = self.model.encode_frames(flat_crops, flat_sides)

        z_seq = enc.z.reshape(bsz, seq_len, 12, -1)
        q_seq = enc.slot_quality.reshape(bsz, seq_len, 12)
        v_seq = enc.frame_valid.reshape(bsz, seq_len, 1)

        scores = v_seq.squeeze(-1) * q_seq.mean(dim=-1)  # (B, T)
        if frame_pad_mask is not None:
            scores = scores.masked_fill(frame_pad_mask, -1.0)

        best_idx = scores.argmax(dim=1)  # (B,)
        batch_arange = torch.arange(bsz, device=dev)

        state = self.model.init_memory(batch_size=bsz, device=dev)
        state = self.model.step_memory(
            state.mem_feat,
            state.mem_mass,
            state.mem_conflict,
            state.last_timestamp_s,
            z_seq[batch_arange, best_idx],
            q_seq[batch_arange, best_idx],
            v_seq[batch_arange, best_idx],
            timestamps_s[batch_arange, best_idx],
        )
        readout = self.model.decode_memory(state.mem_feat, state.mem_mass, state.mem_conflict)
        return _readout_to_predictions(
            readout,
            state,
            self.vocab,
            self.grammar,
            self.constrained,
            selected_indices=[int(i) for i in best_idx.cpu().tolist()],
        )


class PixelAverageEvaluator(_BaseEvaluator):
    """Pixel-space temporal averaging baseline before single-frame decode."""

    @torch.no_grad()
    def evaluate_batch(
        self,
        crops: torch.Tensor,
        timestamps_s: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        frame_pad_mask: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
        source_frame_ids: torch.Tensor | None = None,
    ) -> list[EvalPrediction]:
        self.model.eval()
        bsz, seq_len = crops.shape[:2]
        dev = crops.device

        if frame_pad_mask is not None:
            valid_w = (~frame_pad_mask).to(dtype=crops.dtype)  # (B, T)
            denom = valid_w.sum(dim=1).clamp_min(1.0).reshape(bsz, 1, 1, 1)
            avg_crops = (crops * valid_w.reshape(bsz, seq_len, 1, 1, 1)).sum(dim=1) / denom
        else:
            avg_crops = crops.mean(dim=1)

        sides = (
            plate_sides[:, 0]
            if plate_sides is not None
            else torch.zeros(bsz, dtype=torch.long, device=dev)
        )
        enc = self.model.encode_frames(avg_crops, sides)
        state = self.model.init_memory(batch_size=bsz, device=dev)
        state = self.model.step_memory(
            state.mem_feat,
            state.mem_mass,
            state.mem_conflict,
            state.last_timestamp_s,
            enc.z,
            enc.slot_quality,
            enc.frame_valid,
            timestamps_s[:, 0],
        )
        readout = self.model.decode_memory(state.mem_feat, state.mem_mass, state.mem_conflict)
        return _readout_to_predictions(
            readout,
            state,
            self.vocab,
            self.grammar,
            self.constrained,
        )


class MultiFrameEvaluator(_BaseEvaluator):
    """Full MF-JPPlateOCR latent EvidenceMemory sequence evaluator."""

    @torch.no_grad()
    def evaluate_batch(
        self,
        crops: torch.Tensor,
        timestamps_s: torch.Tensor,
        plate_sides: torch.Tensor | None = None,
        frame_pad_mask: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
        source_frame_ids: torch.Tensor | None = None,
    ) -> list[EvalPrediction]:
        self.model.eval()
        seq_out = self.model.forward_sequence(
            crops=crops,
            timestamps_s=timestamps_s,
            plate_sides=plate_sides,
            frame_pad_mask=frame_pad_mask,
            camera_ids=camera_ids,
            source_frame_ids=source_frame_ids,
        )
        readout = MemoryReadout(
            slot_logits=seq_out.slot_logits,
            plate_type_logits=seq_out.plate_type_logits,
            slot_readability_pred=seq_out.slot_readability_pred,
            status_logits=seq_out.status_logits,
        )
        return _readout_to_predictions(
            readout,
            seq_out.memory,
            self.vocab,
            self.grammar,
            self.constrained,
        )
