"""Multi-task training losses for MF-JPPlateOCR with readability masking & garbage invariance."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from vpocr_ocr.model import MFJPPlateOCR, SequenceOutput


@dataclass
class LossWeights:
    """Scalar weights for multi-task MF-JPPlateOCR optimization."""

    seq_slots: float = 1.0
    aux_frames: float = 1.0
    plate_type: float = 0.25
    calibration: float = 0.85
    status: float = 0.65
    conflict_margin: float = 0.50
    garbage_invariance: float = 0.75


def _safe_bce_prob(pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Autocast-safe elementwise binary cross-entropy in FP32 on [0, 1] probabilities."""
    p = torch.clamp(pred_prob.float(), 1e-6, 1.0 - 1e-6)
    y = target.float()
    return -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))


def compute_effective_slot_readability(
    slot_readability: torch.Tensor,
    target_slots: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    is_foreign_plate: torch.Tensor,
    frame_pad_mask: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """Compute effective per-frame readability for all 12 slots including <PAD> positions.

    For inactive <PAD> slots in region (1..3) or classification (5..6), readability inherits
    the visibility of the field's leading active slot (slot 0 or slot 4) so the model learns
    field termination when visible, while never hallucinating when the row is glared out.
    """
    eff = slot_readability.float().clone()  # (B, T, 12)
    is_pad_slot = (target_slots == pad_id).unsqueeze(1).expand_as(eff)  # (B, T, 12)

    reg_lead = eff[:, :, 0:1].expand(-1, -1, 4)
    eff[:, :, 0:4] = torch.where(
        is_pad_slot[:, :, 0:4],
        torch.maximum(eff[:, :, 0:4], reg_lead),
        eff[:, :, 0:4],
    )

    cls_lead = eff[:, :, 4:5].expand(-1, -1, 3)
    eff[:, :, 4:7] = torch.where(
        is_pad_slot[:, :, 4:7],
        torch.maximum(eff[:, :, 4:7], cls_lead),
        eff[:, :, 4:7],
    )

    valid_non_pad = (~frame_pad_mask).float()
    # Garbage frames (neither valid target nor crisp foreign plate) have 0 readability
    visually_valid = ((frame_valid_mask > 0.5) | is_foreign_plate).float() * valid_non_pad
    eff = eff * visually_valid.unsqueeze(-1)
    return torch.clamp(eff, 0.0, 1.0)


def synthesize_gpu_garbage_frames(
    batch_size: int,
    num_garbage: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize diverse non-plate garbage crops (whiteout, dark noise, side-smear) on GPU."""
    total = batch_size * num_garbage
    # Half whiteout / specular glare, half dark asphalt / extreme horizontal smear
    half = total // 2
    rem = total - half

    # 1. Specular whiteout / overexposure
    base_white = 0.92 + 0.08 * torch.rand((half, 3, 4, 8), device=device)
    whiteout = F.interpolate(base_white, size=(64, 128), mode="bilinear", align_corners=False)
    whiteout = torch.clamp(whiteout + 0.02 * torch.randn_like(whiteout), 0.0, 1.0)

    # 2. Dark sensor noise + extreme horizontal motion smear (non-plate bumper / >80 deg yaw)
    low_freq = 0.35 * torch.rand((rem, 3, 8, 2), device=device)
    smear = F.interpolate(low_freq, size=(64, 128), mode="bilinear", align_corners=False)
    noise = 0.08 * torch.randn((rem, 3, 64, 128), device=device)
    dark_smear = torch.clamp(smear + noise, 0.0, 1.0)

    crops = torch.cat([whiteout, dark_smear], dim=0)
    perm = torch.randperm(total, device=device)
    crops = crops[perm].reshape(batch_size, num_garbage, 3, 64, 128)
    sides = torch.full((batch_size, num_garbage), 2, dtype=torch.long, device=device)
    return crops, sides


class MFPlateLoss(nn.Module):
    """Combined multi-frame OCR objective for Stage 1 and Stage 2 training."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        weights: LossWeights | None = None,
        label_smoothing: float = 0.02,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.weights = weights or LossWeights()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        model: MFJPPlateOCR,
        seq_out: SequenceOutput,
        batch: dict[str, torch.Tensor],
        compute_garbage_invariance: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total weighted loss and scalar diagnostic metrics."""
        target_slots = batch["target_slots"]                # (B, 12) long
        target_plate_type = batch["target_plate_type"]      # (B,) long
        target_status = batch["target_status"]              # (B,) long in {0, 1, 2}
        has_target_plate = batch["has_target_plate"]        # (B,) bool
        frame_valid_mask = batch["frame_valid_mask"]        # (B, T) float
        is_foreign_plate = batch["is_foreign_plate"]        # (B, T) bool
        slot_readability = batch["slot_readability"]        # (B, T, 12) float
        frame_pad_mask = batch["frame_pad_mask"]            # (B, T) bool

        bsz, seq_len = frame_valid_mask.shape
        dev = target_slots.device

        eff_read = compute_effective_slot_readability(
            slot_readability=slot_readability,
            target_slots=target_slots,
            frame_valid_mask=frame_valid_mask,
            is_foreign_plate=is_foreign_plate,
            frame_pad_mask=frame_pad_mask,
            pad_id=self.pad_id,
        )  # (B, T, 12)

        # Readability restricted to genuine non-foreign target frames
        target_frame_mask = (
            (~frame_pad_mask)
            & (~is_foreign_plate)
            & (frame_valid_mask > 0.5)
            & has_target_plate.unsqueeze(1)
        )  # (B, T)
        non_foreign_read = eff_read * target_frame_mask.unsqueeze(-1).float()
        seq_max_read = non_foreign_read.max(dim=1).values  # (B, 12)

        # ------------------------------------------------------------------
        # 1. Sequence 12-Slot Cross-Entropy (masked by seq_slot_readable)
        # ------------------------------------------------------------------
        non_mixed_seq = (target_status != 2) & has_target_plate  # (B,)
        seq_slot_readable = (seq_max_read >= 0.30) & non_mixed_seq.unsqueeze(1)  # (B, 12)

        slot_ce = F.cross_entropy(
            seq_out.slot_logits.float().reshape(-1, self.vocab_size),
            target_slots.reshape(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape(bsz, 12)
        slot_w = seq_max_read * seq_slot_readable.float()
        if slot_w.sum() > 1e-6:
            loss_seq_slots = (slot_ce * slot_w).sum() / slot_w.sum()
        else:
            loss_seq_slots = seq_out.slot_logits.float().sum() * 0.0

        # Plate type loss on sequences with >=1 readable slot
        has_readable_any = (seq_max_read.max(dim=1).values >= 0.30) & non_mixed_seq
        if has_readable_any.any():
            loss_plate_type = F.cross_entropy(
                seq_out.plate_type_logits.float()[has_readable_any],
                target_plate_type[has_readable_any],
            )
        else:
            loss_plate_type = seq_out.plate_type_logits.float().sum() * 0.0

        # ------------------------------------------------------------------
        # 2. Auxiliary Per-Frame Slot Decode Loss (teaches crisp localization)
        # ------------------------------------------------------------------
        aux_logits = model.decode_slot_features(seq_out.frame_z)  # (B, T, 12, V)
        if "frame_slot_targets" in batch:
            frame_targets = batch["frame_slot_targets"]  # (B, T, 12), -1 where invalid
            valid_ft = (frame_targets >= 0) & (~frame_pad_mask).unsqueeze(-1)
            safe_ft = frame_targets.clamp_min(0)
            aux_ce = F.cross_entropy(
                aux_logits.float().reshape(-1, self.vocab_size),
                safe_ft.reshape(-1),
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).reshape(bsz, seq_len, 12)
            # For foreign frames with explicit targets, readability is high (~0.9)
            foreign_read = torch.where(
                is_foreign_plate.unsqueeze(-1),
                torch.full_like(eff_read, 0.90),
                eff_read,
            )
            aux_w = foreign_read * (foreign_read >= 0.25).float() * valid_ft.float()
        else:
            expanded_targets = target_slots.unsqueeze(1).expand(bsz, seq_len, 12)
            aux_ce = F.cross_entropy(
                aux_logits.float().reshape(-1, self.vocab_size),
                expanded_targets.reshape(-1),
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).reshape(bsz, seq_len, 12)
            aux_mask = target_frame_mask.unsqueeze(-1) & (eff_read >= 0.25)
            aux_w = eff_read * aux_mask.float()

        if aux_w.sum() > 1e-6:
            loss_aux_frames = (aux_ce * aux_w).sum() / aux_w.sum()
        else:
            loss_aux_frames = aux_logits.float().sum() * 0.0

        # ------------------------------------------------------------------
        # 3. Quality & Validity Calibration Loss
        # ------------------------------------------------------------------
        non_pad_f = (~frame_pad_mask).float()
        visual_valid_target = (((frame_valid_mask > 0.5) | is_foreign_plate) & (~frame_pad_mask)).float()

        # Raw logit BCE pushes garbage frames well below sigmoid=0.05 floor
        raw_v_bce = F.binary_cross_entropy_with_logits(
            seq_out.frame_raw_valid_logit.float().squeeze(-1),
            visual_valid_target,
            reduction="none",
        )
        v_mse = (seq_out.frame_valid.float().squeeze(-1) - visual_valid_target).pow(2)
        loss_valid_cal = ((raw_v_bce + v_mse) * non_pad_f).sum() / non_pad_f.sum().clamp_min(1.0)

        # Per-slot quality calibration on non-foreign frames (including garbage frames -> 0)
        q_cal_mask = ((~frame_pad_mask) & (~is_foreign_plate)).unsqueeze(-1).float()  # (B, T, 1)
        q_bce = F.binary_cross_entropy_with_logits(
            seq_out.frame_raw_slot_quality_logit.float(),
            eff_read,
            reduction="none",
        )
        q_mse = (seq_out.frame_slot_quality.float() - eff_read).pow(2)
        loss_slot_q_cal = ((q_bce + q_mse) * q_cal_mask).sum() / (
            q_cal_mask.sum() * 12.0
        ).clamp_min(1.0)

        # Sequence slot readability prediction calibration on non-mixed sequences
        seq_read_target = (seq_max_read >= 0.30).float()
        if non_mixed_seq.any():
            sr_bce = _safe_bce_prob(
                seq_out.slot_readability_pred[non_mixed_seq],
                seq_read_target[non_mixed_seq],
            )
            sr_mse = (
                seq_out.slot_readability_pred[non_mixed_seq].float()
                - seq_read_target[non_mixed_seq]
            ).pow(2)
            loss_seq_read_cal = (0.5 * sr_bce + sr_mse).mean()
        else:
            loss_seq_read_cal = seq_out.slot_readability_pred.float().sum() * 0.0

        loss_calibration = loss_valid_cal + loss_slot_q_cal + loss_seq_read_cal

        # ------------------------------------------------------------------
        # 4. Sequence Status CE + Latent Conflict Margin Loss
        # ------------------------------------------------------------------
        loss_status = F.cross_entropy(seq_out.status_logits.float(), target_status)

        max_conflict = seq_out.memory.mem_conflict.float().max(dim=-1).values  # (B,)
        is_mixed_seq = target_status == 2
        conflict_losses = []
        if (~is_mixed_seq).any():
            conflict_losses.append(F.relu(max_conflict[~is_mixed_seq] - 0.18).mean())
        if is_mixed_seq.any():
            conflict_losses.append(F.relu(1.05 - max_conflict[is_mixed_seq]).mean())
        loss_conflict = (
            sum(conflict_losses) / len(conflict_losses)
            if conflict_losses
            else max_conflict.sum() * 0.0
        )

        # ------------------------------------------------------------------
        # 5. Synthetic Garbage-Frame Appending Invariance Loss
        # ------------------------------------------------------------------
        if compute_garbage_invariance:
            num_g = 2
            g_crops, g_sides = synthesize_gpu_garbage_frames(bsz, num_g, dev)
            g_enc = model.encode_frames(
                g_crops.reshape(bsz * num_g, 3, 64, 128),
                g_sides.reshape(bsz * num_g),
            )
            g_raw_bce = F.binary_cross_entropy_with_logits(
                g_enc.raw_valid_logit.float(),
                torch.zeros_like(g_enc.raw_valid_logit.float()),
            )
            g_zero_floor = g_enc.frame_valid.float().pow(2).mean() * 0.5

            g_z = g_enc.z.reshape(bsz, num_g, 12, -1)
            g_q = g_enc.slot_quality.reshape(bsz, num_g, 12)
            g_v = g_enc.frame_valid.reshape(bsz, num_g, 1)

            st_after = seq_out.memory
            base_ts = torch.clamp(st_after.last_timestamp_s, min=0.0)
            for k in range(num_g):
                st_after = model.step_memory(
                    st_after.mem_feat,
                    st_after.mem_mass,
                    st_after.mem_conflict,
                    st_after.last_timestamp_s,
                    g_z[:, k],
                    g_q[:, k],
                    g_v[:, k],
                    base_ts + 0.1 * float(k + 1),
                )
            ro_after = model.decode_memory(
                st_after.mem_feat, st_after.mem_mass, st_after.mem_conflict
            )
            feat_drift = (
                st_after.mem_feat.float() - seq_out.memory.mem_feat.float().detach()
            ).pow(2).mean()
            logit_drift = (
                ro_after.slot_logits.float() - seq_out.slot_logits.float().detach()
            ).pow(2).mean()
            loss_garbage = g_raw_bce + g_zero_floor + 2.0 * feat_drift + 0.5 * logit_drift
        else:
            loss_garbage = seq_out.slot_logits.float().sum() * 0.0

        total_loss = (
            self.weights.seq_slots * loss_seq_slots
            + self.weights.aux_frames * loss_aux_frames
            + self.weights.plate_type * loss_plate_type
            + self.weights.calibration * loss_calibration
            + self.weights.status * loss_status
            + self.weights.conflict_margin * loss_conflict
            + self.weights.garbage_invariance * loss_garbage
        )

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_seq_slots": float(loss_seq_slots.detach().item()),
            "loss_aux_frames": float(loss_aux_frames.detach().item()),
            "loss_plate_type": float(loss_plate_type.detach().item()),
            "loss_calibration": float(loss_calibration.detach().item()),
            "loss_status": float(loss_status.detach().item()),
            "loss_conflict": float(loss_conflict.detach().item()),
            "loss_garbage": float(loss_garbage.detach().item()),
        }
        return total_loss, metrics
