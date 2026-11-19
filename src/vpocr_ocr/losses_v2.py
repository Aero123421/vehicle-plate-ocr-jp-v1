"""Gen-2 multi-task losses (`OCRLossV2`) for `MFJPPlateOCRV2`.

Combines:
1. Masked Final Sequence Slot CE + Per-Frame Auxiliary Slot CE + Status CE + Readability/Valid BCE
   + Monochrome-IR-Masked Color Type CE + Polarity BCE.
2. Anisotropic Gaussian NLL Uncertainty Calibration Loss (`L_unc_cal`) vs clean canonical reference.
3. Multi-View Latent Consistency Loss (`L_mv_cons`).
4. Temporal Prefix Monotonicity Loss (`L_mono`).
5. Counterfactual In-Batch Cross-Vehicle Splice Mahalanobis Margin Loss (`L_cf_conflict`)
   & Zero-Drift Garbage Invariance Loss (`L_garbage`).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from vpocr_ocr.model_v2 import MFJPPlateOCRV2, SequenceOutputV2
from vpocr_synth.chaos_info import (
    coupled_map_lattice_2d,
    focal_self_information_ce,
    shannon_slot_capacity_weights,
)


@contextmanager
def _freeze_bn_running_stats(module: nn.Module):
    """Put BatchNorm layers into eval mode temporarily so clean/garbage aux passes never pollute running stats."""
    bns = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    prev_states = [m.training for m in bns]
    try:
        for m in bns:
            m.eval()
        yield
    finally:
        for m, st in zip(bns, prev_states):
            m.train(st)


@dataclass
class LossWeightsV2:
    """Scalar weights for `OCRLossV2` multi-task optimization."""

    seq_slots: float = 1.0
    aux_frames: float = 0.85
    status: float = 0.65
    calibration: float = 0.75
    color_type: float = 0.35
    polarity: float = 0.25
    unc_cal: float = 0.35
    mv_cons: float = 0.45
    mono: float = 0.30
    cf_conflict: float = 0.55
    garbage_invariance: float = 0.70
    stroke_gestalt: float = 0.25
    statutory_len: float = 0.35
    innovation_nll: float = 0.02
    terminal_risk: float = 0.25
    memory_sr: float = 0.14
    f1_sr: float = 0.08


def _safe_bce_prob_v2(pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(pred_prob.float(), 1e-6, 1.0 - 1e-6)
    y = target.float()
    return -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))


def compute_effective_slot_readability_v2(
    slot_readability: torch.Tensor,
    frame_slot_targets: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    frame_pad_mask: torch.Tensor,
    slot_glare: torch.Tensor | None = None,
    pad_id: int = 0,
) -> torch.Tensor:
    """Propagate leading active-slot readability to `<PAD>` termination slots `(1..3, 5..6)`."""
    eff = slot_readability.float().clone()  # (B, T, 12)
    if slot_glare is not None:
        eff = eff * torch.clamp(1.0 - 0.35 * slot_glare.float(), 0.15, 1.0)

    is_pad_slot = frame_slot_targets == pad_id  # (B, T, 12)
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

    valid_f = ((frame_valid_mask > 0.5) & (~frame_pad_mask)).float()
    return torch.clamp(eff * valid_f.unsqueeze(-1), 0.0, 1.0)


def synthesize_gpu_garbage_frames_v2(
    batch_size: int,
    num_garbage: int,
    hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthesize diverse construction, vehicle-body & roadside non-plate hard-negative crops + geom_meta."""
    h, w = hw
    total = max(1, batch_size * num_garbage)
    k = max(1, total // 10)
    counts: list[int] = []
    rem = total
    for _ in range(9):
        c_i = min(rem, k)
        counts.append(c_i)
        rem -= c_i
    counts.append(max(0, rem))
    n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = counts

    parts: list[torch.Tensor] = []

    # 1. Specular headlight whiteout / IR halation blowout
    if n1 > 0:
        w_low = 0.90 + 0.10 * torch.rand((n1, 3, 4, 8), device=device)
        g1 = F.interpolate(w_low, size=(h, w), mode="bilinear", align_corners=False)
        parts.append(torch.clamp(g1 + 0.02 * torch.randn_like(g1), 0.0, 1.0))

    # 2. Dark asphalt / mud / extreme horizontal motion smear
    if n2 > 0:
        d_low = 0.32 * torch.rand((n2, 3, 8, 2), device=device)
        g2 = F.interpolate(d_low, size=(h, w), mode="bilinear", align_corners=False)
        parts.append(torch.clamp(g2 + 0.06 * torch.randn_like(g2), 0.0, 1.0))

    # 3. Periodic radiator grille / dump-truck tailgate metallic ribs
    if n3 > 0:
        xs = torch.linspace(0.0, 14.0 * math.pi, w, device=device).reshape(1, 1, 1, w)
        phase = torch.rand((n3, 1, 1, 1), device=device) * 6.28
        stripes = 0.42 + 0.32 * torch.sin(xs + phase)
        g3 = stripes.expand(n3, 3, h, w) + 0.05 * torch.randn((n3, 3, h, w), device=device)
        parts.append(torch.clamp(g3, 0.0, 1.0))

    # 4. Red/white construction gate barrier arm & yellow/black hazard chevron stripes
    if n4 > 0:
        yy = torch.linspace(0.0, 1.0, h, device=device).reshape(1, 1, h, 1)
        xx = torch.linspace(0.0, 4.0, w, device=device).reshape(1, 1, 1, w)
        ph4 = torch.rand((n4, 1, 1, 1), device=device) * 4.0
        diag = ((xx + yy + ph4) % 1.0 > 0.5).float()
        is_hazard = (torch.rand((n4, 1, 1, 1), device=device) > 0.40).float()
        col_a = is_hazard * torch.tensor([0.92, 0.78, 0.08], device=device).view(1, 3, 1, 1) + (
            1.0 - is_hazard
        ) * torch.tensor([0.90, 0.14, 0.12], device=device).view(1, 3, 1, 1)
        col_b = is_hazard * torch.tensor([0.10, 0.10, 0.10], device=device).view(1, 3, 1, 1) + (
            1.0 - is_hazard
        ) * torch.tensor([0.94, 0.94, 0.94], device=device).view(1, 3, 1, 1)
        g4 = diag * col_a + (1.0 - diag) * col_b + 0.04 * torch.randn((n4, 3, h, w), device=device)
        parts.append(torch.clamp(g4, 0.0, 1.0))

    # 5. Real Japanese & Latin text-bearing non-plate stencils (`319`, `スケルトン車`, `TOLL GATE`, etc.)
    from vpocr_synth.plate.surface_25d import get_default_nonplate_stencil_atlas

    if n5 > 0:
        stencil_atlas = get_default_nonplate_stencil_atlas()
        g5 = stencil_atlas.sample_crops(n5, (h, w), device)
        parts.append(g5)

    # 6. Kaneko 2D Coupled-Map-Lattice (CML) + extra text-bearing stencil mix
    if n6 > 0:
        n6_st = n6 // 2
        n6_cml = n6 - n6_st
        if n6_st > 0:
            parts.append(get_default_nonplate_stencil_atlas().sample_crops(n6_st, (h, w), device))
        if n6_cml > 0:
            cml_low = coupled_map_lattice_2d(
                n6_cml, max(8, h // 4), max(16, w // 4), steps=6, r=3.97, coupling=0.28, device=device
            )
            cml = F.interpolate(cml_low, size=(h, w), mode="bilinear", align_corners=False)
            tint = 0.75 + 0.45 * torch.rand((n6_cml, 3, 1, 1), device=device)
            g6 = torch.clamp(cml * tint + 0.04 * torch.randn((n6_cml, 3, h, w), device=device), 0.0, 1.0)
            parts.append(g6)

    # 7. Glowing red LED brake-light / amber reflector housing false-positive crops
    if n7 > 0:
        yy7 = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, h, 1)
        xx7 = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, w)
        glow = torch.exp(-1.8 * (xx7 * xx7 * 0.65 + yy7 * yy7 * 1.35))
        red_base = torch.tensor([0.92, 0.10, 0.08], device=device).view(1, 3, 1, 1)
        core = torch.tensor([1.0, 0.42, 0.25], device=device).view(1, 3, 1, 1)
        g7 = (0.14 + 0.86 * glow) * red_base + 0.35 * (glow ** 3) * core
        g7 = torch.clamp(g7 + 0.04 * torch.randn((n7, 3, h, w), device=device), 0.0, 1.0)
        parts.append(g7)

    # 8. Glossy white / pink / pastel specular blobs (common bumper/taillight/signage false positives)
    if n8 > 0:
        yy8 = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, h, 1)
        xx8 = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, w)
        cx8 = (torch.rand((n8, 1, 1, 1), device=device) * 1.2 - 0.6)
        cy8 = (torch.rand((n8, 1, 1, 1), device=device) * 0.8 - 0.4)
        blob = torch.exp(-(((xx8 - cx8) / 0.52) ** 2 + ((yy8 - cy8) / 0.48) ** 2))
        is_pink = (torch.rand((n8, 1, 1, 1), device=device) < 0.55).float()
        pink_col = torch.tensor([0.95, 0.58, 0.72], device=device).view(1, 3, 1, 1)
        white_col = torch.tensor([0.92, 0.93, 0.95], device=device).view(1, 3, 1, 1)
        base_b = is_pink * pink_col + (1.0 - is_pink) * white_col
        bg_b = 0.12 + 0.25 * torch.rand((n8, 3, 1, 1), device=device)
        g8 = bg_b * (1.0 - blob) + (base_b + 0.22 * blob.pow(3)) * blob
        parts.append(torch.clamp(g8 + 0.025 * torch.randn((n8, 3, h, w), device=device), 0.0, 1.0))

    # 9. Pure blank brushed metal / empty plate-colored aluminum panel (zero characters!)
    if n9 > 0:
        gx9 = torch.linspace(-0.5, 0.5, w, device=device).view(1, 1, 1, w)
        gy9 = torch.linspace(-0.5, 0.5, h, device=device).view(1, 1, h, 1)
        grain = 0.03 * torch.randn((n9, 1, h, 1), device=device).expand(n9, 1, h, w)
        base_m = 0.52 + 0.36 * torch.rand((n9, 1, 1, 1), device=device)
        tint_m = torch.tensor([1.0, 0.99, 0.98], device=device).view(1, 3, 1, 1)
        grad_m = 1.0 + 0.18 * gx9 - 0.14 * gy9
        g9 = base_m * tint_m * grad_m + grain + 0.015 * torch.randn((n9, 3, h, w), device=device)
        parts.append(torch.clamp(g9, 0.0, 1.0))

    # 10. Extreme high-frequency Gaussian / salt-and-pepper / blocky sensor noise
    if n10 > 0:
        hi_n = torch.rand((n10, 3, h, w), device=device)
        blk_n = F.interpolate(
            torch.rand((n10, 3, max(4, h // 8), max(8, w // 8)), device=device),
            size=(h, w),
            mode="nearest",
        )
        mix_w = torch.rand((n10, 1, 1, 1), device=device)
        g10 = mix_w * hi_n + (1.0 - mix_w) * blk_n
        parts.append(torch.clamp(g10, 0.0, 1.0))

    all_g = torch.cat(parts, dim=0)[:total]
    perm = torch.randperm(total, device=device)
    crops = all_g[perm].reshape(batch_size, num_garbage, 3, h, w)
    sides = torch.randint(0, 3, (batch_size, num_garbage), dtype=torch.long, device=device)

    # Multi-scale & multi-sharpness geom_meta so close-range gate garbage is also forced to zero-write!
    flat_c = crops.reshape(total, 3, h, w)
    y_lum = 0.299 * flat_c[:, 0] + 0.587 * flat_c[:, 1] + 0.114 * flat_c[:, 2]
    mu_y = y_lum.mean(dim=(1, 2))
    r_clip = (y_lum > 0.96).float().mean(dim=(1, 2))
    chroma = (flat_c.max(dim=1).values - flat_c.min(dim=1).values).reshape(total, -1)
    s_chroma = torch.topk(chroma, k=max(1, (h * w * 15) // 100), dim=-1).values.mean(dim=-1)
    g0_rand = -0.60 + 1.50 * torch.rand_like(mu_y)
    g1_rand = torch.clamp(g0_rand + 0.15 * torch.randn_like(mu_y), -1.0, 1.0)
    sharp_rand = 0.10 + 0.75 * torch.rand_like(mu_y)
    zeros = torch.zeros_like(mu_y)
    geom = torch.stack(
        [g0_rand, g1_rand, zeros, zeros, zeros, torch.full_like(mu_y, 0.35), sharp_rand, mu_y, s_chroma, r_clip],
        dim=-1,
    ).reshape(batch_size, num_garbage, 10)
    return crops, sides, geom


@torch.no_grad()
def offline_terminal_risk_targets(readout, memory, target_slots, target_status,
                                  num_frames, vocab, config, grammar):
    """Actual terminal MBR-policy losses on offline synthetic prefixes only.

    Rejected outcomes incur the declared abstention cost. Accepted partial/full
    outputs incur wrong cost iff a released field or physical identity is wrong.
    No production collecting session invokes this helper.
    """
    from vpocr_ocr.session_v2 import build_decision_v2_from_arrays
    arrays = [a.detach().float().cpu().numpy() for a in readout]
    mem = [a.detach().float().cpu().numpy() for a in memory]
    targets = target_slots.detach().cpu().tolist()
    statuses = target_status.detach().cpu().tolist()
    losses = []
    slices = {"region": slice(0,4), "classification": slice(4,7),
              "hiragana": slice(7,8), "serial": slice(8,12)}
    for i, ids in enumerate(targets):
        dec = build_decision_v2_from_arrays(arrays[0][i], arrays[1][i], arrays[2][i],
            arrays[3][i], arrays[4][i], mem[1][i], mem[2][i], mem[4][i],
            float(mem[6][i]), num_frames, vocab, config, grammar)
        if dec.status != "ACCEPTED":
            loss = config.C_abstain
        else:
            pred_ids = vocab.encode(dec.plate_text)
            released = tuple(slices) if dec.confidence_tier == "FULL_4FIELD" else dec.verified_fields
            wrong = statuses[i] != 0 or any(pred_ids[slices[f]] != ids[slices[f]] for f in released)
            loss = config.C_wrong * float(wrong)
        losses.append(min(1.0, loss/max(config.C_wrong, 1e-8)))
    return torch.tensor(losses, device=target_slots.device, dtype=torch.float32)


class OCRLossV2(nn.Module):
    """NaN-safe FP32 multi-task objective for `MFJPPlateOCRV2`."""

    def __init__(
        self,
        vocab_size: int = 254,
        pad_id: int = 0,
        weights: LossWeightsV2 | None = None,
        label_smoothing: float = 0.006,
        terminal_risk_config=None,
        vocab=None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.weights = weights or LossWeightsV2()
        self.label_smoothing = label_smoothing
        from vpocr_ocr.session_v2 import SessionConfigV2
        from vpocr_ocr.vocab import PlateVocab, JPPlateGrammar
        self.risk_config = terminal_risk_config or SessionConfigV2()
        self.risk_vocab = vocab or PlateVocab()
        self.risk_grammar = JPPlateGrammar(self.risk_vocab)
        sobel_xy = (
            torch.tensor(
                [
                    [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                    [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
                ],
                dtype=torch.float32,
            )
            * 0.25
        )
        self.register_buffer("sobel_xy", sobel_xy, persistent=False)

    def _sobel_xy_grads(self, img_1ch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.sobel_xy.to(device=img_1ch.device, dtype=torch.float32)
        gxy = F.conv2d(F.pad(img_1ch.float(), (1, 1, 1, 1), mode="replicate"), k)
        return gxy[:, 0:1], gxy[:, 1:2]

    def forward(
        self,
        model: MFJPPlateOCRV2,
        seq_out: SequenceOutputV2,
        batch: dict[str, Any],
        compute_garbage_invariance: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total weighted FP32 loss and scalar diagnostic dictionary."""
        crops = batch["crops"]                              # (B, T, 3, H, W)
        target_slots = batch["target_slots"].long()         # (B, 12)
        target_plate_type = batch["target_plate_type"].long()  # (B,)
        target_status = batch["target_status"].long()       # (B,) in {0, 1, 2}
        frame_valid_mask = batch["frame_valid_mask"].float()  # (B, T)
        slot_readability = batch["slot_readability"].float()  # (B, T, 12)

        bsz, seq_len, _, h_img, w_img = crops.shape
        dev = crops.device

        frame_pad_mask = batch.get(
            "frame_pad_mask",
            torch.zeros((bsz, seq_len), dtype=torch.bool, device=dev),
        ).bool()
        slot_glare = batch.get("slot_glare", None)
        frame_slot_targets = batch.get(
            "frame_slot_targets",
            target_slots.unsqueeze(1).expand(bsz, seq_len, 12),
        ).long()
        is_monochrome_ir = batch.get(
            "is_monochrome_ir",
            torch.zeros((bsz, seq_len), dtype=torch.float32, device=dev),
        ).float()
        polarity_target = batch.get("polarity_target", None)

        eff_read = compute_effective_slot_readability_v2(
            slot_readability=slot_readability,
            frame_slot_targets=frame_slot_targets,
            frame_valid_mask=frame_valid_mask,
            frame_pad_mask=frame_pad_mask,
            slot_glare=slot_glare,
            pad_id=self.pad_id,
        )  # (B, T, 12)

        non_pad_mask = ~frame_pad_mask
        valid_frame_bool = non_pad_mask & (frame_valid_mask > 0.5)
        # Frames showing the primary target vehicle (excludes spliced conflicting-vehicle frames)
        same_as_primary = (frame_slot_targets == target_slots.unsqueeze(1)).all(dim=-1)
        is_primary_frame = valid_frame_bool & same_as_primary  # (B, T)

        primary_read = eff_read * is_primary_frame.unsqueeze(-1).float()
        seq_max_read = primary_read.max(dim=1).values  # (B, 12)
        sr_seq_gain = 1.85 if seq_len >= 4 else (1.35 if seq_len >= 2 else 1.0)
        seq_int_read = torch.maximum(
            seq_max_read,
            (sr_seq_gain * torch.linalg.vector_norm(primary_read, ord=2, dim=1)).clamp(0.0, 1.0),
        )  # (B, 12)
        non_mixed_seq = target_status != 2  # (B,)
        is_readable_seq = target_status == 0  # (B,)

        zero_anchor = seq_out.slot_logits.float().sum() * 0.0

        # ------------------------------------------------------------------
        # 1. Final Sequence Slot CE + Per-Frame Auxiliary Slot CE + Status CE
        # ------------------------------------------------------------------
        slot_base_mult = shannon_slot_capacity_weights(dev)
        seq_slot_mult = slot_base_mult.unsqueeze(0).expand(bsz, 12)

        seq_slot_readable = (
            ((seq_int_read >= 0.16) & is_readable_seq.unsqueeze(1))
            | ((seq_int_read >= 0.30) & non_mixed_seq.unsqueeze(1))
        )
        slot_ce = focal_self_information_ce(
            seq_out.slot_logits.float(),
            target_slots,
            gamma=0.45,
            floor_mix=0.72,
            label_smoothing=self.label_smoothing,
        )
        slot_w = seq_int_read.clamp_min(0.20) * seq_slot_readable.float() * seq_slot_mult
        if slot_w.sum() > 1e-6:
            loss_seq_slots = (slot_ce * slot_w).sum() / slot_w.sum()
        else:
            loss_seq_slots = zero_anchor

        # Vault-Door Bypass `<PAD>` supervision (`weight = 0.30`): when `target_status == 1` (`UNREADABLE`)
        # accumulates `mem_P > 0` (including the 50% training vault-door bypass!), explicitly train
        # `memory_readout` `slot_logits -> pad_id` so leaked non-plate text (`319`, `スケルトン車`) abstains!
        unread_leak_mask = (target_status == 1) & (
            seq_out.memory.mem_P.float().mean(dim=(1, 2)) > 1e-4
        )
        if bool(unread_leak_mask.any()):
            n_ul = int(unread_leak_mask.sum().item())
            pad_tgt_ul = torch.full((n_ul * 12,), self.pad_id, dtype=torch.long, device=dev)
            loss_vault_pad = F.cross_entropy(
                seq_out.slot_logits[unread_leak_mask].float().reshape(-1, self.vocab_size),
                pad_tgt_ul,
            )
        else:
            loss_vault_pad = zero_anchor

        aux_logits = model.decode_slot_features(seq_out.frame_z.to(dtype=model.readout_slot_embed.dtype))
        valid_ft = valid_frame_bool.unsqueeze(-1) & (frame_slot_targets >= 0) & (eff_read >= 0.25)
        safe_ft = frame_slot_targets.clamp(0, self.vocab_size - 1)
        aux_slot_mult = slot_base_mult.view(1, 1, 12).expand(bsz, seq_len, 12)
        aux_ce = focal_self_information_ce(
            aux_logits.float(),
            safe_ft,
            gamma=0.40,
            floor_mix=0.75,
            label_smoothing=self.label_smoothing,
        )
        aux_w = eff_read * valid_ft.float() * aux_slot_mult
        if aux_w.sum() > 1e-6:
            loss_aux_frames = (aux_ce * aux_w).sum() / aux_w.sum()
        else:
            loss_aux_frames = zero_anchor

        loss_status_final = F.cross_entropy(seq_out.status_logits.float(), target_status)

        pref_ro = getattr(seq_out, "prefix_readout", None)
        k_pref = int(getattr(seq_out, "prefix_len", 0) or 0)
        has_pref_ro = pref_ro is not None and 1 <= k_pref <= seq_len
        if has_pref_ro:
            pref_prim_read = primary_read[:, :k_pref]                     # (B, k_pref, 12)
            pref_max_read = pref_prim_read.max(dim=1).values              # (B, 12)
            sr_pref_gain = 1.85 if k_pref >= 4 else (1.35 if k_pref >= 2 else 1.0)
            pref_int_read = torch.maximum(
                pref_max_read,
                (sr_pref_gain * torch.linalg.vector_norm(pref_prim_read, ord=2, dim=1)).clamp(0.0, 1.0),
            )
            has_prim_pref = is_primary_frame[:, :k_pref].any(dim=1)       # (B,)
            has_other_pref = (
                valid_frame_bool[:, :k_pref] & (~same_as_primary[:, :k_pref])
            ).any(dim=1)                                                  # (B,)
            pref_is_mixed = (target_status == 2) & has_prim_pref & has_other_pref
            pref_core_readable = (
                (pref_int_read[:, [7, 8, 9, 10, 11]].mean(dim=-1) >= 0.24)
                & (target_status != 1)
                & (~pref_is_mixed)
            )
            pref_target_status = torch.ones_like(target_status)
            pref_target_status = torch.where(
                pref_core_readable, torch.zeros_like(target_status), pref_target_status
            )
            pref_target_status = torch.where(
                pref_is_mixed, torch.full_like(target_status, 2), pref_target_status
            )
            loss_status_pref = F.cross_entropy(pref_ro.status_logits.float(), pref_target_status)
            loss_status = 0.65 * loss_status_final + 0.35 * loss_status_pref

            pref_slot_readable = (pref_int_read >= 0.24) & (target_status != 1).unsqueeze(1) & (~pref_is_mixed).unsqueeze(1)
            pref_slot_ce = F.cross_entropy(
                pref_ro.slot_logits.float().reshape(-1, self.vocab_size),
                target_slots.clamp(0, self.vocab_size - 1).reshape(-1),
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).reshape(bsz, 12)
            pref_slot_w = pref_int_read * pref_slot_readable.float() * seq_slot_mult
            if pref_slot_w.sum() > 1e-6:
                loss_pref_slots = (pref_slot_ce * pref_slot_w).sum() / pref_slot_w.sum()
                loss_seq_slots = 0.80 * loss_seq_slots + 0.20 * loss_pref_slots
        else:
            loss_status = loss_status_final

        # ------------------------------------------------------------------
        # 2. Readability, Frame Validity, IR-Masked Color Type, & Polarity
        # ------------------------------------------------------------------
        non_pad_f = non_pad_mask.float()
        fv_target = ((frame_valid_mask > 0.5) & non_pad_mask).float()
        fv_bce = F.binary_cross_entropy_with_logits(
            seq_out.frame_valid_logit.float(), fv_target, reduction="none"
        )
        fv_mse = (seq_out.frame_valid_gate.float() - fv_target).pow(2)
        loss_valid_cal = ((fv_bce + fv_mse) * non_pad_f).sum() / non_pad_f.sum().clamp_min(1.0)
        invalid_non_pad = non_pad_mask & (frame_valid_mask < 0.5)
        if bool(invalid_non_pad.any()):
            loss_valid_cal = loss_valid_cal + 0.35 * seq_out.frame_lambda_prec[invalid_non_pad].float().mean()

        sq_bce = F.binary_cross_entropy_with_logits(
            seq_out.slot_quality_logits.float(), eff_read, reduction="none"
        )
        loss_sq_cal = (sq_bce * non_pad_f.unsqueeze(-1)).sum() / (
            non_pad_f.sum() * 12.0
        ).clamp_min(1.0)

        seq_read_target = ((seq_int_read >= 0.24) & (target_status != 1).unsqueeze(1)).float()
        if non_mixed_seq.any():
            sr_bce = _safe_bce_prob_v2(
                seq_out.slot_readability_pred[non_mixed_seq], seq_read_target[non_mixed_seq]
            )
            sr_mse = (
                seq_out.slot_readability_pred[non_mixed_seq].float()
                - seq_read_target[non_mixed_seq]
            ).pow(2)
            loss_seq_read_cal = (0.5 * sr_bce + sr_mse).mean()
        else:
            loss_seq_read_cal = zero_anchor

        if has_pref_ro:
            pref_non_mixed = ~pref_is_mixed
            pref_read_target = ((pref_int_read >= 0.24) & (target_status != 1).unsqueeze(1)).float()
            if pref_non_mixed.any():
                psr_bce = _safe_bce_prob_v2(
                    pref_ro.slot_readability_pred[pref_non_mixed],
                    pref_read_target[pref_non_mixed],
                )
                psr_mse = (
                    pref_ro.slot_readability_pred[pref_non_mixed].float()
                    - pref_read_target[pref_non_mixed]
                ).pow(2)
                loss_pref_read_cal = (0.5 * psr_bce + psr_mse).mean()
                loss_seq_read_cal = 0.65 * loss_seq_read_cal + 0.35 * loss_pref_read_cal

        loss_calibration = loss_valid_cal + loss_sq_cal + loss_seq_read_cal

        # Color Type CE strictly masked by (1 - is_monochrome_ir) so B&W IR NEVER hallucinates RGB!
        rgb_frame_mask = is_primary_frame & (is_monochrome_ir < 0.5)  # (B, T)
        color_losses: list[torch.Tensor] = []
        if rgb_frame_mask.any():
            frame_col_logits = model.decode_color_features(
                seq_out.frame_z_color.to(dtype=model.readout_slot_embed.dtype)
            ).float()  # (B, T, 5)
            exp_pt = target_plate_type.unsqueeze(1).expand(bsz, seq_len)
            color_losses.append(
                F.cross_entropy(frame_col_logits[rgb_frame_mask], exp_pt[rgb_frame_mask])
            )
        seq_has_rgb = non_mixed_seq & rgb_frame_mask.any(dim=1) & (seq_int_read.max(dim=1).values >= 0.20)
        if seq_has_rgb.any():
            color_losses.append(
                F.cross_entropy(
                    seq_out.plate_type_logits.float()[seq_has_rgb],
                    target_plate_type[seq_has_rgb],
                )
            )
        loss_color_type = (
            sum(color_losses) / len(color_losses) if color_losses else zero_anchor
        )

        # Polarity BCE (+1.0 bright-bg/dark-char -> 1.0, -1.0 dark-bg/bright-char -> 0.0)
        if polarity_target is not None:
            pol_tgt_bt = (
                polarity_target.unsqueeze(1).expand(bsz, seq_len)
                if polarity_target.ndim == 1
                else polarity_target
            )
            pol_y = (pol_tgt_bt.float() > 0.0).float()
            pol_mask = valid_frame_bool.float()
            if pol_mask.sum() > 0.5:
                pol_f_bce = F.binary_cross_entropy_with_logits(
                    seq_out.frame_pol_logit.float(), pol_y, reduction="none"
                )
                loss_pol_frame = (pol_f_bce * pol_mask).sum() / pol_mask.sum().clamp_min(1.0)
                seq_pol_mask = non_mixed_seq & valid_frame_bool.any(dim=1)
                if seq_pol_mask.any():
                    seq_pol_y = (
                        (pol_tgt_bt.float() * is_primary_frame.float()).sum(dim=1) >= 0.0
                    ).float()
                    loss_pol_seq = F.binary_cross_entropy_with_logits(
                        seq_out.polarity_logit.float()[seq_pol_mask],
                        seq_pol_y[seq_pol_mask],
                    )
                    loss_polarity = 0.5 * (loss_pol_frame + loss_pol_seq)
                else:
                    loss_polarity = loss_pol_frame
            else:
                loss_polarity = zero_anchor
        else:
            loss_polarity = zero_anchor

        # ------------------------------------------------------------------
        # 3. Anisotropic Gaussian NLL Uncertainty Calibration & Multi-View Consistency
        # ------------------------------------------------------------------
        if "canon_clean_crop" in batch and batch["canon_clean_crop"] is not None:
            with torch.no_grad(), _freeze_bn_running_stats(model):
                clean_sides = torch.zeros(bsz, dtype=torch.long, device=dev)
                clean_enc = model.encode_frames(batch["canon_clean_crop"], clean_sides, None)
                z_clean = clean_enc.z.float().detach()        # (B, 12, D)
                zc_clean = clean_enc.z_color.float().detach() # (B, Dc)
            has_clean_ref = True
        else:
            # Stop-gradient readability-weighted sequence mean fallback
            w_ref = primary_read.unsqueeze(-1)
            z_clean = (
                (seq_out.frame_z.float().detach() * w_ref).sum(dim=1)
                / w_ref.sum(dim=1).clamp_min(1e-5)
            )
            zc_clean = seq_out.frame_z_color.float().detach().mean(dim=1)
            has_clean_ref = False

        # Supervise `loss_unc_cal` across 100% of valid primary frames (including blurry `primary_read < 0.20`
        # low-res frames!) so sharp frames (`~20..24`) outweigh sub-Nyquist frames (`~0.3..0.5`) by >30:1
        # while coherent multi-frame low-res observations constructively integrate to high posterior SNR!
        unc_slot_w = is_primary_frame.unsqueeze(-1).float().expand(bsz, seq_len, 12)
        mv_slot_w = primary_read * (primary_read >= 0.20).float()  # (B, T, 12)
        z_pred = seq_out.frame_z.float()
        z_ref_exp = z_clean.unsqueeze(1)  # (B, 1, 12, D)

        if unc_slot_w.sum() > 1e-6:
            probs_aux_det = F.softmax(aux_logits.float().detach(), dim=-1)
            p_gt_det = probs_aux_det.gather(-1, safe_ft.unsqueeze(-1)).squeeze(-1)
            inacc_term = (
                (1.0 - p_gt_det).pow(2) * is_primary_frame.unsqueeze(-1).float()
                + 1.85 * (1.0 - primary_read).pow(2)
            ).unsqueeze(-1)
            err2_det = (
                torch.clamp((z_pred.detach() - z_ref_exp).pow(2), max=6.0)
                + 1.45 * inacc_term
                + (1.0 / 24.0)
            )
            lam_pos = torch.clamp(seq_out.frame_lambda_prec.float(), min=0.02, max=24.0)
            nll_elem = 0.5 * (lam_pos * err2_det - torch.log(lam_pos) + math.log(24.0) - 1.0)
            loss_unc_cal = (nll_elem.mean(dim=-1) * unc_slot_w).sum() / unc_slot_w.sum()

            # Sharp-Frame Dominance Margin Loss: enforce sharp frames (`read >= 0.58`) to command >=18x
            # higher write precision than sub-Nyquist blurry frames (`read <= 0.26`) on the same plate!
            sharp_f = is_primary_frame.unsqueeze(-1) & (primary_read >= 0.58)
            blur_f = is_primary_frame.unsqueeze(-1) & (primary_read <= 0.26)
            both_sb = sharp_f.any(dim=1) & blur_f.any(dim=1)  # (B, 12)
            if bool(both_sb.any()):
                log_lam_m = torch.log(lam_pos.mean(dim=-1).clamp_min(1e-4))  # (B, T, 12)
                log_hi = torch.where(sharp_f, log_lam_m, torch.full_like(log_lam_m, -20.0)).amax(dim=1)
                log_lo = torch.where(blur_f, log_lam_m, torch.full_like(log_lam_m, -20.0)).amax(dim=1)
                dom_margin = F.relu(math.log(18.0) - (log_hi[both_sb] - log_lo[both_sb])).mean()
                loss_unc_cal = loss_unc_cal + 0.35 * dom_margin
        else:
            loss_unc_cal = zero_anchor

        if mv_slot_w.sum() > 1e-6:
            l1_mv = F.smooth_l1_loss(z_pred, z_ref_exp.expand_as(z_pred), reduction="none", beta=0.15).mean(dim=-1)
            zn_p = z_pred / (torch.norm(z_pred, p=2, dim=-1, keepdim=True) + 1e-6)
            zn_r = z_ref_exp / (torch.norm(z_ref_exp, p=2, dim=-1, keepdim=True) + 1e-6)
            cos_mv = 1.0 - (zn_p * zn_r).sum(dim=-1)
            loss_mv_cons = ((l1_mv + 0.5 * cos_mv) * mv_slot_w).sum() / mv_slot_w.sum()

            if has_clean_ref and rgb_frame_mask.any():
                zc_pred = seq_out.frame_z_color.float()[rgb_frame_mask]
                zc_ref = zc_clean.unsqueeze(1).expand(bsz, seq_len, -1)[rgb_frame_mask]
                loss_mv_cons = loss_mv_cons + 0.25 * F.smooth_l1_loss(zc_pred, zc_ref, beta=0.15)
        else:
            loss_mv_cons = zero_anchor

        # ------------------------------------------------------------------
        # 4. Temporal Prefix Monotonicity Loss (L_mono)
        # ------------------------------------------------------------------
        if seq_out.prefix_slot_logits is not None and slot_w.sum() > 1e-6:
            pref_ce = F.cross_entropy(
                seq_out.prefix_slot_logits.float().detach().reshape(-1, self.vocab_size),
                target_slots.clamp(0, self.vocab_size - 1).reshape(-1),
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).reshape(bsz, 12)
            # Final sequence NLL must not degrade relative to early prefix NLL
            mono_viol = F.relu(slot_ce - pref_ce)
            loss_mono = (mono_viol * slot_w).sum() / slot_w.sum()
        else:
            loss_mono = zero_anchor

        # ------------------------------------------------------------------
        # 5. Counterfactual In-Batch Cross-Vehicle Splice Conflict & Garbage Invariance
        # ------------------------------------------------------------------
        max_conflict = seq_out.memory.mem_conflict.float().max(dim=-1).values  # (B,)
        is_mixed_seq = target_status == 2
        cf_terms: list[torch.Tensor] = []
        if (~is_mixed_seq).any():
            cf_terms.append(F.relu(max_conflict[~is_mixed_seq] - 0.18).mean())
        if is_mixed_seq.any():
            cf_terms.append(F.relu(1.55 - max_conflict[is_mixed_seq]).mean())

        # Counterfactual cross-vehicle single-slot Mahalanobis splice margin across batch
        if bsz >= 2:
            perm_b = torch.roll(torch.arange(bsz, device=dev), shifts=1)
            mem_P_f = seq_out.memory.mem_P.float()
            mu_self = seq_out.memory.mem_h.float() / (mem_P_f + 1e-5)  # (B, 12, D)

            # Pick most readable frame of paired vehicle perm_b
            best_t_other = primary_read[perm_b].mean(dim=-1).argmax(dim=-1)  # (B,)
            b_arange = torch.arange(bsz, device=dev)
            z_other = seq_out.frame_z.float()[perm_b, best_t_other]            # (B, 12, D)
            lam_other = seq_out.frame_lambda_prec.float()[perm_b, best_t_other]  # (B, 12, D)

            w_cf = (mem_P_f * lam_other) / (mem_P_f + lam_other + 1e-5)
            w_cf_cap = torch.clamp(w_cf, max=1.0)
            ov_cf = w_cf_cap.mean(dim=-1)  # (B, 12)
            rms_self = torch.sqrt(mu_self.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            rms_other = torch.sqrt(z_other.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            mu_hat_self = mu_self / rms_self
            z_hat_other = z_other / rms_other
            mahal2_cf = 0.5 * (w_cf_cap * (mu_hat_self - z_hat_other).pow(2)).mean(dim=-1)

            diff_slot = (
                (target_slots != target_slots[perm_b])
                & non_mixed_seq.unsqueeze(1)
                & non_mixed_seq[perm_b].unsqueeze(1)
                & (ov_cf > 0.25)
            )
            if diff_slot.any():
                cf_terms.append(F.relu(0.95 - mahal2_cf[diff_slot]).mean())

        loss_cf_conflict = sum(cf_terms) / len(cf_terms) if cf_terms else zero_anchor

        # Synthetic garbage-frame zero-drift invariance (freeze BN running stats so garbage never pollutes BN!)
        if compute_garbage_invariance:
            num_g = 2
            g_crops, g_sides, g_geom = synthesize_gpu_garbage_frames_v2(
                bsz, num_g, (h_img, w_img), dev
            )
            with _freeze_bn_running_stats(model):
                g_enc = model.encode_frames(
                    g_crops.reshape(bsz * num_g, 3, h_img, w_img),
                    g_sides.reshape(bsz * num_g),
                    g_geom.reshape(bsz * num_g, 10),
                )
            g_bce = F.binary_cross_entropy_with_logits(
                g_enc.frame_valid_logit.float(),
                torch.zeros_like(g_enc.frame_valid_logit.float()),
            )
            g_gate_pen = g_enc.frame_valid_gate.float().pow(2).mean()
            g_prec_pen = (
                g_enc.lambda_prec.float().mean()
                + 0.25 * g_enc.lambda_color.float().mean()
            )

            d_dim = model.config.d_model
            dc_dim = model.config.d_color
            gz = g_enc.z.reshape(bsz, num_g, 12, d_dim)
            glam = g_enc.lambda_prec.reshape(bsz, num_g, 12, d_dim)
            gzc = g_enc.z_color.reshape(bsz, num_g, dc_dim)
            glamc = g_enc.lambda_color.reshape(bsz, num_g, dc_dim)
            gpol = g_enc.pol_logit.reshape(bsz, num_g)
            gpolp = g_enc.pol_prec.reshape(bsz, num_g)

            st_g = seq_out.memory
            base_ts = torch.clamp(st_g.last_timestamp_s, min=0.0)
            for k in range(num_g):
                st_g = model.step_memory(
                    mem_h=st_g.mem_h,
                    mem_P=st_g.mem_P,
                    mem_conflict=st_g.mem_conflict,
                    color_h=st_g.color_h,
                    color_P=st_g.color_P,
                    pol_h=st_g.pol_h,
                    pol_P=st_g.pol_P,
                    pose_ema=st_g.pose_ema,
                    last_timestamp_s=st_g.last_timestamp_s,
                    z=gz[:, k],
                    lambda_prec=glam[:, k],
                    z_color=gzc[:, k],
                    lambda_color=glamc[:, k],
                    pol_logit=gpol[:, k],
                    pol_prec=gpolp[:, k],
                    geom_meta=g_geom[:, k],
                    timestamp_s=base_ts + 0.1 * float(k + 1),
                )
            drift_h = (st_g.mem_h.float() - seq_out.memory.mem_h.float().detach()).pow(2).mean()
            drift_p = (st_g.mem_P.float() - seq_out.memory.mem_P.float().detach()).pow(2).mean()
            drift_c = (st_g.color_h.float() - seq_out.memory.color_h.float().detach()).pow(2).mean()
            loss_garbage = g_bce + g_gate_pen + 0.50 * g_prec_pen + 2.0 * (drift_h + drift_p + drift_c)
        else:
            loss_garbage = zero_anchor

        # ------------------------------------------------------------------
        # 6. Training-Only Text Gestalt Stroke-Mask & Full-Res F1 SR Loss
        # ------------------------------------------------------------------
        sm_logits = getattr(seq_out, "stroke_mask_logits", None)
        sm_gt_raw = batch.get("stroke_mask_gt")
        if sm_logits is not None and sm_gt_raw is not None and is_primary_frame.any():
            sm_pred_p = sm_logits[is_primary_frame].float()  # (N_p, 1, H, W)
            sm_tgt_p = sm_gt_raw[is_primary_frame].float().clamp(0.0, 1.0)
            if sm_tgt_p.ndim == 3:
                sm_tgt_p = sm_tgt_p.unsqueeze(1)
            if sm_tgt_p.shape[-2:] != sm_pred_p.shape[-2:]:
                sm_tgt_p = F.interpolate(
                    sm_tgt_p, size=sm_pred_p.shape[-2:], mode="bilinear", align_corners=False
                )
            bce_sm = F.binary_cross_entropy_with_logits(sm_pred_p, sm_tgt_p)
            prob_sm = torch.sigmoid(sm_pred_p)
            inter_sm = (prob_sm * sm_tgt_p).sum(dim=(1, 2, 3))
            denom_sm = (prob_sm + sm_tgt_p).sum(dim=(1, 2, 3)) + 1.0
            dice_sm = (1.0 - (2.0 * inter_sm + 1.0) / denom_sm).mean()
            loss_stroke = 0.60 * bce_sm + 0.40 * dice_sm
        else:
            loss_stroke = zero_anchor

        f1_sr_logits = getattr(seq_out, "f1_sr_logits", None)
        if f1_sr_logits is not None and sm_gt_raw is not None and is_primary_frame.any():
            f1_pred_p = f1_sr_logits[is_primary_frame].float()
            f1_tgt_p = sm_gt_raw[is_primary_frame].float().clamp(0.0, 1.0)
            if f1_tgt_p.ndim == 3:
                f1_tgt_p = f1_tgt_p.unsqueeze(1)
            if f1_tgt_p.shape[-2:] != f1_pred_p.shape[-2:]:
                f1_tgt_p = F.interpolate(
                    f1_tgt_p, size=f1_pred_p.shape[-2:], mode="bilinear", align_corners=False
                )
            bce_f1 = F.binary_cross_entropy_with_logits(f1_pred_p, f1_tgt_p)
            prob_f1 = torch.sigmoid(f1_pred_p)
            inter_f1 = (prob_f1 * f1_tgt_p).sum(dim=(1, 2, 3))
            denom_f1 = (prob_f1 + f1_tgt_p).sum(dim=(1, 2, 3)) + 1.0
            dice_f1 = (1.0 - (2.0 * inter_f1 + 1.0) / denom_f1).mean()
            gx_f1, gy_f1 = self._sobel_xy_grads(prob_f1)
            gx_gt1, gy_gt1 = self._sobel_xy_grads(f1_tgt_p)
            edge_f1 = (gx_f1 - gx_gt1).abs().mean() + (gy_f1 - gy_gt1).abs().mean()
            loss_f1_sr = 0.50 * bce_f1 + 0.35 * dice_f1 + 0.45 * edge_f1
        else:
            loss_f1_sr = zero_anchor

        # ------------------------------------------------------------------
        # 6b. Training-Only Multi-Frame Memory Super-Resolution Loss (L1 + Sobel Gradient)
        #     Directly supervises fused posterior `mu_z = mem_h / mem_P` to reconstruct clean
        #     canonical 80x160 plate luminance + sharp Sobel stroke edges from low-res sequences!
        # ------------------------------------------------------------------
        mem_sr_pred = getattr(seq_out, "memory_sr_pred", None)
        canon_clean = batch.get("canon_clean_crop", None)
        if mem_sr_pred is not None and canon_clean is not None:
            sr_pred_f = mem_sr_pred.float()  # (B, 2, H_sr, W_sr)
            cc = canon_clean.float()
            if cc.shape[-2:] != sr_pred_f.shape[-2:]:
                cc = F.interpolate(cc, size=sr_pred_f.shape[-2:], mode="bilinear", align_corners=False)
            y_clean = (0.299 * cc[:, 0:1] + 0.587 * cc[:, 1:2] + 0.114 * cc[:, 2:3]).clamp(0.0, 1.0)
            gx_c, gy_c = self._sobel_xy_grads(y_clean)
            edge_c = torch.sqrt(gx_c * gx_c + gy_c * gy_c + 1e-6)

            y_hat = sr_pred_f[:, 0:1]
            edge_hat = sr_pred_f[:, 1:2]
            gx_p, gy_p = self._sobel_xy_grads(y_hat)

            sr_seq_mask = is_readable_seq & (seq_out.memory.mem_P.float().mean(dim=(1, 2)) > 1e-4)
            if bool(sr_seq_mask.any()):
                h_sr = sr_pred_f.shape[-2]
                row_y = torch.linspace(0.0, 1.0, h_sr, device=dev, dtype=torch.float32).view(1, 1, h_sr, 1)
                upper_band = ((row_y >= 0.08) & (row_y <= 0.44)).float()
                sp_w = 1.0 + 0.35 * upper_band + 0.25 * (edge_c > 0.10).float()

                geom_b = batch.get("geom_meta", None)
                if geom_b is not None:
                    # Upweight persistent low-res sequences (max projected plate width <= ~36px) by 1.5x
                    is_lr_seq = geom_b[:, :, 0].float().amax(dim=1) < -0.42
                    seq_w_sr = torch.where(is_lr_seq, 1.5, 1.0) * sr_seq_mask.float()
                else:
                    seq_w_sr = sr_seq_mask.float()

                lum_l1 = (y_hat - y_clean).abs()
                sobel_grad_l1 = (gx_p - gx_c).abs() + (gy_p - gy_c).abs()
                sobel_mag_l1 = (edge_hat - edge_c).abs()
                elem_sr = 1.0 * lum_l1 + 1.2 * sobel_grad_l1 + 0.6 * sobel_mag_l1
                tot_w_sr = sp_w * seq_w_sr.view(bsz, 1, 1, 1)
                loss_memory_sr = (elem_sr * tot_w_sr).sum() / tot_w_sr.sum().clamp_min(1e-6)
            else:
                loss_memory_sr = zero_anchor
        else:
            loss_memory_sr = zero_anchor

        # ------------------------------------------------------------------
        # 7. Statutory Length-Conditioned Slot Anchor Supervision (Region 1..4, Class 1..3)
        # ------------------------------------------------------------------
        reg_len_logits = getattr(seq_out, "reg_len_logits", None)
        cls_len_logits = getattr(seq_out, "cls_len_logits", None)
        if reg_len_logits is not None and cls_len_logits is not None:
            reg_cnt = (frame_slot_targets[:, :, 0:4] != self.pad_id).sum(dim=-1)  # (B, T)
            cls_cnt = (frame_slot_targets[:, :, 4:7] != self.pad_id).sum(dim=-1)  # (B, T)
            reg_read_max = eff_read[:, :, 0:4].max(dim=-1).values
            cls_read_max = eff_read[:, :, 4:7].max(dim=-1).values
            reg_mask = valid_frame_bool & (reg_cnt >= 1) & (reg_read_max >= 0.15)
            cls_mask = valid_frame_bool & (cls_cnt >= 1) & (cls_read_max >= 0.15)
            if bool(reg_mask.any()):
                reg_tgt = (reg_cnt[reg_mask] - 1).clamp(0, 3).long()
                reg_ce_vec = F.cross_entropy(
                    reg_len_logits[reg_mask].float(), reg_tgt, reduction="none"
                )
                rw = reg_read_max[reg_mask].clamp_min(0.20)
                loss_reg_len = (reg_ce_vec * rw).sum() / rw.sum().clamp_min(1e-6)
            else:
                loss_reg_len = zero_anchor
            if bool(cls_mask.any()):
                cls_tgt = (cls_cnt[cls_mask] - 1).clamp(0, 2).long()
                cls_ce_vec = F.cross_entropy(
                    cls_len_logits[cls_mask].float(), cls_tgt, reduction="none"
                )
                cw = cls_read_max[cls_mask].clamp_min(0.20)
                loss_cls_len = (cls_ce_vec * cw).sum() / cw.sum().clamp_min(1e-6)
            else:
                loss_cls_len = zero_anchor
            loss_statutory_len = 0.5 * (loss_reg_len + loss_cls_len)
        else:
            loss_statutory_len = zero_anchor

        # Learn nuisance dynamics from the actual correlated observation model.
        loss_innovation = seq_out.innovation_nll if seq_out.innovation_nll is not None else zero_anchor
        loss_risk = zero_anchor
        for risk, readout, memory, count in (
            (seq_out.prefix_risk, seq_out.prefix_readout, seq_out.prefix_memory, seq_out.prefix_len),
            (seq_out.prefix_next_risk, seq_out.next_prefix_readout, seq_out.next_prefix_memory, seq_out.prefix_len+1)):
            if risk is not None and readout is not None and memory is not None:
                target_risk = offline_terminal_risk_targets(readout, memory, target_slots,
                    target_status, count, self.risk_vocab, self.risk_config, self.risk_grammar)
                loss_risk = loss_risk + F.binary_cross_entropy(risk.float().clamp(1e-6,1-1e-6), target_risk)

        total_loss = (
            self.weights.seq_slots * loss_seq_slots
            + 0.30 * loss_vault_pad
            + self.weights.aux_frames * loss_aux_frames
            + self.weights.status * loss_status
            + self.weights.calibration * loss_calibration
            + self.weights.color_type * loss_color_type
            + self.weights.polarity * loss_polarity
            + self.weights.unc_cal * loss_unc_cal
            + self.weights.mv_cons * loss_mv_cons
            + self.weights.mono * loss_mono
            + self.weights.cf_conflict * loss_cf_conflict
            + self.weights.garbage_invariance * loss_garbage
            + self.weights.stroke_gestalt * loss_stroke
            + self.weights.f1_sr * loss_f1_sr
            + self.weights.memory_sr * loss_memory_sr
            + self.weights.statutory_len * loss_statutory_len
            + self.weights.innovation_nll * loss_innovation
            + self.weights.terminal_risk * loss_risk
        )

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_innovation_nll": float(loss_innovation.detach().item()),
            "loss_terminal_risk": float(loss_risk.detach().item()),
            "loss_seq_slots": float(loss_seq_slots.detach().item()),
            "loss_vault_pad": float(loss_vault_pad.detach().item()),
            "loss_aux_frames": float(loss_aux_frames.detach().item()),
            "loss_status": float(loss_status.detach().item()),
            "loss_calibration": float(loss_calibration.detach().item()),
            "loss_color_type": float(loss_color_type.detach().item()),
            "loss_polarity": float(loss_polarity.detach().item()),
            "loss_unc_cal": float(loss_unc_cal.detach().item()),
            "loss_mv_cons": float(loss_mv_cons.detach().item()),
            "loss_mono": float(loss_mono.detach().item()),
            "loss_cf_conflict": float(loss_cf_conflict.detach().item()),
            "loss_garbage": float(loss_garbage.detach().item()),
            "loss_stroke": float(loss_stroke.detach().item()),
            "loss_f1_sr": float(loss_f1_sr.detach().item()),
            "loss_memory_sr": float(loss_memory_sr.detach().item()),
            "loss_statutory_len": float(loss_statutory_len.detach().item()),
        }
        return total_loss, metrics
