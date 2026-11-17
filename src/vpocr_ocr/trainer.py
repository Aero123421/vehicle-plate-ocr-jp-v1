"""Two-stage GPU bfloat16 AMP trainer (Stage 1 single-frame + Stage 2 multi-frame)."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from vpocr_ocr.dataset import collate_plate_sequences
from vpocr_ocr.losses import LossWeights, MFPlateLoss
from vpocr_ocr.model import MFJPPlateOCR, ModelConfig
from vpocr_ocr.vocab import PlateVocab


@dataclass
class TrainConfig:
    """Training hyperparameters for Stage 1 and Stage 2 optimization."""

    batch_size: int = 32
    stage1_steps: int = 300
    stage2_steps: int = 500
    lr_stage1: float = 1.5e-3
    lr_stage2: float = 8.0e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    num_workers: int = 0
    log_every: int = 50
    seed: int = 42


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _make_stage1_single_frame_batch(
    batch: dict[str, Any], max_frames: int = 96
) -> dict[str, Any]:
    """Extract individual non-padded frames as T=1 sequences for rapid Stage 1 pretraining."""
    crops = batch["crops"]  # (B, T, 3, 64, 128)
    bsz, seq_len = crops.shape[:2]
    if seq_len == 1:
        return batch

    pad_mask = batch["frame_pad_mask"]          # (B, T)
    is_foreign = batch["is_foreign_plate"]      # (B, T)
    valid_mask = batch["frame_valid_mask"]      # (B, T)
    readability = batch["slot_readability"]     # (B, T, 12)
    sides = batch["plate_sides"]                # (B, T)

    b_idx, t_idx = torch.nonzero((~pad_mask) & (~is_foreign), as_tuple=True)
    if b_idx.numel() == 0:
        return batch
    if b_idx.numel() > max_frames:
        perm = torch.randperm(b_idx.numel(), device=crops.device)[:max_frames]
        b_idx = b_idx[perm]
        t_idx = t_idx[perm]

    n = b_idx.numel()
    c_1 = crops[b_idx, t_idx].unsqueeze(1)
    r_1 = readability[b_idx, t_idx].unsqueeze(1)
    v_1 = valid_mask[b_idx, t_idx].unsqueeze(1)
    f_1 = torch.zeros((n, 1), dtype=torch.bool, device=crops.device)
    s_1 = sides[b_idx, t_idx].unsqueeze(1)

    t_slots = batch["target_slots"][b_idx].clone()

    # Single-frame status: READABLE if frame is valid and all active slots >= 0.30 else UNREADABLE
    is_active = t_slots != 0
    active_r = torch.where(is_active, r_1.squeeze(1), torch.ones_like(r_1.squeeze(1)))
    min_r = active_r.min(dim=-1).values
    st_1 = torch.where((v_1.squeeze(1) > 0.5) & (min_r >= 0.30), 0, 1).long()

    out: dict[str, Any] = {
        "crops": c_1,
        "timestamps_s": torch.zeros((n, 1), dtype=torch.float32, device=crops.device),
        "source_frame_ids": torch.zeros((n, 1), dtype=torch.long, device=crops.device),
        "camera_ids": torch.zeros((n, 1), dtype=torch.long, device=crops.device),
        "plate_sides": s_1,
        "frame_valid_mask": v_1,
        "is_foreign_plate": f_1,
        "slot_readability": r_1,
        "frame_pad_mask": torch.zeros((n, 1), dtype=torch.bool, device=crops.device),
        "target_slots": t_slots,
        "target_plate_type": batch["target_plate_type"][b_idx],
        "target_status": st_1,
        "has_target_plate": batch["has_target_plate"][b_idx] | (v_1.squeeze(1) > 0.5),
    }
    return out


def save_checkpoint(
    path: str | Path,
    model: MFJPPlateOCR,
    train_config: TrainConfig | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save weights-only-clean PyTorch checkpoint with ModelConfig metadata."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "train_config": asdict(train_config) if train_config is not None else {},
        "metrics": metrics or {},
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[MFJPPlateOCR, PlateVocab, dict[str, Any]]:
    """Load MFJPPlateOCR model and PlateVocab from saved checkpoint."""
    vocab = PlateVocab()
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    cfg_dict = ckpt.get("model_config", {"vocab_size": vocab.size})
    model_cfg = ModelConfig(**cfg_dict)
    model = MFJPPlateOCR(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, ckpt.get("metrics", {})


class OCRTrainer:
    """End-to-end Stage 1 (single-frame) + Stage 2 (multi-frame) GPU AMP trainer."""

    def __init__(
        self,
        model: MFJPPlateOCR,
        vocab: PlateVocab | None = None,
        config: TrainConfig | None = None,
        loss_weights: LossWeights | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.vocab = vocab or PlateVocab()
        self.config = config or TrainConfig()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = MFPlateLoss(
            vocab_size=self.vocab.size,
            pad_id=self.vocab.pad_id,
            weights=loss_weights,
        ).to(self.device)

    def _autocast_ctx(self) -> Any:
        if self.config.use_amp and self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _run_stage(
        self,
        dataset: Dataset[dict[str, Any]],
        num_steps: int,
        max_lr: float,
        is_stage1: bool,
    ) -> list[dict[str, float]]:
        if num_steps <= 0:
            return []

        self.model.train()
        loader = DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, max(1, len(dataset))),  # type: ignore[arg-type]
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_plate_sequences,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(self.config.num_workers > 0),
            multiprocessing_context=("forkserver" if self.config.num_workers > 0 else None),
            drop_last=False,
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=max_lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max(2, num_steps),
            pct_start=0.15 if num_steps >= 20 else 0.3,
            anneal_strategy="cos",
        )

        history: list[dict[str, float]] = []
        step = 0
        while step < num_steps:
            for raw_batch in loader:
                if step >= num_steps:
                    break
                batch = _move_batch_to_device(raw_batch, self.device)
                if is_stage1:
                    batch = _make_stage1_single_frame_batch(batch)

                optimizer.zero_grad(set_to_none=True)
                with self._autocast_ctx():
                    seq_out = self.model.forward_sequence(
                        crops=batch["crops"],
                        timestamps_s=batch["timestamps_s"],
                        plate_sides=batch["plate_sides"],
                        frame_pad_mask=batch["frame_pad_mask"],
                        camera_ids=batch["camera_ids"],
                        source_frame_ids=batch["source_frame_ids"],
                    )
                    loss, metrics = self.criterion(
                        self.model,
                        seq_out,
                        batch,
                        compute_garbage_invariance=True,
                    )

                loss.backward()
                if self.config.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                optimizer.step()
                if step + 1 < max(2, num_steps):
                    scheduler.step()

                metrics["step"] = float(step + 1)
                metrics["lr"] = float(optimizer.param_groups[0]["lr"])
                history.append(metrics)
                step += 1
                if self.config.log_every > 0 and (
                    step % self.config.log_every == 0 or step == num_steps
                ):
                    st_label = "S1" if is_stage1 else "S2"
                    print(
                        f"    [{st_label} {step}/{num_steps}] "
                        f"loss={metrics['loss_total']:.4f} "
                        f"(slots={metrics['loss_seq_slots']:.4f}, aux={metrics['loss_aux_frames']:.4f}, "
                        f"cal={metrics['loss_calibration']:.4f}, st={metrics['loss_status']:.4f}) "
                        f"lr={metrics['lr']:.2e}",
                        flush=True,
                    )

        return history

    def fit(
        self,
        train_dataset: Dataset[dict[str, Any]],
        stage1_dataset: Dataset[dict[str, Any]] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run Stage 1 single-frame pretraining followed by Stage 2 multi-frame training."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        s1_ds = stage1_dataset if stage1_dataset is not None else train_dataset

        stage1_hist = self._run_stage(
            dataset=s1_ds,
            num_steps=self.config.stage1_steps,
            max_lr=self.config.lr_stage1,
            is_stage1=True,
        )
        stage2_hist = self._run_stage(
            dataset=train_dataset,
            num_steps=self.config.stage2_steps,
            max_lr=self.config.lr_stage2,
            is_stage1=False,
        )

        self.model.eval()
        summary: dict[str, Any] = {
            "stage1_steps": len(stage1_hist),
            "stage2_steps": len(stage2_hist),
            "stage1_final_loss": stage1_hist[-1]["loss_total"] if stage1_hist else None,
            "stage2_final_loss": stage2_hist[-1]["loss_total"] if stage2_hist else None,
            "stage1_history": stage1_hist,
            "stage2_history": stage2_hist,
        }
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                self.model,
                train_config=self.config,
                metrics={
                    "stage1_final_loss": summary["stage1_final_loss"],
                    "stage2_final_loss": summary["stage2_final_loss"],
                },
            )
        return summary
