"""
Unified training pipeline for FNO and F-FNO.

All paper training tricks are controllable via flags in TrainConfig:
  - Markov assumption (always on — we train on single-step pairs)
  - Teacher forcing (always on — we use ground truth w_t, not rollout)
  - Gaussian noise injection on input
  - Cosine learning rate decay
  - Input normalization
  - Residual connections (baked into FFNO architecture; FNO has none)

This design lets the ablation study in notebook 03 toggle tricks with a
single argument change.

Loss: normalized MSE (N-MSE), matching Tran et al. 2023 Section 5:
    N-MSE = (1/B) sum_i ||w_pred_i - w_true_i||_2 / ||w_true_i||_2
"""

from dataclasses import dataclass, field
from pathlib import Path
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import FNO2d, FFNO2d
from data.dataset import build_loaders, RolloutDataset


@dataclass
class TrainConfig:
    # Model
    model_type: str = "ffno"              # "fno" or "ffno"
    hidden: int = 32
    modes: int = 12
    n_layers: int = 4
    weight_sharing: bool = False          # only used for ffno
    in_channels: int = 3                  # vorticity + (x, y) coords
    out_channels: int = 1

    # Data
    data_dir: str = "./ns_data"
    batch_size: int = 16

    # Optimization
    n_epochs: int = 50
    lr: float = 2.5e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    use_cosine_decay: bool = True

    # Tricks (for ablation)
    input_noise_std: float = 1e-3         # Gaussian noise on input (0 to disable)

    # Misc
    device: str = "cuda"
    seed: int = 0
    log_every: int = 50
    out_dir: str = "./runs/default"
    # If set, stop training after this many seconds (useful for Colab time limits)
    max_train_seconds: float | None = None


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model_type == "fno":
        return FNO2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            hidden=cfg.hidden,
            modes1=cfg.modes,
            modes2=cfg.modes,
            n_layers=cfg.n_layers,
        )
    elif cfg.model_type == "ffno":
        return FFNO2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            hidden=cfg.hidden,
            modes_x=cfg.modes,
            modes_y=cfg.modes,
            n_layers=cfg.n_layers,
            weight_sharing=cfg.weight_sharing,
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    N-MSE loss from Tran et al. 2023, Section 5:
        (1/B) sum_i ||pred_i - target_i||_2 / ||target_i||_2
    Computed per-sample then averaged. Norms are over all non-batch dims.
    """
    flat_pred = pred.reshape(pred.size(0), -1)
    flat_target = target.reshape(target.size(0), -1)
    num = torch.linalg.norm(flat_pred - flat_target, dim=1)
    den = torch.linalg.norm(flat_target, dim=1).clamp(min=1e-12)
    return (num / den).mean()


def cosine_warmup_lr(step: int, warmup: int, total: int, peak_lr: float, min_lr: float = 1e-6) -> float:
    """
    Linear warmup for `warmup` steps up to `peak_lr`, then cosine decay to
    `min_lr` over the remaining steps. Matches the schedule in the F-FNO paper.
    """
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = normalized_mse(pred, y)
        total += loss.item() * x.size(0)
        count += x.size(0)
    model.train()
    return total / count


@torch.no_grad()
def evaluate_rollout(
    model: nn.Module,
    dataset: RolloutDataset,
    device: str,
    n_steps: int | None = None,
) -> dict:
    """
    Evaluate autoregressive rollout error. Uses the model to predict w_1 from
    w_0, then w_2 from the predicted w_1, etc. Returns per-step N-MSE averaged
    across trajectories.
    """
    model.eval()
    T = dataset.T if n_steps is None else min(n_steps + 1, dataset.T)
    errors_per_step = []

    for i in range(len(dataset)):
        traj, coords = dataset[i]
        traj = traj.to(device)
        coords = coords.to(device)

        w = traj[0].unsqueeze(0)  # (1, H, W), start from ground truth
        per_step = []
        for t in range(1, T):
            x_in = torch.cat([w, coords.unsqueeze(0)], dim=1)  # (1, 3, H, W)
            w_pred = model(x_in).squeeze(1)  # (1, H, W)
            true = traj[t].unsqueeze(0)
            err = torch.linalg.norm(w_pred.flatten() - true.flatten()) / (
                torch.linalg.norm(true.flatten()) + 1e-12
            )
            per_step.append(err.item())
            w = w_pred  # feed prediction back in (no teacher forcing at eval)
        errors_per_step.append(per_step)

    errors = torch.tensor(errors_per_step)  # (N_traj, T-1)
    return {
        "per_step_mean": errors.mean(dim=0).tolist(),
        "per_step_std": errors.std(dim=0).tolist(),
        "final_step_mean": errors[:, -1].mean().item(),
    }


def train(cfg: TrainConfig) -> dict:
    """
    Train a model per the given config. Returns a dict with training history
    and final metrics. Saves best-val checkpoint to cfg.out_dir.
    """
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader, stats = build_loaders(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        input_noise_std=cfg.input_noise_std,
    )

    # Model + optimizer
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{cfg.model_type}] params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = cfg.n_epochs * steps_per_epoch

    history = {
        "step": [], "train_loss": [], "val_loss": [], "lr": [],
        "epoch_time": [], "n_params": n_params,
    }
    best_val = float("inf")
    start_time = time.time()
    step = 0
    stopped_early = False

    for epoch in range(cfg.n_epochs):
        epoch_start = time.time()
        epoch_losses = []

        for x, y in train_loader:
            # Update learning rate
            if cfg.use_cosine_decay:
                lr = cosine_warmup_lr(step, cfg.warmup_steps, total_steps, cfg.lr)
            else:
                lr = cfg.lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = normalized_mse(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            if step % cfg.log_every == 0:
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["lr"].append(lr)
            step += 1

            if cfg.max_train_seconds is not None and time.time() - start_time > cfg.max_train_seconds:
                stopped_early = True
                break

        if stopped_early:
            break

        val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        history["epoch_time"].append(epoch_time)

        # Keep val history aligned with step counter
        history["step"].append(step)
        history["train_loss"].append(sum(epoch_losses) / len(epoch_losses))
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        print(
            f"epoch {epoch + 1:3d}/{cfg.n_epochs} | "
            f"train {sum(epoch_losses) / len(epoch_losses):.4f} | "
            f"val {val_loss:.4f} | "
            f"lr {lr:.2e} | "
            f"{epoch_time:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")

    # Final test eval using best checkpoint
    if (out_dir / "best.pt").exists():
        model.load_state_dict(torch.load(out_dir / "best.pt", weights_only=True))
    test_loss = evaluate(model, test_loader, device)

    total_time = time.time() - start_time
    result = {
        "history": history,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "total_time_sec": total_time,
        "n_params": n_params,
        "stopped_early": stopped_early,
        "config": cfg.__dict__,
    }
    torch.save(result, out_dir / "result.pt")
    print(f"done. test N-MSE={test_loss:.4f} | total {total_time:.1f}s")
    return result
