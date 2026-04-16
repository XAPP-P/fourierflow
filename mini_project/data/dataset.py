"""
Dataset utilities for neural operator training.

The core abstraction is simple: given a saved tensor of vorticity trajectories
of shape (N_traj, T, H, W), we build a dataset of (input, target) pairs where
input is the vorticity at timestep t and target is the vorticity at timestep
t+1. Two spatial coordinate channels are concatenated to the input, since the
F-FNO paper shows "double encoding" of coordinates helps (Fig. 5b).
"""

import math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class NavierStokesDataset(Dataset):
    """
    Markov-style dataset: each sample is a (w_t, w_{t+1}) pair drawn from the
    stored trajectories. Spatial coordinates (x, y) are appended as extra
    input channels.

    Args:
        data_path: path to a .pt file containing a (N, T, H, W) tensor
        input_noise_std: std of Gaussian noise added to input (training only).
                         Set to 0 to disable.
        normalize: whether to normalize vorticity to zero mean / unit std
                   using train-set statistics.
        stats: optional dict with 'mean' and 'std' for normalization (required
               for val/test so they use train-set stats).
    """

    def __init__(
        self,
        data_path: str | Path,
        input_noise_std: float = 0.0,
        normalize: bool = True,
        stats: dict | None = None,
    ):
        self.data = torch.load(data_path, weights_only=True)
        assert self.data.ndim == 4, f"Expected (N, T, H, W), got {self.data.shape}"
        self.N_traj, self.T, self.H, self.W = self.data.shape
        self.input_noise_std = input_noise_std
        self.normalize = normalize

        if normalize:
            if stats is None:
                self.mean = self.data.mean().item()
                self.std = self.data.std().item()
            else:
                self.mean = stats["mean"]
                self.std = stats["std"]
        else:
            self.mean, self.std = 0.0, 1.0

        # Precompute coordinate grid — same for every sample
        x = torch.linspace(0, 1, self.H + 1)[:-1]
        y = torch.linspace(0, 1, self.W + 1)[:-1]
        X, Y = torch.meshgrid(x, y, indexing="ij")
        self.coords = torch.stack([X, Y], dim=0)  # (2, H, W)

        # Each trajectory contributes (T - 1) adjacent pairs
        self.pairs_per_traj = self.T - 1

    def __len__(self) -> int:
        return self.N_traj * self.pairs_per_traj

    def get_stats(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def __getitem__(self, idx: int):
        traj_idx = idx // self.pairs_per_traj
        t = idx % self.pairs_per_traj

        w_t = self.data[traj_idx, t]
        w_next = self.data[traj_idx, t + 1]

        if self.normalize:
            w_t = (w_t - self.mean) / self.std
            w_next = (w_next - self.mean) / self.std

        # Add Gaussian noise to input only during training (F-FNO trick)
        if self.input_noise_std > 0:
            w_t = w_t + self.input_noise_std * torch.randn_like(w_t)

        # Concatenate coordinates: input is (vorticity, x, y) -> 3 channels
        x_in = torch.cat([w_t.unsqueeze(0), self.coords], dim=0)
        y_out = w_next.unsqueeze(0)  # (1, H, W)
        return x_in, y_out


class RolloutDataset(Dataset):
    """
    For evaluation only: returns entire trajectories so we can assess how
    well the operator extrapolates by autoregressive rollout.

    Each item is a full (T, H, W) trajectory plus the coordinate channels.
    """

    def __init__(
        self,
        data_path: str | Path,
        normalize: bool = True,
        stats: dict | None = None,
    ):
        self.data = torch.load(data_path, weights_only=True)
        self.N_traj, self.T, self.H, self.W = self.data.shape
        self.normalize = normalize
        if stats is not None:
            self.mean, self.std = stats["mean"], stats["std"]
        else:
            self.mean, self.std = self.data.mean().item(), self.data.std().item()

        x = torch.linspace(0, 1, self.H + 1)[:-1]
        y = torch.linspace(0, 1, self.W + 1)[:-1]
        X, Y = torch.meshgrid(x, y, indexing="ij")
        self.coords = torch.stack([X, Y], dim=0)

    def __len__(self) -> int:
        return self.N_traj

    def __getitem__(self, idx: int):
        traj = self.data[idx]  # (T, H, W)
        if self.normalize:
            traj = (traj - self.mean) / self.std
        return traj, self.coords


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 16,
    input_noise_std: float = 0.0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Convenience function: build train/val/test DataLoaders with consistent
    normalization statistics.
    """
    data_dir = Path(data_dir)
    train_ds = NavierStokesDataset(data_dir / "train.pt", input_noise_std=input_noise_std)
    stats = train_ds.get_stats()
    val_ds = NavierStokesDataset(data_dir / "val.pt", input_noise_std=0.0, stats=stats)
    test_ds = NavierStokesDataset(data_dir / "test.pt", input_noise_std=0.0, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, stats
