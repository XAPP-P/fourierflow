"""
Scaled-down 2D Navier-Stokes data generator on a periodic domain (torus).

Generates viscosity-fixed NS trajectories using a pseudo-spectral solver.
This mirrors the "TorusLi" setup from Tran et al. 2023 (Section 5.1), but
at a resolution and dataset size that fits in Colab T4 memory and runs in
under ~15 minutes.

Governing equation (vorticity form):
    d w / dt + (u . grad) w = nu * laplacian(w) + f(x, y)
where w is vorticity, u is velocity (divergence-free, recovered from w via
stream function), nu is kinematic viscosity, and f is a fixed forcing term.

Numerical method:
    - Spatial: FFT-based spectral differentiation on a uniform grid
    - Temporal: Crank-Nicolson for diffusion (implicit), Heun's method for
      the nonlinear advection (explicit, 2nd-order)
    - Dealiasing: 2/3 rule (zero out top 1/3 of Fourier modes)

Default settings produce 1000 trajectories at 64x64 with 20 time snapshots
each, which trains in ~30-60 min on a T4 for the architectures used here.
"""

import math
import os
from pathlib import Path

import numpy as np
import torch


def _initial_vorticity(batch: int, N: int, tau: float = 7.0, alpha: float = 2.5, device="cpu") -> torch.Tensor:
    """
    Sample random smooth initial vorticity fields with a Gaussian random-field
    prior (Matern-like). Same distribution used by Li et al. 2021a.

    Returns: (batch, N, N) real tensor
    """
    # Wavenumbers
    k1 = torch.fft.fftfreq(N, d=1.0 / N, device=device)
    k2 = torch.fft.fftfreq(N, d=1.0 / N, device=device)
    K1, K2 = torch.meshgrid(k1, k2, indexing="ij")
    K_squared = K1 ** 2 + K2 ** 2

    # Spectral prior: coefficients ~ N(0, 1) * (|k|^2 + tau^2)^(-alpha/2)
    coef = tau ** (alpha - 1) * (math.pi * (K_squared + tau ** 2)) ** (-alpha / 2.0)

    # Complex Gaussian noise in Fourier space
    noise = torch.randn(batch, N, N, device=device, dtype=torch.cfloat)
    noise.imag = torch.randn(batch, N, N, device=device)
    noise.real = torch.randn(batch, N, N, device=device)

    w_hat = coef.unsqueeze(0) * noise
    # Zero out the mean (k=0 mode) so the field has zero average vorticity
    w_hat[:, 0, 0] = 0.0

    w = torch.fft.ifft2(w_hat).real * (N ** 2)
    return w


def _build_forcing(N: int, device="cpu") -> torch.Tensor:
    """
    Fixed forcing term: f(x, y) = 0.1 * (sin(2 pi (x + y)) + cos(2 pi (x + y)))
    Same as Li et al. 2021a. Domain is [0, 1)^2.
    """
    grid = torch.linspace(0, 1, N + 1, device=device)[:-1]
    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    return 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))


def simulate_navier_stokes(
    batch: int = 32,
    N: int = 64,
    T: int = 20,
    dt: float = 1e-3,
    record_every: int = 100,
    viscosity: float = 1e-3,
    device: str = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    """
    Simulate 2D incompressible Navier-Stokes on the periodic torus [0,1]^2.

    Args:
        batch: number of independent trajectories to simulate in parallel
        N: grid resolution (N x N)
        T: number of recorded snapshots per trajectory
        dt: solver timestep (should be << record interval)
        record_every: record a snapshot every `record_every` solver steps
                      -> snapshot interval = dt * record_every seconds
        viscosity: kinematic viscosity nu
        device: "cpu" or "cuda"
        seed: for reproducibility

    Returns:
        Tensor of shape (batch, T, N, N) containing vorticity snapshots.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device(device)
    w = _initial_vorticity(batch, N, device=device)
    f = _build_forcing(N, device=device)

    # Precompute wavenumbers
    k1 = 2 * math.pi * torch.fft.fftfreq(N, d=1.0 / N, device=device)
    k2 = 2 * math.pi * torch.fft.fftfreq(N, d=1.0 / N, device=device)
    K1, K2 = torch.meshgrid(k1, k2, indexing="ij")
    K_sq = K1 ** 2 + K2 ** 2
    K_sq_safe = K_sq.clone()
    K_sq_safe[0, 0] = 1.0  # avoid division by zero for the k=0 mode

    # Dealiasing mask (2/3 rule)
    kmax = N // 3
    dealias = ((k1.abs() <= kmax).unsqueeze(-1) & (k2.abs() <= kmax).unsqueeze(0)).to(torch.float)

    f_hat = torch.fft.fft2(f)

    # Crank-Nicolson factor for linear (diffusion) part:
    # (1 - 0.5*dt*nu*(-K^2)) w_{n+1} = (1 + 0.5*dt*nu*(-K^2)) w_n + dt*NL
    # Rearranged for convenience:
    cn_num = 1.0 - 0.5 * dt * viscosity * K_sq
    cn_den = 1.0 + 0.5 * dt * viscosity * K_sq

    def nonlinear_term(w_hat_in: torch.Tensor) -> torch.Tensor:
        """Compute -(u . grad) w in Fourier space via pseudo-spectral method."""
        # Stream function: psi_hat = w_hat / K^2
        psi_hat = w_hat_in / K_sq_safe
        psi_hat[..., 0, 0] = 0.0
        # Velocity: u = (d psi / dy, -d psi / dx)
        u_hat = 1j * K2 * psi_hat
        v_hat = -1j * K1 * psi_hat
        # Vorticity gradient
        dwdx_hat = 1j * K1 * w_hat_in
        dwdy_hat = 1j * K2 * w_hat_in
        # To physical space
        u = torch.fft.ifft2(u_hat).real
        v = torch.fft.ifft2(v_hat).real
        dwdx = torch.fft.ifft2(dwdx_hat).real
        dwdy = torch.fft.ifft2(dwdy_hat).real
        # Convective term in physical space, then back to Fourier
        conv = u * dwdx + v * dwdy
        conv_hat = torch.fft.fft2(conv) * dealias  # dealias
        return -conv_hat + f_hat  # include forcing

    snapshots = torch.zeros(batch, T, N, N, device=device)

    w_hat = torch.fft.fft2(w)
    step = 0
    snap_idx = 0
    total_steps = T * record_every

    for step in range(total_steps):
        # Record snapshot first so t=0 is included
        if step % record_every == 0 and snap_idx < T:
            snapshots[:, snap_idx] = torch.fft.ifft2(w_hat).real
            snap_idx += 1

        # Heun's method (RK2) for nonlinear part + CN for linear part
        nl1 = nonlinear_term(w_hat)
        w_hat_star = (cn_num * w_hat + dt * nl1) / cn_den
        nl2 = nonlinear_term(w_hat_star)
        w_hat = (cn_num * w_hat + 0.5 * dt * (nl1 + nl2)) / cn_den

    return snapshots


def generate_and_save(
    out_dir: str,
    n_train: int = 800,
    n_val: int = 100,
    n_test: int = 100,
    N: int = 64,
    T: int = 20,
    dt: float = 1e-3,
    record_every: int = 100,
    viscosity: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """
    Generate train/val/test splits and save as .pt files.

    Returns the dict of metadata saved alongside the data.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = {"train": n_train, "val": n_val, "test": n_test}
    seeds = {"train": seed, "val": seed + 1, "test": seed + 2}

    for split, n in splits.items():
        print(f"Generating {split} set ({n} trajectories)...")
        all_traj = []
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)
            traj = simulate_navier_stokes(
                batch=b,
                N=N,
                T=T,
                dt=dt,
                record_every=record_every,
                viscosity=viscosity,
                device=device,
                seed=seeds[split] + start,
            )
            all_traj.append(traj.cpu())
            print(f"  {start + b}/{n} done")
        data = torch.cat(all_traj, dim=0)
        torch.save(data, out_path / f"{split}.pt")
        print(f"  saved {split}.pt with shape {tuple(data.shape)}")

    meta = dict(
        N=N,
        T=T,
        dt=dt,
        record_every=record_every,
        snapshot_interval=dt * record_every,
        viscosity=viscosity,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )
    torch.save(meta, out_path / "metadata.pt")
    return meta


if __name__ == "__main__":
    # Smoke test: generate a tiny dataset locally
    print("Running smoke test: 4 trajectories at 32x32, 10 steps...")
    traj = simulate_navier_stokes(batch=4, N=32, T=10, seed=0)
    print(f"shape: {tuple(traj.shape)}")
    print(f"vorticity range: [{traj.min():.3f}, {traj.max():.3f}]")
    print(f"mean absolute vorticity over time: {traj.abs().mean(dim=(0, 2, 3)).tolist()}")
