"""
Simplified Fourier Neural Operator (FNO) for 2D problems.

Reference: Li et al. (2021a) "Fourier Neural Operator for Parametric Partial
Differential Equations." https://arxiv.org/abs/2010.08895

This is a faithful but scaled-down re-implementation used as a baseline for
comparison against the F-FNO (Tran et al., 2023). The architecture follows
Eq. (5) of the F-FNO paper:

    L^(l)(z^(l)) = sigma( W^(l) z^(l) + b^(l) + K^(l)(z^(l)) )
    K^(l)(z)     = IFFT( R^(l) . FFT(z) )

where R^(l) is a complex-valued weight matrix over Fourier modes of shape
(hidden, hidden, M, M). Parameter count is O(L * H^2 * M^2) for 2D problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D spectral convolution layer — the core of the original FNO.

    Keeps the lowest `modes1 x modes2` Fourier modes and learns a separate
    complex-valued weight for each (mode_x, mode_y, channel_in, channel_out)
    combination. This is the O(H^2 * M^D) parameter block the F-FNO paper
    targets for factorization.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Fourier modes in x direction (kept lowest)
        self.modes2 = modes2  # Fourier modes in y direction (kept lowest)

        # Scaling factor for initialization — matches Li et al.'s reference impl.
        scale = 1.0 / (in_channels * out_channels)

        # Two separate weight tensors: one for the (+,+) quadrant of modes
        # and one for (-,+). The other two quadrants are conjugate-symmetric
        # because the input is real, so FFT only returns half the spectrum.
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_c, x, y) x (in_c, out_c, x, y) -> (batch, out_c, x, y)
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W) real
        batch = x.size(0)

        # Forward FFT — rfft2 only returns half the spectrum along the last dim
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Multiply relevant Fourier modes, zero out the rest
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Lower modes along x (positive frequencies)
        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        # Upper modes along x (negative frequencies, mirrored)
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Inverse FFT back to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x


class FNOBlock2d(nn.Module):
    """
    A single FNO layer implementing Eq. (5) of Tran et al. 2023:

        output = sigma( W z + b + K(z) )

    where K is a spectral conv and W is a pointwise 1x1 conv in physical space.
    Note: the residual/skip connection is NOT placed inside the block — the
    original FNO has no residual connection. This is one of the things F-FNO
    changes.
    """

    def __init__(self, channels: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        # 1x1 conv = pointwise linear map in physical space (the "W z + b" part)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.pointwise(x))


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for learning a single-step (Markov) update
    of a 2D field on a regular grid.

    Input:  (batch, in_channels, H, W)   e.g. vorticity + (x, y) coords
    Output: (batch, out_channels, H, W)  next-step vorticity prediction

    Notation matches the F-FNO paper (Section 3):
      - P : lifting operator (in_channels -> hidden)
      - L : stack of spectral + pointwise layers
      - Q : projection operator (hidden -> out_channels)
    """

    def __init__(
        self,
        in_channels: int = 3,   # e.g. [vorticity, x-coord, y-coord]
        out_channels: int = 1,  # next-step vorticity
        hidden: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden = hidden

        # P: lifting — takes physical-space input to hidden representation
        self.P = nn.Conv2d(in_channels, hidden, kernel_size=1)

        # L blocks — operator layers
        self.blocks = nn.ModuleList(
            [FNOBlock2d(hidden, modes1, modes2) for _ in range(n_layers)]
        )

        # Q: projection — maps hidden back to physical output.
        # Two-layer MLP (1x1 convs) matching the reference implementation.
        self.Q = nn.Sequential(
            nn.Conv2d(hidden, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.P(x)
        for block in self.blocks:
            x = block(x)
        x = self.Q(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
