"""
Factorized Fourier Neural Operator (F-FNO) for 2D problems.

Reference: Tran et al. (2023) "Factorized Fourier Neural Operators." ICLR 2023.
https://arxiv.org/abs/2111.13802

Key differences from the original FNO (Li et al. 2021a):

1. Factorized spectral layer (Eq. 8):
       K^(l)(z) = sum_{d in D} IFFT_d( R_d^(l) . FFT_d(z) )
   Instead of one joint 2D weight R^(l) of shape (H, H, M, M), we use one
   1D weight per spatial dimension, R_x and R_y, each of shape (H, H, M).
   Parameter count drops from O(L H^2 M^D) to O(L H^2 M D).

2. Improved residual connection (Eq. 7):
       L^(l)(z) = z + sigma( W_2 sigma( W_1 K(z) + b_1 ) + b_2 )
   Residual is added AFTER the nonlinearity (not inside it), and there is a
   two-layer feedforward block inspired by transformers. This is what lets
   the network scale to 24+ layers without diverging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedSpectralConv2d(nn.Module):
    """
    Factorized 2D spectral convolution: apply 1D FFT along each axis
    independently, learn a separate complex weight per axis, sum the results.

    This is the central architectural novelty of F-FNO (Eq. 8 in the paper).
    """

    def __init__(self, channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.channels = channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / channels
        # Two 1D spectral weights — one per spatial dimension.
        # Shape: (in_channels, out_channels, modes) per axis.
        self.weight_x = nn.Parameter(
            scale * torch.rand(channels, channels, modes_x, dtype=torch.cfloat)
        )
        self.weight_y = nn.Parameter(
            scale * torch.rand(channels, channels, modes_y, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul1d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_c, L) x (in_c, out_c, L) -> (batch, out_c, L)
        return torch.einsum("bil,iol->bol", inp, weights)

    def _spectral_1d(self, x: torch.Tensor, weight: torch.Tensor, modes: int, dim: int) -> torch.Tensor:
        """Apply 1D FFT -> multiply -> IFFT along a single spatial dim."""
        size = x.size(dim)
        # 1D real FFT along `dim`
        x_ft = torch.fft.rfft(x, dim=dim, norm="ortho")

        # Multiply the lowest `modes` frequencies, zero the rest
        out_ft = torch.zeros_like(x_ft)

        if dim == -2 or dim == 2:
            # x is (batch, ch, H, W), FFT along H -> ft is (batch, ch, H//2+1, W)
            batch, ch, _, W = x_ft.shape
            # Transpose so modes are along the last dim for _compl_mul1d
            x_ft_sel = x_ft[:, :, :modes, :].permute(0, 3, 1, 2).reshape(batch * W, ch, modes)
            mul = self._compl_mul1d(x_ft_sel, weight)
            mul = mul.reshape(batch, W, ch, modes).permute(0, 2, 3, 1)
            out_ft[:, :, :modes, :] = mul
        else:  # dim == -1 or dim == 3
            # x is (batch, ch, H, W), FFT along W -> ft is (batch, ch, H, W//2+1)
            out_ft[:, :, :, :modes] = self._compl_mul1d(
                x_ft[:, :, :, :modes].reshape(-1, self.channels, modes),
                weight,
            ).reshape(x_ft.size(0), self.channels, x_ft.size(2), modes)

        return torch.fft.irfft(out_ft, n=size, dim=dim, norm="ortho")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum contributions from each spatial dimension (Eq. 8)
        out_x = self._spectral_1d(x, self.weight_x, self.modes_x, dim=-2)
        out_y = self._spectral_1d(x, self.weight_y, self.modes_y, dim=-1)
        return out_x + out_y


class FFNOBlock2d(nn.Module):
    """
    A single F-FNO layer implementing Eq. (7):

        L(z) = z + sigma( W_2 sigma( W_1 K(z) + b_1 ) + b_2 )

    Residual-after-nonlinearity + two-layer feedforward. The K here is the
    factorized spectral conv above.
    """

    def __init__(self, channels: int, modes_x: int, modes_y: int, ff_expansion: int = 2):
        super().__init__()
        self.spectral = FactorizedSpectralConv2d(channels, modes_x, modes_y)
        # Two-layer feedforward in physical space (1x1 convs), transformer-style
        ff_hidden = channels * ff_expansion
        self.W1 = nn.Conv2d(channels, ff_hidden, kernel_size=1)
        self.W2 = nn.Conv2d(ff_hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: residual is OUTSIDE the nonlinearity, per Eq. (7)
        residual = x
        x = self.spectral(x)
        x = F.gelu(self.W1(x))
        x = self.W2(x)
        return residual + F.gelu(x)


class FFNO2d(nn.Module):
    """
    2D Factorized Fourier Neural Operator.

    Same P -> L blocks -> Q structure as FNO, but:
      - Spectral layers are factorized per axis (Eq. 8)
      - Blocks have transformer-style residual + feedforward (Eq. 7)
      - Can optionally share weights across layers (F-FNO-WS variant)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden: int = 32,
        modes_x: int = 12,
        modes_y: int = 12,
        n_layers: int = 4,
        weight_sharing: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden = hidden
        self.weight_sharing = weight_sharing

        self.P = nn.Conv2d(in_channels, hidden, kernel_size=1)

        if weight_sharing:
            # F-FNO-WS variant: one block reused n_layers times
            shared = FFNOBlock2d(hidden, modes_x, modes_y)
            self.blocks = nn.ModuleList([shared for _ in range(n_layers)])
        else:
            self.blocks = nn.ModuleList(
                [FFNOBlock2d(hidden, modes_x, modes_y) for _ in range(n_layers)]
            )

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
        if self.weight_sharing:
            # Count shared block only once
            unique_params = set()
            total = 0
            for p in self.P.parameters():
                total += p.numel()
            for p in self.blocks[0].parameters():
                total += p.numel()
            for p in self.Q.parameters():
                total += p.numel()
            return total
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
