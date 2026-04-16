# Mini-Project: Factorized Fourier Neural Operators (F-FNO)

**INDENG 242B Spring 2026 — UC Berkeley**
Ryan Michael Chekkouri, Yijun Gu

This directory contains a scaled-down replication of Tran et al. (2023),
"Factorized Fourier Neural Operators" (ICLR 2023). It lives alongside the
original `fourierflow` repository but does not modify any of the original
code — everything new is confined to `mini_project/`.

## What's inside

```
mini_project/
├── models/
│   ├── fno.py       # FNO baseline (Li et al. 2021a) in ~150 lines
│   └── ffno.py      # F-FNO with factorized spectral layer + improved residuals
├── data/
│   ├── generate_ns.py  # pseudo-spectral Navier-Stokes solver on torus
│   └── dataset.py      # PyTorch Dataset wrapper for (w_t, w_{t+1}) pairs
├── train.py         # unified training pipeline with all paper tricks as flags
├── notebooks/
│   ├── 01_data_generation.ipynb            # generate and visualize data
│   ├── 02_architecture_comparison.ipynb    # main FNO vs F-FNO experiment
│   └── 03_training_tricks_ablation.ipynb   # ablate each training trick
└── results/         # plots and CSVs go here after running the notebooks
```

## How to run

On Colab (recommended, uses T4 GPU):

1. Clone your fork:
   ```bash
   !git clone https://github.com/<your-username>/fourierflow.git
   %cd fourierflow/mini_project
   ```
2. Open the notebooks in order (01 → 02 → 03). Each notebook is self-
   contained and handles its own imports/paths.

Total runtime on T4:
- Notebook 01 (data generation): ~8 minutes
- Notebook 02 (architecture comparison): ~90-100 minutes (6 runs × ~15 min)
- Notebook 03 (training trick ablation): ~90 minutes (6 runs × ~15 min)

## Scaling decisions vs. original paper

We explicitly stay faithful to the architectural novelty of the paper
but reduce data and compute so a single mini-project member can run
everything on one Colab T4 session.

| Dimension           | Paper                | Ours                  |
|---------------------|----------------------|-----------------------|
| Dataset size        | 1000 train / 200 val | 800 train / 100 val  |
| Resolution          | 64×64                | 64×64 (same)          |
| Training steps      | 100,000              | ~2,000                |
| Max depth tested    | 24 layers            | 12 layers             |
| Hidden dim          | 64                   | 32                    |
| PDE geometries      | Torus + airfoil + elasticity + plasticity | Torus only |
| Optimizer           | Adam + cosine decay  | AdamW + cosine decay  |

Every architectural choice (factorized spectral layer, residual-after-
nonlinearity, two-layer feedforward, weight sharing option) is faithful
to Tran et al. 2023. The core comparison is fair because both FNO and
F-FNO are evaluated under identical data and training recipes.

## Reference

Tran, A., Mathews, A., Xie, L., & Ong, C. S. (2023).
*Factorized Fourier Neural Operators.* ICLR 2023.
[arXiv:2111.13802](https://arxiv.org/abs/2111.13802)
