# Mini-Project: Factorized Fourier Neural Operators (F-FNO)

**INDENG 242B Spring 2026 — UC Berkeley** Ryan Michael Chekkouri, Yijun Gu

This directory contains a scaled-down replication of Tran et al. (2023), ["Factorized Fourier Neural Operators"](https://arxiv.org/abs/2111.13802) (ICLR 2023), plus a small methodological extension exploring anisotropic mode allocation. It lives alongside the original `fourierflow` repository but does not modify any of the original code — everything new is confined to `mini_project/`.

## What we did

1. **Re-implemented FNO and F-FNO from scratch in PyTorch** (~350 lines total) using a shared training pipeline, so every comparison is apples-to-apples.
2. **Reproduced the paper's central claims on 2D Navier-Stokes** at a scale that fits in ~1 hour on a single Colab GPU: the ~10× parameter savings, the depth-scaling behavior, the parameter-efficiency frontier, the training-trick ablation, and the rollout stability.
3. **Proposed and tested a small extension**: per-axis mode allocation (anisotropic F-FNO) on a shear-dominated forcing variant.

All results are documented in the project report. See `results/` for the generated plots and summary CSVs.

## Key findings (scaled-down regime)

| Paper claim                                     | Our result                                                   |
| ----------------------------------------------- | ------------------------------------------------------------ |
| F-FNO uses ~10× fewer parameters                | ✓ 9.94–10.23× ratio across depths                            |
| F-FNO scales with depth; FNO degrades           | ✓ FNO: 0.41% → 0.46% over L=4→12; F-FNO: 0.41% → 0.22%       |
| F-FNO is more parameter-efficient               | ✓ F-FNO-12L (350k params) beats FNO-12L (3.56M params) by 2× |
| Training tricks and architecture are orthogonal | ✓ Cosine LR gives 3× for FNO; architecture alone matches FNO+tricks |
| F-FNO has better rollout stability              | ✓ Lower per-step error across all 19 rollout steps           |

Our extension experiment further shows that **naive hand-specified anisotropic allocation does not beat the symmetric default**, but that **reversed allocation is 3× worse** — so allocation direction matters strongly, and the paper's symmetric design is a principled minimax default. See Section 4 of the report.

## Repository layout

```
mini_project/
├── models/
│   ├── fno.py                              # FNO baseline (Li et al. 2021a)
│   └── ffno.py                             # F-FNO with factorized spectral + improved residuals
├── data/
│   ├── generate_ns.py                      # pseudo-spectral Navier-Stokes solver on the torus
│   └── dataset.py                          # PyTorch Dataset for (w_t, w_{t+1}) pairs
├── train.py                                # unified training pipeline; all paper tricks as flags
├── notebooks/
│   ├── 00_quickstart_runner.ipynb          # end-to-end runner (what we actually used)
│   ├── 01_data_generation.ipynb            # standalone data generation
│   ├── 02_architecture_comparison.ipynb    # standalone Section 3 experiment
│   └── 03_training_tricks_ablation.ipynb   # standalone ablation experiment
├── quickstart.py                           # script version of the runner
├── results/                                # generated plots and summary CSVs
└── runs/                                   # model checkpoints (ignored by git)
```

The four notebooks are redundant by design: `00_quickstart_runner.ipynb` executes the full pipeline end-to-end (data → depth sweep → ablation → plots) and is what we actually used. The three numbered notebooks are retained for interactive exploration of individual stages.

## How to run

On Colab with any GPU (we used an L4; T4 also works, A100 is overkill):

```python
# in a fresh Colab notebook
!git clone https://github.com/XAPP-P/fourierflow.git /content/fourierflow

import sys
from pathlib import Path
PROJECT_ROOT = Path('/content/fourierflow/mini_project')
sys.path.insert(0, str(PROJECT_ROOT))
```

Then open `00_quickstart_runner.ipynb` and run the cells top to bottom. The control-panel cell at the top lets you enable or skip individual stages for reruns.

**Total runtime on L4** (approximate):

| Stage                                                        | Time           |
| ------------------------------------------------------------ | -------------- |
| Data generation (800 train / 100 val / 100 test)             | ~1 min         |
| Depth sweep (6 runs: FNO/F-FNO × L ∈ {4,8,12}, 20 epochs each) | ~50 min        |
| Training-trick ablation (6 runs, 15 epochs each)             | ~35 min        |
| Plots + CSVs                                                 | <1 min         |
| **Total**                                                    | **~1.5 hours** |

On a T4 the total is roughly 2.5–3 hours.

## Scaling decisions vs. the original paper

We stay faithful to every architectural choice in the paper but reduce the training budget so a single team member can run the whole pipeline in one sitting on a consumer-grade Colab GPU.

| Dimension              | Paper                                     | Ours                                                      |
| ---------------------- | ----------------------------------------- | --------------------------------------------------------- |
| Dataset size           | 1000 train / 200 val                      | 800 train / 100 val                                       |
| Resolution             | 64×64                                     | 64×64 (same)                                              |
| Training steps per run | ~100,000                                  | ~16,000 (20 epochs × ~800 steps)                          |
| Max depth tested       | 24 layers                                 | 12 layers                                                 |
| Hidden dim             | 64                                        | 32                                                        |
| PDE geometries         | Torus + airfoil + elasticity + plasticity | Torus only                                                |
| Optimizer              | Adam + cosine decay                       | AdamW + cosine decay                                      |
| Extension              | —                                         | Anisotropic mode allocation on shear-forced Navier-Stokes |

Every architectural choice (factorized spectral layer, residual-after- nonlinearity, two-layer feedforward, weight sharing option) is faithful to Tran et al. (2023). The core comparison is fair because both FNO and F-FNO are evaluated under identical data and training recipes.

## Reference

Tran, A., Mathews, A., Xie, L., & Ong, C. S. (2023). *Factorized Fourier Neural Operators.* ICLR 2023. [arXiv:2111.13802](https://arxiv.org/abs/2111.13802)
