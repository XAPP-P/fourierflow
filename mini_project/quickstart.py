"""
Quickstart: run the entire replication pipeline in one script.

Usage (on Colab with a T4 GPU):

    %cd /content/fourierflow/mini_project
    !python quickstart.py --stage all

Or run one stage at a time:

    !python quickstart.py --stage data      # ~8 min
    !python quickstart.py --stage depth     # ~90 min
    !python quickstart.py --stage ablation  # ~90 min

This is a convenience wrapper around the notebooks — useful when you want
to kick off a long run and walk away. For interactive analysis and plots,
use the notebooks instead.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def stage_data(args):
    from data.generate_ns import generate_and_save

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[data] using device: {device}")
    t0 = time.time()
    meta = generate_and_save(
        out_dir=args.data_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        N=args.N,
        T=args.T,
        dt=1e-3,
        record_every=100,
        viscosity=1e-3,
        batch_size=50,
        device=device,
        seed=42,
    )
    print(f"[data] done in {time.time() - t0:.1f}s; metadata: {meta}")


def stage_depth(args):
    from train import train, TrainConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parent
    runs_dir = project_root / "runs"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    depths = [int(d) for d in args.depths.split(",")]
    all_results = {}

    for model_type in ["fno", "ffno"]:
        for n_layers in depths:
            key = f"{model_type}_L{n_layers}"
            print(f"\n{'=' * 60}\n[depth] {key}\n{'=' * 60}")
            cfg = TrainConfig(
                model_type=model_type,
                hidden=32, modes=12, n_layers=n_layers,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                lr=2.5e-3, weight_decay=1e-4, warmup_steps=500,
                use_cosine_decay=True,
                input_noise_std=1e-3,
                device=device, seed=0,
                out_dir=str(runs_dir / key),
                max_train_seconds=args.max_seconds_per_run,
            )
            all_results[key] = train(cfg)
            torch.save(all_results, results_dir / "depth_sweep_results.pt")
    print(f"[depth] all results saved to {results_dir / 'depth_sweep_results.pt'}")


def stage_ablation(args):
    from train import train, TrainConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parent
    runs_dir = project_root / "runs" / "ablation"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    configs = [
        {"name": "FNO (plain)",              "model_type": "fno",  "noise": 0.0,  "cosine": False},
        {"name": "FNO + cosine LR",          "model_type": "fno",  "noise": 0.0,  "cosine": True},
        {"name": "FNO + cosine + noise",     "model_type": "fno",  "noise": 1e-3, "cosine": True},
        {"name": "F-FNO (plain)",            "model_type": "ffno", "noise": 0.0,  "cosine": False},
        {"name": "F-FNO + cosine LR",        "model_type": "ffno", "noise": 0.0,  "cosine": True},
        {"name": "F-FNO + cosine + noise",   "model_type": "ffno", "noise": 1e-3, "cosine": True},
    ]

    ablation_results = {}
    for i, cfg_dict in enumerate(configs):
        name = cfg_dict["name"]
        print(f"\n{'=' * 60}\n[ablation {i+1}/{len(configs)}] {name}\n{'=' * 60}")
        cfg = TrainConfig(
            model_type=cfg_dict["model_type"],
            hidden=32, modes=12, n_layers=8,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            lr=2.5e-3, weight_decay=1e-4, warmup_steps=500,
            use_cosine_decay=cfg_dict["cosine"],
            input_noise_std=cfg_dict["noise"],
            device=device, seed=0,
            out_dir=str(runs_dir / f"run_{i}"),
            max_train_seconds=args.max_seconds_per_run,
        )
        ablation_results[name] = train(cfg)
        torch.save(ablation_results, results_dir / "ablation_results.pt")
    print(f"[ablation] all results saved to {results_dir / 'ablation_results.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "data", "depth", "ablation"], default="all")
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "ns_data"))
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--N", type=int, default=64, help="grid resolution")
    p.add_argument("--T", type=int, default=20, help="timesteps per trajectory")
    p.add_argument("--depths", default="4,8,12",
                   help="comma-separated list of layer counts to sweep")
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max-seconds-per-run", type=float, default=18 * 60,
                   help="hard cap on training time per run (seconds)")
    args = p.parse_args()

    if args.stage in ("all", "data"):
        stage_data(args)
    if args.stage in ("all", "depth"):
        stage_depth(args)
    if args.stage in ("all", "ablation"):
        stage_ablation(args)
    print("\nAll requested stages done. Open notebooks/02_*.ipynb and 03_*.ipynb for plots.")


if __name__ == "__main__":
    main()
