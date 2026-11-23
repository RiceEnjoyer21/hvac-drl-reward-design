from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hvac_rl.envs import RESULTS_DIR  # noqa: E402
from hvac_rl.evaluate import evaluate_checkpoint  # noqa: E402

SPLITS = ["test", "stress_mixed", "stress_cool"]


def own_reward(cfg: dict) -> str:
    return "R0" if cfg["reward"] == "r0" else f"R1_w{cfg['w']}"


def select_tag(run_dir: Path, cfg: dict, mode: str) -> str:
    if mode == "final":
        return "final"
    val = pd.read_csv(run_dir / "val.csv")
    return str(int(val.loc[val[own_reward(cfg)].idxmax(), "timesteps"]))


def eval_job(run_dir: Path, split: str, mode: str, max_steps: int | None) -> str:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    tag = select_tag(run_dir, cfg, mode)
    out = run_dir / "eval"
    out.mkdir(exist_ok=True)
    metrics = evaluate_checkpoint(run_dir, cfg["algo"], cfg["obs_names"], tag, split, max_steps,
                                  log_path=out / f"{split}__{mode}.csv")
    metrics.update(checkpoint=tag, selection=mode)
    (out / f"{split}__{mode}.json").write_text(json.dumps(metrics, indent=2, default=float))
    return f"{run_dir.name:24s} {split:13s} ckpt={tag:>8s} E={metrics['energy_kwh'] / 1e3:6.2f} MWh " \
           f"disc={metrics['discomfort_occ_Ch']:7.1f} Ch"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run names in results/runs (default: all finished runs)")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--select", choices=["best_val", "final"], default="best_val")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()

    root = RESULTS_DIR / "runs"
    dirs = [root / r for r in args.runs] if args.runs else sorted(p.parent for p in root.glob("*/model_final.zip"))
    jobs = [(d, s) for d in dirs for s in args.splits]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx, max_tasks_per_child=1) as ex:
        futures = [ex.submit(eval_job, d, s, args.select, args.max_steps) for d, s in jobs]
        for f in futures:
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
