from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from hvac_rl.envs import RESULTS_DIR  # noqa: E402
from hvac_rl.agents_v2 import RunConfig  # noqa: E402
from hvac_rl.evaluate_v2 import evaluate_checkpoint  # noqa: E402

RUNS_DIR = RESULTS_DIR / "runs_v2"
SPLITS = ["test", "stress_mixed", "stress_cool"]


def select_tag(run_dir: Path, cfg: dict, mode: str) -> str:
    if mode == "final":
        return "final"
    val = pd.read_csv(run_dir / "val.csv")
    return str(int(val.loc[val[RunConfig.from_dict(cfg).own_reward].idxmax(), "timesteps"]))


def eval_job(run_dir: Path, split: str, mode: str, max_steps: int | None) -> str:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    tag = select_tag(run_dir, cfg, mode)
    out = run_dir / "eval"
    out.mkdir(exist_ok=True)
    m = evaluate_checkpoint(str(run_dir), cfg, tag, split, max_steps, log_path=out / f"{split}__{mode}.csv")
    m.update(checkpoint=tag, selection=mode)
    (out / f"{split}__{mode}.json").write_text(json.dumps(m, indent=2, default=float))
    return f"{run_dir.name:28s} {split:13s} ckpt={tag:>8s} E={m['energy_kwh'] / 1e3:6.2f} MWh " \
           f"disc={m['discomfort_occ_Ch']:7.1f} Ch  J(k50)={m['J_k50'] / 1e3:7.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run names in results/runs_v2 (default: all finished runs)")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--select", choices=["best_val", "final"], default="best_val")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()

    dirs = [RUNS_DIR / r for r in args.runs] if args.runs else sorted(p.parent for p in RUNS_DIR.glob("*/model_final.zip"))
    jobs = [(d, s) for d in dirs for s in args.splits]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        for f in [ex.submit(eval_job, d, s, args.select, args.max_steps) for d, s in jobs]:
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
