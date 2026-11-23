from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hvac_rl.baselines import (  # noqa: E402
    FIXED_BASELINE, RBC5ZONE, fixed, rbc_grid, save_policies, season_sums, select_rbc,
)
from hvac_rl.envs import ENV_IDS, RESULTS_DIR  # noqa: E402
from hvac_rl.evaluate import rollout  # noqa: E402
from hvac_rl.rewards import R1Config  # noqa: E402

OUT = RESULTS_DIR / "baselines"
WEIGHTS = [0.3, 0.5, 0.7]


def grid_job(item):
    pair, setback = item
    sums = season_sums(rollout("train", fixed(*pair, setback=setback), all_zone_temps=False))
    sb = setback or (float("nan"), float("nan"))
    return sums.reset_index().assign(heat=pair[0], cool=pair[1], setback_heat=sb[0], setback_cool=sb[1])


def eval_job(item):
    name, policy, split = item
    rollout(split, policy).to_csv(OUT / f"{split}__{name}.csv", index=False)
    return f"{split:13s} {name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--skip-grid", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.workers) as ex:
        if not args.skip_grid:
            grid = pd.concat(ex.map(grid_job, rbc_grid()), ignore_index=True)
            grid.to_csv(OUT / "rbc_grid_train.csv", index=False)
        grid = pd.read_csv(OUT / "rbc_grid_train.csv")

        tuned = {f"tuned_rbc_w{w}": select_rbc(grid, R1Config(w=w)) for w in WEIGHTS}
        save_policies(tuned, OUT / "tuned_rbc.json")
        for name, p in tuned.items():
            print(name, p, flush=True)

        controllers = {"fixed_21_24": FIXED_BASELINE, "rbc5zone": RBC5ZONE, **tuned}
        jobs = [(n, p, s) for n, p in controllers.items() for s in ENV_IDS]
        for msg in ex.map(eval_job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
