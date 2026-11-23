from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hvac_rl.baselines import EDA_POLICIES  # noqa: E402
from hvac_rl.eda import EDA_DIR  # noqa: E402
from hvac_rl.evaluate import rollout  # noqa: E402

JOBS = [("test", p) for p in EDA_POLICIES] + [
    ("train", "rbc5zone"),
    ("stress_mixed", "rbc5zone"), ("stress_mixed", "off_12_30"),
    ("stress_cool", "rbc5zone"), ("stress_cool", "off_12_30"),
]


def run_job(job: tuple[str, str]) -> str:
    split, policy = job
    df = rollout(split, EDA_POLICIES[policy])
    df.to_csv(EDA_DIR / f"{split}__{policy}.csv", index=False)
    return f"{split:13s} {policy:12s} steps={len(df)} reward_sum={df.reward.sum():.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(run_job, JOBS):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
