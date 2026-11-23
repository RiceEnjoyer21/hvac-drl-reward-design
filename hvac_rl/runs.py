from __future__ import annotations

import json
import re
import time

import pandas as pd
import yaml

from hvac_rl.envs import RESULTS_DIR
from hvac_rl.evaluate import add_derived

RUNS_DIR = RESULTS_DIR / "runs"
LOGS_DIR = RESULTS_DIR / "logs"
PROGRESS_RE = re.compile(r"progress\s+(\d+) steps\s+(\d+)s\s+([\d.]+) steps/s")


def run_info(run_dir) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    label = cfg["reward"].upper() if cfg["reward"] == "r0" else f"R1 w={cfg['w']}"
    return {"run": run_dir.name, "algo": cfg["algo"].upper(), "reward": label, "seed": cfg["seed"]}


def run_dirs(pattern: str = "*"):
    return sorted(p for p in RUNS_DIR.glob(pattern) if (p / "config.yaml").exists())


def load_val(pattern: str = "*") -> pd.DataFrame:
    frames = [pd.read_csv(d / "val.csv").assign(**run_info(d)) for d in run_dirs(pattern) if (d / "val.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_eval(split: str = "test", select: str = "best_val", pattern: str = "*") -> pd.DataFrame:
    rows = []
    for d in run_dirs(pattern):
        f = d / "eval" / f"{split}__{select}.json"
        if f.exists():
            rows.append({**run_info(d), **json.loads(f.read_text())})
    return pd.DataFrame(rows)


def load_eval_log(run: str, split: str = "test", select: str = "best_val") -> pd.DataFrame:
    return add_derived(pd.read_csv(RUNS_DIR / run / "eval" / f"{split}__{select}.csv"))


def progress() -> pd.DataFrame:
    rows = []
    for d in run_dirs():
        log = LOGS_DIR / f"{d.name}.log"
        if not log.exists():
            continue
        text = log.read_text(errors="ignore")
        total = yaml.safe_load((d / "config.yaml").read_text())["total_timesteps"]
        matches = PROGRESS_RE.findall(text)
        steps, elapsed, speed = map(float, matches[-1]) if matches else (0.0, 0.0, float("nan"))
        finished = "\ndone" in text
        if finished and (d / "timing.yaml").exists():
            elapsed = yaml.safe_load((d / "timing.yaml").read_text())["wall_s"]
        idle_min = (time.time() - log.stat().st_mtime) / 60
        status = "done" if finished else ("error" if "Traceback" in text else ("stalled?" if idle_min > 5 else "running"))
        rows.append({
            **run_info(d), "status": status, "steps": int(total if finished else steps),
            "progress_%": 100 * (total if finished else steps) / total, "steps_per_s": speed,
            "elapsed_min": elapsed / 60, "eta_min": 0.0 if finished else (total - steps) / speed / 60,
            "log_idle_min": idle_min,
        })
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()
