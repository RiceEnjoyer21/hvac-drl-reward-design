from __future__ import annotations

import json
import time

import pandas as pd
import yaml

from hvac_rl.envs import RESULTS_DIR
from hvac_rl.evaluate import add_derived
from hvac_rl.runs import PROGRESS_RE

RUNS_DIR = RESULTS_DIR / "runs_v2"
LOGS_DIR = RESULTS_DIR / "logs_v2"


def run_info(run_dir) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    reward = f"R2 κ={cfg['kappa']:g}" if cfg["reward"] == "r2" else cfg["reward"].upper()
    variant = "residual" if cfg.get("residual") else cfg.get("features", "band_time")
    return {"run": run_dir.name, "algo": cfg["algo"].upper(), "reward": reward, "variant": variant,
            "seed": cfg["seed"], "total_timesteps": cfg["total_timesteps"]}


def run_dirs(pattern: str = "*"):
    return sorted(p for p in RUNS_DIR.glob(pattern) if (p / "config.yaml").exists())


def load_val(pattern: str = "*") -> pd.DataFrame:
    frames = [pd.read_csv(d / "val.csv").assign(**run_info(d)) for d in run_dirs(pattern) if (d / "val.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def progress(pattern: str = "*") -> pd.DataFrame:
    rows = []
    for d in run_dirs(pattern):
        log = LOGS_DIR / f"{d.name}.log"
        if not log.exists():
            continue
        info = run_info(d)
        total = info.pop("total_timesteps")
        text = log.read_text(errors="ignore")
        matches = PROGRESS_RE.findall(text)
        steps, elapsed, speed = map(float, matches[-1]) if matches else (0.0, 0.0, float("nan"))
        finished = "\ndone" in text or (d / "model_final.zip").exists()
        if finished and (d / "timing.yaml").exists():
            elapsed = yaml.safe_load((d / "timing.yaml").read_text())["wall_s"]
        idle_min = (time.time() - log.stat().st_mtime) / 60
        status = "done" if finished else ("error" if "Traceback" in text else ("stalled?" if idle_min > 5 else "running"))
        done_steps = total if finished else steps
        rows.append({
            **info, "status": status, "steps": int(done_steps), "progress_%": 100 * done_steps / total,
            "steps_per_s": speed, "elapsed_min": elapsed / 60,
            "eta_min": 0.0 if finished else (total - steps) / speed / 60, "log_idle_min": idle_min,
        })
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()


def load_eval(split: str = "test", select: str = "best_val", pattern: str = "*") -> pd.DataFrame:
    rows = []
    for d in run_dirs(pattern):
        f = d / "eval" / f"{split}__{select}.json"
        if f.exists():
            info = run_info(d)
            info.pop("total_timesteps")
            rows.append({**info, **json.loads(f.read_text())})
    return pd.DataFrame(rows)


def load_eval_log(run: str, split: str = "test", select: str = "best_val") -> pd.DataFrame:
    return add_derived(pd.read_csv(RUNS_DIR / run / "eval" / f"{split}__{select}.csv"))


BASELINES = {"fixed_21_24": ("baseline", "Fixed 21/24"), "rbc5zone": ("baseline", "RBC5Zone (baseline)"),
             "tuned_rbc_w0.5": ("reference", "Tuned RBC (reference)")}
BASELINE_KEY, REFERENCE_KEY = BASELINES["rbc5zone"], BASELINES["tuned_rbc_w0.5"]
FAMILIES = ["baseline", "reference", "v1 R0", "v1 R1", "v2 R2", "v2 residual"]
METRICS = ["energy_kwh", "energy_occ_kwh", "energy_unocc_kwh", "power_peak_kw", "discomfort_occ_Ch",
           "discomfort_occ_cold_Ch", "discomfort_occ_hot_Ch", "discomfort_unocc_Ch", "J_k50", "R0", "R1_w0.5",
           "R2_k50", "frac_heat_at_min", "frac_cool_at_max", "frac_occ_bound_violating"]


def controller_label(info: dict) -> str:
    name = f"{info['algo']} residual" if info["variant"] == "residual" else info["algo"]
    extra = ", base obs" if info["variant"] == "base" else ""
    return f"{name} ({info['reward']}{extra})"


def comparison(split: str, base_logs: dict, select: str = "best_val") -> pd.DataFrame:
    from hvac_rl import runs as runs_v1
    from hvac_rl.evaluate_v2 import metrics_v2

    rows = [{"family": fam, "controller": label, "run": n, "seed": 0, **metrics_v2(base_logs[(split, n)])}
            for n, (fam, label) in BASELINES.items()]
    for d in runs_v1.run_dirs():
        if (d / "eval" / f"{split}__{select}.csv").exists():
            info = runs_v1.run_info(d)
            rows.append({"family": f"v1 {info['reward'].split()[0]}", "controller": f"{info['algo']} ({info['reward']})",
                         "algo": info["algo"], "run": d.name, "seed": info["seed"], **metrics_v2(runs_v1.load_eval_log(d.name, split, select))})
    for d in run_dirs():
        f = d / "eval" / f"{split}__{select}.json"
        if f.exists():
            info = run_info(d)
            family = "v2 residual" if info["variant"] == "residual" else "v2 R2"
            rows.append({"family": family, "controller": controller_label(info), "algo": info["algo"], "run": d.name, "seed": info["seed"],
                         "variant": info["variant"], "kappa": float(info["reward"].split("=")[1]),
                         **json.loads(f.read_text())})
    df = pd.DataFrame(rows)
    df = df.assign(_o=df["family"].map(FAMILIES.index)).sort_values(["_o", "controller", "seed"], kind="stable")
    return df.drop(columns="_o").reset_index(drop=True)


def summarize(df: pd.DataFrame, cols=METRICS) -> pd.DataFrame:
    g = df.groupby(["family", "controller"], sort=False)
    std = g[cols].std().fillna(0.0).add_suffix("_std")
    return pd.concat([g[cols].mean(), std], axis=1).assign(n_seeds=g.size())
