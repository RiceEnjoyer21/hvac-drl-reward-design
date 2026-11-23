from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from hvac_rl.envs import RESULTS_DIR  # noqa: E402
from hvac_rl.evaluate import add_derived, kpis  # noqa: E402
from hvac_rl.agents_v2 import RunConfig, make_train_env, obs_names  # noqa: E402
from hvac_rl.evaluate_v2 import evaluate_policy, metrics_v2  # noqa: E402
from hvac_rl.features import TIME, extra_features  # noqa: E402
from hvac_rl.residual import ResidualConfig, residual_to_physical  # noqa: E402

pd.set_option("display.width", 220)


def check_features() -> None:
    df = pd.read_csv(RESULTS_DIR / "baselines" / "val__tuned_rbc_w0.5.csv")
    f = pd.DataFrame([extra_features(r, TIME) for r in df.to_dict("records")], columns=TIME)
    occ = df["people_occupant"] > 0
    day_occ = df.assign(occ=occ).groupby(["month", "day_of_month"])["occ"].transform("max")
    print("workday == day has occupants:", float((f["workday"].astype(bool) == day_occ).mean()))
    print("hours_to_occ == 0 when occupied:", float((f.loc[occ, "hours_to_occ"] == 0).mean()))
    print("hours_to_occ > 0 when unoccupied:", float((f.loc[~occ, "hours_to_occ"] > 0).mean()))
    print("hours_to_occ range:", f["hours_to_occ"].min(), f["hours_to_occ"].max())


def check_r2_offline() -> None:
    rows = {}
    for name in ["rbc5zone", "fixed_21_24", "tuned_rbc_w0.5"]:
        rows[name] = metrics_v2(add_derived(pd.read_csv(RESULTS_DIR / "baselines" / f"val__{name}.csv")))
    ppo_log = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if ppo_log and ppo_log.exists():
        rows["ppo_r1_w0.5 (v1)"] = metrics_v2(add_derived(pd.read_csv(ppo_log)))
    cols = ["energy_kwh", "discomfort_occ_Ch", "J_k50", "R2_k25", "R2_k50", "R2_k100",
            "r2_energy_sum_k50", "r2_comfort_sum_k50", "r2_smooth_sum_k50"]
    print(pd.DataFrame(rows).T[cols].round(1).to_string())


def check_residual_zero() -> None:
    cfg = ResidualConfig()
    t0 = time.time()
    m = evaluate_policy(lambda o: residual_to_physical(np.zeros(2), o, cfg), "val")
    ref = kpis(add_derived(pd.read_csv(RESULTS_DIR / "baselines" / "val__tuned_rbc_w0.5.csv")))
    print(f"residual u=0: E={m['energy_kwh']:.1f} kWh disc={m['discomfort_occ_Ch']:.2f} Ch "
          f"| tuned RBC: E={ref['energy_kwh']:.1f} disc={ref['discomfort_occ_Ch']:.2f}  ({time.time() - t0:.0f}s)")


def check_envs() -> None:
    for cfg_file in sorted((PROJECT_DIR / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(cfg_file.read_text())
        if cfg["reward"] != "r2":
            continue
        rc = RunConfig.from_dict(cfg)
        env = make_train_env(rc)
        obs = env.reset()
        rs = []
        for _ in range(200):
            obs, r, done, info = env.step(np.array([env.action_space.sample()]))
            rs.append(float(r[0]))
        names = obs_names(env)
        a = info[0].get("r2_energy"), info[0].get("r2_comfort"), info[0].get("r2_smooth")
        print(f"{cfg_file.stem:18s} obs={obs.shape[1]:2d} names_ok={len(names) == obs.shape[1]} "
              f"act={env.action_space.shape} r_mean={np.mean(rs):7.3f} finite={np.isfinite(rs).all()} "
              f"terms={tuple(round(x, 3) for x in a)} own={rc.own_reward}")
        env.close()


if __name__ == "__main__":
    for fn in [check_features, check_r2_offline, check_envs, check_residual_zero]:
        print(f"\n===== {fn.__name__}", flush=True)
        fn()
