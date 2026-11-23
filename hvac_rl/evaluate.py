from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from hvac_rl.envs import CORE_ZONE, STEP_HOURS, ZONES, comfort_bounds, comfort_violation, make_split_env, zone_temp_col
from hvac_rl.rewards import R1Config, R1Reward

Policy = Callable[[dict], np.ndarray]


def rollout(split: str, policy: Policy, r1: R1Config | None = None, all_zone_temps: bool = True,
            max_steps: int | None = None, **env_kwargs) -> pd.DataFrame:
    env = make_split_env(split, all_zone_temps=all_zone_temps, **env_kwargs)
    if r1 is not None:
        env = R1Reward(env, r1)
    names = env.unwrapped.observation_variables
    obs, _ = env.reset()
    rows, done = [], False
    while not done and (max_steps is None or len(rows) < max_steps):
        action = np.asarray(policy(dict(zip(names, obs))), dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        rows.append({
            **dict(zip(names, obs)),
            "a_heat": float(action[0]), "a_cool": float(action[1]), "reward": float(reward),
            "energy_term": float(info.get("energy_term", np.nan)),
            "comfort_term": float(info.get("comfort_term", np.nan)),
            **{k: v for k, v in info.items() if k.startswith("r1")},
        })
        done = term or trunc
    env.close()
    return pd.DataFrame(rows)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        occupied=df["people_occupant"] > 0,
        power_kw=df["HVAC_electricity_demand_rate"] / 1e3,
        energy_kwh=df["total_electricity_HVAC"] / 3.6e6,
    )
    for z in ZONES:
        if zone_temp_col(z) in df:
            df[f"viol_{z}"] = comfort_violation(df[zone_temp_col(z)], df["month"])
    return df


def zone_discomfort(df: pd.DataFrame) -> pd.Series:
    occ = df["occupied"]
    return pd.Series({z: df.loc[occ, f"viol_{z}"].sum() * STEP_HOURS for z in ZONES if f"viol_{z}" in df})


def kpis(df: pd.DataFrame, zone_areas: dict[str, float] | None = None) -> dict:
    occ, v = df["occupied"], df[f"viol_{CORE_ZONE}"]
    out = {
        "energy_kwh": df["energy_kwh"].sum(),
        "power_mean_kw": df["power_kw"].mean(),
        "power_p95_kw": df["power_kw"].quantile(0.95),
        "power_peak_kw": df["power_kw"].max(),
        "discomfort_occ_Ch": v[occ].sum() * STEP_HOURS,
        "frac_occ_outside": (v[occ] > 0).mean(),
        "discomfort_unocc_Ch": v[~occ].sum() * STEP_HOURS,
        "energy_occ_kwh": df.loc[occ, "energy_kwh"].sum(),
        "energy_unocc_kwh": df.loc[~occ, "energy_kwh"].sum(),
        "reward_sum": df["reward"].sum(),
        "reward_energy_sum": df["energy_term"].sum(),
        "reward_comfort_sum": df["comfort_term"].sum(),
    }
    zd = zone_discomfort(df)
    if zone_areas and len(zd) == len(ZONES):
        w = pd.Series(zone_areas)[zd.index]
        out["discomfort_occ_Ch_allzones"] = float((zd * w).sum() / w.sum())
    return out


def policy_metrics(df: pd.DataFrame, weights=(0.3, 0.5, 0.7)) -> dict:
    from hvac_rl.rewards import r1_from_log

    out = kpis(df)
    out["R0"] = out.pop("reward_sum")
    for w in weights:
        out[f"R1_w{w}"] = r1_from_log(df, R1Config(w=w))["r1"].sum()
    heat_min, cool_max, occ = df["a_heat"] <= 12.05, df["a_cool"] >= 29.95, df["occupied"]
    lo, hi = comfort_bounds(df["month"])
    cold, hot = df["air_temperature"] < lo, df["air_temperature"] > hi
    out["discomfort_occ_cold_Ch"] = ((lo - df["air_temperature"]).clip(lower=0)[occ]).sum() * STEP_HOURS
    out["discomfort_occ_hot_Ch"] = ((df["air_temperature"] - hi).clip(lower=0)[occ]).sum() * STEP_HOURS
    out["frac_occ_bound_violating"] = ((heat_min & cold) | (cool_max & hot))[occ].mean()
    out["frac_heat_at_min"] = heat_min.mean()
    out["frac_cool_at_max"] = cool_max.mean()
    out["frac_heat_at_min_occ"] = heat_min[occ].mean()
    out["frac_cool_at_max_occ"] = cool_max[occ].mean()
    return out


def evaluate_checkpoint(run_dir, algo: str, names: list[str], tag: str, split: str,
                        max_steps: int | None = None, log_path=None) -> dict:
    import torch

    from hvac_rl.agents import load_policy

    torch.set_num_threads(1)
    df = add_derived(rollout(split, load_policy(run_dir, algo, names, tag), max_steps=max_steps))
    if log_path is not None:
        df.to_csv(log_path, index=False)
    return policy_metrics(df)
