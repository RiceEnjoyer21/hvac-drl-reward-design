from __future__ import annotations

from pathlib import Path

import pandas as pd

from hvac_rl.evaluate import add_derived, policy_metrics, rollout
from hvac_rl.rewards import KAPPA0_KWH_PER_CH, R2Config, kappa_cost, r2_from_log

KAPPAS = (KAPPA0_KWH_PER_CH, 25.0, 50.0, 100.0)


def metrics_v2(df: pd.DataFrame) -> dict:
    out = policy_metrics(df)
    for k in KAPPAS:
        out[f"J_k{k:g}"] = kappa_cost(out["energy_kwh"], out["discomfort_occ_Ch"], k)
    for k in (25.0, 50.0, 100.0):
        terms = r2_from_log(df, R2Config(kappa=k))
        out[f"R2_k{k:g}"] = terms["r2"].sum()
        if k == 50.0:
            out.update({f"{c}_sum_k50": terms[c].sum() for c in ["r2_energy", "r2_comfort", "r2_smooth"]})
    return out


def evaluate_policy(policy, split: str, max_steps: int | None = None, log_path: Path | None = None) -> dict:
    df = add_derived(rollout(split, policy, max_steps=max_steps))
    if log_path is not None:
        df.to_csv(log_path, index=False)
    return metrics_v2(df)


def evaluate_checkpoint(run_dir, cfg_dict: dict, tag: str, split: str, max_steps: int | None = None,
                        log_path=None) -> dict:
    import torch

    from hvac_rl.agents_v2 import RunConfig, load_policy

    torch.set_num_threads(1)
    cfg = RunConfig.from_dict(cfg_dict)
    policy = load_policy(Path(run_dir), cfg, cfg_dict["obs_names"], tag)
    return evaluate_policy(policy, split, max_steps, log_path)


def behaviour(df: pd.DataFrame) -> dict:
    occ = df["occupied"]
    workday = df.groupby(["month", "day_of_month"])["people_occupant"].transform("max") > 0
    summer = df["month"].isin([6, 7, 8, 9])
    return {
        "mean |Δ heat sp|, °C/step": df["a_heat"].diff().abs().mean(),
        "mean |Δ cool sp|, °C/step": df["a_cool"].diff().abs().mean(),
        "heat sp unocc, °C": df["a_heat"][~occ].mean(),
        "cool sp unocc, °C": df["a_cool"][~occ].mean(),
        "T core occ, summer, °C": df["air_temperature"][occ & summer].mean(),
        "T core occ, winter, °C": df["air_temperature"][occ & df["month"].isin([12, 1, 2])].mean(),
        "E occupied, MWh": df["energy_kwh"][occ].sum() / 1e3,
        "E weekday nights, MWh": df["energy_kwh"][~occ & workday].sum() / 1e3,
        "E weekends, MWh": df["energy_kwh"][~workday].sum() / 1e3,
        "E Jun–Sep, MWh": df["energy_kwh"][summer].sum() / 1e3,
        "E Oct–May, MWh": df["energy_kwh"][~summer].sum() / 1e3,
    }
