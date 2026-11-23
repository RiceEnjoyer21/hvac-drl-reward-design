from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hvac_rl.envs import ACTION_HIGH, ACTION_LOW, STEP_HOURS, comfort_violation, is_summer
from hvac_rl.rewards import ACTION_RANGE, R1Config

Pair = tuple[float, float]


@dataclass(frozen=True)
class SeasonalRBC:
    winter: Pair
    summer: Pair
    winter_setback: Pair | None = None
    summer_setback: Pair | None = None

    def __call__(self, obs: dict) -> np.ndarray:
        summer = bool(is_summer(obs["month"]))
        sp, setback = (self.summer, self.summer_setback) if summer else (self.winter, self.winter_setback)
        if setback is not None and obs["people_occupant"] <= 0:
            sp = setback
        return np.clip(np.asarray(sp, dtype=np.float32), ACTION_LOW, ACTION_HIGH)


def fixed(heat: float, cool: float, setback: Pair | None = None) -> SeasonalRBC:
    return SeasonalRBC((heat, cool), (heat, cool), setback, setback)


RBC5ZONE = SeasonalRBC(winter=(20.0, 23.5), summer=(23.0, 26.0))
FIXED_BASELINE = fixed(21.0, 24.0)

EDA_POLICIES = {
    "rbc5zone": RBC5ZONE,
    "fixed_21_24": FIXED_BASELINE,
    "occ_setback": fixed(21.0, 24.0, setback=(12.0, 30.0)),
    "off_12_30": fixed(12.0, 30.0),
    "tight_2325": fixed(23.25, 23.25),
}

POLICY_LABELS = {
    "rbc5zone": "RBC5Zone",
    "fixed_21_24": "Fixed 21/24",
    "occ_setback": "21/24 occupied, 12/30 otherwise",
    "off_12_30": "Off 12/30",
    "tight_2325": "Tight 23.25/23.25",
    "tuned_rbc_w0.5": "Tuned RBC",
}

GRID_HEAT = [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
GRID_COOL = [23.25, 24.0, 25.0, 26.0, 27.0, 28.0]
GRID_SETBACK: list[Pair | None] = [None, (16.0, 28.0), (12.0, 30.0)]


def rbc_grid() -> list[tuple[Pair, Pair | None]]:
    return [((h, c), sb) for h, c, sb in itertools.product(GRID_HEAT, GRID_COOL, GRID_SETBACK)]


def season_sums(df: pd.DataFrame) -> pd.DataFrame:
    occ = df["people_occupant"] > 0
    viol = comfort_violation(df["air_temperature"], df["month"]) * occ
    smooth = 0.5 * (df["a_heat"].diff().abs().fillna(0) / ACTION_RANGE[0]
                    + df["a_cool"].diff().abs().fillna(0) / ACTION_RANGE[1])
    season = np.where(is_summer(df["month"]), "summer", "winter")
    return pd.DataFrame({
        "season": season, "power_w": df["HVAC_electricity_demand_rate"], "viol_occ_c": viol, "smooth": smooth,
        "energy_kwh": df["total_electricity_HVAC"] / 3.6e6, "discomfort_occ_Ch": viol * STEP_HOURS,
    }).groupby("season").sum()


def r1_score(sums: pd.DataFrame, cfg: R1Config) -> pd.Series:
    return -(cfg.w * sums["power_w"] / cfg.p_ref_w
             + (1 - cfg.w) * sums["viol_occ_c"] / cfg.dt_ref_c
             + cfg.lam * sums["smooth"])


def select_rbc(grid: pd.DataFrame, cfg: R1Config) -> SeasonalRBC:
    scored = grid.assign(score=r1_score(grid, cfg))
    best = scored.loc[scored.groupby("season")["score"].idxmax()].set_index("season")

    def pick(season):
        r = best.loc[season]
        setback = None if pd.isna(r["setback_heat"]) else (float(r["setback_heat"]), float(r["setback_cool"]))
        return (float(r["heat"]), float(r["cool"])), setback

    (w_pair, w_sb), (s_pair, s_sb) = pick("winter"), pick("summer")
    return SeasonalRBC(w_pair, s_pair, w_sb, s_sb)


def save_policies(policies: dict[str, SeasonalRBC], path: Path) -> None:
    path.write_text(json.dumps({k: asdict(v) for k, v in policies.items()}, indent=2))


def load_policies(path: Path) -> dict[str, SeasonalRBC]:
    raw = json.loads(path.read_text())
    return {k: SeasonalRBC(**{f: tuple(v) if v else None for f, v in d.items()}) for k, d in raw.items()}
