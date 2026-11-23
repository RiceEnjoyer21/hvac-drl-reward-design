from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd

from hvac_rl.envs import ACTION_HIGH, ACTION_LOW, comfort_violation

P_REF_W = 2150.0
DT_REF_C = 0.5
ACTION_RANGE = ACTION_HIGH - ACTION_LOW


@dataclass(frozen=True)
class R1Config:
    w: float = 0.5
    lam: float = 0.1
    p_ref_w: float = P_REF_W
    dt_ref_c: float = DT_REF_C


def r1_terms(power_w, temp_c, month, occupied, d_heat, d_cool, cfg: R1Config) -> dict:
    energy = np.asarray(power_w) / cfg.p_ref_w
    comfort = np.asarray(occupied) * comfort_violation(temp_c, month) / cfg.dt_ref_c
    smooth = 0.5 * (np.abs(d_heat) / ACTION_RANGE[0] + np.abs(d_cool) / ACTION_RANGE[1])
    parts = {
        "r1_energy": -cfg.w * energy,
        "r1_comfort": -(1 - cfg.w) * comfort,
        "r1_smooth": -cfg.lam * smooth,
    }
    return {"r1": parts["r1_energy"] + parts["r1_comfort"] + parts["r1_smooth"],
            **parts, "e_norm": energy, "c_norm": comfort, "s_norm": smooth}


def r1_from_log(df: pd.DataFrame, cfg: R1Config) -> pd.DataFrame:
    terms = r1_terms(
        df["HVAC_electricity_demand_rate"], df["air_temperature"], df["month"], df["people_occupant"] > 0,
        df["a_heat"].diff().fillna(0.0), df["a_cool"].diff().fillna(0.0), cfg,
    )
    return pd.DataFrame(terms, index=df.index)


class R1Reward(gym.Wrapper):
    def __init__(self, env: gym.Env, cfg: R1Config):
        super().__init__(env)
        self.cfg = cfg
        self.names = env.unwrapped.observation_variables
        self.prev_action = None

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), ACTION_LOW, ACTION_HIGH)
        obs, _, term, trunc, info = self.env.step(a)
        o = dict(zip(self.names, obs))
        da = np.zeros(2) if self.prev_action is None else a - self.prev_action
        self.prev_action = a
        terms = r1_terms(o["HVAC_electricity_demand_rate"], o["air_temperature"], o["month"],
                         o["people_occupant"] > 0, da[0], da[1], self.cfg)
        terms = {k: float(v) for k, v in terms.items()}
        info.update(terms)
        return obs, terms["r1"], term, trunc, info


KAPPA0_KWH_PER_CH = P_REF_W / DT_REF_C / 1000.0


@dataclass(frozen=True)
class R2Config:
    kappa: float = 50.0
    lam: float = 0.2

    @property
    def comfort_coef(self) -> float:
        return self.kappa / KAPPA0_KWH_PER_CH

    @property
    def equivalent_r1_w(self) -> float:
        return KAPPA0_KWH_PER_CH / (KAPPA0_KWH_PER_CH + self.kappa)


def r2_terms(power_w, temp_c, month, occupied, d_heat, d_cool, cfg: R2Config) -> dict:
    energy = np.asarray(power_w) / P_REF_W
    comfort = np.asarray(occupied) * comfort_violation(temp_c, month) / DT_REF_C
    smooth = 0.5 * (np.abs(d_heat) / ACTION_RANGE[0] + np.abs(d_cool) / ACTION_RANGE[1])
    parts = {
        "r2_energy": -energy,
        "r2_comfort": -cfg.comfort_coef * comfort,
        "r2_smooth": -cfg.lam * smooth,
    }
    return {"r2": parts["r2_energy"] + parts["r2_comfort"] + parts["r2_smooth"], **parts}


def r2_from_log(df: pd.DataFrame, cfg: R2Config) -> pd.DataFrame:
    terms = r2_terms(
        df["HVAC_electricity_demand_rate"], df["air_temperature"], df["month"], df["people_occupant"] > 0,
        df["a_heat"].diff().fillna(0.0), df["a_cool"].diff().fillna(0.0), cfg,
    )
    return pd.DataFrame(terms, index=df.index)


def kappa_cost(energy_kwh: float, discomfort_occ_ch: float, kappa: float) -> float:
    return energy_kwh + kappa * discomfort_occ_ch


class R2Reward(gym.Wrapper):
    def __init__(self, env: gym.Env, cfg: R2Config):
        super().__init__(env)
        self.cfg = cfg
        self.names = env.unwrapped.observation_variables
        self.prev_action = None

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), ACTION_LOW, ACTION_HIGH)
        obs, _, term, trunc, info = self.env.step(a)
        o = dict(zip(self.names, obs))
        da = np.zeros(2) if self.prev_action is None else a - self.prev_action
        self.prev_action = a
        terms = r2_terms(o["HVAC_electricity_demand_rate"], o["air_temperature"], o["month"],
                         o["people_occupant"] > 0, da[0], da[1], self.cfg)
        terms = {k: float(v) for k, v in terms.items()}
        info.update(terms)
        return obs, terms["r2"], term, trunc, info
