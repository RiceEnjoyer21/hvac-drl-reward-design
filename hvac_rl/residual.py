from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import gymnasium as gym
import numpy as np

from hvac_rl.baselines import SeasonalRBC, load_policies
from hvac_rl.envs import ACTION_HIGH, ACTION_LOW, RESULTS_DIR

TUNED_RBC_FILE = RESULTS_DIR / "baselines" / "tuned_rbc.json"
TUNED_RBC_KEY = "tuned_rbc_w0.5"


@dataclass(frozen=True)
class ResidualConfig:
    delta_occ: float = 2.0
    delta_unocc: float = 7.0
    base: str = TUNED_RBC_KEY


@lru_cache(maxsize=4)
def base_policy(key: str = TUNED_RBC_KEY) -> SeasonalRBC:
    return load_policies(TUNED_RBC_FILE)[key]


def residual_to_physical(u, o: dict, cfg: ResidualConfig) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=np.float32), -1.0, 1.0)
    delta = cfg.delta_occ if o["people_occupant"] > 0 else cfg.delta_unocc
    sp = base_policy(cfg.base)(o) + delta * u
    return np.clip(sp, ACTION_LOW, ACTION_HIGH).astype(np.float32)


class ResidualAction(gym.Wrapper):
    def __init__(self, env: gym.Env, cfg: ResidualConfig):
        super().__init__(env)
        self.cfg = cfg
        self.names = list(env.unwrapped.observation_variables)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.last = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last = dict(zip(self.names, obs))
        return obs, info

    def step(self, u):
        obs, reward, term, trunc, info = self.env.step(residual_to_physical(u, self.last, self.cfg))
        self.last = dict(zip(self.names, obs))
        return obs, reward, term, trunc, info
