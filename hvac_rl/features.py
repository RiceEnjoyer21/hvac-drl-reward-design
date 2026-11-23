from __future__ import annotations

import datetime as dt
from functools import lru_cache

import gymnasium as gym
import numpy as np

from hvac_rl.envs import comfort_bounds

CALENDAR_YEAR = 1991
OCC_START_HOUR = 8
OCC_END_HOUR = 21
HOURS_TO_OCC_CAP = 72.0

BAND = ["comfort_lo", "comfort_hi", "margin_lo", "margin_hi"]
TIME = ["hour_sin", "hour_cos", "workday", "hours_to_occ"]
FEATURE_SETS = {"base": [], "band": BAND, "band_time": BAND + TIME}


@lru_cache(maxsize=512)
def _date(month: int, day: int) -> dt.date:
    return dt.date(CALENDAR_YEAR, month, day)


def is_workday(month, day) -> bool:
    return _date(int(month), int(day)).weekday() < 5


def hours_to_occupancy(month, day, hour) -> float:
    d, h = _date(int(month), int(day)), float(hour)
    if d.weekday() < 5 and h < OCC_END_HOUR:
        return max(OCC_START_HOUR - h, 0.0)
    for k in range(1, 5):
        if (d + dt.timedelta(days=k)).weekday() < 5:
            return min(k * 24 + OCC_START_HOUR - h, HOURS_TO_OCC_CAP)
    return HOURS_TO_OCC_CAP


def extra_features(o: dict, names: list[str]) -> np.ndarray:
    if not names:
        return np.zeros(0, dtype=np.float32)
    lo, hi = (float(x) for x in comfort_bounds(o["month"]))
    t, h = float(o["air_temperature"]), float(o["hour"])
    values = {
        "comfort_lo": lo,
        "comfort_hi": hi,
        "margin_lo": t - lo,
        "margin_hi": hi - t,
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "workday": float(is_workday(o["month"], o["day_of_month"])),
        "hours_to_occ": hours_to_occupancy(o["month"], o["day_of_month"], h),
    }
    return np.array([values[n] for n in names], dtype=np.float32)


class FeatureObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, feature_set: str):
        super().__init__(env)
        self.base_names = list(env.unwrapped.observation_variables)
        self.extra_names = FEATURE_SETS[feature_set]
        self.obs_names = self.base_names + self.extra_names
        n = len(self.obs_names)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        o = dict(zip(self.base_names, obs))
        return np.concatenate([np.asarray(obs, dtype=np.float32), extra_features(o, self.extra_names)])
