from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import sinergym.config.modeling as eplus_modeling
import sinergym.simulators.eplus as eplus_simulator
from sinergym.utils.logger import TerminalLogger
from tqdm import tqdm

eplus_simulator.tqdm = partial(tqdm, disable=True)

_get_logger = TerminalLogger.getLogger
TerminalLogger.getLogger = lambda self, name, level, *a, **k: _get_logger(self, name, logging.WARNING, *a, **k)
for _logger in logging.root.manager.loggerDict.values():
    if isinstance(_logger, logging.Logger) and _logger.name.isupper():
        _logger.setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EPLUS_WORKDIR = RESULTS_DIR / "_eplus"

ENV_IDS = {
    "train": "Eplus-5zone-hot-continuous-stochastic-v1",
    "val": "Eplus-5zone-hot-continuous-stochastic-v1",
    "test": "Eplus-5zone-hot-continuous-v1",
    "stress_mixed": "Eplus-5zone-mixed-continuous-v1",
    "stress_cool": "Eplus-5zone-cool-continuous-v1",
}

WEATHER_SEEDS = {"train": 1, "val": 2}

ZONES = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
CORE_ZONE = "SPACE5-1"

COMFORT_WINTER = (20.0, 23.5)
COMFORT_SUMMER = (23.0, 26.0)

ACTION_LOW = np.array([12.0, 23.25], dtype=np.float32)
ACTION_HIGH = np.array([23.25, 30.0], dtype=np.float32)

STEP_HOURS = 0.25


def zone_temp_col(zone: str) -> str:
    return "air_temperature" if zone == CORE_ZONE else f"air_temperature_{zone}"


def is_summer(month):
    return (np.asarray(month) >= 6) & (np.asarray(month) <= 9)


def comfort_bounds(month):
    summer = is_summer(month)
    lo = np.where(summer, COMFORT_SUMMER[0], COMFORT_WINTER[0])
    hi = np.where(summer, COMFORT_SUMMER[1], COMFORT_WINTER[1])
    return lo, hi


def comfort_violation(temp, month):
    lo, hi = comfort_bounds(month)
    temp = np.asarray(temp)
    return np.maximum(lo - temp, 0.0) + np.maximum(temp - hi, 0.0)


def make_env(env_id: str, all_zone_temps: bool = False, **kwargs) -> gym.Env:
    if all_zone_temps:
        variables = dict(gym.spec(env_id).kwargs["variables"])
        for z in ZONES:
            if z != CORE_ZONE:
                variables[zone_temp_col(z)] = ("Zone Air Temperature", z)
        kwargs["variables"] = variables
    EPLUS_WORKDIR.mkdir(parents=True, exist_ok=True)
    eplus_modeling.CWD = str(EPLUS_WORKDIR)
    return gym.make(env_id, **kwargs)


def make_split_env(split: str, all_zone_temps: bool = False, **kwargs) -> gym.Env:
    if split in WEATHER_SEEDS:
        kwargs.setdefault("seed", WEATHER_SEEDS[split])
    return make_env(ENV_IDS[split], all_zone_temps=all_zone_temps, **kwargs)
