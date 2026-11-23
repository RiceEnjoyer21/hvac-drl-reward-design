from __future__ import annotations

import gymnasium as gym
import pandas as pd

from hvac_rl.building import zone_areas
from hvac_rl.envs import ENV_IDS, RESULTS_DIR
from hvac_rl.evaluate import add_derived, kpis

EDA_DIR = RESULTS_DIR / "eda"
BASELINES_DIR = RESULTS_DIR / "baselines"


def load_runs(directory=EDA_DIR) -> dict[tuple[str, str], pd.DataFrame]:
    files = sorted(directory.glob("*__*.csv"))
    if not files:
        raise FileNotFoundError(f"No rollouts in {directory}; run the matching script in scripts/.")
    return {tuple(f.stem.split("__")): add_derived(pd.read_csv(f)) for f in files}


def kpi_table(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    areas = zone_areas()
    out = pd.DataFrame({key: kpis(df, areas) for key, df in runs.items()}).T
    out.index.names = ["split", "policy"]
    return out


def observation_spec(env_id: str = ENV_IDS["test"]) -> pd.DataFrame:
    kw = gym.spec(env_id).kwargs
    rows = [(v, "time", "") for v in kw["time_variables"]]
    rows += [(name, "E+ variable", f"{var} [{key}]") for name, (var, key) in kw["variables"].items()]
    rows += [(name, "E+ meter", meter) for name, meter in kw["meters"].items()]
    return pd.DataFrame(rows, columns=["feature", "type", "source"])
