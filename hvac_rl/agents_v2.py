from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hvac_rl.agents import ALGOS, UnitAction, make_model, to_physical
from hvac_rl.envs import make_split_env
from hvac_rl.rewards import R1Config, R1Reward
from hvac_rl.features import FEATURE_SETS, FeatureObs, extra_features
from hvac_rl.residual import ResidualAction, ResidualConfig, residual_to_physical
from hvac_rl.rewards import R2Config, R2Reward

__all__ = ["ALGOS", "RunConfig", "make_model", "make_train_env", "load_policy", "save_policy"]


@dataclass(frozen=True)
class RunConfig:
    algo: str
    reward: str = "r2"
    kappa: float | None = 50.0
    w: float | None = None
    features: str = "band_time"
    residual: bool = False
    delta_occ: float = 2.0
    delta_unocc: float = 7.0

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @property
    def residual_cfg(self) -> ResidualConfig | None:
        return ResidualConfig(self.delta_occ, self.delta_unocc) if self.residual else None

    @property
    def own_reward(self) -> str:
        if self.reward == "r2":
            return f"R2_k{self.kappa:g}"
        return "R0" if self.reward == "r0" else f"R1_w{self.w}"


def wrap_env(env, cfg: RunConfig):
    if cfg.reward == "r2":
        env = R2Reward(env, R2Config(kappa=cfg.kappa))
    elif cfg.reward == "r1":
        env = R1Reward(env, R1Config(w=cfg.w))
    env = ResidualAction(env, cfg.residual_cfg) if cfg.residual else UnitAction(env)
    return FeatureObs(env, cfg.features)


def make_train_env(cfg: RunConfig, split: str = "train", monitor_dir: Path | None = None) -> VecNormalize:
    def factory():
        env = wrap_env(make_split_env(split), cfg)
        return Monitor(env, str(monitor_dir / "monitor") if monitor_dir else None)

    return VecNormalize(DummyVecEnv([factory]), norm_obs=True, norm_reward=False, clip_obs=10.0)


def obs_names(env: VecNormalize) -> list[str]:
    feat = env.unwrapped.envs[0].env
    while not isinstance(feat, FeatureObs):
        feat = feat.env
    return list(feat.obs_names)


class AgentPolicyV2:
    def __init__(self, model: BaseAlgorithm, vecnorm: VecNormalize, names: list[str], cfg: RunConfig):
        self.model, self.cfg = model, cfg
        self.mean = vecnorm.obs_rms.mean.copy()
        self.std = np.sqrt(vecnorm.obs_rms.var + vecnorm.epsilon)
        self.clip = vecnorm.clip_obs
        self.extra = FEATURE_SETS[cfg.features]
        self.base = [n for n in names if n not in self.extra]
        self.residual = cfg.residual_cfg

    def __call__(self, obs: dict) -> np.ndarray:
        x = np.concatenate([np.array([obs[n] for n in self.base], dtype=np.float32),
                            extra_features(obs, self.extra)])
        x = np.clip((x - self.mean) / self.std, -self.clip, self.clip)
        u, _ = self.model.predict(x, deterministic=True)
        return residual_to_physical(u, obs, self.residual) if self.residual else to_physical(u)


def save_policy(model: BaseAlgorithm, env: VecNormalize, run_dir: Path, tag: str) -> None:
    model.save(run_dir / f"model_{tag}.zip")
    env.save(str(run_dir / f"vecnormalize_{tag}.pkl"))


def load_policy(run_dir: Path, cfg: RunConfig, names: list[str], tag: str = "final") -> AgentPolicyV2:
    model = ALGOS[cfg.algo].load(run_dir / f"model_{tag}.zip", device="cpu")
    with open(run_dir / f"vecnormalize_{tag}.pkl", "rb") as f:
        vecnorm = pickle.load(f)
    return AgentPolicyV2(model, vecnorm, names, cfg)
