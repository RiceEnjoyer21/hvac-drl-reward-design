from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hvac_rl.envs import ACTION_HIGH, ACTION_LOW, make_split_env
from hvac_rl.rewards import R1Config, R1Reward

ALGOS = {"sac": SAC, "ppo": PPO, "td3": TD3}


def to_physical(a) -> np.ndarray:
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    return np.clip(ACTION_LOW + (a + 1) / 2 * (ACTION_HIGH - ACTION_LOW), ACTION_LOW, ACTION_HIGH)


class UnitAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def action(self, a):
        return to_physical(a)


def make_train_env(r1: R1Config | None, split: str = "train", monitor_dir: Path | None = None) -> VecNormalize:
    def factory():
        env = make_split_env(split)
        if r1 is not None:
            env = R1Reward(env, r1)
        env = UnitAction(env)
        return Monitor(env, str(monitor_dir / "monitor") if monitor_dir else None)

    return VecNormalize(DummyVecEnv([factory]), norm_obs=True, norm_reward=False, clip_obs=10.0)


def make_model(algo: str, env: VecNormalize, seed: int) -> BaseAlgorithm:
    kwargs = {}
    if algo == "td3":
        kwargs["action_noise"] = NormalActionNoise(np.zeros(2), 0.1 * np.ones(2))
    return ALGOS[algo]("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **kwargs)


def obs_names(env: VecNormalize) -> list[str]:
    return list(env.unwrapped.envs[0].unwrapped.observation_variables)


class AgentPolicy:
    def __init__(self, model: BaseAlgorithm, vecnorm: VecNormalize, names: list[str]):
        self.model = model
        self.mean = vecnorm.obs_rms.mean.copy()
        self.std = np.sqrt(vecnorm.obs_rms.var + vecnorm.epsilon)
        self.clip = vecnorm.clip_obs
        self.names = names

    def __call__(self, obs: dict) -> np.ndarray:
        x = np.array([obs[n] for n in self.names], dtype=np.float32)
        x = np.clip((x - self.mean) / self.std, -self.clip, self.clip)
        a, _ = self.model.predict(x, deterministic=True)
        return to_physical(a)


def save_policy(model: BaseAlgorithm, env: VecNormalize, run_dir: Path, tag: str) -> None:
    model.save(run_dir / f"model_{tag}.zip")
    env.save(str(run_dir / f"vecnormalize_{tag}.pkl"))


def load_policy(run_dir: Path, algo: str, names: list[str], tag: str = "final") -> AgentPolicy:
    model = ALGOS[algo].load(run_dir / f"model_{tag}.zip", device="cpu")
    with open(run_dir / f"vecnormalize_{tag}.pkl", "rb") as f:
        vecnorm = pickle.load(f)
    return AgentPolicy(model, vecnorm, names)
