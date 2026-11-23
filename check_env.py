import gymnasium as gym
import numpy as np
import sinergym
from sinergym.utils.controllers import RBC5Zone

ids = [e for e in gym.envs.registry if "5zone" in e]
print("5zone envs:", len(ids), ids[:6])

env = gym.make("Eplus-5zone-hot-continuous-v1")
print("obs space:", env.observation_space.shape)
print("action space:", env.action_space)

agent = RBC5Zone(env.unwrapped)
obs, info = env.reset()
total_r = 0.0
for t in range(2000):
    action = agent.act(obs)
    obs, r, term, trunc, info = env.step(action)
    total_r += r
    if term or trunc:
        break
print(f"steps={t+1}, reward_sum={total_r:.2f}")
print("last info keys:", list(info.keys())[:10])
env.close()
print("OK")
