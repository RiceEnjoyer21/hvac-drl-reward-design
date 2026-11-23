from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import yaml
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hvac_rl.agents import make_model, make_train_env, obs_names, save_policy  # noqa: E402
from hvac_rl.envs import PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from hvac_rl.evaluate import evaluate_checkpoint  # noqa: E402
from hvac_rl.rewards import R1Config  # noqa: E402


def isolated(fn, *args, **kwargs):
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args, **kwargs).result()


class ValCallback(BaseCallback):
    def __init__(self, run_dir: Path, cfg: dict, eval_steps: int | None):
        super().__init__()
        self.run_dir, self.cfg, self.eval_steps = run_dir, cfg, eval_steps
        self.rows, self.t0 = [], time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % 10_000 == 0:
            elapsed = time.time() - self.t0
            print(f"  progress {self.num_timesteps:>8d} steps  {elapsed:7.0f}s  "
                  f"{self.num_timesteps / elapsed:6.1f} steps/s", flush=True)
        if self.num_timesteps % self.cfg["eval_freq"] == 0:
            self.evaluate(str(self.num_timesteps))
        return True

    def evaluate(self, tag: str) -> None:
        save_policy(self.model, self.model.get_env(), self.run_dir, tag)
        t_eval = time.time()
        metrics = isolated(evaluate_checkpoint, self.run_dir, self.cfg["algo"], self.cfg["obs_names"],
                           tag, "val", self.eval_steps)
        self.rows.append({"timesteps": self.num_timesteps, "wall_s": time.time() - self.t0,
                          "eval_s": time.time() - t_eval, **metrics})
        pd.DataFrame(self.rows).to_csv(self.run_dir / "val.csv", index=False)
        print(f"[{self.num_timesteps:>8d}] {time.time() - self.t0:7.0f}s  R0={metrics['R0']:9.1f}  "
              f"R1(0.5)={metrics['R1_w0.5']:9.1f}  E={metrics['energy_kwh'] / 1e3:6.2f} MWh  "
              f"disc={metrics['discomfort_occ_Ch']:7.1f} Ch", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--eval-freq", type=int)
    ap.add_argument("--eval-steps", type=int)
    ap.add_argument("--name")
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = yaml.safe_load(args.config.read_text())
    cfg["seed"] = args.seed
    cfg["total_timesteps"] = args.steps or cfg["total_timesteps"]
    cfg["eval_freq"] = args.eval_freq or cfg["eval_freq"]
    run_dir = RESULTS_DIR / "runs" / (args.name or f"{args.config.stem}_s{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    r1 = R1Config(w=cfg["w"]) if cfg["reward"] == "r1" else None
    env = make_train_env(r1, monitor_dir=run_dir)
    cfg["obs_names"] = obs_names(env)
    cfg["config_file"] = str(args.config.resolve().relative_to(PROJECT_ROOT))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    model = make_model(cfg["algo"], env, args.seed)
    callback = ValCallback(run_dir, cfg, args.eval_steps)
    t0 = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"], callback=callback)
    wall = time.time() - t0
    save_policy(model, env, run_dir, "final")
    eval_s = float(sum(r["eval_s"] for r in callback.rows))
    (run_dir / "timing.yaml").write_text(yaml.safe_dump({
        "steps": cfg["total_timesteps"], "wall_s": wall, "eval_s": eval_s,
        "train_steps_per_s": cfg["total_timesteps"] / (wall - eval_s),
    }))
    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
