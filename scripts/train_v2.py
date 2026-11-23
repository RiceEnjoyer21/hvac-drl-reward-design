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

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from hvac_rl.envs import RESULTS_DIR  # noqa: E402
from hvac_rl.agents_v2 import RunConfig, make_model, make_train_env, obs_names, save_policy  # noqa: E402
from hvac_rl.evaluate_v2 import evaluate_checkpoint  # noqa: E402

RUNS_DIR = RESULTS_DIR / "runs_v2"


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
        m = isolated(evaluate_checkpoint, str(self.run_dir), self.cfg, tag, "val", self.eval_steps)
        self.rows.append({"timesteps": self.num_timesteps, "wall_s": time.time() - self.t0,
                          "eval_s": time.time() - t_eval, **m})
        pd.DataFrame(self.rows).to_csv(self.run_dir / "val.csv", index=False)
        print(f"[{self.num_timesteps:>8d}] {time.time() - self.t0:7.0f}s  R2(k50)={m['R2_k50']:10.1f}  "
              f"J(k50)={m['J_k50'] / 1e3:7.2f}  E={m['energy_kwh'] / 1e3:6.2f} MWh  "
              f"E_unocc={m['energy_unocc_kwh'] / 1e3:5.2f}  disc={m['discomfort_occ_Ch']:7.1f} Ch", flush=True)


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
    run_dir = RUNS_DIR / (args.name or f"{args.config.stem}_s{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    rc = RunConfig.from_dict(cfg)
    env = make_train_env(rc, monitor_dir=run_dir)
    cfg["obs_names"] = obs_names(env)
    cfg["config_file"] = str(args.config.resolve().relative_to(PROJECT_DIR))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    model = make_model(rc.algo, env, args.seed)
    callback = ValCallback(run_dir, cfg, args.eval_steps)
    t0 = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"], callback=callback)
    wall = time.time() - t0
    save_policy(model, env, run_dir, "final")
    eval_s = float(sum(r["eval_s"] for r in callback.rows))
    (run_dir / "timing.yaml").write_text(yaml.safe_dump({
        "steps": cfg["total_timesteps"], "wall_s": wall, "eval_s": eval_s,
        "train_steps_per_s": cfg["total_timesteps"] / max(wall - eval_s, 1e-9),
    }))
    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
