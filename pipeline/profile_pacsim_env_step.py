import argparse
import random
import sys
import time
from collections import defaultdict

import numpy as np

from sac_continous_action import Args as SacArgs
from sac_continous_action import make_vector_env
from pacsimEnv import pacsimEnv


class CallProfiler:
    def __init__(self):
        self.stack = []
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def __call__(self, frame, event, arg):
        now = time.perf_counter()
        if event == "c_call":
            name = self._c_name(arg)
            self.stack.append((name, now))
        elif event in ("c_return", "c_exception"):
            if not self.stack:
                return
            name, start = self.stack.pop()
            self.totals[name] += now - start
            self.counts[name] += 1

    @staticmethod
    def _c_name(func):
        module = getattr(func, "__module__", "")
        name = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
        if module and module != "builtins":
            return f"{module}.{name}"
        return name


class Timer:
    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def measure(self, name):
        timer = self

        class _Context:
            def __enter__(self):
                self.start = time.perf_counter()

            def __exit__(self, exc_type, exc, tb):
                timer.totals[name] += time.perf_counter() - self.start
                timer.counts[name] += 1

        return _Context()


def make_raw_env(reward_mode):
    pacsim_args = {
        "cam_sim": False,
        "print_step_status": False,
        "profile_env_step": True,
    }
    if reward_mode == "aggressive":
        pacsim_args.update({
            "lambda_progress": 0.02,
            "lambda_tracking": 0.003,
            "lambda_finish": 10.0,
            "lambda_collition": 10.0,
            "lambda_stand": 0.5,
            "lambda_constant": 0.1,
            "lambda_slipAngle": 0.005,
            "lambda_slipRatio": 0.05,
            "lambda_actionRate": 0.002,
            "lambda_lateral_consistency": 0.001,
            "lambda_longitudinal_consistency": 0.0002,
        })
    elif reward_mode == "conservative":
        pacsim_args.update({
            "lambda_progress": 0.007,
            "lambda_tracking": 0.01,
            "lambda_finish": 10.0,
            "lambda_collition": 10.0,
            "lambda_stand": 0.5,
            "lambda_constant": 0.02,
            "lambda_slipAngle": 0.005,
            "lambda_slipRatio": 0.05,
            "lambda_actionRate": 0.002,
            "lambda_lateral_consistency": 0.001,
            "lambda_longitudinal_consistency": 0.0002,
        })
    return pacsimEnv(pacsim_args)


def print_rows(title, rows, denominator):
    print(f"\n{title}")
    print(f"{'section':44s} {'seconds':>12s} {'share':>9s} {'calls':>8s} {'us/call':>10s}")
    print("-" * 90)
    for name, seconds, calls in rows:
        share = 100.0 * seconds / denominator if denominator > 0 else 0.0
        us_per_call = 1_000_000.0 * seconds / calls if calls else 0.0
        print(f"{name:44s} {seconds:12.6f} {share:8.2f}% {calls:8d} {us_per_call:10.2f}")


def profile_raw_env(args):
    env = make_raw_env(args.reward_mode)
    env.reset(seed=args.seed)
    actions = [env.action_space.sample() for _ in range(args.steps)]

    profiler = CallProfiler()
    start = time.perf_counter()
    for action in actions:
        sys.setprofile(profiler)
        try:
            _, _, terminated, truncated, _ = env.step(action)
        finally:
            sys.setprofile(None)
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start

    print("\nRaw pacsimEnv.step C-call profile")
    print(f"steps={args.steps}, seconds={elapsed:.6f}, ms/step={1000.0 * elapsed / args.steps:.3f}")
    source_rows = [
        (name, seconds, env.profileEnvStepCounts[name])
        for name, seconds in sorted(env.profileEnvStepTotals.items(), key=lambda item: item[1], reverse=True)
        if name != "step_total"
    ]
    source_denominator = env.profileEnvStepTotals.get("step_total", elapsed)
    print_rows("Source-level sections inside pacsimEnv.step", source_rows, source_denominator)

    rows = sorted(profiler.totals.items(), key=lambda item: item[1], reverse=True)
    rows = [(name, seconds, profiler.counts[name]) for name, seconds in rows[: args.top]]
    print_rows("C functions called under raw env.step", rows, elapsed)
    env.close()


def profile_vector_env(args):
    sac_args = SacArgs(
        seed=args.seed,
        total_timesteps=args.steps * args.num_envs,
        num_envs=args.num_envs,
        vector_env_mode=args.vector_env_mode,
        verbose_env=False,
        print_step_status=False,
        reward_mode=args.reward_mode,
    )
    envs = make_vector_env(sac_args)
    envs.reset(seed=args.seed)
    timer = Timer()
    vector_steps = args.steps

    with timer.measure("vector_env_step_total"):
        for _ in range(vector_steps):
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            with timer.measure("vector_env_step_call"):
                envs.step(actions)

    elapsed = timer.totals["vector_env_step_total"]
    envs.close()
    print("\nVector env-step profile")
    print(
        f"num_envs={args.num_envs}, mode={args.vector_env_mode}, "
        f"vector_steps={vector_steps}, env_steps={vector_steps * args.num_envs}"
    )
    print(f"seconds={elapsed:.6f}, ms/vector_step={1000.0 * elapsed / vector_steps:.3f}")
    print(f"env_steps_per_second={(vector_steps * args.num_envs) / elapsed:.2f}")
    rows = [
        (
            "vector_env_step_call",
            timer.totals["vector_env_step_call"],
            timer.counts["vector_env_step_call"],
        )
    ]
    print_rows("Vector wrapper timing", rows, elapsed)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile pacsim env stepping in more detail.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reward-mode", choices=["conservative", "aggressive"], default="conservative")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vector-env-mode", choices=["auto", "sync", "async"], default="auto")
    parser.add_argument("--top", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    profile_raw_env(args)
    profile_vector_env(args)


if __name__ == "__main__":
    main()
