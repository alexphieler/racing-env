"""Benchmark PacSim Gymnasium observation and AsyncVectorEnv throughput.

The benchmark runs the same base environment and frame-stack wrapper used by
the state and vision training entry points.  It intentionally measures only
environment work: actions are generated before timing begins, and terminal
output / renderer profiling are disabled so that they do not pollute timings.
"""

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStackObservation

from pacsimEnv import pacsimEnv


REWARD_PRESETS = {
    "aggressive": {
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
    },
    "conservative": {
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
    },
}


@dataclass(frozen=True)
class BenchmarkResult:
    observation_mode: str
    num_envs: int
    repeat: int
    observation_bytes_per_env: int
    startup_seconds: float
    reset_seconds: float
    step_seconds: float
    steps: int

    @property
    def environment_steps(self):
        return self.steps * self.num_envs

    @property
    def environment_steps_per_second(self):
        return self.environment_steps / self.step_seconds

    @property
    def milliseconds_per_vector_step(self):
        return 1000.0 * self.step_seconds / self.steps

    @property
    def observation_mebibytes_per_env(self):
        return self.observation_bytes_per_env / 1024**2


def make_env(observation_mode, args, index):
    """Return an AsyncVectorEnv-compatible environment factory."""
    if observation_mode not in ("state", "vision"):
        raise ValueError(f"Unknown observation mode: {observation_mode}")

    def thunk():
        params = {
            "cam_sim": observation_mode == "vision",
            "include_camera_obs": observation_mode == "vision",
            "include_last_action": args.include_last_action,
            "print_step_status": False,
            "verbose": False,
            "profile_render": False,
            "preload_track_cache": args.preload_track_cache,
            "use_shadows": not args.no_shadows,
        }
        if args.camera_config_file is not None:
            params["camera_config_file"] = args.camera_config_file
        params.update(REWARD_PRESETS[args.reward_mode])

        env = pacsimEnv(params)
        env = FrameStackObservation(env, stack_size=args.frame_stack)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(args.seed + index)
        return env

    return thunk


def observation_bytes(observation_space):
    """Return the number of bytes in one fully stacked environment observation."""
    if isinstance(observation_space, gym.spaces.Dict):
        return sum(observation_bytes(space) for space in observation_space.spaces.values())
    if isinstance(observation_space, gym.spaces.Box):
        return int(np.prod(observation_space.shape)) * np.dtype(observation_space.dtype).itemsize
    raise TypeError(f"Unsupported observation space: {observation_space!r}")


def benchmark_case(observation_mode, num_envs, args, repeat):
    env_fns = [make_env(observation_mode, args, index) for index in range(num_envs)]

    startup_start = time.perf_counter()
    envs = gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=args.async_shared_memory,
        context=args.async_context,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    startup_seconds = time.perf_counter() - startup_start

    try:
        bytes_per_observation = observation_bytes(envs.single_observation_space)
        reset_start = time.perf_counter()
        envs.reset(seed=[args.seed + index for index in range(num_envs)])
        reset_seconds = time.perf_counter() - reset_start

        action_shape = (args.warmup_steps + args.steps, num_envs, *envs.single_action_space.shape)
        actions = np.random.default_rng(args.seed + repeat).uniform(
            low=-1.0,
            high=1.0,
            size=action_shape,
        ).astype(envs.single_action_space.dtype)

        for action in actions[:args.warmup_steps]:
            envs.step(action)

        step_start = time.perf_counter()
        for action in actions[args.warmup_steps:]:
            envs.step(action)
        step_seconds = time.perf_counter() - step_start
    finally:
        envs.close()

    return BenchmarkResult(
        observation_mode=observation_mode,
        num_envs=num_envs,
        repeat=repeat,
        observation_bytes_per_env=bytes_per_observation,
        startup_seconds=startup_seconds,
        reset_seconds=reset_seconds,
        step_seconds=step_seconds,
        steps=args.steps,
    )


def print_configuration(args):
    print("PacSim AsyncVectorEnv benchmark")
    print(
        "modes={0}; num_envs={1}; steps={2}; warmup_steps={3}; repeats={4}; "
        "shared_memory={5}; frame_stack={6}; async_context={7}".format(
            ",".join(args.observation_modes),
            ",".join(str(value) for value in args.num_envs),
            args.steps,
            args.warmup_steps,
            args.repeats,
            args.async_shared_memory,
            args.frame_stack,
            args.async_context or "default",
        )
    )
    print("Actions are pre-generated; startup, reset, and step timings are separate.")


def print_result(result):
    print(
        "{mode:6s} envs={envs:2d} repeat={repeat:2d}  "
        "obs={obs_mib:6.2f} MiB/env  "
        "startup={startup:7.3f}s  reset={reset:7.3f}s  "
        "step={step:7.3f}s  {rate:9.1f} env-steps/s  {latency:7.3f} ms/vector-step".format(
            mode=result.observation_mode,
            envs=result.num_envs,
            repeat=result.repeat,
            obs_mib=result.observation_mebibytes_per_env,
            startup=result.startup_seconds,
            reset=result.reset_seconds,
            step=result.step_seconds,
            rate=result.environment_steps_per_second,
            latency=result.milliseconds_per_vector_step,
        )
    )


def print_summary(results: Iterable[BenchmarkResult]):
    grouped = {}
    for result in results:
        grouped.setdefault((result.observation_mode, result.num_envs), []).append(result)

    print("\nSummary (mean across repeats)")
    print(
        f"{'observation':12s} {'envs':>4s} {'startup s':>10s} {'reset s':>10s} "
        f"{'obs MiB/env':>12s} {'env-steps/s':>13s} {'stdev':>10s} {'ms/vector':>11s}"
    )
    print("-" * 96)
    for (mode, num_envs), samples in grouped.items():
        rates = [sample.environment_steps_per_second for sample in samples]
        startup = statistics.mean(sample.startup_seconds for sample in samples)
        reset = statistics.mean(sample.reset_seconds for sample in samples)
        ms_per_vector_step = statistics.mean(sample.milliseconds_per_vector_step for sample in samples)
        rate_stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
        print(
            f"{mode:12s} {num_envs:4d} {startup:10.3f} {reset:10.3f} "
            f"{samples[0].observation_mebibytes_per_env:12.2f} "
            f"{statistics.mean(rates):13.1f} {rate_stdev:10.1f} {ms_per_vector_step:11.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark state and vision PacSim observations with Gymnasium AsyncVectorEnv."
    )
    parser.add_argument(
        "--observation-modes",
        choices=("state", "vision"),
        nargs="+",
        default=("state", "vision"),
        help="Observation modes to benchmark (default: state vision).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16, 32),
        help="AsyncVectorEnv worker counts to benchmark (default: 1 2 4 8).",
    )
    parser.add_argument("--steps", type=int, default=250, help="Timed vector steps per run.")
    parser.add_argument("--warmup-steps", type=int, default=25, help="Untimed vector steps per run.")
    parser.add_argument("--repeats", type=int, default=1, help="Independent runs per case.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--reward-mode", choices=tuple(REWARD_PRESETS), default="conservative")
    parser.add_argument("--camera-config-file", default=None)
    parser.add_argument(
        "--async-shared-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AsyncVectorEnv shared-memory observations (default: enabled).",
    )
    parser.add_argument(
        "--async-context",
        choices=("fork", "forkserver", "spawn"),
        default=None,
        help="Optional multiprocessing context passed to AsyncVectorEnv.",
    )
    parser.add_argument(
        "--preload-track-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the normal PacSim track-cache behavior (default: enabled).",
    )
    parser.add_argument(
        "--include-last-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the previous normalized action in observations (default: enabled).",
    )
    parser.add_argument("--no-shadows", action="store_true", help="Disable Vulkan shadows for a renderer-only comparison.")
    args = parser.parse_args()

    if any(num_envs <= 0 for num_envs in args.num_envs):
        parser.error("--num-envs values must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.frame_stack <= 0:
        parser.error("--frame-stack must be positive")
    return args


def main():
    args = parse_args()
    print_configuration(args)
    results = []
    for observation_mode in args.observation_modes:
        for num_envs in args.num_envs:
            for repeat in range(args.repeats):
                result = benchmark_case(observation_mode, num_envs, args, repeat)
                results.append(result)
                print_result(result)
    print_summary(results)


if __name__ == "__main__":
    main()
