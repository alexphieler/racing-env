#!/usr/bin/env python3
"""Run one or both independently schedulable seed-workflow stages."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from artifact_names import replay_buffer_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="Seed shared by every workflow stage.")
    parser.add_argument(
        "--stage",
        choices=("all", "sac", "rest"),
        default="all",
        help=(
            "Workflow portion to run: sac trains the SAC policy; rest runs "
            "fillRB -> BC -> DAgger using that policy; all runs the complete workflow."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=("conservative", "aggressive"),
        default="conservative",
        help="Reward preset used by all stages.",
    )
    parser.add_argument("--noisy", action="store_true", help="Train and use the action-noise SAC variant.")
    parser.add_argument(
        "--include-last-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the previous normalized action in observations.",
    )
    parser.add_argument("--total-timesteps", type=int, help="Optional SAC training-timestep override.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="SAC TensorBoard base directory. Defaults to pipeline/runs.",
    )
    parser.add_argument("--rb-fill-size", type=int, help="Override the number of transitions collected by fillRB.")
    parser.add_argument(
        "--rb-path",
        type=Path,
        help="Replay-buffer path. Defaults to a seed-specific file under $TMPDIR (or /tmp).",
    )
    parser.add_argument("--bc-train-steps", type=int, help="Override the number of BC gradient updates.")
    parser.add_argument("--bc-batch-size", type=int, help="Override the BC batch size.")
    parser.add_argument("--dagger-train-steps", type=int, help="Override the number of DAgger iterations.")
    parser.add_argument("--dagger-batches-per-iter", type=int, help="Override DAgger gradient updates per iteration.")
    parser.add_argument("--dagger-batch-size", type=int, help="Override the total DAgger batch size.")
    parser.add_argument("--dagger-expert-buffer-size", type=int, help="Override the long expert buffer capacity.")
    parser.add_argument("--dagger-expert-short-buffer-size", type=int, help="Override the recent expert buffer capacity.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for child stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def run(command, cwd, dry_run):
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def main():
    args = parse_args()
    pipeline_dir = Path(__file__).resolve().parent
    shared = ["--seed", str(args.seed), "--reward-mode", args.reward_mode]
    if args.noisy:
        shared.append("--noisy")
    if not args.include_last_action:
        shared.append("--no-include-last-action")

    sac_args = ["--seed", str(args.seed), "--reward-mode", args.reward_mode]
    if args.noisy:
        sac_args.append("--noise-augment")
    if not args.include_last_action:
        sac_args.append("--no-include-last-action")
    if args.total_timesteps is not None:
        sac_args.extend(["--total-timesteps", str(args.total_timesteps)])
    log_dir = args.log_dir or pipeline_dir / "runs"
    sac_args.extend(["--log-dir", str(log_dir)])

    fill_args = shared.copy()
    if args.rb_fill_size is not None:
        fill_args.extend(["--buffer-size", str(args.rb_fill_size)])

    default_rb_name = Path(
        replay_buffer_path(args.reward_mode, args.noisy, args.include_last_action, seed=args.seed)
    ).name
    rb_path = args.rb_path or Path(tempfile.gettempdir()) / "nips-env-replay-buffers" / default_rb_name
    fill_args.extend(["--output", str(rb_path)])

    bc_args = shared.copy()
    bc_args.extend(["--buffer", str(rb_path), "--log-dir", str(log_dir)])
    if args.bc_train_steps is not None:
        bc_args.extend(["--train-steps", str(args.bc_train_steps)])
    if args.bc_batch_size is not None:
        bc_args.extend(["--batch-size", str(args.bc_batch_size)])

    dagger_args = shared.copy()
    dagger_args.extend(["--buffer", str(rb_path), "--log-dir", str(log_dir)])
    dagger_overrides = (
        ("--train-steps", args.dagger_train_steps),
        ("--batches-per-iter", args.dagger_batches_per_iter),
        ("--batch-size", args.dagger_batch_size),
        ("--expert-buffer-size", args.dagger_expert_buffer_size),
        ("--expert-short-buffer-size", args.dagger_expert_short_buffer_size),
    )
    for option, value in dagger_overrides:
        if value is not None:
            dagger_args.extend([option, str(value)])

    sac_stage = (("sac_continous_action.py", sac_args),)
    rest_stages = (
        ("fillRB.py", fill_args),
        ("bc.py", bc_args),
        ("dagger.py", dagger_args),
    )
    if args.stage == "sac":
        stages = sac_stage
    elif args.stage == "rest":
        stages = rest_stages
    else:
        stages = sac_stage + rest_stages

    for script, script_args in stages:
        run([args.python, str(pipeline_dir / script), *script_args], pipeline_dir, args.dry_run)


if __name__ == "__main__":
    main()
