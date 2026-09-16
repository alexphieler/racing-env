#!/usr/bin/env python3
"""Evaluate the full seeded state-SAC, BC, and DAgger baseline matrix."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REWARD_MODES = ("conservative", "aggressive")
POLICY_VARIANTS = (("clean", False), ("noisy", True))
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "max_seed",
        type=int,
        help="Evaluate checkpoint seeds 0 through MAX_SEED, inclusive.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for eval.py (default: current interpreter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the eval.py commands without running them.",
    )
    args = parser.parse_args()
    if args.max_seed < 0:
        parser.error("max_seed must be non-negative")
    return args


def run(command, cwd, dry_run):
    print("$ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def evaluation_command(python, eval_script, reward_mode, noisy, model_type, extra_args, seeds):
    command = [
        python,
        str(eval_script),
        "--model-type",
        model_type,
        "--reward-mode",
        reward_mode,
    ]
    if noisy:
        command.append("--noisy")
    command.extend(extra_args)
    command.extend(["--seeds", *map(str, seeds)])
    return command


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    eval_script = script_dir / "eval.py"
    seeds = range(args.max_seed + 1)

    for reward_mode in REWARD_MODES:
        for policy_variant, noisy in POLICY_VARIANTS:
            print(f"\n=== reward={reward_mode} policy={policy_variant} ===", flush=True)

            run(
                evaluation_command(
                    args.python, eval_script, reward_mode, noisy, "state", [], seeds
                ),
                script_dir,
                args.dry_run,
            )

            run(
                evaluation_command(
                    args.python, eval_script, reward_mode, noisy, "vision", [], seeds
                ),
                script_dir,
                args.dry_run,
            )
            run(
                evaluation_command(
                    args.python,
                    eval_script,
                    reward_mode,
                    noisy,
                    "vision",
                    ["--dagger"],
                    seeds,
                ),
                script_dir,
                args.dry_run,
            )


if __name__ == "__main__":
    main()
