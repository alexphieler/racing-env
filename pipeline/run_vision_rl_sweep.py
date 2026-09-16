#!/usr/bin/env python3
"""Launch clean conservative vision RL or state PPO runs on Slurm."""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--methods', nargs='+', choices=['ppo', 'sac', 'asymmetric-sac', 'state-ppo'],
                        default=['ppo', 'sac', 'asymmetric-sac'])
    parser.add_argument('--time', default='12:00:00', help='Time limit per run')
    parser.add_argument('--total-timesteps', type=int, default=None,
                        help='Environment steps per run (defaults: PPO 2M, SAC 1M)')
    parser.add_argument('--cpus-per-task', type=int, default=10)
    parser.add_argument('--mem-per-cpu', type=int, default=8000, help='Slurm memory in MB')
    parser.add_argument('--license', default='horse', help='Slurm license; empty to omit')
    parser.add_argument('--python', default='python3', help='Python executable inside container')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.cpus_per_task, args.mem_per_cpu) < 1 or (
        args.total_timesteps is not None and args.total_timesteps < 1
    ):
        parser.error('Step and resource counts must be positive')
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be distinct non-negative integers')
    if len(set(args.methods)) != len(args.methods):
        parser.error('Methods must be distinct')
    return args


def main():
    args = parse_args()
    root = Path(os.environ.get('NIPS_ENV_ROOT', Path(__file__).resolve().parent.parent)).resolve()
    pipeline = root / 'pipeline'
    image = Path(os.environ.get('NIPS_ENV_IMAGE', str(Path.home() / 'fs-rl-dir'))).expanduser()
    icd = Path(os.environ.get('NIPS_ENV_NVIDIA_ICD', str(root / 'nvidia_icd_apptainer.json')))
    log_dir = pipeline / 'runs' / ('vision_rl_sweep_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if not args.dry_run:
        for executable in ('srun', 'singularity'):
            if shutil.which(executable) is None:
                raise SystemExit(f'Missing executable: {executable}')
        for path in (image, icd, pipeline / 'ppo_vision.py', pipeline / 'sac_vision.py'):
            if not path.exists():
                raise SystemExit(f'Missing path: {path}')
        log_dir.mkdir(parents=True)

    jobs = []
    print(f'Job logs: {log_dir}', flush=True)
    for method in args.methods:
        for seed in args.seeds:
            label = f'{method}_seed_{seed}'
            is_ppo = method in ('ppo', 'state-ppo')
            script = 'ppo_vision.py' if is_ppo else 'sac_vision.py'
            total_timesteps = args.total_timesteps
            if total_timesteps is None:
                total_timesteps = 2_000_000 if is_ppo else 1_000_000
            training = [args.python, '-u', str(pipeline / script), '--seed', str(seed),
                        '--reward-mode', 'conservative', '--no-noise-augment',
                        '--vision-actor', 'structured', '--total-timesteps', str(total_timesteps)]
            if is_ppo:
                training += ['--policy-type', 'state' if method == 'state-ppo' else 'vision']
            else:
                training += ['--asymmetric-critic' if method == 'asymmetric-sac' else '--no-asymmetric-critic']
            # No pretrained paths, critic preloading, or warmup: all runs start from scratch.
            command = ['srun', '--ntasks=1', '--nodes=1', '--gres=gpu:1', '--gpus-per-task=1',
                       f'--cpus-per-task={args.cpus_per_task}', f'--mem-per-cpu={args.mem_per_cpu}',
                       f'--time={args.time}', f'--job-name={label if method == "state-ppo" else "vision_" + label}',
                       f'--output={log_dir / (label + "_%j.out")}',
                       f'--error={log_dir / (label + "_%j.err")}']
            if args.license:
                command += ['--licenses', args.license]
            command += ['singularity', 'exec', '--nv', '--bind', f'{root}:{root}',
                        '--bind', f'{icd}:/tmp/nvidia_icd.json',
                        '--env', 'VK_DRIVER_FILES=/tmp/nvidia_icd.json',
                        '--pwd', str(pipeline), str(image), *training]
            print(f'{label}: {shlex.join(command)}', flush=True)
            jobs.append((label, command))
    if args.dry_run:
        return

    running = []
    failed = False
    try:
        for label, command in jobs:
            running.append((label, subprocess.Popen(command, cwd=pipeline)))
        for label, process in running:
            code = process.wait()
            failed |= code != 0
            print(f'{label}: {"FINISHED" if code == 0 else f"FAILED (exit {code})"}', flush=True)
    finally:
        # Do not leave allocations running if the launcher is interrupted.
        for _, process in running:
            if process.poll() is None:
                process.terminate()
    raise SystemExit(int(failed))


if __name__ == '__main__':
    main()
