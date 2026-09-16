import argparse
import csv
import os
import re
import time
import copy
import random
from pathlib import Path

import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation

import networks
from artifact_names import (
    bc_actor_path,
    dagger_actor_path,
    ppo_actor_path,
    state_actor_path,
    state_ppo_actor_path,
    vision_sac_actor_path,
)
from pacsimEnv import pacsimEnv
from policy_inspection import ACTION_NAMES, save_episode_inspection
from teacher_takeover import run_takeovers


MAP_FILES = [
    "/root/workspace/tracks/FSE22.yaml",
    "/root/workspace/tracks/FSE22_test.yaml",
    "/root/workspace/tracks/FSE23.yaml",
    "/root/workspace/tracks/FSG19.yaml",
    "/root/workspace/tracks/FSG21.yaml",
    "/root/workspace/tracks/FSG23.yaml",
    "/root/workspace/tracks/FSS19.yaml",
    "/root/workspace/tracks/FSS22_V1.yaml",
    "/root/workspace/tracks/FSS22_V2.yaml",
    "/root/workspace/tracks/FSO20.yaml",
    "/root/workspace/tracks/FSI24.yaml",
    "/root/workspace/tracks/FSCZ24.yaml",
    "/root/workspace/tracks/FSG24.yaml",
    "/root/workspace/tracks/FSE24.yaml",
    "/root/workspace/tracks/FSG25.yaml",
    "/root/workspace/tracks/FSCZ25.yaml",
]
LAST_FOUR_TRACKS = ("FSG24", "FSE24", "FSG25", "FSCZ25")

# Keep evaluation returns on the same scale as PPO/SAC training returns.
REWARD_PARAMS = {
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

# Best average successful lap time across evaluation reports.
# TRACK_TIME_REFERENCES = {
#     "FSCZ24": 21.586500,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSCZ25": 21.454556,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSE22": 13.810000,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSE22_test": 11.460000,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSE23": 18.461500,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSE24": 20.299000,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSG19": 20.211500,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSG21": 15.880000,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSG23": 24.539000,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSG24": 29.282333,  # test_aggressive_noisy_last_action_state_sac_actor_1789512748.csv
#     "FSG25": 24.224000,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSI24": 24.509000,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSO20": 27.349000,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSS19": 19.615667,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSS22_V1": 16.324875,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
#     "FSS22_V2": 10.810000,  # test_aggressive_last_action_state_sac_actor_1789512863.csv
# }
TRACK_TIME_REFERENCES = {
      "FSCZ24": 21.587,
      "FSCZ25": 21.455,
      "FSE22": 13.810,
      "FSE22_test": 11.460,
      "FSE23": 18.462,
      "FSE24": 20.299,
      "FSG19": 20.212,
      "FSG21": 15.880,
      "FSG23": 24.539,
      "FSG24": 29.282,
      "FSG25": 24.224,
      "FSI24": 24.509,
      "FSO20": 27.349,
      "FSS19": 19.616,
      "FSS22_V1": 16.325,
      "FSS22_V2": 10.810,
  }



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluator for models")
    parser.add_argument('--track-dir', help='Evaluate all .yaml/.yml tracks directly in this directory, without augmentation')
    parser.add_argument('--inspect-actions', action='store_true',
                        help='Save per-track command/teacher/motion CSVs, PNG plots and error summaries.')
    parser.add_argument('--inspection-teacher', default=None,
                        help='Override state SAC teacher checkpoint used for inspection.')
    parser.add_argument('--teacher-takeover', action='store_true',
                        help='Replay failed episodes and test teacher recovery before failure; enables inspection.')
    parser.add_argument('--takeover-seconds', type=float, nargs='+', default=[1.0, 2.0, 5.0],
                        help='Teacher takeover lead times before the original failure, in seconds.')
    parser.add_argument('--model-type', choices=['state', 'vision'], default='state', help='Type of model to evaluate')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model to load (overrides defaults)')
    parser.add_argument('--reward-mode', choices=['conservative', 'aggressive'], default='conservative', help='Reward preset used in default model names')
    parser.add_argument('--noisy', action='store_true', help='Use noisy variant when applicable')
    parser.add_argument('--dagger', action='store_true', help='If evaluating a vision model, load the DAgger-trained vision model')
    parser.add_argument('--ppo', action='store_true', help='Load the PPO-trained actor for the selected model type')
    parser.add_argument('--sac', action='store_true', help='Load the SAC-trained vision actor')
    parser.add_argument('--dreamer', action='store_true', help='Evaluate a recurrent Dreamer policy')
    parser.add_argument('--dreamer-config', help='YAML training-config overrides for Dreamer architecture/preprocessing')
    parser.add_argument('--dreamer-run-dir', help='Sweep directory containing pacsim_state_seed_N or pacsim_vision_seed_N/latest.pt')
    parser.add_argument(
        '--asymmetric-critic',
        action='store_true',
        help='Load a vision SAC actor trained with a privileged state critic',
    )
    parser.add_argument('--seed', type=int, default=1, help='Seed suffix of the default checkpoint filename')
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        metavar='SEED',
        help='Checkpoint seed suffixes to evaluate together, e.g. --seeds 0 1 2 3 4. Overrides --seed.',
    )
    parser.add_argument('--latent', action='store_true', help='Use the latent/encoder vision model')
    parser.add_argument(
        '--training-episodes',
        type=int,
        default=0,
        metavar='N',
        help=(
            'Run N random episodes from pacsimEnv\'s generated training-track pool, '
            'with the same random flips and start quartiles as training. '
            'Replaces the fixed 16-track evaluation.'
        ),
    )
    parser.add_argument(
        '--eval-seed',
        type=int,
        default=1,
        help='Environment RNG seed for --training-episodes; reused for every evaluated checkpoint.',
    )
    parser.add_argument(
        '--stochastic-actions',
        action='store_true',
        help='Sample policy actions instead of using their deterministic means.',
    )
    parser.add_argument(
        '--include-last-action',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use models/env observations that include the previous normalized action',
    )
    args = parser.parse_args()
    if sum((args.dagger, args.ppo, args.sac, args.dreamer)) > 1:
        parser.error('--dagger, --ppo, --sac, and --dreamer are mutually exclusive')
    if args.dreamer_config and not args.dreamer:
        parser.error('--dreamer-config requires --dreamer')
    if args.dreamer_run_dir and (not args.dreamer or args.model_path):
        parser.error('--dreamer-run-dir requires --dreamer and cannot be combined with --model-path')
    if args.dreamer and (args.latent or args.noisy or args.stochastic_actions or not args.include_last_action):
        parser.error('Dreamer requires last-action observations and does not support --latent, --noisy, or --stochastic-actions')
    if args.asymmetric_critic and not args.sac:
        parser.error('--asymmetric-critic requires --sac')
    if args.model_path and args.seeds:
        parser.error('--model-path cannot be combined with --seeds; multi-seed evaluation uses default seed-suffixed checkpoints')
    if args.training_episodes < 0:
        parser.error('--training-episodes must be non-negative')
    if args.track_dir and args.training_episodes:
        parser.error('--track-dir cannot be combined with --training-episodes')
    try:
        args.map_files = resolve_track_files(args.track_dir)
    except ValueError as error:
        parser.error(str(error))
    if any(not np.isfinite(value) or value <= 0 for value in args.takeover_seconds):
        parser.error('--takeover-seconds must contain finite positive values')
    if args.teacher_takeover:
        args.inspect_actions = True
    return args


def resolve_track_files(directory):
    if directory is None:
        return MAP_FILES
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f'Track directory does not exist: {path}')
    tracks = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in ('.yaml', '.yml'))
    if not tracks:
        raise ValueError(f'No YAML tracks found in: {path}')
    if len({p.stem for p in tracks}) != len(tracks):
        raise ValueError('Track filenames must have unique stems (no duplicate .yaml/.yml names)')
    return [str(p) for p in tracks]


def default_model_path(args, seed):
    if args.dreamer:
        if args.dreamer_run_dir:
            return os.path.join(args.dreamer_run_dir, f'pacsim_{args.model_type}_seed_{seed}', 'latest.pt')
        from dreamer_eval import default_checkpoint
        return default_checkpoint(args.model_type, seed)
    if args.model_type == 'state':
        if args.ppo:
            return state_ppo_actor_path(
                args.reward_mode, args.noisy, args.include_last_action, seed=seed
            )
        return state_actor_path(
            args.reward_mode, args.noisy, args.include_last_action, seed=seed
        )

    if args.ppo:
        return ppo_actor_path(
            args.reward_mode, args.noisy, args.latent, args.include_last_action, seed=seed
        )
    if args.sac:
        return vision_sac_actor_path(
            args.reward_mode,
            args.noisy,
            args.latent,
            args.include_last_action,
            seed=seed,
            asymmetric=args.asymmetric_critic,
        )
    if args.dagger:
        return dagger_actor_path(
            args.reward_mode,
            args.noisy,
            args.latent,
            args.include_last_action,
            seed=seed,
        )
    return bc_actor_path(
        args.reward_mode, args.noisy, args.latent, args.include_last_action, seed=seed
    )


def evaluate_actor(actor, env, args, device, teacher=None, inspection_dir=None):
    """Evaluate fixed test tracks or sampled training-condition episodes."""
    results = {}
    episodic_returns = []
    training_condition_eval = args.training_episodes > 0
    episode_count = args.training_episodes if training_condition_eval else len(args.map_files)
    action_index = 0 if args.stochastic_actions else -1
    def rng_state():
        return (copy.deepcopy(env.unwrapped.np_random.bit_generator.state),
                random.getstate(), np.random.get_state())

    def restore_rng(saved):
        env.unwrapped.np_random.bit_generator.state = copy.deepcopy(saved[0])
        random.setstate(saved[1])
        np.random.set_state(saved[2])

    def replay_state(obs):
        sim = env.unwrapped
        return np.concatenate((np.asarray(sim.position).ravel(), np.asarray(sim.orientation).ravel(),
                               networks.flattenFuncStateSingle(obs).numpy().ravel()))

    for episode_index in range(episode_count):
        if args.teacher_takeover:
            before_reset_rng = rng_state()
        if training_condition_eval:
            # This is exactly the reset path used in training: no map override
            # and no noAugment option, so pacsim samples a generated track, flip,
            # and start quartile. Re-seeding makes checkpoints comparable.
            reset_kwargs = {"seed": args.eval_seed} if episode_index == 0 else {}
            obs = env.reset(**reset_kwargs)[0]
            track_name = env.unwrapped.trackName
            result_key = f"training_episode_{episode_index + 1}"
        else:
            map_file = args.map_files[episode_index]
            reset_kwargs = {'options': {"map_files": [map_file], "noAugment": True}}
            obs = env.reset(**reset_kwargs)[0]
            track_name = os.path.splitext(os.path.basename(map_file))[0]
            result_key = track_name
        if args.dreamer:
            actor.reset()
        done = False
        info = None
        episodic_return = 0.0
        episodic_length = 0
        inspection_rows = []
        replay_actions, replay_states, replay_times = [], [], []
        if args.teacher_takeover:
            replay_states.append(replay_state(obs))
            replay_times.append(float(env.unwrapped.time))
        while not done:
            if args.dreamer:
                action_tensor = actor.action(obs)
            elif args.model_type == 'vision':
                vision, sensors = networks.flattenFuncVisionSingle(obs)
                with torch.no_grad():
                    action_tensor = actor.get_action(vision.to(device), sensors.to(device))[action_index]
            else:
                obs_flattened = networks.flattenFuncStateSingle(obs).to(device)
                with torch.no_grad():
                    action_tensor = actor.get_action(obs_flattened)[action_index]

            action = action_tensor.cpu().numpy()[0]
            if args.teacher_takeover:
                replay_actions.append(action.copy())
            if inspection_dir is not None:
                with torch.no_grad():
                    teacher_mean, _ = teacher(networks.flattenFuncStateSingle(obs).to(device))
                    teacher_action = torch.tanh(teacher_mean).cpu().numpy()[0]
                time_before = float(env.unwrapped.time)
            obs, reward, terminated, truncated, info = env.step(action)
            if args.teacher_takeover:
                replay_states.append(replay_state(obs))
                replay_times.append(float(env.unwrapped.time))
            if inspection_dir is not None:
                sim = env.unwrapped
                velocity = np.asarray(obs['velocity'][-1]) * sim.maxSpeed
                row = {'time_before_s': time_before, 'time_after_s': float(sim.time),
                       'vx_mps': float(velocity[0]), 'vy_mps': float(velocity[1]),
                       'observed_steer_normalized': float(obs['steer'][-1][0]),
                       'yaw_rate_radps': float(obs['imu'][-1][2] * sim.maxYawRate),
                       'center_distance_m': float(abs(sim.curvCoords[1])),
                       'reward': float(reward), 'terminated': bool(terminated),
                       'truncated': bool(truncated), 'collided': bool(info.get('collided', False))}
                row.update({f'action_{name}': float(value) for name, value in zip(ACTION_NAMES, action)})
                row.update({f'teacher_{name}': float(value) for name, value in zip(ACTION_NAMES, teacher_action)})
                inspection_rows.append(row)
            episodic_return += float(reward)
            episodic_length += 1
            done = terminated or truncated

        lap_time = float(info["laptime"])
        episode_label = f"episode={episode_index + 1}, " if training_condition_eval else ""
        print(
            f"{episode_label}track={track_name}, episodic_return={episodic_return:.6f}, "
            f"episodic_length={episodic_length}, lap_time={lap_time:.6f}"
        )
        episodic_returns.append(episodic_return)
        results[result_key] = (lap_time > 0.0, lap_time)
        if inspection_dir is not None:
            save_episode_inspection(inspection_dir, episode_index + 1, track_name,
                                    inspection_rows, lap_time > 0.0, info)
        if args.teacher_takeover and lap_time <= 0:
            after_episode_rng = rng_state()
            torch_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            def replay_reset():
                restore_rng(before_reset_rng)
                return env.reset(**reset_kwargs)[0]

            def teacher_action(observation):
                with torch.no_grad():
                    mean, _ = teacher(networks.flattenFuncStateSingle(observation).to(device))
                    return torch.tanh(mean).cpu().numpy()[0]

            try:
                run_takeovers(replay_reset, env.step, replay_state, teacher_action,
                              replay_actions, replay_states, replay_times, info,
                              args.takeover_seconds, inspection_dir, episode_index + 1, track_name)
            finally:
                restore_rng(after_episode_rng)
                torch.random.set_rng_state(torch_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
    print(
        f"{'training-condition episodes' if training_condition_eval else 'all tracks'}: "
        f"mean_episodic_return={np.mean(episodic_returns):.6f}, episodes={len(episodic_returns)}"
    )
    return results, episodic_returns


def sample_stddev(values):
    """Sample standard deviation, defined as zero for a single checkpoint."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def geometric_mean(values):
    """Geometric mean for positive lap times, or None when there are none."""
    if not values:
        return None
    return float(np.exp(np.mean(np.log(values))))


def normalized_time(track_name, lap_time):
    reference_time = TRACK_TIME_REFERENCES.get(track_name)
    if reference_time is None:
        return None
    if reference_time <= 0:
        raise ValueError(f"Track reference time must be positive: {track_name}={reference_time}")
    return lap_time / reference_time


def track_set_metrics(track_results):
    """Summarize one seed's results for a chosen set of tracks."""
    successes = [success for success, _ in track_results.values()]
    successful_times = [lap_time for success, lap_time in track_results.values() if success]
    successful_normalized_times = [
        normalized_time(track_name, lap_time)
        for track_name, (success, lap_time) in track_results.items()
        if success
    ]
    n_tracks = len(track_results)
    n_success = sum(successes)
    return {
        "n_success": n_success,
        "n_tracks": n_tracks,
        "success_rate": n_success / n_tracks if n_tracks else 0.0,
        "time_gmean": geometric_mean(successful_times),
        "normalized_time_gmean": (
            geometric_mean(successful_normalized_times)
            if all(value is not None for value in successful_normalized_times) else None
        ),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    model_paths = [args.model_path] if args.model_path else [default_model_path(args, seed) for seed in seeds]

    missing_paths = [path for path in model_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError("Expected model(s) at exactly:\n" + "\n".join(missing_paths))

    pacsim_args = {
        "cam_sim": args.model_type == "vision",
        "include_camera_obs": args.model_type == "vision",
        "include_last_action": args.include_last_action,
    }
    pacsim_args.update(REWARD_PARAMS[args.reward_mode])
    env = FrameStackObservation(pacsimEnv(pacsim_args), stack_size=3)
    training_condition_eval = args.training_episodes > 0
    all_results = {os.path.splitext(os.path.basename(path))[0]: [] for path in args.map_files}
    if args.track_dir:
        print(f'Evaluating {len(args.map_files)} tracks from {Path(args.track_dir).resolve()} without augmentation.')
        print('Normalized times are left blank for tracks without reference times.')
    per_seed_results = []
    seed_success_rates = []
    training_condition_returns = []

    for seed, model_path in zip(seeds, model_paths):
        print(f"Loading model: {model_path} (type={args.model_type}, seed={seed})")
        if args.dreamer:
            from dreamer_eval import DreamerEvaluationPolicy
            actor = DreamerEvaluationPolicy(model_path, env, args, device)
        else:
            actor = torch.load(model_path, weights_only=False).to(device)
            actor.eval()
        teacher = None
        inspection_dir = None
        if args.inspect_actions:
            teacher_path = args.inspection_teacher or state_actor_path(
                args.reward_mode, args.noisy, args.include_last_action, seed=seed)
            teacher = torch.load(teacher_path, map_location=device, weights_only=False).eval()
            inspection_dir = os.path.join('eval', f'inspection_{os.path.splitext(os.path.basename(model_path))[0]}_{time.time_ns()}')
            print(f'Inspection teacher: {teacher_path}; output: {inspection_dir}')
        results, episodic_returns = evaluate_actor(actor, env, args, device, teacher, inspection_dir)
        seed_success_rates.append(sum(success for success, _ in results.values()) / len(results))
        if training_condition_eval:
            successes = sum(success for success, _ in results.values())
            training_condition_returns.extend(episodic_returns)
            print(
                "Seed {seed} training-condition summary: passed={passed}, "
                "non-passed={non_passed}, total={total}, mean_episodic_return={mean_return:.6f}".format(
                    seed=seed,
                    passed=successes,
                    non_passed=len(results) - successes,
                    total=len(results),
                    mean_return=np.mean(episodic_returns),
                )
            )
            continue
        for track_name, result in results.items():
            all_results[track_name].append(result)
        all_track_metrics = track_set_metrics(results)
        last_four_metrics = track_set_metrics({track: results[track] for track in LAST_FOUR_TRACKS if track in results})
        print(
            "Seed {seed} summary: passed={passed}, non-passed={non_passed}, total={total}".format(
                seed=seed,
                passed=all_track_metrics["n_success"],
                non_passed=all_track_metrics["n_tracks"] - all_track_metrics["n_success"],
                total=all_track_metrics["n_tracks"],
            )
        )
        per_seed_results.append({
            "seed": seed,
            "all_n_success": all_track_metrics["n_success"],
            "all_n_tracks": all_track_metrics["n_tracks"],
            "all_success_rate": f'{all_track_metrics["success_rate"]:.6f}',
            "all_time_gmean": (
                f'{all_track_metrics["time_gmean"]:.6f}'
                if all_track_metrics["time_gmean"] is not None else ""
            ),
            "all_normalized_time_gmean": (
                f'{all_track_metrics["normalized_time_gmean"]:.6f}'
                if all_track_metrics["normalized_time_gmean"] is not None else ""
            ),
            "last4_n_success": last_four_metrics["n_success"],
            "last4_n_tracks": last_four_metrics["n_tracks"],
            "last4_success_rate": f'{last_four_metrics["success_rate"]:.6f}' if last_four_metrics['n_tracks'] else '',
            "last4_time_gmean": (
                f'{last_four_metrics["time_gmean"]:.6f}'
                if last_four_metrics["time_gmean"] is not None else ""
            ),
            "last4_normalized_time_gmean": (
                f'{last_four_metrics["normalized_time_gmean"]:.6f}'
                if last_four_metrics["normalized_time_gmean"] is not None else ""
            ),
        })

    if len(seed_success_rates) > 1:
        print(
            f'Multi-seed success rate: {100 * np.mean(seed_success_rates):.2f}% '
            f'+/- {100 * sample_stddev(seed_success_rates):.2f}% '
            f'(mean +/- sample stddev across {len(seed_success_rates)} seeds)'
        )

    if training_condition_eval:
        print(
            "Training-condition evaluation summary: mean_episodic_return={mean_return:.6f}, "
            "episodes={episodes}, checkpoints={checkpoints}".format(
                mean_return=np.mean(training_condition_returns),
                episodes=len(training_condition_returns),
                checkpoints=len(model_paths),
            )
        )
        env.close()
        return

    total_passed = sum(
        int(success)
        for track_results in all_results.values()
        for success, _ in track_results
    )
    total_evaluations = sum(len(track_results) for track_results in all_results.values())
    print(
        "Evaluation summary: passed={passed}, non-passed={non_passed}, total={total}".format(
            passed=total_passed,
            non_passed=total_evaluations - total_passed,
            total=total_evaluations,
        )
    )

    os.makedirs('eval', exist_ok=True)
    model_base = os.path.splitext(os.path.basename(model_paths[0]))[0]
    if len(seeds) > 1:
        model_base = re.sub(r'_seed_\d+$', '', model_base)
    if args.dreamer:
        model_base = f'{args.reward_mode}_last_action_{args.model_type}_dreamer'
        if len(seeds) == 1:
            model_base += f'_seed_{seeds[0]}'
    elif args.model_type == 'vision':
        # Default vision checkpoint names do not encode their encoder type.
        model_kind = 'latent' if hasattr(actor, 'vae') else 'structured'
        if model_kind not in model_base:
            model_base = f"{model_base}_{model_kind}"

    timestamp = int(time.time())
    if args.track_dir:
        model_base += '_tracks_' + re.sub(r'[^A-Za-z0-9_-]', '_', Path(args.track_dir).resolve().name)
    csv_fname = f"eval/test_{model_base}_{timestamp}.csv"
    per_seed_csv_fname = f"eval/seed_summary_{model_base}_{timestamp}.csv"
    with open(csv_fname, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Track",
            "success",
            "n_success",
            "n_seeds",
            "success_std",
            "time",
            "time_std",
            "time_reference",
            "normalized_time",
            "normalized_time_std",
        ])
        for track_name, track_results in all_results.items():
            successes = [float(success) for success, _ in track_results]
            successful_times = [lap_time for success, lap_time in track_results if success]
            successful_normalized_times = [
                normalized_time(track_name, lap_time) for lap_time in successful_times
                if track_name in TRACK_TIME_REFERENCES
            ]
            writer.writerow([
                track_name,
                f"{np.mean(successes):.6f}",
                int(sum(successes)),
                len(track_results),
                f"{sample_stddev(successes):.6f}",
                f"{np.mean(successful_times):.6f}" if successful_times else "",
                f"{sample_stddev(successful_times):.6f}" if successful_times else "",
                f"{TRACK_TIME_REFERENCES[track_name]:.6f}" if track_name in TRACK_TIME_REFERENCES else "",
                f"{np.mean(successful_normalized_times):.6f}" if successful_normalized_times else "",
                f"{sample_stddev(successful_normalized_times):.6f}" if successful_normalized_times else "",
            ])

    print(f"Writing evaluation results to: {csv_fname}")
    with open(per_seed_csv_fname, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "seed",
            "all_n_success",
            "all_n_tracks",
            "all_success_rate",
            "all_time_gmean",
            "all_normalized_time_gmean",
            "last4_n_success",
            "last4_n_tracks",
            "last4_success_rate",
            "last4_time_gmean",
            "last4_normalized_time_gmean",
        ])
        writer.writeheader()
        writer.writerows(per_seed_results)
    print(f"Writing per-seed success summary to: {per_seed_csv_fname}")


if __name__ == "__main__":
    main()
