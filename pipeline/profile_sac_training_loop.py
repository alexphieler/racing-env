import argparse
import random
import time
from collections import defaultdict

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import networks
from buffers import DictReplayBuffer
from networks import Actor, SoftQNetwork
from sac_continous_action import (
    copy_next_obs_with_final_obs,
    get_episode_stats_from_infos,
    iter_env_logs,
    make_vector_env,
)
from sac_continous_action import Args as SacArgs


class Timer:
    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, name, elapsed):
        self.totals[name] += elapsed
        self.counts[name] += 1

    def measure(self, name):
        timer = self

        class _Context:
            def __enter__(self):
                self.start = time.perf_counter()

            def __exit__(self, exc_type, exc, tb):
                timer.add(name, time.perf_counter() - self.start)

        return _Context()


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the SAC continuous-action training loop.")
    parser.add_argument("--total-timesteps", type=int, default=1200)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vector-env-mode", choices=["auto", "sync", "async"], default="sync")
    parser.add_argument("--reward-mode", choices=["conservative", "aggressive"], default="conservative")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--noise-augment", action="store_true")
    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def print_table(title, names, timer, denominator):
    print(f"\n{title}")
    print(f"{'section':34s} {'seconds':>12s} {'share':>9s} {'calls':>8s} {'ms/call':>10s}")
    print("-" * 78)
    for name in names:
        seconds = timer.totals.get(name, 0.0)
        calls = timer.counts.get(name, 0)
        share = 100.0 * seconds / denominator if denominator > 0 else 0.0
        ms_per_call = 1000.0 * seconds / calls if calls else 0.0
        print(f"{name:34s} {seconds:12.6f} {share:8.2f}% {calls:8d} {ms_per_call:10.3f}")


def main():
    cli_args = parse_args()
    args = SacArgs(
        seed=cli_args.seed,
        total_timesteps=cli_args.total_timesteps,
        num_envs=cli_args.num_envs,
        vector_env_mode=cli_args.vector_env_mode,
        verbose_env=False,
        print_step_status=False,
        buffer_size=cli_args.buffer_size,
        batch_size=cli_args.batch_size,
        learning_starts=cli_args.learning_starts,
        autotune=cli_args.autotune,
        noise_augment=cli_args.noise_augment,
        reward_mode=cli_args.reward_mode,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = "cpu"

    envs = make_vector_env(args)
    obs_space_filtered = networks.filterObservationForState(envs.single_observation_space)
    action_dim = int(np.prod(envs.single_action_space.shape))
    actor = Actor(obs_space_filtered, action_dim).to(device)
    qf1 = SoftQNetwork(obs_space_filtered, action_dim).to(device)
    qf2 = SoftQNetwork(obs_space_filtered, action_dim).to(device)
    qf1_target = SoftQNetwork(obs_space_filtered, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_space_filtered, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        target_entropy = None
        log_alpha = None
        alpha = args.alpha
        a_optimizer = None

    rb = DictReplayBuffer(
        args.buffer_size,
        gymnasium.spaces.Dict(networks.filterObservationForState(envs.single_observation_space)),
        envs.single_action_space,
        "cpu",
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    timer = Timer()
    with timer.measure("reset"):
        obs, reset_infos = envs.reset(seed=args.seed)
        for _ in iter_env_logs(reset_infos):
            pass

    loop_started = time.perf_counter()
    global_step = 0
    train_updates = 0
    actor_updates = 0

    while global_step < args.total_timesteps:
        iteration_started = time.perf_counter()

        with timer.measure("action_select"):
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                obs_flattened = networks.flattenFuncState(obs)
                actions, _, _ = actor.get_action(torch.Tensor(obs_flattened).to(device))
                actions = actions.detach().cpu().numpy()

        with timer.measure("noise_schedule"):
            steer_noise_base = 0.095
            torque_noise_base = 0.15
            target_max_multiplier = 1.0 if args.noise_augment else 0.0
            anneal_start_step = int(0.3 * args.total_timesteps)
            anneal_end_step = int(0.5 * args.total_timesteps)
            current_multiplier = 0.0

            if target_max_multiplier > 0.0 and global_step > anneal_start_step:
                if global_step >= anneal_end_step:
                    current_multiplier = target_max_multiplier
                else:
                    steps_elapsed = global_step - anneal_start_step
                    ramp_duration = anneal_end_step - anneal_start_step
                    current_multiplier = (steps_elapsed / ramp_duration) * target_max_multiplier

            if current_multiplier > 0:
                base_weights = np.array([steer_noise_base, torque_noise_base])
                current_sigma = base_weights * current_multiplier
                noise = np.random.normal(loc=0, scale=current_sigma, size=actions.shape)
                actions_intended = actions.copy()
                actions = np.clip(actions + noise, -1.0, 1.0)
            else:
                actions_intended = actions.copy()

        with timer.measure("env_step"):
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        with timer.measure("episode_info"):
            for _ in iter_env_logs(infos):
                pass
            episode_stats, episode_mask = get_episode_stats_from_infos(infos)
            if episode_stats is not None and episode_mask is None:
                episode_mask = np.ones(envs.num_envs, dtype=bool)

        with timer.measure("final_obs_copy"):
            dones = np.logical_or(terminations, truncations)
            real_next_obs = copy_next_obs_with_final_obs(next_obs, infos, dones)

        with timer.measure("replay_add"):
            rb.add(obs, real_next_obs, actions_intended, rewards, dones, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            train_updates += 1
            with timer.measure("train_total"):
                with timer.measure("replay_sample"):
                    data = rb.sample(args.batch_size)

                with timer.measure("flatten_batch"):
                    obs_flattened = networks.flattenFuncState(data.observations).to(device)
                    next_obs_flattened = networks.flattenFuncState(data.next_observations).to(device)

                with timer.measure("target_q_no_grad"):
                    with torch.no_grad():
                        next_state_actions, next_state_log_pi, _ = actor.get_action(next_obs_flattened)
                        qf1_next_target = qf1_target(next_obs_flattened, next_state_actions)
                        qf2_next_target = qf2_target(next_obs_flattened, next_state_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                        next_q_value = data.rewards.to(device).flatten() + (
                            1 - data.dones.to(device).flatten()
                        ) * args.gamma * min_qf_next_target.view(-1)

                with timer.measure("critic_forward_loss"):
                    data_actions = data.actions.to(device)
                    qf1_a_values = qf1(obs_flattened, data_actions).view(-1)
                    qf2_a_values = qf2(obs_flattened, data_actions).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                with timer.measure("critic_backward_step"):
                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(qf2.parameters(), max_norm=1.0)
                    q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    with timer.measure("actor_alpha_updates"):
                        for _ in range(args.policy_frequency):
                            actor_updates += 1
                            pi, log_pi, _ = actor.get_action(obs_flattened)
                            qf1_pi = qf1(obs_flattened, pi)
                            qf2_pi = qf2(obs_flattened, pi)
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)
                            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                            actor_optimizer.step()

                            if args.autotune:
                                with torch.no_grad():
                                    _, log_pi, _ = actor.get_action(obs_flattened)
                                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                                a_optimizer.zero_grad()
                                alpha_loss.backward()
                                a_optimizer.step()
                                alpha = log_alpha.exp().item()

                if global_step % args.target_network_frequency == 0:
                    with timer.measure("target_network_update"):
                        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        timer.add("loop_iteration_total", time.perf_counter() - iteration_started)
        global_step += args.num_envs

    loop_elapsed = time.perf_counter() - loop_started
    envs.close()

    top_level = [
        "action_select",
        "noise_schedule",
        "env_step",
        "episode_info",
        "final_obs_copy",
        "replay_add",
        "train_total",
    ]
    accounted = sum(timer.totals[name] for name in top_level)
    timer.totals["other_loop_overhead"] = max(loop_elapsed - accounted, 0.0)
    timer.counts["other_loop_overhead"] = timer.counts["loop_iteration_total"]
    top_level.append("other_loop_overhead")

    train_parts = [
        "replay_sample",
        "flatten_batch",
        "target_q_no_grad",
        "critic_forward_loss",
        "critic_backward_step",
        "actor_alpha_updates",
        "target_network_update",
    ]
    train_accounted = sum(timer.totals[name] for name in train_parts)
    timer.totals["train_other_overhead"] = max(timer.totals["train_total"] - train_accounted, 0.0)
    timer.counts["train_other_overhead"] = train_updates
    train_parts.append("train_other_overhead")

    print("\nSAC training-loop profiling run")
    print(f"timesteps={args.total_timesteps}, num_envs={args.num_envs}, learning_starts={args.learning_starts}")
    print(f"train_updates={train_updates}, actor_updates={actor_updates}, batch_size={args.batch_size}")
    print(f"reset_seconds={timer.totals['reset']:.6f}, loop_seconds={loop_elapsed:.6f}")
    print(f"env_steps_per_second={args.total_timesteps / loop_elapsed:.2f}")

    print_table("Top-level loop time", top_level, timer, loop_elapsed)
    if timer.totals["train_total"] > 0:
        print_table("Training-update time breakdown", train_parts, timer, timer.totals["train_total"])


if __name__ == "__main__":
    main()
