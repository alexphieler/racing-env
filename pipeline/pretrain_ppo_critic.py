"""Warm-start PPO's state critic with PPO-matching GAE rollout targets."""

import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from artifact_names import ppo_critic_path
import ppo_vision as ppo


def optional_float(value):
    return None if value.lower() == "none" else float(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Warm-start PPO's state critic using the same GAE targets as ppo_vision.py."
    )
    parser.add_argument("--actor-path", required=True, help="BC, DAgger, or PPO vision actor checkpoint")
    parser.add_argument("--critic-path", default=None, help="Output critic checkpoint")
    parser.add_argument("--vision-actor", choices=("structured", "latent"), default="structured")
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--reward-mode", choices=("conservative", "aggressive"), default="conservative")
    parser.add_argument("--noisy", action="store_true", help="Use the noisy artifact-name variant")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed and default output artifact seed")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--include-last-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--actor-log-std", type=optional_float, default=-2.0,
        help="Constant log std after mean-only BC/DAgger load; use none to preserve PPO values.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-steps", type=int, default=2048, help="Steps per PPO-matching rollout")
    parser.add_argument("--rollout-iterations", type=int, default=5, help="Collect/update cycles")
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=20, help="Critic-only epochs per rollout")
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="State-only critic batch size; vision features are not retained during updates.",
    )
    parser.add_argument("--verbose-env", action="store_true")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    positive = ("num_envs", "frame_stack", "num_steps", "rollout_iterations", "update_epochs", "batch_size")
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("--num-envs, --frame-stack, --num-steps, --rollout-iterations, --update-epochs, and --batch-size must be positive")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("--gamma must be in (0, 1] and --gae-lambda must be in [0, 1]")
    if args.critic_learning_rate <= 0.0:
        parser.error("--critic-learning-rate must be positive")
    return args


def build_rollout_args(args):
    return ppo.Args(
        seed=args.seed,
        cuda=args.cuda,
        policy_type="vision",
        vision_actor=args.vision_actor,
        vae_path=args.vae_path,
        frame_stack=args.frame_stack,
        num_envs=args.num_envs,
        verbose_env=args.verbose_env,
        include_last_action=args.include_last_action,
        reward_mode=args.reward_mode,
    )


def explained_variance(predictions, targets):
    target_variance = torch.var(targets, unbiased=False)
    if target_variance <= 0:
        return float("nan")
    return (1.0 - torch.var(targets - predictions, unbiased=False) / target_variance).item()


def collect_gae_rollout(agent, envs, obs, next_done, args, device):
    """Collect one rollout and calculate exactly the GAE returns used by PPO."""
    state_batches = []
    rewards = torch.zeros((args.num_steps, envs.num_envs), device=device)
    dones = torch.zeros((args.num_steps, envs.num_envs), device=device)
    values = torch.zeros((args.num_steps, envs.num_envs), device=device)

    for step in range(args.num_steps):
        dones[step] = next_done
        with torch.no_grad():
            state_batch = agent._state_tensor(obs)
            mean, log_std = agent._policy_forward(obs)
            actions, _, _, _ = ppo.tanh_normal_action(mean, log_std)
            values[step] = agent.critic(state_batch).flatten()
        state_batches.append(state_batch.cpu())

        obs, reward, terminations, truncations, _ = envs.step(actions.cpu().numpy())
        rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
        next_done = torch.as_tensor(np.logical_or(terminations, truncations), dtype=torch.float32, device=device)

    with torch.no_grad():
        next_value = agent.get_value(obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for step in reversed(range(args.num_steps)):
            if step == args.num_steps - 1:
                next_nonterminal = 1.0 - next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - dones[step + 1]
                next_values = values[step + 1]
            delta = rewards[step] + args.gamma * next_values * next_nonterminal - values[step]
            advantages[step] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            )
        returns = advantages + values

    return torch.cat(state_batches), returns.reshape(-1).cpu(), values.reshape(-1).cpu(), obs, next_done


def update_critic(critic, optimizer, states, returns, args, device):
    """Fit only the state critic, allowing a batch size independent of the vision actor."""
    indices = torch.arange(len(returns))
    critic.train()
    for _ in range(args.update_epochs):
        shuffled_indices = indices[torch.randperm(len(indices))]
        for start in range(0, len(shuffled_indices), args.batch_size):
            batch_indices = shuffled_indices[start:start + args.batch_size]
            predictions = critic(states[batch_indices].to(device)).view(-1)
            loss = 0.5 * ((predictions - returns[batch_indices].to(device)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()

    critic.eval()
    with torch.no_grad():
        predictions = critic(states.to(device)).view(-1).cpu()
        value_loss = 0.5 * ((predictions - returns) ** 2).mean().item()
        return value_loss, explained_variance(predictions, returns)


def main():
    args = parse_args()
    output_path = args.critic_path or ppo_critic_path(
        args.reward_mode, args.noisy, args.vision_actor == "latent", args.include_last_action, seed=args.seed
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.cuda:
        warnings.filterwarnings("ignore", message="CUDA initialization:.*")
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    rollout_args = build_rollout_args(args)
    envs = ppo.make_vector_env(rollout_args, "pretrain_ppo_critic")
    try:
        agent = ppo.PPOAgent(envs.single_observation_space, envs.single_action_space, rollout_args, device).to(device)
        ppo.load_pretrained_actor(agent.actor, args.actor_path)
        ppo.reset_pretrained_actor_log_std(agent.actor, args.actor_log_std)
        agent.actor.eval()
        for parameter in agent.actor.parameters():
            parameter.requires_grad_(False)

        obs, _ = envs.reset(seed=args.seed)
        next_done = torch.zeros(envs.num_envs, device=device)
        optimizer = optim.Adam(agent.critic.parameters(), lr=args.critic_learning_rate, eps=1e-5)
        print(f"Pretraining PPO-matching GAE critic on {device}")
        for iteration in range(1, args.rollout_iterations + 1):
            states, returns, old_values, obs, next_done = collect_gae_rollout(
                agent, envs, obs, next_done, args, device
            )
            pre_ev = explained_variance(old_values, returns)
            value_loss, post_ev = update_critic(agent.critic, optimizer, states, returns, args, device)
            print(
                f"rollout={iteration}/{args.rollout_iterations}, targets={len(returns)}, "
                f"pre_update_explained_variance={pre_ev:.6f}, value_loss={value_loss:.6f}, "
                f"post_update_explained_variance={post_ev:.6f}"
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(
            {
                "critic": agent.critic.state_dict(),
                "metadata": {
                    "actor_path": args.actor_path,
                    "actor_log_std": args.actor_log_std,
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "num_steps": args.num_steps,
                    "rollout_iterations": args.rollout_iterations,
                    "reward_mode": args.reward_mode,
                    "include_last_action": args.include_last_action,
                    "frame_stack": args.frame_stack,
                },
            },
            output_path,
        )
        print(f"Saved PPO-matching pretrained critic to: {output_path}")
    finally:
        envs.close()


if __name__ == "__main__":
    main()
