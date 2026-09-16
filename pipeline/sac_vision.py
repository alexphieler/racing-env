# Vision SAC with privileged state critics for PacSim continuous actions.
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gymnasium.wrappers import FrameStackObservation
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from artifact_names import (
    state_qf1_path as default_state_qf1_path,
    state_qf2_path as default_state_qf2_path,
    vae_path as default_vae_path,
    vision_sac_actor_path,
    vision_sac_qf1_path,
    vision_sac_qf2_path,
)
from buffers import FrameStackDictReplayBuffer
import networks
from networks import SoftQNetwork
from pacsimEnv import pacsimEnv
from sac_continous_action import (
    EnvLogWrapper,
    copy_next_obs_with_final_obs,
    get_episode_stats_from_infos,
    iter_env_logs,
    maybe_silence,
)

os.environ.pop("NO_COLOR", None)
os.environ.pop("ANSI_COLORS_DISABLED", None)
os.environ.setdefault("FORCE_COLOR", "1")


@dataclass
class Args:
    exp_name: str = "sac_vision"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, CUDA will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb project name"""
    wandb_entity: str | None = None
    """the wandb entity"""
    log_dir: str = "runs"
    """base directory for TensorBoard logs"""

    env_id: str = "pacsimEnv"
    """the environment id of the task"""
    vision_actor: Literal["structured", "latent"] = "structured"
    """vision actor architecture"""
    vae_path: str | None = None
    """path to pretrained VAE weights for the latent vision actor"""
    unfreeze_vae: bool = False
    """allow fine-tuning VAE weights for the latent vision actor"""
    frame_stack: int = 3
    """the number of frames stacked in each observation"""
    num_envs: int = 4
    """the number of parallel game environments"""
    vector_env_mode: Literal["auto", "sync", "async"] = "auto"
    """the vector environment backend"""
    async_shared_memory: bool = True
    """use shared memory for AsyncVectorEnv observations"""
    verbose_env: bool = True
    """allow pacsimEnv stdout/stderr messages during training"""
    print_step_status: bool = False
    """print pacsimEnv periodic per-100-step status lines"""
    include_last_action: bool = True
    """include the previous normalized action in observations"""
    reward_mode: Literal["conservative", "aggressive"] = "conservative"
    """the reward mode for the environment"""
    noise_augment: bool = False
    """apply action noise augmentation while collecting transitions"""

    total_timesteps: int = 1_000_000
    """total timesteps of the experiment"""
    vision_buffer_size: int = 5_000
    """replay capacity; stacked full-resolution camera observations require substantial RAM"""
    vision_batch_size: int = 16
    """replay batch size for full-resolution vision observations"""
    asymmetric_critic: bool = False
    """use a privileged state-observation critic instead of a vision critic"""
    preload_state_critic: bool = False
    """initialize asymmetric critics from matching state-SAC Q checkpoints"""
    preload_state_qf1_path: str | None = None
    """override the state-SAC QF1 checkpoint used for critic preloading"""
    preload_state_qf2_path: str | None = None
    """override the state-SAC QF2 checkpoint used for critic preloading"""
    gamma: float = 0.99
    """the discount factor"""
    tau: float = 0.005
    """target smoothing coefficient"""
    learning_starts: int = 10_000
    """timestep to start learning"""
    freeze_critic_steps: int = 0
    """environment steps after learning starts during which only the actor is updated"""
    policy_lr: float = 3e-4
    """the policy learning rate"""
    q_lr: float = 1e-3
    """the Q-network learning rate"""
    policy_frequency: int = 2
    """the delayed policy-update frequency"""
    target_network_frequency: int = 1
    """the target-network update frequency"""
    alpha: float = 0.2
    """entropy regularization coefficient"""
    autotune: bool = True
    """automatically tune the entropy coefficient"""


def make_env(seed, idx, args, run_name):
    def thunk():
        pacsim_args = {
            "cam_sim": True,
            "print_step_status": args.print_step_status,
            "include_camera_obs": True,
            "include_last_action": args.include_last_action,
        }
        if args.reward_mode == "aggressive":
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
        else:
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

        with maybe_silence(not args.verbose_env):
            pacsim = pacsimEnv(pacsim_args)
        env = EnvLogWrapper(pacsim, emit_logs=args.verbose_env)
        env = FrameStackObservation(env, stack_size=args.frame_stack)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def resolve_vector_env_mode(args):
    use_async = args.vector_env_mode == "async" or (
        args.vector_env_mode == "auto" and args.num_envs > 1
    )
    return "async" if use_async else "sync"


def make_vector_env(args, run_name):
    env_fns = [make_env(args.seed + index, index, args, run_name) for index in range(args.num_envs)]
    autoreset_mode = gym.vector.AutoresetMode.SAME_STEP
    if resolve_vector_env_mode(args) == "sync":
        return gym.vector.SyncVectorEnv(env_fns, autoreset_mode=autoreset_mode)
    return gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=args.async_shared_memory,
        autoreset_mode=autoreset_mode,
    )


def load_vae(args, vision_input_shape):
    vae_path = args.vae_path or default_vae_path(
        args.reward_mode,
        args.noise_augment,
        args.include_last_action,
    )
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"Expected VAE at exactly: {vae_path}")

    vae = networks.ConvVAE(input_spatial=vision_input_shape[-2:])
    checkpoint = torch.load(vae_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        vae = checkpoint
    else:
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "vae"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
        vae.load_state_dict(state_dict)
    print(f"Loaded VAE from: {vae_path}")
    return vae


class VisionSACActor(nn.Module):
    """A PPO-vision-compatible actor with SAC's three-value action API."""

    def __init__(self, obs_space, action_space, args, device):
        super().__init__()
        self.device = device
        self.vision_actor = args.vision_actor
        self.vision_input_shape = self._vision_input_shape(obs_space)
        self.sensor_dim = networks.vision_sensor_dim(obs_space)
        action_dim = int(np.prod(action_space.shape))
        if self.vision_actor == "latent":
            self.policy = networks.LatentHierarchicalActor(
                load_vae(args, self.vision_input_shape),
                self.vision_input_shape,
                self.sensor_dim,
                action_dim,
                freeze_vae=not args.unfreeze_vae,
            )
        else:
            self.policy = networks.StructuredHierarchicalActor(
                self.vision_input_shape,
                self.sensor_dim,
                action_dim,
            )

    @staticmethod
    def _vision_input_shape(obs_space):
        camera_keys = ("cameraLeft", "cameraFront", "cameraRight")
        missing = [key for key in camera_keys if key not in obs_space.spaces]
        if missing:
            raise KeyError(f"Vision policy requires camera observations, missing: {missing}")
        camera_shape = obs_space["cameraFront"].shape
        if len(camera_shape) != 4:
            raise ValueError(f"Expected stacked cameraFront shape (T, C, H, W), got {camera_shape}")
        return (camera_shape[0], len(camera_keys), camera_shape[1], camera_shape[2], camera_shape[3])

    def get_action(self, obs):
        vision, sensors = networks.flattenFuncVision(obs)
        action, log_prob, _, mean = self.policy.get_action(vision.to(self.device), sensors.to(self.device))
        return action, log_prob, mean


class VisionSoftQNetwork(nn.Module):
    """A vision-conditioned Q network for symmetric SAC training."""

    def __init__(self, obs_space, action_space, args, device):
        super().__init__()
        self.device = device
        vision_input_shape = VisionSACActor._vision_input_shape(obs_space)
        sensor_dim = networks.vision_sensor_dim(obs_space)
        action_dim = int(np.prod(action_space.shape))
        self.is_latent = args.vision_actor == "latent"
        if self.is_latent:
            self.encoder = networks.LatentHierarchicalActor(
                load_vae(args, vision_input_shape),
                vision_input_shape,
                sensor_dim,
                action_dim,
                freeze_vae=not args.unfreeze_vae,
            )
        else:
            self.encoder = networks.StructuredHierarchicalActor(
                vision_input_shape,
                sensor_dim,
                action_dim,
            )
        self.fc1 = nn.Linear(2 * action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        vision, sensors = networks.flattenFuncVision(obs)
        vision = vision.to(self.device)
        sensors = sensors.to(self.device)
        # get_action applies the correct image normalization for both the
        # structured and latent encoders; its deterministic output is used as
        # the visual feature for the Q head.
        _, _, _, visual_feature = self.encoder.get_action(vision, sensors)
        x = torch.cat((visual_feature, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def validate_args(args):
    if args.num_envs < 1:
        raise ValueError("num_envs must be at least 1")
    if args.frame_stack < 1:
        raise ValueError("frame_stack must be at least 1")
    if args.vision_buffer_size < 1:
        raise ValueError("vision_buffer_size must be at least 1")
    if args.vision_batch_size < 1:
        raise ValueError("vision_batch_size must be at least 1")
    if args.preload_state_critic and not args.asymmetric_critic:
        raise ValueError("--preload-state-critic requires --asymmetric-critic")
    if (args.preload_state_qf1_path or args.preload_state_qf2_path) and not args.preload_state_critic:
        raise ValueError("--preload-state-qf1-path and --preload-state-qf2-path require --preload-state-critic")
    if args.freeze_critic_steps < 0:
        raise ValueError("freeze_critic_steps must be non-negative")
    if args.freeze_critic_steps and not args.asymmetric_critic:
        raise ValueError("--freeze-critic-steps requires --asymmetric-critic")


def build_run_name(args):
    env_id = args.env_id
    if args.noise_augment:
        env_id += "_noiseAugmented"
    if args.include_last_action:
        env_id += "_lastActionObs"
    env_id += f"_{args.vision_actor}Vision"
    if args.asymmetric_critic:
        env_id += "_asymmetricCritic"
    return f"{env_id}_{args.reward_mode}__{args.exp_name}__{args.seed}__{int(time.time())}"


def replay_buffer_memory_gib(obs_space, buffer_size):
    """Estimated persistent storage for raw episode frames, not eager stacks."""
    observation_bytes = sum(
        int(np.prod(space.shape[1:])) * np.dtype(space.dtype).itemsize
        for space in obs_space.spaces.values()
    )
    return observation_bytes * buffer_size / 1024**3


def action_noise_multiplier(args, global_step):
    if not args.noise_augment:
        return 0.0
    start = int(0.3 * args.total_timesteps)
    end = int(0.5 * args.total_timesteps)
    if global_step <= start:
        return 0.0
    if global_step >= end:
        return 1.0
    return (global_step - start) / max(1, end - start)


def apply_action_noise(actions, args, global_step):
    multiplier = action_noise_multiplier(args, global_step)
    if multiplier <= 0:
        return actions.astype(np.float32, copy=False), multiplier
    base_weights = np.array([0.095, 0.15, 0.15, 0.15, 0.15], dtype=np.float32)
    noise = np.random.normal(0.0, base_weights * multiplier, size=actions.shape)
    return np.clip(actions + noise, -1.0, 1.0).astype(np.float32), multiplier


def critic_value(critic, obs, state, actions, asymmetric_critic):
    if asymmetric_critic:
        return critic(state, actions)
    return critic(obs, actions)


def set_critic_requires_grad(critics, requires_grad):
    for critic in critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(requires_grad)


def load_critic_checkpoint(critic, path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected {label} checkpoint at exactly: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.state_dict() if isinstance(checkpoint, nn.Module) else checkpoint
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    critic.load_state_dict(state_dict)
    print(f"Loaded {label} from: {path}")


def main():
    args = tyro.cli(Args)
    validate_args(args)
    run_name = build_run_name(args)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    log_dir = Path(args.log_dir).expanduser() / run_name
    print(f"TensorBoard log directory: {log_dir}", flush=True)
    writer = SummaryWriter(str(log_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    envs = make_vector_env(args, run_name)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print(args)
    print(f"vector_env_mode: {resolve_vector_env_mode(args)}, num_envs: {envs.num_envs}")
    print(f"device: {device}")

    state_obs_space = gym.spaces.Dict(networks.filterObservationForState(envs.single_observation_space))
    actor = VisionSACActor(envs.single_observation_space, envs.single_action_space, args, device).to(device)
    if args.asymmetric_critic:
        qf1 = SoftQNetwork(state_obs_space).to(device)
        qf2 = SoftQNetwork(state_obs_space).to(device)
        qf1_target = SoftQNetwork(state_obs_space).to(device)
        qf2_target = SoftQNetwork(state_obs_space).to(device)
        critic_description = "state (asymmetric)"
    else:
        qf1 = VisionSoftQNetwork(envs.single_observation_space, envs.single_action_space, args, device).to(device)
        qf2 = VisionSoftQNetwork(envs.single_observation_space, envs.single_action_space, args, device).to(device)
        qf1_target = VisionSoftQNetwork(envs.single_observation_space, envs.single_action_space, args, device).to(device)
        qf2_target = VisionSoftQNetwork(envs.single_observation_space, envs.single_action_space, args, device).to(device)
        critic_description = "vision (symmetric)"
    print(f"critic: {critic_description}")
    if args.preload_state_critic:
        qf1_path = args.preload_state_qf1_path or default_state_qf1_path(
            args.reward_mode,
            args.noise_augment,
            args.include_last_action,
            seed=args.seed,
        )
        qf2_path = args.preload_state_qf2_path or default_state_qf2_path(
            args.reward_mode,
            args.noise_augment,
            args.include_last_action,
            seed=args.seed,
        )
        load_critic_checkpoint(qf1, qf1_path, "state-SAC QF1")
        load_critic_checkpoint(qf2, qf2_path, "state-SAC QF2")
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam([*qf1.parameters(), *qf2.parameters()], lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    memory_gib = replay_buffer_memory_gib(envs.single_observation_space, args.vision_buffer_size)
    print(
        f"Vision replay buffer: {args.vision_buffer_size:,} transitions, approximately "
        f"{memory_gib:.2f} GiB for raw episode frames (stacks are reconstructed when sampled); "
        f"training batch size: {args.vision_batch_size}.",
        flush=True,
    )
    replay_buffer = FrameStackDictReplayBuffer(
        args.vision_buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        n_envs=args.num_envs,
        frame_stack=args.frame_stack,
        handle_timeout_termination=False,
    )
    writer.add_text(
        "model/input",
        (
            f"vision_actor={args.vision_actor}, vision_shape={actor.vision_input_shape}, "
            f"sensor_dim={actor.sensor_dim}, critic={critic_description}, replay_size={args.vision_buffer_size}, "
            f"batch_size={args.vision_batch_size}"
        ),
    )

    obs, reset_infos = envs.reset(seed=args.seed)
    for env_log in iter_env_logs(reset_infos):
        print(env_log)
    progress_bar = tqdm(total=args.total_timesteps, unit="env-step", dynamic_ncols=True)
    global_step = 0
    start_time = time.time()
    actor_loss = None
    alpha_loss = None

    while global_step < args.total_timesteps:
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(obs)
            actions = actions.cpu().numpy()
        actions, noise_multiplier = apply_action_noise(actions, args, global_step)
        if global_step % 1000 == 0:
            writer.add_scalar("debug/noise_multiplier", noise_multiplier, global_step)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        for env_log in iter_env_logs(infos):
            progress_bar.write(env_log)
        episode_stats, episode_mask = get_episode_stats_from_infos(infos)
        if episode_stats is not None:
            if episode_mask is None:
                episode_mask = np.ones(envs.num_envs, dtype=bool)
            for index, finished in enumerate(episode_mask):
                if finished:
                    progress_bar.write(
                        f"global_step={global_step}, episodic_return={episode_stats['r'][index]}, "
                        f"episodic_length={episode_stats['l'][index]}"
                    )
                    writer.add_scalar("charts/episodic_return", episode_stats["r"][index], global_step)
                    writer.add_scalar("charts/episodic_length", episode_stats["l"][index], global_step)
                    break

        dones = np.logical_or(terminations, truncations)
        replay_buffer.add(obs, copy_next_obs_with_final_obs(next_obs, infos, dones), actions, rewards, dones, infos)
        obs = next_obs

        collected_steps = min(global_step + envs.num_envs, args.total_timesteps)
        elapsed = time.time() - start_time
        env_sps = int(collected_steps / elapsed) if elapsed > 0 else 0
        progress_bar.update(collected_steps - global_step)
        progress_bar.set_postfix(env_sps=env_sps)
        if global_step % 100 == 0:
            writer.add_scalar("charts/env_SPS", env_sps, collected_steps)

        if global_step > args.learning_starts:
            data = replay_buffer.sample(args.vision_batch_size)
            state = networks.flattenFuncState(data.observations).to(device)
            critic_frozen = global_step < args.learning_starts + args.freeze_critic_steps
            qf1_values = qf2_values = qf1_loss = qf2_loss = q_loss = None
            if not critic_frozen:
                next_state = networks.flattenFuncState(data.next_observations).to(device)
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(data.next_observations)
                    next_q = torch.min(
                        critic_value(qf1_target, data.next_observations, next_state, next_actions, args.asymmetric_critic),
                        critic_value(qf2_target, data.next_observations, next_state, next_actions, args.asymmetric_critic),
                    ) - alpha * next_log_pi
                    target_q = data.rewards.to(device).flatten() + (
                        1 - data.dones.to(device).flatten()
                    ) * args.gamma * next_q.view(-1)

                qf1_values = critic_value(
                    qf1, data.observations, state, data.actions.to(device), args.asymmetric_critic
                ).view(-1)
                qf2_values = critic_value(
                    qf2, data.observations, state, data.actions.to(device), args.asymmetric_critic
                ).view(-1)
                qf1_loss = F.mse_loss(qf1_values, target_q)
                qf2_loss = F.mse_loss(qf2_values, target_q)
                q_loss = qf1_loss + qf2_loss
                q_optimizer.zero_grad()
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(qf1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(qf2.parameters(), 1.0)
                q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                set_critic_requires_grad((qf1, qf2), False)
                try:
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = actor.get_action(data.observations)
                        min_q_pi = torch.min(
                            critic_value(qf1, data.observations, state, pi, args.asymmetric_critic),
                            critic_value(qf2, data.observations, state, pi, args.asymmetric_critic),
                        )
                        actor_loss = (alpha * log_pi - min_q_pi).mean()
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                            alpha_optimizer.zero_grad()
                            alpha_loss.backward()
                            alpha_optimizer.step()
                            alpha = log_alpha.exp().item()
                finally:
                    set_critic_requires_grad((qf1, qf2), True)

            if not critic_frozen and global_step % args.target_network_frequency == 0:
                for source, target in zip(qf1.parameters(), qf1_target.parameters()):
                    target.data.copy_(args.tau * source.data + (1 - args.tau) * target.data)
                for source, target in zip(qf2.parameters(), qf2_target.parameters()):
                    target.data.copy_(args.tau * source.data + (1 - args.tau) * target.data)

            if global_step % 100 == 0:
                writer.add_scalar("charts/critic_frozen", float(critic_frozen), global_step)
                if q_loss is not None:
                    writer.add_scalar("losses/qf1_values", qf1_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", q_loss.item() / 2.0, global_step)
                if actor_loss is not None:
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                if alpha_loss is not None:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        global_step += envs.num_envs

    progress_bar.close()
    envs.close()
    writer.close()

    actor_path = vision_sac_actor_path(
        args.reward_mode,
        args.noise_augment,
        args.vision_actor == "latent",
        args.include_last_action,
        seed=args.seed,
        asymmetric=args.asymmetric_critic,
    )
    qf1_path = vision_sac_qf1_path(
        args.reward_mode,
        args.noise_augment,
        args.vision_actor == "latent",
        args.include_last_action,
        seed=args.seed,
        asymmetric=args.asymmetric_critic,
    )
    qf2_path = vision_sac_qf2_path(
        args.reward_mode,
        args.noise_augment,
        args.vision_actor == "latent",
        args.include_last_action,
        seed=args.seed,
        asymmetric=args.asymmetric_critic,
    )
    os.makedirs(os.path.dirname(actor_path), exist_ok=True)
    torch.save(actor.policy, actor_path)
    torch.save(qf1, qf1_path)
    torch.save(qf2, qf2_path)
    print(f"Saved actor to: {actor_path}")
    print(f"Saved qf1 to: {qf1_path}")
    print(f"Saved qf2 to: {qf2_path}")


if __name__ == "__main__":
    main()
