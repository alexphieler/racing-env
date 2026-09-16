# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import io
import os
import random
import time
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium.wrappers import FrameStackObservation
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import networks
from artifact_names import (
    ppo_actor_path,
    ppo_critic_path,
    state_ppo_actor_path,
    state_ppo_critic_path,
    vae_path as default_vae_path,
)
from pacsimEnv import pacsimEnv

os.environ.pop("NO_COLOR", None)
os.environ.pop("ANSI_COLORS_DISABLED", None)
os.environ.setdefault("FORCE_COLOR", "1")

ACTION_DIM = 5
LOG_PROB_EPS = 1e-6


@dataclass
class Args:
    exp_name: str = "ppo_vision"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """save the trained actor into the pipeline networks directory"""

    # Environment and model arguments
    env_id: str = "pacsimEnv"
    """the environment id of the task"""
    policy_type: Literal["vision", "state"] = "vision"
    """train a vision actor or a state actor"""
    state_based_actor: bool = False
    """shortcut for --policy-type state"""
    vision_actor: Literal["structured", "latent"] = "structured"
    """vision actor architecture to use when policy_type=vision"""
    pretrained_actor_path: str | None = None
    """Optional BC, DAgger, or PPO vision-actor checkpoint used to initialize PPO fine-tuning"""
    pretrained_critic_path: str | None = None
    """Optional state-value critic checkpoint, e.g. from pretrain_ppo_critic.py"""
    actor_learning_rate: float | None = None
    """Actor learning rate; defaults to learning_rate when omitted"""
    critic_learning_rate: float | None = None
    """Critic learning rate; defaults to learning_rate when omitted"""
    pretrained_actor_log_std: float | None = -2.0
    """Initial log standard deviation after loading a mean-only BC/DAgger actor; None preserves checkpoint values"""
    vae_path: str | None = None
    """path to pretrained VAE weights for the latent vision actor"""
    unfreeze_vae: bool = False
    """allow fine-tuning of the VAE weights for the latent vision actor"""
    frame_stack: int = 3
    """the number of frames stacked in each observation"""
    num_envs: int = 1
    """the number of parallel game environments"""
    vector_env_mode: Literal["auto", "sync", "async"] = "auto"
    """the vector environment backend: auto uses async for multiple envs, sync for one env"""
    async_shared_memory: bool = True
    """use shared memory for AsyncVectorEnv observations"""
    verbose_env: bool = True
    """allow pacsimEnv stdout/stderr messages during training"""
    print_step_status: bool = False
    """print pacsimEnv periodic per-100-step status lines"""
    include_camera_obs: bool = False
    """include camera image observations; forced on for vision policies"""
    include_last_action: bool = True
    """include the previous normalized action in the state/vision observation"""
    reward_mode: Literal["conservative", "aggressive"] = "conservative"
    """the reward mode for the environment (conservative or aggressive)"""
    noise_augment: bool = False
    """apply SAC-style action noise augmentation while collecting rollouts"""

    # PPO arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 128
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """toggles clipped value loss"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    critic_warmup_iterations: int = 0
    """Initial PPO rollout iterations used only to fit the critic to returns from the fixed actor"""
    critic_warmup_epochs: int | None = None
    """Value-only epochs per warmup rollout; defaults to update_epochs"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


@contextmanager
def maybe_silence(enabled):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


class EnvLogWrapper(gym.Wrapper):
    def __init__(self, env, emit_logs):
        super().__init__(env)
        self.emit_logs = emit_logs

    def _logged_call(self, func, *args, **kwargs):
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            result = func(*args, **kwargs)
        return result, buffer.getvalue() if self.emit_logs else ""

    def reset(self, **kwargs):
        (obs, info), output = self._logged_call(self.env.reset, **kwargs)
        if output:
            info = dict(info)
            info["env_log"] = output
        return obs, info

    def step(self, action):
        (obs, reward, terminated, truncated, info), output = self._logged_call(
            self.env.step,
            action,
        )
        if output:
            info = dict(info)
            info["env_log"] = output
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()


def make_env(seed, idx, args, run_name):
    def thunk():
        use_vision = args.policy_type == "vision"
        pacsim_args = {
            "cam_sim": use_vision,
            "print_step_status": args.print_step_status,
            "include_camera_obs": use_vision or args.include_camera_obs,
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
        elif args.reward_mode == "conservative":
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
        if args.capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        return env

    return thunk


def resolve_vector_env_mode(args):
    use_async = args.vector_env_mode == "async" or (
        args.vector_env_mode == "auto" and args.num_envs > 1
    )
    return "async" if use_async else "sync"


def make_vector_env(args, run_name):
    env_fns = [
        make_env(args.seed + i, i, args, run_name)
        for i in range(args.num_envs)
    ]
    vector_env_mode = resolve_vector_env_mode(args)
    autoreset_mode = gym.vector.AutoresetMode.SAME_STEP
    if vector_env_mode == "sync":
        return gym.vector.SyncVectorEnv(env_fns, autoreset_mode=autoreset_mode)
    return gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=args.async_shared_memory,
        autoreset_mode=autoreset_mode,
    )


def get_episode_stats_from_infos(infos):
    episode_stats = infos.get("episode")
    episode_mask = infos.get("_episode")
    if episode_stats is None and isinstance(infos.get("final_info"), dict):
        final_info = infos["final_info"]
        episode_stats = final_info.get("episode")
        episode_mask = final_info.get("_episode")
    return episode_stats, episode_mask


def iter_env_logs(infos):
    log_sources = []
    if "env_log" in infos:
        log_sources.append(infos["env_log"])
    if isinstance(infos.get("final_info"), dict) and "env_log" in infos["final_info"]:
        log_sources.append(infos["final_info"]["env_log"])

    for source in log_sources:
        if isinstance(source, np.ndarray):
            for item in source:
                if item:
                    yield str(item).rstrip()
        elif source:
            yield str(source).rstrip()


def init_obs_storage(num_steps, num_envs, obs_space):
    return {
        key: np.zeros((num_steps, num_envs, *space.shape), dtype=space.dtype)
        for key, space in obs_space.spaces.items()
    }


def store_obs(storage, step, obs):
    for key in storage:
        storage[key][step] = np.asarray(obs[key])


def flatten_obs_storage(storage):
    return {
        key: value.reshape((-1, *value.shape[2:]))
        for key, value in storage.items()
    }


def index_obs(obs, indices):
    return {key: value[indices] for key, value in obs.items()}


def tanh_normal_action(mean, log_std, action=None):
    std = log_std.exp()
    normal = Normal(mean, std)

    if action is None:
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        action_for_log_prob = action
    else:
        action = action.to(device=mean.device, dtype=mean.dtype)
        action_for_log_prob = action.clamp(-1.0 + LOG_PROB_EPS, 1.0 - LOG_PROB_EPS)
        raw_action = torch.atanh(action_for_log_prob)

    log_prob = normal.log_prob(raw_action)
    log_prob -= torch.log(1.0 - action_for_log_prob.pow(2) + LOG_PROB_EPS)
    log_prob = log_prob.sum(1)
    entropy = normal.entropy().sum(1)
    mean_action = torch.tanh(mean)
    return action, log_prob, entropy, mean_action


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
        load_target = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "vae"):
                if key in checkpoint:
                    load_target = checkpoint[key]
                    break
        vae.load_state_dict(load_target)

    print(f"Loaded VAE from: {vae_path}")
    return vae


class PPOAgent(nn.Module):
    def __init__(self, obs_space, action_space, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.policy_type = args.policy_type
        self.vision_actor = args.vision_actor

        action_dim = int(np.prod(action_space.shape))
        state_obs_space = gym.spaces.Dict(networks.filterObservationForState(obs_space))
        self.critic = networks.Critic(state_obs_space)

        self.vision_input_shape = None
        self.sensor_dim = None
        if self.policy_type == "vision":
            self.vision_input_shape = self._vision_input_shape(obs_space)
            self.sensor_dim = networks.vision_sensor_dim(obs_space)
            if self.vision_actor == "latent":
                vae = load_vae(args, self.vision_input_shape)
                self.actor = networks.LatentHierarchicalActor(
                    vae,
                    self.vision_input_shape,
                    self.sensor_dim,
                    action_dim,
                    freeze_vae=not args.unfreeze_vae,
                )
            else:
                self.actor = networks.StructuredHierarchicalActor(
                    self.vision_input_shape,
                    self.sensor_dim,
                    action_dim,
                )
        else:
            self.actor = networks.Actor(state_obs_space)

    @staticmethod
    def _vision_input_shape(obs_space):
        required_keys = ("cameraLeft", "cameraFront", "cameraRight")
        missing = [key for key in required_keys if key not in obs_space.spaces]
        if missing:
            raise KeyError(f"Vision policy requires camera observations, missing: {missing}")

        camera_shape = obs_space["cameraFront"].shape
        if len(camera_shape) != 4:
            raise ValueError(f"Expected stacked cameraFront shape (T, C, H, W), got {camera_shape}")
        return (camera_shape[0], len(required_keys), camera_shape[1], camera_shape[2], camera_shape[3])

    def _state_tensor(self, obs):
        return networks.flattenFuncState(obs).to(self.device)

    def _policy_forward(self, obs):
        if self.policy_type == "vision":
            vision, sensors = networks.flattenFuncVision(obs)
            vision = vision.to(self.device)
            sensors = sensors.to(self.device)
            return self.actor(vision, sensors)
        return self.actor(self._state_tensor(obs))

    def get_value(self, obs):
        return self.critic(self._state_tensor(obs))

    def get_action_and_value(self, obs, action=None):
        mean, log_std = self._policy_forward(obs)
        action, log_prob, entropy, mean_action = tanh_normal_action(mean, log_std, action)
        value = self.get_value(obs)
        return action, log_prob, entropy, value, mean_action


def load_pretrained_actor(actor, checkpoint_path):
    """Load a serialized BC, DAgger, or PPO actor into the PPO actor."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Expected pretrained actor at exactly: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, Mapping):
        state_dict = checkpoint
        for key in ("state_dict", "model", "actor"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
    else:
        raise TypeError(f"Unsupported pretrained actor checkpoint type: {type(checkpoint)!r}")

    if isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "The pretrained checkpoint did not contain an actor state dictionary or actor module."
        )

    try:
        actor.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Pretrained actor architecture does not match the current PPO actor. "
            "Use a checkpoint trained with the same vision architecture, frame stack, "
            "and observation configuration."
        ) from error
    print(f"Loaded pretrained PPO actor: {checkpoint_path}")


def reset_pretrained_actor_log_std(actor, target_log_std):
    """Set a constant initial exploration scale for mean-only BC/DAgger actors."""
    if target_log_std is None:
        return
    if not hasattr(actor, "fc_logstd"):
        raise TypeError("Pretrained log-std reset requires an actor with an fc_logstd head.")
    if not -20.0 < target_log_std < 2.0:
        raise ValueError("pretrained_actor_log_std must lie strictly between -20 and 2.")

    scaled = 2.0 * (target_log_std + 20.0) / 22.0 - 1.0
    pre_tanh_bias = float(np.arctanh(scaled))
    with torch.no_grad():
        actor.fc_logstd.weight.zero_()
        actor.fc_logstd.bias.fill_(pre_tanh_bias)


def load_pretrained_critic(critic, checkpoint_path):
    """Load a state-value critic checkpoint produced by critic pretraining."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Expected pretrained critic at exactly: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, Mapping):
        state_dict = checkpoint
        for key in ("state_dict", "model", "critic"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
    else:
        raise TypeError(f"Unsupported pretrained critic checkpoint type: {type(checkpoint)!r}")

    if isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()
    if not isinstance(state_dict, Mapping):
        raise TypeError("The pretrained checkpoint did not contain a critic state dictionary or module.")

    try:
        critic.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Pretrained critic architecture does not match the current PPO critic. "
            "Use a checkpoint with the same state observation configuration and frame stack."
        ) from error
    print(f"Loaded pretrained PPO critic: {checkpoint_path}")


def action_noise_multiplier(args, global_step):
    if not args.noise_augment:
        return 0.0

    anneal_start_step = int(0.3 * args.total_timesteps)
    anneal_end_step = int(0.5 * args.total_timesteps)
    if global_step <= anneal_start_step:
        return 0.0
    if global_step >= anneal_end_step:
        return 1.0

    ramp_duration = max(1, anneal_end_step - anneal_start_step)
    return (global_step - anneal_start_step) / ramp_duration


def apply_action_noise(actions, args, global_step):
    multiplier = action_noise_multiplier(args, global_step)
    if multiplier <= 0.0:
        return actions, multiplier

    base_weights = np.array([0.095, 0.15, 0.15, 0.15, 0.15], dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=base_weights * multiplier, size=actions.shape)
    noisy_actions = np.clip(actions + noise, -1.0, 1.0).astype(np.float32)
    return noisy_actions, multiplier


def validate_args(args):
    if args.state_based_actor:
        args.policy_type = "state"
    if args.num_envs < 1:
        raise ValueError("num_envs must be at least 1")
    if args.frame_stack < 1:
        raise ValueError("frame_stack must be at least 1")
    if args.num_minibatches < 1:
        raise ValueError("num_minibatches must be at least 1")
    if args.critic_warmup_iterations < 0:
        raise ValueError("critic_warmup_iterations must be non-negative")
    if args.critic_warmup_epochs is not None and args.critic_warmup_epochs < 1:
        raise ValueError("critic_warmup_epochs must be positive when set")
    for name, learning_rate in (
        ("learning_rate", args.learning_rate),
        ("actor_learning_rate", args.actor_learning_rate),
        ("critic_learning_rate", args.critic_learning_rate),
    ):
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError(f"{name} must be positive when set")
    if args.pretrained_actor_path is not None and args.policy_type != "vision":
        raise ValueError("pretrained_actor_path currently supports vision PPO fine-tuning only")
    if args.policy_type != "vision" and args.vision_actor == "latent":
        raise ValueError("vision_actor=latent only applies when policy_type=vision")

    args.batch_size = int(args.num_envs * args.num_steps)
    if args.batch_size % args.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)
    if args.num_iterations < 1:
        raise ValueError("total_timesteps must be at least num_envs * num_steps")
    if args.critic_warmup_iterations >= args.num_iterations:
        raise ValueError("critic_warmup_iterations must be smaller than the total PPO iterations")


def build_run_name(args):
    env_id = args.env_id
    if args.noise_augment:
        env_id = env_id + "_noiseAugmented"
    if args.include_last_action:
        env_id = env_id + "_lastActionObs"
    if args.policy_type == "vision":
        env_id = env_id + f"_{args.vision_actor}Vision"
    else:
        env_id = env_id + "_statePolicy"
    return f"{env_id}_{args.reward_mode}__{args.exp_name}__{args.seed}__{int(time.time())}"


def log_episode_stats(writer, progress_bar, infos, global_step, num_envs):
    episode_stats, episode_mask = get_episode_stats_from_infos(infos)
    if episode_stats is None:
        return
    if episode_mask is None:
        episode_mask = np.ones(num_envs, dtype=bool)
    for idx, finished in enumerate(episode_mask):
        if finished:
            ep_return = episode_stats["r"][idx]
            ep_length = episode_stats["l"][idx]
            progress_bar.write(
                f"global_step={global_step}, episodic_return={ep_return}, "
                f"episodic_length={ep_length}"
            )
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
            break


def save_models(agent, args):
    if not args.save_model:
        return

    noisy = args.noise_augment
    if args.policy_type == "vision":
        actor_path = ppo_actor_path(
            args.reward_mode,
            noisy,
            args.vision_actor == "latent",
            args.include_last_action,
            seed=args.seed,
        )
        critic_path = ppo_critic_path(
            args.reward_mode,
            noisy,
            args.vision_actor == "latent",
            args.include_last_action,
            seed=args.seed,
        )
    else:
        actor_path = state_ppo_actor_path(
            args.reward_mode,
            noisy,
            args.include_last_action,
            seed=args.seed,
        )
        critic_path = state_ppo_critic_path(
            args.reward_mode,
            noisy,
            args.include_last_action,
            seed=args.seed,
        )

    os.makedirs(os.path.dirname(actor_path), exist_ok=True)
    agent.actor.eval()
    agent.critic.eval()
    torch.save(agent.actor, actor_path)
    torch.save(
        {
            "critic": agent.critic.state_dict(),
            "metadata": {
                "policy_type": args.policy_type,
                "vision_actor": args.vision_actor if args.policy_type == "vision" else None,
                "frame_stack": args.frame_stack,
                "include_last_action": args.include_last_action,
                "reward_mode": args.reward_mode,
            },
        },
        critic_path,
    )
    print(f"Saved actor to: {actor_path}")
    print(f"Saved critic to: {critic_path}")


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

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if not args.cuda:
        warnings.filterwarnings("ignore", message="CUDA initialization:.*")
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    envs = make_vector_env(args, run_name)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print(args)
    print(f"vector_env_mode: {resolve_vector_env_mode(args)}, num_envs: {envs.num_envs}")
    print("noise_augment:", args.noise_augment)
    print("device:", device)

    agent = PPOAgent(envs.single_observation_space, envs.single_action_space, args, device).to(device)
    if args.pretrained_actor_path is not None:
        load_pretrained_actor(agent.actor, args.pretrained_actor_path)
        reset_pretrained_actor_log_std(agent.actor, args.pretrained_actor_log_std)
    if args.pretrained_critic_path is not None:
        load_pretrained_critic(agent.critic, args.pretrained_critic_path)

    actor_learning_rate = args.actor_learning_rate or args.learning_rate
    critic_learning_rate = args.critic_learning_rate or args.learning_rate
    optimizer = optim.Adam(
        [
            {"params": agent.actor.parameters(), "lr": actor_learning_rate},
            {"params": agent.critic.parameters(), "lr": critic_learning_rate},
        ],
        eps=1e-5,
    )
    base_learning_rates = (actor_learning_rate, critic_learning_rate)
    critic_warmup_epochs = args.critic_warmup_epochs or args.update_epochs
    writer.add_text(
        "model/input",
        (
            f"policy_type={args.policy_type}, vision_actor={args.vision_actor}, "
            f"vision_shape={agent.vision_input_shape}, sensor_dim={agent.sensor_dim}, "
            f"critic=state, include_last_action={args.include_last_action}, "
            f"pretrained_actor={args.pretrained_actor_path}, "
            f"pretrained_critic={args.pretrained_critic_path}, "
            f"critic_warmup_iterations={args.critic_warmup_iterations}"
        ),
    )

    obs = init_obs_storage(args.num_steps, args.num_envs, envs.single_observation_space)
    actions = torch.zeros((args.num_steps, args.num_envs, *envs.single_action_space.shape), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, reset_infos = envs.reset(seed=args.seed)
    for env_log in iter_env_logs(reset_infos):
        print(env_log)
    next_done = torch.zeros(args.num_envs, device=device)

    global_step = 0
    start_time = time.time()
    progress_bar = tqdm(total=args.num_iterations * args.batch_size, unit="env-step", dynamic_ncols=True)

    for iteration in range(1, args.num_iterations + 1):
        is_critic_warmup = iteration <= args.critic_warmup_iterations
        if args.anneal_lr and not is_critic_warmup:
            policy_iteration = iteration - args.critic_warmup_iterations
            policy_iterations = args.num_iterations - args.critic_warmup_iterations
            frac = 1.0 - (policy_iteration - 1.0) / policy_iterations
            for param_group, base_learning_rate in zip(optimizer.param_groups, base_learning_rates):
                param_group["lr"] = frac * base_learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            store_obs(obs, step, next_obs)
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            env_actions = action.detach().cpu().numpy()
            env_actions, noise_multiplier = apply_action_noise(env_actions, args, global_step)
            if noise_multiplier > 0.0:
                noisy_action = torch.as_tensor(env_actions, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, logprob, _, value, _ = agent.get_action_and_value(next_obs, noisy_action)
                    values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            if global_step % 1000 == 0:
                writer.add_scalar("debug/noise_multiplier", noise_multiplier, global_step)

            next_obs, reward, terminations, truncations, infos = envs.step(env_actions)
            for env_log in iter_env_logs(infos):
                progress_bar.write(env_log)

            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)
            log_episode_stats(writer, progress_bar, infos, global_step, envs.num_envs)

            progress_bar.update(args.num_envs)
            elapsed = time.time() - start_time
            env_sps = int(global_step / elapsed) if elapsed > 0 else 0
            vector_sps = int((global_step // args.num_envs) / elapsed) if elapsed > 0 else 0
            progress_bar.set_postfix(env_sps=env_sps, vec_sps=vector_sps)
            if global_step % 100 == 0:
                writer.add_scalar("charts/env_SPS", env_sps, global_step)
                writer.add_scalar("charts/vector_SPS", vector_sps, global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = flatten_obs_storage(obs)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, *envs.single_action_space.shape))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_epochs = critic_warmup_epochs if is_critic_warmup else args.update_epochs
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = index_obs(b_obs, mb_inds)

                if is_critic_warmup:
                    newvalue = agent.get_value(mb_obs).view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    optimizer.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                    optimizer.step()
                    continue

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    mb_obs,
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv and mb_advantages.numel() > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if not is_critic_warmup and args.target_kl is not None and approx_kl > args.target_kl:
                break

        if is_critic_warmup:
            writer.add_scalar("phase/critic_warmup", 1, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("charts/critic_learning_rate", optimizer.param_groups[1]["lr"], global_step)
            continue

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("phase/critic_warmup", 0, global_step)
        writer.add_scalar("charts/actor_learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/critic_learning_rate", optimizer.param_groups[1]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

    progress_bar.close()
    envs.close()
    save_models(agent, args)
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
