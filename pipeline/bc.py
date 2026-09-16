import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import gymnasium

from gymnasium.wrappers import FrameStackObservation, NormalizeReward, NormalizeObservation

from tqdm.rich import tqdm, trange
from buffers import load_replay_buffer
from artifact_names import bc_actor_path, replay_buffer_path, vae_path as default_vae_path

import networks
from torch.utils.tensorboard import SummaryWriter

action_dim = 5
device = "cuda"

# Actor will be initialized after parsing CLI args so we can choose latent or structured

# CLI args: choose replay buffer path or use noisy buffer
import argparse, os
import random
from pathlib import Path
parser = argparse.ArgumentParser(description="Behavior cloning trainer")
parser.add_argument("--buffer", type=str, default=None, help="Path to replay buffer pickle file")
parser.add_argument("--log-dir", type=str, default="runs", help="Base directory for TensorBoard logs")
parser.add_argument("--seed", type=int, default=1, help="Seed used in the default replay-buffer and BC-model filenames")
parser.add_argument("--noisy", action="store_true", help="Use the noisy replay buffer for the selected reward mode")
parser.add_argument(
    "--reward-mode",
    choices=["conservative", "aggressive"],
    default=None,
    help="Reward preset used in default replay-buffer names. Defaults to conservative.",
)
parser.add_argument("--latent", action="store_true", help="Use LatentHierarchicalActor (uses a ConvVAE encoder)")
parser.add_argument("--vae-path", type=str, default=None, help="Path to pretrained VAE weights (.pt). Optional when using --latent.")
parser.add_argument("--unfreeze-vae", action="store_true", help="Allow fine-tuning of the VAE weights during BC training")
parser.add_argument("--train-steps", type=int, default=10000, help="Number of BC gradient updates.")
parser.add_argument("--batch-size", type=int, default=16, help="Replay-buffer samples per BC gradient update.")
parser.add_argument(
    "--include-last-action",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require replay-buffer observations with last_action.",
)
args = parser.parse_args()

if args.train_steps < 1 or args.batch_size < 1:
    parser.error("--train-steps and --batch-size must both be positive")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

reward_mode = args.reward_mode or "conservative"
if args.buffer:
    rb_path = args.buffer
else:
    rb_path = replay_buffer_path(reward_mode, args.noisy, args.include_last_action, seed=args.seed)

if not os.path.exists(rb_path):
    raise FileNotFoundError(f"Expected replay buffer at exactly: {rb_path}")

rb = load_replay_buffer(rb_path)
print(f"Loaded replay buffer: {rb_path}")

rb_has_last_action = "last_action" in rb.observation_space.spaces
if args.include_last_action and not rb_has_last_action:
    raise ValueError("--include-last-action requires a replay buffer that contains last_action.")

camera_shape = rb.observation_space["cameraFront"].shape
if len(camera_shape) != 4:
    raise ValueError(f"Expected stacked cameraFront shape (T, C, H, W), got {camera_shape}")
imgDim = (camera_shape[0], 3, camera_shape[1], camera_shape[2], camera_shape[3])
sensor_dim = networks.vision_sensor_dim(rb.observation_space)
print(f"Vision input shape: {imgDim}, sensor_dim: {sensor_dim}")

# TensorBoard writer
rb_basename = os.path.splitext(os.path.basename(rb_path))[0]
log_dir = Path(args.log_dir).expanduser() / f"bc_{rb_basename}"
print(f"TensorBoard log directory: {log_dir}", flush=True)
writer = SummaryWriter(log_dir=str(log_dir))
writer.add_text('info', f"replay_buffer: {rb_path}")
writer.add_text('model/input', f"vision_shape={imgDim}, sensor_dim={sensor_dim}, include_last_action={rb_has_last_action}")

count = 0

# Initialize actor depending on CLI flags (structured vs latent)
if getattr(args, 'latent', False):
    # Create ConvVAE and optionally load weights
    vae = networks.ConvVAE(input_spatial=imgDim[-2:])
    vae_loaded = False
    vae_path = args.vae_path or default_vae_path(reward_mode, args.noisy, rb_has_last_action)
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"Expected VAE at exactly: {vae_path}")
    try:
        state = torch.load(vae_path, map_location='cpu', weights_only=False)

        # Support a few common checkpoint layouts
        load_target = None
        if isinstance(state, dict):
            for key in ('state_dict', 'model', 'vae'):
                if key in state:
                    load_target = state[key]
                    break
            if load_target is None:
                load_target = state
        elif isinstance(state, nn.Module):
            vae = state
            load_target = None
        if load_target is not None:
            vae.load_state_dict(load_target)
        vae_loaded = True
        print(f"Loaded VAE from: {vae_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load VAE from {vae_path}: {e}") from e
    actor = networks.LatentHierarchicalActor(
        vae, 
        imgDim, 
        sensor_dim, 
        action_dim,
        freeze_vae=not getattr(args, 'unfreeze_vae', False),
    ).to(device)
    writer.add_text('model/type', f'latent (vae_loaded={vae_loaded}, frozen={not getattr(args, "unfreeze_vae", False)})')
else:
    actor = networks.StructuredHierarchicalActor(
        imgDim, sensor_dim, action_dim
    ).to(device)
    writer.add_text('model/type', 'structured')

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.9998)

train_steps = args.train_steps
batch_size = args.batch_size
for _ in tqdm(range(0, train_steps)):
    bufferSamples = rb.sample(batch_size)
    observations = bufferSamples[0]
    bufferActions = bufferSamples[1].to(device)
    vision, sensors = networks.flattenFuncVision(observations)
    vision = vision.to(device)
    sensors = sensors.to(device)
    # print(f"Vision tensor shape: {vision.shape}")
    # print(f"Sensors tensor shape: {sensors.shape}")

    raw_mean, _ = actor(vision, sensors)
    mean = torch.tanh(raw_mean) * actor.action_scale + actor.action_bias
    actor_loss = torch.nn.functional.mse_loss(mean, bufferActions)
    # Optimize the actor
    print(actor_loss.item())
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optimizer.step()
    scheduler.step()

    # Logging to TensorBoard
    count += 1
    try:
        writer.add_scalar('loss/actor', actor_loss.item(), count)
        writer.add_scalar('lr', scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 0.0, count)
    except Exception as e:
        # Don't crash training if writer fails
        print(f"TensorBoard logging failed: {e}")

    # print(bufferActions)
    # print(mean)

# Save model with the shared artifact naming convention
save_name = bc_actor_path(reward_mode, args.noisy, getattr(args, 'latent', False), rb_has_last_action, seed=args.seed)
os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

torch.save(actor, save_name)
print(f"Saved actor to: {save_name}")
# TensorBoard finalize
try:
    writer.add_text('model/saved', save_name)
    writer.flush()
    writer.close()
except Exception as e:
    print(f"TensorBoard finalize failed: {e}")
