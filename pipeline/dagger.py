import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym

from pacsimEnv import pacsimEnv
from gymnasium.wrappers import FrameStackObservation

import networks
from torch.utils.tensorboard import SummaryWriter
import argparse, os
import random
from pathlib import Path
from buffers import FrameStackDictReplayBuffer, load_replay_buffer
from artifact_names import bc_actor_path, dagger_actor_path, replay_buffer_path, state_actor_path as default_state_actor_path

from tqdm.rich import tqdm, trange
from policy_inspection import log_action_errors


device = "cuda"


# CLI args: choose replay buffer path or use noisy buffer; optionally specify pretrained BC model
parser = argparse.ArgumentParser(description="DAgger trainer using BC pretraining")
parser.add_argument("--buffer", type=str, default=None, help="Path to replay buffer pickle file")
parser.add_argument("--log-dir", type=str, default="runs", help="Base directory for TensorBoard logs")
parser.add_argument("--seed", type=int, default=1, help="Seed used in the default buffer, SAC-teacher, BC, and DAgger-model filenames")
parser.add_argument("--noisy", action="store_true", help="Use the noisy replay buffer and state actor for the selected reward mode")
parser.add_argument(
    "--reward-mode",
    choices=["conservative", "aggressive"],
    default=None,
    help="Reward preset used in default replay-buffer and state-actor names. Defaults to conservative.",
)
parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained BC model to initialize vision actor")
parser.add_argument("--latent", action="store_true", help="Use the latent/encoder BC model")
parser.add_argument("--train-steps", type=int, default=800, help="Number of DAgger rollout/training iterations.")
parser.add_argument("--batches-per-iter", type=int, default=50, help="Gradient updates after each DAgger rollout.")
parser.add_argument("--batch-size", type=int, default=32, help="Total samples in each DAgger gradient update.")
parser.add_argument("--log-interval", type=int, default=80, help="Log training loss every N gradient updates.")
parser.add_argument("--expert-buffer-size", type=int, default=4000, help="Capacity of the long expert replay buffer.")
parser.add_argument("--expert-short-buffer-size", type=int, default=30, help="Capacity of the recent expert replay buffer.")
parser.add_argument(
    "--include-last-action",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require replay-buffer observations with last_action and collect matching env observations.",
)
args = parser.parse_args()

if min(
    args.train_steps,
    args.batches_per_iter,
    args.batch_size,
    args.log_interval,
    args.expert_buffer_size,
    args.expert_short_buffer_size,
) < 1:
    parser.error("DAgger step, batch, and buffer-size options must all be positive")
if args.batch_size < 3:
    parser.error("--batch-size must be at least 3 so BC, expert, and recent-expert samples are all present")

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
include_last_action = rb_has_last_action
pacsim = pacsimEnv({"include_last_action": include_last_action})
env = FrameStackObservation(pacsim, stack_size=3)

# TensorBoard writer
rb_basename = os.path.splitext(os.path.basename(rb_path))[0]
suffix = "_latent" if args.latent else ""
log_dir = Path(args.log_dir).expanduser() / f"dagger_{rb_basename}{suffix}"
print(f"TensorBoard log directory: {log_dir}", flush=True)
writer = SummaryWriter(log_dir=str(log_dir))
writer.add_text('info', f"replay_buffer: {rb_path}")
writer.add_text('model/input', f"include_last_action={include_last_action}")

count = 0

rbExpert = FrameStackDictReplayBuffer(
    args.expert_buffer_size,
    env.observation_space,
    env.action_space,
    "cpu",
    n_envs=1,
    frame_stack=3,
    handle_timeout_termination=False,
)

rbExpertShort = FrameStackDictReplayBuffer(
    args.expert_short_buffer_size,
    env.observation_space,
    env.action_space,
    "cpu",
    n_envs=1,
    frame_stack=3,
    handle_timeout_termination=False,
)

alpha = 0.8
batch_size = args.batch_size
# Keep all three data sources represented. Integer rounding previously allowed
# the recent-expert batch to become zero for small user-provided batch sizes.
batch_size_expert = min(batch_size - 2, max(1, int(batch_size * alpha)))
remaining_batch_size = batch_size - batch_size_expert
batch_size_expert_short = min(remaining_batch_size - 1, max(1, int(remaining_batch_size * 0.3)))
batch_size_bc = batch_size - batch_size_expert - batch_size_expert_short

# Load pretrained vision actor (BC)
if args.pretrained:
    pretrained_path = args.pretrained
else:
    pretrained_path = bc_actor_path(reward_mode, args.noisy, args.latent, include_last_action, seed=args.seed)

if not os.path.isfile(pretrained_path):
    raise FileNotFoundError(f"Expected pretrained BC actor at exactly: {pretrained_path}")

print(f"Loading pretrained vision actor: {pretrained_path}")
actor = torch.load(pretrained_path, weights_only=False)
actor = actor.to(device)
state_actor_path = default_state_actor_path(reward_mode, args.noisy, include_last_action, seed=args.seed)
if not os.path.isfile(state_actor_path):
    raise FileNotFoundError(f"Expected state actor at exactly: {state_actor_path}")

print(f"Loading state actor: {state_actor_path}")
state_actor = torch.load(state_actor_path, weights_only=False).to(device)

expected_state_dim = networks.state_input_dim(env.observation_space)
state_actor_dim = getattr(getattr(state_actor, "fc1", None), "in_features", None)
if state_actor_dim is not None and state_actor_dim != expected_state_dim:
    raise ValueError(
        "State actor input dim ({0}) does not match env state dim ({1}). "
        "Check whether the SAC expert was trained with last_action.".format(
            state_actor_dim,
            expected_state_dim,
        )
    )

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-5)


def deterministic_mean(policy, *inputs):
    """Return the tanh-squashed mean without sampling or log-prob work."""
    mean, _ = policy(*inputs)
    return torch.tanh(mean) * policy.action_scale + policy.action_bias

normalizeVision = True

train_steps = args.train_steps
batches_per_iter = args.batches_per_iter
beta = 0.6
beta_decay = 0.965
log_step = 0
state_actor.eval()
for outer_iter in tqdm(range(0, train_steps)):


    obs = env.reset(seed=args.seed if outer_iter == 0 else None)[0]
    done = False
    actor.eval()
    print("beta: " + str(beta))
    while not done:
        with torch.inference_mode():
            obsFiltered = networks.filterObservationForState(obs)
            obs_flattened = networks.flattenFuncStateSingle(obsFiltered).to(device)
            state_action = deterministic_mean(state_actor, obs_flattened).cpu().numpy()[0]

            vision, sensors = networks.flattenFuncVisionSingle(obs)
            vision = vision.to(device)
            sensors = sensors.to(device)
            vision_action = deterministic_mean(actor, vision, sensors).cpu().numpy()[0]

        pi = beta*state_action + (1.0-beta)*vision_action

        obsNew, reward, terminated, truncated, info = env.step(pi)

        rbExpert.add(obs, obsNew, state_action, reward, terminated, info)
        rbExpertShort.add(obs, obsNew, state_action, reward, terminated, info)
        obs = obsNew.copy()
        done = terminated or truncated
        # count += 1
    actor.train()
    beta = beta*beta_decay

    print("new iter")
    # Log beta value for this iteration
    try:
        writer.add_scalar('misc/beta', beta, outer_iter)
    except Exception as e:
        print(f"TensorBoard logging failed: {e}")

    for i in range(batches_per_iter):
        bufferSamplesBC = rb.sample(batch_size_bc)
        bufferSamplesExpert = rbExpert.sample(batch_size_expert)
        bufferSamplesExpertShort = rbExpertShort.sample(batch_size_expert_short)

        # Preserve the BC/expert/recent-expert mix, but combine it before
        # packing vision so there is one CPU pack and one large vision H2D
        # transfer instead of one for each source.
        sample_batches = (bufferSamplesBC, bufferSamplesExpert, bufferSamplesExpertShort)
        observations = {
            key: torch.cat([samples.observations[key] for samples in sample_batches], dim=0)
            for key in bufferSamplesBC.observations
        }
        actionLabels = torch.cat([samples.actions for samples in sample_batches], dim=0).to(device)
        vision, sensors = networks.flattenFuncVision(observations)
        vision = vision.to(device)
        sensors = sensors.to(device)

        raw_mean, _ = actor(vision, sensors)
        mean = torch.tanh(raw_mean) * actor.action_scale + actor.action_bias

        actor_loss = torch.nn.functional.mse_loss(mean, actionLabels)

        # Optimize the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        log_step += 1
        if log_step % args.log_interval == 0:
            # .item() synchronizes CUDA, so do it only at the configured
            # interval and reuse the value for console and TensorBoard logs.
            loss_value = actor_loss.detach().item()
            print(loss_value)
            try:
                writer.add_scalar('loss/actor', loss_value, log_step)
                log_action_errors(writer, mean, actionLabels, log_step)
                start = 0
                for source, size in (('bc', batch_size_bc), ('dagger', batch_size_expert),
                                     ('recent', batch_size_expert_short)):
                    log_action_errors(writer, mean[start:start + size],
                                      actionLabels[start:start + size], log_step, source)
                    start += size
            except Exception as e:
                print(f"TensorBoard logging failed: {e}")

# Save model with the shared artifact naming convention
save_name = dagger_actor_path(reward_mode, args.noisy, hasattr(actor, 'vae'), include_last_action, seed=args.seed)
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
