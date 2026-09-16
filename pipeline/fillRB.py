import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium
from gymnasium.wrappers import FrameStackObservation

import argparse
import os
import random

from tqdm.rich import tqdm, trange
from artifact_names import replay_buffer_path, state_actor_path

parser = argparse.ArgumentParser(description="fillRB model loader")
parser.add_argument("--model", type=str, default=None, help="Path to model to load (overrides flags)")
parser.add_argument("--seed", type=int, default=1, help="Seed used in the default state-actor and replay-buffer filenames")
parser.add_argument("--buffer-size", type=int, default=8000, help="Number of transitions to collect into the replay buffer")
parser.add_argument("--output", type=str, default=None, help="Replay-buffer output path (overrides the default pipeline data path)")
parser.add_argument("--noisy", action="store_true", help="Load the noisy state actor for the selected reward mode")
parser.add_argument(
    "--include-last-action",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Add the previous normalized action to pacsimEnv observations.",
)
parser.add_argument(
    "--reward-mode",
    choices=["conservative", "aggressive"],
    default=None,
    help="Reward preset used in the saved state actor name. Defaults to conservative unless --model is provided.",
)
args = parser.parse_args()

if args.buffer_size < 1:
    parser.error("--buffer-size must be positive")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

import networks
from pacsimEnv import pacsimEnv
pacsim = pacsimEnv({"include_last_action": args.include_last_action})
env = FrameStackObservation(pacsim, stack_size=3)

from buffers import FrameStackDictReplayBuffer, save_replay_buffer

buffer_size = args.buffer_size
rb = FrameStackDictReplayBuffer(
    buffer_size,
    env.observation_space,
    env.action_space,
    "cpu",
    n_envs=1,
    frame_stack=3,
    handle_timeout_termination=False,
)

if args.model:
    model_path = args.model
else:
    model_path = state_actor_path(args.reward_mode or "conservative", args.noisy, args.include_last_action, seed=args.seed)

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Expected state actor at exactly: {model_path}")

print(f"Loading model: {model_path}")
reward_mode = args.reward_mode or "conservative"
save_filename = args.output or replay_buffer_path(
    reward_mode, args.noisy, args.include_last_action, seed=args.seed
)
os.makedirs(os.path.dirname(save_filename) or ".", exist_ok=True)
model = torch.load(model_path, weights_only=False)
model = model.to("cpu")

expected_state_dim = networks.state_input_dim(env.observation_space)
model_state_dim = getattr(getattr(model, "fc1", None), "in_features", None)
if model_state_dim is not None and model_state_dim != expected_state_dim:
    raise ValueError(
        "State actor input dim ({0}) does not match env state dim ({1}). "
        "Check whether --include-last-action matches the SAC training run.".format(
            model_state_dim,
            expected_state_dim,
        )
    )

pbar = tqdm(total=buffer_size)
while rb.size() < buffer_size:
    obs = env.reset(seed=args.seed if rb.size() == 0 else None)[0]
    done = False
    while not done and rb.size() < buffer_size:
        obsFiltered = networks.filterObservationForState(obs)
        obs_flattened = networks.flattenFuncStateSingle(obsFiltered)
        pi, log_pi, mean = model.get_action(obs_flattened)
        action = mean.detach().numpy()[0]
        obsNew, reward, terminated, truncated, info = env.step(action)

        rb.add(obs, obsNew, action, reward, terminated, info)
        pbar.update(1)
        obs = obsNew.copy()
        done = terminated or truncated
pbar.close()


print("Writing replay buffer to:", save_filename)
save_replay_buffer(save_filename, rb)
