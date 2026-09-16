import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import gymnasium

from pacsimEnv import pacsimEnv
from gymnasium.wrappers import FrameStackObservation, NormalizeReward, NormalizeObservation

import cv2
from tqdm.rich import tqdm, trange

import networks
from artifact_names import bc_actor_path, dagger_actor_path, state_actor_path, vision_sac_actor_path

import subprocess
import argparse
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLI args: choose model type and path
parser = argparse.ArgumentParser(description="Actor inference runner")
parser.add_argument('--model-type', choices=['state', 'vision'], default='state', help='Type of model to run')
parser.add_argument('--model-path', type=str, default=None, help='Path to model to load (overrides defaults)')
parser.add_argument('--reward-mode', choices=['conservative', 'aggressive'], default='conservative', help='Reward preset used in default model names')
parser.add_argument('--noisy', action='store_true', help='Use noisy variant when applicable')
parser.add_argument('--seed', type=int, default=1, help='Seed suffix of the default checkpoint filename')
parser.add_argument('--dagger', action='store_true', help='If running a vision model, load the DAgger-trained vision model')
parser.add_argument('--sac', action='store_true', help='If running a vision model, load the SAC-trained vision model')
parser.add_argument(
    '--asymmetric-critic',
    action='store_true',
    help='Load a vision SAC actor trained with a privileged state critic',
)
parser.add_argument('--latent', action='store_true', help='Use the latent/encoder vision model')
parser.add_argument(
    '--include-last-action',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Use models/env observations that include the previous normalized action',
)
args = parser.parse_args()
if args.dagger and args.sac:
    parser.error('--dagger and --sac are mutually exclusive')
if args.asymmetric_critic and not args.sac:
    parser.error('--asymmetric-critic requires --sac')

# Determine default model path if not provided
if args.model_path:
    model_path = args.model_path
else:
    if args.model_type == 'state':
        model_path = state_actor_path(args.reward_mode, args.noisy, args.include_last_action, seed=args.seed)
    else:
        if args.sac:
            model_path = vision_sac_actor_path(
                args.reward_mode,
                args.noisy,
                args.latent,
                args.include_last_action,
                seed=args.seed,
                asymmetric=args.asymmetric_critic,
            )
        elif args.dagger:
            model_path = dagger_actor_path(
                args.reward_mode, args.noisy, args.latent, args.include_last_action, seed=args.seed
            )
        else:
            model_path = bc_actor_path(
                args.reward_mode, args.noisy, args.latent, args.include_last_action, seed=args.seed
            )

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Expected model at exactly: {model_path}")

print(f"Loading model: {model_path} (type={args.model_type})")
actor = torch.load(model_path, map_location=device, weights_only=False)

actor = actor.to(device)
pacsimArgs = {
    "cam_sim": True,
    "include_last_action": args.include_last_action,
}
pacsim = pacsimEnv(pacsimArgs)
env = FrameStackObservation(pacsim, stack_size=3)

model_base = os.path.splitext(os.path.basename(model_path))[0]
video_name = f"{model_base}_{args.model_type}_{int(time.time())}.mp4"
video_path = os.path.join("..", "viz", video_name)
os.makedirs(os.path.dirname(video_path), exist_ok=True)
print(f"Writing video to: {video_path}")

resetArgs = {
    "map_files" : ["/root/workspace/tracks/FSE23.yaml"],
    "noAugment": True,
}
obs = env.reset(options=resetArgs)[0]

count = 0

def obs_to_image(obs):
    leftImage = np.transpose(obs["cameraLeft"][2], (2, 1, 0))
    frontImage = np.transpose(obs["cameraFront"][2], (2, 1, 0))
    rightImage = np.transpose(obs["cameraRight"][2], (2, 1, 0))
    camImageStitched = np.zeros((frontImage.shape[0], 3*frontImage.shape[1], frontImage.shape[2]), dtype=np.uint8)
    camImageStitched[:,0:frontImage.shape[1]] = leftImage
    camImageStitched[:,frontImage.shape[1]:2*frontImage.shape[1]] = frontImage
    camImageStitched[:,2*frontImage.shape[1]:3*frontImage.shape[1]] = rightImage
    return camImageStitched

im = obs_to_image(obs)

command = [
    'ffmpeg',
    '-y',                     # Overwrite output file
    '-f', 'rawvideo',         # Input format
    '-vcodec', 'rawvideo',
    '-s', f'{im.shape[1]}x{im.shape[0]}', # Size
    '-pix_fmt', 'bgr24',       # OpenCV uses BGR, so we tell FFmpeg to expect BGR
    '-r', str(10),           # Input framerate
    '-i', '-',                # Input comes from a pipe
    '-c:v', 'libx264',        # Video codec to use
    '-pix_fmt', 'yuv420p',    # Output pixel format (standard for compatibility)
    '-preset', 'medium',      # Encoding speed vs compression
    '-crf', '17',             # QUALITY CONTROL: 0 (lossless) to 51 (worst)
    video_path
]
process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

cv2.imwrite("/root/workspace/viz/render/outRender"+str(count)+".png",im)
count += 1

done = False
actor.eval()

actions = []
info = None

while not done:
    # Choose processing depending on model type
    with torch.no_grad():
        if args.model_type == 'vision':
            vision, sensors = networks.flattenFuncVisionSingle(obs)
            vision = vision.to(device)
            sensors = sensors.to(device)
            action = actor.get_action(vision, sensors)[-1].cpu().detach().numpy()[0]
        else:
            obs_flattened = networks.flattenFuncStateSingle(obs).to(device)
            action = actor.get_action(obs_flattened)[-1].cpu().detach().numpy()[0]

    actions.append(action)
    frame = obs_to_image(obs)
    cv2.imwrite("/root/workspace/viz/render/outRender"+str(count)+".png", frame)
    process.stdin.write(frame.tobytes())

    obsNew, reward, terminated, truncated, info = env.step(action)

    obs = obsNew.copy()
    done = terminated or truncated or (count>1000)
    count += 1

print(info)
process.stdin.close()
process.wait()
