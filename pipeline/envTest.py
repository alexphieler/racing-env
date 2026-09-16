import numpy as np

import cv2

import gymnasium as gym

import matplotlib.pyplot as plt

from pacsimEnv import pacsimEnv

import torch.nn as nn

from gymnasium.wrappers import FrameStackObservation

env = pacsimEnv(None)
env = FrameStackObservation(env, stack_size=3)

obs, info = env.reset()
done = False

count = 0

while not done:
    action = [0,0,0,0,0]

    obs, reward, terminated, truncated, info = env.step(action)
    im = env.render()
    cv2.imwrite("/root/workspace/viz/render/outRender"+str(count)+".png",im)

    done = terminated or truncated or (count >= 1000)
    count += 1
    if(count >= 1000):
        done = True