import gymnasium as gym
import numpy as np
from collections import deque

class FrameStack(gym.Wrapper):
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque([], maxlen=n_stack)

        obs_shape = env.observation_space.shape
        stacked_shape = (obs_shape[0] * n_stack, *obs_shape[1:])
        low = np.repeat(self.observation_space.low[np.newaxis, ...], self.n_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], self.n_stack, axis=0
        )
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)