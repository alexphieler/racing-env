import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import numpy as np

from pacsimEnv import pacsimEnv

try:
    import gym
except ImportError:
    import gymnasium as gym


@contextmanager
def maybe_silence(enabled):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def reward_params(reward_mode):
    if reward_mode == "aggressive":
        return {
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
        }
    if reward_mode == "conservative":
        return {
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
        }
    raise ValueError(f"Unsupported reward mode: {reward_mode}")


def _box(space_module, low, high, shape, dtype):
    return space_module.Box(low=low, high=high, shape=shape, dtype=dtype)


def _dict(space_module, spaces):
    return space_module.Dict(spaces)


class PacsimDreamerEnv(gym.Env):
    """Old-Gym API adapter used by NM512/dreamerv3-torch."""

    metadata = {}

    def __init__(
        self,
        gym_module,
        seed=0,
        cam_sim=False,
        include_camera_obs=False,
        include_state_obs=True,
        state_obs_keys="",
        image_size=(64, 64),
        camera_key="cameraFront",
        reward_mode="conservative",
        print_step_status=False,
        verbose_env=False,
        preload_track_cache=True,
        camera_config_file=None,
    ):
        super().__init__()
        self._gym = gym_module
        self._seed = seed
        self._image_size = tuple(int(v) for v in image_size)
        self._monitor_image_size = self._image_size
        self._camera_key = str(camera_key)
        self._include_state_obs = bool(include_state_obs)
        self._state_obs_keys = frozenset(
            key for key in str(state_obs_keys).split("|") if key
        )
        self._verbose_env = bool(verbose_env)

        pacsim_args = {
            "cam_sim": bool(cam_sim),
            "include_camera_obs": bool(include_camera_obs or cam_sim),
            "include_last_action": True,
            "print_step_status": bool(print_step_status),
            "preload_track_cache": bool(preload_track_cache),
        }
        if camera_config_file:
            pacsim_args["camera_config_file"] = camera_config_file
        pacsim_args.update(reward_params(reward_mode))

        with maybe_silence(not self._verbose_env):
            self._env = pacsimEnv(pacsim_args)

        self.action_space = gym_module.spaces.Box(
            low=self._env.action_space.low.astype(np.float32),
            high=self._env.action_space.high.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._build_observation_space(gym_module.spaces)
        self._last_reset_seed = None
        self._reset_velocity_log_stats()

    def _build_observation_space(self, spaces):
        obs_spaces = {}
        for key, space in self._env.observation_space.spaces.items():
            if key in self._camera_keys():
                continue
            if not self._includes_state_obs_key(key):
                continue
            obs_spaces[key] = _box(
                spaces,
                low=np.asarray(space.low, dtype=space.dtype),
                high=np.asarray(space.high, dtype=space.dtype),
                shape=space.shape,
                dtype=space.dtype,
            )
        image_keys = self._selected_camera_keys(self._env.observation_space.spaces)
        for key in image_keys:
            obs_spaces[key] = _box(
                spaces,
                low=0,
                high=255,
                shape=(self._image_size[0], self._image_size[1], 3),
                dtype=np.uint8,
            )
        monitor_width = max(1, len(image_keys)) * self._monitor_image_size[1]
        obs_spaces["image"] = _box(
            spaces,
            low=0,
            high=255,
            shape=(self._monitor_image_size[0], monitor_width, 3),
            dtype=np.uint8,
        )
        return _dict(spaces, obs_spaces)

    def _camera_keys(self):
        return list(getattr(self._env, "cameraObsKeys", []))

    def _includes_state_obs_key(self, key):
        return self._include_state_obs and (
            not self._state_obs_keys or key in self._state_obs_keys
        )

    def _blank_image(self):
        return np.zeros((self._image_size[0], self._image_size[1], 3), dtype=np.uint8)

    def _blank_monitor_image(self):
        return np.zeros((self._monitor_image_size[0], self._monitor_image_size[1], 3), dtype=np.uint8)

    def _extract_image(self, obs):
        keys = self._selected_camera_keys(obs)
        if not keys:
            return self._blank_monitor_image()

        if len(keys) == 1:
            image = self._camera_image(obs, keys[0])
            if image is None:
                return self._blank_monitor_image()
            return self._fit_image(image, self._monitor_image_size)

        return self._compose_camera_strip(obs, keys)

    def _selected_camera_keys(self, obs):
        camera_keys = [key for key in self._camera_keys() if key in obs]
        selector = self._camera_key.strip().lower()
        if selector in {"all", "*", "multi", "strip"}:
            return camera_keys
        if self._camera_key in camera_keys:
            return [self._camera_key]
        return camera_keys[:1]

    def _camera_image(self, obs, key):
        if key not in obs:
            return None
        image = np.asarray(obs[key])
        if image.ndim != 3:
            return None
        if image.shape[0] == 3:
            image = np.transpose(image, (2, 1, 0))
        elif image.shape[-1] != 3:
            return None
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _compose_camera_strip(self, obs, keys):
        target_h, target_w = self._monitor_image_size
        out = np.zeros((target_h, target_w * len(keys), 3), dtype=np.uint8)
        x0 = 0
        for key in keys:
            image = self._camera_image(obs, key)
            if image is not None:
                out[:, x0:x0 + target_w] = self._fit_image(image, self._monitor_image_size)
            x0 += target_w
        return out

    def _resize_image(self, image, image_size):
        target_h, target_w = image_size
        from PIL import Image

        resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
        resized = Image.fromarray(image).resize((target_w, target_h), resample=resample)
        return np.asarray(resized, dtype=np.uint8)

    def _fit_image(self, image, image_size):
        target_h, target_w = image_size
        src_h, src_w = image.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        scale = min(target_w / src_w, target_h / src_h)
        resized_h = max(1, int(round(src_h * scale)))
        resized_w = max(1, int(round(src_w * scale)))
        resized = self._resize_image(image, (resized_h, resized_w))
        out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y0 = (target_h - resized_h) // 2
        x0 = (target_w - resized_w) // 2
        out[y0:y0 + resized_h, x0:x0 + resized_w] = resized
        return out

    def _convert_obs(self, obs, is_first=False, is_terminal=False):
        converted = {}
        image_keys = self._selected_camera_keys(obs)
        for key, value in obs.items():
            if key in self._camera_keys():
                continue
            if not self._includes_state_obs_key(key):
                continue
            converted[key] = np.asarray(value, dtype=np.float32)
        for key in image_keys:
            image = self._camera_image(obs, key)
            if image is None:
                converted[key] = self._blank_image()
            else:
                converted[key] = self._fit_image(image, self._image_size)
        converted["image"] = self._extract_image(obs)
        converted["is_first"] = bool(is_first)
        converted["is_terminal"] = bool(is_terminal)
        return converted

    def _reset_velocity_log_stats(self):
        self._velocity_log_sum = np.zeros(3, dtype=np.float64)
        self._velocity_log_count = 0

    def _velocity_log_values(self, obs):
        velocity = np.asarray(obs.get("velocity", np.zeros(2)), dtype=np.float32)
        imu = np.asarray(obs.get("imu", np.zeros(3)), dtype=np.float32)
        vx = float(velocity[0] * getattr(self._env, "maxSpeed", 1.0))
        vy = float(velocity[1] * getattr(self._env, "maxSpeed", 1.0))
        yaw_rate = float(imu[2] * getattr(self._env, "maxYawRate", 1.0))
        return np.asarray([vx, vy, yaw_rate], dtype=np.float64)

    def _add_velocity_logs(self, converted, obs, done):
        keys = (
            "log_velocity_vx",
            "log_velocity_vy",
            "log_velocity_yaw_rate",
        )
        self._velocity_log_sum += self._velocity_log_values(obs)
        self._velocity_log_count += 1
        values = np.zeros(len(keys), dtype=np.float32)
        if done:
            values = (self._velocity_log_sum / max(1, self._velocity_log_count)).astype(np.float32)
            self._reset_velocity_log_stats()
        for key, value in zip(keys, values):
            converted[key] = value

    def reset(self):
        seed = self._seed if self._last_reset_seed is None else None
        self._last_reset_seed = seed
        self._reset_velocity_log_stats()
        with maybe_silence(not self._verbose_env):
            obs, _ = self._env.reset(seed=seed)
        return self._convert_obs(obs, is_first=True, is_terminal=False)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        with maybe_silence(not self._verbose_env):
            obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        info = dict(info)
        converted = self._convert_obs(obs, is_first=False, is_terminal=terminated)
        self._add_velocity_logs(converted, obs, done)
        info.setdefault("discount", np.array(0.0 if terminated else 1.0, dtype=np.float32))
        return converted, float(reward), done, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        close = getattr(self._env, "close", None)
        if callable(close):
            close()
