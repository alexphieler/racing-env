import argparse
import math
import pathlib
import sys
import types


PIPELINE_ROOT = pathlib.Path(__file__).resolve().parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
DREAMER_ROOT = WORKSPACE_ROOT / "external" / "dreamerv3-torch"


def install_gym_compat():
    try:
        import gym  # noqa: F401
        return sys.modules["gym"]
    except ImportError:
        import gymnasium

        gym_shim = types.ModuleType("gym")

        class Env:
            metadata = {}

            @property
            def unwrapped(self):
                return self

            def reset(self, *args, **kwargs):
                raise NotImplementedError

            def step(self, action):
                raise NotImplementedError

            def close(self):
                pass

        class Wrapper(Env):
            def __init__(self, env):
                self.env = env
                self.action_space = env.action_space
                self.observation_space = env.observation_space
                self.metadata = getattr(env, "metadata", {})

            @property
            def unwrapped(self):
                return getattr(self.env, "unwrapped", self.env)

            def __getattr__(self, name):
                return getattr(self.env, name)

            def reset(self):
                return self.env.reset()

            def step(self, action):
                return self.env.step(action)

            def close(self):
                close = getattr(self.env, "close", None)
                if callable(close):
                    close()

        gym_shim.Env = Env
        gym_shim.Wrapper = Wrapper
        gym_shim.spaces = gymnasium.spaces
        sys.modules["gym"] = gym_shim
        return gym_shim


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


PACSIM_CONFIGS = {
    "pacsim_state": {
        "task": "pacsim_state",
        "logdir": "./runs/dreamer_pacsim_state",
        "device": "cuda:0",
        "compile": False,
        "parallel": False,
        "envs": 1,
        "action_repeat": 1,
        "time_limit": 2000,
        "steps": 1e6,
        "eval_every": 1e3,
        "log_every": 1e3,
        "eval_episode_num": 1,
        "prefill": 2500,
        "dataset_size": 300000,
        "train_ratio": 64,
        "batch_size": 12,
        "batch_length": 64,
        "pretrain": 10,
        "discount": 0.99,
        "imag_horizon": 35,
        "video_pred_log": False,
        "size": [64, 64],
        "encoder": {
            "mlp_keys": "ranges|velocity|rpm|steer|imu|last_action",
            "cnn_keys": "$^",
        },
        "decoder": {
            "mlp_keys": "ranges|velocity|rpm|steer|imu|last_action",
            "cnn_keys": "$^",
        },
        "pacsim_cam_sim": False,
        "pacsim_include_camera_obs": False,
        "pacsim_include_state_obs": True,
        "pacsim_camera_key": "cameraFront",
        "pacsim_reward_mode": "conservative",
        "pacsim_print_step_status": False,
        "pacsim_verbose_env": False,
        "pacsim_preload_track_cache": True,
        "pacsim_camera_config_file": "",
        "pacsim_video_every": 0,
        # Episode files are useful for resuming training, but can be very large
        # with full-resolution camera observations. Launchers may disable them.
        "pacsim_save_episodes": True,
    },
    "pacsim_vision": {
        "task": "pacsim_vision",
        "logdir": "./runs/dreamer_pacsim_vision_fullres",
        "device": "cuda:0",
        "compile": False,
        "parallel": False,
        "envs": 1,
        "action_repeat": 1,
        "time_limit": 2000,
        "steps": 1e6,
        "eval_every": 1e2,
        "log_every": 1e2,
        "eval_episode_num": 1,
        "prefill": 2500,
        "dataset_size": 10000,
        "train_ratio": 32,
        "batch_size": 2,
        "batch_length": 32,
        "pretrain": 10,
        "discount": 0.99,
        "imag_horizon": 35,
        # Video summaries are expensive and can quickly consume disk space.
        # Enable explicitly with --video_pred_log true when needed.
        "video_pred_log": False,
        "size": [256, 306],
        "dyn_hidden": 1000,
        "dyn_deter": 1000,
        "units": 1000,
        "encoder": {
            "mlp_keys": "rpm|imu|steer|last_action",
            "cnn_keys": "cameraLeft|cameraFront|cameraRight",
        },
        "decoder": {
            "mlp_keys": "rpm|imu|steer|last_action",
            "cnn_keys": "cameraLeft|cameraFront|cameraRight",
        },
        "actor": {"layers": 3, "outscale": 0.1, "entropy": 1e-3},
        "critic": {"layers": 3},
        "reward_head": {"layers": 3},
        "cont_head": {"layers": 3},
        "pacsim_cam_sim": True,
        "pacsim_include_camera_obs": True,
        "pacsim_include_state_obs": True,
        "pacsim_state_obs_keys": "rpm|imu|steer|last_action",
        "pacsim_camera_key": "all",
        "pacsim_reward_mode": "conservative",
        "pacsim_print_step_status": False,
        "pacsim_verbose_env": False,
        "pacsim_preload_track_cache": True,
        "pacsim_camera_config_file": "",
        "pacsim_video_every": 0,
        "pacsim_save_episodes": True,
    },
    "pacsim_smoke": {
        "steps": 8,
        "eval_every": 4,
        "eval_episode_num": 0,
        "prefill": 70,
        "pretrain": 1,
        "batch_size": 2,
        "batch_length": 16,
        "train_ratio": 8,
        "dataset_size": 10000,
        "device": "cpu",
        "video_pred_log": False,
    },
}


def load_configs():
    import ruamel.yaml

    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    configs = yaml.load((DREAMER_ROOT / "configs.yaml").read_text())
    configs.update(PACSIM_CONFIGS)
    return configs


def make_pacsim_env_factory(dreamer_module, gym_module):
    original_make_env = dreamer_module.make_env

    def make_env(config, mode, env_id):
        suite, _ = config.task.split("_", 1)
        if suite != "pacsim":
            return original_make_env(config, mode, env_id)

        from pacsim_dreamer_adapter import PacsimDreamerEnv

        image_size = tuple(int(v) for v in config.size)
        camera_config_file = config.pacsim_camera_config_file or None
        env = PacsimDreamerEnv(
            gym_module=gym_module,
            seed=config.seed + env_id,
            cam_sim=config.pacsim_cam_sim,
            include_camera_obs=config.pacsim_include_camera_obs,
            include_state_obs=config.pacsim_include_state_obs,
            state_obs_keys=getattr(config, "pacsim_state_obs_keys", ""),
            image_size=image_size,
            camera_key=config.pacsim_camera_key,
            reward_mode=config.pacsim_reward_mode,
            print_step_status=config.pacsim_print_step_status,
            verbose_env=config.pacsim_verbose_env,
            preload_track_cache=config.pacsim_preload_track_cache,
            camera_config_file=camera_config_file,
        )
        env = dreamer_module.wrappers.TimeLimit(env, config.time_limit)
        env = dreamer_module.wrappers.SelectAction(env, key="action")
        env = dreamer_module.wrappers.UUID(env)
        return env

    return make_env


def patch_dreamer_conv_modules(networks_module, torch_module):
    original_decoder = networks_module.ConvDecoder

    class RectConvEncoder(torch_module.nn.Module):
        def __init__(
            self,
            input_shape,
            depth=32,
            act="SiLU",
            norm=True,
            kernel_size=4,
            minres=4,
        ):
            super().__init__()
            act_cls = getattr(torch_module.nn, act)
            input_h, input_w, input_ch = input_shape
            stages = int(math.log2(min(input_h, input_w)) - math.log2(minres))
            in_dim = input_ch
            out_dim = depth
            layers = []
            for _ in range(stages):
                layers.append(
                    networks_module.Conv2dSamePad(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        kernel_size=kernel_size,
                        stride=2,
                        bias=False,
                    )
                )
                if norm:
                    layers.append(networks_module.ImgChLayerNorm(out_dim))
                layers.append(act_cls())
                in_dim = out_dim
                out_dim *= 2

            self.layers = torch_module.nn.Sequential(*layers)
            self.layers.apply(networks_module.tools.weight_init)
            with torch_module.no_grad():
                dummy = torch_module.zeros(1, input_h, input_w, input_ch)
                dummy = dummy.permute(0, 3, 1, 2)
                out = self.layers(dummy)
            self.outdim = int(out.reshape(1, -1).shape[-1])

        def forward(self, obs):
            obs -= 0.5
            x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
            x = x.permute(0, 3, 1, 2)
            x = self.layers(x)
            x = x.reshape([x.shape[0], -1])
            return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])

    class RectConvDecoder(original_decoder):
        def __init__(self, feat_size, shape=(3, 64, 64), *args, **kwargs):
            self._target_shape = tuple(shape)
            square_shape = (shape[0], shape[1], shape[1])
            networks_module.ConvDecoder = original_decoder
            try:
                original_decoder.__init__(self, feat_size, square_shape, *args, **kwargs)
            finally:
                networks_module.ConvDecoder = RectConvDecoder

        def forward(self, features, dtype=None):
            mean = original_decoder.forward(self, features, dtype=dtype)
            channels, target_h, target_w = self._target_shape
            if mean.shape[-3:-1] == (target_h, target_w):
                return mean
            flat = mean.reshape((-1,) + tuple(mean.shape[-3:]))
            flat = flat.permute(0, 3, 1, 2)
            flat = torch_module.nn.functional.interpolate(
                flat,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            flat = flat.permute(0, 2, 3, 1)
            return flat.reshape(mean.shape[:-3] + (target_h, target_w, channels))

    networks_module.ConvEncoder = RectConvEncoder
    networks_module.ConvDecoder = RectConvDecoder


def patch_dreamer_world_model(models_module, torch_module):
    original_preprocess = models_module.WorldModel.preprocess

    def preprocess(self, obs):
        obs = original_preprocess(self, obs)
        for key, value in list(obs.items()):
            if key == "image":
                continue
            if len(value.shape) >= 4 and value.shape[-1] == 3:
                obs[key] = value / 255.0
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        post_pred = self.heads["decoder"](self.dynamics.get_feat(states))
        init = {key: value[:, -1] for key, value in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        prior_pred = self.heads["decoder"](self.dynamics.get_feat(prior))

        keys = [
            key for key in ("cameraLeft", "cameraFront", "cameraRight", "image")
            if key in data and key in post_pred and key in prior_pred
        ]
        if not keys:
            raise RuntimeError("No decoded image keys available for video prediction.")

        truths = [data[key][:6] for key in keys]
        recons = [post_pred[key].mode()[:6, :5] for key in keys]
        openls = [prior_pred[key].mode() for key in keys]
        truth = torch_module.cat(truths, dim=3)
        model = torch_module.cat(
            [
                torch_module.cat(recons, dim=3),
                torch_module.cat(openls, dim=3),
            ],
            dim=1,
        )
        error = (model - truth + 1.0) / 2.0
        return torch_module.cat([truth, model, error], 2)

    models_module.WorldModel.preprocess = preprocess
    models_module.WorldModel.video_pred = video_pred


def configure_episode_persistence(tools_module, enabled):
    """Optionally avoid persisting the online replay buffer as episode files."""
    if enabled:
        return

    def discard_episodes(directory, episodes):
        # Dreamer keeps these episodes in its in-memory replay cache, so this
        # affects only restart persistence, not training-time replay sampling.
        return True

    tools_module.save_episodes = discard_episodes


def configure_video_logging(tools_module, every):
    """Keep at most one TensorBoard video per requested step interval."""
    if not every:
        return
    original_video = tools_module.Logger.video

    def sparse_video(self, name, value):
        next_step = getattr(self, "_pacsim_next_video_step", every)
        if self.step < next_step:
            return
        original_video(self, name, value)
        self._pacsim_next_video_step = ((self.step // every) + 1) * every

    tools_module.Logger.video = sparse_video


def parse_args(configs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["pacsim_state"])
    known, remaining = parser.parse_known_args()

    defaults = {}
    for name in ["defaults", *known.configs]:
        if name not in configs:
            raise KeyError(f"Unknown Dreamer config '{name}'")
        recursive_update(defaults, configs[name])

    import tools

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda item: item[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(remaining)


def seed_logdir(logdir, seed):
    """Give every seed a stable output directory relative to pipeline."""
    path = pathlib.Path(logdir).expanduser()
    if not path.is_absolute():
        path = PIPELINE_ROOT / path
    suffix = f"_seed_{seed}"
    if path.name.endswith(suffix):
        return str(path)
    return str(path.with_name(f"{path.name}{suffix}"))


def main():
    if not DREAMER_ROOT.exists():
        raise RuntimeError(
            "Missing external/dreamerv3-torch. Run: "
            "git submodule update --init --recursive"
        )
    sys.path.insert(0, str(PIPELINE_ROOT))
    sys.path.insert(0, str(DREAMER_ROOT))
    gym_module = install_gym_compat()

    import dreamer
    import models
    import networks
    import torch
    import tools

    configs = load_configs()
    config = parse_args(configs)
    # dreamer.main seeds Python, NumPy, PyTorch, and CUDA from config.seed.
    # Keep all on-disk outputs separate so a different seed cannot resume or
    # overwrite this run's checkpoint and episode data.
    config.logdir = seed_logdir(config.logdir, config.seed)
    print(f"Dreamer seed: {config.seed}")
    print(f"Video prediction logging: {config.video_pred_log}")
    print(f"TensorBoard video interval: {config.pacsim_video_every or 'unlimited'}")
    print(f"Persist episode replay: {config.pacsim_save_episodes}")
    configure_episode_persistence(tools, config.pacsim_save_episodes)
    configure_video_logging(tools, config.pacsim_video_every)
    patch_dreamer_conv_modules(networks, torch)
    patch_dreamer_world_model(models, torch)
    dreamer.make_env = make_pacsim_env_factory(dreamer, gym_module)
    dreamer.main(config)


if __name__ == "__main__":
    main()
