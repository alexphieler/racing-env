"""Inference-only bridge from stacked evaluation observations to Dreamer."""
import argparse
import importlib
from pathlib import Path
import sys

import numpy as np
import torch


def default_checkpoint(model_type, seed):
    from dreamer_pacsim import PIPELINE_ROOT, seed_logdir
    name = 'dreamer_pacsim_state' if model_type == 'state' else 'dreamer_pacsim_vision_fullres'
    return str(Path(seed_logdir(PIPELINE_ROOT / 'runs' / name, seed)) / 'latest.pt')


def load_dreamer_modules():
    from dreamer_pacsim import DREAMER_ROOT, install_gym_compat
    install_gym_compat()
    # Dreamer and this pipeline both have a top-level networks.py.
    names = ('networks', 'models', 'tools', 'exploration', 'dreamer', 'parallel', 'envs', 'envs.wrappers')
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(DREAMER_ROOT))
        modules = [importlib.import_module(name) for name in ('dreamer', 'models', 'networks')]
    finally:
        sys.path[:] = original_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return modules


class DreamerEvaluationPolicy:
    def __init__(self, checkpoint_path, env, args, device):
        from dreamer_pacsim import (
            load_configs, recursive_update, patch_dreamer_conv_modules,
            patch_dreamer_world_model,
        )
        from pacsim_dreamer_adapter import PacsimDreamerEnv
        import gymnasium as gym

        dreamer, models, dreamer_networks = load_dreamer_modules()
        configs = load_configs()
        values = {}
        for name in ('defaults', 'pacsim_' + args.model_type):
            recursive_update(values, configs[name])
        if args.dreamer_config:
            from ruamel.yaml import YAML
            overrides = YAML(typ='safe').load(Path(args.dreamer_config).read_text())
            if not isinstance(overrides, dict):
                raise ValueError('Dreamer config must be a YAML mapping of training settings')
            recursive_update(values, overrides)
        if bool(values['pacsim_include_camera_obs'] or values['pacsim_cam_sim']) != (args.model_type == 'vision'):
            raise ValueError('Dreamer camera configuration does not match --model-type')
        if values.get('action_repeat', 1) != 1:
            raise ValueError('Dreamer evaluation currently requires action_repeat=1')
        if values['pacsim_reward_mode'] != args.reward_mode:
            raise ValueError('Dreamer training reward preset must match --reward-mode; supply --dreamer-config for non-default training')
        if values.get('pacsim_camera_config_file'):
            raise ValueError('Custom Dreamer camera configurations are not yet supported in eval.py')
        values.update(device=str(device), compile=False, num_actions=int(np.prod(env.action_space.shape)))
        config = argparse.Namespace(**values)
        patch_dreamer_conv_modules(dreamer_networks, torch)
        patch_dreamer_world_model(models, torch)

        # Reuse conversion without constructing a second simulator.
        adapter = PacsimDreamerEnv.__new__(PacsimDreamerEnv)
        adapter._env = env.unwrapped
        adapter._image_size = tuple(config.size)
        adapter._monitor_image_size = tuple(config.size)
        adapter._camera_key = config.pacsim_camera_key
        adapter._include_state_obs = config.pacsim_include_state_obs
        adapter._state_obs_keys = frozenset(filter(None, getattr(config, 'pacsim_state_obs_keys', '').split('|')))
        self.adapter = adapter
        obs_space = adapter._build_observation_space(gym.spaces)
        self.agent = dreamer.Dreamer(obs_space, env.action_space, config,
                                     argparse.Namespace(step=0), dataset=None).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or 'agent_state_dict' not in checkpoint:
            raise ValueError('Expected a Dreamer latest.pt with agent_state_dict')
        self.agent.load_state_dict(checkpoint['agent_state_dict'], strict=True)
        self.agent.requires_grad_(False)
        self.agent.eval()
        self.reset()

    def reset(self):
        self.state = None
        self.is_first = True

    @torch.no_grad()
    def action(self, stacked_obs):
        obs = {key: np.asarray(value)[-1] for key, value in stacked_obs.items()}
        converted = self.adapter._convert_obs(obs, is_first=self.is_first)
        batched = {key: np.expand_dims(value, 0) for key, value in converted.items()}
        output, self.state = self.agent(batched, np.array([self.is_first]),
                                       self.state, training=False)
        self.is_first = False
        return output['action']
