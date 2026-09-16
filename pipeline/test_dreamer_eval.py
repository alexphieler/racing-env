"""Dependency-light checks for the recurrent evaluation bridge."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np


class DreamerBridgeTest(unittest.TestCase):
    def test_history_and_episode_reset(self):
        source = Path(__file__).with_name('dreamer_eval.py')
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        # Test inference/reset without importing the simulator or PyTorch.
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('action', 'reset')]
        namespace = {'np': np, 'torch': SimpleNamespace(no_grad=lambda: lambda fn: fn)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
        policy = namespace['DreamerEvaluationPolicy']()
        calls = []

        def convert(obs, is_first=False):
            np.testing.assert_array_equal(obs['rpm'], [4, 5])
            return dict(obs, is_first=is_first, is_terminal=False)

        def agent(obs, reset, state, training):
            calls.append((bool(obs['is_first'][0]), state, training))
            self.assertEqual(obs['rpm'].shape, (1, 2))
            return {'action': np.zeros((1, 5))}, 'recurrent-state'

        policy.adapter = SimpleNamespace(_convert_obs=convert)
        policy.agent = agent
        obs = {'rpm': np.arange(6).reshape(3, 2)}
        policy.reset()
        self.assertEqual(policy.action(obs).shape, (1, 5))
        policy.action(obs)
        policy.reset()
        policy.action(obs)
        self.assertEqual(calls, [(True, None, False), (False, 'recurrent-state', False), (True, None, False)])


if __name__ == '__main__':
    unittest.main()
