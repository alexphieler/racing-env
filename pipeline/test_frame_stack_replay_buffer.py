"""Regression tests for on-demand frame-stack replay storage.

Run with ``PYTHONPATH=pipeline python -m unittest pipeline/test_frame_stack_replay_buffer.py``.
"""

import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch

from buffers import DictReplayBuffer, FrameStackDictReplayBuffer, load_replay_buffer, save_replay_buffer


def stacked_space(stack_size=3):
    return gym.spaces.Dict(
        {
            "sensor": gym.spaces.Box(-np.inf, np.inf, shape=(stack_size, 2), dtype=np.float32),
            "cameraFront": gym.spaces.Box(0, 255, shape=(stack_size, 1, 2, 3), dtype=np.uint8),
        }
    )


def stack(history, frame, stack_size=3):
    history = (history + [frame])[-stack_size:]
    history = [history[0]] * (stack_size - len(history)) + history
    return history, history


def observation(frames):
    values = np.asarray(frames, dtype=np.float32)
    camera = np.asarray(frames, dtype=np.uint8)[:, None, None, None]
    camera = np.broadcast_to(camera, (len(frames), 1, 2, 3)).copy()
    return {"sensor": np.stack((values, values + 0.5), axis=1)[None], "cameraFront": camera[None]}


class FrameStackDictReplayBufferTest(unittest.TestCase):
    def test_reconstructs_reset_and_terminal_stacks_exactly(self):
        space = stacked_space()
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        eager = DictReplayBuffer(12, space, action_space, device="cpu", n_envs=1, handle_timeout_termination=False)
        compact = FrameStackDictReplayBuffer(
            12, space, action_space, device="cpu", n_envs=1, frame_stack=3, handle_timeout_termination=False
        )

        # Two episodes.  The first ends after frame 3; the second begins at 10.
        episodes = ([0, 1, 2, 3], [10, 11, 12])
        for episode in episodes:
            history = []
            for index, frame in enumerate(episode[:-1]):
                history, current = stack(history, frame)
                _, following = stack(history, episode[index + 1])
                done = index == len(episode) - 2
                args = (
                    observation(current),
                    observation(following),
                    np.array([[0.25]], dtype=np.float32),
                    np.array([1.0], dtype=np.float32),
                    np.array([done]),
                    [{}],
                )
                eager.add(*args)
                compact.add(*args)

        for transition in range(compact.size()):
            eager_data = eager._get_samples(np.array([transition]))
            compact_data = compact._get_samples(np.array([transition]))
            for key in space.spaces:
                torch.testing.assert_close(eager_data.observations[key], compact_data.observations[key])
                torch.testing.assert_close(eager_data.next_observations[key], compact_data.next_observations[key])

    def test_uses_about_one_sixth_of_eager_observation_storage(self):
        space = stacked_space()
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        eager = DictReplayBuffer(12, space, action_space, device="cpu", n_envs=1, handle_timeout_termination=False)
        compact = FrameStackDictReplayBuffer(
            12, space, action_space, device="cpu", n_envs=1, frame_stack=3, handle_timeout_termination=False
        )
        eager_observation_bytes = sum(value.nbytes for value in eager.observations.values()) + sum(
            value.nbytes for value in eager.next_observations.values()
        )
        raw_frame_bytes = compact.observation_storage_nbytes // compact.raw_buffer_size
        self.assertEqual(compact.observation_storage_nbytes, raw_frame_bytes * (12 + 3 - 1))
        self.assertLess(compact.observation_storage_nbytes, eager_observation_bytes / 5)

    def test_reconstructs_history_across_the_ring_buffer_wrap(self):
        space = stacked_space()
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        compact = FrameStackDictReplayBuffer(
            4, space, action_space, device="cpu", n_envs=1, frame_stack=3, handle_timeout_termination=False
        )
        history = []
        # Keep a single long episode so the sampled stacks must cross the
        # physical end of the circular array.
        for frame in range(8):
            history, current = stack(history, frame)
            _, following = stack(history, frame + 1)
            compact.add(
                observation(current),
                observation(following),
                np.array([[0.0]], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([False]),
                [{}],
            )

        # Slots 0..3 now contain transitions whose current frames are 4..7.
        for slot, current_frame in enumerate(range(4, 8)):
            data = compact._get_samples(np.array([slot]))
            expected = np.array([current_frame - 2, current_frame - 1, current_frame], dtype=np.float32)
            np.testing.assert_array_equal(data.observations["sensor"].numpy()[0, :, 0], expected)
            np.testing.assert_array_equal(data.next_observations["sensor"].numpy()[0, :, 0], expected + 1)

    def test_fillrb_pickle_contract_remains_usable_by_bc_and_dagger(self):
        """The offline consumers rely on observation_space plus sample()[0:2]."""
        space = stacked_space()
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        buffer = FrameStackDictReplayBuffer(
            4, space, action_space, device="cpu", n_envs=1, frame_stack=3, handle_timeout_termination=False
        )
        history = []
        for frame in range(2):
            history, current = stack(history, frame)
            _, following = stack(history, frame + 1)
            current_obs = {key: value[0] for key, value in observation(current).items()}
            following_obs = {key: value[0] for key, value in observation(following).items()}
            buffer.add(
                current_obs, following_obs, np.array([0.0], dtype=np.float32),
                np.float32(0.0), False, {},
            )

        with tempfile.NamedTemporaryFile(suffix=".pkl") as file:
            save_replay_buffer(file.name, buffer)
            restored = load_replay_buffer(file.name)

        self.assertEqual(restored.observation_space["cameraFront"].shape, (3, 1, 2, 3))
        sample = restored.sample(1)
        # BC and DAgger both access these tuple positions before flattening.
        self.assertEqual(sample[0]["cameraFront"].shape, (1, 3, 1, 2, 3))
        self.assertEqual(sample[1].shape, (1, 1))

if __name__ == "__main__":
    unittest.main()
