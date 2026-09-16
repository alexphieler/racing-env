"""Replay-verified teacher takeover experiments on failed learner episodes."""
import csv
from pathlib import Path

import numpy as np


def run_takeovers(reset, step, state, teacher_action, actions, states, times,
                  baseline_info, offsets, directory, episode, track):
    """Callbacks expose the environment; states include pose and stacked sensors.

    A full learner-action replay must reproduce the failure before interventions
    are interpreted. Every intervention also verifies its own replay prefix.
    """
    def matches(obs, index):
        return np.allclose(state(obs), states[index], rtol=1e-6, atol=1e-5)

    obs = reset()
    replay_ok = matches(obs, 0)
    for index, action in enumerate(actions):
        if not replay_ok:
            break
        obs, _, terminated, truncated, info = step(action.copy())
        replay_ok = matches(obs, index + 1)
        if terminated or truncated:
            replay_ok &= index == len(actions) - 1
            replay_ok &= all(info.get(key) == baseline_info.get(key)
                             for key in ('collided', 'timeout', 'crossed_line', 'laptime'))
            break
        if index == len(actions) - 1:
            replay_ok = False

    rows = []
    for offset in offsets:
        # Take over no later than the requested lead time, clamped to reset.
        target = max(times[0], times[-1] - offset)
        switch = max(0, min(len(actions) - 1, int(np.searchsorted(times, target, side='right') - 1)))
        row = dict(episode=episode, track=track, requested_lead_s=offset,
                   actual_lead_s=times[-1] - times[switch], takeover_time_s=times[switch],
                   takeover_step=switch, baseline_failure_time_s=times[-1],
                   baseline_replay_verified=bool(replay_ok), status='invalid_baseline_replay',
                   passed='', reason='', episodic_return='', teacher_steps=0)
        if replay_ok:
            obs = reset()
            prefix_ok = matches(obs, 0)
            total_reward = 0.0
            for index in range(switch):
                if not prefix_ok:
                    break
                obs, reward, terminated, truncated, _ = step(actions[index].copy())
                total_reward += float(reward)
                prefix_ok = matches(obs, index + 1) and not (terminated or truncated)
            row['status'] = 'invalid_prefix_replay'
            if prefix_ok:
                done = False
                while not done:
                    obs, reward, terminated, truncated, info = step(teacher_action(obs))
                    total_reward += float(reward)
                    row['teacher_steps'] += 1
                    done = terminated or truncated
                row.update(status='valid', passed=float(info.get('laptime', 0)) > 0,
                           reason=','.join(k for k in ('collided', 'timeout', 'crossed_line') if info.get(k)) or 'other',
                           episodic_return=total_reward)
        rows.append(row)
        print(f"Takeover {track}, lead={row['actual_lead_s']:.2f}s: "
              f"{row['status']}, passed={row['passed']}, reason={row['reason']}")
    path = Path(directory) / 'takeover_summary.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return rows
