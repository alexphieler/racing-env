"""Per-action training metrics and evaluation trace exports."""
import csv
from pathlib import Path

import numpy as np

ACTION_NAMES = ('steer', 'wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr')


def log_action_errors(writer, predicted, target, step, group='all'):
    error = (predicted.detach() - target.detach()).float()
    metrics = (error.square().mean(0).cpu().tolist(),
               error.abs().mean(0).cpu().tolist(), error.mean(0).cpu().tolist())
    for metric, values in zip(('mse', 'mae', 'bias'), metrics):
        for name, value in zip(ACTION_NAMES, values):
            writer.add_scalar(f'action_errors/{group}/{metric}/{name}', value, step)


def save_episode_inspection(directory, episode, track, rows, success, info):
    """Export all episodes, including the final transition, without auto-diagnosing failures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f'{episode:03d}_{Path(track).stem}_{"passed" if success else "failed"}'
    with stem.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    t = data['time_before_s']
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    for i, name in enumerate(ACTION_NAMES):
        ax = axes[0 if i == 0 else 1]
        line, = ax.plot(t, data[f'action_{name}'], label=name)
        ax.plot(t, data[f'teacher_{name}'], '--', color=line.get_color(), alpha=0.65)
    axes[0].set_ylabel('Steering command')
    axes[1].set_ylabel('Wheel commands')
    for ax in axes[:2]:
        ax.set_ylim(-1.05, 1.05)
    # Motion is measured after applying the command, so use its own timestamp.
    axes[2].plot(data['time_after_s'], data['vx_mps'], label='vx')
    axes[2].plot(data['time_after_s'], data['vy_mps'], label='vy')
    axes[2].set_ylabel('Velocity (m/s)')
    axes[3].plot(data['time_after_s'], data['center_distance_m'], label='distance to centerline')
    axes[3].set_ylabel('Distance (m)')
    axes[3].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left')
        ax.axvspan(max(t[0], data['time_after_s'][-1] - 5), data['time_after_s'][-1],
                   color='orange', alpha=0.08)
    reason = ', '.join(key for key in ('collided', 'timeout', 'crossed_line') if info.get(key)) or 'other'
    fig.suptitle(f'{track}: {"passed" if success else "failed"} ({reason})\n'
                 'Solid: executed policy commands; dashed: teacher at the same policy states; shaded: final 5 s')
    fig.tight_layout()
    fig.savefig(stem.with_suffix('.png'), dpi=140)
    plt.close(fig)
    last = t >= data['time_after_s'][-1] - 5
    summary = {'episode': episode, 'track': track, 'passed': success, 'reason': reason,
               'return': float(data['reward'].sum()), 'steps': len(rows)}
    for name in ACTION_NAMES:
        error = data[f'action_{name}'] - data[f'teacher_{name}']
        summary[f'mse_{name}'] = float(np.mean(error ** 2))
        summary[f'last5s_mae_{name}'] = float(np.mean(np.abs(error[last]))) if last.any() else float('nan')
    summary['max_speed_mps'] = float(np.max(np.hypot(data['vx_mps'], data['vy_mps'])))
    summary['final_center_distance_m'] = float(data['center_distance_m'][-1])
    path = directory / 'summary.csv'
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)
    print(f'Inspection: {stem}.png')
