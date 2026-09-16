#!/usr/bin/env python3
"""Plot a grid of PacSim YAML tracks without importing the simulator.

Examples:
    python3 pipeline/plot_tracks.py
    python3 pipeline/plot_tracks.py --tracks FSG25 FSI24 FSG19 --columns 3
    python3 pipeline/plot_tracks.py --track-dir pipeline/generated_tracks_subset
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACK_DIR = REPO_ROOT / "tracks"
REFERENCE_ORDER = (
    "FSG25", "FSI24", "FSG19", "FSE22",
    "FSS19", "FSG21", "FSS22_V1", "FSO20",
    "FSG23", "FSE23", "FSCZ24", "FSE24",
    "FSS22_V2", "FSE22_test", "FSG24", "FSCZ25",
)


def track_files(directory, names=None):
    """Preserve explicit order; otherwise use reference order or directory order."""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Track directory does not exist: {directory}")
    if names is None and directory == DEFAULT_TRACK_DIR.resolve():
        names = REFERENCE_ORDER
    if names is None:
        files = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        )
    else:
        files = []
        for name in names:
            path = directory / name
            if path.suffix.lower() in {".yaml", ".yml"}:
                matches = [path] if path.is_file() else []
            else:
                matches = [p for p in (path.with_suffix(".yaml"), path.with_suffix(".yml"))
                           if p.is_file()]
            if len(matches) != 1:
                raise ValueError(f"Expected one YAML file for {name!r} in {directory}; found {len(matches)}")
            files.append(matches[0])
    if not files:
        raise ValueError(f"No YAML tracks found in {directory}")
    return files


def load_track(path):
    """Read lane coordinates and the declared start pose in the track frame."""
    try:
        with path.open(encoding="utf-8") as stream:
            document = yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        track = document["track"]
        lanes = []
        for side in ("left", "right"):
            points = np.asarray([cone["position"] for cone in track[side]], dtype=float)
            if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
                raise ValueError(f"{side} lane must contain a nonempty list of positions")
            if not np.isfinite(points[:, :2]).all():
                raise ValueError(f"{side} lane contains non-finite coordinates")
            lanes.append(points[:, :2])
        start = np.asarray(track["start"]["position"], dtype=float)
        if start.ndim != 1 or start.size < 2 or not np.isfinite(start[:2]).all():
            raise ValueError("start.position must contain finite x and y coordinates")
        return lanes[0], lanes[1], start[:2]
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid track {path}: {exc}") from exc


def plot_tracks(files, *, columns=4, figsize=None, cone_size=8, start_size=18,
                title_size=12, show_titles=True, show_start=True, show_frame=True, common_scale=False):
    """Return the figure; coordinates keep their original orientation and aspect."""
    tracks = [(path, *load_track(path)) for path in files]
    rows = math.ceil(len(tracks) / columns)
    # Reserve title space in physical units. With automatic sizing, larger
    # titles grow the canvas while every track panel stays 2.2 inches tall.
    title_space = (title_size + 5) / 72 + 0.10 if show_titles else 0.12
    bottom_space = 0.18
    figsize = figsize or (4 * columns, rows * (2.2 + title_space) + bottom_space)
    panel_height = (figsize[1] - bottom_space - rows * title_space) / rows
    if panel_height <= 0:
        raise ValueError("Figure height is too small for the titles; increase --figsize height")
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    fig.subplots_adjust(left=0.012, right=0.993,
                        bottom=bottom_space / figsize[1], top=1 - title_space / figsize[1],
                        wspace=0.035, hspace=title_space / panel_height)

    # Fit equal x/y units into equally sized rectangular panels. Expanding the
    # limits ourselves avoids stretching tracks or shrinking individual axes.
    panel = axes[0, 0].get_position()
    panel_ratio = (panel.width * figsize[0]) / (panel.height * figsize[1])
    bounds = []
    for _, left, right, start in tracks:
        points = np.vstack((left, right, start)) if show_start else np.vstack((left, right))
        low, high = points.min(axis=0), points.max(axis=0)
        extent = high - low
        width = max(extent[0], extent[1] * panel_ratio, 1.0) * 1.10
        bounds.append(((low + high) / 2, width))
    shared_width = max(width for _, width in bounds)

    for ax, (path, left, right, start), (center, width) in zip(axes.flat, tracks, bounds):
        ax.scatter(left[:, 0], left[:, 1], s=cone_size, color="blue", edgecolors="none")
        ax.scatter(right[:, 0], right[:, 1], s=cone_size, color="gold", edgecolors="none")
        if show_start:
            ax.scatter(*start, s=start_size, color="red", edgecolors="none", zorder=3)
        if common_scale:
            width = shared_width
        height = width / panel_ratio
        ax.set_xlim(center[0] - width / 2, center[0] + width / 2)
        ax.set_ylim(center[1] - height / 2, center[1] + height / 2)
        ax.set_aspect("equal", adjustable="box")
        if show_titles:
            ax.set_title(path.stem.replace("_", " "), fontsize=title_size, fontweight="bold", pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(show_frame)
            spine.set_color("#333333")
            spine.set_linewidth(0.7)
    for ax in list(axes.flat)[len(tracks):]:
        ax.set_visible(False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR,
                        help="Directory of PacSim .yaml/.yml files (default: repository tracks/).")
    parser.add_argument("--tracks", nargs="+", metavar="NAME",
                        help="Track names or filenames in plotting order; defaults to the reference's 16 tracks, "
                             "or all YAML files sorted by name when using a different directory.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "paper" / "tracks")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf", "svg"])
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Plot only the first N tracks in the selected order.")
    parser.add_argument("--figsize", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Fixed figure size in inches; default keeps panels 2.2 inches tall "
                             "and adds title space automatically, with 4 inches per column.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution (default: 180).")
    parser.add_argument("--cone-size", type=float, default=8, help="Cone marker area in points squared (default: 8).")
    parser.add_argument("--start-size", type=float, default=18, help="Start marker area in points squared.")
    parser.add_argument("--title-size", type=float, default=12, help="Track-title font size in points (default: 12).")
    parser.add_argument("--no-titles", action="store_true", help="Hide track names and remove title spacing.")
    parser.add_argument("--no-start", action="store_true", help="Hide the red YAML start-position markers.")
    parser.add_argument("--no-frame", action="store_true", help="Hide panel borders.")
    parser.add_argument("--common-scale", action="store_true",
                        help="Use the same coordinate scale across all panels; default fits each track independently.")
    args = parser.parse_args()
    if args.columns < 1 or args.dpi < 1:
        parser.error("--columns and --dpi must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    sizes = [args.cone_size, args.start_size, args.title_size, *(args.figsize or [])]
    if any(not math.isfinite(value) or value <= 0 for value in sizes):
        parser.error("Figure dimensions, marker sizes, and title size must be finite and positive")

    try:
        files = track_files(args.track_dir, args.tracks)
        if args.limit is not None:
            files = files[:args.limit]
        fig = plot_tracks(files, columns=args.columns, figsize=args.figsize,
                          cone_size=args.cone_size, start_size=args.start_size, title_size=args.title_size,
                          show_titles=not args.no_titles, show_start=not args.no_start, show_frame=not args.no_frame,
                          common_scale=args.common_scale)
        try:
            output_dir = args.output_dir.expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            for extension in dict.fromkeys(args.formats):
                output = output_dir / f"tracks_overview.{extension}"
                fig.savefig(output, dpi=args.dpi, facecolor="white")
                print(f"Saved {len(files)} tracks: {output}")
        finally:
            plt.close(fig)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
