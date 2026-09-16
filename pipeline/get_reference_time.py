"""Derive per-track reference times from evaluation reports.

For each track, this script selects the lowest valid aggregate `time` from the
CSV reports in eval/. It ignores per-seed summary reports and failed tracks,
whose time cell is blank in multi-seed reports (or zero in legacy reports).
"""

import argparse
import csv
import glob
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print the best aggregate evaluation time for every track."
    )
    parser.add_argument(
        "--eval-dir",
        default="eval",
        help="Directory containing eval.py CSV reports (default: eval).",
    )
    return parser.parse_args()


def valid_time(row):
    """Return a positive report time, or None for a failure/malformed row."""
    try:
        lap_time = float(row.get("time", ""))
    except (TypeError, ValueError):
        return None
    return lap_time if lap_time > 0 else None


def main():
    args = parse_args()
    pattern = os.path.join(args.eval_dir, "*.csv")
    report_paths = [
        path
        for path in sorted(glob.glob(pattern))
        if not os.path.basename(path).startswith("seed_summary_")
    ]
    if not report_paths:
        raise SystemExit(f"No aggregate evaluation reports found at: {pattern}")

    reference_times = {}
    source_reports = {}
    skipped_reports = []
    for report_path in report_paths:
        with open(report_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames or "Track" not in reader.fieldnames or "time" not in reader.fieldnames:
                skipped_reports.append(report_path)
                continue
            for row in reader:
                track = (row.get("Track") or "").strip()
                lap_time = valid_time(row)
                if not track or lap_time is None:
                    continue
                if lap_time < reference_times.get(track, float("inf")):
                    reference_times[track] = lap_time
                    source_reports[track] = os.path.basename(report_path)

    if not reference_times:
        raise SystemExit("No positive aggregate lap times found in evaluation reports.")

    print("# Best average successful lap time across evaluation reports.")
    print("TRACK_TIME_REFERENCES = {")
    for track in sorted(reference_times):
        print(f'    "{track}": {reference_times[track]:.6f},  # {source_reports[track]}')
    print("}")

    if skipped_reports:
        print(
            f"\nSkipped {len(skipped_reports)} CSV report(s) without Track/time columns:",
            file=sys.stderr,
        )
        for report_path in skipped_reports:
            print(f"  {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
