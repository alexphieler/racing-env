#!/usr/bin/env bash
# Evaluate state SAC, vision BC, and vision DAgger for seeds 0–9.
set -euo pipefail

dry_run=false
case "${1:-}" in
    --dry-run) dry_run=true; shift ;;
    -h|--help)
        echo "Usage: bash run_eval_sweep.sh [--dry-run]"
        echo "Evaluates seeds 0–9, conservative/aggressive, clean/noisy, state SAC/vision BC/vision DAgger."
        echo "Set NIPS_ENV_PYTHON to override python3."
        exit 0
        ;;
esac
if [[ $# -ne 0 ]]; then
    echo "Usage: bash run_eval_sweep.sh [--dry-run]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"
python_bin=${NIPS_ENV_PYTHON:-python3}
failures=()
for reward in conservative aggressive; do
    for noise in clean noisy; do
        for method in state-sac bc dagger; do
            args=(--reward-mode "$reward" --seeds 0 1 2 3 4 5 6 7 8 9)
            if [[ $noise == noisy ]]; then
                args+=(--noisy)
            fi
            case "$method" in
                state-sac) args+=(--model-type state) ;;
                bc) args+=(--model-type vision) ;;
                dagger) args+=(--model-type vision --dagger) ;;
            esac
            command=("$python_bin" "$script_dir/eval.py" "${args[@]}")
            printf '%q ' "${command[@]}"
            printf '\n'
            if "$dry_run"; then
                continue
            fi
            if "${command[@]}"; then
                echo "FINISHED: $method $reward $noise"
            else
                failures+=("$method $reward $noise")
                echo "FAILED: $method $reward $noise" >&2
            fi
        done
    done
done

if [[ ${#failures[@]} -gt 0 ]]; then
    printf 'Failed configuration: %s\n' "${failures[@]}" >&2
    exit 1
fi
