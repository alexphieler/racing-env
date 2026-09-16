#!/usr/bin/env bash
# Run every reward/noise/seed combination through one workflow stage on Slurm.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_seed_workflow_sweep.sh MAX_SEED [sac|rest|all] [--time TIME]

Runs seeds 0 through MAX_SEED (inclusive), for conservative and aggressive
rewards and for both clean and noisy SAC policies. All combinations are started
in parallel. The optional workflow stage defaults to all; use sac and rest as
separate shorter Slurm allocations.

Options:
  -t, --time TIME  Slurm time limit (default: 12:00:00)

Environment overrides:
  NIPS_ENV_ROOT   Repository location (default: /home/s2738360/nips-env)
  NIPS_ENV_IMAGE  Singularity image or sandbox (default: $HOME/fs-rl-dir)
  NIPS_ENV_NVIDIA_ICD  Host Vulkan ICD JSON (default: $NIPS_ENV_ROOT/nvidia_icd_apptainer.json)
EOF
}

time_limit=12:00:00
positionals=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--time)
            if [[ $# -lt 2 || -z $2 ]]; then
                echo "$1 requires a time limit" >&2
                exit 2
            fi
            time_limit=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            positionals+=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            positionals+=("$1")
            shift
            ;;
    esac
done

if [[ ${#positionals[@]} -lt 1 || ${#positionals[@]} -gt 2 ]]; then
    usage >&2
    exit 2
fi

max_seed=${positionals[0]}
stage=${positionals[1]:-all}
if ! [[ $max_seed =~ ^[0-9]+$ ]]; then
    echo "MAX_SEED must be a non-negative integer; got: $max_seed" >&2
    exit 2
fi
case $stage in
    sac|rest|all) ;;
    *)
        echo "Stage must be sac, rest, or all; got: $stage" >&2
        exit 2
        ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${NIPS_ENV_ROOT:-"$(cd -- "$script_dir/.." && pwd)"}
pipeline_dir="$repo_root/pipeline"
image=${NIPS_ENV_IMAGE:-"$HOME/fs-rl-dir"}
nvidia_icd=${NIPS_ENV_NVIDIA_ICD:-"$repo_root/nvidia_icd_apptainer.json"}

if [[ ! -r $image ]]; then
    echo "Cannot read Singularity image or sandbox: $image" >&2
    echo "Set NIPS_ENV_IMAGE to a valid container image or sandbox directory." >&2
    exit 65
fi
if [[ ! -r $nvidia_icd ]]; then
    echo "Cannot read Vulkan ICD JSON: $nvidia_icd" >&2
    exit 66
fi
if [[ ! -x $pipeline_dir/singularity_entry.sh ]]; then
    echo "Singularity entry script is not executable: $pipeline_dir/singularity_entry.sh" >&2
    exit 69
fi

pids=()
labels=()

for reward_mode in conservative aggressive; do
    for noise_mode in clean noisy; do
        for ((seed = 0; seed <= max_seed; seed++)); do
            workflow_args=(
                --stage "$stage"
                --reward-mode "$reward_mode"
                --seed "$seed"
            )
            if [[ $noise_mode == noisy ]]; then
                workflow_args+=(--noisy)
            fi

            label="stage=$stage reward=$reward_mode noise=$noise_mode seed=$seed"
            echo "=== starting $label ==="
            (
                argument_dir=$(mktemp -d "$pipeline_dir/.seed-workflow-args.XXXXXX")
                argument_file="$argument_dir/args"
                trap 'rm -rf -- "$argument_dir"' EXIT
                printf '%s\0' "${workflow_args[@]}" > "$argument_file"

                command=(
                    srun --ntasks=1 --cpus-per-task=10 "--time=$time_limit"
                    --mem-per-cpu=8000 --nodes=1 --gres=gpu:1 --gpus-per-task=1 -L horse
                    singularity shell --nv
                    --bind "$nvidia_icd:/tmp/nvidia_icd.json"
                    --bind "$pipeline_dir:$pipeline_dir"
                    --env VK_DRIVER_FILES=/tmp/nvidia_icd.json
                    --env "NIPS_WORKFLOW_ARG_FILE=$argument_file"
                    --pwd "$pipeline_dir"
                    --shell "$pipeline_dir/singularity_entry.sh"
                    "$image"
                )
                printf '$ '
                printf '%q ' "${command[@]}"
                printf '\n'
                "${command[@]}"
            ) &
            pids+=("$!")
            labels+=("$label")
        done
    done
done

exit_status=0
for index in "${!pids[@]}"; do
    if wait "${pids[index]}"; then
        echo "=== finished ${labels[index]} ==="
    else
        echo "=== failed ${labels[index]} ===" >&2
        exit_status=1
    fi
done
exit "$exit_status"
