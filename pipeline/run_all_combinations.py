import argparse
import os
import subprocess
import sys
from pathlib import Path

from artifact_names import bc_actor_path, dagger_actor_path, replay_buffer_path, state_actor_path, vae_path


REWARD_MODES = ("conservative", "aggressive")
POLICY_VARIANTS = ("clean", "noisy")
ACTOR_TYPES = ("structured", "latent")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full state -> replay buffer -> VAE -> BC -> DAgger pipeline for selected combinations."
    )
    parser.add_argument(
        "--reward-modes",
        nargs="+",
        choices=REWARD_MODES,
        default=list(REWARD_MODES),
        help="Reward presets to run.",
    )
    parser.add_argument(
        "--policy-variants",
        nargs="+",
        choices=POLICY_VARIANTS,
        default=list(POLICY_VARIANTS),
        help="State-policy variants to run: clean or noisy.",
    )
    parser.add_argument(
        "--actor-types",
        nargs="+",
        choices=ACTOR_TYPES,
        default=list(ACTOR_TYPES),
        help="Vision actor variants to train with BC and DAgger.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child scripts.")
    parser.add_argument("--state-total-timesteps", type=int, default=None, help="Override SAC total timesteps.")
    parser.add_argument("--vae-epochs", type=int, default=None, help="Override VAE training epochs.")
    parser.add_argument("--vae-batch-size", type=int, default=None, help="Override VAE batch size.")
    parser.add_argument(
        "--include-last-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the previous normalized action in state/vision observations.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip a stage when its expected output already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with later combinations if a stage fails.")
    return parser.parse_args()


def run_command(cmd, cwd, dry_run):
    print()
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def maybe_run(cmd, cwd, output_path, skip_existing, dry_run):
    if skip_existing and output_path.exists():
        print(f"Skipping existing output: {output_path}", flush=True)
        return
    run_command(cmd, cwd, dry_run)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    os.chdir(repo_root)

    for reward_mode in args.reward_modes:
        for policy_variant in args.policy_variants:
            noisy = policy_variant == "noisy"
            rb_path = repo_root / replay_buffer_path(reward_mode, noisy, args.include_last_action)
            vae_model_path = repo_root / vae_path(reward_mode, noisy, args.include_last_action)

            print()
            print(f"=== reward={reward_mode} policy={policy_variant} ===", flush=True)

            state_cmd = [
                args.python,
                str(script_dir / "sac_continous_action.py"),
                "--reward-mode",
                reward_mode,
            ]
            if noisy:
                state_cmd.append("--noise-augment")
            if not args.include_last_action:
                state_cmd.append("--no-include-last-action")
            if args.state_total_timesteps is not None:
                state_cmd.extend(["--total-timesteps", str(args.state_total_timesteps)])
            state_output = repo_root / state_actor_path(reward_mode, noisy, args.include_last_action)

            fill_cmd = [
                args.python,
                str(script_dir / "fillRB.py"),
                "--reward-mode",
                reward_mode,
            ]
            if noisy:
                fill_cmd.append("--noisy")
            if not args.include_last_action:
                fill_cmd.append("--no-include-last-action")

            vae_cmd = [
                args.python,
                str(script_dir / "vae.py"),
                "--buffer",
                str(rb_path),
                "--reward-mode",
                reward_mode,
                "--save-path",
                str(vae_model_path),
            ]
            if noisy:
                vae_cmd.append("--noisy")
            if not args.include_last_action:
                vae_cmd.append("--no-include-last-action")
            if args.vae_epochs is not None:
                vae_cmd.extend(["--epochs", str(args.vae_epochs)])
            if args.vae_batch_size is not None:
                vae_cmd.extend(["--batch-size", str(args.vae_batch_size)])

            try:
                maybe_run(state_cmd, repo_root, state_output, args.skip_existing, args.dry_run)
                maybe_run(fill_cmd, repo_root, rb_path, args.skip_existing, args.dry_run)
                maybe_run(vae_cmd, repo_root, vae_model_path, args.skip_existing, args.dry_run)

                for actor_type in args.actor_types:
                    print()
                    print(f"--- vision actor={actor_type} ---", flush=True)
                    latent = actor_type == "latent"
                    bc_output = repo_root / bc_actor_path(reward_mode, noisy, latent, args.include_last_action)
                    dagger_output = repo_root / dagger_actor_path(reward_mode, noisy, latent, args.include_last_action)

                    bc_cmd = [
                        args.python,
                        str(script_dir / "bc.py"),
                        "--buffer",
                        str(rb_path),
                        "--reward-mode",
                        reward_mode,
                    ]
                    dagger_cmd = [
                        args.python,
                        str(script_dir / "dagger.py"),
                        "--buffer",
                        str(rb_path),
                        "--reward-mode",
                        reward_mode,
                    ]
                    if noisy:
                        bc_cmd.append("--noisy")
                        dagger_cmd.append("--noisy")
                    if not args.include_last_action:
                        bc_cmd.append("--no-include-last-action")
                        dagger_cmd.append("--no-include-last-action")
                    if latent:
                        bc_cmd.extend(["--latent", "--vae-path", str(vae_model_path)])
                        dagger_cmd.append("--latent")

                    maybe_run(bc_cmd, repo_root, bc_output, args.skip_existing, args.dry_run)
                    maybe_run(dagger_cmd, repo_root, dagger_output, args.skip_existing, args.dry_run)
            except subprocess.CalledProcessError as exc:
                print(f"Stage failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", flush=True)
                if not args.continue_on_error:
                    raise


if __name__ == "__main__":
    main()
