# General

Reward presets (conservative or aggressive) and clean/noisy variants are included in generated filenames. Model checkpoints always use `.pt`.

Artifact naming convention:

- `networks/{conservative|aggressive}_{noisy_}state_sac_actor.pt`
- `data_{conservative|aggressive}_{noisy_}state_sac_replay_buffer.pkl`
- `networks/{conservative|aggressive}_{noisy_}vision_vae.pt`
- `networks/{conservative|aggressive}_{noisy_}vision_bc_{resnet|encoder}.pt`
- `networks/{conservative|aggressive}_{noisy_}vision_dagger_{resnet|encoder}.pt`
- `networks/{conservative|aggressive}_{noisy_}vision_ppo_{resnet|encoder}.pt`
- `networks/{conservative|aggressive}_{noisy_}vision_ppo_{resnet|encoder}_critic.pt`
- `networks/{conservative|aggressive}_{noisy_}vision_sac_{resnet|encoder}[_asymmetric]_seed_<seed>.pt`

If args for a script are unclear, run the script with -h to get list of arguments

Run tensorboard: `python3 -m tensorboard.main --logdir runs/ --host 0.0.0.0`

Tensorboard runs are in runs directory

Inspect per-action errors and evaluation failures:

`python3 eval.py --model-type vision --dagger --reward-mode conservative --noisy --inspect-actions`

This loads the matching state SAC teacher for comparison at the learner's states
(override with `--inspection-teacher PATH`). Each episode writes a trace CSV and
PNG to `eval/inspection_<model>_<timestamp>/`, alongside `summary.csv`. Plots
compare executed steering/wheel commands to teacher commands and show velocity
and centerline distance, highlighting the final five seconds. CSVs also record
observed steering and yaw rate. Teacher comparisons are diagnostic only; the
learner controls the entire rollout. Summaries contain per-action MSE and final
five-second MAE, success and termination reason.

New DAgger runs log `action_errors/{all,bc,dagger,recent}/{mse,mae,bias}/`
for steering and each wheel. These are sampled-minibatch diagnostics at the
existing log interval, not held-out evaluation errors. Existing logs cannot
recover this breakdown retroactively.

Test whether the state teacher can recover from failed learner trajectories:

`python3 eval.py --model-type vision --dagger --reward-mode conservative --noisy --teacher-takeover --takeover-seconds 1 2 5`

This enables action inspection and adds `takeover_summary.csv` in the same
inspection directory. For each failed episode it first replays all recorded
learner commands to verify the original failure. It then repeats the recorded
prefix and switches permanently to the deterministic teacher at each requested
lead time. Prefix verification compares pose and stacked state observations
with tolerances `rtol=1e-6`, `atol=1e-5`. Non-reproducing trials are marked invalid.
Lead times longer than the episode are clamped to reset; actual takeover times
are recorded. RNG states are restored after trials to preserve subsequent
learner evaluations. Takeover outcomes never count toward learner success.
Teacher failure shows that this teacher did not recover, not that recovery is
physically impossible. This option adds several rollouts per failed track.

# Run the full sweep

Run all reward presets, clean/noisy state policies, and structured/latent BC + DAgger variants:

`python3 run_all_combinations.py`

The scripts use exact artifact names for each combination and fail if the expected file is missing.

Preview the command matrix without starting training:

`python3 run_all_combinations.py --dry-run`

Useful filters:

`python3 run_all_combinations.py --reward-modes aggressive --policy-variants noisy --actor-types latent --skip-existing`

# Run one seeded SAC/BC/DAgger workflow in two HPC jobs

`run_seed_workflow.py` can be split immediately after SAC. Submit the first
command to train the state policy, then submit the second command to create the
replay buffer and train BC and DAgger. Pass the same seed and policy options to
both jobs. Use an explicit shared `--rb-path` for the second job when its
temporary directory should not hold the replay buffer.

```bash
python3 run_seed_workflow.py --stage sac --seed 1
python3 run_seed_workflow.py --stage rest --seed 1 --rb-path /shared/nips-env/replay_seed_1.pkl
```

`--stage all` (the default) retains the original full workflow behavior.
SAC, BC, and DAgger logs are passed explicitly to `pipeline/runs` by default;
override their base directory with `--log-dir /shared/tensorboard-runs`.

To run the complete reward/noise/seed matrix with Slurm, use the included
launcher. It runs seeds from `0` through the supplied maximum, inclusive, and
uses one `srun` allocation per combination:

```bash
pipeline/run_seed_workflow_sweep.sh 9 sac --time 02:00:00
pipeline/run_seed_workflow_sweep.sh 9 rest --time 04:00:00
```

The launcher starts all combinations concurrently, then waits and returns a
failure status if any allocation fails. It defaults to `/home/s2738360/nips-env`
and the `fs-rl-dir` Singularity sandbox in `$HOME`. Override either location
with `NIPS_ENV_ROOT` or `NIPS_ENV_IMAGE` if your HPC paths differ. `--time` (or
`-t`) controls the Slurm time limit and defaults to `12:00:00`; omitting the
stage runs `all`.

For one workflow invocation, use `run_seed_workflow_seed.sh`. It defaults to
the same Slurm/Singularity setup; pass `--naive` to run Python directly on the
host instead:

```bash
pipeline/run_seed_workflow_seed.sh -- --seed 1 --stage sac
pipeline/run_seed_workflow_seed.sh --naive -- --seed 1 --stage sac
```

# Train state based policy

## Vision RL Slurm sweep

Launch seeds 1–5 of vision PPO, vision SAC, and asymmetric vision SAC from
scratch, using conservative rewards without action-noise augmentation:

```bash
python3 pipeline/run_vision_rl_sweep.py --dry-run
python3 pipeline/run_vision_rl_sweep.py --time 12:00:00
```

Run these commands from the repository root on the cluster. The launcher starts
15 concurrent `srun` requests (one GPU each), waits for them, and returns nonzero
if any run fails. Keep the launcher alive until completion. Defaults are one
million environment steps per run, 10 CPUs and 8000 MB per CPU, and license
`horse`. Other trainer hyperparameters retain their defaults. PPO uses its
existing privileged state critic; plain SAC uses vision critics and asymmetric
SAC uses state critics. No pretrained models are loaded.

Override container/repository/ICD paths with `NIPS_ENV_IMAGE`, `NIPS_ENV_ROOT`,
and `NIPS_ENV_NVIDIA_ICD`, as for the existing sweep. This launcher uses
`singularity exec` directly and requires a container where Python and pacsim
are available without a custom shell entry script. `--python` selects the
container Python executable if needed.

Per-job stdout/stderr go to a timestamped `pipeline/runs/vision_rl_sweep_*`
directory. TensorBoard logs remain under `pipeline/runs`; checkpoints use the
existing method/seed-specific names in `pipeline/networks` and may overwrite
previous checkpoints with the same names. This launches training, not evaluation.
Use `--methods ppo`, `--seeds 3 4 5`, or `--total-timesteps N` to restrict a run.

## State policy commands

Conservative without noise augmentation:

`python3 sac_continous_action.py`

Conservative with noise augmentation:

`python3 sac_continous_action.py --noise-augment`

Aggressive examples:

`python3 sac_continous_action.py --reward-mode aggressive`

`python3 sac_continous_action.py --reward-mode aggressive --noise-augment`

# Fill replay buffer

Trajectories are driven by state based policy

Default (clean model):

`python3 fillRB.py`

Using the noise augmented state policy:

`python3 fillRB.py --noisy`

# Train autoencoder

`python vae.py`

# Train vision based policy using BC

`python3 bc.py`

`python3 bc.py --noisy`

Latent/encoder variant:

`python3 bc.py --latent`

# Train vision based policy using DAgger (using state actor as expert)

Default (clean state expert):

`python3 dagger.py`

Using the regular noise augmented state expert:

`python3 dagger.py --noisy`

# Train vision based policy using PPO

Structured vision actor:

`python3 ppo_vision.py`

Latent/encoder actor:

`python3 ppo_vision.py --vision-actor latent`

State-based actor:

`python3 ppo_vision.py --state-based-actor`

Fine-tune a matching BC or DAgger vision actor with a short, fixed-policy critic
warmup (the actor is loaded from the supplied checkpoint):

```bash
python3 ppo_vision.py \
  --pretrained-actor-path networks/conservative_last_action_vision_dagger_resnet_seed_1.pt \
  --pretrained-actor-log-std -2 \
  --critic-warmup-iterations 5 --critic-warmup-epochs 10 \
  --actor-learning-rate 1e-5 --critic-learning-rate 3e-4
```

BC and DAgger train only the action mean. The default `--pretrained-actor-log-std
-2` therefore replaces their untrained log-standard-deviation head before PPO
starts collecting stochastic rollouts. Set it to `None` only when loading a PPO
checkpoint whose exploration head should be kept.

To pretrain the PPO state critic separately with PPO-matching GAE rollout
targets from a frozen imitation actor, then load it for fine-tuning:

```bash
python3 pretrain_ppo_critic.py \
  --actor-path networks/conservative_noisy_last_action_vision_dagger_resnet_seed_1.pt \
  --noisy --actor-log-std -3 --rollout-iterations 5 --update-epochs 20 --batch-size 512

python3 ppo_vision.py \
  --pretrained-actor-path networks/conservative_noisy_last_action_vision_dagger_resnet_seed_1.pt \
  --pretrained-actor-log-std -3 \
  --pretrained-critic-path networks/conservative_noisy_last_action_vision_ppo_resnet_critic_seed_1.pt
```

The critic pretrainer uses the generated training-track pool and its ordinary
random flips/start quartiles. It uses the same `gamma`, GAE lambda, and
2,048-step rollout target as PPO. Its large batch size applies only to the
cheap state critic; it does not increase vision-model VRAM use.

# Train vision based policy using SAC

Structured vision actor with vision Q critics:

`python3 sac_vision.py`

Use asymmetric SAC with privileged state-only Q critics:

`python3 sac_vision.py --asymmetric-critic`

Initialize those Q critics from a matching seeded state-SAC run:

`python3 sac_vision.py --asymmetric-critic --preload-state-critic --seed 1`

Keep those pretrained critics frozen while the actor catches up for the first
50k post-warm-up environment steps:

`python3 sac_vision.py --asymmetric-critic --preload-state-critic --freeze-critic-steps 50000 --seed 1`

The preload command resolves the corresponding `state_sac_qf1/qf2` files from
the seed, reward mode, noise setting, and last-action setting. Use
`--preload-state-qf1-path` and `--preload-state-qf2-path` to select a different
state-SAC source run explicitly.

Latent/encoder vision actor (requires a pretrained VAE):

`python3 sac_vision.py --vision-actor latent`

Vision SAC stores stacked, full-resolution camera images in replay. Its default
`--vision-buffer-size 5000` needs several GiB of RAM; increase it deliberately
on machines with enough memory. It also uses a vision-specific batch size of
16 (`--vision-batch-size`) to avoid GPU memory spikes from full-resolution
camera batches.

# Train DreamerV3 with PacSim

This repo integrates `NM512/dreamerv3-torch` as a git submodule at
`external/dreamerv3-torch`. The launcher keeps the upstream submodule unchanged
and adapts `pacsimEnv` to the old Gym API expected by that implementation.

Initialize submodules before building the Docker image:

`git submodule update --init --recursive`

Rebuild and start the container:

`docker compose up -d --build`

Run a short state-observation smoke test inside Docker:

`docker exec -it --workdir /root/workspace/pipeline fs-rl python3 dreamer_pacsim.py --configs pacsim_state pacsim_smoke`

Start state-observation training:

`docker exec -it --workdir /root/workspace/pipeline fs-rl python3 dreamer_pacsim.py --configs pacsim_state`

Start camera-observation training:

`docker exec -it --workdir /root/workspace/pipeline fs-rl python3 dreamer_pacsim.py --configs pacsim_vision`

Relative logdirs (including `--logdir` overrides) resolve under `pipeline/`,
and `_seed_<seed>` is appended to the run directory name. For example,
`--logdir runs/dreamer_vision_test_10k --seed 40` saves to
`pipeline/runs/dreamer_vision_test_10k_seed_40/`, which is mounted on the host
when using Docker Compose. Absolute logdirs are used as given, with the same
seed suffix. Existing runs in other locations are not moved automatically.

The default vision config uses all enabled PacSim cameras. The adapter exposes
`cameraLeft`, `cameraFront`, and `cameraRight` as separate native-resolution
`256x306` image observations for Dreamer's CNN. It also keeps a same-resolution
side-by-side `image` panorama for TensorBoard rollout videos. The default vision
logdir is `runs/dreamer_pacsim_vision_fullres`, replay is capped at 10k
transitions, batch size is 2, and the vision model uses a moderately wider
RSSM/CNN than the upstream defaults. It uses camera images together with RPM,
IMU, steering, and previous-action observations; rangefinder and velocity are
excluded from the environment and model inputs. Do not reuse older vision
logdirs with different observation shapes. To run the older front-only setup, add
`--pacsim_camera_key cameraFront`. To downscale, override `--size 128,153` or
`--size 64,64`.

Useful overrides:

- `--device cpu` to run without CUDA.
- `--steps 100000 --eval_every 5000` to change the training budget.
- `--pacsim_reward_mode aggressive` to use the aggressive reward preset.
- `--pacsim_verbose_env True` to show PacSim reset and termination logs.

# Evaluate models

- Clean: `python3 eval.py --model-type state`
- Noise augmented: `python3 eval.py --model-type state --noisy`

Evaluate one policy configuration across several training seeds. This resolves
the usual seed-suffixed checkpoint names and writes one aggregate CSV:

`python3 eval.py --model-type state --noisy --reward-mode conservative --seeds 0 1 2 3 4`

For each track, `success` and `success_std` are the mean and sample standard
deviation of the per-seed completion indicators. `time` and `time_std` are the
mean and sample standard deviation of lap times among successful seeds; blank
time fields mean that no seed completed that track. `n_success` and `n_seeds`
give the absolute successful and attempted policy-track evaluations. A companion
`seed_summary_*.csv` records one global row per seed. It contains success
counts, success rates, and geometric mean lap times for all tracks and for
`FSG24`, `FSE24`, `FSG25`, and `FSCZ25`. Geometric mean times only include that
seed's successful tracks; blank values mean no track in that group was
completed. `eval.py` also records normalized times using the per-track
`TRACK_TIME_REFERENCES` map (currently initialized to `1.0` for every track).
A single `--seed` evaluation remains supported.

To replace the fixed evaluation tracks with a directory of tracks:

```bash
python3 eval.py --model-type vision --dagger --seeds 0 1 2 3 4 --track-dir generated_tracks_eval
```

`--track-dir` accepts a path relative to the working directory or an absolute
path. All directly contained `.yaml`/`.yml` files are evaluated in sorted order,
without augmentation, for every seed. It works for all supported model types
and cannot be combined with `--training-episodes`. Output filenames include
the track directory name. Raw lap times and success rates are always reported;
normalized times are blank where reference times are unavailable.

Dreamer can be evaluated on the same tracks with the same return/CSV reporting:

```bash
python3 eval.py --model-type vision --dreamer --seeds 0 1 2 3 4
python3 eval.py --model-type state --dreamer --seed 0
```

Defaults resolve `runs/dreamer_pacsim_vision_fullres_seed_N/latest.pt` or
`runs/dreamer_pacsim_state_seed_N/latest.pt`. For a custom run, use
`--model-path /path/to/latest.pt` (single checkpoint). The matching pacsim
Dreamer preset supplies architecture and preprocessing settings; pass
`--dreamer-config /path/to/overrides.yaml` with the training settings if they
differ. This is a flat YAML mapping using the same keys as `dreamer_pacsim.py`.
Checkpoint weights are loaded strictly; optimizer states are not restored.
The recurrent state resets for every episode and advances once per step using
the newest observation, without frame stacking inside Dreamer. Action selection
matches upstream `training=False` evaluation (latent inference can still be
stochastic). Custom camera config files and action repetition other than one
are currently unsupported, as are `--stochastic-actions`, `--noisy`, `--latent`,
and `--no-include-last-action` for Dreamer. Install the Dreamer dependencies and
initialize `external/dreamerv3-torch` before running.

To derive reference times from existing aggregate reports, run
`python3 get_reference_time.py`. It prints a ready-to-paste
`TRACK_TIME_REFERENCES` dictionary, choosing the best (lowest) average
successful lap time per track across `eval/test_*.csv` reports.

`createEvalTable.py` uses the matching `seed_summary_*.csv` file to report
seed-level global metrics as mean `±` sample standard deviation: normalized
geometric mean time and success rate for all tracks, plus success rate for the
last four tracks. It also reports a performance score calculated per seed as
`all_success_rate / all_normalized_time_gmean`, then summarizes that score as
mean `±` sample standard deviation. Older reports without a matching seed
summary retain their aggregate values without an uncertainty term.

Use `python3 createEvalTable.py --decimal-places 2` to use a compact, common
precision in both generated tables. To control just the displayed success-rate
percentage precision, use `--success-decimal-places 1` (the default).

To evaluate the full baseline matrix over seeds `0` through `9`, run:

`python3 run_full_baseline_eval.py 9`

This runs state SAC plus structured BC/DAgger for both reward modes and
clean/noisy policies, writing one aggregate evaluation report and one per-seed
summary for each configuration. Preview its commands with
`python3 run_full_baseline_eval.py 9 --dry-run`.

Evaluate a DAgger-trained vision model:

- DAgger (vision): `python3 eval.py --model-type vision --dagger`
- DAgger (vision, noise augmented): `python3 eval.py --model-type vision --dagger --noisy`

Evaluate PPO-trained models:

- PPO (state): `python3 eval.py --model-type state --ppo`
- PPO (vision): `python3 eval.py --model-type vision --ppo`
- PPO (vision, latent): `python3 eval.py --model-type vision --ppo --latent`

To compare returns under the training environment distribution rather than the
fixed test tracks, sample generated training-track episodes with their normal
random flips and start quartiles:

```bash
python3 eval.py --model-type vision --ppo --training-episodes 100 --eval-seed 1
```

This mode evaluates deterministic mean actions by default. Add
`--stochastic-actions` to sample the policy, matching PPO rollout actions.

Evaluate SAC-trained vision models:

- SAC (vision): `python3 eval.py --model-type vision --sac`
- SAC (vision, asymmetric critic): `python3 eval.py --model-type vision --sac --asymmetric-critic`
- SAC (vision, latent): `python3 eval.py --model-type vision --sac --latent`

Evaluation report file is create in eval directory

# Create video

`python3 actorInference.py`
