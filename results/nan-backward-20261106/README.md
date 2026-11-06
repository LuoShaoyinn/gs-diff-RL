# Backward NaN diagnosis

**Cause identified: the long backward checkpoint reconstruction diverges from
the original feedback trajectory.** Shortening the replay interval to 25
control steps makes the previously failing full 1,000-step backward pass
complete with finite gradients, using public Genesis and no detach.

The reproduction and previous results were committed as `4a32c70` before this
investigation. These are diagnostic replays of the failed training calculation,
not additional benchmark evaluations. All simulator runs use public PyPI
Genesis 1.4.5 on A843-2x4090, physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`, internal
`cuda:0`). No simulator source or custom operator was installed or modified.

## Checkpoint replay is the source of the amplification

Genesis 1.4.5's `Simulator.load_ckpt()` restores the beginning of an interval
and reconstructs it by calling `step(in_backward=True)` with the taped control
inputs. This is an open-loop replay: it does not rerun the feedback policy on
the reconstructed observations. A balanced feedback trajectory therefore
does not imply that this reconstruction is numerically stable.

Comparing the original and reconstructed caches directly:

| Quantity | One 1,000-step replay interval | Forty 25-step replay intervals |
| --- | ---: | ---: |
| Forward state cost | 3.2905103580560535e-5 | 3.2905103580560535e-5 |
| Largest qpos reconstruction error | 1.5894553 | 4.656613e-10 |
| Largest velocity reconstruction error | 5.5305394 | 1.862645e-9 |
| Backward | NaN at reverse step 818 | Completes, all parameter gradients finite |

The restored initial state matches exactly. After the first physics substep,
the maximum qpos difference is only `4.656612873e-10`, with velocity difference
`1.862645149e-9`. During the long replay these float32-scale discrepancies grow:
the reconstructed pole reaches `1.59112215` radians, whereas the original
never exceeds `0.00990091` radians. The reconstructed angular velocity reaches
`5.53054` rad/s and angular acceleration `593.53845` rad/s². Backward thus
linearizes around a substantially different trajectory; its exploding
gradients do not describe the original balanced rollout.

The 25-step case processes all 40 checkpoints in one `loss.backward()` and
takes 18.77 seconds for the measured forward/backward computation. Its policy
gradients are:

```text
dL/dW = [-2.367087e-7, 1.628868e-6, 6.562585e-7, -6.518917e-7]
dL/db = -0.0011094274
```

This changes checkpoint replay length, **not** the differentiation horizon.
There is no internal state detach, truncated gradient, replacement derivative,
optimizer update, or additional benchmark evaluation in this experiment.
The exact operation producing the first rounding discrepancy was not isolated.

## Use the public checkpoint option

The trainer now accepts `--checkpoint-steps`. For a 1,000-step rollout use
`--horizon 1000 --checkpoint-steps 25`. This sets the public
`SimOptions(substeps_local=4*25)` while keeping all 1,000 policy steps in the
autograd graph. The interval must divide the horizon because this Genesis
version's input-tape indexing fails for incomplete final intervals. Omitting
the option preserves the earlier training recipe's checkpoint configuration.

To reproduce the two diagnostic backward calculations on the designated GPU,
from the NAS checkout, choose a fresh output directory:

```sh
PHYSICAL_GPU=1 bash scripts/run_cuda.sh scripts/diagnose_inverted_pendulum_replay.py \
  --out runs/my-checkpoint-replay-diagnosis
```

`checkpoint-replay/` retains the original/reconstructed arrays, per-checkpoint
errors, and both outcomes. The failed 256-step line-search candidate from the
earlier experiment was not preserved, so this diagnosis does not establish
the cause of that separate failure.

## Reproduced failure

Replay the final `matched-h128/latest.pt` policy, reset seed 49, 64 environments,
1,000 control steps, four physics substeps per control step, float32. The
checkpoint SHA-256 is
`72217f7ac25369b38972bf76e69f49c627e4e36b3bec5ed79ccd1839b712a588`.
No optimizer step is involved.

The complete forward rollout is finite:

- Mean state cost: `3.2905103580560535e-05`.
- Maximum absolute state components `[cart, angle, cart velocity, angular velocity]`:
  `[0.0179101340, 0.00990090985, 0.0724756643, 0.126131371]`.
- Maximum absolute action: `0.195691526`, within the action bound of 3.

The first observed nonfinite value is **infinity in a qpos adjoint**, after
reversing physics substep 3275 (control step 818). Another backward substep
produces NaNs. Genesis subsequently raises
`Nan grad in qpos or dofs_vel found at step 818`.
Its validity check tests `isnan`, so the preceding infinity is not reported by
that exception.

The preceding qpos adjoint maxima grow by approximately 1.798 per control step:

| Reversed physics substep | Maximum finite absolute qpos adjoint |
| --- | ---: |
| 3960 | 1.2741e-6 |
| 3880 | 0.157676 |
| 3800 | 1.9815e4 |
| 3720 | 2.4902e9 |
| 3640 | 3.1295e14 |
| 3560 | 3.9330e19 |
| 3480 | 4.9426e24 |
| 3400 | 6.2115e29 |
| 3320 | 7.8061e34 |
| 3280 | 2.7673e37 |

The immediate numerical cause is backward overflow, rather than a nonfinite
forward trajectory or an optimizer update. Clipping policy gradients after
`backward()` cannot prevent an exception raised inside that backward pass.
These observations alone do not establish whether the amplified derivatives
are correct.

## Constraint bypass

An otherwise identical replay uses the public option
`RigidOptions(disable_constraint=True)`. The complete saved observation and
action arrays are byte-identical (`forward_array_equality.json`), and it has
zero active constraint rows.
The backward still overflows, this time at control step 924. Disabling the
constraint solver therefore does not fix the failure. The two modes take
different upstream backward implementations for the mass solve.

## Derivative comparison

One-step state Jacobians were measured at 16 states sampled from the original
rollout, including its settled tail. They were compared with central forward
perturbations of `1e-4` and `1e-5` in each state component. At step 999,
environment 0, the largest eigenvalue magnitude is 0.9352008 from autodiff and
0.9352078 from the `1e-4` perturbation. The largest entrywise discrepancy there
is 0.0002023. Smaller perturbations have greater float32 subtraction error.

Repeating the derivative measurement across several consecutive feedback
steps from the settled state also gives close agreement:

| Control steps | Largest absolute autodiff entry | Largest entrywise discrepancy from forward perturbations |
| --- | ---: | ---: |
| 1 | 10.38669 | 0.000573 |
| 2 | 9.79058 | 0.000690 |
| 4 | 4.55971 | 0.001710 |
| 8 | 1.04196 | 0.001579 |
| 16 | 0.98119 | 0.001974 |

This evidence does not support interpreting the full-rollout overflow as the
true local instability of the balanced controller. The recorded summaries
are in `local-sensitivity-h1/` and `chain-sensitivity/`.

## Artifacts

`fp32-trace/` and `no-constraints/` contain the configuration, summary, first
nonfinite observation, per-substep adjoint samples, and Torch gradient samples.
Instrumentation reads gradient fields and registers read-only Torch hooks; it
does not clip, replace, or detach the training gradient. The original full
forward trajectories are retained on the shared NAS under
`~/diskstation/gs-diff-RL/runs/nan-backward/`.

All 36 installed Python source files under `genesis/engine/solvers/rigid/` and
`genesis/grad/` match their distribution `RECORD` hashes
(`distribution_record.json`).
