# Differentiable policy learning in Genesis

The current environment uses **public PyPI `genesis-world==1.4.5`**, with no
Genesis fork or simulator patches. CUDA and ROCm are mutually exclusive extras;
`uv.lock` records the resolved package versions.

The Go2 scripts contained a confirmed joint-order bug: PD feedback read
FL/FR/RL/RR while forces were sent in FR/FL/RR/RL order. Feedback now uses the
same named joints as the force commands. See the
[runtime replay and corrected training](results/public-go2-joint-order/README.md).
Old Go2 checkpoints need retraining; `--legacy-pd` on an evaluator explicitly
reproduces their original controller.

## InvertedPendulum-v5

The policy trained with 64- then 128-step physics gradients achieves
**1,000/1,000 return in all 100 unmodified Gymnasium/MuJoCo episodes**, and all
64 held-out Genesis episodes. The zero-output baseline averages 24.1/1,000.
See [the recorded experiments](results/public-inverted-pendulum/README.md).

On **A843-2x4090, physical GPU 1**, run from `~/diskstation/gs-diff-RL`:

```sh
PHYSICAL_GPU=1 bash scripts/reproduce_inverted_pendulum.sh runs/my-reproduction
```

The recipe starts from zero output, uses L-BFGS with full simulator gradients,
and increases the horizon from 64 to 128 steps. Each stage uses a fresh optimizer
with the preceding learned weights. L-BFGS performs multiple gradient rollouts
per outer step; logs count every line-search rollout in `environment_steps`.

The wrapper runs `uv sync --extra cuda` on the server and places the
environment, uv downloads, Python installation, temporary files, and compiler
caches on the NAS. Transfer source only, **never `.venv`**:

```sh
rsync -a --exclude='.venv/' --exclude='.git/' --exclude='runs/' \
  --exclude='.runtime/' --exclude='__pycache__/' ./ ~/diskstation/gs-diff-RL/
```

Physical GPU 1 appears as `cuda:0` inside the isolated process. Choose a fresh
output prefix for each run. The ROCm extra and `--backend amdgpu` remain
available, but the new reproduction was validated on CUDA.

The actor is trainable affine feedback followed by tanh, initialized to zero
output on a fresh run. It learns from randomized resets and a smooth state-only balancing cost.
Every policy gradient goes through Genesis; there is no critic, PPO, learned
dynamics, imitation data, or detach within a rollout. The horizons are 2.56 and
5.12 seconds; evaluation lasts 40 seconds. A separate attempt to differentiate
all 1,000 steps failed with a NaN gradient on the balanced policy's first
backward pass. This reproduction does **not** claim full-episode gradients.
Genesis's default added rotor inertia is explicitly set
to zero to preserve the benchmark XML's armature.

The script evaluates the final checkpoint for 1,000 steps in the same Genesis
solver mode and separately on unmodified Gymnasium/MuJoCo InvertedPendulum-v5,
using its original survival reward and termination. Genesis and MuJoCo use
different integrators. Results include a zero-output baseline, fresh evaluation
resets, per-episode scores, a checkpoint, exact versions, observed first training
resets, loaded model parameters, training metrics, and a rollout trace. A resumed run's baseline is its starting
checkpoint, rather than zero output. Training cost is a surrogate for the
benchmark reward; this is a task reproduction, not a replication of SHAC scores.

Evaluate a saved policy again on a separate set of benchmark resets:

```sh
uv run --no-sync --extra cuda python evaluate_inverted_pendulum.py \
  runs/my-reproduction-h128/latest.pt --seed 20000 --episodes 100 \
  --out runs/inverted-pendulum-evaluation
```

Add `--video` with `MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1` for a headless replay on
physical GPU 1 (up to 40 seconds, terminating on failure). The evaluation
policy and MuJoCo physics run on CPU; training runs on the selected GPU.

See [the diagnosis](results/diagnosis-20261106/README.md) for the existing Go2
failures and relevant SHAC, AHAC, and Open-DiffLoco papers. Validation uses actual
training and evaluation experiments.

## Historical Go2 experiments

A short-horizon Go2 state-feedback policy trained by backpropagating the reward through Genesis rigid-body simulation and contacts. This is direct pathwise policy optimization, not PPO or a learned dynamics model. It is an initial experiment, not a claim of stable walking.

These historical artifacts used Genesis fork `2f5b5305` because its rigid-body differentiable path was newer than the then-released 1.4.4 package. The current public environment supersedes that dependency. The commands below run the corrected controller; the historical results retain their original controller and solver provenance:

```sh
uv sync --extra rocm
uv run --no-sync --extra rocm python train_go2.py --backend amdgpu --iterations 100 --envs 8 --horizon 32
```

The default policy interval is 20 ms (50 Hz), with 20 physics substeps of 1 ms (1000 Hz). The policy produces a torque held for those substeps. The 32-step horizon is 0.64 s.

The run writes a JSONL training log and latest policy/optimizer checkpoint under `runs/go2`. Each iteration starts from the same standing pose and uses a fixed batch of forward speed commands, so the training metrics do not establish generalization or long-horizon walking.

On the A843 RX 9070 XT, noninteractive shells need `QD_ENABLE_HIP_MEMPOOL=0`. Its ROCm 7.2.4 linker requires the unavailable `libxml2.so.2`; the experiment used the system linker without changing the installation:

```sh
mkdir -p /tmp/gs-diff-rocm/llvm/bin
ln -sf /usr/bin/ld.lld /tmp/gs-diff-rocm/llvm/bin/ld.lld
export ROCM_PATH=/tmp/gs-diff-rocm QD_ENABLE_HIP_MEMPOOL=0
```

The first run used the pinned Genesis checkout with the existing srt2 ROCm virtual environment because fresh `uv sync` did not complete promptly. The host's `xsb_github` SSH identity was used for GitHub access.

Compare a trained checkpoint with the initial zero-output controller using a two-second forward-only rollout. Timing is read from the checkpoint (older checkpoints fall back to 10 ms and two substeps):

```sh
uv run --no-sync --extra rocm python evaluate_go2.py runs/go2/latest.pt --backend amdgpu --steps 100
```

Evaluation uses Genesis's forward-only rigid solver, which can differ from the differentiable solver used in training. The two environments use the same initial pose and command, and the policy's normalized time observation is held at 1 after the 32-step training horizon. See `results/go2-1khz-50hz/README.md` for the 1 kHz results and `results/go2-first/README.md` for the earlier run.

## Short-horizon actor-critic experiment

`train_shac_go2.py` adapts Open-DiffLoco's smooth reference-free reward components to Genesis with 50 Hz control and 20 physics substeps. A frozen target critic bootstraps a 16-step pathwise actor objective; the critic fits discounted segment returns. States are carried between segments as detached generalized positions and velocities, and low or tilted robots reset at segment boundaries. This is a small independent SHAC-inspired experiment, not a reproduction of Open-DiffLoco: it uses Genesis, a smaller batch, a simplified command range, no action noise, and no lambda-return critic target. It also reconstructs simulator state at each boundary rather than retaining internal contact/warmstart state. The original Genesis solver is unchanged.

```sh
uv run --no-sync --extra rocm python train_shac_go2.py --backend amdgpu --iterations 100 --envs 16 --horizon 16
uv run --no-sync --extra rocm python evaluate_shac_go2.py runs/go2-shac/latest.pt --backend amdgpu --steps 250
```

The evaluation uses the forward-only solver, compares the trained controller with a zero-output controller at 0.4 m/s, and logs their base position, velocity, and height. Training checkpoints include optimizer and rollout state but this script does not yet resume them.
