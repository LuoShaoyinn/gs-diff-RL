# Genesis differentiable RL experiments

A short-horizon Go2 state-feedback policy trained by backpropagating the reward through Genesis rigid-body simulation and contacts. This is direct pathwise policy optimization, not PPO or a learned dynamics model. It is an initial experiment, not a claim of stable walking.

The Genesis fork is pinned because the rigid-body differentiable path used here is newer than the released 1.4.4 package. ROCm 7.2.4 follows the existing srt2 configuration. On the RX 9070 XT host:

```sh
uv sync --extra rocm
uv run --no-sync --extra rocm python train_go2.py --iterations 100 --envs 8 --horizon 32
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
uv run --no-sync --extra rocm python evaluate_go2.py runs/go2/latest.pt --steps 100
```

Evaluation uses Genesis's forward-only rigid solver, which can differ from the differentiable solver used in training. The two environments use the same initial pose and command, and the policy's normalized time observation is held at 1 after the 32-step training horizon. See `results/go2-first/README.md` for the observed results and numerical failures.
