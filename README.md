# Genesis differentiable RL experiments

A short-horizon Go2 state-feedback policy trained by backpropagating the reward through Genesis rigid-body simulation and contacts. This is direct pathwise policy optimization, not PPO or a learned dynamics model. It is an initial experiment, not a claim of stable walking.

The Genesis fork is pinned because the rigid-body differentiable path used here is newer than the released 1.4.4 package. ROCm 7.2.4 follows the existing srt2 configuration. On the RX 9070 XT host:

```sh
uv sync --extra rocm
uv run --no-sync --extra rocm python train_go2.py --iterations 100 --envs 8 --horizon 32
```

The run writes a JSONL training log and latest policy/optimizer checkpoint under `runs/go2`. Each iteration starts from the same standing pose and uses a fixed batch of forward speed commands, so the training metrics do not establish generalization or long-horizon walking.
