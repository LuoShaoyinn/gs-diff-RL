# Go2 named-joint feedback correction

Public Genesis 1.4.5, Torch 2.14.1+cu130, A843-2x4090 physical GPU 1.
These are actual simulation/training experiments, not local tests.

The runtime order in `mapping.json` is FL, FR, RL, RR, while the force-command
order is FR, FL, RR, RL. `trace.jsonl` records all 100 steps at 20 ms, with
20 physics substeps. Each policy comparison uses identical resets, observations,
and simulator mode; only PD feedback indexing changes.

```sh
PHYSICAL_GPU=1 bash scripts/run_cuda.sh scripts/diagnose_go2_joint_order.py \
  results/go2-1khz-50hz/latest.pt --steps 100 --out runs/go2-joint-order-public
PHYSICAL_GPU=1 bash scripts/run_cuda.sh train_go2.py \
  --iterations 10 --envs 8 --horizon 32 --out runs/go2-named-pd-public
```

The four trace columns are `zero_legacy_pd`, `zero_aligned_pd`,
`trained_legacy_pd`, `trained_aligned_pd`. At two seconds their heights are
0.20237, 0.29786, 0.06439, and 0.20211 m. The torque-difference field is a
difference between *requested* torques before simulator force limiting.

`training/` contains the new corrected controller's ten-update log and final
checkpoint. Cost decreases from 0.38532 to 0.26769, with finite gradients. These
short runs demonstrate the wiring error and execution of its fix. They do not
establish stable walking, or isolate the historical fork/ROCm solver failure.
