# 1 kHz physics / 50 Hz policy experiment

On A843 RX 9070 XT, Genesis fork `2f5b530582d06241947f640fc2bb2e842ddbd8c4`, Quadrants 1.3.3 and PyTorch `2.11.0+rocm7.2.4.git5fbd98f3` ran 100 updates with 8 environments, 32 policy steps per rollout, seed 42 and fixed speed commands from 0.2 to 0.6 m/s. Genesis `dt=0.02` with `substeps=20` gives 1 ms physics substeps; policy and torque commands are updated every 20 ms and held through those substeps. Each training rollout covers 0.64 s.

The training objective decreased from 0.38520 at update 0 to 0.10139 at update 99; all logged gradients were finite. Mean endpoint forward velocity rose from -0.144 to 0.443 m/s, and endpoint height was 0.298 m at update 99. These are training-batch statistics, not a held-out evaluation. The final checkpoint is saved after its corresponding optimizer update, whereas the metrics describe the preceding rollout. The run took about 1081 seconds, including initial compilation.

The two-second evaluation used the checkpoint timing, a forward-only rigid solver, identical initial poses, and a 0.4 m/s command for the trained and zero-output controllers. At 0.4 s their x positions were 0.074 m and -0.035 m, and heights were 0.272 m and 0.240 m respectively. Before the 0.8 s report the solver raised `Invalid constraint forces causing 'nan'`; the batched error does not identify the failing environment. This does not establish sustained walking, and forward-only solver behavior can differ from the differentiable solver. See `evaluation.log` for the full output.

`metrics.jsonl` contains the training log and `latest.pt` contains the policy and optimizer state. The host used the existing srt2 virtual environment and the ROCm linker workaround in the project README.
