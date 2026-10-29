# Go2 NaN diagnosis

On A843 RX 9070 XT, the final `go2-1khz-50hz` checkpoint was replayed in Genesis fork `2f5b530582d06241947f640fc2bb2e842ddbd8c4`, with the same forward-only solver, 20 ms policy interval, 20 substeps, and 0.4 m/s command. `scripts/diagnose_go2_nan.py` records the complete pre-step state, observation, raw policy output, bounded action and torque and reads each environment's solver error flags immediately after every policy step.

- Paired replay: the first error appeared after zero-based step 37, corresponding to the interval from 0.74 to 0.76 s. Errno was `[0, 96]`: the trained environment failed and baseline did not.
- Separate trained replay: the same step and flag (`[96]`) reproduced. All observations, raw outputs, actions and torques were finite through the input to that step. At that input, max absolute raw output was 0.885, bounded action 0.177 rad, torque 26.44 Nm, and velocity 10.89 rad/s.
- Separate zero-output replay: completed 100 steps (2 s) without solver errno or non-finite input. This is not proof of good locomotion or numerical stability in other situations.

In this Genesis revision, errno 96 combines `INVALID_FORCE_NAN` (32), set when the constraint solve produces a NaN `qacc`, and `INVALID_ACC_NAN` (64), set when the next integrated position or velocity contains NaN. Genesis keeps the last valid state for the affected environment when it detects the latter, so a finite post-step state does not rule out an internal solver NaN. The evidence locates the first detected failure inside the physics step after finite control input; it does not identify the precise contact, solver operation, or numerical cause, nor exclude the policy producing a difficult physical configuration. It also does not validate gradients or simulator stability generally.

The three log files contain full input/output snapshots. To repeat, run the diagnostic using the pinned Genesis environment and `--mode paired`, `--mode trained`, or `--mode baseline` with the checkpoint path. The flag polling forces GPU synchronization and can affect performance.
