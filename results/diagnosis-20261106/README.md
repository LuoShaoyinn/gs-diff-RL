# Differentiable training diagnosis

## Confirmed Go2 controller bug and fix

The PD feedback used the wrong joints. The URDF stores joints in **FL, FR, RL,
RR** order, but actions and `control_dofs_force` use **FR, FL, RR, RL**. The old
code subtracted `qpos[:, 7:]` and `dofs_vel[:, 6:]` from named action targets,
then applied those forces to a different leg. In particular, front-right
actions used front-left feedback. Symmetric initial positions concealed this
mistake; later asymmetric motion produces incorrect feedback.

All four Go2 training/evaluation scripts now obtain both position and velocity
indices from the same named joints used for force commands. The observation
layout is preserved. New checkpoints record `controller_version=named_pd_v1`;
the direct trainer rejects resuming a legacy checkpoint. Evaluators expose
`--legacy-pd` for reproducing the old controller explicitly.

An actual two-second replay on public Genesis 1.4.5, A843-2x4090 physical GPU 1,
changed only the feedback indexing within each policy comparison:

| Policy | Old feedback: final height | Correct feedback: final height |
| --- | ---: | ---: |
| Zero output | 0.2024 m | 0.2979 m |
| Historical trained policy | 0.0644 m | 0.2021 m |

The runtime mapping and full trace are in
[public-go2-joint-order](../public-go2-joint-order/). The historical policy still
does not walk after this fix. A fresh ten-update differentiable training run
with corrected feedback completed on the public CUDA package: cost
0.38532 to 0.26769, finite gradient norms 0.53–0.55. This validates execution of
the corrected control/gradient path, not a locomotion solution.

## What the existing evidence establishes

The existing Go2 code does backpropagate through the simulator. Its failures do
not establish that differentiable policy learning cannot work, or that all
Genesis gradients are broken. Several distinct problems were combined in the
same experiment.

1. **The direct optimizer learns a short transient.** `train_go2.py` resets to
   exactly the same pose every update, uses fixed commands, and optimizes only
   32 x 20 ms = 0.64 s. It has neither continuing training states nor a terminal
   value function. The recorded cost decreases from 0.38520 to 0.10139, but the
   subsequent two-second rollout asks for a behavior never optimized. The time
   observation is also clamped to a value outside the training sequence.
2. **Evaluation changes the simulator path.** Training sets
   `requires_grad=True`; both Go2 evaluators use the forward-only rigid solver.
   Those modes are not guaranteed to produce identical dynamics. A failure in
   one mode cannot, by itself, diagnose the other mode's gradients.
3. **There was a separate numerical failure on the old fork/ROCm stack.** The
   tracked `results/nan-diagnosis/trained.log` records errno 96 at zero-based
   control step 37, after finite observations/actions/forces. The ignored
   `runs/nan-investigation/` artifacts further associate this with the forward
   incremental Newton Hessian path: rebuilding the Hessian completed that replay,
   but the robot still fell. This is historical evidence from that revision and
   GPU, not a finding about the public release used for the new experiment.
4. **Contact events make the pathwise objective difficult.** The saved
   `runs/gradient-check/reward-only.log` compares one output-bias direction at
   eight steps: AD 0.371648 versus central difference 0.370502 at epsilon 0.001.
   At nine steps, the corresponding AD is 0.503109 versus FD 50.28367 at epsilon
   0.0001. Contact snapshots differ, both final alive masks agree, and this case
   uses no critic contribution. This demonstrates sensitivity around a contact
   event; it does not distinguish an incorrect derivative from a genuinely
   nonsmooth branch change. Finite gradients alone are insufficient evidence.
5. **The actor-critic experiment is a substantial algorithm change.**
   `train_shac_go2.py` resets only at segment boundaries, evaluates termination
   only at the endpoint, reconstructs physics from qpos/qvel, and uses simple
   segment returns rather than TD-lambda. Its observation function accepts but
   omits the previous action even though the reward depends on it. The local
   Open-DiffLoco reference also uses observation normalization and action noise.
   The existing run contains only 25,600 environment steps, versus the local
   reference configuration's eight million. These are implementation and budget
   differences, not a controlled ablation identifying a single cause.
6. **Some logged metrics hide failures.** In the actor-critic script, height and
   forward velocity are recorded after resetting failed environments. A reset
   contributes height 0.42 m and velocity zero to those averages. The revised
   100-update log has 635 environment resets and a maximum gradient norm of
   1,134.43 before clipping to 1. Its five-second evaluation falls and stops.

The ignored investigation artifacts were inspected in this workspace; they are
not part of the tracked historical results. Apart from the explicit feedback
replay and corrected training above, the historical failures were not rerun as
matched GPU or Genesis-version comparisons.

## First reproduction

The new [InvertedPendulum result](../public-inverted-pendulum/README.md) reaches
1,000/1,000 in all 100 unmodified Gymnasium/MuJoCo episodes and all 64 Genesis
evaluation episodes. The zero-output baseline averages 24.1. This uses complete
gradients through 64- then 128-step rollouts, with a fresh zero-output policy at
the start of the curriculum. It does not establish full-episode gradients.

`train_inverted_pendulum.py` uses public PyPI Genesis 1.4.5, the MJCF distributed
with Gymnasium's InvertedPendulum-v5, and a trainable affine state-feedback
policy bounded with tanh. The policy starts at zero output. All policy updates
come from differentiating simulated states; the objective has no direct action
penalty that could produce a gradient without going through physics.

The smooth training cost is

```
mean(x^2 + 10 * theta^2 + 0.1 * (xdot^2 + thetadot^2))
```

The benchmark's survival reward is locally constant and its termination is
discontinuous, so directly differentiating that reward supplies no useful
balancing gradient. Training therefore uses this surrogate; evaluation uses the
unmodified Gymnasium reward and termination, capped at 1,000 steps. This is a
reproduction of a standard RL **task** using differentiable policy optimization,
not a reproduction of SHAC's published scores or its CartPole swing-up task.

Within every training rollout, the entire policy/state/physics chain remains
attached. There is no PPO, critic, learned dynamics model, imitation data, or
handwritten stabilizing controller. Genesis uses approximate implicit dynamics;
the official MuJoCo benchmark uses RK4. Evaluation in both simulators makes this
difference explicit. Fresh uniform resets are used from the first update and
are recorded in `first_reset.json`; current runs also save the observed state
after reset in `first_reset_observed.json`.

The initial pendulum experiments exposed another concrete model mismatch:
Genesis adds `default_armature=0.1` to zero-armature joints by default, including
the passive pole. The benchmark XML specifies zero. The trainer now explicitly
sets `default_armature=0.0` and records the actual loaded armature, damping, and
force limits. This uses the public API without changing Genesis source.
Earlier `seed42-h64`, `seed42-h64-continued`, `seed42-h256`, and `seed42-lbfgs`
results retain the original added inertia and must not be described as an exact
benchmark model match.

A fresh public-CUDA continuation also failed: jumping from the 64-step policy
to 256-step L-BFGS training at learning rate 1.0 raised `Nan grad in qpos or
dofs_vel found at step 25` during a strong-Wolfe line-search trial. The full
[traceback and configuration](../public-inverted-pendulum/failed-h256/) are
retained. That run did not save the failed candidate's weights or full state
trace, so the precise numerical cause is not established. It is a backward
failure on the public CUDA release, distinct from the historical forward
constraint-force failure on the fork/ROCm stack. Later trainer revisions save
failed candidate weights and reset seeds and still stop on the error.

After the task was solved, a separate 1,000-step training attempt failed on its
**first backward pass**, before any policy update or line-search trial. The
forward state cost was finite (`3.29051e-5`); Genesis reported a NaN gradient at
reverse step 818. The [saved failing candidate](../public-inverted-pendulum/failed-h1000/failed_rollout.json)
is exactly the successful policy, with reset seed 49. A finite forward rollout
and successful evaluation therefore do not guarantee a usable long-rollout
gradient on this public CUDA configuration. The subsequent
[checkpoint-replay investigation](../nan-backward-20261106/README.md) isolates
the cause: roundoff-scale discrepancies grow during the 1,000-step open-loop
reconstruction used for backward, so its simulated states differ substantially
from the original feedback trajectory. The same full backward calculation
completes with finite gradients using 25-step checkpoint intervals. This uses
the public `substeps_local` option without a simulator patch, replacement
gradient, or detach at checkpoint boundaries.

## Relevant papers

- [Xu et al., SHAC, ICLR 2022](https://arxiv.org/abs/2204.07137): a terminal
  critic and short differentiable windows address difficult long-horizon
  optimization. Resetting every short window to the same standing pose does not
  reproduce this algorithm.
- [Georgiev et al., AHAC, ICML 2024](https://research.nvidia.com/publication/2024-07_adaptive-horizon-actor-critic-policy-learning-contact-rich-differentiable):
  studies problems with first-order gradients around stiff contact dynamics and
  adapts the differentiation horizon.
- [Opat, Open-DiffLoco](https://diffloco.martin-opat.com/): a relevant quadruped
  reference using SHAC/MJX. Porting its reward alone does not reproduce its
  simulator, algorithm, reset semantics, or training budget.
- [Gymnasium InvertedPendulum-v5 specification](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/):
  the independent task definition used for the new evaluation.
