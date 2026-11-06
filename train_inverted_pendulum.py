"""Learn InvertedPendulum feedback by backpropagating through public Genesis.

There is no critic, score-function estimator, demonstration, or state detach
inside a rollout. The objective depends ONLY on simulated states, so every
policy gradient must pass through the simulator. The benchmark survival reward
is used for evaluation; a smooth state cost supplies the training gradient.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import genesis as gs
import gymnasium as gym
import gymnasium.envs.mujoco
import numpy as np
import torch
from torch import nn


class Policy(nn.Module):
    """Trainable affine feedback, smoothly bounded to the benchmark action range."""

    def __init__(self):
        super().__init__()
        self.feedback = nn.Linear(4, 1)
        nn.init.zeros_(self.feedback.weight)
        nn.init.zeros_(self.feedback.bias)

    def forward(self, obs):
        return 3.0 * torch.tanh(self.feedback(obs))


def model_path():
    return Path(gymnasium.envs.mujoco.__file__).parent / "assets" / "inverted_pendulum.xml"


def random_resets(seed, count, noise=0.01):
    # Matches the benchmark's independent uniform qpos/qvel reset distribution.
    return np.random.default_rng(seed).uniform(-noise, noise, (count, 4)).astype(np.float32)


class GenesisPendulum:
    def __init__(self, envs, horizon, substeps=4, requires_grad=True, default_armature=0.0):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.04, substeps=substeps,
                substeps_local=substeps * horizon if requires_grad else 1,
                requires_grad=requires_grad,
            ),
            rigid_options=gs.options.RigidOptions(enable_collision=False),
            show_viewer=False,
        )
        # Genesis otherwise adds 0.1 rotor inertia to zero-armature joints,
        # including this benchmark's passive pole. Keep the authored zero here.
        self.robot = self.scene.add_entity(gs.morphs.MJCF(
            file=str(model_path()), default_armature=default_armature))
        self.scene.build(n_envs=envs)
        self.slider = self.robot.get_joint("slider").dof_start
        if self.robot.n_qs != 2 or self.robot.n_dofs != 2:
            raise RuntimeError("Expected the benchmark's two generalized coordinates")

    def reset(self, states):
        self.scene.reset()
        states = torch.as_tensor(states, device=gs.device)
        self.robot.set_qpos(states[:, :2], zero_velocity=True)
        self.robot.set_dofs_velocity(states[:, 2:])
        return self.observation()

    def observation(self):
        state = self.scene.rigid_solver.get_state()
        return torch.cat((state.qpos, state.dofs_vel), dim=-1)

    def step(self, action):
        # The benchmark MJCF motor has gear=100; force control takes joint force,
        # whereas Gymnasium takes the pre-transmission motor control in [-3,3].
        self.robot.control_dofs_force(100.0 * action, [self.slider])
        self.scene.step()
        return self.observation()


def state_cost(obs):
    return obs[:, 0].square() + 10 * obs[:, 1].square() + 0.1 * obs[:, 2:].square().sum(-1)


@torch.no_grad()
def evaluate_genesis(env, policy, states, steps):
    obs = env.reset(states)
    alive = torch.ones(len(states), dtype=torch.bool, device=gs.device)
    scores = torch.zeros(len(states), dtype=torch.long, device=gs.device)
    max_angle = torch.zeros(len(states), device=gs.device)
    for _ in range(steps):
        obs = env.step(policy(obs))
        alive &= torch.isfinite(obs).all(-1) & (obs[:, 1].abs() <= 0.2)
        scores += alive.long()
        max_angle = torch.maximum(max_angle, obs[:, 1].abs())
    return {
        "episodes": len(states), "steps": steps,
        "mean_return": scores.float().mean().item(),
        "min_return": scores.min().item(),
        "successes": (scores == steps).sum().item(),
        "returns": scores.cpu().tolist(),
        "max_abs_angle_rad": max_angle.max().item(),
    }


@torch.no_grad()
def evaluate_gym(policy, seed, episodes, trace_path=None):
    """Unmodified benchmark, independent resets, original reward and termination."""
    cpu_policy = Policy().cpu()
    cpu_policy.load_state_dict({k: v.cpu() for k, v in policy.state_dict().items()})
    env = gym.make("InvertedPendulum-v5")
    records = []
    trace = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        initial = obs.tolist()
        score = 0.0
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action = cpu_policy(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            steps += 1
            if episode == 0 and trace_path:
                trace.append([steps, *obs.tolist(), float(action[0]), float(reward)])
        records.append({"seed": seed + episode, "initial_state": initial,
                        "return": score, "steps": steps,
                        "terminated": terminated, "truncated": truncated})
    env.close()
    if trace_path:
        np.savetxt(trace_path, trace, delimiter=",", comments="",
                   header="step,cart_m,angle_rad,cart_velocity_m_s,angular_velocity_rad_s,action,reward")
    return {
        "environment": "InvertedPendulum-v5", "episodes": episodes,
        "mean_return": float(np.mean([r["return"] for r in records])),
        "min_return": min(r["return"] for r in records),
        "successes": sum(r["truncated"] and not r["terminated"] for r in records),
        "records": records,
    }


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--default-armature", type=float, default=0.0,
                        help="Keep 0 for the benchmark; 0.1 reproduces the initial model mismatch")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    parser.add_argument("--lbfgs-max-iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--validation-steps", type=int, default=250)
    parser.add_argument("--backend", choices=["cuda", "amdgpu"], default="cuda")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", help="Continue policy/optimizer; horizon may be increased explicitly")
    initialization.add_argument("--initial-policy", help="Start from saved weights with a fresh optimizer")
    parser.add_argument("--out", default="runs/inverted-pendulum")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.jsonl").exists():
        raise FileExistsError(f"Choose a fresh run directory: {out}")
    torch.set_num_threads(1)
    gs.init(backend=getattr(gs, args.backend), precision="32", seed=args.seed, logging_level="warning")
    torch.manual_seed(args.seed)
    config = {
        "args": vars(args), "host": platform.node(), "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(gs.device), "gpu": torch.cuda.get_device_name(0),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("genesis-world", "quadrants", "torch", "gymnasium", "mujoco")},
        "genesis_path": gs.__file__,
        "model_sha256": hashlib.sha256(model_path().read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_cost": "mean(x^2 + 10*angle^2 + 0.1*(xdot^2+angledot^2))",
        "control_dt_s": 0.04, "physics_dt_s": 0.04 / args.substeps,
        "gradient": "full rollout BPTT through Genesis; no critic or internal state detach",
    }
    write_json(out / "config.json", config)
    print(json.dumps({"event": "config", **config}), flush=True)
    env = GenesisPendulum(args.envs, args.horizon, args.substeps,
                          default_armature=args.default_armature)
    write_json(out / "model_observed.json", {
        "armature": env.robot.get_dofs_armature().cpu().tolist(),
        "damping": env.robot.get_dofs_damping().cpu().tolist(),
        "force_range": [x.cpu().tolist() for x in env.robot.get_dofs_force_range()],
    })
    policy = Policy().to(gs.device)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.LBFGS(policy.parameters(), lr=args.lr,
                                     max_iter=args.lbfgs_max_iter,
                                     line_search_fn="strong_wolfe")
    start_iteration = 0
    prior_steps = 0
    if args.resume or args.initial_policy:
        previous = torch.load(args.resume or args.initial_policy, map_location=gs.device, weights_only=False)
        if previous["config"]["model_sha256"] != config["model_sha256"]:
            raise ValueError("Cannot resume a different benchmark model")
        policy.load_state_dict(previous["policy"])
        if args.resume:
            if previous["config"]["args"].get("optimizer", "adam") != args.optimizer:
                raise ValueError("Use --initial-policy when changing optimizers")
            optimizer.load_state_dict(previous["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_iteration = previous["iteration"] + 1
        old_args = previous["config"]["args"]
        prior_steps = previous.get("environment_steps", start_iteration * old_args["envs"] * old_args["horizon"])
    write_json(out / "first_reset.json", random_resets(args.seed + start_iteration, args.envs).tolist())
    # Validation seeds are disjoint from both training and final evaluation seeds.
    validation = random_resets(500000, args.envs)
    baseline = evaluate_gym(policy, args.eval_seed, args.eval_episodes)
    write_json(out / "baseline_gym.json", baseline)
    print(json.dumps({"event": "baseline_gym", **{k: v for k, v in baseline.items() if k != "records"}}), flush=True)
    best_score = -1.0
    start = time.monotonic()
    training_steps = prior_steps
    with (out / "metrics.jsonl").open("w") as log:
        for local_iteration in range(args.iterations):
            iteration = start_iteration + local_iteration
            resets = random_resets(args.seed + iteration, args.envs)
            evaluations = []

            def closure():
                obs = env.reset(resets)
                if local_iteration == 0 and not evaluations:
                    write_json(out / "first_reset_observed.json", obs.detach().cpu().tolist())
                costs = []
                for _ in range(args.horizon):
                    obs = env.step(policy(obs))
                    costs.append(state_cost(obs))
                objective = torch.stack(costs).mean()
                optimizer.zero_grad(set_to_none=True)
                try:
                    if not torch.isfinite(objective):
                        raise RuntimeError(f"Nonfinite state cost at update {iteration}")
                    objective.backward()
                except Exception as error:
                    # Preserve the actual failed optimizer candidate; never replace
                    # a failed physics gradient with a fabricated/zero gradient.
                    write_json(out / "failed_rollout.json", {
                        "iteration": iteration, "completed_rollouts": evaluations,
                        "state_cost": objective.item(), "error": str(error),
                        "policy": {k: v.detach().cpu().tolist() for k, v in policy.state_dict().items()},
                        "reset_seed": args.seed + iteration,
                    })
                    raise
                # A line search must receive the true derivative of its objective.
                clip = 1.0 if args.optimizer == "adam" else float("inf")
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip, error_if_nonfinite=True)
                evaluations.append((objective.item(), norm.item()))
                return objective

            if args.optimizer == "adam":
                closure()
                optimizer.step()
            else:
                optimizer.step(closure)
            training_steps += len(evaluations) * args.envs * args.horizon
            loss_value, grad_value = evaluations[0]
            record = {"iteration": iteration, "state_cost": loss_value,
                      "grad_norm_before_clip": grad_value,
                      "elapsed_s": time.monotonic() - start,
                      "environment_steps": training_steps,
                      "rollout_evaluations": len(evaluations),
                      "closure_costs": [value for value, _ in evaluations],
                      "weights": policy.feedback.weight.detach().cpu().tolist(),
                      "bias": policy.feedback.bias.detach().cpu().tolist()}
            checkpoint = {"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                          "iteration": iteration, "config": config,
                          "environment_steps": record["environment_steps"]}
            torch.save(checkpoint, out / "latest.pt")
            if (local_iteration + 1) % args.eval_every == 0 or local_iteration == args.iterations - 1:
                result = evaluate_genesis(env, policy, validation, args.validation_steps)
                record["validation"] = result
                if result["mean_return"] > best_score:
                    best_score = result["mean_return"]
                    torch.save(checkpoint, out / "best.pt")
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    # Always evaluate the final update. No selection on the held-out benchmark.
    heldout = random_resets(args.eval_seed, args.envs)
    genesis_result = evaluate_genesis(env, policy, heldout, args.eval_steps)
    gym_result = evaluate_gym(policy, args.eval_seed, args.eval_episodes, out / "gym_trace.csv")
    result = {"genesis_same_solver": genesis_result, "gymnasium_mujoco": gym_result,
              "baseline_gym_mean_return": baseline["mean_return"],
              "evaluated_checkpoint": "latest.pt", "training_elapsed_s": time.monotonic() - start}
    write_json(out / "evaluation.json", result)
    print(json.dumps({"event": "final", "genesis_same_solver": genesis_result,
                      "gymnasium_mujoco": {k: v for k, v in gym_result.items() if k != "records"},
                      "elapsed_s": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
