"""Replay the actual Go2 controller with legacy and named-joint PD feedback."""

import argparse
import json
from pathlib import Path

import genesis as gs
import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--backend", choices=["cuda", "amdgpu"], default="cuda")
    parser.add_argument("--out", default="runs/go2-joint-order")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    gs.init(backend=getattr(gs, args.backend), precision="32", logging_level="warning")
    dt = checkpoint["args"].get("control_dt", 0.02)
    substeps = checkpoint["args"].get("substeps", 20)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=substeps), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0, 0.42)))
    scene.build(n_envs=4)
    names = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL")
             for part in ("hip", "thigh", "calf")]
    dofs = [robot.get_joint(name).dof_start for name in names]
    q_indices = [robot.get_joint(name).q_start for name in names]
    mapping = {"genesis": gs.__version__, "device": str(gs.device),
               "control_names": names, "control_dof_indices": dofs,
               "control_q_indices": q_indices,
               "legacy_velocity_indices": list(range(6, 18)),
               "legacy_position_indices": list(range(7, 19)),
               "raw_joint_order": [name for _, name in sorted(zip(q_indices, names))]}
    (out / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps({"event": "mapping", **mapping}), flush=True)
    nominal = torch.tensor([0, .8, -1.5, 0, .8, -1.5, 0, 1., -1.5, 0, 1., -1.5], device=gs.device)
    q0 = torch.tensor([0, 0, .42, 1, 0, 0, 0] + nominal.tolist(), device=gs.device).repeat(4, 1)
    robot.set_qpos(q0, zero_velocity=True)
    policy = nn.Sequential(nn.Linear(38, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 12)).to(gs.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    command = torch.full((4, 1), .4, device=gs.device)
    order = ["zero_legacy_pd", "zero_aligned_pd", "trained_legacy_pd", "trained_aligned_pd"]
    with torch.no_grad(), (out / "trace.jsonl").open("w") as log:
        for step in range(args.steps):
            state = scene.rigid_solver.get_state()
            q, v = state.qpos, state.dofs_vel
            # Keep the old policy's observation convention. Only PD feedback is changed.
            obs = torch.cat((q[:, 3:7], v[:, :6], q[:, 7:] - nominal, v[:, 6:], command,
                             torch.full_like(command, min(step / checkpoint["args"]["horizon"], 1.0)),
                             q[:, 2:3], torch.ones_like(command)), -1)
            action = .25 * torch.tanh(policy(obs))
            action[:2] = 0
            legacy = 20 * (nominal + action - q[:, 7:]) - .5 * v[:, 6:]
            aligned = 20 * (nominal + action - q[:, q_indices]) - .5 * v[:, dofs]
            torque = legacy.clone()
            torque[[1, 3]] = aligned[[1, 3]]
            mismatch = (legacy - aligned).abs().amax(-1).tolist()
            robot.control_dofs_force(torque, dofs)
            scene.step()
            state = scene.rigid_solver.get_state()
            record = {"step": step + 1, "time_s": (step + 1) * dt, "order": order,
                      "height_m": state.qpos[:, 2].tolist(), "x_m": state.qpos[:, 0].tolist(),
                      "vx_m_s": state.dofs_vel[:, 0].tolist(),
                      "max_legacy_vs_aligned_torque_difference_nm": mismatch}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if (step + 1) % 20 == 0 or step == args.steps - 1:
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
