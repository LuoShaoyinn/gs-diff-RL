"""Short-horizon pathwise policy optimization through Genesis rigid dynamics."""
import argparse
import json
import time
from pathlib import Path

import genesis as gs
import torch
from torch import nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--envs', type=int, default=8)
    p.add_argument('--horizon', type=int, default=32)
    p.add_argument('--control-dt', type=float, default=0.02)
    p.add_argument('--substeps', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--out', default='runs/go2')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    gs.init(backend=gs.amdgpu, precision='32', logging_level='warning')
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.control_dt, substeps=args.substeps, requires_grad=True),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file='urdf/go2/urdf/go2.urdf', pos=(0, 0, 0.42)))
    scene.build(n_envs=args.envs)
    names = [f'{leg}_{part}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for part in ('hip', 'thigh', 'calf')]
    dofs = [robot.get_joint(name).dof_start for name in names]
    default = torch.tensor([0, .8, -1.5, 0, .8, -1.5, 0, 1., -1.5, 0, 1., -1.5], device=gs.device)
    qpos = torch.tensor([0, 0, .42, 1, 0, 0, 0] + default.tolist(), device=gs.device).repeat(args.envs, 1)
    policy = nn.Sequential(nn.Linear(38, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 12)).to(gs.device)
    nn.init.zeros_(policy[-1].weight)
    nn.init.zeros_(policy[-1].bias)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(json.dumps({'config': vars(args), 'device': str(gs.device), 'torch': torch.__version__, 'hip': torch.version.hip, 'genesis': gs.__version__}), flush=True)
    start = time.time()
    with (out / 'metrics.jsonl').open('w') as log:
        for iteration in range(args.iterations):
            scene.reset()
            robot.set_qpos(qpos, zero_velocity=True)
            command = torch.linspace(0.2, 0.6, args.envs, device=gs.device).unsqueeze(-1)
            loss = 0.0
            for step in range(args.horizon):
                state = scene.rigid_solver.get_state()
                pos, vel = state.qpos, state.dofs_vel
                joint_pos, joint_vel = pos[:, 7:], vel[:, 6:]
                obs = torch.cat((pos[:, 3:7], vel[:, :6], joint_pos - default, joint_vel, command,
                                 torch.full_like(command, step / args.horizon), pos[:, 2:3],
                                 torch.ones_like(command)), dim=-1)
                action = torch.tanh(policy(obs)) * 0.25
                torque = 20 * (default + action - joint_pos) - 0.5 * joint_vel
                robot.control_dofs_force(torque, dofs)
                scene.step()
                state = scene.rigid_solver.get_state()
                q, v = state.qpos, state.dofs_vel
                loss = loss + ((v[:, 0:1] - command).square().mean() + 2 * (q[:, 2] - .42).square().mean()
                               + 2 * q[:, 4:6].square().sum(dim=-1).mean() + 0.001 * torque.square().mean()) / args.horizon
            optim.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError(f'non-finite loss/gradient at iteration {iteration}: {loss.item()}, {norm.item()}')
            optim.step()
            record = {'iteration': iteration, 'loss': loss.item(), 'grad_norm': norm.item(),
                      'forward_velocity': v[:, 0].mean().item(), 'height': q[:, 2].mean().item(), 'elapsed_s': time.time() - start}
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()
            torch.save({'policy': policy.state_dict(), 'optimizer': optim.state_dict(), 'iteration': iteration, 'args': vars(args)}, out / 'latest.pt')


if __name__ == '__main__':
    main()
