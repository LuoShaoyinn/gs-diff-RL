"""Compare a trained policy and its zero-output initial controller over a longer rollout."""
import argparse
import json

import genesis as gs
import torch
from torch import nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--command', type=float, default=0.4)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    gs.init(backend=gs.amdgpu, precision='32', logging_level='warning')
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.01, substeps=2), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file='urdf/go2/urdf/go2.urdf', pos=(0, 0, .42)))
    scene.build(n_envs=2)
    names = [f'{leg}_{part}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for part in ('hip', 'thigh', 'calf')]
    dofs = [robot.get_joint(name).dof_start for name in names]
    default = torch.tensor([0, .8, -1.5, 0, .8, -1.5, 0, 1., -1.5, 0, 1., -1.5], device=gs.device)
    qpos = torch.tensor([0, 0, .42, 1, 0, 0, 0] + default.tolist(), device=gs.device).repeat(2, 1)
    robot.set_qpos(qpos, zero_velocity=True)
    policy = nn.Sequential(nn.Linear(38, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 12)).to(gs.device)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()
    command = torch.full((2, 1), args.command, device=gs.device)
    minimum = torch.full((2,), float('inf'), device=gs.device)
    with torch.no_grad():
        for step in range(args.steps):
            state = scene.rigid_solver.get_state()
            pos, vel = state.qpos, state.dofs_vel
            joint_pos, joint_vel = pos[:, 7:], vel[:, 6:]
            phase = torch.full_like(command, min(step / ckpt['args']['horizon'], 1.0))
            obs = torch.cat((pos[:, 3:7], vel[:, :6], joint_pos - default, joint_vel, command,
                             phase, pos[:, 2:3], torch.ones_like(command)), dim=-1)
            action = torch.tanh(policy(obs)) * .25
            action[0] = 0  # initial controller; environment 1 uses the trained policy
            torque = 20 * (default + action - joint_pos) - .5 * joint_vel
            robot.control_dofs_force(torque, dofs)
            scene.step()
            state = scene.rigid_solver.get_state()
            minimum = torch.minimum(minimum, state.qpos[:, 2])
            if (step + 1) % 20 == 0 or step == args.steps - 1:
                print(json.dumps({'step': step + 1, 'time_s': (step + 1) * .01,
                                  'x': state.qpos[:, 0].tolist(), 'height': state.qpos[:, 2].tolist(),
                                  'vx': state.dofs_vel[:, 0].tolist(), 'min_height': minimum.tolist(),
                                  'order': ['zero_output', 'trained']}), flush=True)


if __name__ == '__main__':
    main()
