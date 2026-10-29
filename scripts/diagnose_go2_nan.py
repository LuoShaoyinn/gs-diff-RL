"""Instrument the real Go2 evaluation, checking each policy step and per-env solver errno."""
import argparse
import json

import genesis as gs
import torch
from torch import nn
from genesis.utils.misc import qd_to_torch


def stats(x):
    x = x.detach()
    return {'finite': bool(torch.isfinite(x).all().item()), 'values': x.cpu().tolist(),
            'max_abs': float(x.abs().max().item())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--mode', choices=['paired', 'trained', 'baseline'], default='paired')
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dt = ckpt['args']['control_dt']
    substeps = ckpt['args']['substeps']
    gs.init(backend=gs.amdgpu, precision='32', logging_level='warning')
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=substeps), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file='urdf/go2/urdf/go2.urdf', pos=(0, 0, .42)))
    n = 2 if args.mode == 'paired' else 1
    scene.build(n_envs=n)
    names = [f'{leg}_{part}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for part in ('hip', 'thigh', 'calf')]
    dofs = [robot.get_joint(name).dof_start for name in names]
    default = torch.tensor([0, .8, -1.5, 0, .8, -1.5, 0, 1., -1.5, 0, 1., -1.5], device=gs.device)
    qpos = torch.tensor([0, 0, .42, 1, 0, 0, 0] + default.tolist(), device=gs.device).repeat(n, 1)
    robot.set_qpos(qpos, zero_velocity=True)
    policy = nn.Sequential(nn.Linear(38, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 12)).to(gs.device)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()
    command = torch.full((n, 1), .4, device=gs.device)
    def emit(data):
        print(json.dumps(data, allow_nan=True), flush=True)
    emit({'event': 'config', 'mode': args.mode, 'control_dt': dt, 'substeps': substeps,
          'parameter_finite': all(torch.isfinite(p).all().item() for p in policy.parameters())})
    with torch.no_grad():
        for step in range(args.steps):
            state = scene.rigid_solver.get_state()
            pos, vel = state.qpos, state.dofs_vel
            joint_pos, joint_vel = pos[:, 7:], vel[:, 6:]
            phase = torch.full_like(command, min(step / ckpt['args']['horizon'], 1.0))
            obs = torch.cat((pos[:, 3:7], vel[:, :6], joint_pos - default, joint_vel, command,
                             phase, pos[:, 2:3], torch.ones_like(command)), dim=-1)
            raw = policy(obs)
            action = torch.tanh(raw) * .25
            if args.mode in ('paired', 'baseline'):
                action[0] = 0
            torque = 20 * (default + action - joint_pos) - .5 * joint_vel
            record = {'event': 'pre', 'step': step, 'time_s': step * dt, 'pos': stats(pos), 'vel': stats(vel),
                      'obs': stats(obs), 'raw': stats(raw), 'action': stats(action), 'torque': stats(torque)}
            emit(record)
            if not all(record[k]['finite'] for k in ('pos', 'vel', 'obs', 'raw', 'action', 'torque')):
                emit({'event': 'nonfinite_input', 'step': step})
                return
            robot.control_dofs_force(torque, dofs)
            try:
                scene.step()
                errno = qd_to_torch(scene.rigid_solver._errno).cpu().tolist()
                after = scene.rigid_solver.get_state()
                emit({'event': 'post', 'step': step, 'errno': errno, 'pos': stats(after.qpos), 'vel': stats(after.dofs_vel)})
                if any(errno):
                    emit({'event': 'solver_errno', 'step': step, 'errno': errno})
                    return
            except Exception as e:
                emit({'event': 'exception', 'step': step, 'error': str(e),
                      'errno': qd_to_torch(scene.rigid_solver._errno).cpu().tolist()})
                return
    emit({'event': 'complete', 'steps': args.steps})


if __name__ == '__main__':
    main()
