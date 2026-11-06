"""Compare original physics states with checkpoint reconstruction in the failed rollout."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import genesis as gs
import numpy as np
import torch
from genesis.utils.misc import qd_to_numpy
from train_inverted_pendulum import GenesisPendulum, Policy, model_path, random_resets, state_cost, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', default='runs/nan-backward/checkpoint-replay')
args = parser.parse_args()
root = Path(args.out)
root.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
gs.init(backend=gs.cuda, precision='32', seed=42, logging_level='warning')
checkpoint = torch.load('results/public-inverted-pendulum/matched-h128/latest.pt', map_location=gs.device, weights_only=False)
for window in (1000, 25):
    out = root / f'window-{window}'
    out.mkdir()
    env = GenesisPendulum.__new__(GenesisPendulum)
    env.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.04, substeps=4,
        substeps_local=4*window, requires_grad=True),
        rigid_options=gs.options.RigidOptions(enable_collision=False), show_viewer=False)
    env.robot = env.scene.add_entity(gs.morphs.MJCF(file=str(model_path()), default_armature=0.0))
    env.scene.build(n_envs=64)
    env.slider = env.robot.get_joint('slider').dof_start
    solver = env.scene.rigid_solver
    sim = env.scene.sim
    policy = Policy().to(gs.device)
    policy.load_state_dict(checkpoint['policy'])
    original_load = sim.load_ckpt
    replay_records = []
    def observed_load():
        start = sim.cur_substep_global - sim._substeps_local
        reference = solver._ckpt[str(start)]
        original_load()
        row = {'start_substep': start, 'window_steps': window}
        arrays = {}
        for name in ('qpos', 'dofs_vel', 'dofs_acc'):
            expected = reference[name]
            actual = qd_to_numpy(getattr(solver._rigid_adjoint_cache, name), copy=True)
            delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
            per_substep = np.max(delta, axis=(1,2))
            nonzero = np.flatnonzero(per_substep > 0)
            row[name] = {'max_absolute_difference': float(np.max(delta)),
                'original_max_abs': float(np.max(np.abs(expected))),
                'replayed_max_abs': float(np.max(np.abs(actual))),
                'first_different_substep': int(nonzero[0]) if len(nonzero) else None,
                'difference_at_start': float(per_substep[0]),
                'difference_at_end': float(per_substep[-1])}
            arrays[name+'_original'] = expected
            arrays[name+'_replayed'] = actual
            arrays[name+'_max_error_per_substep'] = per_substep
        if window == 1000 or start == 3900:
            np.savez_compressed(out/f'replay-{start}.npz', **arrays)
        replay_records.append(row)
        write_json(out/'replay.json', replay_records)
        print(json.dumps({'event':'replay', **row}), flush=True)
    sim.load_ckpt = observed_load
    start_time = time.monotonic()
    obs = env.reset(random_resets(49,64))
    costs = []
    for step in range(1000):
        obs = env.step(policy(obs))
        costs.append(state_cost(obs))
    loss = torch.stack(costs).mean()
    report = {'window_steps':window, 'horizon':1000, 'pid':os.getpid(),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'state_cost':loss.item()}
    print(json.dumps({'event':'forward', **report}), flush=True)
    try:
        loss.backward()
        report['backward_completed'] = True
        report['gradients'] = {name:p.grad.detach().cpu().tolist() for name,p in policy.named_parameters()}
        report['gradients_finite'] = all(torch.isfinite(p.grad).all().item() for p in policy.parameters())
    except Exception as error:
        report['backward_completed'] = False
        report['error'] = str(error)
    report['elapsed_s'] = time.monotonic()-start_time
    write_json(out/'summary.json',report)
    print(json.dumps({'event':'summary', **report}),flush=True)
