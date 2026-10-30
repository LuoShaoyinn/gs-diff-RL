"""Short-horizon actor-critic Go2 experiment inspired by Open-DiffLoco."""
import argparse
import json
import time
from pathlib import Path

import genesis as gs
import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='runs/go2-shac')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    gs.init(backend=gs.amdgpu, precision='32', logging_level='warning')
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.02, substeps=20, requires_grad=True), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file='urdf/go2/urdf/go2.urdf', pos=(0, 0, .42)))
    scene.build(n_envs=args.envs)
    names = [f'{leg}_{part}_joint' for leg in ('FR','FL','RR','RL') for part in ('hip','thigh','calf')]
    dofs = [robot.get_joint(name).dof_start for name in names]
    default = torch.tensor([0,.8,-1.5,0,.8,-1.5,0,1.,-1.5,0,1.,-1.5], device=gs.device)
    initial_q = torch.tensor([0,0,.42,1,0,0,0]+default.tolist(), device=gs.device).repeat(args.envs,1)
    q, v = initial_q.clone(), torch.zeros(args.envs,18,device=gs.device)
    previous = torch.zeros(args.envs,12,device=gs.device)
    command = torch.empty(args.envs,1,device=gs.device).uniform_(.2,.6)
    age = torch.zeros(args.envs,device=gs.device,dtype=torch.long)
    def network(out_dim):
        return nn.Sequential(nn.Linear(38,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,out_dim)).to(gs.device)
    actor, critic, target = network(12), network(1), network(1)
    nn.init.zeros_(actor[-1].weight); nn.init.zeros_(actor[-1].bias)
    target.load_state_dict(critic.state_dict()); target.requires_grad_(False)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=.001)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=.0005)
    gamma = .99
    def obs(q,v,prev,cmd):
        return torch.cat((q[:,3:7],v[:,:6],q[:,7:]-default,v[:,6:],cmd,q[:,2:3],torch.ones_like(cmd),torch.zeros_like(cmd)),dim=-1)
    def reward(q,v,act,prev,cmd):
        # Adapted from Open-DiffLoco's smooth, reference-free objective.
        qw,qx,qy,qz = q[:,3],q[:,4],q[:,5],q[:,6]
        gz = 1 - 2*(qx.square()+qy.square())
        return (-(v[:,0]-cmd[:,0]).square()-v[:,1].square()-v[:,5].square()
                + torch.exp(-10*(q[:,2]-.3).square()) - .5*v[:,2].square() + .5*gz
                - .3*(q[:,7:]-default).square().sum(-1) - .02*(act-prev).square().sum(-1)
                - .05*act.square().sum(-1) - .05*v[:,3:5].square().sum(-1))
    start=time.time()
    with (out/'metrics.jsonl').open('w') as log:
        for it in range(args.iterations):
            scene.reset(); robot.set_qpos(q,zero_velocity=True); robot.set_dofs_velocity(v)
            observations=[]; rewards=[]
            for t in range(args.horizon):
                state=scene.rigid_solver.get_state(); q,v=state.qpos,state.dofs_vel
                ob=obs(q,v,previous,command); act=torch.tanh(actor(ob))*.25
                torque=20*(default+act-q[:,7:])-.5*v[:,6:]
                robot.control_dofs_force(torque,dofs); scene.step()
                state=scene.rigid_solver.get_state(); q,v=state.qpos,state.dofs_vel
                observations.append(ob.detach()); rewards.append(reward(q,v,act,previous,command)); previous=act
            final_obs=obs(q,v,previous,command)
            upright = 1 - 2*(q[:,4].square() + q[:,5].square())
            alive=((q[:,2]>.22) & (upright>.5)).float().detach()
            # Critic parameters are frozen, while its derivative w.r.t. terminal state remains in actor loss.
            actor_loss=-(sum((gamma**t)*r for t,r in enumerate(rewards)) + gamma**args.horizon*alive*target(final_obs).squeeze(-1)).mean()
            actor_opt.zero_grad(set_to_none=True); actor_loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
            if not torch.isfinite(actor_loss) or not torch.isfinite(gn): raise RuntimeError(f'nonfinite actor iteration {it}: {actor_loss} {gn}')
            actor_opt.step()
            with torch.no_grad():
                returns=alive*target(final_obs.detach()).squeeze(-1)
                targets=[]
                for r in reversed(rewards):
                    returns=r.detach()+gamma*returns; targets.append(returns)
                targets=torch.stack(targets[::-1]).flatten()
                obs_batch=torch.stack(observations).reshape(-1,38)
            for _ in range(16):
                critic_opt.zero_grad(set_to_none=True)
                critic_loss=(critic(obs_batch).squeeze(-1)-targets).square().mean()
                critic_loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(),10.); critic_opt.step()
            with torch.no_grad():
                for p,tp in zip(critic.parameters(),target.parameters()): tp.lerp_(p,.2)
                q=q.detach().clone(); v=v.detach().clone(); previous=previous.detach().clone(); age+=args.horizon
                done=(alive == 0)|(age>=1000)|(~torch.isfinite(q).all(-1))|(~torch.isfinite(v).all(-1))
                q[done]=initial_q[done]; v[done]=0; previous[done]=0; age[done]=0
                command[done]=torch.empty((int(done.sum()),1),device=gs.device).uniform_(.2,.6)
            rec={'iteration':it,'reward':torch.stack(rewards).mean().item(),'actor_loss':actor_loss.item(),
                 'critic_loss':critic_loss.item(),'grad_norm':gn.item(),'height':q[:,2].mean().item(),
                 'vx':v[:,0].mean().item(),'resets':int(done.sum()),'elapsed_s':time.time()-start}
            print(json.dumps(rec),flush=True); log.write(json.dumps(rec)+'\n'); log.flush()
            torch.save({'actor':actor.state_dict(),'critic':critic.state_dict(),'target':target.state_dict(),
                        'actor_optimizer':actor_opt.state_dict(),'critic_optimizer':critic_opt.state_dict(),
                        'q':q,'v':v,'previous':previous,'command':command,'age':age,'iteration':it,'args':vars(args)},out/'latest.pt')


if __name__=='__main__': main()
