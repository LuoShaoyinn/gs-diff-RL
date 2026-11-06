"""Forward-only evaluation of the short-horizon actor-critic Go2 policy."""
import argparse
import json
import genesis as gs
import torch
from torch import nn

p=argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('--steps',type=int,default=250)
p.add_argument('--backend', choices=['cuda','amdgpu'], default='cuda')
p.add_argument('--legacy-pd', action='store_true', help='Reproduce the old cross-wired PD controller'); args=p.parse_args()
ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
gs.init(backend=getattr(gs,args.backend),precision='32',logging_level='warning')
scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.02,substeps=20),show_viewer=False)
scene.add_entity(gs.morphs.Plane()); robot=scene.add_entity(gs.morphs.URDF(file='urdf/go2/urdf/go2.urdf',pos=(0,0,.42)));scene.build(n_envs=2)
names=[f'{leg}_{part}_joint' for leg in ('FR','FL','RR','RL') for part in ('hip','thigh','calf')]
dofs=[robot.get_joint(name).dof_start for name in names]
q_indices=[robot.get_joint(name).q_start for name in names]
default=torch.tensor([0,.8,-1.5,0,.8,-1.5,0,1.,-1.5,0,1.,-1.5],device=gs.device)
q=torch.tensor([0,0,.42,1,0,0,0]+default.tolist(),device=gs.device).repeat(2,1);robot.set_qpos(q,zero_velocity=True)
actor=nn.Sequential(nn.Linear(38,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,12)).to(gs.device);actor.load_state_dict(ckpt['actor']);actor.eval()
command=torch.full((2,1),.4,device=gs.device); minheight=torch.full((2,),float('inf'),device=gs.device)
with torch.no_grad():
 for step in range(args.steps):
  state=scene.rigid_solver.get_state(); q,v=state.qpos,state.dofs_vel
  ob=torch.cat((q[:,3:7],v[:,:6],q[:,7:]-default,v[:,6:],command,q[:,2:3],torch.ones_like(command),torch.zeros_like(command)),dim=-1)
  action=torch.tanh(actor(ob))*.25; action[0]=0
  pd_q=q[:,7:] if args.legacy_pd else q[:,q_indices]
  pd_v=v[:,6:] if args.legacy_pd else v[:,dofs]
  robot.control_dofs_force(20*(default+action-pd_q)-.5*pd_v,dofs); scene.step()
  state=scene.rigid_solver.get_state(); q,v=state.qpos,state.dofs_vel; minheight=torch.minimum(minheight,q[:,2])
  if (step+1)%20==0 or step==args.steps-1:
   print(json.dumps({'step':step+1,'time_s':(step+1)*.02,'x':q[:,0].tolist(),'height':q[:,2].tolist(),'vx':v[:,0].tolist(),'min_height':minheight.tolist(),'order':['zero_output','trained'],'controller':'legacy' if args.legacy_pd else 'named_pd_v1'}),flush=True)
