"""Coordinate and timing regressions only; no scene or sample screening."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_geometry import URDF,apply
from fullcycle_scene import SceneFrame
from close_inplace_stage4.common import FullPose,baseT
from random_initial_observe.frame_solver import FrameSolver
from random_initial_observe.reference import reference
from close_inplace_stage8.reference import reference as original
from random_initial_observe.controller import LiveExpert
from newexpert_stage4.prepare import sha,write
run=Path(sys.argv[1]).resolve();manifest=json.loads((run/'manifest.json').read_text())
robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf',include_visual=True)
q=np.array([.2,1.5,-1.1,-.2,.1,0.]);B=np.array([1.26,.18,4.71]);old=FullPose(robot)
F=baseT(B)@old.fk(q);p=F[:3,3]+F[:3,:3]@old.pad;R=F[:3,:3]
s=FrameSolver(robot,SceneFrame());qa,ia=s.solve(B,p,R,q+.01,80);qb,ib=old.solve(B,p,R,q+.01,80)
assert np.max(abs(qa-qb))<1e-8
checks=[]
for trial in manifest['trials']:
    e=trial['effective_sampling'];sc=SceneFrame((2.467+e['door_x']-2.5,-.463+e['door_y'],.01),base_z=e['robot_z'],base_pitch=e['robot_pitch'])
    solver=FrameSolver(robot,sc);F=sc.base_transform(B)@solver.fk(q);p=F[:3,3]+F[:3,:3]@solver.pad
    x,info=solver.solve(B,p,F[:3,:3],q,80);assert info['position_error_m']<1e-7
    p2=apply(solver.mapping(B),np.array([p]))[0];assert np.max(abs(p2-(baseT(B)@solver.fk(q))[:3,3]-(baseT(B)@solver.fk(q))[:3,:3]@solver.pad))<1e-12
    assert np.max(abs(sc.to_reference_base(sc.to_world_base(B))-B))<1e-12
    checks.append(dict(seed=trial['seed'],height=e['robot_z'],pitch=e['robot_pitch'],coordinate_error_m=info['position_error_m']))
# Same parking polynomial, endpoint and initial/final speed, on a sampled heading.
for yaw in [np.pi,3.12,3.16]:
    dt=float(np.float32(.02));L=.18;v=.12;T=np.ceil(2*L/v/dt)*dt;a=v*T;u=np.linspace(0,1,10001)
    travel=a*u+(10*L-6*a)*u**3+(-15*L+8*a)*u**4+(6*L-3*a)*u**5
    speed=(a+3*(10*L-6*a)*u**2+4*(-15*L+8*a)*u**3+5*(6*L-3*a)*u**4)/T
    assert abs(speed[0]-v)<1e-12 and abs(speed[-1])<1e-12 and min(speed)>-1e-12
    assert abs(travel[-1]-L)<1e-12 and max(travel)<=L+1e-12
for t in np.linspace(0,30,1501):assert reference(t,1.4)==original(t,1.4)
for n in ['controller.py','early_arm.py','runtime.py','execute.py','prepare.py','frame_solver.py']:compile((Path(__file__).parent/n).read_text(),n,'exec')
# Author physics/assets and base control are identical to stage8.
changed=subprocess.check_output(['git','diff','--name-only','90f6f52','HEAD'],cwd=ROOT,text=True).splitlines()
assert all(n.startswith('migration/wc4_act/random_initial_observe/') for n in changed)
write(run/'cpu-check.json',dict(passed=True,scene_creations=0,scope='coordinate equivalence, parking continuity, unchanged reference and protected sources; no reachability screening',sample_coordinates=checks,nominal_ik_joint_difference=float(np.max(abs(qa-qb)))))
print((run/'cpu-check.json').read_text())
