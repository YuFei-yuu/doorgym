"""CPU-only replay of the failing clock boundary, never physical state replay."""
import argparse,json,sys,hashlib,subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from author_fast5.geometry import smooth
from author_fast5.trajectory import CompleteExpert,Measured

def main():
 import os
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 prior=ROOT/'.repro/runs/wc4-author-fast5-20261028';rows=[json.loads(l) for l in (prior/'attempt_000/states.jsonl').open()]
 effective=json.loads((prior/'attempt_000/effective-config.json').read_text());ref=json.loads((prior/'attempt_000/reference.json').read_text())
 dt=json.loads((prior/'physics-start.json').read_text())['dt'];first=next(r for r in rows if r['phase']=='approach')
 anchor=rows[first['step']-2];failure=rows[-1]
 # Phase enter occurs in the last deploy pre-physics sample = this preceding post sample.
 def obs(r):return Measured(r['t'],np.array(r['actual_base']),np.array(r['arm_q']),np.array(r['arm_qd']),np.array(r['ee_actual']),
  Rotation.from_quat(r['ee_quat']).as_matrix(),r['door_q'][0],r['door_qd'][0],True,not r['nonzero_door_contact'],r['base_measured_fd_speed'],r['base_measured_fd_yaw_rate'],False)
 st=NS(args=NS(**effective),door=NS(actor_position_offset=[.11,0,0],bounding={'min':[0,0,0]},actor_scale=1.,actor_yaw=np.pi/2),
  home_positions=np.array(ref['home']),base_traverse=np.array(ref['old_expert_final_base']),yaw_traverse=ref['old_expert_final_yaw'])
 e=CompleteExpert(NS(),st,ref['dof_names']);A=obs(anchor);o=obs(failure);e.enter('approach',A)
 u=(o.time-A.time)/e.cfg['approach_s'];old=lambda x:x*x*x*(10+x*(-15+6*x));old_s=old(u)
 assert old_s>1 and smooth(u)==1
 result=e.tick(failure['step']+1,None,o);assert np.isfinite(result.rotation).all() and e.phase=='approach'
 result2=e.tick(failure['step']+2,None,replace(o,time=o.time+dt));assert e.phase=='close'
 # Test all relevant durations at every actual step and immediately adjacent floats.
 values=[]
 for duration in [.5,1.,1.5,2.,4.,5.,6.,120.]:
  values.extend((np.arange(6001)*dt/duration).tolist())
 values.extend([0.,1.,np.nextafter(1.,0.),np.nextafter(1.,2.),u])
 bounded=np.array([smooth(x) for x in values]);assert np.all((bounded>=0)&(bounded<=1))
 Slerp([0,1],Rotation.from_euler('z',[0.,.7]))(bounded)
 old_values=np.array([old(float(np.clip(x,0,1))) for x in values]);delta=float(max(abs(bounded-old_values)));assert delta<1e-12
 cfg=Path(__file__).with_name('config.json');old_cfg=subprocess.check_output(['git','show','ea1a56a:migration/wc4_act/author_fast5/config.json'],cwd=ROOT)
 assert cfg.read_bytes()==old_cfg
 out=dict(passed=True,physics_steps=0,actual_dt=dt,anchor_time=A.time,failure_time=o.time,input_u=u,old_s=old_s,repaired_s=smooth(u),
  former_failing_tick_returns=True,next_tick_enters_close=True,domain_samples=len(values),maximum_polynomial_change=delta,
  trajectory_configuration_unchanged=True,config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),
  scope='Actual failed approach timestamp and full 120-second float32 clock domain; no new scene, no trajectory tuning')
 (a.out/'repair-regression.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
