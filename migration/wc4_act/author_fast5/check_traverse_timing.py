"""CPU-only timing/IK check from the accepted nominal traversal entry.

Reuses the original author traversal interpolation and accepted entry state.
Ideal arm tracking is a kinematic check, not a replay or a new physical rollout.
"""
import argparse, ast, hashlib, json, math, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from author_fast5.trajectory import CompleteExpert,Measured
from author_fast5.check_design import AuthorDLS
from author_fast5.geometry import DoorFrame,transform,traverse_clock
from fullcycle_geometry import URDF

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    a.out.mkdir(parents=True,exist_ok=False)
    baseline=ROOT/'.repro/runs/wc4-contact-inset-20261028/attempt_000'
    rows=[json.loads(l) for l in (baseline/'states.jsonl').open()]
    effective=json.loads((baseline/'effective-config.json').read_text())
    ref=json.loads((baseline/'reference.json').read_text());names=ref['dof_names']
    start=next(i for i,r in enumerate(rows) if r['phase']=='traverse_door')
    last=next(i-1 for i,r in enumerate(rows) if i>start and r['phase']!='traverse_door')
    prior=rows[start-1];P=np.array(prior['ee_target']);R=Rotation.from_quat(prior['ee_target_quat']).as_matrix()
    held_gripper=float(prior['arm_command'][names.index('jointGripper')])
    dt=float(np.float32(.02))
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf')
    frame=DoorFrame(transform([2.61,0,.01],[0,0,math.pi/2]),base_z=effective['robot_z'],base_pitch=effective['robot_pitch'])
    ik=AuthorDLS(robot,effective)
    old=json.loads(subprocess.check_output(['git','show','482152f:migration/wc4_act/author_fast5/config.json'],cwd=ROOT,text=True))
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    source=ROOT/'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py'
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='trajectory_targets')
    branch=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.comparators[0],ast.Name) and n.test.comparators[0].id=='traverse_end')
    code=compile(ast.Module(body=branch.body,type_ignores=[]),str(source),'exec')
    def run(c):
        args=NS(**effective)
        args.sim_dt=dt;args.door_z_offset=.01
        st=NS(args=args,door=NS(actor_position_offset=[.11,0,0],bounding={'min':[0,0,0]},actor_scale=1.,actor_yaw=math.pi/2),
            home_positions=np.array(ref['home']),base_traverse=np.array(rows[last]['command_base'][:2]),yaw_traverse=rows[last]['command_base'][2],
            env=None,door_actor=None,ik_state=None,base_start=None,base_stop=None,base_push=None,yaw_start=None,yaw_push=None,traj={})
        def author(step,*unused):
            if step<last+1:
                ns=dict(np=np,args=args,step=step,push_end=start,traverse_steps=last-start+1,
                    base_traverse=st.base_traverse.astype(np.float32),yaw_traverse=st.yaw_traverse,gripper_closed=held_gripper,
                    lerp=lambda x,y,t:(1-t)*x+t*y,
                    traj=dict(traverse_start_base_xy=np.asarray(prior['command_base'][:2],dtype=np.float32),traverse_start_yaw=st.yaw_traverse,
                        traverse_hold_target_pos=P.copy(),traverse_hold_target_quat=Rotation.from_matrix(R).as_quat(),last_gripper=held_gripper))
                exec(code,ns)
                return (ns['phase'],ns['base_xy'],ns['yaw'],ns['target_pos'],ns['target_quat'],ns['gripper'],None)
            return ('hold_home',st.base_traverse.copy(),st.yaw_traverse,P.copy(),Rotation.from_matrix(R).as_quat(),held_gripper,None)
        e=CompleteExpert(NS(trajectory_targets=author),st,names,c)
        q=np.array(prior['arm_q']);B=np.array(prior['actual_base']);F=transform(prior['ee_actual']);F[:3,:3]=Rotation.from_quat(prior['ee_quat']).as_matrix()
        trace=[];prevq=q.copy()
        for step in range(start,start+1800):
            o=Measured(step*dt,B.copy(),q.copy(),(q-prevq)/dt,F[:3,3].copy(),F[:3,:3].copy(),math.pi/2,0.,True,True,0.,0.,False)
            target=e.tick(step,None,o);assert target.failed is None
            prevq=q.copy();B=target.base.copy()
            if target.joints is not None:q=target.joints.copy()
            else:q[:6]=ik.step(B,q[:6],target.position,target.rotation,frame)
            q[6]=target.gripper
            F,_=ik.fk_jacobian(B,q[:6],frame)
            elapsed=(step-start)*dt
            mode='joint' if target.joints is not None else 'cartesian'
            if not c.get('traverse_extra_s',0.) and step<=last:assert np.max(abs(B-np.asarray(rows[step]['command_base'])))<1e-6
            trace.append(dict(t=step*dt,elapsed=elapsed,phase=target.phase,mode=mode,base=B.tolist(),q=q.tolist(),
                target=(F[:3,3] if mode=='joint' else target.position).tolist(),ee=F[:3,3].tolist(),
                position_error_m=0. if mode=='joint' else float(np.linalg.norm(F[:3,3]-target.position)),
                joint_margin_rad=float(np.min(np.r_[q[:6]-ik.lo,ik.hi-q[:6]])),joint_rate_rad_s=float(np.max(abs(q[:6]-prevq[:6]))/dt)))
            if e.phase=='deploy':break
        assert e.phase=='deploy'
        return trace
    traces=[run(old),run(cfg)]
    result={}
    for label,tr in zip(['accepted','tuned'],traces):
        cart=[r for r in tr if r['mode']=='cartesian']
        result[label]=dict(withdrawal_duration_s=next(r['elapsed'] for r in tr if r['mode']=='joint'),cartesian_max_position_error_m=max(r['position_error_m'] for r in cart),
            minimum_joint_margin_rad=min(r['joint_margin_rad'] for r in tr),max_joint_rate_rad_s=max(r['joint_rate_rad_s'] for r in tr),
            first_joint_fold_time_s=next(r['t'] for r in tr if r['mode']=='joint'),turn_start_s=next(r['t'] for r in tr if r['phase']=='turn_for_close'),deploy_entry_s=tr[-1]['t'])
        with (a.out/(label+'-trace.jsonl')).open('w') as f:
            for r in tr:f.write(json.dumps(r)+'\n')
    # Same normalized pose path and half velocity for both isolated base turns.
    from author_fast5.geometry import smooth
    turns={}
    for key,delta in [('turn_s',math.radians(80)),('restore_s',math.pi)]:
        u=np.linspace(0,1,2001);yaw=np.array([delta*smooth(x) for x in u]);v0=np.diff(yaw)/(old[key]/2000);v1=np.diff(yaw)/(cfg[key]/2000)
        assert cfg[key]==2*old[key] and np.allclose(v1,.5*v0)
        turns[key]=dict(old_s=old[key],new_s=cfg[key],old_peak_rad_s=float(max(abs(v0))),new_peak_rad_s=float(max(abs(v1))),speed_ratio=.5)
    elapsed=np.linspace(0,6,6001);clock=np.array([traverse_clock(cfg,t) for t in elapsed]);rate=np.diff(clock)/.001
    assert min(rate)>0 and abs(clock[-1]-(elapsed[-1]-cfg['traverse_extra_s']))<1e-12
    result.update(base_turns=turns,original_base_traversal_geometry_preserved=True,minimum_traverse_clock_rate=float(min(rate)),physics_steps=0,
        baseline_states_sha256=hashlib.sha256((baseline/'states.jsonl').read_bytes()).hexdigest(),
        limitation='Nominal kinematics with ideal arm tracking. Original author traversal interpolation executed with fractional reference steps. No contact forces or new physical door/foot result; actual hold-open effect requires a later authorized rollout.')
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for label,tr in zip(['Accepted','Tuned'],traces):
        t=np.array([r['elapsed'] for r in tr]);B=np.array([r['base'] for r in tr]);ee=np.array([r['ee'] for r in tr])
        axs[0].plot(t,np.linalg.norm(ee-ee[0],axis=1),label=label)
        axs[1].plot(t,np.degrees(B[:,2]),label=label)
        axs[2].plot(t,[r['joint_margin_rad'] for r in tr],label=label)
    axs[0].set_ylabel('EE displacement (m)');axs[1].set_ylabel('Base yaw (deg)');axs[2].set_ylabel('Joint margin (rad)');axs[2].set_xlabel('Seconds after traverse starts')
    for ax in axs:ax.axvline(cfg['traverse_slow_window_s'],ls=':',color='gray');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Coordinated slower withdrawal and preparation turn — ideal CPU check, NOT a rollout');fig.tight_layout();fig.savefig(a.out/'timing-check.png',dpi=150)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
