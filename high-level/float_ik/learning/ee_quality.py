#!/usr/bin/env python3
"""CPU trajectory quality audit, with separate command/IK/servo diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from dataset_contract import validate_episode
from schema import EE_SCHEMA

# Engineering screening limits for this smoke dataset, fixed before collection.
LIMITS = dict(position_p95_m=.08, position_max_m=.15, target_step_max_m=.08,
              servo_position_p95_m=.08, rotation_p95_rad=.6, fk_max_m=.005)

def stats(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),p50=float(np.percentile(x,50)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),max=float(x.max()))

def world_positions(states, metrics):
    yaw=np.asarray([float(x['base_yaw']) for x in metrics]);c=np.cos(yaw);s=np.sin(yaw)
    local=states[:,21:24]
    return np.column_stack([c*local[:,0]-s*local[:,1]+[float(x['base_x']) for x in metrics],s*local[:,0]+c*local[:,1]+[float(x['base_y']) for x in metrics],local[:,2]+.6])

def audit(folder, plot=False, limits=None):
    limits = dict(LIMITS if limits is None else limits)
    folder = Path(folder)
    summary=json.loads((folder/'summary.json').read_text())
    if summary['status']!='completed':return dict(path=str(folder),passed=False,status=summary['status'],seed=summary['seed'])
    with np.load(folder/'episode.npz') as z:
        validate_episode(z,EE_SCHEMA)
        state=z['state'];nstate=z['next_state'];action=z['action'];jt=z['joint_position_target'];q=z['actual_joint_position']
        aligned=bool(np.array_equal(state[1:],nstate[:-1]))
    with np.load(folder/'trajectory.npz') as z:phase=z['phase']
    rows=list(csv.DictReader((folder/'metrics.csv').open()))
    col=lambda k:np.asarray([float(x[k]) for x in rows])
    pos=col('ee_command_pos_error_m');rot=col('ee_command_rot_error_rad');servo=col('fk_ee_tracking_m')
    target_step=np.linalg.norm(np.diff(action[:,3:6],axis=0),axis=1)
    aq=action[:,6:10].astype(float);aq/=np.linalg.norm(aq,axis=1,keepdims=True)
    angle_step=2*np.arccos(np.clip(abs((aq[1:]*aq[:-1]).sum(1)),0,1))
    actual=world_positions(nstate,rows)
    actual_step=np.linalg.norm(np.diff(actual,axis=0),axis=1)
    # Check the stored frame convention against the simulator's direct EE error.
    position_audit=float(abs(np.linalg.norm(action[:,3:6]-actual,axis=1)-pos).max())
    d=dict(path=str(folder),seed=summary['seed'],status=summary['status'],success=summary['success'],reach=summary['reach'],grasp=summary['grasp'],unlocked=summary['unlocked'],final_angle_deg=summary['signed_angle_deg'],rear_clearance_m=summary['rear_clearance_m'],steps=summary['steps_completed'],position_error_m=stats(pos),orientation_error_rad=stats(rot),servo_position_error_m=stats(servo),target_step_m=stats(target_step),target_rotation_step_rad=stats(angle_step),actual_ee_step_m=stats(actual_step),joint_target_step_rad=stats(np.max(abs(np.diff(jt[:,:6],axis=0)),axis=1)),joint_servo_error_rad=stats(np.max(abs(jt[:,:6]-q[:,:6]),axis=1)),joint_velocity_radps=stats(np.max(abs(nstate[:,14:20]),axis=1)),ik_correction_rad=stats(col('ik_joint_delta_max_rad')),fk_max_m=float(col('fk_validation_m').max()),world_frame_error_audit_max_m=position_audit,frame_alignment=aligned)
    d['by_phase']={name:dict(position_error_m=stats(pos[phase==name]),orientation_error_rad=stats(rot[phase==name]),servo_position_error_m=stats(servo[phase==name])) for name in sorted(set(phase))}
    checks=dict(completed_requested_steps=len(action)==summary.get('steps',2850),frame_alignment=aligned,world_frame_audit=position_audit<1e-5,task_success=bool(summary['success'] and summary['reach'] and summary['grasp']),position_p95=stats(pos)['p95']<=limits['position_p95_m'],position_max=pos.max()<=limits['position_max_m'],target_step_max=target_step.max()<=limits['target_step_max_m'],servo_position_p95=stats(servo)['p95']<=limits['servo_position_p95_m'],rotation_p95=stats(rot)['p95']<=limits['rotation_p95_rad'],fk_max=d['fk_max_m']<=limits['fk_max_m'])
    checks={key:bool(value) for key,value in checks.items()}
    d['checks']=checks;d['passed']=all(checks.values())
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t=np.arange(len(action))/60
        fig,axes=plt.subplots(3,2,figsize=(13,10))
        for j,k in enumerate('xyz'):
            axes[0,0].plot(t,action[:,3+j],label=k+' target',lw=1)
            axes[0,0].plot(t,actual[:,j],'--',label=k+' actual',lw=.8)
        axes[0,0].set(ylabel='World position (m)',title='EE target and post-step actual');axes[0,0].legend(ncol=3,fontsize=7)
        axes[0,1].plot(t,pos,label='Target to actual EE');axes[0,1].plot(t,servo,label='FK(joint command) to actual EE');axes[0,1].set(ylabel='Position error (m)');axes[0,1].legend(fontsize=8)
        axes[1,0].plot(t[1:],target_step*1000,label='Target');axes[1,0].plot(t[1:],actual_step*1000,label='Actual');axes[1,0].set(ylabel='EE displacement / 60 Hz step (mm)');axes[1,0].legend()
        axes[1,1].plot(t,rot*180/np.pi);axes[1,1].set(ylabel='Orientation tracking error (degrees)')
        axes[2,0].plot(t[1:],np.max(abs(np.diff(jt[:,:6],axis=0)),axis=1),label='Joint target jump');axes[2,0].plot(t,col('ik_joint_delta_max_rad'),label='IK command minus current joint');axes[2,0].set(ylabel='Max arm joint change (rad)');axes[2,0].legend(fontsize=8)
        axes[2,1].plot(t,col('signed_angle_deg'));axes[2,1].set(ylabel='Door angle (degrees)')
        for ax in axes.flat:ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2)
        fig.suptitle(folder.name+' | '+summary['mode']+' | '+str(summary.get('checkpoint_step','expert')))
        fig.tight_layout();fig.savefig(folder/'quality.png',dpi=140);plt.close(fig)
    return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--limits',help='Optional JSON with all quality limits fixed before collection');p.add_argument('stage',choices=['pilot','dataset','evaluation']);a=p.parse_args();r=Path(a.root).resolve();plan=json.loads((r/'plan.json').read_text())
    out=r/({'pilot':'pilot_quality.json','dataset':'dataset_quality.json','evaluation':'evaluation_quality.json'}[a.stage])
    if out.exists():raise RuntimeError('Audit already exists; preserve it')
    folders=[r/'pilot/expert_nominal'] if a.stage=='pilot' else sorted((r/('data' if a.stage=='dataset' else 'validation')).glob('*'))
    limits=json.loads(Path(a.limits).read_text()) if a.limits else LIMITS
    if set(limits)!=set(LIMITS):raise ValueError('All quality limit keys are required')
    details=[audit(f,limits=limits,plot=(a.stage!='dataset')) for f in folders if (f/'summary.json').exists()]
    result=dict(limits=limits,episodes=details,passed=all(x['passed'] for x in details) and bool(details),meaning='Expert data screening only; policy failures do not invalidate the evaluation')
    if a.stage=='dataset':
        manifest=[]
        for d in details:
            if d['passed']:
                seed=d['seed'];split='train' if seed in plan['train_seeds'] else 'val'
                assert seed in plan['train_seeds']+plan['offline_val_seeds']
                manifest.append(dict(episode=str(Path(d['path'])/'episode.npz'),seed=seed,split=split,success=d['success'],status=d['status'],quality_pass=True))
        counts={s:sum(x['split']==s for x in manifest) for s in ['train','val']}
        result.update(accepted_counts=counts,attempted=len(details),rejected=[x['seed'] for x in details if not x['passed']])
        result['passed']=counts['train']>=8 and counts['val']>=2
        with (r/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
