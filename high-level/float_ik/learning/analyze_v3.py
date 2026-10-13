"""Plots and geometric audits for source whole-body floating-backend records."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from kinematics import FK


def analyze(folder):
    folder = Path(folder)
    rows = [json.loads(s) for s in (folder/'execution.jsonl').read_text().splitlines()]
    t = np.array([r['sim_time'] for r in rows])
    actual = np.array([r['ee']['actual_pos_world'] for r in rows])
    target = np.array([r['ee']['target_pos_world'] for r in rows])
    qt = np.array([r['ee']['target_quat'] for r in rows])
    qa = np.array([r['ee']['actual_quat'] for r in rows])
    dots = np.sum(qt*qa, axis=1)/(np.linalg.norm(qt,axis=1)*np.linalg.norm(qa,axis=1))
    orientation = 2*np.arccos(np.clip(np.abs(dots),0,1))
    angles = -np.rad2deg([r['door_after'][0] for r in rows])
    q = np.array([r['q_after'] for r in rows])
    servo = np.array([r['servo_target'] for r in rows])
    base = np.array([r['base_after'] for r in rows])
    phases = np.array([r['phase_id'] for r in rows])
    errors = np.linalg.norm(target-actual,axis=1)
    fig, axs = plt.subplots(3,2,figsize=(13,10), constrained_layout=True)
    dt=float(t[1]-t[0]) if len(t)>1 else .02
    axs[0,0].plot(t+dt,angles,label='signed door angle (after step)')
    axs[0,0].plot(t,np.rad2deg([r['door_after'][1] for r in rows]),label='handle angle',alpha=.7)
    axs[0,0].axhline(80,color='red',linestyle='--',label='source pass threshold')
    axs[0,0].set(ylabel='Degrees')
    axs[0,1].plot(base[:,0],base[:,1],label='base XY')
    cfg=json.loads((folder/'config.json').read_text())['args']
    axs[0,1].scatter([cfg['door_x']],[cfg['door_y']],label='door root / navigation reference')
    if any(r['extra']['pass_started'] for r in rows):
        through=next(r['extra']['through_target'] for r in rows if r['extra']['pass_started'])
        axs[0,1].scatter(*through,label='source through target')
    axs[0,1].set(xlabel='World X (m)',ylabel='World Y (m)')
    axs[0,1].set_aspect('equal',adjustable='datalim')
    for i,name in enumerate('xyz'):
        axs[1,0].plot(t,target[:,i],label=name+' target')
        axs[1,0].plot(t,actual[:,i],'--',label=name+' actual')
    axs[1,0].set(ylabel='World EE position (m)')
    axs[1,1].plot(t,errors,label='position norm (m)')
    axs[1,1].plot(t,orientation,label='orientation angle (rad)',alpha=.7)
    axs[1,1].set(ylabel='EE tracking diagnostics')
    axs[2,0].plot(t,[r['applied_action'][0] for r in rows],label='body vx command')
    axs[2,0].plot(t,[r['base']['lin_vel'][0] for r in rows],'--',label='achieved body vx')
    axs[2,0].plot(t,[r['applied_action'][1] for r in rows],label='yaw rate command')
    axs[2,0].set(ylabel='m/s or rad/s')
    axs[2,1].step(t,phases,where='post',label='source phase')
    axs[2,1].plot(t,q[:,-1],label='gripper actual')
    axs[2,1].plot(t,servo[:,-1],'--',label='gripper target')
    contact_available=all(r['extra'].get('contact_pairs_available',False) for r in rows)
    if contact_available:
        axs[2,1].step(t,[r.get('finger_handle_contact',False) for r in rows],where='post',label='force-bearing finger/handle contact')
    for ax in axs.flat:
        ax.grid(alpha=.2);ax.legend(fontsize=7)
        if ax is not axs[0,1]:ax.set_xlabel('Simulation time (s)')
    fig.suptitle(folder.name)
    fig.savefig(folder/'task_diagnostics.png',dpi=150);plt.close(fig)
    fig, axs=plt.subplots(7,1,figsize=(12,13),sharex=True,constrained_layout=True)
    for i,ax in enumerate(axs):
        ax.plot(t,servo[:,i],label='servo target');ax.plot(t+dt,q[:,i],'--',label='actual after step')
        if i<6:
            raw_ik=np.array([r['q_before'][i]+r['ik_delta'][i] for r in rows])
            ax.plot(t,raw_ik,':',alpha=.6,label='IK target before source freeze / limits')
        ax.set_ylabel('q%d (rad)'%(i+1));ax.grid(alpha=.2);ax.legend(fontsize=7)
    axs[-1].set_xlabel('Simulation time (s)');fig.savefig(folder/'joints.png',dpi=150);plt.close(fig)
    raw=np.array([r['dp_action_raw'] for r in rows]);applied=np.array([r['applied_action'] for r in rows])
    labels=['vx (m/s)','yaw rate (rad/s)','EE x (m)','EE y (m)','EE z (m)','qx','qy','qz','qw','gripper (rad)']
    replans=t[[bool(r['extra'].get('dp_replanned',False)) for r in rows]]
    fig,axs=plt.subplots(5,2,figsize=(13,12),sharex=True,constrained_layout=True)
    for i,ax in enumerate(axs.flat):
        ax.plot(t,raw[:,i],label='raw action');ax.plot(t,applied[:,i],'--',label='applied')
        low,high=ax.get_ylim()
        if len(replans):ax.vlines(replans,low,high,color='gray',alpha=.15,linewidth=.4)
        ax.set_ylabel(labels[i]);ax.grid(alpha=.2);ax.legend(fontsize=6)
    axs[-1,0].set_xlabel('Simulation time (s)');axs[-1,1].set_xlabel('Simulation time (s)')
    fig.savefig(folder/'actions_10d.png',dpi=150);plt.close(fig)
    with (folder/'metrics.csv').open('w') as f:
        out=csv.writer(f);out.writerow(['frame','sim_time','signed_angle_deg','ee_position_error_m','ee_orientation_error_rad','base_x','base_y','phase_id','pass_started','pass_done','dp_action_applied'])
        for i,r in enumerate(rows):out.writerow([r['step'],t[i],angles[i],errors[i],orientation[i],*base[i,:2],phases[i],r['extra']['pass_started'],r['extra']['pass_done'],r['extra']['dp_action_applied']])
    repo=Path(__file__).resolve().parents[3]
    urdf=folder/'assets/z1_arm_visual_flip.urdf'
    if not urdf.exists():urdf=repo/'low-level/resources/robots/b1z1/urdf/b1z1_basearn.urdf'
    names=['joint%d'%i for i in range(1,7)]+['jointGripper']
    fk=FK(urdf,'ee_gripper_link',names)
    fk_errors=[];jac_errors=[]
    for r in rows[1::50]:
        state=np.array(r['q_before']);pose=fk.position(state)
        # Use measured world EE quaternion and URDF-relative orientation to
        # recover the root rotation, also covering navigation turns.
        _,local_quat=fk.pose(state)
        rot=(Rotation.from_quat(r['root_before'][3:7]) if 'root_before' in r
             else Rotation.from_quat(r['ee']['actual_quat'])*Rotation.from_quat(local_quat).inv())
        world=rot.apply(pose)+np.array(r['base']['xy']+[r['base']['height']])
        fk_errors.append(np.linalg.norm(world-r['ee']['actual_pos_world']))
        jac=np.stack([(fk.position(state+np.eye(7)[j]*1e-5)-pose)/1e-5 for j in range(6)],axis=1)
        jac_errors.append(np.linalg.norm(rot.as_matrix()@jac-np.array(r['jacobian_world'])[:3]))
    result=dict(frames=len(rows),max_signed_angle_deg=float(max(angles)),
                position_error_p95_m=float(np.percentile(errors,95)),orientation_error_p95_rad=float(np.percentile(orientation,95)),
                fk_error_max_m=float(max(fk_errors)),jacobian_position_error_max=float(max(jac_errors)),
                contact_pairs_available=contact_available,
                finger_handle_contact_frames=(sum(r.get('finger_handle_contact',False) for r in rows) if contact_available else None))
    (folder/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(folder=str(folder),**result)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folders',nargs='+');args=parser.parse_args()
    for folder in args.folders:analyze(folder)
