#!/usr/bin/env python3
"""CPU-only plots and provenance for a completed or budget-limited smoke run."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--validation-count',type=int,default=5)
p.add_argument('--test-seed',type=int,default=3000)
p.add_argument('--validation-seed',type=int,default=2000)
a=p.parse_args()
r=Path(a.root).resolve()
plots=r/'plots'
plots.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    fig.tight_layout()
    fig.savefig(plots/(name+'.png'))
    fig.savefig(plots/(name+'.pdf'))
    plt.close(fig)

def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))

rows=[json.loads(x) for x in (r/'training/train.jsonl').read_text().splitlines()] if (r/'training/train.jsonl').exists() else []
if rows:
    with (r/'training/train.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in rows))))
        w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].plot([x['step'] for x in rows],[x['loss'] for x in rows],lw=.6,alpha=.65,label='train: raw noise MSE')
    val=[x for x in rows if 'val_loss' in x]
    axs[0].plot([x['step'] for x in val],[x['val_loss'] for x in val],'-o',ms=3,label='fixed validation noise MSE')
    axs[0].set(xlabel='Optimizer step',ylabel='Noise MSE',title='Diffusion loss (not task success)');axs[0].legend(fontsize=8)
    axs[1].plot([x['elapsed_sec']/60 for x in rows],[x['step'] for x in rows])
    axs[1].set(xlabel='Training elapsed (min)',ylabel='Optimizer step',title='Training throughput')
    save(fig,'training_validation')

summary=[]
for category in ['validation','test','data']:
    for path in sorted((r/category).glob('*/summary.json')):
        s=json.loads(path.read_text())
        detail = table(path.parent/'metrics.csv') if (path.parent/'metrics.csv').exists() else []
        diagnostics = dict(mean_clipped_fraction=float(np.mean([float(x['clipped_fraction']) for x in detail])) if detail else None, min_ee_to_handle_m=min([float(x['ee_to_handle_m']) for x in detail],default=None), max_fk_validation_m=max([float(x.get('fk_validation_m',0)) for x in detail],default=None))
        diagnostics.update(ever_open80=s.get('ever_open80',False), legacy_open_80deg_actual20=s.get('legacy_open_80deg_actual20',False), legacy_pass20=s.get('legacy_pass20',False), first_inference_ms=s.get('first_inference_ms'), simulation_seconds_per_wall_second=s.get('sim_time',0)/s['elapsed_sec'] if s.get('elapsed_sec') else None, execution_observe_p50_ms=None, execution_observe_p95_ms=None)
        if (path.parent/'trajectory.npz').exists():
            with np.load(path.parent/'trajectory.npz') as trajectory:
                if 'execution_observe_sec' in trajectory and len(trajectory['execution_observe_sec']):
                    diagnostics.update(execution_observe_p50_ms=float(np.median(trajectory['execution_observe_sec'])*1000), execution_observe_p95_ms=float(np.percentile(trajectory['execution_observe_sec'],95)*1000))
        summary.append(dict(**diagnostics, category=category,episode=path.parent.name,seed=s.get('seed'),mode=s.get('mode'),checkpoint_step=s.get('checkpoint_step'),status=s.get('status'),success=s.get('success',False),open80=s.get('open80',False),reach=s.get('reach',False),grasp=s.get('grasp',False),unlocked=s.get('unlocked',False),signed_angle_deg=s.get('signed_angle_deg'),rear_clearance_m=s.get('rear_clearance_m'),failure_type=s.get('failure_type'),inference_p50_ms=s.get('inference_p50_ms'),inference_p95_ms=s.get('inference_p95_ms'),elapsed_sec=s.get('elapsed_sec')))
if summary:
    with (r/'episode_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (r/'episode_results.json').write_text(json.dumps(summary,indent=2))

def interval(k,n):
    z=1.96;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-h),min(1.,c+h)]

groups=[]
for cat in ['validation','test','data']:
    keys=sorted(set((s['mode'],s['checkpoint_step'] if s['checkpoint_step'] is not None else -1) for s in summary if s['category']==cat))
    for mode,step in keys:
        group=[s for s in summary if s['category']==cat and s['mode']==mode and (s['checkpoint_step'] if s['checkpoint_step'] is not None else -1)==step]
        k=sum(s['success'] for s in group);n=len(group)
        final_angles=[s['signed_angle_deg'] for s in group if s['signed_angle_deg'] is not None]
        groups.append(dict(category=cat,mode=mode,step=step,attempts=n,successes=k,success_rate=k/n,wilson95=interval(k,n),open80_count=sum(s['open80'] for s in group),reach_count=sum(s['reach'] for s in group),grasp_count=sum(s['grasp'] for s in group),unlocked_count=sum(s['unlocked'] for s in group),final_angle_samples=len(final_angles),mean_final_angle_deg=float(np.mean(final_angles)) if final_angles else None))
(r/'aggregate.json').write_text(json.dumps(groups,indent=2))
t=[g for g in groups if g['category']=='test']
if t:
    fig,ax=plt.subplots(figsize=(7,4))
    rates=np.asarray([100*g['success_rate'] for g in t])
    bounds=np.asarray([g['wilson95'] for g in t])*100
    labels=['Expert' if g['mode']=='expert' else 'DP step %d'%g['step'] for g in t]
    ax.bar(labels,rates,color=['#27846b' if g['mode']=='expert' else '#416ea5' for g in t],alpha=.8)
    ax.errorbar(np.arange(len(t)),rates,yerr=np.maximum(0,np.stack([rates-bounds[:,0],bounds[:,1]-rates])),fmt='none',ecolor='black',capsize=5)
    for i,g in enumerate(t):ax.text(i,106,'%d/%d'%(g['successes'],g['attempts']),ha='center')
    complete=all(g['attempts']==30 for g in t) and all(s['status']=='completed' for s in summary if s['category']=='test')
    ax.set(ylabel='Final success (%)',ylim=(-3,115),title=('Independent test' if complete else 'Partial independent test; stopped early')+'; Wilson 95% CI')
    save(fig,'test_comparison')
v=[g for g in groups if g['category']=='validation' and g['attempts']==a.validation_count]
if v:
    fig,ax=plt.subplots(figsize=(7,4))
    rates=np.asarray([100*g['success_rate'] for g in v])
    intervals=np.asarray([[100*g['wilson95'][0],100*g['wilson95'][1]] for g in v])
    ax.errorbar([g['step'] for g in v],rates,yerr=np.stack([rates-intervals[:,0],intervals[:,1]-rates]),fmt='-o',capsize=4,label='Final success + Wilson 95% CI')
    ax.plot([g['step'] for g in v],[100*g['open80_count']/g['attempts'] for g in v],'-s',label='Final open >=80 deg')
    ax.set(xlabel='Checkpoint training step',ylabel='Success (%)',ylim=(-3,103),title='Closed-loop validation, same %d seeds'%a.validation_count)
    ax.legend(fontsize=8);save(fig,'closed_loop_success')

candidates=[r/'adapter-nominal']
candidates+=list(sorted((r/'test').glob('*seed_%d'%a.test_seed)))
if len(candidates)==1:
    expert=r/('diagnostic/expert_seed_%d'%a.validation_seed)
    if expert.exists():candidates.append(expert)
    points=[json.loads(x.read_text()) for x in sorted(r.glob('validation_point_*.json'))]
    if points:
        selected=json.loads((r/'selection.json').read_text()) if (r/'selection.json').exists() else {}
        steps=sorted(set([min(x['step'] for x in points),max(x['step'] for x in points)]+([selected['best_step']] if selected else [])))
        for step in steps:
            path=r/('validation/step_%06d_seed_%d'%(step,a.validation_seed))
            if path.exists() and path not in candidates:candidates.append(path)
        if selected:
            successful=[path.parent for path in sorted((r/'validation').glob('step_%06d*/summary.json'%selected['best_step'])) if json.loads(path.read_text()).get('success')]
            if successful and successful[0] not in candidates:candidates.append(successful[0])
    else:candidates+=list(sorted((r/'validation').glob('*seed_%d'%a.validation_seed)))
fig,axes=plt.subplots(2,2,figsize=(12,8))
for path in candidates:
    if not (path/'metrics.csv').exists():continue
    s=json.loads((path/'summary.json').read_text());m=table(path/'metrics.csv')
    label='%s/%s seed %s'%(s['mode'],s.get('checkpoint_step','script'),s['seed'])
    t=[float(x['sim_time']) for x in m]
    for ax,key,unit in zip(axes.flat,['signed_angle_deg','handle_angle_deg','ee_to_handle_m','rear_clearance_m'],['deg','deg','m','m']):
        ax.plot(t,[float(x[key]) for x in m],label=label,lw=1,color='C%d'%candidates.index(path))
        ax.set(xlabel='Simulation time (s)',ylabel=unit,title=key)
axes[0,0].axhline(80,color='gray',ls='--',lw=.8)
axes[1,1].axhline(.52,color='gray',ls='--',lw=.8)
axes[0,0].legend(fontsize=7)
save(fig,'rollout_metrics')
fig,axes=plt.subplots(2,2,figsize=(12,8))
for path in candidates:
    if not (path/'metrics.csv').exists():continue
    s=json.loads((path/'summary.json').read_text());m=table(path/'metrics.csv')
    label='%s/%s seed %s'%(s['mode'],s.get('checkpoint_step','script'),s['seed'])
    t=[float(x['sim_time']) for x in m]
    for ax,key,unit in zip(axes.flat,['ee_to_handle_m','fk_goal_to_handle_m','fk_ee_tracking_m','joint_tracking_l2_rad'],['m','m','m','rad']):
        if not m or key not in m[0]:continue  # Early baseline predates FK diagnostics.
        ax.plot(t,[float(x[key]) for x in m],label=label,lw=1,color='C%d'%candidates.index(path))
        ax.set(xlabel='Simulation time (s)',ylabel=unit,title=key)
axes[0,0].legend(fontsize=7)
save(fig,'tracking_errors')
for path in candidates:
    if not (path/'trajectory.npz').exists():continue
    z=np.load(path/'trajectory.npz');actions=z['action'];t=np.arange(len(actions))/60
    if not len(actions):continue
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for j in range(3):axs[0].plot(t,actions[:,j],label=['vx (m/s)','vy (m/s)','yaw rate (rad/s)'][j])
    for j in range(6):axs[1].plot(t,actions[:,3+j],label='joint%d'%(j+1))
    axs[2].plot(t,actions[:,-1],label='gripper target (rad)')
    for ax in axs:ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.15)
    axs[1].set_ylabel('Target (rad)');axs[2].set_xlabel('Simulation time (s)')
    fig.suptitle(path.name,fontsize=11)
    save(fig,'actions_'+path.name)
    rates=np.diff(actions,axis=0)*60
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for j in range(3):axs[0].plot(t[1:],rates[:,j],label=['dvx/dt (m/s^2)','dvy/dt (m/s^2)','dwz/dt (rad/s^2)'][j])
    for j in range(6):axs[1].plot(t[1:],rates[:,3+j],label='joint%d'%(j+1))
    axs[2].plot(t[1:],rates[:,-1],label='gripper target rate (rad/s)')
    for ax in axs:ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.15)
    axs[1].set_ylabel('Target rate (rad/s)');axs[2].set_xlabel('Simulation time (s)')
    fig.suptitle(path.name+' applied action rates',fontsize=11)
    save(fig,'action_rates_'+path.name)
    raw=z['raw_action']
    fig,axs=plt.subplots(2,1,figsize=(10,5),sharex=True)
    axs[0].plot(t,np.linalg.norm(raw-actions,axis=1),label='L2(raw - applied), mixed units')
    axs[0].set_ylabel('Clipping magnitude');axs[0].legend(fontsize=8)
    axs[1].plot(t,np.mean(np.abs(raw-actions)>1e-7,axis=1)*100)
    axs[1].set(xlabel='Simulation time (s)',ylabel='Clipped channels (%)',ylim=(-3,103))
    fig.suptitle(path.name+' action constraints',fontsize=11)
    save(fig,'constraints_'+path.name)

# Full-file decode rather than metadata-only video checks.
import cv2
import imageio_ffmpeg
videos=[]
for original in r.rglob('rollout.mp4'):
    h264=original.with_name('rollout_labeled_h264.mp4')
    if not h264.exists():
        # Label real decoded rollout frames; preserve every frame at its recorded 30 Hz.
        cap=cv2.VideoCapture(str(original))
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','640x360','-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(h264)]
        with original.with_name('video-encoding.log').open('w') as log:
            encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
            while True:
                ok,frame=cap.read()
                if not ok:break
                cv2.putText(frame,'run_id='+r.name,(8,346),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                encoder.stdin.write(frame.tobytes())
            cap.release();encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('H264 encoding failed: '+str(original))

for path in r.rglob('rollout_labeled_h264.mp4'):
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if count in [0,expected//2,expected-1]:cv2.imwrite(str(path.parent/('video_check_%04d.png'%count)),frame)
        count+=1
    cap.release()
    videos.append(dict(path=str(path),fps=fps,decoded_frames=count,metadata_frames=expected,duration_sec=count/fps if fps else None,passed=count==1425 and abs(fps-30)<1e-6,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(r/'video_checks.json').write_text(json.dumps(videos,indent=2))
contact_rows=[]
for item in videos:
    directory=Path(item['path']).parent
    if directory.parent.name!='test':continue
    if not item['passed']:continue
    frames=[cv2.resize(cv2.imread(str(directory/('video_check_%04d.png'%i))),(480,270)) for i in (0,712,1424)]
    contact_rows.append(np.hstack(frames))
if contact_rows:cv2.imwrite(str(r/'video_contact_sheet.png'),np.vstack(contact_rows))
print(json.dumps(dict(groups=groups,videos=videos)),flush=True)
