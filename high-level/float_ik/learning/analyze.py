#!/usr/bin/env python3
"""CPU-only Cartesian run tables, plots and explicit paired-video comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifacts import write_new_json
from dataset_contract import layout


def rows(path):
    with path.open() as f: return list(csv.DictReader(f))


def save(fig, output, name):
    fig.tight_layout()
    for suffix in ('png','pdf'): fig.savefig(output/(name+'.'+suffix),dpi=140)
    plt.close(fig)


def episode_plot(folder, output, name):
    m=rows(folder/'metrics.csv')
    with np.load(folder/'trajectory.npz') as z:
        indices=layout(z['action_names'].tolist());action=z['action'].astype(float);raw=z['raw_action'].astype(float)
        joints=z['joint_position_target'].astype(float);actual_q=z['actual_joint_position'].astype(float)
        ik=z['ik_joint_position_target'].astype(float) if 'ik_joint_position_target' in z else None
    if len(m)!=len(action): raise ValueError('Metrics/action length mismatch: '+str(folder))
    if not m: return
    t=np.array([float(x['sim_time']) for x in m]);col=lambda key:np.array([float(x[key]) for x in m])
    fig,axs=plt.subplots(3,2,figsize=(13,10))
    for j,key in enumerate('xyz'): axs[0,0].plot(t,action[:,indices['position'][j]],label=key+' command')
    if 'ee_actual_x' in m[0]:
        for key in 'xyz':axs[0,0].plot(t,col('ee_actual_'+key),'--',label=key+' actual')
    axs[0,0].set(ylabel='World EE position (m)')
    for key in ['ee_command_pos_error_m','fk_ee_tracking_m']:axs[0,1].plot(t,col(key),label=key)
    axs[0,1].set(ylabel='Position error (m)')
    for j in range(6):
        axs[1,0].plot(t,joints[:,j],label='q%d target'%(j+1))
        axs[1,0].plot(t,actual_q[:,j],'--',alpha=.6)
    if ik is not None:
        for j in range(6):axs[1,0].plot(t,ik[:,j],':',alpha=.4)
    axs[1,0].set(ylabel='Servo target / dashed actual / dotted raw IK (rad)')
    axs[1,1].plot(t,col('ee_command_rot_error_rad')*180/np.pi,label='Orientation tracking')
    axs[1,1].set(ylabel='Degrees')
    axs[2,0].plot(t,col('base_x'),label='base x');axs[2,0].plot(t,col('base_y'),label='base y')
    axs[2,0].set(ylabel='World position (m)')
    axs[2,1].plot(t,col('signed_angle_deg'),label='Door angle');axs[2,1].set(ylabel='Degrees')
    for ax in axs.flat:ax.legend(fontsize=6,ncol=2);ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2)
    fig.suptitle(name);save(fig,output,'trajectory_'+name)
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for j,label in enumerate(['vx m/s','yaw rate rad/s'] if action.shape[1]==10 else ['vx m/s','vy m/s','yaw rate rad/s']):axs[0].plot(t,action[:,j],label=label)
    if len(t)>1:
        axs[1].plot(t[1:],np.linalg.norm(np.diff(action[:,indices['position']],axis=0),axis=1)*1000,label='EE target displacement mm/step')
        axs[2].plot(t[1:],np.max(np.abs(np.diff(joints[:,:6],axis=0)),axis=1),label='Max arm target jump rad/step')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    axs[-1].set_xlabel('Simulation time (s)');save(fig,output,'command_changes_'+name)


def compare_pair(expert, policy, output, index, video=False):
    import cv2
    summaries=[json.loads((p/'summary.json').read_text()) for p in (expert,policy)]
    if summaries[0]['mode']!='expert' or summaries[1]['mode']!='policy':raise ValueError('--pair expects expert then policy')
    profiles=[s.get('execution_profile','default') for s in summaries]
    if profiles[0]!=profiles[1]:raise ValueError('Cannot present different execution profiles as a paired comparison')
    if summaries[0].get('execution_wrapper_sha256')!=summaries[1].get('execution_wrapper_sha256') or summaries[0].get('source_hashes')!=summaries[1].get('source_hashes'):
        raise ValueError('Paired executor source versions differ')
    with np.load(expert/'episode.npz') as e, np.load(policy/'episode.npz') as p:
        same=bool(np.array_equal(e['state'][0],p['state'][0]) and np.array_equal(e['images'][0],p['images'][0]))
    if not same:raise ValueError('Paired initial observations differ')
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for folder,label in [(expert,'Expert'),(policy,'DP step %s'%summaries[1].get('checkpoint_step'))]:
        m=rows(folder/'metrics.csv');t=[float(x['sim_time']) for x in m]
        for ax,key in zip(axs.flat,['ee_command_pos_error_m','ee_command_rot_error_rad','signed_angle_deg','rear_clearance_m']):
            ax.plot(t,[float(x[key]) for x in m],label=label);ax.set(title=key,xlabel='Simulation time (s)');ax.legend();ax.grid(alpha=.2)
    name='comparison_%02d'%index;fig.suptitle('Seed %s | profile %s'%(summaries[0]['seed'],profiles[0]));save(fig,output,name)
    result=dict(expert=str(expert),policy=str(policy),initial_state_images_equal=same,execution_profile=profiles[0],statuses=[s['status'] for s in summaries],successes=[s.get('success',False) for s in summaries],reach=[s.get('reach',False) for s in summaries],grasp=[s.get('grasp',False) for s in summaries])
    if not video:return result
    import imageio_ffmpeg
    paths=[p/'rollout.mp4' for p in (expert,policy)]
    # Decode both complete files first; the common-prefix output is explicitly labelled.
    counts=[]; rates=[]
    for path in paths:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
        cap=cv2.VideoCapture(str(path));rates.append(cap.get(cv2.CAP_PROP_FPS));count=0
        while cap.read()[0]:count+=1
        cap.release();counts.append(count)
    if min(counts)<=0 or rates[0]<=0 or abs(rates[0]-rates[1])>1e-6:raise ValueError('Empty videos or mismatched fps')
    caps=[cv2.VideoCapture(str(p)) for p in paths];frames=min(counts);fps=rates[0];out=output/(name+'.mp4')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1280x360','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i in range(frames):
            read=[cap.read() for cap in caps]
            if not all(ok for ok,_ in read):raise RuntimeError('Decode ended before counted frame')
            frame=np.hstack([cv2.resize(f,(640,360)) for _,f in read])
            label='DP step %s | %s'%(summaries[1].get('checkpoint_step'),profiles[0])
            for x,text in [(8,'EXPERT | '+profiles[0]),(648,label)]:
                cv2.rectangle(frame,(x,328),(x+625,359),(0,0,0),-1);cv2.putText(frame,text,(x+2,348),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
            if counts[0]!=counts[1]:cv2.putText(frame,'COMMON PREFIX; one rollout ended early',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            proc.stdin.write(frame.tobytes())
            if i in {0,frames//2,frames-1}:cv2.imwrite(str(output/(name+'_frame_%04d.png'%i)),frame)
    finally:
        for cap in caps:cap.release()
        proc.stdin.close();rc=proc.wait()
    if rc:raise RuntimeError('Video encoding failed')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-i',str(out),'-f','null','-'],check=True)
    cap=cv2.VideoCapture(str(out));decoded=0
    while cap.read()[0]:decoded+=1
    cap.release()
    if decoded!=frames:raise ValueError('Paired output frame count mismatch')
    result.update(video=str(out),source_frames=counts,output_frames=decoded,fps=fps,common_prefix_only=counts[0]!=counts[1],sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--episode',action='append',default=[],help='Plot a run-relative episode; repeat. Default: first recorded attempt in each category/profile/mode.')
    p.add_argument('--pair',nargs=2,action='append',default=[],metavar=('EXPERT','POLICY'));p.add_argument('--videos',action='store_true')
    a=p.parse_args();r=Path(a.root).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    records=[];default_plots={}
    for path in sorted(r.rglob('summary.json')):
        s=json.loads(path.read_text())
        if 'mode' not in s or 'steps_completed' not in s:continue
        rel=path.parent.relative_to(r);category=rel.parts[0];profile=s.get('execution_profile','default')
        rec=dict(episode=str(rel),category=category,execution_profile=profile,mode=s.get('mode'),seed=s.get('seed'),step=s.get('checkpoint_step'),status=s.get('status'),steps_completed=s.get('steps_completed'),success=bool(s.get('success',False) and s.get('status')=='completed'),reach=s.get('reach',False),grasp=s.get('grasp',False),signed_angle_deg=s.get('signed_angle_deg'),failure_type=s.get('failure_type'))
        records.append(rec)
        if (path.parent/'trajectory.npz').exists() and (path.parent/'metrics.csv').exists():default_plots.setdefault((category,profile,rec['mode']),rel)
    if not records:raise ValueError('No Cartesian attempts found')
    write_new_json(out/'episode_results.json',records)
    with (out/'episode_results.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    groups={}
    for row in records:groups.setdefault((row['category'],row['execution_profile'],row['mode'],row['step']),[]).append(row)
    aggregate=[]
    for key,items in groups.items():
        n=len(items);k=sum(x['success'] for x in items);rate=k/n;z=1.96;den=1+z*z/n;center=(rate+z*z/(2*n))/den;half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
        aggregate.append(dict(category=key[0],execution_profile=key[1],mode=key[2],step=key[3],attempts=n,completed=sum(x['status']=='completed' for x in items),errors=sum(x['status']=='error' for x in items),successes=k,reach=sum(x['reach'] for x in items),grasp=sum(x['grasp'] for x in items),success_rate=rate,wilson95=[max(0,center-half),min(1,center+half)]))
    write_new_json(out/'aggregate.json',aggregate)
    log=r/'training/train.jsonl'
    if log.exists():
        train=[json.loads(x) for x in log.read_text().splitlines()];val=[x for x in train if 'val_loss' in x]
        fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].plot([x['step'] for x in train],[x['loss'] for x in train],alpha=.5,label='train')
        axs[0].plot([x['step'] for x in val],[x['val_loss'] for x in val],label='offline val');axs[0].set(xlabel='Optimizer step',ylabel='Noise MSE (not task success)');axs[0].legend()
        axs[1].plot([x['elapsed_sec'] for x in train],[x['step'] for x in train]);axs[1].set(xlabel='Training seconds',ylabel='Optimizer step');save(fig,out,'training')
    for i,rel in enumerate(a.episode or default_plots.values()):episode_plot(r/rel,out,'%02d_%s'%(i,Path(rel).name))
    pairs=[compare_pair(r/e,r/p,out,i,a.videos) for i,(e,p) in enumerate(a.pair)]
    write_new_json(out/'analysis.json',dict(schemas=sorted({json.loads(p.read_text())['schema'] for p in r.rglob('summary.json') if json.loads(p.read_text()).get('schema')}),root=str(r),attempts=len(records),pairs=pairs,meaning='Recorded task results, trajectories and paired videos; groups retain execution versions.'))
    print(json.dumps(dict(attempts=len(records),groups=len(aggregate),pairs=len(pairs),output=str(out))))


if __name__=='__main__':main()
