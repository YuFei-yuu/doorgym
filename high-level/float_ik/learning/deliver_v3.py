"""Task tables, source training curves and a complete paired rollout video."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def paired_video(expert,policy,output,updates=5000):
    import cv2
    import imageio_ffmpeg
    cv2.setNumThreads(2)
    folders=[expert,policy]
    configs=[json.loads((p/'config.json').read_text()) for p in folders]
    summaries=[json.loads((p/'summary.json').read_text()) for p in folders]
    if configs[0]['source_hashes']!=configs[1]['source_hashes'] or summaries[0]['seed']!=summaries[1]['seed']:
        raise ValueError('Paired rollouts must share execution source and seed')
    with np.load(expert/'trajectory.npz') as a,np.load(policy/'trajectory.npz') as b:
        keys=['state','wrist_handle_mask','wrist_masked_depth','front_handle_mask','front_masked_depth']
        if not all(np.array_equal(a[k][0],b[k][0]) for k in keys):raise ValueError('Paired initial observations differ')
    videos=[p/'rollout.mp4' for p in folders]
    readers=[imageio_ffmpeg.read_frames(str(p),input_params=['-threads','2']) for p in videos]
    meta=[next(r) for r in readers]
    if any(abs(m['fps']-25)>1e-5 for m in meta):raise ValueError('Expected source 25fps video sampling')
    counts=[s['frames']//2 for s in summaries]
    total=max(counts);last=[None,None]
    seed=summaries[0]['seed']
    destination=output/('comparison_seed_%d.mp4'%seed)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
         '-s','1200x400','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2',
         '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for frame in range(total):
            panels=[]
            for j,(reader,m) in enumerate(zip(readers,meta)):
                if frame<counts[j]:
                    raw=next(reader);w,h=m['size'];last[j]=np.frombuffer(raw,np.uint8).reshape(h,w,3).copy()
                panel=cv2.resize(last[j],(600,400))
                label=('EXPERT' if j==0 else 'DP - %d updates'%updates)+' | seed %d'%seed
                cv2.rectangle(panel,(0,372),(599,399),(0,0,0),-1)
                cv2.putText(panel,label,(8,391),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,0),1)
                if frame>=counts[j]:
                    cv2.putText(panel,'EPISODE ENDED - FINAL FRAME HELD',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,0),1)
                panels.append(panel)
            joined=np.hstack(panels);proc.stdin.write(joined.tobytes())
            if frame in {0,total//2,total-1}:cv2.imwrite(str(output/('pair_frame_%04d.png'%frame)),joined[:,:,::-1])
    finally:
        for reader in readers:reader.close()
        proc.stdin.close();code=proc.wait()
    if code:raise RuntimeError('Paired video encoding failed')
    for path in videos+[destination]:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-threads','2','-i',str(path),'-f','null','-'],check=True)
    return dict(video=str(destination),source_frames=counts,output_frames=total,shorter_episode_final_frame_held=counts[0]!=counts[1],
                identical_initial_observations=True,sha256=hashlib.sha256(destination.read_bytes()).hexdigest())


def main(root):
    root=Path(root).resolve();out=root/'analysis';out.mkdir(exist_ok=True)
    rows=[]
    for path in sorted((root/'evaluation').glob('*/summary.json')):
        s=json.loads(path.read_text());s['episode']=str(path.parent.relative_to(root));s['group']=path.parent.name.split('_')[0]
        log=[json.loads(x) for x in (path.parent/'execution.jsonl').read_text().splitlines()]
        latency=[r['extra']['inference_sec']*1000 for r in log if r['extra'].get('dp_replanned') and r['extra'].get('inference_sec') is not None]
        s['replan_latency_p50_ms']=float(np.percentile(latency,50)) if latency else None
        s['replan_latency_p95_ms']=float(np.percentile(latency,95)) if latency else None
        rows.append(s)
    (out/'episodes.json').write_text(json.dumps(rows,indent=2))
    fields=['episode','group','seed','mode','frames','opened','pass_done','success','max_signed_door_deg','final_signed_door_deg',
            'through_distance','first_open_frame','first_pass_frame','dp_frames','script_frames','invalid_image_frames',
            'replan_latency_p50_ms','replan_latency_p95_ms','end_reason']
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    aggregate={}
    for group in ['validation','test']:
        items=[s for s in rows if s['group']==group]
        aggregate[group]=dict(attempts=len(items),opened=sum(s['opened'] for s in items),
                              completed=sum(s['success'] for s in items),seeds=[s['seed'] for s in items])
    (out/'task_results.json').write_text(json.dumps(aggregate,indent=2))
    train=[json.loads(x) for x in (root/'training/metrics.jsonl').read_text().splitlines()]
    with (root/'training/metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(train[0]));w.writeheader();w.writerows(train)
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    steps=[r['train/step'] for r in train]
    for ax,key,label in zip(axs,['train/loss','train/lr','train/elapsed_sec'],['Source epsilon noise MSE','Learning rate','Training elapsed seconds']):
        ax.plot(steps,[r[key] for r in train]);ax.set(xlabel='Optimizer update',ylabel=label);ax.grid(alpha=.2)
    fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    with np.load(root/'evaluation/offline_predictions.npz') as z:
        pred=z['prediction'];truth=z['expert'];rotation=z['orientation_error_rad']
        names=z['action_names'].tolist();position=[names.index('ee_'+x) for x in 'xyz']
        rmse=np.sqrt(np.mean((pred-truth)**2,axis=(0,1)))
        position_error=np.linalg.norm((pred-truth)[:,:,position],axis=-1).ravel()
        fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        axs[0].bar(names,rmse);axs[0].tick_params(axis='x',rotation=65);axs[0].set(ylabel="Raw action RMSE (each channel's native unit)")
        axs[1].hist(position_error,bins=40);axs[1].set(xlabel='World EE action error norm (m)',ylabel='Executed-action samples')
        finite_rotation=np.sort(np.rad2deg(rotation[np.isfinite(rotation)]))
        axs[2].plot(finite_rotation,np.arange(1,len(finite_rotation)+1)/max(1,len(finite_rotation)))
        axs[2].set(xlabel='Normalized quaternion angle error (degrees)',ylabel='Cumulative fraction')
        fig.savefig(out/'offline_errors.png',dpi=160);plt.close(fig)
        window=len(pred)//2;fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True)
        for ax,j,label in zip(axs,position,'XYZ'):
            ax.plot(truth[window,:,j],label='expert');ax.plot(pred[window,:,j],'--',label='DP prediction')
            ax.set_ylabel('World '+label+' (m)');ax.legend();ax.grid(alpha=.2)
        axs[-1].set_xlabel('Action offset in the executed 16-step chunk');fig.savefig(out/'offline_action_window.png',dpi=160);plt.close(fig)
    full=[s for s in rows if s['group'] in ['validation','test']]
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    ax.bar([str(s['seed']) for s in full],[s['max_signed_door_deg'] for s in full]);ax.axhline(80,color='red',linestyle='--',label='Source 80-degree threshold')
    ax.set(xlabel='Held-out rollout seed',ylabel='Maximum signed door angle (degrees)');ax.legend();fig.savefig(out/'task_angles.png',dpi=160);plt.close(fig)
    pair=paired_video(root/'evaluation/expert_14400',root/'evaluation/test_14400',out)
    (out/'paired_video.json').write_text(json.dumps(pair,indent=2))
    checkpoint=root/'training/frozen_step_005000.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    lines=['# Floating-base source DP: 5000-update result','',
           'The source whole-body task uses the source floating pose API, CPU tensor pipeline and GPU PhysX.',
           'Source task/navigation -> DP overwrite -> floating execution order and snap_arm_to_zero are retained.',
           'Training uses the original uncached LeRobot sequence reader, source network, noise MSE, AdamW and 100-step DDPM.',
           '',f'Checkpoint SHA256: `{digest}`','',
           '| Set | Opened >=80 degrees | Completed original pass flow | Attempts |','|---|---:|---:|---:|']
    for name,r in aggregate.items():lines.append('| %s | %d | %d | %d |'%(name,r['opened'],r['completed'],r['attempts']))
    lines += ['', 'Success is latched when the original task has opened the door in the correct direction and reached its through target.',
              'The through target side is latched at first 80 degrees, as in the source; final target distance is reported separately.',
              'All attempted full rollouts remain in the denominator. Short rollouts are interface diagnostics.',
              '',f'Last logged source training noise MSE: {train[-1]["train/loss"]:.6f}.',
              '', 'Artifacts: training_curves.png, task_angles.png, episodes.csv, paired_video.json and comparison_seed_14400.mp4.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(aggregate=aggregate,paired_video=pair,checkpoint_sha256=digest)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
