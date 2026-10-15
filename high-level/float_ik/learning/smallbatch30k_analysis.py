"""Diagnostics and postprocessing only; preserve source time, real tails and errors."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'high-level/dp'))


def offline(root,step):
    from isaacgym import gymapi
    import torch
    from door_dp_common import DoorDPPolicyController
    from smallbatch30k import IMAGES,verify,save
    from branch16040_analysis import physical_metrics
    from forward_report import write_csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(2);verify(root)
    plan=json.loads((root/'plan.json').read_text());out=root/'offline'/('step_%06d'%step);out.mkdir(parents=True,exist_ok=False)
    controller=DoorDPPolicyController(root/'training/checkpoints'/('step_%06d.pt'%step),device='cuda:0',num_inference_steps=100,action_horizon=32)
    with np.load(root/'raw/episode_000000.npz',allow_pickle=True) as z:
        states=z['state'].copy();actions=z['action'].copy();images=[z[k].copy() for k in IMAGES]
    windows={w['center']:w for w in plan['windows']};predictions=[];targets=[];meta=[]
    for frame in range(max(windows)+1):
        controller.append_observation(states[frame],*[im[frame] for im in images])
        if frame not in windows:continue
        for seed in plan['diffusion_seeds']:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);controller.sample_action_chunk()
            pred=np.asarray(list(controller.action_queue),dtype=np.float32)
            if pred.shape!=(32,10) or not np.isfinite(pred).all():raise FloatingPointError('Invalid diagnostic prediction')
            predictions.append(pred);targets.append(actions[frame:frame+32]);meta.append(dict(windows[frame],noise_seed=seed))
    predictions=np.asarray(predictions);targets=np.asarray(targets);metrics=[]
    np.savez_compressed(out/'predictions.npz',prediction=predictions,target=targets,
        center=np.array([m['center'] for m in meta]),noise_seed=np.array([m['noise_seed'] for m in meta]))
    for stage in ['all',*dict.fromkeys(m['stage'] for m in meta)]:
        ids=[i for i,m in enumerate(meta) if stage=='all' or m['stage']==stage]
        for length in [16,32]:metrics.append(dict(stage=stage,horizon=length,**physical_metrics(predictions[ids,:length],targets[ids,:length])))
    save(out/'summary.json',dict(optimizer_updates=step,metrics=metrics,windows=plan['windows'],diffusion_seeds=plan['diffusion_seeds'],
        actual_executed_prefix=16,diagnostic_prediction_horizon=32,causal_pre_action_history=True,
        note='Fixed shared-action phase windows from previous validated run; stage/targets are not model inputs. No observation after center fed to its prediction.'))
    write_csv(out/'metrics.csv',metrics)
    selected=[i for i,m in enumerate(meta) if m['noise_seed']==plan['diffusion_seeds'][0]]
    fig,axes=plt.subplots(5,2,figsize=(13,12),constrained_layout=True)
    for ch,ax in enumerate(axes.flat):
        for i in selected:
            t=(meta[i]['center']+np.arange(32))*.02;ax.plot(t,targets[i,:,ch],color='black',alpha=.6);ax.plot(t,predictions[i,:,ch],color='tab:blue',alpha=.7)
        ax.set(xlabel='True source time s',ylabel=['vx m/s','yaw rad/s','EE x m','EE y m','EE z m','qx','qy','qz','qw','gripper rad'][ch]);ax.grid(alpha=.2)
    fig.suptitle('SingleB random batch8; optimizer updates%d; black target, blue prediction'%step)
    fig.savefig(out/'actions_full.png',dpi=125);plt.close(fig);verify(root)


def pair_video(root,folder,out):
    import cv2
    import imageio_ffmpeg
    from stop_audit import inspect_video
    paths=[root/'reference/rollout.mp4',folder/'rollout.mp4'];caps=[cv2.VideoCapture(str(p)) for p in paths]
    counts=[int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    if counts!=[723,1250] or any(abs(c.get(cv2.CAP_PROP_FPS)-25)>1e-5 for c in caps):raise ValueError('Paired source video rate/length mismatch')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1280x360','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i in range(1250):
            panels=[]
            for j,c in enumerate(caps):
                if i<counts[j]:
                    ok,frame=c.read()
                    if not ok:raise ValueError('Paired decode ended early')
                    frame=cv2.resize(frame,(640,360))
                else:
                    frame=np.zeros((360,640,3),dtype=np.uint8)
                    cv2.putText(frame,'REFERENCE ENDED - NO REAL DATA',(10,170),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                cv2.putText(frame,('Pure10D expert' if j==0 else 'Random batch8 policy')+' t=%.2fs'%(i/25),(10,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2);panels.append(frame)
            process.stdin.write(np.concatenate(panels,axis=1).tobytes())
    finally:
        process.stdin.close()
        for c in caps:c.release()
    if process.wait():raise RuntimeError('Paired encoder failed')
    check=inspect_video(out,1250)
    if not check['passed']:raise RuntimeError('Paired complete decode failed')
    return dict(check,reference_source_frames=723,policy_source_frames=1250,reference_ended_black=True)


def closed(root,step):
    from smallbatch30k import verify,save
    from branch16040_analysis import series,compare_metrics,angle_error
    from forward_report import write_csv,rollout_plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verify(root);out=root/'comparison'/('step_%06d'%step);out.mkdir(parents=True,exist_ok=False)
    folder=root/'pilot/evaluation'/('step_%06d'%step)/'seed_16040';model=series(folder);reference=series(root/'reference')
    with np.load(root/'raw/episode_000000.npz',allow_pickle=True) as z:reference['action']=z['action'].copy()
    metrics=[];timing=[];n=min(len(model['t']),len(reference['t']));ids=np.arange(n)
    metrics.append(dict(alignment='true_time',stage='all',**compare_metrics(model,reference,ids,ids)))
    for phase in sorted(set(reference['phase'])):
        mi=np.where(model['phase']==phase)[0];ri=np.where(reference['phase']==phase)[0]
        entry=dict(stage=int(phase),reference_start_sec=float(reference['t'][ri[0]]),reference_duration_sec=len(ri)*.02,model_missing_stage=not bool(len(mi)))
        if not len(mi):timing.append(entry);continue
        entry.update(model_start_sec=float(model['t'][mi[0]]),model_duration_sec=len(mi)*.02,stage_start_delay_sec=float(model['t'][mi[0]]-reference['t'][ri[0]]))
        timing.append(entry);length=max(len(mi),len(ri));a=mi[np.rint(np.linspace(0,len(mi)-1,length)).astype(int)];b=ri[np.rint(np.linspace(0,len(ri)-1,length)).astype(int)]
        metrics.append(dict(entry,alignment='stage_progress',**compare_metrics(model,reference,a,b)))
    position_error=np.linalg.norm(model['actual_post']-model['action'][:,2:5],axis=-1);orient=angle_error(model['orientation_post'],model['action'][:,5:9])
    full=dict(real_frames=len(model['t']),own_post_tracking_rmse_m=float(np.sqrt(np.mean(position_error**2))),
        own_post_tracking_p95_m=float(np.percentile(position_error,95)),own_post_tracking_max_m=float(position_error.max()),
        own_post_orientation_rmse_deg=float(np.sqrt(np.mean(orient**2))),own_post_orientation_p95_deg=float(np.percentile(orient,95)),
        total_base_path_m=float(np.sum(model['measured_speed'])*json.loads((folder/'config.json').read_text())['dt']),
        reconstruction_vs_next_log_max_m=model['reconstruction_error_m'])
    result=json.loads((folder/'result.json').read_text())
    save(out/'summary.json',dict(optimizer_updates=step,metrics=metrics,stage_timing=timing,full_record_tracking=full,
        result=result,reference_real_frames=1446,reference_extended=False,
        no_new_imitation_pass_threshold=True,definitions='True-time error uses real common prefix only. Full record tracking compares same-step target and actual post-step EE. Phase timing uses observed task phase; no missing phase interpolation.'))
    write_csv(out/'metrics.csv',metrics);write_csv(out/'stage_timing.csv',timing)
    def curves(data,ids):
        a=data['action'][ids];p=data['actual_post'][ids];base=data['base'][ids]
        return [a[:,0],a[:,1],data['measured_speed'][ids],*[a[:,j] for j in range(2,5)],*[p[:,j] for j in range(3)],
            np.linalg.norm(p-a[:,2:5],axis=-1),angle_error(data['orientation_post'][ids],a[:,5:9]),a[:,9],base[:,0],base[:,1],
            np.cumsum(data['measured_speed'][ids])*.02]
    labels=['command vx m/s','command yaw rad/s','measured base speed m/s','target EE x m','target EE y m','target EE z m',
        'actual post EE x m','actual post EE y m','actual post EE z m','post target tracking m','post orientation tracking deg','gripper command rad','base X m','base Y m','cumulative path m']
    fig,axes=plt.subplots(5,3,figsize=(15,16),constrained_layout=True)
    for label,d in [('Pure10D reference',reference),('Policy%d'%step,model)]:
        idx=np.arange(len(d['t']))
        for ax,y,title in zip(axes.flat,curves(d,idx),labels):ax.plot(d['t'],y,label=label);ax.set(xlabel='True simulation time s',ylabel=title);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.savefig(out/'true_time_full.png',dpi=125);plt.close(fig)
    for phase in sorted(set(reference['phase'])):
        fig,axes=plt.subplots(4,3,figsize=(13,11),constrained_layout=True)
        for label,d in [('Pure10D reference',reference),('Policy',model)]:
            idx=np.where(d['phase']==phase)[0]
            if not len(idx):continue
            for ax,y,title in zip(axes.flat,curves(d,idx)[:12],labels[:12]):
                ax.plot(np.linspace(0,1,len(idx)),y,label=label+' %.2fs'%(len(idx)*.02));ax.set(xlabel='Task phase progress',ylabel=title);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=6);fig.suptitle('Phase%d; delays/missing phases in stage_timing.csv'%phase)
        fig.savefig(out/('phase_%d.png'%phase),dpi=110);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
    for label,d in [('Pure10D reference',reference),('Policy',model)]:ax.plot(d['base'][:,0],d['base'][:,1],label=label)
    ax.set(xlabel='World X m',ylabel='World Y m');ax.axis('equal');ax.legend();ax.grid(alpha=.2);fig.savefig(out/'base_path.png',dpi=125);plt.close(fig)
    save(out/'video_verification.json',pair_video(root,folder,out/'expert_policy.mp4'))
    rollout_plots(folder);verify(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--step',type=int,required=True);group=p.add_mutually_exclusive_group(required=True);group.add_argument('--offline',action='store_true');group.add_argument('--closed',action='store_true');a=p.parse_args()
    (offline if a.offline else closed)(Path(a.root).resolve(),a.step)
