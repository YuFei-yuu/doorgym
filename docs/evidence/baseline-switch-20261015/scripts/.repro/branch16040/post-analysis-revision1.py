"""Fixed causal full-action cross evaluation and separately reported trajectory quality."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'high-level/dp'))


def angle_error(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    an=np.linalg.norm(a,axis=-1);bn=np.linalg.norm(b,axis=-1)
    if np.any(an<1e-6) or np.any(bn<1e-6):raise ValueError('Degenerate orientation cannot be scored as valid')
    dot=np.sum(a*b,axis=-1)/(an*bn)
    return np.rad2deg(2*np.arccos(np.clip(np.abs(dot),0,1)))


def physical_metrics(pred,target):
    diff=np.asarray(pred,dtype=np.float64)-target
    return dict(vx_rmse_m_s=float(np.sqrt(np.mean(diff[...,0]**2))),
        yaw_rmse_rad_s=float(np.sqrt(np.mean(diff[...,1]**2))),
        ee_position_rmse_m=float(np.sqrt(np.mean(np.sum(diff[...,2:5]**2,axis=-1)))),
        ee_xyz_rmse_m=np.sqrt(np.mean(diff[...,2:5]**2,axis=tuple(range(diff.ndim-1)))).tolist(),
        ee_orientation_rmse_deg=float(np.sqrt(np.mean(angle_error(pred[...,5:9],target[...,5:9])**2))),
        gripper_rmse_rad=float(np.sqrt(np.mean(diff[...,9]**2))),
        channel_rmse=np.sqrt(np.mean(diff**2,axis=tuple(range(diff.ndim-1)))).tolist(),
        draws=int(pred.shape[0]))


def offline(root,step):
    from isaacgym import gymapi
    import torch
    from door_dp_common import DoorDPPolicyController
    from branch16040 import SOURCE,IMAGES,verify,save
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(2);verify(root)
    plan=json.loads((root/'plan.json').read_text());out=root/'offline'/('step_%06d'%step);out.mkdir(parents=True,exist_ok=False)
    all_metrics=[]
    labels=['A'] if step==0 else ['A','B']
    for model_label in labels:
        cp=SOURCE if step==0 else root/model_label/'training/checkpoints'/('step_%06d.pt'%step)
        # Diagnostic only: expose all 32 predicted steps. Deployment still executes 16.
        controller=DoorDPPolicyController(cp,device='cuda:0',num_inference_steps=100,action_horizon=32)
        for obs_label in ['A','B']:
            with np.load(root/obs_label/'raw/episode_000000.npz',allow_pickle=True) as z:
                states=z['state'].copy();actions=z['action'].copy();images=[z[k].copy() for k in IMAGES]
            controller.obs_buffer.clear();controller.action_queue.clear()
            windows={w['center']:w for w in plan['windows']};preds=[];targets=[];meta=[]
            for frame in range(max(windows)+1):
                controller.append_observation(states[frame],*[im[frame] for im in images])
                if frame not in windows:continue
                for seed in plan['diffusion_seeds']:
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    controller.sample_action_chunk()
                    p=np.asarray(list(controller.action_queue),dtype=np.float32)
                    if p.shape!=(32,10) or not np.isfinite(p).all():raise ValueError('Invalid full action prediction')
                    preds.append(p);targets.append(actions[frame:frame+32]);meta.append(dict(windows[frame],noise_seed=seed))
            preds=np.asarray(preds);targets=np.asarray(targets)
            np.savez_compressed(out/('%s_on_%s.npz'%(model_label,obs_label)),prediction=preds,target=targets,
                center=np.array([m['center'] for m in meta]),noise_seed=np.array([m['noise_seed'] for m in meta]))
            for stage in ['all',*dict.fromkeys(m['stage'] for m in meta)]:
                idx=[i for i,m in enumerate(meta) if stage=='all' or m['stage']==stage]
                for horizon in [16,32]:
                    all_metrics.append(dict(model=model_label,observation=obs_label,stage=stage,horizon=horizon,
                        **physical_metrics(preds[idx,:horizon],targets[idx,:horizon])))
            # Each stage keeps its actual source time on these selected queue curves.
            selected=[i for i,m in enumerate(meta) if m['noise_seed']==plan['diffusion_seeds'][0]]
            fig,axes=plt.subplots(5,2,figsize=(13,12),constrained_layout=True)
            for ch,ax in enumerate(axes.flat):
                for i in selected:
                    t=(meta[i]['center']+np.arange(32))*.02
                    ax.plot(t,targets[i,:,ch],color='black',alpha=.6)
                    ax.plot(t,preds[i,:,ch],color='tab:blue',alpha=.7)
                ax.set(xlabel='True source time (s)',ylabel=['vx m/s','yaw rad/s','EE x m','EE y m','EE z m','qx','qy','qz','qw','gripper rad'][ch]);ax.grid(alpha=.2)
            fig.suptitle('%s model on %s observations; black common expert label, blue prediction; new updates %d'%(model_label,obs_label,step))
            fig.savefig(out/('%s_on_%s.png'%(model_label,obs_label)),dpi=125);plt.close(fig)
    if step==0:
        # Both models are the exact same source at this node; no independent draws invented.
        all_metrics += [dict(m,model='B',identical_source_prediction=True) for m in list(all_metrics)]
        for obs in ['A','B']:
            import shutil
            shutil.copy2(out/('A_on_%s.npz'%obs),out/('B_on_%s.npz'%obs))
    save(out/'summary.json',dict(new_updates=step,source_updates=5000,metrics=all_metrics,windows=plan['windows'],
        diffusion_seeds=plan['diffusion_seeds'],history_causal=True,controller_original=True,
        note='Full32 diagnostic queue; deployment uses first16. Baseline A/B identical source. Quaternion metrics use physical sign-invariant angle.'))
    from forward_report import write_csv
    write_csv(out/'metrics.csv',all_metrics);verify(root)


def series(folder):
    rows=[json.loads(x) for x in (folder/'execution.jsonl').read_text().splitlines()]
    with np.load(folder/'trajectory.npz',allow_pickle=True) as z:
        action=z['action'].copy();next_state=z['next_state'].copy()
    cfg=json.loads((folder/'config.json').read_text())
    base=np.asarray([r['base_after'] for r in rows],dtype=np.float64)
    quat=base[:,3:7];u=quat[:,:3];w=quat[:,3:4]
    # The float backend has yaw-only root orientation, so its yaw quaternion is
    # the recorded root quaternion. Reconstruct the actual post-step world EE.
    v=next_state[:,19:22]+np.asarray(cfg['arm_base_offset'],dtype=np.float64)[0]
    post=base[:,:3]+v+2*(w*np.cross(u,v)+np.cross(u,np.cross(u,v)))
    pre=np.array([r['ee']['actual_pos_world'] for r in rows])
    regular=np.array([r['pre_observation_event']=='none' for r in rows[1:]])
    reconstruction_error=float(np.max(np.abs(post[:-1][regular]-pre[1:][regular]))) if regular.any() else None
    if reconstruction_error is not None and reconstruction_error>1.5e-5:
        raise ValueError('Post-step EE reconstruction inconsistent with next logged observation')
    return dict(rows=rows,t=np.array([r['sim_time'] for r in rows]),action=action,
        base=base[:,:3],actual=pre,actual_post=post,orientation_post=next_state[:,22:26],
        reconstruction_error_m=reconstruction_error,
        orientation=np.array([r['ee']['actual_quat'] for r in rows]),
        phase=np.array([r['phase_id'] for r in rows]),
        measured_speed=np.array([np.linalg.norm(np.asarray(r['base_after'][:3])-r['root_before'][:3])/.02 for r in rows]))


def compare_metrics(model,ref,mi,ri):
    p,t=model['action'][mi],ref['action'][ri]
    metrics=physical_metrics(p,t)
    metrics.update(actual_ee_trajectory_rmse_m=float(np.sqrt(np.mean(np.sum((model['actual'][mi]-ref['actual'][ri])**2,axis=-1)))),
        actual_post_step_ee_trajectory_rmse_m=float(np.sqrt(np.mean(np.sum((model['actual_post'][mi]-ref['actual_post'][ri])**2,axis=-1)))),
        own_post_step_target_tracking_rmse_m=float(np.sqrt(np.mean(np.sum((model['actual_post'][mi]-p[:,2:5])**2,axis=-1)))),
        own_post_step_target_orientation_rmse_deg=float(np.sqrt(np.mean(angle_error(model['orientation_post'][mi],p[:,5:9])**2))),
        actual_ee_orientation_trajectory_rmse_deg=float(np.sqrt(np.mean(angle_error(model['orientation'][mi],ref['orientation'][ri])**2))),
        own_pre_action_target_tracking_rmse_m=float(np.sqrt(np.mean(np.sum((model['actual'][mi]-p[:,2:5])**2,axis=-1)))),
        own_pre_action_target_orientation_rmse_deg=float(np.sqrt(np.mean(angle_error(model['orientation'][mi],p[:,5:9])**2))),
        measured_base_speed_rmse_m_s=float(np.sqrt(np.mean((model['measured_speed'][mi]-ref['measured_speed'][ri])**2))),
        base_path_position_rmse_m=float(np.sqrt(np.mean(np.sum((model['base'][mi]-ref['base'][ri])**2,axis=-1)))),samples=len(mi))
    return metrics


def video_grid(folders,names,out):
    import cv2
    import imageio_ffmpeg
    caps=[cv2.VideoCapture(str(p/'rollout.mp4')) for p in folders]
    counts=[int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    fps=[c.get(cv2.CAP_PROP_FPS) for c in caps]
    if counts!=[723,723,1250,1250] or any(abs(f-25)>1e-5 for f in fps):raise ValueError('Unexpected comparison source video length/rate')
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p',str(out)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i in range(1250):
            panels=[]
            for j,cap in enumerate(caps):
                if i<counts[j]:
                    ok,im=cap.read()
                    if not ok:raise ValueError('Comparison decode failed')
                    im=cv2.resize(im,(640,360))
                else:
                    im=np.zeros((360,640,3),dtype=np.uint8)
                    cv2.putText(im,'REFERENCE ENDED - NO REAL HOLD DATA',(12,160),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                cv2.putText(im,names[j]+' t=%.2fs'%(i/25),(10,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
                panels.append(im)
            frame=np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        for c in caps:c.release()
    if proc.wait():raise RuntimeError('Comparison encoder failed')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-i',str(out),'-f','null','-'],check=True)
    return dict(source_frames=counts,output_frames=1250,fps=25,reference_end_black=True,full_decode=True)


def closed(root,step):
    from branch16040 import REF_A,REF_B,save,verify
    from forward_report import write_csv,rollout_plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verify(root);out=root/'comparison'/('step_%06d'%step);out.mkdir(parents=True,exist_ok=False)
    folders=[REF_A,REF_B,*[root/l/'pilot/evaluation'/('step_%06d'%step)/'seed_16040' for l in ['A','B']]]
    names=['Original expert','Pure10D replay','Model A','Model B'];data=[series(p) for p in folders];records=[]
    for label,m in zip(['A','B'],data[2:]):
        for ref_label,r in zip(['A','B'],data[:2]):
            n=min(len(m['t']),len(r['t']));mi=np.arange(n);ri=np.arange(n)
            records.append(dict(model=label,reference=ref_label,alignment='true_time',stage='all',
                comparison_end_sec=float(r['t'][n-1]),**compare_metrics(m,r,mi,ri)))
            for phase in sorted(set(r['phase'])):
                ma=np.where(m['phase']==phase)[0];ra=np.where(r['phase']==phase)[0]
                record=dict(model=label,reference=ref_label,alignment='stage_progress',stage=int(phase),
                    reference_start_sec=float(r['t'][ra[0]]),reference_end_sec=float(r['t'][ra[-1]]),reference_duration_sec=len(ra)*.02)
                if not len(ma):records.append(dict(record,model_missing_stage=True));continue
                record.update(model_start_sec=float(m['t'][ma[0]]),model_end_sec=float(m['t'][ma[-1]]),model_duration_sec=len(ma)*.02,
                    stage_start_delay_sec=float(m['t'][ma[0]]-r['t'][ra[0]]))
                n=max(len(ma),len(ra));ii=ma[np.rint(np.linspace(0,len(ma)-1,n)).astype(int)];jj=ra[np.rint(np.linspace(0,len(ra)-1,n)).astype(int)]
                records.append(dict(record,**compare_metrics(m,r,ii,jj)))
        rollout_plots(root/label/'pilot/evaluation'/('step_%06d'%step)/'seed_16040')
    for alignment in ['true_time','stage_progress']:
        phases=[None] if alignment=='true_time' else sorted(set(data[0]['phase']))
        fig,axes=plt.subplots(6,len(phases),figsize=(max(12,3*len(phases)),15),squeeze=False,constrained_layout=True)
        for col,phase in enumerate(phases):
            for name,d in zip(names,data):
                ids=np.arange(len(d['t'])) if phase is None else np.where(d['phase']==phase)[0]
                if not len(ids):continue
                x=d['t'][ids] if phase is None else np.linspace(0,1,len(ids))
                vals=[d['action'][ids,0],d['measured_speed'][ids],d['action'][ids,2],d['actual'][ids,0],
                    np.linalg.norm(d['actual'][ids]-d['action'][ids,2:5],axis=-1),d['base'][ids,0]]
                for row,y in enumerate(vals):axes[row,col].plot(x,y,label=name)
            for row,ylabel in enumerate(['command vx m/s','measured base speed m/s','target EE x m','actual EE x m','own pre-action tracking m','base X m']):
                axes[row,col].set(xlabel='True time s' if phase is None else 'Stage progress (duration separately reported)',ylabel=ylabel,title='' if phase is None else 'phase '+str(phase));axes[row,col].grid(alpha=.2)
            axes[0,col].legend(fontsize=6)
        fig.savefig(out/(alignment+'.png'),dpi=125);plt.close(fig)
    video=video_grid(folders,names,out/'four_way.mp4');save(out/'video_verification.json',video)
    save(out/'summary.json',dict(new_updates=step,metrics=records,
        tracking_definition='Actual EE from pre-action execution record versus current target; not post-step same-action tracking.',
        timing='Time comparisons stop at reference end; no repeated end frames contribute to metrics. Stage comparison also reports actual delays/durations.',
        imitation_acceptance='Continuous errors only; no newly invented pass thresholds. Task events scored separately.'))
    write_csv(out/'metrics.csv',records);verify(root)


def final_report(root,output):
    """Re-score preserved predictions/trajectories; never rerun inference or physics."""
    from branch16040 import REF_A,REF_B,save,verify
    from forward_report import write_csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verify(root);output.mkdir(parents=True,exist_ok=False)
    cross=[];tasks=[];trajectories=[];timing=[]
    references={'A':series(REF_A),'B':series(REF_B)}
    for step in [0,500,1000,2000]:
        value=json.loads((root/'offline'/('step_%06d'%step)/'summary.json').read_text())
        cross.extend(dict(m,new_updates=step) for m in value['metrics'])
    for step in [1000,2000]:
        models={l:series(root/l/'pilot/evaluation'/('step_%06d'%step)/'seed_16040') for l in ['A','B']}
        for label,m in models.items():
            folder=root/label/'pilot/evaluation'/('step_%06d'%step)/'seed_16040'
            result=json.loads((folder/'result.json').read_text())
            stopping=result['stopping'];evaluable=bool(result['pass_done'])
            task=dict(branch=label,new_updates=step,opened=result['opened'],passed=result['pass_done'],
                stop_evaluable=evaluable,stopped=result['stopped'] if evaluable else None,joint=result['success'],
                max_signed_door_deg=result.get('max_signed_door_deg'),first_pass_step=stopping['first_pass_step'],
                complete_2500_steps=result['execution_ok'],post_pass_total_path_m=stopping.get('post_pass_total_path_m') if evaluable else None,
                settle_latency_sec=stopping.get('settle_latency_sec') if evaluable else None,
                max_two_second_drift_m=stopping.get('max_two_second_drift_m') if evaluable else None,
                reconstruction_vs_next_log_max_m=m['reconstruction_error_m'],video=str(folder/'rollout.mp4'))
            tasks.append(task)
            for ref_label,r in references.items():
                n=min(len(m['t']),len(r['t']));ids=np.arange(n)
                trajectories.append(dict(branch=label,reference=ref_label,new_updates=step,alignment='true_time',stage='all',
                    **compare_metrics(m,r,ids,ids)))
                for phase in sorted(set(r['phase'])):
                    ma=np.where(m['phase']==phase)[0];ra=np.where(r['phase']==phase)[0]
                    event=dict(branch=label,reference=ref_label,new_updates=step,stage=int(phase),
                        reference_start_sec=float(r['t'][ra[0]]),reference_duration_sec=len(ra)*.02,
                        model_missing_stage=not bool(len(ma)))
                    if not len(ma):timing.append(event);continue
                    event.update(model_start_sec=float(m['t'][ma[0]]),model_duration_sec=len(ma)*.02,
                        stage_start_delay_sec=float(m['t'][ma[0]]-r['t'][ra[0]]))
                    timing.append(event);n=max(len(ma),len(ra))
                    mi=ma[np.rint(np.linspace(0,len(ma)-1,n)).astype(int)];ri=ra[np.rint(np.linspace(0,len(ra)-1,n)).astype(int)]
                    trajectories.append(dict(event,alignment='stage_progress',**compare_metrics(m,r,mi,ri)))
        sets={**{'Expert A':references['A'],'Replay B':references['B']},**{'Model '+l:m for l,m in models.items()}}
        # Full physical trajectory curves, with original simulation time retained.
        fig,axs=plt.subplots(5,3,figsize=(15,16),constrained_layout=True)
        for name,d in sets.items():
            t=d['t'];act=d['action'];post=d['actual_post'];base=d['base']
            values=[act[:,0],act[:,1],d['measured_speed'],*[act[:,i] for i in range(2,5)],
                *[post[:,i] for i in range(3)],np.linalg.norm(post-act[:,2:5],axis=-1),
                angle_error(d['orientation_post'],act[:,5:9]),act[:,9],
                *[base[:,i] for i in range(2)],np.r_[0,np.cumsum(np.linalg.norm(np.diff(base,axis=0),axis=-1))]]
            labels=['command vx m/s','command yaw rad/s','measured speed m/s','target EE x m','target EE y m','target EE z m',
                'actual post EE x m','actual post EE y m','actual post EE z m','own post EE tracking m','own post orientation error deg',
                'gripper command rad','base x m','base y m','cumulative path m']
            for ax,y,title in zip(axs.flat,values,labels):ax.plot(t,y,label=name);ax.set(xlabel='True simulation time s',ylabel=title);ax.grid(alpha=.2)
        axs[0,0].legend(fontsize=7);fig.savefig(output/('true_time_full_%d.png'%step),dpi=130);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
        for name,d in sets.items():ax.plot(d['base'][:,0],d['base'][:,1],label=name)
        ax.set(xlabel='World X m',ylabel='World Y m');ax.axis('equal');ax.legend();ax.grid(alpha=.2)
        fig.savefig(output/('base_path_%d.png'%step),dpi=130);plt.close(fig)
        for phase in sorted(set(references['A']['phase'])):
            fig,axs=plt.subplots(4,3,figsize=(13,11),constrained_layout=True)
            for name,d in sets.items():
                ids=np.where(d['phase']==phase)[0]
                if not len(ids):continue
                progress=np.linspace(0,1,len(ids));a=d['action'][ids];p=d['actual_post'][ids]
                values=[a[:,0],a[:,1],d['measured_speed'][ids],*[a[:,i] for i in range(2,5)],*[p[:,i] for i in range(3)],
                    np.linalg.norm(p-a[:,2:5],axis=-1),angle_error(d['orientation_post'][ids],a[:,5:9]),a[:,9]]
                for ax,y,title in zip(axs.flat,values,labels[:12]):
                    ax.plot(progress,y,label=name+' %.2fs'%(len(ids)*.02));ax.set(xlabel='Task stage progress',ylabel=title);ax.grid(alpha=.2)
            axs[0,0].legend(fontsize=6);fig.suptitle('New updates %d; task phase %d; actual start delays/durations in stage_timing.csv'%(step,phase))
            fig.savefig(output/('stage_%d_%d.png'%(step,phase)),dpi=115);plt.close(fig)
    training=[]
    for label in ['A','B']:
        rows=[json.loads(x) for x in (root/label/'training/metrics.jsonl').read_text().splitlines()]
        training.extend(dict(r,branch=label) for r in rows)
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for label in ['A','B']:
        rows=[r for r in training if r['branch']==label]
        for ax,key,title in zip(axs,['train/loss','train/lr','train/elapsed_sec'],['Noise MSE','Learning rate','Optimization seconds']):
            ax.plot([r['train/step'] for r in rows],[r[key] for r in rows],label=label);ax.set(xlabel='Branch new updates (source weights at5000)',ylabel=title);ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'training_curves.png',dpi=140);plt.close(fig)
    for name,records in [('cross_actions',cross),('closed_loop_tasks',tasks),('trajectory_errors',trajectories),('stage_timing',timing),('training_metrics',training)]:
        write_csv(output/(name+'.csv'),records);save(output/(name+'.json'),records)
    save(output/'definitions.json',dict(predicted_target='Actual applied full10D target; position is identical to requested DP position, quaternion application normalized.',
        tracking='Actual post-step EE reconstructed from next_state and base_after versus same-step target; also retain original pre-action metrics.',
        reconstruction='Yaw-only float root; original config arm_base_offset. Checked against next pre-action log except explicit snap; no future observation used for inference.',
        alignment='True-time comparisons only real common prefix. Stage comparisons report original start delay and duration separately; missing stage is not filled.',
        physical_orientation='Sign-invariant normalized quaternion angle; raw component RMSE is only diagnostic and sign dependent.',
        stop_not_evaluable='No original pass event => stopped/latency/drift/path are null, joint failure.',
        scope='Only postprocessing existing successful inference and simulations; no re-execution or new training.'))
    verify(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--step',type=int,choices=[0,500,1000,2000]);p.add_argument('--closed-loop',action='store_true');p.add_argument('--report-output');a=p.parse_args()
    if a.report_output:final_report(Path(a.root).resolve(),Path(a.report_output).resolve())
    elif a.step is None:p.error('--step required without --report-output')
    else:(closed if a.closed_loop else offline)(Path(a.root).resolve(),a.step)
