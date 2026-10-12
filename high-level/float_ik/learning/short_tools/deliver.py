"""CPU-only delivery for the fixed 2000-step Cartesian smoke experiment."""
import csv, hashlib, json, pathlib, subprocess, sys
import numpy as np
import cv2
import imageio_ffmpeg
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
repo=pathlib.Path('/data/zyf/video2door');root=repo/'experiments/float-push/20261012-ee-short-refined'
sys.path.insert(0,str(repo/'high-level/float_ik/learning'))
from ee_quality import world_positions, audit
from dataset_contract import validate_episode
from schema import EE_SCHEMA
from policy import Policy

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
plan=load(root/'plan.json');frozen=load(root/'frozen_model.json');ckpath=pathlib.Path(frozen['checkpoint'])
assert sha(ckpath)==frozen['sha256']
ck=torch.load(ckpath,map_location='cpu');assert ck['step']==2000 and ck['schema']==EE_SCHEMA
manifest=load(root/'manifest.json');assert ck['manifest']==manifest
train=[]
for item in manifest:
    with np.load(item['episode']) as z:
        validate_episode(z,EE_SCHEMA)
        if item['split']=='train':train.append({k:z[k] for k in ['state','action']})
for k in ['state','action']:
    values=torch.from_numpy(np.concatenate([x[k] for x in train]))
    assert torch.allclose(ck['stats'][k+'_mean'],values.mean(0))
    assert torch.allclose(ck['stats'][k+'_std'],values.std(0).clamp_min(.01))
assert set(plan['train_seeds']).isdisjoint(plan['offline_val_seeds'])
assert set(plan['rollout_val_seeds']).isdisjoint(plan['train_seeds']+plan['offline_val_seeds'])
assert not torch.cuda.is_initialized()
prior=load(repo/plan['predecessor']/'preflight.json')
assert all(sha(repo/p)==v for p,v in prior['old_files'].items())
expected=load(root/'source_sha256.json');assert all(sha(repo/p)==v for p,v in expected.items())
quality=dict(profile='ee_reference_limited_v1',episodes=[audit(f,plot=True) for f in sorted((root/'execution_diagnostic').glob('*')) if (f/'summary.json').exists()]);(root/'execution_diagnostic_quality.json').write_text(json.dumps(quality,indent=2));details={pathlib.Path(d['path']).name:d for d in quality['episodes']}
rows=[json.loads(x) for x in (root/'training/train.jsonl').read_text().splitlines()]
assert [x['step'] for x in rows]==list(range(1,2001))
assert all(np.isfinite(x['loss']) and np.isfinite(x['grad_norm']) for x in rows)
plots=root/'plots';plots.mkdir(exist_ok=True)
fig,axs=plt.subplots(1,2,figsize=(11,4))
axs[0].plot([x['step'] for x in rows],[x['loss'] for x in rows],alpha=.5,lw=.7,label='Training noise MSE')
v=[x for x in rows if 'val_loss' in x];axs[0].plot([x['step'] for x in v],[x['val_loss'] for x in v],'-o',label='Fixed offline validation MSE');axs[0].legend();axs[0].set(xlabel='Optimizer step',ylabel='Noise MSE')
axs[1].plot([x['elapsed_sec'] for x in rows],[x['step'] for x in rows]);axs[1].set(xlabel='Training-loop elapsed seconds',ylabel='Optimizer step');fig.tight_layout();fig.savefig(plots/'training.png',dpi=150);plt.close(fig)

def metrics(folder):return list(csv.DictReader((folder/'metrics.csv').open()))
def encode(original):
    output=original.with_name('rollout_h264.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-i',str(original),'-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
    return output

def video_check(path, expected=1425):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        n+=1
    cap.release();assert n==expected and abs(fps-30)<1e-6
    return dict(path=str(path.relative_to(root)),frames=n,fps=fps,seconds=n/fps,sha256=sha(path))

videos=[];pairs=[]
for seed in plan['rollout_val_seeds']:
    folders=[root/'execution_diagnostic'/('limited_%s_seed_%d'%(mode,seed)) for mode in ['expert','policy']]
    summaries=[load(f/'summary.json') for f in folders]
    assert all(s['status']=='completed' and s['steps_completed']==2850 for s in summaries)
    assert summaries[0]['randomization']==summaries[1]['randomization']
    assert all(s['execution_profile']=='ee_reference_limited_v1' and s['execution_wrapper_sha256']==sha(root/'limited_rollout.py') for s in summaries)
    assert summaries[1]['checkpoint_step']==2000 and summaries[1]['expert_call_guard_passed']
    assert summaries[1]['phases']=={'policy':2850}
    data=[]
    for folder in folders:
        with np.load(folder/'episode.npz') as z:data.append({k:z[k] for k in ['state','next_state','action','images','joint_position_target','actual_joint_position']})
    # Same environment seed plus policy-independent sensors/reset.
    assert np.array_equal(data[0]['state'][0],data[1]['state'][0])
    assert np.array_equal(data[0]['images'][0],data[1]['images'][0])
    cols=[metrics(f) for f in folders]
    pos=[world_positions(d['next_state'],m) for d,m in zip(data,cols)]
    t=np.arange(2850)/60
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for label,d,m,w in zip(['Expert + limits','DP 2000 + limits'],data,cols,pos):
        axs[0,0].plot(w[:,0],w[:,2],label=label);axs[0,1].plot(t,[float(x['signed_angle_deg']) for x in m],label=label)
        axs[1,0].plot(t,[float(x['ee_command_pos_error_m']) for x in m],label=label)
        axs[1,1].plot(t[1:],np.max(abs(np.diff(d['joint_position_target'][:,:6],axis=0)),axis=1),label=label)
    axs[0,0].set(xlabel='World x (m)',ylabel='World z (m)',title='Actual EE path: x-z projection')
    axs[0,1].set(xlabel='Simulation time (s)',ylabel='Door angle (degrees)')
    axs[1,0].set(xlabel='Simulation time (s)',ylabel='EE command tracking error (m)')
    axs[1,1].set(xlabel='Simulation time (s)',ylabel='Max arm target jump (rad / step)')
    for ax in axs.flat:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Paired reset seed %d | DP 2000 + generic reference limits (diagnostic)'%seed);fig.tight_layout();fig.savefig(plots/('comparison_%d.png'%seed),dpi=150);plt.close(fig)
    for folder in folders:videos.append(video_check(encode(folder/'rollout.mp4')))
    output=root/('comparison_seed_%d.mp4'%seed)
    caps=[cv2.VideoCapture(str(f/'rollout.mp4')) for f in folders]
    proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1280x360','-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    contact=[]
    for i in range(1425):
        read=[c.read() for c in caps];assert all(ok for ok,_ in read)
        frame=np.hstack([f for _,f in read]);cv2.putText(frame,'EXPERT + SAME LIMITS',(8,350),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1);cv2.putText(frame,'DP 2000 + REFERENCE LIMITS (diagnostic)',(648,350),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1);proc.stdin.write(frame.tobytes())
        if i in [0,300,600,900,1200,1424]:contact.append(cv2.resize(frame,(960,270)))
    for cap in caps:cap.release()
    proc.stdin.close();assert proc.wait()==0
    videos.append(video_check(output));cv2.imwrite(str(root/('contact_seed_%d.png'%seed)),np.vstack(contact))
    jitter={}
    for label,d,w in zip(['expert','policy'],data,pos):
        accel=np.diff(w,n=2,axis=0)*60**2;jerk=np.diff(w,n=3,axis=0)*60**3
        jitter[label]=dict(initial_target_jump_m=float(np.linalg.norm(d['action'][0,3:6]-d['state'][0,31:34])),actual_path_length_m=float(np.linalg.norm(np.diff(w,axis=0),axis=1).sum()),actual_acceleration_rms_mps2=float(np.sqrt((accel**2).sum(1).mean())),actual_jerk_rms_mps3=float(np.sqrt((jerk**2).sum(1).mean())))
    pairs.append(dict(seed=seed,initial_state_images_equal=True,expert=details[folders[0].name],policy=details[folders[1].name],jitter_raw_60hz=jitter))
videos.append(video_check(encode(root/'pilot/expert_nominal/rollout.mp4')))
raw_failure=load(root/'validation/policy_seed_8500/summary.json')
assert raw_failure['status']=='error' and raw_failure['steps_completed']==368
videos.append(video_check(encode(root/'validation/policy_seed_8500/rollout.mp4'),expected=184))
videos.append(video_check(encode(root/'validation/expert_seed_8500/rollout.mp4')))
budget=[json.loads(s) for s in (root/'budget.jsonl').read_text().splitlines()]
result=dict(checks=dict(checkpoint_frozen=True,exactly_2000_updates=True,train_only_normalization=True,split_disjoint=True,policy_expert_guard=True,paired_initial_state_images=True,all_diagnostic_complete=True,full_video_decode=True,old_experiment_unchanged=True,source_hashes_match=True,cpu_delivery_no_cuda_context=not torch.cuda.is_initialized()),schema=EE_SCHEMA,checkpoint=frozen,plan=plan,execution_profile='ee_reference_limited_v1',direct_validation=dict(started=1,completed=0,errors=1,unstarted_seeds=[8501,8502],failure_steps=368),execution_diagnostic_plan=load(root/'execution_diagnostic_plan.json'),dataset=load(root/'dataset_quality.json'),pilot=load(root/'pilot_quality.json'),pairs=pairs,training=dict(first_loss=rows[0]['loss'],last_loss=rows[-1]['loss'],first_val_loss=v[0]['val_loss'],last_val_loss=v[-1]['val_loss'],train_loop_seconds=rows[-1]['elapsed_sec'],optimizer_steps=2000),video_checks=videos,budget_seconds=sum(x['elapsed_sec'] for x in budget),budget_cap=3600,predecessor_pilot_included=True,offline_action_diagnostics=load(root/'offline_action_diagnostics.json'),failure_audit=load(root/'failure_audit_final.json'))
assert all(result['checks'].values());assert result['budget_seconds']<=3600
(root/'delivery_checks.json').write_text(json.dumps(result,indent=2,allow_nan=False))
files=[p for p in root.rglob('*') if p.is_file() and p.name not in ['artifact_sha256.json','delivery.log'] and p.name not in ['driver.lock','budget.lock']]
(root/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in sorted(files)},indent=2))
print(json.dumps({k:result[k] for k in ['checks','training','budget_seconds']},indent=2))
print('PAIRS',json.dumps([dict(seed=x['seed'],expert_success=x['expert']['success'],policy_success=x['policy']['success'],policy_angle=x['policy']['final_angle_deg']) for x in pairs]))
