"""Final-version-only audit; saved prior failures are comparison, not acceptance."""
import ast,json,math,hashlib,subprocess,sys,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_compact import Arm
from fullcycle_geometry import URDF
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
run=Path(sys.argv[1]);out=run/'audit';out.mkdir(exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
base='12058c18573d100d0524d47da9c1f5c5e8060273';implementation=git('rev-parse','63024ac');changed=git('diff','--name-only',base).splitlines()
assert all(p=='migration/wc4_act/fullcycle_compact.py' or p.startswith('migration/wc4_act/newexpert_stage3/') for p in changed)
old=git('show',base+':migration/wc4_act/fullcycle_compact.py');new=(ROOT/'migration/wc4_act/fullcycle_compact.py').read_text()
protected=dict(base=base,implementation=implementation,head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),main=git('rev-parse','main'),worktree=git('status','--porcelain=v1'),changed=changed,
    before_rendezvous_unchanged=old.split('        elif self.phase=="compact_rendezvous":')[0]==new.split('        elif self.phase=="compact_rendezvous":')[0],
    seek_close_release_final_unchanged=old.split('        elif self.phase=="compact_seek":')[1].rstrip()==new.split('        elif self.phase=="compact_seek":')[1].rstrip(),
    measured_arrival_thresholds_unchanged='deployed=(np.linalg.norm(actual_pad-goal_pad)<.008\n                          and np.dot(actual[:3,0],endpoint[:3,0])>math.cos(.01))' in new,
    all_other_tracked_files_unchanged=True)
assert all(protected[k] for k in ['before_rendezvous_unchanged','seek_close_release_final_unchanged','measured_arrival_thresholds_unchanged'])
write('protected-scope.json',protected);(out/'source.diff').write_text(git('diff',base))
arm=Arm(URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf'));cases={};traces={};binding={}
for name in ['nominal','high-zero','low-pitch','high-pitch','drift45']:
 d=run/name;cmd=read(d/'command.json');assert cmd['head']==implementation
 for n,h in cmd['source_sha256'].items():assert sha(ROOT/n)==h
 binding[name]=cmd;rows=[json.loads(s) for s in (d/'trace.jsonl').open()];traces[name]=rows;s=read(d/'summary.json')
 dep=[r for r in rows if r['phase_before']=='compact_rendezvous'];sw=[r for r in dep if r['phase']=='compact_seek'];assert len(sw)==1
 gate=sw[0]['arrival_gate_pre'];assert gate['pad_error_m']<.008 and gate['axis_error_rad']<.01 and gate['base_error_m']<=.008 and gate['v_previous']<.01
 ready=[r for r in dep if r['arrival_gate_pre']['pad_error_m']<.008 and r['arrival_gate_pre']['axis_error_rad']<.01 and r['arrival_gate_pre']['base_error_m']<=.008 and r['arrival_gate_pre']['v_previous']<.01 and abs((r['base_before'][2]-math.radians(75)+math.pi)%(2*math.pi)-math.pi)<.008]
 assert ready and ready[0]['t']==sw[0]['t']
 a=np.array([r['action'] for r in rows]);R=np.array([r['ee_rotation'] for r in rows]);dr=Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).magnitude()
 pad=np.array([r['pad'] for r in dep]);q=np.array([r['q'][:6] for r in dep]);dq=np.diff(q,axis=0)
 active=[r for r in dep if r['ik'].get('progress',0)>.05];qa=np.array([r['q'][:6] for r in active]);slack=np.minimum(qa-arm.lo,arm.hi-qa)
 tracking=[r['target_tracking_m'] for r in dep if r['target_tracking_m'] is not None]
 ik=[r['ik'] for r in rows if r['ik'] and r['ik'].get('constraint')!='joint_arc_tracking_not_ik']
 z=[r['z'] for r in rows];assert max(z)==min(z)
 clear=read(run/('clearance-'+name)/'clearance.json');mins=clear['minimums']
 nongrip={k:v for k,v in mins.items() if not k.startswith('compact_unload/') and not k.endswith('/end_panel') and v['gap_m']<0}
 cases[name]=dict(complete_code_path=s['complete_code_path'],suffix_seconds=s['suffix_seconds'],steps=len(rows),base_height=z[0],
    deploy_start=dep[0]['t'],deploy_switch=sw[0]['t'],arrival_gate=gate,wait_after_actual_ready_s=sw[0]['t']-ready[0]['t'],
    deployment_pad_height_range=[float(pad[:,2].min()),float(pad[:,2].max())],deployment_reference_tracking_max_m=float(max(tracking)),
    deployment_min_slack_after_5percent_rad=float(slack.min()),deployment_max_joint_step_rad=float(abs(dq).max()),
    full_max_joint_rate_rad_s=float(abs(np.diff(a[:,2:],axis=0)).max()/.02),full_max_orientation_step_rad=float(dr.max()),
    true_ik_max_position_error_m=max(v['position_error_m'] for v in ik),true_ik_max_orientation_or_axis_error_rad=max(v['orientation_error_rad'] for v in ik),
    final_home_error_rad=s['final_home_error_rad'],final_yaw_error_rad=s['final_yaw_error_rad'],
    finite_wall_min_m=min(v['gap_m'] for k,v in mins.items() if 'finite_walls' in k),nonend_overlap_after_unload=nongrip,
    clearance_minimums=mins)
write('version-bindings.json',binding);write('cases.json',cases)
ref=read(run/'reference/new-reference.json')['sweep_samples'];write('static-summary.json',dict(samples=len(ref),max_position_m=max(r['position_error_m'] for r in ref),min_joint_slack_rad=min(r['joint_slack_rad'] for r in ref)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=traces['nominal'];t=np.array([x['t'] for x in r]);b=np.array([x['base'] for x in r]);a=np.array([x['action'] for x in r]);pad=np.array([x['pad'] for x in r]);refpad=[]
from newexpert_stage2.push_geometry import PAD_LOCAL
for x in r:
 T=x['target'];refpad.append([np.nan]*3 if T is None else (np.array(T)[:3,3]+np.array(T)[:3,:3]@PAD_LOCAL).tolist())
refpad=np.array(refpad);dep=np.array([x['phase_before']=='compact_rendezvous' for x in r]);zoom=(t>=24)&(t<=34)
fig,ax=plt.subplots(2,2,figsize=(14,9));ax[0,0].plot(b[:,0],b[:,1],label='base');ax[0,0].plot(pad[:,0],pad[:,1],alpha=.6,label='actual pad (synthetic)')
from matplotlib.patches import Rectangle
for i,w in enumerate(read(run/'clearance-nominal/finite-wall-binding.json')['0']['actors']):
 c=w['center'];s=w['size'];ax[0,0].add_patch(Rectangle((c[0]-s[0]/2,c[1]-s[1]/2),s[0],s[1],color='brown',alpha=.3,label='finite wall' if i==0 else None))
ax[0,0].set_aspect('equal');ax[0,0].set(xlabel='Reference x (m)',ylabel='Reference y (m)',title='Complete suffix + finite walls')
ax[0,1].plot(t,a[:,0],label='v (m/s)');ax[0,1].plot(t,a[:,1],label='yaw rate (rad/s)');ax[0,1].set(title='Base commands',xlabel='Synthetic time (s)')
ax[1,0].plot(t,[x['z'] for x in r],label='base z');ax[1,0].plot(t,pad[:,2],label='pad z');ax[1,0].plot(t[dep],refpad[dep,2],'--',label='deployment reference z');ax[1,0].set(title='Constant base height; necessary arm arc',xlabel='Synthetic time (s)',ylabel='Height (m)')
for i in range(6):ax[1,1].plot(t,a[:,2+i],label='joint%d'%(i+1))
ax[1,1].set(title='Continuous bounded joint commands',xlabel='Synthetic time (s)',ylabel='rad')
for v in ax.flat:v.grid(alpha=.2);v.legend(fontsize=7)
fig.suptitle('Stage 3 final version | Complete SYNTHETIC suffix, NOT physical success');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/'full-suffix.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(14,9))
for i,n in enumerate(['x','y','z']):ax[0,0].plot(t[zoom],pad[zoom,i],label='actual '+n);ax[0,0].plot(t[zoom],refpad[zoom,i],'--',alpha=.6,label='reference '+n)
ax[0,0].set(title='Contact centre: arc unfolding, not Cartesian straight line',ylabel='m')
for i in range(6):ax[0,1].plot(t[zoom],a[zoom,2+i],label='joint%d'%(i+1))
ax[0,1].set(title='Joint unfolding',ylabel='rad')
ax[1,0].plot(t[zoom],[x['arrival_gate_pre']['pad_error_m'] for x in r if 24<=x['t']<=34],label='pad error to hover');ax[1,0].axhline(.008,color='red',linestyle='--',label='unchanged 8 mm');ax[1,0].set(title='Measured arrival before transition',ylabel='m')
ax[1,1].plot(t[zoom],a[zoom,0],label='base v');ax[1,1].plot(t[zoom],[x['ik'].get('progress',np.nan) for x in r if 24<=x['t']<=34],label='smooth deployment progress');ax[1,1].set(title='Base/arm coordination')
for v in ax.flat:v.axvline(cases['nominal']['deploy_switch'],color='black',linestyle=':',label='switch to seek');v.grid(alpha=.2);v.legend(fontsize=7);v.set_xlabel('Synthetic time (s)')
fig.suptitle('Final deployment detail | actual gate, unchanged thresholds');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/'deployment-detail.png',dpi=170);plt.close(fig)
# Prior failure appears only as a separately labelled comparison.
failed=ROOT/'.repro/reviews/wc4-newexpert-stage2-20261025-01/new-linear-nominal/trace.jsonl'
old=[json.loads(s) for s in failed.open()];fig,axes=plt.subplots(1,2,figsize=(12,5))
for label,rs in [('stage2 failed straight',old),('stage3 final joint arc',r)]:
 tt=[x['t'] for x in rs];axes[0].plot(tt,[x['pad'][2] for x in rs],label=label);axes[1].plot(tt,[x['q'][0] for x in rs],label=label)
axes[0].set(ylabel='Contact centre height (m)',title='Saved failure vs final synthetic suffix');axes[1].set(ylabel='Joint1 (rad)',title='No persistent limit trap in final deployment')
for v in axes:v.set_xlabel('Synthetic time (s)');v.grid(alpha=.2);v.legend(fontsize=8)
fig.tight_layout();fig.savefig(out/'failure-comparison.png',dpi=170);plt.close(fig)
files=[p for p in (ROOT/'migration/wc4_act').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
write('source-sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
pre=ROOT/'.repro/reviews/wc4-newexpert-stage3-20261025-01'
write('budget.json',dict(physical_episodes=0,approved_six_trials_consumed=0,collection=0,updates=0,policy_evaluations=0,gpu_locks=0,monitors=0,
    final_synthetic_steps={n:len(rs) for n,rs in traces.items()},final_reference_ik_samples=3006,final_clearance_cases=5,final_behavior_tests=7,final_legacy_tests=8,
    retained_3second_control_steps={p.parent.name:sum(1 for _ in p.open()) for p in pre.glob('*/trace.jsonl')},retained_3second_reference_samples=3006,retained_3second_behavior_tests=7,retained_3second_legacy_tests=8))
issues={n:c['nonend_overlap_after_unload'] for n,c in cases.items() if c['nonend_overlap_after_unload']}
write('decision.json',dict(status='FROZEN_CPU_CHECKS_COMPLETE' if not issues and all(c['complete_code_path'] for c in cases.values()) else 'FROZEN_REVIEW_RISKS',
    final_implementation=implementation,complete_cases=sum(c['complete_code_path'] for c in cases.values()),total_cases=len(cases),source_overlap_risks=issues,
    physical_success_claim=False,live_budget_consumed=0,limitations=['Synthetic perfect tracking/contact feedback is not physics.','Finite source geometry sampled every0.5s and phase boundaries, first4s every0.1s; not continuous swept/self-collision certification.','Only one historical post-pass boundary reused; height/pitch endpoints do not re-execute randomized opening.','drift45 is an additional artificial door drift stress case, not a new initial distribution.']))
print(json.dumps({n:{k:v for k,v in c.items() if k!='clearance_minimums'} for n,c in cases.items()},indent=2))
