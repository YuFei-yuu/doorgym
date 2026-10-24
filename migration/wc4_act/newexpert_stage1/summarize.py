"""Summarize frozen CPU evidence without granting live-execution approval."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_compact import Arm
from fullcycle_geometry import URDF
BASE='5b782e0ba0374d6e2864a0e5863469ced1cdc146'
run=Path(sys.argv[1]);out=run/'audit';out.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
def write(n,obj):(out/n).write_text(json.dumps(obj,indent=2)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
changed=git('diff','--name-only',BASE).splitlines()
assert all(p=='migration/wc4_act/fullcycle_compact.py' or p.startswith('migration/wc4_act/newexpert_stage1/') for p in changed)
old=git('show',BASE+':migration/wc4_act/fullcycle_compact.py');new=(ROOT/'migration/wc4_act/fullcycle_compact.py').read_text()
markers=['        elif self.phase=="compact_rendezvous":','        elif self.phase=="release":']
protected=dict(changed_files=changed,all_other_tracked_files_unchanged=True,
    compact_prefix_through_bypass_unchanged=old.split(markers[0])[0]==new.split(markers[0])[0],
    release_retract_exit_final_limits_unchanged=old.split(markers[1])[1]==new.split(markers[1])[1],
    base=BASE,head=git('rev-parse','HEAD').strip(),main=git('rev-parse','main').strip(),worktree=git('status','--porcelain=v1'),
    explanation='Complete git diff restricted to CompactExpert approach/closing limits and new stage directory. Opening, full-body pass, assets, physics, R11, recording, float64 reader fix, ACT, receipts/checkpoint/monitor code unchanged.')
assert protected['compact_prefix_through_bypass_unchanged'] and protected['release_retract_exit_final_limits_unchanged']
write('protected-scope.json',protected);(out/'source.diff').write_text(git('diff',BASE));(out/'git-refs.txt').write_text(git('show-ref','--heads'))
arm=Arm(URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf'))
reference={}
for label in ['old','new']:
    d=read(run/'reference'/(label+'-reference.json'));rows=d['sweep_samples'];worst=max(rows,key=lambda r:r['position_error_m']);q=np.array(worst['q'])[:6];slack=np.minimum(q-arm.lo,arm.hi-q)
    saturated=[r for r in rows if r['joint_slack_rad']<1e-6]
    reference[label]=dict(worst_sample=worst,worst_joint_slack=slack.tolist(),saturated_samples=len(saturated),total_samples=len(rows),saturated_cases=sorted({r['case'] for r in saturated}))
write('reachability-risk.json',reference)
continuity={}
for path in sorted(run.glob('*/trace.jsonl')):
    rows=[json.loads(l) for l in path.open()];a=np.array([r['action'] for r in rows]);b=np.array([r['base_before'] for r in rows]);after=np.array([r['base'] for r in rows]);delta=after[:,:2]-b[:,:2]
    lateral=-np.sin(b[:,2])*delta[:,0]+np.cos(b[:,2])*delta[:,1]
    events=[]
    for prev,r in zip(rows,rows[1:]):
        if prev['phase']!=r['phase']:
            events.append(dict(t=r['t'],previous=prev['phase'],phase=r['phase'],joint_command_delta_rad=float(max(abs(np.array(r['q'])-prev['q']))),ee_displacement_m=float(np.linalg.norm(np.array(r['ee'])-prev['ee'])),v_delta=r['action'][0]-prev['action'][0],w_delta=r['action'][1]-prev['action'][1]))
    continuity[path.parent.name]=dict(lateral_step_residual_m=float(max(abs(lateral))),integration_forward_residual_m=float(max(abs(np.sum(delta*np.c_[np.cos(b[:,2]),np.sin(b[:,2])],axis=1)-a[:,0]*.02))),events=events,
        note='First measured-to-command difference is inherited at handoff; per-step deltas here describe commanded continuity, not actuator dynamics. Rendezvous target field is the fixed endpoint, not the interpolated joint-space deployment target.')
write('continuity.json',continuity)
files=[]
for part in ['migration/wc4_act','high-level/data/asset/a2wz1','high-level/data/asset/door_set/wc4','high-level/data/cfg/b1z1_opendoor.yaml','high-level/float_ik/config/a2w_float_ik_push_door_parallel.yaml']:
    p=ROOT/part;files.extend([x for x in p.rglob('*') if x.is_file() and '__pycache__' not in str(x)] if p.is_dir() else [p])
write('source-asset-sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))})
write('budget.json',dict(physical_episodes=0,collection_episodes=0,optimizer_updates=0,policy_evaluations=0,gpu_leases=0,monitors_started=0,reference_ik_samples=sum(v['total_samples'] for v in reference.values()),synthetic_cases=len(continuity),synthetic_steps={p.parent.name:sum(1 for _ in p.open()) for p in run.glob('*/trace.jsonl')},clearance_cases=len(list(run.glob('clearance*/completed.json'))),behavior_checks=4,legacy_tests=8,failed_invocations=[dict(run='wc4-newexpert-stage1-20261025-01',mode='reference',reason='Python3.8 relative source path receipt ValueError; numerical validation not entered; kept unchanged')]))
write('decision.json',dict(status='FROZEN_OFFLINE_REVIEW_ONLY_STOP_BEFORE_LIVE',live_authorized=False,physical_success_claim=False,
    blockers=['Joint-limit saturation persists in configured height/pitch range; exact pose reachability unresolved.', 'Historical handoff has conservative wrist-panel overlap in nominal source geometry; inherited but not resolved.', 'Infinite worst-wall-plane overlap is inconclusive because finite wall openings are not represented; wall separation not certified.', 'No dynamic/contact or continuous swept/self-collision certification.'],
    next_step='Independent review and a new user task sheet; no automatic parameter search, retry, collection or training.'))
print(json.dumps({'audit':str(out),'status':'frozen offline risks','changed':changed},indent=2))
