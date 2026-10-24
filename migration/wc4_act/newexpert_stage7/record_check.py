"""Read-only per-attempt gate. Failures are retained, never replaced."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from adapters import record_contract
from common import sha


def validate(run,index,destination):
    p=run/'attempt_000';result=json.loads((p/'result.json').read_text())
    states=[json.loads(l) for l in (p/'states.jsonl').open()]
    clock=[json.loads(l) for l in (p/'recording-clock.jsonl').open()]
    systemic=[];steps=result['steps'];dt=states[0]['t'] if states else .02
    if len(states)!=steps or [s['step'] for s in states]!=list(range(steps)):systemic.append('noncontinuous_state_log')
    rawfiles=list((p/'raw').glob('episode_*.npz'))
    if len(rawfiles)!=1:raise RuntimeError('systemic_missing_raw_npz')
    raw=rawfiles[0]
    with np.load(raw,allow_pickle=True) as z:
        T=int(z['state'].shape[0]);state=z['state'].copy();action=z['action'].copy()
        if state.shape!=(T,9) or action.shape!=(T,9) or not np.isfinite(np.r_[state.ravel(),action.ravel()]).all():systemic.append('bad_joint9')
        if T!=math.ceil(steps/2) or len(clock)!=T:systemic.append('record_clock_length')
        indices=np.arange(T)*2
        np.testing.assert_allclose(state[1:,:2],action[:-1,:2],atol=1e-7,rtol=0)
        np.testing.assert_allclose(state[:,2:],z['replay_dof_pos'][:,12:19],atol=1e-7,rtol=0)
        predicted=[]
        for f,k in enumerate(indices):
            s=states[k];q=np.array(s['arm_command'],dtype=np.float32)
            if f==0:base=[0.,0.]
            else:
                old=np.array(states[k-2]['command_base']);cur=np.array(s['command_base'])
                delta=(cur[:2].astype(np.float32)-old[:2].astype(np.float32))/(2*dt)
                heading=np.array([math.cos(old[2]),math.sin(old[2])],dtype=np.float32)
                base=[float(delta@heading),((cur[2]-old[2]+math.pi)%(2*math.pi)-math.pi)/(2*dt)]
            predicted.append(np.r_[base,q])
            if clock[f]['source_step']!=k or abs(clock[f]['physics_post_time_s']-s['t'])>1e-8:systemic.append('frame_timestamp_mismatch')
            np.testing.assert_allclose(state[f,2:],np.asarray(s['arm_q'],np.float32),atol=1e-7,rtol=0)
        np.testing.assert_allclose(action,np.asarray(predicted),atol=2e-6,rtol=0)
        views={}
        for v in ['front','wrist']:
            image=z[v+'_masked_depth']
            if image.dtype!=np.uint8 or image.shape!=(T,480,640):systemic.append('raw_depth_schema')
            for f in range(T):
                if hashlib.sha256(image[f].tobytes()).hexdigest()!=clock[f]['raw_depth_sha256'][v]:systemic.append('raw_depth_digest')
            if not np.any(np.diff(image[:,::16,::16].astype(np.int16),axis=0)):systemic.append('frozen_'+v+'_depth')
            views[v]=dict(shape=list(image.shape),dtype=str(image.dtype),raw_bytes=image.nbytes)
    videos=[]
    if result.get('error') or result['reason'].startswith('infrastructure_or_recording_error'):systemic.append(result['reason'])
    contacts=[json.loads(line) for line in (p/'contacts.jsonl').open()]
    render=json.loads((run/'raw-render-receipt.json').read_text())
    contract=record_contract(steps,[s['step'] for s in states],[c['step'] for c in contacts],
        [c['source_step'] for c in clock],json.loads((run/'depth-encoding-receipt.json').read_text()),
        render['rendered_steps'],[str(x) for x in p.rglob('*.mp4')])
    systemic+=contract['failures']
    for i,c in enumerate(contacts):
        if abs(c['interval_s'][1]-states[i]['t'])>1e-8:systemic.append('contact_timestamp_mismatch')
    if result['reason'].startswith(('stage7_','nonfinite','missing_depth_observation','forbidden_unlocked_hinge_write','nonzero_auxiliary_hinge_effort')):
        systemic.append('systemic_guard_stop:'+result['reason'])
    nominal=None  # R11 does not require reproducing an expert path.
    # Core terminal result is provisional until aligned physical push review.
    out=dict(trial=json.loads((p/'frozen-trial.json').read_text())['index'],seed=json.loads((p/'frozen-trial.json').read_text())['seed'],steps=steps,frames=T,
        sim_seconds=result['sim_seconds'],systemic_errors=sorted(set(systemic)),recording_checks_pass=not systemic,no_video_raw_contract=contract,video_recording_authorized=False,
        automatic_final_conditions_met=result.get('automatic_final_conditions_met',False),monitor=result.get('independent_task_monitor'),
        nominal_comparison=nominal,views=views,videos=videos,raw_npz=str(raw),raw_npz_bytes=raw.stat().st_size,
        raw_npz_sha256=sha(raw),
        physical_acceptance='requires_aligned_contact_motion_and_source_review; video explicitly waived',reason=result['reason'])
    (destination/'record-validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if systemic:raise RuntimeError('STOP_SYSTEMIC: '+str(systemic))



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'audit';out.mkdir(exist_ok=True);validate(a.run,0,out)
