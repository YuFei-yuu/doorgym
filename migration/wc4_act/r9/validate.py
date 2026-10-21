"""Read-only per-attempt gate. Failures are retained, never replaced."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import av


def validate(run,index):
    p=run/('trial_%02d'%index);result=json.loads((p/'result.json').read_text())
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
    for path in [p/'overview/env_0000/rollout.mp4',p/'front_depth.mp4',p/'wrist_depth.mp4']:
        with av.open(str(path)) as f:
            stream=f.streams.video[0];fps=float(stream.average_rate);count=sum(1 for _ in f.decode(video=0))
        if fps!=25 or count!=T:systemic.append('video_count_fps')
        videos.append(dict(path=str(path),frames=count,fps=fps,fully_decoded=True,duration_s=count/fps))
    if result.get('error') or result['reason'].startswith('infrastructure_or_recording_error'):systemic.append(result['reason'])
    # Exact nominal comparison is a recording noninterference check, not a
    # new success criterion for randomized trials.
    nominal=None
    if json.loads((p/'frozen-trial.json').read_text())['nominal']:
        old=run.parents[0]/'wc4-fullcycle-compact-r7-20261021/attempt_000/states.jsonl'
        oldstates=[json.loads(l) for l in old.open()]
        n=min(len(oldstates),len(states))
        diffs={k:float(np.max(np.abs(np.asarray([s[k] for s in states[:n]])-np.asarray([s[k] for s in oldstates[:n]])))) for k in ['action','actual_base','door_q','arm_q']}
        nominal=dict(reference_steps=len(oldstates),new_steps=len(states),max_abs_differences=diffs)
        if len(states)!=len(oldstates) or diffs['action']>1e-5 or diffs['door_q']>1e-5:systemic.append('nominal_behavior_changed')
    # Core terminal result is provisional until aligned physical push review.
    out=dict(trial=index,seed=json.loads((p/'frozen-trial.json').read_text())['seed'],steps=steps,frames=T,
        sim_seconds=result['sim_seconds'],systemic_errors=sorted(set(systemic)),recording_checks_pass=not systemic,
        automatic_final_conditions_met=result.get('automatic_final_conditions_met',False),monitor=result.get('independent_task_monitor'),
        nominal_comparison=nominal,views=views,videos=videos,raw_npz=str(raw),raw_npz_bytes=raw.stat().st_size,
        physical_acceptance='requires_aligned_contact_and_video_review',reason=result['reason'])
    (p/'record-validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if systemic:raise RuntimeError('STOP_SYSTEMIC: '+str(systemic))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args();validate(a.run,a.index)
