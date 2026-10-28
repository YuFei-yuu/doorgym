"""CPU audit of all actual logs/raw buffers; no simulation or derived frame dumps."""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from success_random5.common import read,write,sha

def audit(out):
    result=read(out/'result.json');clock=[json.loads(x) for x in (out/'recording-clock.jsonl').open()]
    records={};phases={};previous=None;max_angle=0.;opened=None;passed=None;last=None;count=0
    first_latched=None;separated_after_latch=None
    for line in (out/'states.jsonl').open():
        row=json.loads(line);assert row['step']==count
        assert row['t']==row['timing']['physics_post_time_s'] if 'physics_post_time_s' in row['timing'] else row['t']>0
        max_angle=max(max_angle,abs(row['door_q'][0]))
        if opened is None and abs(row['door_q'][0])>=math.radians(80):opened=row['t']
        if row['passed'] and passed is None:passed=row['t']
        if row['latched'] is not None and first_latched is None:first_latched=row
        if first_latched is not None and not row['nonzero_door_contact'] and separated_after_latch is None:separated_after_latch=row['t']
        phase=phases.setdefault(row['phase'],dict(first_post_s=row['t'],last_post_s=row['t']))
        phase['last_post_s']=row['t']
        if count%2==0:records[count]=dict(arm=row['arm_q'],command=row['arm_command'],t=row['t'])
        if first_latched is row:pre_latch=previous
        previous=row;last=row;count+=1
    assert count==result['steps'] and [x['source_step'] for x in clock]==list(range(0,count,2))
    assert sum(1 for _ in (out/'contacts.jsonl').open())==count
    lock_count=0;first_lock=None;max_delta=0.
    for line in (out/'lock-writes.jsonl').open():
        row=json.loads(line);max_delta=max(max_delta,abs(row['delta']))
        if row['kind']=='relocked':
            lock_count+=1
            if first_lock is None:first_lock=row
    legal=False
    if first_lock:
        assert pre_latch is not None
        gate=pre_latch['task_monitor']
        assert not gate['returned'] and gate['close_span_s']>=.5-1e-9
        assert abs(first_lock['latched']-pre_latch['door_q'][0])<1e-7
        assert first_latched['step']==first_lock['step']
        legal=last['latched'] is not None and not last['unlocked']
    success=bool(opened is not None and legal)
    assert success==result['success']
    assert abs(math.degrees(max_angle)-result['simplified_score']['max_abs_open_deg'])<1e-9
    assert max_delta<1e-7
    paths=list((out/'raw').glob('*.npz'));assert len(paths)==1
    frames=len(clock);errs=dict(state=0.,action=0.,previous_command=0.);views={}
    with np.load(paths[0],allow_pickle=True) as d:
        state=d['state'];action=d['action'];assert state.shape==action.shape==(frames,9)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        for i,c in enumerate(clock):
            rec=records[c['source_step']];assert c['frame']==i and c['physics_post_time_s']==rec['t']
            errs['state']=max(errs['state'],float(np.max(abs(state[i,2:]-np.array(rec['arm'])))))
            errs['action']=max(errs['action'],float(np.max(abs(action[i,2:]-np.array(rec['command'])))))
            prior=np.zeros(2) if i==0 else action[i-1,:2]
            errs['previous_command']=max(errs['previous_command'],float(np.max(abs(state[i,:2]-prior))))
        for view in ['front','wrist']:
            ims=d[view+'_masked_depth'];assert ims.shape==(frames,480,640) and ims.dtype==np.uint8
            for i,c in enumerate(clock):assert hashlib.sha256(ims[i].tobytes()).hexdigest()==c['raw_depth_sha256'][view]
            views[view]=dict(frames=frames,shape=list(ims.shape),dtype=str(ims.dtype),raw_buffer_sha_matches=True)
        assert int(d['record_sample_stride'])==2 and int(d['fps'])==25
    assert max(errs.values())<1e-6
    receipt=read(out/'raw-render-receipt.json')
    assert receipt['rendered_steps']==list(range(0,count,2))
    assert not list(out.rglob('*.mp4'))
    frozen=read(out/'frozen-trial.json');binding=read(out/'physical-camera-binding.json')
    return dict(index=frozen['index'],seed=frozen['seed'],success=success,
        max_abs_open_deg=math.degrees(max_angle),opened_80_time_s=opened,
        legal_relock_time_s=None if first_lock is None else first_lock['time_s'],
        final_legal_relocked=legal,whole_body_pass_time_s=passed,
        contact_separation_after_latch_s=separated_after_latch,phases=phases,
        expert_sequence_complete=result['expert_sequence_complete'],sim_seconds=result['sim_seconds'],steps=count,
        stop_reason=result['reason'],failure_reasons=result['simplified_score']['failures'],execution_error=result['error'],
        initial_state=read(out/'initial-state.json'),effective_randomization=binding['randomization'],
        actual_camera_poses=binding['camera_local_poses'],frames=frames,joint9_max_error=errs,
        views=views,raw_sha256=sha(paths[0]),lock_writes=lock_count,max_lock_write_delta=max_delta,
        raw_clock_and_render_verified=True,no_video=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    rows=[]
    for index in range(5):
        attempts=sorted((run/('trial_%02d'%index)).glob('attempt_[0-9][0-9][0-9]'))
        valid=[p for p in attempts if (p/'result.json').exists() and not read(p/'result.json')['error']]
        assert len(valid)==1
        row=audit(valid[0]);row['attempts']=[str(p.relative_to(run)) for p in attempts];rows.append(row)
    write(run/'audit.json',dict(successes=sum(x['success'] for x in rows),denominator=5,trials=rows))
    with (run/'results.csv').open('x') as f:
        keys=['index','seed','success','max_abs_open_deg','legal_relock_time_s','whole_body_pass_time_s','sim_seconds','stop_reason']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:x[k] for k in keys} for x in rows)
    print(json.dumps({'successes':sum(x['success'] for x in rows),'denominator':5,'results':[{k:x[k] for k in ['seed','success','max_abs_open_deg','legal_relock_time_s','whole_body_pass_time_s','stop_reason']} for x in rows]},indent=2))

if __name__=='__main__':main()
