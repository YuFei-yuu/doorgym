import json,sys,math,collections
from pathlib import Path
run=Path(sys.argv[1]);a=run/'attempt_000';out=run/'review';rows=[json.loads(l) for l in (out/'compact-actual-trace.jsonl').open()];phases={};first=None;last=None;sample=[];counts=collections.Counter();angles=[]
for line in (a/'contacts.jsonl').open():
    c=json.loads(line);r=rows[c['step']];phase=r['phase']
    if phase not in ['deploy_a','deploy_b','approach','close','release','retract','restore_heading','final_hold']:continue
    for x in c['contacts']:
        names=[y['body'] for y in x['names']];lam=float(x.get('lambda',0))
        if lam==0:continue
        if 'door_panel' in names and any(y in names for y in ['gripperStator','gripperMover']):
            record=dict(step=c['step'],time_s=r['t'],phase=phase,door_deg=math.degrees(r['door_q'][0]),contact=x)
            if first is None:first=record
            last=record;counts[phase]+=1
            if len(sample)<4:sample.append(record)
        if x['classification']=='unexpected_external_loaded':sample.append(dict(step=c['step'],time_s=r['t'],phase=phase,offtarget=x))
result=dict(first_postdeploy_loaded_panel_contact=first,last_postdeploy_loaded_panel_contact=last,loaded_record_count_by_phase=dict(counts),samples=sample[:30],
    final_door_deg=math.degrees(rows[-1]['door_q'][0]),task_monitor=rows[-1]['task_monitor'],meaning='Loaded gripper-panel records, including approach. Backside/contact work must also use live RGB and raw geometry; close-only runtime counters omit approach contact.')
(out/'contact-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
