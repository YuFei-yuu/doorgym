"""Read-only evidence analysis; never starts Isaac Gym or replays state."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
import av
import numpy as np
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args()
    run=Path(a.run);attempt=run/'attempt_000';out=run/'audit';out.mkdir(exist_ok=False)
    states=[json.loads(l) for l in (attempt/'states.jsonl').open()]
    result=json.loads((attempt/'result.json').read_text())
    references=json.loads((attempt/'reference.json').read_text())
    bodymap=json.loads((attempt/'body-map.json').read_text())
    dt=states[0]['t'];cmd=np.array([s['action'] for s in states]);bases=np.array([s['command_base'] for s in states]);actual=np.array([s['actual_base'] for s in states])
    expected=np.empty_like(bases);initial=json.loads((attempt/'initial-state.json').read_text())
    xy=np.array(initial['base_start']);yaw=initial['yaw']
    for i,action in enumerate(cmd):
        xy=np.asarray(xy,dtype=np.float32)+np.asarray([math.cos(yaw),math.sin(yaw)],dtype=np.float32)*(float(action[0])*dt)
        yaw+=float(action[1])*dt;expected[i]=[*xy,yaw]
    robot_indices={int(k) for k,v in bodymap.items() if v['actor_handle'] in [0,1]}
    unexpected=[];pairs=Counter();contact_steps=0
    with (attempt/'contacts.jsonl').open() as f:
        for i,line in enumerate(f):
            row=json.loads(line)
            assert row['step']==i
            if row['contacts']:contact_steps+=1
            for c in row['contacts']:
                n0,n1=c['names'];pair=' / '.join(v['actor']+':'+v['body'] for v in [n0,n1] if 'actor' in v)
                pairs[pair]+=1
                if c['body0'] in robot_indices: rob,other=n0,n1
                elif c['body1'] in robot_indices: rob,other=n1,n0
                else: continue
                if other.get('actor_handle') in [0,1] or 'actor' not in other:continue
                if rob['body'] in ['gripperStator','gripperMover'] and other['actor_handle']==2 and other['body'] in ['door_panel','lever_handle']:continue
                unexpected.append(dict(step=row['step'],pair=pair,lambda_value=c.get('lambda'),initialOverlap=c.get('initialOverlap'),minDist=c.get('minDist'),normal=c.get('normal'),localPos0=c.get('localPos0'),localPos1=c.get('localPos1')))
    contacts_rows=i+1
    lockwrites=[json.loads(l) for l in (attempt/'lock-writes.jsonl').open()]
    stats=dict(steps=len(states),states_contiguous=[s['step'] for s in states]==list(range(len(states))),contacts_rows=contacts_rows,phase_counts=dict(Counter(s['phase'] for s in states)),max_door_angle_deg=float(np.rad2deg(max(s['door_q'][0] for s in states))),max_handle_angle_deg=float(np.rad2deg(max(s['door_q'][1] for s in states))),unlocked_steps=sum(s['unlocked'] for s in states),pass_steps=sum(s['passed'] for s in states),new_closing_steps=sum(s['phase']=='close' for s in states),relock_writes=sum(x['kind']=='relocked' for x in lockwrites),max_initial_lock_angle_correction=max([abs(x['delta']) for x in lockwrites]+[0]),old_yaw_reference_rad=references['old_expert_final_yaw'],joint9_command_integrator_max_error=float(np.max(np.abs(expected-bases))),actual_command_xy_max_error_m=float(np.max(np.linalg.norm(actual[:,:2]-bases[:,:2],axis=1))),max_command_joint_increment_rad=np.max(np.abs(np.diff(cmd[:,2:],axis=0)),axis=0).tolist(),max_arm_position_command_error_rad=np.max(np.abs(np.array([s['arm_q'] for s in states])-cmd[:,2:]),axis=0).tolist(),max_ee_target_position_error_m=max(float(np.linalg.norm(np.array(s['ee_target'])-s['ee_actual'])) for s in states),unexpected_contact_records=unexpected,contact_pair_counts=dict(pairs),stop=result)
    videos=[];tiles=[]
    for path in [attempt/'overview/env_0000/rollout.mp4',attempt/'front_depth.mp4',attempt/'wrist_depth.mp4']:
        container=av.open(str(path));stream=container.streams.video[0]
        fps=float(stream.average_rate);n=0;times=[];selected_frames={}
        wanted=set(np.linspace(0,result['video_frames']['0']-1,8,dtype=int).tolist())
        for frame in container.decode(video=0):
            times.append(float(frame.pts*frame.time_base))
            if n in wanted and 'rollout' in path.name:
                im=frame.to_image();im.thumbnail((960,320));canvas=Image.new('RGB',(960,346),'white');canvas.paste(im,(0,26));ImageDraw.Draw(canvas).text((10,5),'ORIGINAL LIVE RGB | frame %d | %.2fs'%(n,n/25),fill='black');selected_frames[n]=canvas
            n+=1
        container.close()
        videos.append(dict(path=str(path),fps=fps,decoded_frames=n,duration_s=n/fps,expected_frames=math.ceil(len(states)/2),complete=bool(n==math.ceil(len(states)/2) and abs(fps-25)<1e-9),timestamps_regular=bool(len(times)<2 or np.max(np.abs(np.diff(times)-.04))<1e-6),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if selected_frames: tiles=[selected_frames[i] for i in sorted(selected_frames)]
    if tiles:
        sheet=Image.new('RGB',(1920,346*math.ceil(len(tiles)/2)),'white')
        for i,t in enumerate(tiles):sheet.paste(t,((i%2)*960,(i//2)*346))
        sheet.save(out/'original-rgb-contact-sheet.jpg',quality=90)
    stats['video_validation']=videos
    stats['evidence_complete']=bool(stats['states_contiguous'] and contacts_rows==len(states) and len(states)==result['steps'] and all(v['complete'] for v in videos))
    stats['contact_interpretation']='Stop is conservative: a PhysX contact manifold may be within the configured contact distance. Zero lambda does not establish impact or a force-bearing collision. Raw fields are preserved; do not infer force from initialOverlap alone.'
    stats['depth_stage_disambiguation']='UNVERIFIED: no actual closing phase occurred, so no opening/closing paired observations exist.'
    (out/'audit.json').write_text(json.dumps(stats,indent=2)+'\n')
    ledger=dict(attempt=0,status='stopped',success=False,reason=result['reason'],sim_steps=len(states),sim_seconds=result['sim_seconds'],evidence_complete=stats['evidence_complete'],subsequent_attempts=0)
    with (run/'attempts.jsonl').open('a') as f:f.write(json.dumps(ledger)+'\n')
    print(json.dumps({k:stats[k] for k in ['steps','phase_counts','max_door_angle_deg','unlocked_steps','joint9_command_integrator_max_error','actual_command_xy_max_error_m','max_command_joint_increment_rad','evidence_complete','unexpected_contact_records']}))


if __name__=='__main__':main()
