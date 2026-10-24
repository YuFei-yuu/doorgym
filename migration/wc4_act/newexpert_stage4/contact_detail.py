"""Read-only supplementary contact face, latch and terminal details."""
import argparse, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
r=a.run;o=r/'audit';states=[json.loads(l) for l in (r/'attempt_000/states.jsonl').open()]
events=[json.loads(l) for l in (o/'post-pass-door-contacts.jsonl').open()]
groups={}
for e in events:
    if e['lambda']!=0:groups.setdefault(e['phase']+' | '+e['pair'],[]).append(e)
report={'loaded_contact_groups':{k:{'records':len(v),'first_step':v[0]['step'],'last_step':v[-1]['step'],
    'first_time_s':v[0]['time_s'],'last_time_s':v[-1]['time_s']} for k,v in groups.items()}}
points=[]
for e in events:
    if e['lambda']==0 or e['phase']!='close' or not e['end_panel']:continue
    c=e['raw_contact'];j=next(i for i,x in enumerate(c['names']) if x['body']=='gripperStator')
    points.append([c['localPos'+str(j)][k] for k in 'xyz'])
v=np.array(points)
report['close_stator_local_contact']={'records':len(v),'min_xyz_m':v.min(0).tolist(),'max_xyz_m':v.max(0).tolist(),
    'source_box_center_m':[.135,0,-.020],'source_box_size_m':[.030,.060,.020],'intended_positive_x_face_m':.150,
    'max_distance_from_source_x_face_m':float(abs(v[:,0]-.15).max()),
    'meaning':'Raw reported points near distal fixed-finger outward face; collision shape IDs are absent, so source-face matching is geometric support, not exact cooked-shape attribution.'}
locks=[json.loads(l) for l in (r/'attempt_000/lock-writes.jsonl').open()]
rel=next(x for x in locks if x['kind']=='relocked');s=states[rel['step']-1]
report['latch']={'step':rel['step'],'pre_time_s':rel['time_s'],'latched_rad':rel['latched'],
    'previous_measured_rad':s['door_q'][0],'first_write_delta':rel['delta'],
    'stored_previous_actual_exactly':rel['latched']==s['door_q'][0] and rel['delta']==0}
report['final']={k:states[-1][k] for k in ['t','arm_qd','base_measured_fd_speed','base_measured_fd_yaw_rate',
    'whole_geometry_max_x','pass_plane_x','door_q','door_qd']}
report['budget']={'scene_claims':1,'episodes':1,'physics_steps':len(states),'sim_seconds':states[-1]['t'],
    'unused_sim_seconds':150-states[-1]['t'],'additional_episodes':0,'training_updates':0,'policy_evaluations':0}
target=o/'contact-and-terminal-detail.json'
value=json.dumps(report,indent=2)+'\n'
if target.exists():
    assert json.loads(target.read_text())==report,'Supplementary result differs from original stdin audit'
    (o/'contact-detail-reproducibility.json').write_text(json.dumps({'matches_original_stdin_audit':True})+'\n')
else:target.write_text(value)
print(value)
