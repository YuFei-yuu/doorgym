import json,hashlib,sys
from pathlib import Path
import numpy as np
run=Path(sys.argv[1]);out=run/'attempt_000';review=run/'review_final'
rows=[json.loads(l) for l in (out/'states.jsonl').open()];clock=[json.loads(l) for l in (out/'recording-clock.jsonl').open()]
contact=[json.loads(l) for l in (out/'contacts.jsonl').open()];n=len(rows);frames=len(clock)
errors=[];state_error=action_error=prior_error=0.;hashes={'front':True,'wrist':True}
path=next((out/'raw').glob('*.npz'))
with np.load(path,allow_pickle=True) as archive:
 d={k:archive[k] for k in ['state','action','replay_root_state','replay_dof_pos','replay_dof_vel','front_masked_depth','wrist_masked_depth']}
 assert d['state'].shape==d['action'].shape==(frames,9)
 for k in ['state','action','replay_root_state','replay_dof_pos','replay_dof_vel']:assert np.isfinite(d[k]).all()
 for i,c in enumerate(clock):
  step=c['source_step'];r=rows[step]
  assert c['frame']==i and step==2*i and c['physics_post_time_s']==r['t']
  state_error=max(state_error,float(np.max(abs(d['state'][i,2:]-np.array(r['arm_q'])))))
  action_error=max(action_error,float(np.max(abs(d['action'][i,2:]-np.array(r['arm_command'])))))
  prior=np.zeros(2) if i==0 else d['action'][i-1,:2]
  prior_error=max(prior_error,float(np.max(abs(d['state'][i,:2]-prior))))
  assert np.array_equal(d['replay_dof_pos'][i,:12],np.zeros(12))
  for view in hashes:
   im=d[view+'_masked_depth'][i];assert im.shape==(480,640) and im.dtype==np.uint8
   hashes[view]&=hashlib.sha256(im.tobytes()).hexdigest()==c['raw_depth_sha256'][view]
 assert all(hashes.values()) and max(state_error,action_error,prior_error)<1e-6
robot_contact={};ground={}
for row,c in zip(rows,contact):
 for x in c['contacts']:
  if x.get('lambda',0)==0:continue
  names=x['names']
  for i,a in enumerate(names):
   b=names[1-i];body=a.get('body','')
   if body.endswith('_wheel'):
    group=ground if b.get('body')=='ground_or_unknown' else robot_contact
    key=body+' / '+b.get('body','');z=group.setdefault(key,dict(records=0,first_s=row['t'],last_s=row['t'],max_abs_lambda=0.))
    z['records']+=1;z['last_s']=row['t'];z['max_abs_lambda']=max(z['max_abs_lambda'],abs(float(x['lambda'])))
base=np.array([r['actual_base'] for r in rows if r['phase'] in ['traverse_fold_finish','turn_for_close','deploy','approach']])
res=dict(states=n,contacts=len(contact),frames=frames,state_joint_error=state_error,action_joint_error=action_error,previous_25hz_base_command_error=prior_error,
 raw_depth_hashes_match=hashes,leg_zero_placeholder_preserved=True,all_finite=True,raw_npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
 observed_preparation_base_xy_max_difference_m=float(np.max(abs(base[:,:2]-base[0,:2]))),wheel_ground_loaded_records=ground,wheel_other_loaded_records=robot_contact,
 video_expected_frames=(n+1)//2,source='raw original physical logs and saved author recorder buffers')
(review/'data-audit.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
