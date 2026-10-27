import json,sys,collections,hashlib
from pathlib import Path
import numpy as np
run=Path(sys.argv[1]);summary={};contacts={}
for i in range(2):
 p=run/('trial_%02d'%i)/'attempt_000';result=json.loads((p/'result.json').read_text());clock=[json.loads(l) for l in (p/'recording-clock.jsonl').open()];n=(result['steps']+1)//2
 z=np.load(p/'raw/episode_000000.npz',allow_pickle=True);shapes={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in ['state','action','wrist_masked_depth','front_masked_depth']}
 assert z['state'].shape==z['action'].shape==(n,9)
 assert np.isfinite(z['state']).all() and np.isfinite(z['action']).all()
 assert all(z[k].shape==(n,480,640) and z[k].dtype==np.uint8 for k in ['wrist_masked_depth','front_masked_depth'])
 assert [r['source_step'] for r in clock]==list(range(0,result['steps'],2))
 counts=collections.Counter();examples={};loaded_steps={side:set() for side in ['FL','FR','RL','RR']}
 for line in (p/'contacts.jsonl').open():
  row=json.loads(line)
  for c in row['contacts']:
   ns=[o['body'] for o in c['names']]
   if float(c['lambda']) and any(n.startswith(('FL_','FR_','RL_','RR_')) for n in ns):
    key=' / '.join(ns);counts[key]+=1
    if key not in examples:examples[key]=dict(time_s=row['interval_s'][1],record=c)
   if min(int(c['body0']),int(c['body1']))<0 and float(c['lambda']):
    for name in ns:
     if name[:2] in loaded_steps and name.endswith('_wheel'):loaded_steps[name[:2]].add(row['step'])
 contacts[str(i)]=dict(loaded_leg_contacts=dict(counts),first_records=examples,ground_loaded_unique_steps={k:len(v) for k,v in loaded_steps.items()})
 summary[str(i)]=dict(frames=n,shapes=shapes,clock_contiguous=True,joint9_finite=True,training_leg_placeholders_all_zero=bool(np.all(z['replay_dof_pos'][:,:12]==0)),actual_legs_source='states.jsonl base_dof_actual and base_body_states, not training placeholders',model=None,training=0,conversion=0,physics_episodes=1,result=result)
 z.close()
(run/'observation/recording-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
(run/'observation/leg-contact-evidence.json').write_text(json.dumps(contacts,indent=2)+'\n')
print(json.dumps(dict(recordings=summary,contacts=contacts),indent=2))
