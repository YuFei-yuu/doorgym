import json,sys,math
from pathlib import Path
import numpy as np
run=Path(sys.argv[1]);out=run/'attempt_000';dest=run/'review_final'
rows=[json.loads(x) for x in (out/'states.jsonl').open()];contacts=[json.loads(x) for x in (out/'contacts.jsonl').open()]
summary={};events=[];groups={};previous=None
for r,c in zip(rows,contacts):
 if r['phase'] not in ['deploy','approach','close','release','fold','restore_heading','final_hold']:continue
 step_pairs=[];end=False;other=False
 for x in c['contacts']:
  if not x.get('lambda',0):continue
  ns=x['names'];door=next((z for z in ns if z.get('actor')=='door_0'),None)
  robot=next((z for z in ns if z.get('actor','').startswith(('b1_base','z1_arm'))),None)
  if not door or not robot:continue
  key=robot['body']+' / '+door['body'];step_pairs.append(key)
  z=summary.setdefault(key,dict(records=0,steps=set(),first_s=r['t'],last_s=r['t'],max_abs_lambda=0.,phases=set()))
  z['records']+=1;z['steps'].add(r['step']);z['last_s']=r['t'];z['max_abs_lambda']=max(z['max_abs_lambda'],abs(x['lambda']));z['phases'].add(r['phase'])
  if robot['body'] in ['gripperStator','gripperMover'] and door['body']=='door_panel':end=True
  else:other=True
 if r['phase']=='close':
  kind='both' if end and other else 'end_only' if end else 'other_only' if other else 'no_loaded_door_contact'
  delta=r['pre_physics']['door_q'][0]-r['door_q'][0]
  z=groups.setdefault(kind,dict(steps=0,signed_door_drop_deg=0.,positive_door_drop_deg=0.))
  z['steps']+=1;z['signed_door_drop_deg']+=math.degrees(delta);z['positive_door_drop_deg']+=max(0.,math.degrees(delta))
 if step_pairs:events.append(dict(step=r['step'],t=r['t'],phase=r['phase'],door_deg=math.degrees(r['door_q'][0]),loaded_pairs=sorted(set(step_pairs)),ee=r['ee_actual']))
for x in summary.values():x['step_count']=len(x['steps']);x['steps']=sorted(x['steps']);x['phases']=sorted(x['phases'])
result=dict(loaded_pair_summary=summary,close_groups=groups,events=events,interpretation='Timing diagnostics only: simultaneous contact, momentum and constraints prevent causal attribution of angle change to any one body. No force/work/coverage threshold is applied.')
(dest/'closing-contact-audit.json').write_text(json.dumps(result,indent=2)+'\n')
# Original RGB frames around contact onset, loaded non-end events and closure.
import imageio
from PIL import Image,ImageDraw
first=next((r['t'] for r in rows if r['phase']=='close'),None);last=next((r['t'] for r in rows if r['latched'] is not None),None)
times=sorted(set([x for x in [first,last] if x is not None]+([events[0]['t'],events[-1]['t']] if events else [])+list(np.linspace(first or 0,last or rows[-1]['t'],12))))
reader=imageio.get_reader(str(out/'overview/env_0000/rollout.mp4'));tiles=[]
for t in times:
 i=min((len(rows)-1)//2,max(0,round((t-.02)/.04)));im=Image.fromarray(reader.get_data(i)).resize((960,320));ImageDraw.Draw(im).text((5,300),f'Original RGB frame {i}; physical sample t={(2*i+1)*.02:.2f}s',fill='yellow');tiles.append(im)
reader.close();sheet=Image.new('RGB',(1920,320*math.ceil(len(tiles)/2)),(240,240,240))
for i,im in enumerate(tiles):sheet.paste(im,((i%2)*960,(i//2)*320))
sheet.save(dest/'closing-keyframes.jpg',quality=92)
print(json.dumps({'pairs':{k:{n:v for n,v in x.items() if n!='steps'} for k,x in summary.items()},'close_groups':groups},indent=2))
