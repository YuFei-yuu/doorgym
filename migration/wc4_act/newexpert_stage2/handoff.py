"""Read-only historic handoff audit; never a new physical episode."""
import hashlib,json,sys,math,subprocess,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_geometry import URDF,transform
from fullcycle_collision import collision_pieces,signed_gap
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
out=Path(sys.argv[1]);out.mkdir(exist_ok=False,parents=True)
src=ROOT/'.repro/runs/wc4-fullcycle-compact-r7-20261021/attempt_000'
rows=[];first=None
with (src/'states.jsonl').open() as f:
 for line in f:
  r=json.loads(line)
  if first is None and r['passed']:first=r['step']
  if first is not None:
   if r['step']>first+220:break
   rows.append(r)
assert rows
(out/'historical-states-excerpt.json').write_text(json.dumps(rows)+'\n')
raw=[];pairs={}
with (src/'contacts.jsonl').open() as f:
 for line in f:
  r=json.loads(line)
  if r['step']<first:continue
  if r['step']>rows[-1]['step']:break
  selected=[]
  for c in r['contacts']:
   bodies=[n.get('body','') for n in c.get('names',[])]
   if 'door_panel' in bodies and any(b in bodies for b in ['wrist_down_mount','wrist_up_mount','link06','gripperStator','gripperMover']):
    selected.append(c);key='/'.join(bodies);p=pairs.setdefault(key,dict(records=0,nonzero_lambda_records=0,first_step=r['step'],last_step=r['step'],max_abs_lambda=0.))
    p['records']+=1;p['last_step']=r['step'];p['max_abs_lambda']=max(p['max_abs_lambda'],abs(c['lambda']));p['nonzero_lambda_records']+=int(abs(c['lambda'])>0)
  raw.append(dict(step=r['step'],interval_s=r.get('interval_s'),contacts=selected))
(out/'historical-contacts-excerpt.json').write_text(json.dumps(raw)+'\n')
robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');door=URDF(ROOT/'high-level/data/asset/door_set/wc4/model.urdf')
R=[p for p in collision_pieces(robot.path) if p.link in ['wrist_down_mount','wrist_up_mount','link06','gripperStator','gripperMover']]
D=[p for p in collision_pieces(door.path) if p.link=='door_panel']
samples=[]
for r in rows[::5]:
 b=r['actual_base'];rp=robot.fk(dict(zip(['joint%d'%i for i in range(1,7)]+['jointGripper'],r['arm_q'])),transform([*b[:2],.5],[0,0,b[2]]))
 dp=door.fk({'door_hinge_joint':r['door_q'][0],'handle_spindle_joint':r['door_q'][1]},transform([2.61,0,.01],[0,0,math.pi/2]))
 gaps={}
 for p in R:
  cat='end' if p.link.startswith('gripper') else 'wrist';A=p.world(rp[p.link])
  for d in D:
   g=signed_gap(A,d.world(dp[d.link]))
   if cat not in gaps or g<gaps[cat]['gap_m']:gaps[cat]=dict(gap_m=g,link=p.link,piece=p.name,door_piece=d.name)
 samples.append(dict(step=r['step'],t=r['t'],phase=r['phase'],base=b,q=r['arm_q'],minimums=gaps))
checks=dict(source=str(src),historical_only=True,new_physical_episodes=0,first_step=first,last_step=rows[-1]['step'],contact_pairs=pairs,
    phases=[dict(step=r['step'],t=r['t'],phase=r['phase']) for i,r in enumerate(rows) if i==0 or r['phase']!=rows[i-1]['phase']],
    samples=samples,interpreter=sys.executable,argv=sys.argv,
    limitations=['Historical raw contact lambda is diagnostic, not alone a task success/failure criterion.', 'FK reconstruction and conservative source convex overlap are distinct from raw physical contact records.', 'No new physical validation of modified contact geometry or extraction; initial inherited commands remain unchanged.'])
(out/'handoff.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps({k:v for k,v in checks.items() if k!='samples'},indent=2))
