"""Targeted final-mode deployment directly from a recorded physical folded pose."""
import json,sys,os,math,hashlib,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_compact import CompactExpert
from fullcycle_geometry import URDF
from newexpert_stage2.push_geometry import PAD_LOCAL
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
out=Path(sys.argv[1]);out.mkdir(exist_ok=False,parents=True)
p=ROOT/'.repro/runs/wc4-fullcycle-compact-r7-20261021/attempt_000/states.jsonl'
with p.open() as f:
 b=next(r for r in map(json.loads,f) if r['phase']=='compact_rendezvous')
(out/'recorded-start.json').write_text(json.dumps(dict(source=str(p),row=b),indent=2)+'\n')
c=json.loads(Path(__file__).with_name('geometry.json').read_text())['compact_expert'];robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');q=np.array(b['arm_q']);base=np.array(b['actual_base']);home=np.r_[np.zeros(6),-1.1775]
x=CompactExpert(robot,c,base,q,b['action'],b['door_q'][0],0,home,math.pi,.02);x.phase='compact_rendezvous'
rows=[]
for k in range(1000):
 t=k*.02;E=x.scene.base_transform(base)@x.arm.fk(q);pad=E[:3,3]+E[:3,:3]@PAD_LOCAL;target=x.curve.tool(math.pi/2,c['hover_distance']);goalpad=target[:3,3]+target[:3,:3]@PAD_LOCAL
 gate=dict(pad_error_m=float(np.linalg.norm(pad-goalpad)),axis_error_rad=float(np.arccos(np.clip(np.dot(E[:3,0],target[:3,0]),-1,1))),v_previous=x.v)
 a,d=x.command(t,base,q,b['door_q'][0],0,False,False,False);base[:2]+=a[0]*.02*np.array([math.cos(base[2]),math.sin(base[2])]);base[2]+=a[1]*.02;q=a[2:].copy()
 rows.append(dict(t=t,phase=x.phase,base=base.tolist(),q=q.tolist(),action=a.tolist(),gate_pre=gate,pad=pad.tolist(),diagnostics=d))
 if x.phase=='compact_seek':break
(out/'trace.json').write_text(json.dumps(rows)+'\n')
r=dict(complete=x.phase=='compact_seek',interpreter=sys.executable,argv=sys.argv,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),initial_measured_q=b['arm_q'],initial_command_q=b['action'][2:],seconds=rows[-1]['t'],arrival=rows[-1]['gate_pre'],physical_episodes=0,meaning='Recorded physical folded pose retained directly; subsequent feedback is synthetic exact joint tracking; this is a targeted deployment check, not another full physical episode.',geometry_sha256=hashlib.sha256(Path(__file__).with_name('geometry.json').read_bytes()).hexdigest())
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
assert r['complete'] and r['arrival']['pad_error_m']<.008 and r['arrival']['axis_error_rad']<.01
