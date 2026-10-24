"""One bounded design probe (three constant heights); no controller/physics."""
import sys,os,json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_geometry import URDF,transform
from fullcycle_collision import collision_pieces
from fullcycle_compact import Arm,ClosingCurve
from newexpert_stage2.push_geometry import PadArm,PadCurve,PAD_LOCAL
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
c=json.loads(Path(__file__).with_name('geometry.json').read_text())['compact_expert'];robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');arm=PadArm(robot)
rows=[]
heights=[float(x) for x in sys.argv[2:]] or [.82,.85,.90]
for height in heights:
 for pitch in [0,2.5,5]:
  for z in [.47,.5,.53]:
   curve=PadCurve(dict(c,height=height));q=np.array([0,1.7,-1.3,-.4,0,0.])
   for theta in np.linspace(math.pi/2,0,31):
    ref=curve.at(theta);B=transform([*ref[:2],z],[0,math.radians(pitch),ref[2]])
    q,err=arm.solve(np.linalg.inv(B)@curve.tool(theta),q,1.,120)
    rows.append(dict(height=height,pitch=pitch,z=z,theta=theta,q=q.tolist(),**err))
summary={str(h):dict(max_position_m=max(r['position_error_m'] for r in rows if r['height']==h),max_axis_rad=max(r['orientation_error_rad'] for r in rows if r['height']==h),min_joint_slack=min(r['joint_slack_rad'] for r in rows if r['height']==h)) for h in heights}
pieces=collision_pieces(robot.path);poses=robot.fk(dict(zip(arm.names+["jointGripper"],[0]*7)));E=poses['ee_gripper_link'];geom=[]
for p in pieces:
 if p.link in ['gripperStator','gripperMover','link06','wrist_down_mount','wrist_up_mount']:
  vertices=p.world(poses[p.link])[0];v=(vertices-E[:3,3])@E[:3,:3]
  geom.append(dict(link=p.link,name=p.name,lo=v.min(0).tolist(),hi=v.max(0).tolist()))
d=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),argv=sys.argv,interpreter=sys.executable,cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),summary=summary,rows=rows,contact_piece_bounds_in_ee=geom,pad_local=PAD_LOCAL.tolist(),kind='bounded static design probe only')
(out/'probe.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(summary,indent=2))
