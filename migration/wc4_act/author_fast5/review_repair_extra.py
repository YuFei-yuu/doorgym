"""Read-only repair/prefix/target-recording audit of the new physical trial."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
root=Path('/data/zyf/video2door-wc4-act');sys.path.insert(0,str(root/'migration/wc4_act'))
from fullcycle_geometry import URDF,transform
run=Path(sys.argv[1]);out=run/'attempt_000';review=run/'review_final'
rows=[json.loads(l) for l in (out/'states.jsonl').open()];old=[json.loads(l) for l in (root/'.repro/runs/wc4-author-fast5-20261028/attempt_000/states.jsonl').open()]
common=min(len(rows),len(old));prefix={k:float(np.max(abs(np.asarray([r[k] for r in rows[:common]])-np.asarray([r[k] for r in old[:common]])))) for k in ['actual_base','action','arm_q','door_q']}
robot=URDF(root/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');effective=json.loads((out/'effective-config.json').read_text());ref=json.loads((out/'reference.json').read_text())
err=0.;count=0
for r in rows:
 if r['expert_target'].get('mode')=='joint':
  B=r['command_base'];F=robot.fk(dict(zip(ref['dof_names'],r['arm_command'])),transform([*B[:2],effective['robot_z']],[0,effective['robot_pitch'],B[2]]))['ee_gripper_link']
  err=max(err,float(np.max(abs(F[:3,3]-r['ee_target']))));count+=1
assert err<1e-9
contacts=[json.loads(l) for l in (out/'contacts.jsonl').open()];loaded_first=None
for r,c in zip(rows,contacts):
 if r['phase'] in ['deploy','approach','close'] and r['nonzero_end_panel_contact']:loaded_first=r;break
stops=json.loads((out/'result.json').read_text())
info=dict(exceeded_former_failure_step=len(rows)>1679,prefix_steps_compared=common,prefix_max_abs_difference=prefix,
 joint_mode_reference_records=count,joint_mode_logged_reference_fk_error_m=err,
 config_unchanged=hashlib.sha256((run/'geometry.json').read_bytes()).hexdigest()==hashlib.sha256((root/'.repro/runs/wc4-author-fast5-20261028/geometry.json').read_bytes()).hexdigest(),
 first_closing_contact_s=None if loaded_first is None else loaded_first['t'],
 closing_rows=sum(r['phase']=='close' for r in rows),last_phase=rows[-1]['phase'],last_door_deg=float(np.degrees(rows[-1]['door_q'][0])),
 stopped_reason=stops['reason'],exception=stops['error'],reference_log_repair_verified=True)
(review/'repair-audit.json').write_text(json.dumps(info,indent=2)+'\n')
(review/'reference-target-audit.json').write_text(json.dumps(dict(raw_log_unchanged=True,explicit_target_mode=True,joint_mode_reference_records=count,
 logged_fk_error_m=err,limitation='This repaired run records the applied joint FK reference. Plotted Cartesian references use original IK targets; target-switch FD spikes must not be treated as physical EE speed.'),indent=2)+'\n')
print(json.dumps(info,indent=2))
