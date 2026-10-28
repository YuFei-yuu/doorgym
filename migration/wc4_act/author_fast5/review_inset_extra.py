"""Read-only target and unchanged-base audit; no scene creation."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

root = Path('/data/zyf/video2door-wc4-act')
sys.path.insert(0, str(root / 'migration/wc4_act'))
from fullcycle_geometry import URDF, transform

run = Path(sys.argv[1])
out = run / 'attempt_000'
review = run / 'review_final'
rows = [json.loads(line) for line in (out / 'states.jsonl').open()]
baseline = root / '.repro/runs/wc4-slow15-base-profile-20261028'
old = [json.loads(line) for line in (baseline / 'attempt_000/states.jsonl').open()]
common = min(len(rows), len(old))
diffs = {key: float(np.max(abs(np.asarray([r[key] for r in rows[:common]]) - np.asarray([r[key] for r in old[:common]])))) for key in ['actual_base', 'base_dof_actual']}
robot = URDF(root / 'high-level/data/asset/a2wz1/urdf/a2wz1.urdf')
effective = json.loads((out / 'effective-config.json').read_text())
ref = json.loads((out / 'reference.json').read_text())
err = 0.
count = 0
for r in rows:
    if r['expert_target'].get('mode') == 'joint':
        B = r['command_base']
        F = robot.fk(dict(zip(ref['dof_names'], r['arm_command'])), transform([*B[:2], effective['robot_z']], [0, effective['robot_pitch'], B[2]]))['ee_gripper_link']
        err = max(err, float(np.max(abs(F[:3, 3] - r['ee_target']))))
        count += 1
assert err < 1e-9
(review / 'reference-target-audit.json').write_text(json.dumps(dict(raw_log_unchanged=True, explicit_target_mode=True, joint_mode_reference_records=count, logged_fk_error_m=err, limitation='Joint-mode curves use applied joint FK; Cartesian curves use IK targets. Target-switch finite differences are not physical EE speed.'), indent=2) + '\n')
cfg = json.loads((out / 'effective-expert.json').read_text())
oldcfg = json.loads((baseline / 'attempt_000/effective-expert.json').read_text())
changes = {k: [oldcfg.get(k), cfg.get(k)] for k in sorted(set(cfg) | set(oldcfg)) if oldcfg.get(k) != cfg.get(k)}
info = dict(common_physics_steps=common, maximum_difference=diffs, config_changes=changes, joint_mode_reference_records=count, logged_fk_error_m=err)
(review / 'inset-extra-audit.json').write_text(json.dumps(info, indent=2) + '\n')
video = out / 'overview/env_0000/rollout.mp4'
v = json.loads((review / 'video-check.json').read_text())
v['sha256'] = hashlib.sha256(video.read_bytes()).hexdigest()
(review / 'video-check.json').write_text(json.dumps(v, indent=2) + '\n')
print(json.dumps(info, indent=2))
