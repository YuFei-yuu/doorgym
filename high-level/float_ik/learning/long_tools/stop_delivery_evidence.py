"""CPU-only evidence from the saved failed attempt and recorded video anchors."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root).resolve()
failure=r/'test/untrained_seed_5007'
summary=json.loads((failure/'summary.json').read_text())
log=r/'logs/0166_test_untrained_seed_5007.log'
lines=log.read_text().splitlines()
matches=[dict(line=i+1,text=line) for i,line in enumerate(lines) if 'internal error' in line or 'illegal memory access' in line]
with np.load(failure/'trajectory.npz') as z:
    actions={key:dict(shape=list(z[key].shape),all_finite=bool(np.isfinite(z[key]).all()),max_abs=float(np.abs(z[key]).max())) for key in ('raw_action','action','rejected_or_interrupted_action')}
    assert all(value['all_finite'] for value in actions.values())
    assert actions['action']['shape']==[294,10] and actions['rejected_or_interrupted_action']['shape']==[1,10]
tests=[json.loads(path.read_text()) for path in (r/'test').glob('*/summary.json')]
result=dict(episode='test/untrained_seed_5007',classification='PhysX CUDA failure followed by nonfinite observation assertion',
    root_cause_confirmed=False,failed_post_state_saved=False,exact_nonfinite_channels_known=False,
    completed_transitions=summary['steps_completed'],attempted_transition=summary['steps_completed']+1,
    saved_actions=actions,log=str(log.relative_to(r)),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    first_error_messages=matches[:8],illegal_access_messages=[x for x in matches if 'illegal memory access' in x['text']][:4],
    traceback=summary['traceback'],test_counts=dict(planned=120,started=len(tests),completed=sum(x['status']=='completed' for x in tests),not_started=120-len(tests)),
    no_retry=True,remaining_budget_unused_due_to_stop_rule=True)
(r/'stop_diagnostics.json').write_text(json.dumps(result,indent=2))
alignment=[]
for video in json.loads((r/'video_checks.json').read_text()):
    folder=Path(video['path']).parent
    with (folder/'metrics.csv').open() as f: metrics={int(row['step']):row for row in csv.DictReader(f)}
    anchors=[]
    for frame,step in ((0,2),(712,1426),(1424,2850)):
        row=metrics[step]
        anchors.append(dict(frame=frame,simulation_step=step,simulation_time_sec=step/60,
            expected_time_overlay=f'{step/60:.2f}s',expected_door_overlay=f'{float(row["signed_angle_deg"]):.1f}deg',
            expected_success_overlay=row['success'],metrics=row,
            image=str((folder/f'video_check_{frame:04d}.png').relative_to(r))))
    alignment.append(dict(video=str(Path(video['path']).relative_to(r)),decoded=video['passed'],
        mapping='zero-based video frame f captures completed simulation step 2*(f+1)',anchors=anchors))
(r/'video_frame_alignment.json').write_text(json.dumps(alignment,indent=2))
print(json.dumps(dict(test_counts=result['test_counts'],saved_actions=actions,video_anchors=len(alignment)*3)))
