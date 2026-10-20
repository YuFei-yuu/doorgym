"""Independent raw-log criterion, original live video selection, no extra trials."""
import argparse,hashlib,json,math,shutil
from collections import defaultdict
from pathlib import Path
import av,numpy as np
from common import *
a=argparse.ArgumentParser();a.add_argument('step',type=int);a.add_argument('batch',type=int);a.add_argument('--recorded',action='store_true');args=a.parse_args()
CP=RUN/'checkpoints'/f'{args.step:06d}';directory=CP/'evaluation'/f'batch_{args.batch:03d}'
records=defaultdict(list)
for line in (directory/'batch_0000.jsonl').read_text().splitlines():
 row=json.loads(line);records[int(row['controlled_env_id'])].append(row)
assert sorted(records)==list(range(16))
selection=json.loads((CP/'video-selection.json').read_text()) if (CP/'video-selection.json').exists() else {}
results=[];disposition=[]
manifest=CP/'door-checkpoint/model_latest.pt';model_sha=json.loads((RUN/f'checkpoint-ready-{args.step:06d}.json').read_text())['model_sha256']
for env,rows in sorted(records.items()):
 assert [v['step'] for v in rows]==list(range(1100))
 assert all(v['dp_action_names']==['vx','yaw','joint1','joint2','joint3','joint4','joint5','joint6','jointGripper'] for v in rows)
 assert all(v['phase_name']=='dp_policy' and v['action_mode']=='a2w_joint_action9' and len(v['dp_action_raw'])==len(v['state'])==9 for v in rows)
 assert all(np.isfinite(v['dp_action_raw']).all() and np.isfinite(v['state']).all() for v in rows)
 angles=np.array([abs(math.degrees(v['door']['dof'][0])) for v in rows]);assert np.isfinite(angles).all()
 randomization=rows[0]['ikpush_randomization'];assert randomization['enabled'] and 0<=randomization['robot_pitch']<=np.deg2rad(5)
 maximum=float(angles.max());success=maximum>=80;role='success' if success else 'failure'
 item={'checkpoint_step':args.step,'trial_index':args.batch*16+env,'batch_index':args.batch,'seed':615455575+args.batch,'env_id':env,'env_seed':randomization['env_seed'],'success':success,'max_abs_open_deg':maximum,'records':1100,'phase':'dp_policy','checkpoint_manifest':str(manifest),'checkpoint_manifest_sha256':sha(manifest),'model_sha256':model_sha,'log':str(directory/'batch_0000.jsonl')}
 if args.recorded:
  video=directory/'batch_0000_overview'/f'env_{env:04d}'/'rollout.mp4';receipt=json.loads(video.with_name('recording.json').read_text())
  assert receipt['complete'] and receipt['frames']==receipt['expected_frames']==550 and receipt['source_indices']==list(range(0,1100,2))
  digest=sha(video);entry={'trial_index':item['trial_index'],'source':str(video),'sha256':digest,'role':role,'frames':550}
  if role not in selection:
   with av.open(str(video)) as container:
    stream=container.streams.video[0];assert float(stream.average_rate)==25 and (stream.width,stream.height)==(1920,640)
    pts=[float(frame.pts*frame.time_base) for frame in container.decode(stream)]
   assert len(pts)==550 and np.max(np.abs(np.array(pts)-np.arange(550)/25))<1e-6
   dest=CP/'videos'/role;dest.mkdir(parents=True,exist_ok=False)
   video.rename(dest/'rollout.mp4');shutil.copy2(video.with_name('recording.json'),dest/'source-recording.json')
   binding={**item,'video':str(dest/'rollout.mp4'),'video_sha256':digest,'frames':550,'fps':25,'duration_s':22,'origin':'Synchronous original policy execution within the 64 trials; no replay or rerun'}
   save(dest/'binding.json',binding);selection[role]=binding;entry['disposition']='retained';entry['retained_as']=str(dest/'rollout.mp4')
  else:
   # Only this run's newly created temporary candidate clip is discarded.
   # All historical evidence, raw logs, recording receipts and clip hashes remain.
   assert RUN in video.parents and video.is_file();video.unlink();entry['disposition']='temporary candidate not retained; class already represented'
  disposition.append(entry)
 results.append(item)
save(CP/'video-selection.json',selection)
if args.recorded:save(directory/'temporary-video-disposition.json',disposition)
out={'status':'passed','step':args.step,'batch':args.batch,'trials':16,'successes':sum(v['success'] for v in results),'records':17600,'recorded_candidates':args.recorded,'selected_video_classes':sorted(selection),'criterion':'max absolute logged hinge angle >=80 degrees','no_expert_fallback_observed':True,'results':results}
save(directory/'independent-check.json',out);print(json.dumps({k:v for k,v in out.items() if k!='results'},indent=2))

import sys,subprocess
subprocess.run([sys.executable,'-B',str(RUN/'tools/log_eval_progress.py'),str(args.batch)],check=True)
