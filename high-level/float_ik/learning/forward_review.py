"""Full video decode and paired same-initial-state evidence for the 5000 review."""
import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

from deliver_v3 import paired_video


def review(root):
    root=Path(root);suite=root/'evaluation/step_005000'
    rows=json.loads((suite/'results.json').read_text())
    out=root/'analysis/video_review';out.mkdir(parents=True,exist_ok=True)
    decoded=[];errors=[]
    for r in rows:
        video=Path(r['folder'])/'rollout.mp4'
        if not video.exists():
            errors.append(dict(seed=r['seed'],error='Video missing'));continue
        code=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-threads','2',
                             '-i',str(video),'-f','null','-']).returncode
        if code:errors.append(dict(seed=r['seed'],error='Video decode failed',exit_code=code))
        else:decoded.append(r['seed'])
    selected=[r for r in rows if r['robot_x']==4.1 and r['robot_y']==0.]
    for success in (True,False):
        candidate=next((r for r in rows if r['success']==success and r not in selected),None)
        if candidate:selected.append(candidate)
    comparisons=[]
    for r in selected:
        expert=root/'dataset'/('val_seed_%d'%r['seed']);policy=Path(r['folder'])
        if (expert/'rollout.mp4').exists() and (policy/'rollout.mp4').exists():
            pair_out=out/('seed_%d'%r['seed']);pair_out.mkdir(exist_ok=True)
            comparisons.append(paired_video(expert,policy,pair_out,updates=5000))
    (out/'review.json').write_text(json.dumps(dict(decoded_seeds=decoded,errors=errors,
             comparisons=comparisons,human_review_required=True,objective_complete=False),indent=2))
    print(json.dumps(dict(decoded=len(decoded),errors=errors,comparisons=len(comparisons))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();review(a.root)
