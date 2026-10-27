import json,sys,hashlib
from pathlib import Path
import numpy as np
run=Path(sys.argv[1]);a=run/'attempt_000';out=run/'review';clock=[json.loads(l) for l in (a/'recording-clock.jsonl').open()];z=np.load(a/'raw/episode_000000.npz',allow_pickle=False);n=len(clock)
r=dict(frames=n,state_shape=list(z['state'].shape),action_shape=list(z['action'].shape),state_dtype=str(z['state'].dtype),action_dtype=str(z['action'].dtype),views={},failures=[])
for view in ['front','wrist']:
    im=z[view+'_masked_depth'];bad=[i for i in range(n) if hashlib.sha256(im[i].tobytes()).hexdigest()!=clock[i]['raw_depth_sha256'][view]]
    r['views'][view]=dict(shape=list(im.shape),dtype=str(im.dtype),hash_mismatch_frames=bad)
    if bad or im.dtype!=np.uint8 or im.shape!=(n,480,640):r['failures'].append(view)
s=z['state'];u=z['action'];action_error=max(np.max(abs(u[i]-np.array(clock[i]['action']))) for i in range(n));r['action_clock_max_error']=float(action_error)
r['actual_joint_replay_max_error']=float(np.max(abs(s[:,2:]-z['replay_dof_pos'][:,12:19])))
r['previous_25hz_command_max_error']=float(np.max(abs(s[1:,:2]-u[:-1,:2])))
r['initial_previous_command']=s[0,:2].tolist();r['finite']=bool(np.isfinite(s).all() and np.isfinite(u).all())
if action_error>1e-7 or r['actual_joint_replay_max_error']>1e-7 or r['previous_25hz_command_max_error']>1e-7 or not r['finite']:r['failures'].append('joint9_or_clock')
r['passed']=not r['failures'];(out/'raw-contract-validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
