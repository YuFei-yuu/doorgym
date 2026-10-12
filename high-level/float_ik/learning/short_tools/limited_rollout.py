"""Explicit execution ablation: generic bounded Cartesian references, no teacher aid."""
import hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
REPO=next(p for p in Path(__file__).resolve().parents if (p/'high-level/float_ik/learning/rollout.py').is_file())
HERE=REPO/'high-level/float_ik/learning';sys.path.insert(0,str(HERE))
import float_env
from ee_actions import normalize_ee_action,expert_pose_reference
BaseEnv=float_env.FloatEnv
PROFILE='ee_reference_limited_v1'
class LimitedEnv(BaseEnv):
    def step(self,action):
        normalized=normalize_ee_action(action,self.action_lower[-1],self.action_upper[-1],self.previous[6:10])
        measured,_=float_env.demo.get_body_pose(self.gym,self.env,self.arm,self.ee_index)
        # Pure position/quaternion arithmetic; no expert_action, door state or phase.
        reference=expert_pose_reference(normalized,self.previous,measured)
        obs,info,done,applied=super().step(reference)
        self.last_clipped_channels=np.asarray(action)!=applied;self.last_clipped_channels[6:10]=False
        info['clipped_fraction']=float(self.last_clipped_channels.mean())
        info['raw_ee_target_error_m']=float(np.linalg.norm(normalized[3:6].astype(float)-measured))
        info['reference_position_correction_m']=float(np.linalg.norm(normalized[3:6].astype(float)-applied[3:6]))
        return obs,info,done,applied
float_env.FloatEnv=LimitedEnv
out=Path(sys.argv[sys.argv.index('--output')+1])
try:
    runpy.run_path(str(HERE/'rollout.py'),run_name='__main__')
finally:
    for name in ['summary.json','config.json']:
        p=out/name
        if p.exists():
            d=json.loads(p.read_text());d.update(execution_profile=PROFILE,execution_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference_limits=dict(position_step_m=.006,position_lookahead_m=.06,rotation_step_rad=.035),comparison_scope='Explicit generic execution limits, diagnostic only; never pool with default execution results.')
            p.write_text(json.dumps(d,indent=2))
