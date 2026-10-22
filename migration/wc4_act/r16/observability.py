"""Descriptive failure-input aliases across measured headings; no action truth transfer.
Only the 224 already-frozen observations; no policy or additional forward.
"""
from common import *
import numpy as np,math
obs=[p for p in read(RUN/'observations.json')['points'] if p['kind']=='failure'];state=[];small=[];scale=np.asarray([.01,.01]+[.03]*7)
for p in obs:
 with np.load(RUN/p['input']) as a:state.append(a['state'].copy());small.append(np.stack([a[v][::16,::16,0].astype(np.float32)/255 for v in ['front','wrist']]))
state=np.stack(state);small=np.stack(small);pairs=[]
for i,p in enumerate(obs):
 for j in range(i+1,len(obs)):
  q=obs[j];goal_gap=abs(float(wrap(p['heading_error_rad']-q['heading_error_rad'])))
  if goal_gap<math.radians(30):continue
  sd=float(np.max(abs(state[i]-state[j])/scale));rms=np.sqrt(((small[i]-small[j])**2).mean(axis=(1,2)))
  if sd>1 or rms.max()>.02:continue
  pairs.append(dict(a=p['id'],b=q['id'],seed_a=p['seed'],seed_b=q['seed'],heading_error_deg=[math.degrees(p['heading_error_rad']),math.degrees(q['heading_error_rad'])],heading_error_difference_deg=math.degrees(goal_gap),one_near_target_other_far=(min(abs(p['heading_error_rad']),abs(q['heading_error_rad']))<=math.radians(2)),state_scaled_max=sd,coarse_depth_rmse=rms.tolist(),action_truth_available=False))
ranked=sorted(pairs,key=lambda r:(not r['one_near_target_other_far'],max(r['coarse_depth_rmse']),r['state_scaled_max'],r['a'],r['b']))[:12];idx={p['id']:p for p in obs}
for p in ranked:
 with np.load(RUN/idx[p['a']]['input']) as a,np.load(RUN/idx[p['b']]['input']) as b:p['full_depth_rmse']=[float(np.sqrt(np.mean(((a[v].astype(np.float32)-b[v])/255)**2))) for v in ['front','wrist']]
write(RUN/'failure-observability-risk.json',dict(scope='Post-hoc descriptive pairs within the same pre-frozen 224 failure inputs. No new observation/prediction or expert-label transfer.',thresholds=dict(state_scale=scale.tolist(),depth_rmse=.02,measured_heading_difference_deg=30,near_target_deg=2),coarse_near_pairs=len(pairs),coarse_near_target_vs_far_pairs=sum(p['one_near_target_other_far'] for p in pairs),ranked_full_resolution_examples=ranked,not_proof_of_unlearnability=True,no_correct_action_labels_for_failure=True,new_forward_calls=0))
print('Failure input goal-condition risk:',len(pairs),'coarse pairs;',sum(p['one_near_target_other_far'] for p in pairs),'near vs far; no action truth assigned',flush=True)
