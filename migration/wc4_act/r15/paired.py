"""Post-evaluation diagnostics and paired comparison. No new model/scenes."""
import json,math
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
def record_diagnostics(output,states,score,seed,step):
 ref=read(output/'reference.json');tail=states[-130:]
 def held(pred):
  if not pred(tail[-1]):return False
  i=len(tail)-1
  while i>0 and pred(tail[i-1]):i-=1
  return tail[-1]['t']-tail[i]['t']>=2
 gates=dict(opening80=score['opening_max_abs_deg']>=80,whole_body_passed_no_return=score['whole_body_pass_step'] is not None and not score['returned'],
 measured_end_push_screen=score['first_aligned_end_push_step'] is not None,legal_relock=score['close_hold'] is not None,
 final_common2s=score['final_common_span_s']>=2,automatic_candidates=score['measured_task_gates_pass'],
 final_home2s=held(lambda s:all(abs(a-b)<=.03 for a,b in zip(s['arm_q'][:6],ref['home'][:6])) and all(abs(v)<=.05 for v in s['arm_qd'][:6])),
 final_heading2s=held(lambda s:abs(wrap(s['actual_base'][2]-ref['old_expert_final_yaw']))<=math.radians(2)),
 final_base_still2s=held(lambda s:s['base_measured_fd_speed']<=.01 and s['base_measured_fd_yaw_rate']<=.01))
 passage=score['whole_body_pass_step'];turn=None
 if passage is not None:
  yaw=states[passage]['actual_base'][2];turn=max(abs(wrap(s['actual_base'][2]-yaw)) for s in states[passage:])
 write(output/'paired-diagnostics.json',dict(seed=seed,step=step,gates=gates,failures=score['failures'],max_abs_turn_after_pass_rad=turn,
  note='Turning/contact flags are diagnostics, not stage inputs, contact proof, collision proof or extra success gates'))
def summarize(run,step):
 base=run/'evaluation'/f'{step:06d}';order=read(run/'TRAINING_ORDER.json')
 baseline={t['seed']:t for t in read(run/'baseline-030000.json')['trials']}
 assert len(baseline)==16
 records=[];counts={};prior_counts={}
 for d in sorted(base.glob('trial_*')):
  if not d.is_dir():continue
  row=read(d/'paired-diagnostics.json');prior=baseline[row['seed']]['gates']
  row['R14_030000_gates']={k:prior[k] for k in row['gates']}
  row['improved_items']=[k for k,v in row['gates'].items() if v and not prior[k]]
  row['regressed_items']=[k for k,v in row['gates'].items() if not v and prior[k]]
  for k,v in row['gates'].items():counts[k]=counts.get(k,0)+int(v);prior_counts[k]=prior_counts.get(k,0)+int(prior[k])
  records.append(row)
 assert len(records)==16
 write(base/'paired-summary.json',dict(checkpoint_step=step,denominator=16,automatic_candidates=counts['automatic_candidates'],
  reviewed_successes=0 if not counts['automatic_candidates'] else None,counts=counts,R14_030000_same16_counts=prior_counts,records=records,
  original_video_review_required=True,generalization_claim=False,new_30000_evaluation=False))
