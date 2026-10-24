"""Offline measured closing-route diagnostics. No policies or simulator import."""
import pathlib,json,math,sys,time
import numpy as np
ROOT=pathlib.Path('/data/zyf/video2door-wc4-act');SUPPORT=ROOT/'migration/wc4_act';sys.path.insert(0,str(SUPPORT))
from fullcycle_compact import ClosingCurve
R=ROOT/'.repro/runs/wc4-fulltask-act-r21-cont150155-s1000-20261024'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi
PLAN=dict(scope='Descriptive measured diagnostics only; no extra failure gates or policy input. Geometric passage is a provisional cohort; independently review original RGB before claiming valid passage.',fold='after geometric pass, arm home max6joint error<=.03rad and maxspeed<=.05rad/s continuously>=.5s',turn='first >=.2s continuous abs applied yaw command>.05rad/s after geometric pass',deployment='first >=.2s continuous max6joint home error>.10rad after detected fold',staging='fixed expert nominal90 staging pose: distance<=.10m and yawerror<=10deg',rendezvous='fixed expert nominal90 rendezvous pose:distance<=.10m and yawerror<=10deg',hover='after staging, EE distance to fixed nominal90 hover<=.10m',push='existing R11 aligned measured end push; not physical push certification',contact_loss='after first aligned push before latch, unlatched door angle>1deg and no nonzero end-panel record; durations descriptive, SDK zero lambda not causal contact failure',post_collision_rotation='only separate secondary description after nonend nonzero record during measured door closing; not a causal collision diagnosis',prediction_or_scene_calls=0)
def analyze(tag,d,seed):
 states=[json.loads(l) for l in (d/'states.jsonl').open()];sco=read(d/'r11-score.json');ref=read(d/'reference.json');frame=read(d/'scene-frame.json');N=len(states)
 t=np.array([s['t'] for s in states]);dt=float(np.median(np.diff(t)));p=sco['whole_body_pass_step'];push=sco['first_aligned_end_push_step'];end=next((s['step'] for s in states if s['latched'] is not None),N)
 X=np.asarray([s['actual_base'] for s in states]);A=np.asarray([s['arm_q'][:6] for s in states]);V=np.asarray([s['arm_qd'][:6] for s in states]);home=np.asarray(ref['home'][:6]);qerr=abs(A-home).max(1)
 W=np.asarray(frame['world_from_reference']);b=ClosingCurve().at(math.pi/2)[:3];stage=b.copy();stage[:2]-=.22*np.array([math.cos(b[2]),math.sin(b[2])])
 yawdelta=math.atan2(W[1,0],W[0,0])
 def world(p):return np.r_[(W@np.r_[p[:2],0.,1.])[:2],p[2]+yawdelta]
 bw,sw=world(b),world(stage);hover=ClosingCurve().tool(math.pi/2,.04)[:3,3];hover=(W@np.r_[hover,1.])[:3]
 def reach(goal):return (np.linalg.norm(X[:,:2]-goal[:2],axis=1)<=.10)&(abs(wrap(X[:,2]-goal[2]))<=math.radians(10))
 def first(mask,start,stop=N,hold=.0):
  if start is None:return None
  begin=None
  for i in range(start,stop):
   if mask[i]:
    if begin is None:begin=i
    if t[i]-t[begin]+1e-9>=hold:return int(begin)
   else:begin=None
  return None
 turn=first(np.array([abs(s['action'][1])>.05 for s in states]),p,hold=.2)
 fold=first((qerr<=.03)&(abs(V).max(1)<=.05),p,hold=.5)
 deploy=first(qerr>.10,fold,hold=.2)
 staging=first(reach(sw),p);rv=first(reach(bw),p);approach=first(np.linalg.norm(np.array([s['ee_actual'] for s in states])-hover,axis=1)<=.10,staging)
 off=first(np.array([s['nonend_nonzero_record_during_measured_closing'] for s in states]),p)
 missing=np.array([not s['nonzero_end_panel_contact'] and s['latched'] is None and abs(s['door_q'][0])>math.radians(1) for s in states]);gaps=[]
 if push is not None:
  z=None
  for i in range(push,end):
   if missing[i] and z is None:z=i
   if z is not None and (not missing[i] or i==end-1):
    stop=i if not missing[i] else i+1;gaps.append((stop-z)*dt);z=None
 def event(i):return None if i is None else dict(step=i,time_s=float(t[i]),base=X[i].tolist(),arm_home_max_error_rad=float(qerr[i]),door_deg=math.degrees(states[i]['door_q'][0]))
 row=dict(model=tag,seed=seed,directory=str(d),full_candidate=sco['measured_task_gates_pass'],failures=sco['failures'],opening_max_deg=sco['opening_max_abs_deg'],geometric_pass=event(p),returned=sco['returned'],fold=event(fold),turn=event(turn),fold_before_turn=(fold<=turn if fold is not None and turn is not None else None),staging=event(staging),deployment=event(deploy),rendezvous=event(rv),hover_approach=event(approach),first_aligned_push=event(push),legal_relock=sco['close_hold'],first_offtarget_closing_nonzero=event(off),post_offtarget_abs_command_turn_deg=(math.degrees(sum(abs(s['action'][1])*dt for s in states[off:])) if off is not None else None),unloaded_end_contact_gap_count_ge_halfsec=sum(g>=.5 for g in gaps),unloaded_end_contact_gap_max_s=max(gaps,default=None),push_loaded_end_fraction=(sum(s['nonzero_end_panel_contact'] for s in states[push:end])/(end-push) if push is not None and end>push else None),video=str(d/'overview/env_0000/rollout.mp4'),video_time_convention='frame f corresponds to physical state near .04f+.02s; original clip, not state replay')
 if p is not None:
  stop=max(p+1,push+1 if push is not None else end)
  route=np.asarray(read(R/'route-reference.json')['points_xy']);inv=np.linalg.inv(W);refxy=(inv@np.c_[X[:,:2],np.zeros(N),np.ones(N)].T).T[:,:2]
  def progress(stop):
   z=refxy[p:max(p+1,stop)];dist=np.linalg.norm(z[:,None,:]-route[None,:,:],axis=2);prefix=0
   for point in z:
    while prefix<len(route) and np.linalg.norm(point-route[prefix])<=.15:prefix+=1
   return dict(ordered_reference_points=prefix,total_reference_points=len(route),ordered_prefix_fraction=prefix/len(route),visited_fraction=float((dist.min(axis=0)<=.15).mean()),min_endpoint_distance_m=float(dist[:,-1].min()))
  row['bypass_reference_before_push']=progress(stop)
  row['bypass_reference_before_deployment']=None if deploy is None else progress(deploy)
  row['rendezvous_before_deployment']=None if deploy is None else bool(rv is not None and rv<=deploy)
  row.update(min_staging_distance_before_push_m=float(np.linalg.norm(X[p:stop,:2]-sw[:2],axis=1).min()),min_rendezvous_distance_before_push_m=float(np.linalg.norm(X[p:stop,:2]-bw[:2],axis=1).min()))
 return row
if __name__=='__main__':
 step=int(sys.argv[1]);assert step in [100000,150000,200000]
 stage=R/'evaluation'/f'{step:06d}';assert read(stage/'complete.json')['consumed']==16
 out=R/'closing-analysis'/f'{step:06d}';out.mkdir(parents=True,exist_ok=False);write(out/'diagnostic-contract.json',PLAN)
 # Reuse the already-successful frozen R20 historical64 records, no model/control rerun.
 historical=read(R/'historical-closing-records.json')['records'];rows=list(historical)
 if step==100000:
  for i,seed in enumerate(range(62200,62216)):
   row=analyze('R18_uniform100000',ROOT/'.repro/runs/wc4-fulltask-act-r18-s1000-20261023/evaluation/100000'/f'trial_{i:02d}',seed);rows.append(row);write(out/f'R18_uniform100000-{seed}.json',row)
 for i,seed in enumerate(range(62200,62216)):
  row=analyze(f'R21_{step:06d}',stage/f'trial_{i:02d}',seed);rows.append(row);write(out/f'R21_{step:06d}-{seed}.json',row);print(step,seed,flush=True)
 write(out/'all-records.json',dict(definitions=PLAN,records=rows,new_model_forwards=0,new_scenes=0,reused_historical_records=64,equal_update_uniform_comparison=(step==100000),later_stages_branch_trend_only=(step!=100000)))
