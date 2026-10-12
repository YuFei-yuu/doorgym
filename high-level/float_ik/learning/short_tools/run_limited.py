import fcntl,json,pathlib,shlex,statistics,subprocess,sys,time,hashlib
repo=pathlib.Path('/data/zyf/video2door');r=repo/'experiments/float-push/20261012-ee-short-refined'
sys.path.insert(0,str(repo/'high-level/float_ik/learning/long_tools'))
from bounded import run,used,atomic_json
lock=(r/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
seed=int(sys.argv[1]);assert seed in [8500,8501,8502]
ck=json.loads((r/'frozen_model.json').read_text());assert hashlib.sha256(pathlib.Path(ck['checkpoint']).read_bytes()).hexdigest()==ck['sha256']
plan=dict(profile='ee_reference_limited_v1',seeds=[8500,8501,8502],model=ck,limit_arguments=dict(position_step_m=.006,position_lookahead_m=.06,rotation_step_rad=.035),no_optimizer_updates=True,raw_failure_retained='validation/policy_seed_8500',rule='Stop on any simulator failure; no replacement of failed direct-execution validation')
if not (r/'execution_diagnostic_plan.json').exists():atomic_json(r/'execution_diagnostic_plan.json',plan)
else:assert json.loads((r/'execution_diagnostic_plan.json').read_text())==plan
for mode in ['expert','policy']:
 name='limited_%s_seed_%d'%(mode,seed);out=r/'execution_diagnostic'/name
 if out.exists():raise RuntimeError('Attempt already exists: '+name)
 samples=[]
 for i in range(3):
  s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  samples.append([[v.strip() for v in line.split(',')] for line in s.splitlines()])
  if i<2:time.sleep(2)
 eligible=[]
 for j,s in enumerate(samples[0]):
  if min(float(x[j][2]) for x in samples)>=16384 and statistics.mean(float(x[j][3]) for x in samples)<70:eligible.append((statistics.mean(float(x[j][3]) for x in samples),int(s[0]),s[1]))
 atomic_json(r/'resources'/(name+'.json'),dict(samples=samples,eligible=eligible))
 if not eligible:raise RuntimeError('No eligible GPU')
 _,gpu,uuid=min(eligible)
 args=['python',r/'limited_rollout.py','--schema','float_push_ee_world_v1','--mode',mode,'--seed',seed,'--split','execution_diagnostic','--record','--video','--diagnostic-steps',60,'--output',out]
 if mode=='policy':args+=['--checkpoint',ck['checkpoint'],'--inference-steps',25]
 shell='export VIDEO2DOOR_GPU=%d; source /data/zyf/video2door/.repro/blackwell/activate.sh || exit; exec nice -n 10 '%gpu+shlex.join(list(map(str,args)))
 code=run(r,name,['bash','-c',shell],timeout=180,total=3600,gpu=gpu,gpu_uuid=uuid,log_name=name+'.log')
 if code:raise RuntimeError('Failed diagnostic; stop and preserve: '+name)
 s=json.loads((out/'summary.json').read_text());print('RESULT',name,s['status'],s['success'],s['signed_angle_deg'],flush=True)
print('USED',used(r))
