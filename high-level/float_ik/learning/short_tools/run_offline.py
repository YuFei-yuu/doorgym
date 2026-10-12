import fcntl,json,pathlib,shlex,statistics,subprocess,sys,time
repo=pathlib.Path('/data/zyf/video2door');r=repo/'experiments/float-push/20261012-ee-short-refined';sys.path.insert(0,str(repo/'high-level/float_ik/learning/long_tools'))
from bounded import run,atomic_json
lock=(r/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
samples=[]
for i in range(3):
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 samples.append([[v.strip() for v in line.split(',')] for line in s.splitlines()])
 if i<2:time.sleep(2)
eligible=[]
for j,row in enumerate(samples[0]):
 if min(float(s[j][2]) for s in samples)>=16384 and statistics.mean(float(s[j][3]) for s in samples)<70:eligible.append((statistics.mean(float(s[j][3]) for s in samples),int(row[0]),row[1]))
assert eligible
_,gpu,uuid=min(eligible);atomic_json(r/'resources/offline_action_diagnostics.json',dict(samples=samples,eligible=eligible))
args=['python',repo/'high-level/float_ik/learning/offline_eval.py','--root',r]
shell='export VIDEO2DOOR_GPU=%d; source /data/zyf/video2door/.repro/blackwell/activate.sh || exit; exec nice -n 10 '%gpu+shlex.join(list(map(str,args)))
rc=run(r,'offline_action_diagnostics',['bash','-c',shell],timeout=90,total=3600,gpu=gpu,gpu_uuid=uuid,log_name='offline_action_diagnostics.log');sys.exit(rc)
