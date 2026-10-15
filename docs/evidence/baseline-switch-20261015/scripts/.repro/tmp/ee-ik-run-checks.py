import fcntl,json,pathlib,shlex,statistics,subprocess,sys,time
repo=pathlib.Path('/data/zyf/video2door');r=repo/'experiments/float-push/20261012-ee-ik-adapter-check'
r.mkdir(exist_ok=True);sys.path.insert(0,str(repo/'high-level/float_ik/learning/long_tools'))
from bounded import run,used,atomic_json
lock=(r/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
phase=sys.argv[1]
samples=[]
for i in range(3):
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 samples.append([[v.strip() for v in s.split(',')] for s in lines.splitlines()])
 if i<2:time.sleep(2)
candidates=[]
for j,v in enumerate(samples[0]):
 if min(float(s[j][2]) for s in samples)>=16384 and statistics.mean(float(s[j][3]) for s in samples)<70:candidates.append((statistics.mean(float(s[j][3]) for s in samples),int(v[0]),v[1]))
atomic_json(r/('resource_'+phase+'.json'),dict(samples=samples,candidates=candidates))
if not candidates:raise SystemExit('No eligible GPU')
_,gpu,uuid=min(candidates)
learning=repo/'high-level/float_ik/learning'
commands={
 'cpu':['python',learning/'test_ee_contract.py','-v'],
 'expert':['python',learning/'rollout.py','--schema','float_push_ee_world_v1','--mode','expert','--nominal','--seed','7000','--record','--video','--diagnostic-steps','60','--output',r/'expert_nominal'],
 'replay':['python',learning/'rollout.py','--schema','float_push_ee_world_v1','--mode','replay','--nominal','--seed','7000','--record','--diagnostic-steps','60','--replay',r/'expert_nominal/episode.npz','--output',r/'replay_nominal'],
 'probe':['python',learning/'check_ee_sim.py','--output',r/'policy_probe.json'],
}
cmd=commands[phase.split("_")[0]]
if phase.endswith("_final"):
 cmd=[str(x).replace("expert_nominal", "expert_nominal_final").replace("replay_nominal", "replay_nominal_final").replace("policy_probe.json", "policy_probe_final.json") for x in cmd]
shell='export VIDEO2DOOR_GPU=%d; source /data/zyf/video2door/.repro/blackwell/activate.sh || exit; exec nice -n 10 '%gpu+shlex.join(list(map(str,cmd)))
rc=run(r,phase,['bash','-c',shell],timeout=180 if phase.split('_')[0] in ['expert','replay'] else 90,total=600,gpu=gpu,gpu_uuid=uuid,log_name=phase+'.log')
print('USED',used(r));print((r/'logs'/(phase+'.log')).read_text()[-3000:]);sys.exit(rc)
