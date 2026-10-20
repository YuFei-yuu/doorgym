from common import *
assert json.loads((RUN/'accept-results-exit.json').read_text())['returncode']==0
assert json.loads((RUN/'acceptance.json').read_text())['policy_trials']==64
assert not list(RUN.glob('stop*.json'))
# Freeze API watcher logs before sealing; no training/simulation is started here.
for _ in range(15):
 if (RUN/'wandb-eval-monitor-finished.json').exists():break
 time.sleep(5)
assert json.loads((RUN/'wandb-eval-monitor-finished.json').read_text())['evaluation_complete']
source_check();repos={}
for repo in [ROOT,Path('/data/zyf/video2door')]:
 repos[str(repo)]={k:subprocess.check_output(['git',*args],cwd=repo,text=True).strip() for k,args in [('commit',['rev-parse','HEAD']),('branch',['branch','--show-current']),('status',['status','--porcelain=v1']),('diff',['diff','--stat'])]}
 assert not repos[str(repo)]['status']
assert repos['/data/zyf/video2door']['commit']=='64b1bcc162c4997dbe39fd3f76b61500573d69a8'
active=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
 uuid,pid,mem=[v.strip() for v in line.split(',')];p=Path('/proc')/pid
 try:cmd=(p/'cmdline').read_bytes().decode().replace('\0',' ');cwd=(p/'cwd').resolve()
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if str(ROOT) in cmd or cwd==ROOT:active.append({'uuid':uuid,'pid':pid,'cmd':cmd})
assert not active,active
mon={}
for name in ['wandb-monitor.jsonl','wandb-eval-monitor.jsonl']:
 rows=[json.loads(s) for s in (RUN/name).read_text().splitlines()];mon[name]={'events':len(rows),'failures':[v for v in rows if not v.get('online')]}
a=json.loads((RUN/'analysis-existing/manifest.json').read_text())
for v in a['files']:assert sha(RUN/'analysis-existing'/v['path'])==v['sha256']
result={'at':time.time(),'source_states':repos,'start_step':40000,'end_step':50000,'new_updates':10000,'new_policy_trials':64,'evaluated_steps':[50000],'project_gpu_processes':active,'all_training_and_simulation_stopped':True,'monitors':mon,'pre50k_analysis_artifacts_unchanged':True,'fixed_evaluation_set_only':True,'independent_confirmation_run':False}
save(RUN/'final-state-check.json',result);print(json.dumps(result,indent=2))
