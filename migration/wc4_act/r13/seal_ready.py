"""Seal CPU-qualified readiness, not permission to train. No model or environment."""
import sys,subprocess,ast,os,time
from runtime import *
run=ROOT/'.repro/runs/wc4-engineering-r13-20261022'
if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise RuntimeError('Clean committed code required')
for name,digest in read(run/'protected-before.json').items():
 if sha(Path(name))!=digest:raise RuntimeError('Protected data/history/source changed: '+name)
last=sorted(run.glob('regression_*'))[-1];result=read(last/'result.json')
if not result['passed']:raise RuntimeError('Latest CPU regression failed')
for p in (SUPPORT/'r13').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
# Explicit immutable-contract guard beyond the original R12 binding.
paths=['high-level/lerobot','high-level/float_ik','high-level/data','low-level','third_party',
 'migration/wc4_act/r11','migration/wc4_act/r12']
subprocess.run(['git','diff','--exit-code','3c713de','HEAD','--',*paths],cwd=ROOT,check=True)
master=read(run/'manifest.json');assert master['trials']==read(R12/'manifest.json')['trials']
source=read(run/'source-binding.json')
for p in [*sorted((SUPPORT/'r13').iterdir()),ROOT/'high-level/dp/door_action_evidence.py',ROOT/'high-level/dp/door_policy_backend.py',ROOT/'high-level/dp/door_policy_worker.py']:
 if p.is_file():source[str(p.relative_to(ROOT))]=sha(p)
master['source_sha256']=source;write(run/'manifest.json',master)
for p in (run/'frozen').glob('*/manifest.json'):
 one=read(p);assert one['trials']==[t for t in master['trials'] if t['seed']==one['seeds'][0]]
 one['source_sha256']=source;write(p,one)
write(run/'source-binding.json',source)
write(run/'frozen-files.json',{str(p.relative_to(run)):sha(p) for p in [run/'manifest.json',run/'scope.json',*sorted((run/'frozen').glob('*/manifest.json'))]})
normalization=read(last/'normalization/result.json')
write(run/'readiness.json',dict(status='CPU-engineering-passed-await-independent-acceptance',
    training_authorization=False,formal_ACT_updates=0,real_policy_inference=0,simulation=0,closed_loop_evaluation=0,
    final_regression=str(last),passed_commands=len(result['commands']),
    frozen_initial_camera_trials=72,all_trial_tuples_equal_to_r12=True,
    protected_files_rehashed=len(read(run/'protected-before.json')),protected_all_equal=True,
    real_normalization_check=normalization,
    tested_runtime_head='f1a8b9f',sealed_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    unverified=['real ACT/GPU forward and exported model equivalence','CUDA/Vulkan scenario binding',
      'actual Gym target/physics timing, latch legality and physical push/release video',
      'GPU RNG/save/restore and live W&B stop synchronization',
      'exact resumed multiworker DataLoader sample order is not serialized by the author checkpoint'],
    r12_budget_consumption=[1122,1123],no_automatic_training_or_resume=True))
write(run/'resource-final.json',dict(at=time.time(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True),
    process_snapshot=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)))
print('Sealed CPU-only readiness; all protected hashes and frozen trials unchanged; no training authorization')
