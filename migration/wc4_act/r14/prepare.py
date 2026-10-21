"""R14 fresh run, exact R13 inputs and frozen trials; no training or simulation."""
import sys,os,subprocess,shutil,ast,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];SUPPORT=ROOT/'migration/wc4_act'
sys.path[:0]=[str(SUPPORT/'r13'),str(SUPPORT)]
from runtime import read,write,sha,training_argv,DATA,REPO,R12
ID='wc4-fulltask-act-r14-s1000-20261022';RUN=ROOT/'.repro/runs'/ID
R13=ROOT/'.repro/runs/wc4-engineering-r13-20261022'
assert not RUN.exists(),'No replacement or retry'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
assert shutil.disk_usage(ROOT).free>=180*1024**3,'Insufficient reserved headroom; do not delete history'
for name,digest in read(R13/'source-binding.json').items():assert sha(ROOT/name)==digest,name
for name,digest in read(R13/'protected-before.json').items():assert sha(Path(name))==digest,name
assert read(R13/'readiness.json')['status']=='CPU-engineering-passed-await-independent-acceptance'
RUN.mkdir(parents=True)
source=read(R13/'source-binding.json')
for p in Path(__file__).parent.iterdir():
    if p.is_file():
        if p.suffix=='.py':ast.parse(p.read_text())
        source[str(p.relative_to(ROOT))]=sha(p)
master=read(R13/'manifest.json');master['source_sha256']=source
assert master['trials']==read(R12/'manifest.json')['trials']
write(RUN/'manifest.json',master);write(RUN/'source-binding.json',source)
for trial in master['trials']:
    folder=RUN/'frozen'/str(trial['seed']);folder.mkdir(parents=True)
    write(folder/'manifest.json',dict(master,trials=[trial],seeds=[trial['seed']]))
for name in ['frozen-r7-physics.yaml','geometry.json','dataset-sha256.json','collection-disjoint.json']:
    shutil.copyfile(R12/name,RUN/name)
contract=read(R12/'contract.json');contract['task']='R14 fresh full-task ACT baseline';contract['r12_loss_updates']=[1122,1123]
write(RUN/'contract.json',contract)
write(RUN/'TRAINING_ORDER.json',dict(task='R14 user order',authorized_by_new_user_task=True,
    r12_consumed_budget_acknowledged=True,r12_updates_separately_lost=[1122,1123],
    seed=1000,steps=30000,start_from_scratch=True,resume_checkpoint=None,
    intermediate_seeds=list(range(62300,62308)),final_seeds=list(range(62200,62264)),
    evaluation_rounds={'010000':8,'020000':8,'030000':64},total_evaluation_budget=80,
    max_episode_seconds=180,no_extra_smoke_or_retry=True,no_automatic_resume=True))
write(RUN/'capacity.json',dict(at=time.time(),free_bytes=shutil.disk_usage(ROOT).free,
    logical_reserved_bytes=180*1024**3,stop_below_bytes=40*1024**3,
    basis='80 x 180s state/contact/input/video evidence; three checkpoints/exports and safety headroom',
    df=subprocess.check_output(['df','-h',str(ROOT)],text=True)))
write(RUN/'budget.json',dict(planned_updates=30000,completed_updates=0,evaluation_budget=80,evaluation_invocations=0,
    r12_separate_loss=[1122,1123],status='prepared-no-updates-no-scenes'))
# Parse the exact recipe only; no policy/model/optimizer constructed here.
import draccus,lerobot
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
cmd=training_argv(RUN);cfg=draccus.parse(TrainPipelineConfig,args=cmd[3:]);cfg.validate()
assert not cfg.resume and cfg.policy.pretrained_path is None and cfg.policy.pretrained_backbone_weights=='ResNet18_Weights.IMAGENET1K_V1'
assert cfg.seed==1000 and cfg.steps==30000 and cfg.batch_size==16 and cfg.policy.optimizer_lr==1e-5
assert cfg.policy.chunk_size==100 and cfg.policy.n_action_steps==50 and cfg.policy.n_obs_steps==1
assert cfg.dataset.episodes is None and str(cfg.dataset.root)==str(DATA)
assert read(DATA/'meta/info.json')['total_frames']==159529 and read(DATA/'meta/info.json')['total_episodes']==72
assert ROOT/'high-level/lerobot/src' in Path(lerobot.__file__).parents
write(RUN/'cpu-preflight.json',dict(argv=cmd,parsed_config=cfg.to_dict(),python=sys.executable,lerobot=lerobot.__file__,
    no_model_optimizer_or_scene=True,frozen_trial_tuples_equal=True))
# Exercise the new passive parser on the already accepted real-IPC CPU fixture.
sys.path.insert(0,str(Path(__file__).parent))
from audit import verify_journal
proof=verify_journal(R13/'regression_004/ipc-gym/applied.jsonl')
assert proof['emitted']==75 and proof['completed_physics_steps']==150
write(RUN/'passive-audit-cpu-check.json',proof)
write(RUN/'execution-version.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    main=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip(),worktree_clean=True,
    accepted_R13='821662dbd963ca16db372c65751aeb9c857fca5b'))
write(RUN/'frozen-files.json',{str(p.relative_to(RUN)):sha(p) for p in [RUN/'manifest.json',RUN/'TRAINING_ORDER.json',RUN/'contract.json',
    RUN/'frozen-r7-physics.yaml',RUN/'geometry.json',RUN/'dataset-sha256.json',*sorted((RUN/'frozen').glob('*/manifest.json'))]})
print('R14 fresh order and exact frozen trials ready; zero formal updates/scenes; passive CPU parser passed')
