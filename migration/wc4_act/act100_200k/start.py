from common import *
sys.path.insert(0,str(SUPPORT))
from commands import build
verify()
if len(sys.argv)>1:
 cp=Path(sys.argv[1]).resolve();assert cp.is_dir() and RUN in cp.parents
 step=read(cp/'training_state/training_step.json')['step'];assert 0<step<200000
 cmd=[sys.executable,'-B',str(Path(__file__).with_name('train.py')),'--config_path='+str(cp/'pretrained_model/train_config.json'),'--resume=true','--steps=200000','--save_freq=20000','--output_dir='+str(RUN/'train'),'--job_name='+RUN.name]
 write(RUN/('resume-command-'+str(step)+'-'+str(__import__('time').time_ns())+'.json'),dict(argv=cmd,parent_checkpoint=str(cp),parent_step=step,target_updates=200000,remaining_updates=200000-step))
else:
 cmd=build('train',RUN.name,gpu=0,seed=1000,train_steps=200000)['argv']
 cmd[:2]=[sys.executable,'-B',str(Path(__file__).with_name('train.py'))]
 cmd=[('--dataset.repo_id='+REPO if x.startswith('--dataset.repo_id=') else '--dataset.root='+str(DATA) if x.startswith('--dataset.root=') else '--save_freq=20000' if x.startswith('--save_freq=') else '--job_name='+RUN.name if x.startswith('--job_name=') else x) for x in cmd]
 write(RUN/'train-command.json',dict(argv=cmd,start_step=0,target_updates=200000,old_task_weights_loaded=False))
os.execv(sys.executable,cmd)
