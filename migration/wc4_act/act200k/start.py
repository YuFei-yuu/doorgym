from common import *
verify()
cp=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else PARENT
assert cp==PARENT or RUN in cp.parents
step=read(cp/'training_state/training_step.json')['step'];assert 50000<=step<200000
cadence=100000 if step<100000 else 20000
cmd=[sys.executable,'-B',str(Path(__file__).with_name('train.py')),'--config_path='+str(cp/'pretrained_model/train_config.json'),'--resume=true','--steps=200000','--save_freq='+str(cadence),'--output_dir='+str(RUN/'train'),'--job_name='+RUN.name]
write(RUN/('train-command.json' if cp==PARENT else 'resume-command-'+str(step)+'.json'),dict(argv=cmd,parent_checkpoint=str(cp),parent_step=step,target_updates=200000,remaining_updates=200000-step))
os.execv(sys.executable,cmd)
