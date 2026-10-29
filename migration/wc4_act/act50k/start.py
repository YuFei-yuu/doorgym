from common import *
import subprocess
sys.path.insert(0,str(SUPPORT))
from commands import build
verify()
cmd=build('train',RUN.name,gpu=0,seed=1000,train_steps=50000)['argv']
cmd[:2]=[sys.executable,'-B',str(Path(__file__).with_name('train.py'))]
cmd=[('--dataset.repo_id='+REPO if x.startswith('--dataset.repo_id=') else '--dataset.root='+str(DATA) if x.startswith('--dataset.root=') else x) for x in cmd]
if len(sys.argv)>1:
 cp=Path(sys.argv[1]).resolve()
 assert cp.is_dir() and RUN in cp.parents
 cfg=read(cp/'pretrained_model/train_config.json');cfg.update(resume=True,steps=50000,save_freq=10000)
 write(RUN/'resume-config.json',cfg)
 cmd=[sys.executable,'-B',str(Path(__file__).with_name('train.py')),'--config_path='+str(cp/'pretrained_model/train_config.json'),'--resume=true','--steps=50000','--save_freq=10000']
write(RUN/('resume-command.json' if len(sys.argv)>1 else 'train-command.json'),cmd)
os.execv(sys.executable,cmd)
