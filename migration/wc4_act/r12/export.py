import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from torch.utils.data import DataLoader,Subset

from common import *
step=int(sys.argv[1]);assert step in [10000,20000,30000]
R=RUN/'checkpoints'/f'{step:06d}';R.mkdir(parents=True,exist_ok=False)
TRAIN_R=RUN
SOURCE=RUN/'train/checkpoints'/f'{step:06d}'/'pretrained_model';WRAPPED=R/'door-checkpoint'
D=OLD;DATA_ID=REPO
assert json.loads((SOURCE.parent/'training_state/training_step.json').read_text())['step']==step
assert not WRAPPED.exists()
command=[sys.executable,'high-level/dp/export_official_lerobot_act_to_door_checkpoint.py','--official_checkpoint',str(SOURCE),'--root',str(D/'lerobot'),'--repo_id',DATA_ID,'--out_dir',str(WRAPPED/'model_latest'),'--depth_only','--device','cpu','--action_horizon','25']
(R/'export-command.json').write_text(json.dumps(command,indent=2)+'\n')
subprocess.run(command,cwd=ROOT,check=True)
sys.path.insert(0,str(ROOT/'high-level/dp'))
from door_policy_backend import load_door_policy_backend,load_lerobot_policy_normalizer_stats
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.factory import make_dataset

cfg=PreTrainedConfig.from_pretrained(str(SOURCE));cfg.device='cpu';cfg.pretrained_backbone_weights=None
official=ACTPolicy.from_pretrained(SOURCE,config=cfg,local_files_only=True,strict=True).eval()
pre,post=make_pre_post_processors(cfg,pretrained_path=str(SOURCE),
    preprocessor_overrides={'device_processor':{'device':'cpu'},'normalizer_processor':{'device':'cpu'}},
    postprocessor_overrides={'unnormalizer_processor':{'device':'cpu'}})
backend=load_door_policy_backend(WRAPPED/'model_latest.pt',device='cpu',action_horizon=25)
assert backend.action_dim==9 and backend.horizon==100 and backend.action_horizon==25
meta=json.loads((WRAPPED/'model_latest/door_policy_meta.json').read_text())
assert meta['policy_config']['stats_source']=='official_lerobot_preprocessor'
assert backend.sidecar_config['action']==['vx','yaw','joint1','joint2','joint3','joint4','joint5','joint6','jointGripper']
for name,value in official.state_dict().items():
    assert torch.equal(value,backend.policy.state_dict()[name]),name
stats=load_lerobot_policy_normalizer_stats(SOURCE);assert stats
for key,values in stats.items():
    if key in backend.stats:
        for name,value in values.items():
            if name in backend.stats[key]:assert torch.equal(value.cpu(),backend.stats[key][name].cpu()),(key,name)
data_cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id=DATA_ID,root=str(DATA),use_imagenet_stats=False),policy=ACTConfig(device='cpu',chunk_size=100,n_action_steps=50,push_to_hub=False,use_action_loss_weight=False))
dataset=make_dataset(data_cfg)
indices=[0,100,2215,79764,159528]
batch=next(iter(DataLoader(Subset(dataset,indices),batch_size=len(indices),num_workers=0)))
batch_pre=pre({k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in batch.items()})
policy_batch={k:batch[k] for k in ['observation.state',*backend.image_keys]}
batch_backend=backend.normalizer.normalize_batch(policy_batch,include_action=False)
errors={}
for key in ['observation.state',*backend.image_keys]:
    error=float((batch_pre[key]-batch_backend[key]).abs().max());errors[key]=error
    assert error<=1e-6,(key,error)
with torch.no_grad():
    normalized=official.predict_action_chunk(batch_pre)
    expected=post(normalized.clone())
    got=backend.predict_action_chunks_from_batch(policy_batch)
assert tuple(got.shape)==(len(indices),100,9) and torch.isfinite(got).all()
error=float((expected-got).abs().max());assert error<=1e-5,error
checkpoints=[]
for saved_step in [step]:
    directory=TRAIN_R/'train/checkpoints'/f'{saved_step:06d}'
    assert directory.is_dir()
    checkpoints.append({'step':saved_step,'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(directory.rglob('*')) if p.is_file()]})
wrapped_files=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(WRAPPED.rglob('*')) if p.is_file()]
result={'status':'passed','source':str(SOURCE),'wrapped':str(WRAPPED/'model_latest.pt'),'sample_indices':indices,'policy_weights_equal':True,'normalizer_stats_equal':True,'preprocessing_max_error':errors,'physical_action_max_error':error,'prediction_shape':list(got.shape),'action_horizon':25,'checkpoints':checkpoints,'wrapped_files':wrapped_files,'no_simulation':True}
(R/'export-consistency.json').write_text(json.dumps(result,indent=2)+'\n')
torch.save({'indices':indices,'batch':batch,'expected_actions':expected,'normalization_keys':list(errors)},R/'export-reference.pt')
print(json.dumps({k:result[k] for k in ['status','preprocessing_max_error','physical_action_max_error','prediction_shape','action_horizon']},indent=2))
