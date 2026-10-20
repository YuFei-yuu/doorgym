"""Run unchanged author training with fail-fast finite-value guards and receipts."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

from common import ROOT,RUN,SOURCE,save,sha
import torch
from accelerate import Accelerator
from lerobot.policies.act.modeling_act import ACTPolicy

from lerobot.configs.train import TrainPipelineConfig
original_validate=TrainPipelineConfig.validate
def validate_frozen(cfg):
    result=original_validate(cfg)
    assert cfg.resume and cfg.steps==50000 and cfg.save_freq==2000
    assert cfg.batch_size==16 and cfg.seed==1000 and cfg.num_workers==3
    assert cfg.checkpoint_path==SOURCE and cfg.policy.pretrained_path==SOURCE/'pretrained_model'
    assert str(cfg.dataset.root).endswith('/wc4-joint9-skill50-s62000-20261015/lerobot/local/wc4_joint9_skill_wc4-joint9-skill50-s62000-20261015')
    old=json.loads((SOURCE/'pretrained_model/train_config.json').read_text());new=cfg.to_dict()
    def diff(a,b,p=''):
        out=[]
        if isinstance(a,dict) and isinstance(b,dict):
            for key in sorted(set(a)|set(b)):out+=diff(a.get(key),b.get(key),p+'.'+key if p else key)
        elif a!=b:out.append({'path':p,'before':a,'after':b})
        return out
    changes=diff(old,new)
    allowed={'resume','steps','save_freq','output_dir','job_name','wandb.run_id','policy.pretrained_path','checkpoint_path'}
    assert all(v['path'] in allowed for v in changes),changes
    save(RUN/'effective-training-config.json',new);save(RUN/'training-config-diff.json',changes)
    return result
TrainPipelineConfig.validate=validate_frozen

forward=ACTPolicy.forward
def checked_forward(*args,**kwargs):
    result=forward(*args,**kwargs)
    if not torch.isfinite(result[0]).all().item():
        (RUN/'stop-nonfinite.json').write_text(json.dumps({'at':time.time(),'where':'forward loss before backward'})+'\n')
        raise FloatingPointError('STOP: non-finite loss before backward; no optimizer update applied')
    return result
ACTPolicy.forward=checked_forward
clip=Accelerator.clip_grad_norm_
def checked_clip(*args,**kwargs):
    norm=clip(*args,**kwargs)
    if not torch.isfinite(norm).all().item():
        (RUN/'stop-nonfinite.json').write_text(json.dumps({'at':time.time(),'where':'gradient norm before optimizer step'})+'\n')
        raise FloatingPointError('STOP: non-finite gradient norm before optimizer update')
    return norm
Accelerator.clip_grad_norm_=checked_clip

spec=importlib.util.spec_from_file_location('wc4_author_train',ROOT/'high-level/lerobot/src/lerobot/scripts/lerobot_train.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
logger_init=module.WandBLogger.__init__
def init_logger(self,cfg):
    logger_init(self,cfg)
    import wandb
    assert wandb.run is not None and not wandb.run.offline,'W&B online is required'
    (RUN/'wandb-run.json').write_text(json.dumps({'id':wandb.run.id,'url':wandb.run.url,'entity':wandb.run.entity,'project':wandb.run.project,'group':wandb.run.group,'created_at':time.time()},indent=2)+'\n')
module.WandBLogger.__init__=init_logger
# Verify the author resume loader and model before the first new update.
original_make=module.make_policy
def make_verified(*args,**kwargs):
    policy=original_make(*args,**kwargs)
    from safetensors.torch import load_file
    expected=load_file(str(SOURCE/'pretrained_model/model.safetensors'),device='cpu')
    actual=policy.state_dict()
    assert set(expected)==set(actual)
    for key,value in expected.items():assert torch.equal(value,actual[key].detach().cpu()),key
    save(RUN/'restored-policy.json',{'source':str(SOURCE),'sha256':sha(SOURCE/'pretrained_model/model.safetensors'),'all_weights_equal_before_update':True})
    return policy
module.make_policy=make_verified
original_processors=module.make_pre_post_processors
def processors_verified(*args,**kwargs):
    pre,post=original_processors(*args,**kwargs)
    from lerobot.processor.normalize_processor import NormalizerProcessorStep,UnnormalizerProcessorStep
    from safetensors.torch import load_file
    checked=[]
    for pipeline,cls,name in [(pre,NormalizerProcessorStep,'policy_preprocessor_step_3_normalizer_processor.safetensors'),(post,UnnormalizerProcessorStep,'policy_postprocessor_step_0_unnormalizer_processor.safetensors')]:
        candidates=[v for v in pipeline.steps if isinstance(v,cls)];assert len(candidates)==1
        actual=candidates[0].state_dict();expected=load_file(str(SOURCE/'pretrained_model'/name),device='cpu')
        assert set(actual)==set(expected)
        for key,value in expected.items():assert torch.equal(value,actual[key].cpu()),key
        checked.append({'source':name,'tensors':len(expected),'all_equal':True})
    save(RUN/'restored-statistics.json',{'checked_before_first_update':True,'processors':checked})
    return pre,post
module.make_pre_post_processors=processors_verified
original_restore=module.load_training_state
def restore_verified(*args,**kwargs):
    result=original_restore(*args,**kwargs);step,opt,scheduler=result
    assert step==40000 and scheduler is None
    steps=[float(v['step']) for v in opt.state.values() if 'step' in v]
    assert steps and set(steps)=={40000.0}
    from lerobot.utils.random_utils import serialize_rng_state
    from safetensors.torch import load_file
    expected_rng=load_file(str(SOURCE/'training_state/rng_state.safetensors'),device='cpu')
    actual_rng=serialize_rng_state()
    assert set(expected_rng)==set(actual_rng)
    for key,value in expected_rng.items():assert torch.equal(value,actual_rng[key].cpu()),key
    save(RUN/'restored-rng-check.json',{'all_rng_tensors_equal_after_author_restore':True,'keys':sorted(expected_rng)})
    save(RUN/'restored-training-state.json',{'step':step,'optimizer_steps':sorted(set(steps)),'optimizer_step_tensors':len(steps),'rng_restored_by_author_loader':True,'rng_file_sha256':sha(SOURCE/'training_state/rng_state.safetensors')})
    return result
module.load_training_state=restore_verified
original_last=module.update_last_checkpoint
def checkpoint_ready(directory):
    result=original_last(directory);step=int(directory.name)
    assert step in [42000,44000,46000,48000,50000]
    assert json.loads((directory/'training_state/training_step.json').read_text())['step']==step
    save(RUN/f'checkpoint-ready-{step:06d}.json',{'step':step,'directory':str(directory),'at':time.time(),'model_sha256':sha(directory/'pretrained_model/model.safetensors')})
    return result
module.update_last_checkpoint=checkpoint_ready
update=module.update_policy;updates=[40000]
metrics=(RUN/'updates.jsonl').open('a',buffering=1)
def update_logged(*args,**kwargs):
    result=update(*args,**kwargs);updates[0]+=1;m=result[0]
    row={'update':updates[0],'at':time.time(),'loss':float(m.loss.val),'grad_norm':float(m.grad_norm.val),'lr':float(m.lr.val),'update_s':float(m.update_s.val)}
    assert all(math.isfinite(row[k]) for k in ['loss','grad_norm','lr'])
    metrics.write(json.dumps(row)+'\n')
    if updates[0]%100==0 or updates[0]==40001:
        p=RUN/'train-progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(p)
    return result
module.update_policy=update_logged
try:
    module.main()
    assert updates[0]==50000,updates[0]
    (RUN/'training-complete.json').write_text(json.dumps({'updates':updates[0],'new_updates':updates[0]-40000,'at':time.time(),'finite_guard_each_update':True},indent=2)+'\n')
finally:metrics.close()
