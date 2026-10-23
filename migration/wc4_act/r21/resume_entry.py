"""Explicitly user-authorized R20 continuation; verify restored state before update."""
import sys,os,runpy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R13=ROOT/'migration/wc4_act/r13';SUPPORT=ROOT/'migration/wc4_act'
sys.path[:0]=[str(R13),str(SUPPORT)]
import overview
from safetensors.torch import load_file
import torch
from lerobot.utils.random_utils import serialize_rng_state
original_load=overview.load;held={}

def load(name,relative):
    module=original_load(name,relative)
    if name!='r13_author_train':return module
    make=module.make_policy;processors=module.make_pre_post_processors;restore=module.load_training_state
    def policy(*a,**k):
        result=make(*a,**k);held['policy']=result;return result
    def prepost(*a,**k):
        result=processors(*a,**k);held['processors']=result;return result
    def restored(*a,**k):
        result=restore(*a,**k);step,opt,scheduler=result
        assert step==75000 and scheduler is None
        checkpoint=Path(a[0]);expected=load_file(str(checkpoint/'pretrained_model/model.safetensors'),device='cpu')
        actual=held['policy'].state_dict();assert set(expected)==set(actual)
        for key,value in expected.items():assert torch.equal(value,actual[key].detach().cpu()),key
        from lerobot.optim.optimizers import flatten_dict
        state=opt.state_dict();live_optimizer=flatten_dict({'state':state['state']})
        saved_optimizer=load_file(str(checkpoint/'training_state/optimizer_state.safetensors'),device='cpu')
        assert set(live_optimizer)==set(saved_optimizer)
        for key,value in saved_optimizer.items():assert torch.equal(value,live_optimizer[key].detach().cpu()),key
        assert json.loads(json.dumps(state['param_groups']))==json.loads((checkpoint/'training_state/optimizer_param_groups.json').read_text())
        steps=[float(v['step']) for v in opt.state.values() if 'step' in v];assert steps and set(steps)=={75000.}
        expected_rng=load_file(str(checkpoint/'training_state/rng_state.safetensors'),device='cpu');actual_rng=serialize_rng_state()
        assert set(expected_rng)==set(actual_rng)
        for key,value in expected_rng.items():assert torch.equal(value,actual_rng[key].cpu()),key
        from lerobot.processor.normalize_processor import NormalizerProcessorStep,UnnormalizerProcessorStep
        checked=[]
        for pipeline,cls,file in [(held['processors'][0],NormalizerProcessorStep,'policy_preprocessor_step_3_normalizer_processor.safetensors'),
          (held['processors'][1],UnnormalizerProcessorStep,'policy_postprocessor_step_0_unnormalizer_processor.safetensors')]:
            live=next(p for p in pipeline.steps if isinstance(p,cls)).state_dict();saved=load_file(str(checkpoint/'pretrained_model'/file),device='cpu')
            assert set(live)==set(saved)
            for key,value in saved.items():assert torch.equal(value,live[key].cpu()),key
            checked.append(file)
        run=Path(os.environ['WC4_EXECUTION_RUN'])
        (run/'restoration-before-first-update.json').write_text(json.dumps(dict(step=step,model_all_tensors_equal=True,
          optimizer_step_values=sorted(set(steps)),optimizer_step_tensors=len(steps),optimizer_all_tensors_equal=True,optimizer_tensor_count=len(saved_optimizer),optimizer_param_groups_equal=True,RNG_all_tensors_equal=True,
          CUDA_RNG_present='torch_cuda_rng_state' in actual_rng,processor_states_equal=checked,
          source=str(checkpoint),dataloader_prefetch_cursor_restored=False,original_author_resume=True),indent=2)+'\n')
        return result
    module.make_policy=policy;module.make_pre_post_processors=prepost;module.load_training_state=restored
    return module
overview.load=load
runpy.run_path(str(Path(__file__).with_name('train_entry.py')),run_name='__main__')
