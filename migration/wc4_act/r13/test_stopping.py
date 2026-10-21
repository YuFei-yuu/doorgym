"""CPU toy update fixtures; no ACT construction/training. Real author serializer."""
import os,sys,json,signal,subprocess,random
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from types import SimpleNamespace
from stopping import StopBoundary,BoundaryStop,commit_checkpoint,request_stop
from lerobot.utils.train_utils import save_checkpoint,load_training_state
from lerobot.utils.random_utils import serialize_rng_state,deserialize_rng_state,set_seed


class Toy(torch.nn.Module):
    def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor([1.,2.]))
    def save_pretrained(self,p):
        p=Path(p);p.mkdir(parents=True,exist_ok=True);save_file(self.state_dict(),str(p/'model.safetensors'))

class Config:
    peft=None
    def save_pretrained(self,p):(Path(p)/'train_config.json').write_text('{"fixture_only":true}\n')

class Processor:
    def __init__(self,name):self.name=name
    def save_pretrained(self,p):save_file({'state.mean':torch.arange(9.)},str(Path(p)/(self.name+'.safetensors')))


def exercise(out,trigger):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    set_seed(19);model=Toy();opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=2,gamma=.5)
    cfg=Config();pre=Processor('pre');post=Processor('post')
    boundary=None
    def writer(directory,step):save_checkpoint(directory,step,cfg,model,opt,scheduler,pre,post)
    def save(step):return commit_checkpoint(out/'checkpoints',step,writer)
    boundary=StopBoundary(out,save)
    boundary.install_signals()
    if trigger=='before':request_stop(out,'fixture before update')
    def toy_update():
        opt.zero_grad();loss=((model.w-torch.rand(2))**2).sum()
        if trigger=='forward':os.kill(os.getpid(),signal.SIGTERM)
        loss.backward()
        if trigger=='backward':os.kill(os.getpid(),signal.SIGINT)
        opt.step()
        if trigger=='optimizer':os.kill(os.getpid(),signal.SIGTERM)
        scheduler.step()
        if trigger in ['after','ignored_int']:os.kill(os.getpid(),signal.SIGINT)
        return float(loss)
    try:
        for _ in range(3):boundary.update(toy_update)
    except BoundaryStop:pass
    else:raise AssertionError('Stop not honored')
    finally:boundary.close()
    expected=0 if trigger=='before' else 1
    assert boundary.completed==expected
    stopped=json.loads((out/'STOPPED.json').read_text());assert stopped['uncertain_updates']==0
    checkpoint=Path(stopped['checkpoint']['path']);manifest=json.loads((checkpoint/'COMMITTED.json').read_text())
    assert manifest['step']==expected
    weights={k:v.detach().clone() for k,v in model.state_dict().items()};rng=serialize_rng_state()
    for k,v in load_file(str(checkpoint/'pretrained_model/model.safetensors')).items():assert torch.equal(v,weights[k])
    restored=Toy();restored.load_state_dict(load_file(str(checkpoint/'pretrained_model/model.safetensors')))
    restored_opt=torch.optim.AdamW(restored.parameters(),lr=1e-3)
    restored_scheduler=torch.optim.lr_scheduler.StepLR(restored_opt,step_size=2,gamma=.5)
    step,restored_opt,_=load_training_state(checkpoint,restored_opt,restored_scheduler)
    assert step==expected and restored_scheduler.state_dict()==scheduler.state_dict()
    actual_rng=serialize_rng_state();assert all(torch.equal(v,actual_rng[k]) for k,v in rng.items())
    assert all(float(s['step'])==expected for s in restored_opt.state.values())
    # Identical next TOY transition from the saved boundary; not an ACT update.
    def next_step(m,o,s):
        o.zero_grad();loss=((m.w-torch.rand(2))**2).sum();loss.backward();o.step();s.step()
    deserialize_rng_state(rng);next_step(model,opt,scheduler)
    deserialize_rng_state(rng);next_step(restored,restored_opt,restored_scheduler)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    assert torch.equal(load_file(str(checkpoint/'pretrained_model/pre.safetensors'))['state.mean'],torch.arange(9.))
    return dict(trigger=trigger,completed=expected,uncertain=0,model_optimizer_scheduler_rng_processors_equal=True,next_toy_transition_exact=True)

if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[2]=='child':
        # Parent deliberately ignores SIGINT; installed handler must override it.
        result=exercise(sys.argv[1],'ignored_int');print(json.dumps(result));raise SystemExit(0)
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    results=[exercise(out/k,k) for k in ['before','forward','backward','optimizer','after']]
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    with (out/'nohup-child.log').open('x') as log:
        subprocess.run([sys.executable,'-B',__file__,str(out/'ignored_int'),'child'],stdout=log,stderr=subprocess.STDOUT,check=True)
    results.append(json.loads((out/'nohup-child.log').read_text().splitlines()[-1]))
    # Serialization failure leaves no published target or STOPPED clean receipt.
    failed=out/'save-failure';failed.mkdir()
    def broken(p,s):
        (p/'partial').write_text('not committed');raise OSError('fixture serialization failure')
    try:commit_checkpoint(failed,7,broken)
    except OSError:pass
    else:raise AssertionError('Expected serialization failure')
    assert not (failed/'stop-000007').exists()
    # Unexpected partial update is explicitly unsafe, never clean/resumable.
    unsafe=out/'unsafe';unsafe.mkdir();b=StopBoundary(unsafe,lambda step:None)
    def crash():raise RuntimeError('fixture optimizer failure')
    try:b.update(crash)
    except RuntimeError:pass
    finally:b.close()
    assert (unsafe/'UNSAFE_STOP.json').exists() and not (unsafe/'STOPPED.json').exists()
    (out/'result.json').write_text(json.dumps(dict(status='passed',cases=results,failed_save_not_published=True,
        unexpected_fault_not_claimed_clean=True,ACT_training_updates=0,real_policy_inference=0),indent=2)+'\n')
    print('PASS: CPU toy stop before/forward/backward/optimizer/after/nohup; exact restore and next toy transition; failed-save/unsafe guards')
