"""Exactly one deployed-mode chunk100 per frozen point/model, with durable claims.
The model forward pre-hook is the budget boundary; no training/VAE future inputs.
"""
import os,sys,time,json,gc
from common import *
import numpy as np
import torch
from safetensors.torch import load_file
sys.path.insert(0,str(ROOT/'high-level/dp'))
from door_policy_backend import DoorPolicyController,load_lerobot_policy_normalizer_stats
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
assert not torch.cuda.is_available() and not os.environ.get('CUDA_VISIBLE_DEVICES')
torch.set_num_threads(2);torch.set_num_interop_threads(1)
frozen=read(RUN/'FROZEN.json');assert frozen['status']=='sealed_before_any_R16_policy_forward'
for rel,h in frozen['files'].items():assert sha(RUN/rel)==h,rel
for rel,h in read(RUN/'source-binding.json').items():assert sha(ROOT/rel)==h,rel
binding=read(RUN/'model-binding.json')
for model in binding.values():
 for name,h in model['files'].items():assert sha(name)==h,name
points=read(RUN/'observations.json')['points'];assert len(points)==512
ledger=(RUN/'forward-budget.jsonl').open('x',buffering=1);outdir=RUN/'predictions';outdir.mkdir(exist_ok=False)
count=0;active={};captured={};started=time.time();prechecks=[]
def record(x):ledger.write(json.dumps(x,allow_nan=False)+'\n');ledger.flush();os.fsync(ledger.fileno())
def claim(module,args):
 global count
 assert count<BUDGET and active and args[0]['observation.state'].shape[0]==1
 assert 'action' not in args[0] and 'action_is_pad' not in args[0]
 count+=1;record(dict(event='forward_claim',sequence=count,at=time.time(),**active,batch_size=1,chunk=100,device='cpu',deployment_eval=not module.training,future_action_input=False))
def capture(module,args,output):captured['normalized']=output[0].detach().cpu().clone()
for step in [30000,40000,50000]:
 model=binding[str(step)];ctl=DoorPolicyController(model['export'],device='cpu',action_horizon=25)
 backend=ctl.backend;assert ctl.obs_horizon==1 and ctl.pred_horizon==100 and ctl.action_horizon==25 and ctl.action_dim==9
 assert not ctl.pointcloud_conditioning and not ctl.plucker_conditioning and not ctl.end_signal_prediction
 official=Path(model['official']);weights=load_file(str(official/'model.safetensors'),device='cpu')
 assert len(weights)==234
 for k,v in backend.policy.state_dict().items():assert torch.equal(v.cpu(),weights[k]),k
 del weights
 cfg=PreTrainedConfig.from_pretrained(str(official));cfg.device='cpu'
 pre,post=make_pre_post_processors(cfg,pretrained_path=str(official),preprocessor_overrides={'device_processor':{'device':'cpu'},'normalizer_processor':{'device':'cpu'}},postprocessor_overrides={'unnormalizer_processor':{'device':'cpu'}})
 backend.policy.eval();hook=backend.policy.model.register_forward_pre_hook(claim);hook2=backend.policy.model.register_forward_hook(capture)
 stat=load_lerobot_policy_normalizer_stats(official)
 for key,vals in stat.items():
  for key2,v in vals.items():
   if key in backend.stats and key2 in backend.stats[key]:assert torch.equal(v.cpu(),backend.stats[key][key2].cpu())
 maxpre=0.;maxpost=0.;maxsanitize=0.;hashes=[]
 for i,p in enumerate(points):
  assert sha(RUN/p['input'])==p['input_sha256']
  with np.load(RUN/p['input']) as a:raw={k:a[k].copy() for k in a.files}
  item=ctl._make_item(raw['state'],None,raw['wrist'],front_second_rgb=raw['front'])
  batch=ctl._batch_from_windows([[item]])
  assert set(batch)=={'observation.state',*backend.image_keys}
  off=pre({k:v.clone() for k,v in batch.items()});norm=backend.normalizer.normalize_batch(batch,include_action=False)
  dif={k:float((off[k]-norm[k]).abs().max()) for k in batch};maxpre=max(maxpre,*dif.values());assert max(dif.values())<=1e-6,dif
  active=dict(model_step=step,observation=p['id'],input_sha256=p['input_sha256']);captured.clear()
  t=time.time();physical=ctl.predict_action_chunks_from_batch(batch).cpu().numpy();elapsed=time.time()-t
  normalized=captured['normalized'];expected=post(normalized.clone()).cpu().numpy();rawphysical=backend.normalizer.denormalize_action(normalized).cpu().numpy()
  assert physical.shape==(1,100,9) and np.isfinite(physical).all()
  maxpost=max(maxpost,float(abs(expected-rawphysical).max()));assert abs(expected-rawphysical).max()<=1e-6
  maxsanitize=max(maxsanitize,float(abs(physical-rawphysical).max()))
  path=outdir/f'{step:06d}-{p["id"]}.npz';np.savez_compressed(path,physical=physical[0].astype(np.float32),normalized=normalized.numpy()[0].astype(np.float32))
  record(dict(event='forward_completed',sequence=count,at=time.time(),**active,elapsed_s=elapsed,prediction=str(path.relative_to(RUN)),prediction_sha256=sha(path),preprocessing_max_error=dif,normalized_input_sha256={k:ahash(v.cpu().numpy()) for k,v in norm.items()},state_preprocess_max_change=float(abs(item['observation.state'].cpu().numpy()-raw['state']).max())))
  if (i+1)%32==0:print('progress',step,i+1,'total',count,'elapsed_s',round(time.time()-started,1),flush=True)
 hook.remove();hook2.remove()
 prechecks.append(dict(step=step,model_weights_equal=234,normalization_states_equal=True,official_preprocess_max_error=maxpre,official_postprocess_max_error=maxpost,controller_sanitize_max_change=maxsanitize,config=ctl.config,sidecar=ctl.sidecar_config))
 del ctl,backend,pre,post;gc.collect()
assert count==1536
write(RUN/'inference-complete.json',dict(status='budget_complete',forward_claims=count,completed=count,model_observation_pairs=count,unique_observations=len(points),batch_size=1,elapsed_s=time.time()-started,device='cpu',training=0,simulation=0,closed_loop=0,repeats=0,checks=prechecks))
ledger.close();print('CPU INFERENCE BUDGET COMPLETE',count,flush=True)
