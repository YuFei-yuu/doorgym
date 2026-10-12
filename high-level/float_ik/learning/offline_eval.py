"""Expert-observation diagnostics through the original DoorDPPolicyController."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from artifacts import resolve_recorded_path,write_new_json
from dataset_contract import validate_episode,layout


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42);p.add_argument('--episodes',type=int,default=4);p.add_argument('--windows',type=int,default=32);a=p.parse_args();root=Path(a.root)
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'dp'))
    from door_dp_common import DoorDPPolicyController
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    controller=DoorDPPolicyController(resolve_recorded_path(root,a.checkpoint),device=a.device)
    manifest=json.loads((root/'manifest.json').read_text());errors=[];rotations=[];samples=[];names=None
    for item in [x for x in manifest if x['split']=='val' and x['success']][:a.episodes]:
        with np.load(resolve_recorded_path(root,item['episode'])) as z:
            meta=validate_episode(z);names=meta['action_names'];idx=layout(names)
            if len(names)!=controller.action_dim or len(meta['state_names'])!=controller.config['state_dim']:raise ValueError('Checkpoint/recording dimensions differ')
            if controller.action_horizon>controller.pred_horizon:raise ValueError('Execution window exceeds prediction window')
            first=controller.obs_horizon-1;last=len(z['action'])-controller.pred_horizon
            if first>last:raise ValueError('Episode is too short for original sequence windows')
            for center in np.unique(np.linspace(first,last,a.windows,dtype=int)):
                controller.obs_buffer.clear();controller.action_queue.clear()
                for frame in range(center-first,center+1):
                    if 'images' in z:rgb=[np.repeat(im[:,:,None],3,-1) for im in z['images'][frame]]
                    else:rgb=[z[k][frame] for k in ('wrist_handle_mask','wrist_masked_depth','front_handle_mask','front_masked_depth')]
                    controller.append_observation(z['state'][frame],*rgb)
                controller.sample_action_chunk();pred=np.asarray(list(controller.action_queue),float);truth=z['action'][center:center+len(pred)].astype(float)
                errors.extend(pred-truth)
                for action,target in zip(pred,truth):
                    q=action[idx['quaternion']];norm=np.linalg.norm(q)
                    rotations.append(2*np.arccos(np.clip(abs(np.dot(q/norm,target[idx['quaternion']])),0,1)) if norm>1e-8 else np.nan)
                samples.append(dict(seed=item['seed'],frame=int(center)))
    if not errors:raise ValueError('No accepted validation windows')
    error=np.asarray(errors);rmse=np.sqrt(np.mean(error**2,axis=0))
    result=dict(checkpoint=a.checkpoint,action_names=names,samples=len(error),windows=samples,rmse_per_channel=rmse,position_vector_rmse_m=np.linalg.norm(rmse[layout(names)['position']]),orientation_p95_rad=np.percentile(rotations,95),invalid_rotation_count=int((~np.isfinite(rotations)).sum()),inference_steps=len(controller.noise_scheduler.timesteps),clip_sample=bool(controller.noise_scheduler.config.clip_sample))
    write_new_json(a.output,result);print(json.dumps(dict(output=a.output,samples=len(error))))


if __name__=='__main__':main()
