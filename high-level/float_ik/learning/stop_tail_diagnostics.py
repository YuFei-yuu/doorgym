"""Fixed expert windows anchored to observed replans; separate inference process."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'high-level/dp'))


def select_windows(first_stop,replans,last_center):
    preceding=[c for c in replans if c<=first_stop]
    if not preceding:raise ValueError('No actual replan before the first expert stop')
    first=preceding[-1]
    return [dict(center=c,stage='conversion' if c<=first_stop else 'hold',first_stop=first_stop,
                 stop_offset=first_stop-c,history_first=max(0,c-15),history_last=c)
            for c in replans if first<=c<=last_center]


def run(root,step):
    # Deployment Gym must load Isaac Gym before its custom Torch runtime.
    from isaacgym import gymapi
    import numpy as np
    import torch
    from door_dp_common import DoorDPPolicyController
    from artifacts import finite_json
    from diagnostic_integrity import verify_frozen
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(2)
    verify_frozen(root,REPO)
    plan=json.loads((root/'diagnostic_plan.json').read_text())
    output=root/'diagnostics'/('step_%06d'%step);output.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(plan['baseline_checkpoint']) if step==1000 else root/'pilot/training/checkpoints'/('step_%06d.pt'%step)
    controller=DoorDPPolicyController(checkpoint,device='cuda:0',num_inference_steps=100,action_horizon=16)
    predictions=[];targets=[];metadata=[];errors={'conversion':[],'hold':[]}
    for episode in plan['episodes']:
        with np.load(episode['trajectory'],allow_pickle=True) as raw:
            state=raw['state'];actions=raw['action'];images=[raw[key] for key in ['wrist_handle_mask','wrist_masked_depth','front_handle_mask','front_masked_depth']]
            controller.obs_buffer.clear();controller.action_queue.clear();windows={w['center']:w for w in episode['windows']}
            for frame in range(max(windows)+1):
                controller.append_observation(state[frame],*[image[frame] for image in images])
                if frame not in windows:continue
                window=windows[frame]
                for noise_seed in plan['diffusion_seeds']:
                    torch.manual_seed(noise_seed);torch.cuda.manual_seed_all(noise_seed)
                    controller.action_queue.clear();controller.sample_action_chunk()
                    predicted=np.asarray(list(controller.action_queue),dtype=np.float32)
                    target=actions[frame:frame+16]
                    if predicted.shape!=(16,10) or not np.isfinite(predicted).all():raise ValueError('Invalid diagnostic predictions')
                    error=predicted[:,:2]-target[:,:2];errors[window['stage']].append(error)
                    predictions.append(predicted[:,:2]);targets.append(target[:,:2])
                    metadata.append(dict(window,seed=episode['seed'],diffusion_seed=noise_seed))
    stage_metrics={key:dict(vx_rmse=float(np.sqrt(np.mean(np.asarray(value)[:,:,0]**2))),
        yaw_rmse=float(np.sqrt(np.mean(np.asarray(value)[:,:,1]**2))),windows=len(value),executed_offsets=16) for key,value in errors.items()}
    np.savez_compressed(output/'predictions.npz',prediction=np.asarray(predictions),expert=np.asarray(targets),
        center=np.asarray([v['center'] for v in metadata]),seed=np.asarray([v['seed'] for v in metadata]),noise_seed=np.asarray([v['diffusion_seed'] for v in metadata]))
    value=dict(step=step,checkpoint=str(checkpoint),stages=stage_metrics,windows=metadata,
        original_controller=True,inference_steps=100,executed_action_prefix=16,prediction_horizon=32,
        same_windows_and_diffusion_seeds=True,history_is_causal=True,
        note='Expert observations only up to the actual anchored replan center; no action-execution-time future observation is fed to an earlier queue prediction.')
    (output/'summary.json').write_text(json.dumps(finite_json(value),indent=2)+'\n')
    for seed in [16040,16041,16042]:
        selected=[i for i,v in enumerate(metadata) if v['seed']==seed and v['diffusion_seed']==plan['diffusion_seeds'][0]]
        fig,axes=plt.subplots(2,len(selected),figsize=(3*len(selected),5),squeeze=False,constrained_layout=True)
        for col,index in enumerate(selected):
            for ch in range(2):
                ax=axes[ch,col];ax.plot(range(16),targets[index][:,ch],label='Expert');ax.plot(range(16),predictions[index][:,ch],label='DP')
                ax.set_title('center%d %s'%(metadata[index]['center'],metadata[index]['stage']));ax.grid(alpha=.2)
                ax.set_ylabel('vx m/s' if ch==0 else 'yaw rad/s');ax.set_xlabel('Executed offset0-15')
                if col==0:ax.legend(fontsize=7)
        fig.savefig(output/('seed_%d_predictions.png'%seed),dpi=130);plt.close(fig)
    verify_frozen(root,REPO)
    print(json.dumps(dict(step=step,stages=stage_metrics,draws=len(metadata))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--step',type=int,choices=[1000,2000,3000,5000],required=True)
    args=parser.parse_args();run(Path(args.root).resolve(),args.step)
