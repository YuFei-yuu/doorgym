"""Read-only forensics of the stopped factory read; never bypasses its guard."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,pathlib,traceback
import numpy as np
import torch
from PIL import Image
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets import video_utils
from lerobot.datasets.utils import dataset_to_policy_features
R=pathlib.Path('/data/zyf/video2door-wc4-act/.repro/runs/wc4-fulltask-expand-r9-20261021')
F=R/'reader-failure';F.mkdir(exist_ok=False)
cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='local/wc4_fulltask_expand_r9_20261021',root=str(R/'lerobot/local/wc4_fulltask_expand_r9_20261021'),use_imagenet_stats=False),
    policy=ACTConfig(device='cpu',chunk_size=100,n_action_steps=50,use_action_loss_weight=False,plucker_conditioning=False,interaction_state_conditioning=False,camera_input_gating=False,handle_latent_aux=False,point_cloud_conditioning=False,end_signal_prediction=False),batch_size=16,num_workers=3)
ds=make_dataset(cfg);global_index=41661;record=ds.hf_dataset[global_index]
episode=int(record['episode_index']);ep=ds.meta.episodes[episode]
first=int(ep['dataset_from_index']);local=global_index-first
item=json.loads((R/'candidate-index.json').read_text())['episodes'][episode]
raw_path=pathlib.Path(item['candidate_npz']);captured=[];original=video_utils.decode_video_frames_torchcodec


def observe(path,timestamps,tolerance_s,*args,**kwargs):
    decoder=video_utils._default_decoder_cache.get_decoder(str(path));fps=decoder.metadata.average_fps
    indices=[round(float(t)*fps) for t in timestamps]
    frames=decoder.get_frames_at(indices=indices)
    native=frames.pts_seconds.detach().cpu().to(torch.float64)
    query64=torch.tensor([float(t) for t in timestamps],dtype=torch.float64)
    query_default=torch.tensor(timestamps);loaded_default=torch.tensor([p.item() for p in frames.pts_seconds])
    qdist=torch.cdist(query_default[:,None],loaded_default[:,None],p=1).min(1).values
    dist64=torch.cdist(query64[:,None],native[:,None],p=1).min(1).values
    view='wrist' if 'wrist' in str(path) else 'front'
    decoded=frames.data[0].permute(1,2,0).numpy()
    with np.load(raw_path,allow_pickle=True) as z:
        raw=z[view+'_masked_depth'][local].copy();phase=str(z['phase_names'][int(z['subtask_index'][local,0])])
    Image.fromarray(raw).save(F/(view+'-raw-depth.png'));Image.fromarray(decoded).save(F/(view+'-converted-decoded.png'))
    captured.append(dict(video=str(path),view=view,fps=fps,frame_indices=indices,queried_float64=query64.tolist(),native_pts_float64=native.tolist(),native_pts_dtype=str(frames.pts_seconds.dtype),
        comparator_query_dtype=str(query_default.dtype),comparator_loaded_dtype=str(loaded_default.dtype),comparator_query=query_default.tolist(),comparator_loaded=loaded_default.tolist(),
        default_dtype_distance_s=qdist.tolist(),diagnostic_float64_distance_s=dist64.tolist(),unchanged_tolerance_s=tolerance_s,
        raw_vs_selected_decoded_normalized_mae=float(np.mean(abs(decoded.astype(float)-raw[:,:,None])/255)),phase=phase))
    # Original function and threshold still decide. No successful fallback.
    return original(path,timestamps,tolerance_s,*args,**kwargs)


video_utils.decode_video_frames_torchcodec=observe
error=None
try:ds[global_index]
except Exception as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
finally:video_utils.decode_video_frames_torchcodec=original
features=dataset_to_policy_features(ds.features)
report=dict(global_dataset_index=global_index,episode=episode,source_trial=item['trial'],source_run=item['source_run'],source_seed=item['seed'],local_frame=local,
    local_dataset_timestamp=float(record['timestamp']),raw_physics_post_time_s=.02+.04*local,raw_npz=str(raw_path),captured=captured,error=error,
    policy_feature_parser={k:dict(type=v.type.value,shape=list(v.shape)) for k,v in features.items()},
    guard_changed=False,threshold_changed=False,backend_changed=False,dataset_changed=False,raw_evidence_changed=False,model_constructed=False,
    interpretation='Diagnostic float64 distances are calculations only; the unmodified actual factory call still raises and is NOT accepted. No repacking or reader repair applied.')
(F/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
assert error and error['type']=='FrameTimestampError'
print(json.dumps(report,indent=2))
