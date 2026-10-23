"""Freeze one existing expert full-bypass geometric reference, diagnostics only."""
import json,pathlib,hashlib
import numpy as np
def freeze_route(run,root):
 r11=root/'.repro/runs/wc4-fulltask-reaudit-r11-20261021';prov=json.loads((r11/'unified-provenance.json').read_text())['episodes'][0]
 ledger=json.loads((r11/'measured-ledger.json').read_text())['trials'];entry=next(x for x in ledger if x['source_run']==prov['source_run'] and x['trial']==prov['trial'])
 scene=pathlib.Path(entry['source_dir'])/'scene-frame.json';W=np.asarray(json.loads(scene.read_text())['world_from_reference']);p=r11/'candidates/episode_000000.npz'
 with np.load(p,allow_pickle=True) as a:
  phase=a['subtask_index'].reshape(-1);ids=np.flatnonzero(phase==list(a['phase_names']).index('compact_bypass'));xy=a['replay_root_state'][ids,:2]
 xyz=np.c_[xy,np.zeros(len(xy)),np.ones(len(xy))];ref=(np.linalg.inv(W)@xyz.T).T[:,:2];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(ref,axis=0),axis=1))];assert arc[-1]>1
 keep=np.r_[True,np.diff(arc)>1e-10];grid=np.linspace(0,arc[-1],51);points=np.c_[np.interp(grid,arc[keep],ref[keep,0]),np.interp(grid,arc[keep],ref[keep,1])]
 x=dict(source_episode=0,source_sha256=prov['sha256'],scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),frame='actual door reference frame via inverse recorded world_from_reference',points_xy=points.tolist(),arclength_m=float(arc[-1]),selection='deterministic expert episode0, frozen before any R20 model result; no action truth transferred',diagnostic_only=True,visit_radius_m=.15,ordered_progress='greedy time-ordered visit of the51 reference points in increasing arclength; report reached prefix fraction, visited fraction, and closest distance before first push and before first post-fold deployment separately; no added success gate')
 (run/'route-reference.json').write_text(json.dumps(x,indent=2)+'\n')
