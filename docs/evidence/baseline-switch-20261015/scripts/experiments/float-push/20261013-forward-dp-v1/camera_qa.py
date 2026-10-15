"""Visual inspection montage from the first successful new expert trajectory."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=Path(a.root)
keys=['wrist_handle_mask','wrist_masked_depth','front_handle_mask','front_masked_depth']
with np.load(root/'dataset/train_seed_16040/trajectory.npz') as z:
    n=len(z['state']);frames=[0,n//3,2*n//3,n-1]
    fig,axs=plt.subplots(4,4,figsize=(12,8),constrained_layout=True)
    for i,frame in enumerate(frames):
        for j,key in enumerate(keys):
            axs[i,j].imshow(z[key][frame]);axs[i,j].axis('off')
            axs[i,j].set_title('%s, frame %d'%(key,frame),fontsize=8)
    (root/'analysis').mkdir(exist_ok=True)
    fig.savefig(root/'analysis/camera_qa.png',dpi=160)
