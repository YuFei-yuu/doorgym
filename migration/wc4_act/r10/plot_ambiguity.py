import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RUN, OLD, read
R=OLD
index=read(OLD/'candidate-index.json')['episodes']
F=RUN/'dataset-check'
if (F/'learnability-diagnostic.json').exists():
    report=read(F/'learnability-diagnostic.json')
    for h in [1,25,100]:
        possible=[c for c in report['reviewed_by_horizon'][str(h)] if c['near_full_depth'] and c['significant_by_horizon'][str(h)]]
        if not possible:continue
        c=possible[0];samples=[]
        for suffix in ['a','b']:
            ep=c['episode_'+suffix];frame=c['frame_'+suffix]
            with np.load(index[ep]['candidate_npz'],allow_pickle=True) as raw:
                samples.append(dict(ep=ep,frame=frame,front=raw['front_masked_depth'][frame],wrist=raw['wrist_masked_depth'][frame],chunk=raw['action'][np.minimum(frame+np.arange(100),len(raw['action'])-1)]))
        fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
        for col,s in enumerate(samples):
            for row,view in enumerate(['front','wrist']):
                axs[row,col].imshow(s[view],cmap='gray',vmin=0,vmax=255);axs[row,col].set_title('%s ep%d frame%d'%(view,s['ep'],s['frame']));axs[row,col].axis('off')
        for j,s in enumerate(samples):
            for k in range(2):axs[2,k].plot(np.arange(c['common_valid_future_frames'])/25,s['chunk'][:c['common_valid_future_frames'],k],label='A' if j==0 else 'B')
        for k in range(2):
            axs[2,k].axvspan(0,24/25,color='gray',alpha=.1);axs[2,k].set(xlabel='Future offset (s)',ylabel='vx (m/s)' if k==0 else 'yaw rate (rad/s)');axs[2,k].legend();axs[2,k].grid(alpha=.2)
        fig.suptitle('Near-observation diagnostic | significant horizon %d frames\nShaded: first25 | full curves: chunk100 | no added inputs'%h)
        fig.savefig(F/('alias-horizon-%03d.png'%h),dpi=160);plt.close(fig)
