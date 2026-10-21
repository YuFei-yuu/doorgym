"""Standalone audit figures from measured states and original depth pixels."""
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collect import ROOT,read
R=ROOT/'.repro/runs/wc4-fulltask-expand-r9-20261021'
rows=[json.loads(l) for l in (R/'attempts.jsonl').open()]
fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
for a in rows:
    p=R/('trial_%02d'%a['trial']);tf=np.asarray(read(p/'scene-frame.json')['world_from_reference'])
    xy=[]
    for n,line in enumerate((p/'states.jsonl').open()):
        if n%5:continue
        s=json.loads(line)
        if s['passed']:xy.append(s['actual_base'][:2])
    if xy:
        xy=np.asarray(xy)-tf[:2,3]
        axes[0].plot(xy[:,0],xy[:,1],alpha=.25,lw=.8,color='#23836d' if a['status']=='candidate' else '#ae8135')
    initial=a['parameters']['effective_sampling']
    axes[1].scatter(initial['robot_x'],np.degrees(initial['robot_pitch']),s=23,color='#23836d' if a['status']=='candidate' else '#ae8135',marker='o' if a['status']=='candidate' else 'x')
axes[0].set_aspect('equal');axes[0].set(xlabel='Door-reference x (m)',ylabel='Door-reference y (m)',title='Original measured far-side paths')
axes[1].set(xlabel='Initial robot x (m)',ylabel='Initial robot pitch (deg)',title='Green: candidates | amber: withheld/failed')
index=read(R/'candidate-index.json')['episodes']
axes[2].bar(np.arange(len(index)),[e['frames'] for e in index],color=['#517db4']*6+['#23836d']*(len(index)-6))
axes[2].set(xlabel='Candidate episode',ylabel='25Hz frames',title='Variable length | first six retained R8')
for ax in axes:ax.grid(alpha=.2)
fig.savefig(R/'coverage-paths-lengths.png',dpi=160);plt.close(fig)
F=R/'dataset-check'
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
            for k in range(2):axs[2,k].plot(np.arange(100)/25,s['chunk'][:,k],label='A' if j==0 else 'B')
        for k in range(2):
            axs[2,k].axvspan(0,24/25,color='gray',alpha=.1);axs[2,k].set(xlabel='Future offset (s)',ylabel='vx (m/s)' if k==0 else 'yaw rate (rad/s)');axs[2,k].legend();axs[2,k].grid(alpha=.2)
        fig.suptitle('Near-observation diagnostic | significant horizon %d frames\nShaded: first25 | full curves: chunk100 | no added inputs'%h)
        fig.savefig(F/('alias-horizon-%03d.png'%h),dpi=160);plt.close(fig)
