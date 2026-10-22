"""Publication-style static diagnostics from sealed predictions only; CPU, no policy."""
from common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'figure.dpi':140,'savefig.dpi':180})
figdir=RUN/'figures';figdir.mkdir(exist_ok=True)
e=read(RUN/'expert-physical-errors.json')['models'];groups=['exit_turn','heading_restore','stop_switch','final_hold'];steps=[30000,40000,50000];colors=['#4169a1','#d08b24','#309266']
fig,ax=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
for q,step in enumerate(steps):
 entries={x['group']:x for x in e[str(step)] if x['horizon']=='first25'}
 for k,col in enumerate([0,1]):ax[k].bar(np.arange(4)+(q-1)*.24,[entries[g]['physical_mae'][col] for g in groups],width=.23,color=colors[q],label=f'{step//1000}k')
 ax[2].bar(np.arange(4)+(q-1)*.24,[entries[g]['predicted_stop_on_true_stop']/max(1,entries[g]['truth_stop']) for g in groups],width=.23,color=colors[q],label=f'{step//1000}k')
for a,title in zip(ax,['Forward speed MAE (m/s)','Yaw rate MAE (rad/s)','Stop recall on true stop frames']):
 a.set_title(title);a.set_xticks(range(4),['Exit turn','Restore','Stop switch','Final hold'],rotation=20);a.legend()
fig.suptitle('72 expert episodes, 4 frozen observations each; deployed inference, first 25 valid actions')
fig.savefig(figdir/'expert-errors.png');fig.savefig(figdir/'expert-errors.svg');plt.close(fig)
chunks=read(RUN/'chunk-target-timelines.json');fig,axes=plt.subplots(4,1,figsize=(11,10),layout='constrained')
for ax,seed in zip(axes,SEEDS):
 rows=[x for x in chunks if x['seed']==seed];t=[x['time_s'] for x in rows]
 ax.axhspan(-2,2,color='#409b70',alpha=.15,label='Heading tolerance +/-2 deg');ax.plot(t,[r['pre_error_deg'] for r in rows],color='#26384a',label='Wrapped measured heading error at replan');ax.set_ylabel('Heading error (deg)');ax.set_title(f'50k seed {seed}: actual post-relock committed segments')
 near=[r for r in rows if r['near_at_replan']];ax.scatter([r['time_s'] for r in near],[r['pre_error_deg'] for r in near],color='#d4493e',s=24,label='Near target at replan',zorder=3)
 twin=ax.twinx();twin.plot(t,[r['first25_mean_yaw_rate'] for r in rows],color='#b66d19',alpha=.7,label='Worker mean yaw command, next 25');twin.set_ylabel('Yaw rate (rad/s)')
 if seed==SEEDS[0]:ax.legend(loc='upper left',fontsize=7);twin.legend(loc='upper right',fontsize=7)
axes[-1].set_xlabel('Existing rollout time (s)');fig.savefig(figdir/'actual-heading-timing.png');fig.savefig(figdir/'actual-heading-timing.svg');plt.close(fig)
obs=read(RUN/'observations.json')['points'];pick=[p for p in obs if p['kind']=='failure' and abs(p['heading_error_rad'])<=np.deg2rad(2)]
case=[p for p in obs if p['kind']=='failure' and p['seed']==62202];pick=[min(case,key=lambda p:abs(p['heading_error_rad'])),*pick]
fig,axes=plt.subplots(len(pick),1,figsize=(10,2.2*len(pick)),layout='constrained')
for ax,p in zip(np.atleast_1d(axes),pick):
 for step,c in zip(steps,colors):
  with np.load(RUN/'predictions'/f'{step:06d}-{p["id"]}.npz') as z:y=z['physical']
  ax.plot(np.arange(100)/25,y[:,1],color=c,label=f'{step//1000}k')
 ax.axvline(1,color='black',ls='--',lw=1,label='25-action boundary');ax.axhspan(-.01,.01,color='#888888',alpha=.15);ax.set_title(f"{p['id']} | t={p['time_s']:.2f}s | measured heading error={np.rad2deg(p['heading_error_rad']):.3f} deg")
 ax.set_ylabel('Predicted yaw rate (rad/s)');ax.legend(ncol=4,fontsize=7)
axes[-1].set_xlabel('Predicted offset (s); later 75 actions are not executed as a single chunk')
fig.savefig(figdir/'near-target-predicted-chunks.png');fig.savefig(figdir/'near-target-predicted-chunks.svg');plt.close(fig)
print('Static figures complete; no policy constructed or forwarded',flush=True)
# Original policy-depth pair panels, not simulated images; nearest expert is descriptive only.
sim=read(RUN/'observation-similarity.json');near_map={r['failure_id']:r for r in sim['failure_nearest']};byid={p['id']:p for p in obs}
fig,axes=plt.subplots(4,4,figsize=(12,11),layout='constrained')
for row,seed in enumerate(SEEDS):
 p=min([p for p in obs if p['kind']=='failure' and p['seed']==seed],key=lambda p:abs(p['heading_error_rad']));match=near_map[p['id']];q=byid[match['expert_id']]
 with np.load(RUN/p['input']) as a,np.load(RUN/q['input']) as b:
  for col,(z,v,label) in enumerate([(a,'wrist','Failure wrist'),(a,'front','Failure front'),(b,'wrist','Nearest expert wrist'),(b,'front','Nearest expert front')]):
   ax=axes[row,col];ax.imshow(z[v][:,:,0],cmap='gray',vmin=0,vmax=255);ax.set_axis_off();ax.set_title(f"{label}\n{p['id'] if col<2 else q['id']}",fontsize=8)
fig.suptitle('Original frozen policy depth; nearest expert supplies no action truth for failures')
fig.savefig(figdir/'input-depth-comparison.png');plt.close(fig)
