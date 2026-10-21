"""Standalone evidence plots from original raw pixels and source actions."""
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path('/data/zyf/video2door-wc4-act/.repro/runs/wc4-fulltask-pilot-r8-20261021')
F=R/'dataset-check';report=json.loads((F/'learnability-diagnostic.json').read_text())
pairs=report['confirmed_pairs'];index=json.loads((R/'candidate-index.json').read_text())['episodes']
if pairs:
    # Prefer the nominal same-episode example with exact front-depth equality.
    c=next((x for x in pairs if x['episode_a']==0 and x['episode_b']==0),pairs[0])
    samples=[]
    for suffix in ['a','b']:
        ep=c['episode_'+suffix];frame=c['frame_'+suffix]
        with np.load(index[ep]['candidate_npz'],allow_pickle=True) as z:
            samples.append(dict(ep=ep,frame=frame,state=z['state'][frame],action=z['action'][frame],
                front=z['front_masked_depth'][frame],wrist=z['wrist_masked_depth'][frame],
                chunk=z['action'][np.minimum(frame+np.arange(100),len(z['action'])-1)],
                phase=str(z['phase_names'][int(z['subtask_index'][frame,0])]),physical_time=.02+frame*.04))
    fig,axs=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
    for col,s in enumerate(samples):
        axs[0,col].imshow(s['front'],cmap='gray',vmin=0,vmax=255);axs[0,col].set_title('Front | ep%d frame%d | %s'%(s['ep'],s['frame'],s['phase']))
        axs[1,col].imshow(s['wrist'],cmap='gray',vmin=0,vmax=255);axs[1,col].set_title('Wrist | observation physical t~%.2fs'%s['physical_time'])
        for ax in axs[:2,col]:ax.axis('off')
    for j,s in enumerate(samples):
        axs[2,0].plot(np.arange(100)/25,s['chunk'][:,0],label='A' if j==0 else 'B')
        axs[2,1].plot(np.arange(100)/25,s['chunk'][:,1],label='A' if j==0 else 'B')
    axs[2,0].set(xlabel='Future offset (s)',ylabel='vx (m/s)',title='Different future forward-speed supervision')
    axs[2,1].set(xlabel='Future offset (s)',ylabel='yaw rate (rad/s)',title='Different future turning supervision')
    for ax in axs[2,:]:ax.legend();ax.grid(alpha=.25)
    fig.suptitle('Near-identical dual depth / Joint9 state, different ACT chunk100\nDiagnostic only; no added stage/history inputs or model changes')
    fig.savefig(F/'learnability-example.png',dpi=160);fig.savefig(F/'learnability-example.svg');plt.close(fig)
    (F/'learnability-example.json').write_text(json.dumps(c,indent=2)+'\n')
ledger=[json.loads(l) for l in (R/'attempts.jsonl').open()]
fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
for x in ledger:
    i=x['trial'];p=R/('trial_%02d'%i);rows=[json.loads(l) for l in (p/'states.jsonl').open()]
    tf=np.asarray(json.loads((p/'scene-frame.json').read_text())['world_from_reference']);xy=np.array([s['actual_base'][:2] for s in rows if s['passed']])-tf[:2,3]
    axs[0].plot(xy[:,0],xy[:,1],ls='-' if i not in [4,7] else '--',label='trial %d%s'%(i,' (withheld)' if i in [4,7] else ''))
axs[0].set_aspect('equal');axs[0].set(xlabel='Door-reference x (m)',ylabel='Door-reference y (m)',title='All eight original measured paths');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
axs[1].bar([x['trial'] for x in ledger],[x['frames'] for x in ledger],color=['#c99b37' if x['trial'] in [4,7] else '#288565' for x in ledger])
axs[1].set(xlabel='Frozen trial index',ylabel='25Hz frames',title='Variable-length complete records (no 550-frame cut)');axs[1].grid(axis='y',alpha=.2)
fig.savefig(R/'pilot-paths-and-lengths.png',dpi=160);plt.close(fig)
