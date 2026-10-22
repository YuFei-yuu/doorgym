"""Existing RGB/frame extraction and static paired charts only. No replay/scene/model."""
from common import *
import os,time,math
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import av
from PIL import Image,ImageDraw
F=RUN/'FINAL';(F/'review-frames').mkdir(exist_ok=True);(F/'figures').mkdir(exist_ok=True)
review=[]
for seed in [62215,62203,62209]:
 d=RUN/'evaluation'/f'trial_{seed-62200:02d}';scored=read(d/'r11-score.json');video=d/'overview/env_0000/rollout.mp4';validation=read(d/'video-validation.json')
 assert sha(video)==validation['sha256'];n=validation['decoded_frames']
 with np.load(F/'timelines'/f'h5-s{seed}.npz') as a:rows=a['rows'];target=float(a['target_yaw'])
 events=[0,n-1]
 if scored['whole_body_pass_step'] is not None:
  frame=scored['whole_body_pass_step']//2;events += [frame+off for off in [-75,-25,0,25,75,250,500]]
 opened=np.flatnonzero(abs(rows[:,6])>=math.radians(80))
 if len(opened):events += [int(opened[0]//2)+off for off in [-25,0,25]]
 events += list(np.linspace(0,n-1,32,dtype=int))
 chosen=sorted(set(int(np.clip(f,0,n-1)) for f in events));dest=F/'review-frames'/str(seed);dest.mkdir(exist_ok=True)
 thumbs=[];kept=[]
 with av.open(str(video)) as c:
  stream=c.streams.video[0];stream.codec_context.thread_count=2;assert float(stream.average_rate)==25
  for i,frame in enumerate(c.decode(video=0)):
   if i in chosen:
    im=frame.to_image().convert('RGB');im.save(dest/f'{i:04d}.jpg',quality=90)
    thumb=Image.new('RGB',(400,248),'white');thumb.paste(im.resize((400,225)),(0,23));ImageDraw.Draw(thumb).text((6,5),f'seed {seed} | RGB frame {i} | video {i/25:.2f}s',fill='black');thumbs.append(thumb);kept.append(i)
   if i>=chosen[-1]:break
 sheet=Image.new('RGB',(1600,248*math.ceil(len(thumbs)/4)),'white')
 for i,im in enumerate(thumbs):sheet.paste(im,((i%4)*400,(i//4)*248))
 sheet.save(F/'figures'/f'failure-seed{seed}-event-sheet.jpg',quality=90)
 review.append(dict(seed=seed,original=str(video),sha256=validation['sha256'],frames=n,fps=25,selected_original_frames=kept,derived_event_sheet=f'figures/failure-seed{seed}-event-sheet.jpg',frame_time_label='Encoded RGB timestamp index/25; physical50Hz event times are in separate audited state timing',no_replay=True,new_scene_calls=0,visual_review_status='awaiting executor inspection'))
write(F/'video-review-index.json',review)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'figure.dpi':140,'savefig.dpi':180})
counts=read(F/'paired-results.json')['counts'];keys=['opening80','whole_body_passed_no_return','measured_end_push_screen','legal_relock','final_home2s','final_heading2s','final_base_still2s','final_common2s'];labels=['Open80','Far/no return','End-push screen','Legal relock','Home2s','Heading2s','Still2s','Common2s']
fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
for shift,h,c in [(-.18,25,'#4374aa'),(.18,5,'#cf7c32')]:
 b=ax.bar(np.arange(len(keys))+shift,[counts[str(h)][k] for k in keys],width=.34,color=c,label=f'Execution horizon {h}')
 ax.bar_label(b,fontsize=8)
ax.set_ylim(0,18);ax.set_xticks(range(len(keys)),labels,rotation=18);ax.set_ylabel('Episodes / same16 initial states');ax.set_title('Fixed R15 50k model: h25 original vs h5 new; full successes 0/16 in both');ax.legend();fig.savefig(F/'figures/paired-gates.png');fig.savefig(F/'figures/paired-gates.svg');plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained')
for h,c in [(25,'#4374aa'),(5,'#cf7c32')]:
 with np.load(F/'timelines'/f'h{h}-s62215.npz') as z:a=z['rows'];target=float(z['target_yaw'])
 err=(a[:,1]-target+np.pi)%(2*np.pi)-np.pi
 axes[0].plot(a[:,0],np.rad2deg(a[:,6]),label=f'h{h}',color=c)
 axes[1].plot(a[:,0],np.rad2deg(err),label=f'h{h}',color=c)
 axes[2].plot(a[:,0],a[:,3],label=f'h{h}',color=c)
axes[0].set_ylabel('Door angle (deg)');axes[1].set_ylabel('Wrapped heading error (deg)');axes[2].set_ylabel('Applied yaw rate (rad/s)');axes[2].set_xlabel('Actual simulation time (s)')
for ax in axes:ax.legend()
axes[1].axhspan(-2,2,color='green',alpha=.1);fig.suptitle('seed62215 paired existing rollouts; neither h5 nor h25 achieved legal relock')
fig.savefig(F/'figures/seed62215-paired-timing.png');fig.savefig(F/'figures/seed62215-paired-timing.svg');plt.close(fig)
print('Original RGB review sheets and static charts generated; new model/scene calls0',flush=True)
