"""Read-only motion audit. Run in the existing pro5000 Gym CPU environment."""
import argparse, collections, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import imageio
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

ROOT=Path('/data/zyf/video2door-wc4-act')
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
sys.path.insert(0,str(ROOT/'migration/wc4_act/r11'))
from criteria import score
from fullcycle_geometry import URDF, transform
from newexpert_stage2.push_geometry import PadArm, PadCurve, PAD_LOCAL

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    run=a.run;attempt=run/'attempt_000';out=run/'audit';out.mkdir(exist_ok=True)
    assert (run/'entry-exit.json').exists(),'No audit while live'
    states=[json.loads(l) for l in (attempt/'states.jsonl').open()]
    locks=[json.loads(l) for l in (attempt/'lock-writes.jsonl').open()]
    result=read(attempt/'result.json');ref=read(attempt/'reference.json')
    n=len(states);t=np.array([s['t'] for s in states]);dt=t[0]
    phases=[s['phase'] for s in states]
    intervals=[]
    for i,s in enumerate(states):
        if not i or phases[i]!=phases[i-1]:
            if intervals:intervals[-1]['last_step']=i-1;intervals[-1]['last_time_s']=t[i-1]
            intervals.append({'phase':phases[i],'first_step':i,'first_time_s':t[i]})
    intervals[-1].update(last_step=n-1,last_time_s=t[-1]);write(out/'phase-intervals.json',intervals)
    after_pass=result['whole_body_pass_step']
    contact_pairs=collections.Counter();loaded_pairs=collections.Counter();events=[]
    rows=0
    for i,line in enumerate((attempt/'contacts.jsonl').open()):
        row=json.loads(line);assert row['step']==i;rows+=1;s=states[i]
        s.update(end_panel_raw_contact=False,loaded_door_contact=False,zero_lambda_door_records=0,nonend_door_records=0)
        for c in row['contacts']:
            door=[x for x in c['names'] if x.get('actor_handle')==2]
            robot=[x for x in c['names'] if x.get('actor_handle') in [0,1]]
            if not door or not robot:continue
            panel=door[0]['body']=='door_panel';end=robot[0]['body'] in ['gripperStator','gripperMover']
            s['end_panel_raw_contact']|=panel and end;s['loaded_door_contact']|=c['lambda']!=0
            s['zero_lambda_door_records']+=c['lambda']==0;s['nonend_door_records']+=not end
            if after_pass is not None and i>=after_pass:
                pair=robot[0]['body']+' / '+door[0]['body'];contact_pairs[pair]+=1
                if c['lambda']!=0:loaded_pairs[pair]+=1
                events.append({'step':i,'time_s':t[i],'phase':phases[i],'pair':pair,'lambda':c['lambda'],
                    'end_panel':panel and end,'door_delta_rad':s['door_q'][0]-s['pre_physics']['door_q'][0],
                    'raw_contact':c})
    assert rows==n
    with (out/'post-pass-door-contacts.jsonl').open('w') as f:
        for x in events:f.write(json.dumps(x)+'\n')
    scoring=score(states,locks,ref);write(out/'r11-score.json',scoring)
    arm=PadArm(URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf'))
    geom=read(run/'geometry.json');curve=PadCurve(geom['compact_expert'])
    scene=read(attempt/'scene-frame.json');world=np.array(scene['world_from_reference'])
    base=np.array([s['actual_base'] for s in states]);cmd_base=np.array([s['command_base'] for s in states])
    q=np.array([s['arm_q'][:6] for s in states]);qc=np.array([s['arm_command'][:6] for s in states])
    ee=np.array([s['ee_actual'] for s in states]);door=np.array([s['door_q'][0] for s in states])
    rots=Rotation.from_quat(np.array([s['ee_quat'] for s in states])).as_matrix()
    pads=ee+np.einsum('nij,j->ni',rots,PAD_LOCAL)
    goals=[];axes=[];targets=[];diagnostic_target=[]
    hover=world@curve.tool(math.pi/2,geom['compact_expert']['hover_distance'])
    hover_pad=hover[:3,3]+hover[:3,:3]@PAD_LOCAL
    error=np.linalg.norm(pads-hover_pad,axis=1)
    axis_error=np.arccos(np.clip(rots[:,:,0]@hover[:3,0],-1,1))
    for s in states:
        d=s.get('compact_expert') or {};target=d.get('target_world')
        if target is not None:
            T=np.array(target);diagnostic_target.append((T[:3,3]+T[:3,:3]@PAD_LOCAL).tolist())
        else:diagnostic_target.append([float('nan')]*3)
    diagnostic_target=np.array(diagnostic_target)
    starts={x['phase']:x['first_step'] for x in intervals}
    deploy=np.array([p=='compact_rendezvous' for p in phases]);post=np.arange(n)>=(after_pass if after_pass is not None else n)
    rootz=np.array([s['root_rb']['pose']['p']['z'] for s in states])
    seek=starts.get('compact_seek');gate=None
    if seek is not None:
        i=seek;s=states[i];prev=cmd_base[i-1] if i else np.array(read(attempt/'initial-state.json')['base_start']+[math.pi])
        refbase=np.r_[(np.linalg.inv(world)@np.r_[prev[:2],0,1])[:2],prev[2]-math.atan2(world[1,0],world[0,0])]
        B=transform([*refbase[:2],scene['base_z_reference']],[0,scene['base_pitch'],refbase[2]])
        E=world@B@arm.fk(np.array(s['observation_pre']['state'][2:8]))
        P=E[:3,3]+E[:3,:3]@PAD_LOCAL
        goalbase=curve.at(math.pi/2)[:3]
        gate={'step':i,'command_time_s':i*dt,'logged_post_time_s':t[i],
            'reconstructed_pre_gate_pad_error_m':float(np.linalg.norm(P-hover_pad)),
            'reconstructed_pre_gate_axis_error_rad':float(np.arccos(np.clip(E[:3,0]@hover[:3,0],-1,1))),
            'reconstructed_pre_gate_base_distance_m':float(np.linalg.norm(refbase[:2]-goalbase[:2])),
            'previous_velocity_command_m_s':float(states[i-1]['action'][0]),
            'thresholds':{'pad_m':.008,'axis_rad':.01,'base_m':.008,'v_m_s':.01},
            'source_semantics':'Reconstruct exact prior command-base and observation_pre q used by frozen CompactExpert; post-physics measurements also retained',
            'previous_post_measured_pad_error_m':float(error[i-1]),'previous_post_measured_axis_error_rad':float(axis_error[i-1])}
    first_push=scoring['first_aligned_end_push_step']
    first_loaded=next((e for e in events if e['end_panel'] and e['lambda']!=0 and e['phase']=='close'),None)
    suffix_nonend=[e for e in events if e['phase']=='close' and not e['end_panel'] and e['lambda']!=0]
    summary={'source_head':read(run/'manifest.json')['start_commit'],'result':result,
        'r11_measured_gates_pass':scoring['measured_task_gates_pass'],'r11_failures':scoring['failures'],
        'steps':n,'duration_s':t[-1],'initial_frozen_matches':read(attempt/'initial-frozen-comparison.json')['matches_frozen'],
        'root_z_range_m':[float(rootz.min()),float(rootz.max())],
        'post_pass_root_z_range_m':[float(rootz[post].min()),float(rootz[post].max())] if post.any() else None,
        'max_base_tracking_xy_m':float(np.linalg.norm(base[:,:2]-cmd_base[:,:2],axis=1).max()),
        'post_pass_max_joint_command_step_rad':float(np.max(abs(np.diff(qc[post],axis=0)))) if sum(post)>1 else None,
        'deployment_max_joint_command_step_rad':float(np.max(abs(np.diff(qc[deploy],axis=0)))) if sum(deploy)>1 else None,
        'deployment_max_joint_command_tracking_rad':float(np.max(abs(q[deploy]-qc[deploy]))) if deploy.any() else None,
        'deployment_max_pad_reference_tracking_m':float(np.nanmax(np.linalg.norm(pads[deploy]-diagnostic_target[deploy],axis=1))) if deploy.any() else None,
        'deployment_duration_s':float(sum(deploy)*dt),'deployment_arrival_gate':gate,
        'first_r11_aligned_push_step':first_push,'first_close_loaded_end_contact':first_loaded,
        'close_loaded_nonend_records':len(suffix_nonend),'post_pass_contact_pairs':dict(contact_pairs),
        'post_pass_loaded_contact_pairs':dict(loaded_pairs),
        'close_min_angle_deg':float(np.rad2deg(door[[p=='close' for p in phases]]).min()) if 'close' in starts else None,
        'final_door_angle_deg':float(np.rad2deg(door[-1])),
        'final_joint_home_error_rad':(abs(q[-1]-np.array(ref['home'][:6]))).tolist(),
        'final_heading_error_rad':float((base[-1,2]-ref['old_expert_final_yaw']+math.pi)%(2*math.pi)-math.pi),
        'physical_acceptance':'Independent original-video/contact/code review required; automated gates are separated.'}
    write(out/'summary.json',summary)
    expected_z=read(attempt/'initial-frozen-comparison.json')['expected']['robot_z']
    assert np.max(abs(rootz-expected_z))<1e-7,'SYSTEMIC: base height changed within episode'
    # Event evidence always references complete original movie; no replay or extra simulation.
    event_index={'start':0,'whole_body_pass':after_pass,'deployment_start':starts.get('compact_rendezvous'),
        'seek_start':seek,'close_start':starts.get('close'),'first_aligned_push':first_push,
        'first_loaded_end_close':first_loaded['step'] if first_loaded else None,
        'latch':next((x['step'] for x in locks if x['kind']=='relocked'),None),
        'release':starts.get('release'),'retract':starts.get('retract'),'final_hold':starts.get('final_hold'),'stop':n-1}
    index={k:None if v is None else {'step':v,'measured_time_s':t[v],'original_rgb_frame':v//2,
           'raw_state_file':'attempt_000/states.jsonl','raw_contact_file':'attempt_000/contacts.jsonl',
           'original_movie':'attempt_000/overview/env_0000/rollout.mp4'} for k,v in event_index.items()}
    write(out/'event-index.json',index)
    wanted={0,(n-1)//2};wanted.update(v//2 for v in event_index.values() if v is not None)
    if deploy.any():wanted.update(int(i)//2 for i in np.linspace(np.where(deploy)[0][0],np.where(deploy)[0][-1],5))
    if 'close' in starts:wanted.update(int(i)//2 for i in np.linspace(starts['close'],n-1,9))
    images=[]
    with imageio.get_reader(str(attempt/'overview/env_0000/rollout.mp4')) as f:
        for i,frame in enumerate(f):
            if i in wanted:
                im=Image.fromarray(frame);im.save(out/('original-frame-%05d.jpg'%i),quality=94)
                tile=im.copy();tile.thumbnail((768,256));sheet=Image.new('RGB',(768,286),'white');sheet.paste(tile,(0,30))
                ImageDraw.Draw(sheet).text((8,8),'ORIGINAL RGB frame %d / source step %d / t=%.2fs / %s'%(i,2*i,t[min(2*i,n-1)],phases[min(2*i,n-1)]),fill='black');images.append(sheet)
    sheet=Image.new('RGB',(1536,286*math.ceil(len(images)/2)),(230,230,230))
    for i,im in enumerate(images):sheet.paste(im,((i%2)*768,(i//2)*286))
    sheet.save(out/'original-evidence-sheet.jpg',quality=92)
    fig,axs=plt.subplots(3,2,figsize=(15,13),constrained_layout=True)
    ax=axs[0,0];ax.plot(base[:,0],base[:,1],label='actual base');ax.plot(pads[:,0],pads[:,1],label='actual contact pad',alpha=.8)
    if (attempt/'compact-route.npy').exists():route=np.load(attempt/'compact-route.npy');ax.plot(route[:,0],route[:,1],'k--',label='fixed bypass reference')
    hinge=(world@np.array([2.467,-.463,.01,1]))[:2]
    for i in sorted({0,starts.get('close',0),n-1}):
        end=hinge+np.array([-math.sin(door[i]),math.cos(door[i])])*.93;ax.plot([hinge[0],end[0]],[hinge[1],end[1]],label='door t=%.1f'%t[i])
    ax.set_aspect('equal');ax.set(xlabel='world x (m)',ylabel='world y (m)',title='Original live route / pad / door');ax.legend(fontsize=8)
    ax=axs[0,1];ax.plot(t,np.rad2deg(door));ax.set(xlabel='measured time (s)',ylabel='door angle (deg)',title='Measured door motion')
    ax=axs[1,0];ax.plot(t,rootz,label='actual root z');ax.plot(t,pads[:,2],label='actual pad z');ax.set(xlabel='time (s)',ylabel='height (m)');ax.legend()
    ax=axs[1,1];ax.plot(t,np.linalg.norm(pads-diagnostic_target,axis=1)*1000,label='actual pad vs current reference');ax.set(xlabel='time (s)',ylabel='error (mm)',title='Pad tracking, reference changes by phase');ax.legend()
    ax=axs[2,0]
    for j in range(6):ax.plot(t,q[:,j],label='q%d actual'%(j+1));ax.plot(t,qc[:,j],'--',alpha=.6)
    ax.set(xlabel='time (s)',ylabel='rad',title='Actual solid / commanded dashed joints');ax.legend(ncol=3,fontsize=8)
    ax=axs[2,1];ax.plot(t,np.array([s['action'][0] for s in states]),label='vx command');ax.plot(t,np.array([s['base_measured_fd_speed'] for s in states]),label='measured speed');ax.set(xlabel='time (s)',ylabel='m/s');ax.legend()
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.savefig(out/'actual-motion-comparison.png',dpi=160);plt.close(fig)
    if deploy.any():
        ix=np.where(deploy)[0];sl=slice(max(0,ix[0]-10),min(n,ix[-1]+30));fig,axs=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
        axs[0].plot(t[sl],error[sl]*1000,label='actual pad vs fixed hover');axs[0].axhline(8,color='k',ls='--',label='8 mm gate');axs[0].set(ylabel='mm');axs[0].legend()
        axs[1].plot(t[sl],axis_error[sl],label='actual push axis vs fixed hover');axs[1].axhline(.01,color='k',ls='--');axs[1].set(ylabel='rad');axs[1].legend()
        for j in range(6):axs[2].plot(t[sl],q[sl,j],label='q%d'%(j+1));axs[2].plot(t[sl],qc[sl,j],'--',alpha=.6)
        axs[2].set(xlabel='measured time (s)',ylabel='rad');axs[2].legend(ncol=6)
        for ax in axs:
            if seek is not None:ax.axvline(t[seek],color='r',ls=':')
            ax.grid(alpha=.2)
        fig.savefig(out/'deployment-detail.png',dpi=160);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['r11_measured_gates_pass','r11_failures','steps','duration_s','deployment_arrival_gate','close_loaded_nonend_records','final_door_angle_deg']},indent=2))

if __name__=='__main__':main()
