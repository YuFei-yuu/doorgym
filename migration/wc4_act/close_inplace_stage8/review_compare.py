"""Predeclared before/after metrics from ACTUAL states; never replay physics."""
import sys,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
run=Path(sys.argv[1]);out=run/'review';root=Path('/data/zyf/video2door-wc4-act');baseline=root/'.repro/runs/wc4-close-inplace-stage7-20261027'
KEYS=['step','t','phase','actual_base','action','arm_q','arm_qd','ee_actual','ee_target','ee_quat','early_arm','nonzero_door_contact','nonzero_end_panel_contact','compact_expert','door_q']
def read(p):
    rows=[]
    for l in (p/'attempt_000/states.jsonl').open():
        r=json.loads(l);rows.append({k:r[k] for k in KEYS if k in r})
    return rows
def analyze(rows):
    T=np.array([r['t'] for r in rows]);Q=np.array([r['arm_q'] for r in rows]);QD=np.array([r['arm_qd'] for r in rows]);B=np.array([r['actual_base'] for r in rows]);EE=np.array([r['ee_actual'] for r in rows]);REF=[]
    for r in rows:
        d=r.get('compact_expert') or {}
        REF.append(np.array(d['target_pad'])-np.array(d['target_rotation'])@np.array([.015,.03,-.02]) if 'target_pad' in d else np.array(r['ee_target']))
    REF=np.array(REF);V=np.r_[np.nan,np.linalg.norm(np.diff(EE,axis=0),axis=1)/np.diff(T)];VR=np.r_[np.nan,np.linalg.norm(np.diff(REF,axis=0),axis=1)/np.diff(T)]
    first=next((r for r in rows if r['phase'] in ['deploy_a','deploy_b','approach','close'] and r['nonzero_end_panel_contact']),None)
    time=lambda predicate:next((r['t'] for r in rows if predicate(r)),None)
    times=dict(traverse=time(lambda r:r['phase']=='traverse_door'),withdraw=time(lambda r:r['phase']=='withdraw' or r.get('early_arm',{}).get('stage')=='parallel_withdraw'),fold=time(lambda r:r['phase']=='fold' or r.get('early_arm',{}).get('stage')=='parallel_fold'),parking=time(lambda r:r['phase']=='endpoint_hold'),turn=time(lambda r:r['phase']=='turn'),first_closing_contact=first['t'] if first else None)
    settled_start=None;actual_fold=None
    for r in rows:
        if times['fold'] is None or r['t']<times['fold']:continue
        good=max(abs(np.array(r['arm_q'])[:6]))<=.03 and max(abs(np.array(r['arm_qd'])[:6]))<=.05 and not r['nonzero_door_contact']
        if good:
            if settled_start is None:settled_start=r['t']
            if r['t']-settled_start>=.2:actual_fold=dict(first_qualified_s=settled_start,confirmed_after_0p2s=r['t']);break
        else:settled_start=None
    windows={}
    if first:
        for lo,hi in [(-.5,0),(0,1),(1,2),(-.5,1)]:
            mask=(T-first['t']>=lo)&(T-first['t']<hi);key='%g_to_%g_s'%(lo,hi)
            windows[key]=dict(samples=int(mask.sum()),actual_mean_mps=float(np.mean(V[mask])),actual_peak_mps=float(np.max(V[mask])),reference_mean_mps=float(np.mean(VR[mask])),reference_peak_mps=float(np.max(VR[mask])))
    close=[r for r in rows if (r.get('compact_expert') or {}).get('phase')=='close'];ref_samples=[]
    for r in close:
        ref=(r.get('compact_expert') or {}).get('reference',{});ref_samples.append((r['t'],ref.get('turn_progress',0),ref.get('follow_progress',0)))
    def range_time(col):
        begin=next((x[0] for x in ref_samples if x[col]>0),None);end=next((x[0] for x in ref_samples if x[col]>=1),None);return dict(start_s=begin,end_s=end,duration_s=None if begin is None or end is None else end-begin)
    motion={}
    base_v=np.r_[np.nan,np.linalg.norm(np.diff(B[:,:2],axis=0),axis=1)/np.diff(T)]
    yaw_v=np.r_[np.nan,abs(np.diff(np.unwrap(B[:,2])))/np.diff(T)]
    for name,col in [('turn',1),('follow',2)]:
        timing=range_time(col)
        if timing['start_s'] is not None and timing['end_s'] is not None:
            mask=(T>=timing['start_s'])&(T<=timing['end_s'])
            motion[name]=dict(timing=timing,actual_ee_mean_mps=float(np.mean(V[mask])),reference_ee_mean_mps=float(np.mean(VR[mask])),actual_base_mean_mps=float(np.mean(base_v[mask])),actual_base_peak_mps=float(np.max(base_v[mask])),actual_yaw_mean_rps=float(np.mean(yaw_v[mask])),actual_yaw_peak_rps=float(np.max(yaw_v[mask])))
    clip=dict(base_rate=sum(bool(r['compact_expert'].get('base_rate_clipped')) for r in close),base_acceleration=sum(bool(r['compact_expert'].get('base_acceleration_clipped')) for r in close),joint_rate=sum(bool(r['compact_expert'].get('joint_rate_clipped')) for r in close),early_arm=sum(bool(r.get('early_arm',{}).get('rate_clipped')) for r in rows))
    for key,field in [('base_rate','base_rate_clipped'),('base_acceleration','base_acceleration_clipped'),('joint_rate','joint_rate_clipped')]:
        if not any(field in r['compact_expert'] for r in close):clip[key]=None
    if not any('early_arm' in r for r in rows):clip['early_arm']=None
    return dict(actual_later_motion=motion,times=times,actual_fold=actual_fold,contact_speed_windows=windows,turn_reference=range_time(1),follow_reference=range_time(2),clipped_steps=clip,total_s=float(T[-1])),(T,Q,QD,B,EE,REF,V,VR)
old=read(baseline);new=read(run);A,aa=analyze(old);B,bb=analyze(new)
ratio={k:{name:(B['contact_speed_windows'][k][name]/v-1)*100 if v else None for name,v in A['contact_speed_windows'][k].items() if name.endswith('mps')} for k in B['contact_speed_windows'] if k in A['contact_speed_windows']}
limit=min(len(new),1240);base_diff=float(np.max(abs(np.array([r['actual_base'] for r in new[:limit]])-np.array([r['actual_base'] for r in old[:limit]]))))
opening_diff={k:max(float(np.max(abs(np.array(x[k])-np.array(y[k])))) for x,y in zip(new[:875],old[:875])) for k in ['actual_base','action','arm_q','door_q']}
later_percent={k:(B['actual_later_motion'][k]['actual_ee_mean_mps']/A['actual_later_motion'][k]['actual_ee_mean_mps']-1)*100 for k in ['turn','follow'] if k in B['actual_later_motion'] and k in A['actual_later_motion']}
report=dict(actual_later_ee_mean_percent_change=later_percent,baseline=str(baseline),current=str(run),baseline_metrics=A,current_metrics=B,percent_change=ratio,base_through_parking_max_difference=base_diff,opening_through_step874_max_difference=opening_diff,
    method='50Hz backward finite difference of logged actual ee_gripper_link world position. Reference EE = pre-physics desired pad minus desired rotation*pad; plotted at interval end. Windows fixed before rollout, aligned on each actual first loaded closing contact. No smoothing or phase-only contact proxy.')
(out/'before-after-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
fig,ax=plt.subplots(3,1,figsize=(12,10),sharex=True)
for name,M,data in [('stage7',A,aa),('stage8',B,bb)]:
    T,Q,QD,base,*_=data;ax[0].plot(T,base[:,0],label=name);ax[1].plot(T,np.max(abs(Q[:,:6]),axis=1),label=name);ax[2].plot(T,np.max(abs(QD[:,:6]),axis=1),label=name)
    for key,style in [('withdraw',':'),('fold','--'),('parking','-.'),('turn',':')]:
        if M['times'][key] is not None:ax[1].axvline(M['times'][key],ls=style,alpha=.35)
ax[0].set_ylabel('Actual base x (m)');ax[1].set_ylabel('Actual max |q-home| (rad)');ax[1].axhline(.03,color='gray');ax[2].set_ylabel('Actual max |qd| (rad/s)');ax[2].axhline(.05,color='gray');ax[-1].set_xlabel('Simulation time (s)');ax[-1].set_xlim(16,36)
for a in ax:a.grid(alpha=.3);a.legend()
fig.tight_layout();fig.savefig(out/'parallel-fold-comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(4,1,figsize=(12,12),sharex=True)
for name,M,data in [('stage7',A,aa),('stage8',B,bb)]:
    T,Q,QD,base,EE,ref,V,VR=data;anchor=M['times']['first_closing_contact']
    if anchor is None:continue
    mask=(T-anchor>=-1)&(T-anchor<=30)
    ax[0].plot((T-anchor)[mask],VR[mask],label=name+' reference');ax[1].plot((T-anchor)[mask],V[mask],label=name+' actual');ax[2].plot((T-anchor)[mask],np.unwrap(base[:,2])[mask],label=name);ax[3].plot((T-anchor)[mask],base[mask,0],label=name)
ax[0].set_ylabel('Reference EE speed (m/s)');ax[1].set_ylabel('Actual EE speed (m/s)');ax[2].set_ylabel('Actual base yaw (rad)');ax[3].set_ylabel('Actual base x (m)');ax[3].set_xlabel('Time from each actual first closing contact (s)');ax[3].set_xlim(-1,30)
for a in ax:a.axvline(0,color='gray',ls=':');a.grid(alpha=.3);a.legend()
fig.tight_layout();fig.savefig(out/'contact-speed-base-comparison.png',dpi=160);ax[3].set_xlim(-.75,3)
for j,column in [(0,7),(1,6)]:
    peaks=[]
    for M,data in [(A,aa),(B,bb)]:
        if M['times']['first_closing_contact'] is None:continue
        rel=data[0]-M['times']['first_closing_contact'];peaks.append(float(np.nanmax(data[column][(rel>=-.75)&(rel<=3)])))
    if peaks:ax[j].set_ylim(0,max(peaks)*1.1)
fig.savefig(out/'first-contact-speed-detail.png',dpi=160)
print(json.dumps(report),flush=True)
