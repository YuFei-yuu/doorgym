import json,sys,math
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);rows=[json.loads(l) for l in (p/'review/compact-actual-trace.jsonl').open()];eligible=[]
for r in rows:
    d=r.get('compact_expert') or {}
    if d.get('phase')=='approach' and r['step']>=3321:
        eligible.append(dict(step=r['step'],t=r['t'],door_deg=math.degrees(r['door_q'][0]),position_m=d.get('measured_position_error'),orientation_rad=d.get('measured_orientation_error'),speed_rad_s=d.get('measured_joint_speed_max'),joint_margin=d.get('joint_margin')))
r=dict(eligible_samples=len(eligible),last=eligible[-1] if eligible else None,thresholds=dict(position_m=.008,orientation_rad=.01,speed_rad_s=.05),satisfied_counts={})
for key,lim in r['thresholds'].items():r['satisfied_counts'][key]=sum(e[key]<=lim for e in eligible)
r['all_satisfied_count']=sum(all(e[k]<=v for k,v in r['thresholds'].items()) for e in eligible)
r['eligible_extrema']={k:dict(min=min(e[k] for e in eligible),max=max(e[k] for e in eligible)) for k in r['thresholds']} if eligible else {}
endpoint=next(x for x in rows if x['phase']=='endpoint_hold');post=[x for x in rows if x['t']>=endpoint['t']];B=np.array([x['actual_base'] for x in post]);r['post_endpoint_translation_m']=float(np.linalg.norm(np.diff(B[:,:2],axis=0),axis=1).sum());r['endpoint_t']=endpoint['t'];r['endpoint']=endpoint['actual_base'];r['never_closed_phase']=all(x['phase']!='close' for x in rows)
(p/'review/approach-gate-diagnosis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
for a,(key,lim) in zip(ax,r['thresholds'].items()):
    a.plot([e['t'] for e in eligible],[e[key] for e in eligible]);a.axhline(lim,color='red',ls='--',label='unchanged gate');a.set_ylabel(key);a.grid(alpha=.3);a.legend()
ax[-1].set_xlabel('Actual simulation time (s)');fig.suptitle('Approach gate: actual feedback; no threshold relaxation');fig.tight_layout();fig.savefig(p/'review/approach-gate-curves.png',dpi=160)
