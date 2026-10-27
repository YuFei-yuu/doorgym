"""Read recorded demand/clip telemetry; missing values remain unknown."""
import json,sys
from pathlib import Path
from collections import defaultdict
p=Path(sys.argv[1]);stats=defaultdict(lambda:dict(samples=0,max_requested_joint_rate=0.,joint_clipped=0,base_rate_clipped=0,base_acceleration_clipped=0))
for line in (p/'review/compact-actual-trace.jsonl').open():
    r=json.loads(line);d=r.get('compact_expert') or {}
    if 'requested_joint_rate_max' in d:
        s=stats[d['phase']];s['samples']+=1;s['max_requested_joint_rate']=max(s['max_requested_joint_rate'],d['requested_joint_rate_max']);s['joint_clipped']+=bool(d.get('joint_rate_clipped'));s['base_rate_clipped']+=bool(d.get('base_rate_clipped'));s['base_acceleration_clipped']+=bool(d.get('base_acceleration_clipped'))
result=dict(by_diagnostic_phase=dict(stats),limits=dict(joint_rad_s=.7,base_m_s=.04,base_rad_s=.2,base_acceleration_m_s2=.1,base_acceleration_rad_s2=.3),meaning='Recorded pre-rate-limit command demand; actual joint velocities/positions remain in raw state. Baseline missing flags are not assumed zero.')
(p/'review/limit-demand-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
