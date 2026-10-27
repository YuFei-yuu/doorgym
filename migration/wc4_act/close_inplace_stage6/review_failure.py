import sys,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);out=p/'review';rows=[json.loads(l) for l in (out/'compact-actual-trace.jsonl').open()];close=[r for r in rows if r['phase']=='close'];contacts=json.loads((out/'contact-review.json').read_text());metrics=json.loads((out/'metrics.json').read_text())
def small(r):
    d=r.get('compact_expert') or {}
    return dict(step=r['step'],post_time_s=r['t'],door_deg=math.degrees(r['door_q'][0]),door_speed_rad_s=r['door_qd'][0],base=r['actual_base'],action_vw=r['action'][:2],loaded_contact=r['nonzero_end_panel_contact'],position_error_m=d.get('measured_position_error'),orientation_error_rad=d.get('measured_orientation_error'),ik=d.get('ik'),joint_margin=d.get('joint_margin'))
threshold={str(deg):next((small(r) for r in close if math.degrees(r['door_q'][0])<=deg),None) for deg in [70,35,10,1]}
bad=[r for r in close if (r.get('compact_expert') or {}).get('measured_position_error',0)>.06 or (r.get('compact_expert') or {}).get('measured_orientation_error',0)>.15]
report=dict(angle_crossings=threshold,first_tracking_violation=small(bad[0]) if bad else None,last=small(rows[-1]),last_loaded_contact=contacts['last_postdeploy_loaded_panel_contact'],
    maximum_actual_closing_speed_rad_s=max(-r['door_qd'][0] for r in close),base_yaw_rate_limit_rad_s=.2,follow_heading_tolerance_rad=.003,
    interpretation='Measured door advances faster than yaw recovery; at <=35deg the heading gate is not met, so translation stays zero. Contact ends at last_loaded_contact; later door motion is not evidence of continuing end-effector contact. No parameter changes or retry.')
(out/'failure-localization.json').write_text(json.dumps(report,indent=2)+'\n')
T=[r['t'] for r in close];fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
ax[0].plot(T,[math.degrees(r['door_q'][0]) for r in close],label='actual door deg');ax[0].plot(T,[math.degrees(r['actual_base'][2]) for r in close],label='actual base yaw deg');ax[0].legend()
ax[1].plot(T,[(r.get('compact_expert') or {}).get('measured_position_error',float('nan')) for r in close]);ax[1].axhline(.06,color='red',ls='--');ax[1].set_ylabel('Tracking error (m)')
ax[2].plot(T,[(r.get('compact_expert') or {}).get('measured_orientation_error',float('nan')) for r in close]);ax[2].axhline(.15,color='red',ls='--');ax[2].set_ylabel('Orientation error (rad)');ax[2].set_xlabel('Actual post-physics time (s); error is preceding command evaluation')
for a in ax:a.grid(alpha=.3);a.axvline(contacts['last_postdeploy_loaded_panel_contact']['time_s'],color='gray',ls=':')
fig.tight_layout();fig.savefig(out/'tracking-failure.png',dpi=160);print(json.dumps(report),flush=True)
