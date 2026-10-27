"""Targeted recorded-feedback handoff check, not physics or a full candidate."""
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from close_inplace_stage6.controller import LiveExpert
from close_inplace_stage6.dynamic_gate import DynamicContactGate
from close_inplace_stage4.common import FullPose
from fullcycle_geometry import URDF
p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=False)
source=ROOT/'.repro/runs/wc4-close-inplace-stage5-20261027-retry2/attempt_000/states.jsonl'
rows=[]
for line in source.open():
    r=json.loads(line)
    if 2918<=r['step']<=3720:rows.append(r)
robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf',include_visual=True)
c=LiveExpert.__new__(LiveExpert);c.S=FullPose(robot);c.cfg={'radius':.905,'height':.85};c.dt=.019999999552965164
first=next(r for r in rows if r['step']==2920);prior=next(r for r in rows if r['step']==2919)
c.cmd=np.array(first['action'][2:]);c.command_velocity=(c.cmd-np.array(prior['action'][2:]))/c.dt;c.prevq=np.array(first['pre_physics']['arm_dof']) if False else np.array(prior['arm_q'])
c.home=np.r_[np.zeros(6),-1.1775000095367432];c.finalyaw=math.pi;c.endpoint=np.array([1.2599997520446777,.18000000715255737]);c.previous_twist=np.zeros(2);c.route=None;c.follow0=None;c.handoff=None;c.dynamic_gate=DynamicContactGate()
c.enter('approach',2920*c.dt,np.array(prior['arm_q']),np.array(first['actual_base']))
# Reproduce the exact measured pre-physics state from the transition's predecessor.
# First post-transition command uses the next observed post-physics state.
trace=[];transition=None;previous=first
for r in rows:
    if r['step']<=2920:continue
    q=np.array(previous['arm_q']);B=np.array(previous['actual_base']);t=r['step']*c.dt
    action,d=c.command(t,B,q,float(previous['door_q'][0]),float(previous['door_qd'][0]),previous['nonzero_end_panel_contact'],previous['nonzero_door_contact'],False)
    trace.append(dict(step=r['step'],t=t,phase=c.phase,action=action.tolist(),diagnostic=d,feedback='frozen actual measurements; counterfactual commands, NOT a rollout'))
    if d.get('dynamic_handoff') and transition is None:transition=len(trace)-1
    if transition is not None and t-trace[transition]['t']>=2:break
    previous=r
assert transition is not None,'Recorded dynamic contact was not accepted'
# Adversarial gate inputs are variations of observations, not route/parameter searches.
cases={k:DynamicContactGate() for k in ['normal','no_contact','single_frame_contact','position_mismatch','orientation_mismatch','door_not_moving','door_opening']};accepted={k:[] for k in cases}
for r in rows:
    d=r.get('compact_expert') or {}
    if d.get('phase')!='approach' or 'measured_position_error' not in d:continue
    for name,g in cases.items():
        loaded=r['nonzero_end_panel_contact'];pe=d['measured_position_error'];oe=d['measured_orientation_error'];omega=float(r['door_qd'][0])
        if name=='no_contact':loaded=False
        if name=='single_frame_contact':loaded=r['step']==3350
        if name=='position_mismatch':pe=.05
        if name=='orientation_mismatch':oe=.2
        if name=='door_not_moving':omega=0
        if name=='door_opening':omega=abs(omega)
        if g.observe(r['t'],float(r['door_q'][0]),omega,loaded,pe,oe,ready=r['step']>=3321):accepted[name].append(r['step'])
assert accepted['normal'] and all(not v for k,v in accepted.items() if k!='normal')
A=np.array([r['action'] for r in trace]);rate=np.diff(A[:,2:],axis=0)/c.dt
before=rate[transition-1];after=rate[transition]
report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Only recorded approach and first 2s after counterfactual handoff; no physical rollout or complete route',
    transition=trace[transition],first_close_command=trace[transition+1],maximum_joint_command_rate=float(np.max(abs(rate))),boundary_command_change_rad=float(np.max(abs(A[transition+1,2:]-A[transition,2:]))),
    boundary_joint_velocity_change=float(np.max(abs(after-before))),gate_cases={k:dict(accepted_count=len(v),first_step=v[0] if v else None) for k,v in accepted.items()},passed=True)
assert report['maximum_joint_command_rate']<=.700001
(p/'targeted-check.json').write_text(json.dumps(report,indent=2)+'\n');(p/'counterfactual-handoff.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trace));print(json.dumps({k:v for k,v in report.items() if k not in ['transition','first_close_command']}),flush=True)
