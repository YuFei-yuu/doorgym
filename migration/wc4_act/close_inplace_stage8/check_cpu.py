"""One necessary nominal time-reference check. Synthetic tracking, no physics."""
import sys,json,math,hashlib,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from close_inplace_stage8.controller import LiveExpert
from close_inplace_stage8.reference import reference,TOTAL
from close_inplace_stage4.common import FullPose
from fullcycle_geometry import URDF
from close_inplace_stage6.dynamic_gate import DynamicContactGate
p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=False)
from close_inplace_stage8.early_arm import EarlyArm
source=ROOT/'.repro/runs/wc4-close-inplace-stage7-20261027/attempt_000/states.jsonl'
with source.open() as f:
    row=next(json.loads(l) for l in f if (json.loads(l).get('compact_expert') or {}).get('dynamic_handoff'))
h=row['compact_expert']['dynamic_handoff'];dt=.019999999552965164
c=LiveExpert.__new__(LiveExpert);c.S=FullPose(URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf'));c.cfg={'radius':.905,'height':.85};c.dt=dt
c.cmd=np.array(h['previous_joint_command']);c.command_velocity=np.array(h['previous_joint_velocity']);c.prevq=np.array(h['measured_q']);c.home=np.r_[np.zeros(6),-1.1775000095367432];c.finalyaw=math.pi;c.endpoint=np.array([1.2599997520446777,.18000000715255737]);c.previous_twist=np.zeros(2);c.route=None;c.follow0=None;c.handoff=h;c.dynamic_gate=DynamicContactGate()
B=np.array(row['actual_base']);q=np.array(h['measured_q']);t0=h['time_s'];c.enter('close',t0,q,B)
trace=[];independence=[];clipping=[];oldaction=np.r_[0,0,c.cmd];maxrate=0.;maxik=0.;wmax=vmax=0.;accmax=np.zeros(2)
for i in range(1,math.ceil((TOTAL+.52)/dt)+1):
    t=t0+i*dt
    if i in [1,100,650,1000,1300]:
        probe=copy.deepcopy(c);a,d=probe.command(t,B,q,0.,0.,False,False,False)
        probe=copy.deepcopy(c);b,e=probe.command(t,B,q,math.pi/2,-.3,True,True,False)
        assert np.array_equal(a,b);assert d['reference']==e['reference'];independence.append(i)
    action,d=c.command(t,B,q,h['theta'],0.,False,False,False)
    if d['base_rate_clipped'] or d['base_acceleration_clipped'] or d['joint_rate_clipped']:clipping.append(i)
    maxrate=max(maxrate,float(np.max(abs(action[2:]-oldaction[2:]))/dt));maxik=max(maxik,d['ik']['position_error_m']);vmax=max(vmax,abs(action[0]));wmax=max(wmax,abs(action[1]));accmax=np.maximum(accmax,abs(action[:2]-oldaction[:2])/dt)
    if i==1:boundary=float(np.max(abs((action[2:]-oldaction[2:])/dt-np.array(h['previous_joint_velocity']))))
    B[:2]+=action[0]*dt*np.array([math.cos(B[2]),math.sin(B[2])]);B[2]+=action[1]*dt;q=action[2:].copy();oldaction=action.copy()
    trace.append(dict(i=i,t=t,base=B.tolist(),q=q.tolist(),action=action.tolist(),diagnostic=d,kind='ideal synthetic actuator tracking for one reference limit check; NOT physics'))
assert maxrate<=.700001 and vmax<=.040001 and wmax<=.200001
assert np.max(abs(B[:2]-(c.endpoint+[.45,0])))<1e-5 and abs((B[2]+math.pi)%(2*math.pi)-math.pi)<1e-5
assert trace[-1]['diagnostic']['reference_complete'] and abs(trace[-1]['action'][0])<.001
report=dict(passed=True,base_clipped_steps=clipping,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='One complete timed pushing reference only; no rollout, suffix or alternative search',durations_s=[3.3,9.4,12.4],total_s=TOTAL,
    maximum_command_joint_rate=maxrate,maximum_ik_position_residual=maxik,maximum_base_speed=vmax,maximum_base_yaw_rate=wmax,maximum_base_acceleration=accmax.tolist(),boundary_joint_velocity_change=boundary,
    endpoint_base=B.tolist(),door_angle_and_contact_independence_steps=independence,
    control_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__).resolve().with_name('controller.py'),Path(__file__).resolve().with_name('reference.py'),Path(__file__).resolve().with_name('early_arm.py')]})
assert not clipping
# Single measured opening endpoint -> withdrawal/fold continuity check.
with source.open() as f:prefix=[r for r in map(json.loads,f) if r['step'] in [873,874]]
a0,a1=prefix
E=EarlyArm(c.S.robot if hasattr(c.S,'robot') else URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf'),np.array(a1['actual_base']),np.array(a1['arm_q']),np.array(a1['action'][2:]),np.array(a0['action'][2:]),875*dt,dt,c.home)
Q=np.array(a1['arm_q']);et=[];emax=0.;eclip=[];bounds=[]
for i in range(math.ceil((E.tw+E.tf+.1)/dt)):
    target=E.command(E.t0+i*dt,Q,False);emax=max(emax,E.latest['requested_rate_max'])
    if E.latest['rate_clipped']:eclip.append(i)
    if np.min(np.r_[target[:6]-E.S.lo,E.S.hi-target[:6]])<-1e-6:bounds.append(i)
    et.append(dict(i=i,t=E.t0+i*dt,q=target.tolist(),diagnostic=E.latest.copy(),kind='ideal synthetic joint tracking only'));Q=target
assert not eclip and not bounds and E.plan['ik']['position_error_m']<=.008
report['early_arm']=dict(plan=E.plan,max_requested_rate=emax,clipped_steps=eclip,outside_scaled_limits=bounds,withdraw_begin_s=E.t0,fold_begin_s=E.t0+E.tw,home_reference_s=E.t0+E.tw+E.tf,baseline_parking_s=24.799999445676804)
(p/'early-arm-check.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in et))
(p/'check.json').write_text(json.dumps(report,indent=2)+'\n');(p/'reference-check.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace));print(json.dumps(report),flush=True)
