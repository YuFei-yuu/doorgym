"""CPU-only design checks. Synthetic door feedback is NOT physics evidence."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from fullcycle_compact import DEFAULT, Arm, ClosingCurve, CompactExpert, rounded_route
from fullcycle_geometry import URDF, transform, apply

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-run",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--reference-repo",required=True)
    p.add_argument("--geometry",default=None)
    p.add_argument("--door-scenario",choices=["nominal","drift45"],default="nominal")
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    recipe=dict(DEFAULT)
    if a.geometry:recipe.update(json.loads(Path(a.geometry).read_text())["compact_expert"])
    robot=URDF(ROOT/"high-level/data/asset/a2wz1/urdf/a2wz1.urdf",include_visual=True)
    arm=Arm(robot);curve=ClosingCurve(recipe)
    source=Path(a.source_run)
    boundary=None
    for line in (source/"attempt_000/states.jsonl").open():
        r=json.loads(line)
        if r["passed"]:boundary=r;break
    (out/"boundary.json").write_text(json.dumps(boundary,indent=2)+"\n")
    names=arm.names+["jointGripper"]
    legs={side+suffix:val for side in ["FL","FR","RL","RR"] for suffix,val in
          [("_hip_joint",0),("_thigh_joint",.8),("_calf_joint",-1.5)]}
    # Full-pose IK throughout the close curve, including source collision and
    # visual bounds. Do not infer PhysX contact forces from these bounds.
    sweep=[];seed=np.array([0,1.7,-1.3,-.4,0,math.pi/2])
    for theta in np.linspace(math.radians(95),0,191):
        ref=curve.at(theta);B=transform([*ref[:2],.5],[0,0,ref[2]])
        seed,err=arm.solve(np.linalg.inv(B)@curve.tool(theta),seed,1.,120)
        poses=robot.fk(dict(legs,**dict(zip(names,np.r_[seed,0.]))),B)
        cloud=robot.cloud(poses)
        sweep.append(dict(theta=theta,base=ref[:3].tolist(),q=seed.tolist(),**err,
                          whole_max_x=float(cloud[:,0].max()),
                          forward_per_progress=ref[6],leftward_per_progress=ref[7]))
    (out/"closing-sweep.json").write_text(json.dumps(sweep,indent=2)+"\n")
    # Exercise the complete controller with exact joint tracking and explicitly
    # synthetic door-angle/contact input. This checks code paths, budget,
    # unicycle commands and handoffs, not dynamic success or actual contact.
    base=np.array(boundary["actual_base"]);q=np.array(boundary["arm_q"])
    theta=boundary["door_q"][0];home=np.r_[np.zeros(6),-1.1775];dt=.02
    ctl=CompactExpert(robot,recipe,base,q,boundary["action"],theta,0.,home,math.pi,dt)
    traces=[];phase_events=[];prev_phase=None;hold=0.;latched=False;end_time=None
    for k in range(8000):
        t=k*dt
        contact=False
        # No contact oracle is used during withdrawal. It does not certify
        # that actual contact will clear; the live controller still waits.
        old_theta=theta
        if a.door_scenario=="drift45" and ctl.phase not in ("close","release","retract",
                "compact_exit_turn","compact_exit_arc","restore_heading","final_hold"):
            theta=math.radians(90-45*np.clip((t-6)/19,0,1))
        if ctl.phase=="compact_seek":
            contact=ctl.seek_offset<=0 and abs(theta-math.pi/2)<.01
        if ctl.phase=="close" and ctl.close_parameter<=theta+.001:
            contact=True;theta=min(theta,max(0.,ctl.close_parameter))
        td=(theta-old_theta)/dt
        if ctl.phase=="close" and theta==0:
            hold+=dt
            if hold>=.54:latched=True
        action,diag=ctl.command(t,base,q,theta,td,contact,contact,latched)
        # Identical planar integration convention to the existing Joint9 base.
        base[:2]+=action[0]*dt*np.array([math.cos(base[2]),math.sin(base[2])])
        base[2]+=action[1]*dt;q=action[2:].copy()
        B=transform([*base[:2],.5],[0,0,base[2]])
        poses=robot.fk(dict(legs,**dict(zip(names,q))),B)
        cloud=robot.cloud(poses)
        if ctl.phase!=prev_phase:
            phase_events.append(dict(t=t,phase=ctl.phase,base=base.tolist(),theta=theta,
                                     ik=diag["ik"]))
            print(json.dumps(phase_events[-1]),flush=True);prev_phase=ctl.phase
        traces.append(dict(t=t,phase=ctl.phase,base=base.tolist(),q=q.tolist(),
            action=action.tolist(),theta=theta,whole_max_x=float(cloud[:,0].max()),
            bounds=[cloud.min(0).tolist(),cloud.max(0).tolist()],ik=diag["ik"]))
        if ctl.phase=="final_hold":
            if end_time is None:end_time=t
            if t-end_time>=2.02:break
    (out/"synthetic-command-trace.json").write_text(json.dumps(traces)+"\n")
    (out/"phase-events.json").write_text(json.dumps(phase_events,indent=2)+"\n")
    def fn(src,name):
        node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.dump(node,include_attributes=False)
    old=subprocess.check_output(["git","-C",a.reference_repo,"show",
         "e3cd684:migration/wc4_act/fullcycle_expert.py"],text=True)
    new=Path(__file__).with_name("fullcycle_expert.py").read_text()
    checks=dict(
        opening_unchanged=fn(old,"opening_command")==fn(new,"opening_command"),
        close_full_pose_position=max(s["position_error_m"] for s in sweep)<.005,
        close_full_pose_orientation=max(s["orientation_error_rad"] for s in sweep)<.035,
        close_whole_body_far=max(s["whole_max_x"] for s in sweep)<2.5,
        close_forward=min(s["forward_per_progress"] for s in sweep)>0,
        right_to_left_0_to_90=min(s["leftward_per_progress"] for s in sweep if s["theta"]<=math.pi/2)>0,
        complete_controller_code_path=ctl.phase=="final_hold",
        command_no_reverse=min(s["action"][0] for s in traces)>=0,
        command_whole_body_far=max(s["whole_max_x"] for s in traces)<2.5,
        nominal_time_under_180=traces[-1]["t"]+boundary["t"]+dt<180,
    )
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(checks=checks,all_pass=all(checks.values()),recipe=recipe,
        max_sweep_position_error_m=max(s["position_error_m"] for s in sweep),
        max_sweep_orientation_error_rad=max(s["orientation_error_rad"] for s in sweep),
        synthetic_suffix_seconds=traces[-1]["t"]+dt,
        maximum_source_geometry_x=max(s["whole_max_x"] for s in traces),
        phases=phase_events,door_scenario=a.door_scenario,physical_execution_verified=False,
        limitations=["Synthetic door/contact input and exact joint tracking; not a simulated episode.",
                     "URDF geometry is not cooked PhysX geometry or self-contact certification."])
    (out/"checks.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    if not result["all_pass"]:raise AssertionError("Compact CPU check failed")


if __name__=="__main__":main()
