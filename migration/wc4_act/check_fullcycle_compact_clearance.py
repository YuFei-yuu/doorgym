"""Source convex-geometry diagnostics for the compact command integration.

Not PhysX import/contact certification. Inherited self-contact is out of scope.
Only new robot-vs-door/frame geometry is compared; intended distal contact and
initial disengagement are reported separately from folded bypass clearance.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from fullcycle_geometry import URDF, transform
from fullcycle_collision import collision_pieces, signed_gap

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument("--trace",required=True)
    p.add_argument("--output",required=True);a=p.parse_args()
    rp=ROOT/"high-level/data/asset/a2wz1/urdf/a2wz1.urdf"
    dp=ROOT/"high-level/data/asset/door_set/wc4/model.urdf"
    robot=URDF(rp);door=URDF(dp);R=collision_pieces(rp);D=collision_pieces(dp)
    legs={side+suffix:val for side in ["FL","FR","RL","RR"] for suffix,val in
          [("_hip_joint",0),("_thigh_joint",.8),("_calf_joint",-1.5)]}
    names=["joint%d"%i for i in range(1,7)]+["jointGripper"]
    trace=json.loads(Path(a.trace).read_text());result={};sampled=[]
    for i,r in enumerate(trace):
        if i%10 and i!=len(trace)-1:continue
        B=transform([*r["base"][:2],.5],[0,0,r["base"][2]])
        poses=robot.fk(dict(legs,**dict(zip(names,r["q"]))),B)
        doorposes=door.fk({"door_hinge_joint":r["theta"],"handle_spindle_joint":0.},
                         transform([2.61,0,.01],[0,0,math.pi/2]))
        dw=[(d,d.world(doorposes[d.link])) for d in D]
        best={}
        for part in R:
            pw=part.world(poses[part.link])
            for d,x in dw:
                category=("end_panel" if d.link=="door_panel" and part.link in ("gripperStator","gripperMover")
                          else "wrist_panel" if d.link=="door_panel" and part.link in ("link06","wrist_down_mount","wrist_up_mount")
                          else "other_external")
                gap=signed_gap(pw,x)
                if category not in best or gap<best[category]["gap"]:
                    best[category]=dict(gap=gap,robot_link=part.link,door_link=d.link,
                                       robot_piece=part.name,door_piece=d.name,t=r["t"])
        phase=r["phase"]
        for cat,value in best.items():
            key=phase+"/"+cat
            if key not in result or value["gap"]<result[key]["gap"]:result[key]=value
        sampled.append(dict(t=r["t"],phase=phase,minimums=best))
    out=Path(a.output);out.mkdir(exist_ok=False)
    (out/"source-clearance.json").write_text(json.dumps(dict(
        meaning="Conservative source convex bounds, sampled every 0.2 s; no cooked contact certification",
        minimums=result,samples=sampled),indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
