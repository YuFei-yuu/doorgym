"""Pad-centre + push-axis IK. Source kinematics only; no state/asset changes."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from fullcycle_compact import Arm, ClosingCurve
from fullcycle_geometry import transform

# Centre of the outward x face of gripperStator collision box #2,
# expressed in ee_gripper_link. Box centre (0.135,0,-.020), size
# (.030,.060,.020), ee origin (0.135,0,0), all in stator frame.
PAD_LOCAL=np.array([.015,0.,-.020])

def deployment_pose(start,goal,amount):
    """Straight contact-centre travel with shortest continuous axis rotation."""
    a=float(np.clip(amount,0,1))
    p0=start[:3,3]+start[:3,:3]@PAD_LOCAL
    p1=goal[:3,3]+goal[:3,:3]@PAD_LOCAL
    R=Slerp([0.,1.],Rotation.from_matrix(np.stack([start[:3,:3],goal[:3,:3]])))(a).as_matrix()
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=(1-a)*p0+a*p1-R@PAD_LOCAL
    return T

class PadArm(Arm):
    def pad(self,q):
        F=self.fk(q)
        return F[:3,3]+F[:3,:3]@PAD_LOCAL

    def solve(self,target,seed,orientation_weight=1.,iterations=40):
        # Preserve inherited withdrawal/release target semantics. Only the
        # contact approach and push use the relaxed contact-axis constraint.
        if orientation_weight<.9:
            return super().solve(target,seed,orientation_weight,iterations)
        seed=np.clip(np.asarray(seed)[:6],self.lo+1e-8,self.hi-1e-8)
        goal=target[:3,3]+target[:3,:3]@PAD_LOCAL
        axis=target[:3,0]
        def residual(q):
            F=self.fk(q);point=F[:3,3]+F[:3,:3]@PAD_LOCAL
            return np.r_[4*(point-goal),F[:3,0]-axis,.001*(q-seed)]
        r=least_squares(residual,seed,bounds=(self.lo,self.hi),max_nfev=iterations,
                        ftol=1e-7,xtol=1e-7,gtol=1e-7)
        F=self.fk(r.x);point=F[:3,3]+F[:3,:3]@PAD_LOCAL
        error=float(np.arccos(np.clip(np.dot(F[:3,0],axis),-1,1)))
        return r.x,dict(position_error_m=float(np.linalg.norm(point-goal)),
            orientation_error_rad=error,constraint='pad centre plus tool x push axis; roll free',
            ee_origin_reference_error_m=float(np.linalg.norm(F[:3,3]-target[:3,3])),
            joint_slack_rad=float(np.min(np.r_[r.x-self.lo,self.hi-r.x])))

class PadCurve(ClosingCurve):
    def tool(self,theta,extra_offset=0):
        # A constant-height circular contact path. Reference roll is merely
        # a coordinate convention: IK constrains the push axis, not that roll.
        P=self.point(theta,self.c['pad_offset']+extra_offset)
        T=transform([*P,self.c['height']],[0,0,theta])
        T[:3,3]-=T[:3,:3]@PAD_LOCAL
        return T
