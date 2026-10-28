"""Asset-derived door frame; no simulator mutation, no historical control imports."""
from pathlib import Path
import math
import numpy as np
from scipy.spatial.transform import Rotation
from fullcycle_geometry import URDF, transform, apply, origin, mesh_bounds
ROOT = Path(__file__).resolve().parents[3]


def smooth(u):
    u = float(np.clip(u, 0, 1))
    # The polynomial can overshoot 1 by a few ulps for u just below 1.
    # Slerp requires a closed [0, 1] domain, including the real float32 clock.
    return float(np.clip(u*u*u*(10+u*(-15+6*u)), 0., 1.))


def angle_delta(a, b):
    return (b-a+math.pi) % (2*math.pi)-math.pi


class DoorFrame:
    def __init__(self, actor_transform, scale=1., base_z=.5, base_pitch=0.):
        self.robot_z, self.robot_pitch = float(base_z), float(base_pitch)
        self.scale = float(scale)
        path = ROOT/'high-level/data/asset/door_set/wc4/model.urdf'
        self.door = URDF(path)
        hinge = next(j for j in self.door.joints if j[0] == 'door_hinge_joint')
        self.actor = np.asarray(actor_transform, float)
        self.hinge = self.actor[:3, 3]+self.actor[:3, :3]@(hinge[4][:3, 3]*scale)
        # WC4 closed slab tangent = actor +X; closing normal = actor -Y.
        self.tangent = self.actor[:3, 0]
        self.normal = -self.actor[:3, 1]
        self.R = np.column_stack((self.normal, self.tangent, self.actor[:3, 2]))
        self.yaw = math.atan2(self.normal[1], self.normal[0])
        slab = self.door.root.find("link[@name='door_panel']/collision[@name='door_slab']")
        mesh = slab.find('geometry/mesh')
        lo, hi = mesh_bounds(path.parent/mesh.get('filename'))
        # The mesh's near/far y faces are actual WC4 surfaces, not door_x.
        self.slab_lo, self.slab_hi = lo*scale, hi*scale
        self.back_normal_offset = -float(hi[1])*scale

    @classmethod
    def runtime(cls, st):
        d, a = st.door, st.args
        xyz = np.array([a.door_x, a.door_y, a.door_z_offset])+np.asarray(d.actor_position_offset)
        xyz[2] -= d.bounding['min'][2]*d.actor_scale
        return cls(transform(xyz, [0, 0, d.actor_yaw]), d.actor_scale, a.robot_z, a.robot_pitch)

    def base(self, local_xy, yaw):
        p = self.hinge+self.R@np.r_[local_xy, 0.]
        return np.r_[p[:2], yaw+self.yaw]

    def base_transform(self, B):
        return transform([*B[:2], self.robot_z], [0, self.robot_pitch, B[2]])

    def contact(self, cfg, theta, clearance=0.):
        n = self.R@np.array([math.cos(theta), math.sin(theta), 0.])
        tangent = self.R@np.array([-math.sin(theta), math.cos(theta), 0.])
        p = (self.hinge+cfg['contact_radius_m']*tangent+
             (self.back_normal_offset-clearance)*n+
             self.R@np.array([0., 0., cfg['contact_height_above_hinge_m']]))
        # Front-side reach around the free edge, with a prescribed wrist pose.
        wrist_yaw=math.radians(cfg['contact_yaw_at_open_deg'])*theta/(math.pi/2)
        R = Rotation.from_rotvec(n*math.radians(cfg['contact_roll_deg'])).as_matrix()@self.R@Rotation.from_euler('z',wrist_yaw).as_matrix()
        # Source stator box 2 is centered at EE + [0,0,-.02], size [.03,.06,.02].
        # Use the outer support corner, not its projection onto the slab plane:
        # the tangential offset keeps the wrist mount outside the free edge.
        pad=np.array([.015,.03,-.02])
        return p-R@pad, R, p, n, pad


def closing_reference(cfg, frame, t):
    # theta is a clock-driven path parameter, never a measured door angle.
    s=progress(t,cfg['close_s'],cfg['progress_ramp_s'])
    theta=math.radians(cfg['reference_open_angle_deg'])*(1-s)
    yaw=math.radians(cfg['closing_start_yaw_deg'])*(1-smooth(s/cfg['turn_progress_end']))
    follow=cfg['follow_m']*smooth((s-cfg['follow_progress_start'])/(1-cfg['follow_progress_start']))
    xy=np.asarray(cfg['closing_start_hinge_xy'])+[follow,0.]
    B = frame.base(xy, yaw)
    p, R, pad, n, local = frame.contact(cfg, theta)
    return B, p, R, theta, pad


def progress(t,duration,ramp):
    """One velocity plateau with quintic start/stop ramps; C2 at joins."""
    t=float(np.clip(t,0,duration))
    integral=lambda x:2.5*x**4-3*x**5+x**6
    if t<ramp:return ramp*integral(t/ramp)/(duration-ramp)
    if t>duration-ramp:return 1-ramp*integral((duration-t)/ramp)/(duration-ramp)
    return (t-ramp/2)/(duration-ramp)
