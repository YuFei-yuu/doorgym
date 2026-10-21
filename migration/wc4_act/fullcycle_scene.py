"""Environment-to-R7 design frame. Geometry only; no simulator mutations."""
import math
import numpy as np
from fullcycle_geometry import transform, apply, origin


class SceneFrame:
    def __init__(self, hinge_world=(2.467,-.463,.01), yaw_delta=0.,
                 base_z=.5, base_pitch=0., door_plane_x=2.5, wall_plane_x=2.5):
        self.yaw_delta=float(yaw_delta)
        self.world_from_reference=transform([0,0,0],[0,0,self.yaw_delta])
        nominal=np.array([2.467,-.463,.01])
        self.world_from_reference[:3,3]=np.asarray(hinge_world)-self.world_from_reference[:3,:3]@nominal
        if np.max(abs(self.world_from_reference-np.eye(4)))<1e-12:self.world_from_reference=np.eye(4)
        self.reference_from_world=np.linalg.inv(self.world_from_reference)
        self.base_z=float(base_z)-self.world_from_reference[2,3]
        self.base_pitch=float(base_pitch)
        self.door_plane_x=float(door_plane_x);self.wall_plane_x=float(wall_plane_x)
        self.pass_plane_x=min(self.door_plane_x,self.wall_plane_x)

    def to_reference_base(self,base):
        p=apply(self.reference_from_world,np.array([[base[0],base[1],0.]]))[0]
        return np.r_[p[:2],float(base[2])-self.yaw_delta]

    def to_world_base(self,base):
        p=apply(self.world_from_reference,np.array([[base[0],base[1],0.]]))[0]
        return np.r_[p[:2],float(base[2])+self.yaw_delta]

    def base_transform(self,base):
        return transform([*base[:2],self.base_z],[0,self.base_pitch,base[2]])

    @classmethod
    def from_runtime(cls,m,st):
        # Use the imported asset's real hinge origin and runtime actor pose.
        from fullcycle_geometry import URDF
        door=st.door
        path=m.HIGH_LEVEL_ROOT/'data/asset'/door.spec['path']
        # Specs are relative to door_set, matching the source loader.
        if not path.exists():path=m.HIGH_LEVEL_ROOT/'data/asset/door_set'/door.spec['path']
        u=URDF(path)
        j=next(j for j in u.joints if j[0]==door.spec.get('door_dof_name','door_hinge_joint'))
        local=j[4][:3,3]
        actor=np.array([st.args.door_x,st.args.door_y,st.args.door_z_offset],float)+np.asarray(door.actor_position_offset,float)
        actor[2]-=float(getattr(door,'bounding',{'min':[0,0,0]})['min'][2])*float(door.actor_scale)
        R=transform(rpy=[0,0,float(door.actor_yaw)])[:3,:3]
        hinge=actor+R@(local*float(door.actor_scale))
        return cls(hinge,float(door.actor_yaw)-math.pi/2,
            st.args.robot_z,st.args.robot_pitch,st.args.door_x,
            st.args.door_x+st.args.door_wall_x_offset)

    def metadata(self):
        return dict(world_from_reference=self.world_from_reference.tolist(),
            base_z_reference=self.base_z,base_pitch=self.base_pitch,
            door_plane_x=self.door_plane_x,wall_plane_x=self.wall_plane_x,
            pass_plane_x=self.pass_plane_x,meaning='Fixed R7 geometry in the actual door frame; no angle-based endpoint replanning')
