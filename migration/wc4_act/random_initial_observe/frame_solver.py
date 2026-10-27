"""Exact change of coordinates into the unchanged nominal IK solver."""
import numpy as np
from close_inplace_stage4.common import FullPose,baseT
from fullcycle_geometry import apply
class FrameSolver(FullPose):
    def __init__(self,robot,scene):
        super().__init__(robot);self.scene=scene
    def mapping(self,B):return baseT(B)@np.linalg.inv(self.scene.base_transform(B))
    def solve(self,B,p,R,seed,nfev):
        M=self.mapping(B)
        return super().solve(B,apply(M,np.array([p]))[0],M[:3,:3]@R,seed,nfev)
    def solve_point(self,B,p,u,seed,nfev,full_rotation=None,ee=False):
        M=self.mapping(B)
        return super().solve_point(B,apply(M,np.array([p]))[0],None if u is None else M[:3,:3]@u,seed,nfev,
            None if full_rotation is None else M[:3,:3]@full_rotation,ee)
