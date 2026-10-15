"""Check per-env frame coverage and one render per recorded time step, without physics."""
from types import SimpleNamespace
from pathlib import Path
import overview

def main():
    calls=[];recorders=[]
    class Recorder:
        def __init__(self,*args):self.count=0;self.expected=None;self.indices=[];self.renders=0;recorders.append(self)
        def capture(self,index,text='',render=False):self.count+=1;self.indices.append(index);self.renders+=int(render)
    states=[SimpleNamespace(index=i,env=i,args=SimpleNamespace(sim_dt=0.02,steps=1100),last_phase='push_door') for i in range(2)]
    result=(states,'depth_only')
    module=SimpleNamespace(create_parallel_env_states=lambda *args:result,update_door_twin_tracker=lambda *args:calls.append(args),gymapi=None)
    original=overview.Overview
    try:
        overview.Overview=Recorder
        cameras=overview.install_live(module,Path('/not-created'))
        assert module.create_parallel_env_states(None,None) is result
        for step in range(1100):
            for st in states:module.update_door_twin_tracker(st,step,[1.0,0.0])
        assert len(calls)==2200 and len(cameras)==2
        for camera in cameras.values():assert camera.count==camera.expected==550 and camera.indices==list(range(0,1100,2))
        assert sum(c.renders for c in cameras.values())==550
        cmd=['python','/repo/isaacgym_float_ik_a2w_basearn_push_door_parallel.py','--num_envs','2']
        wrapped=overview.wrap_live_command(cmd,'/output')
        assert wrapped[-2:]==cmd[-2:] and wrapped[3]=='--'
    finally:overview.Overview=original
    print('PASS: both environments record all 550 frames; tracker calls unchanged; shared render only once per frame')

if __name__=='__main__':main()
