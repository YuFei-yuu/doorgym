"""No-video rendering/encoding adapters. No commands, physics or scoring."""
from pathlib import Path
import numpy as np

def install_raw_render(module):
    context={};original_create=module.create_parallel_env_states;original_update=module.update_door_twin_tracker
    receipt={'overview_video':False,'depth_video':False,'rendered_steps':[]}
    def create(*args,**kwargs):
        result=original_create(*args,**kwargs)
        context['gym'],context['sim']=args[:2]
        return result
    def update(st,step,door_pos):
        result=original_update(st,step,door_pos)
        # Exact post-physics render point formerly supplied by Overview.capture.
        context['gym'].step_graphics(context['sim'])
        context['gym'].render_all_camera_sensors(context['sim'])
        receipt['rendered_steps'].append(int(step))
        return result
    module.create_parallel_env_states=create;module.update_door_twin_tracker=update
    return receipt

class DepthSink:
    def __init__(self,name,counters):self.name=name;self.counters=counters;counters[name]=0
    def append_data(self,image):
        image=np.asarray(image)
        if image.shape[:2]!=(480,640) or image.dtype!=np.uint8:raise RuntimeError('stage7_invalid_depth_buffer')
        self.counters[self.name]+=1
    def close(self):pass

def no_video_writer(output,counters):
    def writer(path,*args,**kwargs):
        path=Path(path)
        if path.parent==output and path.name in ['front_depth.mp4','wrist_depth.mp4']:
            return DepthSink(path.name,counters)
        raise RuntimeError('stage7_unexpected_video_writer:'+str(path))
    return writer

def record_contract(steps,state_steps,contact_steps,clock_steps,receipt,rendered_steps,video_files):
    n=(steps+1)//2;errors=[]
    for name,actual,expected in [('states',state_steps,list(range(steps))),('contacts',contact_steps,list(range(steps))),
        ('record_clock',clock_steps,list(range(0,steps,2))),('raw_render',rendered_steps,list(range(0,steps,2)))]:
        if actual!=expected:errors.append(name+'_continuity')
    if receipt.get('encoded') is not False or receipt.get('raw_depth_observations_retained') is not True:
        errors.append('encoding_receipt')
    if receipt.get('buffers_observed')!={'front_depth.mp4':n,'wrist_depth.mp4':n}:errors.append('depth_buffer_count')
    if video_files:errors.append('unexpected_video_files')
    return {'passed':not errors,'failures':errors,'frames':n,'video_required':False}
