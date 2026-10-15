"""Observer-camera recording hooks; never supplies policy inputs or commands."""
import importlib.util
import json
import math
from pathlib import Path
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]

def load(name, relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

class Overview:
    def __init__(self,gym,sim,env,gymapi,output,label):
        self.gym,self.sim,self.env=gym,sim,env
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        cp=gymapi.CameraProperties();cp.width=960;cp.height=640;cp.horizontal_fov=65
        self.eyes=[[4.5,-3.5,2.5],[0.0,3.5,2.5]];self.target=[2.4,0.0,0.8];self.cameras=[]
        for eye in self.eyes:
            camera=gym.create_camera_sensor(env,cp)
            if camera<0:raise RuntimeError('Overview camera creation failed')
            gym.set_camera_location(camera,env,gymapi.Vec3(*eye),gymapi.Vec3(*self.target))
            self.cameras.append(camera)
        self.color=gymapi.IMAGE_COLOR;self.label=label;self.count=0;self.indices=[];self.expected=None
        self.writer=imageio.get_writer(str(self.output/'rollout.mp4'),fps=25,codec='libx264',quality=7,
            macro_block_size=1,ffmpeg_params=['-threads','2','-movflags','+faststart'])
    def capture(self,index,text='',render=False):
        if render:
            self.gym.step_graphics(self.sim);self.gym.render_all_camera_sensors(self.sim)
        views=[]
        for camera in self.cameras:
            rgba=self.gym.get_camera_image(self.sim,self.env,camera,self.color)
            views.append(np.asarray(rgba,dtype=np.uint8).reshape(640,960,4)[:,:,:3].copy())
        rgb=np.concatenate(views,axis=1)
        im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
        for j,view_name in enumerate(['APPROACH SIDE','EXIT SIDE']):
            x=j*960;draw.rectangle((x,0,x+960,38),fill=(20,20,20))
            draw.text((x+10,5),self.label+' | '+view_name,fill=(255,235,100))
            draw.text((x+10,22),f'frame {self.count:04d} | t={self.count/25:.2f}s | {text}',fill='white')
        self.writer.append_data(np.asarray(im));self.count+=1;self.indices.append(int(index))
    def close(self):
        self.writer.close()
        receipt={'video':str(self.output/'rollout.mp4'),'label':self.label,'frames':self.count,'fps':25,
            'duration_s':self.count/25,'resolution':[1920,640],'camera_eyes':self.eyes,'camera_target':self.target,
            'source_indices':self.indices,'expected_frames':self.expected,
            'complete':self.expected is None or self.count==self.expected,
            'meaning':'Observer RGB. State replay is visualization, not a fresh execution result.'}
        (self.output/'recording.json').write_text(json.dumps(receipt,indent=2)+'\n')

def install_live(module,output):
    cameras={};rendered=[None];original_create=module.create_parallel_env_states;original_update=module.update_door_twin_tracker
    def create(*args,**kwargs):
        result=original_create(*args,**kwargs)
        gym,sim=args[:2]
        for st in result[0]:
            cameras[st.index]=Overview(gym,sim,st.env,module.gymapi,Path(output)/f'env_{st.index:04d}','LIVE EXECUTION | observer RGB')
            stride=max(1,int(round(1/(25*float(st.args.sim_dt)))))
            cameras[st.index].expected=int(math.ceil(st.args.steps/stride))
        return result
    def update(st,step,door_pos):
        result=original_update(st,step,door_pos)
        dt=float(st.args.sim_dt);stride=max(1,int(round(1/(25*dt))))
        if step%stride==0:
            angle=float(np.rad2deg(np.asarray(door_pos)[0]))
            cameras[st.index].capture(step,f'env={st.index} phase={st.last_phase} hinge={angle:.1f}deg',render=rendered[0]!=step)
            rendered[0]=step
        return result
    module.create_parallel_env_states=create;module.update_door_twin_tracker=update
    return cameras

def live_main():
    output=Path(sys.argv[1]);assert sys.argv[2]=='--'
    author=ROOT/'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py'
    sys.path.insert(0,str(author.parent));sys.argv=[str(author),*sys.argv[3:]]
    module=load('wc4_live_author',str(author.relative_to(ROOT)));cameras=install_live(module,output)
    try:module.main()
    finally:
        for camera in cameras.values():camera.close()
    if not cameras or any(v.count!=v.expected for v in cameras.values()):
        raise RuntimeError('Missing or incomplete overview video; retain partial recording for diagnosis')

def wrap_live_command(command,output):
    expected='isaacgym_float_ik_a2w_basearn_push_door_parallel.py'
    if Path(command[1]).name!=expected:
        raise RuntimeError('Overview wrapper is limited to the frozen A2W ikpush route')
    return [command[0],str(Path(__file__).with_name('live_with_overview.py')),str(output),'--',*command[2:]]

def record_main():
    module=load('wc4_record_author','high-level/dp/record/record_door_dp_dataset_a2w_state10.py')
    original=module.run_logged_subprocess;counter=[0]
    def run(command,args,cwd,check=True):
        batch=counter[0];counter[0]+=1
        output=Path(args.run_metadata_dir or args.raw_root)/f'overview_batch_{batch:03d}'
        return original(wrap_live_command(command,output),args,cwd,check)
    module.run_logged_subprocess=run;module.main()

def play_main():
    module=load('wc4_play_author','high-level/dp/play/play_door_policy.py');original=module.subprocess.run
    def run(command,**kwargs):
        if len(command)<2 or Path(command[1]).name!='isaacgym_float_ik_a2w_basearn_push_door_parallel.py':
            return original(command,**kwargs)
        flag='--dp_log_path';log=Path(command[command.index(flag)+1])
        return original(wrap_live_command(command,log.parent/(log.stem+'_overview')),**kwargs)
    module.subprocess.run=run
    try:module.main()
    finally:module.subprocess.run=original

def evaluate_main():
    module=load('wc4_eval_author','high-level/dp/eval/eval_door_policy_success.py')
    module.PLAY_SCRIPT=Path(__file__).with_name('play_with_overview.py');module.main()
