"""One frozen perturbed trial. Depth buffers retained; only video encoding suppressed."""
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from newexpert_stage4.prepare import sha, write

def verify(run):
    m=json.loads((run/'manifest.json').read_text())
    assert len(m['trials'])==8 and m['maximum_sim_seconds']==150
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,'source mismatch: '+n
    assert sha(m['yaml'])==m['yaml_sha256']
    c=json.loads((run/'preparation-receipt.json').read_text())
    assert sha(run/'geometry.json')==c['geometry_sha256']
    assert sha(run/'manifest.json')==c['manifest_sha256']
    assert sha(run/'initial-coverage.json')==c['initial_coverage_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['start_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();batch=a.run.resolve();index=a.index;assert 0<=index<8;run=batch/('trial_%02d'%index)
    manifest=verify(batch);assert sha(run/'geometry.json')==sha(batch/'geometry.json');out=run/'attempt_000';out.mkdir(exist_ok=False)
    assert 'FLOCK' in Path('/proc/self/fdinfo/9').read_text()
    uuid=Path(os.readlink('/proc/self/fd/9')).stem
    assert uuid==subprocess.check_output(['nvidia-smi','-i',os.environ['VIDEO2DOOR_GPU'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    from overview import load, install_live
    import fullcycle_expert as expert
    sys.path.insert(0,str(ROOT/'high-level/float_ik'))
    m=load('stage5_author','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    from fullcycle_pilot import install_frozen_trial, audit_effective_trial
    install_frozen_trial(m,batch/'manifest.json',index,out)
    cams=install_live(m,out/'overview',post_step_timestamps=True)
    create=m.create_parallel_env_states
    def create_with_camera(*args,**kwargs):
        result=create(*args,**kwargs);gym=args[0]
        for st in result[0]:
            cp=m.gymapi.CameraProperties();cp.width=1280;cp.height=720;cp.horizontal_fov=55
            st._fullcycle_stop_camera=gym.create_camera_sensor(st.env,cp)
            assert st._fullcycle_stop_camera>=0
        return result
    m.create_parallel_env_states=create_with_camera
    parse=m.parse_args
    def parsed():
        args=parse();args.show_camera_images=True
        assert args.steps==7500 and args.num_envs==1
        assert not args.dp_policy_checkpoint and not args.record_dp_dataset
        return args
    m.parse_args=parsed
    plan=json.loads((run/'geometry.json').read_text())
    def runner(gym,sim,states,viewer,args,dt,dof_names):
        assert len(states)==1 and int(args.steps)*dt<=150.000001
        st=states[0];audit_effective_trial(gym,st,out)
        expected=manifest['trials'][index]['expected_spawn']
        current={'door':expert.serial(gym.get_actor_dof_states(st.env,st.door_actor,m.gymapi.STATE_ALL)),
                 'arm':expert.serial(gym.get_actor_dof_states(st.env,st.arm_actor,m.gymapi.STATE_ALL)),
                 'root':expert.serial(gym.get_actor_rigid_body_states(st.env,st.arm_actor,m.gymapi.STATE_ALL)[0]),
                 'base_start':st.base_start.tolist(),'yaw':st.yaw_start}
        current=json.loads(json.dumps(current))
        actual=current['root']['pose'];position=actual['p'];quat=actual['r']
        import numpy as np
        match=bool(np.max(abs(np.array(current['base_start'])-expected['base_start']))<1e-7 and abs(position['z']-expected['robot_z'])<1e-7 and abs(current['yaw']-expected['yaw'])<1e-12 and np.max(abs(np.array([quat[k] for k in ['x','y','z','w']])-expected['root_quat']))<1e-7)
        write(out/'initial-frozen-comparison.json',{'matches_frozen':match,'actual':current,'expected':expected})
        assert match,'Actual spawn differs from frozen initial list'
        write(run/'physics-start.json',{'at':time.time(),'dt':dt,'maximum_steps':7500,'source_verified':True})
        class GuardGym:
            def __init__(self):self.steps=0;self.allowed={os.getpid()}
            def __getattr__(self,name):return getattr(gym,name)
            def simulate(self,simulation):
                if self.steps%50==0:
                    free=shutil.disk_usage(ROOT).free
                    if free<40*1024**3:raise expert.StopAttempt('stage5_capacity_below_40GiB')
                    if self.steps%250==0:
                        try:verify(batch)
                        except BaseException as exc:raise expert.StopAttempt('stage5_source_binding_mismatch:'+repr(exc))
                        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                        conflicts=[line for line in apps.splitlines() if line.split(',')[0].strip()==uuid and int(line.split(',')[1]) not in self.allowed]
                        with (run/'resource-monitor.jsonl').open('a') as f:f.write(json.dumps({'step':self.steps,'free_bytes':free,'compute_apps':apps,'conflicts':conflicts,'at':time.time()})+'\n')
                        if conflicts:raise expert.StopAttempt('stage5_gpu_uuid_conflict')
                    if self.steps and self.steps%50==0:
                        required=['states.jsonl','contacts.jsonl','recording-clock.jsonl']
                        if any(not (out/n).exists() or (out/n).stat().st_size==0 for n in required):
                            raise expert.StopAttempt('stage5_critical_log_missing')
                if self.steps>=7500:raise expert.StopAttempt('stage5_150s_budget')
                self.steps+=1
                return gym.simulate(simulation)
        expert.run(m,cams,out,plan,GuardGym(),sim,states,viewer,args,dt,dof_names)
    import imageio
    original_writer=imageio.get_writer
    depth_counts={}
    class DepthSink:
        def __init__(self,name):self.name=name;depth_counts[name]=0
        def append_data(self,image):
            import numpy as np
            assert np.asarray(image).shape[:2]==(480,640)
            depth_counts[self.name]+=1
        def close(self):pass
    def writer(path,*args,**kwargs):
        name=Path(path).name
        if Path(path).parent==out and name in ['front_depth.mp4','wrist_depth.mp4']:return DepthSink(name)
        return original_writer(path,*args,**kwargs)
    imageio.get_writer=writer
    m.run_parallel_demo=runner
    original_sim=m.base_ik.create_sim
    def one_scene(*args,**kwargs):
        verify(batch)
        write(run/'scene-claim.json',{'episodes':1,'max_seconds':150,'time':time.time(),'pid':os.getpid()})
        write(out/'scene-creation-receipt.json',{'scenes':1,'episodes':1,'max_sim_seconds':150,'wall_time':time.time()})
        return original_sim(*args,**kwargs)
    m.base_ik.create_sim=one_scene
    cli=manifest['trials'][index]['argv']
    sys.argv=[str(ROOT/'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py'),*cli]
    write(out/'command.json',{'argv':sys.argv,'entry_argv':[sys.executable,__file__,'--run',str(batch),'--index',str(index)],'cwd':str(ROOT),
         'head':manifest['start_commit'],'gpu':{k:os.environ.get(k) for k in ['VIDEO2DOOR_GPU','CUDA_VISIBLE_DEVICES','VK_LAYER_PATH','VK_INSTANCE_LAYERS']},'uuid':uuid,'start_time':time.time()})
    code=0
    try:m.main()
    except BaseException:
        code=1;(run/'entry-error.txt').write_text(traceback.format_exc());raise
    finally:
        imageio.get_writer=original_writer
        write(run/'depth-encoding-receipt.json',{'encoded':False,'buffers_observed':depth_counts,'raw_depth_observations_retained':True})
        for camera in cams.values():
            if not (camera.output/'recording.json').exists():camera.close()
        write(run/'entry-exit.json',{'returncode':code,'time':time.time(),'scene_claimed':(run/'scene-claim.json').exists()})

if __name__=='__main__':main()
