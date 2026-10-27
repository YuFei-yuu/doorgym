"""One-scene adapter around the unchanged fullcycle_expert.run()."""
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from newexpert_stage4.prepare import sha, write

def verify(run):
    m=json.loads((run/'manifest.json').read_text())
    assert len(m['trials'])==1 and m['maximum_sim_seconds']==240
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,'source mismatch: '+n
    assert sha(m['yaml'])==m['yaml_sha256']
    c=json.loads((run/'consistency.json').read_text())
    assert sha(run/'geometry.json')==c['geometry_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['start_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    manifest=verify(run);out=run/'attempt_000';out.mkdir(exist_ok=False)
    assert 'FLOCK' in Path('/proc/self/fdinfo/9').read_text()
    uuid=Path(os.readlink('/proc/self/fd/9')).stem
    assert uuid==subprocess.check_output(['nvidia-smi','-i',os.environ['VIDEO2DOOR_GPU'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    from overview import load, install_live
    from close_inplace_stage6 import runtime as expert
    sys.path.insert(0,str(ROOT/'high-level/float_ik'))
    m=load('stage6_author','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    from fullcycle_pilot import install_frozen_trial, audit_effective_trial
    install_frozen_trial(m,run/'manifest.json',0,out)
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
        assert args.steps==12000 and args.num_envs==1
        assert not args.dp_policy_checkpoint and not args.record_dp_dataset
        return args
    m.parse_args=parsed
    plan=json.loads((run/'geometry.json').read_text())
    def runner(gym,sim,states,viewer,args,dt,dof_names):
        assert len(states)==1 and int(args.steps)*dt<=240.000001
        st=states[0];audit_effective_trial(gym,st,out)
        expected=json.loads((run/'r7-initial-state.json').read_text())
        current={'door':expert.serial(gym.get_actor_dof_states(st.env,st.door_actor,m.gymapi.STATE_ALL)),
                 'arm':expert.serial(gym.get_actor_dof_states(st.env,st.arm_actor,m.gymapi.STATE_ALL)),
                 'root':expert.serial(gym.get_actor_rigid_body_states(st.env,st.arm_actor,m.gymapi.STATE_ALL)[0]),
                 'base_start':st.base_start.tolist(),'yaw':st.yaw_start}
        current=json.loads(json.dumps(current))
        from newexpert_stage6.bindings import compare_spawn
        er=expected['root']['pose']
        frozen=dict(base_start=expected['base_start'],robot_z=er['p']['z'],root_quat=[er['r'][k] for k in ['x','y','z','w']],yaw=expected['yaw'])
        comparison=compare_spawn(current,frozen)
        comparison['dofs_identical']=current['door']==expected['door'] and current['arm']==expected['arm']
        write(out/'initial-r7-comparison.json',dict(comparison,actual=current,expected=expected))
        assert comparison['matches_frozen'] and comparison['dofs_identical'],'Initial state differs from nominal'
        write(run/'physics-start.json',{'at':time.time(),'dt':dt,'maximum_steps':12000,'source_verified':True})
        class GuardGym:
            def __init__(self):self.steps=0;self.allowed={os.getpid()}
            def __getattr__(self,name):return getattr(gym,name)
            def simulate(self,simulation):
                if self.steps%50==0:
                    free=shutil.disk_usage(ROOT).free
                    if free<40*1024**3:raise expert.StopAttempt('stage6_capacity_below_40GiB')
                    if self.steps%250==0:
                        try:verify(run)
                        except BaseException as exc:raise expert.StopAttempt('stage6_source_binding_mismatch:'+repr(exc))
                        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                        conflicts=[line for line in apps.splitlines() if line.split(',')[0].strip()==uuid and int(line.split(',')[1]) not in self.allowed]
                        with (run/'resource-monitor.jsonl').open('a') as f:f.write(json.dumps({'step':self.steps,'free_bytes':free,'compute_apps':apps,'conflicts':conflicts,'at':time.time()})+'\n')
                        if conflicts:raise expert.StopAttempt('stage6_gpu_uuid_conflict')
                    if self.steps and self.steps%50==0:
                        required=['states.jsonl','contacts.jsonl','recording-clock.jsonl']
                        if any(not (out/n).exists() or (out/n).stat().st_size==0 for n in required):
                            raise expert.StopAttempt('stage6_critical_log_missing')
                if self.steps>=12000:raise expert.StopAttempt('stage6_150s_budget')
                self.steps+=1
                return gym.simulate(simulation)
        expert.run(m,cams,out,plan,GuardGym(),sim,states,viewer,args,dt,dof_names)
    m.run_parallel_demo=runner
    original_sim=m.base_ik.create_sim
    def one_scene(*args,**kwargs):
        verify(run)
        write(run/'scene-claim.json',{'episodes':1,'max_seconds':240,'time':time.time(),'pid':os.getpid()})
        write(out/'scene-creation-receipt.json',{'scenes':1,'episodes':1,'max_sim_seconds':240,'wall_time':time.time()})
        return original_sim(*args,**kwargs)
    m.base_ik.create_sim=one_scene
    cli=manifest['trials'][0]['argv']
    sys.argv=[str(ROOT/'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py'),*cli]
    write(out/'command.json',{'argv':sys.argv,'entry_argv':[sys.executable,__file__,'--run',str(run)],'cwd':str(ROOT),
         'head':manifest['start_commit'],'gpu':{k:os.environ.get(k) for k in ['VIDEO2DOOR_GPU','CUDA_VISIBLE_DEVICES','VK_LAYER_PATH','VK_INSTANCE_LAYERS']},'uuid':uuid,'start_time':time.time()})
    code=0
    try:
        m.main()
        result=json.loads((out/'result.json').read_text())
        if result.get('error'):raise RuntimeError('Recorded runtime exception: '+result['reason'])
    except BaseException:
        code=1;(run/'entry-error.txt').write_text(traceback.format_exc());raise
    finally:
        for camera in cams.values():
            if not (camera.output/'recording.json').exists():camera.close()
        write(run/'entry-exit.json',{'returncode':code,'time':time.time(),'scene_claimed':(run/'scene-claim.json').exists()})

if __name__=='__main__':main()
