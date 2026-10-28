"""One frozen random initial state, accepted expert, raw depth, <=120s."""
import argparse,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from expert50.common import read,write,verify,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--attempt',type=int,default=0)
    a=p.parse_args();batch=a.run.resolve();root_manifest=verify(batch);index=0
    run=batch/'attempts'/('seed%d'%a.seed);manifest=read(run/'manifest.json');out=run/('attempt_%03d'%a.attempt)
    assert len(manifest['trials'])==1 and manifest['trials'][0]['seed']==a.seed
    assert manifest['root_manifest_sha256']==sha(batch/'manifest.json')
    if a.attempt:
        previous=run/('attempt_%03d'%(a.attempt-1))
        receipt=read(previous/'entry-exit.json')
        assert receipt['returncode']!=0,'Cannot retry a completed task failure/success'
        assert not (previous/'result.json').exists() or read(previous/'result.json')['error']
    out.mkdir(exist_ok=False)
    assert 'FLOCK' in Path('/proc/self/fdinfo/9').read_text()
    uuid=Path(os.readlink('/proc/self/fd/9')).stem
    assert uuid==subprocess.check_output(['nvidia-smi','-i',os.environ['VIDEO2DOOR_GPU'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    from overview import load
    from newexpert_stage7.adapters import install_raw_render
    from legdiag_random5 import runtime as expert
    from fullcycle_pilot import install_frozen_trial,audit_effective_trial
    sys.path.insert(0,str(ROOT/'high-level/float_ik'))
    m=load('random5_author','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    install_frozen_trial(m,run/'manifest.json',index,out)
    render_receipt=install_raw_render(m)
    parse=m.parse_args
    def parsed():
        args=parse();args.show_camera_images=True
        assert args.steps==6000 and args.num_envs==1
        assert not args.dp_policy_checkpoint and not args.record_dp_dataset
        return args
    m.parse_args=parsed
    plan=read(batch/'geometry.json')
    def runner(gym,sim,states,viewer,args,dt,dof_names):
        assert len(states)==1 and int(args.steps)*dt<=120.000001
        audit_effective_trial(gym,states[0],out)
        st=states[0];ga=m.gymapi
        current=dict(door=expert.serial(gym.get_actor_dof_states(st.env,st.door_actor,ga.STATE_ALL)),
            arm=expert.serial(gym.get_actor_dof_states(st.env,st.arm_actor,ga.STATE_ALL)),
            root=expert.serial(gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)[0]),
            base_start=st.base_start.tolist(),yaw=st.yaw_start)
        current=json.loads(json.dumps(current))
        write(out/'sampled-actual-initial.json',current)
        write(out/'physics-start.json',dict(at=time.time(),dt=dt,maximum_steps=6000))
        class GuardGym:
            def __init__(self):self.steps=0
            def __getattr__(self,name):return getattr(gym,name)
            def simulate(self,simulation):
                if self.steps>=6000:raise expert.StopAttempt('120s_budget')
                if self.steps%250==0:
                    verify(batch)
                    if shutil.disk_usage(ROOT).free<40*1024**3:raise RuntimeError('capacity_below_40GiB')
                    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                    conflicts=[x for x in apps.splitlines() if x.split(',')[0].strip()==uuid and int(x.split(',')[1])!=os.getpid()]
                    with (out/'resource-monitor.jsonl').open('a') as f:f.write(json.dumps(dict(step=self.steps,free_bytes=shutil.disk_usage(ROOT).free,compute_apps=apps,conflicts=conflicts))+'\n')
                self.steps+=1
                return gym.simulate(simulation)
        expert.run(m,{},out,plan,GuardGym(),sim,states,viewer,args,dt,dof_names)
    m.run_parallel_demo=runner
    original_sim=m.base_ik.create_sim
    def one_scene(*args,**kwargs):
        verify(batch)
        write(out/'scene-claim.json',dict(index=index,attempt=a.attempt,seed=manifest['trials'][index]['seed'],max_seconds=120,time=time.time(),pid=os.getpid()))
        return original_sim(*args,**kwargs)
    m.base_ik.create_sim=one_scene
    sys.argv=[str(ROOT/'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py'),*manifest['trials'][index]['argv']]
    write(out/'command.json',dict(argv=sys.argv,entry_argv=[sys.executable,__file__,'--run',str(batch),'--seed',str(a.seed),'--attempt',str(a.attempt)],cwd=str(ROOT),head=manifest['start_commit'],uuid=uuid,
        gpu={k:os.environ.get(k) for k in ['VIDEO2DOOR_GPU','CUDA_VISIBLE_DEVICES','VK_LAYER_PATH','VK_INSTANCE_LAYERS']},time=time.time()))
    code=0
    try:
        m.main()
        result=read(out/'result.json')
        assert read(out/'initial-state.json')==read(out/'sampled-actual-initial.json')
        if result['error']:raise RuntimeError(result['error'])
    except BaseException:
        code=1;(out/'entry-error.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'raw-render-receipt.json',render_receipt)
        write(out/'entry-exit.json',dict(returncode=code,time=time.time(),scene_claimed=(out/'scene-claim.json').exists()))

if __name__=='__main__':main()
