"""Author sampling only, on CPU; no scenes and no reachability filtering."""
import argparse,copy,json,math,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from success_random5.common import BASE,sha,write,read,head

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','diff',BASE,'--','migration/wc4_act/author_fast5','migration/wc4_act/fullcycle_task_monitor.py'],cwd=ROOT,text=True)
    run.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'.repro/runs/wc4-delay-only-20261028'
    old=read(prior/'manifest.json');cli=copy.deepcopy(old['trials'][0]['argv'])
    config=run/'frozen-physics.yaml';config.write_bytes((prior/'frozen-physics.yaml').read_bytes())
    assert sha(config)==sha(ROOT/'migration/wc4_act/author_fast5/frozen-r7-physics.yaml')
    cli=[x for x in cli if x not in ['--no_ikpush_env_randomization','--no_enable_depth_camera_randomization','--enable_depth_camera_randomization']]
    cli+=['--enable_depth_camera_randomization']
    def put(args,k,v):
        if k in args:args[args.index(k)+1]=str(v)
        else:args.extend([k,str(v)])
    put(cli,'--a2w_float_ik_config',config);put(cli,'--steps',6000);put(cli,'--num_envs',1)
    write(run/'geometry.json',dict(expert=read(ROOT/'migration/wc4_act/author_fast5/config.json'),live_joint9_recording=True))
    from overview import load
    sys.path.insert(0,str(ROOT/'high-level/float_ik'))
    m=load('random5_prepare_author','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    trials=[]
    for index,seed in enumerate(range(62100,62105)):
        argv=cli.copy();put(argv,'--seed',seed);sys.argv=['random5_prepare',*argv]
        args=m.parse_args();args.sim_dt=float(np.float32(.02))
        env=m.make_env_args(args,0);before=json.loads(env.ikpush_randomization_json)
        assert before['enabled']
        np.random.seed(seed)
        for name in ['wrist','front']:
            rot=m.dc.wrist_camera_rotation_radians_from_args(env) if name=='wrist' else [math.radians(getattr(env,'front_camera_'+axis+'_deg')) for axis in ['yaw','pitch','roll']]
            m.cached_local_camera_transform_from_cfg(name,m.dc.camera_cfg_for_args(name,env),rot,args=env)
        poses={k:{n:np.asarray(v).tolist() for n,v in pose.items()} for k,pose in env._camera_axis_local_poses.items()}
        trials.append(dict(index=index,seed=seed,nominal=False,argv=argv,
                           sampling_before_asset_overrides=before,camera_local_poses=poses))
        (run/('trial_%02d'%index)).mkdir()
    tracked=subprocess.check_output(['git','ls-files','migration/wc4_act','high-level/float_ik','high-level/dp','high-level/data'],cwd=ROOT,text=True).splitlines()
    sources={n:sha(ROOT/n) for n in tracked if (ROOT/n).is_file()}
    write(run/'manifest.json',dict(start_commit=head(),accepted_execution='f85f0b44724cc7abb271ce2f8213dc063b583f6e',accepted_freeze=BASE,
        maximum_scenes=5,maximum_sim_seconds=120,trials=trials,seeds=[t['seed'] for t in trials],
        yaml=str(config),yaml_sha256=sha(config),geometry_sha256=sha(run/'geometry.json'),source_sha256=sources,
        r7_physics_after_asset_overrides=old['r7_physics_after_asset_overrides'],
        sampling='Author make_env_args(args, env_index=0), seeds 62100..62104, one process/scene per seed; no filtering. Accepted frozen physics; original pose distribution and author independent camera jitter +/-0.02m,+/-5deg.',
        video=False,conversion=False,training=False,automatic_retry=False))
    write(run/'initial-list.json',trials)
    print(json.dumps({'run':str(run),'head':head(),'seeds':[t['seed'] for t in trials],'scenes_created':0}))

if __name__=='__main__':main()
