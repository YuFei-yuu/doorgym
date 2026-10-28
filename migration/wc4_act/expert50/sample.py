"""Unfiltered author initial-state sampling; each seed is prepared exactly once."""
import copy,json,math,sys
import numpy as np
from expert50.common import ROOT,read,write,sha

class Sampler:
    def __init__(self,run):
        self.run=run;self.manifest=read(run/'manifest.json')
        sys.path.insert(0,str(ROOT/'high-level/float_ik'))
        from overview import load
        self.author=load('expert50_sampling_author','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    def prepare(self,seed):
        root=self.manifest;m=self.author;run=self.run
        target=run/'attempts'/('seed%d'%seed);target.mkdir(exist_ok=False)
        argv=copy.deepcopy(root['author_argv_template'])
        argv[argv.index('--seed')+1]=str(seed)
        argv[argv.index('--a2w_float_ik_config')+1]=root['yaml']
        sys.argv=['expert50_sample',*argv];args=m.parse_args();args.sim_dt=float(np.float32(.02))
        env=m.make_env_args(args,0);sample=json.loads(env.ikpush_randomization_json);assert sample['enabled']
        np.random.seed(seed)
        for name in ['wrist','front']:
            rot=m.dc.wrist_camera_rotation_radians_from_args(env) if name=='wrist' else [math.radians(getattr(env,'front_camera_'+axis+'_deg')) for axis in ['yaw','pitch','roll']]
            m.cached_local_camera_transform_from_cfg(name,m.dc.camera_cfg_for_args(name,env),rot,args=env)
        poses={k:{n:np.asarray(v).tolist() for n,v in pose.items()} for k,pose in env._camera_axis_local_poses.items()}
        trial=dict(index=0,seed=seed,nominal=False,argv=argv,sampling_before_asset_overrides=sample,camera_local_poses=poses)
        manifest=dict(start_commit=root['start_commit'],trials=[trial],yaml=root['yaml'],yaml_sha256=root['yaml_sha256'],
            source_sha256=root['source_sha256'],r7_physics_after_asset_overrides=root['r7_physics_after_asset_overrides'],
            root_manifest_sha256=sha(run/'manifest.json'))
        write(target/'manifest.json',manifest)
        with (run/'prepared-seeds.jsonl').open('a') as f:f.write(json.dumps(dict(seed=seed,manifest_sha256=sha(target/'manifest.json'),sampling_calls=1,filtered=False))+'\n')
        return target
