"""Freeze all 72 distinct initial conditions before any model or scene execution."""
import sys,subprocess,shutil,json,math
from pathlib import Path
from types import SimpleNamespace
from common import *
sys.path.insert(0,str(SUPPORT))
R8=ROOT/'.repro/runs/wc4-fulltask-pilot-r8-20261021'

def prepare():
    import numpy as np,yaml
    from overview import load
    from fullcycle_pilot import PHYS
    from fullcycle_scene import SceneFrame
    assert not RUN.exists(),'New run only'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
    assert shutil.disk_usage(ROOT).free>=180*1024**3,'Reserve evidence/model headroom without deleting history'
    run=RUN;run.mkdir(parents=True)
    write(run/'capacity-reservation.json',dict(free_bytes=shutil.disk_usage(ROOT).free,
        reserved_budget_bytes=180*1024**3,basis='80 x 180s full state/contact journals, temporary RGB, three checkpoints and exports; logical reserve'))
    old = read(R8/'manifest.json')
    source = read(R8/'consistency.json')['source_sha256'].copy()
    for name, digest in source.items():
        assert sha(ROOT/name) == digest, 'Frozen R8 source changed: '+name
    for p in (SUPPORT/'r12').glob('*.py'):source[str(p.relative_to(ROOT))]=sha(p)
    source['migration/wc4_act/r11/criteria.py']=sha(SUPPORT/'r11/criteria.py')
    for name in ['geometry.json', 'frozen-r7-physics.yaml']:
        shutil.copyfile(R8/name, run/name)
    expected = old['r7_physics_after_asset_overrides']
    cfg = yaml.safe_load((run/'frozen-r7-physics.yaml').read_text())
    assert cfg['randomization_absolute_ranges'] == old['physical_absolute_ranges']
    seed_list = list(range(62300,62308))+list(range(62200,62264))
    r9=read(ROOT/'.repro/runs/wc4-fulltask-expand-r9-20261021/manifest.json')
    assert not set(seed_list)&set(old['seeds']+r9['seeds'])
    assert len(seed_list)==len(set(seed_list))
    sys.path.insert(0, str(ROOT/'high-level/float_ik'))
    m = load('r12_prepare_author', 'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    dc = yaml.safe_load((ROOT/'high-level/data/cfg/b1z1_opendoor.yaml').read_text())
    idx, spec = next((int(i), s) for i, s in dc['env']['asset']['trainAssets']['opensource_leverdoor'].items() if s.get('name') == 'wc4')
    trials = []
    for i, seed in enumerate(seed_list):
        cli = old['trials'][1]['argv'].copy()  # R8 randomized trial, never nominal.
        for key, val in [('--seed', seed), ('--a2w_float_ik_config', run/'frozen-r7-physics.yaml'),('--steps',9000)]:
            cli[cli.index(key)+1] = str(val)
        assert '--no_ikpush_env_randomization' not in cli and '--enable_depth_camera_randomization' in cli
        sys.argv = ['r9_prepare', *cli]
        args = m.parse_args(); args.sim_dt = float(np.float32(.02))
        env = m.make_env_args(args, 0)
        before = json.loads(env.ikpush_randomization_json)
        np.random.seed(seed)
        jitter = []; original = m.dc.jitter_camera_pose_for_args
        def capture(pos, rot, args):
            p, r = original(pos, rot, args)
            jitter.append(dict(position_delta=(np.asarray(p)-pos).tolist(), rotation_delta_rad=(np.asarray(r)-rot).tolist()))
            return p, r
        m.dc.jitter_camera_pose_for_args = capture
        try:
            for name in ['wrist', 'front']:
                rot = m.dc.wrist_camera_rotation_radians_from_args(env) if name == 'wrist' else [math.radians(getattr(env, 'front_camera_'+axis+'_deg')) for axis in ['yaw','pitch','roll']]
                m.cached_local_camera_transform_from_cfg(name, m.dc.camera_cfg_for_args(name, env), rot, args=env)
        finally:
            m.dc.jitter_camera_pose_for_args = original
        poses = {k:{n:np.asarray(v).tolist() for n,v in p.items()} for k,p in env._camera_axis_local_poses.items()}
        fake = SimpleNamespace(spec=spec, asset_index=idx, door_motion_sign_multiplier=spec['door_motion_sign_multiplier'], actor_scale=spec['actor_scale'], actor_yaw=math.pi/2, actor_position_offset=spec['actor_position_offset'])
        m.dc.apply_door_runtime_overrides(env, fake)
        assert all(abs(getattr(env,k)-expected[k])<1e-12 for k in PHYS)
        assert not env.enable_depth_noise and not env.enable_depth_gaussian_blur
        frame = SceneFrame.from_runtime(m, SimpleNamespace(args=env, door=fake))
        trials.append(dict(index=i, seed=seed, nominal=False, argv=cli, sampling_before_asset_overrides=before,
            effective_sampling=json.loads(env.ikpush_randomization_json), camera_local_poses=poses, camera_jitter=jitter, scene_frame=frame.metadata()))
    write(run/'source-binding.json',source)
    master=dict(trials=trials,seeds=seed_list,yaml=str(run/'frozen-r7-physics.yaml'),
        yaml_sha256=sha(run/'frozen-r7-physics.yaml'),source_sha256=source,
        r7_physics_after_asset_overrides=expected,physical_absolute_ranges=old['physical_absolute_ranges'])
    write(run/'manifest.json',master)
    for i,trial in enumerate(trials):
        p=run/'frozen'/str(trial['seed']);p.mkdir(parents=True)
        one=master.copy();one.update(trials=[trial],seeds=[trial['seed']])
        write(p/'manifest.json',one)
    write(run/'contract.json',dict(training_seed=1000,optimizer_updates=30000,
        intermediate_seeds=seed_list[:8],final_seeds=seed_list[8:],
        evaluation_rounds={'010000':8,'020000':8,'030000':64},total_round_budget=80,
        data_frames=159529,data_episodes=72,chunk_size=100,train_n_action_steps=50,
        execute_actions=25,fps=25,batch_size=16,lr=1e-5,
        initial_observation='physical t=0; next observation t=0.04 before next two held physics substeps',
        joint_target_limits='physical actor DOF bounds; raw/applied/delta logged; no IK soft clamp or gripper latch',
        latch='after observed opening, actual angle <=1deg and speed <=1deg/s held >=0.5s; capture last actual angle',
        stop='no automatic resume/retry; 30k updates and 80 invocations maximum'))
    write(run/'dataset-sha256.json',{str(p.relative_to(DATA)):sha(p) for p in sorted(DATA.rglob('*')) if p.is_file()})
    write(run/'collection-disjoint.json',dict(training_seeds=old['seeds']+r9['seeds'],
        evaluation_seeds=seed_list,seed_sets_disjoint=True,
        actual_parameter_fingerprints={str(t['seed']):__import__('hashlib').sha256(json.dumps([t['sampling_before_asset_overrides'],t['camera_local_poses']],sort_keys=True).encode()).hexdigest() for t in trials}))
    # Compare full randomized initial/camera tuples, not only seed numbers.
    keys=lambda t:json.dumps([t['sampling_before_asset_overrides'],t['camera_local_poses']],sort_keys=True)
    assert len(set(map(keys,trials)))==72
    assert not set(map(keys,trials))&set(map(keys,old['trials']+r9['trials']))
    write(run/'frozen-files.json',{str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='source-binding.json'})
    write(run/'budget.json',dict(updates=0,evaluation_invocations=0,status='prepared-no-scenes'))
    print('Frozen 8 intermediate and 64 final initial/camera states; no simulation or model constructed.')

if __name__=='__main__':prepare()
