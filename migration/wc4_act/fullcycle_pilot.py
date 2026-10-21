"""Frozen eight-trial pilot configuration; no simulation in prepare()."""
import argparse,copy,hashlib,json,math,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
PHYS=['door_joint_friction','door_joint_damping','handle_joint_friction','handle_joint_damping',
      'handle_spring_stiffness','handle_spring_damping','door_open_resistance','door_open_damping',
      'door_auto_open_force','door_lock_force','handle_unlock_ratio','door_auto_open_sign','door_auto_open_target_ratio']
RANDOM_PHYS=PHYS[:6]


def install_frozen_trial(m,manifest_path,index,output):
    manifest=json.loads(manifest_path.read_text())
    if index is None or not 0<=index<len(manifest['trials']) or len(manifest['trials'])>8:
        raise ValueError('Invalid frozen pilot trial / budget')
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Pilot source changed: '+name)
    if hashlib.sha256(Path(manifest['yaml']).read_bytes()).hexdigest()!=manifest['yaml_sha256']:
        raise RuntimeError('Pilot effective YAML changed')
    trial=manifest['trials'][index];original=m.make_env_args
    def make(args,env_index):
        env=original(args,env_index)
        actual=json.loads(env.ikpush_randomization_json)
        if actual!=trial['sampling_before_asset_overrides']:
            raise RuntimeError('systemic_frozen_initial_state_mismatch')
        env._camera_axis_local_poses={k:{n:np.array(v,dtype=np.float32) for n,v in pose.items()}
            for k,pose in trial['camera_local_poses'].items()}
        env._pilot_trial=trial;env._pilot_physics=manifest['r7_physics_after_asset_overrides']
        return env
    m.make_env_args=make
    (output/'frozen-trial.json').write_text(json.dumps(trial,indent=2)+'\n')


def audit_effective_trial(gym,st,output):
    if not hasattr(st.args,'_pilot_trial'):return
    expected=st.args._pilot_physics
    values={k:float(getattr(st.args,k)) for k in PHYS}
    if any(abs(values[k]-expected[k])>1e-12 for k in PHYS):raise RuntimeError('systemic_physics_not_frozen_to_r7')
    if st.args.enable_depth_noise or st.args.enable_depth_gaussian_blur:raise RuntimeError('systemic_depth_augmentation_enabled')
    frozen=st.args._pilot_trial['camera_local_poses']
    actual={k:{n:np.asarray(v).tolist() for n,v in pose.items()} for k,pose in st.args._camera_axis_local_poses.items()}
    if actual!=frozen:raise RuntimeError('systemic_camera_poses_not_frozen')
    props=gym.get_actor_dof_properties(st.env,st.door_actor)
    door_props={n:props[n].tolist() for n in props.dtype.names}
    (output/'physical-camera-binding.json').write_text(json.dumps(dict(physics=values,r7_physics=expected,
        all_physics_match=True,door_dof_properties=door_props,camera_local_poses=actual,
        randomization=json.loads(st.args.ikpush_randomization_json)),indent=2)+'\n')


def prepare(run):
    from overview import load
    run=Path(run)
    if (run/'manifest.json').exists() or (run/'scene-claims').exists():raise RuntimeError('Frozen/live pilot cannot be regenerated')
    run.mkdir(exist_ok=True,parents=True)
    r7=ROOT/'.repro/runs/wc4-fullcycle-compact-r7-20261021/attempt_000'
    effective=json.loads((r7/'effective-config.json').read_text())
    base=yaml.safe_load((ROOT/'high-level/float_ik/config/a2w_float_ik_push_door_parallel.yaml').read_text())
    door_cfg=yaml.safe_load((ROOT/'high-level/data/cfg/b1z1_opendoor.yaml').read_text())
    specs=door_cfg['env']['asset']['trainAssets']['opensource_leverdoor']
    idx,spec=next((int(i),s) for i,s in specs.items() if s.get('name')=='wc4')
    expected={k:float(effective[k]) for k in PHYS}
    pre={k:expected[k]/float(spec.get('controller_multipliers',{}).get(k,1)) for k in PHYS}
    base['door_and_handle_physics'].update(pre)
    # Replace absolute physical ranges, not merely the half-width flags.
    base['randomization_absolute_ranges']={k:[pre[k],pre[k]] for k in RANDOM_PHYS}
    for k in RANDOM_PHYS:base['randomization']['ikpush_'+k+'_rand']=0.
    base['randomization'].update(ikpush_door_open_resistance_prob=0.,ikpush_door_open_resistance_min=0.,ikpush_door_open_resistance_max=0.)
    config=run/'frozen-r7-physics.yaml';config.write_text(yaml.safe_dump(base,sort_keys=False))
    geom=json.loads((ROOT/'migration/wc4_act/fullcycle_compact.json').read_text());geom['live_joint9_recording']=True
    (run/'geometry.json').write_text(json.dumps(geom,indent=2)+'\n')
    original=json.loads((r7/'command.json').read_text())['argv'][1:]
    remove=['--no_ikpush_env_randomization','--no_enable_depth_camera_randomization','--enable_depth_camera_randomization']
    original=[x for x in original if x not in remove]
    def put(a,k,value):
        if k in a:a[a.index(k)+1]=str(value)
        else:a.extend([k,str(value)])
    put(original,'--a2w_float_ik_config',config)
    for k,value in pre.items():put(original,'--'+k,value)
    original+=['--no_keyframe_loss_weights']
    sys.path.insert(0,str(ROOT/'high-level/float_ik'))
    m=load('pilot_prepare_author','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    trials=[];feasibility=[]
    from fullcycle_geometry import URDF
    from fullcycle_scene import SceneFrame
    from fullcycle_compact import Arm,ClosingCurve
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');arm=Arm(robot);curve=ClosingCurve(geom['compact_expert'])
    for i,seed in enumerate(range(62000,62008)):
        cli=original.copy();put(cli,'--seed',seed)
        cli+=['--no_ikpush_env_randomization','--no_enable_depth_camera_randomization'] if i==0 else ['--enable_depth_camera_randomization']
        put(cli,'--depth_camera_pos_rand_m',.02);put(cli,'--depth_camera_rot_rand_deg',5.)
        sys.argv=['pilot_prepare',*cli];args=m.parse_args();args.sim_dt=float(np.float32(.02))
        env=m.make_env_args(args,0);before=json.loads(env.ikpush_randomization_json)
        np.random.seed(seed)
        jitters=[];old=m.dc.jitter_camera_pose_for_args
        def capture(pos,rot,args):
            p,r=old(pos,rot,args);jitters.append(dict(position_delta=(np.asarray(p)-pos).tolist(),rotation_delta_rad=(np.asarray(r)-rot).tolist()));return p,r
        m.dc.jitter_camera_pose_for_args=capture
        for name in ['wrist','front']:
            rot=m.dc.wrist_camera_rotation_radians_from_args(env) if name=='wrist' else [math.radians(getattr(env,'front_camera_'+axis+'_deg')) for axis in ['yaw','pitch','roll']]
            m.cached_local_camera_transform_from_cfg(name,m.dc.camera_cfg_for_args(name,env),rot,args=env)
        m.dc.jitter_camera_pose_for_args=old
        poses={k:{n:np.asarray(v).tolist() for n,v in p.items()} for k,p in env._camera_axis_local_poses.items()}
        fake=SimpleNamespace(spec=spec,asset_index=idx,door_motion_sign_multiplier=spec['door_motion_sign_multiplier'],actor_scale=spec['actor_scale'],actor_yaw=math.pi/2,actor_position_offset=spec['actor_position_offset'])
        m.dc.apply_door_runtime_overrides(env,fake)
        if any(abs(getattr(env,k)-expected[k])>1e-12 for k in PHYS):raise RuntimeError('Offline physical binding mismatch')
        frame=SceneFrame.from_runtime(m,SimpleNamespace(args=env,door=fake))
        q=np.array([0.,1.7,-1.3,-.4,0.,math.pi/2]);errors=[]
        for theta in np.linspace(math.pi/2,0,31):
            ref=curve.at(theta);q,err=arm.solve(np.linalg.inv(frame.base_transform(ref[:3]))@curve.tool(theta),q,1.,120);errors.append(err)
        feasible=max(e['position_error_m'] for e in errors)<.015 and max(e['orientation_error_rad'] for e in errors)<.08
        feasibility.append(dict(trial=i,feasible=feasible,max_position_error=max(e['position_error_m'] for e in errors),max_orientation_error=max(e['orientation_error_rad'] for e in errors)))
        trials.append(dict(index=i,seed=seed,nominal=i==0,argv=cli,sampling_before_asset_overrides=before,
            effective_sampling=json.loads(env.ikpush_randomization_json),camera_local_poses=poses,camera_jitter=jitters,
            scene_frame=frame.metadata()))
    (run/'offline-reachability.json').write_text(json.dumps(feasibility,indent=2)+'\n')
    files=[p for p in (ROOT/'migration/wc4_act').glob('*.py')]+[ROOT/'migration/wc4_act/fullcycle_compact.json']
    manifest=dict(start_commit='664aa7c41893967f1f839f85273fa0cf73671177',maximum_scenes=8,maximum_sim_seconds=180,
        seeds=[t['seed'] for t in trials],trials=trials,r7_physics_after_asset_overrides=expected,
        yaml=str(config),yaml_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        physical_absolute_ranges=base['randomization_absolute_ranges'],all_offline_reachability_pass=all(x['feasible'] for x in feasibility),
        cameras='Author independent uniform +/-0.02m and +/-5deg per axis, frozen before scenes; nominal first trial unperturbed',
        policy_inputs=['state9','front_masked_depth','wrist_masked_depth'],new_policy_inputs=[])
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    receipt=dict(status='source_route_consistent',geometry_sha256=hashlib.sha256((run/'geometry.json').read_bytes()).hexdigest(),source_sha256=manifest['source_sha256'])
    (run/'consistency.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),seeds=manifest['seeds'],feasibility=feasibility),indent=2))
    if not manifest['all_offline_reachability_pass']:raise RuntimeError('STOP: existing R7 poses incompatible with full confirmed distribution')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare');p.add_argument('--seal');a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    elif a.seal:
        run=Path(a.seal)
        if (run/'scene-claims').exists():raise RuntimeError('No source reseal after the first scene claim')
        path=run/'manifest.json';manifest=json.loads(path.read_text())
        fields=['seeds','trials','r7_physics_after_asset_overrides','yaml_sha256','physical_absolute_ranges']
        parameters={k:manifest[k] for k in fields};frozen=run/'frozen-parameters.json'
        if frozen.exists():
            if json.loads(frozen.read_text())!=parameters:raise RuntimeError('Frozen trial parameters changed')
        else:frozen.write_text(json.dumps(parameters,indent=2)+'\n')
        if not (run/'manifest-before-source-seal.json').exists():
            (run/'manifest-before-source-seal.json').write_bytes(path.read_bytes())
        old=json.loads((ROOT/'.repro/reviews/wc4-r7-design-20261021/consistency.json').read_text())
        files=[ROOT/n for n in old['source_sha256']]
        files=list(set(files+list((ROOT/'migration/wc4_act').glob('*.py'))+list((ROOT/'migration/wc4_act').glob('*.sh'))))
        manifest['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        receipt=json.loads((run/'consistency.json').read_text());receipt['source_sha256']=manifest['source_sha256']
        (run/'consistency.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('Source sealed without resampling or changing frozen trial parameters; no scenes created.')
    else:p.error('Choose --prepare or --seal')
