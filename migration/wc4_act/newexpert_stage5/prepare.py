"""Freeze eight consecutive seeds before any scene; preserve author sampling."""
import argparse, copy, hashlib, json, math, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from newexpert_stage4.prepare import sha, write
from fullcycle_pilot import PHYS

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    run.mkdir(parents=True,exist_ok=False)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    previous=ROOT/'.repro/runs/wc4-newexpert-stage4-20261025'
    old=json.loads((previous/'manifest.json').read_text())
    config=run/'frozen-r7-physics.yaml';config.write_bytes((previous/'frozen-r7-physics.yaml').read_bytes())
    geom=json.loads((previous/'geometry.json').read_text())
    geom['execution_authorization']={'stage':5,'episodes':8,'maximum_sim_seconds':150,'dual_depth_video_encoding':False,
        'basis':'User fifth-round eight perturbed-initial-state task; no expert/physics changes'}
    write(run/'geometry.json',geom)
    sys.path.insert(0,str(ROOT/'high-level/float_ik'))
    from overview import load
    m=load('stage5_cpu_prepare','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    assets=yaml.safe_load((ROOT/'high-level/data/cfg/b1z1_opendoor.yaml').read_text())['env']['asset']['trainAssets']['opensource_leverdoor']
    idx,spec=next((int(i),s) for i,s in assets.items() if s.get('name')=='wc4')
    fake=SimpleNamespace(spec=spec,asset_index=idx,door_motion_sign_multiplier=spec['door_motion_sign_multiplier'],
        actor_scale=spec['actor_scale'],actor_yaw=math.pi/2,actor_position_offset=spec['actor_position_offset'],
        handle_bounding=json.loads((ROOT/'high-level/data/asset/door_set/wc4/handle_bounding.json').read_text()))
    cli0=copy.deepcopy(old['trials'][0]['argv'])
    cli0=[x for x in cli0 if x not in ['--no_ikpush_env_randomization','--no_enable_depth_camera_randomization']]
    cli0+=['--enable_depth_camera_randomization']
    cli0[cli0.index('--a2w_float_ik_config')+1]=str(config)
    trials=[]
    for i,seed in enumerate(range(62500,62508)):
        cli=cli0.copy();cli[cli.index('--seed')+1]=str(seed)
        sys.argv=['stage5_prepare',*cli];args=m.parse_args();args.sim_dt=float(np.float32(.02))
        env=m.make_env_args(args,0);before=json.loads(env.ikpush_randomization_json)
        np.random.seed(seed)
        for name in ['wrist','front']:
            rotation=m.dc.wrist_camera_rotation_radians_from_args(env) if name=='wrist' else [math.radians(getattr(env,'front_camera_'+axis+'_deg')) for axis in ['yaw','pitch','roll']]
            m.cached_local_camera_transform_from_cfg(name,m.dc.camera_cfg_for_args(name,env),rotation,args=env)
        poses={k:{n:np.asarray(v).tolist() for n,v in pose.items()} for k,pose in env._camera_axis_local_poses.items()}
        m.dc.apply_door_runtime_overrides(env,fake)
        assert all(abs(float(getattr(env,k))-old['r7_physics_after_asset_overrides'][k])<1e-12 for k in PHYS)
        yaw,heading,base,stop=m.dc.compute_base_walk_targets(env,fake)
        trial={'index':i,'seed':seed,'nominal':False,'argv':cli,'sampling_before_asset_overrides':before,
            'effective_sampling':json.loads(env.ikpush_randomization_json),'camera_local_poses':poses,
            'expected_spawn':{'base_start':base.tolist(),'robot_z':float(np.float32(env.robot_z)),
                'robot_pitch':env.robot_pitch,'yaw':yaw,'root_quat':m.dc.robot_base_quat_np(env.robot_pitch,yaw).tolist()}}
        trials.append(trial)
    sources=dict(old['source_sha256'])
    for folder in ['newexpert_stage4','newexpert_stage5']:
        for path in (ROOT/'migration/wc4_act'/folder).glob('*'):
            if path.is_file():sources[str(path.relative_to(ROOT))]=sha(path)
    assert all(sha(ROOT/n)==h for n,h in sources.items())
    for n in ['fullcycle_expert.py','fullcycle_compact.py','newexpert_stage2/push_geometry.py','newexpert_stage3/geometry.json']:
        name='migration/wc4_act/'+n
        assert subprocess.check_output(['git','show','671020b97dde6222f115edb2db3a9ee12c033b22:'+name],cwd=ROOT)==(ROOT/name).read_bytes()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest={'start_commit':head,'maximum_scenes':8,'maximum_sim_seconds':150,'seeds':[t['seed'] for t in trials],
        'trials':trials,'r7_physics_after_asset_overrides':old['r7_physics_after_asset_overrides'],
        'yaml':str(config),'yaml_sha256':sha(config),'source_sha256':sources,'prior_stage4_head':'671020b97dde6222f115edb2db3a9ee12c033b22',
        'selection':'62500..62507 chosen consecutively before sampling; no seed screening, filtering or replacement',
        'depth_encoding_disabled_only':True,'live_joint9_raw_recording':True}
    write(run/'manifest.json',manifest)
    ranges={'robot_x':[3.4,4.1],'robot_y':[-.10,.04],'robot_z':[.47,.53],'robot_pitch':[0,math.radians(5)],
        'robot_yaw':[math.pi-.03,math.pi+.03],'door_x':[2.47,2.53],'door_y':[-.03,.03],'door_wall_x_offset':[-.03,.03]}
    coverage={k:{'existing_range':r,'sampled_min':min(t['effective_sampling'][k] for t in trials),
                 'sampled_max':max(t['effective_sampling'][k] for t in trials)} for k,r in ranges.items()}
    for k,c in coverage.items():assert ranges[k][0]<=c['sampled_min']<=c['sampled_max']<=ranges[k][1]
    write(run/'initial-coverage.json',{'axes':coverage,'trials':[{'index':t['index'],'seed':t['seed'],
        'sampled':{k:t['effective_sampling'][k] for k in ranges},'expected_spawn':t['expected_spawn']} for t in trials],
        'limitations':'Eight independent uniform samples are not corner coverage or a distribution success-rate estimate. Sampled robot_y is an offset; actual y includes door/handle alignment. Each actual pre-physics spawn is checked against this frozen list.'})
    for i in range(8):
        dest=run/('trial_%02d'%i);dest.mkdir();(dest/'geometry.json').write_bytes((run/'geometry.json').read_bytes())
    write(run/'preparation-receipt.json',{'scene_creations':0,'head':head,'manifest_sha256':sha(run/'manifest.json'),
        'geometry_sha256':sha(run/'geometry.json'),'initial_coverage_sha256':sha(run/'initial-coverage.json')})
    print(json.dumps({'head':head,'seeds':manifest['seeds'],'coverage':coverage},indent=2))

if __name__=='__main__':main()
