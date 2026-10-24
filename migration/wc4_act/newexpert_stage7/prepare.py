"""Freeze 42 unfiltered new samples and bind 8 byte-identical reused sources."""
import argparse,copy,json,math,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from newexpert_stage7.common import BASE,sha,read,write
from fullcycle_pilot import PHYS

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    assert read(a.preflight)['all_passed']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    oldrun=ROOT/'.repro/runs/wc4-newexpert-stage6-20261025';old=read(oldrun/'manifest.json');oldsha=read(oldrun/'SHA256.json')
    for name,h in oldsha.items():assert sha(oldrun/name)==h,'Prior evidence changed: '+name
    for name,h in old['source_sha256'].items():assert sha(ROOT/name)==h,'Baseline source changed: '+name
    for name in ['fullcycle_expert.py','fullcycle_compact.py','newexpert_stage2/push_geometry.py','newexpert_stage3/geometry.json']:
        n='migration/wc4_act/'+name
        assert subprocess.check_output(['git','show',BASE+':'+n],cwd=ROOT)==(ROOT/n).read_bytes()
    run.mkdir(parents=True,exist_ok=False)
    (run/'preflight.json').write_bytes(a.preflight.read_bytes())
    config=run/'frozen-r7-physics.yaml';config.write_bytes((oldrun/'frozen-r7-physics.yaml').read_bytes())
    geom=read(oldrun/'geometry.json');geom['execution_authorization']={'stage':7,'new_episodes':42,'reused_episodes':8,
        'maximum_sim_seconds':150,'video_encoding':False,'masked_depth_raw_retained':True,
        'basis':'User seventh-round body explicitly authorizes 8 reused +42 new =50 source episodes; title 32 is not applied'}
    write(run/'geometry.json',geom)
    reused=[]
    for i in range(8):
        src=oldrun/('trial_%02d'%i);required=['attempt_000/raw/episode_000000.npz','attempt_000/states.jsonl',
            'attempt_000/contacts.jsonl','attempt_000/lock-writes.jsonl','attempt_000/recording-clock.jsonl',
            'attempt_000/result.json','attempt_000/command.json','attempt_000/initial-state.json','attempt_000/frozen-trial.json',
            'attempt_000/physical-camera-binding.json','audit/record-validation.json','audit/r11-score.json','audit/summary.json']
        files={n:{'path':str(src/n),'sha256':sha(src/n),'bytes':(src/n).stat().st_size} for n in required}
        for n,f in files.items():assert f['sha256']==oldsha['trial_%02d/'%i+n]
        q=read(src/'audit/record-validation.json');s=read(src/'audit/r11-score.json')
        assert q['recording_checks_pass'] and s['measured_task_gates_pass']
        reused.append({'source_index':i,'source_id':'stage6-seed%d'%(62500+i),'seed':62500+i,'kind':'reused_stage6_original',
            'source_dir':str(src),'execution_commit':read(src/'attempt_000/command.json')['head'],
            'files':files,'frames':q['frames'],'automatic_terminal':read(src/'attempt_000/result.json')['automatic_final_conditions_met'],
            'complete_task_review':'stage6_joint_review_previously_accepted_by_user','recording_checks_pass':True,
            'physical_reexecution':False,'prior_full_manifest_sha256':sha(oldrun/'SHA256.json')})
    write(run/'reuse-index.json',{'sources':reused,'source_count':8,'physics_reexecutions':0,
        'prior_joint_review':{'path':str(oldrun/'visual-review.json'),'sha256':sha(oldrun/'visual-review.json')},
        'prior_stage6_files_verified':len(oldsha),'original_sha_manifest':str(oldrun/'SHA256.json')})
    sys.path.insert(0,str(ROOT/'high-level/float_ik'))
    from overview import load
    module=load('stage7_cpu_prepare','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    assets=yaml.safe_load((ROOT/'high-level/data/cfg/b1z1_opendoor.yaml').read_text())['env']['asset']['trainAssets']['opensource_leverdoor']
    idx,spec=next((int(i),s) for i,s in assets.items() if s.get('name')=='wc4')
    fake=SimpleNamespace(spec=spec,asset_index=idx,door_motion_sign_multiplier=spec['door_motion_sign_multiplier'],
        actor_scale=spec['actor_scale'],actor_yaw=math.pi/2,actor_position_offset=spec['actor_position_offset'],
        robot_y_offset=float(spec.get('robot_y_offset',0)),handle_bounding=read(ROOT/'high-level/data/asset/door_set/wc4/handle_bounding.json'))
    cli0=copy.deepcopy(old['trials'][0]['argv']);cli0[cli0.index('--a2w_float_ik_config')+1]=str(config)
    trials=[]
    for i,seed in enumerate(range(62700,62742)):
        cli=cli0.copy();cli[cli.index('--seed')+1]=str(seed)
        sys.argv=['stage7_prepare',*cli];args=module.parse_args();args.sim_dt=float(np.float32(.02))
        env=module.make_env_args(args,0);before=json.loads(env.ikpush_randomization_json);np.random.seed(seed)
        for name in ['wrist','front']:
            rotation=module.dc.wrist_camera_rotation_radians_from_args(env) if name=='wrist' else [math.radians(getattr(env,'front_camera_'+axis+'_deg')) for axis in ['yaw','pitch','roll']]
            module.cached_local_camera_transform_from_cfg(name,module.dc.camera_cfg_for_args(name,env),rotation,args=env)
        poses={k:{n:np.asarray(v).tolist() for n,v in pose.items()} for k,pose in env._camera_axis_local_poses.items()}
        module.dc.apply_door_runtime_overrides(env,fake)
        assert all(abs(float(getattr(env,k))-old['r7_physics_after_asset_overrides'][k])<1e-12 for k in PHYS)
        yaw,heading,base,stop=module.dc.compute_base_walk_targets(env,fake)
        trials.append({'index':i,'source_index':8+i,'seed':seed,'nominal':False,'argv':cli,
            'sampling_before_asset_overrides':before,'effective_sampling':json.loads(env.ikpush_randomization_json),
            'camera_local_poses':poses,'expected_spawn':{'base_start':base.tolist(),'robot_z':float(np.float32(env.robot_z)),
                'robot_pitch':env.robot_pitch,'yaw':yaw,'root_quat':module.dc.robot_base_quat_np(env.robot_pitch,yaw).tolist()}})
    assert len({t['seed'] for t in trials})==42 and not ({t['seed'] for t in trials}&set(old['seeds']))
    sources=dict(old['source_sha256'])
    for folder in ['newexpert_stage6','newexpert_stage7']:
        for path in (ROOT/'migration/wc4_act'/folder).glob('*'):
            if path.is_file():sources[str(path.relative_to(ROOT))]=sha(path)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    m={'start_commit':head,'handoff_base':BASE,'maximum_scenes':42,'reused_sources':8,'total_source_slots':50,
        'maximum_sim_seconds':150,'seeds':[t['seed'] for t in trials],'trials':trials,
        'r7_physics_after_asset_overrides':old['r7_physics_after_asset_overrides'],'yaml':str(config),'yaml_sha256':sha(config),
        'source_sha256':sources,'selection':'Consecutive 62700..62741 selected before sampling, exactly one CPU sample per seed; no filtering or replacement',
        'recording':'No video; original front/wrist masked-depth uint8 buffers and Joint9 live NPZ retained'}
    write(run/'manifest.json',m)
    ranges={'robot_x':[3.4,4.1],'robot_y':[-.1,.04],'robot_z':[.47,.53],'robot_pitch':[0,math.radians(5)],
        'robot_yaw':[math.pi-.03,math.pi+.03],'door_x':[2.47,2.53],'door_y':[-.03,.03],'door_wall_x_offset':[-.03,.03]}
    coverage={k:{'existing_range':v,'sampled_min':min(t['effective_sampling'][k] for t in trials),
        'sampled_max':max(t['effective_sampling'][k] for t in trials)} for k,v in ranges.items()}
    for k,v in coverage.items():assert ranges[k][0]<=v['sampled_min']<=v['sampled_max']<=ranges[k][1]
    write(run/'initial-coverage.json',{'axes':coverage,'trials':[{'index':t['index'],'seed':t['seed'],
        'expected_spawn':t['expected_spawn'],'sampled':{k:t['effective_sampling'][k] for k in ranges}} for t in trials]})
    bindings={}
    for trial in trials:
        dest=run/('trial_%02d'%trial['index']);dest.mkdir();(dest/'geometry.json').write_bytes((run/'geometry.json').read_bytes())
        mini=dict(m,trials=[trial],maximum_scenes=1,global_trial_index=trial['index'],local_install_slot=0)
        write(dest/'manifest.json',mini)
        bindings[str(trial['index'])]={'manifest_sha256':sha(dest/'manifest.json'),'geometry_sha256':sha(dest/'geometry.json')}
    write(run/'trial-bindings.json',bindings)
    # Estimate from every original stage-six completed step; retain original 40 GiB reserve, never delete history.
    per_step=max(sum((Path(x['source_dir'])/'attempt_000'/n).stat().st_size for n in ['states.jsonl','contacts.jsonl','recording-clock.jsonl','lock-writes.jsonl'])/read(Path(x['source_dir'])/'attempt_000/result.json')['steps'] for x in reused)
    raw_per_frame=max(x['files']['attempt_000/raw/episode_000000.npz']['bytes']/x['frames'] for x in reused)
    import shutil
    estimated=math.ceil(1.2*(42*7500*per_step+42*3750*raw_per_frame));free=shutil.disk_usage(ROOT).free
    write(run/'capacity-plan.json',{'free_bytes':free,'estimated_new_full_raw_bytes_20percent_margin':estimated,
        'required_headroom_bytes':40*1024**3,'fits_estimate_and_headroom':free>=estimated+40*1024**3,
        'note':'Viability estimate, not a file quota. All attempts retained; ongoing 40GiB guard remains.'})
    assert free>=estimated+40*1024**3,'Capacity cannot safely retain all 42 attempts plus 40GiB reserve'
    names=['manifest.json','geometry.json','initial-coverage.json','reuse-index.json','trial-bindings.json','preflight.json']
    write(run/'preparation-receipt.json',{'head':head,'scene_creations':0,'new_seed_samples':42,'reused_physics_runs':0,
        'file_sha256':{n:sha(run/n) for n in names}})
    print(json.dumps({'head':head,'new_seeds':m['seeds'],'reused':8,'frozen_new':42,'capacity_estimate_bytes':estimated,'free_bytes':free}))
if __name__=='__main__':main()
