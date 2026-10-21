"""R9 collection orchestration only. The R8 execution modules stay untouched."""
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SUPPORT = ROOT / 'migration/wc4_act'
sys.path.insert(0, str(SUPPORT))
OLD = ROOT / '.repro/runs/wc4-fulltask-pilot-r8-20261021'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def prepare(run):
    import numpy as np
    import yaml
    from overview import load
    from fullcycle_pilot import PHYS
    from fullcycle_scene import SceneFrame
    if run.exists():
        raise FileExistsError('New run only; never regenerate a frozen run')
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    free = shutil.disk_usage(ROOT).free
    assert free >= 140 * 1024**3, 'Insufficient reserved capacity; do not clean history'
    run.mkdir(parents=True)
    write(run/'capacity-reservation.json', dict(free_bytes=free, planned_capacity_bytes=140*1024**3,
        basis='R8 full evidence rate scaled to 64 x 180s, plus conversion/temp/headroom; logical reservation, no deletion',
        disk_snapshot=subprocess.check_output(['df', '-h', str(ROOT)], text=True)))
    old = read(OLD/'manifest.json')
    source = read(OLD/'consistency.json')['source_sha256'].copy()
    for name, digest in source.items():
        assert sha(ROOT/name) == digest, 'Frozen R8 source changed: '+name
    for p in Path(__file__).parent.iterdir():
        if p.is_file(): source[str(p.relative_to(ROOT))] = sha(p)
    for name in ['geometry.json', 'frozen-r7-physics.yaml']:
        shutil.copyfile(OLD/name, run/name)
    expected = old['r7_physics_after_asset_overrides']
    cfg = yaml.safe_load((run/'frozen-r7-physics.yaml').read_text())
    assert cfg['randomization_absolute_ranges'] == old['physical_absolute_ranges']
    seed_list = list(range(62100, 62164))
    assert not set(seed_list) & set(old['seeds'])
    sys.path.insert(0, str(ROOT/'high-level/float_ik'))
    m = load('r9_prepare_author', 'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    dc = yaml.safe_load((ROOT/'high-level/data/cfg/b1z1_opendoor.yaml').read_text())
    idx, spec = next((int(i), s) for i, s in dc['env']['asset']['trainAssets']['opensource_leverdoor'].items() if s.get('name') == 'wc4')
    trials = []
    for i, seed in enumerate(seed_list):
        cli = old['trials'][1]['argv'].copy()  # R8 randomized trial, never nominal.
        for key, val in [('--seed', seed), ('--a2w_float_ik_config', run/'frozen-r7-physics.yaml')]:
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
    master = dict(start_commit='b8dda1939c6902423fbcd8ddd941085e3908634e', execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        maximum_scenes=64, maximum_sim_seconds=180, target_candidates=50, retained_r8_trials=[0,1,2,3,5,6], permanently_withheld_r8_trials=[4,7],
        seeds=seed_list, trials=trials, source_sha256=source, yaml=str(run/'frozen-r7-physics.yaml'), yaml_sha256=sha(run/'frozen-r7-physics.yaml'),
        r7_physics_after_asset_overrides=expected, physical_absolute_ranges=old['physical_absolute_ranges'],
        geometry_sha256=sha(run/'geometry.json'), all_new_trials_randomized=True)
    write(run/'manifest.json', master)
    write(run/'consistency.json', dict(status='source_route_consistent', source_sha256=source, geometry_sha256=master['geometry_sha256']))
    for i, trial in enumerate(trials):
        p = run/'frozen'/('%03d'%i); p.mkdir(parents=True)
        one = master.copy(); one.update(trials=[trial], maximum_scenes=1, seeds=[trial['seed']])
        write(p/'manifest.json', one)
    write(run/'frozen-files.json', {str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file()})
    (run/'claims').mkdir(); (run/'candidates').mkdir()
    oldindex = read(OLD/'candidate-index.json')
    episodes = []
    for e, item in enumerate(oldindex['episodes']):
        src = Path(item['candidate_npz']); dest = run/'candidates'/('episode_%06d.npz'%e)
        assert sha(src) == item['sha256'], 'R8 retained candidate digest changed'
        shutil.copyfile(src, dest); assert sha(src) == sha(dest)
        copy = item.copy()
        copy.update(episode_index=e, source_run=str(OLD), original_npz=item['source_npz'], source_npz=str(src), candidate_npz=str(dest), sha256=sha(dest), provenance='R8 retained byte-for-byte')
        episodes.append(copy)
    for p in (OLD/'candidates').iterdir():
        if p.suffix != '.npz' and p.is_file(): shutil.copyfile(p, run/'candidates'/p.name)
    write(run/'candidate-index.json', dict(episodes=episodes, target=50, retained_r8=6, r8_withheld=[4,7]))
    print('Frozen 64 randomized seeds and parameters; retained six original candidates; no scenes created.', flush=True)


def verify(run):
    for name, digest in read(run/'frozen-files.json').items():
        assert sha(run/name) == digest, 'Frozen manifest/config changed: '+name
    for name, digest in read(run/'manifest.json')['source_sha256'].items():
        assert sha(ROOT/name) == digest, 'Source changed: '+name


def launch(run, index, mode):
    verify(run)
    m = read(run/'manifest.json')
    assert 0 <= index < 64
    if mode == 'attempt':
        assert not (run/'SYSTEMIC_STOP.json').exists(), 'Systemic stop remains active'
        assert len(read(run/'candidate-index.json')['episodes']) < 50
        assert len(list((run/'claims').iterdir())) < 64
        assert shutil.disk_usage(ROOT).free > 20*1024**3, 'Storage headroom stop; never delete history'
        with (run/'claims'/('%03d.json'%index)).open('x') as f:
            json.dump(dict(index=index,seed=m['seeds'][index],wall_time=time.time(),meaning='Invocation consumed; scene receipt determines whether creation began'), f)
    name = ('trial' if mode=='attempt' else 'preflight')+'_%02d'%index
    output = run/name
    assert not output.exists(), 'No replacement or retry'
    command = [sys.executable,'-B',str(SUPPORT/'fullcycle_expert.py'),'--output',str(output),
        '--geometry',str(run/'geometry.json'),'--validation',str(run/'consistency.json'),
        '--pilot-manifest',str(run/'frozen'/('%03d'%index)/'manifest.json'),'--trial-index','0']
    if mode == 'preflight': command.append('--preflight')
    command += m['trials'][index]['argv']
    write(run/(name+'-command.json'),dict(argv=command,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),index=index,mode=mode,start_time=time.time()))
    with (run/(name+'.log')).open('x') as f:
        result = subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    write(run/(name+'-exit.json'),dict(returncode=result.returncode,end_time=time.time()))
    if result.returncode:
        write(run/'SYSTEMIC_STOP.json',dict(index=index,reason='execution process failed; inspect before any further attempt',returncode=result.returncode))
        raise SystemExit(result.returncode)
    print(name, 'complete', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--mode',choices=['prepare','preflight','attempt'],required=True);p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.mode=='prepare': prepare(a.run)
    else: launch(a.run,a.index,a.mode)
