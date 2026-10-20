"""Exactly one initialized scene, property reads only, NEVER a dynamics run.

Reuses frozen author loaders and nominal initialization. No author main, IK
setup, expert controller, policy, replay, camera or loop is called. A persistent
creation receipt prevents another scene; a proxy forbids stepping throughout
and all scene writes after initialization. Raw metadata is not a certificate
of cooked collision geometry or of a successful task.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
GEOMETRY_SHA='872dd74761d51bd241caa89682ccff285321a70ded53d3bc0988cd80fd16be47'
FORBIDDEN={'simulate','fetch_results','step_graphics','render_all_camera_sensors',
           'set_rigid_transform','set_actor_root_state_tensor','set_actor_root_state_tensor_indexed',
           'apply_actor_dof_efforts','apply_rigid_body_force_tensors','apply_rigid_body_force_at_pos_tensors'}


def value(obj,depth=0):
    if obj is None or isinstance(obj,(str,bool,int,float)):return obj
    if isinstance(obj,np.void):return {n:value(obj[n],depth+1) for n in obj.dtype.names}
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,np.ndarray):
        return [value(x,depth+1) for x in obj] if obj.dtype.names else obj.tolist()
    if isinstance(obj,dict):return {str(k):value(v,depth+1) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [value(x,depth+1) for x in obj]
    if depth>6:return repr(obj)
    fields=[n for n in dir(obj) if not n.startswith('_')]
    result={}
    for n in fields:
        try:
            x=getattr(obj,n)
            if not callable(x):result[n]=value(x,depth+1)
        except Exception as e:result[n]={'unavailable':repr(e)}
    return result or repr(obj)


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value(obj),indent=2)+'\n')


def archive_urdf(root,file,destination):
    """Preserve generated URDF plus referenced mesh/material dependency closure."""
    root=Path(root);src=root/file;target=Path(destination)/(Path(file).name if Path(file).is_absolute() else file)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    records=[];queue=[];seen=set()
    for mesh in ET.parse(src).getroot().findall('.//mesh'):
        queue.append((src.parent/mesh.get('filename'),target.parent/mesh.get('filename')))
    while queue:
        source,dest=queue.pop(0);source=source.resolve();dest=dest.resolve()
        if Path(destination).resolve() not in dest.parents:
            # Preserve an external/absolute dependency by manifest, never write
            # back into a source asset directory or outside this output tree.
            dest=(Path(destination)/'external'/source.name).resolve()
        if str(dest) in seen:continue
        seen.add(str(dest))
        if not source.is_file():
            records.append(dict(source=str(source),destination=str(dest),missing=True));continue
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        records.append(dict(source=str(source),destination=str(dest),bytes=source.stat().st_size,
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        if source.suffix.lower()=='.obj':
            for line in source.read_text(errors='replace').splitlines():
                if line.startswith('mtllib '):
                    queue.extend((source.parent/n,dest.parent/n) for n in line.split()[1:])
        elif source.suffix.lower()=='.mtl':
            for line in source.read_text(errors='replace').splitlines():
                if line.startswith(('map_Kd ','map_Ka ','map_Bump ','bump ')):
                    n=line.split()[-1];queue.append((source.parent/n,dest.parent/n))
        elif source.suffix.lower()=='.dae':
            for n in ET.parse(source).getroot().iter():
                if n.tag.endswith('init_from') and n.text:
                    candidate=source.parent/n.text
                    if candidate.is_file():queue.append((candidate,dest.parent/n.text))
    return dict(original_urdf=str(src),archived_urdf=str(target),dependencies=records,
                urdf_sha256=hashlib.sha256(src.read_bytes()).hexdigest())


class PropertyOnlyGym:
    def __init__(self,raw,out):
        self._raw=raw;self.out=Path(out);self.sealed=False;self.calls=[];self.assets=[];self.actor_creations=[]
        self.creation_calls=0;self.forbidden_attempts=0;self.asset_ids={};self._loaded_assets=[]

    def seal(self):self.sealed=True

    def __getattr__(self,name):
        if name in FORBIDDEN or (self.sealed and name.startswith(('set_','create_','load_','apply_','add_','prepare_'))):
            self.forbidden_attempts+=1;self.calls.append(dict(method=name,blocked=True,sealed=self.sealed))
            raise RuntimeError('Forbidden in property-only inspection: '+name)
        fn=getattr(self._raw,name)
        if not callable(fn):return fn
        def call(*args,**kwargs):
            self.calls.append(dict(method=name,sealed=self.sealed))
            if name=='create_sim':
                if self.creation_calls:raise RuntimeError('One-scene budget exhausted')
                # Exclusive receipt is written BEFORE the only SDK attempt.
                with (self.out/'scene-creation-receipt.json').open('x') as f:
                    json.dump(dict(time=time.time(),create_sim_attempts=1,simulate_calls=0,trajectory_calls=0),f,indent=2)
                self.creation_calls+=1
                write(self.out/'sim-parameters.json',dict(compute_device=args[0],graphics_device=args[1],physics_engine=args[2],parameters=args[3]))
            if name=='load_asset':
                i=len(self.assets);root,file,options=args[1:4]
                record=dict(id=i,root=str(root),file=str(file),options=value(options),archive=archive_urdf(root,file,self.out/'imported-assets'/('asset_%03d'%i)))
                self.assets.append(record);write(self.out/'asset-imports.json',self.assets)
                result=fn(*args,**kwargs);self.asset_ids[id(result)]=i;self._loaded_assets.append(result);return result
            if name=='create_actor':
                self.actor_creations.append(dict(name=args[3],asset_id=self.asset_ids.get(id(args[1])),pose=value(args[2]),
                    collision_group=args[4] if len(args)>4 else None,collision_filter=args[5] if len(args)>5 else None,
                    segmentation_id=args[6] if len(args)>6 else None))
            if name.startswith(('set_','apply_')):
                # Initialization writes are retained, not passed off as reads.
                with (self.out/'initialization-writes.jsonl').open('a') as f:
                    f.write(json.dumps(dict(method=name,arguments=[value(x) for x in args[1:]]))+'\n')
            return fn(*args,**kwargs)
        return call


def guard_test():
    class Fake:
        def __getattr__(self,n):raise AssertionError('SDK function must not be accessed: '+n)
    p=PropertyOnlyGym(Fake(),Path('/does-not-create-a-scene'))
    rejected=[]
    for n in FORBIDDEN:
        try:getattr(p,n)
        except RuntimeError:rejected.append(n)
    p.seal()
    for n in ['set_actor_dof_states','create_sim','load_asset','create_actor','prepare_sim']:
        try:getattr(p,n)
        except RuntimeError:rejected.append(n)
    assert len(rejected)==len(FORBIDDEN)+5
    return dict(blocked_method_lookup_count=len(rejected),SDK_functions_accessed=0,scenes_created=0)


def snapshot(gym,sim,env,ga,out):
    actors=[]
    for ai in range(gym.get_actor_count(env)):
        actor=gym.get_actor_handle(env,ai);names=list(gym.get_actor_rigid_body_names(env,actor))
        ranges=gym.get_actor_rigid_body_shape_indices(env,actor)
        shapes=gym.get_actor_rigid_shape_properties(env,actor)
        bodies=gym.get_actor_rigid_body_properties(env,actor)
        states=gym.get_actor_rigid_body_states(env,actor,ga.STATE_ALL)
        dof_names=list(gym.get_actor_dof_names(env,actor))
        record=dict(actor_index=ai,actor_handle=actor,name=gym.get_actor_name(env,actor),body_names=names,
            body_properties=value(bodies),shape_properties=value(shapes),body_shape_ranges=value(ranges),
            shape_count=gym.get_actor_rigid_shape_count(env,actor),unstepped_body_states=value(states),
            dof_names=dof_names,dof_properties=value(gym.get_actor_dof_properties(env,actor)) if dof_names else [],
            unstepped_dof_states=value(gym.get_actor_dof_states(env,actor,ga.STATE_ALL)) if dof_names else [],body_indices=[])
        for bi,name in enumerate(names):
            handle=gym.get_actor_rigid_body_handle(env,actor,bi)
            record['body_indices'].append(dict(name=name,actor_index=bi,handle=handle,
                environment_index=gym.get_actor_rigid_body_index(env,actor,bi,ga.DOMAIN_ENV),
                simulation_index=gym.get_actor_rigid_body_index(env,actor,bi,ga.DOMAIN_SIM),
                get_rigid_transform=value(gym.get_rigid_transform(env,handle))))
        actors.append(record);write(out/'actor-properties.json',actors)
    write(out/'actor-creation-arguments.json',gym.actor_creations)
    asset_properties=[]
    for i,a in enumerate(gym._loaded_assets):
        asset_properties.append(dict(asset_id=i,body_names=list(gym.get_asset_rigid_body_names(a)),
            body_shape_ranges=value(gym.get_asset_rigid_body_shape_indices(a)),shape_properties=value(gym.get_asset_rigid_shape_properties(a)),
            shape_count=gym.get_asset_rigid_shape_count(a),dof_names=list(gym.get_asset_dof_names(a))))
    write(out/'asset-properties.json',asset_properties)
    write(out/'api-capabilities.json',dict(methods=sorted(n for n in dir(type(gym._raw)) if not n.startswith('_')),
        limitation='Rigid shape properties and ranges do not themselves expose cooked vertices, per-shape local geometry transforms or shape type. Soft/deformable triangle APIs are not rigid collision geometry exports.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--geometry',required=True);p.add_argument('--preflight',action='store_true')
    a,argv=p.parse_known_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    geometry=Path(a.geometry)
    if hashlib.sha256(geometry.read_bytes()).hexdigest()!=GEOMETRY_SHA:raise RuntimeError('Reviewed route hash mismatch')
    plan=json.loads(geometry.read_text())
    if plan.get('navigation_revision')!='nominal_compact_reverse_v1':raise RuntimeError('Wrong navigation revision')
    if not a.preflight and (out/'scene-creation-receipt.json').exists():
        raise RuntimeError('Authorized scene creation already consumed; do not retry')
    write(out/('preflight-command.json' if a.preflight else 'command.json'),dict(argv=sys.argv,cwd=str(Path.cwd()),
        environment={n:os.environ.get(n) for n in ['VIDEO2DOOR_GPU','CUDA_VISIBLE_DEVICES','VK_LAYER_PATH','VK_INSTANCE_LAYERS','LD_LIBRARY_PATH']},
        geometry=str(geometry),geometry_sha256=GEOMETRY_SHA,meaning='PROPERTY INSPECTION ONLY; no dynamics or trajectory'))
    if a.preflight:
        write(out/'preflight.json',dict(guard_test=guard_test(),geometry_sha256=GEOMETRY_SHA,scene_creations=0))
        print('Guard and geometry preflight passed; no simulator created.');return
    sys.path.insert(0,str(ROOT/'high-level/float_ik'))
    import isaacgym_float_ik_a2w_basearn_push_door_parallel as m
    sys.argv=[str(ROOT/'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py'),*argv]
    args=m.parse_args()
    if args.num_envs!=1 or not args.no_ikpush_env_randomization or args.dp_policy_checkpoint or args.record_dp_dataset or args.expert_action_replay_raw_episode:
        raise RuntimeError('Inspection must be single nominal env, no policy, replay or recorder')
    m.resolve_seed(args)
    write(out/'effective-args-before-creation.json',{k:v for k,v in vars(args).items() if isinstance(v,(int,float,str,bool,list,tuple,type(None)))})
    shutil.copy2(geometry,out/'geometry-compact.json')
    src=out/'source';src.mkdir(exist_ok=False)
    for file in [Path(__file__),Path(__file__).with_name('fullcycle_navigation.py'),ROOT/'AGENTS.md',
        ROOT/'high-level/float_ik/door_common.py',ROOT/'high-level/float_ik/isaacgym_visualize_b1z1_basearn.py',
        ROOT/'high-level/float_ik/isaacgym_a2w_ik_push_door_parallel.py',ROOT/'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py']:
        shutil.copy2(file,src/file.name)
    ga=m.gymapi;gym=PropertyOnlyGym(ga.acquire_gym(),out);sim=None
    result=dict(scene_create_attempts=0,simulate_calls=0,trajectory_calls=0,scene_state_mutations_after_seal=0)
    try:
        sim,dt=m.base_ik.create_sim(gym,args);args.sim_dt=float(dt)
        ground=ga.PlaneParams();ground.normal=ga.Vec3(0,0,1);gym.add_ground(sim,ground)
        base,arm=m.load_a2wz1_float_robot_assets(gym,sim,args,out/'generated-assets')
        doors=m.load_door_assets(gym,sim,args)
        base_data=m.configure_a2w_base_visual_dofs(gym,base,args)
        dof_names,props,states,targets,lower,upper,defaults,speeds,selected=m.base_ik.configure_dofs(gym,arm,args)
        if 'jointGripper' in dof_names:
            j=dof_names.index('jointGripper');states['pos'][j]=args.gripper_open;targets[j]=args.gripper_open
        door=m.clone_door_runtime(doors[0]);ea=m.make_env_args(args,0)
        env,aa,handles,da,_=m.create_parallel_env_actors(gym,sim,base,arm,door,props,states,ea,0,1)
        m.apply_a2w_base_visual_dofs(gym,env,handles,base_data)
        gym.prepare_sim(sim)
        gym.seal()
        write(out/'effective-args-after-asset-overrides.json',{k:v for k,v in vars(ea).items() if isinstance(v,(int,float,str,bool,list,tuple,type(None)))})
        snapshot(gym,sim,env,ga,out)
        cache=[]
        for path in (Path.home()/'.isaacgym/vhacd').glob('*video2door_wc4_act*door_set_wc4*'):
            if path.is_file():
                dest=out/'cooked-cache'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
                cache.append(dict(source=str(path),copy=str(dest),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),meaning='Opaque existing VHACD cache, not a documented per-shape geometry export'))
        write(out/'cooked-cache-provenance.json',cache)
        result['status']='property_snapshot_complete'
    except BaseException as e:
        result['status']='inspection_stopped';result['error']=repr(e)
        raise
    finally:
        result['scene_create_attempts']=gym.creation_calls
        result['forbidden_method_lookups']=gym.forbidden_attempts
        write(out/'api-call-audit.json',gym.calls)
        write(out/'inspection-result.json',result)
        # Destroy is cleanup of the sole scene, never a reset/recreation.
        if sim is not None:
            gym.destroy_sim(sim);result['scene_destroyed']=True
            write(out/'api-call-audit.json',gym.calls);write(out/'inspection-result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
