"""Validate a recorded nominal handoff against a new approach without Gym."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np

from fullcycle_geometry import URDF
from fullcycle_smooth_approach import build_approach, integrate_commands

ROOT=Path(__file__).resolve().parents[2]


def phase_body(source, name):
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node,ast.If) and isinstance(node.test,ast.Compare)
                and isinstance(node.test.left,ast.Name) and node.test.left.id=='phase'
                and len(node.test.comparators)==1
                and isinstance(node.test.comparators[0],ast.Constant)
                and node.test.comparators[0].value==name):
            return node.body
    raise ValueError("Missing phase "+name)


def canonical(nodes):
    return ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True);parser.add_argument('--boundary',required=True)
    parser.add_argument('--reference-repo',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(exist_ok=False)
    plan=json.loads(Path(args.plan).read_text());b=json.loads(Path(args.boundary).read_text())
    dt=b['pass']['t']/(b['pass']['step']+1)
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf',include_visual=True)
    p=build_approach(b['pass']['command_base'],b['pass']['action'],b['pass_previous']['action'],
                     plan['smooth_approach'],np.r_[plan['selected']['q'],0.],dt,
                     robot,b['pass']['arm_q'],start_step=b['pass']['step']+1)
    a=p['actions'];prebrake=p['time']<p['continuous_curve_duration_s']-1
    checks={
        'finite_joint9':bool(a.shape[1]==9 and np.isfinite(a).all()),
        'no_reverse':bool(np.all(a[:,0]>=0)),
        'no_in_place_turn':bool(not np.any((abs(a[:,0])<1e-5)&(abs(a[:,1])>1e-4))),
        'no_intermediate_stop':bool(np.all(a[prebrake,0]>0)),
        'existing_base_limits':bool(a[:,0].max()<=.25 and abs(a[:,1]).max()<=.6),
        'counterclockwise_net_heading':bool(p['base'][-1,2]-p['base'][0,2]>math.pi),
        'contact_pose_preserved':bool(p['contact_position_error_m']<.003 and p['contact_heading_error_rad']<.004),
        'contact_joint_command_preserved':bool(np.max(abs(a[-1,2:]-np.array(b['contact']['action'][2:])))<1e-6),
        'joint9_integrator_matches':bool(np.array_equal(p['base'],integrate_commands(a,b['pass']['command_base'],dt))),
        'deploy_cadence_aligned':bool((b['pass']['step']+1+len(a))%2==0),
    }
    new=Path(__file__).with_name('fullcycle_expert.py').read_text()
    old=subprocess.check_output(['git','-C',args.reference_repo,'show',
        '12f5ee8:migration/wc4_act/fullcycle_expert.py'],text=True)
    for name in ['deploy','release','retract','restore_heading','final_hold']:
        checks['unchanged_'+name+'_controller']=canonical(phase_body(new,name))==canonical(phase_body(old,name))
    checks['close_controller_only_hold_gate_changed']=canonical(phase_body(new,'close')[:-1])==canonical(phase_body(old,'close')[:-1])
    def fn(src,name):
        return next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
    checks['unchanged_opening_command']=ast.dump(fn(new,'opening_command'),include_attributes=False)==ast.dump(fn(old,'opening_command'),include_attributes=False)
    result={'checks':checks,'all_pass':all(checks.values()),'duration_s':p['duration_s'],
            'nominal_pass_to_contact_r5_s':b['contact']['t']-b['pass']['t'],
            'max_base_speed':float(a[:,0].max()),'max_abs_yaw_rate':float(abs(a[:,1]).max()),
            'minimum_forward_speed_before_braking':p['minimum_forward_speed_before_final_braking'],
            'contact_position_error_m':p['contact_position_error_m'],'contact_heading_error_rad':p['contact_heading_error_rad'],
            'max_command_step_delta':abs(np.diff(a,axis=0)).max(0).tolist(),
            'withdraw_position_error_m':p['withdraw_position_error_m'],
            'withdraw_orientation_error_rad':p['withdraw_orientation_error_rad'],
            'geometry_sha256':hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('fullcycle*.py')},
            'physical_execution_verified':False}
    np.savez_compressed(out/'planned-approach.npz',**{k:v for k,v in p.items() if isinstance(v,np.ndarray)})
    (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not result['all_pass']:raise AssertionError('Smooth approach regression failed')


if __name__=='__main__':main()
