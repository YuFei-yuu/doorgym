"""CPU regression using the frozen false-positive sample and positive/negative controls."""
import argparse,copy,json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bindings import compare_spawn,quaternion_rotation_error,recording_contract,EPS32,ROTATION_ATOL_RAD,UNIT_NORM_ATOL

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    fixture=json.loads(a.fixture.read_text());actual=fixture['actual'];expected=fixture['expected'];cases=[]
    def check(name,c,e,want):
        result=compare_spawn(c,e);assert result['matches_frozen']==want,(name,result)
        cases.append({'name':name,'expected':want,'result':result})
    check('frozen_stage5_one_ULP_false_positive',actual,expected,True)
    for label,scale in [('q_minus_q',-1),('float32_norm_rounding',1+2*EPS32)]:
        c=copy.deepcopy(actual)
        for k in ['x','y','z','w']:c['root']['pose']['r'][k]*=scale
        check(label,c,expected,True)
    c=copy.deepcopy(actual);c['yaw']+=2*math.pi;check('yaw_2pi_equivalence',c,expected,True)
    for angle in [4*ROTATION_ATOL_RAD,.001,.1,math.pi]:
        c=copy.deepcopy(actual);q=(Rotation.from_rotvec([angle,0,0])*Rotation.from_quat(expected['root_quat'])).as_quat()
        c['root']['pose']['r']=dict(zip(['x','y','z','w'],q.tolist()));check('true_rotation_bias_'+str(angle),c,expected,False)
    for label,values in [('zero_norm',[0]*4),('bad_norm',[0,0,0,2]),('tiny_norm',[0,0,0,1e-20]),('nan',[0,0,float('nan'),1]),('inf',[0,0,0,float('inf')])]:
        c=copy.deepcopy(actual);c['root']['pose']['r']=dict(zip(['x','y','z','w'],values));check(label,c,expected,False)
    for label,mutation in [('root_xy_bias',lambda c:c['root']['pose']['p'].update(x=c['root']['pose']['p']['x']+.001)),
                           ('root_z_bias',lambda c:c['root']['pose']['p'].update(z=c['root']['pose']['p']['z']+.001)),
                           ('metadata_xy_bias',lambda c:c['base_start'].__setitem__(0,c['base_start'][0]+.001)),
                           ('nonfinite_position',lambda c:c['root']['pose']['p'].update(y=float('nan'))),
                           ('nonfinite_yaw',lambda c:c.update(yaw=float('inf'))),
                           ('true_yaw_bias',lambda c:c.update(yaw=c['yaw']+.001))]:
        c=copy.deepcopy(actual);mutation(c);check(label,c,expected,False)
    manifest=json.loads((a.fixture.parents[2]/'manifest.json').read_text())
    for i,trial in enumerate(manifest['trials']):
        e=trial['expected_spawn'];c=copy.deepcopy(actual)
        c['base_start']=e['base_start'];c['yaw']=e['yaw']
        c['root']['pose']['p']=dict(zip(['x','y','z'],[*e['base_start'],e['robot_z']]))
        q=np.asarray(e['root_quat'],float);q=q/np.linalg.norm(q)
        c['root']['pose']['r']=dict(zip(['x','y','z','w'],q.tolist()))
        check('inherited_spawn_normalization_%02d'%i,c,e,True)
        for view,pose in trial['camera_local_poses'].items():
            for name,values in pose.items():assert np.array_equal(np.asarray(values),np.asarray(values,np.float32)),(i,view,name)
    negative_reference=copy.deepcopy(expected);negative_reference['root_quat'][0]=float('nan');check('nonfinite_reference',actual,negative_reference,False)
    d={'encoded':False,'raw_depth_observations_retained':True,'buffers_observed':{'front_depth.mp4':2,'wrist_depth.mp4':2}}
    base=dict(steps=3,state_rows=[0,1,2],contact_rows=[0,1,2],clock_steps=[0,2],rgb_frames=2,rgb_fps=25,depth_receipt=d,depth_video_exists=False)
    recording=[]
    for name,changes,want in [('depth_videos_absent_but_raw_and_rgb_present',{},True),('missing_rgb',{'rgb_frames':0},False),
        ('missing_contact_row',{'contact_rows':[0,2]},False),('wrong_clock',{'clock_steps':[0,1]},False),
        ('missing_depth_receipt',{'depth_receipt':{}},False),('unexpected_depth_video',{'depth_video_exists':True},False),
        ('wrong_rgb_rate',{'rgb_fps':24},False)]:
        result=recording_contract(**dict(base,**changes));assert result['passed']==want,(name,result)
        recording.append({'name':name,'expected':want,'result':result})
    result={'all_passed':True,'pose_cases':cases,'recording_cases':recording,'fixture':str(a.fixture),
        'tolerance_basis':{'float32_eps':EPS32,'rotation_atol_rad':ROTATION_ATOL_RAD,'unit_norm_atol':UNIT_NORM_ATOL,
            'position':'4*eps32*max(1,abs(coordinate)) meters; actual root xy/z and metadata both checked'},
        'scene_creations':0,'physics_steps':0}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'pose_cases':len(cases),'recording_cases':len(recording)}))

if __name__=='__main__':main()
