"""CPU controls for no-video render order, buffer checks and original pose guard."""
import argparse,copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from newexpert_stage7.adapters import install_raw_render,no_video_writer,record_contract
from newexpert_stage6.bindings import compare_spawn
from newexpert_stage7.common import write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    events=[]
    gym=SimpleNamespace(step_graphics=lambda s:events.append('graphics'),render_all_camera_sensors=lambda s:events.append('render'))
    module=SimpleNamespace(create_parallel_env_states=lambda *a:('states','mode'),update_door_twin_tracker=lambda *a:events.append('tracker'))
    r=install_raw_render(module);assert module.create_parallel_env_states(gym,'sim')==('states','mode')
    module.update_door_twin_tracker(None,0,None);module.update_door_twin_tracker(None,2,None)
    assert events==['tracker','graphics','render']*2 and r['rendered_steps']==[0,2]
    counters={};writer=no_video_writer(Path('/tmp/unit'),counters)
    for name in ['front_depth.mp4','wrist_depth.mp4']:
        sink=writer('/tmp/unit/'+name);sink.append_data(np.zeros((480,640),np.uint8));sink.append_data(np.ones((480,640),np.uint8));sink.close()
    negatives=0
    for path in ['/tmp/unit/rollout.mp4','/tmp/other/front_depth.mp4']:
        try:writer(path)
        except RuntimeError:negatives+=1
    assert negatives==2
    try:writer('/tmp/unit/front_depth.mp4').append_data(np.zeros((1,1),np.uint8))
    except RuntimeError:negatives+=1
    assert negatives==3
    receipt={'encoded':False,'raw_depth_observations_retained':True,'buffers_observed':{'front_depth.mp4':2,'wrist_depth.mp4':2}}
    args=dict(steps=3,state_steps=[0,1,2],contact_steps=[0,1,2],clock_steps=[0,2],receipt=receipt,rendered_steps=[0,2],video_files=[])
    cases=[]
    for name,changes,expected in [('no_videos_raw_complete',{},True),('missing_render',{'rendered_steps':[0]},False),
        ('missing_contact',{'contact_steps':[0,2]},False),('wrong_clock',{'clock_steps':[0,1]},False),
        ('video_created',{'video_files':['rollout.mp4']},False),('missing_receipt',{'receipt':{}},False)]:
        result=record_contract(**dict(args,**changes));assert result['passed']==expected
        cases.append({'name':name,'expected':expected,'result':result})
    fixture=json.loads(a.fixture.read_text());assert compare_spawn(fixture['actual'],fixture['expected'])['matches_frozen']
    bad=copy.deepcopy(fixture['actual']);bad['root']['pose']['p']['x']+=.001
    assert not compare_spawn(bad,fixture['expected'])['matches_frozen']
    write(a.output,{'all_passed':True,'render_order':events,'negative_writer_controls':negatives,'recording_cases':cases,
        'frozen_false_positive_fixture_sha256':sha(a.fixture),'pose_guard_unchanged':True,'scene_creations':0,'physics_steps':0})
    print('Stage7 no-video adapters and controls passed; no scene created')
if __name__=='__main__':main()
