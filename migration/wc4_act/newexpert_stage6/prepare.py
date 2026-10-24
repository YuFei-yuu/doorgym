"""Inherit frozen stage-five samples without importing any sampling function."""
import argparse,copy,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from newexpert_stage4.prepare import sha,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--regression',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();old=ROOT/'.repro/runs/wc4-newexpert-stage5-20261025'
    original=json.loads((old/'manifest.json').read_text());m=copy.deepcopy(original)
    regression=json.loads(a.regression.read_text());assert regression['all_passed']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    for n,h in original['source_sha256'].items():assert sha(ROOT/n)==h,'Inherited protected source changed: '+n
    for n,h in json.loads((old/'SHA256.json').read_text()).items():assert sha(old/n)==h,'Stage-five evidence changed: '+n
    run.mkdir(parents=True,exist_ok=False)
    (run/'regression.json').write_bytes(a.regression.read_bytes())
    (run/'frozen-r7-physics.yaml').write_bytes((old/'frozen-r7-physics.yaml').read_bytes())
    (run/'initial-coverage.json').write_bytes((old/'initial-coverage.json').read_bytes())
    (run/'inherited-stage5-manifest.json').write_bytes((old/'manifest.json').read_bytes())
    for trial in m['trials']:
        cli=trial['argv'];cli[cli.index('--a2w_float_ik_config')+1]=str(run/'frozen-r7-physics.yaml')
    inherited=[]
    for before,after in zip(original['trials'],m['trials']):
        b=copy.deepcopy(before);b['argv'][b['argv'].index('--a2w_float_ik_config')+1]=str(run/'frozen-r7-physics.yaml')
        assert b==after
        inherited.append({'index':after['index'],'seed':after['seed'],'all_nonpath_trial_fields_identical':True,
            'expected_spawn_identical':before['expected_spawn']==after['expected_spawn'],
            'camera_identical':before['camera_local_poses']==after['camera_local_poses'],
            'effective_sampling_identical':before['effective_sampling']==after['effective_sampling']})
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    m['start_commit']=head;m['yaml']=str(run/'frozen-r7-physics.yaml');assert sha(m['yaml'])==original['yaml_sha256']
    for folder in ['newexpert_stage5','newexpert_stage6']:
        for path in (ROOT/'migration/wc4_act'/folder).glob('*'):
            if path.is_file():m['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    m['inherited_stage5_manifest_sha256']=sha(old/'manifest.json')
    m['stage6_changes']='SO(3) spawn identity check and explicit no-depth-video recording validation only; no resampling'
    geom=json.loads((old/'geometry.json').read_text());geom['execution_authorization']={
        'stage':6,'episodes':8,'maximum_sim_seconds':150,'dual_depth_video_encoding':False,
        'basis':'User sixth-round correction and exact frozen-trial restoration order'}
    write(run/'geometry.json',geom);write(run/'manifest.json',m)
    write(run/'inheritance.json',{'source':str(old),'source_manifest_sha256':sha(old/'manifest.json'),
        'seeds_identical':m['seeds']==original['seeds'],'trials':inherited,'physics_yaml_byte_identical':True,
        'coverage_byte_identical':True,'compact_expert_identical':geom['compact_expert']==json.loads((old/'geometry.json').read_text())['compact_expert'],
        'sampling_calls':0,'prior_infrastructure_attempt':{'stage':5,'scene_creations':1,'physics_steps':0,'expert_failures':0}})
    for i in range(8):
        dest=run/('trial_%02d'%i);dest.mkdir();(dest/'geometry.json').write_bytes((run/'geometry.json').read_bytes())
    write(run/'preparation-receipt.json',{'scene_creations':0,'head':head,'manifest_sha256':sha(run/'manifest.json'),
        'geometry_sha256':sha(run/'geometry.json'),'initial_coverage_sha256':sha(run/'initial-coverage.json'),
        'regression_sha256':sha(run/'regression.json'),'inheritance_sha256':sha(run/'inheritance.json')})
    print(json.dumps({'head':head,'seeds':m['seeds'],'sampling_calls':0,'inherited_trials':len(inherited)}))

if __name__=='__main__':main()
