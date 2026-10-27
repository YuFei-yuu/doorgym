"""Prepare exactly one nominal physical attempt; no simulator imports."""
import argparse,copy,json,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from newexpert_stage4.prepare import sha,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    check=ROOT/'.repro/reviews/wc4-close-inplace-stage6-20261027-check/targeted-check.json'
    receipt=json.loads(check.read_text());assert receipt['passed']
    # Numerical check was executed at 414f96c; bind the unchanged tested control to this run.
    import hashlib
    tested={}
    for n in ['controller.py','dynamic_gate.py']:
        rel='migration/wc4_act/close_inplace_stage6/'+n
        tested[rel]=hashlib.sha256(subprocess.check_output(['git','show','414f96c:'+rel],cwd=ROOT)).hexdigest()
        assert tested[rel]==sha(ROOT/rel)
    before=(ROOT/'migration/wc4_act/close_inplace_stage5/controller.py').read_text()
    after=(ROOT/'migration/wc4_act/close_inplace_stage6/controller.py').read_text()
    prefix=lambda x:x[x.index("        if phase=='retimed_traverse':"):x.index("        elif phase=='close':")]
    assert prefix(before)==prefix(after),'Approved prefix changed'
    run.mkdir(parents=True,exist_ok=False)
    write(run/'targeted-check-binding.json',dict(check=str(check),sha256=sha(check),tested_controller_sha256=tested,prefix_identical=True,
        source_frozen_log_sha256=receipt['source_sha256'],negative_cases=receipt['gate_cases'],scope='Recorded-feedback CPU checks, not physical success'))
    old=ROOT/'.repro/runs/wc4-newexpert-stage4-20261025'
    prior=json.loads((old/'manifest.json').read_text());trial=copy.deepcopy(prior['trials'][0])
    assert trial['seed']==62000 and trial['nominal']
    yaml=run/'frozen-physics.yaml';yaml.write_bytes(Path(prior['yaml']).read_bytes())
    cli=trial['argv'];cli[cli.index('--steps')+1]='12000';cli[cli.index('--a2w_float_ik_config')+1]=str(yaml)
    assert cli[cli.index('--num_envs')+1]=='1'
    protected={n:h for n,h in prior['source_sha256'].items() if n.startswith(('high-level/data/','high-level/float_ik/'))}
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    files=[Path(n) for n in subprocess.check_output(['git','ls-files','migration/wc4_act','high-level/float_ik'],cwd=ROOT,text=True).splitlines() if (ROOT/n).is_file()]
    sources={str(n):sha(ROOT/n) for n in files};sources.update(protected)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    cfg=dict(radius=.905,height=.85,follow_m=.45,nominal_seed=62000,maximum_seconds=240,
        normal_roll_start_deg=13.5,normal_roll_after80_deg=15.5,joint_command_rate=.7,
        basis='Round6 sustained dynamic contact transition; unchanged nominal physics and prefix',dynamic_contact_hold_s=.30,dynamic_contact_motion_rad=.003,handoff_blend_s=2.)
    plan={'compact_expert':cfg,'selected':{},'live_joint9_recording':True}
    write(run/'geometry.json',plan)
    write(run/'manifest.json',dict(start_commit=head,maximum_scenes=1,maximum_sim_seconds=240,seeds=[62000],trials=[trial],
        r7_physics_after_asset_overrides=prior['r7_physics_after_asset_overrides'],yaml=str(yaml),yaml_sha256=sha(yaml),source_sha256=sources,
        provenance={'nominal_manifest':str(old/'manifest.json'),'sha256':sha(old/'manifest.json')}))
    write(run/'consistency.json',dict(geometry_sha256=sha(run/'geometry.json'),status='source_route_consistent',scope='Source/interface binding; not a physical success or full geometry certificate'))
    for n in ['initial-state.json','effective-config.json']:(run/('r7-'+n)).write_bytes((old/'attempt_000'/n).read_bytes())
    write(run/'budget.json',dict(episodes=1,max_seconds=240,retry=False,training=0,batch_collection=0,reserve_bytes=40*1024**3,planned_saving_bytes=10*1024**3))
    write(run/'source-binding.json',dict(head=head,main=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip(),protected=protected))
    print(json.dumps(dict(run=str(run),head=head,physics_started=False)))
if __name__=='__main__':main()
