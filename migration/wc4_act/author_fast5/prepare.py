"""Bind one nominal scene, 120 s maximum, without creating a simulator."""
import argparse,copy,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from newexpert_stage4.prepare import sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--check',type=Path,required=True);a=p.parse_args()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
 run=a.run.resolve();run.mkdir(parents=True,exist_ok=False)
 old=ROOT/'.repro/runs/wc4-newexpert-stage4-20261025';prior=json.loads((old/'manifest.json').read_text());trial=copy.deepcopy(prior['trials'][0])
 assert trial['seed']==62000 and trial['nominal']
 config=run/'frozen-physics.yaml';config.write_bytes((ROOT/'migration/wc4_act/author_fast5/frozen-r7-physics.yaml').read_bytes())
 assert sha(config)==sha(old/'frozen-r7-physics.yaml')
 cli=trial['argv'];cli[cli.index('--steps')+1]='6000';cli[cli.index('--a2w_float_ik_config')+1]=str(config)
 protected={n:h for n,h in prior['source_sha256'].items() if n.startswith(('high-level/data/','high-level/float_ik/'))}
 assert all(sha(ROOT/n)==h for n,h in protected.items())
 files=subprocess.check_output(['git','ls-files','migration/wc4_act','high-level/float_ik'],cwd=ROOT,text=True).splitlines()
 sources={n:sha(ROOT/n) for n in files if (ROOT/n).is_file()};sources.update(protected)
 cfg=json.loads((Path(__file__).parent/'config.json').read_text());plan={'expert':cfg,'live_joint9_recording':True}
 write(run/'geometry.json',plan)
 check=json.loads((a.check/'preflight.json').read_text())
 for f in ['config.json','geometry.py','trajectory.py','adapter.py','runtime.py','recording.py']:
  assert check['source_sha256'][f]==sha(Path(__file__).parent/f),f
 write(run/'preflight-binding.json',{'path':str(a.check),'preflight':check,'sha256':sha(a.check/'preflight.json')})
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 write(run/'manifest.json',dict(start_commit=head,maximum_scenes=1,maximum_sim_seconds=120,seeds=[62000],trials=[trial],
  yaml=str(config),yaml_sha256=sha(config),source_sha256=sources,r7_physics_after_asset_overrides=prior['r7_physics_after_asset_overrides'],
  prior_manifest=str(old/'manifest.json'),prior_manifest_sha256=sha(old/'manifest.json')))
 write(run/'preparation-receipt.json',{'geometry_sha256':sha(run/'geometry.json'),'scenes_created':0,'head':head})
 write(run/'budget.json',{'episodes':1,'max_seconds':120,'max_steps':6000,'automatic_retry':False,'training':False,'seed':62000,'nominal':True})
 print(json.dumps({'run':str(run),'head':head,'physics_started':False}))
if __name__=='__main__':main()
