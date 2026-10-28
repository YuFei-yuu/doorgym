"""Reproduce the stopped run's interpolation endpoint from its saved clock only."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from author_fast5.geometry import smooth

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
 rows=[json.loads(x) for x in (run/'attempt_000/states.jsonl').open()]
 first=next(r for r in rows if r['phase']=='approach');anchor=rows[first['step']-1]['observation_pre']['time_s']
 result=json.loads((run/'attempt_000/result.json').read_text());dt=json.loads((run/'physics-start.json').read_text())['dt']
 elapsed=result['attempted_step']*dt-anchor;u=elapsed/2.;s=smooth(u)
 d=dict(actual_dt=dt,approach_anchor_pre_s=anchor,attempted_step=result['attempted_step'],elapsed_s=elapsed,input_u=u,
  quintic_s=s,overshoot_above_one=s-1,within_slerp_domain=bool(0<=s<=1),source_unchanged=True,physical_reruns=0,
  cpu_check_clock='i * 0.02; did not exercise actual float32 dt boundary')
 (run/'interpolation-diagnosis.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
