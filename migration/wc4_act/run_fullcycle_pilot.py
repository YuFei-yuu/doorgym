"""One frozen attempt per invocation. Never retries, trains or runs a policy."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','attempt'],required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    manifest=a.run/'manifest.json';m=json.loads(manifest.read_text())
    if not 0<=a.index<8 or len(m['trials'])!=8:raise ValueError('Maximum eight frozen trials')
    name=('preflight' if a.mode=='preflight' else 'trial')+'_%02d'%a.index
    output=a.run/name
    if output.exists():raise FileExistsError('No replacement or retry: '+str(output))
    cmd=[sys.executable,'-B',str(ROOT/'migration/wc4_act/fullcycle_expert.py'),'--output',str(output),
        '--geometry',str(a.run/'geometry.json'),'--validation',str(a.run/'consistency.json'),
        '--pilot-manifest',str(manifest),'--trial-index',str(a.index)]
    if a.mode=='preflight':cmd.append('--preflight')
    cmd+=m['trials'][a.index]['argv']
    record=dict(argv=cmd,trial=a.index,mode=a.mode,seed=m['trials'][a.index]['seed'],start_time=time.time(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    with (a.run/(name+'-command.json')).open('x') as f:json.dump(record,f,indent=2)
    with (a.run/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    (a.run/(name+'-exit.json')).write_text(json.dumps(dict(returncode=r.returncode,end_time=time.time()))+'\n')
    print(name,'returncode',r.returncode,flush=True)
    if r.returncode:sys.exit(r.returncode)


if __name__=='__main__':main()
