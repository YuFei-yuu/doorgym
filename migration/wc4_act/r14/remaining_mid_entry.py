"""Only the six untouched 010000 seeds; reuse already verified 010000 export."""
import sys,os,subprocess,runpy,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R13=ROOT/'migration/wc4_act/r13'
sys.path[:0]=[str(R13),str(ROOT/'migration/wc4_act')]
import common
base_read=common.read

def read(path):
    result=base_read(path)
    if Path(path)==common.RUN/'contract.json':
        result=dict(result);result['intermediate_seeds']=list(range(62302,62308))
    return result
common.read=read
original=subprocess.run

def reused(command,*a,**k):
    if isinstance(command,(tuple,list)) and str(R13/'export.py') in command:
        source=ROOT/'.repro/runs/wc4-fulltask-act-r14-s1000-20261022/checkpoints/010000'
        proof=base_read(source/'export-consistency.json');assert proof['status']=='passed'
        (common.RUN/'reused-010000-export.json').write_text(json.dumps(dict(source=str(source),existing_consistency=proof,
          new_model_forward=False,new_export=False),indent=2)+'\n')
        if k.get('stdout'):k['stdout'].write('Reused immutable original 010000 export with passed consistency; no additional forward.\n')
        return subprocess.CompletedProcess(command,0)
    return original(command,*a,**k)
subprocess.run=reused
sys.argv=[str(Path(__file__).with_name('evaluate_entry.py')),'10000']
runpy.run_path(str(Path(__file__).with_name('evaluate_entry.py')),run_name='__main__')
