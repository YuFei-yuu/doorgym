"""R13 training unchanged; add W&B reporting and audited evaluation subprocess entry."""
import sys,os,time,runpy,subprocess,shutil,resource,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R13=ROOT/'migration/wc4_act/r13'
sys.path[:0]=[str(R13),str(ROOT/'migration/wc4_act')]
from lerobot.rl.wandb_utils import WandBLogger
import torch
log_dict=WandBLogger.log_dict

def reporting(self,d,step,*a,**k):
    d=dict(d)
    if 'update_s' in d:
        denominator=float(d['update_s'])+float(d.get('dataloading_s',0))
        if denominator>0:d['samples_per_s']=16/denominator
        d['disk_free_gib']=shutil.disk_usage(ROOT).free/1024**3
        d['cuda_allocated_gib']=torch.cuda.memory_allocated()/1024**3
        d['cuda_reserved_gib']=torch.cuda.memory_reserved()/1024**3
        d['process_maxrss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        row=dict(update=step,at=time.time(),**{key:d[key] for key in ['samples_per_s','disk_free_gib','cuda_allocated_gib','cuda_reserved_gib','process_maxrss_gib'] if key in d})
        with (Path(os.environ['WC4_EXECUTION_RUN'])/'reporting.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return log_dict(self,d,step,*a,**k)
WandBLogger.log_dict=reporting
original=subprocess.run

def audited(command,*a,**k):
    if isinstance(command,(list,tuple)) and str(R13/'evaluate.py') in command:
        command=list(command);command[command.index(str(R13/'evaluate.py'))]=str(Path(__file__).with_name('evaluate_entry.py'))
    return original(command,*a,**k)
subprocess.run=audited
runpy.run_path(str(R13/'train.py'),run_name='__main__')
