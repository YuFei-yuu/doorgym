"""Pause launches for offline report repair; completed physics is never retried."""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from newexpert_stage7.common import write
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--reason',required=True);a=p.parse_args()
try:write(a.run/'DIAGNOSTIC_HOLD.json',{'index':a.index,'reason':a.reason,'at':time.time(),'physics_retry_allowed':False})
except FileExistsError:pass
