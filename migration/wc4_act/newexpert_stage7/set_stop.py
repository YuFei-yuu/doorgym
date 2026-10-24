"""Signal only this batch; does not kill, reset or restart processes."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from newexpert_stage7.common import stop
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--stage',required=True);p.add_argument('--reason',required=True);a=p.parse_args()
stop(a.run,a.index,a.stage,a.reason)
