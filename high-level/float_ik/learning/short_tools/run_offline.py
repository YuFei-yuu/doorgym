"""Budgeted offline diagnostic; no fixed experiment path or seed allocation."""
import argparse,sys
from pathlib import Path
here=Path(__file__).resolve().parents[1];sys.path.insert(0,str(here/'run_tools'))
from launch import launch
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--stage',required=True);p.add_argument('--timeout',required=True,type=float);p.add_argument('--total-seconds',required=True,type=float);p.add_argument('extra',nargs=argparse.REMAINDER);a=p.parse_args()
extra=a.extra[1:] if a.extra[:1]==['--'] else a.extra
raise SystemExit(launch(a.root,a.stage,['python',here/'offline_eval.py','--root',a.root,'--output',a.output]+extra,a.timeout,a.total_seconds))
