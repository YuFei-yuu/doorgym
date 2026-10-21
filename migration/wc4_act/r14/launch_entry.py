"""Bind the accepted R13 supervisor to the R14 reporting entry, same train args."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R13=ROOT/'migration/wc4_act/r13'
sys.path[:0]=[str(R13),str(ROOT/'migration/wc4_act')]
import runtime
build=runtime.training_argv

def argv(run):
    result=build(run);result[2]=str(Path(__file__).with_name('train_entry.py'));return result
runtime.training_argv=argv
runpy.run_path(str(R13/'launch_train.py'),run_name='__main__')
