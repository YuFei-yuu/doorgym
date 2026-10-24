"""Unmodified R11 data/video check in existing ACT CPU environment."""
import argparse, json, sys, traceback
from pathlib import Path
ROOT=Path('/data/zyf/video2door-wc4-act')
sys.path.insert(0,str(ROOT/'migration/wc4_act/r11'))
from validate_recordings import validate
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
assert (a.run/'entry-exit.json').exists()
out=a.run/'audit';out.mkdir(exist_ok=True)
class TrialRoot:
    def __truediv__(self,name):
        assert name=='trial_00';return a.run/'attempt_000'
try:validate(TrialRoot(),0,out)
except BaseException:
    (out/'record-validation-error.txt').write_text(traceback.format_exc());raise
