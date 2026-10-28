"""CPU-only admission-bound and source-isolation checks."""
import ast,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from expert50.collect import can_launch
for successes in range(46):
    for inflight in range(5):
        assert can_launch(successes,inflight)==(successes+inflight<45)
assert not subprocess.check_output(['git','diff','405356a','--','migration/wc4_act/author_fast5','migration/wc4_act/legdiag_random5','migration/wc4_act/success_random5','migration/wc4_act/fullcycle_task_monitor.py'],cwd=ROOT,text=True)
for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
print('PASS: success/inflight admission bound; accepted runtime/action/physics/scoring source unchanged; Python syntax')
