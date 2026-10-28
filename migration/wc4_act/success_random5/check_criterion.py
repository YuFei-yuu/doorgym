"""CPU truth-table regression: extra task events cannot alter the score."""
import math
from success_random5.criterion import SuccessMonitor

for opened in [False, True]:
    for authorized in [False, True]:
        for final_locked in [False, True]:
            s=SuccessMonitor()
            s.observe(.02,math.radians(80 if opened else 79.999),False)
            if authorized:s.authorize_relock(3.,.001,{'unchanged_gate':True})
            s.observe(4.,.001,final_locked)
            assert s.result()['success'] == (opened and authorized and final_locked)
# No traversal/release/fold/heading/hold inputs exist; success survives their absence.
s=SuccessMonitor();s.observe(1.,-math.radians(80),False)
s.authorize_relock(2.,0.,{});s.observe(2.02,0.,True)
assert s.result()['success'] and s.result()['max_abs_open_deg']==80.
s.observe(3.,0.,False);assert not s.result()['success']
print('PASS: eight truth-table cases, absolute 80-degree boundary, final unlock, no auxiliary event inputs')
