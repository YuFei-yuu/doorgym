"""Controlled clock/API receipts, no W&B connections, models, optimizer, or scenes."""
from monitor_state import ProgressWatch
w=ProgressWatch();assert not w.observe(0,150100,149900)['stale']
assert not w.observe(590,150100,149900)['stale']
assert not w.observe(10000,150000,149900,paused=True)['stale']
assert not w.observe(20000,150100,149900)['stale']
assert not w.observe(20599,150100,149900)['stale']
assert w.observe(20601,150100,149900)['stale']
assert not w.observe(20602,150400,150300)['stale']
assert not w.observe(30000,150400,150300)['stale'] # within allowed reporting cadence
assert not w.observe(40000,150500,150300)['stale'] # start only when behind
assert w.observe(40601,150500,150300)['stale']
f=ProgressWatch()
for i in range(5):assert not f.observe(i*60,150000,online=False,paused=True)['api_failed']
assert f.observe(300,150000,online=False,paused=True)['api_failed']
assert not f.observe(301,150000,149900,online=True,paused=True)['api_failed']
print('{"status":"passed","normal_pause_resume":true,"fresh_upload_window":true,"true_stale_600s":true,"API_failures_6_even_paused":true,"timeout_not_relaxed":true,"model_forwards":0,"scenes":0}')
