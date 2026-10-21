"""Manual stop request only; does not kill, train, evaluate or resume anything."""
import argparse
from stopping import request_stop
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--reason',required=True);a=p.parse_args()
request_stop(a.run,a.reason,source='operator')
print('Stop request persisted; wait for STOPPED.json and committed checkpoint. No automatic resume.')
