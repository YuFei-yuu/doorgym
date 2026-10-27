import json,sys
from pathlib import Path
root=Path('/data/zyf/video2door-wc4-act');run=Path(sys.argv[1]);result={}
for name,p in [('stage7',root/'.repro/runs/wc4-close-inplace-stage7-20261027'),('stage8',run)]:
    rows=[]
    with (p/'attempt_000/states.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if 17.49<=r['t']<=26:rows.append({k:r[k] for k in ['step','t','phase','nonzero_door_contact','any_door_contact']})
    start=next(r['t'] for r in rows if r['phase']=='traverse_door');events={}
    for field in ['nonzero_door_contact','any_door_contact']:
        began=None
        for r in rows:
            if r['t']<start:continue
            if not r[field]:
                if began is None:began=r['t']
                if r['t']-began>=.2:events[field]=dict(first_clear_s=began,confirmed_after_0p2s=r['t']);break
            else:began=None
    result[name]=events
(run/'review/traversal-contact-separation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
