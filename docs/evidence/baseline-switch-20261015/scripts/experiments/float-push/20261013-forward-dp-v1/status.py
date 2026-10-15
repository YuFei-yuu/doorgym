"""Small read-only status snapshot for the running experiment."""
import argparse
import json
import time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
def read(path, default):
    path=root/path
    return json.loads(path.read_text()) if path.exists() else default
state=read(Path('pipeline_state.json'),{})
manifest=read(Path('manifest.json'),[])
accepted=[r for r in manifest if r['success']]
metrics=root/'training/metrics.jsonl'
result=dict(phase=state.get('phase'),supervisor_pid=state.get('pid'),
            supervisor_alive=Path('/proc/%s'%state.get('pid')).exists(),
            collection_successes=len(accepted),collection_recorded=len(manifest),
            collected_frames=sum(r['frames'] for r in accepted),
            case=state.get('case'),training=read(Path('training/progress.json'),None),
            last_training_metric=json.loads(metrics.read_text().splitlines()[-1]) if metrics.exists() else None,
            checkpoint_results={path.parent.name:json.loads(path.read_text()) for path in sorted((root/'evaluation').glob('*/aggregate.json'))},
            error=state.get('error'),sampled_at=time.time())
print(json.dumps(result,ensure_ascii=False))
