"""Refresh a local scientific plot when the source 100-update log advances."""
import argparse
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    root=Path(root);last=-1
    while True:
        try:
            rows=[json.loads(s) for s in (root/'training/metrics.jsonl').read_text().splitlines()]
            if rows and rows[-1]['train/step']!=last:
                fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
                steps=[r['train/step'] for r in rows]
                for ax,key,label in zip(axs,['train/loss','train/lr','train/elapsed_sec'],['Source epsilon noise MSE','Learning rate','Training elapsed seconds']):
                    ax.plot(steps,[r[key] for r in rows]);ax.set(xlabel='Optimizer update',ylabel=label);ax.grid(alpha=.2)
                temporary=root/'training/live_curves.tmp.png'
                fig.savefig(temporary,dpi=150);plt.close(fig)
                temporary.replace(root/'training/live_curves.png')
                last=rows[-1]['train/step'];print(json.dumps(dict(plotted_step=last)),flush=True)
            phase=json.loads((root/'pipeline_state.json').read_text())['phase']
            if phase!='training5000':return
        except (FileNotFoundError,json.JSONDecodeError):pass
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
