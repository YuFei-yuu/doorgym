"""Refresh training/checkpoint tables and complete rollout diagnostic plots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def rollout_plots(folder):
    path=folder/'execution.jsonl'
    if not path.exists(): return
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    if not rows: return
    t=np.array([r['sim_time'] for r in rows]);base=np.array([r['base_after'] for r in rows])
    angles=-np.rad2deg([r['door_after'][0] for r in rows])
    raw=np.array([r['dp_action_raw'] for r in rows]);applied=np.array([r['applied_action'] for r in rows])
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axs[0].plot(t,angles);axs[0].axhline(80,color='red',linestyle='--');axs[0].set(xlabel='Simulation time (s)',ylabel='Signed door angle (degrees)')
    axs[1].plot(base[:,0],base[:,1]);axs[1].scatter(base[0,0],base[0,1],label='Start')
    for r in rows:
        if r['extra'].get('pass_started'):
            axs[1].scatter(*r['extra']['through_target'],label='Through target');break
    axs[1].set(xlabel='World X (m)',ylabel='World Y (m)');axs[1].axis('equal');axs[1].legend()
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(folder/'door_and_path.png',dpi=140);plt.close(fig)
    labels=['vx (m/s)','yaw rate (rad/s)','EE x (m)','EE y (m)','EE z (m)','qx','qy','qz','qw','gripper (rad)']
    fig,axs=plt.subplots(5,2,figsize=(12,11),sharex=True,constrained_layout=True)
    for i,ax in enumerate(axs.flat):
        ax.plot(t,raw[:,i],label='DP raw');ax.plot(t,applied[:,i],'--',label='Applied')
        ax.set_ylabel(labels[i]);ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.savefig(folder/'actions_10d.png',dpi=130);plt.close(fig)


def report(root, suite=None):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True)
    metrics=root/'training/metrics.jsonl'
    if metrics.exists():
        train=[json.loads(x) for x in metrics.read_text().splitlines()]
        write_csv(root/'training/metrics.csv',train)
        fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for ax,key,label in zip(axs,['train/loss','train/lr','train/elapsed_sec'],
                               ['Epsilon noise MSE','Learning rate','Training elapsed (s)']):
            ax.plot([r['train/step'] for r in train],[r[key] for r in train])
            ax.set(xlabel='Completed updates',ylabel=label);ax.grid(alpha=.2)
        temporary=out/'training_curves.tmp.png';fig.savefig(temporary,dpi=150);plt.close(fig)
        temporary.replace(out/'training_curves.png')
    suites=[]
    for path in sorted((root/'evaluation').glob('*/aggregate.json')):
        result=json.loads(path.read_text());result['suite']=path.parent.name;suites.append(result)
        if suite==path.parent.name:
            rows=json.loads((path.parent/'results.json').read_text());write_csv(path.parent/'results.csv',rows)
            for r in rows:rollout_plots(Path(r['folder']))
            fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
            ax.bar([str(r['seed']) for r in rows],[r['max_signed_door_deg'] or 0 for r in rows],
                   color=['seagreen' if r['success'] else 'tomato' for r in rows])
            ax.axhline(80,color='red',linestyle='--');ax.set(xlabel='Predeclared seed',ylabel='Maximum signed angle (degrees)')
            ax.tick_params(axis='x',rotation=60);fig.savefig(path.parent/'task_angles.png',dpi=150);plt.close(fig)
    write_csv(out/'checkpoint_success.csv',suites)
    checkpoint_suites=[r for r in suites if r['suite'].startswith('step_')]
    if checkpoint_suites:
        fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
        steps=[int(r['suite'].split('_')[-1]) for r in checkpoint_suites]
        for key,label in [('opening_rate','Opened >=80 degrees'),('passing_rate','Pass done'),('success_rate','Autonomous joint success')]:
            ax.plot(steps,[r[key] for r in checkpoint_suites],marker='o',label=label)
        ax.set(xlabel='Completed updates',ylabel='Rate',ylim=(-.03,1.03));ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(out/'checkpoint_success.png',dpi=150);plt.close(fig)
    state=json.loads((root/'pipeline_state.json').read_text()) if (root/'pipeline_state.json').exists() else {}
    lines=['# Forward floating-base DP training','',
           'Source baseline: `0acc2f050479fb54a0670ccb90bc72b98b52a54a`.',
           'State: `'+state.get('phase','unknown')+'`. Objective is not complete until nominal >=9/10 and video confirmation.',
           '', '| Checkpoint / set | Opened | Passed | Autonomous successes | Attempts / scheduled |',
           '|---|---:|---:|---:|---:|']
    for r in suites:
        lines.append('| %s | %d | %d | %d | %d / %d |'%(r['suite'],r['opened'],r['passed'],r['successes'],r['attempts'],r['scheduled']))
    lines += ['', 'All attempted episodes, process errors and timeouts stay in the denominator.',
              'Success requires correct-direction 80 degrees, source pass_done within 0.12 m, and no script fallback.',
              'Reverse travel, turning and pass_align frames are diagnostics; full videos determine direct forward passage.',
              'Final test and nominal acceptance are held out until candidate/video review.',
              '', 'See plan.json, recording_contract.json, training/checkpoints, evaluation/*/results.csv and full rollout.mp4 files.']
    temporary=out/'REPORT.md.tmp';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(out/'REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--suite');a=p.parse_args()
    report(a.root,a.suite)
