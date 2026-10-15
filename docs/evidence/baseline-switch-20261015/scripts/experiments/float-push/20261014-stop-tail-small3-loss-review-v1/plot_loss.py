"""Read-only loss review; original run and logs are never modified."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
source, output = Path(args.input), Path(args.output)
output.mkdir(parents=True, exist_ok=True)
digest = hashlib.sha256(source.read_bytes()).hexdigest()
rows = [json.loads(line) for line in source.read_text().splitlines()]
steps = np.asarray([row['train/step'] for row in rows])
loss = np.asarray([row['train/loss'] for row in rows])
assert np.array_equal(steps, np.arange(10, 1001, 10)), 'Missing/duplicate/out-of-order metrics'
assert np.isfinite(loss).all() and (loss > 0).all()
smooth = np.full(len(loss), np.nan)
smooth[4:] = np.convolve(loss, np.ones(5)/5, mode='valid')

def describe(low, high):
    values = loss[(steps >= low) & (steps <= high)]
    return dict(logged_batches=len(values), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()), std=float(values.std()))

stats = dict(input=str(source), input_sha256=digest, samples=len(rows), logged_every_updates=10,
    loss_definition='Per-logged-batch DDPM epsilon prediction MSE, F.mse_loss(pred, noise); no validation loss',
    smoothing='Trailing arithmetic mean of5 logged batches; raw points retained; no smoothing of original data',
    first_logged_loss=float(loss[0]), final_logged_loss=float(loss[-1]),
    checkpoints={str(step): dict(batch_loss=float(loss[steps == step][0]), trailing5_mean=float(smooth[steps == step][0])) for step in [250, 500, 1000]},
    intervals={label: describe(low, high) for label, low, high in [('10-100',10,100),('610-800',610,800),('810-1000',810,1000)]},
    learning_rates=sorted(set(row['train/lr'] for row in rows)),
    optimizer_seconds=rows[-1]['train/elapsed_sec'])
stats['late_mean_change_percent'] = 100*(stats['intervals']['810-1000']['mean']/stats['intervals']['610-800']['mean']-1)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), constrained_layout=True)
for ax in axes:
    ax.plot(steps, loss, color='#8ba8bf', marker='o', markersize=2.7, linewidth=.9, alpha=.85, label='Logged batch (every 10 updates)')
    ax.plot(steps, smooth, color='#125875', linewidth=2.1, label='Trailing mean of 5 logged batches')
    ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Epsilon prediction MSE')
    ax.grid(True, alpha=.2)
    for step in [250, 500, 1000]:
        ax.axvline(step, color='#8d6d4e', linestyle=':', linewidth=1)
    ax.set_xticks([100,250,500,750,1000])
axes[0].set_yscale('log'); axes[0].set_xlim(0, 1020)
axes[0].yaxis.set_major_locator(FixedLocator([.02, .05, .1, .2, .5]))
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, pos: '%g' % value))
axes[0].set_title('Entire run (logarithmic loss axis)')
axes[1].set_xlim(100,1020)
tail = loss[steps >= 100]
axes[1].set_ylim(0, float(tail.max()*1.1))
axes[1].set_title('Updates 100-1000 (linear loss axis)')
axes[0].legend(loc='upper right', fontsize=8, frameon=False)
for step in [250, 500, 1000]:
    value = float(loss[steps == step][0])
    axes[0].scatter([step], [value], s=35, color='#be5c35', zorder=5)
    axes[0].annotate('%d: %.4f' % (step,value), (step,value), xytext=(-12,14), textcoords='offset points',
                     ha='right' if step == 1000 else 'center', fontsize=8, color='#984425')
fig.suptitle('Stop-tail small3: seed 42 | batch 64 | lr=1e-4 | 1000 updates', fontsize=13)
fig.savefig(output/'loss_review.png', dpi=220)
fig.savefig(output/'loss_review.svg')
plt.close(fig)
assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, 'Original metrics changed'
(output/'loss_statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
print(json.dumps(stats))
