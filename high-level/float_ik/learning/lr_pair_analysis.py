"""Fixed-observation prediction mean bias and population variation across three fixed seeds."""
import argparse,json
from pathlib import Path
import numpy as np
from lr_pair import OLD,save

def variation(folder,plan):
    with np.load(folder/'predictions.npz') as z:
        print('prediction keys',z.files,flush=True)
        pred=z['prediction'];target=z['target'];centers=z['center'];seeds=z['noise_seed']
    rows=[]
    for stage in ['all',*dict.fromkeys(w['stage'] for w in plan['windows'])]:
        selected=[w['center'] for w in plan['windows'] if stage=='all' or w['stage']==stage]
        for horizon in [16,32]:
            draws=[];truth=[]
            for center in selected:
                ids=np.flatnonzero(centers==center)
                if len(ids)!=3 or set(seeds[ids].tolist())!=set(plan['diffusion_seeds']):raise ValueError('Unpaired fixed diffusion draws')
                ids=sorted(ids,key=lambda i:int(seeds[i]));draws.append(pred[ids,:horizon,:]);truth.append(target[ids[0],:horizon,:])
                if not all(np.array_equal(target[i],target[ids[0]]) for i in ids):raise ValueError('Targets differ across seeds')
            draws=np.asarray(draws);truth=np.asarray(truth);mean=draws.mean(axis=1);variance=((draws-mean[:,None])**2).mean(axis=1)
            row=dict(stage=stage,horizon=horizon,windows=len(selected),seeds=plan['diffusion_seeds'],population_ddof=0)
            for j,name in [(0,'vx'),(1,'yaw')]:
                err=mean[:,:,j]-truth[:,:,j]
                row[name]=dict(mean_prediction_rmse=float(np.sqrt(np.mean(err**2))),signed_mean_error=float(err.mean()),
                    seed_variation_rms_std=float(np.sqrt(variance[:,:,j].mean())),
                    per_seed_rmse=[float(np.sqrt(np.mean((draws[:,k,:,j]-truth[:,:,j])**2))) for k in range(3)],
                    draw_rmse=float(np.sqrt(np.mean((draws[:,:,:,j]-truth[:,None,:,j])**2))))
            rows.append(row)
    save(folder/'bias_variation.json',dict(metrics=rows,definition='Three fixed inference seeds only; mean bias and population seed variance separated; no generalization confidence claim'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--step',type=int);p.add_argument('--baseline',action='store_true');a=p.parse_args();r=Path(a.root)
    if a.baseline:
        folder=r/'baseline';folder.mkdir(exist_ok=True)
        import shutil
        for n in ['predictions.npz','summary.json']:shutil.copy2(OLD/'offline/step_030000'/n,folder/n)
        variation(folder,json.loads((r/'plan.json').read_text()))
        save(folder/'provenance.json',dict(source=str(OLD),offline_reused=True,simulation_reused=True,no_new_baseline_simulation=True))
    else:variation(r/'offline'/('step_%06d'%a.step),json.loads((r/'plan.json').read_text()))
