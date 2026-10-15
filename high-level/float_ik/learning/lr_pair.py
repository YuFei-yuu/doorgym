"""Paired exact-state continuation; only optimizer LR differs, +10k each."""
import argparse,copy,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import smallbatch30k as base
from branch16040 import completed_stage,cpu_stage,gpu_stage
from diagnostic_integrity import sha256,source_paths,phase_lock
HERE=Path(__file__).resolve().parent;REPO=base.REPO
OLD=REPO/'experiments/float-push/20261015-B16040-batch8-random30k-v1'
SOURCE=OLD/'training/checkpoints/step_030000.pt'
EXPECTED='a75adbc811384e4ae6fc81d64b0f0cba4fcb9525d1371a4aa2bea2216ec75a8d'
NODES=[32000,35000,40000];LRS={'C':1e-4,'L':1e-5};save=base.save

def prepare(root):
    if sha256(SOURCE)!=EXPECTED:raise ValueError('Source checkpoint SHA differs')
    inputs,dirs={},{};base.shared.register_input(OLD,inputs,dirs)
    sources={str(p.relative_to(REPO)):sha256(p) for p in source_paths(REPO)}
    for name in sources:
        target=root/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,target)
    plan=json.loads((OLD/'plan.json').read_text());plan.update(source_hashes=sources,input_hashes=inputs,input_directories=dirs,
        experiment_kind='paired_complete_resume_lr_only',source_checkpoint=str(SOURCE),source_sha256=EXPECTED,
        nodes=NODES,source_updates=30000,added_updates=10000,lrs=LRS,max_updates=40000)
    save(root/'plan.json',plan)
    for label in LRS:
        r=root/label;r.mkdir();(r/'logs').mkdir();save(r/'plan.json',plan)
        shutil.copytree(OLD/'raw',r/'raw');shutil.copytree(OLD/'reference',r/'reference')
    (root/'initial_git_status.txt').write_bytes(base.shared.git('status','--short','--branch'))
    (root/'initial_worktree.diff').write_bytes(base.shared.git('diff','--binary','HEAD'))
    base.verify(root)

def data_check(root):
    import torch
    from train_door_dp import DoorDPSequenceDataset,_as_chw_u8
    from action_contract import FORWARD_V3,canonicalize_sequence
    cp=torch.load(SOURCE,map_location='cpu',weights_only=False)
    data=Path(cp['config']['root']);hashes=json.loads((OLD/'data_hashes.json').read_text())
    if any(sha256(data/n)!=h for n,h in hashes.items()):raise ValueError('Accepted converted data changed')
    ds=DoorDPSequenceDataset(data,cp['config']['repo_id'],16,32,FORWARD_V3)
    if (len(ds),ds.length)!=(1415,1446):raise ValueError('Data dimensions differ')
    with np.load(OLD/'raw/episode_000000.npz',allow_pickle=True) as z:
        states=canonicalize_sequence(z['state'],slice(-5,-1));actions=canonicalize_sequence(z['action'],slice(5,9))
        images={key:z[key].copy() for key in base.IMAGES}
        for j,c in enumerate(ds.indices):
            item=ds[j];ids=np.maximum(0,np.arange(c-15,c+1))
            if not np.array_equal(item['state'].numpy(),states[ids]) or not np.array_equal(item['action'].numpy(),actions[c:c+32]):raise ValueError('Window alignment differs')
            for key,out in zip(base.IMAGES,['mask','masked_depth','front_mask','front_masked_depth']):
                if not np.array_equal(item[out].numpy(),images[key][ids].transpose(0,3,1,2)):raise ValueError('Window image differs')
    from train_door_dp import exact_state
    if not exact_state(cp['stats'],torch.load(OLD/'training_only_stats.pt',weights_only=False)):raise ValueError('Source statistics mismatch')
    save(root/'input_acceptance.json',dict(passed=True,frames=1446,windows=1415,all_actual_windows_checked=True,converted_hashes_exact=True,
        stats_inherited_exact=True,stats_recomputed=False,source_sha256=sha256(SOURCE)))

def command(r,step,audit=False):
    lr=LRS[r.name];previous=SOURCE if step==32000 else r/'training/checkpoints'/('step_%06d.pt'%NODES[NODES.index(step)-1])
    args=['python','-u',str(REPO/'high-level/dp/train_door_dp.py'),'--root',str(OLD/'data/lerobot/local/single_B16040'),
        '--repo_id','local/single_B16040','--output_dir',str(r/('restore_audit' if audit else 'training')),
        '--run_name',r.parent.name+'-'+r.name,'--steps',str(step),'--batch_size','8','--num_workers','0','--lr',str(lr),'--seed','42',
        '--action_contract','forward_v3','--obs_horizon','16','--pred_horizon','32','--action_horizon','16','--num_diffusion_iters','100',
        '--log_interval','10','--save_interval',str(step),'--recording_metadata',str(OLD/'recording_contract.json'),
        '--stop_on_nonfinite_loss','--resume',str(previous)]
    if step==32000 and r.name=='L':args+=['--resume_lr_from','0.0001']
    if audit:args+=['--resume_audit_only']
    elif step==32000:args+=['--verify_restored_state',str(r/'restore_audit/checkpoints/step_030000.pt')]
    return args

def pair_check(root,step):
    import torch
    from train_door_dp import exact_state
    source=torch.load(SOURCE,map_location='cpu',weights_only=False);cs=[];records=[]
    for label,lr in LRS.items():
        r=root/label;p=r/('restore_audit' if step==30000 else 'training')/'checkpoints'/('step_%06d.pt'%step)
        c=torch.load(p,map_location='cpu',weights_only=False);cs.append(c)
        if c['config']['completed_updates']!=step or c['config']['lr']!=lr:raise ValueError('Step/LR mismatch')
        if not exact_state(c['stats'],source['stats']):raise ValueError('Normalization changed')
        if len(c['optimizer']['state'])!=130 or any(int(v['step'])!=step for v in c['optimizer']['state'].values()):raise ValueError('AdamW steps not complete')
        values=list(c['model'].values())+[v for q in c['optimizer']['state'].values() for v in q.values() if isinstance(v,torch.Tensor)]
        if any(not torch.isfinite(v).all() for v in values):raise FloatingPointError('Nonfinite trained state')
        if any(g['lr']!=lr for g in c['optimizer']['param_groups']):raise ValueError('Effective LR differs')
        if step==30000:
            opt=copy.deepcopy(c['optimizer'])
            for g in opt['param_groups']:g['lr']=1e-4
            if not exact_state(opt,source['optimizer']) or not exact_state(c['model'],source['model']) or not exact_state(c['training_state'],source['training_state']):raise ValueError('Full source restoration failed')
        records.append(dict(branch=label,sha256=sha256(p),step=step,lr=lr,optimizer_states=130,sampler_cursor=c['training_state']['sampler']['cursor']))
    for k in ['sampler','rng']:
        if not exact_state(cs[0]['training_state'][k],cs[1]['training_state'][k]):raise ValueError('Paired '+k+' differs')
    save(root/('pair_%06d.json'%step),dict(passed=True,records=records,paired_sampler_and_rng_exact=True,
        source_model_optimizer_exact_except_lr=step==30000))

def run(root,repair=False):
    if not repair:
        root.mkdir(parents=True);(root/'logs').mkdir();prepare(root)
    else:
        # Recover engineering failures only, never repeat successful compute.
        trained=any((root/l/'training/metrics.jsonl').exists() for l in LRS)
        plan=json.loads((root/'plan.json').read_text());changed={n:sha256(REPO/n) for n,h in plan['source_hashes'].items() if sha256(REPO/n)!=h}
        allowed={str((HERE/n).relative_to(REPO)) for n in ['lr_pair_online.py','lr_pair_analysis.py']}
        if trained and set(changed)-allowed:raise ValueError('Cannot change training dependencies after start')
        rev=root/'revisions'/str(time.time_ns());rev.mkdir(parents=True);save(rev/'prior_plan.json',plan)
        for n in changed:
            p=rev/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,p)
        save(rev/'changes.json',changed);plan['source_hashes'].update(changed)
        for r in [root,*[root/l for l in LRS]]:
            save(r/'plan.json',plan)
            for n in ['failure.json','completion.json']:
                if (r/n).exists():shutil.copy2(r/n,rev/(r.name+'_'+n));(r/n).unlink()
    loggers=[]
    try:
        with phase_lock(root,'lr_pair'):
            cpu_stage(root,'tests',['-m','unittest','discover','-s',REPO/'high-level/dp','-p','test_lr_resume.py','-v'])
            cpu_stage(root,'input_check',[__file__,'--root',root,'--worker','data'])
            for label in LRS:gpu_stage(root/label,'full_restore_preflight',command(root/label,32000,True),environment='train')
            cpu_stage(root,'pair_30000',[__file__,'--root',root,'--worker','pair','--step',30000])
            for label in LRS:
                r=root/label;log=(r/'logs'/('online_'+str(time.time_ns())+'.log')).open('x')
                proc=subprocess.Popen([str(REPO/'.repro/envs/door-dp-train/bin/python'),'-u',str(HERE/'lr_pair_online.py'),'--root',str(r)],stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
                loggers.append((proc,log));deadline=time.time()+180
                while not (r/'online/ready.json').exists():
                    if proc.poll() is not None or time.time()>deadline:raise RuntimeError('Online preflight failed '+label)
                    time.sleep(3)
            cpu_stage(root,'baseline_variation',[HERE/'lr_pair_analysis.py','--root',root,'--baseline'])
            save(root/'state.json',dict(phase='ready_links',links={l:json.loads((root/l/'online/ready.json').read_text())['url'] for l in LRS}))
            while not (root/'links_delivered.json').exists():time.sleep(2)
            for r in [root,*[root/l for l in LRS]]:
                if not (r/'training_source_freeze.json').exists():save(r/'training_source_freeze.json',dict(at=time.time(),source_hashes=json.loads((r/'plan.json').read_text())['source_hashes']))
            for step in NODES:
                for label in LRS:
                    r=root/label;base.verify(r);save(root/'state.json',dict(phase='training',branch=label,cumulative_target=step,added_target=step-30000))
                    gpu_stage(r,'train_%d'%step,command(r,step),environment='train')
                cpu_stage(root,'pair_%d'%step,[__file__,'--root',root,'--worker','pair','--step',step])
                for label in LRS:
                    r=root/label;save(root/'state.json',dict(phase='evaluation',branch=label,cumulative_step=step))
                    gpu_stage(r,'offline_%d'%step,['python','-u',str(HERE/'smallbatch30k_analysis.py'),'--root',r,'--step',str(step),'--offline'],environment='gym')
                    cpu_stage(r,'variation_%d'%step,[HERE/'lr_pair_analysis.py','--root',r,'--step',step])
                    base.evaluate(r,step)
                    cpu_stage(r,'comparison_%d'%step,[HERE/'smallbatch30k_analysis.py','--root',r,'--step',step,'--closed'])
            base.verify(root);save(root/'final_input_preservation.json',base.shared.verify_inputs(json.loads((root/'plan.json').read_text())))
            for r in [root,*[root/l for l in LRS]]:save(r/'completion.json',dict(phase='completed',added_updates=10000,cumulative_updates=40000,new_rollouts=6,auto_expand=False))
            for proc,_ in loggers:
                proc.wait(timeout=120)
                if proc.returncode:raise RuntimeError('Online final verification failed')
            save(root/'state.json',dict(phase='completed',added_updates_each=10000,new_rollouts=6))
    except BaseException as exc:
        for r in [root,*[root/l for l in LRS]]:save(r/'completion.json',dict(phase='failed_stopped',error=repr(exc)))
        save(root/'failure.json',dict(error=repr(exc),at=time.time()));raise
    finally:
        for proc,log in loggers:
            if proc.poll() is None:
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=15)
            log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--worker',choices=['data','pair']);p.add_argument('--step',type=int);p.add_argument('--continue-repair',action='store_true');a=p.parse_args();r=Path(a.root).resolve()
    if a.worker=='data':data_check(r)
    elif a.worker=='pair':pair_check(r,a.step)
    else:run(r,a.continue_repair)
