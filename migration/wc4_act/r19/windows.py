"""Read existing expert evidence only; freeze four equally weighted event pools."""
import pathlib,json,hashlib,collections,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3]
R11=ROOT/'.repro/runs/wc4-fulltask-reaudit-r11-20261021'
RUN=ROOT/'.repro/runs/wc4-fulltask-act-r19-windows-s1000-20261023'
LABELS=['fold_to_bypass','bypass_to_rendezvous','rendezvous_to_seek','first_measured_end_push']
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 RUN.mkdir(exist_ok=False)
 prov=read(R11/'unified-provenance.json')['episodes'];ledger={(x['source_run'],x['trial']):x for x in read(R11/'measured-ledger.json')['trials']}
 events=[];pools=[[] for _ in LABELS];offset=0;sources={};overlaps=[]
 for ep,e in enumerate(prov):
  p=R11/'candidates'/f'episode_{ep:06d}.npz';assert sha(p)==e['sha256'];sources[str(p)]=e['sha256']
  x=ledger[e['source_run'],e['trial']];d=pathlib.Path(x['source_dir']);assert x['seed']==e['seed'] and x['measured_task_gates_pass']
  with np.load(p,allow_pickle=True) as a:
   phase=a['subtask_index'].reshape(-1);names=list(a['phase_names']);n=len(phase);assert n==e['frames']
   boundaries=[]
   for prev,nxt in [('compact_fold','compact_bypass'),('compact_bypass','compact_rendezvous'),('compact_rendezvous','compact_seek')]:
    ids=np.flatnonzero(phase==names.index(nxt));assert len(ids)>0
    f=int(ids[0]);assert f>0 and names[int(phase[f-1])]==prev
    boundaries.append(f)
   clock=[json.loads(l) for l in (d/'recording-clock.jsonl').open()];assert len(clock)==n
   assert all(c['frame']==i and c['source_step']==2*i for i,c in enumerate(clock))
   push=x['first_aligned_end_push_step'];assert push is not None
   f=next(i for i,c in enumerate(clock) if c['source_step']>=push);assert 0<=clock[f]['source_step']-push<=1
   boundaries.append(f)
   assert boundaries==sorted(boundaries) and len(set(boundaries))==4
   # Source state at all four anchors, plus actual first push, independently bound.
   needed={clock[f]['source_step'] for f in boundaries}|{push};states={}
   for line in (d/'states.jsonl').open():
    s=json.loads(line)
    if s['step'] in needed:states[s['step']]=s
   assert len(states)==len(needed)
   first=states[clock[boundaries[0]]['source_step']];assert first['passed']
   s=states[push];assert s['end_panel_contact'] and s['passed'] and s['door_q'][0]<s['pre_physics']['door_q'][0]
   assert names[int(phase[f])] in ['compact_seek','close']
   for path in [d/'recording-clock.jsonl',d/'states.jsonl',R11/'unified-provenance.json',R11/'measured-ledger.json']:
    if str(path) not in sources:sources[str(path)]=sha(path)
   sets=[]
   for label,center,pool in zip(LABELS,boundaries,pools):
    lo=max(0,center-25);hi=min(n-1,center+25);indices=list(range(offset+lo,offset+hi+1));pool.extend(indices);sets.append(set(indices))
    events.append(dict(episode=ep,seed=e['seed'],label=label,frame=center,index=offset+center,time_s=clock[center]['physics_post_time_s'],source_step=clock[center]['source_step'],phase=names[int(phase[center])],lo=lo,hi_inclusive=hi,frames=len(indices),clipped=(lo!=center-25 or hi!=center+25),indices=indices,measured_anchor={k:states[clock[center]['source_step']][k] for k in ['actual_base','arm_q','passed','door_q']},first_push_step=push if label==LABELS[-1] else None))
   overlaps.append(dict(episode=ep,pairwise={LABELS[i]+' / '+LABELS[j]:len(sets[i]&sets[j]) for i in range(4) for j in range(i+1,4)}))
  offset+=n;print('events',ep,boundaries,flush=True)
 assert len(prov)==72 and offset==159529
 out=dict(definitions=dict(zip(LABELS,['first recorded compact_bypass after compact_fold: measured fold-completion/turn onset','first recorded compact_rendezvous after compact_bypass: route endpoint reached, final approach/deployment begins','first recorded compact_seek after compact_rendezvous: base arrived and arm hover deployment reached; end approach begins','first R11 aligned measured end-panel contact with measured decreasing door angle; map to first 25Hz sample at/after its 50Hz step (<=20ms)'])),window='inclusive center +/-25 frames at25Hz; clip only episode endpoints; supervision remains original chunk100 with original episode padding',mixture='probability .5 uniform all159529 frames, .125 per event class uniform over its window-frame occurrences; overlap keeps class membership and additive mass',labels=LABELS,events=events,pools=pools,pool_sizes=list(map(len,pools)),overlaps=overlaps,clipped_windows=sum(x['clipped'] for x in events),unique_window_frames=len(set(sum(pools,[]))),sources=sources,episodes=72,frames=offset,frozen_at=time.time(),new_model_results_seen=False)
 (RUN/'windows.json').write_text(json.dumps(out,indent=2)+'\n');print('SUMMARY', {k:out[k] for k in ['pool_sizes','clipped_windows','unique_window_frames']},flush=True)
 (RUN/'WINDOWS-FROZEN.json').write_text(json.dumps(dict(sha256=sha(RUN/'windows.json'),frozen_at=time.time(),training_updates=0,model_forwards=0,scenes=0),indent=2)+'\n')
if __name__=='__main__':main()
