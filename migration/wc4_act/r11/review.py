"""Re-score every R8/R9 trial, writing exclusively to new R11 evidence."""
import json,pathlib,collections,math
import av
import numpy as np
from PIL import Image,ImageDraw
from criteria import score
ROOT=pathlib.Path('/data/zyf/video2door-wc4-act')
RUN=ROOT/'.repro/runs/wc4-fulltask-reaudit-r11-20261021'


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')


def main():
    old=read(ROOT/'.repro/runs/wc4-fulltask-expand-r9-20261021/candidate-index.json')['episodes']
    accepted_seeds={e['seed']:i for i,e in enumerate(old)}
    ledger=[]
    for source,count in [('wc4-fulltask-pilot-r8-20261021',8),('wc4-fulltask-expand-r9-20261021',64)]:
      for trial in range(count):
        p=ROOT/'.repro/runs'/source/('trial_%02d'%trial)
        frozen=read(p/'frozen-trial.json');seed=frozen['seed'];out=RUN/('r%d-trial_%02d'%(8 if count==8 else 9,trial));out.mkdir(exist_ok=False)
        states=[json.loads(l) for l in (p/'states.jsonl').open()]
        contacts=0;coupled=[];pairs=collections.Counter()
        for i,line in enumerate((p/'contacts.jsonl').open()):
          row=json.loads(line);assert row['step']==i
          x=states[i];x.update(end_panel_raw_contact=False,loaded_door_contact=False,zero_lambda_door_records=0,nonend_door_records=0)
          assert abs(row['interval_s'][1]-x['t'])<1e-8
          for c in row['contacts']:
            names=c['names'];door=[a for a in names if a.get('actor_handle')==2]
            robot=[a for a in names if a.get('actor_handle') in [0,1]]
            if door and robot:
              panel=door[0]['body']=='door_panel';end=robot[0]['body'] in ['gripperStator','gripperMover']
              x['end_panel_raw_contact']|=panel and end
              x['loaded_door_contact']|=c['lambda']!=0
              x['zero_lambda_door_records']+=c['lambda']==0
              x['nonend_door_records']+=not end
              pairs[robot[0]['body']+' / '+door[0]['body']]+=1
          contacts+=1
        assert contacts==len(states)
        locks=[json.loads(l) for l in (p/'lock-writes.jsonl').open()]
        result=score(states,locks,read(p/'reference.json'))
        result.update(source_run=source,trial=trial,seed=seed,source_dir=str(p),old_candidate_episode=accepted_seeds.get(seed),
            was_isolated=seed not in accepted_seeds,automatic_terminal_pass=read(p/'result.json').get('automatic_final_conditions_met',False),
            recording_checks_pass=read(p/'record-validation.json')['recording_checks_pass'],
            source_record_validation=str(p/'record-validation.json'),raw_npz=str(p/'raw/episode_000000.npz'),
            frames=read(p/'record-validation.json')['frames'],raw_contact_pairs=dict(pairs),
            visual_review='pending',complete_task_review='pending',data_eligible='pending')
        write(out/'measured-review.json',result)
        # Original RGB at event times and densely during push. No simulator or replay.
        push=result['aligned_end_push_steps'];k0=push[0] if push else 0;k1=push[-1] if push else len(states)-1
        latch=next((x['step'] for x in locks if x['kind']=='relocked'),len(states)-1)
        chosen={0,(result['whole_body_pass_step'] or 0)//2,(result['old_contact_separation_step'] or 0)//2,latch//2,(len(states)-1)//2}
        chosen.update(int(x)//2 for x in np.linspace(k0,k1,7));chosen.update([max(0,k0//2-12),min((len(states)-1)//2,k1//2+25)])
        tiles=[];fps=None
        with av.open(str(p/'overview/env_0000/rollout.mp4')) as f:
          fps=float(f.streams.video[0].average_rate);n=0
          for n,frame in enumerate(f.decode(video=0)):
            if n in chosen:
              im=frame.to_image();im.thumbnail((640,360));tile=Image.new('RGB',(640,385),'white');tile.paste(im,(0,25))
              ImageDraw.Draw(tile).text((5,5),'R%d trial%02d ORIGINAL frame%d t=%.2fs'%(8 if count==8 else 9,trial,n,states[min(2*n,len(states)-1)]['t']),fill='black');tiles.append(tile)
          frames=n+1
        assert fps==25 and frames==result['frames']
        sheet=Image.new('RGB',(1920,385*math.ceil(len(tiles)/3)),'white')
        for j,im in enumerate(tiles):sheet.paste(im,((j%3)*640,(j//3)*385))
        sheet.save(out/'original-evidence-sheet.jpg',quality=90)
        result['original_video_decode']=dict(frames=frames,fps=fps,complete=True)
        write(out/'measured-review.json',result);ledger.append(result)
        print(source,trial,result['measured_task_gates_pass'],result['failures'],flush=True)
    write(RUN/'measured-ledger.json',dict(trials=ledger,old_candidates=46,isolated=26))


if __name__=='__main__':main()
