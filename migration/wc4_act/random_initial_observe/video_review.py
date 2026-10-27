import av,json,sys,math
from pathlib import Path
from PIL import Image,ImageDraw
run=Path(sys.argv[1]);out=run/'observation';comparison=json.loads((out/'comparison.json').read_text());receipts={}
for label,report in comparison.items():
 p=Path(report['source']);video=p/'overview/env_0000/rollout.mp4';events=json.loads((out/(label+'-events.json')).read_text());times=[0,max(0,report['result']['sim_seconds']-.04)]
 for phase in report['phases']:
  if phase['phase'] in ['push_door','turn','close','release','restore_heading','final_hold']:times.extend([phase['start_s'],phase['end_s']])
 for side,parts in report['feet'].items():
  for part,v in parts.items():
   if v['max_deviation_rad']>(.08 if part=='hip' else .15):times.extend([max(0,v['max_time_s']-.5),v['max_time_s']])
 n_expected=(report['result']['steps']+1)//2;indices={min(n_expected-1,max(0,round(t*25))):t for t in times};selected=[]
 c=av.open(str(video));stream=c.streams.video[0];fps=float(stream.average_rate);n=0
 for i,f in enumerate(c.decode(stream)):
  n+=1;size=[f.width,f.height]
  if i in indices:
   im=f.to_image();im.save(out/(label+'-live-%06d.png'%i));selected.append((i,im))
 c.close();w=640;h=round(w*size[1]/size[0]);sheet=Image.new('RGB',(w*2,(h+28)*math.ceil(len(selected)/2)),'white');draw=ImageDraw.Draw(sheet)
 for j,(i,im) in enumerate(selected):
  x=(j%2)*w;y=(j//2)*(h+28);sheet.paste(im.resize((w,h)),(x,y));draw.text((x+5,y+h+5),label+' LIVE original | frame %d | %.2fs'%(i,i/25),fill='black')
 sheet.save(out/(label+'-live-sheet.jpg'),quality=90)
 receipts[label]=dict(path=str(video),decoded_frames=n,expected_frames=n_expected,fps=fps,size=size,duration_s=n/fps,count_matches=n==n_expected,all_decoded=True)
 assert n==n_expected and fps==25
(out/'video-validation.json').write_text(json.dumps(receipts,indent=2)+'\n');print(json.dumps(receipts,indent=2))
