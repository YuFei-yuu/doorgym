import av,json,sys,math
from pathlib import Path
from PIL import Image,ImageDraw
run=Path(sys.argv[1]);out=run/'review';video=run/'attempt_000/overview/env_0000/rollout.mp4'
events=json.loads((out/'event-index.json').read_text());m=json.loads((out/'metrics.json').read_text())
times=[0,12,17.62]
for p in ['endpoint_hold','turn','deploy_b','approach','close','release','final_hold']:
    x=next((x for x in events if x.get('phase')==p),None)
    if x:times.append(x['time_s'])
times.append(max(0,m['result']['sim_seconds']-.04));times=sorted(set(times));indices={max(0,round(t*25)):t for t in times};selected=[]
c=av.open(str(video));stream=c.streams.video[0];fps=float(stream.average_rate);n=0;size=None
for i,f in enumerate(c.decode(stream)):
    n+=1;size=[f.width,f.height]
    if i in indices:
        im=f.to_image();name='live-%06d.png'%i;im.save(out/name);selected.append((i,im))
c.close();w=640;h=round(w*size[1]/size[0]);sheet=Image.new('RGB',(w*2,(h+30)*math.ceil(len(selected)/2)),(245,245,245));draw=ImageDraw.Draw(sheet)
for j,(i,im) in enumerate(selected):
    x=(j%2)*w;y=(j//2)*(h+30);sheet.paste(im.resize((w,h)),(x,y));draw.text((x+8,y+h+7),'LIVE original RGB | %.2f s | frame %d'%(i/25,i),fill='black')
sheet.save(out/'live-contact-sheet.jpg',quality=90)
receipt=dict(path=str(video),decoded_frames=n,fps=fps,size=size,expected_frames=(m['result']['steps']+1)//2,all_decoded=True,duration_s=n/fps,original_live=True)
receipt['count_matches']=receipt['decoded_frames']==receipt['expected_frames'];(out/'video-validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
