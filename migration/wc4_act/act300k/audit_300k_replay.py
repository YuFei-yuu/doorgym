"""Decode complete replay videos and save sparse visual QA contact sheets."""
import argparse,json,hashlib
from pathlib import Path
import av
from PIL import Image,ImageDraw

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();results=[]
 for name in ['300000-success','300000-failure']:
  d=a.root/name;video=d/'overview/rollout.mp4';samples=[];times=[]
  with av.open(str(video)) as c:
   s=c.streams.video[0];rate=float(s.average_rate);assert rate==25
   for i,f in enumerate(c.decode(s)):
    assert (f.width,f.height)==(1920,640)
    times.append(float(f.pts*f.time_base))
    if i in [0,250,500,1000,1250,2999]:
     im=f.to_image().resize((960,320));ImageDraw.Draw(im).text((8,45),'%s | %.2fs'%(name,i/25),fill='yellow');samples.append(im)
  assert len(times)==3000 and all(abs(t-i/25)<1e-6 for i,t in enumerate(times))
  sheet=Image.new('RGB',(960,320*len(samples)))
  for i,im in enumerate(samples):sheet.paste(im,(0,i*320))
  sheet.save(d/'contact-sheet.jpg',quality=90)
  receipt=json.loads((d/'provenance.json').read_text())
  assert receipt['frames']==3000 and receipt['physics_integration_steps']==0 and receipt['max_body_pose_restore_error']<1e-5
  results.append(dict(name=name,seed=receipt['seed'],success=receipt['original_result']['success'],video=str(video),sha256=sha(video),bytes=video.stat().st_size,decoded_frames=len(times),fps=rate,duration_s=120,width=1920,height=640,pts_uniform=True))
 (a.root/'video-audit.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
