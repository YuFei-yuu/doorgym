import av,sys,math
from pathlib import Path
from PIL import Image,ImageDraw
run=Path(sys.argv[1]);out=run/'review';old=Path('/data/zyf/video2door-wc4-act/.repro/runs/wc4-close-inplace-stage7-20261027');times=[17.52,18.92,21.,24.,24.80,25.40,34.40]
w=800;h=round(w/3);sheet=Image.new('RGB',(2*w,(h+28)*len(times)),(245,245,245));draw=ImageDraw.Draw(sheet)
for col,p in enumerate([old,run]):
    c=av.open(str(p/'attempt_000/overview/env_0000/rollout.mp4'));s=c.streams.video[0]
    for j,t in enumerate(times):
        c.seek(int(t/s.time_base),stream=s,backward=True);im=None
        for f in c.decode(s):
            if float(f.pts*s.time_base)>=t-.025:im=f.to_image();break
        if im is None:continue
        sheet.paste(im.resize((w,h)),(col*w,j*(h+28)));draw.text((col*w+8,j*(h+28)+h+6),('Stage 7' if col==0 else 'Stage 8')+' original RGB | video %.2f s'%t,fill='black')
    c.close()
sheet.save(out/'parallel-fold-keyframes.jpg',quality=93)
