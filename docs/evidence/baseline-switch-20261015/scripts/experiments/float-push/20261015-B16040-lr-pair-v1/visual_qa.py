"""Inspect frames of completed comparison videos only."""
import json,sys
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
r=Path(sys.argv[1]);out=r/'visual_qa';out.mkdir(exist_ok=True)
for t in [25.9,49.9]:
    panels=[]
    for step in [32000,35000,40000]:
        row=[]
        for label in ['C','L']:
            p=r/label/'comparison'/('step_%06d'%step)/'expert_policy.mp4'
            cap=cv2.VideoCapture(str(p));cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read();cap.release()
            assert ok,(label,step,t)
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((640,280))
            panel=Image.new('RGB',(640,im.height+24),'white');panel.paste(im,(0,24));ImageDraw.Draw(panel).text((8,5),'%s %dk  true time %.2fs (reference | policy)'%(label,step//1000,t),fill='black');row.append(panel)
        panels.append(row)
    height=sum(max(p.height for p in row) for row in panels);grid=Image.new('RGB',(1280,height),'white');y=0
    for row in panels:
        for col,p in enumerate(row):grid.paste(p,(col*640,y))
        y+=max(p.height for p in row)
    grid.save(out/('comparison_frames_t%.1f.png'%t))
print('Two six-video inspection grids saved')
