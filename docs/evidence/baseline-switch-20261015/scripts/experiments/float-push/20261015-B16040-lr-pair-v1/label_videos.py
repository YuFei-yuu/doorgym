"""Caption-only derivative, preserving original videos and all completed compute."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import cv2,imageio_ffmpeg
r=Path(sys.argv[1]);sys.path.insert(0,str(Path.cwd()/'high-level/float_ik/learning'));from stop_audit import inspect_video
records=[]
for label,lr in [('C','1e-4'),('L','1e-5')]:
 for step in [32000,35000,40000]:
  original=r/label/'comparison'/('step_%06d'%step)/'expert_policy.mp4';out=original.with_name('expert_policy_labeled.mp4')
  cap=cv2.VideoCapture(str(original));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==1250
  command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1280x360','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
  proc=subprocess.Popen(command,stdin=subprocess.PIPE)
  try:
   for i in range(1250):
    ok,frame=cap.read();assert ok
    cv2.rectangle(frame,(640,0),(1279,43),(0,0,0),-1)
    caption='%s FULL RESUME 30k+%dk lr%s t=%.2fs'%(label,(step-30000)//1000,lr,i/25)
    cv2.putText(frame,caption,(650,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
    proc.stdin.write(frame.tobytes())
  finally:proc.stdin.close();cap.release()
  assert proc.wait()==0
  check=inspect_video(out,1250);assert check['passed']
  records.append(dict(branch=label,cumulative_updates=step,original=str(original),original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
   labeled_video=str(out),labeled_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),check=check))
(r/'labeled_video_index.json').write_text(json.dumps(dict(at=time.time(),scope='Caption-only derivative; original generic Random batch8 title retained in original; full-state branch provenance clarified',no_new_simulation_or_inference=True,records=records),indent=2)+'\n')
print('Six caption-only corrected videos, each1250 frames, verified')
