"""Real old/new reader comparisons plus synthetic guard regressions."""
import ast,json,subprocess,types,unittest
from common import *
import numpy as np,torch
from lerobot.datasets.factory import make_dataset
from lerobot.datasets import video_utils as vu


def old_reader():
 source=subprocess.check_output(['git','show','d8a72fa:high-level/lerobot/src/lerobot/datasets/video_utils.py'],cwd=ROOT,text=True)
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='decode_video_frames_torchcodec')
 namespace=vu.__dict__.copy();exec(compile(ast.Module(body=[node],type_ignores=[]),'<d8a72fa TorchCodec reader>','exec'),namespace)
 return namespace[node.name]


class FakeDecoder:
 def __init__(self,offset=0.):self.metadata=types.SimpleNamespace(average_fps=25.);self.offset=offset;self.calls=[]
 def get_frames_at(self,indices):
  self.calls.append(indices)
  return types.SimpleNamespace(data=torch.stack([torch.full((3,2,2),i%255,dtype=torch.uint8) for i in indices]),pts_seconds=torch.tensor([i/25+self.offset for i in indices],dtype=torch.float64))
class FakeCache:
 def __init__(self,d):self.d=d
 def get_decoder(self,path):return self.d


def main():
 old=old_reader();new=vu.decode_video_frames_torchcodec;dtype=torch.get_default_dtype();assert dtype==torch.float32
 synthetic=[]
 for offset,expected in [(0.,True),(.000099,True),(.0001,False),(.0002,False)]:
  d=FakeDecoder(offset);passed=True
  try:new('fake',[0.],1e-4,decoder_cache=FakeCache(d))
  except vu.FrameTimestampError:passed=False
  assert passed==expected;synthetic.append(dict(offset_s=offset,read_passed=passed,expected=expected))
 queries=[2000.08,2000.,2000.04,2000.08]
 d=FakeDecoder();out=new('fake',queries,1e-4,decoder_cache=FakeCache(d))
 assert d.calls==[[50002,50000,50001,50002]] and len(out)==len(queries)
 np.testing.assert_allclose(out[:,0,0,0].numpy(),np.array([i%255 for i in d.calls[0]],np.float32)/255,rtol=0,atol=0)
 assert out.dtype==torch.float32
 ds=make_dataset(config());timestamps=np.asarray(ds.hf_dataset['timestamp'],dtype=np.float64).reshape(-1)
 episodes=np.asarray(ds.hf_dataset['episode_index']).reshape(-1)
 tests={0,1,100,41660,41661,41662,len(ds)-1,len(ds)-2}
 # Both sides of float32 absolute-time precision transitions (seconds).
 for seconds in [512,1024,2048]:tests.update(range(seconds*25-2,seconds*25+3))
 for e in range(46):
  ids=np.flatnonzero(episodes==e);tests.update([int(ids[0]),int(ids[-1])])
 real=[];short_equal=0;old_failures=0
 for i in sorted(tests):
  e=int(episodes[i]);ep=ds.meta.episodes[e]
  for key in ds.meta.video_keys:
   offset=ep['videos/'+key+'/from_timestamp'];assert not isinstance(offset,torch.Tensor) or offset.dtype==torch.float64
   q=[float(offset)+float(timestamps[i])];path=DATA/ds.meta.get_video_file_path(e,key)
   decoder=vu._default_decoder_cache.get_decoder(str(path));selected=[round(t*decoder.metadata.average_fps) for t in q]
   native=decoder.get_frames_at(indices=selected)
   current=new(path,q,1e-4);assert current.dtype==torch.float32
   np.testing.assert_array_equal(current.numpy(),(native.data/255.).to(torch.float32).numpy())
   before_ok=True;before_error=None
   try:
    previous=old(path,q,1e-4);torch.testing.assert_close(previous,current,rtol=0,atol=0)
    if q[0]<512:short_equal+=1
   except vu.FrameTimestampError as ex:before_ok=False;before_error=str(ex);old_failures+=1
   real.append(dict(index=i,episode=e,key=key,query=q,selected_frames=selected,native_pts=native.pts_seconds.tolist(),old_passed=before_ok,new_passed=True,old_error=before_error))
 # Real multi-time query with reversed order and duplicates.
 key=ds.meta.video_keys[0];path=DATA/ds.meta.get_video_file_path(0,key)
 q=[2000.08,28.12,1666.440003356934,2000.08,0.]
 many=new(path,q,1e-4)
 for i,t in enumerate(q):torch.testing.assert_close(many[i],new(path,[t],1e-4)[0],rtol=0,atol=0)
 assert many.shape[0]==5 and torch.get_default_dtype()==dtype
 assert any(x['index']==41661 and not x['old_passed'] for x in real) and short_equal>0
 write(RUN/'regression.json',dict(passed=True,real_single_reads=len(real),old_failures_reproduced=old_failures,short_old_new_exact_equal=short_equal,
  synthetic=synthetic,synthetic_order_queries=queries,synthetic_selected=d.calls,real_multi_queries=q,real_multi_count=len(many),dtype_before=str(dtype),dtype_after=str(torch.get_default_dtype()),returned_image_dtype=str(many.dtype),tolerance_s=1e-4,comparison_rule='strict <; unchanged',frame_selection='round(timestamp * average_fps); unchanged',real=real,no_model_constructed=True))
 print('Regression passed',len(real),'real reads;',old_failures,'old precision errors reproduced',flush=True)


if __name__=='__main__':main()
