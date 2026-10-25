"""Unchanged strict tolerance and direct float64 query/PTS regression."""
from common import *
import types,torch
from lerobot.datasets import video_utils as vu

class Decoder:
    metadata=types.SimpleNamespace(average_fps=25.)
    def __init__(self,offset):self.offset=offset;self.indices=None
    def get_frames_at(self,indices):
        self.indices=indices
        return types.SimpleNamespace(data=torch.zeros((len(indices),3,2,2),dtype=torch.uint8),pts_seconds=torch.tensor([i/25+self.offset for i in indices],dtype=torch.float64))
class Cache:
    def __init__(self,d):self.d=d
    def get_decoder(self,path):return self.d

results=[]
for q,offset,expect in [([0.],0.,True),([0.],.000099,True),([0.],.0001,False),([0.],.0002,False),([4000.00009999],0.,True),([4000.00010001],0.,False),([4000.],.00009999,True),([4000.],.00010001,False),([0.,.0402],0.,False)]:
    d=Decoder(offset);passed=True
    try:im=vu.decode_video_frames_torchcodec('fake',q,1e-4,decoder_cache=Cache(d))
    except vu.FrameTimestampError:passed=False
    assert passed==expect
    results.append(dict(queries=q,offset=offset,expected=expect,passed_read=passed,selected=d.indices))
d=Decoder(0.)
vu.decode_video_frames_torchcodec('fake',[2000.08,2000.,2000.04,2000.08],1e-4,decoder_cache=Cache(d))
assert d.indices==[50002,50000,50001,50002] and torch.get_default_dtype()==torch.float32
write(RUN/'precision-guard.json',dict(passed=True,cases=results,ordering_and_duplicates=True,default_dtype=str(torch.get_default_dtype()),tolerance_s=1e-4,comparison='strict < unchanged'))
print('Precision guard passed',flush=True)
