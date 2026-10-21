"""Reject precision-destroying float32-then-float64 implementations too."""
from common import *
import torch
from lerobot.datasets import video_utils as vu
from lerobot.datasets.factory import make_dataset
from regression import FakeDecoder,FakeCache,old_reader

new=vu.decode_video_frames_torchcodec;old=old_reader();before=torch.get_default_dtype()
results=[]
for q,offset,expected,name in [([4000.00009999],0.,True,'long_query_within'),([4000.00010001],0.,False,'long_query_beyond'),
 ([4000.],.00009999,True,'long_native_pts_within'),([4000.],.00010001,False,'long_native_pts_beyond'),
 ([0.,.0402],0.,False,'multi_one_wrong_timestamp')]:
 d=FakeDecoder(offset);passed=True
 try:new('fake',q,1e-4,decoder_cache=FakeCache(d))
 except vu.FrameTimestampError:passed=False
 assert passed==expected;results.append(dict(name=name,queries=q,native_offset=offset,passed_read=passed,expected=expected,selected=d.calls))
ds=make_dataset(config());key=ds.meta.video_keys[0];path=DATA/ds.meta.get_video_file_path(0,key)
q=[.08,28.12,0.,.08,100.]
a=old(path,q,1e-4);b=new(path,q,1e-4);torch.testing.assert_close(a,b,rtol=0,atol=0)
assert torch.get_default_dtype()==before==torch.float32 and b.dtype==torch.float32
write(RUN/'extra-regression.json',dict(passed=True,guard_cases=results,normal_real_multi_queries=q,old_new_multi_outputs_exact=True,count=len(b),dtype_before=str(before),dtype_after=str(torch.get_default_dtype()),output_dtype=str(b.dtype),
 purpose='Long negative cases would incorrectly pass if original query/PTS were first rounded to float32 then promoted to float64. No global dtype change.'))
print('Additional direct-precision and multi-query regressions passed.')
