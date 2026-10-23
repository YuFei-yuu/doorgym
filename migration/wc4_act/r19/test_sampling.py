"""CPU fixture: Accelerate loader preserves frozen indices and batch grouping."""
import os,pathlib,json
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from sampling import install,audit_batch,schedule
from accelerate import Accelerator
class IndexFixture(torch.utils.data.Dataset):
 def __len__(self):return 159529
 def __getitem__(self,i):return {'index':i,'observation.state':torch.zeros(9)}
install();loader=torch.utils.data.DataLoader(IndexFixture(),batch_size=16,num_workers=3,drop_last=False,shuffle=True,prefetch_factor=2)
a=Accelerator(cpu=True);loader=a.prepare(loader);seen=0;counts=np.zeros(5,dtype=np.int64)
for i,b in enumerate(loader):
 proof=audit_batch(b,i);counts+=proof['group_counts'];seen+=len(b['index'])
assert seen==400000 and counts.tolist()==[200000,50000,50000,50000,50000]
# Negative interface check must refuse reordering before optimizer execution.
try:audit_batch({'index':torch.tensor(schedule()[0][0][::-1].copy())},0)
except AssertionError:pass
else:raise AssertionError('corrupt batch accepted')
print(json.dumps(dict(status='passed',batches=i+1,samples=seen,groups=counts.tolist(),model_forwards=0,scenes=0,optimizer_updates=0)))
