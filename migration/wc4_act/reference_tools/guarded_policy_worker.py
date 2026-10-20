"""Fail on non-finite normalized inputs/raw model outputs before source sanitization."""
from pathlib import Path
import runpy
import sys
import torch
from lerobot.policies.act.modeling_act import ACTPolicy

original=ACTPolicy.predict_action_chunk
def checked(self,batch):
    for key in self.config.input_features:
        if key in batch and not torch.isfinite(batch[key]).all().item():
            raise FloatingPointError('STOP: non-finite normalized inference input '+key)
    result=original(self,batch)
    if not torch.isfinite(result).all().item():
        raise FloatingPointError('STOP: non-finite raw ACT action before sanitization')
    return result
ACTPolicy.predict_action_chunk=checked
source=Path(sys.argv[1]);sys.argv=sys.argv[1:]
sys.path.insert(0,str(source.parent))
runpy.run_path(str(source),run_name='__main__')
