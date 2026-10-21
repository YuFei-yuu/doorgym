"""Install a safe boundary around the author update/serialization interfaces.

This module never builds an ACT model or optimizer. The caller supplies the
unchanged author trainer and captures its actual policy/processors/config.
"""
from pathlib import Path
from stopping import StopBoundary,commit_checkpoint


class TrainingGuard:
    def __init__(self,module,run,cfg_getter):
        self.module=module;self.run=Path(run);self.cfg_getter=cfg_getter
        self.context=None;self.processors=None
        self.boundary=StopBoundary(run,self.save)
        original_processors=module.make_pre_post_processors
        def processors(*a,**k):
            self.processors=original_processors(*a,**k)
            return self.processors
        module.make_pre_post_processors=processors
        original_update=module.update_policy
        def update(*a,**k):
            # Signature is the existing author update_policy metrics,policy,batch,
            # optimizer,grad_clip; accelerator/scheduler remain the same objects.
            self.context=dict(policy=a[1],optimizer=a[3],accelerator=k['accelerator'],scheduler=k.get('lr_scheduler'))
            return self.boundary.update(original_update,*a,**k)
        module.update_policy=update

    def save(self,step):
        if self.context is None or self.processors is None:raise RuntimeError('No initialized consistent trainer context')
        import torch
        from lerobot.utils.random_utils import serialize_rng_state
        context=self.context
        if torch.cuda.is_available():torch.cuda.synchronize()
        before=serialize_rng_state()
        def writer(directory,completed):
            self.module.save_checkpoint(checkpoint_dir=directory,step=completed,cfg=self.cfg_getter(),
                policy=context['accelerator'].unwrap_model(context['policy']),optimizer=context['optimizer'],
                scheduler=context['scheduler'],preprocessor=self.processors[0],postprocessor=self.processors[1])
            after=serialize_rng_state()
            if set(before)!=set(after) or not all(torch.equal(before[k],after[k]) for k in before):raise RuntimeError('Serializer changed RNG; checkpoint not published')
        receipt=commit_checkpoint(self.run/'stopped-checkpoints',step,writer)
        after=serialize_rng_state()
        if set(before)!=set(after) or not all(torch.equal(before[k],after[k]) for k in before):raise RuntimeError('Checkpoint serialization changed trainer RNG')
        return receipt


def metric_values(metrics):
    """Author MetricsTracker meters and simple engineering fixtures share a path."""
    return {name:float(getattr(getattr(metrics,name),'val',getattr(metrics,name)))
            for name in ['loss','lr','grad_norm','update_s']}
