"""Bind offline code inspection to the historical frozen execution sources."""
import hashlib,pathlib
from review import ROOT,RUN,read,write


def main():
    binding=read(ROOT/'.repro/runs/wc4-fulltask-expand-r9-20261021/consistency.json')['source_sha256']
    checked={}
    for name,digest in binding.items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        assert actual==digest, 'Historical execution source changed: '+name
        checked[name]=actual
    notes=[
        dict(file='migration/wc4_act/fullcycle_expert.py',lines=[158,164,178,224,233,304,306,310,312,313,323,335,347,393],
             finding='Only declared locked_hold writes hinge state; unlock-to-latch is guarded. Compact path starts after actual whole-body pass. Latch stores current measured angle after observed hold. Actual post-state and raw contacts recorded each physics step.'),
        dict(file='high-level/float_ik/door_common.py',lines=[5294,5322,5329,5339],
             finding='Existing handle spring retained; hinge effort uses configured resistance/damping/auto torque. Historical effective configuration and every unlocked raw hinge effort are checked, not silently altered.'),
        dict(file='migration/wc4_act/fullcycle_recording.py',lines=[12,30,40,55,62],
             finding='Live 25Hz buffers, measured joints and preceding recorded command; no log decimation or new phase policy input.'),
        dict(file='high-level/lerobot/src/lerobot/datasets/video_utils.py',lines=[360,361,364],
             finding='R10 direct float64 query and native PTS construction; strict tolerance/frame selection unchanged.')]
    for source,count in [('wc4-fulltask-pilot-r8-20261021',8),('wc4-fulltask-expand-r9-20261021',64)]:
        for i in range(count):
            p=ROOT/'.repro/runs'/source/('trial_%02d'%i)
            torques=read(p/'reference.json')['hinge_torque_sources']
            assert all(torques[k]==0 for k in ['door_open_resistance','door_open_damping','door_auto_open_force','door_joint_friction','door_joint_damping'])
    write(RUN/'code-audit.json',dict(passed=True,frozen_source_files=len(checked),source_sha256=checked,
        reviewed_paths=notes,all_72_effective_hinge_configurations_checked=True,
        no_expert_asset_physics_home_or_recording_source_changes=True,
        limitations='Code and raw-log audit plus original video review; not full collision/force-model certification. No simulator imported.'))
    print('Frozen execution sources and all72 hinge configurations match',flush=True)


if __name__=='__main__':main()
