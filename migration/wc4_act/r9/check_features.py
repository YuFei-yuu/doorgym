"""Use the real policy factory's feature parser without constructing a policy."""
import json,pathlib
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
R=pathlib.Path('/data/zyf/video2door-wc4-act/.repro/runs/wc4-fulltask-expand-r9-20261021')
F=R/'dataset-check';report=json.loads((F/'factory-loader-report.json').read_text())
features=dataset_to_policy_features(report['features'])
outputs={k:v for k,v in features.items() if v.type is FeatureType.ACTION}
inputs={k:v for k,v in features.items() if k not in outputs}
expected={'observation.state','observation.images.front_masked_depth','observation.images.wrist_masked_depth'}
assert set(inputs)==expected and set(outputs)=={'action'}
assert tuple(inputs['observation.state'].shape)==(9,) and tuple(outputs['action'].shape)==(9,)
for key in expected-{'observation.state'}:assert tuple(inputs[key].shape)==(3,480,640)
def serial(items):return {k:dict(type=v.type.value,shape=list(v.shape)) for k,v in items.items()}
(F/'resolved-policy-feature-map.json').write_text(json.dumps(dict(parser='lerobot.datasets.utils.dataset_to_policy_features (same function used by make_policy)',
    inputs=serial(inputs),outputs=serial(outputs),no_policy_constructed=True,stage_camera_pose_door_contact_are_not_inputs=True),indent=2)+'\n')
print('Only dual depth and Joint9 state inputs; action9 output; no policy constructed.')
