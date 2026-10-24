"""Read-only stopped-run diagnosis. Does not change guards or create a scene."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
x=json.loads((r/'trial_00/attempt_000/initial-frozen-comparison.json').read_text())
e=np.array(x['expected']['root_quat']);q=np.array([x['actual']['root']['pose']['r'][k] for k in ['x','y','z','w']])
d={'max_abs_component_difference':float(abs(q-e).max()),'expected_norm':float(np.linalg.norm(e)),
   'actual_norm':float(np.linalg.norm(q)),
   'normalized_rotation_difference_rad':float((Rotation.from_quat(q)*Rotation.from_quat(e).inv()).magnitude()),
   'base_start_identical':x['actual']['base_start']==x['expected']['base_start'],
   'root_z_identical':x['actual']['root']['pose']['p']['z']==x['expected']['robot_z'],
   'yaw_identical':x['actual']['yaw']==x['expected']['yaw']}
original=json.loads((r/'initial-binding-diagnosis.json').read_text())
assert all(original[k]==v for k,v in d.items())
(r/'diagnosis-reproduction.json').write_text(json.dumps({'matches_first_cpu_diagnosis':True,'values':d},indent=2)+'\n')
print(json.dumps(d,indent=2))
