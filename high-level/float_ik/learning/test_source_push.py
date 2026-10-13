"""Compare the shared floating push helper against the fixed upstream task."""
import ast
import copy
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'high-level/float_ik'))
import isaacgym_float_ik_b1z1_basearn_push_door as source


def check():
    relative='high-level/float_ik/isaacgym_float_ik_b1z1_basearn_push_door.py'
    original=subprocess.check_output(['git','show','0acc2f050479fb54a0670ccb90bc72b98b52a54a:'+relative],cwd=ROOT,text=True)
    tree=ast.parse(original)
    defaults={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Dict):
            values={k.value:v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant)}
            if 'name' in values and isinstance(values['name'],ast.Constant) and str(values['name'].value).startswith('--'):
                name=values['name'].value[2:]
                defaults[name]=(eval(compile(ast.Expression(values['default']),relative,'eval'),vars(source)) if 'default' in values else False)
    args=SimpleNamespace(**defaults)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='trajectory_targets')
    namespace=dict(vars(source))
    handle_pos=np.array([2.5,.03,.8],np.float32)
    handle_quat=np.array([0,0,0,1],np.float32)
    angle=[0.]
    namespace['get_body_pose']=lambda *a:(handle_pos.copy(),handle_quat.copy())
    namespace['get_actor_dof_state']=lambda *a:(np.array([angle[0],0.],np.float32),np.zeros(2,np.float32))
    namespace['door_hinge_open_ratio']=lambda door,a,args:abs(a)/(math.pi/2)
    exec(compile(ast.Module(body=[function],type_ignores=[]),relative,'exec'),namespace)
    reference=namespace['trajectory_targets']
    adapted=types.FunctionType(source.trajectory_targets.__code__,namespace)
    rotate_end=sum(getattr(args,k) for k in ['walk_steps','initial_hold_steps','grasp_steps','grasp_hold_steps','gripper_close_steps','handle_rotate_steps'])
    cases=[]
    for opened,ratio,progress,unidoor in [(False,0,.1,True),(False,0,.7,False),(True,.2,.1,True),(True,.5,.6,True),(True,.87,.7,True),(True,.96,.8,True)]:
        args.unidoor_style_push=unidoor;angle[0]=-ratio*math.pi/2
        door=SimpleNamespace(open_stage=opened,handle_body_index=0,handle_goal_offset=np.zeros(3,np.float32))
        ik=SimpleNamespace(current_pos_np=np.array([2.4,0,.8],np.float32),target_quat_np=handle_quat.copy())
        step=rotate_end+int(progress*args.door_push_steps)-1
        common=(step,args,door,None,None,None,ik,np.array([4.1,0],np.float32),np.array([3.3,0],np.float32),np.array([1.,0],np.float32),math.pi,math.pi)
        a=reference(*common,{})
        b=adapted(*common,{})
        for x,y in zip(a,b):
            if isinstance(x,str):assert x==y
            elif x is None:assert y is None
            else:assert np.allclose(x,y,rtol=0,atol=1e-7),(x,y)
        cases.append(dict(opened=opened,ratio=ratio,progress=progress,unidoor=unidoor,matched=True))
    return dict(passed=True,cases=cases)


if __name__=='__main__':
    result=check()
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
