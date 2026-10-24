"""Execute the author's wall construction against a CPU geometry recorder.

No isaacgym imports. The exact extracted source constructs the two finite
boxes; a fake API records asset size and actor transforms without simulation.
"""
import ast,hashlib,math,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from fullcycle_geometry import corners,transform
from fullcycle_collision import Piece
ROOT=Path(__file__).resolve().parents[3]

class Vec3:
    def __init__(self,x,y,z):self.x=x;self.y=y;self.z=z
class Transform:
    def __init__(self):self.p=Vec3(0,0,0);self.r=None
class Recorder:
    def __init__(self):self.actors=[]
    def load_asset(self,sim,directory,name,opts):
        xml=ET.parse(Path(directory)/name)
        return np.fromstring(xml.find('.//visual/geometry/box').get('size'),sep=' ')
    def create_actor(self,env,asset,pose,name,*args):
        self.actors.append(dict(name=name,size=asset.tolist(),center=[pose.p.x,pose.p.y,pose.p.z]))
        return len(self.actors)-1
    def set_rigid_body_color(self,*args):pass

def walls(effective,bbox,wall_offset=None):
    source=ROOT/'high-level/float_ik/door_common.py';text=source.read_text()
    names=['door_side_walls_enabled','door_wall_opening_axis','door_wall_bounds_yaw_offset',
           'create_door_side_wall_asset','door_world_xy_bounds_from_bbox','create_door_side_walls']
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    with tempfile.TemporaryDirectory(prefix='wc4-stage2-walls-') as folder:
        ns=dict(math=math,Path=Path,tempfile=SimpleNamespace(gettempdir=lambda:folder),
            gymapi=SimpleNamespace(Vec3=Vec3,Transform=Transform,Quat=lambda *x:x,AssetOptions=SimpleNamespace,MESH_VISUAL=0),
            _DOOR_SIDE_WALL_ASSET_CACHE={},door_asset_family=lambda d:'wc4')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
        args=SimpleNamespace(**effective)
        if wall_offset is not None:args.door_wall_x_offset=wall_offset
        door=SimpleNamespace(spec={'name':'wc4'},actor_scale=1.,actor_yaw=math.pi/2,actor_position_offset=[.11,0,0],bounding=bbox)
        g=Recorder();ns['create_door_side_walls'](g,object(),object(),door,args,99)
        result=[]
        for a in g.actors:
            size=np.array(a['size']);piece=Piece(a['name'],'finite_wall',corners(-size/2,size/2),'box')
            result.append((piece,piece.world(transform(a['center']))))
        return result,dict(author_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),functions=names,actors=g.actors,
            method='Unmodified author AST with CPU asset/actor recorder; no simulator; visual box dimensions as constructed by author asset source')
