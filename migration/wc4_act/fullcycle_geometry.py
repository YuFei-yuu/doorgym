"""CPU-only URDF geometry/FK for the nominal full-cycle experiment.

Uses existing asset geometry; no simulator state, physical properties or assets
are modified. Mesh AABBs are conservative, including convex decomposition.
"""
import itertools
import math
import struct
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def vec(s, default='0 0 0'):
    return np.fromstring(s or default, sep=' ')


def rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*K + (1-math.cos(angle))*(K@K)


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    T = np.eye(4)
    T[:3, 3] = xyz
    T[:3, :3] = rot([0,0,1], rpy[2])@rot([0,1,0], rpy[1])@rot([1,0,0], rpy[0])
    return T


def origin(e):
    o = e.find('origin')
    return np.eye(4) if o is None else transform(vec(o.get('xyz')), vec(o.get('rpy')))


def corners(lo, hi):
    return np.array(list(itertools.product(*zip(lo, hi))), dtype=float)


def apply(T, p):
    return p@T[:3,:3].T + T[:3,3]


def mesh_bounds(path):
    path = Path(path)
    if path.suffix.lower() == '.obj':
        points = np.array([list(map(float, l.split()[1:4])) for l in path.read_text().splitlines() if l.startswith('v ')])
    elif path.suffix.lower() == '.stl':
        b = path.read_bytes()
        n = struct.unpack('<I', b[80:84])[0]
        if len(b) == 84+n*50:
            d = np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
            points = np.frombuffer(b, dtype=d, count=n, offset=84)['v'].reshape(-1,3)
        else:
            points = np.array([list(map(float,l.split()[1:4])) for l in b.decode().splitlines() if l.strip().startswith('vertex ')])
    elif path.suffix.lower() == '.dae':
        root = ET.parse(path).getroot()
        ns = {'c':'http://www.collada.org/2005/11/COLLADASchema'}
        all_points = []
        for src in root.findall('.//c:mesh/c:source', ns):
            a = src.find('c:float_array', ns)
            if a is not None and ('position' in src.get('id','').lower()):
                all_points.append(np.fromstring(a.text, sep=' ').reshape(-1,3))
        points = np.concatenate(all_points)
        unit = root.find('c:asset/c:unit', ns)
        points *= 1 if unit is None else float(unit.get('meter','1'))
    else:
        raise ValueError(path)
    return points.min(0), points.max(0)


class URDF:
    def __init__(self, path, include_visual=False):
        self.path = Path(path)
        self.root = ET.parse(path).getroot()
        self.links = {e.get('name'):e for e in self.root.findall('link')}
        self.joints = []
        self.clouds = {}
        for j in self.root.findall('joint'):
            self.joints.append((j.get('name'), j.get('type'), j.find('parent').get('link'), j.find('child').get('link'), origin(j), vec(j.find('axis').get('xyz')) if j.find('axis') is not None else np.array([1.,0,0])))
        for name,l in self.links.items():
            points = []
            for g in list(l.findall('collision')) + (list(l.findall('visual')) if include_visual else []):
                geom = g.find('geometry')
                if geom is None: continue
                e = list(geom)[0]
                if e.tag == 'mesh':
                    lo,hi = mesh_bounds(self.path.parent/e.get('filename'))
                    scale = vec(e.get('scale'), '1 1 1')
                    p = corners(lo,hi)*scale
                elif e.tag == 'box':
                    size = vec(e.get('size'))/2
                    p = corners(-size,size)
                elif e.tag in ('cylinder','sphere'):
                    radius = float(e.get('radius'))
                    half = radius if e.tag == 'sphere' else float(e.get('length'))/2
                    p = corners([-radius,-radius,-half],[radius,radius,half])
                else: raise ValueError(e.tag)
                points.append(apply(origin(g),p))
            if points: self.clouds[name] = np.concatenate(points)
        self.root_links = set(self.links)-{j[3] for j in self.joints}

    def fk(self, q, root_T=None):
        poses = {n:np.eye(4) if root_T is None else root_T for n in self.root_links}
        todo = self.joints[:]
        while todo:
            progress = False
            for j in todo[:]:
                name,kind,parent,child,T,axis = j
                if parent not in poses: continue
                M = np.eye(4)
                if kind in ('revolute','continuous'): M[:3,:3] = rot(axis,float(q.get(name,0)))
                elif kind == 'prismatic': M[:3,3] = axis*float(q.get(name,0))
                poses[child] = poses[parent]@T@M
                todo.remove(j); progress = True
            if not progress: raise ValueError('Disconnected URDF')
        return poses

    def cloud(self, poses, exclude=()):
        return np.concatenate([apply(poses[n],p) for n,p in self.clouds.items() if n not in exclude])

    def arm_fk(self, values):
        return self.fk(dict(zip(['joint%d'%i for i in range(1,7)],values)))['ee_gripper_link']


def close_pose(angle, radius, distance, z):
    phi = math.pi/2 + angle
    u = np.array([math.cos(phi), math.sin(phi)])
    n = np.array([math.sin(phi), -math.cos(phi)])
    hinge = np.array([2.467,-.463])
    point = hinge + radius*u
    base = point-distance*n
    velocity = radius*n+distance*u
    yaw = math.atan2(velocity[1],velocity[0])
    T = transform([*base,.5],[0,0,yaw])
    # End effector's x axis points along the closing normal, roll as author.
    ee = transform([*point,z],[math.pi/2,0,math.atan2(n[1],n[0])])
    return T, ee
