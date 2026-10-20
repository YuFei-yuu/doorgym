"""Offline convex collision pieces from actual URDF collision elements.

Boxes remain separate pieces. Meshes use their convex hull, matching the
conservative single-convex robot import; the door slab convex hull is an outer
bound on VHACD pieces and is never sufficient to prove penetration of a cavity.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from fullcycle_geometry import origin, vec, corners, apply


def mesh_points(path):
    path=Path(path)
    if path.suffix.lower()=='.obj':
        return np.array([list(map(float,l.split()[1:4])) for l in path.read_text().splitlines() if l.startswith('v ')])
    if path.suffix.lower()=='.stl':
        import struct
        b=path.read_bytes();n=struct.unpack('<I',b[80:84])[0]
        if len(b)==84+n*50:
            d=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
            return np.frombuffer(b,dtype=d,count=n,offset=84)['v'].reshape(-1,3).astype(float)
        return np.array([list(map(float,l.split()[1:4])) for l in b.decode().splitlines() if l.strip().startswith('vertex ')])
    raise ValueError('Collision mesh format must be OBJ/STL: '+str(path))


def unique_axes(a):
    a=np.asarray(a,float).reshape(-1,3);a=a[np.linalg.norm(a,axis=1)>1e-9]
    if not len(a): return a
    a=a/np.linalg.norm(a,axis=1,keepdims=True)
    for x in a:
        i=np.flatnonzero(abs(x)>1e-8)[0]
        if x[i]<0:x*=-1
    _,indices=np.unique(np.round(a,7),axis=0,return_index=True)
    return a[np.sort(indices)]


class Piece:
    def __init__(self,link,name,points,kind):
        self.link=link;self.name=name;self.kind=kind
        h=ConvexHull(points);self.vertices=points[h.vertices]
        self.normals=unique_axes(h.equations[:,:3])
        edge_faces={}
        for i,tri in enumerate(h.simplices):
            for u,v in zip(tri,np.roll(tri,-1)):
                edge_faces.setdefault(tuple(sorted([int(u),int(v)])),[]).append(i)
        edges=[]
        for (u,v),faces in edge_faces.items():
            if len(faces)!=2 or abs(np.dot(h.equations[faces[0],:3],h.equations[faces[1],:3]))<1-1e-7:
                edges.append(points[v]-points[u])
        self.edges=unique_axes(edges)

    def world(self,T):
        return (apply(T,self.vertices),self.normals@T[:3,:3].T,self.edges@T[:3,:3].T)


def collision_pieces(urdf):
    path=Path(urdf);root=ET.parse(path).getroot();pieces=[]
    for link in root.findall('link'):
        for i,c in enumerate(link.findall('collision')):
            g=list(c.find('geometry'))[0]
            if g.tag=='box':
                d=vec(g.get('size'))/2;p=corners(-d,d)
            elif g.tag=='mesh':
                p=mesh_points(path.parent/g.get('filename'))*vec(g.get('scale'),'1 1 1')
            elif g.tag=='cylinder':
                r=float(g.get('radius'));half=float(g.get('length'))/2
                # Circumscribed 32-gon, conservative by <0.5% in radius.
                a=np.arange(32)*2*np.pi/32;r/=np.cos(np.pi/32)
                p=np.array([[r*np.cos(x),r*np.sin(x),z] for x in a for z in [-half,half]])
            elif g.tag=='sphere':
                r=float(g.get('radius'));p=corners([-r]*3,[r]*3)
            else:raise ValueError(g.tag)
            pieces.append(Piece(link.get('name'),c.get('name',str(i)),apply(origin(c),p),g.tag))
    return pieces


def signed_gap(A,B):
    """Exact SAT sign for convex polyhedra; positive is separation proof.

    Positive value is a projected lower bound on Euclidean surface distance.
    Negative value signals convex overlap, not necessarily mesh concavity.
    """
    av,an,ae=A;bv,bn,be=B
    broad=np.maximum(bv.min(0)-av.max(0),av.min(0)-bv.max(0))
    if broad.max()>.05:return float(broad.max())
    cross=np.cross(ae[:,None,:],be[None,:,:]).reshape(-1,3)
    axes=unique_axes(np.concatenate([an,bn,cross]))
    aa=av@axes.T;bb=bv@axes.T
    gaps=np.maximum(bb.min(0)-aa.max(0),aa.min(0)-bb.max(0))
    return float(gaps.max())


def minimum_gap(pieces_a,poses_a,pieces_b,poses_b):
    best=(float('inf'),None)
    for a in pieces_a:
        A=a.world(poses_a[a.link])
        for b in pieces_b:
            gap=signed_gap(A,b.world(poses_b[b.link]))
            if gap<best[0]:best=(gap,(a.link,a.name,b.link,b.name,a.kind,b.kind))
    return best
