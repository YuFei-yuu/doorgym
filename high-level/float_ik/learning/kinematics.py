"""URDF forward kinematics for diagnostics only; never writes simulator state."""
import math
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x,y,z = axis
    k = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(theta)*k+(1-math.cos(theta))*(k@k)


class FK:
    def __init__(self, path, tip, names):
        root = ET.parse(path).getroot()
        parents = {j.find('child').attrib['link']:j for j in root.findall('joint')}
        chain=[]
        while tip in parents:
            j=parents[tip]
            origin=j.find('origin')
            attrs=origin.attrib if origin is not None else {}
            xyz=np.fromstring(attrs.get('xyz','0 0 0'),sep=' ')
            rpy=np.fromstring(attrs.get('rpy','0 0 0'),sep=' ')
            t=np.eye(4)
            t[:3,3]=xyz
            t[:3,:3]=rotation([0,0,1],rpy[2])@rotation([0,1,0],rpy[1])@rotation([1,0,0],rpy[0])
            axis=j.find('axis')
            axis=np.fromstring(axis.attrib['xyz'],sep=' ') if axis is not None else np.array([1.,0,0])
            index=names.index(j.attrib['name']) if j.attrib['name'] in names else None
            chain.append((t,axis,index,j.attrib['type']))
            tip=j.find('parent').attrib['link']
        self.chain=chain[::-1]

    def position(self,q):
        t=np.eye(4)
        for origin,axis,index,kind in self.chain:
            t=t@origin
            if index is not None:
                motion=np.eye(4)
                if kind == 'prismatic':
                    motion[:3,3]=axis*q[index]
                else:
                    motion[:3,:3]=rotation(axis,q[index])
                t=t@motion
        return t[:3,3]
