"""The original raw schema has pre-action fields, not backend transitions."""
import unittest
import numpy as np
from branch16040 import IMAGES, verify_raw_trajectory


class RawSchemaTests(unittest.TestCase):
    def setUp(self):
        self.raw={k:np.zeros((2,1),dtype=np.float32) for k in ['state','action','subtask_index',*IMAGES]}
        self.trajectory={k:v.copy() for k,v in self.raw.items()}
        self.trajectory['next_state']=np.ones((2,1),dtype=np.float32)

    def test_original_raw_schema(self):
        verify_raw_trajectory(self.raw,self.trajectory)

    def test_action_mismatch_still_rejected(self):
        self.trajectory['action'][0,0]=1
        with self.assertRaises(ValueError):verify_raw_trajectory(self.raw,self.trajectory)

    def test_missing_image_still_rejected(self):
        del self.raw[IMAGES[0]]
        with self.assertRaises(KeyError):verify_raw_trajectory(self.raw,self.trajectory)


if __name__=='__main__':unittest.main()
