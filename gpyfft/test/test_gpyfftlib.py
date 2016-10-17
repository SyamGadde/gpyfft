from __future__ import print_function
import unittest
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from gpyfft import gpyfftlib


"""
Some basic tests
"""

def get_contexts():
    ALL_DEVICES = []
    for platform in cl.get_platforms():
        ALL_DEVICES += platform.get_devices(device_type = cl.device_type.GPU)
    contexts = [ cl.Context([device]) for device in ALL_DEVICES ]
    return contexts


class test_basic(unittest.TestCase):


    
    def test_basic(self):
        G = gpyfftlib.GpyFFT()
        print('clFFT version:', G.get_version())

    def test_create_plan(self):
        G = gpyfftlib.GpyFFT()

        ctx = get_contexts()[0]
        queue = cl.CommandQueue(ctx)
        nd_data = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8]],
                           dtype=np.complex64)
        cl_data = cla.to_device(queue, nd_data)
        cl_data_transformed = cla.zeros_like(cl_data)

        plan = G.create_plan(ctx, cl_data.shape)

        print('plan.strides_in', plan.strides_in)
        print('plan.strides_out', plan.strides_out)
        print('plan.distances', plan.distances)
        print('plan.batch_size', plan.batch_size)

        
        
        

if __name__ == '__main__':
    unittest.main()
