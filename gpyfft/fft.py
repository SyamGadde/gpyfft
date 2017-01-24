from __future__ import absolute_import, division, print_function
from .gpyfftlib import GpyFFT
import gpyfft.gpyfftlib as gfft
import pyopencl as cl
GFFT = GpyFFT(debug=False)

import pyopencl as cl
import numpy as np

class FFT(object):
    def __init__(self, context, queue, in_array, out_array=None, axes = None,
                 fast_math = False,
                 real=False,
                 in_shape = None,
                 out_shape = None,
    ):
        self.context = context
        self.queue = queue

        if in_shape is None:
            in_shape = in_array.shape
        t_strides_in, t_distance_in, t_batchsize_in, t_shape, axes_transform = self.calculate_transform_strides(
            axes, in_shape, in_array.strides, in_array.dtype)

        if out_array is not None:
            # if both arrays point to the same underlying buffer,
            # assume in-place
            t_inplace = (in_array.base_data is out_array.base_data)
            if out_shape is None:
                out_shape = out_array.shape
            t_strides_out, t_distance_out, t_batchsize_out, t_shape_out, foo = self.calculate_transform_strides(
                axes, out_shape, out_array.strides, out_array.dtype)
            if t_inplace:
                out_array = None

            #assert t_batchsize_out == t_batchsize_in and t_shape == t_shape_out, 'input and output size does not match' #TODO: fails for real-to-complex
            
        else:
            t_inplace = True
            t_strides_out, t_distance_out = t_strides_in, t_distance_in

        
        #assert np.issubclass(in_array.dtype, np.complexfloating) and \
        #    np.issubclass(in_array.dtype, np.complexfloating), \
                
        #precision (+ fast_math!)
        #complex64 <-> complex64
        #complex128 <-> complex128

        if in_array.dtype in (np.float32, np.complex64):
            precision = gfft.CLFFT_SINGLE
        elif in_array.dtype in (np.float64, np.complex128):
            precision = gfft.CLFFT_DOUBLE

        #TODO: add assertions that precision match
        if in_array.dtype in (np.float32, np.float64):
            layout_in = gfft.CLFFT_REAL
            layout_out = gfft.CLFFT_HERMITIAN_INTERLEAVED

            expected_out_shape = list(in_shape)
            expected_out_shape[axes_transform[0]] = expected_out_shape[axes_transform[0]]//2 + 1
            assert out_shape == tuple(expected_out_shape), \
                'output array shape %s does not match expected shape: %s'%(out_shape,expected_out_shape)

        elif in_array.dtype in (np.complex64, np.complex128):
            if not real:
                layout_in = gfft.CLFFT_COMPLEX_INTERLEAVED
                layout_out = gfft.CLFFT_COMPLEX_INTERLEAVED
            else:
                # complex-to-real transform
                layout_in = gfft.CLFFT_HERMITIAN_INTERLEAVED
                layout_out = gfft.CLFFT_REAL
                t_shape = t_shape_out

        self.t_shape = t_shape
        self.batchsize = t_batchsize_in

        plan = GFFT.create_plan(context, t_shape)
        plan.inplace = t_inplace
        plan.strides_in = t_strides_in
        plan.strides_out = t_strides_out
        plan.distances = (t_distance_in, t_distance_out)
        plan.batch_size = self.batchsize
        plan.precision = precision
        plan.layouts = (layout_in, layout_out)

        if False:
            print('axes', axes        )
            print('in_array.shape:          ', in_array.shape)
            print('in_array.strides/itemsize', tuple(s // in_array.dtype.itemsize for s in in_array.strides))
            print('in_shape:                ', in_shape)
            print('shape transform          ', t_shape)
            print('t_strides                ', t_strides_in)
            print('distance_in              ', t_distance_in)
            print('batchsize                ', t_batchsize_in)
            print('t_stride_out             ', t_strides_out)
            print('inplace                  ', t_inplace)

        plan.bake(self.queue)
        temp_size = plan.temp_array_size
        if temp_size:
            #print 'temp_size:', plan.temp_array_size
            self.temp_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size = temp_size)
        else:
            self.temp_buffer = None

        self.plan = plan
        self.data = in_array
        self.result = out_array

    @classmethod
    def calculate_transform_strides(cls,
                                    axes,
                                    shape,
                                    strides,
                                    dtype,
                                   ):
        ddim = len(shape) #dimensionality data
        if axes is None:
            axes = range(ddim)

        tdim = len(axes) #dimensionality transform
        assert tdim <= ddim

        # transform negative axis values (e.g. -1 for last axis) to positive
        axes_transform = tuple(a + ddim if a<0 else a for a in axes)

        axes_notransform = set(range(ddim)).difference(axes_transform)
        assert len(axes_notransform) < 2, 'more than one non-transformed axis'
        #TODO: collapse non-transformed axes if possible

        t_shape = [shape[i] for i in axes_transform]
        dsize = dtype.itemsize
        t_strides = [strides[i]//dsize for i in axes_transform]

        t_distance = [strides[i]//dsize for i in axes_notransform]
        if not t_distance:
            t_distance = 0
        else:
            t_distance = t_distance[0] #TODO

        batchsize = 1
        for a in axes_notransform:
            batchsize *= shape[a]

        return (tuple(t_strides), t_distance, batchsize, tuple(t_shape), axes_transform)

    def enqueue(self, forward = True):
        """enqueue transform"""
        if self.result is not None:
            events = self.plan.enqueue_transform((self.queue,), (self.data.data,), (self.result.data),
                                        direction_forward = forward, temp_buffer = self.temp_buffer)
        else:
            events = self.plan.enqueue_transform((self.queue,), (self.data.data,),
                                        direction_forward = forward, temp_buffer = self.temp_buffer)
        return events

    def update_arrays(self, input_array, output_array):
        pass
