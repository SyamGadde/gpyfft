# File:         batch.py
# Description:  Utility classes for batching arbitrary gpyfft transforms
#               subject to clFFT and CL device restrictions.
# Author:       gadde@biac.duke.edu

import collections
import fractions
import itertools
import numpy
import operator

CACHED_PLANS = {}

# These are here so that we can delay importing gpyfft and pyopencl as
# late as possible, to avoid hiccups if used with multiprocessing.
# To be clear, do not expect pyopencl to work in a subprocess if it is
# imported before the fork().
my_gpyfft = None
my_pyopencl = None
clfftRadices = None
def import_my_gpyfft():
    global my_gpyfft
    global my_pyopencl
    global clfftRadices
    if my_gpyfft is None:
        import gpyfft
        my_gpyfft = gpyfft
        clfftVersion = my_gpyfft.GFFT.get_version()
        (major, minor, patch) = clfftVersion
        if major > 2 or (major == 2 and minor >= 12):
            clfftRadices = set([2, 3, 5, 7, 11, 13])
        elif (major == 2 and minor >= 8):
            clfftRadices = set([2, 3, 5, 7])
        else:
            clfftRadices = set([2, 3, 5])
    if my_pyopencl is None:
        import pyopencl
        my_pyopencl = pyopencl

def good_gpyfft_size(trylen):
    global clfftRadices
    while trylen != 1:
        foundradix = None
        for radix in clfftRadices:
            if trylen % radix == 0:
                foundradix = radix
        if foundradix is not None:
            trylen /= foundradix
        else:
            break
    if trylen == 1:
        return True
    return False

class BatchPlan(object):
    Rules = collections.namedtuple(
        "Rules",
        (
            "collapse", # sequence: each element contains a sequence of (neighboring) dimensions in the original array that should be collapsed into a single dimension
            "ndim", # int: number of dimensions in collapsed array
            "axes", # tuple: these are the axes in the collapsed input that can be currently processed in one call to clFFT

            # The following fields will be set to None if no batch dimension
            "batchdim", # int: we can process multiple elements along this dimension at once
            "batchdimsize", # int: size of the batch dimension
            "batchchunk", # int: this is how many elements along batchdim to process at once

            "alignsteps", # tuple: for each dimension, the number of elements needed to jump to ensure alignment.  Only used internally.
            "firstlooper", # int: first dimension along which we should loop
            "bufsize", # int: the maximum size of the chunk of data that will be passed to clFFT
            "initindices", # sequence: for each dimension, the initial index (or slice) used to start processing
            
            "outviewtype", # dtype: if not None, input array should be converted to a view with the given type before proceeding with the next Rule
            "outviewaxis", # dtype: if not None, axis along which view should be taken
        )
    )
    
    def __init__(self, queue, in_array, axes, maxmem=None, keeparrays=True, out_array=None):
        """
        Usage:
        
        bp = BatchPlan(queue, in_array, axes, maxmem=None, keeparrays=True)

        This prepares a plan for processing a GPU array in_array given
        a list of potential transform axes (axes).  First it attempts
        to collapse neighboring non-transform dimensions as long as it
        is possible to do so without copying the data (i.e. if you can
        traverse a set of dimensions using a single stride).

        Then it determines whether whole dataset can be processed at
        once or whether it needs to be processed in chunks.  Only one
        non-transformed dimension can be chunked -- this is the
        "batch" dimension.  All other dimensions are looped one item
        at a time.

        This class tries to make a best guess of how big the chunks
        should be based on the memory capacity of the device, but it
        assumes, of course, that nothing else is taking up space.  You
        can override its guess of how much device memory is available
        by specifying the 'maxmem' keyword argument.  This is a very
        loose estimate (does not take into account temporary space
        needed by the transform, etc.).

        If all axes in 'axes' cannot be processed in a batch (only
        reason would be device alignment issues), then the axes will
        be divided into groups and processed separately, if possible.

        Data arrays can be batch-processed using the method
        batch_loop().

        The batches (i.e. slices of the input array(s)) resulting from
        the plan can be retrieved directly using the get_batches()
        generator.

        The original input/output array is stored for use by
        batch_loop, if needed.  If you will be sending your own
        array(s) to batch_loop, and don't want BatchPlan to keep a
        reference to the original array(s), then either send
        keeparrays=False, or call bp.release_arrays().
        """
        #print "BatchPlan(in_array(shape=%s strides=%s), out_array(shape=%s strides=%s), axes=%s)" % (in_array.shape, in_array.strides, out_array.shape if out_array is not None else None, out_array.strides if out_array is not None else None, axes)

        in_array_orig = in_array

        assert len(axes) > 0, "Empty axes!"
        intype = in_array.dtype
        inkind = intype.kind
        outtype = None
        outkind = None
        outviewtype = None
        outviewaxis = None
        if out_array is not None:
            outtype = out_array.dtype
            outkind = outtype.kind
            if outkind == 'f':
                # complex-to-real
                assert inkind == 'c', "real output requires complex input"
                if in_array.base_data is out_array.base_data:
                    assert in_array.strides[axes[0]] == intype.itemsize and out_array.strides[axes[0]] == outtype.itemsize, "in-place complex-to-real transforms require elements along first transformed axis to be contiguous!"
                    outviewtype = outtype
                    outviewaxis = axes[0]
        if inkind == 'f':
            # real-to-complex
            if out_array is None or in_array.base_data is out_array.base_data:
                #print "axes[0]=%s in_array.strides=%s intype=%s intype.itemsize=%s" % (axes[0], in_array.strides, intype, intype.itemsize)
                assert in_array.strides[axes[0]] == intype.itemsize, "in-place real-to-complex transforms require elements along first transformed axis to be contiguous!"
                outviewtype = numpy.dtype('=c' + str(intype.itemsize * 2))
                outviewaxis = axes[0]
            if out_array is None:
                out_array = self.force_view(in_array, outviewtype, outviewaxis)
            assert outkind is None or outkind == 'c', "real input requires complex output (if output is specified)"
        del intype, inkind, outtype, outkind

        # get an idea of how much memory is available on device and its
        # alignment requirement (in case we need to use sub-buffers)
        device = queue.get_info(my_pyopencl.command_queue_info.DEVICE)
        bytesavailable = device.get_info(my_pyopencl.device_info.GLOBAL_MEM_SIZE)
        if maxmem is None:
            maxmem = device.get_info(my_pyopencl.device_info.MAX_MEM_ALLOC_SIZE)
        bufalign = device.get_info(my_pyopencl.device_info.MEM_BASE_ADDR_ALIGN) / 8

        self.rules_list = []
        while len(axes) > 0:
            tryaxes = axes
            axes = []
    
            while len(tryaxes) > 0:
                (collapse, newshape, newstrides, newaxes) = self._calc_collapse(in_array.shape, in_array.strides, leavedims=tryaxes)
    
                newndim = len(newshape)
    
                # for all dimensions, determine minimum appropriate batch step sizes
                # (i.e., if we choose to step along that dimension, how big must the
                # steps be to ensure that the beginning of each step follows the
                # device alignment?).
                stepmins = list(newshape)
                for dimind in range(newndim):
                    curstride = newstrides[dimind]
                    cursize = newshape[dimind]
                    bufoffset = in_array.offset
                    if bufoffset % bufalign != 0:
                        # starting points of all buffers must be aligned
                        raise Exception("Dimension %d has buffer offset %d, but buffers on this device must be aligned to %d bytes!" % (dimind, bufoffset, bufalign))
    
                    gcd = fractions.gcd(curstride, bufalign) # greatest common denominator
                    # A batch of:
                    #    stepmin * curstride
                    # bytes where:
                    #    stepmin = (alignment / gcd)
                    # will ensure start of each batch is aligned.  For example,
                    # if curstride is a multiple of alignment, then gcd == alignment,
                    # and stepmin will be 1.  If not, then stepmin is the minimum
                    # multiple of curstride that will be a multiple of alignment.
                    stepmin = bufalign / gcd
                    if stepmin < cursize:
                        stepmins[dimind] = stepmin
                #print "stepmins=%s" % (stepmins,)
    
                # Any dimensions that are not in tryaxes have to be looped or
                # batched, but at most one can be batched.  Only dimensions
                # with stepmins[dim] == 1 can be looped (otherwise we will skip
                # processing of some data).  Look for candidates.
                nonaxes = [ x for x in range(newndim) if x not in newaxes ]
                # nonaxes is sorted by stride (from _calc_collapse()), so we
                # can easily prefer batching for smallest strides and looping
                # for largest strides
                nonloop_dims = []
                loop_dims = []
                batchdim = None
                for x in nonaxes:
                    if stepmins[x] == 1:
                        loop_dims.append(x)
                    else:
                        if batchdim is None:
                            batchdim = x
                        else:
                            nonloop_dims.append(x)
                if len(nonloop_dims) > 0:
                    raise Exception("Found a dimension that can't be looped or batched! (shape=%s strides=%s axes=%s stepmins=%s device_align_bytes=%d)" % (newshape, newstrides, tryaxes, stepmins, bufalign))
                if batchdim is None and loop_dims:
                    batchdim = loop_dims.pop(0)
                if loop_dims:
                    firstlooper = loop_dims[0]
                else:
                    firstlooper = newndim
                #print "tryaxes=%s batchdim=%s firstlooper=%s" % (tryaxes, batchdim, firstlooper)
    
                # Start by assuming we will fully process all dims, then work back
                # if we don't have enough memory.
                # XXX How do we know how much memory clFFT needs on device in
                # XXX addition to the already uploaded array?  Shouldn't we include
                # XXX the size of the current input array?
                batchchunk = None
                bufsize = in_array.dtype.itemsize
                for axis in newaxes:
                    bufsize *= newshape[axis]
                if batchdim is not None:
                    bufsize *= newshape[batchdim]
                #print "bufsize=%s maxmem=%s" % (bufsize, maxmem)
                if bufsize > maxmem:
                    if batchdim is not None and stepmins[batchdim] != newshape[batchdim]:
                        # batch dim can be chunked -- see if it's enough
                        trybufsize = (bufsize / newshape[batchdim]) * stepmins[batchdim]
                        if trybufsize <= maxmem:
                            factor = maxmem // trybufsize
                            batchchunk = stepmins[batchdim] * factor
                            bufsize = trybufsize * factor
                            #print "chunking batchdim %s, batchchunk=%s" % (batchdim, batchchunk)
                #print "bufsize=%s maxmem=%s" % (bufsize, maxmem)
                if bufsize > maxmem:
                    # now try removing last axis (unless it's the only axis)
                    if len(newaxes) > 1:
                        axes.insert(0, tryaxes[-1]) # transform next time
                        tryaxes = tryaxes[:-1]
                        # loop around in case new non-transformed axis can
                        # be collapsed with others
                        continue
    
                #print "bufsize=%s maxmem=%s" % (bufsize, maxmem)
                if bufsize > maxmem:
                    raise Exception("Won't be able to allocate enough memory for transform!")
    
                batchdimsize = None
                if batchdim is not None:
                    batchdimsize = newshape[batchdim]
    
                """
                Create an initial list of parameters for processing an array X
                (X itself need not be specified here) in chunks.
                The returned parameters will be used by batch_loop().
                """
    
                # set up initial loop indices
                loopindices = [0] * (newndim + 1) # loopindices[newndim] will be sentinel
                for axis in newaxes:
                    loopindices[axis] = slice(0, None) # allows for the same slices to support looping through padded/unpadded slices of the same base array
                if batchdim is not None:
                    if batchchunk is None:
                        batchchunk = batchdimsize
                    #assert batchchunk <= batchdimsize # not necessary to assert this?
                    batchremainder = batchdimsize % batchchunk
                    if batchchunk is not None:
                        loopindices[batchdim] = slice(0, batchchunk)
                    else:
                        loopindices[batchdim] = slice(0, batchdimsize)
    
                self.rules_list.append(BatchPlan.Rules(collapse, newndim, newaxes, batchdim, batchdimsize, batchchunk, stepmins, firstlooper, bufsize, loopindices, outviewtype, outviewaxis))

                tryaxes = [] # finished with this group
            
            if outviewtype is not None and out_array is not None:
                # if we go around again, we have to use new view on array
                in_array = out_array
                outviewtype = None
                outviewaxis = None

        self.queue = queue
        self.in_array = None
        self.out_array = None
        if keeparrays:
            self.in_array = in_array_orig
            self.out_array = out_array

    def release_arrays(self):
        self.in_array = None
        self.out_array = None

    @staticmethod
    def force_view(in_array, outviewtype, axis):
        """
        numpy.ndarray.view() won't work for arrays that are not pure C-
        or fortran-contiguous when the new type is of different size,
        even if the fastest/altered axis (the only one that really
        matters) is contiguous.  So we provide this function to trick it.
        """
        # pre-condition: axis is a contiguous dimension
        insize = in_array.dtype.itemsize
        outsize = outviewtype.itemsize
        newstrides = list(in_array.strides)
        newshape = list(in_array.shape)
        if insize > outsize:
            assert insize % outsize == 0
            factor = insize / outsize
            assert newstrides[axis] % factor == 0
            newstrides[axis] //= factor
            newshape[axis] *= factor
        else:
            assert outsize % insize == 0
            factor = outsize / insize
            newstrides[axis] *= factor
            assert newshape[axis] % factor == 0
            newshape[axis] //= factor
        # trick numpy into allowing view by assigning fake contiguous strides
        out_array = in_array.view()
        out_array.strides = numpy.cumprod(numpy.hstack([[1], in_array.shape[:-1]])) * in_array.dtype.itemsize
        out_array = in_array.view(outviewtype)
        out_array.strides = tuple(newstrides)
        out_array.shape = tuple(newshape)
        return out_array
    
    @staticmethod
    def _calc_collapse(shape, strides, leavedims=[]):
        # post-condition: dimensions in returned collapse rule will be
        # ordered by increasing stride.
        #print "_calc_collapse: shape=%s strides=%s leavedims=%s" % (shape, strides, leavedims)
        if set(leavedims) == set(range(len(shape))):
            return ([[x] for x in range(len(shape))], shape, strides, leavedims)
        oldndim = len(oldshape)
        oldshape = shape
        oldstrides = strides

        # this will store the dims sorted by stride
        olddimsbystride = zip(*sorted(zip([x for x in range(oldndim)], strides), key=operator.itemgetter(1)))[0]
        #print "_calc_collapse: olddimsbystride=%s" % (olddimsbystride,)

        newndim = oldndim
        curnewdim = 0
        curoldind = 0
        curolddim = olddimsbystride[curoldind]
        newshape = [oldshape[curolddim]]
        newstrides = [oldstrides[curolddim]]
        newleavedims = []
        if curolddim in leavedims:
            newleavedims.append(curnewdim)
        rule_collapse = [[curolddim]]
        while curoldind < oldndim - 1:
            curolddim = olddimsbystride[curoldind]
            # we will attempt to collapse current dimension with next
            # precondition: curolddim is a member of rule_collapse[-1]
            cursize = newshape[curnewdim]
            nextolddim = olddimsbystride[curoldind + 1]
            nextnewdim = curnewdim + 1
            #print "_calc_collapse: enter loop curoldind=%s curolddim=%s curnewdim=%s nextolddim=%s nextnewdim=%s cursize=%s rule_collapse=%s" % (curoldind, curolddim, curnewdim, nextolddim, nextnewdim, cursize, rule_collapse)
            if curolddim not in leavedims and nextolddim not in leavedims and nextnewdim < newndim and newshape[curnewdim] * newstrides[curnewdim] == oldstrides[nextolddim]:
                # collapsing with next dimension will maintain current stride
                rule_collapse[-1].append(nextolddim)
                nextsize = oldshape[nextolddim]
                newsize = cursize * nextsize
                newshape[curnewdim] = newsize
                #print " collapsed dim: rule_collapse=%s newshape=%s" % (rule_collapse, newshape)
                #print "Collapsing dim %d (size=%d) with next dim %d (size=%d) to make new dimension %d (size=%d)" % (curnewdim, cursize, nextnewdim, nextsize, curnewdim, newsize)
                curolddim = nextolddim
                curoldind += 1
                newndim -= 1
                # don't increment curnewdim, just continue
                continue
            if nextolddim < oldndim:
                rule_collapse.append([nextolddim])
                newshape.append(oldshape[nextolddim])
                newstrides.append(oldstrides[nextolddim])
                if nextolddim in leavedims:
                    newleavedims.append(nextnewdim)
            #print " no collapse: rule_collapse=%s" % (rule_collapse,)
            curoldind += 1
            curnewdim = nextnewdim
        #print "returning rule_collapse=%s newshape=%s newstrides=%s leavedims=%s" % (rule_collapse, newshape, newstrides, newleavedims)
        return (rule_collapse, newshape, newstrides, newleavedims)

    @profile
    def collapse_array(self, x, rule_collapse):
        """
        Return a view of the array that is collapsed along the non-transformed
        dimensions.
        pre-condition: dimensions in rule_collapse must be sorted by
        increasing strides.
        """
        #print "collapse_array: rule_collapse=%s x.shape=%s x.strides=%s" % (rule_collapse, x.shape, x.strides)
        oldshape = x.shape
        x = x.transpose(list(itertools.chain(*rule_collapse)))
        #print "                transposed.shape=%s strides=%s" % (x.shape, x.strides)
        newshape = [1] * len(rule_collapse)
        for (newdim, dimgroup) in enumerate(rule_collapse):
            for olddim in dimgroup:
                newshape[newdim] *= oldshape[olddim]
        x = x.reshape(newshape, order='F')
        #print "                retval.shape=%s strides=%s" % (x.shape, x.strides)
        return x

    @profile
    def get_batches(self, *arrays):
        return self._get_batches_aux(*arrays)

    @profile
    def get_unique_batches(self, *arrays):
        return self._get_batches_aux(*arrays, unique_only=True)

    @profile
    def _get_batches_aux(self, *arrays, unique_only=False):
        startenum = 0
        for (ruleind, rules) in enumerate(self.rules_list):
            nextenum = startenum + 1

            loopindices = list(rules.initindices)
            axes = rules.axes
            firstlooper = rules.firstlooper
            batchdim = rules.batchdim
            batchchunk = rules.batchchunk
            outviewtype = rules.outviewtype
            outviewaxis = rules.outviewaxis

            #print "get_batches_aux: ruleind=%d loopindices=%s shapes=%s axes=%s firstlooper=%s batchdim=%s batchchunk=%s" % (ruleind, loopindices, [x.shape for x in arrays], axes, firstlooper, batchdim, batchchunk)

            assert batchchunk is None or batchchunk > 0
            newarrays = [self.collapse_array(x, rules.collapse) for x in arrays]
            # make sure all arrays match in non-transformed dimensions
            ndim = newarrays[0].ndim
            shape0 = newarrays[0].shape
            for array in newarrays[1:]:
                assert ndim == array.ndim
                for dim in range(ndim):
                    if dim not in axes:
                        assert newarrays[0].shape[dim] == array.shape[dim]
            if batchchunk is not None and firstlooper > batchdim:
                # we will "loop" through batchdim too
                firstlooper = batchdim
            stepsizes = [1] * (ndim + 1)
            batchdimsize = None
            if batchdim is not None:
                batchdimsize = shape0[batchdim]
                if batchchunk is not None:
                    stepsizes[batchdim] = batchchunk
                else:
                    stepsizes[batchdim] = shape0[batchdim]
            numsteps = 1
            for dimind in range(ndim):
                if dimind in axes:
                    continue
                numsteps *= int(ceil(1. * shape0[dimind] / stepsizes[dimind]))
            # there will be at most two unique batch shape/stride combinations
            # in the batch dimension -- the last chunk in the batch dimension
            # may be smaller.
            #print "numsteps=%s" % (numsteps,)
            incdim = -1
            stepnum = 0
            while loopindices[-1] == 0:
                stepnum += 1
                #print "loop step: %d/%d\r" % (stepnum, numsteps),
                #print "loopindices: ", loopindices
                #tmpdata = xinput_g.get()[tuple(loopindices[0:ndim])] ; print "input: mean=%s max=%s\n" % (numpy.mean(tmpdata), numpy.max(tmpdata)), xinput_g.get()[tuple(loopindices[0:ndim])]
                li = tuple(loopindices[0:ndim])
                subarrays = tuple( x[li] for x in newarrays )
                #subarray = subarrays[0]; print "xinput_g shape=%s strides=%s base_data.ptr=%s base_data.size=%s offset=%s" % (subarray.shape, subarray.strides, subarray.base_data.ptr, subarray.base_data.size, subarray.offset)
                enum = startenum
                #print "incdim=%s batchdim=%s batchchunk=%s loopindices[batchdim]=%s batchdimsize=%s" % (incdim, batchdim, batchchunk, loopindices[batchdim] if batchdim is not None else None, batchdimsize)
                if incdim == batchdim and batchchunk is not None and loopindices[batchdim].stop > batchdimsize:
                    enum = startenum + 1
                    nextenum = startenum + 2
                #print "enum=%d subarrays (shapes, strides, base_data.ptr, base_data.size, offset)=%s" % (enum, "[" + ", ".join("(%s, %s, %s, %s, %s)" % (xinput_sub_g.shape, xinput_sub_g.strides, xinput_sub_g.base_data.ptr, xinput_sub_g.base_data.size, xinput_sub_g.offset) for xinput_sub_g in subarrays) + "]",)
                yield (enum, rules, subarrays)
                #time.sleep(0.5)
                #print "xinput_returned_g shape=%s strides=%s base_data.ptr=%s offset=%s" % (xinput_sub_g.shape, xinput_sub_g.strides, xinput_sub_g.base_data.ptr, xinput_sub_g.offset)
                #del subarrays
                #my_pyopencl.wait_for_events(es); tmpdata = xinput_g.get()[tuple(loopindices[0:ndim])]; print "output: mean=%s max=%s\n" % (numpy.mean(tmpdata), numpy.max(tmpdata)), xinput_g.get()[tuple(loopindices[0:ndim])]
                incdim = firstlooper
                if unique_only:
                    #print "unique only: enum=%s" % (enum,)
                    if batchdim is not None and batchchunk is not None and (batchdimsize % batchchunk) != 0:
                        if enum != startenum:
                            # did both batches
                            return
                        # fast-forward to last batch
                        #print "fast-forwarding"
                        batchslice = loopindices[batchdim]
                        while batchslice.stop < batchdimsize:
                            batchslice = slice(batchslice.stop, batchslice.stop + batchchunk)
                        loopindices[batchdim] = batchslice
                        incdim = batchdim
                        continue
                    else:
                        # all batches are the same
                        return
                while incdim < ndim + 1:
                    if incdim in axes:
                        incdim += 1
                        continue
                    #print "old loopindices: ", loopindices, " incdim: ", incdim
                    if incdim == batchdim:
                        batchslice = loopindices[incdim]
                        batchslice = slice(batchslice.stop, batchslice.stop + batchchunk)
                        #print " new loopindices: ", loopindices, " incdim: ", incdim
                        if batchslice.start < batchdimsize:
                            # done!
                            loopindices[incdim] = batchslice
                            break
                        else:
                            # we wrapped around this dimension
                            loopindices[incdim] = slice(0, batchchunk)
                    else:
                        loopindices[incdim] += stepsizes[incdim]
                        #print " new loopindices: ", loopindices, " incdim: ", incdim
                        if incdim == ndim or loopindices[incdim] < shape0[incdim]:
                            # done!
                            break
                        else:
                            # we wrapped around this dimension
                            loopindices[incdim] = 0
                    #print " new loopindices: ", loopindices, " incdim: ", incdim
                    incdim += 1
            startenum = nextenum
            #print "Looped %d steps" % (stepnum,)

            if outviewtype is not None:
                arrays = [self.force_view(x, outviewtype, outviewaxis) for x in arrays]

    @profile
    def batch_loop(self, callfunc, args, arrays=None):
        """
        Call the function callfunc for each batch as so:

        for batch in batches:
            args = callfunc(batchenum, rules, subarrays, args)
        
        where subarrays is a iterable containing the same slice of
        each of the input arrays which come either from those sent to
        the BatchPlan constructor or overridden by the 'arrays'
        keyword argument of this function.  batchenum is an integer
        that corresponds to unique batch shapes.  rules is a
        namedtuple containing several fields used to coordinate
        batching of the original array.  The return value of callfunc
        is passed as the new value of 'args' for the next iteration,
        much like the reduce() pattern.
        """
        if arrays is None:
            if self.in_array is None:
                raise Exception("No default array available!")
            arrays = [ self.in_array ]
        for batchentry in self.get_batches(*arrays):
            (curbatchenum, currules, curbatch) = batchentry
            args = callfunc(curbatchenum, currules, curbatch, args)
        return args

class FFTBatchPlan(BatchPlan):
    def __init__(self, context, queue, in_array, axes, **kwargs):
        BatchPlan.__init__(self, queue, in_array, axes, **kwargs)
        self.context = context
        self.queue = queue
        arrays = [in_array]
        if 'out_array' in kwargs:
            out_array = kwargs['out_array']
            if out_array is not None:
                arrays.append(out_array)
        import_my_gpyfft()
        self.fftobjs = self.get_fft_objs(arrays)
        if 'keeparrays' not in kwargs or kwargs['keeparrays']:
            self.arrays = arrays

    @profile
    def get_fft_objs(self, arrays):
        batchgen = self.get_unique_batches(*arrays)
        return [
            self._get_plan(self.context, self.queue, *batcharrays, axes=batchrules.axes)
            for (batchenum, batchrules, batcharrays) in batchgen
        ]

    @profile
    def fft_loop(self, *no_args_allowed, callfunc=None, args=None, arrays=None, forward=None, wait_for=None):
        """
        Same as BatchPlan.batch_loop, but all arguments must be keyword
        arguments, to allow callfunc and args to be optional to override
        default FFT calling function.  Also adds 'forward' keyword argument
        to specify direction to default FFT calling function.
        """
        if len(no_args_allowed) > 0:
            raise Exception("All arguments to FFTBatchPlan.batch_loop must be keyword arguments!")
        if forward is None:
            forward = True
        if args is None:
            if wait_for is not None:
                args = wait_for
            else:
                args = []
        if callfunc is None:
            @profile
            def call(batchenum, rules, batcharrays, es, forward_=forward):
                sub_g = batcharrays[0]
                result_g = None
                if len(batcharrays) > 1:
                    result_g = batcharrays[1]
                #print "batchenum=%s sub_g.shape=%s result_g.shape=%s" % (batchenum, sub_g.shape, result_g.shape if result_g is not None else None)
                #wait_for_events(wait_for); print "sub_g mean(abs) before: %s" % (numpy.mean(numpy.abs(sub_g.get())),)
                es = self.fftobjs[batchenum].enqueue_arrays(data=sub_g, result=result_g, forward=forward_, wait_for_events=es)
                #wait_for_events(es); print "sub_g mean(abs) after: %s" % (numpy.mean(numpy.abs(sub_g.get())),)
                del sub_g
                del result_g
                return es
            callfunc = call
        if arrays is None:
            try:
                arrays = self.arrays
            except:
                raise Exception("If you don't send arrays to FFTBatchPlan constructor, you must send 'arrays' keyword argument to FFTBatchPlan.batch_loop()!")
        
        return super(FFTBatchPlan, self).batch_loop(callfunc, args, arrays)

    @profile
    def _get_plan(self, context, queue, *arrays, axes=None, nocache=False, nooutput=True):
        global my_gpyfft
        assert len(arrays) == 1 or len(arrays) == 2
        output = None
        a = arrays[0]
        if len(arrays) == 2:
            output = arrays[1]
        shape = a.shape
        outtypestr = ''
        real = False
        if a.dtype.kind == 'f':
            real = True
        inplace = True
        if output is not None:
            if output.dtype.kind == 'f':
                real = True
            outtypestr = '->%s' % (output.dtype,)
            inplace = (a.base_data is output.base_data)
        planid = "%s%s %s %s %s inplace=%s" % (a.dtype, outtypestr, a.shape, a.strides, axes, inplace)
        if not nocache and planid in CACHED_PLANS:
            plan = CACHED_PLANS[planid]
        else:
            if not nooutput:
                print "pid=%d devtype=%s, Compiling new plan for planid '%s'" % (os.getpid(), queue.device.type, planid,)
            if output is not None and output.dtype.kind == 'f':
                # output is real, input is complex hermitian, need to use
                # output shape as FFT size
                testshape = output.shape
            else:
                testshape = a.shape
            # gpyfft (and clFFT?) fails with dimensions that are not factors
            # of 2, 3 or 5 (or combinations of such) but does not tell you why,
            # so we check for it here.
            for dimind in axes:
                if not good_gpyfft_size(testshape[dimind]):
                    raise Exception("Dimension %d with shape %d has prime factors other than %s! (gpyfft/clFFT does not support that)" % (dimind, testshape[dimind], self.clfftRadices))
            plan = my_gpyfft.FFT(context, queue, a, axes=axes, out_array=output, real=real)
            CACHED_PLANS[planid] = plan
        return plan
        
