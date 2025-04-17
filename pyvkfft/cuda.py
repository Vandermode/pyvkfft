# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import ctypes
import numpy as np
from numpy.ctypeslib import as_ctypes
from .tune import tune_vkfft

try:
    import pycuda.driver as cu_drv

    has_pycuda = True
except ImportError:
    has_pycuda = False
try:
    import cupy as cp
    import jax

    has_cupy = True
except ImportError:
    has_cupy = False
    import sys

    if has_pycuda is False and 'sphinx' not in sys.modules:
        raise ImportError("You need either PyCUDA or CuPy to use pyvkfft.cuda.")

from .base import load_library, VkFFTApp as VkFFTAppBase, check_vkfft_result, ctype_int_size_p

try:
    library = _vkfft_cuda = load_library("_vkfft_cuda")
    
    # Define constants from VkFFT
    VKFFT_MAX_FFT_DIMENSIONS = _vkfft_cuda.vkfft_max_fft_dimensions()
    
    # Basic type definitions matching VkFFT types
    pfINT = ctypes.c_int64
    pfUINT = ctypes.c_uint64
    pfLD = ctypes.c_longdouble  # long double equivalent
    pfUINT_array = pfUINT * VKFFT_MAX_FFT_DIMENSIONS
    

    class VkFFTConfiguration(ctypes.Structure):
        """Python ctypes representation of VkFFTConfiguration structure for CUDA backend."""
        _fields_ = [
            # Required parameters
            ("FFTdim", pfUINT),  # FFT dimensionality (1, 2 or 3)
            ("size", pfUINT_array),  # WHD -system dimensions
            ("device", ctypes.POINTER(ctypes.c_void_p)),  # pointer to CUDA device
            ("stream", ctypes.POINTER(ctypes.c_void_p)),  # pointer to CUDA streams
            ("num_streams", pfUINT),  # number of streams for asynchronous execution
            
            # Data parameters
            ("userTempBuffer", pfUINT),  # manual user allocation
            ("bufferNum", pfUINT),  # number of buffers passed
            ("tempBufferNum", pfUINT),  # number of temp buffers passed
            ("inputBufferNum", pfUINT),  # number of input buffers passed
            ("outputBufferNum", pfUINT),  # number of output buffers passed
            ("kernelNum", pfUINT),  # number of kernel buffers passed
            
            # Buffer size arrays
            ("bufferSize", ctypes.POINTER(pfUINT)),  # buffer sizes in bytes
            ("tempBufferSize", ctypes.POINTER(pfUINT)),  # temp buffer sizes
            ("inputBufferSize", ctypes.POINTER(pfUINT)),  # input buffer sizes
            ("outputBufferSize", ctypes.POINTER(pfUINT)),  # output buffer sizes
            ("kernelSize", ctypes.POINTER(pfUINT)),  # kernel buffer sizes
            
            # Buffer pointers
            ("buffer", ctypes.POINTER(ctypes.c_void_p)),  # computation buffers
            ("tempBuffer", ctypes.POINTER(ctypes.c_void_p)),  # temp buffers
            ("inputBuffer", ctypes.POINTER(ctypes.c_void_p)),  # input buffers
            ("outputBuffer", ctypes.POINTER(ctypes.c_void_p)),  # output buffers
            ("kernel", ctypes.POINTER(ctypes.c_void_p)),  # kernel buffers
            
            # Offset specifications
            ("specifyOffsetsAtLaunch", pfUINT),  # specify offsets at launch
            ("bufferOffset", pfUINT),  # buffer offset
            ("tempBufferOffset", pfUINT),  # temp buffer offset
            ("inputBufferOffset", pfUINT),  # input buffer offset
            ("outputBufferOffset", pfUINT),  # output buffer offset
            ("kernelOffset", pfUINT),  # kernel offset
            
            # Complex component handling
            ("bufferSeparateComplexComponents", pfUINT),  # manage buffer complex numbers as separate R and I
            ("tempBufferSeparateComplexComponents", pfUINT),  # manage temp buffer complex numbers as separate R and I
            ("inputBufferSeparateComplexComponents", pfUINT),  # manage input buffer complex numbers as separate R and I
            ("outputBufferSeparateComplexComponents", pfUINT),  # manage output buffer complex numbers as separate R and I
            ("kernelSeparateComplexComponents", pfUINT),  # manage kernel complex numbers as separate R and I
            
            # Imaginary buffer offsets
            ("bufferOffsetImaginary", pfUINT),  # imaginary buffer offset
            ("tempBufferOffsetImaginary", pfUINT),  # imaginary temp buffer offset
            ("inputBufferOffsetImaginary", pfUINT),  # imaginary input buffer offset
            ("outputBufferOffsetImaginary", pfUINT),  # imaginary output buffer offset
            ("kernelOffsetImaginary", pfUINT),  # imaginary kernel offset
            
            # Performance parameters
            ("coalescedMemory", pfUINT),  # coalesced memory in bytes
            ("aimThreads", pfUINT),  # target threads per block
            ("numSharedBanks", pfUINT),  # number of shared memory banks
            ("inverseReturnToInputBuffer", pfUINT),  # return inverse transform to input buffer
            ("numberBatches", pfUINT),  # number of batches
            ("useUint64", pfUINT),  # use 64-bit addressing
            ("omitDimension", pfUINT_array),  # disable FFT for specific dimensions
            ("performBandwidthBoost", ctypes.c_int),  # reduce coalesced number for strided axes
            ("groupedBatch", pfUINT_array),  # FFTs per threadblock per dimension
            ("optimizePow2StridesTempBuffer", ctypes.c_int),  # optimize power of 2 strides
            ("inStridePadTempBuffer", pfUINT),  # pad elements for optimized strides
            ("outStridePadTempBuffer", pfUINT),  # pad elements to for optimized strides
            
            # Precision parameters
            ("doublePrecision", pfUINT),  # double precision calculations
            ("quadDoubleDoublePrecision", pfUINT),  # double-double quad precision calculations
            ("quadDoubleDoublePrecisionDoubleMemory", pfUINT),  # double-double quad precision with FP64 storage
            ("halfPrecision", pfUINT),  # half precision calculations
            ("halfPrecisionMemoryOnly", pfUINT),  # half precision for I/O only
            ("doublePrecisionFloatMemory", pfUINT),  # FP64 calculation with FP32 storage
            
            # Transform parameters
            ("performR2C", pfUINT),  # R2C/C2R decomposition
            ("performDCT", pfUINT),  # DCT transformation
            ("performDST", pfUINT),  # DST transformation
            ("performR2R", pfUINT_array),  # DCT/DST per axis
            ("disableMergeSequencesR2C", pfUINT),  # disable merging of real sequences
            ("forceCallbackVersionRealTransforms", pfUINT),  # force callback for R2C/R2R
            
            # Algorithm controls
            ("normalize", pfUINT),  # normalize inverse transform
            ("disableReorderFourStep", pfUINT),  # disable Four step algorithm unshuffling
            ("useLUT", pfINT),  # use lookup tables
            ("useLUT_4step", pfINT),  # use lookup tables for Four-step FFT
            ("makeForwardPlanOnly", pfUINT),  # forward FFT only
            ("makeInversePlanOnly", pfUINT),  # inverse FFT only
            
            # Buffer layout parameters
            ("bufferStride", pfUINT_array),  # buffer strides
            ("isInputFormatted", pfUINT),  # input buffer formatting
            ("isOutputFormatted", pfUINT),  # output buffer formatting
            ("inputBufferStride", pfUINT_array),  # input buffer strides
            ("outputBufferStride", pfUINT_array),  # output buffer strides
            ("swapTo2Stage4Step", pfUINT),  # switch to 2 upload 4-step FFT
            ("swapTo3Stage4Step", pfUINT),  # switch to 3 upload 4-step FFT
            
            # Debug and optimization parameters
            ("considerAllAxesStrided", pfUINT),  # treat non-strided axes as strided
            ("keepShaderCode", pfUINT),  # keep and print shader code
            ("printMemoryLayout", pfUINT),  # print buffer order
            ("saveApplicationToString", pfUINT),  # save compiled binaries
            ("loadApplicationFromString", pfUINT),  # load from binaries
            ("loadApplicationString", ctypes.c_void_p),  # binary data pointer
            ("disableSetLocale", pfUINT),  # disable locale setting
            
            # Bluestein algorithm parameters
            ("fixMaxRadixBluestein", pfUINT),  # control sequence padding in Bluestein
            ("forceBluesteinSequenceSize", pfUINT),  # force specific sequence size
            ("useCustomBluesteinPaddingPattern", pfUINT),  # use custom padding pattern
            ("primeSizes", ctypes.POINTER(pfUINT)),  # non-decomposable sizes
            ("paddedSizes", ctypes.POINTER(pfUINT)),  # padding sizes
            
            # Rader algorithm parameters
            ("fixMinRaderPrimeMult", pfUINT),  # start direct multiplication
            ("fixMaxRaderPrimeMult", pfUINT),  # end direct multiplication
            ("fixMinRaderPrimeFFT", pfUINT),  # start FFT convolution
            ("fixMaxRaderPrimeFFT", pfUINT),  # end FFT convolution
            ("fixMaxRaderRadixFFT", pfUINT),  # limit Rader to specific radix
            
            # Zero padding parameters
            ("performZeropadding", pfUINT_array),  # enable zero padding
            ("fft_zeropad_left", pfUINT_array),  # left zero pad boundaries
            ("fft_zeropad_right", pfUINT_array),  # right zero pad boundaries
            ("frequencyZeroPadding", pfUINT),  # frequency domain padding
            
            # Convolution parameters
            ("performConvolution", pfUINT),  # perform convolution
            ("conjugateConvolution", pfUINT),  # conjugate during convolution
            ("crossPowerSpectrumNormalization", pfUINT),  # normalize frequency multiplication
            ("coordinateFeatures", pfUINT),  # feature vector dimension
            ("matrixConvolution", pfUINT),  # matrix-vector convolution
            ("symmetricKernel", pfUINT),  # symmetric convolution kernel
            ("numberKernels", pfUINT),  # number of kernels
            ("singleKernelMultipleBatches", pfUINT),  # one kernel for multiple batches
            ("kernelConvolution", pfUINT),  # create kernel for convolution
            
            # Register parameters
            ("registerBoost", pfUINT),  # use register file to extend shared memory
            ("registerBoostNonPow2", pfUINT),  # register boost for non-power-of-2
            ("registerBoost4Step", pfUINT),  # register boost for large sequences
            
            # Memory paging parameters
            ("devicePageSize", pfUINT),  # GPU page size in KB
            ("localPageSize", pfUINT),  # local page size in KB
            
            # Device capabilities (auto-filled but can be overridden)
            ("computeCapabilityMajor", pfUINT),  # CUDA compute capability major
            ("computeCapabilityMinor", pfUINT),  # CUDA compute capability minor
            ("maxComputeWorkGroupCount", pfUINT_array),  # max work group count
            ("maxComputeWorkGroupSize", pfUINT_array),  # max work group size
            ("maxThreadsNum", pfUINT),  # max threads
            ("sharedMemorySizeStatic", pfUINT),  # static shared memory size
            ("sharedMemorySize", pfUINT),  # total shared memory size
            ("sharedMemorySizePow2", pfUINT),  # power of 2 <= shared memory size
            ("warpSize", pfUINT),  # threads per warp
            ("halfThreads", pfUINT),  # Intel fix
            ("allocateTempBuffer", pfUINT),  # auto-allocated temp buffer
            ("reorderFourStep", pfUINT),  # unshuffle Four step algorithm
            ("maxCodeLength", pfINT),  # max code generation buffer size
            ("maxTempLength", pfINT),  # max temp string buffer size
            ("autoCustomBluesteinPaddingPattern", pfUINT),  # auto padding pattern
            ("useRaderUintLUT", pfUINT),  # use LUT for Rader g_pow
            ("vendorID", pfUINT),  # GPU vendor ID
            
            # CUDA-specific fields
            ("stream_event", ctypes.POINTER(ctypes.c_void_p)),  # stream events
            ("streamCounter", pfUINT),  # stream counter
            ("streamID", pfUINT),  # stream ID
        ]
        
    class VkFFTLaunchParams(ctypes.Structure):
        _fields_ = [
            # Pointers to buffers
            ("buffer", ctypes.POINTER(ctypes.c_void_p)),                 # void* const*
            ("tempBuffer", ctypes.POINTER(ctypes.c_void_p)),             # void**
            ("inputBuffer", ctypes.POINTER(ctypes.c_void_p)),            # void* const*
            ("outputBuffer", ctypes.POINTER(ctypes.c_void_p)),           # void* const*
            ("kernel", ctypes.POINTER(ctypes.c_void_p)),                 # void* const*
            
            # Offsets for buffers (in bytes)
            ("bufferOffset", pfUINT),                           # pfUINT
            ("tempBufferOffset", pfUINT),                       # pfUINT
            ("inputBufferOffset", pfUINT),                      # pfUINT
            ("outputBufferOffset", pfUINT),                     # pfUINT
            ("kernelOffset", pfUINT),                           # pfUINT
            
            # Offsets for imaginary parts when using separate complex components
            ("bufferOffsetImaginary", pfUINT),                  # pfUINT
            ("tempBufferOffsetImaginary", pfUINT),              # pfUINT
            ("inputBufferOffsetImaginary", pfUINT),             # pfUINT
            ("outputBufferOffsetImaginary", pfUINT),            # pfUINT
            ("kernelOffsetImaginary", pfUINT),                  # pfUINT
        ]
        
    class VkFFTAxis(ctypes.Structure):
        pass
        
    class VkFFTPlan(ctypes.Structure):
        _fields_ = [
            ("actualFFTSizePerAxis", (pfUINT_array) * VKFFT_MAX_FFT_DIMENSIONS),
            ("numAxisUploads", pfUINT_array),
            ("axisSplit", (pfUINT * 4) * VKFFT_MAX_FFT_DIMENSIONS),
            ("axes", (VkFFTAxis * 4) * VKFFT_MAX_FFT_DIMENSIONS),
            ("bigSequenceEvenR2C", pfUINT),
            ("actualPerformR2CPerAxis", pfUINT_array),
            ("R2Cdecomposition", VkFFTAxis),
            ("inverseBluesteinAxes", (VkFFTAxis * 4) * VKFFT_MAX_FFT_DIMENSIONS)
        ]

    class VkFFTApplication(ctypes.Structure):
        _fields_ = [
            ("configuration", VkFFTConfiguration),
            ("localFFTPlan", ctypes.POINTER(VkFFTPlan)),
            ("localFFTPlan_inverse", ctypes.POINTER(VkFFTPlan)),
            ("actualNumBatches", pfUINT),
            ("firstAxis", pfUINT),
            ("lastAxis", pfUINT),
            ("useBluesteinFFT", pfUINT_array),
            ("bufferRaderUintLUT", ((ctypes.c_void_p) * 4) * VKFFT_MAX_FFT_DIMENSIONS),
            ("bufferBluestein", ctypes.c_void_p * VKFFT_MAX_FFT_DIMENSIONS),
            ("bufferBluesteinFFT", ctypes.c_void_p * VKFFT_MAX_FFT_DIMENSIONS),
            ("bufferBluesteinIFFT", ctypes.c_void_p * VKFFT_MAX_FFT_DIMENSIONS),
            ("bufferRaderUintLUTSize", (pfUINT * 4) * VKFFT_MAX_FFT_DIMENSIONS),
            ("bufferBluesteinSize", pfUINT_array),
            ("applicationBluesteinString", ctypes.c_void_p * VKFFT_MAX_FFT_DIMENSIONS),
            ("applicationBluesteinStringSize", pfUINT_array),
            ("numRaderFFTPrimes", pfUINT),
            ("rader_primes", pfUINT * 30),
            ("rader_buffer_size", pfUINT * 30),
            ("raderFFTkernel", ctypes.c_void_p * 30),
            ("applicationStringOffsetRader", pfUINT),
            ("currentApplicationStringPos", pfUINT),
            ("applicationStringSize", pfUINT),
            ("saveApplicationString", ctypes.c_void_p)
        ]

    class _types:
        """Aliases"""
        vkfft_config_p = ctypes.POINTER(VkFFTConfiguration)
        vkfft_app_p = ctypes.POINTER(VkFFTApplication)
        stream = ctypes.c_void_p


    _vkfft_cuda.make_config.restype = _types.vkfft_config_p
    _vkfft_cuda.make_config.argtypes = [ctype_int_size_p, ctypes.c_size_t,
                                        ctypes.c_void_p, ctypes.c_void_p, _types.stream,
                                        ctypes.c_int, ctypes.c_size_t, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t,
                                        ctype_int_size_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctype_int_size_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    _vkfft_cuda.init_app.restype = _types.vkfft_app_p
    _vkfft_cuda.init_app.argtypes = [_types.vkfft_config_p, ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_size_t),
                                     ctype_int_size_p, ctype_int_size_p]
    
    _vkfft_cuda.init_app_from_config.restype = _types.vkfft_app_p
    _vkfft_cuda.init_app_from_config.argtypes = [_types.vkfft_config_p]

    _vkfft_cuda.fft.restype = ctypes.c_int
    _vkfft_cuda.fft.argtypes = [_types.vkfft_app_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    _vkfft_cuda.ifft.restype = ctypes.c_int
    _vkfft_cuda.ifft.argtypes = [_types.vkfft_app_p, ctypes.c_void_p, ctypes.c_void_p]

    _vkfft_cuda.free_app.restype = None
    _vkfft_cuda.free_app.argtypes = [_types.vkfft_app_p]

    _vkfft_cuda.free_config.restype = None
    _vkfft_cuda.free_config.argtypes = [_types.vkfft_config_p]

    _vkfft_cuda.vkfft_max_fft_dimensions.restype = ctypes.c_uint32
    _vkfft_cuda.vkfft_max_fft_dimensions.argtypes = None
except OSError:
    # This is used for doc generation
    import sys

    if 'sphinx' in sys.modules:
        pass
    else:
        raise


class VkFFTApp(VkFFTAppBase):
    """
    VkFFT application interface, similar to a cuFFT plan.
    """

    def __init__(self, shape, dtype: type, ndim=None, inplace=True, stream=None, norm=1,
                 r2c=False, dct=False, dst=False, axes=None, strides=None, tune_config=None,
                 r2c_odd=False, convolve=False, convolve_conj=0, convolve_norm=False,
                 convolve_shape=None, verbose=False, **kwargs):
        """

        :param shape: the shape of the array to be transformed. The number
            of dimensions of the array can be larger than the FFT dimensions,
            but only for 1D and 2D transforms. 3D FFT transforms can only
            be done on 3D arrays.
        :param dtype: the numpy dtype of the source array (can be complex64 or complex128)
        :param ndim: the number of dimensions to use for the FFT. By default,
            uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
            array to perform a batched 3D FFT on all the layers. The FFT
            is always performed along the last axes if the array's number
            of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
            on the x and y axes for ndim=2.
        :param inplace: if True (the default), performs an inplace transform and
            the destination array should not be given in fft() and ifft().
        :param stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
            for the transform. This can also be the pointer/handle (int) to the
            cuda stream object. If None, the default stream will be used.
        :param norm: if 0 (unnormalised), every transform multiplies the L2
            norm of the array by its size (or the size of the transformed
            array if ndim<d.ndim).
            if 1 (the default) or "backward", the inverse transform divides
            the L2 norm by the array size, so FFT+iFFT will keep the array norm.
            if "ortho", each transform will keep the L2 norm, but that will
            involve an extra read & write operation.
        :param r2c: if True, will perform a real->complex transform, where the
            complex destination is a half-hermitian array.
            For an inplace transform, if the input data shape is (...,nx), the input
            float array should have a shape of (..., nx+2) if nx is even
            or (..., nx+1) if nx is odd, the last one or two columns
            being ignored in the input data, and the resulting
            complex array (using pycuda's GPUArray.view(dtype=np.complex64) to
            reinterpret the type) will have a shape (..., nx//2 + 1).
            For an out-of-place transform, if the input (real) shape is (..., nx),
            the output (complex) shape should be (..., nx//2+1).
            Note that for C2R transforms with ndim>=2, the source (complex) array
            is modified.
            For an inplace transform with an odd-sized x-axis, see the r2c_odd
            parameter.
        :param dct: used to perform a Direct Cosine Transform (DCT) aka a R2R transform.
            An integer can be given to specify the type of DCT (1, 2, 3 or 4).
            if dct=True, the DCT type 2 will be performed, following scipy's convention.
        :param dst: used to perform a Direct Sine Transform (DST) aka a R2R transform.
            An integer can be given to specify the type of DST (1, 2, 3 or 4).
            if dst=True, the DST type 2 will be performed, following scipy's convention.
        :param axes: a list or tuple of axes along which the transform should be made.
            if None, the transform is done along the ndim fastest axes, or all
            axes if ndim is None. For R2C transforms, the fast axis must be
            transformed.
        :param strides: the array strides - needed if not C-ordered.
        :param tune_config: this can be used to automatically generate an
            optimised set of VkFFT parameters by testing various configurations
            and measuring the FFT speed, in a manner similar to fftw's FFTW_MEASURE.
            This should be a dictionary including the backend used and the parameter
            values which will be tested.
            This is EXPERIMENTAL, as wrong parameters may lead to crashes.
            Note that this will allocate temporary GPU arrays, unless the arrays
            to used have been passed as parameters ('dest' and 'src').
            Examples:
            tune={'backend':'cupy} - minimal example, will automatically test a small
            set of parameters (4 to 10 tests). Recommended !
            tune={'backend':'cupy, 'warpSize':[8,16,32,64,128]}: this will test
            5 possible values for the warpSize.
            tune={'backend':'cupy, 'groupedBatch':[[-1,-1,-1],[8,8,8], [4,16,16}:
            this will test 3 possible values for groupedBatch. This one is more
            tricky to use.
            tune={'backend':'cupy, 'warpSize':[8,16,32,64,128], 'src':a}: this
            will test 5 possible values for the warpSize, with a given source GPU
            array. This would only be valid for an inplace transform as no
            destination array is given.
        :param r2c_odd: this should be set to True to perform an inplace r2c/c2r
            transform with an odd-sized fast (x) axis.
            Explanation: to perform a 1D inplace transform of an array with 100
                elements, the input array should have a 100+2 size, resulting in
                a half-Hermitian array of size 51. If the input data has a size
                of 101, the input array should also be padded to 102 (101+1), and
                the resulting half-Hermitian array also has a size of 51. A
                flag is thus needed to differentiate the cases of 100+2 or 101+1.
        :param convolve: create a VkFFTApp to perform an on-the-fly convolution
            using a supplied kernel. Calling the app's fft() will perform
            the complete convolution (fft+multiplication by kernel+ifft),
            bypassing saving the arrays after the FT, resulting in a ~2X
            speedup for the convolution. The kernel must be supplied
            when calling the fft().
            This supports C2C (any dimensions) and R2C (ndim>1 and inplace),
            only for radix sizes and single-upload transforms (so allowed
            sizes depend on the GPU cache size).
        :param convolve_conj: if 1, use the conjugate of the transformed
            input array. If 2, use the conjugate of the kernel.
        :param convolve_norm: if True, normalise the kernel multiplication
            (crossPowerSpectrumNormalization).
        :param convolve_shape: by default (None), the convolution kernel
            must have the same shape as the transformed array.
            Alternatively, a batch convolution can be performed e.g.
            with kernel and array shapes respectively equal to
            (ny, nx) and (n_batch, ny, nx) for a 2D batch transform.
            It is also possible to use a kernel size of shape (nz, ny, nx),
            as long as n_batch is a multiple of nz.
        :param verbose: if True, print a 1-string info about this VkFFTApp.
            See __str__ for details.

        :raises RuntimeError: if the initialisation fails, e.g. if the CUDA
            driver has not been properly initialised, or if the transform dimensions
            are not allowed by VkFFT.
        """
        if tune_config is not None:
            kwargs = tune_vkfft(tune_config, shape=shape, dtype=dtype, ndim=ndim, inplace=inplace, stream=stream,
                                norm=norm, r2c=r2c, dct=dct, dst=dst, axes=axes, strides=strides, verbose=False,
                                r2c_odd=r2c_odd, **kwargs)[0]
        super().__init__(shape, dtype, ndim=ndim, inplace=inplace, norm=norm, r2c=r2c,
                         dct=dct, dst=dst, axes=axes, strides=strides, r2c_odd=r2c_odd,
                         convolve=convolve, convolve_norm=convolve_norm,
                         convolve_conj=convolve_conj, convolve_shape=convolve_shape, **kwargs)

        self.stream = stream

        self.config = self.make_config()
        if self.config is None:
            raise RuntimeError("Error creating VkFFTConfiguration. Was the CUDA context properly initialised ?")

        # res = ctypes.c_int(0)
        # # Size of tmp buffer allocated by VkFFT - if any
        # tmp_buffer_nbytes = ctypes.c_size_t(0)
        # # 0 or 1 for each axis, only if the Bluestein algorithm is used
        # use_bluestein_fft = np.zeros(VKFFT_MAX_FFT_DIMENSIONS, dtype=int)
        # # number of axis upload per dimension
        # num_axis_upload = np.zeros(VKFFT_MAX_FFT_DIMENSIONS, dtype=int)

        # self.app = _vkfft_cuda.init_app(self.config, ctypes.byref(res),
        #                                 ctypes.byref(tmp_buffer_nbytes),
        #                                 use_bluestein_fft, num_axis_upload)
        
        if self.performZeropadding is not None:
            self.config.contents.performZeropadding = as_ctypes(self.performZeropadding) if isinstance(self.performZeropadding, np.ndarray) else self.performZeropadding
        if self.fft_zeropad_left is not None:
            self.config.contents.fft_zeropad_left = as_ctypes(self.fft_zeropad_left) if isinstance(self.fft_zeropad_left, np.ndarray) else self.fft_zeropad_left
        if self.fft_zeropad_right is not None:
            self.config.contents.fft_zeropad_right = as_ctypes(self.fft_zeropad_right) if isinstance(self.fft_zeropad_right, np.ndarray) else self.fft_zeropad_right
            
        # self.config.contents.printMemoryLayout = pfUINT(1)
        
        self.app = _vkfft_cuda.init_app_from_config(self.config)

        # check_vkfft_result(res, shape, dtype, ndim, inplace, norm, r2c, dct, dst, axes, "cuda")

        if self.app is None:
            raise RuntimeError("Error creating VkFFTApplication. Was the CUDA driver initialised ?")
        if has_pycuda:
            # TODO: This is a kludge to keep a reference to the context, so that it is deleted
            #  after the app in __delete__, which throws an error if the context does not exist
            #  anymore. Except that we cannot be sure this is the right context, if a stream
            #  has been given because we don't have access to cuStreamGetCtx from python...
            self._ctx = cu_drv.Context.get_current()

        # self.tmp_buffer_nbytes = np.int64(tmp_buffer_nbytes)
        # self.use_bluestein_fft = [bool(n) for n in use_bluestein_fft[:len(self.shape)]]
        # self.nb_axis_upload = [int(num_axis_upload[i] * (self.skip_axis[i] is False))
        #                        for i in range(len(self.shape))]
        
        # if convolve and max(self.nb_axis_upload) > 1:
        #     raise RuntimeError(f"On-the-fly convolution is not supported with axis multi-upload [{self.__str__()}]")
        if verbose:
            print(self)
    
    @property
    def buffer_size(self):
        res = np.uint64(self.config.contents.bufferSize[0])
        return res

    @property
    def tmp_buffer_nbytes(self):
        if bool(self.config.contents.allocateTempBuffer):
            res = np.uint64(self.config.contents.tempBufferSize[0])
        else:
            res = 0
        return res
    
    @property
    def use_bluestein_fft(self):
        use_bluestein_fft = np.array(self.app.contents.useBluesteinFFT)
        res = [bool(n) for n in use_bluestein_fft[:len(self.shape)]]
        return res
    
    @property
    def nb_axis_upload(self):
        num_axis_upload = np.array(self.app.contents.localFFTPlan.contents.numAxisUploads)
        res = [int(num_axis_upload[i] * (self.skip_axis[i] is False)) for i in range(len(self.shape))]
        return res
    
    @property
    def axis_split(self):
        _axis_split = np.array(self.app.contents.localFFTPlan.contents.axisSplit)
        return _axis_split[:len(self.shape)]

    def __del__(self):
        """ Takes care of deleting allocated memory in the underlying
        VkFFTApplication and VkFFTConfiguration.
        """
        if self.app is not None:
            _vkfft_cuda.free_app(self.app)
        if self.config is not None:
            _vkfft_cuda.free_config(self.config)

    def make_config(self):
        """ Create a vkfft configuration for a FFT transform"""
        if len(self.shape) > VKFFT_MAX_FFT_DIMENSIONS:
            raise RuntimeError(f"Too many FFT dimensions after collapsing non-transform axes: "
                               f"{len(self.shape)}>{VKFFT_MAX_FFT_DIMENSIONS}")

        shape = np.ones(VKFFT_MAX_FFT_DIMENSIONS, dtype=int)
        shape[:len(self.shape)] = self.shape

        skip = np.zeros(VKFFT_MAX_FFT_DIMENSIONS, dtype=int)
        skip[:len(self.skip_axis)] = self.skip_axis

        grouped_batch = np.empty(VKFFT_MAX_FFT_DIMENSIONS, dtype=int)
        grouped_batch.fill(-1)
        grouped_batch[:len(self.groupedBatch)] = self.groupedBatch

        if self.r2c and self.inplace:
            # the last one or two columns are ignored in the R array, and will be used
            # in the C array with a size nx//2+1
            if self.r2c_odd:
                shape[0] -= 1
            else:
                shape[0] -= 2

        s = 0
        if self.stream is not None:
            if has_pycuda:
                if isinstance(self.stream, cu_drv.Stream):
                    s = self.stream.handle
            if has_cupy:
                if isinstance(self.stream, cp.cuda.Stream):
                    s = self.stream.ptr
            if s == 0 and isinstance(self.stream, int):
                # Assume the ptr or handle was passed
                s = self.stream

        if self.norm == "ortho":
            norm = 0
        else:
            norm = self.norm

        # We pass fake buffer pointer addresses to VkFFT. The real ones will be
        # given when performing the actual FFT.
        dest_gpudata = 2
        if self.inplace:
            dest_gpudata = 0

        config = _vkfft_cuda.make_config(shape, self.ndim, 1, dest_gpudata, s,
                                       norm, self.precision, int(self.r2c),
                                       int(self.dct), int(self.dst),
                                       int(self.disableReorderFourStep), int(self.registerBoost),
                                       int(self.use_lut), int(self.keepShaderCode),
                                       self.n_batch, skip,
                                       int(self.coalescedMemory), int(self.numSharedBanks),
                                       int(self.aimThreads), int(self.performBandwidthBoost),
                                       int(self.registerBoostNonPow2), int(self.registerBoost4Step),
                                       int(self.warpSize), grouped_batch,
                                       int(self.forceCallbackVersionRealTransforms),
                                       int(self._convolve), int(self._convolve_conj),
                                       int(self._convolve_norm), int(self._coordinateFeatures),
                                       int(self._singleKernelMultipleBatches))
        return config

    def fft(self, src, dest=None, convolve_kernel=None):
        """
        Compute the forward FFT

        :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
        :param dest: the destination GPU array. Should be None for an inplace transform
        :param convolve_kernel: the convolution kernel, only if the application
            is configured to perform a convolution.
        :raises RuntimeError: in case of a GPU kernel launch error
        :return: the transformed array. For a R2C inplace transform, the complex view of the
            array is returned. If this is a convolution application, the full
            convolution is performed (FT, kernel multiplication, IFT)
        """
        use_cupy = False
        if has_cupy:
            # if isinstance(src, (cp.ndarray, jax.numpy.ndarray)):
            if isinstance(src, cp.ndarray):
                use_cupy = True
        if use_cupy:
            src_ptr = src.__cuda_array_interface__['data'][0]
        else:
            # Must cast the gpudata to int as it can either be a DeviceAllocation object
            # or an int (e.g. when using a view of another array)
            src_ptr = int(src.gpudata)
        if dest is not None:
            if use_cupy:
                dest_ptr = dest.__cuda_array_interface__['data'][0]
            else:
                dest_ptr = int(dest.gpudata)
        else:
            dest_ptr = src_ptr

        conv_k_ptr = 0
        if convolve_kernel is not None:
            if use_cupy:
                conv_k_ptr = convolve_kernel.__cuda_array_interface__['data'][0]
            else:
                conv_k_ptr = int(convolve_kernel.gpudata)

        if self._convolve:
            if conv_k_ptr == 0:
                raise RuntimeError("VkFFTApp.fft: convolve=True but not convolution kernel was given")
            if self._coordinateFeatures or self._singleKernelMultipleBatches:
                if convolve_kernel.shape != self._convolve_shape:
                    raise RuntimeError(f"VkFFTApp.fft: the convolution kernel shape "
                                       f"{convolve_kernel.shape} does not match "
                                       f"the one used to create the app {self._convolve_shape}")

        if self.inplace:
            if src_ptr != dest_ptr:
                raise RuntimeError("VkFFTApp.fft: dest is not None but this is an inplace transform")
            res = _vkfft_cuda.fft(self.app, int(src_ptr), int(src_ptr), int(conv_k_ptr))
            check_vkfft_result(res, src.shape, src.dtype, self.ndim, self.inplace, self.norm, self.r2c,
                               self.dct, self.dst, backend="cuda")
            if self.norm == "ortho":
                src *= self._get_fft_scale(norm=0)
            if self.r2c and not self._convolve:
                if src.dtype == np.float32:
                    return src.view(dtype=np.complex64)
                elif src.dtype == np.float64:
                    return src.view(dtype=np.complex128)
            return src
        else:
            if dest is None:
                raise RuntimeError("VkFFTApp.fft: dest is None but this is an out-of-place transform")
            if src_ptr == dest_ptr:
                raise RuntimeError("VkFFTApp.fft: dest and src are identical but this is an out-of-place transform")
            if self.r2c and not self._convolve:
                assert (dest.size == src.size // src.shape[self.fast_axis] * (src.shape[self.fast_axis] // 2 + 1))
            res = _vkfft_cuda.fft(self.app, int(src_ptr), int(dest_ptr), int(conv_k_ptr))
            check_vkfft_result(res, src.shape, src.dtype, self.ndim, self.inplace, self.norm, self.r2c,
                               self.dct, self.dst, backend="cuda")
            if self.norm == "ortho":
                dest *= self._get_fft_scale(norm=0)
            return dest

    def ifft(self, src, dest=None):
        """
        Compute the backward FFT

        :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
        :param dest: the destination GPU array. Should be None for an inplace transform
        :raises RuntimeError: in case of a GPU kernel launch error
        :return: the transformed array. For a C2R inplace transform, the float view of the
            array is returned.
        """
        if self._convolve:
            raise RuntimeError("VkFFTApp.ifft: only fft() can be used when convolve=True")
        use_cupy = False
        if has_cupy:
            if isinstance(src, cp.ndarray):
                use_cupy = True
        if use_cupy:
            src_ptr = src.__cuda_array_interface__['data'][0]
        else:
            # Must cast the gpudata to int as it can either be a DeviceAllocation object
            # or an int (e.g. when using a view of another array)
            src_ptr = int(src.gpudata)
        if dest is not None:
            if use_cupy:
                dest_ptr = dest.__cuda_array_interface__['data'][0]
            else:
                dest_ptr = int(dest.gpudata)
        else:
            dest_ptr = src_ptr
        if self.inplace:
            if dest is not None:
                if src_ptr != dest_ptr:
                    raise RuntimeError("VkFFTApp.fft: dest!=src but this is an inplace transform")
            res = _vkfft_cuda.ifft(self.app, int(src_ptr), int(src_ptr))
            check_vkfft_result(res, src.shape, src.dtype, self.ndim, self.inplace, self.norm, self.r2c,
                               self.dct, self.dst, backend="cuda")
            if self.norm == "ortho":
                src *= self._get_ifft_scale(norm=0)
            if self.r2c:
                if src.dtype == np.complex64:
                    return src.view(dtype=np.float32)
                elif src.dtype == np.complex128:
                    return src.view(dtype=np.float64)
            return src
        else:
            if dest is None:
                raise RuntimeError("VkFFTApp.ifft: dest is None but this is an out-of-place transform")
            if src_ptr == dest_ptr:
                raise RuntimeError("VkFFTApp.ifft: dest and src are identical but this is an out-of-place transform")
            if self.r2c:
                assert (src.size == dest.size // dest.shape[self.fast_axis] * (dest.shape[self.fast_axis] // 2 + 1))
                # Special case, src and dest buffer sizes are different,
                # VkFFT is configured to go back to the source buffer
                res = _vkfft_cuda.ifft(self.app, int(dest_ptr), int(src_ptr))
            else:
                res = _vkfft_cuda.ifft(self.app, int(src_ptr), int(dest_ptr))
            check_vkfft_result(res, src.shape, src.dtype, self.ndim, self.inplace, self.norm, self.r2c,
                               self.dct, self.dst, backend="cuda")
            if self.norm == "ortho":
                dest *= self._get_ifft_scale(norm=0)
            return dest


def vkfft_version():
    """
    Get VkFFT version
    :return: version as X.Y.Z
    """
    int_ver = _vkfft_cuda.vkfft_version()
    return "%d.%d.%d" % (int_ver // 10000, (int_ver % 10000) // 100, int_ver % 100)


def cuda_runtime_version(raw=False):
    """
    Get CUDA runtime version

    :param raw: if True, return the version as X*1000+Y*10+Z
    :return: version as X.Y.Z
    """
    int_ver = _vkfft_cuda.cuda_runtime_version()
    if raw:
        return raw
    return "%d.%d.%d" % (int_ver // 1000, (int_ver % 1000) // 10, int_ver % 10)


def cuda_driver_version(raw=False):
    """
    Get CUDA driver version

    :param raw: if True, return the version as X*1000+Y*10+Z
    :return: version as X.Y.Z
    """
    int_ver = _vkfft_cuda.cuda_driver_version()
    if raw:
        return raw
    return "%d.%d.%d" % (int_ver // 1000, (int_ver % 1000) // 10, int_ver % 10)


def cuda_compile_version(raw=False):
    """
    Get CUDA version against which pyvkfft was compiled

    :param raw: if True, return the version as X*1000+Y*10+Z
    :return: version as X.Y.Z
    """
    if raw:
        return raw
    int_ver = _vkfft_cuda.cuda_compile_version()
    return "%d.%d.%d" % (int_ver // 1000, (int_ver % 1000) // 10, int_ver % 10)
