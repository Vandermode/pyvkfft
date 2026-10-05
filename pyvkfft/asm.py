"""Experimental prepared complex64 angular-spectrum propagation on one CUDA GPU.

The operator is center_crop(ifft2(fft2(center_pad(x)) * H)); both FFT
dimensions are exactly doubled. Input/output buffers must remain alive until
the bound stream completes. The full complex phase is preserved.

Set PYVKFFT_ASM_LIBRARY to the isolated library built by examples/asm_build.py
to enable native boundary fusion. Without it the explicit-padding VkFFT path
remains available. No build or tuning is performed implicitly.
"""
import ctypes
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import threading

import numpy as np


CUDA_HELPERS = r'''
__device__ float2 asm_coefficient(long long y, long long x, long long ny, long long nx,
                                  double dy, double dx, double z, double wavelength,
                                  int bandlimit) {
    double fy = (y < (ny+1)/2 ? y : y-ny) / (ny*dy);
    double fx = (x < (nx+1)/2 ? x : x-nx) / (nx*dx);
    double invlambda = 1./wavelength;
    double q = invlambda*invlambda - (fx*fx + fy*fy);
    if (q < 0.) return make_float2(0.f, 0.f);
    if (bandlimit) {
        double tx=2.*z/(nx*dx), ty=2.*z/(ny*dy);
        if (fabs(fx)>invlambda/sqrt(1.+tx*tx) || fabs(fy)>invlambda/sqrt(1.+ty*ty))
            return make_float2(0.f, 0.f);
    }
    double sn, cs;
    sincos(6.283185307179586476925286766559*z*sqrt(q), &sn, &cs);
    return make_float2((float)cs, (float)sn);
}
'''

CUDA_KERNELS = CUDA_HELPERS + r'''
extern "C" __global__ void prepare_transfer(const float2* source, float2* dest,
    long long ny, long long nx, long long ay, long long by, long long ax, long long bx,
    int centered) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ny*nx) return;
    long long y=i/nx, x=i%nx;
    y=(y%ay)*by+y/ay; x=(x%ax)*bx+x/ax;
    if(centered) { y=(y+ny/2)%ny; x=(x+nx/2)%nx; }
    dest[i]=source[y*nx+x];
}
extern "C" __global__ void prepare_transfer_transposed(const float2* source, float2* dest,
    long long ny, long long nx, long long ay, long long by, long long ax, long long bx,
    int centered) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ny*nx) return;
    long long y=i/nx, x=i%nx;
    y=(y%ay)*by+y/ay; x=(x%ax)*bx+x/ax;
    if(centered) { y=(y+ny/2)%ny; x=(x+nx/2)%nx; }
    // Source axes are reversed relative to this plan. Pack directly, without
    // materializing a full transposed transfer function.
    dest[i]=source[x*ny+y];
}
extern "C" __global__ void prepare_transfer_transposed_tiled(const float2* source, float2* dest,
    long long ny, long long nx, long long ay, long long by, long long ax, long long bx,
    int centered) {
    __shared__ float2 tile[32][33];
    long long group=blockIdx.x;
    long long at=group%((ax+31)/32); group/=(ax+31)/32;
    long long dt=group%((by+31)/32); group/=(by+31)/32;
    long long b=group%bx, c=group/bx;
    long long d=dt*32+threadIdx.x;
    // The source's contiguous digit is d; the destination's is a. Exchange
    // these in shared memory while preserving both FFT factor permutations.
    for(int j=0;j<32;j+=8) {
        long long a=at*32+threadIdx.y+j;
        if(a<ax && d<by) {
            long long x=a*bx+b, y=c*by+d;
            if(centered) { y=(y+ny/2)%ny; x=(x+nx/2)%nx; }
            tile[threadIdx.y+j][threadIdx.x]=source[x*ny+y];
        }
    }
    __syncthreads();
    long long a=at*32+threadIdx.x;
    for(int j=0;j<32;j+=8) {
        d=dt*32+threadIdx.y+j;
        if(a<ax && d<by) dest[(d*ay+c)*nx+b*ax+a]=tile[threadIdx.x][threadIdx.y+j];
    }
}
extern "C" __global__ void generate_transfer(float2* dest,
    long long ny, long long nx, long long ay, long long by, long long ax, long long bx,
    double dy, double dx, double z, double wavelength, int bandlimit) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ny*nx) return;
    long long y=i/nx, x=i%nx;
    y=(y%ay)*by+y/ay; x=(x%ax)*bx+x/ax;
    dest[i]=asm_coefficient(y,x,ny,nx,dy,dx,z,wavelength,bandlimit);
}
extern "C" __global__ void pad_input(const float2* source, float2* dest,
                                     long long h, long long w) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=4*h*w) return;
    long long y=i/(2*w), x=i%(2*w);
    dest[i]=(y>=h/2 && y<h/2+h && x>=w/2 && x<w/2+w)
        ? source[(y-h/2)*w+x-w/2] : make_float2(0.f,0.f);
}
extern "C" __global__ void crop_output(const float2* source, float2* dest,
                                       long long h, long long w) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<h*w) dest[i]=source[(i/w+h/2)*(2*w)+i%w+w/2];
}
'''


@lru_cache(maxsize=8)
def _load_native(path):
    lib = ctypes.CDLL(path)
    if hasattr(lib, 'asm_ffi_abi_version'):
        raise RuntimeError('This is a JAX FFI backend; use pyvkfft.jax_asm and '
                           'PYVKFFT_ASM_FFI_LIBRARY instead of ASMPlan')
    lib.asm_abi_version.restype = ctypes.c_uint32
    if lib.asm_abi_version() != 2:
        raise RuntimeError('Incompatible experimental ASM library ABI')
    lib.asm_last_error.restype = ctypes.c_char_p
    lib.asm_create.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_double, ctypes.c_double,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.asm_create.restype = ctypes.c_void_p
    lib.asm_info.argtypes = [ctypes.c_void_p]
    lib.asm_info.restype = ctypes.c_char_p
    lib.asm_execute.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_double, ctypes.c_double]
    lib.asm_execute.restype = ctypes.c_int
    lib.asm_destroy.argtypes = [ctypes.c_void_p]
    lib.asm_destroy.restype = None
    return lib


class ASMPlan:
    """Prepared 2D center-padded ASM with caller-allocated compact output.

    ``mode='static'`` requires prepare_transfer(H) or prepare_asm_transfer(...).
    ``mode='dynamic'`` requires finite z and positive wavelength (SI units) on
    each execution. Pixel pitch is fixed at construction. Evanescent components
    are zeroed. ``bandlimit='rectangular'`` adds the documented per-axis cutoff.

    tuning_profile accepts a dict or JSON file with implementation ('explicit'
    or 'native'), transfer ('materialized', 'fused', 'symmetric', 'cached_phase',
    or 'cached_phase_symmetric'), aimThreads,
    coalescedMemory, groupedBatch, prune, and axis_order ('row' or 'column').
    Column order includes two compact transposes and should be selected only
    after complete-operator benchmarking. Fused dynamic H requires the native build.
    A plan serializes host submission and binds all work to its creation stream.
    close() synchronizes that stream before releasing owned resources.
    """

    def __init__(self, shape, pixel_pitch, mode='static', bandlimit='none',
                 stream=None, tuning_profile=None):
        import cupy as cp
        self._closed = True
        self._handle = None
        self._delegate = None
        if len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
            raise ValueError('shape must contain two positive integers')
        self.shape = tuple(map(int, shape))
        self.padded_shape = tuple(2*n for n in self.shape)
        if math.prod(self.padded_shape) > np.iinfo(np.int64).max // 8:
            raise ValueError('Shape byte count exceeds signed 64-bit indexing')
        if len(pixel_pitch) != 2 or not all(math.isfinite(v) and v > 0 for v in pixel_pitch):
            raise ValueError('pixel_pitch must contain two finite positive values in meters')
        if mode not in ('static', 'dynamic') or bandlimit not in ('none', 'rectangular'):
            raise ValueError('Invalid transfer mode or bandlimit policy')
        self.pixel_pitch = tuple(map(float, pixel_pitch))
        self.mode, self.bandlimit = mode, bandlimit
        self.device = cp.cuda.Device().id
        self.stream = stream if stream is not None else cp.cuda.get_current_stream()
        self._lock = threading.Lock()
        profile = tuning_profile or {}
        if isinstance(profile, (str, Path)):
            profile = json.loads(Path(profile).read_text())
        profile = dict(profile)
        allowed = {'implementation', 'transfer', 'aimThreads', 'coalescedMemory', 'groupedBatch', 'prune', 'axis_order'}
        if set(profile) - allowed:
            raise ValueError(f'Unknown tuning options: {set(profile)-allowed}')
        self.transfer_strategy = profile.get('transfer', 'materialized')
        cached = self.transfer_strategy in ('cached_phase', 'cached_phase_symmetric')
        symmetric = self.transfer_strategy == 'symmetric'
        native_variable = 'PYVKFFT_ASM_SYMMETRIC_LIBRARY' if symmetric else 'PYVKFFT_ASM_LIBRARY'
        native_path = os.environ.get(native_variable)
        self.implementation = profile.get('implementation', 'native' if native_path or cached or symmetric else 'explicit')
        if self.implementation not in ('explicit', 'native') or self.transfer_strategy not in ('materialized', 'fused', 'symmetric', 'cached_phase', 'cached_phase_symmetric'):
            raise ValueError('Unsupported implementation or transfer strategy')
        if (self.transfer_strategy == 'fused' or cached) and (mode != 'dynamic' or self.implementation != 'native'):
            raise ValueError('Fused transfer evaluation requires dynamic mode and the native implementation')
        if symmetric and (mode != 'static' or self.implementation != 'native'):
            raise ValueError('Symmetric transfer storage requires static mode and the native implementation')
        axis_order = profile.get('axis_order', 'row')
        if axis_order not in ('row', 'column'):
            raise ValueError('axis_order must be row or column')
        tuning = {k: v for k, v in profile.items() if k not in ('implementation', 'transfer', 'prune', 'axis_order')}
        for key in ('aimThreads', 'coalescedMemory'):
            if key in tuning and (not isinstance(tuning[key], int) or tuning[key] <= 0):
                raise ValueError(f'{key} must be a positive integer')
        if 'groupedBatch' in tuning:
            groups = tuning['groupedBatch']
            if len(groups) != 2 or any(not isinstance(v, int) or v < 0 for v in groups):
                raise ValueError('groupedBatch must contain two nonnegative integers')
        prune = bool(profile.get('prune', False))
        if prune and self.implementation != 'native':
            raise ValueError('Pruning requires the native backend')
        if self.implementation == 'native' and not native_path and not cached:
            builder = 'asm_symmetric_build.py' if symmetric else 'asm_build.py'
            raise RuntimeError(f'Build examples/{builder} and set {native_variable}')
        if axis_order == 'column':
            from .asm_axis import ColumnFirstASMPlan
            inner_profile = {k: v for k, v in profile.items() if k != 'axis_order'}
            self._delegate = ColumnFirstASMPlan(
                self.shape, self.pixel_pitch, mode=mode, bandlimit=bandlimit,
                stream=self.stream, tuning_profile=inner_profile)
            self._info = self._delegate.info
            self._work = self._delegate._plan._work
            self._closed = False
            return
        if cached:
            from .asm_phase import CachedPhaseASMPlan
            self._delegate = CachedPhaseASMPlan(
                self.shape, self.pixel_pitch, mode=mode, bandlimit=bandlimit,
                stream=self.stream, tuning_profile=profile)
            self._work = self._delegate._work
            self._closed = False
            return
        cuda_source = CUDA_KERNELS
        if symmetric:
            from .asm_symmetric import CUDA_SYMMETRIC, table_shape
            cuda_source += CUDA_SYMMETRIC
        self._module = cp.RawModule(code=cuda_source, options=('--std=c++11',))
        self._kernels = {name: self._module.get_function(name) for name in
                         ('prepare_transfer', 'prepare_transfer_transposed', 'prepare_transfer_transposed_tiled',
                          'generate_transfer', 'pad_input', 'crop_output')}
        if symmetric:
            self._kernels['prepare_symmetric_transfer'] = self._module.get_function('prepare_symmetric_transfer')
        with self.stream:
            self._work = cp.empty(self.padded_shape, cp.complex64)
            self._transfer = (cp.empty(table_shape(self.shape), cp.complex64) if symmetric else
                              None if self.transfer_strategy == 'fused' else cp.empty_like(self._work))
            if self.implementation == 'native':
                self._native = _load_native(str(Path(native_path).resolve()))
                if hasattr(self._native, 'asm_phase_abi_version'):
                    raise RuntimeError(f'{native_variable} selects a cached-phase backend; '
                                       'use PYVKFFT_ASM_PHASE_LIBRARY with transfer="cached_phase"')
                has_symmetric = hasattr(self._native, 'asm_symmetric_abi_version')
                if symmetric:
                    from .asm_symmetric import validate_library
                    validate_library(self._native)
                elif has_symmetric:
                    raise RuntimeError('PYVKFFT_ASM_LIBRARY selects a symmetric-table backend; '
                                       'use PYVKFFT_ASM_SYMMETRIC_LIBRARY with its matching transfer strategy')
                grouped = list(tuning.get('groupedBatch', [0, 0]))
                if len(grouped) != 2:
                    raise ValueError('groupedBatch must have two entries, fastest axis first')
                self._handle = self._native.asm_create(
                    *self.shape, *self.pixel_pitch, int(self.transfer_strategy == 'fused'),
                    int(bandlimit == 'rectangular'), int(prune), self.stream.ptr,
                    tuning.get('aimThreads', 128), tuning.get('coalescedMemory', 32), *grouped)
                if not self._handle:
                    raise RuntimeError(self._native.asm_last_error().decode())
                self._closed = False
                info = json.loads(self._native.asm_info(self._handle))
                info['library_sha256'] = hashlib.sha256(Path(native_path).read_bytes()).hexdigest()
            else:
                from .cuda import VkFFTApp
                # The legacy C wrapper accumulates element counts in a signed int.
                if math.prod(self.padded_shape) > np.iinfo(np.int32).max:
                    raise ValueError('Use the 64-bit native ASM backend for this shape')
                self._app = VkFFTApp(self.padded_shape, np.complex64, inplace=True, norm=1,
                                     convolve=True, disableReorderFourStep=True,
                                     stream=self.stream, **tuning)
                if any(self._app.use_bluestein_fft) or max(self._app.nb_axis_upload) > 2:
                    raise ValueError('ASM supports radix FFTs with at most two uploads per axis')
                config = self._app.app.contents.configuration
                info = {'uploads': self._app.nb_axis_upload, 'axis_split': self._app.axis_split[:, :2].tolist(),
                        'temporary_bytes': int(config.tempBufferSize[0]) if config.allocateTempBuffer else 0}
        ay, by = info['axis_split'][1] if info['uploads'][1] == 2 else (self.padded_shape[0], 1)
        ax, bx = info['axis_split'][0] if info['uploads'][0] == 2 else (self.padded_shape[1], 1)
        self._layout = tuple(np.int64(v) for v in (*self.padded_shape, ay, by, ax, bx))
        self._shape_args = tuple(np.int64(v) for v in self.shape)
        self._info = dict(info, shape=self.shape, padded_shape=self.padded_shape,
                          axis_order='row',
                          implementation=self.implementation, transfer_strategy=self.transfer_strategy,
                          workspace_bytes=self._work.nbytes,
                          transfer_bytes=0 if self._transfer is None else self._transfer.nbytes,
                          tuning=tuning, device=self.device)
        if symmetric:
            self._info.update(symmetric_abi=1, symmetric_layout=1,
                              transfer_table_shape=self._transfer.shape,
                              transfer_preparation='Analytic ASM only; arbitrary H uses materialized storage')
        self._prepared = False
        self._closed = False

    @property
    def info(self):
        if self._delegate is not None:
            return self._delegate.info
        return json.loads(json.dumps(self._info))

    @property
    def _supports_compact_reuse(self):
        return self.implementation == 'native' and (
            self._delegate is None or hasattr(self._delegate, '_execute_private_compact'))

    def _validate_array(self, array, shape, name):
        import cupy as cp
        if self._closed:
            raise RuntimeError('ASM plan is closed')
        if cp.cuda.Device().id != self.device:
            raise ValueError('ASM plan must execute on its creation device')
        if not isinstance(array, cp.ndarray) or array.shape != shape or array.dtype != cp.complex64:
            raise ValueError(f'{name} must be a complex64 CuPy array of shape {shape}')
        if array.device.id != self.device or not array.flags.c_contiguous:
            raise ValueError(f'{name} must be contiguous and on the plan device')

    def _launch(self, name, size, args):
        self._kernels[name](((size+255)//256,), (256,), args, stream=self.stream)

    def prepare_transfer(self, transfer, order='fft'):
        """Copy a static H into the plan's layout; enqueued on the bound stream."""
        if self._delegate is not None:
            return self._delegate.prepare_transfer(transfer, order=order)
        self._prepare_transfer_array(transfer, order)

    def _prepare_transfer_array(self, transfer, order, transposed=False):
        """Pack an ordinary or axis-reversed source without an intermediate."""
        if self.transfer_strategy == 'symmetric':
            raise ValueError('Symmetric storage requires prepare_asm_transfer; use materialized storage for supplied H')
        if self.mode != 'static':
            raise ValueError('prepare_transfer is only valid in static mode')
        if order not in ('fft', 'centered'):
            raise ValueError('Transfer order must be fft or centered')
        shape = self.padded_shape[::-1] if transposed else self.padded_shape
        self._validate_array(transfer, shape, 'transfer')
        with self._lock:
            if self._closed:
                raise RuntimeError('ASM plan is closed')
            kernel = 'prepare_transfer_transposed' if transposed else 'prepare_transfer'
            args = (transfer, self._transfer, *self._layout, np.int32(order == 'centered'))
            _, _, ay, by, ax, bx = map(int, self._layout)
            if transposed and by >= 32 and ax >= 32:
                groups = ((ax+31)//32)*((by+31)//32)*bx*ay
                self._kernels['prepare_transfer_transposed_tiled']((groups,), (32, 8),
                                                                  args, stream=self.stream)
            else:
                self._launch(kernel, self._work.size, args)
            self._prepared = True

    def prepare_asm_transfer(self, *, z, wavelength):
        """Generate static analytic ASM H directly in packed storage.

        Uses the plan's pixel pitch and band-limit policy. No full-grid source
        or transpose temporary is allocated. This is preparation: subsequent
        static executions reuse H until another preparation call replaces it.
        """
        if self.mode != 'static':
            raise ValueError('prepare_asm_transfer is only valid in static mode')
        if z is None or wavelength is None or not math.isfinite(z) or not math.isfinite(wavelength) or wavelength <= 0:
            raise ValueError('Static ASM preparation requires finite z and positive wavelength')
        if self._delegate is not None:
            return self._delegate.prepare_asm_transfer(z=z, wavelength=wavelength)
        with self._lock:
            if self._closed:
                raise RuntimeError('ASM plan is closed')
            kernel = 'prepare_symmetric_transfer' if self.transfer_strategy == 'symmetric' else 'generate_transfer'
            self._launch(kernel, self._transfer.size,
                         (self._transfer, *self._layout, *map(np.float64, self.pixel_pitch),
                          np.float64(z), np.float64(wavelength), np.int32(self.bandlimit == 'rectangular')))
            self._prepared = True

    def execute(self, source, dest, *, z=None, wavelength=None):
        """Enqueue propagation without allocation or host synchronization."""
        if self._delegate is not None:
            return self._delegate.execute(source, dest, z=z, wavelength=wavelength)
        return self._execute_arrays(source, dest, z=z, wavelength=wavelength)

    def _execute_private_compact(self, buffer, *, z=None, wavelength=None):
        """Reuse column-wrapper scratch after its first kernel has consumed it.

        Native and cached-phase plans share the first-read/final-write
        schedule. The wrapper owns this buffer; public input is never aliased.
        """
        if not self._supports_compact_reuse:
            raise RuntimeError('Private compact-buffer reuse requires a supported native plan')
        if self._delegate is not None:
            return self._delegate._execute_private_compact(buffer, z=z, wavelength=wavelength)
        return self._execute_arrays(buffer, buffer, z=z, wavelength=wavelength,
                                    private_alias=True)

    def _execute_arrays(self, source, dest, *, z=None, wavelength=None, private_alias=False):
        self._validate_array(source, self.shape, 'source')
        self._validate_array(dest, self.shape, 'dest')
        overlap = source.data.ptr < dest.data.ptr+dest.nbytes and dest.data.ptr < source.data.ptr+source.nbytes
        if overlap and not (private_alias and source is dest):
            raise ValueError('Input and output must not overlap')
        if self.mode == 'static':
            if not self._prepared or z is not None or wavelength is not None:
                raise ValueError('Static mode requires a prepared H and no optical parameters at execute')
            z, wavelength = 0., 1.
        elif z is None or wavelength is None or not math.isfinite(z) or not math.isfinite(wavelength) or wavelength <= 0:
            raise ValueError('Dynamic mode requires finite z and finite positive wavelength')
        with self._lock, self.stream:
            if self._closed:
                raise RuntimeError('ASM plan is closed')
            if self.mode == 'dynamic' and self.transfer_strategy == 'materialized':
                self._launch('generate_transfer', self._work.size,
                             (self._transfer, *self._layout, *map(np.float64, self.pixel_pitch),
                              np.float64(z), np.float64(wavelength), np.int32(self.bandlimit == 'rectangular')))
            if self.implementation == 'native':
                result = self._native.asm_execute(self._handle, source.data.ptr, dest.data.ptr,
                                                  self._work.data.ptr,
                                                  self._transfer.data.ptr if self._transfer is not None else 0,
                                                  z, wavelength)
                if result:
                    raise RuntimeError(f'Native ASM execution failed: VkFFT error {result}')
            else:
                self._launch('pad_input', self._work.size, (source, self._work, *self._shape_args))
                self._app.fft(self._work, convolve_kernel=self._transfer)
                self._launch('crop_output', source.size, (self._work, dest, *self._shape_args))
        return dest

    def close(self):
        if getattr(self, '_closed', True):
            return
        import cupy as cp
        with self._lock, cp.cuda.Device(self.device):
            if self._closed:
                return
            if self._delegate is not None:
                self._delegate.close()
                self._work = None
                self._closed = True
                return
            self.stream.synchronize()
            if self._handle:
                self._native.asm_destroy(self._handle)
                self._handle = None
            self._app = None
            self._work = self._transfer = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Explicit close reports errors. Interpreter shutdown may already
            # have destroyed the CUDA context or Python module globals.
            pass
