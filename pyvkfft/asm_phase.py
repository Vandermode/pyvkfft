"""Optional full or symmetric phase caches for distance-dominant ASM workloads.

The isolated phase backend has its own capability marker. It does not replace
or modify the standard ABI2 library used for ordinary fused propagation.
"""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import threading

import numpy as np


CUDA_PHASE_SOURCE = r'''
extern "C" __global__ void asm_prepare_phase(double* phase,
    long long ny, long long nx, long long ay, long long by, long long ax, long long bx,
    double dy, double dx, double wavelength) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ny*nx) return;
    long long y=i/nx, x=i%nx;
    y=(y%ay)*by+y/ay; x=(x%ax)*bx+x/ax;
    double fy=(y<(ny+1)/2 ? y : y-ny)/(ny*dy);
    double fx=(x<(nx+1)/2 ? x : x-nx)/(nx*dx);
    double il=1./wavelength, q=il*il-(fx*fx+fy*fy);
    phase[i]=q<0. ? -1. : sqrt(q);
}
'''


class CachedPhaseASMPlan:
    """Internal ASMPlan delegate with exact wavelength-keyed FP64 phase reuse.

    A wavelength change prepares the phase table on the bound stream before
    propagation. The preparation is part of execute and its cost must be timed.
    Changing distance only reuses the table. Rectangular masks are unsupported
    by this optional backend and must use the ordinary fused implementation.
    """

    def __init__(self, shape, pixel_pitch, mode='dynamic', bandlimit='none',
                 stream=None, tuning_profile=None):
        import cupy as cp
        from .asm import _load_native, CUDA_HELPERS
        self._closed = True
        self._handle = None
        if mode != 'dynamic' or bandlimit != 'none':
            raise ValueError('Cached phase requires dynamic mode and bandlimit="none"')
        if len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
            raise ValueError('shape must contain two positive integers')
        self.shape = tuple(map(int, shape))
        self.padded_shape = tuple(2*n for n in self.shape)
        if math.prod(self.padded_shape) > np.iinfo(np.int64).max//8:
            raise ValueError('Shape byte count exceeds signed 64-bit indexing')
        if len(pixel_pitch) != 2 or not all(math.isfinite(v) and v > 0 for v in pixel_pitch):
            raise ValueError('pixel_pitch must contain two finite positive values')
        self.pixel_pitch = tuple(map(float, pixel_pitch))
        self.mode, self.bandlimit = mode, bandlimit
        self.implementation, self.transfer_strategy = 'native', 'cached_phase'
        self.device = cp.cuda.Device().id
        self.stream = stream if stream is not None else cp.cuda.get_current_stream()
        self._lock = threading.Lock()
        profile = tuning_profile or {}
        if isinstance(profile, (str, Path)):
            profile = json.loads(Path(profile).read_text())
        profile = dict(profile)
        allowed = {'implementation', 'transfer', 'prune', 'aimThreads', 'coalescedMemory', 'groupedBatch', 'axis_order'}
        if set(profile)-allowed:
            raise ValueError(f'Unknown tuning options: {set(profile)-allowed}')
        strategy = profile.get('transfer', 'cached_phase')
        if profile.get('implementation', 'native') != 'native' or strategy not in ('cached_phase', 'cached_phase_symmetric'):
            raise ValueError('Cached phase requires its dedicated native backend')
        symmetric = strategy == 'cached_phase_symmetric'
        self.transfer_strategy = strategy
        if profile.get('axis_order', 'row') != 'row':
            raise ValueError('Select column order through ASMPlan, not the cached-phase delegate')
        tuning = {k: v for k, v in profile.items() if k in ('aimThreads', 'coalescedMemory', 'groupedBatch')}
        for key in ('aimThreads', 'coalescedMemory'):
            if key in tuning and (not isinstance(tuning[key], int) or tuning[key] <= 0):
                raise ValueError(f'{key} must be a positive integer')
        grouped = list(tuning.get('groupedBatch', [0, 0]))
        if len(grouped) != 2 or any(not isinstance(v, int) or v < 0 for v in grouped):
            raise ValueError('groupedBatch must contain two nonnegative integers')
        variable = 'PYVKFFT_ASM_SYMMETRIC_LIBRARY' if symmetric else 'PYVKFFT_ASM_PHASE_LIBRARY'
        library_path = os.environ.get(variable)
        if not library_path:
            builder = 'asm_symmetric_build.py' if symmetric else 'asm_cached_phase_build.py'
            raise RuntimeError(f'Build examples/{builder} and set {variable}')
        path = Path(library_path).resolve()
        self._native = _load_native(str(path))
        if symmetric:
            from .asm_symmetric import CUDA_SYMMETRIC, table_shape, validate_library
            validate_library(self._native)
            if hasattr(self._native, 'asm_phase_abi_version'):
                raise RuntimeError('A symmetric backend must not advertise the full cached-phase ABI')
            phase_shape = table_shape(self.shape)
            preparation_source = CUDA_HELPERS + CUDA_SYMMETRIC
            preparation_kernel = 'prepare_symmetric_phase'
        else:
            if hasattr(self._native, 'asm_symmetric_abi_version'):
                raise RuntimeError('The selected library is a symmetric backend, not a full cached-phase backend')
            try:
                capability = self._native.asm_phase_abi_version
            except AttributeError as exc:
                raise RuntimeError('The selected library is not a cached-phase backend') from exc
            capability.argtypes = []
            capability.restype = ctypes.c_uint32
            if capability() != 1:
                raise RuntimeError('Incompatible cached-phase capability ABI')
            phase_shape = self.padded_shape
            preparation_source = CUDA_PHASE_SOURCE
            preparation_kernel = 'asm_prepare_phase'
        with self.stream:
            self._work = cp.empty(self.padded_shape, cp.complex64)
            self._phase = cp.empty(phase_shape, cp.float64)
            self._module = cp.RawModule(code=preparation_source, options=('--std=c++11',))
            self._prepare = self._module.get_function(preparation_kernel)
            self._handle = self._native.asm_create(
                *self.shape, *self.pixel_pitch, 1, 0, int(bool(profile.get('prune', False))),
                self.stream.ptr, tuning.get('aimThreads', 128), tuning.get('coalescedMemory', 32), *grouped)
            if not self._handle:
                raise RuntimeError(self._native.asm_last_error().decode())
            self._closed = False
            info = json.loads(self._native.asm_info(self._handle))
        ay, by = info['axis_split'][1] if info['uploads'][1] == 2 else (self.padded_shape[0], 1)
        ax, bx = info['axis_split'][0] if info['uploads'][0] == 2 else (self.padded_shape[1], 1)
        self._layout = tuple(np.int64(v) for v in (*self.padded_shape, ay, by, ax, bx))
        self._phase_wavelength = None
        self._phase_preparations = 0
        self._info = dict(info, shape=self.shape, padded_shape=self.padded_shape,
                          implementation='native', transfer_strategy=self.transfer_strategy, axis_order='row',
                          library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          phase_table_bytes=self._phase.nbytes, phase_table_shape=self._phase.shape,
                          workspace_bytes=self._work.nbytes, transfer_bytes=self._phase.nbytes,
                          wavelength_change_cost='Phase preparation included in execute',
                          tuning=tuning, device=self.device)
        if symmetric:
            self._info.update(symmetric_abi=1, symmetric_layout=1)
        else:
            self._info.update(phase_abi=1)

    @property
    def info(self):
        info = dict(self._info, prepared_wavelength=self._phase_wavelength,
                    phase_preparations=self._phase_preparations)
        return json.loads(json.dumps(info))

    def _validate_array(self, array, shape, name):
        from .asm import ASMPlan
        ASMPlan._validate_array(self, array, shape, name)

    def prepare_transfer(self, transfer, order='fft'):
        raise ValueError('Cached-phase mode prepares its own wavelength-dependent phase')

    def execute(self, source, dest, *, z=None, wavelength=None):
        return self._execute_arrays(source, dest, z=z, wavelength=wavelength)

    def _execute_private_compact(self, buffer, *, z=None, wavelength=None):
        """Reuse wrapper scratch between the first read and final write kernels.

        Phase preparation touches only the separate phase table. The native
        compact input is consumed by the first forward row kernel, before the
        final inverse row kernel overwrites this private buffer on the same
        stream. Public execute continues to reject every input/output overlap.
        """
        return self._execute_arrays(buffer, buffer, z=z, wavelength=wavelength,
                                    private_alias=True)

    def _execute_arrays(self, source, dest, *, z=None, wavelength=None, private_alias=False):
        self._validate_array(source, self.shape, 'source')
        self._validate_array(dest, self.shape, 'dest')
        overlap = source.data.ptr < dest.data.ptr+dest.nbytes and dest.data.ptr < source.data.ptr+source.nbytes
        if overlap and not (private_alias and source is dest):
            raise ValueError('Input and output must not overlap')
        if z is None or wavelength is None or not math.isfinite(z) or not math.isfinite(wavelength) or wavelength <= 0:
            raise ValueError('Dynamic mode requires finite z and finite positive wavelength')
        wavelength = float(wavelength)
        with self._lock, self.stream:
            if self._closed:
                raise RuntimeError('ASM plan is closed')
            if self.stream.is_capturing():
                raise RuntimeError('Cached-phase execution does not support CUDA Graph capture')
            if wavelength != self._phase_wavelength:
                self._prepare(((self._phase.size+255)//256,), (256,),
                              (self._phase, *self._layout, *map(np.float64, self.pixel_pitch),
                               np.float64(wavelength)), stream=self.stream)
                self._phase_wavelength = wavelength
                self._phase_preparations += 1
            result = self._native.asm_execute(self._handle, source.data.ptr, dest.data.ptr,
                                              self._work.data.ptr, self._phase.data.ptr, z, wavelength)
            if result:
                raise RuntimeError(f'Native cached-phase execution failed: VkFFT error {result}')
        return dest

    def close(self):
        if getattr(self, '_closed', True):
            return
        import cupy as cp
        with self._lock, cp.cuda.Device(self.device):
            if self._closed:
                return
            self.stream.synchronize()
            if self._handle:
                self._native.asm_destroy(self._handle)
                self._handle = None
            self._work = self._phase = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
