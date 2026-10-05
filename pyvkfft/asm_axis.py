"""Optional column-first composition for the experimental ASM operator."""
import threading
import json
from pathlib import Path
import numpy as np

TRANSPOSE_SOURCE = r'''
extern "C" __global__ void asm_transpose(const float2* src, float2* dst,
                                         long long h, long long w) {
    __shared__ float2 tile[32][33];
    long long x=(long long)blockIdx.x*32+threadIdx.x;
    long long y=(long long)blockIdx.y*32+threadIdx.y;
    for(int j=0;j<32;j+=8)
        tile[threadIdx.y+j][threadIdx.x]=(x<w && y+j<h)
            ? src[(y+j)*w+x] : make_float2(0.f,0.f);
    __syncthreads();
    x=(long long)blockIdx.y*32+threadIdx.x;
    y=(long long)blockIdx.x*32+threadIdx.y;
    for(int j=0;j<32;j+=8)
        if(x<h && y+j<w) dst[(y+j)*h+x]=tile[threadIdx.x][threadIdx.y+j];
}
'''


class ColumnFirstASMPlan:
    """Same compact ASM contract, with both runtime compact transposes included.

    The underlying FFT runs on reversed axes and reversed pixel pitches.
    Static H is packed directly from its original axes. Native plans
    reuse one private compact buffer (one quarter padded grid); other backends
    use two. No allocation occurs during execution.
    Use only as an explicit profile after benchmarking the complete operator.
    """

    def __init__(self, shape, pixel_pitch, mode='static', bandlimit='none',
                 stream=None, tuning_profile=None):
        import cupy as cp
        from .asm import ASMPlan
        self._closed = True
        if isinstance(tuning_profile, (str, Path)):
            tuning_profile = json.loads(Path(tuning_profile).read_text())
        tuning_profile = dict(tuning_profile or {})
        tuning_profile.pop('axis_order', None)
        self._plan = ASMPlan(tuple(shape)[::-1], tuple(pixel_pitch)[::-1], mode=mode,
                             bandlimit=bandlimit, stream=stream, tuning_profile=tuning_profile)
        self.shape = self._plan.shape[::-1]
        self.padded_shape = self._plan.padded_shape[::-1]
        self.pixel_pitch = self._plan.pixel_pitch[::-1]
        self.mode, self.bandlimit = mode, bandlimit
        self.stream, self.device = self._plan.stream, self._plan.device
        self._lock = threading.Lock()
        self._reuse_compact = self._plan._supports_compact_reuse
        with self.stream:
            self._input = cp.empty(self.shape[::-1], cp.complex64)
            self._output = self._input if self._reuse_compact else cp.empty_like(self._input)
            self._transpose_bytes = self._input.nbytes*(1 if self._reuse_compact else 2)
            self._module = cp.RawModule(code=TRANSPOSE_SOURCE)
            self._transpose = self._module.get_function('asm_transpose')
        self._closed = False

    @property
    def info(self):
        info = self._plan.info
        info.update(shape=self.shape, padded_shape=self.padded_shape, axis_order='column',
                    native_axis_mapping=['physical_y', 'physical_x'],
                    compact_buffer_reused=self._reuse_compact,
                    compact_transpose_bytes=self._transpose_bytes)
        info['workspace_bytes'] += info['compact_transpose_bytes']
        return info

    def _validate_array(self, array, shape, name):
        if self._closed:
            raise RuntimeError('ASM plan is closed')
        self._plan._validate_array(array, shape, name)

    def _copy_transposed(self, source, dest):
        h, w = source.shape
        self._transpose(((w+31)//32, (h+31)//32), (32, 8),
                        (source, dest, np.int64(h), np.int64(w)), stream=self.stream)

    def prepare_transfer(self, transfer, order='fft'):
        self._validate_array(transfer, self.padded_shape, 'transfer')
        if self.mode != 'static' or order not in ('fft', 'centered'):
            raise ValueError('Static transfer preparation requires fft or centered order')
        with self._lock, self.stream:
            if self._closed:
                raise RuntimeError('ASM plan is closed')
            self._plan._prepare_transfer_array(transfer, order, transposed=True)

    def prepare_asm_transfer(self, *, z, wavelength):
        with self._lock:
            if self._closed:
                raise RuntimeError('ASM plan is closed')
            self._plan.prepare_asm_transfer(z=z, wavelength=wavelength)

    def execute(self, source, dest, *, z=None, wavelength=None):
        self._validate_array(source, self.shape, 'source')
        self._validate_array(dest, self.shape, 'dest')
        if source.data.ptr < dest.data.ptr+dest.nbytes and dest.data.ptr < source.data.ptr+source.nbytes:
            raise ValueError('Input and output must not overlap')
        with self._lock, self.stream:
            if self._closed:
                raise RuntimeError('ASM plan is closed')
            self._copy_transposed(source, self._input)
            if self._reuse_compact:
                self._plan._execute_private_compact(self._input, z=z, wavelength=wavelength)
            else:
                self._plan.execute(self._input, self._output, z=z, wavelength=wavelength)
            self._copy_transposed(self._output, dest)
        return dest

    def close(self):
        if getattr(self, '_closed', True):
            return
        with self._lock:
            if self._closed:
                return
            self._plan.close()
            self._input = self._output = None
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
