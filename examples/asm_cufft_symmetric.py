"""Fair cuFFT control for full or quadrant static analytic ASM transfer storage.

Uses the same pruned transforms and optional crop callback as asm_cufft. The column
orientation reuses one private compact buffer for both timed transposes.
Transfer preparation is separate from execute; caller input remains immutable.
This is an experimental benchmark control, not a general transfer-table API."""
import hashlib
import cupy as cp
import numpy as np
from asm_cufft import PrunedCuFFT
from pyvkfft.asm import CUDA_HELPERS
from pyvkfft.asm_axis import TRANSPOSE_SOURCE

class CuFFTStaticPlan:

    def __init__(self, shape, pitch, mask='none', order='row', compressed=False, stream=None, crop_callback=True):
        self.shape = tuple(shape)
        self.padded_shape = tuple((2 * n for n in shape))
        self.pitch = tuple(pitch)
        self.mask, self.order, self.compressed = (mask, order, compressed)
        self.stream = stream or cp.cuda.get_current_stream()
        self.native_shape = self.shape if order == 'row' else self.shape[::-1]
        self.native_pitch = self.pitch if order == 'row' else self.pitch[::-1]
        h, w = self.native_shape
        ny, nx = (2 * h, 2 * w)
        with self.stream:
            dummy = cp.zeros((1, 1), cp.complex64)
            source = dummy if compressed else cp.broadcast_to(dummy, (ny, nx))
            self.core = PrunedCuFFT(self.native_shape, self.native_pitch, source, dynamic=compressed, bandlimit=mask == 'rectangular', crop_callback=crop_callback)
            if compressed:
                self.core.transfer = cp.empty((w + 1, h + 1), cp.complex64)
                self.core.dynamic = False
                code = f'''
extern "C" __global__ void multiply_quadrant(float2* data,const float2* table,
 long long unused_ny,long long unused_nx,double unused_dy,double unused_dx,
 double unused_z,double unused_wavelength,int unused_dynamic,int unused_mask) {{
 unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>={ny * nx}ull)return;
 unsigned long long y=i%{ny}ull,x=i/{ny}ull;
 y=y<={h}ull?y:{ny}ull-y;x=x<={w}ull?x:{nx}ull-x;
 float2 a=data[i],b=table[x*{h + 1}ull+y];
 const float inv=1.f/{ny * nx}.f;
 data[i]=make_float2((a.x*b.x-a.y*b.y)*inv,(a.x*b.y+a.y*b.x)*inv);
}}
'''
                self.multiply_module = cp.RawModule(code=code)
                self.core.kernels['multiply_columns'] = self.multiply_module.get_function('multiply_quadrant')
                self.core.multiply_source_hash = hashlib.sha256(code.encode()).hexdigest()
            self.compact = cp.empty(self.native_shape, cp.complex64) if order == 'column' else None
            self.transpose = cp.RawModule(code=TRANSPOSE_SOURCE).get_function('asm_transpose') if self.compact is not None else None
            rows, cols = self.core.transfer.shape
            code = CUDA_HELPERS + f"""
extern "C" __global__ void prepare_static(float2* table,double z,double wavelength) {{
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>={rows * cols}ll)return;
 long long y=i%{cols}ll,x=i/{cols}ll;
 table[i]=asm_coefficient(y,x,{ny}ll,{nx}ll,{self.native_pitch[0]:.17g},{self.native_pitch[1]:.17g},z,wavelength,{int(mask == 'rectangular')});
}}
"""
            self.prepare_module = cp.RawModule(code=code)
            self.prepare = self.prepare_module.get_function('prepare_static')
        self.transfer = self.core.transfer
        self.work = self.core.columns
        self.info = dict(self.core.info, shape=self.shape, axis_order=order, transfer_strategy='static_analytic_quadrant' if compressed else 'static_analytic_full', transfer_layout='transposed natural frequency quadrant' if compressed else 'transposed full natural frequency grid', compact_buffer_reused=order == 'column', compact_transpose_bytes=0 if self.compact is None else self.compact.nbytes)
        self.info['workspace_bytes'] += self.info['compact_transpose_bytes']

    def prepare_asm_transfer(self, *, z, wavelength):
        self.prepare(((self.core.transfer.size + 255) // 256,), (256,), (self.core.transfer, np.float64(z), np.float64(wavelength)), stream=self.stream)

    def copy_transposed(self, source, dest):
        h, w = source.shape
        self.transpose(((w + 31) // 32, (h + 31) // 32), (32, 8), (source, dest, np.int64(h), np.int64(w)), stream=self.stream)

    def execute(self, x, out):
        with self.stream:
            if self.order == 'column':
                self.copy_transposed(x, self.compact)
                self.core.execute(self.compact, self.compact, 0.0, 1.0)
                self.copy_transposed(self.compact, out)
            else:
                self.core.execute(x, out, 0.0, 1.0)

    def close(self):
        self.stream.synchronize()
        self.core = self.compact = self.transfer = self.work = None
