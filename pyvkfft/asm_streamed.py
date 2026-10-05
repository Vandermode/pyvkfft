"""Exact complex64 ASM with a compact workspace and explicit FFT tiles.

The capacity-oriented schedule preserves full complex phase and input values.
It uses caller output as temporary spectrum storage while an operation is in
flight. Callers must keep input/output alive until the bound stream completes.
Build the isolated ordinary FFT library with examples/asm_streamed_build.py and
set PYVKFFT_STREAMED_LIBRARY to its libvkfft_streamed.so.
"""
import ctypes
from functools import lru_cache
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import threading

import numpy as np
from .asm import CUDA_HELPERS


STREAMED_KERNELS = r'''
// A cyclic frequency window stores only columns where the transfer may be
// nonzero. Defaults preserve the full-plane CuPy and legacy FFI schedules.
#ifndef ACTIVE_COLUMNS
#define ACTIVE_COLUMNS NX
#define EXTRA_COLUMNS W
#define ACTIVE_ORIGIN_X 0
#define TRANSFER_HEIGHT NY
#define TRANSFER_WIDTH NX
#define TRANSFER_ORIGIN_Y 0
#define TRANSFER_ORIGIN_X 0
#define TRANSFER_LAYOUT 0
#endif
__device__ __forceinline__ long long active_frequency_x(long long x) {
    long long value=x+ACTIVE_ORIGIN_X;
    return value>=NX?value-NX:value;
}
extern "C" __global__ void pad_row_tile(const float2* source, float2* scratch, long long offset) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)R*NX)return;
    long long x=i%NX,y=i/NX+offset;
    scratch[i]=(y<H && x>=X0 && x<X0+W)?source[(long long)y*W+x-X0]:make_float2(0,0);
}
extern "C" __global__ void scatter_row_tile(const float2* scratch, float2* first, float2* second, long long offset) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)R*ACTIVE_COLUMNS)return;
    long long x=i%ACTIVE_COLUMNS,y=i/ACTIVE_COLUMNS+offset;
    if(y<H) {
        float2 value=scratch[(i/ACTIVE_COLUMNS)*NX+active_frequency_x(x)];
        if(x<W)first[(long long)y*W+x]=value;
        else second[(long long)y*EXTRA_COLUMNS+x-W]=value;
    }
}
extern "C" __global__ void gather_row_tile(const float2* first, const float2* second, float2* scratch, long long offset) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)R*NX)return;
    long long x=i%NX,y=i/NX+offset;
    long long k=x-ACTIVE_ORIGIN_X;
    if(k<0)k+=NX;
    scratch[i]=(y<H && k<ACTIVE_COLUMNS)
        ?(k<W?first[(long long)y*W+k]:second[(long long)y*EXTRA_COLUMNS+k-W])
        :make_float2(0,0);
}
extern "C" __global__ void crop_row_tile(const float2* scratch, float2* dest, long long offset) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)R*W)return;
    long long y=i/W+offset;
    if(y<H)dest[(long long)y*W+i%W]=scratch[(i/W)*NX+i%W+X0];
}
extern "C" __global__ void gather_split_columns(const float2* first, const float2* second, float2* dest, long long offset) {
    __shared__ float2 tile[32][33];
    long long gx=blockIdx.x%((B+31)/32),gy=blockIdx.x/((B+31)/32);
    long long x=gx*32+threadIdx.x;
    long long y=gy*32+threadIdx.y;
    #pragma unroll
    for(int j=0;j<32;j+=8) {
        long long yy=y+j;
        float2 value=make_float2(0,0);
        if(x<B && x+offset<ACTIVE_COLUMNS && yy>=Y0 && yy<Y0+H) {
            long long xx=x+offset;
            value=xx<W?first[(long long)(yy-Y0)*W+xx]:second[(long long)(yy-Y0)*EXTRA_COLUMNS+xx-W];
        }
        tile[threadIdx.y+j][threadIdx.x]=value;
    }
    __syncthreads();
    x=gx*32+threadIdx.y;
    y=gy*32+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8)
        if(x+j<B && y<NY)dest[(long long)(x+j)*NY+y]=tile[threadIdx.x][threadIdx.y+j];
}
extern "C" __global__ void scatter_split_columns(const float2* source, float2* first, float2* second, long long offset) {
    __shared__ float2 tile[32][33];
    long long gx=blockIdx.x%((H+31)/32),gy=blockIdx.x/((H+31)/32);
    long long y=gx*32+threadIdx.x;
    long long x=gy*32+threadIdx.y;
    #pragma unroll
    for(int j=0;j<32;j+=8)
        tile[threadIdx.y+j][threadIdx.x]=(x+j<B && y<H)
            ?source[(long long)(x+j)*NY+y+Y0]:make_float2(0,0);
    __syncthreads();
    y=gx*32+threadIdx.y;
    x=gy*32+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8)
        if(x<B && x+offset<ACTIVE_COLUMNS && y+j<H) {
            long long xx=x+offset;
            if(xx<W)first[(long long)(y+j)*W+xx]=tile[threadIdx.x][threadIdx.y+j];
            else second[(long long)(y+j)*EXTRA_COLUMNS+xx-W]=tile[threadIdx.x][threadIdx.y+j];
        }
}
extern "C" __global__ void multiply_transfer(float2* data, long long offset, double z, double wavelength) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)B*NY)return;
    long long px=i/NY+offset,py=i%NY;
    if(px>=NX){data[i]=make_float2(0,0);return;}
    long long x=(px%AX)*BX*CX+(px/AX%BX)*CX+px/(AX*BX);
    long long y=(py%AY)*BY*CY+(py/AY%BY)*CY+py/(AY*BY);
    float2 h=asm_coefficient(y,x,NY,NX,DY,DX,z,wavelength,BANDLIMIT);
    float2 v=data[i];
    data[i]=make_float2(v.x*h.x-v.y*h.y,v.x*h.y+v.y*h.x);
}

// Coalesced natural-order transfer loads, transposed into column FFT storage.
extern "C" __global__ void multiply_packed(float2* data, const unsigned int* transfer,
                                          long long offset, float scale, int reverse) {
    __shared__ float2 tile[32][33];
    long long gx=blockIdx.x%((B+31)/32), gy=blockIdx.x/((B+31)/32);
    long long x=gx*32+threadIdx.x, py=gy*32+threadIdx.y;
    #pragma unroll
    for(int j=0;j<32;j+=8) {
        long long yy=py+j;
        long long y=(yy%AY)*BY*CY+(yy/AY%BY)*CY+yy/(AY*BY), xx=active_frequency_x(x+offset);
        if(reverse) { y=y?NY-y:0; xx=xx?NX-xx:0; }
        #if TRANSFER_LAYOUT == 1
        long long ty=y<NY-y?y:NY-y, tx=xx<NX-xx?xx:NX-xx;
        #else
        long long ty=y-TRANSFER_ORIGIN_Y, tx=xx-TRANSFER_ORIGIN_X;
        if(ty<0)ty+=NY;
        if(tx<0)tx+=NX;
        #endif
        float2 h=make_float2(0,0);
        if(x<B && x+offset<ACTIVE_COLUMNS && yy<NY && ty<TRANSFER_HEIGHT && tx<TRANSFER_WIDTH) {
            unsigned int bits=transfer[ty*TRANSFER_WIDTH+tx];
            int re=int(bits&65535u),im=int(bits>>16);
            re=re>=32768?re-65536:re; im=im>=32768?im-65536:im;
            h=make_float2(re*scale,im*scale);
        }
        tile[threadIdx.y+j][threadIdx.x]=h;
    }
    __syncthreads();
    x=gx*32+threadIdx.y; py=gy*32+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8) if(x+j<B && py<NY) {
        long long i=(x+j)*NY+py;
        float2 v=data[i],h=tile[threadIdx.x][threadIdx.y+j];
        data[i]=make_float2(v.x*h.x-v.y*h.y,v.x*h.y+v.y*h.x);
    }
}

extern "C" __global__ void pad_row_tile_transposed(const float2* source,float2* scratch,long long offset) {
    __shared__ float2 tile[32][33];
    long long gx=blockIdx.x%((NX+31)/32),gy=blockIdx.x/((NX+31)/32);
    long long x=gx*32+threadIdx.y,y=gy*32+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8)
        tile[threadIdx.y+j][threadIdx.x]=(y<R && y+offset<H && x+j>=X0 && x+j<X0+W)
            ?source[(long long)(x+j-X0)*H+y+offset]:make_float2(0,0);
    __syncthreads();
    x=gx*32+threadIdx.x;y=gy*32+threadIdx.y;
    #pragma unroll
    for(int j=0;j<32;j+=8)
        if(x<NX && y+j<R)scratch[(long long)(y+j)*NX+x]=tile[threadIdx.x][threadIdx.y+j];
}
extern "C" __global__ void transpose_final(const float2* source,float2* dest) {
    __shared__ float2 tile[32][33];
    long long gx=blockIdx.x%((W+31)/32),gy=blockIdx.x/((W+31)/32);
    long long x=gx*32+threadIdx.x,y=gy*32+threadIdx.y;
    #pragma unroll
    for(int j=0;j<32;j+=8)
        tile[threadIdx.y+j][threadIdx.x]=(x<W && y+j<H)?source[(long long)(y+j)*W+x]:make_float2(0,0);
    __syncthreads();
    x=gx*32+threadIdx.y;y=gy*32+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8)
        if(x+j<W && y<H)dest[(long long)(x+j)*H+y]=tile[threadIdx.x][threadIdx.y+j];
}
'''


@lru_cache(maxsize=None)
def _streamed_library(path):
    try:
        library = ctypes.CDLL(path)
        if hasattr(library, 'streamed_ffi_abi_version'):
            raise RuntimeError('This is a JAX FFI backend; use its core/libvkfft_streamed.so for CuPy')
        library.streamed_fft_abi_version.restype = ctypes.c_uint32
        library.streamed_fft_capabilities.restype = ctypes.c_uint64
        if library.streamed_fft_abi_version() != 1 or library.streamed_fft_capabilities() != 1:
            raise RuntimeError('Incompatible streamed FFT ABI or capabilities')
        library.streamed_fft_last_error.restype = ctypes.c_char_p
        library.streamed_fft_create.argtypes = [ctypes.c_uint64] * 3
        library.streamed_fft_create.restype = ctypes.c_void_p
        library.streamed_fft_create_ex.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int]
        library.streamed_fft_create_ex.restype = ctypes.c_void_p
        library.streamed_fft_info.argtypes = [ctypes.c_void_p]
        library.streamed_fft_info.restype = ctypes.c_char_p
        library.streamed_fft_execute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        library.streamed_fft_execute.restype = ctypes.c_int
        library.streamed_fft_destroy.argtypes = [ctypes.c_void_p]
        library.streamed_fft_destroy.restype = None
    except (OSError, AttributeError) as exc:
        raise RuntimeError('The library is not a compatible ordinary streamed FFT backend') from exc
    return library, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


class StreamedASMPlan:
    """Prepared exact ASM using one compact private buffer and shared FFT tiles.

    Input and output are distinct, contiguous CuPy complex64 arrays of ``shape``.
    Both physical dimensions are doubled for the operator. ``bandlimit`` is
    ``'none'`` or ``'rectangular'``; evanescent components are always zeroed.

    ``axis_order='column'`` changes the internal orientation using tiled input
    gathering and one final compact transpose, without another allocation.
    ``tile_columns`` counts frequency columns in the internal orientation.
    ``tile_rows=None`` chooses a row tile fitting the column tile allocation.
    Tiles are explicit capacity/performance choices; no autotuning is performed.
    Radix decompositions with up to three uploads per axis are supported.
    ``transfer_mode='packed'`` accepts a natural-order uint32 SNORM16 plane
    through ``execute(..., transfer=payload, scale=..., transpose=False)``.
    This mode uses ordered row tiles and decodes the fixed transfer directly;
    ``transpose=True`` reverses frequencies for a bilinear field transpose.
    Caller-owned transfer storage is additional to ``info['total_operator_bytes']``.

    ``execute(source, dest, z=..., wavelength=...)`` is asynchronous and does
    not allocate GPU arrays. Every call supplies optical parameters, including
    callers which reuse fixed parameters. The plan serializes host submission
    on its bound stream. Arrays must remain alive and ready on that stream until
    execution completes; output contains intermediate data until completion.
    ``close()`` synchronizes before releasing owned storage and FFT handles.
    """

    def __init__(self, shape, pixel_pitch, bandlimit='none', stream=None,
                 tile_columns=256, tile_rows=None, axis_order='row', transfer_mode='analytic'):
        import cupy as cp
        self._closed = True
        self._row_handle = self._column_handle = None
        self._extra = self._scratch = self._row_view = self._column_view = None
        self._lock = threading.Lock()
        if len(shape) != 2:
            raise ValueError('shape must have two positive integer dimensions')
        self.shape = tuple(_positive_integer(v, 'shape dimension') for v in shape)
        if math.prod(self.shape) > np.iinfo(np.int64).max // 32:
            raise ValueError('Padded shape exceeds signed 64-bit byte arithmetic')
        if (len(pixel_pitch) != 2 or any(not isinstance(v, Real) or isinstance(v, bool) or
                                         not math.isfinite(v) or v <= 0 for v in pixel_pitch)):
            raise ValueError('pixel_pitch must contain two finite positive values')
        if bandlimit not in ('none', 'rectangular'):
            raise ValueError('bandlimit must be none or rectangular')
        if axis_order not in ('row', 'column'):
            raise ValueError('axis_order must be row or column')
        if transfer_mode not in ('analytic', 'packed'):
            raise ValueError('transfer_mode must be analytic or packed')
        if transfer_mode == 'packed' and axis_order != 'row':
            raise ValueError('Packed transfers currently require row axis order')
        self.transfer_mode = transfer_mode
        self.pixel_pitch = tuple(map(float, pixel_pitch))
        self.bandlimit, self.axis_order = bandlimit, axis_order
        self.padded_shape = tuple(2*v for v in self.shape)
        self.logical_shape = self.shape if axis_order == 'row' else self.shape[::-1]
        logical_pitch = self.pixel_pitch if axis_order == 'row' else self.pixel_pitch[::-1]
        h, w = self.logical_shape
        ny, nx = 2*h, 2*w
        self.tile_columns = b = min(_positive_integer(tile_columns, 'tile_columns'), nx)
        requested_rows = max(1, b*ny//nx) if tile_rows is None else _positive_integer(tile_rows, 'tile_rows')
        self.tile_rows = r = min(requested_rows, h)
        scratch_elements = max(r*nx, b*ny)
        if scratch_elements > np.iinfo(np.int64).max // 8:
            raise ValueError('Tile storage exceeds signed 64-bit byte arithmetic')
        path = os.environ.get('PYVKFFT_STREAMED_LIBRARY')
        if not path:
            raise RuntimeError('Set PYVKFFT_STREAMED_LIBRARY to the isolated ordinary FFT library')
        self._native, library_hash = _streamed_library(str(Path(path).resolve()))
        self.device = cp.cuda.Device().id
        self.stream = stream if stream is not None else cp.cuda.get_current_stream()
        if not isinstance(self.stream, cp.cuda.Stream):
            raise ValueError('stream must be a CuPy CUDA stream')
        stream_device = getattr(self.stream, 'device_id', -1)
        if stream_device not in (-1, self.device):
            raise ValueError('The stream belongs to a different CUDA device')
        self._nx, self._ny = nx, ny
        self._closed = False
        try:
            with self.stream:
                self._extra = cp.empty(self.logical_shape, cp.complex64)
                self._scratch = cp.empty(scratch_elements, cp.complex64)
                self._row_view = self._scratch[:r*nx].reshape(r, nx)
                self._column_view = self._scratch[:b*ny].reshape(b, ny)
                self._row_handle, row_info = self._create_fft(nx, r, ordered=transfer_mode == 'packed')
                self._column_handle, column_info = self._create_fft(ny, b)
                def factors(info, length):
                    if info.get('ordered'):
                        return length, 1, 1
                    values = info['axis_split'][:info['uploads']]
                    values += [1] * (3-len(values))
                    if math.prod(values) != length:
                        raise RuntimeError('Invalid initialized FFT frequency permutation')
                    return values
                ax, bx, cx = factors(row_info, nx)
                ay, by, cy = factors(column_info, ny)
                dimensions = dict(H=h, W=w, NY=ny, NX=nx, X0=w//2, Y0=h//2,
                                  B=b, R=r, AX=ax, BX=bx, CX=cx, AY=ay, BY=by, CY=cy)
                defines = ''.join(f'#define {key} ({value}LL)\n' for key, value in dimensions.items())
                defines += f'#define DY ({logical_pitch[0]!r})\n#define DX ({logical_pitch[1]!r})\n'
                defines += f'#define BANDLIMIT ({int(bandlimit == "rectangular")})\n'
                self._module = cp.RawModule(code=defines + CUDA_HELPERS + STREAMED_KERNELS)
                names = ('pad_row_tile', 'scatter_row_tile', 'gather_row_tile', 'crop_row_tile',
                         'gather_split_columns', 'scatter_split_columns', 'multiply_transfer',
                         'pad_row_tile_transposed', 'transpose_final', 'multiply_packed')
                self._kernels = {name: self._module.get_function(name) for name in names}
            temporary = row_info['temporary_bytes'] + column_info['temporary_bytes']
            workspace = self._extra.nbytes + self._scratch.nbytes + temporary
            compact_bytes = math.prod(self.shape)*8
            self.info = dict(shape=self.shape, padded_shape=self.padded_shape, logical_shape=self.logical_shape,
                             axis_order=axis_order, mode=transfer_mode, implementation='streamed',
                             tile_columns=b, tile_rows=r, column_tiles=math.ceil(nx/b), row_tiles=math.ceil(h/r),
                             extra_compact_bytes=self._extra.nbytes, shared_tile_bytes=self._scratch.nbytes,
                             temporary_bytes=temporary, workspace_bytes=workspace,
                             caller_input_bytes=compact_bytes, caller_output_bytes=compact_bytes,
                             total_operator_bytes=workspace+2*compact_bytes, row_fft=row_info, column_fft=column_info,
                             library_sha256=library_hash)
        except Exception:
            self.close()
            raise

    def _create_fft(self, length, batch, ordered=False):
        handle = self._native.streamed_fft_create_ex(length, batch, self.stream.ptr, int(ordered))
        if not handle:
            raise RuntimeError(self._native.streamed_fft_last_error().decode())
        try:
            info = json.loads(self._native.streamed_fft_info(handle))
            if info['uploads'] not in (1, 2, 3):
                raise RuntimeError('Unsupported FFT upload count')
            return handle, info
        except Exception:
            self._native.streamed_fft_destroy(handle)
            raise

    def _fft(self, handle, array, inverse=False):
        result = self._native.streamed_fft_execute(handle, array.data.ptr, int(inverse))
        if result:
            raise RuntimeError(f'Streamed FFT execution failed: VkFFT error {result}')

    def _validate_array(self, array, name):
        import cupy as cp
        if (not isinstance(array, cp.ndarray) or array.shape != self.shape or
                array.dtype != np.complex64 or not array.flags.c_contiguous):
            raise ValueError(f'{name} must be a contiguous CuPy complex64 array of shape {self.shape}')
        if array.device.id != self.device:
            raise ValueError(f'{name} belongs to a different CUDA device')

    def execute(self, source, dest, *, z=0., wavelength=532e-9, transfer=None, scale=1/32767, transpose=False):
        """Enqueue the exact operator with caller-owned, non-overlapping output."""
        import cupy as cp
        with self._lock:
            if self._closed:
                raise RuntimeError('Streamed ASM plan is closed')
            if cp.cuda.Device().id != self.device:
                raise RuntimeError('Current CUDA device differs from the plan device')
            self._validate_array(source, 'source')
            self._validate_array(dest, 'dest')
            if source.data.ptr < dest.data.ptr+dest.nbytes and dest.data.ptr < source.data.ptr+source.nbytes:
                raise ValueError('Input and output must not overlap')
            for array in (source, dest):
                for workspace in (self._extra, self._scratch):
                    if (array.data.ptr < workspace.data.ptr+workspace.nbytes and
                            workspace.data.ptr < array.data.ptr+array.nbytes):
                        raise ValueError('Caller arrays must not overlap private workspace')
            if (not isinstance(z, Real) or not isinstance(wavelength, Real) or
                    not math.isfinite(z) or not math.isfinite(wavelength) or wavelength <= 0):
                raise ValueError('Require finite z and finite positive wavelength')
            if self.transfer_mode == 'packed':
                if (not isinstance(transfer, cp.ndarray) or transfer.dtype != np.uint32 or
                        transfer.shape != self.padded_shape or not transfer.flags.c_contiguous or
                        transfer.device.id != self.device):
                    raise ValueError('transfer must be a contiguous uint32 padded plane on the plan device')
                for array in (dest, self._extra, self._scratch):
                    if (transfer.data.ptr < array.data.ptr+array.nbytes and
                            array.data.ptr < transfer.data.ptr+transfer.nbytes):
                        raise ValueError('transfer must not overlap output or workspace')
                if not isinstance(transpose, (bool, np.bool_)):
                    raise ValueError('transpose must be a boolean')
                if not isinstance(scale, Real) or not math.isfinite(scale):
                    raise ValueError('scale must be finite')
            h, w = self.logical_shape
            r, b, nx, ny = self.tile_rows, self.tile_columns, self._nx, self._ny
            kernels = self._kernels
            with self.stream:
                for offset in range(0, h, r):
                    index = np.int64(offset)
                    if self.axis_order == 'row':
                        kernels['pad_row_tile'](((r*nx+255)//256,), (256,), (source, self._row_view, index))
                    else:
                        kernels['pad_row_tile_transposed']((((nx+31)//32)*((r+31)//32),), (32, 8),
                                                          (source, self._row_view, index))
                    self._fft(self._row_handle, self._row_view)
                    kernels['scatter_row_tile'](((r*nx+255)//256,), (256,),
                                                (self._row_view, dest, self._extra, index))
                for offset in range(0, nx, b):
                    index = np.int64(offset)
                    kernels['gather_split_columns']((((b+31)//32)*((ny+31)//32),), (32, 8),
                                                    (dest, self._extra, self._column_view, index))
                    self._fft(self._column_handle, self._column_view)
                    if self.transfer_mode == 'packed':
                        kernels['multiply_packed']((((b+31)//32)*((ny+31)//32),), (32, 8),
                                                   (self._column_view, transfer, index, np.float32(scale), np.int32(transpose)))
                    else:
                        kernels['multiply_transfer'](((b*ny+255)//256,), (256,),
                                                     (self._column_view, index, np.float64(z), np.float64(wavelength)))
                    self._fft(self._column_handle, self._column_view, inverse=True)
                    kernels['scatter_split_columns']((((h+31)//32)*((b+31)//32),), (32, 8),
                                                     (self._column_view, dest, self._extra, index))
                target = dest if self.axis_order == 'row' else self._extra
                for offset in range(0, h, r):
                    index = np.int64(offset)
                    kernels['gather_row_tile'](((r*nx+255)//256,), (256,),
                                              (dest, self._extra, self._row_view, index))
                    self._fft(self._row_handle, self._row_view, inverse=True)
                    kernels['crop_row_tile'](((r*w+255)//256,), (256,), (self._row_view, target, index))
                if self.axis_order == 'column':
                    kernels['transpose_final']((((w+31)//32)*((h+31)//32),), (32, 8), (self._extra, dest))
        return dest

    def close(self):
        """Wait for the bound stream and release owned device resources."""
        if getattr(self, '_closed', True):
            return
        import cupy as cp
        with self._lock, cp.cuda.Device(self.device):
            if self._closed:
                return
            self.stream.synchronize()
            for name in ('_row_handle', '_column_handle'):
                handle = getattr(self, name, None)
                if handle:
                    self._native.streamed_fft_destroy(handle)
                    setattr(self, name, None)
            self._row_view = self._column_view = self._extra = self._scratch = None
            self._kernels = self._module = None
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Streamed ASM plan is closed')
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
