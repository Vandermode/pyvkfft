"""Experimental separable cuFFT ASM baseline; all execution storage is prepared."""
import math
import hashlib
import numpy as np


class PrunedCuFFT:
    """Skip zero input row FFTs and unobserved output row inverse FFTs.

    Column data uses a transposed contiguous layout. The two transpose kernels
    also embed/extract the supported rows. Their cost is included in execute.
    """
    def __init__(self, shape, pitch, transfer, dynamic=False, bandlimit=False,
                 specialize=True, crop_callback=False):
        import cupy as cp
        from cupy.cuda import cufft
        from pyvkfft.asm import CUDA_HELPERS
        self.cp, self.cufft = cp, cufft
        self.h, self.w = shape
        self.pitch, self.dynamic, self.bandlimit = pitch, dynamic, bandlimit
        self.specialized = specialize
        self.crop_callback = crop_callback
        self.inverse_row_plan = None
        self._last_output_pointer = None
        self.rows = cp.empty((self.h, 2*self.w), cp.complex64)
        self.columns = cp.empty((2*self.w, 2*self.h), cp.complex64)
        # Both FFTs use contiguous batches. Transpose traffic is explicit.
        self.row_plan = cufft.Plan1d(2*self.w, cufft.CUFFT_C2C, self.h)
        self.col_plan = cufft.Plan1d(2*self.h, cufft.CUFFT_C2C, 2*self.w)
        self.transfer = transfer if dynamic else cp.ascontiguousarray(transfer.T)
        self.module = cp.RawModule(code=CUDA_HELPERS + r'''
extern "C" __global__ void pad_rows(const float2* src, float2* dst, long long h, long long w) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=h*2*w) return;
 long long x=i%(2*w), y=i/(2*w);
 dst[i]=(x>=w/2 && x<w/2+w) ? src[y*w+x-w/2] : make_float2(0,0);
}
extern "C" __global__ void embed_columns(const float2* src, float2* dst, long long h, long long w) {
 __shared__ float2 tile[32][33];
 long long x=(long long)blockIdx.x*32+threadIdx.x;
 long long y=(long long)blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8) {
  long long yy=y+j;
  tile[threadIdx.y+j][threadIdx.x]=(x<2*w && yy>=h/2 && yy<h/2+h)
   ? src[(yy-h/2)*2*w+x] : make_float2(0,0);
 }
 __syncthreads();
 x=(long long)blockIdx.y*32+threadIdx.x;
 y=(long long)blockIdx.x*32+threadIdx.y;
 for(int j=0;j<32;j+=8) if(x<2*h && y+j<2*w)
  dst[(y+j)*2*h+x]=tile[threadIdx.x][threadIdx.y+j];
}
extern "C" __global__ void extract_rows(const float2* src, float2* dst, long long h, long long w) {
 __shared__ float2 tile[32][33];
 long long x=(long long)blockIdx.x*32+threadIdx.x;
 long long y=(long long)blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)
  tile[threadIdx.y+j][threadIdx.x]=(x<h && y+j<2*w)
   ? src[(y+j)*2*h+x+h/2] : make_float2(0,0);
 __syncthreads();
 x=(long long)blockIdx.y*32+threadIdx.x;
 y=(long long)blockIdx.x*32+threadIdx.y;
 for(int j=0;j<32;j+=8) if(x<2*w && y+j<h)
  dst[(y+j)*2*w+x]=tile[threadIdx.x][threadIdx.y+j];
}
extern "C" __global__ void crop_rows(const float2* src, float2* dst, long long h, long long w) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<h*w) dst[i]=src[(i/w)*2*w+i%w+w/2];
}
extern "C" __global__ void set_output_pointer(unsigned long long* dest, unsigned long long address) {
 dest[0]=address;
}
extern "C" __global__ void multiply_columns(float2* data, const float2* transfer,
 long long ny, long long nx, double dy, double dx, double z, double wavelength,
 int dynamic, int bandlimit) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=ny*nx) return;
 long long fy=i%ny, fx=i/ny;
 float2 a=data[i];
 float2 b=dynamic ? asm_coefficient(fy,fx,ny,nx,dy,dx,z,wavelength,bandlimit) : transfer[i];
 float inv=1.f/(ny*nx);
 data[i]=make_float2((a.x*b.x-a.y*b.y)*inv,(a.x*b.y+a.y*b.x)*inv);
}
''')
        self.kernels={n:self.module.get_function(n) for n in
                      ('pad_rows','embed_columns','extract_rows','crop_rows','multiply_columns','set_output_pointer')}
        self.multiply_source_hash = None
        if specialize:
            # Shape, pitch, mode and mask are plan constants for both vendors.
            # Leave distance and wavelength as runtime launch arguments.
            ny, nx = 2*self.h, 2*self.w
            coefficient = (
                f'asm_coefficient(i%{ny}ll,i/{ny}ll,{ny}ll,{nx}ll,'
                f'{pitch[0]:.17g},{pitch[1]:.17g},z,wavelength,{int(bandlimit)})'
                if dynamic else 'transfer[i]')
            source = CUDA_HELPERS.replace('__device__ float2 asm_coefficient',
                                           '__device__ __forceinline__ float2 asm_coefficient') + f'''
extern "C" __global__ void multiply_specialized(float2* data,const float2* transfer,
 long long unused_ny,long long unused_nx,double unused_dy,double unused_dx,
 double z,double wavelength,int unused_dynamic,int unused_bandlimit) {{
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>={ny*nx}ll) return;
 float2 a=data[i],b={coefficient};
 const float inv=1.f/{ny*nx}.f;
 data[i]=make_float2((a.x*b.x-a.y*b.y)*inv,(a.x*b.y+a.y*b.x)*inv);
}}
'''
            self.specialized_module = cp.RawModule(code=source)
            self.kernels['multiply_columns'] = self.specialized_module.get_function('multiply_specialized')
            self.multiply_source_hash = hashlib.sha256(source.encode()).hexdigest()
        if crop_callback:
            self.output_pointer = cp.empty(1, cp.uint64)
            callback = f'''
__device__ void asm_pruned_crop_pointer(void* data,unsigned long long i,float2 value,void* info,void* shared) {{
 unsigned long long x=i%{2*self.w}ull,y=i/{2*self.w}ull;
 if(x>={self.w//2}ull && x<{self.w//2+self.w}ull)
  ((float2*)(*((unsigned long long*)info)))[y*{self.w}ull+x-{self.w//2}ull]=value;
}}
'''
            with cp.fft.config.set_cufft_callbacks(cb_store=callback,
                    cb_store_name='asm_pruned_crop_pointer',
                    cb_store_data=self.output_pointer.data, cb_ver='jit'):
                # CuPy 14's get_fft_plan 1D branch still expects legacy callback
                # fields; use the JIT protocol already used by its ND branch.
                manager = cp.fft.config.get_current_callback_manager()
                handle = manager.set_callbacks(cufft.CUFFT_C2C)
                self.inverse_row_plan = manager.create_plan(handle,
                    ('Plan1d', (2*self.w, cufft.CUFFT_C2C, self.h)))

    @property
    def info(self):
        return {'schedule':'H forward rows, 2W forward/inverse columns, H inverse rows',
                'workspace_bytes':self.rows.nbytes+self.columns.nbytes,
                'transfer_bytes':0 if self.dynamic else self.transfer.nbytes,
                'transfer_layout':'transposed once during static preparation',
                'transpose':'tiled 32x32; padding and row extraction included',
                'transfer_strategy':'fused evaluation' if self.dynamic else 'prepared transposed static H',
                'shape_pitch_mode_mask_specialized':self.specialized,
                'crop_callback':self.crop_callback,
                'multiply_source_sha256':self.multiply_source_hash,
                'cufft_scratch_bytes':sum(p.work_area.mem.size for p in
                    (self.row_plan,self.col_plan,self.inverse_row_plan) if p is not None)}

    def execute(self, source, dest, z, wavelength):
        h,w=np.int64(self.h),np.int64(self.w)
        def launch(name,n,args): self.kernels[name](((n+255)//256,), (256,), args)
        launch('pad_rows',self.rows.size,(source,self.rows,h,w))
        self.row_plan.fft(self.rows,self.rows,self.cufft.CUFFT_FORWARD)
        self.kernels['embed_columns'](((2*self.w+31)//32,(2*self.h+31)//32),(32,8),
                                      (self.rows,self.columns,h,w))
        self.col_plan.fft(self.columns,self.columns,self.cufft.CUFFT_FORWARD)
        launch('multiply_columns',self.columns.size,(self.columns,self.transfer,2*h,2*w,
               *map(np.float64,self.pitch),np.float64(z),np.float64(wavelength),
               np.int32(self.dynamic),np.int32(self.bandlimit)))
        self.col_plan.fft(self.columns,self.columns,self.cufft.CUFFT_INVERSE)
        self.kernels['extract_rows'](((self.h+31)//32,(2*self.w+31)//32),(32,8),
                                     (self.columns,self.rows,h,w))
        if self.crop_callback:
            if self._last_output_pointer != dest.data.ptr:
                self.kernels['set_output_pointer']((1,), (1,),
                    (self.output_pointer, np.uint64(dest.data.ptr)))
                self._last_output_pointer = dest.data.ptr
            self.inverse_row_plan.fft(self.rows,self.rows,self.cufft.CUFFT_INVERSE)
        else:
            self.row_plan.fft(self.rows,self.rows,self.cufft.CUFFT_INVERSE)
            launch('crop_rows',source.size,(self.rows,dest,h,w))


class CuFFTASMPlan:
    """Prepared benchmark operator with explicit axis order and crop strategy.

    All execution storage and FFT plans are prepared here. Call execute on the
    same CUDA stream used during construction. Input is preserved; output
    pointers may change between calls. Column order includes both compact
    transpose kernels in execute. No tuning runs implicitly.
    """
    def __init__(self, shape, pitch, transfer, dynamic=False, bandlimit=False,
                 axis_order='row', crop_callback=False, specialize=True):
        import cupy as cp
        if axis_order not in ('row', 'column'):
            raise ValueError('axis_order must be row or column')
        self.shape, self.axis_order = tuple(shape), axis_order
        if axis_order == 'column':
            self.input_transposed = cp.empty(self.shape[::-1], cp.complex64)
            self.output_transposed = cp.empty_like(self.input_transposed)
            self.core = PrunedCuFFT(self.shape[::-1], pitch[::-1], transfer.T,
                dynamic, bandlimit, specialize=specialize, crop_callback=crop_callback)
            self.transpose = cp.RawKernel(r'''
extern "C" __global__ void transpose_compact(const float2* src,float2* dst,long long h,long long w) {
 __shared__ float2 tile[32][33];
 long long x=(long long)blockIdx.x*32+threadIdx.x,y=(long long)blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)tile[threadIdx.y+j][threadIdx.x]=(x<w && y+j<h)?src[(y+j)*w+x]:make_float2(0,0);
 __syncthreads();x=(long long)blockIdx.y*32+threadIdx.x;y=(long long)blockIdx.x*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(x<h && y+j<w)dst[(y+j)*h+x]=tile[threadIdx.x][threadIdx.y+j];
}''', 'transpose_compact')
        else:
            self.core = PrunedCuFFT(self.shape, pitch, transfer, dynamic,
                bandlimit, specialize=specialize, crop_callback=crop_callback)

    @property
    def info(self):
        result = dict(self.core.info, shape=self.shape, fft_axis_order=self.axis_order)
        result['compact_transpose_bytes'] = (0 if self.axis_order == 'row' else
            self.input_transposed.nbytes + self.output_transposed.nbytes)
        result['workspace_bytes'] += result['compact_transpose_bytes']
        return result

    def execute(self, source, dest, z, wavelength):
        if self.axis_order == 'row':
            self.core.execute(source, dest, z, wavelength)
        else:
            h, w = self.shape
            self.transpose(((w+31)//32, (h+31)//32), (32,8),
                (source, self.input_transposed, np.int64(h), np.int64(w)))
            self.core.execute(self.input_transposed, self.output_transposed, z, wavelength)
            self.transpose(((h+31)//32, (w+31)//32), (32,8),
                (self.output_transposed, dest, np.int64(w), np.int64(h)))
