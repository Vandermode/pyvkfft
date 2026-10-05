"""Alternative separable cuFFT schedule with strided column transforms."""
import numpy as np


class StridedCuFFT:
    def __init__(self,shape,pitch,transfer,dynamic=False,bandlimit=False):
        import cupy as cp
        from cupy.cuda import cufft
        from pyvkfft.asm import CUDA_KERNELS
        self.cufft=cufft
        self.h,self.w=shape;self.ny,self.nx=2*self.h,2*self.w
        self.dynamic,self.bandlimit=dynamic,bandlimit
        self.pitch,self.transfer=pitch,transfer
        self.work=cp.empty((self.ny,self.nx),cp.complex64)
        self.rows=self.work[self.h//2:self.h//2+self.h]
        self.row_plan=cufft.Plan1d(self.nx,cufft.CUFFT_C2C,self.h)
        self.col_plan=cufft.PlanNd((self.ny,),(self.ny,),self.nx,1,
                                 (self.ny,),self.nx,1,cufft.CUFFT_C2C,self.nx,'C',0,self.ny)
        self.module=cp.RawModule(code=CUDA_KERNELS+r'''
extern "C" __global__ void multiply_strided(float2* data,const float2* transfer,
 long long ny,long long nx,double dy,double dx,double z,double wavelength,int dynamic,int bandlimit) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=ny*nx) return;
 float2 a=data[i], b=dynamic?asm_coefficient(i/nx,i%nx,ny,nx,dy,dx,z,wavelength,bandlimit):transfer[i];
 float inv=1.f/(ny*nx);
 data[i]=make_float2((a.x*b.x-a.y*b.y)*inv,(a.x*b.y+a.y*b.x)*inv);
}
''')
        self.kernels={n:self.module.get_function(n) for n in ('pad_input','crop_output','multiply_strided')}

    @property
    def info(self):
        return {'schedule':'H populated row FFTs; strided full columns; H observed row IFFTs',
                'workspace_bytes':self.work.nbytes,
                'cufft_scratch_bytes':sum(p.work_area.mem.size for p in (self.row_plan,self.col_plan)),
                'shared_transfer_bytes':0 if self.dynamic else self.transfer.nbytes,
                'transfer_strategy':'fused evaluation' if self.dynamic else 'natural-order static H'}

    def execute(self,source,dest,z,wavelength):
        def launch(name,n,values):self.kernels[name](((n+255)//256,),(256,),values)
        dims=np.int64(self.h),np.int64(self.w)
        launch('pad_input',self.work.size,(source,self.work,*dims))
        self.row_plan.fft(self.rows,self.rows,self.cufft.CUFFT_FORWARD)
        self.col_plan.fft(self.work,self.work,self.cufft.CUFFT_FORWARD)
        launch('multiply_strided',self.work.size,(self.work,self.transfer,np.int64(self.ny),np.int64(self.nx),
              *map(np.float64,self.pitch),np.float64(z),np.float64(wavelength),np.int32(self.dynamic),np.int32(self.bandlimit)))
        self.col_plan.fft(self.work,self.work,self.cufft.CUFFT_INVERSE)
        self.row_plan.fft(self.rows,self.rows,self.cufft.CUFFT_INVERSE)
        launch('crop_output',source.size,(self.work,dest,*dims))
