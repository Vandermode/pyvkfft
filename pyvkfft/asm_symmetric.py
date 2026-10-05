"""Shared storage for exact even-frequency analytic ASM transfer tables.

Both static complex64 H and cached FP64 sqrt(q) use eight bytes per entry.
This format applies to analytic ASM, not arbitrary user-supplied H. Compile
CUDA_SYMMETRIC after pyvkfft.asm.CUDA_HELPERS for the static coefficient helper.
"""
import ctypes
import operator


SYMMETRIC_ABI_VERSION = 1
SYMMETRIC_LAYOUT_VERSION = 1


def validate_library(library):
    """Reject ordinary, full-phase, or incompatible compressed backends."""
    for name, expected in (('asm_symmetric_abi_version', SYMMETRIC_ABI_VERSION),
                           ('asm_symmetric_layout_version', SYMMETRIC_LAYOUT_VERSION)):
        try:
            capability = getattr(library, name)
        except AttributeError as exc:
            raise RuntimeError('The selected library is not a symmetric ASM backend') from exc
        capability.argtypes = []
        capability.restype = ctypes.c_uint32
        if capability() != expected:
            raise RuntimeError(f'Incompatible symmetric ASM capability: {name}')


def table_shape(compact_shape):
    """Return positive-frequency rows and the physical padded row stride."""
    if len(compact_shape) != 2 or any(isinstance(n, bool) for n in compact_shape):
        raise ValueError('compact_shape must contain two positive integers')
    height, width = map(operator.index, compact_shape)
    if height <= 0 or width <= 0:
        raise ValueError('compact_shape must contain two positive integers')
    columns = width + 1
    stride = columns if columns < 1024 else ((columns + 31)//32)*32
    return height + 1, stride


CUDA_SYMMETRIC = r'''
__device__ __forceinline__ long long asm_symmetric_stride(long long nx) {
    long long columns=nx/2+1;
    return columns<1024 ? columns : ((columns+31)/32)*32;
}
__device__ __forceinline__ long long asm_symmetric_unfold(long long p,
                                                         long long a,long long b) {
    long long half=a/2, bulk=half*b;
    if(!half) return p;
    long long digit, row;
    if(p<bulk) { digit=p/half; row=p%half; }
    else { digit=p-bulk; row=half; }
    return row*b+digit;
}
extern "C" __global__ void prepare_symmetric_transfer(float2* table,
    long long ny,long long nx,long long ay,long long by,long long ax,long long bx,
    double dy,double dx,double z,double wavelength,int bandlimit) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long rows=ny/2+1, columns=nx/2+1, stride=asm_symmetric_stride(nx);
    if(i>=rows*stride) return;
    long long py=i/stride, px=i%stride;
    if(px>=columns) { table[i]=make_float2(0.f,0.f); return; }
    long long y=asm_symmetric_unfold(py,ay,by), x=asm_symmetric_unfold(px,ax,bx);
    table[i]=asm_coefficient(y,x,ny,nx,dy,dx,z,wavelength,bandlimit);
}
extern "C" __global__ void prepare_symmetric_phase(double* table,
    long long ny,long long nx,long long ay,long long by,long long ax,long long bx,
    double dy,double dx,double wavelength) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long rows=ny/2+1, columns=nx/2+1, stride=asm_symmetric_stride(nx);
    if(i>=rows*stride) return;
    long long py=i/stride, px=i%stride;
    if(px>=columns) { table[i]=-1.; return; }
    long long y=asm_symmetric_unfold(py,ay,by), x=asm_symmetric_unfold(px,ax,bx);
    double fy=y/(ny*dy), fx=x/(nx*dx);
    double invlambda=1./wavelength, q=invlambda*invlambda-(fx*fx+fy*fy);
    table[i]=q<0. ? -1. : sqrt(q);
}
'''
