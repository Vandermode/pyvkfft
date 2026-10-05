"""Measure distance-only phase-table traffic versus recomputing dynamic H.

This isolates coefficient generation, not complete propagation. An FP64 phase
table is valid only for its exact wavelength and doubles transfer storage.
"""
import argparse
import json
import os
import time
from pathlib import Path
import cupy as cp
import numpy as np
from pyvkfft.asm import ASMPlan, CUDA_KERNELS

CODE = r'''
extern "C" __global__ void phase_prepare(double* phase,long long ny,long long nx,
    long long ay,long long by,long long ax,long long bx,double dy,double dx,double wavelength) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ny*nx) return;
    long long y=i/nx,x=i%nx;
    y=(y%ay)*by+y/ay;x=(x%ax)*bx+x/ax;
    double fy=(y<(ny+1)/2?y:y-ny)/(ny*dy),fx=(x<(nx+1)/2?x:x-nx)/(nx*dx);
    double il=1./wavelength,q=il*il-(fx*fx+fy*fy);
    phase[i]=q<0.?-1.:sqrt(q);
}
extern "C" __global__ void frequency_evaluate(const double* fytable,const double* fxtable,
    float2* out,long long ny,long long nx,long long ay,long long by,long long ax,long long bx,
    double z,double wavelength) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ny*nx)return;
    long long y=i/nx,x=i%nx;
    y=(y%ay)*by+y/ay;x=(x%ax)*bx+x/ax;
    double fx=fxtable[x],fy=fytable[y],il=1./wavelength,q=il*il-(fx*fx+fy*fy);
    if(q<0.){out[i]=make_float2(0.f,0.f);return;}
    double sn,cs;sincos(6.283185307179586476925286766559*z*sqrt(q),&sn,&cs);
    out[i]=make_float2((float)cs,(float)sn);
}
extern "C" __global__ void phase_evaluate(const double* phase,float2* out,long long n,double z) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    double v=phase[i],sn,cs;
    if(v<0.) {out[i]=make_float2(0.f,0.f);return;}
    sincos(6.283185307179586476925286766559*z*v,&sn,&cs);
    out[i]=make_float2((float)cs,(float)sn);
}
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--shape',nargs=2,type=int,default=[8192,32768])
    p.add_argument('--cached-fused-library',type=Path,
                   help='Research-only library from asm_cached_phase_build.py')
    p.add_argument('--pitch',nargs=2,type=float,default=[6.4e-6,6.4e-6])
    p.add_argument('--sustain',action='store_true',help='Four alternating five-second blocks of fused versus cached-fused, varying distance')
    args=p.parse_args()
    ny,nx=args.shape;n=ny*nx
    if min(args.shape)<=0 or ny%2 or nx%2:p.error('Padded shape must contain positive even dimensions')
    if min(args.pitch)<=0:p.error('Pixel pitch must be positive')
    if args.sustain and not args.cached_fused_library:p.error('--sustain requires --cached-fused-library')
    module=cp.RawModule(code=CUDA_KERNELS+CODE)
    phase=cp.empty(args.shape,cp.float64)
    plan=ASMPlan((ny//2,nx//2),args.pitch,tuning_profile={'implementation':'native','prune':True,'groupedBatch':[0,32]})
    fused=ASMPlan(plan.shape,plan.pixel_pitch,mode='dynamic',tuning_profile={'implementation':'native','prune':True,'groupedBatch':[0,32],'transfer':'fused'})
    cached_fused=None
    if args.cached_fused_library:
        original_library=os.environ['PYVKFFT_ASM_LIBRARY']
        try:
            os.environ['PYVKFFT_ASM_LIBRARY']=str(args.cached_fused_library.resolve())
            cached_fused=ASMPlan(plan.shape,plan.pixel_pitch,mode='dynamic',tuning_profile={'implementation':'native','prune':True,'groupedBatch':[0,32],'transfer':'fused'})
        finally:
            os.environ['PYVKFFT_ASM_LIBRARY']=original_library
        if cached_fused._layout != plan._layout:raise RuntimeError('Cached-phase permutation mismatch')
    out=plan._transfer
    dimensions=plan._layout
    iy=cp.arange(ny,dtype=cp.float64);ix=cp.arange(nx,dtype=cp.float64)
    fy=cp.where(iy<ny//2,iy,iy-ny)/(ny*args.pitch[0])
    fx=cp.where(ix<nx//2,ix,ix-nx)/(nx*args.pitch[1])
    physical=tuple(map(np.float64,(*args.pitch,.1,532e-9)))
    geometry=((n+255)//256,), (256,)
    prepare=lambda: module.get_function('phase_prepare')(*geometry,(phase,*dimensions,*map(np.float64,args.pitch),np.float64(532e-9)))
    materialize=lambda:module.get_function('generate_transfer')(*geometry,(out,*dimensions,*physical,np.int32(0)))
    cached=lambda:module.get_function('phase_evaluate')(*geometry,(phase,out,np.int64(n),np.float64(.1)))
    compact=lambda:module.get_function('frequency_evaluate')(*geometry,(fy,fx,out,*dimensions,np.float64(.1),np.float64(532e-9)))
    prepare();materialize();ref=out.copy();cached()
    error=float((cp.linalg.norm(out-ref)/cp.linalg.norm(ref)).get())
    compact()
    frequency_error=float((cp.linalg.norm(out-ref)/cp.linalg.norm(ref)).get())
    if error>2e-5: raise RuntimeError(f'Phase error {error}')
    start,stop=cp.cuda.Event(),cp.cuda.Event()
    results={'shape':args.shape,'pitch':args.pitch,'phase_table_bytes':phase.nbytes,'frequency_table_bytes':fx.nbytes+fy.nbytes,'relative_l2':error,'frequency_relative_l2':frequency_error,'blocks_ms':{}}
    cp.random.seed(701)
    x=(cp.random.random(plan.shape,dtype=cp.float32)+1j*cp.random.random(plan.shape,dtype=cp.float32)).astype(cp.complex64)
    dest=cp.empty_like(x)
    plan._prepared=True
    def propagate(fn):
        fn();plan.execute(x,dest)
    fused_execute=lambda:fused.execute(x,dest,z=.1,wavelength=532e-9)
    fused_execute();propagation_ref=dest.copy();propagate(cached)
    results['propagation_relative_l2']=float((cp.linalg.norm(dest-propagation_ref)/cp.linalg.norm(propagation_ref)).get())
    if results['propagation_relative_l2']>2e-5 or frequency_error>2e-5:raise RuntimeError('Alternative mismatch')
    functions=[('prepare',prepare),('recompute',materialize),('cached',cached),('frequency',compact),
               ('propagation_fused',fused_execute),('propagation_recompute',lambda:propagate(materialize)),
               ('propagation_cached',lambda:propagate(cached)),('propagation_frequency',lambda:propagate(compact))]
    if cached_fused:
        def execute_cached_fused(z=.1):
            result=cached_fused._native.asm_execute(cached_fused._handle,x.data.ptr,dest.data.ptr,
                                                   cached_fused._work.data.ptr,phase.data.ptr,z,532e-9)
            if result:raise RuntimeError(f'Cached-phase native error {result}')
        errors=[]
        for z in (.01,.1,1.,-.1):
            fused.execute(x,dest,z=z,wavelength=532e-9)
            reference=dest.copy()
            cached_fused._work.fill(cp.nan)
            execute_cached_fused(z)
            error=float((cp.linalg.norm(dest-reference)/cp.linalg.norm(reference)).get())
            if not np.isfinite(error) or error>2e-5:raise RuntimeError(f'Cached-phase mismatch {error}')
            errors.append(error)
        results['cached_fused_relative_l2']=errors
        results['cached_fused_info']=cached_fused.info
        functions.append(('propagation_cached_fused',execute_cached_fused))
    if args.sustain:
        results['protocol']='Four alternating five-second blocks; fixed wavelength532nm, distance cycles0.01/0.1/1m; parameter updates included; immutable source'
        labels=['propagation_fused','propagation_cached_fused']
        results['blocks_ms']={label:[] for label in labels}
        for block in range(4):
            for label in labels if block%2==0 else reversed(labels):
                total,count=0.,0
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    start.record()
                    for i in range(21):
                        z=(.01,.1,1.)[i%3]
                        if label=='propagation_fused':fused.execute(x,dest,z=z,wavelength=532e-9)
                        else:execute_cached_fused(z)
                    stop.record();stop.synchronize()
                    total+=cp.cuda.get_elapsed_time(start,stop)
                    count+=21
                results['blocks_ms'][label].append(total/count)
                print(block,label,total/count,flush=True)
    else:
        for name,fn in functions:
            for _ in range(5):fn()
            blocks=[]
            for _ in range(4):
                start.record()
                for _ in range(50):fn()
                stop.record();stop.synchronize()
                blocks.append(cp.cuda.get_elapsed_time(start,stop)/50)
            results['blocks_ms'][name]=blocks
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    plan.close();fused.close()
    if cached_fused:cached_fused.close()


if __name__=='__main__':main()
