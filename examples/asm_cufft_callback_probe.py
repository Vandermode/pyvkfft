"""Bounded cuFFT separable 1D callback study with equivalent full ASM output."""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    devices=parser.add_mutually_exclusive_group(required=True)
    devices.add_argument('--gpu',type=int,choices=(1,2,3))
    devices.add_argument('--slurm',action='store_true')
    parser.add_argument('--shape',type=int,nargs=2,required=True)
    parser.add_argument('--mode',choices=('static','dynamic_both'),default='static')
    parser.add_argument('--seconds-per-block',type=float,default=1)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--cufft-library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variants',nargs='+',choices=('explicit','pad','crop','multiply','all_supported','native'))
    parser.add_argument('--vkfft-options',type=json.loads,default={})
    args=parser.parse_args()
    if args.slurm != bool(os.environ.get('SLURM_JOB_ID')):
        parser.error('Use --slurm inside Slurm and --gpu outside it')
    if args.gpu is not None:
        uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
    ctypes.CDLL(str(args.cufft_library.resolve()),mode=ctypes.RTLD_GLOBAL)
    import cupy as cp
    import numpy as np
    bus=cp.cuda.runtime.deviceGetPCIBusId(0)
    if isinstance(bus,bytes):bus=bus.decode()
    uuid=subprocess.check_output(['nvidia-smi','-i',bus,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    from cupy.cuda import cufft
    from cupyx.scipy.fft import get_fft_plan
    from asm_cufft import PrunedCuFFT
    from pyvkfft.asm import ASMPlan,CUDA_KERNELS,CUDA_HELPERS
    from issue205_sustained import Monitor
    h,w=args.shape;ny,nx=2*h,2*w;n=ny*nx;dynamic=args.mode!='static'
    free,total=cp.cuda.runtime.memGetInfo()
    estimated=n*8*(8+((1 if dynamic else 2) if args.variants and 'native' in args.variants else 0))
    if estimated>.8*free: raise MemoryError('Callback probe exceeds conservative 80% memory budget')
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        rng=cp.random.RandomState(36)
        source=cp.empty((h,w),cp.complex64)
        source.real=rng.standard_normal((h,w),dtype=cp.float32)
        source.imag=rng.standard_normal((h,w),dtype=cp.float32)
        dest=cp.empty_like(source);reference=cp.empty_like(source)
        transfer=cp.empty((ny,nx),cp.complex64)
        parameters=cp.empty(2,cp.float64)
        module=cp.RawModule(code=CUDA_KERNELS+r'''
extern "C" __global__ void probe_parameters(double* p,double z,double wavelength) {p[0]=z;p[1]=wavelength;}
''')
        gen=module.get_function('generate_transfer');set_params=module.get_function('probe_parameters')
        gen(((n+255)//256,),(256,),(transfer,*map(np.int64,(ny,nx,ny,1,nx,1)),
             np.float64(6.4e-6),np.float64(6.4e-6),np.float64(.01),np.float64(532e-9),np.int32(0)))
        base=PrunedCuFFT((h,w),(6.4e-6,6.4e-6),transfer,dynamic)
        pad_source=f'''__device__ float2 asm_pruned_row_load(void* data,unsigned long long i,void* info,void* shared) {{
 unsigned long long x=i%{nx}ull,y=i/{nx}ull;
 return x>={w//2}ull && x<{w//2+w}ull ? ((float2*)info)[y*{w}ull+x-{w//2}ull] : make_float2(0,0);
}}'''
        crop_source=f'''__device__ void asm_pruned_row_store(void* data,unsigned long long i,float2 value,void* info,void* shared) {{
 unsigned long long x=i%{nx}ull,y=i/{nx}ull;
 if(x>={w//2}ull && x<{w//2+w}ull) ((float2*)info)[y*{w}ull+x-{w//2}ull]=value;
}}'''
        coefficient=(f'asm_coefficient(i%{ny}ull,i/{ny}ull,{ny}ll,{nx}ll,6.4e-6,6.4e-6,((double*)info)[0],((double*)info)[1],0)' if dynamic else '((float2*)info)[i]')
        mul_source=(CUDA_HELPERS if dynamic else '')+f'''\n__device__ float2 asm_pruned_column_load(void* data,unsigned long long i,void* info,void* shared) {{
 float2 a=((float2*)data)[i],b={coefficient};float inv=1.f/{n}.f;
 return make_float2((a.x*b.x-a.y*b.y)*inv,(a.x*b.y+a.y*b.x)*inv);
}}'''
        specs={
            'pad':(base.rows,dict(cb_load=pad_source,cb_load_name='asm_pruned_row_load',cb_load_data=source.data)),
            'crop':(base.rows,dict(cb_store=crop_source,cb_store_name='asm_pruned_row_store',cb_store_data=dest.data)),
            'multiply':(base.columns,dict(cb_load=mul_source,cb_load_name='asm_pruned_column_load',cb_load_data=(parameters if dynamic else base.transfer).data))}
        callback_plans={};unsupported={}
        for label,(array,config) in specs.items():
            try:
                with cp.fft.config.set_cufft_callbacks(**config,cb_ver='jit'):
                    # CuPy14 get_fft_plan's 1D branch still accesses legacy
                    # cb_load_aux_arr fields. Use its JIT manager's native
                    # preallocated-handle protocol (same as the ND branch).
                    manager=cp.fft.config.get_current_callback_manager()
                    handle=manager.set_callbacks(cufft.CUFFT_C2C)
                    callback_plans[label]=manager.create_plan(handle,
                        ('Plan1d',(array.shape[1],cufft.CUFFT_C2C,array.shape[0])))
            except Exception as exc:
                unsupported[label]={'stage':'1d_callback_planning','error':repr(exc)}
                print(label,repr(exc),flush=True)
        native_plan=None
        if args.variants and 'native' in args.variants:
            native_plan=ASMPlan((h,w),(6.4e-6,6.4e-6),mode='dynamic' if dynamic else 'static',stream=stream,
                tuning_profile=dict(args.vkfft_options,implementation='native',prune=True,transfer='fused' if dynamic else 'materialized'))
            if not dynamic:native_plan.prepare_transfer(transfer)
    stream.synchronize()
    def parameters_at(i):
        return ((.01,.1,1.)[i%3],(450e-9,532e-9,633e-9)[i%3]) if dynamic else (.01,532e-9)
    def execute(features,i):
        z,wl=parameters_at(i)
        with stream:
            if features==('native',):
                if dynamic:native_plan.execute(source,dest,z=z,wavelength=wl)
                else:native_plan.execute(source,dest)
                return
            dims=np.int64(h),np.int64(w)
            def launch(name,size,values):base.kernels[name](((size+255)//256,),(256,),values)
            if 'pad' not in features: launch('pad_rows',base.rows.size,(source,base.rows,*dims))
            callback_plans.get('pad',base.row_plan).fft(base.rows,base.rows,cufft.CUFFT_FORWARD) if 'pad' in features else base.row_plan.fft(base.rows,base.rows,cufft.CUFFT_FORWARD)
            base.kernels['embed_columns'](((nx+31)//32,(ny+31)//32),(32,8),(base.rows,base.columns,*dims))
            base.col_plan.fft(base.columns,base.columns,cufft.CUFFT_FORWARD)
            if 'multiply' in features:
                if dynamic:set_params((1,),(1,),(parameters,np.float64(z),np.float64(wl)))
                callback_plans['multiply'].fft(base.columns,base.columns,cufft.CUFFT_INVERSE)
            else:
                launch('multiply_columns',n,(base.columns,base.transfer,np.int64(ny),np.int64(nx),
                    np.float64(6.4e-6),np.float64(6.4e-6),np.float64(z),np.float64(wl),np.int32(dynamic),np.int32(0)))
                base.col_plan.fft(base.columns,base.columns,cufft.CUFFT_INVERSE)
            base.kernels['extract_rows'](((h+31)//32,(nx+31)//32),(32,8),(base.columns,base.rows,*dims))
            (callback_plans['crop'] if 'crop' in features else base.row_plan).fft(base.rows,base.rows,cufft.CUFFT_INVERSE)
            if 'crop' not in features:launch('crop_rows',h*w,(base.rows,dest,*dims))
    variants={'explicit':()}
    for name in callback_plans:variants[name]=(name,)
    if len(callback_plans)>1:variants['all_supported']=tuple(callback_plans)
    if native_plan is not None:variants['native']=('native',)
    if args.variants:variants={name:features for name,features in variants.items() if name in args.variants}
    errors={}
    for i in range(3 if dynamic else 1):
        execute((),i)
        with stream:cp.copyto(reference,dest)
        for label,features in list(variants.items()):
            execute(features,i)
            with stream:
                num=den=0.
                for start in range(0,dest.size,2**20):
                    a=dest.ravel()[start:start+2**20];b=reference.ravel()[start:start+2**20]
                    num+=float(cp.sum(cp.abs(a-b)**2,dtype=cp.float64));den+=float(cp.sum(cp.abs(b)**2,dtype=cp.float64))
            err=math.sqrt(num/den) if den else math.sqrt(num)
            if not math.isfinite(err) or err>2e-5:
                unsupported[label+'_execution']={'stage':'correctness','parameter_index':i,'relative_l2':err}
                del variants[label]
                print('Rejected correctness',label,i,err,flush=True)
                continue
            errors[label]=max(err,errors.get(label,0.))
    result={'shape':args.shape,'mode':args.mode,'gpu_uuid':uuid,'cufft_version':cufft.getVersion(),
            'protocol':'Full compact ASM, separable 1D callbacks independently composed, static preparation excluded, runtime H included',
            'variants':variants,'unsupported':unsupported,'correctness_relative_l2':errors,'blocks':[],
            'base_plan':dict(base.info,cufft_scratch_bytes=sum(p.work_area.mem.size for p in (base.row_plan,base.col_plan))),
            'callback_scratch_bytes':{label:p.work_area.mem.size for label,p in callback_plans.items()},
            'memory_pool_used_bytes':cp.get_default_memory_pool().used_bytes(),
            'native_plan':None if native_plan is None else native_plan.info}
    for features in variants.values():
        for i in range(3):execute(features,i)
    stream.synchronize()
    monitor=Monitor(uuid)
    monitor.thread.start()
    for round_index in range(args.rounds):
        order=list(variants)
        if round_index%2:order.reverse()
        for label in order:
            a,b=cp.cuda.Event(),cp.cuda.Event();a.record(stream)
            for i in range(5):execute(variants[label],i)
            b.record(stream);b.synchronize();estimate=cp.cuda.get_elapsed_time(a,b)/5
            count=max(10,math.ceil(1000*args.seconds_per_block/estimate))
            events=[cp.cuda.Event() for _ in range(count+1)];events[0].record(stream)
            for i in range(count):execute(variants[label],i);events[i+1].record(stream)
            events[-1].synchronize()
            samples=[cp.cuda.get_elapsed_time(events[i],events[i+1]) for i in range(count)]
            result['blocks'].append({'label':label,'round':round_index,'samples_ms':samples})
            print(label,float(np.median(samples)),flush=True)
    monitor.close()
    result['telemetry'],result['telemetry_errors']=monitor.rows,monitor.errors
    result['median_ms']={label:float(np.median([x for b in result['blocks'] if b['label']==label for x in b['samples_ms']])) for label in variants}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    if native_plan is not None:native_plan.close()


if __name__=='__main__':main()
