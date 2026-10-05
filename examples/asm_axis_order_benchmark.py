"""Compare row-first/column-first separable ASM including compact transposes.

This isolates transform order. Both paths use the same cuFFT implementation;
column-first input/output transpose kernels are inside the timed region.
"""
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
    parser.add_argument('--seconds-per-block',type=float,default=2)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--cufft-library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-strided',action='store_true')
    args=parser.parse_args()
    if min(args.shape)<=0 or args.rounds<1 or args.seconds_per_block<=0:
        parser.error('Positive dimensions, rounds and duration required')
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
    from asm_cufft import PrunedCuFFT
    from pyvkfft.asm import CUDA_KERNELS
    shape=tuple(args.shape);padded=tuple(2*n for n in shape)
    free,total=cp.cuda.runtime.memGetInfo()
    estimated=int(math.prod(padded)*8*((10 if args.include_strided else 6.75) if args.mode=='static' else (7.5 if args.include_strided else 4.75)))
    if estimated>.8*free: raise MemoryError('Axis-order study exceeds conservative 80% memory budget')
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        random=cp.random.RandomState(16)
        source=cp.empty(shape,cp.complex64)
        source.real=random.standard_normal(shape,dtype=cp.float32)
        source.imag=random.standard_normal(shape,dtype=cp.float32)
        dest=cp.empty_like(source);reference=cp.empty_like(source)
        source_t=cp.empty(shape[::-1],cp.complex64);dest_t=cp.empty_like(source_t)
        transfer=cp.empty(padded if args.mode=='static' else (1,),cp.complex64)
        module=cp.RawModule(code=CUDA_KERNELS)
        gen=module.get_function('generate_transfer')
        if args.mode=='static':
            gen(((transfer.size+255)//256,),(256,),(transfer,*map(np.int64,(*padded,padded[0],1,padded[1],1)),
                np.float64(6.4e-6),np.float64(6.4e-6),np.float64(.01),np.float64(532e-9),np.int32(0)))
        row=PrunedCuFFT(shape,(6.4e-6,6.4e-6),transfer,args.mode!='static')
        column=PrunedCuFFT(shape[::-1],(6.4e-6,6.4e-6),transfer.T,args.mode!='static')
        if args.include_strided:
            from asm_cufft_strided import StridedCuFFT
            strided=StridedCuFFT(shape,(6.4e-6,6.4e-6),transfer,args.mode!='static')
    stream.synchronize()
    transpose=cp.RawKernel(r'''
extern "C" __global__ void transpose_compact(const float2* src,float2* dst,long long h,long long w) {
 __shared__ float2 tile[32][33];
 long long x=(long long)blockIdx.x*32+threadIdx.x,y=(long long)blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)tile[threadIdx.y+j][threadIdx.x]=(x<w && y+j<h)?src[(y+j)*w+x]:make_float2(0,0);
 __syncthreads();x=(long long)blockIdx.y*32+threadIdx.x;y=(long long)blockIdx.x*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(x<h && y+j<w)dst[(y+j)*h+x]=tile[threadIdx.x][threadIdx.y+j];
}''','transpose_compact')
    def params(i):
        return ((.01,.1,1.)[i%3],(450e-9,532e-9,633e-9)[i%3]) if args.mode!='static' else (.01,532e-9)
    def execute(label,i):
        with stream:
            if label=='row_first': row.execute(source,dest,*params(i))
            elif label=='strided_columns':strided.execute(source,dest,*params(i))
            else:
                transpose(((shape[1]+31)//32,(shape[0]+31)//32),(32,8),(source,source_t,*map(np.int64,shape)))
                column.execute(source_t,dest_t,*params(i))
                transpose(((shape[0]+31)//32,(shape[1]+31)//32),(32,8),(dest_t,dest,*map(np.int64,shape[::-1])))
    labels=['row_first','column_first']+(['strided_columns'] if args.include_strided else [])
    errors=[]
    for i in range(3 if args.mode!='static' else 1):
        execute('row_first',i)
        with stream: cp.copyto(reference,dest)
        execute('column_first',i)
        with stream:
            num,den=0.,0.
            for start in range(0,dest.size,2**20):
                a=dest.ravel()[start:start+2**20];b=reference.ravel()[start:start+2**20]
                num+=float(cp.sum(cp.abs(a-b)**2,dtype=cp.float64))
                den+=float(cp.sum(cp.abs(b)**2,dtype=cp.float64))
        error=math.sqrt(num/den) if den else math.sqrt(num)
        if not math.isfinite(error) or error>2e-5: raise AssertionError(error)
        errors.append(error)
    if args.include_strided:
        for i in range(3 if args.mode!='static' else 1):
            execute('row_first',i)
            with stream:cp.copyto(reference,dest)
            execute('strided_columns',i)
            with stream:
                num=den=0.
                for start in range(0,dest.size,2**20):
                    a=dest.ravel()[start:start+2**20];b=reference.ravel()[start:start+2**20]
                    num+=float(cp.sum(cp.abs(a-b)**2,dtype=cp.float64));den+=float(cp.sum(cp.abs(b)**2,dtype=cp.float64))
            error=math.sqrt(num/den) if den else math.sqrt(num)
            if not math.isfinite(error) or error>2e-5:raise AssertionError(('strided_columns',error))
            errors.append(error)
    result={'shape':shape,'mode':args.mode,'gpu_uuid':uuid,'relative_l2':errors,
            'protocol':'Full compact input/output; column-first includes both compact transpose kernels; all dynamic H included',
            'row_first_plan':row.info,'column_first_plan':column.info,'blocks':[]}
    if args.include_strided:result['strided_plan']=strided.info
    for label in labels:
        for i in range(3): execute(label,i)
    stream.synchronize()
    for round_index in range(args.rounds):
        order=list(labels)
        if round_index%2: order.reverse()
        for label in order:
            a,b=cp.cuda.Event(),cp.cuda.Event()
            a.record(stream)
            for i in range(5): execute(label,i)
            b.record(stream);b.synchronize()
            estimate=cp.cuda.get_elapsed_time(a,b)/5
            count=max(10,math.ceil(1000*args.seconds_per_block/estimate))
            events=[cp.cuda.Event() for _ in range(count+1)]
            events[0].record(stream)
            for i in range(count):
                execute(label,i);events[i+1].record(stream)
            events[-1].synchronize()
            samples=[cp.cuda.get_elapsed_time(events[i],events[i+1]) for i in range(count)]
            result['blocks'].append({'label':label,'round':round_index,'samples_ms':samples})
            print(label,float(np.median(samples)),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
