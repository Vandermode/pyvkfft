"""Verify a large exact ASM identity within an explicit device memory budget.

Run in a fresh process with a reserved CUDA_VISIBLE_DEVICES GPU and a built
PYVKFFT_STREAMED_LIBRARY. Default 32768-square caller I/O and workspace occupy
about 24 GiB, below the default 32 GiB cap. No full reference grid is allocated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import cupy as cp
import numpy as np
from pyvkfft.asm_streamed import StreamedASMPlan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape',type=int,nargs=2,default=(32768,32768))
    parser.add_argument('--tile',type=int,default=32)
    parser.add_argument('--cap-gib',type=float,default=32)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    h,w=args.shape
    if min(h,w,args.tile,args.cap_gib)<=0:
        parser.error('Shape, tile and cap must be positive')
    cp.cuda.Device().synchronize()
    free,total=cp.cuda.runtime.memGetInfo()
    cap=min(args.cap_gib*2**30,.8*free)
    # This probe fixes both tile extents; allow 256 MiB for modules/planning.
    predicted=3*h*w*8+max(args.tile*2*h,args.tile*2*w)*8
    if predicted+256*2**20>cap:
        raise MemoryError(f'Preflight {predicted+256*2**20} bytes exceeds {cap}')
    rows=[]
    def checkpoint(label):
        cp.cuda.Device().synchronize()
        consumed=free-cp.cuda.runtime.memGetInfo()[0]
        rows.append(dict(label=label,driver_bytes=consumed,cupy_live_bytes=cp.get_default_memory_pool().used_bytes(),cupy_reserved_bytes=cp.get_default_memory_pool().total_bytes()))
        if consumed>cap:raise MemoryError(f'{label}: {consumed} exceeds {cap}')
    x=cp.empty((h,w),cp.complex64)
    out=cp.empty_like(x)
    fill=cp.RawKernel(r'''extern "C" __global__ void fill(float2* x,long long n) {
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n)x[i]=make_float2(((i*17+13)%1021-510)/511.0f,((i*29+7)%1019-509)/510.0f);
    }''','fill')
    fill(((h*w+255)//256,),(256,),(x,np.int64(h*w)))
    checkpoint('compact_io_nonconstant_input')
    def source_hash():
        digest=hashlib.sha256()
        for y in range(0,h,64):digest.update(x[y:y+64].get().tobytes())
        return digest.hexdigest()
    before=source_hash()
    with StreamedASMPlan((h,w),(6.4e-6,6.4e-6),tile_columns=args.tile,tile_rows=args.tile) as plan:
        checkpoint('plan_created')
        used=cp.get_default_memory_pool().used_bytes()
        reserved=cp.get_default_memory_pool().total_bytes()
        timings=[]
        for i in range(3):
            start,end=cp.cuda.Event(),cp.cuda.Event()
            start.record()
            with patch.object(cp,'empty',side_effect=AssertionError('execute allocation')):
                plan.execute(x,out,z=0.,wavelength=532e-9)
            end.record();end.synchronize()
            timings.append(cp.cuda.get_elapsed_time(start,end))
            checkpoint('execute_'+str(i))
            assert used==cp.get_default_memory_pool().used_bytes()
            assert reserved==cp.get_default_memory_pool().total_bytes()
        numerator=denominator=maximum_error=maximum_source=0.
        for y in range(0,h,64):
            delta=cp.abs(out[y:y+64]-x[y:y+64]);mag=cp.abs(x[y:y+64])
            numerator+=float(cp.sum(delta**2,dtype=cp.float64).get())
            denominator+=float(cp.sum(mag**2,dtype=cp.float64).get())
            maximum_error=max(maximum_error,float(delta.max().get()))
            maximum_source=max(maximum_source,float(mag.max().get()))
        error=(numerator/denominator)**.5
        linf=maximum_error/maximum_source
        assert error<2e-5 and linf<1e-4,(error,linf)
        assert source_hash()==before,'Input changed'
        checkpoint('chunked_reference_verification')
        root=Path(__file__).resolve().parents[1]
        library=Path(os.environ['PYVKFFT_STREAMED_LIBRARY'])
        result=dict(shape=[h,w],cap_bytes=cap,initial_free_bytes=free,total_bytes=total,predicted_operator_bytes=predicted,checkpoints=rows,peak_driver_bytes=max(r['driver_bytes'] for r in rows),info=plan.info,identity_relative_l2=error,identity_max_error_over_source_max=linf,input_sha256=before,input_preserved=True,no_execute_allocation=True,execute_ms=timings,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),root/'pyvkfft/asm_streamed.py',root/'src/vkfft_streamed.cpp']},gpu_metadata=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),note='Fresh process, nonconstant exact complex input, z=0 identity, three full executions, chunked reference and source hash; driver checkpoints include compact caller I/O.')
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
