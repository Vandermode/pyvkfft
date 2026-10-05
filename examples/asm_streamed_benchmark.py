"""Sustained paired dynamic ASM comparison. Run each replicate in a fresh process.

Set CUDA_VISIBLE_DEVICES to one reserved GPU UUID and both library environment
variables. Example: python examples/asm_streamed_benchmark.py --shape 4096 16384
--tile 256 --output result.json. Defaults: four alternating five-second blocks
per variant. Latency uncertainty must be computed across process replicates.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from unittest.mock import patch
import sys
import time

import cupy as cp
import numpy as np
LIBRARY=Path(os.environ["PYVKFFT_STREAMED_LIBRARY"])
from pyvkfft.asm_streamed import StreamedASMPlan
from pyvkfft.asm import ASMPlan

from issue205_sustained import Monitor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tile',type=int,required=True)
    parser.add_argument('--shape',type=int,nargs=2,default=(4096,16384))
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--axis-order',choices=('row','column'),default='row')
    args=parser.parse_args()
    if min(args.shape)<=0 or args.seconds<5 or args.rounds<4:
        parser.error('Require positive shape, at least five seconds, and four rounds')
    shape=tuple(args.shape);pitch=(6.4e-6,6.4e-6)
    free=cp.cuda.runtime.memGetInfo()[0]
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        cp.random.seed(963)
        x=cp.empty(shape,cp.complex64)
        x.real=cp.random.standard_normal(shape,dtype=cp.float32)
        x.imag=cp.random.standard_normal(shape,dtype=cp.float32)
        out,expected=cp.empty_like(x),cp.empty_like(x)
        plans={'native':ASMPlan(shape,pitch,mode='dynamic',stream=stream,
                               tuning_profile=({'implementation':'native','prune':True,'transfer':'fused','axis_order':'column'} if shape[0]>shape[1] else {'implementation':'native','prune':True,'transfer':'fused','groupedBatch':[0,32]})),
               'streamed':StreamedASMPlan(shape,pitch,tile_columns=args.tile,stream=stream,axis_order=args.axis_order)}
        def run(label,i,dest=out):
            plans[label].execute(x,dest,z=(.01,.1,1.)[i%3],wavelength=(450e-9,532e-9,633e-9)[i%3])
        def source_hash():
            result=hashlib.sha256()
            for y in range(0,shape[0],64):
                result.update(x[y:y+64].get().tobytes())
            return result.hexdigest()
        input_before=source_hash()
        errors=[]
        for i in range(3):
            run('native',i,expected);run('streamed',i)
            stream.synchronize()
            num=den=0.;maximum_error=maximum_expected=0.
            for y in range(0,shape[0],64):
                a,b=out[y:y+64],expected[y:y+64]
                delta=cp.abs(a-b);mag=cp.abs(b)
                num+=float(cp.sum(delta**2,dtype=cp.float64).get());den+=float(cp.sum(mag**2,dtype=cp.float64).get())
                maximum_error=max(maximum_error,float(delta.max().get()))
                maximum_expected=max(maximum_expected,float(mag.max().get()))
            e=float(np.sqrt(num/max(den,1e-30)))
            linf=maximum_error/max(maximum_expected,1e-30)
            assert np.isfinite(e) and e<2e-5 and linf<1e-4,(e,linf)
            errors.append(dict(relative_l2=e,max_abs_over_reference_max=linf))
        del expected,delta,mag,a,b
        stream.synchronize();cp.get_default_memory_pool().free_all_blocks()
        used_before=cp.get_default_memory_pool().used_bytes()
        total_before=cp.get_default_memory_pool().total_bytes()
        with patch.object(cp, 'empty', side_effect=AssertionError('execute allocation')):
            for i in range(3):run('streamed',i)
        stream.synchronize()
        assert used_before==cp.get_default_memory_pool().used_bytes()
        assert total_before==cp.get_default_memory_pool().total_bytes()
        paired_driver_bytes=free-cp.cuda.runtime.memGetInfo()[0]
        assert paired_driver_bytes<.8*free
        blocks=[]
        monitor=Monitor(os.environ['CUDA_VISIBLE_DEVICES']);monitor.thread.start()
        labels=['native','streamed']
        for repeat in range(args.rounds):
            for label in labels if repeat%2==0 else labels[::-1]:
                start,end=cp.cuda.Event(),cp.cuda.Event();start.record(stream)
                for i in range(4):run(label,i)
                end.record(stream);end.synchronize()
                estimate=cp.cuda.get_elapsed_time(start,end)/4
                count=max(12,math.ceil(args.seconds*1100/estimate))
                events=[cp.cuda.Event() for _ in range(count+1)]
                t0=time.monotonic();events[0].record(stream)
                for i in range(count):run(label,i);events[i+1].record(stream)
                events[-1].synchronize();t1=time.monotonic()
                samples=[cp.cuda.get_elapsed_time(events[i],events[i+1]) for i in range(count)]
                assert sum(samples)>=args.seconds*1000,(label,sum(samples))
                blocks.append(dict(label=label,round=repeat,samples_ms=samples,start_monotonic=t0,end_monotonic=t1,median_ms=statistics.median(samples),gpu_duration_ms=sum(samples)))
                print(args.tile,repeat,label,statistics.median(samples),flush=True)
        monitor.close()
        input_after=source_hash()
        assert input_before==input_after,'Source changed'
        for block in blocks:
            rows=[r for r in monitor.rows if block['start_monotonic']+.5<=r['monotonic']<=block['end_monotonic']-.5]
            block['trimmed_telemetry']=dict(samples=len(rows),min_gpu_percent=min((r['gpu_percent'] for r in rows),default=0),mean_gpu_percent=statistics.mean(r['gpu_percent'] for r in rows) if rows else 0)
        root=Path(__file__).resolve().parents[1]
        result=dict(shape=shape,tile=args.tile,errors=errors,info={k:p.info for k,p in plans.items()},
                    initial_free_bytes=free,paired_driver_bytes=paired_driver_bytes,blocks=blocks,
                    telemetry=monitor.rows,telemetry_errors=monitor.errors,
                    median_ms={label:statistics.median(statistics.median(b['samples_ms']) for b in blocks if b['label']==label) for label in labels},
                    library=str(LIBRARY),library_sha256=hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
                    native_library_sha256=hashlib.sha256(Path(os.environ['PYVKFFT_ASM_LIBRARY']).read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),root/'pyvkfft/asm_streamed.py',root/'src/vkfft_streamed.cpp',root/'pyvkfft/asm.py']},
                    input_sha256=input_before,input_preserved=True,no_execute_allocation=True,
                    gpu_metadata=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
                    protocol='Paired alternating complete operators, independent immutable input, both z/wavelength vary, minimum5second blocks, four rounds, trim telemetry .5 seconds at both edges; repeat in three fresh processes')
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(result['median_ms'],flush=True)
        for p in plans.values():p.close()


if __name__=='__main__':main()
