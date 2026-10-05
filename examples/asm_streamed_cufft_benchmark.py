"""Compare packed tiled VkFFT and cuFFT with the same compact storage schedule.

Both preserve input, use two compact spectrum halves, transpose/pad/crop the
same tiles, decode the same packed H, and include all launches in event timing.
cuFFT emits natural frequency order and is normalized at transfer multiplication.
Run fresh processes on an otherwise idle GPU for repeated timing measurements.
"""
import argparse
import hashlib
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import cupy as cp
from cupy.cuda import cufft
import numpy as np
from pyvkfft.asm_streamed import StreamedASMPlan


class CuFFTStreamedPlan(StreamedASMPlan):
    def _create_fft(self, length, batch, ordered=False):
        self._native = SimpleNamespace(streamed_fft_destroy=lambda handle: None)
        plan = cufft.Plan1d(length, cufft.CUFFT_C2C, batch)
        return plan, dict(length=length, batch=batch, uploads=1, ordered=1,
                          axis_split=[length, 1, 1], temporary_bytes=plan.work_area.mem.size)

    def _fft(self, handle, array, inverse=False):
        handle.fft(array, array, cufft.CUFFT_INVERSE if inverse else cufft.CUFFT_FORWARD)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native = SimpleNamespace(streamed_fft_destroy=lambda handle: None)

    def execute(self, source, dest, **kwargs):
        kwargs['scale'] = kwargs.get('scale', 1/32767)/math.prod(self.padded_shape)
        return super().execute(source, dest, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--tile', type=int, default=1024)
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.size < 2 or args.tile < 1 or args.seconds < 1 or args.rounds < 2:
        parser.error('Positive size/tile, seconds >=1 and rounds >=2 required')
    n = args.size
    source = cp.empty((n,n),cp.complex64)
    output = cp.empty_like(source)
    fill = cp.RawKernel(r'''extern "C" __global__ void fill(float2* x, long long n) {
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n) { unsigned int a=(unsigned int)i*747796405u+2891336453u;
            unsigned int b=(a^(a>>16))*277803737u;
            x[i]=make_float2((int)(a&65535u)/32768.f-1.f,(int)(b&65535u)/32768.f-1.f); }
    }''','fill')
    fill(((n*n+255)//256,), (256,), (source,np.int64(n*n)))
    # General non-even payload, generated without dense transfer allocation.
    payload = cp.empty((2*n,2*n),cp.uint32)
    generate = cp.RawKernel(r'''extern "C" __global__ void generate(unsigned int* p,long long n) {
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n){unsigned int a=(unsigned int)i*747796405u+2891336453u;
            p[i]=(a^(a>>16))*277803737u;}
    }''','generate')
    generate(((4*n*n+255)//256,), (256,), (payload,np.int64(4*n*n)))
    plans = {'vkfft':StreamedASMPlan((n,n),(1.,1.),tile_columns=args.tile,transfer_mode='packed'),
             'cufft':CuFFTStreamedPlan((n,n),(1.,1.),tile_columns=args.tile,transfer_mode='packed')}
    # Compare samples distributed over the full output without a second plane.
    indices = cp.asarray(np.random.default_rng(71).integers(0,n*n,65536,dtype=np.int64))
    input_sample = source.ravel()[indices].get()
    references = {}
    for name,p in plans.items():
        p.execute(source,output,transfer=payload)
        references[name] = output.ravel()[indices].get()
    error = float(np.linalg.norm(references['vkfft']-references['cufft'])/np.linalg.norm(references['cufft']))
    assert error < 2e-5, error
    blocks = []
    for repeat in range(args.rounds):
        for name in (('vkfft','cufft') if repeat%2==0 else ('cufft','vkfft')):
            p=plans[name]
            a,b=cp.cuda.Event(),cp.cuda.Event();a.record();p.execute(source,output,transfer=payload);b.record();b.synchronize()
            count=max(3,math.ceil(args.seconds*1100/cp.cuda.get_elapsed_time(a,b)))
            events=[cp.cuda.Event() for _ in range(count+1)];events[0].record()
            start=time.perf_counter()
            for i in range(count):p.execute(source,output,transfer=payload);events[i+1].record()
            events[-1].synchronize()
            samples=[cp.cuda.get_elapsed_time(events[i],events[i+1]) for i in range(count)]
            row=dict(backend=name,round=repeat,samples_ms=samples,median_ms=statistics.median(samples),wall_seconds=time.perf_counter()-start)
            blocks.append(row);print(name,repeat,row['median_ms'],flush=True)
    np.testing.assert_array_equal(source.ravel()[indices].get(), input_sample)
    record=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_sample_preserved=True, size=n,tile=args.tile,sampled_field_relative_l2=error,blocks=blocks,
                median_ms={name:statistics.median(r['median_ms'] for r in blocks if r['backend']==name) for name in plans},
                info={name:p.info for name,p in plans.items()},gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cupy_version=cp.__version__,cufft_version=cufft.getVersion(),
                protocol='Same packed uint32 H, same compact tiled schedule, natural-order cuFFT and no-reorder VkFFT columns, both normalized once per axis or equivalently in multiply. Prepared plans, immutable input, all padding/transposes/decode/crop included. Alternating sustained blocks, sampled numerical comparison; no physical transfer preparation.')
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(record['median_ms'],flush=True)
    for p in plans.values():p.close()


if __name__=='__main__':main()
