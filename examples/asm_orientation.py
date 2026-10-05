"""Compare row-first ASM with column-first ASM, including compact transposes."""
from pyvkfft.asm import ASMPlan
import numpy as np
import cupy as cp
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
p.add_argument('--shape', type=int, nargs=2, required=True)
p.add_argument('--mode', choices=('static', 'dynamic'), default='dynamic')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds-per-block', type=float, default=2)
p.add_argument('--rounds', type=int, default=4)
a = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = subprocess.check_output(
    ['nvidia-smi', '-i', str(a.gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
h, w = a.shape
free_initial, _ = cp.cuda.runtime.memGetInfo()
# Static peak is 7.25 padded grids: two work buffers, two prepared H,
# five compact arrays (1.25 grids), natural H and its transpose.
# Another half-grid bounds setup temporaries and memory-pool fragments.
peak_estimated_bytes = int(32*h*w*(7.75 if a.mode == 'static' else 3.75))
if peak_estimated_bytes > .8*free_initial:
    raise MemoryError(
        'Orientation experiment exceeds conservative 80% VRAM preflight')
stream = cp.cuda.Stream(non_blocking=True)
transpose = cp.RawKernel(r'''
extern "C" __global__ void transpose(const float2* src,float2* dst,long long h,long long w) {
 __shared__ float2 t[32][33];
 long long x=(long long)blockIdx.x*32+threadIdx.x,y=(long long)blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8) t[threadIdx.y+j][threadIdx.x]=(x<w && y+j<h)?src[(y+j)*w+x]:make_float2(0,0);
 __syncthreads();
 x=(long long)blockIdx.y*32+threadIdx.x;y=(long long)blockIdx.x*32+threadIdx.y;
 for(int j=0;j<32;j+=8) if(x<h && y+j<w) dst[(y+j)*h+x]=t[threadIdx.x][threadIdx.y+j];
}''', 'transpose')
with stream:
    rng = cp.random.RandomState(43)
    x = cp.empty((h, w), cp.complex64)
    x.real = rng.standard_normal((h, w), dtype=cp.float32)
    x.imag = rng.standard_normal((h, w), dtype=cp.float32)
    out = cp.empty_like(x)
    reference = cp.empty_like(x)
    xt = cp.empty((w, h), cp.complex64)
    yt = cp.empty_like(xt)
    profile = {'implementation': 'native', 'prune': True,
               'transfer': 'fused' if a.mode == 'dynamic' else 'materialized'}
    row = ASMPlan((h, w), (6.4e-6, 6.4e-6), mode=a.mode,
                  stream=stream, tuning_profile=profile)
    col = ASMPlan((w, h), (6.4e-6, 6.4e-6), mode=a.mode,
                  stream=stream, tuning_profile=profile)
    if a.mode == 'static':
        fill_h = cp.RawKernel(r'''
extern "C" __global__ void fill_h(float2* dest,long long n) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n) {double sn,cs;sincos((double)i/123.,&sn,&cs);dest[i]=make_float2((float)cs,(float)sn);}
}''', 'fill_h')
        transfer = cp.empty((2*h, 2*w), cp.complex64)
        fill_h(((transfer.size+255)//256,), (256,),
               (transfer, np.int64(transfer.size)), stream=stream)
        row.prepare_transfer(transfer)
        transfer_t = cp.ascontiguousarray(transfer.T)
        col.prepare_transfer(transfer_t)
        stream.synchronize()
        del transfer, transfer_t
        cp.get_default_memory_pool().free_all_blocks()

    def execute(plan, src, dst):
        if a.mode == 'dynamic':
            plan.execute(src, dst, z=.1, wavelength=532e-9)
        else:
            plan.execute(src, dst)

    def row_first(): execute(row, x, out)

    def column_first():
        transpose(((w+31)//32, (h+31)//32), (32, 8),
                  (x, xt, np.int64(h), np.int64(w)), stream=stream)
        execute(col, xt, yt)
        transpose(((h+31)//32, (w+31)//32), (32, 8),
                  (yt, out, np.int64(w), np.int64(h)), stream=stream)
    execute(row, x, reference)
    column_first()
    stream.synchronize()
    num = den = 0.
    for offset in range(0, x.size, 2**20):
        v = out.ravel()[offset:offset+2**20]
        r = reference.ravel()[offset:offset+2**20]
        num += float(cp.sum(cp.abs(v-r)**2, dtype=cp.float64))
        den += float(cp.sum(cp.abs(r)**2, dtype=cp.float64))
    error = (num/den)**.5
    assert np.isfinite(error) and error <= 2e-5, error
    result = {'shape': a.shape, 'mode': a.mode, 'relative_l2': error, 'protocol': 'column-first includes both tiled compact transposes; prepared static H transpose excluded',
              'blocks_ms': {'row_first': [], 'column_first': []}, 'plans': {'row': row.info, 'column': col.info},
              'peak_estimated_bytes': peak_estimated_bytes,
              'active_pool_bytes': cp.get_default_memory_pool().used_bytes()}
    for round_index in range(a.rounds):
        variants = [('row_first', row_first), ('column_first', column_first)]
        if round_index % 2:
            variants.reverse()
        for label, fn in variants:
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record(stream)
            for _ in range(5):
                fn()
            end.record(stream)
            end.synchronize()
            estimate = cp.cuda.get_elapsed_time(start, end)/5
            count = max(10, int(1000*a.seconds_per_block/estimate))
            start.record(stream)
            for _ in range(count):
                fn()
            end.record(stream)
            end.synchronize()
            result['blocks_ms'][label].append(
                cp.cuda.get_elapsed_time(start, end)/count)
    result['median_ms'] = {k: float(np.median(v))
                           for k, v in result['blocks_ms'].items()}
    print(json.dumps(result['median_ms']), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    row.close()
    col.close()
