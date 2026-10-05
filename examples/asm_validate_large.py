"""Large-grid L2 and scaled max-error checks against an independent cuFFT schedule."""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
p.add_argument('--shape', type=int, nargs=2, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cufft-library', type=Path, required=True)
a = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = subprocess.check_output(
    ['nvidia-smi', '-i', str(a.gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
ctypes.CDLL(str(a.cufft_library), mode=ctypes.RTLD_GLOBAL)
import cupy as cp
import numpy as np
from cupy.cuda import cufft
from pyvkfft.asm import ASMPlan, CUDA_KERNELS

shape = tuple(a.shape)
padded = tuple(2*n for n in shape)
size = math.prod(padded)
free_initial, _ = cp.cuda.runtime.memGetInfo()
if 7*size*8 > .8*free_initial:
    raise MemoryError('Correctness preflight exceeds 80% available VRAM')
stream = cp.cuda.Stream(non_blocking=True)
result = {'shape': shape, 'padded_shape': padded, 'checks': []}
with stream:
    rng = cp.random.RandomState(629)
    x = cp.empty(shape, cp.complex64)
    x.real = rng.standard_normal(shape, dtype=cp.float32)
    x.imag = rng.standard_normal(shape, dtype=cp.float32)
    original = x.copy()
    out, reference = cp.empty_like(x), cp.empty_like(x)
    work = cp.empty(padded, cp.complex64)
    transfer = cp.empty_like(work)
    cuplan = cufft.PlanNd(padded, padded, 1, size, padded, 1, size, cufft.CUFFT_C2C, 1, 'C', 1, padded[-1])
    module = cp.RawModule(code=CUDA_KERNELS+r'''
    extern "C" __global__ void multiply(float2* x,const float2* h,long long n) {
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n) {float2 u=x[i],v=h[i];float scale=1.f/n;
            x[i]=make_float2((u.x*v.x-u.y*v.y)*scale,(u.x*v.y+u.y*v.x)*scale);}
    }
    ''')
    kernels = {name: module.get_function(name) for name in
               ('generate_transfer', 'pad_input', 'crop_output', 'multiply')}
    def launch(name, count, args):
        kernels[name](((count+255)//256,), (256,), args, stream=stream)
    dims = tuple(map(np.int64, shape))
    layout = tuple(map(np.int64, (*padded, padded[0], 1, padded[1], 1)))
    for mask in ('none', 'rectangular'):
        for mode in ('static', 'dynamic'):
            profile = {'implementation': 'native', 'prune': True,
                       'transfer': 'fused' if mode == 'dynamic' else 'materialized'}
            with ASMPlan(shape, (6.4e-6, 6.4e-6), mode=mode, bandlimit=mask,
                         stream=stream, tuning_profile=profile) as plan:
                for z, wavelength in ((.01, 450e-9), (.1, 532e-9), (1., 633e-9)):
                    launch('generate_transfer', size, (transfer, *layout, np.float64(6.4e-6), np.float64(6.4e-6),
                           np.float64(z), np.float64(wavelength), np.int32(mask == 'rectangular')))
                    launch('pad_input', size, (x, work, *dims))
                    cuplan.fft(work, work, cufft.CUFFT_FORWARD)
                    launch('multiply', size, (work, transfer, np.int64(size)))
                    cuplan.fft(work, work, cufft.CUFFT_INVERSE)
                    launch('crop_output', x.size, (work, reference, *dims))
                    if mode == 'static': plan.prepare_transfer(transfer)
                    plan._work.fill(cp.nan)
                    plan.execute(x, out, **({'z': z, 'wavelength': wavelength} if mode == 'dynamic' else {}))
                    num = den = max_error = max_reference = 0.
                    for i in range(0, x.size, 2**20):
                        actual = out.ravel()[i:i+2**20]
                        expected = reference.ravel()[i:i+2**20]
                        difference = cp.abs(actual-expected)
                        num += float(cp.sum(difference**2, dtype=cp.float64))
                        den += float(cp.sum(cp.abs(expected)**2, dtype=cp.float64))
                        max_error = max(max_error, float(cp.max(difference)))
                        max_reference = max(max_reference, float(cp.max(cp.abs(expected))))
                    l2 = math.sqrt(num/den) if den else math.sqrt(num)
                    scaled_max = max_error/max_reference if max_reference else max_error
                    assert math.isfinite(l2) and l2 <= 2e-5, l2
                    assert math.isfinite(scaled_max) and scaled_max <= 1e-4, scaled_max
                    result['checks'].append(dict(mode=mode, bandlimit=mask, z=z, wavelength=wavelength,
                                                 relative_l2=l2, scaled_max_error=scaled_max))
            cp.get_default_memory_pool().free_all_blocks()
    assert bool(cp.array_equal(x, original)), 'Input was modified'
    stream.synchronize()
    result['pool_reserved_after_checks_bytes'] = cp.get_default_memory_pool().total_bytes()
    result['initial_free_bytes'] = free_initial
result['maximum_relative_l2'] = max(r['relative_l2'] for r in result['checks'])
result['maximum_scaled_error'] = max(r['scaled_max_error'] for r in result['checks'])
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'}), flush=True)
