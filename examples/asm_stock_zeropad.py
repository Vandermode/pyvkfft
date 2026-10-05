"""Audit stock VkFFT zero-interval controls for the exact padded ASM operator.

A simultaneous circular translation of embedding and crop leaves convolution
unchanged; leading-corner embedding exposes one trailing zero interval per axis.
"""
from pyvkfft.cuda import VkFFTApp, VKFFT_MAX_FFT_DIMENSIONS
from pyvkfft.asm import ASMPlan, CUDA_KERNELS
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
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = subprocess.check_output(
    ['nvidia-smi', '-i', str(a.gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
shape = tuple(a.shape)
h, w = shape
padded = (2*h, 2*w)
free, _ = cp.cuda.runtime.memGetInfo()
if 8*32*h*w > .8*free:
    raise MemoryError('Preflight exceeds 80% available VRAM')
stream = cp.cuda.Stream(non_blocking=True)
with stream:
    rng = cp.random.RandomState(59)
    x = cp.empty(shape, cp.complex64)
    x.real = rng.standard_normal(shape, dtype=cp.float32)
    x.imag = rng.standard_normal(shape, dtype=cp.float32)
    out = cp.empty_like(x)
    reference = cp.empty_like(x)
    work = cp.empty(padded, cp.complex64)
    transfer = cp.empty_like(work)
    transfer.real = rng.uniform(-1, 1, padded, dtype=cp.float32)
    transfer.imag = rng.uniform(-1, 1, padded, dtype=cp.float32)
    flags, left, right = [
        np.zeros(VKFFT_MAX_FFT_DIMENSIONS, np.uint64) for _ in range(3)]
    flags[:2] = 1
    left[:2] = (w, h)
    right[:2] = (2*w, 2*h)
    app = VkFFTApp(padded, np.complex64, inplace=True, norm=1, convolve=True, disableReorderFourStep=True, stream=stream,
                   performZeropadding=flags, fft_zeropad_left=left, fft_zeropad_right=right)
    packed = cp.empty_like(work)
    ay, by = app.axis_split[1, :2] if app.nb_axis_upload[1] == 2 else (2*h, 1)
    ax, bx = app.axis_split[0, :2] if app.nb_axis_upload[0] == 2 else (2*w, 1)
    pack = cp.RawModule(code=CUDA_KERNELS).get_function('prepare_transfer')
    pack(((work.size+255)//256,), (256,), (transfer, packed, *
         map(np.int64, (*padded, ay, by, ax, bx)), np.int32(0)), stream=stream)
    plan = ASMPlan(shape, (6.4e-6, 6.4e-6), stream=stream,
                   tuning_profile={'implementation': 'native', 'prune': True})
    plan.prepare_transfer(transfer)
    plan.execute(x, reference)

    def stock():
        cp.copyto(work[:h, :w], x)
        app.fft(work, convolve_kernel=packed)
        cp.copyto(out, work[:h, :w])
    errors = []
    for value in (cp.nan, 17+23j):
        work.fill(value)
        stock()
        stream.synchronize()
        num = den = 0.
        for i in range(0, x.size, 2**20):
            y = out.ravel()[i:i+2**20]
            r = reference.ravel()[i:i+2**20]
            num += float(cp.sum(cp.abs(y-r)**2, dtype=cp.float64))
            den += float(cp.sum(cp.abs(r)**2, dtype=cp.float64))
        errors.append((num/den)**.5)
    result = {'shape': shape, 'relative_l2_dirty_workspace': errors, 'correct': bool(all(np.isfinite(e) and e <= 2e-5 for e in errors)),
              'stock_uploads': list(map(int, app.nb_axis_upload)), 'stock_axis_split': app.axis_split.tolist(),
              'semantics': 'Zero interval [compact_length,padded_length), leading-corner embedding and same crop; equivalent by circular translation'}
    if result['correct']:
        result['blocks_ms'] = {'stock_flags': [], 'native_pruned': []}
        for round_index in range(4):
            variants = [('stock_flags', stock),
                        ('native_pruned', lambda: plan.execute(x, out))]
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
                count = max(10, int(1000/estimate))
                start.record(stream)
                for _ in range(count):
                    fn()
                end.record(stream)
                end.synchronize()
                result['blocks_ms'][label].append(
                    cp.cuda.get_elapsed_time(start, end)/count)
        result['median_ms'] = {k: float(np.median(v))
                               for k, v in result['blocks_ms'].items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    plan.close()
