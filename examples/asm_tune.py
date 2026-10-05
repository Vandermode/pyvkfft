"""Bounded, generated-source-deduplicated experimental ASM tuning.

CUDA_VISIBLE_DEVICES must select an authorized idle GPU. Every candidate gets
a fresh process so a failed generated kernel cannot contaminate later trials.
Screening timings are exploratory; use asm_benchmark.py for sustained evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import cupy as cp
    import numpy as np
    from pyvkfft.asm import ASMPlan
    profile = json.loads(args.profile)
    shape = tuple(args.shape)
    cp.random.seed(715)
    x = (cp.random.random(shape, dtype=cp.float32) +
         1j*cp.random.random(shape, dtype=cp.float32)).astype(cp.complex64)
    out = cp.empty_like(x)
    plan = ASMPlan(shape, (6.4e-6, 6.4e-6), mode=args.mode,
                   tuning_profile=profile)
    h, w = shape
    padded = cp.zeros((2*h, 2*w), cp.complex64)
    padded[h//2:h//2+h, w//2:w//2+w] = x
    optical = {'z': .1, 'wavelength': 532e-9} if args.mode == 'dynamic' else {}
    if args.mode == 'static':
        transfer = cp.exp(1j*cp.arange(4*h*w, dtype=cp.float32).reshape(2*h, 2*w)*1e-5)
        plan.prepare_transfer(transfer)
    else:
        from pyvkfft.asm import CUDA_KERNELS
        module = cp.RawModule(code=CUDA_KERNELS)
        transfer = cp.empty_like(padded)
        module.get_function('generate_transfer')(((transfer.size+255)//256,), (256,),
            (transfer, *map(np.int64, (2*h, 2*w, 2*h, 1, 2*w, 1)),
             np.float64(6.4e-6), np.float64(6.4e-6), np.float64(.1),
             np.float64(532e-9), np.int32(0)))
    ref = cp.fft.ifft2(cp.fft.fft2(padded)*transfer)[h//2:h//2+h, w//2:w//2+w].copy()
    del padded, transfer
    cp.get_default_memory_pool().free_all_blocks()
    plan._work.fill(cp.nan)
    plan.execute(x, out, **optical)
    error = float((cp.linalg.norm(out-ref)/cp.linalg.norm(ref)).get())
    if not np.isfinite(error) or error > 2e-5:
        raise RuntimeError(f'Incorrect candidate: relative L2 {error}')
    info = plan.info
    fingerprints = [k['source_fnv1a64'] for k in info['kernels']]
    fingerprint = hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest()
    previous = set(json.loads(Path(args.seen).read_text())) if Path(args.seen).exists() else set()
    result = dict(profile=profile, info=info, fingerprint=fingerprint, relative_l2=error)
    if fingerprint in previous:
        result['duplicate'] = True
    else:
        for _ in range(4):
            plan.execute(x, out, **optical)
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        blocks = []
        for _ in range(args.blocks):
            start.record()
            for _ in range(args.iterations):
                plan.execute(x, out, **optical)
            stop.record(); stop.synchronize()
            blocks.append(cp.cuda.get_elapsed_time(start, stop)/args.iterations)
        result['milliseconds'] = blocks
        result['median_ms'] = float(np.median(blocks))
    Path(args.result).write_text(json.dumps(result, indent=2)+'\n')
    plan.close()


def main(args):
    if args.profile:
        worker(args)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base = dict(implementation='native', prune=True,
                transfer='fused' if args.mode == 'dynamic' else 'materialized',
                aimThreads=128, coalescedMemory=32)
    candidates = [base]
    candidates += [dict(base, aimThreads=t) for t in (64, 256, 512)]
    candidates += [dict(base, coalescedMemory=c) for c in (64, 128)]
    candidates += [dict(base, groupedBatch=[x, y]) for x,y in
                   ((4,0), (8,0), (16,0), (32,0), (0,4), (0,8), (0,16), (0,32))]
    seen_path = output/'seen.json'
    seen, results = [], []
    for i, profile in enumerate(candidates):
        seen_path.write_text(json.dumps(seen))
        result = output/f'candidate-{i:02d}.json'
        command = [sys.executable, __file__, '--shape', *map(str,args.shape),
                   '--mode', args.mode, '--profile', json.dumps(profile),
                   '--result', str(result), '--seen', str(seen_path),
                   '--blocks', str(args.blocks), '--iterations', str(args.iterations)]
        with (output/f'candidate-{i:02d}.log').open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            item = dict(profile=profile, failed=True, returncode=process.returncode)
        else:
            item = json.loads(result.read_text())
            seen.append(item['fingerprint'])
        results.append(item)
        (output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        print(i, profile, item.get('median_ms', 'duplicate' if item.get('duplicate') else 'failed'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', nargs=2, type=int, default=[4096, 16384])
    p.add_argument('--mode', choices=['static','dynamic'], default='static')
    p.add_argument('--output', default='asm-tuning')
    p.add_argument('--profile')
    p.add_argument('--result')
    p.add_argument('--seen', default='/nonexistent')
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--iterations', type=int, default=20)
    main(p.parse_args())
