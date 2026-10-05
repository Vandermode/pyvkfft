"""Benchmark static analytic transfer compression against native and fair cuFFT.

Run each comparison in three independent processes with --rounds 4 --seconds 5.
The primary family compares full native, compressed native and compressed cuFFT.
For large column-oriented shapes use separate native and cufft families to bound
combined live memory. cuFFT uses quadrant H and one reused private column buffer;
the measured faster crop method is callback for row order and explicit for column.
Both column transposes are timed. Transfer preparation and validation are untimed.

Select the ordinary and symmetry libraries through CLI flags or their usual
PYVKFFT_ASM_LIBRARY / PYVKFFT_ASM_SYMMETRIC_LIBRARY environment variables. Pass
--cufft-library explicitly for cuFFT comparisons to avoid inadvertently benchmarking
a bundled cuFFT version. CUDA_VISIBLE_DEVICES (or --gpu) must be one GPU UUID.
Owned memory includes plan arrays and cuFFT scratch, excluding caller input/output;
planned_pair_bytes is postvalidation process memory, not a per-variant peak.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np


def benchmark(args):
    shape = tuple(args.shape)
    pitch = (6.4e-06, 6.4e-06)
    stream = cp.cuda.Stream(non_blocking=True)
    groups = [0, 32] if args.order == 'row' else [0, 0]
    free = cp.cuda.runtime.memGetInfo()[0]
    with stream:
        cp.random.seed(928)
        x = cp.empty(shape, cp.complex64)
        x.real = cp.random.standard_normal(shape, dtype=cp.float32)
        x.imag = cp.random.standard_normal(shape, dtype=cp.float32)
        out = cp.empty_like(x)
        expected = cp.empty_like(x)
        plans = {}
        owned = {}
        labels = {'primary': ['full', 'quadrant', 'cufft'], 'native': ['full', 'quadrant'], 'cufft': ['cufft', 'quadrant']}[args.family]
        for label in labels:
            used = cp.get_default_memory_pool().used_bytes()
            if label == 'cufft':
                p = CuFFTStaticPlan(shape, pitch, args.mask, args.order, True, stream, crop_callback=args.order == 'row')
            else:
                profile = {'implementation': 'native', 'axis_order': args.order, 'prune': True, 'groupedBatch': groups}
                if label == 'quadrant':
                    profile['transfer'] = 'symmetric'
                p = ASMPlan(shape, pitch, mode='static', bandlimit=args.mask, stream=stream, tuning_profile=profile)
            plans[label] = p
            owned[label] = cp.get_default_memory_pool().used_bytes() - used
        baseline = plans[labels[0]]
        errors = []
        for z, w in ((0.01, 5.32e-07), (-0.1, 4.5e-07), (100.0, 6.33e-07)):
            for p in plans.values():
                p.prepare_asm_transfer(z=z, wavelength=w)
            baseline.execute(x, expected)
            for label, candidate in plans.items():
                if candidate is baseline:
                    continue
                with patch('cupy.empty', side_effect=AssertionError('execute allocation')), patch('cupy.empty_like', side_effect=AssertionError('execute allocation')):
                    candidate.execute(x, out)
                stream.synchronize()
                num = den = peak = mx = 0.0
                for y in range(0, shape[0], 64):
                    a, b = (out[y:y + 64], expected[y:y + 64])
                    num += float(cp.sum(cp.abs(a - b) ** 2, dtype=cp.float64).get())
                    den += float(cp.sum(cp.abs(b) ** 2, dtype=cp.float64).get())
                    mx = max(mx, float(cp.max(cp.abs(a - b)).get()))
                    peak = max(peak, float(cp.max(cp.abs(b)).get()))
                e = float(np.sqrt(num / max(den, 1e-30)))
                re = mx / max(peak, 1e-30)
                assert np.isfinite(e) and e < 2e-05 and (re < 0.0001), (e, re)
                errors.append({'label': label, 'reference': labels[0], 'z': z, 'wavelength': w, 'relative_l2': e, 'relative_max': re})
        for p in plans.values():
            p.prepare_asm_transfer(z=0.1, wavelength=5.32e-07)
        del expected
        stream.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        planned = free - cp.cuda.runtime.memGetInfo()[0]
        assert planned < 0.8 * free
        blocks = []
        labels = list(plans)
        monitor = Monitor(os.environ['CUDA_VISIBLE_DEVICES'])
        monitor.thread.start()
        for r in range(args.rounds):
            for label in labels if r % 2 == 0 else labels[::-1]:
                p = plans[label]
                a, b = (cp.cuda.Event(), cp.cuda.Event())
                a.record(stream)
                for i in range(10):
                    p.execute(x, out)
                b.record(stream)
                b.synchronize()
                estimate = cp.cuda.get_elapsed_time(a, b) / 10
                count = max(20, math.ceil(args.seconds * 1000 / estimate))
                events = [cp.cuda.Event() for _ in range(count + 1)]
                t0 = time.monotonic()
                events[0].record(stream)
                for i in range(count):
                    p.execute(x, out)
                    events[i + 1].record(stream)
                events[-1].synchronize()
                t1 = time.monotonic()
                times = [cp.cuda.get_elapsed_time(events[i], events[i + 1]) for i in range(count)]
                blocks.append(dict(label=label, round=r, samples_ms=times, start_monotonic=t0, end_monotonic=t1))
                print(shape, args.order, args.mask, r, label, statistics.median(times), flush=True)
        monitor.close()
        result = {'shape': shape, 'mask': args.mask, 'order': args.order, 'info': {k: p.info for k, p in plans.items()}, 'correctness': errors, 'owned_device_bytes': owned, 'family': args.family, 'cufft_version': cp.cuda.cufft.getVersion(), 'planned_pair_bytes': planned, 'initial_free_bytes': free, 'blocks': blocks, 'telemetry': monitor.rows, 'telemetry_errors': monitor.errors, 'median_ms': {label: statistics.median((statistics.median(b['samples_ms']) for b in blocks if b['label'] == label)) for label in labels}, 'baseline': 'Frozen public ASMPlan ordinary/full and common symmetric ABI1/layout1 versus specialized pruned cuFFT 11.4.1 with screened-fastest crop (primary callback / tall explicit) and quarter analytic H; column variants reuse private compact buffer', 'protocol': 'Complete compact operator; immutable random source; static analytic H prepared outside timing; no execute allocation; paired alternating blocks', 'prepared_transfer_parameters': {'z': 0.1, 'wavelength': 5.32e-07}}
        for p in plans.values():
            p.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--family', choices=['primary', 'native', 'cufft'], required=True)
    parser.add_argument('--shape', nargs=2, type=int, default=(4096, 16384))
    parser.add_argument('--order', choices=['row', 'column'], default='row')
    parser.add_argument('--mask', choices=['none', 'rectangular'], default='none')
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=5,
                        help='Nominal block seconds, estimated from ten warm-up executions')
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        help='One GPU UUID, default CUDA_VISIBLE_DEVICES')
    parser.add_argument('--library', type=Path, default=os.environ.get('PYVKFFT_ASM_LIBRARY'))
    parser.add_argument('--symmetric-library', type=Path,
                        default=os.environ.get('PYVKFFT_ASM_SYMMETRIC_LIBRARY'))
    parser.add_argument('--cufft-library', type=Path,
                        default=os.environ.get('PYVKFFT_CUFFT_LIBRARY'),
                        help='Exact libcuFFT to preload before importing CuPy; required for fair cuFFT families')
    parser.add_argument('--python-snapshot', type=Path,
                        help='Optional directory containing a frozen pyvkfft package')
    args = parser.parse_args()
    if min(args.shape) < 1 or args.rounds < 1 or args.seconds <= 0:
        parser.error('Shape, rounds, and seconds must be positive')
    if not args.gpu or not args.gpu.startswith('GPU-') or ',' in args.gpu:
        parser.error('--gpu or CUDA_VISIBLE_DEVICES must select one GPU UUID')
    for name in ('library', 'symmetric_library'):
        value = getattr(args, name)
        if value is None or not value.is_file():
            parser.error(f'--{name.replace("_", "-")} must identify an existing library')
    if args.family != 'native' and args.cufft_library is None:
        parser.error('--cufft-library is required for cuFFT comparisons')
    if args.cufft_library is not None and not args.cufft_library.is_file():
        parser.error('--cufft-library must identify an existing library')
    if args.python_snapshot is not None and not (args.python_snapshot / 'pyvkfft').is_dir():
        parser.error('--python-snapshot must contain the frozen pyvkfft package')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['PYVKFFT_ASM_LIBRARY'] = str(args.library.resolve())
    os.environ['PYVKFFT_ASM_SYMMETRIC_LIBRARY'] = str(args.symmetric_library.resolve())
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    if args.python_snapshot is not None:
        sys.path.insert(0, str(args.python_snapshot.resolve()))
    if args.cufft_library is not None:
        ctypes.CDLL(str(args.cufft_library.resolve()), mode=ctypes.RTLD_GLOBAL)
    global cp, ASMPlan, CuFFTStaticPlan, Monitor
    import cupy as cp
    from pyvkfft.asm import ASMPlan
    from asm_cufft_symmetric import CuFFTStaticPlan
    from issue205_sustained import Monitor
    import pyvkfft.asm as asm_module
    import pyvkfft.asm_axis as axis_module
    import pyvkfft.asm_symmetric as symmetric_module
    import asm_cufft
    import asm_cufft_symmetric

    result = benchmark(args)
    # The original result string describes the published 11.4.1 runs; record the
    # selected implementation/version accurately for portable reproductions.
    result['baseline'] = ('Public ASMPlan full and symmetric transfer versus pruned '
                          f'cuFFT {result["cufft_version"]} with quadrant analytic H; '
                          'row crop callback / column explicit crop, one private column buffer')
    result['helper_sha256'] = hashlib.sha256(Path(asm_cufft_symmetric.__file__).read_bytes()).hexdigest()
    result['study_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['source_files'] = {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                              for m in (asm_module, axis_module, symmetric_module, asm_cufft, asm_cufft_symmetric)}
    result['libraries'] = {name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                           for name, path in (('native', args.library), ('symmetric', args.symmetric_library),
                                              ('cufft', args.cufft_library)) if path is not None}
    result['python_snapshot'] = str(args.python_snapshot.resolve()) if args.python_snapshot else None
    result['gpu_uuid'] = args.gpu
    result['mode'] = 'static'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k not in ('blocks', 'telemetry', 'info')}, flush=True)


if __name__ == '__main__':
    main()
