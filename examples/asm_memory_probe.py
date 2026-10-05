"""Measure isolated ASM allocation peaks and prepared execution storage.

The CuPy allocator high-water mark covers live allocations routed through that
allocator. Driver observations are samples, not a guarantee of catching every
short-lived internal CUDA allocation. Construction and preparation are included.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--shape', type=int, nargs=2, required=True)
    parser.add_argument('--variant', choices=('native', 'cufft', 'cufft_symmetric'), required=True)
    parser.add_argument('--mode', choices=('static', 'dynamic'), required=True)
    parser.add_argument('--axis-order', choices=('row', 'column'), default='row')
    parser.add_argument('--crop-callback', action='store_true')
    parser.add_argument('--analytic-static', action='store_true',
                        help='Generate native static ASM H directly in packed storage')
    parser.add_argument('--native-transfer', choices=('materialized', 'fused', 'symmetric',
                                                      'cached_phase', 'cached_phase_symmetric'))
    parser.add_argument('--grouped-batch', type=int, nargs=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.analytic_static and (args.variant not in ('native', 'cufft_symmetric') or args.mode != 'static'):
        parser.error('--analytic-static requires native/cufft_symmetric and --mode static')
    if args.variant == 'cufft_symmetric' and (args.mode != 'static' or not args.analytic_static):
        parser.error('cufft_symmetric requires --mode static --analytic-static')
    if args.native_transfer and args.variant != 'native':
        parser.error('--native-transfer requires --variant native')
    if args.native_transfer == 'symmetric' and not args.analytic_static:
        parser.error('Symmetric native transfer requires --analytic-static')
    import subprocess
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    import ctypes
    cuda = Path(os.environ['CUDA_PATH'])
    ctypes.CDLL(str(cuda/'targets/x86_64-linux/lib/libcufft.so.11'), mode=ctypes.RTLD_GLOBAL)
    import cupy as cp
    from pyvkfft.asm import ASMPlan
    from asm_cufft import CuFFTASMPlan

    cp.cuda.Device(0).use()
    cp.cuda.runtime.free(0)
    free_start, total = cp.cuda.runtime.memGetInfo()
    shape = tuple(args.shape)
    grid_bytes = shape[0]*shape[1]*32
    if 6*grid_bytes > .8*free_start:
        raise MemoryError('Conservative isolated-probe preflight exceeds 80% of free VRAM')
    pool = cp.cuda.MemoryPool()
    peaks = dict(live_pool_bytes=0, reserved_pool_bytes=0, observed_driver_delta_bytes=0)

    def observe():
        used, reserved = pool.used_bytes(), pool.total_bytes()
        driver = free_start-cp.cuda.runtime.memGetInfo()[0]
        peaks['live_pool_bytes'] = max(peaks['live_pool_bytes'], used)
        peaks['reserved_pool_bytes'] = max(peaks['reserved_pool_bytes'], reserved)
        peaks['observed_driver_delta_bytes'] = max(peaks['observed_driver_delta_bytes'], driver)
        return dict(live_pool_bytes=used, reserved_pool_bytes=reserved, driver_delta_bytes=driver)

    def allocate(size):
        ptr = pool.malloc(size)
        observe()
        return ptr

    cp.cuda.set_allocator(allocate)
    snapshots = {}
    x = cp.ones(shape, cp.complex64)
    output = cp.empty_like(x)
    # Dynamic cuFFT kernels need an argument pointer but never read H; a dummy
    # avoids retaining the benchmark harness's unused full reference H array.
    transfer = (None if args.analytic_static else
                cp.ones(tuple(2*n for n in shape), cp.complex64) if args.mode == 'static' else
                cp.empty(1, cp.complex64))
    snapshots['caller_arrays'] = observe()
    if args.variant == 'native':
        profile = dict(implementation='native', prune=True, axis_order=args.axis_order,
                       transfer=args.native_transfer or ('fused' if args.mode == 'dynamic' else 'materialized'))
        if args.grouped_batch is not None:
            profile['groupedBatch'] = args.grouped_batch
        plan = ASMPlan(shape, (6.4e-6, 6.4e-6), mode=args.mode, tuning_profile=profile)
        snapshots['constructed'] = observe()
        if args.mode == 'static':
            if args.analytic_static:
                plan.prepare_asm_transfer(z=0., wavelength=532e-9)
            else:
                plan.prepare_transfer(transfer)
    elif args.variant == 'cufft_symmetric':
        from asm_cufft_symmetric import CuFFTStaticPlan
        plan = CuFFTStaticPlan(shape, (6.4e-6, 6.4e-6), order=args.axis_order,
                              compressed=True, crop_callback=args.crop_callback)
        snapshots['constructed'] = observe()
        plan.prepare_asm_transfer(z=0., wavelength=532e-9)
    else:
        plan = CuFFTASMPlan(shape, (6.4e-6, 6.4e-6), transfer,
                           dynamic=args.mode == 'dynamic', axis_order=args.axis_order,
                           crop_callback=args.crop_callback)
        snapshots['constructed'] = observe()
    cp.cuda.get_current_stream().synchronize()
    snapshots['prepared_source_H_retained'] = observe()
    del transfer
    pool.free_all_blocks()
    snapshots['prepared_source_H_released'] = observe()
    if args.variant in ('native', 'cufft_symmetric'):
        plan.execute(x, output, **({'z': 0., 'wavelength': 532e-9} if args.mode == 'dynamic' else {}))
    else:
        plan.execute(x, output, 0., 532e-9)
    cp.cuda.get_current_stream().synchronize()
    snapshots['executed'] = observe()
    allocation_peaks = dict(peaks)
    # Identity-H sanity check after ending the memory measurement interval.
    error = float(cp.max(cp.abs(output[::64, ::64]-1)))
    if error > 2e-5:
        raise AssertionError(error)
    result = dict(shape=shape, padded_grid_bytes=grid_bytes, variant=args.variant,
                  mode=args.mode, axis_order=args.axis_order, crop_callback=args.crop_callback,
                  analytic_static=args.analytic_static,
                  native_transfer=args.native_transfer,
                  gpu_uuid=uuid, initial_free_bytes=free_start, total_bytes=total,
                  cufft_version=int(cp.cuda.cufft.getVersion()),
                  peaks=allocation_peaks, snapshots=snapshots, plan=plan.info,
                  identity_sample_max_error=error,
                  scope='Fresh process; input/output included; original static H released after preparation. Allocator high-water includes construction/preparation, excludes final sanity-check allocations. Driver peak is sampled at allocator events and stage boundaries.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), peaks=allocation_peaks,
                         prepared=snapshots['prepared_source_H_released'])), flush=True)


if __name__ == '__main__':
    main()
