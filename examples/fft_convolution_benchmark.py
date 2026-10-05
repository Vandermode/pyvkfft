"""Compare prepared, in-place 2D C2C convolution using cuFFT and VkFFT.

Local runs select physical GPUs 1-3. Slurm runs respect the assigned devices.
cuFFT's inverse is unnormalized: its precomputed filter includes 1 / prod(shape).
The optional cuFFT LTO load callback fuses that filter into the inverse load.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from issue205_sustained import Monitor


CALLBACK = r'''
__device__ float2 convolution_multiply_load(void* dataIn, unsigned long long offset,
                                          void* callerInfo, void* sharedPtr) {
    float2 a = ((float2*)dataIn)[offset];
    float2 k = ((float2*)callerInfo)[offset];
    return make_float2(a.x*k.x-a.y*k.y, a.x*k.y+a.y*k.x);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument('--gpu', type=int, choices=(1, 2, 3))
    device.add_argument('--slurm', action='store_true')
    parser.add_argument('--shape', type=int, nargs=2, required=True)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--cufft-library', type=Path,
                        help='Preload an exact cuFFT library before CuPy chooses a wheel copy')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=[
        'cufft', 'cufft_callback', 'vkfft_ordered', 'vkfft_unordered', 'vkfft_native'],
        choices=['cufft', 'cufft_callback', 'vkfft_ordered', 'vkfft_unordered', 'vkfft_native'])
    parser.add_argument('--seconds-per-block', type=float, default=3)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--vkfft-options', type=json.loads, default={},
                        help='JSON with VkFFT tuning parameters, e.g. {"aimThreads":128}')
    parser.add_argument('--profile', action='store_true',
                        help='Capture one prepared convolution per variant via CUDA profiler API')
    args = parser.parse_args()
    if args.slurm != bool(os.environ.get('SLURM_JOB_ID')):
        parser.error('Use --slurm inside Slurm and --gpu outside Slurm')
    if args.gpu is not None:
        uuid = subprocess.check_output([
            'nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid', '--format=csv,noheader'
        ], text=True).strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    if min(args.shape) <= 0 or args.seconds_per_block <= 0 or args.rounds <= 0:
        parser.error('Dimensions, block duration, and rounds must be positive')

    if args.cufft_library:
        cufft_library = ctypes.CDLL(str(args.cufft_library.resolve()), mode=ctypes.RTLD_GLOBAL)
    import cupy as cp
    import numpy as np
    from cupy.cuda import cufft
    from cupyx.scipy.fft import get_fft_plan
    if args.library:
        import pyvkfft.base
        original_loader = pyvkfft.base.load_library
        pyvkfft.base.load_library = lambda name: (
            ctypes.CDLL(str(args.library.resolve())) if name == '_vkfft_cuda'
            else original_loader(name))
    from pyvkfft.cuda import VkFFTApp, _vkfft_cuda, vkfft_version

    cp.cuda.Device(0).use()
    pci_bus_id = cp.cuda.runtime.deviceGetPCIBusId(0)
    if isinstance(pci_bus_id, bytes):
        pci_bus_id = pci_bus_id.decode()
    uuid = subprocess.check_output([
        'nvidia-smi', '-i', pci_bus_id, '--query-gpu=uuid', '--format=csv,noheader'
    ], text=True).strip()
    shape = tuple(args.shape)
    size = math.prod(shape)
    rng = cp.random.RandomState(205)
    # Preparation uses float32 components to bound temporary storage.
    x = cp.empty(shape, cp.complex64)
    x.real = rng.standard_normal(shape, dtype=cp.float32)
    x.imag = rng.standard_normal(shape, dtype=cp.float32)
    kernel = cp.empty_like(x)
    phase = rng.uniform(-np.pi, np.pi, size=shape).astype(cp.float32)
    cp.ElementwiseKernel('float32 phase', 'complex64 k',
                         'k=complex<float>(cosf(phase),sinf(phase));',
                         'prepare_phase_filter')(phase, kernel)
    del phase
    cp.get_default_memory_pool().free_all_blocks()
    scaled_kernel = kernel / size
    buffer = cp.empty_like(x)
    cuplan = cufft.PlanNd(shape, shape, 1, size, shape, 1, size,
                         cufft.CUFFT_C2C, 1, 'C', 1, shape[-1])
    reference = x.copy()
    cuplan.fft(reference, reference, cufft.CUFFT_FORWARD)
    cp.multiply(reference, scaled_kernel, out=reference)
    cuplan.fft(reference, reference, cufft.CUFFT_INVERSE)

    plans = {}
    spectra = {}
    callables = {}
    plan_info = {}
    if 'cufft' in args.variants:
        def cu_convolution():
            cuplan.fft(buffer, buffer, cufft.CUFFT_FORWARD)
            cp.multiply(buffer, scaled_kernel, out=buffer)
            cuplan.fft(buffer, buffer, cufft.CUFFT_INVERSE)
        callables['cufft'] = cu_convolution
    if 'cufft_callback' in args.variants:
        with cp.fft.config.set_cufft_callbacks(
                cb_load=CALLBACK, cb_load_name='convolution_multiply_load',
                cb_load_data=scaled_kernel.data, cb_ver='jit'):
            callback_plan = get_fft_plan(buffer, axes=(0, 1), value_type='C2C')

        def cu_callback_convolution():
            cuplan.fft(buffer, buffer, cufft.CUFFT_FORWARD)
            callback_plan.fft(buffer, buffer, cufft.CUFFT_INVERSE)
        callables['cufft_callback'] = cu_callback_convolution

    pack = cp.ElementwiseKernel(
        'raw complex64 natural, int64 width, int64 ay, int64 by, int64 ax, int64 bx',
        'complex64 packed', r'''
        long long y = i / width, x = i % width;
        long long sy = (y % ay) * by + y / ay;
        long long sx = (x % ax) * bx + x / ax;
        packed = natural[sy * width + sx];
    ''', 'pack_convolution_spectrum')
    packed_cache = {}
    for label in args.variants:
        if not label.startswith('vkfft'):
            continue
        native = label == 'vkfft_native'
        app = VkFFTApp(shape, np.complex64, inplace=True, norm=1,
                       convolve=native, disableReorderFourStep=(label != 'vkfft_ordered'),
                       **args.vkfft_options)
        plans[label] = app
        config = app.app.contents.configuration
        plan_info[label] = {'uploads': app.nb_axis_upload, 'axis_split': app.axis_split.tolist(),
                            'temporary_bytes': int(config.tempBufferSize[0]) if config.allocateTempBuffer else 0}
        if label == 'vkfft_ordered':
            spectra[label] = kernel
        else:
            if any(app.use_bluestein_fft) or max(app.nb_axis_upload) > 2:
                raise ValueError('Spectrum packing supports one/two-upload radix FFTs')
            ay, by = (map(int, app.axis_split[1, :2]) if app.nb_axis_upload[1] == 2 else (shape[0], 1))
            ax, bx = (map(int, app.axis_split[0, :2]) if app.nb_axis_upload[0] == 2 else (shape[1], 1))
            key = (ay, by, ax, bx)
            if key not in packed_cache:
                packed_cache[key] = cp.empty_like(kernel)
                pack(kernel, shape[1], ay, by, ax, bx, packed_cache[key])
            spectra[label] = packed_cache[key]
        if native:
            callables[label] = lambda a=app, k=spectra[label]: a.fft(buffer, convolve_kernel=k)
        else:
            def vk_convolution(a=app, k=spectra[label]):
                a.fft(buffer)
                cp.multiply(buffer, k, out=buffer)
                a.ifft(buffer)
            callables[label] = vk_convolution

    def error(actual, expected):
        numerator, denominator = 0., 0.
        for offset in range(0, size, 2**20):
            a = actual.ravel()[offset:offset + 2**20]
            b = expected.ravel()[offset:offset + 2**20]
            numerator += float(cp.sum(cp.abs(a-b)**2, dtype=cp.float64))
            denominator += float(cp.sum(cp.abs(b)**2, dtype=cp.float64))
        return math.sqrt(numerator / denominator)

    correctness = {}
    for label, execute in callables.items():
        cp.copyto(buffer, x)
        execute()
        correctness[label] = error(buffer, reference)
        if not math.isfinite(correctness[label]) or correctness[label] >= 2e-5:
            raise AssertionError((label, correctness[label]))
    del reference
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    result = {'shape': shape, 'dtype': 'complex64', 'gpu_uuid': uuid,
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'cufft_version': cufft.getVersion(), 'vkfft_version': vkfft_version(),
              'cufft_library': str(args.cufft_library.resolve()) if args.cufft_library else 'CuPy default',
              'cupy_version': cp.__version__, 'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
              'library_sha256': hashlib.sha256(Path(_vkfft_cuda._name).read_bytes()).hexdigest(),
              'plans': plan_info, 'correctness_relative_l2': correctness,
              'vkfft_options': args.vkfft_options,
              'profile_only': args.profile, 'blocks': [],
              'protocol': {'prepared_filter': 'unit-modulus, normalized offline for cuFFT',
                           'timed': 'FFT, multiplication (separate or fused), IFFT',
                           'reset': 'outside warm-up and timed regions',
                           'order': 'alternate forward/reverse variant order across rounds'}}
    print(json.dumps({k: result[k] for k in ('shape', 'device', 'cufft_version', 'correctness_relative_l2')}), flush=True)

    if args.profile:
        for execute in callables.values():
            cp.copyto(buffer, x)
            for _ in range(10):
                execute()
        cp.cuda.Stream.null.synchronize()
        cp.cuda.profiler.start()
        for label, execute in callables.items():
            cp.cuda.nvtx.RangePush(label)
            execute()
            cp.cuda.nvtx.RangePop()
        cp.cuda.Stream.null.synchronize()
        cp.cuda.profiler.stop()
    else:
        monitor = Monitor(uuid)
        monitor.thread.start()
        try:
            labels = list(callables)
            for round_index in range(args.rounds):
                order = labels if round_index % 2 == 0 else labels[::-1]
                for label in order:
                    execute = callables[label]
                    cp.copyto(buffer, x)
                    start, end = cp.cuda.Event(), cp.cuda.Event()
                    start.record()
                    for _ in range(20):
                        execute()
                    end.record()
                    end.synchronize()
                    estimate = cp.cuda.get_elapsed_time(start, end) / 20
                    count = max(20, math.ceil(args.seconds_per_block * 1000 / estimate))
                    events = [cp.cuda.Event() for _ in range(count + 1)]
                    t0 = time.monotonic()
                    events[0].record()
                    for i in range(count):
                        execute()
                        events[i + 1].record()
                    events[-1].synchronize()
                    t1 = time.monotonic()
                    times = [cp.cuda.get_elapsed_time(events[i], events[i+1]) for i in range(count)]
                    result['blocks'].append({'label': label, 'round': round_index,
                                             'start_monotonic': t0, 'end_monotonic': t1,
                                             'samples_ms': times})
                    print(f'{label}: {np.median(times):.3f} ms ({count} convolutions)', flush=True)
        finally:
            monitor.close()
        result['telemetry'] = monitor.rows
        result['telemetry_errors'] = monitor.errors
        result['median_ms'] = {label: float(np.median([t for b in result['blocks']
                                                     if b['label'] == label for t in b['samples_ms']]))
                               for label in callables}
        print(json.dumps(result['median_ms'], indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
