"""Prepared-data C2C convolution benchmark with continuous GPU telemetry.

Uses a dense unit-modulus spectral filter so repeated convolutions stay bounded.
All preparation, correctness checks, input resets and warm-ups are outside timing.
Example: python examples/issue205_sustained.py --gpu 2 --shape 32768 32768
    --library /tmp/205-build/fixed.so --output /tmp/205-sustained.json
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time


class Monitor:
    class Utilization(ctypes.Structure):
        _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]

    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ('total', 'free', 'used')]

    def __init__(self, uuid):
        self.nvml = ctypes.CDLL('libnvidia-ml.so.1')
        self.check(self.nvml.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self.check(self.nvml.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.handle)))
        self.rows = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f'NVML error {status}')

    def sample(self):
        util, memory = self.Utilization(), self.Memory()
        power, clock = ctypes.c_uint(), ctypes.c_uint()
        self.check(self.nvml.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util)))
        self.check(self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)))
        self.check(self.nvml.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power)))
        self.check(self.nvml.nvmlDeviceGetClockInfo(self.handle, 1, ctypes.byref(clock)))
        return {'monotonic': time.monotonic(), 'gpu_percent': util.gpu,
                'memory_busy_percent': util.memory, 'memory_used_bytes': memory.used,
                'power_w': power.value / 1000, 'sm_clock_mhz': clock.value}

    def run(self):
        while not self.stop.is_set():
            try:
                self.rows.append(self.sample())
            except Exception as error:
                self.errors.append(str(error))
            self.stop.wait(0.1)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.nvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--shape', type=int, nargs=2, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds-per-block', type=float, default=8)
    args = parser.parse_args()
    if min(args.shape) <= 0 or args.seconds_per_block < 4:
        parser.error('Use positive dimensions and at least four seconds per block')
    uuid = subprocess.check_output([
        'nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid', '--format=csv,noheader'
    ], text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid

    import cupy as cp
    import numpy as np
    import pyvkfft.base
    original_loader = pyvkfft.base.load_library
    pyvkfft.base.load_library = lambda name: (
        ctypes.CDLL(str(args.library.resolve())) if name == '_vkfft_cuda'
        else original_loader(name))
    from pyvkfft.cuda import VkFFTApp, vkfft_version

    cp.cuda.Device(0).use()
    shape = tuple(args.shape)
    size = math.prod(shape)
    array_bytes = size * 8
    free, total = cp.cuda.runtime.memGetInfo()
    if 8 * array_bytes > free:
        raise RuntimeError(f'Need room for up to eight arrays ({8 * array_bytes / 2**30:.1f} GiB); '
                           f'{free / 2**30:.1f} GiB free')
    print(f'GPU {args.gpu}: preparing {shape}, {array_bytes / 2**30:.1f} GiB per array', flush=True)
    plans = {label: VkFFTApp(shape, np.complex64, inplace=True,
                           disableReorderFourStep=(label == 'disabled'))
             for label in ('normal', 'disabled')}
    for app in plans.values():
        if any(app.use_bluestein_fft) or max(app.nb_axis_upload) > 2:
            raise ValueError('This benchmark supports one/two-upload radix layouts')

    # Generate dense deterministic input and an arbitrary phase-only filter.
    # This is preparation; sin/cos and index permutations never run inside timing.
    generate = cp.ElementwiseKernel('', 'complex64 x, complex64 spectrum', r'''
        unsigned int h = (unsigned int)i * 747796405u + 2891336453u;
        h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
        h = (h >> 22u) ^ h;
        float phase = (float)(h & 65535u) * (6.283185307179586f / 65536.0f);
        spectrum = complex<float>(cosf(phase), sinf(phase));
        x = complex<float>((float)(h & 65535u) / 32768.0f - 1.0f,
                           (float)(h >> 16u) / 32768.0f - 1.0f);
    ''', 'issue205_prepare')
    x = cp.empty(shape, cp.complex64)
    kernels = {'normal': cp.empty_like(x)}
    generate(x, kernels['normal'])
    # One direct gather packs both dimensions without large intermediate arrays.
    disabled = plans['disabled']
    ay, by = (map(int, disabled.axis_split[1, :2])
              if disabled.nb_axis_upload[1] == 2 else (shape[0], 1))
    ax, bx = (map(int, disabled.axis_split[0, :2])
              if disabled.nb_axis_upload[0] == 2 else (shape[1], 1))
    pack = cp.ElementwiseKernel(
        'raw complex64 natural, int64 width, int64 a_y, int64 b_y, int64 a_x, int64 b_x',
        'complex64 packed', r'''
        long long y = i / width, x = i % width;
        long long natural_y = (y % a_y) * b_y + y / a_y;
        long long natural_x = (x % a_x) * b_x + x / a_x;
        packed = natural[natural_y * width + natural_x];
    ''', 'issue205_pack_spectrum')
    kernels['disabled'] = cp.empty_like(x)
    pack(kernels['normal'], shape[1], ay, by, ax, bx, kernels['disabled'])
    buffer = cp.empty_like(x)
    reference_spectrum = cp.fft.fftn(x)
    reference_spectrum *= kernels['normal']
    reference = cp.fft.ifftn(reference_spectrum)
    del reference_spectrum
    cp.get_default_memory_pool().free_all_blocks()

    def convolution(label):
        plans[label].fft(buffer)
        cp.multiply(buffer, kernels[label], out=buffer)
        plans[label].ifft(buffer)

    def relative_error(actual, expected):
        # Bound reduction scratch memory even for multi-GiB arrays.
        squared_error, squared_reference = 0.0, 0.0
        for first in range(0, size, 2**20):
            observed = actual.ravel()[first:first + 2**20]
            target = expected.ravel()[first:first + 2**20]
            squared_error += float(cp.sum(cp.abs(observed - target)**2, dtype=cp.float64))
            squared_reference += float(cp.sum(cp.abs(target)**2, dtype=cp.float64))
        return math.sqrt(squared_error / squared_reference)

    correctness = {}
    for label in plans:
        cp.copyto(buffer, x)
        convolution(label)
        correctness[label] = relative_error(buffer, reference)
        if not correctness[label] < 2e-5:
            raise AssertionError((label, correctness[label]))
    del reference
    cp.fft.config.get_plan_cache().clear()
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    print(f'GPU {args.gpu}: correctness {correctness}; starting sustained runs', flush=True)

    result = {
        'gpu_index': args.gpu, 'gpu_uuid': uuid, 'shape': shape, 'dtype': 'complex64',
        'array_bytes': array_bytes, 'device_total_bytes': total,
        'library': str(args.library.resolve()),
        'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
        'vkfft': vkfft_version(), 'cupy': cp.__version__, 'correctness_relative_l2': correctness,
        'plans': {label: {'uploads': app.nb_axis_upload, 'axis_split': app.axis_split.tolist()}
                  for label, app in plans.items()},
        'protocol': {
            'included': ['FFT', 'prepared-spectrum multiplication', 'IFFT'],
            'excluded': ['allocations', 'plan creation', 'spectrum preparation/permutation',
                         'input reset', 'correctness checks', 'warm-up'],
            'filter': 'dense complex unit-modulus spectrum to keep repeated applications bounded',
            'order': ['normal', 'disabled', 'disabled', 'normal'],
            'reset': 'once before warm-up per block; no resets inside timed execution',
            'monitor_interval_seconds': 0.1,
            'telemetry_window': 'omit first 1 second and final 0.2 seconds of each block',
        }, 'blocks': [],
    }
    monitor = Monitor(uuid)
    monitor.thread.start()
    try:
        for label in result['protocol']['order']:
            cp.copyto(buffer, x)
            warm_start, warm_end = cp.cuda.Event(), cp.cuda.Event()
            warm_start.record()
            for _ in range(20):
                convolution(label)
            warm_end.record()
            warm_end.synchronize()
            estimated_ms = cp.cuda.get_elapsed_time(warm_start, warm_end) / 20
            count = max(20, math.ceil(args.seconds_per_block * 1000 / estimated_ms))
            events = [cp.cuda.Event() for _ in range(count + 1)]
            t0 = time.monotonic()
            events[0].record()
            for iteration in range(count):
                convolution(label)
                events[iteration + 1].record()
            events[-1].synchronize()
            t1 = time.monotonic()
            times = [cp.cuda.get_elapsed_time(events[i], events[i + 1]) for i in range(count)]
            block = {'label': label, 'iterations': count, 'start_monotonic': t0,
                     'end_monotonic': t1, 'median_ms': float(np.median(times)),
                     'mean_ms': float(np.mean(times)), 'samples_ms': times}
            result['blocks'].append(block)
            print(f'GPU {args.gpu}: {label}, {count} convolutions, median {block["median_ms"]:.3f} ms',
                  flush=True)
        # Check that repeated application did not produce NaNs/infinities or underflow.
        result['final_sample_max_abs'] = float(cp.abs(buffer.ravel()[::4096]).max())
        assert math.isfinite(result['final_sample_max_abs']) and result['final_sample_max_abs'] > 0
    finally:
        monitor.close()
    result['telemetry'] = monitor.rows
    result['telemetry_errors'] = monitor.errors
    for block in result['blocks']:
        samples = [row for row in monitor.rows
                   if block['start_monotonic'] + 1 <= row['monotonic'] <= block['end_monotonic'] - 0.2]
        if not samples:
            raise RuntimeError('No steady-state telemetry samples')
        block['telemetry'] = {'samples': len(samples)}
        for metric in ('gpu_percent', 'memory_busy_percent', 'power_w', 'sm_clock_mhz', 'memory_used_bytes'):
            values = [row[metric] for row in samples]
            block['telemetry'][metric] = {'min': min(values), 'mean': float(np.mean(values)),
                                          'max': max(values)}
        block['telemetry']['fraction_gpu_at_least_99'] = float(np.mean(
            [row['gpu_percent'] >= 99 for row in samples]))
    result['summary'] = {}
    for label in plans:
        blocks = [block for block in result['blocks'] if block['label'] == label]
        result['summary'][label] = {
            'median_ms': float(np.median([t for block in blocks for t in block['samples_ms']])),
            'telemetry_blocks': [block['telemetry'] for block in blocks],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
