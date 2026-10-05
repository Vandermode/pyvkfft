"""Investigate VkFFT #205 with deterministic C2C convolution checks.

Run from the project root using the jax environment, for example:
    python examples/issue205_repro.py --gpu 1 --shape 65536 --output /tmp/205.json
Only physical GPUs 1, 2, and 3 may be selected. This script does not change plans
or the installed library. Layout conversion covers one/two-upload radix FFTs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--shape', type=int, nargs='+', required=True)
    parser.add_argument('--dtype', choices=('complex64', 'complex128'), default='complex64')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, help='Use an isolated experimental CUDA library')
    parser.add_argument('--benchmark-repeats', type=int, default=0,
                        help='Time FFT/multiply/IFFT with a precomputed kernel')
    parser.add_argument('--out-of-place', action='store_true',
                        help='Diagnostic only: the proposed patch targets in-place C2C')
    args = parser.parse_args()
    if args.out_of_place and args.benchmark_repeats:
        parser.error('Benchmarking currently supports in-place transforms only')
    if args.benchmark_repeats < 0 or any(n <= 0 for n in args.shape):
        parser.error('Shapes must be positive and benchmark repeats nonnegative')
    uuid = subprocess.check_output([
        'nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid', '--format=csv,noheader'
    ], text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid

    import cupy as cp
    import numpy as np
    if args.library:
        import ctypes
        import pyvkfft.base
        original_loader = pyvkfft.base.load_library
        pyvkfft.base.load_library = lambda name: (
            ctypes.CDLL(str(args.library.resolve())) if name == '_vkfft_cuda'
            else original_loader(name))
    from pyvkfft.cuda import VkFFTApp, _vkfft_cuda, vkfft_version

    cp.cuda.Device(0).use()
    shape = tuple(args.shape)
    dtype = np.dtype(args.dtype)
    rng = cp.random.RandomState(205)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    k = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    fx, fk = cp.fft.fftn(x), cp.fft.fftn(k)
    reference = cp.fft.ifftn(fx * fk)
    tolerance = 2e-5 if dtype == np.complex64 else 1e-11

    def transform(app, array, inverse=False):
        operation = app.ifft if inverse else app.fft
        return operation(array) if app.inplace else operation(array, cp.empty_like(array))

    def error(actual, expected):
        delta = cp.abs(actual - expected).astype(cp.float64)
        magnitude = cp.abs(expected).astype(cp.float64)
        relative = float(cp.sqrt(cp.sum(delta**2) / cp.sum(magnitude**2)))
        return {'relative_l2': relative,
                'max_abs_over_reference_max': float(delta.max() / magnitude.max()),
                'passes': bool(relative < tolerance)}

    def layout(array, app, inverse=False):
        """Pack/unpack the two-factor frequency permutation, separately per axis."""
        if any(app.use_bluestein_fft) or max(app.nb_axis_upload) > 2:
            raise ValueError('Layout helper supports only one/two-upload radix FFTs')
        result = array
        for axis in range(len(shape)):
            vk_axis = len(shape) - 1 - axis
            if app.nb_axis_upload[vk_axis] == 1:
                continue
            factors = tuple(int(v) for v in app.axis_split[vk_axis, :2])
            if inverse:
                factors = factors[::-1]
            expanded = result.shape[:axis] + factors + result.shape[axis + 1:]
            result = cp.ascontiguousarray(
                result.reshape(expanded).swapaxes(axis, axis + 1)).reshape(shape)
        return result

    result = {
        'gpu_index': args.gpu, 'gpu_uuid': uuid, 'shape': shape, 'dtype': str(dtype),
        'inplace': not args.out_of_place,
        'cupy': cp.__version__, 'vkfft': vkfft_version(),
        'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
        'library': _vkfft_cuda._name,
        'library_sha256': hashlib.sha256(Path(_vkfft_cuda._name).read_bytes()).hexdigest(),
        'tolerance_relative_l2': tolerance, 'checks': {}, 'plans': {},
    }
    checks = result['checks']
    plans = {}
    for disabled in (False, True):
        label = 'disabled' if disabled else 'normal'
        app = plans[label] = VkFFTApp(shape, dtype.type, inplace=not args.out_of_place,
                                      disableReorderFourStep=disabled)
        result['plans'][label] = {
            'uploads': app.nb_axis_upload, 'axis_split': app.axis_split.tolist(),
            'bluestein': app.use_bluestein_fft,
        }
        spectrum = transform(app, x.copy())
        checks[label + '_forward_natural'] = error(spectrum, fx)
        if disabled:
            checks['disabled_forward_packed'] = error(spectrum, layout(fx, app))
            inverse_natural = transform(app, fx.copy(), inverse=True)
            checks['disabled_inverse_natural_to_packed'] = error(inverse_natural, layout(x, app))
        checks[label + '_roundtrip'] = error(transform(app, spectrum.copy(), inverse=True), x)
        kernel = transform(app, k.copy())
        product = spectrum * kernel
        checks[label + '_convolution'] = error(transform(app, product.copy(), inverse=True), reference)
        if disabled:
            natural_product = layout(product, app, inverse=True)
            checks['unpack_then_normal_inverse'] = error(
                transform(plans['normal'], natural_product.copy(), inverse=True), reference)
            checks['unpack_then_disabled_inverse_then_unpack'] = error(
                layout(transform(app, natural_product.copy(), inverse=True), app, inverse=True), reference)

    native = VkFFTApp(shape, dtype.type, inplace=True, convolve=True)
    result['plans']['native'] = {
        'uploads': native.nb_axis_upload, 'axis_split': native.axis_split.tolist(),
    }
    checks['native_convolution'] = error(
        native.fft(x.copy(), convolve_kernel=layout(fk, native)), reference)
    if args.benchmark_repeats:
        result['timings_ms'] = {}
        prepared = {}
        for label, app in plans.items():
            if not checks[label + '_convolution']['passes']:
                continue
            # Prepare each multiplication spectrum in that plan's output layout.
            # All preparation finishes before warming up or measuring either path.
            prepared[label] = (app, app.fft(k.copy()))
        buffer = cp.empty_like(x)
        start, after_fft, after_multiply, stop = (cp.cuda.Event() for _ in range(4))
        cp.cuda.Stream.null.synchronize()
        labels = list(prepared)
        durations = {label: {stage: [] for stage in ('total', 'fft', 'multiply', 'ifft')}
                     for label in labels}
        warmups = 20
        result['benchmark_protocol'] = {
            'warmups_per_variant': warmups,
            'repeats_per_variant': args.benchmark_repeats,
            'order': 'paired, alternating normal/disabled then disabled/normal',
            'timer': 'CUDA events on the default stream',
            'included': ['forward FFT', 'pointwise multiplication', 'inverse FFT'],
            'excluded': ['plan creation', 'allocations', 'kernel FFT/layout preparation',
                         'input reset', 'warmups', 'correctness validation'],
            'shared_input_work_buffer': True,
        }
        for iteration in range(args.benchmark_repeats + warmups):
            order = labels if iteration % 2 == 0 else labels[::-1]
            for label in order:
                app, kernel = prepared[label]
                # Reset precedes the start event on the same stream, outside timing.
                cp.copyto(buffer, x)
                start.record()
                app.fft(buffer)
                after_fft.record()
                buffer *= kernel
                after_multiply.record()
                app.ifft(buffer)
                stop.record()
                stop.synchronize()
                if iteration >= warmups:
                    for stage, first, last in (
                            ('total', start, stop), ('fft', start, after_fft),
                            ('multiply', after_fft, after_multiply),
                            ('ifft', after_multiply, stop)):
                        durations[label][stage].append(cp.cuda.get_elapsed_time(first, last))
        for label in labels:
            result['timings_ms'][label] = {
                'median': float(np.median(durations[label]['total'])),
                'samples': durations[label]['total'],
                'stages': {stage: {'median': float(np.median(values)),
                                   'p10': float(np.percentile(values, 10)),
                                   'p90': float(np.percentile(values, 90)),
                                   'samples': values}
                           for stage, values in durations[label].items()},
            }
    cp.cuda.Stream.null.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
