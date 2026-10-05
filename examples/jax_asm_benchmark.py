"""Validate and time a JIT-compiled phase-design loss and its gradient.

Select one GPU with CUDA_VISIBLE_DEVICES before running. Use fresh processes for
each backend to compare compiler memory estimates without retained allocator
storage from another executable. The nonsymmetric complex H describes an exact
two-tap spatial operator, so full-size forward/gradient accuracy can be checked
independently without keeping a second FFT implementation's workspace alive.

python examples/jax_asm_benchmark.py --backend vkfft --shape 18000 18000 --output result.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('vkfft', 'jax'), required=True)
    parser.add_argument('--shape', type=int, nargs=2, default=(512, 512))
    parser.add_argument('--grouped-batch', type=int, default=32)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    h, w = args.shape
    if min(h, w) < 8 or args.samples < 1:
        parser.error('Dimensions must be >=8 and samples positive')
    jax.config.update('jax_enable_x64', True)
    if jax.devices()[0].platform != 'gpu':
        raise RuntimeError('CUDA GPU required')
    from pyvkfft.jax_asm import convolve, cache_info

    @jax.jit
    def initialize(seed):
        index = jnp.arange(h*w, dtype=jnp.uint32).reshape(h, w) + seed
        bits = (index ^ (index >> 16)) * jnp.uint32(0x7feb352d)
        bits = (bits ^ (bits >> 15)) * jnp.uint32(0x846ca68b)
        bits ^= bits >> 16
        phase = (bits & 65535).astype(jnp.float32) * np.float32(2*np.pi/65536)
        return phase

    @jax.jit
    def make_transfer(shift_y, shift_x):
        fy = jnp.fft.fftfreq(2*h)[:, None]
        fx = jnp.fft.fftfreq(2*w)[None, :]
        return (1 + .25j*jnp.exp(-2j*jnp.pi*(shift_y*fy+shift_x*fx))).astype(jnp.complex64)

    phase = initialize(jnp.asarray(17, jnp.uint32))
    transfer = make_transfer(jnp.asarray(3., jnp.float64), jnp.asarray(-5., jnp.float64))
    jax.block_until_ready((phase, transfer))

    def propagate(value, table):
        if args.backend == 'vkfft':
            return convolve(value, table, grouped_batch=args.grouped_batch)
        padded = jnp.pad(value, ((h//2, h-h//2), (w//2, w-w//2)))
        full = jnp.fft.ifft2(jnp.fft.fft2(padded)*table)
        return full[h//2:h//2+h, w//2:w//2+w]

    def objective(design, table):
        field = jnp.exp(1j*design)
        output = propagate(field, table)
        return jnp.mean(jnp.abs(output)**2), output

    function = jax.jit(jax.value_and_grad(objective, has_aux=True))
    compiled = function.lower(phase, transfer).compile()
    memory = compiled.memory_analysis()
    memory_info = {name: getattr(memory, name) for name in (
        'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes',
        'alias_size_in_bytes')}
    compiler_peak = (memory.argument_size_in_bytes + memory.output_size_in_bytes
                     + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
    available = jax.devices()[0].memory_stats().get('bytes_limit', 0)
    if available and compiler_peak > .8*available:
        raise MemoryError(f'Compiler allocation estimate {compiler_peak} exceeds 80% of device limit {available}')
    print('compiled', args.backend, memory_info, flush=True)
    jax.block_until_ready(compiled(phase, transfer))
    timings = []
    for index in range(args.samples):
        if index:
            del actual
        start = time.perf_counter()
        actual = jax.block_until_ready(compiled(phase, transfer))
        timings.append(1000*(time.perf_counter()-start))
    execution_stats = jax.devices()[0].memory_stats()

    def stencil_objective(design):
        field = jnp.exp(1j*design)
        shifted = jnp.pad(field[:-3, 5:], ((3, 0), (0, 5)))
        output = field + .25j*shifted
        return jnp.mean(jnp.abs(output)**2), output

    expected = jax.block_until_ready(jax.jit(jax.value_and_grad(stencil_objective, has_aux=True))(phase))

    @jax.jit
    def chunk_error(a, b):
        return (jnp.sum(jnp.abs(a-b)**2, dtype=jnp.float64),
                jnp.sum(jnp.abs(b)**2, dtype=jnp.float64),
                jnp.max(jnp.abs(a-b)), jnp.max(jnp.abs(b)))

    def relative(a, b):
        error = norm = maximum = peak = 0.
        for first in range(0, h, 128):
            values = chunk_error(a[first:first+128], b[first:first+128])
            n, d, m, p = map(float, values)
            error += n
            norm += d
            maximum, peak = max(maximum, m), max(peak, p)
        return dict(relative_l2=np.sqrt(error/max(norm, 1e-300)),
                    relative_max=maximum/max(peak, 1e-300))

    forward_error = relative(actual[0][1], expected[0][1])
    gradient_error = relative(actual[1], expected[1])
    assert forward_error['relative_l2'] < 2e-5, forward_error
    assert gradient_error['relative_l2'] < 2e-5, gradient_error
    np.testing.assert_allclose(actual[0][0], expected[0][0], rtol=2e-5)
    hlo = str(function.lower(phase, transfer).compiler_ir())
    if args.backend == 'vkfft':
        assert '@pyvkfft_asm_v3' in hlo and 'stablehlo.fft' not in hlo
    result = dict(backend=args.backend, shape=args.shape, dtype='complex64',
                  grouped_batch=args.grouped_batch, jax_version=jax.__version__,
                  gpu=str(jax.devices()[0]), gpu_kind=jax.devices()[0].device_kind,
                  gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  samples_ms=timings, median_ms=statistics.median(timings),
                  memory_analysis=memory_info, compiler_allocation_estimate_bytes=compiler_peak,
                  execution_memory_stats=execution_stats,
                  forward_error=forward_error, phase_gradient_error=gradient_error,
                  loss=float(actual[0][0]), expected_loss=float(expected[0][0]),
                  native_plan_info=cache_info() if args.backend == 'vkfft' else [],
                  protocol='Fresh process, one compilation/warm-up, synchronized full loss+field+phase-gradient calls. '
                           'Memory stats captured before independent stencil validation. '
                           'Timings are a smoke benchmark, not a sustained confidence-interval study.',
                  validation='Full nonconstant 2-D phase field; asymmetric complex two-tap transfer; '
                             'independent spatial stencil verifies complete forward and phase gradient.')
    sources = [Path(__file__)]
    import pyvkfft.jax_asm as implementation
    sources.append(Path(implementation.__file__))
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('median_ms', 'compiler_allocation_estimate_bytes',
                                                 'forward_error', 'phase_gradient_error')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
