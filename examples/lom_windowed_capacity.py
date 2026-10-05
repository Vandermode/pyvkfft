"""Execute large physical ASM with sampled, independent direct-DFT validation.

The timed graph returns a selected output's real value, the full propagated
field, and the full phase gradient for a nonconstant phase design. A direct
double-precision inverse DFT of the compact physical transfer independently
checks sampled gradients. A separate donated impulse propagation checks
sampled forward values. No full padded reference FFT plane is allocated.
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
from scalax.sharding import MeshShardingHelper
from functional.propagation import PropagationConfig, propagation_constant_factory, propagation_function_factory
from pyvkfft.jax_asm import PackedTransferWindow, convolve_packed, streamed_cache_info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size', type=int, required=True)
    p.add_argument('--tile', type=int, default=128)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert a.size > 1 and a.size % 2 == 0 and a.samples > 0
    assert jax.config.x64_enabled, 'Enable JAX x64 for the independent reference, while fields remain complex64'
    n = a.size
    mesh = MeshShardingHelper([1]*6, ['zo', 'wvl', 'angle', 'zi', 'polar', 'tp'])
    cfg = PropagationConfig(nx=n, ny=n, dx=.0036/n, dy=.0036/n, wvl=520.6e-9,
                            reference_plane_to_sensor_distance=.01008, theta=0., phi=0.,
                            sensor_center_mode='on_axis', aperture_type='rectangular',
                            spatial_wave_fields='scalar_demodulated', transfer_storage='snorm16_compact',
                            propagation_backend='vkfft', vkfft_tile_columns=a.tile, vkfft_transfer_layout='window')
    table, incident, post, ratio = propagation_constant_factory(mesh, cfg, return_values=True)
    forward = propagation_function_factory(mesh, cfg)[0]

    def objective(phase, transfer, target):
        field = forward(phase, incident, transfer, post)
        value = jax.lax.dynamic_slice(field.reshape(n, n), (target[0], target[1]), (1, 1))[0, 0].real
        return value, field

    fn = jax.jit(jax.value_and_grad(objective, has_aux=True))
    spec = jax.ShapeDtypeStruct((1,)*5+(n, n), jnp.float32)
    target = jnp.asarray((n//3, n//5), jnp.int32)
    compiled = fn.lower(spec, table, target).compile()
    m = compiled.memory_analysis()
    memory = {k: int(getattr(m, k)) for k in ('argument_size_in_bytes', 'output_size_in_bytes',
                                            'temp_size_in_bytes', 'alias_size_in_bytes')}
    estimate = memory['argument_size_in_bytes']+memory['output_size_in_bytes']+memory['temp_size_in_bytes']-memory['alias_size_in_bytes']
    device = jax.devices()[0]
    record = dict(size=n, logical_fft_shape=[2*n, 2*n], transfer_shape=list(table.shape),
                  transfer_origin=list(table.origin), transfer_bytes=int(table.nbytes),
                  compiled_memory=memory, compiler_allocation_estimate_bytes=estimate,
                  allocator_limit=device.memory_stats()['bytes_limit'], executed=not a.plan_only,
                  gpu=device.device_kind, jax_version=jax.__version__,
                  physical_aperture_m=.0036, wavelength_m=520.6e-9, distance_m=.01008,
                  passband_ratio=float(ratio), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(record), flush=True)
    if a.plan_only:
        a.output.write_text(json.dumps(record, indent=2)+'\n')
        return
    assert estimate < record['allocator_limit']-512*2**20, 'Insufficient allocator headroom'

    @jax.jit
    def initialize(seed):
        i = jnp.arange(n, dtype=jnp.uint32)[:, None]*jnp.uint32(n)+jnp.arange(n, dtype=jnp.uint32)[None, :]+seed
        v = (i^(i >> 16))*jnp.uint32(0x7feb352d)
        v = (v^(v >> 15))*jnp.uint32(0x846ca68b)
        v ^= v >> 16
        return ((v & 65535).astype(jnp.float32)*np.float32(2*np.pi/65536)).reshape(spec.shape)

    phase = initialize(jnp.asarray(17, jnp.uint32))
    result = jax.block_until_ready(compiled(phase, table, target))
    samples = []
    for _ in range(a.samples):
        del result
        start = time.perf_counter()
        result = jax.block_until_ready(compiled(phase, table, target))
        samples.append(1000*(time.perf_counter()-start))
    measured = device.memory_stats()
    print(json.dumps(dict(median_ms=statistics.median(samples), execution_memory=measured)), flush=True)
    rows = np.unique(np.r_[np.linspace(0, n-1, 24, dtype=np.int64), 0, 1, n-2, n-1, n//3, n//3+1])
    cols = np.unique(np.r_[np.linspace(0, n-1, 24, dtype=np.int64), 0, 1, n-2, n-1, n//5, n//5+1])

    @jax.jit
    def sample(value):
        return value.reshape(n, n)[jnp.asarray(rows)[:, None], jnp.asarray(cols)[None, :]]

    phase_samples = np.asarray(sample(phase))
    actual_gradient = np.asarray(sample(result[1]))
    del phase, result
    kh, kw = table.shape[-2:]
    oy, ox = table.origin

    @jax.jit
    def direct_dft(payload, lag_y, lag_x):
        bits = payload.reshape(kh, kw)
        real = (bits & 65535).astype(jnp.int32)
        imag = (bits >> 16).astype(jnp.int32)
        real = jnp.where(real >= 32768, real-65536, real).astype(jnp.float32)*np.float32(1/32767)
        imag = jnp.where(imag >= 32768, imag-65536, imag).astype(jnp.float32)*np.float32(1/32767)
        coefficient = jax.lax.complex(real, imag).astype(jnp.complex128)
        ky = (jnp.arange(kh, dtype=jnp.int64)+oy) % (2*n)
        kx = (jnp.arange(kw, dtype=jnp.int64)+ox) % (2*n)
        ky = jnp.where(ky >= n, ky-2*n, ky).astype(jnp.float64)
        kx = jnp.where(kx >= n, kx-2*n, kx).astype(jnp.float64)
        ey = jnp.exp((2j*np.pi/(2*n))*lag_y[:, None]*ky[None, :])
        ex = jnp.exp((2j*np.pi/(2*n))*kx[:, None]*lag_x[None, :])
        return jnp.matmul(jnp.matmul(ey, coefficient, precision=jax.lax.Precision.HIGHEST),
                          ex, precision=jax.lax.Precision.HIGHEST)/float((2*n)**2)

    reference_backward = np.asarray(direct_dft(table.payload, jnp.asarray(n//3-rows, jnp.float64),
                                               jnp.asarray(n//5-cols, jnp.float64)))
    expected_gradient = -(np.exp(1j*phase_samples)*np.float32(incident.reshape(()))*reference_backward).imag
    gradient_error = float(np.linalg.norm(actual_gradient-expected_gradient)/np.linalg.norm(expected_gradient))
    assert gradient_error < 3e-5, gradient_error
    print('GRADIENT_VALIDATED', gradient_error, flush=True)

    @jax.jit
    def impulse(position):
        return ((jnp.arange(n)[:, None] == position[0]) &
                (jnp.arange(n)[None, :] == position[1])).astype(jnp.complex64)

    x = impulse(target)
    ptr = x.unsafe_buffer_pointer()
    impulse_table = PackedTransferWindow(table.payload.reshape(kh, kw), table.full_shape, table.origin)
    output = jax.jit(lambda field, h: convolve_packed(field, h, tile_columns=a.tile),
                     donate_argnums=(0,))(x, impulse_table)
    output.block_until_ready()
    assert x.is_deleted() and output.unsafe_buffer_pointer() == ptr
    actual_forward = np.asarray(sample(output))
    del output
    reference_forward = np.asarray(direct_dft(table.payload, jnp.asarray(rows-n//3, jnp.float64),
                                              jnp.asarray(cols-n//5, jnp.float64)))
    field_error = float(np.linalg.norm(actual_forward-reference_forward)/np.linalg.norm(reference_forward))
    assert field_error < 3e-5, field_error
    record.update(samples_ms=samples, median_ms=statistics.median(samples), execution_memory=measured,
                  gradient_sample_relative_l2=gradient_error, impulse_sample_relative_l2=field_error,
                  sample_rows=rows.tolist(), sample_columns=cols.tolist(),
                  plans=streamed_cache_info(), donation_reused_pointer=True,
                  protocol='Physical Hann-bandlimited ASM, directly prepared packed window. Timed graph returns a single-pixel linear loss, '
                           'full field and full phase gradient for a nonconstant dense phase design. Independent complex128 direct inverse DFT '
                           'of packed H validates the sampled phase gradient, followed by sampled validation of a separate donated impulse forward. '
                           'This is sampled large-grid validation; complete field/gradient comparisons are performed at smaller sizes. '
                           'The reported execution peak precedes validation; no optimizer state or extra propagation planes.')
    a.output.write_text(json.dumps(record, indent=2)+'\n')
    print('PASS', gradient_error, field_error, flush=True)


if __name__ == '__main__':
    main()
