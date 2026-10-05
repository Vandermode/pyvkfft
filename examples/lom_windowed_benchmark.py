"""Measure physical LOM ASM with full or directly prepared windowed packed H.

Use a fresh process per configuration. The default optical aperture is 3.6 mm
at every resolution, with 10.08 mm propagation distance and a Hann band limit.
The objective is weighted output energy with a nonconstant phase design.
"""
import argparse
from dataclasses import replace
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
import functional.propagation as propagation
from pyvkfft import jax_asm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--layout', choices=('full', 'window', 'quadrant'), default='full')
    parser.add_argument('--loss-only', action='store_true')
    parser.add_argument('--tile', type=int, default=128)
    parser.add_argument('--aperture', type=float, default=.0036)
    parser.add_argument('--pitch', type=float)
    parser.add_argument('--distance', type=float, default=.01008)
    parser.add_argument('--theta', type=float, default=0.)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--validate', action='store_true', help='Compare full field/gradient to full-table execution after measuring memory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.size < 2 or args.size % 2 or args.samples < 1 or args.blocks < 1 or args.seconds < 0:
        parser.error('Require positive even size and positive sample/block counts')
    if args.validate and args.layout == 'full':
        parser.error('--validate compares compressed execution to the full-table path; select window or quadrant')
    n = args.size
    mesh = MeshShardingHelper([1]*6, ['zo', 'wvl', 'angle', 'zi', 'polar', 'tp'])
    cfg = propagation.PropagationConfig(
        nx=n, ny=n, dx=args.pitch or args.aperture/n, dy=args.pitch or args.aperture/n,
        wvl=520.6e-9, reference_plane_to_sensor_distance=args.distance,
        theta=args.theta, sensor_center_mode='on_axis', bandlimit_window='hann',
        aperture_type='rectangular', spatial_wave_fields='scalar_demodulated',
        propagation_backend='vkfft', transfer_storage='snorm16_compact',
        vkfft_tile_columns=args.tile, vkfft_transfer_layout=args.layout)
    begin = time.perf_counter()
    table, incident, post, passband = propagation.propagation_constant_factory(mesh, cfg, return_values=True)
    jax.block_until_ready((table, incident, post, passband))
    preparation_seconds = time.perf_counter()-begin
    device = jax.devices()[0]
    after_preparation = device.memory_stats()

    @jax.jit
    def initialize(seed):
        i = jnp.arange(n, dtype=jnp.uint32)[:, None]*jnp.uint32(n)+jnp.arange(n, dtype=jnp.uint32)[None, :]+seed
        v = (i^(i >> 16))*jnp.uint32(0x7feb352d)
        v = (v^(v >> 15))*jnp.uint32(0x846ca68b)
        v ^= v >> 16
        return ((v & 65535).astype(jnp.float32)*np.float32(2*np.pi/65536)).reshape((1,)*5+(n, n))

    phase = initialize(jnp.asarray(17, jnp.uint32))
    weights = .3+.7*jnp.cos(jnp.arange(n, dtype=jnp.float32)*np.float32(2*np.pi/n))**2

    def operation(options):
        forward = propagation.propagation_function_factory(mesh, options)[0]
        def objective(value, transfer):
            field = forward(value, incident, transfer, post)
            loss = jnp.sum(jnp.abs(field)**2*weights)
            return loss if args.loss_only else (loss, field)
        return jax.jit(jax.value_and_grad(objective, has_aux=not args.loss_only))

    function = operation(cfg)
    compiled = function.lower(phase, table).compile()
    m = compiled.memory_analysis()
    memory = {k: int(getattr(m, k)) for k in ('argument_size_in_bytes', 'output_size_in_bytes',
                                            'temp_size_in_bytes', 'alias_size_in_bytes')}
    estimate = memory['argument_size_in_bytes']+memory['output_size_in_bytes']+memory['temp_size_in_bytes']-memory['alias_size_in_bytes']
    assert estimate < device.memory_stats()['bytes_limit']-512*2**20, 'Insufficient allocator headroom'
    result = jax.block_until_ready(compiled(phase, table))
    blocks = []
    for _ in range(args.blocks):
        samples = []
        start_block = time.perf_counter()
        while len(samples) < args.samples or time.perf_counter()-start_block < args.seconds:
            del result
            begin = time.perf_counter()
            result = jax.block_until_ready(compiled(phase, table))
            samples.append(1000*(time.perf_counter()-begin))
        blocks.append(samples)
    measured = device.memory_stats()
    summary = dict(size=n, layout=args.layout, loss_only=args.loss_only, tile=args.tile,
                   pitch=cfg.dx, aperture=cfg.dx*n, distance=args.distance, theta=args.theta,
                   wavelength=520.6e-9, bandlimit='hann',
                   transfer_shape=list(table.shape), transfer_bytes=int(table.nbytes),
                   transfer_origin=list(table.origin) if args.layout == 'window' else None,
                   passband_ratio=float(passband), preparation_seconds=preparation_seconds,
                   after_preparation_memory=after_preparation, compiled_memory=memory,
                   compiler_allocation_estimate_bytes=estimate, execution_memory=measured,
                   blocks_ms=blocks, median_ms=statistics.median([v for block in blocks for v in block]),
                   plans=jax_asm.streamed_cache_info(), jax_version=jax.__version__,
                   gpu=device.device_kind, visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   protocol='Fresh process; actual shared LOM phase factory; directly prepared physical packed ASM H; '
                            'nonconstant phase and weighted output energy; synchronized blocks after warmup. '
                            'Execution peak recorded before optional full-table validation; includes earlier H preparation high water.')
    print(json.dumps({k: summary[k] for k in ('size', 'layout', 'loss_only', 'transfer_bytes', 'median_ms', 'execution_memory')}), flush=True)
    if args.validate:
        baseline = replace(cfg, vkfft_transfer_layout='full')
        full_h, _, _, full_passband = propagation.propagation_constant_factory(mesh, baseline, return_values=True)
        expected = jax.block_until_ready(operation(baseline)(phase, full_h))
        @jax.jit
        def relative_error(a, b):
            difference = (a-b).astype(jnp.complex128)
            reference = b.astype(jnp.complex128)
            return jnp.sqrt(jnp.sum(jnp.abs(difference)**2)/jnp.maximum(jnp.sum(jnp.abs(reference)**2), 1e-300))
        actual_loss = result[0] if args.loss_only else result[0][0]
        expected_loss = expected[0] if args.loss_only else expected[0][0]
        errors = dict(gradient_relative_l2=float(relative_error(result[1], expected[1])),
                      loss_relative=float(abs(actual_loss-expected_loss)/jnp.maximum(abs(expected_loss), 1e-30)),
                      field_relative_l2=None if args.loss_only else float(relative_error(result[0][1], expected[0][1])))
        assert all(value < 3e-5 for value in errors.values() if value is not None), errors
        np.testing.assert_allclose(passband, full_passband, rtol=3e-6, atol=1e-8)
        summary['errors_vs_full_table'] = errors
        print('VALIDATION', json.dumps(errors), flush=True)
    selected = Path(os.environ.get('PYVKFFT_STREAMED_FFI_LIBRARY', Path(jax_asm.__file__).with_name('libvkfft_streamed_ffi.so')))
    summary['library_sha256'] = hashlib.sha256(selected.read_bytes()).hexdigest()
    summary['source_sha256'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), Path(jax_asm.__file__), Path(propagation.__file__))}
    args.output.write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
