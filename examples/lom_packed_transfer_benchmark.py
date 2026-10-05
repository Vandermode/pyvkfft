"""Isolated LOM ASM/RSC packed-transfer forward/gradient memory measurements.

Run with LOM and the proxy's compact/scaled-transfer modules on PYTHONPATH.
First --prepare one physical transfer, then benchmark --storage dense/packed
in fresh processes on the same GPU. Dense storage decodes the SAME quantized
table, separating FFT accuracy/memory from the pre-existing quantization loss.
Input generation and reference validation are excluded from measured execution
memory; transfer preparation runs in its own process.
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

from functional.propagation import (
    PropagationConfig, complex_field_propagation_function_factory,
    propagation_constant_factory,
)
from neuraldoe_proxy.compact_transfer import pack_complex_values
from neuraldoe_proxy.scaled_transfer import ScaledTransfer, pack_scaled_transfer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--method', choices=('asm', 'rsc'), required=True)
    parser.add_argument('--size', type=int, default=18000)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--backend', choices=('vkfft', 'jax'), default='vkfft')
    parser.add_argument('--storage', choices=('dense', 'packed'), default='packed')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--grouped-batch', type=int, default=0)
    parser.add_argument('--tile-columns', type=int, default=0)
    args = parser.parse_args()
    if args.size < 8 or args.size % 2 or args.samples < 1:
        parser.error('Use a positive even size >=8 and at least one sample')
    jax.config.update('jax_enable_x64', True)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise RuntimeError('Expose one CUDA GPU')
    mesh = MeshShardingHelper([1]*6, ['zo', 'wvl', 'angle', 'zi', 'polar', 'tp'])
    cfg = PropagationConfig(
        nx=args.size, ny=args.size, dx=.2e-6, dy=.2e-6, wvl=520.6e-9,
        reference_plane_to_sensor_distance=.01008, theta=np.arcsin(.7), phi=0.,
        sensor_center_mode='on_axis', method=args.method, pad_factor=1,
        aperture_type='rectangular', spatial_wave_fields='scalar_demodulated',
        propagation_backend=args.backend, vkfft_grouped_batch=args.grouped_batch,
    )
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        if (directory/'prepared.json').exists():
            raise ValueError('Use a fresh preparation directory')
        start = time.perf_counter()
        h, _, post, _ = propagation_constant_factory(mesh, cfg, return_values=True)
        assert post.shape == (1,)*7
        if args.method == 'asm':
            packed, scale = pack_complex_values(h), jnp.asarray(1/32767, jnp.float32)
        else:
            encoded = pack_scaled_transfer(h)
            packed, scale = encoded.payload, encoded.scale
        payload = np.asarray(packed)
        scale_host = np.asarray(scale)
        np.save(directory/'payload.npy', payload)
        np.save(directory/'scale.npy', scale_host)
        # Decode in host chunks, avoiding a second full GPU table during prep.
        dense = np.lib.format.open_memmap(directory/'decoded.npy', mode='w+',
                                         dtype=np.complex64, shape=payload.shape)
        raw = payload.reshape(2*args.size, 2*args.size)
        target = dense.reshape(raw.shape)
        original_host = np.asarray(h).reshape(raw.shape)
        quantization_error = quantization_norm = 0.
        for first in range(0, raw.shape[0], 128):
            bits = raw[first:first+128]
            re = (bits & np.uint32(65535)).astype(np.uint16).view(np.int16).astype(np.float32)
            im = (bits >> np.uint32(16)).astype(np.uint16).view(np.int16).astype(np.float32)
            decoded = (re + np.complex64(1j)*im)*np.float32(scale_host.squeeze())
            target[first:first+128] = decoded
            original = original_host[first:first+128]
            quantization_error += float(np.sum(np.abs(decoded-original)**2, dtype=np.float64))
            quantization_norm += float(np.sum(np.abs(original)**2, dtype=np.float64))
        dense.flush()
        write_json(directory/'prepared.json', dict(method=args.method, size=args.size,
            payload_bytes=payload.nbytes, dense_bytes=dense.nbytes,
            scale=float(scale_host.squeeze()),
            packed_sha256=hashlib.sha256(payload.tobytes()).hexdigest(),
            transfer_quantization_relative_l2=np.sqrt(quantization_error/max(quantization_norm, 1e-300)),
            seconds=time.perf_counter()-start,
            geometry='0.2 um pitch, 520.6 nm wavelength, z=10.08 mm, NA=0.70 incident, on-axis sensor, Hann ASM window'))
        print('prepared', directory, flush=True)
        return

    prepared = json.loads((directory/'prepared.json').read_text())
    assert prepared['method'] == args.method and prepared['size'] == args.size
    packed = args.storage == 'packed'
    cfg = replace(cfg, vkfft_tile_columns=args.tile_columns, transfer_storage=(
        ('snorm16_compact' if args.method == 'asm' else 'snorm16_scaled') if packed else 'complex64'))
    data = jax.device_put(np.load(directory/('payload.npy' if packed else 'decoded.npy'), mmap_mode='r'))
    transfer = (ScaledTransfer(data, jax.device_put(np.load(directory/'scale.npy')))
                if packed and args.method == 'rsc' else data)
    post = jnp.ones((1,)*7, jnp.complex64)
    n = args.size

    @jax.jit
    def initialize(seed):
        index = jnp.arange(n*n, dtype=jnp.uint32).reshape((1,)*5+(n, n)) + seed
        bits = (index ^ (index >> 16))*jnp.uint32(0x7feb352d)
        bits = (bits ^ (bits >> 15))*jnp.uint32(0x846ca68b)
        bits ^= bits >> 16
        return (bits & 65535).astype(jnp.float32)*np.float32(2*np.pi/65536)

    phase = initialize(jnp.asarray(17, jnp.uint32))
    jax.block_until_ready((phase, transfer))

    def make_function(options):
        forward = complex_field_propagation_function_factory(mesh, options)
        def objective(design, h):
            output = forward(jnp.exp(1j*design), h, post)
            return jnp.mean(jnp.real(output*jnp.conj(output))), output
        return jax.jit(jax.value_and_grad(objective, has_aux=True))

    function = make_function(cfg)
    compiled = function.lower(phase, transfer).compile()
    analysis = compiled.memory_analysis()
    compiler_memory = {name: int(getattr(analysis, name)) for name in (
        'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
    estimate = sum(compiler_memory[name] for name in (
        'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes')) - compiler_memory['alias_size_in_bytes']
    limit = jax.devices()[0].memory_stats().get('bytes_limit', 0)
    if limit and estimate > .85*limit:
        raise MemoryError(f'Compiler estimate {estimate} exceeds 85% of {limit}')
    before = jax.devices()[0].memory_stats()
    jax.block_until_ready(compiled(phase, transfer))
    times = []
    for index in range(args.samples):
        if index:
            del actual
        start = time.perf_counter()
        actual = jax.block_until_ready(compiled(phase, transfer))
        times.append(1000*(time.perf_counter()-start))
    stats = jax.devices()[0].memory_stats()
    print('measured', args.method, args.backend, args.storage, statistics.median(times), stats, flush=True)
    result = dict(method=args.method, backend=args.backend, storage=cfg.transfer_storage,
        size=n, grouped_batch=args.grouped_batch, tile_columns=args.tile_columns,
        jax_version=jax.__version__, gpu=jax.devices()[0].device_kind,
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        median_ms=statistics.median(times), samples_ms=times, transfer_bytes=transfer.nbytes,
        compiled_memory=compiler_memory, compiler_allocation_estimate_bytes=estimate,
        before_execution_memory=before, execution_memory=stats, loss=float(actual[0][0]),
        transfer_quantization_relative_l2=prepared['transfer_quantization_relative_l2'],
        protocol='Fresh process per case, host-loaded precomputed physical H, one warmup, five synchronized calls; reference after memory sample. Dense H is the same quantized table decoded to complex64. Single-plane outer propagation only.')
    ir = str(function.lower(phase, transfer).compiler_ir())
    if args.backend == 'vkfft':
        assert ('@pyvkfft_streamed_v' if args.tile_columns else '@pyvkfft_asm_v3') in ir and 'stablehlo.fft' not in ir
        if packed:
            assert 'ui32' in ir and 'stablehlo.shift_right' not in ir
        # Same packed H through the independently decoded JAX/cuFFT pipeline.
        reference = jax.block_until_ready(make_function(replace(cfg, propagation_backend='jax', vkfft_tile_columns=0))(phase, transfer))
        def relative(a, b):
            a, b = np.asarray(a).reshape(n, n), np.asarray(b).reshape(n, n)
            numerator = denominator = 0.
            for first in range(0, n, 128):
                aa, bb = a[first:first+128], b[first:first+128]
                numerator += float(np.sum(np.abs(aa-bb)**2, dtype=np.float64))
                denominator += float(np.sum(np.abs(bb)**2, dtype=np.float64))
            return float(np.sqrt(numerator/max(denominator, 1e-300)))
        result['field_relative_l2'] = relative(actual[0][1], reference[0][1])
        result['gradient_relative_l2'] = relative(actual[1], reference[1])
        assert result['field_relative_l2'] < 2e-5, result
        assert result['gradient_relative_l2'] < 3e-5, result
        from pyvkfft.jax_asm import cache_info, streamed_cache_info
        result['plans'] = streamed_cache_info() if args.tile_columns else cache_info()
    source = Path(__file__)
    result['benchmark_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    suffix = f'-tile{args.tile_columns}' if args.tile_columns else ''
    write_json(directory/f'{args.backend}-{args.storage}{suffix}.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
