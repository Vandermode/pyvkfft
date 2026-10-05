"""Fresh-process physical ASM / FFT timing and per-device memory measurements."""
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from pyvkfft.jax_distributed import fft2_factory, windowed_asm_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--mode', choices=['fft', 'window', 'dense'], default='window')
    parser.add_argument('--backend', choices=['jax', 'vkfft'], default='vkfft')
    parser.add_argument('--gradient', action='store_true')
    parser.add_argument('--donate', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--unpacked-exchange', action='store_true')
    parser.add_argument('--tile-rows', type=int, default=128)
    parser.add_argument('--tile-columns', type=int, default=64)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--blocks', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    n = args.size
    devices = jax.devices()
    p = len(devices)
    assert n % p == 0 and n % 2 == 0 and args.samples > 0
    assert not (args.donate and args.gradient)
    mesh = Mesh(np.array(devices), ('tp',))
    row = NamedSharding(mesh, P('tp', None))
    replicated = NamedSharding(mesh, P())
    print('IDENTITY', jax.__version__, devices, vars(args), flush=True)
    if args.mode == 'fft':
        transfer = jax.device_put(jnp.zeros((1, 1), jnp.uint32), replicated)
        raw = fft2_factory(mesh, backend=args.backend, tile=args.tile_rows)
        operation = lambda field, payload: raw(field)
        origin = None
        scale = 1.
        passband = None
    else:
        from scalax.sharding import MeshShardingHelper
        from functional.propagation import PropagationConfig, propagation_constant_factory
        helper = MeshShardingHelper([1, 1, 1, 1, 1, p], ['zo', 'wvl', 'angle', 'zi', 'polar', 'tp'])
        cfg = PropagationConfig(nx=n, ny=n, dx=.0036/n, dy=.0036/n, wvl=520.6e-9,
                                reference_plane_to_sensor_distance=.01008,
                                theta=0., sensor_center_mode='on_axis', bandlimit_window='hann',
                                aperture_type='rectangular', spatial_wave_fields='scalar_demodulated',
                                propagation_backend='vkfft', transfer_storage='snorm16_compact',
                                vkfft_transfer_layout='window', vkfft_tile_columns=args.tile_columns)
        table, incident, post, passband = propagation_constant_factory(helper, cfg, return_values=True,
                                                                       H_sharding=None)
        jax.block_until_ready((table, incident, post, passband))
        transfer = jax.device_put(table.payload.reshape(table.shape[-2:]), replicated)
        origin = tuple(table.origin)
        scale = np.float32(1/32767)
        if args.mode == 'window':
            raw = windowed_asm_factory(mesh, origin=origin, backend=args.backend,
                                       tile_rows=args.tile_rows, tile_columns=args.tile_columns,
                                       packed_exchange=not args.unpacked_exchange)
            operation = lambda field, payload: raw(field, payload, scale)
        else:
            forward = fft2_factory(mesh, backend=args.backend, tile=args.tile_rows)
            backward = fft2_factory(mesh, backend=args.backend, inverse=True, tile=args.tile_rows)
            # Decode directly into a column-sharded dense spectrum. The compact
            # argument is retained for identical physical coefficients.
            @jax.jit
            def expand(payload):
                re = (payload & 65535).astype(jnp.int32)
                im = (payload >> 16).astype(jnp.int32)
                values = (jnp.where(re>=32768, re-65536, re)+
                          1j*jnp.where(im>=32768, im-65536, im))*scale
                out = jnp.zeros((2*n, 2*n), jnp.complex64)
                return out.at[(jnp.arange(payload.shape[0])+origin[0])[:, None] % (2*n),
                              (jnp.arange(payload.shape[1])+origin[1])[None, :] % (2*n)].set(values)
            transfer = jax.jit(expand, out_shardings=NamedSharding(mesh, P(None, 'tp')))(transfer)
            def operation(field, payload):
                padded = jnp.pad(field, ((n//2, n//2), (n//2, n//2)))
                return backward(forward(padded)*payload)[n//2:n//2+n, n//2:n//2+n]
    transfer.block_until_ready()
    weights = .3+.7*jnp.cos(jnp.arange(n, dtype=jnp.float32)*np.float32(2*np.pi/n))**2
    if args.gradient:
        field_from_phase = jax.checkpoint(lambda phase: jnp.exp(1j*phase).astype(jnp.complex64)/np.float32(n))
        def objective(phase, payload):
            field = operation(field_from_phase(phase), payload)
            return jnp.sum(jnp.abs(field)**2*weights), field
        function = jax.jit(jax.value_and_grad(objective, has_aux=True),
                           in_shardings=(row, transfer.sharding))
        dtype = jnp.float32
    else:
        function = jax.jit(operation, in_shardings=(row, transfer.sharding),
                           donate_argnums=(0,) if args.donate else ())
        dtype = jnp.complex64
    spec = jax.ShapeDtypeStruct((n, n), dtype, sharding=row)
    begin = time.perf_counter()
    compiled = function.lower(spec, transfer).compile()
    compile_seconds = time.perf_counter()-begin
    mem = compiled.memory_analysis()
    compiled_memory = {name: int(getattr(mem, name)) for name in
                       ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
    estimate = compiled_memory['argument_size_in_bytes']+compiled_memory['output_size_in_bytes']+compiled_memory['temp_size_in_bytes']-compiled_memory['alias_size_in_bytes']
    report = dict(size=n, mode=args.mode, backend=args.backend, gradient=args.gradient,
                  donate=args.donate, device_count=p, devices=[str(d) for d in devices],
                  gpu=devices[0].device_kind, jax=jax.__version__, transfer_shape=list(transfer.shape),
                  transfer_bytes=int(transfer.nbytes), transfer_origin=origin,
                  passband_ratio=None if passband is None else float(passband),
                  compiled_memory=compiled_memory, per_device_compiler_estimate=estimate,
                  compile_seconds=compile_seconds, memory_before_execution=[d.memory_stats() for d in devices],
                  tile_rows=args.tile_rows, tile_columns=args.tile_columns,
                  packed_exchange=not args.unpacked_exchange,
                  phase_checkpoint=args.gradient,
                  job=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print('PLAN', json.dumps(report), flush=True)
    (args.output.with_suffix('.hlo.txt')).write_text(compiled.as_text())
    if not args.plan_only:
        assert estimate < min(d.memory_stats()['bytes_limit'] for d in devices)-1024*2**20, 'Insufficient allocator headroom'
        @partial_jit(row)
        def initialize(seed):
            i = jnp.arange(n, dtype=jnp.uint32)[:, None]*jnp.uint32(n)+jnp.arange(n, dtype=jnp.uint32)[None, :]+seed
            v = (i^(i >> 16))*jnp.uint32(0x7feb352d)
            v = (v^(v >> 15))*jnp.uint32(0x846ca68b)
            v ^= v >> 16
            phase = (v & 65535).astype(jnp.float32)*np.float32(2*np.pi/65536)
            return phase if args.gradient else jnp.exp(1j*phase).astype(jnp.complex64)
        value = initialize(jnp.asarray(17, jnp.uint32)).block_until_ready()
        pointers = [s.data.unsafe_buffer_pointer() for s in value.addressable_shards]
        result = jax.block_until_ready(compiled(value, transfer))
        alias = ([s.data.unsafe_buffer_pointer() for s in result.addressable_shards] == pointers) if args.donate else None
        blocks = []
        for _ in range(args.blocks):
            samples = []
            block_start = time.perf_counter()
            while len(samples)<args.samples or time.perf_counter()-block_start<args.seconds:
                del result
                if args.donate:
                    value = initialize(jnp.asarray(17, jnp.uint32)).block_until_ready()
                begin = time.perf_counter()
                result = jax.block_until_ready(compiled(value, transfer))
                samples.append(1000*(time.perf_counter()-begin))
            blocks.append(samples)
        samples = [x for block in blocks for x in block]
        report.update(samples_ms=samples, blocks_ms=blocks, median_ms=statistics.median(samples),
                      memory_after_execution=[d.memory_stats() for d in devices], donation_reused_pointers=alias)
        if args.gradient:
            report['loss'] = float(result[0][0])
        else:
            report['sample'] = [float(jnp.real(result[n//3, n//5])), float(jnp.imag(result[n//3, n//5]))]
        print('MEASURED', json.dumps({k: report[k] for k in ('median_ms', 'memory_after_execution')}), flush=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(args.output)
    print('ALL_COMPLETED', flush=True)


def partial_jit(sharding):
    return lambda fn: jax.jit(fn, out_shardings=sharding)


if __name__ == '__main__':
    main()
