"""Large distributed physical ASM with independent sampled direct-DFT checks."""
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
from scalax.sharding import MeshShardingHelper
from functional.propagation import PropagationConfig, propagation_constant_factory
from pyvkfft.jax_distributed import windowed_asm_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--gradient', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--tile-rows', type=int, default=128)
    parser.add_argument('--tile-columns', type=int, default=64)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.x64_enabled
    devices = jax.devices()
    p, n = len(devices), args.size
    assert n % p == 0 and n % 2 == 0
    mesh = Mesh(np.array(devices), ('tp',))
    row, replicated = NamedSharding(mesh, P('tp', None)), NamedSharding(mesh, P())
    # Initialize the communication clique before allocating a very large field.
    # NCCL owns some memory outside XLA's allocator.
    from jax.experimental.shard_map import shard_map
    warmup = jax.jit(shard_map(lambda x: jax.lax.psum(x, 'tp'), mesh=mesh,
                              in_specs=P('tp'), out_specs=P(), check_rep=False))
    warmup(jax.device_put(np.ones(p, np.float32), NamedSharding(mesh, P('tp')))).block_until_ready()
    helper = MeshShardingHelper([1, 1, 1, 1, 1, p], ['zo', 'wvl', 'angle', 'zi', 'polar', 'tp'])
    cfg = PropagationConfig(nx=n, ny=n, dx=.0036/n, dy=.0036/n, wvl=520.6e-9,
                            reference_plane_to_sensor_distance=.01008, theta=0., phi=0.,
                            sensor_center_mode='on_axis', aperture_type='rectangular',
                            spatial_wave_fields='scalar_demodulated', transfer_storage='snorm16_compact',
                            propagation_backend='vkfft', vkfft_tile_columns=args.tile_columns,
                            vkfft_transfer_layout='window')
    table, incident, _, ratio = propagation_constant_factory(helper, cfg, return_values=True, H_sharding=None)
    kh, kw = table.shape[-2:]
    payload = jax.device_put(table.payload.reshape(kh, kw), replicated)
    oy, ox = table.origin
    raw = windowed_asm_factory(mesh, origin=(oy, ox), tile_rows=args.tile_rows, tile_columns=args.tile_columns)
    scale = np.float32(1/32767)
    target = (n//3, n//5)
    modes = ((.37, -.61, 1.), (2.11, 1.29, .3))
    forward = lambda x, h: raw(x, h, scale)
    if args.gradient:
        field_from_phase = jax.checkpoint(lambda phase: jnp.exp(1j*phase).astype(jnp.complex64)/np.float32(n))
        def objective(phase, h):
            field = forward(field_from_phase(phase), h)
            return field[target[0], target[1]].real, field
        fn = jax.jit(jax.value_and_grad(objective, has_aux=True), in_shardings=(row, replicated))
        dtype = jnp.float32
    else:
        fn = jax.jit(forward, in_shardings=(row, replicated), out_shardings=row, donate_argnums=(0,))
        dtype = jnp.complex64
    compiled = fn.lower(jax.ShapeDtypeStruct((n, n), dtype, sharding=row), payload).compile()
    m = compiled.memory_analysis()
    memory = {k: int(getattr(m, k)) for k in ('argument_size_in_bytes', 'output_size_in_bytes',
                                            'temp_size_in_bytes', 'alias_size_in_bytes')}
    estimate = memory['argument_size_in_bytes']+memory['output_size_in_bytes']+memory['temp_size_in_bytes']-memory['alias_size_in_bytes']
    record = dict(size=n, gradient=args.gradient, device_count=p, gpu=devices[0].device_kind,
                  devices=[str(d) for d in devices], jax=jax.__version__, logical_fft_shape=[2*n, 2*n],
                  transfer_shape=[kh, kw], transfer_origin=[oy, ox], transfer_bytes=int(payload.nbytes),
                  per_device_compiler_estimate=estimate, compiled_memory=memory,
                  allocator_limits=[d.memory_stats()['bytes_limit'] for d in devices],
                  tile_rows=args.tile_rows, tile_columns=args.tile_columns,
                  phase_checkpoint=args.gradient,
                  job=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                  aperture_m=.0036, wavelength_m=520.6e-9, distance_m=.01008, bandlimit='hann',
                  passband_ratio=float(ratio), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print('PLAN', json.dumps(record), flush=True)
    args.output.with_suffix('.hlo.txt').write_text(compiled.as_text())
    if args.plan_only:
        args.output.write_text(json.dumps(record, indent=2)+'\n')
        return
    assert estimate < min(record['allocator_limits'])-1024*2**20, 'Insufficient allocator headroom'

    @jax.jit
    def phase_values(seed):
        i = jnp.arange(n, dtype=jnp.uint32)[:, None]*jnp.uint32(n)+jnp.arange(n, dtype=jnp.uint32)[None, :]+seed
        v = (i^(i >> 16))*jnp.uint32(0x7feb352d)
        v = (v^(v >> 15))*jnp.uint32(0x846ca68b)
        v ^= v >> 16
        return (v & 65535).astype(jnp.float32)*np.float32(2*np.pi/65536)

    def plane_values(offset):
        y = jnp.arange(n, dtype=jnp.float32)[:, None]/np.float32(n)
        x = jnp.arange(n, dtype=jnp.float32)[None, :]/np.float32(n)
        return sum(np.float32(weight)*jnp.exp(jnp.complex64(1j)*(
            np.float32(ay)*y+np.float32(ax)*x+offset)) for ay, ax, weight in modes)

    initialize = jax.jit(phase_values if args.gradient else plane_values, out_shardings=row)
    seed = jnp.asarray(17, jnp.uint32) if args.gradient else jnp.asarray(0., jnp.float32)
    field = initialize(seed).block_until_ready()
    samples, alias_checks = [], []
    for repeat in range(args.samples+1):
        if repeat and not args.gradient:
            field = initialize(seed).block_until_ready()
        pointers = [s.data.unsafe_buffer_pointer() for s in field.addressable_shards]
        begin = time.perf_counter()
        result = jax.block_until_ready(compiled(field, payload))
        elapsed = 1000*(time.perf_counter()-begin)
        if not args.gradient:
            alias = field.is_deleted() and pointers == [s.data.unsafe_buffer_pointer() for s in result.addressable_shards]
            assert alias, 'Donated distributed field was not reused'
            alias_checks.append(alias)
        if repeat:
            samples.append(elapsed)
        if repeat < args.samples:
            del result
    record.update(samples_ms=samples, median_ms=statistics.median(samples),
                  memory_after_execution=[d.memory_stats() for d in devices], donation_checks=alias_checks)
    print('MEASURED', json.dumps({k: record[k] for k in ('median_ms', 'memory_after_execution')}), flush=True)
    rows = np.unique(np.r_[np.linspace(0, n-1, 28, dtype=np.int64), 1, n-2, target[0], target[0]+1])
    cols = np.unique(np.r_[np.linspace(0, n-1, 28, dtype=np.int64), 1, n-2, target[1], target[1]+1])
    def sample_local(x):
        first = jax.lax.axis_index('tp')*(n//p)
        local_rows = jnp.asarray(rows)-first
        selected = x[jnp.clip(local_rows, 0, n//p-1)[:, None], jnp.asarray(cols)[None, :]]
        selected = jnp.where(((local_rows>=0) & (local_rows<n//p))[:, None], selected, 0)
        return jax.lax.psum(selected, 'tp')
    sample = jax.jit(shard_map(sample_local, mesh=mesh, in_specs=P('tp', None), out_specs=P(), check_rep=False))
    if args.gradient:
        phase_samples = np.asarray(sample(field))
        actual = np.asarray(sample(result[1]))
        record['linear_loss'] = float(result[0][0])
    else:
        actual = np.asarray(sample(result))
    del field, result
    # The independent reference needs only the small transfer window and sampled
    # positions. Evaluate after recording the benchmark's GPU-memory high water.
    bits = payload.addressable_shards[0].data

    def unpack(bits):
        re = (bits & 65535).astype(jnp.int32)
        im = (bits >> 16).astype(jnp.int32)
        re = jnp.where(re>=32768, re-65536, re).astype(jnp.float32)*scale
        im = jnp.where(im>=32768, im-65536, im).astype(jnp.float32)*scale
        return jax.lax.complex(re, im).astype(jnp.complex128)

    def frequencies():
        ky = (jnp.arange(kh, dtype=jnp.int64)+oy) % (2*n)
        kx = (jnp.arange(kw, dtype=jnp.int64)+ox) % (2*n)
        return (jnp.where(ky>=n, ky-2*n, ky).astype(jnp.float64)*(np.pi/n),
                jnp.where(kx>=n, kx-2*n, kx).astype(jnp.float64)*(np.pi/n))

    @jax.jit
    def direct_impulse(bits, lag_y, lag_x):
        wy, wx = frequencies()
        ey = jnp.exp(1j*lag_y[:, None]*wy[None, :])
        ex = jnp.exp(1j*wx[:, None]*lag_x[None, :])
        return (ey @ unpack(bits) @ ex)/float((2*n)**2)

    @jax.jit
    def direct_planes(bits):
        wy, wx = frequencies()
        def finite_sum(t):
            return jnp.exp(1j*(n-1)*t/2)*jnp.sin(n*t/2)/jnp.sin(t/2)
        h = unpack(bits)
        output = jnp.zeros((len(rows), len(cols)), jnp.complex128)
        for ay, ax, weight in modes:
            ey = jnp.exp(1j*jnp.asarray(rows, jnp.float64)[:, None]*wy[None, :])*finite_sum(ay/n-wy)[None, :]
            ex = jnp.exp(1j*wx[:, None]*jnp.asarray(cols, jnp.float64)[None, :])*finite_sum(ax/n-wx)[:, None]
            output += weight*(ey @ h @ ex)/float((2*n)**2)
        return output

    if args.gradient:
        kernel = np.asarray(direct_impulse(bits, jnp.asarray(target[0]-rows, jnp.float64),
                                          jnp.asarray(target[1]-cols, jnp.float64)))
        reference = -(np.exp(1j*phase_samples)*kernel/np.float32(n)).imag
    else:
        reference = np.asarray(direct_planes(bits))
    error = float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
    assert error < 3e-5, error
    record.update(sample_relative_l2=error, sample_rows=rows.tolist(), sample_columns=cols.tolist(),
                  protocol='Fresh process; fixed physical packed ASM window; all fields distributed over GPU memory. '
                           'Five synchronized samples after warmup; initialization and reference excluded. '
                           'Gradient mode retains phase input, full output field and full phase gradient for a single-pixel linear loss. '
                           'Forward mode donates a dense sum of two plane waves. '
                           'Independent complex128 direct DFT validates sampled gradients or dense forward values; no reference FFT.')
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(args.output)
    print('PASS', error, flush=True)


if __name__ == '__main__':
    main()
