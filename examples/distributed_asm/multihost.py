"""Two-node Slurm check: distributed VkFFT, AD and communication timing.

Run one process and one allocated GPU per node. The large timing uses a
synthetic packed window of the same 9305x9305 size as the physical study.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import jax
jax.distributed.initialize(local_device_ids=[0], initialization_timeout=120)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from pyvkfft.jax_distributed import fft2_factory, windowed_asm_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=32768)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mesh = Mesh(np.array(jax.devices()), ('tp',))
    row, col, rep = (NamedSharding(mesh, spec) for spec in (P('tp', None), P(None, 'tp'), P()))
    rank, p = jax.process_index(), jax.device_count()
    print('IDENTITY', rank, jax.__version__, jax.devices(), jax.local_devices(), flush=True)
    def put(host, sharding):
        host = np.asarray(host)
        return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])
    def scalar(value):
        return float(np.asarray(value.addressable_shards[0].data))
    relative = jax.jit(lambda a, b: jnp.sqrt(jnp.sum(jnp.abs(a-b)**2)/jnp.sum(jnp.abs(b)**2)), out_shardings=rep)
    rng = np.random.default_rng(137)
    shape = (32*p, 48*p)
    host = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
    x = put(host, row)
    fft = jax.jit(fft2_factory(mesh, tile=512))
    inverse = jax.jit(fft2_factory(mesh, inverse=True, tile=512))
    expected_fft = put(np.fft.fft2(host).astype(np.complex64), col)
    errors = dict(fft_relative_l2=scalar(relative(fft(x), expected_fft)),
                  roundtrip_relative_l2=scalar(relative(inverse(fft(x)), x)))
    bits = rng.integers(0, 2**32, (19, 23), dtype=np.uint32)
    packed = put(bits, rep)
    scale = np.float32(1/32767)
    origin = (2*shape[0]-9, 2*shape[1]-11)
    raw = windowed_asm_factory(mesh, origin=origin, tile_rows=13, tile_columns=7)
    real, imag = (bits & 65535).astype(np.int32), (bits >> 16).astype(np.int32)
    values = (np.where(real>=32768, real-65536, real)+1j*np.where(imag>=32768, imag-65536, imag))*scale
    transfer = np.zeros(tuple(2*d for d in shape), np.complex128)
    transfer[np.ix_((np.arange(19)+origin[0]) % (2*shape[0]), (np.arange(23)+origin[1]) % (2*shape[1]))] = values
    def dense(value, h):
        padded = np.pad(value, tuple((d//2, d-d//2) for d in shape))
        return np.fft.ifft2(np.fft.fft2(padded)*h)[shape[0]//2:shape[0]//2+shape[0], shape[1]//2:shape[1]//2+shape[1]]
    expected = dense(host, transfer)
    reverse = np.roll(np.flip(transfer, (0, 1)), (1, 1), (0, 1))
    expected_gradient = dense(2*np.conj(expected), reverse)
    fn = jax.jit(lambda v, h: raw(v, h, scale))
    gradient = jax.jit(jax.grad(lambda v, h: jnp.sum(jnp.abs(fn(v, h))**2)))
    errors.update(field_relative_l2=scalar(relative(fn(x, packed), put(expected.astype(np.complex64), row))),
                  gradient_relative_l2=scalar(relative(gradient(x, packed), put(expected_gradient.astype(np.complex64), row))))
    print('VALIDATED', json.dumps(errors), flush=True)
    assert max(errors.values()) < 3e-5, errors
    del x, packed, expected_fft
    jax.clear_caches()
    n = args.size
    seed = put(np.asarray(17, np.uint32), rep)
    def initialize(seed):
        i = jnp.arange(n, dtype=jnp.uint32)[:, None]*jnp.uint32(n)+jnp.arange(n, dtype=jnp.uint32)[None, :]+seed
        v = (i^(i >> 16))*jnp.uint32(0x7feb352d)
        v = (v^(v >> 15))*jnp.uint32(0x846ca68b)
        v ^= v >> 16
        return (v & 65535).astype(jnp.float32)*np.float32(2*np.pi/65536)
    phase = jax.jit(initialize, out_shardings=row)(seed)
    field = jax.jit(lambda v: jnp.exp(1j*v).astype(jnp.complex64), out_shardings=row)(phase)
    records = []
    def measure(name, function, *operands):
        compiled = jax.jit(function).lower(*operands).compile()
        result = jax.block_until_ready(compiled(*operands))
        multihost_utils.sync_global_devices(name+'-start')
        samples = []
        for _ in range(args.samples):
            del result
            start = time.perf_counter()
            result = jax.block_until_ready(compiled(*operands))
            samples.append(1000*(time.perf_counter()-start))
        multihost_utils.sync_global_devices(name+'-end')
        m = compiled.memory_analysis()
        record = dict(name=name, samples_ms=samples, median_ms=statistics.median(samples),
                      memory=[d.memory_stats() for d in jax.local_devices()],
                      compiled_memory={k: int(getattr(m, k)) for k in ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')})
        print('MEASURED', json.dumps(record), flush=True)
        records.append(record)
    for backend in ('jax', 'vkfft'):
        measure('fft-'+backend, fft2_factory(mesh, backend=backend, tile=2048), field)
    del field
    k = min(9305, 2*n)
    def table(seed):
        i = jnp.arange(k, dtype=jnp.uint32)[:, None]*jnp.uint32(k)+jnp.arange(k, dtype=jnp.uint32)[None, :]+seed
        return (i^(i >> 16))*jnp.uint32(0x7feb352d)
    packed = jax.jit(table, out_shardings=rep)(seed)
    field_from_phase = jax.checkpoint(lambda v: jnp.exp(1j*v).astype(jnp.complex64)/np.float32(n))
    for backend in ('jax', 'vkfft'):
        op = windowed_asm_factory(mesh, origin=(2*n-k//2, 2*n-k//2), backend=backend,
                                  tile_rows=512, tile_columns=128)
        def loss(v, h):
            result = op(field_from_phase(v), h, scale)
            return jnp.sum(jnp.abs(result)**2), result
        measure('window-gradient-'+backend, jax.value_and_grad(loss, has_aux=True), phase, packed)
    report = dict(rank=rank, process_count=jax.process_count(), device_count=p, size=n,
                  node=os.environ.get('SLURMD_NODENAME'), job=os.environ.get('SLURM_JOB_ID'),
                  validation=errors, cases=records, synthetic_window_shape=[k, k])
    path = args.output.with_name(args.output.stem+f'-rank{rank}.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    multihost_utils.sync_global_devices('finished')
    print('ALL_COMPLETED', flush=True)
    jax.distributed.shutdown()


if __name__ == '__main__':
    main()
