"""Measure single-plane LOM forward/design-gradient capacity in a fresh process.

Uses a nonconstant phase design and a constant complex packed transfer, giving
an analytic large-grid oracle without a padded reference allocation. This is a
capacity/FFT correctness probe, not a physical propagation benchmark. Set one
CUDA_VISIBLE_DEVICES GPU and choose an allocator memory fraction suitable for
that reserved device. Near capacity, preallocation avoids pool fragmentation. Includes output and design gradient.
"""
import argparse
from functools import partial
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
from functional.propagation import PropagationConfig, complex_field_propagation_function_factory, propagation_function_factory
from pyvkfft.jax_asm import streamed_cache_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--tile', type=int, default=1024)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('vkfft', 'jax'), default='vkfft')
    parser.add_argument('--phase-factory', action='store_true', help='Use the shared phase factory with rematerialized phase modulation')
    parser.add_argument('--plan-only', action='store_true', help='Report compiled memory without allocating large arrays')
    parser.add_argument('--loss-only', action='store_true', help='Return loss and design gradient without retaining the full field')
    args = parser.parse_args()
    n = args.size
    if n < 2 or n % 2 or not 1 <= args.tile <= 65536 or args.samples < 1:
        parser.error('Require a positive even shape, tile in 1..65536, and samples >=1')
    mesh = MeshShardingHelper([1]*6, ['zo', 'wvl', 'angle', 'zi', 'polar', 'tp'])
    cfg = PropagationConfig(nx=n, ny=n, dx=6.4e-6, dy=6.4e-6, wvl=532e-9,
                            transfer_storage='snorm16_compact', propagation_backend=args.backend,
                            vkfft_tile_columns=args.tile if args.backend == 'vkfft' else 0,
                            spatial_wave_fields='scalar_demodulated', aperture_type='rectangular')
    forward = (propagation_function_factory(mesh, cfg)[0] if args.phase_factory
               else complex_field_propagation_function_factory(mesh, cfg))
    incident = jnp.ones((1,)*7, jnp.float32)
    post = jnp.ones((1,)*7, jnp.complex64)

    def operation(phase, payload):
        y = (forward(phase, incident, payload, post) if args.phase_factory
             else forward(jnp.exp(1j*phase), payload, post))
        loss = jnp.mean(jnp.real(y))
        return loss if args.loss_only else (loss, y)
    function = jax.jit(jax.value_and_grad(operation, has_aux=not args.loss_only))
    phase_spec = jax.ShapeDtypeStruct((1,)*5+(n,n), jnp.float32)
    transfer_spec = jax.ShapeDtypeStruct((1,)*5+(2*n,2*n), jnp.uint32)
    compiled = function.lower(phase_spec, transfer_spec).compile()
    analysis = compiled.memory_analysis()
    memory = {key:int(getattr(analysis,key)) for key in ('argument_size_in_bytes','output_size_in_bytes',
                                                       'temp_size_in_bytes','alias_size_in_bytes')}
    estimate = memory['argument_size_in_bytes']+memory['output_size_in_bytes']+memory['temp_size_in_bytes']-memory['alias_size_in_bytes']
    device = jax.devices()[0]
    limit = device.memory_stats()['bytes_limit']
    print(json.dumps(dict(size=n, compiled_memory=memory, estimate=estimate, allocator_limit=limit)), flush=True)
    if args.plan_only:
        record = dict(size=n, tile=args.tile, backend=args.backend, loss_only=args.loss_only, phase_factory=args.phase_factory,
                      compiled_memory=memory, compiler_allocation_estimate_bytes=estimate,
                      allocator_limit=limit, executed=False)
        args.output.write_text(json.dumps(record, indent=2)+'\n')
        return
    if estimate > limit-512*2**20:
        raise MemoryError('Compiled allocation leaves less than 512 MiB allocator headroom')

    @jax.jit
    def initialize(seed):
        i = jnp.arange(n*n,dtype=jnp.uint32).reshape(phase_spec.shape)+seed
        value = (i^(i>>16))*jnp.uint32(0x7feb352d)
        value = (value^(value>>15))*jnp.uint32(0x846ca68b)
        value ^= value>>16
        return (value & 65535).astype(jnp.float32)*np.float32(2*np.pi/65536)
    phase = initialize(jnp.asarray(17,jnp.uint32))
    bits = np.uint32(19660 | (26214 << 16))
    payload = jax.jit(lambda value:jnp.full(transfer_spec.shape,value,jnp.uint32))(jnp.asarray(bits))
    jax.block_until_ready((phase,payload))
    before = device.memory_stats()
    result = jax.block_until_ready(compiled(phase,payload))
    samples=[]
    for _ in range(args.samples):
        del result
        start=time.perf_counter();result=jax.block_until_ready(compiled(phase,payload));samples.append(1000*(time.perf_counter()-start))
    measured = device.memory_stats()
    print(json.dumps(dict(size=n, median_ms=statistics.median(samples), execution_memory=measured)),flush=True)
    # Host chunks bound validation storage and leave the measured high water intact.
    coef=np.complex64((np.float32(19660)+1j*np.float32(26214))*np.float32(1/32767))
    sums=np.zeros(4,np.float64)
    actual=None if args.loss_only else result[0][1].reshape(n,n)
    gradient=result[1].reshape(n,n);design=phase.reshape(n,n)
    @partial(jax.jit, static_argnums=(2,))
    def host_chunk(array, first, rows):
        return jax.lax.dynamic_slice_in_dim(array, first, rows, axis=0)
    for first in range(0,n,64):
        rows=min(64,n-first)
        start=jnp.asarray(first,jnp.int32)
        values=np.asarray(host_chunk(design,start,rows));expected=(np.exp(1j*values)*coef).astype(np.complex64)
        reference_gradient=(-expected.imag/np.float32(n*n)).astype(np.float32)
        a=expected if actual is None else np.asarray(host_chunk(actual,start,rows))
        g=np.asarray(host_chunk(gradient,start,rows))
        sums += [np.sum(np.abs(a-expected)**2,dtype=np.float64),np.sum(np.abs(expected)**2,dtype=np.float64),
                 np.sum((g-reference_gradient)**2,dtype=np.float64),np.sum(reference_gradient**2,dtype=np.float64)]
    errors=dict(field_relative_l2=float(np.sqrt(sums[0]/sums[1])),gradient_relative_l2=float(np.sqrt(sums[2]/sums[3])))
    assert errors['field_relative_l2']<2e-5 and errors['gradient_relative_l2']<3e-5,errors
    record=dict(size=n,tile=args.tile,backend=args.backend,loss_only=args.loss_only, phase_factory=args.phase_factory,samples_ms=samples,median_ms=statistics.median(samples),
                compiled_memory=memory,compiler_allocation_estimate_bytes=estimate,allocator_limit=limit,
                before_execution_memory=before,execution_memory=measured,errors=errors,
                plans=streamed_cache_info() if args.backend=='vkfft' else [],gpu=device.device_kind,
                gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),jax_version=jax.__version__,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                protocol='Fresh process; actual shared LOM factory; nonconstant phase; constant complex packed transfer; forward field plus nonzero phase gradient; analytic chunked validation; no reference FFT allocation. Capacity probe, not physical-H performance.')
    if args.loss_only:
        record['errors']['field_relative_l2'] = None
        record['protocol'] = record['protocol'].replace('forward field plus nonzero phase gradient', 'loss plus nonzero phase gradient; full field is not retained')
    args.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(errors),flush=True)


if __name__=='__main__':main()
