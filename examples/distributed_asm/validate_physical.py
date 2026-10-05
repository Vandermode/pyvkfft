"""Compare complete physical ASM fields and phase gradients to dense JAX FFTs."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scalax.sharding import MeshShardingHelper
from functional.propagation import PropagationConfig, propagation_constant_factory
from pyvkfft.jax_distributed import fft2_factory, windowed_asm_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=18000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    n, devices = args.size, jax.devices()
    p = len(devices)
    mesh = Mesh(np.array(devices), ('tp',))
    row, rep = NamedSharding(mesh, P('tp', None)), NamedSharding(mesh, P())
    helper = MeshShardingHelper([1, 1, 1, 1, 1, p], ['zo', 'wvl', 'angle', 'zi', 'polar', 'tp'])
    cfg = PropagationConfig(nx=n, ny=n, dx=.0036/n, dy=.0036/n, wvl=520.6e-9,
                            reference_plane_to_sensor_distance=.01008, theta=0., sensor_center_mode='on_axis',
                            aperture_type='rectangular', spatial_wave_fields='scalar_demodulated',
                            propagation_backend='vkfft', transfer_storage='snorm16_compact',
                            vkfft_transfer_layout='window', vkfft_tile_columns=64)
    table, _, _, _ = propagation_constant_factory(helper, cfg, return_values=True, H_sharding=None)
    payload = jax.device_put(table.payload.reshape(table.shape[-2:]), rep)
    oy, ox = table.origin
    scale = np.float32(1/32767)
    forward = fft2_factory(mesh, backend='jax')
    backward = fft2_factory(mesh, backend='jax', inverse=True)
    def dense(x, bits):
        re, im = (bits & 65535).astype(jnp.int32), (bits >> 16).astype(jnp.int32)
        value = (jnp.where(re>=32768, re-65536, re)+1j*jnp.where(im>=32768, im-65536, im))*scale
        full = jnp.zeros((2*n, 2*n), jnp.complex64)
        full = full.at[(jnp.arange(bits.shape[0])+oy)[:, None] % (2*n),
                       (jnp.arange(bits.shape[1])+ox)[None, :] % (2*n)].set(value)
        y = backward(forward(jnp.pad(x, ((n//2, n//2), (n//2, n//2))))*full)
        return y[n//2:n//2+n, n//2:n//2+n]
    compact = windowed_asm_factory(mesh, origin=(oy, ox), tile_rows=512, tile_columns=128)
    weights = .3+.7*jnp.cos(jnp.arange(n, dtype=jnp.float32)*np.float32(2*np.pi/n))**2
    def loss_factory(op):
        field_from_phase = jax.checkpoint(lambda phase: jnp.exp(1j*phase).astype(jnp.complex64)/np.float32(n))
        def loss(phase, bits):
            field = op(field_from_phase(phase), bits)
            return jnp.sum(jnp.abs(field)**2*weights), field
        return jax.jit(jax.value_and_grad(loss, has_aux=True), in_shardings=(row, rep))
    initialize = jax.jit(lambda: (jnp.sin(jnp.arange(n, dtype=jnp.float32)[:, None]*.01)+
                                 jnp.cos(jnp.arange(n, dtype=jnp.float32)[None, :]*.017)).astype(jnp.float32),
                         out_shardings=row)
    phase = initialize()
    actual = jax.block_until_ready(loss_factory(lambda x, h: compact(x, h, scale))(phase, payload))
    expected = jax.block_until_ready(loss_factory(dense)(phase, payload))
    @jax.jit
    def relative(a, b):
        return jnp.sqrt(jnp.sum(jnp.abs((a-b).astype(jnp.complex128))**2)/jnp.sum(jnp.abs(b.astype(jnp.complex128))**2))
    record = dict(size=n, device_count=p, jax=jax.__version__, transfer_shape=list(payload.shape),
                  field_relative_l2=float(relative(actual[0][1], expected[0][1])),
                  phase_gradient_relative_l2=float(relative(actual[1], expected[1])),
                  loss_relative=float(abs(actual[0][0]-expected[0][0])/abs(expected[0][0])))
    print(json.dumps(record), flush=True)
    assert max(record[k] for k in ('field_relative_l2', 'phase_gradient_relative_l2', 'loss_relative')) < 3e-5
    args.output.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
