"""Independent distributed FFT, cyclic-window and complex AD validation."""
import argparse
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from pyvkfft.jax_distributed import fft2_factory, windowed_asm_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='vkfft', choices=['jax', 'vkfft'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    devices = jax.devices()
    print(jax.__version__, devices, flush=True)
    mesh = Mesh(np.array(devices), ('tp',))
    p = len(devices)
    rng = np.random.default_rng(301)
    records = []
    def error(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-20))
    for h, w in [(8*p, 12*p), (32*p, 48*p), (192*p, 250*p), (8*p, 12288), (12288, 8*p)]:
        x = (rng.normal(size=(h, w))+1j*rng.normal(size=(h, w))).astype(np.complex64)
        fft = jax.jit(fft2_factory(mesh, backend=args.backend))
        ifft = jax.jit(fft2_factory(mesh, backend=args.backend, inverse=True))
        y = fft(x)
        rec = dict(shape=[h, w], fft_error=error(y, np.fft.fft2(x)), roundtrip_error=error(ifft(y), x))
        fft_grad = jax.jit(jax.grad(lambda z: jnp.sum(jnp.abs(fft(z))**2)))(x)
        rec['fft_gradient_error'] = error(fft_grad, 2*h*w*np.conj(x))
        th, tw = min(19, 2*h), min(23, 2*w)
        for oy, ox in [(2*h-th//2, 2*w-tw//2), (3, 5)]:
            packed = rng.integers(0, 2**32, (th, tw), dtype=np.uint32)
            scale = np.float32(1/32767)
            real = (packed & 65535).astype(np.int32)
            imag = (packed >> 16).astype(np.int32)
            transfer = (np.where(real>=32768, real-65536, real)+
                        1j*np.where(imag>=32768, imag-65536, imag)).astype(np.complex64)*scale
            full = np.zeros((2*h, 2*w), np.complex64)
            full[np.ix_((np.arange(th)+oy) % (2*h), (np.arange(tw)+ox) % (2*w))] = transfer
            def reference(z):
                padded = jnp.pad(z, ((h//2, h-h//2), (w//2, w-w//2)))
                return jnp.fft.ifft2(jnp.fft.fft2(padded)*full)[h//2:h//2+h, w//2:w//2+w]
            op = windowed_asm_factory(mesh, origin=(oy, ox), backend=args.backend,
                                       tile_rows=13, tile_columns=7)
            fun = jax.jit(lambda z: op(z, packed, scale))
            expected = jax.jit(reference)(x)
            actual = fun(x)
            weights = rng.uniform(.2, 1.2, size=(h, w)).astype(np.float32)
            loss = lambda z: jnp.sum(weights*jnp.abs(fun(z))**2)
            ref_loss = lambda z: jnp.sum(weights*jnp.abs(reference(z))**2)
            grad = jax.jit(jax.grad(loss))(x)
            expected_grad = jax.jit(jax.grad(ref_loss))(x)
            tangent = (rng.normal(size=(h, w))+1j*rng.normal(size=(h, w))).astype(np.complex64)
            jvp = jax.jit(lambda z, dz: jax.jvp(fun, (z,), (dz,))[1])(x, tangent)
            entry = dict(rec, origin=[oy, ox], transfer_shape=[th, tw],
                         field_error=error(actual, expected), gradient_error=error(grad, expected_grad),
                         jvp_error=error(jvp, jax.jit(reference)(tangent)))
            print(json.dumps(entry), flush=True)
            assert max(v for k, v in entry.items() if k.endswith('_error')) < 3e-5, entry
            records.append(entry)
        jax.clear_caches()
    report = dict(backend=args.backend, jax=jax.__version__, devices=[str(d) for d in devices],
                  job=os.environ.get('SLURM_JOB_ID'), cases=records)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(args.output)
    print('ALL_VALIDATED', flush=True)


if __name__ == '__main__':
    main()
