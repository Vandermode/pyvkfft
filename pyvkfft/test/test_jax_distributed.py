"""Distributed slab FFT, cyclic-window packing and complex AD regressions.

CPU checks: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4
Native checks additionally require the distributed FFI library and CUDA GPUs.
"""
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from pyvkfft.jax_distributed import fft2_factory, windowed_asm_factory


class TestDistributedReference(unittest.TestCase):
    backend = 'jax'

    @classmethod
    def setUpClass(cls):
        devices = jax.devices()
        if len(devices)<2:
            raise unittest.SkipTest('At least two devices required')
        if cls.backend == 'vkfft' and (devices[0].platform != 'gpu' or
                                       not os.environ.get('PYVKFFT_DISTRIBUTED_FFI_LIBRARY')):
            raise unittest.SkipTest('CUDA GPUs and isolated FFI build required')
        cls.mesh = Mesh(np.array(devices), ('tp',))
        cls.p = len(devices)
        cls.row = NamedSharding(cls.mesh, P('tp', None))
        cls.rep = NamedSharding(cls.mesh, P())

    def close(self, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        relative = np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-20)
        self.assertLess(relative, 3e-5)

    def case(self, full=False):
        rng = np.random.default_rng(42)
        shape = (16*self.p, 21)
        x = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
        window = tuple(2*d for d in shape) if full else (19, 23)
        bits = rng.integers(0, 2**32, window, dtype=np.uint32)
        origin = (0, 0) if full else (2*shape[0]-9, 2*shape[1]-11)
        real, imag = (bits & 65535).astype(np.int32), (bits >> 16).astype(np.int32)
        value = (np.where(real>=32768, real-65536, real)+
                 1j*np.where(imag>=32768, imag-65536, imag)).astype(np.complex64)/np.float32(32767)
        transfer = np.zeros(tuple(2*d for d in shape), np.complex64)
        transfer[np.ix_((np.arange(window[0])+origin[0]) % (2*shape[0]),
                        (np.arange(window[1])+origin[1]) % (2*shape[1]))] = value
        def reference(a):
            padded = jnp.pad(a, tuple((d//2, d-d//2) for d in shape))
            return jnp.fft.ifft2(jnp.fft.fft2(padded)*transfer)[shape[0]//2:shape[0]//2+shape[0],
                                                             shape[1]//2:shape[1]//2+shape[1]]
        op = windowed_asm_factory(self.mesh, origin=origin, tile_rows=13, tile_columns=7, backend=self.backend)
        return x, bits, lambda a: op(a, bits, np.float32(1/32767)), reference

    def test_fft_normalization_and_complex_transpose(self):
        shape = (24*self.p, 32*self.p)
        rng = np.random.default_rng(19)
        x = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
        fft = jax.jit(fft2_factory(self.mesh, backend=self.backend, tile=31))
        inverse = jax.jit(fft2_factory(self.mesh, backend=self.backend, inverse=True, tile=31))
        self.close(fft(x), np.fft.fft2(x))
        self.close(inverse(fft(x)), x)
        gradient = jax.jit(jax.grad(lambda a: jnp.sum(jnp.abs(fft(a))**2)))(x)
        self.close(gradient, 2*np.prod(shape)*np.conj(x))

    def test_asymmetric_window_padding_jvp_and_phase_hessian(self):
        x, _, op, reference = self.case()
        self.close(jax.jit(op)(x), jax.jit(reference)(x))
        self.close(jax.jvp(op, (jnp.asarray(x),), (jnp.asarray(x*.3j),))[1], reference(x*.3j))
        weights = jnp.linspace(.2, 1., x.shape[1])
        loss = lambda phase: jnp.sum(jnp.abs(op(jnp.exp(1j*phase)))**2*weights)
        ref_loss = lambda phase: jnp.sum(jnp.abs(reference(jnp.exp(1j*phase)))**2*weights)
        phase, tangent = jnp.asarray(x.real), jnp.asarray(x.imag)
        self.close(jax.jit(jax.grad(loss))(phase), jax.jit(jax.grad(ref_loss))(phase))
        actual = jax.jit(lambda a, da: jax.jvp(jax.grad(loss), (a,), (da,))[1])(phase, tangent)
        expected = jax.jit(lambda a, da: jax.jvp(jax.grad(ref_loss), (a,), (da,))[1])(phase, tangent)
        self.close(actual, expected)

    def test_batched_data_parallel_broadcast_transpose(self):
        if len(jax.devices()) not in (2, 4):
            self.skipTest('Use two or four devices')
        mesh = Mesh(np.asarray(jax.devices()).reshape(-1, 2), ('wvl', 'tp'))
        h, w = 32, 21
        rng = np.random.default_rng(82)
        x = (rng.normal(size=(1, 3, h, w))+1j*rng.normal(size=(1, 3, h, w))).astype(np.complex64)
        bits = rng.integers(0, 2**32, (2, 1, 19, 23), dtype=np.uint32)
        real, imag = (bits & 65535).astype(np.int32), (bits >> 16).astype(np.int32)
        value = (np.where(real>=32768, real-65536, real)+
                 1j*np.where(imag>=32768, imag-65536, imag)).astype(np.complex64)/np.float32(32767)
        origin = (2*h-9, 2*w-11)
        transfer = np.zeros((2, 1, 2*h, 2*w), np.complex64)
        iy, ix = (np.arange(19)+origin[0]) % (2*h), (np.arange(23)+origin[1]) % (2*w)
        transfer[..., iy[:, None], ix] = value
        def reference(a):
            a = jnp.pad(a, ((0, 0), (0, 0), (h//2, h//2), (w//2, w-w//2)))
            a = jnp.fft.ifft2(jnp.fft.fft2(a)*transfer)
            return a[..., h//2:h//2+h, w//2:w//2+w]
        op = windowed_asm_factory(mesh, origin=origin, batch_axes=('wvl', None),
                                  backend=self.backend, tile_rows=13, tile_columns=7)
        native = lambda a: op(a, bits, np.float32(1/32767))
        self.close(jax.jit(native)(x), jax.jit(reference)(x))
        weights = jnp.linspace(.2, 1., w)
        loss = lambda a: jnp.sum(jnp.abs(native(a))**2*weights)
        expected_loss = lambda a: jnp.sum(jnp.abs(reference(a))**2*weights)
        self.close(jax.jit(jax.grad(loss))(x), jax.jit(jax.grad(expected_loss))(x))

    def test_full_support_odd_width_and_reused_input(self):
        x, _, op, reference = self.case(full=True)
        result, unchanged = jax.jit(lambda a: (op(a), a*3))(x)
        self.close(result, reference(x))
        np.testing.assert_array_equal(unchanged, x*3)

    def test_donated_field_reuses_each_local_buffer(self):
        host, bits, _, reference = self.case()
        expected = np.asarray(jax.jit(reference)(host))
        x = jax.device_put(host.copy(), self.row)
        payload = jax.device_put(bits, self.rep)
        origin = (2*host.shape[0]-9, 2*host.shape[1]-11)
        op = windowed_asm_factory(self.mesh, origin=origin, backend=self.backend, tile_rows=13, tile_columns=7)
        pointers = [s.data.unsafe_buffer_pointer() for s in x.addressable_shards]
        result = jax.jit(lambda a, h: op(a, h, np.float32(1/32767)), donate_argnums=(0,),
                         in_shardings=(self.row, self.rep), out_shardings=self.row)(x, payload)
        result.block_until_ready()
        self.close(result, expected)
        if self.backend == 'vkfft':
            self.assertTrue(x.is_deleted())
            self.assertEqual(pointers, [s.data.unsafe_buffer_pointer() for s in result.addressable_shards])

    def test_single_plane_batch_retains_donation(self):
        host, bits, _, reference = self.case()
        expected = np.asarray(jax.jit(reference)(host))
        host = host.reshape((1,)*5+host.shape)
        bits = bits.reshape((1,)*5+bits.shape)
        row = NamedSharding(self.mesh, P(None, None, None, None, None, 'tp', None))
        x = jax.device_put(host.copy(), row)
        payload = jax.device_put(bits, self.rep)
        origin = (2*host.shape[-2]-9, 2*host.shape[-1]-11)
        op = windowed_asm_factory(self.mesh, origin=origin, backend=self.backend, tile_rows=13, tile_columns=7, batch_axes=(None,)*5)
        pointers = [s.data.unsafe_buffer_pointer() for s in x.addressable_shards]
        result = jax.jit(lambda a, h: op(a, h, np.float32(1/32767)), donate_argnums=(0,),
                         in_shardings=(row, self.rep), out_shardings=row)(x, payload)
        result.block_until_ready()
        self.close(result.reshape(expected.shape), expected)
        if self.backend == 'vkfft':
            self.assertTrue(x.is_deleted())
            self.assertEqual(pointers, [s.data.unsafe_buffer_pointer() for s in result.addressable_shards])

    def test_invalid_arguments_and_fixed_scale(self):
        with self.assertRaises(ValueError):
            fft2_factory(self.mesh, tile=0)
        with self.assertRaises(ValueError):
            fft2_factory(self.mesh)(jnp.zeros((8*self.p, 8*self.p), jnp.float32))
        with self.assertRaises(ValueError):
            windowed_asm_factory(self.mesh, origin=(0, 0), tile_rows=65537)
        op = windowed_asm_factory(self.mesh, origin=(0, 0), backend=self.backend)
        field = jnp.ones((8*self.p, 12), jnp.complex64)
        bits = jnp.ones((3, 5), jnp.uint32)
        with self.assertRaisesRegex(TypeError, 'fixed parameters'):
            jax.grad(lambda scale: jnp.sum(op(field, bits, scale).real))(jnp.float32(1.))


class TestDistributedNative(TestDistributedReference):
    backend = 'vkfft'


if __name__ == '__main__':
    unittest.main()
