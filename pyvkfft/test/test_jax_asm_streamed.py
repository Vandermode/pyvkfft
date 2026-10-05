"""Tiled JAX propagation: reuse transfer/AD contracts and verify tile boundaries."""
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace
import unittest

import numpy as np
import jax
import jax.numpy as jnp
from test_jax_asm import TestJAXASM, reference


@unittest.skipUnless(os.environ.get('PYVKFFT_STREAMED_FFI_LIBRARY'), 'Streamed FFI build required')
class TestJAXStreamed(unittest.TestCase):
    field = TestJAXASM.field
    packed = TestJAXASM.packed
    close = TestJAXASM.close
    test_packed_forward_extremes_scales_and_complex_transpose = TestJAXASM.test_packed_forward_extremes_scales_and_complex_transpose
    test_packed_broadcast_vmap_jvp_and_second_field_derivative = TestJAXASM.test_packed_broadcast_vmap_jvp_and_second_field_derivative

    @classmethod
    def setUpClass(cls):
        from pyvkfft import jax_asm
        jax.config.update('jax_enable_x64', True)
        if jax.devices()[0].platform != 'gpu':
            raise unittest.SkipTest('CUDA required')
        cls.native = jax_asm
        cls.engine = SimpleNamespace(convolve_packed=partial(jax_asm.convolve_packed,
                                                             tile_columns=7, tile_rows=3))

    def test_multi_upload_partial_tiles_cache_and_immutable_input(self):
        for shape in ((24, 18000), (18000, 24)):
            x = self.field(shape)
            saved = np.asarray(x).copy()
            payload, raw = self.packed(tuple(2*n for n in shape))
            op = partial(self.native.convolve_packed, tile_columns=127, tile_rows=31)
            function = jax.jit(jax.value_and_grad(lambda v, h: jnp.sum(jnp.abs(op(v, h))**2)))
            expected = jax.value_and_grad(lambda v: jnp.sum(jnp.abs(reference(v, jnp.asarray(raw/np.float32(32767))))**2))(x)
            result = function(x, payload)
            self.close(result[1], expected[1])
            ir = str(function.lower(x, payload).compiler_ir())
            self.assertIn('@pyvkfft_streamed_v', ir)
            self.assertNotIn('stablehlo.fft', ir)
            self.assertNotIn('stablehlo.shift_right', ir)
            self.native.clear_streamed_plan_cache()
            self.close(function(x, payload)[1], expected[1])
            np.testing.assert_array_equal(x, saved)

    def test_concurrent_submissions_use_independent_workspaces(self):
        x = self.field((32, 64))
        payload, raw = self.packed((64, 128))
        function = jax.jit(partial(self.native.convolve_packed, tile_columns=17, tile_rows=7))
        function(x, payload).block_until_ready()
        def submit(index):
            source = x * np.complex64(index+1j)
            return function(source, payload).block_until_ready(), source
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(submit, range(1, 7)))
        for output, source in results:
            self.close(output, reference(source, jnp.asarray(raw/np.float32(32767))))

    def test_reused_input_is_immutable_with_multiple_consumers(self):
        x = self.field((17, 30))
        saved = np.asarray(x).copy()
        payload, raw = self.packed((34, 60))
        op = partial(self.native.convolve_packed, tile_columns=13, tile_rows=5)
        @jax.jit
        def operation(value, table):
            propagated = op(value, table)
            return propagated, value*3, op(value*(1+2j), table)+propagated
        expected = reference(x, jnp.asarray(raw/np.float32(32767)))
        for _ in range(2):
            output, unchanged, combined = operation(x, payload)
            self.close(output, expected)
            self.close(unchanged, saved*3)
            self.close(combined, expected*(2+2j))
            np.testing.assert_array_equal(x, saved)

    def test_explicit_donation_reuses_the_field_buffer(self):
        x = self.field((17, 30))
        payload, raw = self.packed((34, 60))
        expected = reference(x, jnp.asarray(raw/np.float32(32767)))
        expected.block_until_ready()
        pointer = x.unsafe_buffer_pointer()
        function = jax.jit(partial(self.native.convolve_packed, tile_columns=13, tile_rows=5),
                           donate_argnums=(0,))
        output = function(x, payload)
        output.block_until_ready()
        self.close(output, expected)
        self.assertEqual(output.unsafe_buffer_pointer(), pointer)
        self.assertTrue(x.is_deleted())
        if self.native._streamed_library().streamed_ffi_abi_version() >= 2:
            ir = str(function.lower(jax.ShapeDtypeStruct(output.shape, output.dtype), payload).compiler_ir())
            self.assertIn('output_operand_aliases', ir)

    def test_larger_planned_axes_with_general_transfer(self):
        # Exercise both long-axis orientations at the projected square-grid
        # limits without requiring an otherwise idle 80 GB device.
        for shape in ((48, 41472), (41472, 48), (48, 45360), (45360, 48)):
            x = self.field(shape)
            payload, raw = self.packed(tuple(2*n for n in shape))
            op = partial(self.native.convolve_packed, tile_columns=4093, tile_rows=4093)
            actual = jax.jit(op)(x, payload)
            self.close(actual, reference(x, jnp.asarray(raw/np.float32(32767))))

    def test_tile_guards(self):
        x = self.field((8, 12))
        payload, _ = self.packed((16, 24))
        for options in ({'tile_columns': -1}, {'tile_columns': True}, {'tile_rows': 3},
                        {'tile_columns': 8, 'grouped_batch': 32}, {'tile_columns': 8, 'tile_rows': 0}):
            with self.assertRaises(ValueError):
                self.native.convolve_packed(x, payload, **options)

    def test_windowed_transfer_forward_and_complex_transpose(self):
        # Asymmetric, wrapped supports distinguish the transpose from both
        # conjugation and reusing the primal window origin.
        for shape in ((7, 10), (18, 15)):
            h, w = shape
            x = self.field(shape)
            weight = self.field(shape)*(0.3+0.7j)
            for support in ((1, 1), (h+1, w-1), (2*h-1, w+3), (2*h, 2*w)):
                for origin in ((0, 0), (2*h-3, 2*w-4), (3, 5)):
                    payload, raw = self.packed(support)
                    full = np.zeros((2*h, 2*w), np.complex64)
                    iy = (np.arange(support[0])+origin[0]) % (2*h)
                    ix = (np.arange(support[1])+origin[1]) % (2*w)
                    full[np.ix_(iy, ix)] = raw/np.float32(32767)
                    op = partial(self.native.convolve_packed, tile_columns=7,
                                 tile_rows=3, transfer_origin=origin)
                    actual = jax.jit(op)(x, payload)
                    self.close(actual, reference(x, jnp.asarray(full)))
                    objective = lambda v: jnp.sum(jnp.real(op(v, payload)*weight))
                    expected = lambda v: jnp.sum(jnp.real(reference(v, jnp.asarray(full))*weight))
                    self.close(jax.jit(jax.grad(objective))(x), jax.grad(expected)(x))

    def test_windowed_broadcast_jvp_and_donation_without_extra_spectrum(self):
        shape = (17, 30)
        payload, raw = self.packed((13, 19))
        origin = (29, 53)
        full = np.zeros((34, 60), np.complex64)
        full[np.ix_((np.arange(13)+29) % 34, (np.arange(19)+53) % 60)] = raw/np.float32(32767)
        x = self.field(shape)
        op = partial(self.native.convolve_packed, tile_columns=11, tile_rows=5,
                     transfer_origin=origin)
        batched = jnp.stack((x, x*(2+1j)))
        result, tangent = jax.jit(lambda v: jax.jvp(lambda a: op(a, payload), (v,), (v*0.5,)))(batched)
        expected = jnp.stack([reference(v, jnp.asarray(full)) for v in batched])
        self.close(result, expected)
        self.close(tangent, expected*0.5)
        saved = np.asarray(x).copy()
        immutable = jax.jit(lambda v: (op(v, payload), v*3))(x)
        self.close(immutable[0], expected[0])
        np.testing.assert_array_equal(x, saved)
        self.close(immutable[1], saved*3)
        pointer = x.unsafe_buffer_pointer()
        output = jax.jit(op, donate_argnums=(0,))(x, payload)
        output.block_until_ready()
        self.close(output, expected[0])
        self.assertTrue(x.is_deleted())
        self.assertEqual(pointer, output.unsafe_buffer_pointer())
        plans = [p for p in self.native.streamed_cache_info()
                 if p['shape'] == list(shape) and p['transfer_shape'] == [13, 19]]
        self.assertTrue(plans)
        self.assertTrue(all(p['extra_spectrum_bytes'] == 0 for p in plans))

    def test_windowed_multi_upload_and_partial_tiles(self):
        for shape, support, origin in (((24, 18000), (31, 917), (39, 35409)),
                                       ((18000, 24), (917, 31), (35409, 39))):
            x = self.field(shape)
            payload, raw = self.packed(support)
            full = np.zeros(tuple(2*n for n in shape), np.complex64)
            iy = (np.arange(support[0])+origin[0]) % full.shape[0]
            ix = (np.arange(support[1])+origin[1]) % full.shape[1]
            full[np.ix_(iy, ix)] = raw/np.float32(32767)
            op = partial(self.native.convolve_packed, tile_columns=127, tile_rows=11,
                         transfer_origin=origin)
            self.close(jax.jit(op)(x, payload), reference(x, jnp.asarray(full)))
            self.close(jax.jit(jax.grad(lambda v: jnp.mean(jnp.real(op(v, payload)))))(x),
                       jax.grad(lambda v: jnp.mean(jnp.real(reference(v, jnp.asarray(full)))))(x))

    def test_windowed_guards(self):
        x = self.field((8, 12))
        payload, _ = self.packed((7, 9))
        for options in ({}, {'transfer_origin': (0, 0)},
                        {'tile_columns': 8, 'transfer_origin': (0.5, 1)},
                        {'tile_columns': 8, 'transfer_origin': (True, 0)},
                        {'tile_columns': 8, 'transfer_origin': (1,)},
                        {'tile_columns': 8, 'transfer_origin': 1}):
            with self.assertRaises(ValueError):
                self.native.convolve_packed(x, payload, **options)
        with self.assertRaises(ValueError):
            self.native.convolve_packed(x, jnp.zeros((17, 9), jnp.uint32),
                                       tile_columns=8, transfer_origin=(0, 0))

    def test_packed_quadrant_forward_gradient_and_donation(self):
        for shape in ((17, 30), (24, 18000), (18000, 24)):
            h, w = shape
            x = self.field(shape)
            payload, raw = self.packed((h+1, w+1))
            iy = np.minimum(np.arange(2*h), 2*h-np.arange(2*h))
            ix = np.minimum(np.arange(2*w), 2*w-np.arange(2*w))
            full = jnp.asarray(raw[np.ix_(iy, ix)]/np.float32(32767))
            op = partial(self.native.convolve_packed, tile_columns=127, tile_rows=11,
                         transfer_layout='quadrant')
            self.close(jax.jit(op)(x, payload), reference(x, full))
            weight = self.field(shape)*(.3+.7j)
            self.close(jax.jit(jax.grad(lambda v: jnp.sum(jnp.real(op(v, payload)*weight))))(x),
                       jax.grad(lambda v: jnp.sum(jnp.real(reference(v, full)*weight)))(x))
            expected = reference(x, full)
            expected.block_until_ready()
            pointer = x.unsafe_buffer_pointer()
            output = jax.jit(op, donate_argnums=(0,))(x, payload)
            output.block_until_ready()
            self.close(output, expected)
            self.assertTrue(x.is_deleted())
            self.assertEqual(output.unsafe_buffer_pointer(), pointer)

    def test_packed_quadrant_guards_and_window_metadata(self):
        x = self.field((8, 12))
        payload, _ = self.packed((9, 13))
        for options in ({'transfer_layout': 'quadrant'},
                        {'tile_columns': 8, 'transfer_layout': 'unknown'},
                        {'tile_columns': 8, 'transfer_layout': 'quadrant', 'transfer_origin': (0, 0)}):
            with self.assertRaises(ValueError):
                self.native.convolve_packed(x, payload, **options)
        window = self.native.PackedTransferWindow(payload, (16, 24), (13, 20))
        self.close(jax.jit(partial(self.native.convolve_packed, tile_columns=8))(x, window),
                   jax.jit(partial(self.native.convolve_packed, tile_columns=8,
                                   transfer_origin=(13, 20)))(x, payload))
        for table, options in ((self.native.PackedTransferWindow(payload, (15, 24), (0, 0)), {}),
                               (window, {'transfer_origin': (0, 0)}),
                               (window, {'transfer_layout': 'quadrant'})):
            with self.assertRaises(ValueError):
                self.native.convolve_packed(x, table, tile_columns=8, **options)


del TestJAXASM

if __name__ == '__main__':
    unittest.main(verbosity=2)
