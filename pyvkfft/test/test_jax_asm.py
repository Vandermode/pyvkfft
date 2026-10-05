"""CUDA FFI regression tests; run directly or with unittest discovery.

Set PYVKFFT_ASM_FFI_LIBRARY and select a CUDA device before starting Python.
No CuPy or legacy pyvkfft.cuda extension is used by these tests.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import unittest

import numpy as np
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None


def reference(field, transfer):
    h, w = field.shape[-2:]
    padded = jnp.pad(field, [(0, 0)]*(field.ndim-2) + [(h//2, h-h//2), (w//2, w-w//2)])
    result = jnp.fft.ifft2(jnp.fft.fft2(padded)*transfer)
    return result[..., h//2:h//2+h, w//2:w//2+w]


def analytic_reference(field, z, wavelength, *, pitch, bandlimit='none'):
    h, w = field.shape
    fy = jnp.fft.fftfreq(2*h, pitch[0])[:, None]
    fx = jnp.fft.fftfreq(2*w, pitch[1])[None, :]
    q = 1/wavelength**2 - (fx*fx + fy*fy)
    valid = q >= 0
    # Strictly positive clipped branch avoids 0*inf AD artifacts outside support.
    root = jnp.sqrt(jnp.where(valid, q, 1.))
    transfer = jnp.exp(2j*jnp.pi*z*root)
    if bandlimit == 'rectangular':
        valid &= ((jnp.abs(fx) <= 1/wavelength/jnp.sqrt(1+(z/(w*pitch[1]))**2))
                  & (jnp.abs(fy) <= 1/wavelength/jnp.sqrt(1+(z/(h*pitch[0]))**2)))
    return reference(field, jnp.where(valid, transfer, 0).astype(jnp.complex64))


@unittest.skipUnless(jax is not None and os.environ.get('PYVKFFT_ASM_FFI_LIBRARY'),
                     'JAX and an ASM FFI library are required')
class TestJAXASM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update('jax_enable_x64', True)
        if jax.devices()[0].platform != 'gpu':
            raise unittest.SkipTest('CUDA device required')
        from pyvkfft import jax_asm
        cls.engine = jax_asm

    def field(self, shape, seed=37):
        rng = np.random.default_rng(seed)
        return jnp.asarray((rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(np.complex64))

    def close(self, actual, expected, tolerance=6e-6):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = np.linalg.norm((actual-expected).ravel()) / max(np.linalg.norm(expected.ravel()), 1e-15)
        self.assertTrue(np.isfinite(error))
        self.assertLess(error, tolerance, (error, tolerance))

    def test_general_transfer_forward_shapes_and_immutable_input(self):
        for shape in ((8, 12), (9, 15), (1, 6), (6, 1), (30, 50), (64, 48)):
            with self.subTest(shape=shape):
                x = self.field(shape)
                saved = np.asarray(x).copy()
                transfer = self.field(tuple(2*n for n in shape), 43)
                actual = jax.jit(self.engine.convolve)(x, transfer)
                # NumPy provides an independent double-precision FFT reference.
                h, w = shape
                xp = np.pad(saved, ((h//2, h-h//2), (w//2, w-w//2)))
                expected = np.fft.ifft2(np.fft.fft2(xp)*np.asarray(transfer))[h//2:h//2+h, w//2:w//2+w]
                self.close(actual, expected)
                np.testing.assert_array_equal(x, saved)

    def test_identity_zero_and_changed_transfer(self):
        x = self.field((16, 24))
        execute = jax.jit(self.engine.convolve)
        h = jnp.ones((32, 48), jnp.complex64)
        self.close(execute(x, h), x)
        np.testing.assert_array_equal(execute(x, h*0), np.zeros(x.shape))
        self.close(execute(x, h*(.3+.8j)), x*(.3+.8j))
        self.close(execute(x, h), x)

    def packed(self, shape, seed=42):
        rng = np.random.default_rng(seed)
        re = rng.integers(-32768, 32768, size=shape, dtype=np.int32)
        im = rng.integers(-32768, 32768, size=shape, dtype=np.int32)
        re.flat[:5] = [-32768, -32767, -1, 0, 32767]
        im.flat[:5] = [32767, 0, -1, -32767, -32768]
        bits = (re & 65535).astype(np.uint32) | ((im & 65535).astype(np.uint32) << 16)
        decoded = re.astype(np.float32) + np.complex64(1j)*im.astype(np.float32)
        return jnp.asarray(bits), decoded

    def test_packed_forward_extremes_scales_and_complex_transpose(self):
        for shape in ((8, 12), (9, 15), (1, 6), (6, 1)):
            x = self.field(shape)
            bits, raw = self.packed(tuple(2*n for n in shape))
            saved = np.asarray(bits).copy()
            execute = jax.jit(self.engine.convolve_packed)
            probe = self.field(shape, 93)
            for scale in (np.float32(1/32767), np.float32(3.7/32767), np.float32(0)):
                transfer = raw*scale
                h, w = shape
                xp = np.pad(np.asarray(x), ((h//2, h-h//2), (w//2, w-w//2)))
                expected = np.fft.ifft2(np.fft.fft2(xp)*transfer)[h//2:h//2+h, w//2:w//2+w]
                actual = execute(x, bits, scale)
                if scale == 0:
                    np.testing.assert_array_equal(actual, np.zeros(shape))
                else:
                    self.close(actual, expected)
                    native_grad = jax.jit(jax.grad(lambda value: jnp.real(jnp.sum(
                        self.engine.convolve_packed(value, bits, scale)*probe))))(x)
                    ref_grad = jax.jit(jax.grad(lambda value: jnp.real(jnp.sum(
                        reference(value, jnp.asarray(transfer))*probe))))(x)
                    self.close(native_grad, ref_grad)
            self.close(execute(x, bits), reference(x, jnp.asarray(raw*np.float32(1/32767))))
            np.testing.assert_array_equal(bits, saved)

    def test_packed_broadcast_vmap_jvp_and_second_field_derivative(self):
        x = self.field((2, 1, 8, 12))
        bits, raw = self.packed((1, 3, 16, 24))
        scales = jnp.asarray([0., 1/32767, 4.5/32767], jnp.float32).reshape(1, 3, 1, 1)
        transfer = jnp.asarray(raw)*scales
        native = lambda value: self.engine.convolve_packed(value, bits, scales)
        ref = lambda value: reference(value, transfer)
        self.close(jax.jit(native)(x), ref(x))
        direction = self.field(x.shape, 69)
        gradients = []
        for op in (native, ref):
            self.close(jax.jvp(op, (x,), (direction,))[1], op(direction))
            grad = jax.grad(lambda value: jnp.sum(jnp.abs(op(value))**2))
            gradients.append((jax.jit(grad)(x), jax.jit(lambda v, d: jax.jvp(grad, (v,), (d,))[1])(x, direction)))
        for actual, expected in zip(*gradients):
            self.close(actual, expected)
        mapped = jax.jit(jax.vmap(lambda value: self.engine.convolve_packed(value, bits[0], scales[0])))(x[:, 0])
        self.close(mapped, ref(x))
        # Scale belongs to the fixed quantized transfer, not the design.
        scale_grad = jax.grad(lambda s: jnp.real(jnp.sum(self.engine.convolve_packed(x, bits, s))))(scales)
        np.testing.assert_array_equal(scale_grad, np.zeros(scales.shape))

    def test_packed_multi_upload_axes_and_native_ir(self):
        for shape in ((24, 18000), (18000, 24)):
            x = self.field(shape)
            bits, raw = self.packed(tuple(2*n for n in shape))
            scale = np.float32(2/32767)
            probe = self.field(shape, 58)
            function = jax.jit(jax.value_and_grad(lambda value, payload: jnp.real(jnp.sum(
                self.engine.convolve_packed(value, payload, scale)*probe))))
            actual = function(x, bits)
            ref = jax.jit(jax.value_and_grad(lambda value: jnp.real(jnp.sum(
                reference(value, jnp.asarray(raw*scale))*probe))))(x)
            self.close(actual[1], ref[1])
            np.testing.assert_allclose(actual[0], ref[0], rtol=2e-5, atol=.005)
            ir = str(function.lower(x, bits).compiler_ir())
            self.assertIn('@pyvkfft_asm_v3', ir)
            self.assertIn('ui32', ir)
            self.assertNotIn('stablehlo.fft', ir)
            self.assertNotIn('stablehlo.shift_right', ir)
            info = [p for p in self.engine.cache_info() if p['shape'] == list(shape) and p['transfer_mode'] == 3]
            self.assertTrue(any(max(p['plan']['uploads']) == 2 for p in info), info)

    def test_packed_validation(self):
        x = self.field((8, 12))
        bits, _ = self.packed((16, 24))
        with self.assertRaises(ValueError):
            self.engine.convolve_packed(x, bits.astype(jnp.int32))
        with self.assertRaises(ValueError):
            self.engine.convolve_packed(x, bits[:, :-1])
        with self.assertRaises(TypeError):
            self.engine.convolve_packed(x, bits, jnp.asarray(1., jnp.float64))
        with self.assertRaises(ValueError):
            self.engine.convolve_packed(x, bits, jnp.ones(3, jnp.float32))

    def test_multi_upload_axis_forward_and_gradient(self):
        for shape in ((24, 18000), (18000, 24)):
            with self.subTest(shape=shape):
                x = self.field(shape)
                transfer = self.field(tuple(2*n for n in shape), 45)
                function = jax.jit(self.engine.convolve)
                self.close(function(x, transfer), reference(x, transfer))
                probe = self.field(x.shape, 9)
                for f in (self.engine.convolve, reference):
                    gradient = jax.jit(jax.grad(lambda value: jnp.real(jnp.sum(f(value, transfer)*probe))))(x)
                    if f is self.engine.convolve:
                        actual = gradient
                    else:
                        self.close(actual, gradient)
                info = [p for p in self.engine.cache_info() if p['shape'] == list(x.shape)]
                self.assertTrue(any(max(p['plan']['uploads']) == 2 for p in info), info)
                pitch = (.2e-6, .2e-6)
                op = partial(self.engine.asm, pixel_pitch=pitch)
                self.close(jax.jit(op)(x, .001, 532e-9),
                           analytic_reference(x, .001, 532e-9, pitch=pitch))

    def test_field_and_nonradial_transfer_gradients(self):
        x, h = self.field((12, 20)), self.field((24, 40), 12)
        target = self.field(x.shape, 98)
        gradients = []
        for function in (self.engine.convolve, reference):
            loss = lambda x, h: jnp.mean(jnp.abs(function(x, h)-target)**2)
            gradients.append(jax.jit(jax.grad(loss, argnums=(0, 1)))(x, h))
        for actual, expected in zip(*gradients):
            self.close(actual, expected)

    def test_field_directional_finite_difference(self):
        x, h, direction = self.field((12, 20)), self.field((24, 40), 8), self.field((12, 20), 49)
        loss = jax.jit(lambda value: jnp.mean(jnp.abs(self.engine.convolve(value, h))**2))
        gradient = jax.grad(loss)(x)
        expected = jnp.real(jnp.sum(gradient*direction))
        step = .01
        actual = (loss(x+step*direction)-loss(x-step*direction))/(2*step)
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-5)

    def test_quadrant_forward_gradient_and_jvp(self):
        x, q = self.field((9, 15)), self.field((10, 16), 61)
        iy, ix = jnp.arange(18), jnp.arange(30)
        iy, ix = jnp.minimum(iy, 18-iy), jnp.minimum(ix, 30-ix)
        native = partial(self.engine.convolve, transfer_layout='quadrant')
        ref = lambda x, q: reference(x, q[iy[:, None], ix[None, :]])
        self.close(jax.jit(native)(x, q), ref(x, q))
        for op in (native, ref):
            gradient = jax.jit(jax.grad(lambda x, q: jnp.mean(jnp.abs(op(x, q))**2), argnums=(0, 1)))(x, q)
            if op is native:
                actual = gradient
            else:
                for a, b in zip(actual, gradient):
                    self.close(a, b)
        self.close(jax.jvp(native, (x, q), (x, q))[1], jax.jvp(ref, (x, q), (x, q))[1])

    def test_jvp_and_field_hessian_vector(self):
        x, h = self.field((12, 20)), self.field((24, 40), 18)
        direction = self.field(x.shape, 54)
        self.close(jax.jvp(self.engine.convolve, (x, h), (direction, h))[1],
                   jax.jvp(reference, (x, h), (direction, h))[1])
        results = []
        for op in (self.engine.convolve, reference):
            grad = jax.grad(lambda x: jnp.mean(jnp.abs(op(x, h))**2))
            results.append(jax.jit(lambda x, d: jax.jvp(grad, (x,), (d,))[1])(x, direction))
        self.close(*results)

    def test_broadcast_and_vmap_gradients(self):
        x = self.field((2, 1, 8, 12))
        h = self.field((1, 3, 16, 24), 42)
        self.close(jax.jit(self.engine.convolve)(x, h), reference(x, h))
        ours = jax.jit(jax.grad(lambda h: jnp.mean(jnp.abs(self.engine.convolve(x, h))**2)))(h)
        expected = jax.grad(lambda h: jnp.mean(jnp.abs(reference(x, h))**2))(h)
        self.close(ours, expected)
        values = self.field((8, 2, 12), 53)
        transfer = h[0, 0]
        mapped = jax.jit(jax.vmap(lambda a: self.engine.convolve(a, transfer), in_axes=1, out_axes=1))
        self.close(mapped(values), jax.vmap(lambda a: reference(a, transfer), in_axes=1, out_axes=1)(values))

    def test_analytic_modes_and_dynamic_values(self):
        x = self.field((12, 20))
        for pitch, mask in (((6.4e-6, 5e-6), 'none'), ((.2e-6, .24e-6), 'none'),
                            ((.2e-6, .24e-6), 'rectangular')):
            op = jax.jit(partial(self.engine.asm, pixel_pitch=pitch, bandlimit=mask))
            for z, wave in ((0., 532e-9), (1e-6, 633e-9), (-1e-6, 450e-9)):
                self.close(op(x, z, wave), analytic_reference(x, z, wave, pitch=pitch, bandlimit=mask))

    def test_analytic_parameter_gradients_jvp_and_second_derivatives(self):
        x, probe = self.field((12, 20)), self.field((12, 20), 18)
        pitch = (6.4e-6, 5e-6)
        native = partial(self.engine.asm, pixel_pitch=pitch)
        ref = partial(analytic_reference, pitch=pitch)
        z, w = jnp.asarray(.01, jnp.float64), jnp.asarray(532e-9, jnp.float64)
        results = []
        for op in (native, ref):
            loss = lambda x, z, w: jnp.real(jnp.sum(op(x, z, w)*probe))
            grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(x, z, w)
            tangent = jax.jit(lambda x, z, w: jax.jvp(op, (x, z, w), (x, jnp.asarray(1e-8), jnp.asarray(1e-13)))[1])(x, z, w)
            second = [jax.jit(jax.grad(jax.grad(loss, argnums=i), argnums=j))(x, z, w)
                      for i, j in ((1, 1), (1, 2), (2, 2))]
            results.append((grads, tangent, second))
        for a, b in zip(results[0][0], results[1][0]):
            self.close(a, b, 2e-5)
        self.close(results[0][1], results[1][1], 2e-5)
        for a, b in zip(results[0][2], results[1][2]):
            self.close(a, b, 2e-5)

    def test_paraxial_intensity_parameter_gradients(self):
        x = self.field((64, 96))
        z, wavelength = .01, 532e-9
        h, w = x.shape
        for pitch in (6.4e-6, 50e-6):
            fy = np.fft.fftfreq(2*h, pitch)[:, None]
            fx = np.fft.fftfreq(2*w, pitch)[None, :]
            frequency_squared = fx*fx + fy*fy
            root = np.sqrt(1/wavelength**2-frequency_squared)
            residual = -2*np.pi*frequency_squared/(root+1/wavelength)
            slope = residual/wavelength**2/root
            spectrum = np.fft.fft2(np.pad(np.asarray(x), ((h//2, h//2), (w//2, w//2))))
            spectrum *= np.exp(2j*np.pi*z*root)
            crop = lambda a: a[h//2:h//2+h, w//2:w//2+w]
            output = crop(np.fft.ifft2(spectrum))
            expected = [2*np.sum(np.real(np.conj(output)*crop(np.fft.ifft2(spectrum*1j*d))))
                        for d in (residual, z*slope)]

            def loss(z, wavelength):
                y = self.engine.asm(x, z, wavelength, pixel_pitch=(pitch, pitch))
                # Sum avoids rounding a non-power-of-two mean's cotangent
                # before cancellation of the physically irrelevant carrier.
                return jnp.sum(jnp.real(y*jnp.conj(y)))

            actual = jax.jit(jax.grad(loss, argnums=(0, 1)))(jnp.asarray(z), jnp.asarray(wavelength))
            for a, b in zip(actual, expected):
                self.close(a, b, 1e-5)

    def test_grazing_mode_distance_derivative(self):
        x = self.field((8, 8))
        # Exactly representable q=0 at axis Nyquist frequencies.
        op = partial(self.engine.asm, pixel_pitch=(1., 1.))
        reference_op = partial(analytic_reference, pitch=(1., 1.))
        native = jax.jit(lambda z: jax.jvp(lambda z: op(x, z, 2.), (z,), (jnp.ones_like(z),))[1])
        expected = jax.jit(lambda z: jax.jvp(lambda z: reference_op(x, z, 2.), (z,), (jnp.ones_like(z),))[1])
        self.close(native(jnp.asarray(.1)), expected(jnp.asarray(.1)), 1e-5)

    def test_analytic_batched_parameters_and_float32(self):
        x = self.field((8, 12))
        pitch = (6.4e-6, 5e-6)
        z = jnp.asarray([.001, -.002], jnp.float32)
        wavelength = jnp.asarray(532e-9, jnp.float32)
        op = partial(self.engine.asm, pixel_pitch=pitch)
        expected = jax.vmap(lambda z: analytic_reference(x, z.astype(jnp.float64), wavelength.astype(jnp.float64), pitch=pitch))(z)
        self.close(jax.jit(op)(x, z, wavelength), expected)
        self.close(jax.jit(jax.vmap(op, in_axes=(None, 0, None)))(x, z, wavelength), expected)

    def test_compiled_field_gradient_uses_native_calls(self):
        x, h = self.field((12, 20)), self.field((24, 40), 23)
        compiled = jax.jit(jax.value_and_grad(lambda x, h: jnp.sum(jnp.abs(self.engine.convolve(x, h))**2)))
        text = str(compiled.lower(x, h).compiler_ir())
        self.assertIn('@pyvkfft_asm_v3', text)
        self.assertNotIn('stablehlo.fft', text)
        self.assertNotIn('stablehlo.pad', text)

    def test_concurrent_submissions_and_cache_clear(self):
        x, h = self.field((12, 20)), self.field((24, 40), 29)
        function = jax.jit(self.engine.convolve)
        function(x, h).block_until_ready()
        values = [x*(i+1) for i in range(8)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(lambda value: np.asarray(function(value, h)), values))
        for value, output in zip(values, outputs):
            self.close(output, reference(value, h))
        pending = function(x, h)
        self.engine.clear_plan_cache()
        self.close(pending, reference(x, h))
        self.close(function(x, h), reference(x, h))

    def test_validation_and_nonfinite_dynamic_parameters(self):
        x, h = self.field((12, 20)), self.field((24, 40))
        with self.assertRaises(TypeError):
            self.engine.convolve(x.astype(jnp.complex128), h)
        with self.assertRaises(ValueError):
            self.engine.convolve(x, h[:, :-1])
        with self.assertRaises(ValueError):
            self.engine.asm(x, .01, -1., pixel_pitch=(1., 1.))
        with self.assertRaises(ValueError):
            self.engine.asm(x, .01, 1., pixel_pitch=(0., 1.))
        with self.assertRaises(ValueError):
            self.engine.convolve(x, h, grouped_batch=-1)
        function = jax.jit(partial(self.engine.asm, pixel_pitch=(6.4e-6, 6.4e-6)))
        result = function(x, jnp.asarray(.01), jnp.asarray(-1.))
        self.assertTrue(np.isnan(result).all())
        from pyvkfft.asm import _load_native
        with self.assertRaisesRegex(RuntimeError, 'JAX FFI backend'):
            _load_native(os.environ['PYVKFFT_ASM_FFI_LIBRARY'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
