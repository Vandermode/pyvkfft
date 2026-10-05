"""ASM correctness: independent complex128 NumPy reference and CUDA contracts.

Select an authorized GPU with CUDA_VISIBLE_DEVICES before running this module.
Native cases require PYVKFFT_ASM_LIBRARY; explicit VkFFT cases run without it.
"""
import os
import ctypes
import unittest
from unittest.mock import patch

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from pyvkfft.asm import ASMPlan, CUDA_HELPERS


def reference_transfer(shape, pitch, z, wavelength, bandlimit=False):
    fy = np.fft.fftfreq(shape[0], pitch[0])[:, None]
    fx = np.fft.fftfreq(shape[1], pitch[1])[None, :]
    q = (1. / wavelength)**2 - (fx**2 + fy**2)
    mask = q >= 0
    if bandlimit:
        lx, ly = shape[1]*pitch[1], shape[0]*pitch[0]
        mask = mask & (np.abs(fx) <= 1./(wavelength*np.sqrt(1.+(2*z/lx)**2)))
        mask = mask & (np.abs(fy) <= 1./(wavelength*np.sqrt(1.+(2*z/ly)**2)))
    result = np.zeros(shape, dtype=np.complex128)
    result[mask] = np.exp(2j*np.pi*z*np.sqrt(q[mask]))
    return result


def reference_propagate(source, transfer):
    h, w = source.shape
    padded = np.zeros((2*h, 2*w), np.complex128)
    padded[h//2:h//2+h, w//2:w//2+w] = source
    propagated = np.fft.ifft2(np.fft.fft2(padded)*transfer)
    return propagated[h//2:h//2+h, w//2:w//2+w]


@unittest.skipIf(cp is None, 'CuPy unavailable')
class TestASM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        except Exception as exc:
            raise unittest.SkipTest(f'CUDA unavailable: {exc}')

    def profiles(self, dynamic=False):
        yield {'implementation': 'explicit'}
        if os.environ.get('PYVKFFT_ASM_LIBRARY'):
            yield {'implementation': 'native'}
            yield {'implementation': 'native', 'prune': True}
            yield {'implementation': 'native', 'prune': True, 'axis_order': 'column'}
            if dynamic:
                yield {'implementation': 'native', 'transfer': 'fused'}
                yield {'implementation': 'native', 'transfer': 'fused', 'prune': True}
                yield {'implementation': 'native', 'transfer': 'fused', 'prune': True, 'axis_order': 'column'}

    def assert_close(self, actual, expected):
        actual = cp.asnumpy(actual)
        self.assertTrue(np.isfinite(actual).all())
        norm = np.linalg.norm(expected)
        if norm:
            self.assertLess(np.linalg.norm(actual-expected)/norm, 2e-5)
            self.assertLess(np.max(np.abs(actual-expected))/np.max(np.abs(expected)), 1e-4)
        else:
            self.assertEqual(np.max(np.abs(actual)), 0.)

    def test_analytic_static_preparation(self):
        rng = np.random.default_rng(926)
        for shape in ((15, 21), (32, 8192), (8192, 32)):
            source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
            x, out = cp.asarray(source), cp.empty(shape, cp.complex64)
            pitch = (0.3e-6, 0.25e-6)
            for profile in self.profiles():
                for mask in ('none', 'rectangular'):
                    with ASMPlan(shape, pitch, bandlimit=mask, tuning_profile=profile) as plan:
                        for z, wavelength in ((0., 500e-9), (1., 532e-9), (-1e-6, 633e-9)):
                            expected = reference_propagate(source, reference_transfer(
                                plan.padded_shape, pitch, z, wavelength, mask == 'rectangular'))
                            before = cp.get_default_memory_pool().used_bytes()
                            plan.prepare_asm_transfer(z=z, wavelength=wavelength)
                            plan.execute(x, out)
                            self.assert_close(out, expected)
                            self.assertEqual(before, cp.get_default_memory_pool().used_bytes())
                        for z, wavelength in ((None, 532e-9), (0., None), (float('nan'), 532e-9), (0., 0.)):
                            with self.assertRaises(ValueError):
                                plan.prepare_asm_transfer(z=z, wavelength=wavelength)
                    with self.assertRaises(RuntimeError):
                        plan.prepare_asm_transfer(z=0., wavelength=532e-9)
        with ASMPlan((8, 16), pitch, mode='dynamic') as plan:
            with self.assertRaises(ValueError):
                plan.prepare_asm_transfer(z=0., wavelength=532e-9)

    def test_static_shapes_and_order(self):
        rng = np.random.default_rng(713)
        for shape in ((8, 16), (15, 21), (32, 8192), (8192, 32)):
            source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
            padded_shape = tuple(2*n for n in shape)
            transfer = (rng.uniform(.2, 1.4, padded_shape) *
                        np.exp(1j*rng.uniform(-np.pi, np.pi, padded_shape))).astype(np.complex64)
            expected = reference_propagate(source, transfer)
            for profile in self.profiles():
                for order in ('fft', 'centered'):
                    with self.subTest(shape=shape, profile=profile, order=order):
                        with ASMPlan(shape, (6.4e-6, 6.4e-6), tuning_profile=profile) as plan:
                            gpu_source = cp.asarray(source)
                            gpu_dest = cp.empty_like(gpu_source)
                            prepared = np.fft.fftshift(transfer) if order == 'centered' else transfer
                            plan.prepare_transfer(cp.asarray(prepared), order=order)
                            plan._work.fill(cp.nan)
                            plan.execute(gpu_source, gpu_dest)
                            self.assert_close(gpu_dest, expected)
                            np.testing.assert_array_equal(cp.asnumpy(gpu_source), source)
                            # Stale workspace must not affect a new input or output buffer.
                            plan._work.fill(17+23j)
                            other = cp.empty_like(gpu_dest)
                            plan.execute(-gpu_source, other)
                            self.assert_close(other, -expected)

    def test_identity_impulses_zero(self):
        shape = (15, 16)
        for profile in self.profiles():
            with ASMPlan(shape, (1e-6, 2e-6), tuning_profile=profile) as plan:
                plan.prepare_transfer(cp.ones(plan.padded_shape, cp.complex64))
                for location in (None, (0, 0), (7, 8), (14, 15)):
                    source = np.zeros(shape, np.complex64)
                    if location:
                        source[location] = 1+2j
                    actual = cp.empty(shape, cp.complex64)
                    plan.execute(cp.asarray(source), actual)
                    self.assert_close(actual, source)

    def test_dynamic_parameters(self):
        rng = np.random.default_rng(907)
        shape = (32, 48)
        source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
        for pitch in ((6.4e-6, 6.4e-6), (0.2e-6, 0.25e-6)):
            for bandlimit in ('none', 'rectangular'):
                for profile in self.profiles(dynamic=True):
                    with self.subTest(pitch=pitch, bandlimit=bandlimit, profile=profile):
                        with ASMPlan(shape, pitch, mode='dynamic', bandlimit=bandlimit, tuning_profile=profile) as plan:
                            x, out = cp.asarray(source), cp.empty(shape, cp.complex64)
                            for z, wavelength in ((0., 532e-9), (.01, 450e-9), (-.1, 633e-9), (1., 532e-9)):
                                transfer = reference_transfer(plan.padded_shape, pitch, z, wavelength,
                                                              bandlimit == 'rectangular')
                                plan.execute(x, out, z=z, wavelength=wavelength)
                                self.assert_close(out, reference_propagate(source, transfer))

    def test_stream_and_allocation_contract(self):
        for profile in self.profiles(dynamic=True):
            stream = cp.cuda.Stream(non_blocking=True)
            with stream, ASMPlan((16, 32), (6.4e-6, 6.4e-6), mode='dynamic',
                                  stream=stream, tuning_profile=profile) as plan:
                x = cp.ones(plan.shape, cp.complex64)
                out = cp.empty_like(x)
                plan.execute(x, out, z=0., wavelength=532e-9)
                stream.synchronize()
                used = cp.get_default_memory_pool().used_bytes()
                for _ in range(3):
                    plan.execute(x, out, z=0., wavelength=532e-9)
                stream.synchronize()
                self.assertEqual(used, cp.get_default_memory_pool().used_bytes())
                self.assert_close(out, np.ones(plan.shape))
                with self.assertRaises(ValueError):
                    plan.execute(x, x, z=0., wavelength=532e-9)
                with self.assertRaises(ValueError):
                    plan.execute(x, out, z=0., wavelength=-1)
            with self.assertRaises(RuntimeError):
                plan.execute(x, out, z=0., wavelength=532e-9)

    def test_validation_contract(self):
        for shape in ((0, 8), (-1, 8), (True, 8), (1.5, 8), (8,), (2**62, 8)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                ASMPlan(shape, (1e-6, 1e-6))
        for pitch in ((0., 1e-6), (-1., 1e-6), (np.nan, 1e-6), (np.inf, 1e-6)):
            with self.subTest(pitch=pitch), self.assertRaises(ValueError):
                ASMPlan((8, 16), pitch)
        for profile in ({'typo': 1}, {'aimThreads': 0}, {'groupedBatch': [-1, 0]},
                        {'implementation': 'explicit', 'prune': True},
                        {'implementation': 'explicit', 'transfer': 'fused'}):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                ASMPlan((8, 16), (1e-6, 1e-6), tuning_profile=profile)
        for profile in self.profiles():
            with ASMPlan((8, 16), (1e-6, 1e-6), tuning_profile=profile) as plan:
                x, out = cp.ones(plan.shape, cp.complex64), cp.empty(plan.shape, cp.complex64)
                with self.assertRaises(ValueError):
                    plan.execute(x, out)
                with self.assertRaises(ValueError):
                    plan.prepare_transfer(cp.ones(plan.shape, cp.complex64))
                plan.prepare_transfer(cp.ones(plan.padded_shape, cp.complex64))
                for bad in (cp.ones(plan.shape, cp.complex128), x[:, ::-1], np.ones(plan.shape, np.complex64)):
                    with self.assertRaises(ValueError):
                        plan.execute(bad, out)
                with self.assertRaises(ValueError):
                    plan.execute(x, out, z=0., wavelength=532e-9)
            plan.close()  # Repeated close is harmless.

    def test_null_native_configuration_is_reported(self):
        from pyvkfft.cuda import VkFFTApp, _types
        # Checked size arithmetic can reject before allocating a configuration.
        # A ctypes null pointer is not Python None and must never be dereferenced
        # during construction or passed to free_config during destruction.
        with patch.object(VkFFTApp, 'make_config', return_value=_types.vkfft_config_p()):
            with self.assertRaises(RuntimeError):
                VkFFTApp((8, 16), np.complex64)

    def test_initialized_fft_scratch_is_reported(self):
        from pyvkfft.cuda import VkFFTApp
        context_anchor = cp.empty(1, cp.complex64)
        # A two-upload ordered transform allocates scratch during planning.
        # The initial configuration is not the authoritative allocation record.
        app = VkFFTApp((4, 32768), np.complex64, ndim=1, inplace=True,
                       disableReorderFourStep=False)
        config = app.app.contents.configuration
        self.assertTrue(config.allocateTempBuffer)
        self.assertGreater(int(config.tempBufferSize[0]), 0)
        self.assertEqual(int(app.tmp_buffer_nbytes), int(config.tempBufferSize[0]))

    def test_evanescent_cutoff_and_negative_distance(self):
        # Nyquist x is exactly at the propagating cutoff at wavelength=2*dx.
        # Adjacent wavelengths exercise either side without a loose cutoff mask.
        shape, pitch = (16, 32), (0.3e-6, 0.25e-6)
        rng = np.random.default_rng(410)
        source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
        for profile in self.profiles(dynamic=True):
            with ASMPlan(shape, pitch, mode='dynamic', tuning_profile=profile) as plan:
                x, out = cp.asarray(source), cp.empty(shape, cp.complex64)
                for wavelength in (499.999e-9, 500e-9, 500.001e-9):
                    for z in (0., -1e-6, 1e-6):
                        transfer = reference_transfer(plan.padded_shape, pitch, z, wavelength)
                        plan.execute(x, out, z=z, wavelength=wavelength)
                        self.assert_close(out, reference_propagate(source, transfer))

    def test_coefficients_independent_reference(self):
        code = CUDA_HELPERS + r'''
        extern "C" __global__ void coefficients(float2* out, double z, double wavelength, int mask) {
            int i=blockIdx.x*blockDim.x+threadIdx.x;
            if(i<32*64) out[i]=asm_coefficient(i/64,i%64,32,64,0.3e-6,0.25e-6,z,wavelength,mask);
        }
        '''
        kernel = cp.RawKernel(code, 'coefficients')
        out = cp.empty((32, 64), cp.complex64)
        for mask in (False, True):
            for z, wavelength in ((0., 500e-9), (-1e-6, 499.999e-9), (.1, 633e-9)):
                kernel((8,), (256,), (out, np.float64(z), np.float64(wavelength), np.int32(mask)))
                expected = reference_transfer(out.shape, (.3e-6, .25e-6), z, wavelength, mask)
                self.assert_close(out, expected)

    def test_compact_spatial_filter_control(self):
        shape = (24, 32)
        source = np.exp(1j*np.arange(np.prod(shape)).reshape(shape)/17.).astype(np.complex64)
        spatial = np.zeros(tuple(2*n for n in shape), np.complex128)
        spatial[0, 0], spatial[0, 1], spatial[1, 0] = .5, .25, .25
        transfer = np.fft.fft2(spatial).astype(np.complex64)
        expected = reference_propagate(source, transfer)
        for profile in self.profiles():
            with ASMPlan(shape, (1e-6, 1e-6), tuning_profile=profile) as plan:
                plan.prepare_transfer(cp.asarray(transfer))
                out = cp.empty(shape, cp.complex64)
                plan.execute(cp.asarray(source), out)
                self.assert_close(out, expected)

    @unittest.skipUnless(os.environ.get('PYVKFFT_SIZE64_LIBRARY'), 'isolated 64-bit wrapper not configured')
    def test_checked_wrapper_size_arithmetic(self):
        from pyvkfft.cuda import _vkfft_cuda
        context_anchor = cp.zeros(1)  # make_config queries the current driver context.
        library = ctypes.CDLL(os.environ['PYVKFFT_SIZE64_LIBRARY'])
        for name in ('make_config', 'free_config'):
            getattr(library, name).argtypes = getattr(_vkfft_cuda, name).argtypes
            getattr(library, name).restype = getattr(_vkfft_cuda, name).restype
        dimensions = library.vkfft_max_fft_dimensions()
        zeros = np.zeros(dimensions, np.int64)
        for shape in ((65536, 32768), (65536, 65536), (2**62, 8), (-1, 8), (0, 8)):
            size = np.ones(dimensions, np.int64)
            size[:2] = shape
            config = library.make_config(size, 2, 1, 0, 0, 1, 4, 0, 0, 0, 0, -1, -1, 0,
                                         1, zeros, -1, -1, -1, -1, -1, -1, -1, zeros,
                                         -1, 0, 0, 0, 1, 0)
            if 0 < shape[0] < 2**60:
                self.assertTrue(config)
                try:
                    self.assertEqual(config.contents.bufferSize[0], shape[0]*shape[1]*8)
                finally:
                    library.free_config(config)
            else:
                self.assertFalse(config)


if __name__ == '__main__':
    unittest.main()
