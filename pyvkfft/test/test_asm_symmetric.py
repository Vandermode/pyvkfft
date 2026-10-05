"""Static analytic ASM quadrant storage: accuracy, memory, and API boundaries.

Run directly with PYVKFFT_ASM_SYMMETRIC_LIBRARY set to the optional family.
PYVKFFT_ASM_LIBRARY enables exact comparisons with ordinary full-table plans.
"""
import os
import unittest
from unittest.mock import patch

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from pyvkfft.asm import ASMPlan
from pyvkfft.asm_symmetric import table_shape

if __package__:
    from .test_asm import reference_propagate, reference_transfer
else:
    from test_asm import reference_propagate, reference_transfer


class TestSymmetricLayout(unittest.TestCase):
    def test_table_sizes_and_thin_shapes(self):
        self.assertEqual(table_shape((1, 1)), (2, 2))
        self.assertEqual(table_shape((8192, 1)), (8193, 2))
        self.assertEqual(table_shape((15, 21)), (16, 22))
        self.assertEqual(table_shape((8, 1022)), (9, 1023))
        self.assertEqual(table_shape((8, 1023)), (9, 1024))
        self.assertEqual(table_shape((8, 1024)), (9, 1056))
        self.assertEqual(table_shape((4096, 16384)), (4097, 16416))
        for shape in ((0, 1), (1, -1), (True, 2), (1,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                table_shape(shape)


@unittest.skipIf(cp is None or not os.environ.get('PYVKFFT_ASM_SYMMETRIC_LIBRARY'),
                 'CuPy or optional symmetric backend unavailable')
class TestStaticSymmetricASM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        except Exception as exc:
            raise unittest.SkipTest(f'CUDA unavailable: {exc}')

    def test_reference_repreparation_and_allocation(self):
        """336 complex128 reference cases, including cutoffs and signed 100 m."""
        rng = np.random.default_rng(928)
        parameters = ((0., 500e-9), (1e-6, 499.999e-9), (-1e-6, 500.001e-9),
                      (.1, 532e-9), (-1., 450e-9), (100., 450e-9), (-100., 450e-9))
        for shape in ((1, 1), (1, 15), (15, 21), (16, 32), (32, 8192), (8192, 32)):
            host = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
            for pitch in ((6.4e-6, 6.4e-6), (.3e-6, .25e-6)):
                for mask in ('none', 'rectangular'):
                    for order in ('row', 'column'):
                        with self.subTest(shape=shape, pitch=pitch, mask=mask, order=order):
                            stream = cp.cuda.Stream(non_blocking=True)
                            with stream, ASMPlan(shape, pitch, bandlimit=mask, stream=stream,
                                    tuning_profile={'transfer': 'symmetric', 'prune': True,
                                                    'axis_order': order, 'groupedBatch': [0, 32]}) as plan:
                                x, out = cp.asarray(host), cp.empty(shape, cp.complex64)
                                inner = plan if order == 'row' else plan._delegate._plan
                                native_shape = shape if order == 'row' else shape[::-1]
                                self.assertEqual(inner._transfer.shape, table_shape(native_shape))
                                self.assertEqual(plan.info['transfer_bytes'], inner._transfer.nbytes)
                                self.assertLessEqual(inner._transfer.nbytes, inner._work.nbytes)
                                allocated = cp.get_default_memory_pool().used_bytes()
                                for z, wavelength in parameters:
                                    transfer = reference_transfer(plan.padded_shape, pitch, z, wavelength,
                                                                  mask == 'rectangular')
                                    expected = reference_propagate(host, transfer)
                                    inner._transfer.fill(cp.nan)
                                    plan.prepare_asm_transfer(z=z, wavelength=wavelength)
                                    plan._work.fill(cp.nan)
                                    with patch('cupy.empty', side_effect=AssertionError('execute allocation')), \
                                         patch('cupy.empty_like', side_effect=AssertionError('execute allocation')):
                                        plan.execute(x, out)
                                    actual = out.get(stream=stream)
                                    self.assertTrue(np.isfinite(actual).all())
                                    relative_l2 = np.linalg.norm(actual-expected)/(np.linalg.norm(expected) or 1.)
                                    relative_max = np.max(np.abs(actual-expected))/(np.max(np.abs(expected)) or 1.)
                                    self.assertLess(relative_l2, 2e-5)
                                    self.assertLess(relative_max, 1e-4)
                                np.testing.assert_array_equal(x.get(stream=stream), host)
                                self.assertEqual(cp.get_default_memory_pool().used_bytes(), allocated)

    @unittest.skipUnless(os.environ.get('PYVKFFT_ASM_LIBRARY'), 'ordinary native backend unavailable')
    def test_bitwise_full_table_equivalence(self):
        rng = np.random.default_rng(901)
        for shape in ((15, 21), (32, 8192), (8192, 32)):
            source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
            for order in ('row', 'column'):
                for mask in ('none', 'rectangular'):
                    stream = cp.cuda.Stream(non_blocking=True)
                    profile = {'implementation': 'native', 'axis_order': order,
                               'prune': True, 'groupedBatch': [0, 32]}
                    with stream, ASMPlan(shape, (.3e-6, .25e-6), bandlimit=mask, stream=stream,
                                         tuning_profile=profile) as full, \
                         ASMPlan(shape, (.3e-6, .25e-6), bandlimit=mask, stream=stream,
                                 tuning_profile=dict(profile, transfer='symmetric')) as compressed:
                        x = cp.asarray(source)
                        expected, actual = cp.empty_like(x), cp.empty_like(x)
                        for z, wavelength in ((.1, 532e-9), (-.1, 450e-9), (100., 633e-9)):
                            full.prepare_asm_transfer(z=z, wavelength=wavelength)
                            compressed.prepare_asm_transfer(z=z, wavelength=wavelength)
                            full.execute(x, expected)
                            compressed.execute(x, actual)
                            np.testing.assert_array_equal(actual.get(stream=stream), expected.get(stream=stream))

    def test_configuration_and_library_rejection(self):
        for mode, implementation in (('dynamic', 'native'), ('static', 'explicit')):
            with self.assertRaises(ValueError):
                ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode=mode,
                        tuning_profile={'implementation': implementation, 'transfer': 'symmetric'})
        with patch.dict(os.environ, {'PYVKFFT_ASM_SYMMETRIC_LIBRARY': ''}):
            with self.assertRaisesRegex(RuntimeError, 'PYVKFFT_ASM_SYMMETRIC_LIBRARY'):
                ASMPlan((8, 16), (6.4e-6, 6.4e-6), tuning_profile={'transfer': 'symmetric'})
        ordinary = os.environ.get('PYVKFFT_ASM_LIBRARY')
        if ordinary:
            with patch.dict(os.environ, {'PYVKFFT_ASM_SYMMETRIC_LIBRARY': ordinary}):
                with self.assertRaisesRegex(RuntimeError, 'not a symmetric'):
                    ASMPlan((8, 16), (6.4e-6, 6.4e-6), tuning_profile={'transfer': 'symmetric'})
        symmetric = os.environ['PYVKFFT_ASM_SYMMETRIC_LIBRARY']
        with patch.dict(os.environ, {'PYVKFFT_ASM_LIBRARY': symmetric}):
            for mode, transfer in (('static', 'materialized'), ('dynamic', 'fused')):
                with self.assertRaisesRegex(RuntimeError, 'symmetric-table backend'):
                    ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode=mode,
                            tuning_profile={'implementation': 'native', 'transfer': transfer})

    def test_analytic_only_lifetime_and_validation(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            plan = ASMPlan((8, 16), (6.4e-6, 6.4e-6), stream=stream,
                           tuning_profile={'transfer': 'symmetric'})
            x, out = cp.ones(plan.shape, cp.complex64), cp.empty(plan.shape, cp.complex64)
            with self.assertRaises(ValueError):
                plan.execute(x, out)
            with self.assertRaisesRegex(ValueError, 'prepare_asm_transfer'):
                plan.prepare_transfer(cp.ones(plan.padded_shape, cp.complex64))
            for z, wavelength in ((None, 532e-9), (.1, 0.), (float('nan'), 532e-9)):
                with self.assertRaises(ValueError):
                    plan.prepare_asm_transfer(z=z, wavelength=wavelength)
            plan.prepare_asm_transfer(z=0., wavelength=532e-9)
            with self.assertRaises(ValueError):
                plan.execute(x, x)
            plan.execute(x, out)
            plan.close()
            plan.close()
            np.testing.assert_allclose(out.get(stream=stream), np.ones(plan.shape), atol=1e-6)
            with self.assertRaises(RuntimeError):
                plan.execute(x, out)


if __name__ == '__main__':
    unittest.main()
