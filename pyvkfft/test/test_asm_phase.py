"""Public cached-phase ASM regression tests, including exact cache invalidation.

Run directly with CUDA_VISIBLE_DEVICES and PYVKFFT_ASM_PHASE_LIBRARY set.
PYVKFFT_ASM_LIBRARY is optional and is used only by the wrong-backend check.
Set PYVKFFT_ASM_SYMMETRIC_LIBRARY to include compressed-table and reuse tests.
"""
import os
from math import prod as math_prod
import unittest
from unittest.mock import patch

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from pyvkfft.asm import ASMPlan

if __package__:
    from .test_asm import reference_propagate, reference_transfer
else:
    from test_asm import reference_propagate, reference_transfer


@unittest.skipIf(cp is None or not os.environ.get('PYVKFFT_ASM_PHASE_LIBRARY'),
                 'CuPy or optional phase backend unavailable')
class TestCachedPhaseASM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        except Exception as exc:
            raise unittest.SkipTest(f'CUDA unavailable: {exc}')

    def assert_reference(self, source, output, shape, pitch, z, wavelength, stream):
        transfer = reference_transfer(tuple(2*n for n in shape), pitch, z, wavelength)
        expected = reference_propagate(source, transfer)
        actual = output.get(stream=stream)
        self.assertTrue(np.isfinite(actual).all())
        norm = np.linalg.norm(expected)
        if norm:
            self.assertLess(np.linalg.norm(actual-expected)/norm, 2e-5)
            self.assertLess(np.max(np.abs(actual-expected))/np.max(np.abs(expected)), 1e-4)
        else:
            np.testing.assert_array_equal(actual, expected)

    def test_reference_and_invalidation(self):
        """48 independent complex128 reference cases, including evanescence."""
        rng = np.random.default_rng(112)
        parameters = ((.01, 532e-9), (.1, 532e-9), (-.1, 450e-9),
                      (0., 450e-9), (1., 633e-9), (.1, 532e-9))
        expected_preparations = (1, 1, 2, 2, 3, 4)
        for shape in ((8, 16), (15, 21), (32, 8192), (8192, 32)):
            for pitch in ((6.4e-6, 6.4e-6), (.2e-6, .25e-6)):
                with self.subTest(shape=shape, pitch=pitch):
                    source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
                    stream = cp.cuda.Stream(non_blocking=True)
                    with stream, ASMPlan(shape, pitch, mode='dynamic', stream=stream,
                            tuning_profile={'transfer': 'cached_phase', 'prune': True,
                                            'groupedBatch': [0, 32]}) as plan:
                        x = cp.asarray(source)
                        output = cp.empty_like(x)
                        allocated = cp.get_default_memory_pool().used_bytes()
                        for (z, wavelength), count in zip(parameters, expected_preparations):
                            plan._work.fill(cp.nan)
                            plan.execute(x, output, z=z, wavelength=wavelength)
                            self.assert_reference(source, output, shape, pitch, z, wavelength, stream)
                            self.assertEqual(plan.info['phase_preparations'], count)
                            self.assertEqual(plan.info['prepared_wavelength'], wavelength)
                        self.assertEqual(plan.info['axis_order'], 'row')
                        self.assertEqual(plan.info['phase_table_bytes'], np.prod(plan.padded_shape)*8)
                        self.assertEqual(cp.get_default_memory_pool().used_bytes(), allocated)
                        np.testing.assert_array_equal(x.get(stream=stream), source)

    def test_column_order_and_phase_only_environment(self):
        rng = np.random.default_rng(901)
        for shape in ((15, 21), (32, 48)):
            source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
            pitch = (.2e-6, .25e-6)
            with patch.dict(os.environ, {'PYVKFFT_ASM_LIBRARY': ''}):
                with ASMPlan(shape, pitch, mode='dynamic', tuning_profile={
                        'transfer': 'cached_phase', 'axis_order': 'column', 'prune': True}) as plan:
                    x, output = cp.asarray(source), cp.empty(shape, cp.complex64)
                    for z, wavelength in ((.01, 532e-9), (-.1, 450e-9), (.1, 532e-9)):
                        plan._work.fill(cp.nan)
                        plan.execute(x, output, z=z, wavelength=wavelength)
                        self.assert_reference(source, output, shape, pitch, z, wavelength, plan.stream)
                    self.assertEqual(plan.info['axis_order'], 'column')
                    self.assertEqual(plan.info['phase_preparations'], 3)
                    self.assertEqual(tuple(plan.info['shape']), shape)
                    self.assertTrue(plan.info['compact_buffer_reused'])
                    self.assertEqual(plan.info['compact_transpose_bytes'], source.nbytes)

    def test_invalid_configuration_and_capability(self):
        profile = {'implementation': 'native', 'transfer': 'cached_phase'}
        for mode, bandlimit in (('static', 'none'), ('dynamic', 'rectangular')):
            with self.assertRaises(ValueError):
                ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode=mode, bandlimit=bandlimit,
                        tuning_profile=profile)
        with patch.dict(os.environ, {'PYVKFFT_ASM_PHASE_LIBRARY': ''}):
            with self.assertRaisesRegex(RuntimeError, 'PYVKFFT_ASM_PHASE_LIBRARY'):
                ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode='dynamic', tuning_profile=profile)
        standard = os.environ.get('PYVKFFT_ASM_LIBRARY')
        if standard:
            with patch.dict(os.environ, {'PYVKFFT_ASM_PHASE_LIBRARY': standard}):
                with self.assertRaisesRegex(RuntimeError, 'not a cached-phase'):
                    ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode='dynamic', tuning_profile=profile)

    def test_phase_library_rejected_by_ordinary_native(self):
        phase = os.environ['PYVKFFT_ASM_PHASE_LIBRARY']
        with patch.dict(os.environ, {'PYVKFFT_ASM_LIBRARY': phase}):
            for mode, transfer in (('static', 'materialized'),
                                   ('dynamic', 'materialized'), ('dynamic', 'fused')):
                with self.subTest(mode=mode, transfer=transfer):
                    with self.assertRaisesRegex(RuntimeError, 'selects a cached-phase backend'):
                        ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode=mode,
                                tuning_profile={'implementation': 'native', 'transfer': transfer})

    def test_runtime_validation_and_close(self):
        plan = ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode='dynamic',
                       tuning_profile={'transfer': 'cached_phase'})
        x, output = cp.zeros(plan.shape, cp.complex64), cp.empty(plan.shape, cp.complex64)
        with self.assertRaises(ValueError):
            plan.execute(x, x, z=.1, wavelength=532e-9)
        for z, wavelength in ((None, 532e-9), (.1, None), (float('nan'), 532e-9), (.1, 0.)):
            with self.assertRaises(ValueError):
                plan.execute(x, output, z=z, wavelength=wavelength)
        plan.execute(x, output, z=-.1, wavelength=450e-9)
        np.testing.assert_array_equal(output.get(stream=plan.stream), np.zeros(plan.shape, np.complex64))
        plan.close()
        plan.close()
        with self.assertRaises(RuntimeError):
            plan.execute(x, output, z=.1, wavelength=532e-9)


@unittest.skipIf(cp is None or not os.environ.get('PYVKFFT_ASM_SYMMETRIC_LIBRARY'),
                 'CuPy or optional symmetric backend unavailable')
class TestSymmetricCachedPhaseASM(unittest.TestCase):
    """Exact folded tables, public contracts, and private column-buffer reuse."""

    assert_reference = TestCachedPhaseASM.assert_reference

    @classmethod
    def setUpClass(cls):
        try:
            cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        except Exception as exc:
            raise unittest.SkipTest(f'CUDA unavailable: {exc}')

    @unittest.skipUnless(os.environ.get('PYVKFFT_ASM_PHASE_LIBRARY'),
                         'Full phase backend required for bitwise table comparison')
    def test_reference_tables_and_cache_invalidation(self):
        """96 independent references, including singleton/Nyquist and large z."""
        from pyvkfft.asm_symmetric import table_shape
        rng = np.random.default_rng(951)
        parameters = ((0., 532e-9), (.01, 532e-9), (-.1, 633e-9),
                      (1000., 450e-9), (-1000., 450e-9), (.1, 532e-9))
        counts = (1, 1, 2, 3, 3, 4)
        shapes = ((1, 1), (1, 15), (15, 1), (15, 21), (32, 48),
                  (8, 1024), (32, 8192), (8192, 32))
        for shape in shapes:
            for pitch in ((6.4e-6, 7.1e-6), (.2e-6, .25e-6)):
                with self.subTest(shape=shape, pitch=pitch):
                    source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
                    stream = cp.cuda.Stream(non_blocking=True)
                    profile = {'transfer': 'cached_phase_symmetric', 'prune': True,
                               'groupedBatch': [0, 32]}
                    with stream, ASMPlan(shape, pitch, mode='dynamic', stream=stream,
                            tuning_profile=profile) as plan, ASMPlan(shape, pitch, mode='dynamic',
                            stream=stream, tuning_profile=dict(profile, transfer='cached_phase')) as full:
                        x = cp.asarray(source)
                        output, reference_output = cp.empty_like(x), cp.empty_like(x)
                        self.assertEqual(tuple(plan.info['phase_table_shape']), table_shape(shape))
                        self.assertEqual(plan.info['phase_table_bytes'], math_prod(table_shape(shape))*8)
                        self.assertEqual(plan.info['symmetric_abi'], 1)
                        self.assertEqual(plan.info['symmetric_layout'], 1)
                        ny, nx, ay, by, ax, bx = map(int, full._delegate._layout)

                        def folded_rank(length, a, b):
                            coordinate = np.arange(length)
                            frequency = (coordinate % a)*b+coordinate//a
                            magnitude = np.minimum(frequency, length-frequency)
                            digit, row = magnitude % b, magnitude//b
                            return np.where(row < a//2, digit*(a//2)+row, (a//2)*b+digit)

                        y, xx = folded_rank(ny, ay, by), folded_rank(nx, ax, bx)
                        allocated = cp.get_default_memory_pool().used_bytes()
                        for (z, wavelength), count in zip(parameters, counts):
                            plan._work.fill(cp.nan)
                            full._work.fill(cp.nan)
                            full.execute(x, reference_output, z=z, wavelength=wavelength)
                            with patch('cupy.empty', side_effect=AssertionError('Execution allocation')), \
                                    patch('cupy.empty_like', side_effect=AssertionError('Execution allocation')):
                                plan.execute(x, output, z=z, wavelength=wavelength)
                            stream.synchronize()
                            table = plan._delegate._phase.get(stream=stream)
                            expanded = table[y[:, None], xx[None, :]]
                            original = full._delegate._phase.get(stream=stream)
                            np.testing.assert_array_equal(expanded.view(np.uint64), original.view(np.uint64))
                            np.testing.assert_array_equal(output.get(stream=stream), reference_output.get(stream=stream))
                            self.assert_reference(source, output, shape, pitch, z, wavelength, stream)
                            self.assertEqual(plan.info['phase_preparations'], count)
                            self.assertEqual(plan.info['prepared_wavelength'], wavelength)
                        self.assertEqual(cp.get_default_memory_pool().used_bytes(), allocated)
                        np.testing.assert_array_equal(x.get(stream=stream), source)

    def test_column_private_reuse_and_symmetric_only_environment(self):
        rng = np.random.default_rng(602)
        for shape in ((15, 21), (32, 8192), (8192, 32)):
            with self.subTest(shape=shape):
                source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
                pitch = (.2e-6, .25e-6)
                stream = cp.cuda.Stream(non_blocking=True)
                with patch.dict(os.environ, {'PYVKFFT_ASM_LIBRARY': '', 'PYVKFFT_ASM_PHASE_LIBRARY': ''}):
                    with stream, ASMPlan(shape, pitch, mode='dynamic', stream=stream,
                            tuning_profile={'transfer': 'cached_phase_symmetric',
                                            'axis_order': 'column', 'prune': True}) as plan:
                        x, output = cp.asarray(source), cp.empty(shape, cp.complex64)
                        for z, wavelength in ((.01, 532e-9), (-.1, 633e-9), (1000., 450e-9), (0., 532e-9)):
                            plan._work.fill(cp.nan)
                            with patch('cupy.empty', side_effect=AssertionError('Execution allocation')), \
                                    patch('cupy.empty_like', side_effect=AssertionError('Execution allocation')):
                                plan.execute(x, output, z=z, wavelength=wavelength)
                            self.assert_reference(source, output, shape, pitch, z, wavelength, stream)
                        self.assertTrue(plan.info['compact_buffer_reused'])
                        self.assertEqual(plan.info['compact_transpose_bytes'], source.nbytes)
                        self.assertEqual(plan.info['phase_preparations'], 4)
                        np.testing.assert_array_equal(x.get(stream=stream), source)
                        with self.assertRaises(ValueError):
                            plan.execute(x, x, z=.1, wavelength=532e-9)

    def test_capability_and_runtime_contract(self):
        profile = {'transfer': 'cached_phase_symmetric', 'implementation': 'native'}
        for mode, mask in (('static', 'none'), ('dynamic', 'rectangular')):
            with self.assertRaises(ValueError):
                ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode=mode, bandlimit=mask, tuning_profile=profile)
        with patch.dict(os.environ, {'PYVKFFT_ASM_SYMMETRIC_LIBRARY': ''}):
            with self.assertRaisesRegex(RuntimeError, 'PYVKFFT_ASM_SYMMETRIC_LIBRARY'):
                ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode='dynamic', tuning_profile=profile)
        for variable in ('PYVKFFT_ASM_LIBRARY', 'PYVKFFT_ASM_PHASE_LIBRARY'):
            wrong = os.environ.get(variable)
            if wrong:
                with patch.dict(os.environ, {'PYVKFFT_ASM_SYMMETRIC_LIBRARY': wrong}):
                    with self.assertRaisesRegex(RuntimeError, 'not a symmetric ASM backend'):
                        ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode='dynamic', tuning_profile=profile)
        with patch.dict(os.environ, {'PYVKFFT_ASM_PHASE_LIBRARY': os.environ['PYVKFFT_ASM_SYMMETRIC_LIBRARY']}):
            with self.assertRaisesRegex(RuntimeError, 'not a full cached-phase backend'):
                ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode='dynamic',
                        tuning_profile={'transfer': 'cached_phase'})
        plan = ASMPlan((8, 16), (6.4e-6, 6.4e-6), mode='dynamic', tuning_profile=profile)
        storage = cp.zeros(math_prod(plan.shape)+1, cp.complex64)
        x = storage[:-1].reshape(plan.shape)
        overlapping = storage[1:].reshape(plan.shape)
        output = cp.empty_like(x)
        with self.assertRaises(ValueError):
            plan.execute(x, overlapping, z=.1, wavelength=532e-9)
        with self.assertRaises(ValueError):
            plan._delegate._execute_arrays(x, x.view(), z=.1, wavelength=532e-9, private_alias=True)
        for z, wavelength in ((None, 532e-9), (.1, None), (float('nan'), 532e-9), (.1, 0.)):
            with self.assertRaises(ValueError):
                plan.execute(x, output, z=z, wavelength=wavelength)
        plan.execute(x, output, z=-.1, wavelength=450e-9)
        np.testing.assert_array_equal(output.get(stream=plan.stream), np.zeros(plan.shape, np.complex64))
        plan.close()
        plan.close()
        with self.assertRaises(RuntimeError):
            plan.execute(x, output, z=.1, wavelength=532e-9)


if __name__ == '__main__':
    unittest.main()
