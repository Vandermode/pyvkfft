"""Independent checks for column-first composition; run directly with CUDA visibility set."""
import os
import unittest
from unittest.mock import patch
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from pyvkfft.asm_axis import ColumnFirstASMPlan
if __package__:
    from .test_asm import reference_propagate, reference_transfer
else:
    from test_asm import reference_propagate, reference_transfer


@unittest.skipUnless(os.environ.get('PYVKFFT_ASM_LIBRARY'), 'isolated native library required')
@unittest.skipIf(cp is None, 'CuPy unavailable')
class TestColumnFirstASM(unittest.TestCase):
    def test_direct_packing_arbitrary_transfer(self):
        rng = np.random.default_rng(619)
        for shape in ((15, 21), (32, 8192), (8192, 32), (45, 8192)):
            source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
            padded = tuple(2*n for n in shape)
            transfer = (rng.normal(size=padded)+1j*rng.normal(size=padded)).astype(np.complex64)
            expected = reference_propagate(source, transfer)
            with ColumnFirstASMPlan(shape, (6.4e-6, 7.1e-6),
                                    tuning_profile={'implementation': 'native', 'prune': True}) as plan:
                x, out = cp.asarray(source), cp.empty(shape, cp.complex64)
                for order in ('fft', 'centered'):
                    supplied = np.fft.fftshift(transfer) if order == 'centered' else transfer
                    gpu_h = cp.asarray(supplied)
                    with (patch.object(cp, 'empty', side_effect=AssertionError('preparation allocated')),
                          patch.object(cp, 'empty_like', side_effect=AssertionError('preparation allocated'))):
                        plan.prepare_transfer(gpu_h, order=order)
                    plan._plan._work.fill(cp.nan)
                    plan.execute(x, out)
                    actual = cp.asnumpy(out)
                    self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), 2e-5)
                    np.testing.assert_array_equal(cp.asnumpy(gpu_h), supplied)

    def test_static_dynamic_orders_and_storage(self):
        rng = np.random.default_rng(814)
        for shape in ((15, 21), (32, 8192), (8192, 32)):
            source = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
            pitch = (0.3e-6, 0.25e-6)
            stream = cp.cuda.Stream(non_blocking=True)
            for mode in ('static', 'dynamic'):
                profile = {'implementation': 'native', 'prune': True,
                           'transfer': 'fused' if mode == 'dynamic' else 'materialized'}
                with stream, ColumnFirstASMPlan(shape, pitch, mode=mode, bandlimit='rectangular',
                                                stream=stream, tuning_profile=profile) as plan:
                    self.assertIs(plan._input, plan._output)
                    self.assertTrue(plan.info['compact_buffer_reused'])
                    self.assertEqual(plan.info['compact_transpose_bytes'], np.prod(shape)*8)
                    x, out = cp.asarray(source), cp.empty(shape, cp.complex64)
                    for index, (z, wavelength) in enumerate(((0., 500e-9), (1e-6, 532e-9), (-1e-6, 633e-9))):
                        transfer = reference_transfer(plan.padded_shape, pitch, z, wavelength, True)
                        expected = reference_propagate(source, transfer)
                        if mode == 'static':
                            order = 'centered' if index % 2 else 'fft'
                            plan.prepare_transfer(cp.asarray(np.fft.fftshift(transfer) if order == 'centered' else transfer,
                                                             dtype=cp.complex64), order=order)
                        plan._plan._work.fill(cp.nan)
                        kwargs = {'z': z, 'wavelength': wavelength} if mode == 'dynamic' else {}
                        plan.execute(x, out, **kwargs)
                        actual = cp.asnumpy(out)
                        self.assertTrue(np.isfinite(actual).all())
                        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), 2e-5)
                        self.assertLess(np.max(np.abs(actual-expected))/np.max(np.abs(expected)), 1e-4)
                        np.testing.assert_array_equal(cp.asnumpy(x), source)
                    stream.synchronize()
                    used = cp.get_default_memory_pool().used_bytes()
                    plan.execute(x, out, **kwargs)
                    stream.synchronize()
                    self.assertEqual(used, cp.get_default_memory_pool().used_bytes())
                    with self.assertRaises(ValueError):
                        plan.execute(x, x, **kwargs)
                    with self.assertRaises(ValueError):
                        plan._plan.execute(plan._input, plan._input, **kwargs)
                self.assertEqual(plan.info['axis_order'], 'column')
                with self.assertRaises(RuntimeError):
                    plan.execute(x, out, **kwargs)


if __name__ == '__main__':
    unittest.main()
