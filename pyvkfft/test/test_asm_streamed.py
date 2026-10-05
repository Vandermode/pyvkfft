"""Independent references and CUDA contracts for the optional streamed plan."""
from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

from pyvkfft.asm_streamed import StreamedASMPlan
from test_asm import reference_propagate, reference_transfer


@unittest.skipUnless(cp is not None and os.environ.get('PYVKFFT_STREAMED_LIBRARY'),
                     'CuPy and an isolated streamed FFT library are required')
class TestStreamedASM(unittest.TestCase):
    def assert_close(self, actual, expected):
        actual = cp.asnumpy(actual)
        self.assertTrue(np.isfinite(actual).all())
        norm = np.linalg.norm(expected)
        if norm:
            self.assertLess(np.linalg.norm(actual-expected)/norm, 2e-5)
            self.assertLess(np.max(np.abs(actual-expected))/np.max(np.abs(expected)), 1e-4)
        else:
            self.assertEqual(np.max(np.abs(actual)), 0.)

    def test_reference_and_orientations(self):
        rng = np.random.default_rng(610)
        pitch = (0.3e-6, 0.25e-6)
        for shape, tile in (((15, 21), 17), ((32, 8192), 128), ((8192, 32), 17), ((45, 8192), 512)):
            host = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
            for axis in ('row', 'column'):
                for mask in ('none', 'rectangular'):
                    stream = cp.cuda.Stream(non_blocking=True)
                    with stream, StreamedASMPlan(shape, pitch, mask, stream, tile, axis_order=axis) as plan:
                        source, dest = cp.asarray(host), cp.empty(shape, cp.complex64)
                        for z, wavelength in ((0., 500e-9), (1e-6, 532e-9), (-1e-6, 633e-9), (1., 532e-9)):
                            with self.subTest(shape=shape, axis=axis, mask=mask, z=z, wavelength=wavelength):
                                expected = reference_propagate(host, reference_transfer(
                                    plan.padded_shape, pitch, z, wavelength, mask == 'rectangular'))
                                plan._extra.fill(cp.nan)
                                plan._scratch.fill(19+23j)
                                plan.execute(source, dest, z=z, wavelength=wavelength)
                                self.assert_close(dest, expected)
                                np.testing.assert_array_equal(cp.asnumpy(source), host)
                        expected_workspace = plan._extra.nbytes + plan._scratch.nbytes + plan.info['temporary_bytes']
                        self.assertEqual(plan.info['workspace_bytes'], expected_workspace)
                        self.assertEqual(plan.info['total_operator_bytes'], expected_workspace + 2*host.nbytes)

    def test_multi_upload_roundtrip_and_propagation(self):
        rng = np.random.default_rng(611)
        pitch = (6.4e-6, 7.1e-6)
        for shape in ((16, 16384), (32768, 8)):
            host = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
            source, dest = cp.asarray(host), cp.empty(shape, cp.complex64)
            with StreamedASMPlan(shape, pitch, tile_columns=16) as plan:
                self.assertIn(2, (plan.info['row_fft']['uploads'], plan.info['column_fft']['uploads']))
                for z in (0., -.01):
                    expected = reference_propagate(host, reference_transfer(plan.padded_shape, pitch, z, 532e-9))
                    plan.execute(source, dest, z=z, wavelength=532e-9)
                    self.assert_close(dest, expected)

    def test_contract_and_no_execute_allocation(self):
        shape = (15, 21)
        stream = cp.cuda.Stream(non_blocking=True)
        for axis in ('row', 'column'):
            with stream:
                source = cp.ones(shape, cp.complex64)
                dest = cp.empty(shape, cp.complex64)
                plan = StreamedASMPlan(shape, (6.4e-6, 7.1e-6), stream=stream,
                                       tile_columns=17, tile_rows=4, axis_order=axis)
                plan.execute(source, dest, z=.01, wavelength=532e-9)
                stream.synchronize()
                used = cp.get_default_memory_pool().used_bytes()
                with (patch.object(cp, 'empty', side_effect=AssertionError('execute allocated')),
                      patch.object(cp, 'empty_like', side_effect=AssertionError('execute allocated')),
                      patch.object(cp, 'asarray', side_effect=AssertionError('execute allocated')),
                      patch.object(cp, 'ascontiguousarray', side_effect=AssertionError('execute allocated'))):
                    self.assertIs(plan.execute(source, dest, z=-.1, wavelength=633e-9), dest)
                stream.synchronize()
                self.assertEqual(used, cp.get_default_memory_pool().used_bytes())
                with self.assertRaises(ValueError):
                    plan.execute(source, source, z=0., wavelength=532e-9)
                overlap = cp.empty(source.size+1, cp.complex64)
                with self.assertRaises(ValueError):
                    plan.execute(overlap[:-1].reshape(shape), overlap[1:].reshape(shape), z=0., wavelength=532e-9)
                with self.assertRaises(ValueError):
                    plan.execute(source, plan._extra.reshape(shape), z=0., wavelength=532e-9)
                for bad in (cp.ones(shape, cp.complex128), source[:, ::-1], np.ones(shape, np.complex64)):
                    with self.assertRaises(ValueError):
                        plan.execute(bad, dest, z=0., wavelength=532e-9)
                for z, wavelength in ((None, 532e-9), (float('nan'), 532e-9), (0., 0.), (0., float('inf'))):
                    with self.assertRaises(ValueError):
                        plan.execute(source, dest, z=z, wavelength=wavelength)
                with patch.object(cp.cuda, 'Device', return_value=SimpleNamespace(id=plan.device+1)):
                    with self.assertRaisesRegex(RuntimeError, 'device'):
                        plan.execute(source, dest, z=0., wavelength=532e-9)
                plan.close()
                plan.close()
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    plan.execute(source, dest, z=0., wavelength=532e-9)

    def test_configuration_and_native_size_checks(self):
        if os.environ.get('PYVKFFT_STREAMED_FFI_LIBRARY'):
            from pyvkfft.asm_streamed import _streamed_library
            with self.assertRaisesRegex(RuntimeError, 'JAX FFI backend'):
                _streamed_library(os.environ['PYVKFFT_STREAMED_FFI_LIBRARY'])
        for shape in ((0, 8), (True, 8), (8., 8), (1 << 62, 8)):
            with self.assertRaises(ValueError):
                StreamedASMPlan(shape, (6.4e-6, 6.4e-6))
        for options in ({'tile_columns': 0}, {'tile_rows': -1}, {'axis_order': 'auto'}, {'bandlimit': 'circular'}):
            with self.assertRaises(ValueError):
                StreamedASMPlan((8, 16), (6.4e-6, 6.4e-6), **options)
        for pitch in ((0., 1.), (float('nan'), 1.)):
            with self.assertRaises(ValueError):
                StreamedASMPlan((8, 16), pitch)
        with patch.dict(os.environ, {'PYVKFFT_STREAMED_LIBRARY': ''}):
            with self.assertRaisesRegex(RuntimeError, 'PYVKFFT_STREAMED_LIBRARY'):
                StreamedASMPlan((8, 16), (6.4e-6, 6.4e-6))
        with StreamedASMPlan((8, 16), (6.4e-6, 6.4e-6)) as plan:
            self.assertEqual(plan._native.streamed_fft_capabilities(), 1)
            for length, batch in ((0, 1), (1, 0), (1 << 62, 8), ((1 << 64)-1, 1)):
                self.assertFalse(plan._native.streamed_fft_create(length, batch, plan.stream.ptr))
            self.assertNotEqual(plan._native.streamed_fft_execute(None, None, 0), 0)

    def test_packed_transfer_partial_tiles_and_transpose(self):
        rng = np.random.default_rng(818)
        for shape in ((17, 23), (32, 18000), (18000, 32)):
            h, w = shape
            host = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
            real = rng.integers(-32768, 32768, (2*h, 2*w), dtype=np.int32)
            imag = rng.integers(-32768, 32768, (2*h, 2*w), dtype=np.int32)
            payload = cp.asarray((real.astype(np.uint32)&65535)|(imag.astype(np.uint32)<<16))
            scale = np.float32(3.5/32767)
            transfer = (real+1j*imag).astype(np.complex64)*scale
            source, output = cp.asarray(host), cp.empty(shape, cp.complex64)
            with StreamedASMPlan(shape, (1., 1.), transfer_mode='packed',
                                 tile_columns=127, tile_rows=31) as plan:
                for reverse in (False, True):
                    table = np.roll(np.flip(transfer), (1, 1), (0, 1)) if reverse else transfer
                    expected = reference_propagate(host, table)
                    plan.execute(source, output, transfer=payload, scale=scale, transpose=reverse)
                    self.assert_close(output, expected)
                np.testing.assert_array_equal(cp.asnumpy(source), host)
                with self.assertRaises(ValueError):
                    plan.execute(source, output, transfer=payload.astype(cp.int32))
                alias = payload.view(cp.complex64).ravel()[:h*w].reshape(shape)
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    plan.execute(source, alias, transfer=payload)

    @unittest.skipUnless(os.environ.get('PYVKFFT_TEST_LARGE_1D'), 'Explicit large 1-D validation required')
    def test_three_upload_frequency_permutation_and_roundtrip(self):
        from pyvkfft.asm_streamed import _streamed_library
        import json
        library, _ = _streamed_library(os.environ['PYVKFFT_STREAMED_LIBRARY'])
        length = 1 << 26
        source = cp.empty(length, cp.complex64)
        fill = cp.RawKernel(r'''extern "C" __global__ void fill(float2* x, long long n) {
            long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
            if(i<n) {
                unsigned int a=(unsigned int)i*747796405u+2891336453u;
                unsigned int b=(a^(a>>16))*277803737u;
                x[i]=make_float2((int)(a&65535u)/32768.f-1.f,(int)(b&65535u)/32768.f-1.f);
            }
        }''', 'fill')
        fill(((length+255)//256,), (256,), (source, np.int64(length)))
        actual = source.copy()
        handle = library.streamed_fft_create(length, 1, cp.cuda.get_current_stream().ptr)
        self.assertTrue(handle)
        try:
            info = json.loads(library.streamed_fft_info(handle))
            self.assertEqual(info['uploads'], 3)
            self.assertEqual(library.streamed_fft_execute(handle, actual.data.ptr, 0), 0)
            expected = cp.fft.fft(source)
            index = cp.arange(1, length, 313, dtype=cp.int64)  # exclude DC
            a, b, c = info['axis_split']
            natural = (index%a)*b*c+(index//a%b)*c+index//(a*b)
            error = float(cp.linalg.norm(actual[index]-expected[natural])/cp.linalg.norm(expected[natural]))
            self.assertLess(error, 2e-5)
            self.assertEqual(library.streamed_fft_execute(handle, actual.data.ptr, 1), 0)
            error = float(cp.linalg.norm(actual-source)/cp.linalg.norm(source))
            self.assertLess(error, 2e-5)
        finally:
            library.streamed_fft_destroy(handle)

    def test_serialized_host_submission(self):
        shape, pitch = (15, 21), (6.4e-6, 7.1e-6)
        rng = np.random.default_rng(612)
        hosts = [(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64) for _ in range(2)]
        with StreamedASMPlan(shape, pitch, tile_columns=17, tile_rows=4) as plan:
            sources = [cp.asarray(host) for host in hosts]
            outputs = [cp.empty(shape, cp.complex64) for _ in hosts]
            def submit(index):
                with cp.cuda.Device(plan.device):
                    for _ in range(3):
                        plan.execute(sources[index], outputs[index], z=.01*(index+1), wavelength=532e-9)
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(submit, (0, 1)))
            plan.stream.synchronize()
            for index in range(2):
                expected = reference_propagate(hosts[index], reference_transfer(
                    plan.padded_shape, pitch, .01*(index+1), 532e-9))
                self.assert_close(outputs[index], expected)


if __name__ == '__main__':
    unittest.main()
