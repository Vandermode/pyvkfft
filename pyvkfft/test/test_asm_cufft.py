"""Independent reference coverage for the experimental separable cuFFT baseline.

Run with CUDA_VISIBLE_DEVICES set to an authorized GPU before importing CuPy.
"""
import importlib.util
from pathlib import Path
import unittest
import numpy as np


class TestPrunedCuFFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if not cp.cuda.runtime.getDeviceCount(): raise RuntimeError('No CUDA device')
        except Exception as exc:
            raise unittest.SkipTest(str(exc))
        path=Path(__file__).resolve().parents[2]/'examples'/'asm_cufft.py'
        spec=importlib.util.spec_from_file_location('asm_cufft',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.cp,cls.Plan=cp,module.PrunedCuFFT
        cls.Wrapper=module.CuFFTASMPlan

    def test_axis_and_crop_callbacks(self):
        cp=self.cp
        shape=(15,24);padded_shape=tuple(2*n for n in shape)
        rng=np.random.default_rng(192)
        original=(rng.standard_normal(shape)+1j*rng.standard_normal(shape)).astype(np.complex64)
        transfer=(rng.standard_normal(padded_shape)+1j*rng.standard_normal(padded_shape)).astype(np.complex64)
        h,w=shape;crop=(slice(h//2,h//2+h),slice(w//2,w//2+w))
        for axis_order in ('row','column'):
            for callback in (False,True):
                with self.subTest(axis_order=axis_order,callback=callback):
                    plan=self.Wrapper(shape,(.2e-6,.3e-6),cp.asarray(transfer),
                                      axis_order=axis_order,crop_callback=callback)
                    source=cp.asarray(original)
                    outputs=[cp.empty_like(source),cp.empty_like(source)]
                    for index,scale in ((0,1.),(1,-.25),(0,.75)):
                        source.set(original*scale)
                        plan.core.rows.fill(cp.nan);plan.core.columns.fill(cp.nan)
                        plan.execute(source,outputs[index],0.,532e-9)
                        padded=np.zeros(padded_shape,np.complex128);padded[crop]=original*scale
                        expected=np.fft.ifft2(np.fft.fft2(padded)*transfer)[crop]
                        np.testing.assert_allclose(cp.asnumpy(outputs[index]),expected,rtol=2e-5,atol=2e-6)
                        np.testing.assert_array_equal(cp.asnumpy(source),original*scale)
                    # Anisotropic dynamic frequencies must swap along with axes.
                    dynamic=self.Wrapper(shape,(.2e-6,.3e-6),cp.asarray(transfer),True,
                                         axis_order=axis_order,crop_callback=callback)
                    z,wl=-.01,450e-9
                    fy=np.fft.fftfreq(padded_shape[0],.2e-6)[:,None]
                    fx=np.fft.fftfreq(padded_shape[1],.3e-6)[None,:]
                    q=wl**-2-fx*fx-fy*fy
                    spectral=np.exp(2j*np.pi*z*np.sqrt(np.maximum(q,0)))*(q>=0)
                    dynamic.execute(source,outputs[1],z,wl)
                    expected=np.fft.ifft2(np.fft.fft2(padded)*spectral)[crop]
                    np.testing.assert_allclose(cp.asnumpy(outputs[1]),expected,rtol=2e-5,atol=2e-6)

    def test_static_analytic_quadrant(self):
        import sys
        from unittest.mock import patch
        examples = str(Path(__file__).resolve().parents[2] / 'examples')
        with patch.object(sys, 'path', [examples] + sys.path):
            from asm_cufft_symmetric import CuFFTStaticPlan
        cp = self.cp
        shape, pitch = (15, 21), (.3e-6, .25e-6)
        padded_shape = tuple(2 * n for n in shape)
        rng = np.random.default_rng(617)
        original = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
        h, w = shape
        crop = (slice(h // 2, h // 2 + h), slice(w // 2, w // 2 + w))
        padded = np.zeros(padded_shape, np.complex128)
        padded[crop] = original
        fy = np.fft.fftfreq(padded_shape[0], pitch[0])[:, None]
        fx = np.fft.fftfreq(padded_shape[1], pitch[1])[None, :]
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            source = cp.asarray(original)
            outputs = [cp.empty_like(source), cp.empty_like(source)]
            for mask in ('none', 'rectangular'):
                for order in ('row', 'column'):
                    for compressed in (False, True):
                        for callback in (False, True):
                            with self.subTest(mask=mask, order=order, compressed=compressed, callback=callback):
                                plan = CuFFTStaticPlan(shape, pitch, mask, order, compressed, stream,
                                                       crop_callback=callback)
                                for i, (z, wavelength) in enumerate(((0., 500e-9), (-1e-6, 532e-9), (100., 450e-9))):
                                    q = wavelength ** -2 - fx * fx - fy * fy
                                    keep = q >= 0
                                    if mask == 'rectangular':
                                        keep &= abs(fx) <= 1 / (wavelength * np.sqrt(1 + (2 * z / (padded_shape[1] * pitch[1])) ** 2))
                                        keep &= abs(fy) <= 1 / (wavelength * np.sqrt(1 + (2 * z / (padded_shape[0] * pitch[0])) ** 2))
                                    transfer = np.exp(2j * np.pi * z * np.sqrt(np.maximum(q, 0))) * keep
                                    expected = np.fft.ifft2(np.fft.fft2(padded) * transfer)[crop]
                                    plan.prepare_asm_transfer(z=z, wavelength=wavelength)
                                    plan.core.rows.fill(cp.nan)
                                    plan.work.fill(cp.nan)
                                    if plan.compact is not None:
                                        plan.compact.fill(cp.nan)
                                    with patch('cupy.empty', side_effect=AssertionError('execute allocation')):
                                        plan.execute(source, outputs[i % 2])
                                    stream.synchronize()
                                    actual = cp.asnumpy(outputs[i % 2])
                                    error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                                    self.assertLess(error, 2e-5)
                                    np.testing.assert_array_equal(cp.asnumpy(source), original)
                                plan.close()

    def test_static_dirty_workspace_and_input_preservation(self):
        cp=self.cp
        for shape in ((7,11),(16,32),(31,65),(64,16)):
            with self.subTest(shape=shape):
                rng=np.random.default_rng(82)
                x=(rng.standard_normal(shape)+1j*rng.standard_normal(shape)).astype(np.complex64)
                sh=tuple(2*n for n in shape)
                transfer=(rng.standard_normal(sh)+1j*rng.standard_normal(sh)).astype(np.complex64)
                plan=self.Plan(shape,(6.4e-6,6.4e-6),cp.asarray(transfer))
                source=cp.asarray(x);dest=cp.empty_like(source)
                for scale in (1.,-.75):
                    source.set(x*scale)
                    plan.rows.fill(cp.nan);plan.columns.fill(cp.nan)
                    plan.execute(source,dest,0.,532e-9)
                    padded=np.zeros(sh,np.complex128)
                    h,w=shape;crop=(slice(h//2,h//2+h),slice(w//2,w//2+w))
                    padded[crop]=x*scale
                    expected=np.fft.ifft2(np.fft.fft2(padded)*transfer)[crop]
                    np.testing.assert_allclose(cp.asnumpy(dest),expected,rtol=2e-5,atol=2e-6)
                    np.testing.assert_array_equal(cp.asnumpy(source),x*scale)

    def test_dynamic_parameters_and_masks(self):
        cp=self.cp;shape=(31,48);sh=tuple(2*n for n in shape)
        rng=np.random.default_rng(53)
        x=(rng.standard_normal(shape)+1j*rng.standard_normal(shape)).astype(np.complex64)
        h,w=shape;crop=(slice(h//2,h//2+h),slice(w//2,w//2+w))
        padded=np.zeros(sh,np.complex128);padded[crop]=x
        for pitch in ((6.4e-6,6.4e-6),(.2e-6,.3e-6)):
            fy=np.fft.fftfreq(sh[0],pitch[0])[:,None]
            fx=np.fft.fftfreq(sh[1],pitch[1])[None,:]
            for bandlimit in (False,True):
                plan=self.Plan(shape,pitch,cp.empty(sh,cp.complex64),True,bandlimit)
                source=cp.asarray(x);dest=cp.empty_like(source)
                for z,wl in ((0.,532e-9),(.01,450e-9),(-.025,633e-9)):
                    with self.subTest(pitch=pitch,bandlimit=bandlimit,z=z,wl=wl):
                        q=wl**-2-fx*fx-fy*fy
                        mask=q>=0
                        if bandlimit:
                            mask &= (abs(fx)<=1/(wl*np.sqrt(1+(2*z/(sh[1]*pitch[1]))**2)))
                            mask &= (abs(fy)<=1/(wl*np.sqrt(1+(2*z/(sh[0]*pitch[0]))**2)))
                        transfer=np.exp(2j*np.pi*z*np.sqrt(np.maximum(q,0)))*mask
                        expected=np.fft.ifft2(np.fft.fft2(padded)*transfer)[crop]
                        plan.rows.fill(cp.nan);plan.columns.fill(cp.nan)
                        plan.execute(source,dest,z,wl)
                        np.testing.assert_allclose(cp.asnumpy(dest),expected,rtol=2e-5,atol=2e-6)


if __name__=='__main__': unittest.main()
