import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt
from scipy.datasets import ascent

import cupy
from pyvkfft.base import primes, primes_str
from pyvkfft.fft import rfftn, irfftn, fftn, ifftn
from pyvkfft.cuda import VkFFTApp

from skimage.transform import resize


def main():
    img = ascent()[:256,:256] / 255.
    size = 256
    
    # size = int(2**15)
    print(size)
    img = resize(img, (size, size), anti_aliasing=True)
    
    ny, nx = img.shape
    x, y = np.meshgrid(np.arange(-nx//2,nx//2), np.arange(-ny//2, ny//2), indexing='xy')
    
    sigma = size // 100
    kernel = np.fft.fftshift(np.exp(-(x**2+y**2) / (2*sigma**2)))
    kernel /= kernel.sum()

    # Numpy convolution
    img, kernel = img.astype(np.complex64), kernel.astype(np.complex64)
    
    gd_np = np.fft.ifftn(np.fft.fftn(img) * np.fft.fftn(kernel), img.shape)
    
    # move data to GPU
    d_gpu = cupy.asarray(img)
    g_gpu = cupy.asarray(kernel)
    
    k_gpu = fftn(g_gpu)

    # vkfft convolution
    # d_gpu = ifftn(fftn(d_gpu) * fftn(g_gpu))
    app = VkFFTApp(img.shape, dtype=np.float32, ndim=2, inplace=True, r2c=False, convolve=True, convolve_conj=0)
    app.fft(d_gpu, convolve_kernel=k_gpu)
    
    gd0 = d_gpu.get()
    
    print(np.abs(gd0 - gd_np).max())
    print('np.allclose(gd0, gd_np): ',  np.allclose(gd0, gd_np, rtol=1e-6, atol=gd_np.max()*1e-6))

    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(img.real, cmap='gray')
    plt.subplot(132)
    plt.imshow(np.fft.fftshift(kernel).real, cmap='gray')
    plt.subplot(133)
    plt.imshow(gd0.real, cmap='gray', vmin=0, vmax=1)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()