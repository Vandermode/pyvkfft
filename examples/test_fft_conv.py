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
from pyvkfft.cuda import VkFFTApp, VKFFT_MAX_FFT_DIMENSIONS, pfUINT, pfUINT_array

from skimage.transform import resize

from ctypes import cast
from numpy.ctypeslib import as_ctypes, as_array


def test_1d():
    size = 256
    input = ascent()[:size,0] / 255.
    
    performZeropadding = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    # performZeropadding = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    print(performZeropadding)
    # performZeropadding = as_ctypes(performZeropadding)
    fft_zeropad_left = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    fft_zeropad_left[0] = size // 4 * 1
    # fft_zeropad_left = as_ctypes(fft_zeropad_left)
    fft_zeropad_right = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    fft_zeropad_right[0] = size // 4 * 3
    
    input[fft_zeropad_left[0]:fft_zeropad_right[0]] = 0
    
    nx, = input.shape
    x = np.arange(-nx//2,nx//2)
    
    sigma = size / 100
    kernel = np.fft.fftshift(np.exp(-(x**2) / (2*sigma**2)))
    kernel /= kernel.sum()

    # Numpy convolution
    input, kernel = input.astype(np.complex64), kernel.astype(np.complex64)
    
    K_np = np.fft.fftn(kernel)
    gd_np = np.fft.ifftn(np.fft.fftn(input) * K_np, input.shape)
    
    # move data to GPU
    d_gpu = cupy.asarray(input)
    k_gpu = cupy.asarray(kernel)
    
    K_gpu = fftn(k_gpu)

    # vkfft convolution
    # d_gpu = ifftn(fftn(d_gpu) * fftn(k_gpu))
    d_gpu = ifftn(fftn(d_gpu, performZeropadding=performZeropadding, fft_zeropad_left=fft_zeropad_left, fft_zeropad_right=fft_zeropad_right) * fftn(k_gpu))
    
    # app = VkFFTApp(
    #     input.shape, dtype=np.complex64, ndim=1, inplace=True, r2c=False, convolve=True, convolve_conj=0, 
    #     performZeropadding=performZeropadding,
    #     fft_zeropad_left=fft_zeropad_left,
    #     fft_zeropad_right=fft_zeropad_right)
    
    # app = VkFFTApp(input.shape, dtype=np.complex64, ndim=1, inplace=True, r2c=False, convolve=True, convolve_conj=0)
    
    # # print(app.is_radix_transform())
    # print(app)
    # print('app.nb_axis_upload:', app.nb_axis_upload)
    # print('app.use_bluestein_fft:', app.use_bluestein_fft)
    # print('app.tmp_buffer_nbytes:', app.tmp_buffer_nbytes)
    # print('app.axis_split', app.axis_split)
    # print(as_array(app.config.contents.performZeropadding))
    # print(as_array(app.config.contents.fft_zeropad_left))
    # print(as_array(app.config.contents.fft_zeropad_right))
    # app.fft(d_gpu, convolve_kernel=K_gpu)
    
    
    gd0 = d_gpu.get()
    
    print(np.abs(gd0 - gd_np).max())
    print('np.allclose(gd0, gd_np): ',  np.allclose(gd0, gd_np, rtol=1e-6, atol=gd_np.max()*1e-6))

    plt.figure(figsize=(20,4))
    plt.subplot(141)
    plt.plot(input.real)
    plt.subplot(142)
    plt.plot(np.fft.fftshift(kernel).real)
    plt.subplot(143)
    plt.plot(gd_np.real)
    plt.plot(gd0.real, '--')
    plt.tight_layout()    
    # plt.subplot(144)
    # plt.tight_layout()

    plt.show()


def main():
    img = ascent()[:256,:256] / 255.
    # size = 8192
    size = 1024
    # size = 1234
    
    performZeropadding = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    # performZeropadding = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    print(performZeropadding)
    # performZeropadding = as_ctypes(performZeropadding)
    fft_zeropad_left = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    fft_zeropad_left[0] = 1024 // 4 * 1
    fft_zeropad_left[1] = 1024 // 4 * 1
    # fft_zeropad_left = as_ctypes(fft_zeropad_left)
    fft_zeropad_right = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    fft_zeropad_right[0] = 1024 // 4 * 3
    fft_zeropad_right[1] = 1024 // 4 * 3
    
    # size = int(2**15)
    print(size)
    img = resize(img, (size, size), anti_aliasing=True)
    
    img[fft_zeropad_left[1]:fft_zeropad_right[1], :] = 0
    img[:, fft_zeropad_left[0]:fft_zeropad_right[0]] = 0
    
    ny, nx = img.shape
    x, y = np.meshgrid(np.arange(-nx//2,nx//2), np.arange(-ny//2, ny//2), indexing='xy')
    
    sigma = size // 100
    kernel = np.fft.fftshift(np.exp(-(x**2+y**2) / (2*sigma**2)))
    kernel /= kernel.sum()

    # Numpy convolution
    img, kernel = img.astype(np.complex64), kernel.astype(np.complex64)
    
    K_np = np.fft.fftn(kernel)
    gd_np = np.fft.ifftn(np.fft.fftn(img) * K_np, img.shape)
    
    # move data to GPU
    d_gpu = cupy.asarray(img)
    k_gpu = cupy.asarray(kernel)
    
    K_gpu = fftn(k_gpu)

    # vkfft convolution
    # d_gpu = ifftn(fftn(d_gpu) * fftn(k_gpu))
    d_gpu = ifftn(fftn(d_gpu, performZeropadding=performZeropadding, fft_zeropad_left=fft_zeropad_left, fft_zeropad_right=fft_zeropad_right) * fftn(k_gpu))
    
    # app = VkFFTApp(
    #     img.shape, dtype=np.float32, ndim=2, inplace=True, r2c=False, convolve=True, convolve_conj=0, 
    #     performZeropadding=performZeropadding,
    #     fft_zeropad_left=fft_zeropad_left,
    #     fft_zeropad_right=fft_zeropad_right)
    
    # # print(app.is_radix_transform())
    # print(app)
    # print('app.nb_axis_upload:', app.nb_axis_upload)
    # print('app.use_bluestein_fft:', app.use_bluestein_fft)
    # print('app.tmp_buffer_nbytes:', app.tmp_buffer_nbytes)
    # print('app.axis_split', app.axis_split)
    
    # app.fft(d_gpu, convolve_kernel=K_gpu)
    
    gd0 = d_gpu.get()
    
    print(np.abs(gd0 - gd_np).max())
    print('np.allclose(gd0, gd_np): ',  np.allclose(gd0, gd_np, rtol=1e-6, atol=gd_np.max()*1e-6))

    plt.figure(figsize=(12,4))
    plt.subplot(141)
    plt.imshow(img.real, cmap='gray')
    plt.subplot(142)
    plt.imshow(np.fft.fftshift(kernel).real, cmap='gray')
    plt.subplot(143)
    plt.imshow(gd_np.real, cmap='gray', vmin=0, vmax=1)
    plt.tight_layout()    
    plt.subplot(144)
    plt.imshow(gd0.real, cmap='gray', vmin=0, vmax=1)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    # main()
    test_1d()