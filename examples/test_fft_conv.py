import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt
from scipy.datasets import ascent
from skimage.transform import resize

import cupy
from pyvkfft.base import primes, primes_str
from pyvkfft.fft import rfftn, irfftn, fftn, ifftn
from pyvkfft.cuda import VkFFTApp, VKFFT_MAX_FFT_DIMENSIONS, pfUINT, pfUINT_array
from ctypes import cast
from numpy.ctypeslib import as_ctypes, as_array

import time


def crop_center_2d(img, shape):
    cropy, cropx = shape
    y, x = img.shape[-2:]
    
    # Check if the given shape is larger than the image shape
    if cropy > y or cropx > x:
        # Return the original image if the crop shape is larger
        return img
    
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[..., starty:starty + cropy, startx:startx + cropx]


def crop_center_1d(inp, cropx):
    x, = inp.shape[-1:]
    startx = x // 2 - (cropx // 2)
    return inp[..., startx:startx + cropx]


def test_1d():
    # size = 2**12
    size = 512
    input = ascent()[:size,0] / 255.
    input = np.random.rand(size).astype(np.float32)
    print(input.shape)
    
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
    
    gd_np = np.fft.fftshift(gd_np)
    gd_np = crop_center_1d(gd_np, size//2)
    
    # move data to GPU
    d_gpu = cupy.asarray(input)
    k_gpu = cupy.asarray(kernel)
    
    K_gpu = fftn(k_gpu)

    # vkfft convolution
    # d_gpu = ifftn(fftn(d_gpu) * fftn(k_gpu))
    
    app = VkFFTApp(input.shape, dtype=np.complex64, ndim=1, inplace=True)
    app_zp = VkFFTApp(input.shape, dtype=np.complex64, ndim=1, inplace=True,
                   performZeropadding=performZeropadding,
                   fft_zeropad_left=fft_zeropad_left,
                   fft_zeropad_right=fft_zeropad_right)
    
    
    # d_gpu = app_zp.ifft(app_zp.fft(d_gpu) * K_gpu)
    # d_gpu = app.ifft(app.fft(d_gpu) * K_gpu)
    
    app_zp.fft(d_gpu)
    d_gpu *= K_gpu
    app_zp.ifft(d_gpu)
    
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
    gd0 = np.fft.fftshift(gd0)
    gd0 = crop_center_1d(gd0, size//2)
    
    print(np.abs(gd0 - gd_np).max())
    print('np.allclose(gd0, gd_np): ',  np.allclose(gd0, gd_np, rtol=1e-6, atol=gd_np.max()*1e-6))

    plt.figure(figsize=(20,4))
    plt.subplot(141)
    plt.plot(input.real)
    plt.title('(inner) padded input')
    plt.subplot(142)
    plt.plot(np.fft.fftshift(kernel).real)
    plt.title('kernel')
    plt.subplot(143)
    plt.plot(gd_np.real, label='numpy fft')
    plt.plot(gd0.real, '--', label='vkfft')
    plt.legend()
    plt.title('FFT + multiplication + IFFT')
    plt.tight_layout()
    # plt.subplot(144)
    # plt.tight_layout()

    plt.show()


def test_2d():
    img = ascent()[:256,:256] / 255.
    # size = 8192
    size = 1024
    # size = 1234
    
    performZeropadding = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    # performZeropadding = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    print(performZeropadding)
    # performZeropadding = as_ctypes(performZeropadding)
    fft_zeropad_left = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    fft_zeropad_left[0] = size // 4 * 1
    fft_zeropad_left[1] = size // 4 * 1
    # fft_zeropad_left = as_ctypes(fft_zeropad_left)
    fft_zeropad_right = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    fft_zeropad_right[0] = size // 4 * 3
    fft_zeropad_right[1] = size // 4 * 3
    
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
    
    gd_np = np.fft.fftshift(gd_np, axes=(-2, -1))
    gd_np = crop_center_2d(gd_np, (size//2, size//2))
    
    
    # move data to GPU
    d_gpu = cupy.asarray(img)
    k_gpu = cupy.asarray(kernel)
    
    K_gpu = fftn(k_gpu)

    # vkfft convolution
    # d_gpu = ifftn(fftn(d_gpu) * fftn(k_gpu))
    
    app_zp = VkFFTApp(
        img.shape, dtype=np.complex64, ndim=2, inplace=True, r2c=False, convolve=False, convolve_conj=0, 
        performZeropadding=performZeropadding,
        fft_zeropad_left=fft_zeropad_left,
        fft_zeropad_right=fft_zeropad_right)
    
    app = VkFFTApp(img.shape, dtype=np.complex64, ndim=2, inplace=True, r2c=False, convolve=False, convolve_conj=0)
    
    # # print(app.is_radix_transform())
    # print(app)
    # print('app.nb_axis_upload:', app.nb_axis_upload)
    # print('app.use_bluestein_fft:', app.use_bluestein_fft)
    # print('app.tmp_buffer_nbytes:', app.tmp_buffer_nbytes)
    # print('app.axis_split', app.axis_split)
    
    # app.fft(d_gpu, convolve_kernel=K_gpu)
    app_zp.fft(d_gpu)
    d_gpu *= K_gpu
    app_zp.ifft(d_gpu)
    
    gd0 = d_gpu.get()    
    gd0 = np.fft.fftshift(gd0, axes=(-2, -1))
    gd0 = crop_center_2d(gd0, (size//2, size//2))
    
    print(np.abs(gd0 - gd_np).max())
    print('np.allclose(gd0, gd_np): ',  np.allclose(gd0, gd_np, rtol=1e-6, atol=gd_np.max()*1e-6))

    plt.figure(figsize=(12,4))
    plt.subplot(141)
    plt.imshow(img.real, cmap='gray')
    plt.title('(inner) padded input')
    plt.subplot(142)
    plt.imshow(np.fft.fftshift(kernel).real, cmap='gray')
    plt.title('kernel')
    plt.subplot(143)
    plt.imshow(gd_np.real, cmap='gray', vmin=0, vmax=1)
    plt.title('numpy FFTs results')
    plt.tight_layout()    
    plt.subplot(144)
    plt.imshow(gd0.real, cmap='gray', vmin=0, vmax=1)
    plt.title('vKFFT results')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    test_1d()
    test_2d()
