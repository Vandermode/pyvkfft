from typing import Optional, Tuple, Union, Any
import numpy as np
import jax
import jax.numpy as jnp
import ctypes

from .cuda import VkFFTApp as VkFFTApp_cuda, library


jax.ffi.register_ffi_target("vkfft_fft", jax.ffi.pycapsule(library.VkFFTForward), platform="CUDA")
jax.ffi.register_ffi_target("vkfft_ifft", jax.ffi.pycapsule(library.VkFFTInverse), platform="CUDA")


class VkFFTApp(VkFFTApp_cuda):
    """JAX-compatible wrapper for VkFFT."""
    
    def __init__(self, shape, dtype, ndim=None, inplace=True, stream=None, norm=1,
                 r2c=False, dct=False, dst=False, axes=None, strides=None, tune_config=None,
                 r2c_odd=False, convolve=False, convolve_conj=0, convolve_norm=False,
                 convolve_shape=None, verbose=False, **kwargs):
        super().__init__(shape, dtype, ndim, inplace, stream, norm,
                         r2c, dct, dst, axes, strides, tune_config,
                         r2c_odd, convolve, convolve_conj, convolve_norm,
                         convolve_shape, verbose, **kwargs)

    def jax_fft(self, x: jnp.ndarray, kernel: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply forward FFT to input array.
        
        Args:
            x: Input array
            kernel: Optional convolution kernel
            
        Returns:
            Output array after FFT
        """
        # Define output type
        output_type = jax.ShapeDtypeStruct(x.shape, x.dtype)        
        app_void_ptr_address = ctypes.cast(self.app, ctypes.c_void_p).value
                
        input_output_aliases = {}
        if self.inplace:
            input_output_aliases[0] = 0
        
        # Prepare kernel if provided
        if kernel is not None:
            return jax.ffi.ffi_call("vkfft_fft", output_type, input_output_aliases=input_output_aliases)(x, kernel, app=app_void_ptr_address)
        else:
            # Pass an empty array as kernel
            empty_kernel = jnp.zeros((0,), dtype=self.dtype)
            return jax.ffi.ffi_call("vkfft_fft", output_type, input_output_aliases=input_output_aliases)(x, empty_kernel, app=app_void_ptr_address)
    
    def jax_ifft(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply inverse FFT to input array.
        
        Args:
            x: Input array
            
        Returns:
            Output array after inverse FFT
        """
        # Define output type
        output_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
        app_void_ptr_address = ctypes.cast(self.app, ctypes.c_void_p).value
        
        input_output_aliases = {}
        if self.inplace:
            input_output_aliases[0] = 0
        
        return jax.ffi.ffi_call("vkfft_ifft", output_type, input_output_aliases=input_output_aliases)(x, app=app_void_ptr_address)
