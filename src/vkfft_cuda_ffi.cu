#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "vkfft_cuda.cu"


namespace ffi = xla::ffi;

// Forward FFT implementation function
ffi::Error VkFFTForwardImpl(void *app, ffi::AnyBuffer input,
                            ffi::AnyBuffer kernel, // This could be an empty buffer
                            ffi::Result<ffi::AnyBuffer> output)
{
    // Cast the void* to VkFFTApplication*
    VkFFTApplication *vkapp = static_cast<VkFFTApplication*>(app);
    
    // Only use kernel if it has elements
    void *kernel_ptr = (kernel.element_count() > 0) ? kernel.untyped_data() : NULL;

    // Call the VkFFT fft function with possibly NULL kernel
    int result = fft(vkapp,
                     input.untyped_data(),
                     output->untyped_data(),
                     kernel_ptr);

    if (result != 0)
    {
        return ffi::Error::Internal("VkFFT forward execution failed with code: " + std::to_string(result));
    }

    return ffi::Error::Success();
}

// Inverse FFT implementation function
ffi::Error VkFFTInverseImpl(void *app, ffi::AnyBuffer input,
                            ffi::Result<ffi::AnyBuffer> output)
{
    // Cast the void* to VkFFTApplication*
    VkFFTApplication *vkapp = static_cast<VkFFTApplication*>(app);

    // Call the VkFFT ifft function
    int result = ifft(vkapp,
                      input.untyped_data(),
                      output->untyped_data());

    if (result != 0)
    {
        return ffi::Error::Internal("VkFFT inverse execution failed with code: " + std::to_string(result));
    }

    return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(VkFFTForward, VkFFTForwardImpl,
                              ffi::Ffi::Bind()
                                .Attr<ffi::Pointer<void>>("app")
                                .Arg<ffi::AnyBuffer>() // input
                                .Arg<ffi::AnyBuffer>() // kernel
                                .Ret<ffi::AnyBuffer>() // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(VkFFTInverse, VkFFTInverseImpl,
                              ffi::Ffi::Bind()
                                .Attr<ffi::Pointer<void>>("app")
                                .Arg<ffi::AnyBuffer>() // input
                                .Ret<ffi::AnyBuffer>() // output
);
