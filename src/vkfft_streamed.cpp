// Ordinary, opaque 1D FFT handles for the exact streamed ASM implementation.
// Build only against a staged VkFFT tree with the inverse upload-order fix.
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#define VKFFT_BACKEND 1
#include "vkFFT.h"

struct StreamedFFT {
    VkFFTApplication app{};
    CUdevice device{};
    CUstream stream{};
    uint64_t length=0, batch=0, bytes=0;
    void* buffer=reinterpret_cast<void*>(1);
    void* temporary=reinterpret_cast<void*>(2);
    bool initialized=false;
    std::string info;
    ~StreamedFFT() { if (initialized) deleteVkFFT(&app); }
};
static thread_local std::string streamed_error;

extern "C" {
uint32_t streamed_fft_abi_version() { return 1; }
// Bit 0 identifies the ordinary no-reorder API built with the inverse fix.
uint64_t streamed_fft_capabilities() { return 1; }
const char* streamed_fft_last_error() { return streamed_error.c_str(); }

void* streamed_fft_create_ex(uint64_t length, uint64_t batch, uint64_t stream, int ordered) {
    streamed_error.clear();
    try {
        if (!length || !batch || (ordered != 0 && ordered != 1) ||
                length > static_cast<uint64_t>(INT64_MAX)/batch/8)
            throw std::invalid_argument("FFT dimensions exceed checked byte arithmetic");
        std::unique_ptr<StreamedFFT> p(new StreamedFFT);
        p->length=length; p->batch=batch; p->bytes=length*batch*8;
        p->stream=reinterpret_cast<CUstream>(stream);
        if (cuCtxGetDevice(&p->device) != CUDA_SUCCESS)
            throw std::runtime_error("No current CUDA context");
        VkFFTConfiguration c{};
        c.FFTdim=1; c.size[0]=length; c.numberBatches=batch;
        c.device=&p->device; c.stream=&p->stream; c.num_streams=1;
        c.buffer=&p->buffer; c.bufferSize=&p->bytes;
        c.normalize=1; c.disableReorderFourStep=!ordered;
#ifdef VKFFT_STREAMED_FFI
        c.userTempBuffer=1; c.tempBuffer=&p->temporary; c.tempBufferSize=&p->bytes;
#endif
        VkFFTResult result=initializeVkFFT(&p->app, c);
        if (result != VKFFT_SUCCESS)
            throw std::runtime_error("VkFFT initialization failed: " + std::to_string(result));
        p->initialized=true;
        auto* plan=p->app.localFFTPlan;
        if (p->app.useBluesteinFFT[0] || plan->numAxisUploads[0] > 3)
            throw std::invalid_argument("Streamed ASM requires radix FFTs with at most three uploads");
        std::ostringstream info;
        info << "{\"length\":" << length << ",\"batch\":" << batch
             << ",\"buffer_bytes\":" << p->bytes << ",\"uploads\":" << plan->numAxisUploads[0]
             << ",\"axis_split\":[" << plan->axisSplit[0][0] << ',' << plan->axisSplit[0][1]
             << ',' << plan->axisSplit[0][2]
             << "],\"ordered\":" << ordered << ",\"temporary_bytes\":"
             << (p->app.configuration.allocateTempBuffer ? p->app.configuration.tempBufferSize[0] : 0)
             << ",\"vkfft_version\":" << VkFFTGetVersion() << '}';
        p->info=info.str();
        return p.release();
    } catch (const std::exception& e) {
        streamed_error=e.what();
        return nullptr;
    }
}

void* streamed_fft_create(uint64_t length, uint64_t batch, uint64_t stream) {
    return streamed_fft_create_ex(length, batch, stream, 0);
}

const char* streamed_fft_info(void* handle) {
    return handle ? static_cast<StreamedFFT*>(handle)->info.c_str() : nullptr;
}

int streamed_fft_execute(void* handle, void* buffer, int inverse) {
    auto* p=static_cast<StreamedFFT*>(handle);
    if (!p || !buffer || (inverse != 0 && inverse != 1)) return -1;
    VkFFTLaunchParams launch{};
    launch.buffer=&buffer;
    return static_cast<int>(VkFFTAppend(&p->app, inverse ? 1 : -1, &launch));
}

void streamed_fft_destroy(void* handle) {
    auto* p=static_cast<StreamedFFT*>(handle);
    if (p) {
        cuStreamSynchronize(p->stream);
        delete p;
    }
}
}
