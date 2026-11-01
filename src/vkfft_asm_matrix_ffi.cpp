// Typed XLA FFI for matrix-kernel windowed convolutions (VkFFT native matrix convolution).
// Independent of the ABI 4 engine (vkfft_asm.cpp / vkfft_asm_ffi.cpp): own hooks, own plans, own target.
//
// expand: out[v,b,k,o] = window(IFFT2(sum_i M_k[o,i] FFT2(embed(field[v,b,i]))))
//         coordinateFeatures = matrixConvolution = M, numberKernels = K: M forward FFTs, the K matrix
//         products in the column convolution step, then K*M inverse FFTs.
// reduce: out[v,b,o] = sum_k window(IFFT2(sum_i M_k[o,i] FFT2(embed(field[v,b,k,i]))))
//         fused (K*M <= 8): the K*M input planes are the features of one (K*M) x (K*M) matrix convolution whose
//         rows >= M are zero; only the M output rows are inverse transformed (patched coordinate loop) and
//         dispatched. Otherwise one single-kernel pass per k, the output store accumulating over k.
// Kernel entries M_k[o,i] are read from planes[component_map[k][o][i]] (natural FFT order, -1 = zero), so a
// symmetric 2x2 kernel stored as (xx, xy, yy) and diagonal kernels share one code path. Execution buffers
// (output, workspace, VkFFT scratch) belong to XLA; cached plans own compiled kernels only. Expand uses K
// single-kernel passes when the K-kernel convolution kernel spills or the workspace holds only M canvases.
// Requires the three VKFFT_ASM_MATRIX_HOOKS header patches of examples/asm_matrix_ffi_build.py: the kernel-entry
// report; per-kernel stores only in the convolution step (stock VkFFT repeats every kernel's store K times, which
// re-applies the four-step twiddle and is wrong for K > 1 on two-upload column axes); the output-row limit of a
// fused reduce.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

static bool asm_read_hook(void*, void*, void*, void*);
static bool asm_write_hook(void*, void*, void*, void*);
static bool asm_is_output(void*);
static bool asm_compile_hook(void*, void*);
static const void* asm_launch_arguments();
static void asm_dispatch_dimensions(void*, void*);
static void asm_matrix_component(uint64_t, uint64_t, uint64_t);
static long long asm_matrix_outputs(void*, long long);
#define VKFFT_BACKEND 1
#define VKFFT_ASM_HOOKS 1
#define VKFFT_ASM_MATRIX_HOOKS 1
#include "vkFFT.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

constexpr int MAX_KERNELS = 16, MAX_FEATURES = 4, MAX_FUSED = 8, MAX_ENTRIES = 256;

// Must match the device-side definition in device_helpers().
struct MatrixArgs {
    const void* input;
    void* output;
    const void* kernel;
    uint64_t ax, bx, ay, by;
    int32_t accumulate, unused;
    int32_t plane[MAX_ENTRIES];   // [kernel][output row][input feature] -> plane index, -1 = zero
};
struct MatrixPlan {
    VkFFTApplication app{};
    CUdevice device{};
    CUstream stream{};
    // Each of the M field planes (height x width) is zero-embedded at (in_y, in_x) in a canvas_h x canvas_w
    // FFT canvas; every output plane is the out_h x out_w window at (out_y, out_x).
    // features = matrix size (VkFFT coordinates); outputs = transformed output rows (< features: fused reduce).
    uint64_t height, width, features, kernels, outputs, bytes;
    uint64_t canvas_h, canvas_w, in_y, in_x, out_h, out_w, out_y, out_x;
    bool transpose;
    void* buffer = reinterpret_cast<void*>(1);
    void* kernel = reinterpret_cast<void*>(2);
    void* temporary = reinterpret_cast<void*>(3);
    bool initialized = false;
    bool needs_temporary = false;
    bool conv_spills = false;   // the column convolution kernel spills registers to local memory
    std::string info;
    struct KernelSource { uint64_t axis, upload, inverse, hash; };
    std::vector<KernelSource> sources;
    ~MatrixPlan() { if (initialized) deleteVkFFT(&app); }
};
static thread_local MatrixPlan* building = nullptr;
static thread_local MatrixPlan* executing = nullptr;
static thread_local MatrixArgs arguments{};
static thread_local std::string last_error;
// Matrix entry (output j, input l) and VkFFT kernel component of the kernel load being generated.
static thread_local uint64_t component_j = 0, component_l = 0, component_k = 0;
static thread_local bool component_pending = false;

static void asm_matrix_component(uint64_t j, uint64_t l, uint64_t k) {
    component_j = j; component_l = l; component_k = k; component_pending = true;
}
// Inverse transforms of the convolution step: only the output rows (all rows unless a fused reduce).
static long long asm_matrix_outputs(void* opaque, long long count) {
    auto* sc = static_cast<VkFFTSpecializationConstantsLayout*>(opaque);
    return building && sc->convolutionStep && building->outputs < uint64_t(count) ? (long long)building->outputs
                                                                                  : count;
}
static bool asm_is_output(void* opaque) {
    auto* sc = static_cast<VkFFTSpecializationConstantsLayout*>(opaque);
    return building && sc->actualInverse && sc->axis_id == 0 && sc->axis_upload_id == sc->numAxisUploads - 1;
}
static bool asm_read_hook(void* opaque, void* out_ptr, void* buffer_ptr, void* index_ptr) {
    if (!building) return false;
    auto* sc = static_cast<VkFFTSpecializationConstantsLayout*>(opaque);
    auto* out = static_cast<PfContainer*>(out_ptr);
    auto* buffer = static_cast<PfContainer*>(buffer_ptr);
    auto* index = static_cast<PfContainer*>(index_ptr);
    bool input = !sc->actualInverse && sc->axis_id == 0 && sc->axis_upload_id == sc->numAxisUploads - 1 &&
                 !std::strcmp(buffer->name, sc->inputsStruct.name);
    bool kernel = sc->convolutionStep && !std::strcmp(buffer->name, sc->kernelStruct.name);
    if (!input && !kernel) return false;
    if (kernel) {
        if (!component_pending) {
            // Unpatched headers or an unexpected kernel load: never guess the matrix entry.
            last_error = "Kernel load without a matrix component (missing VKFFT_ASM_MATRIX_HOOKS patch)";
            sc->res = VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
            return true;
        }
        component_pending = false;
        // VkFFT addresses entry k of kernel batch b at canvas index + k*stride[dims] + b*stride[dims+1];
        // remove that codegen-time offset and address the planes through the component map instead.
        uint64_t offset = component_k*sc->inputStride[sc->numFFTdims].data.i +
                          uint64_t(sc->batchID.data.i)*sc->inputStride[sc->numFFTdims+1].data.i;
        sc->tempLen = std::snprintf(sc->tempStr, 4096, "%s = asm_kernel(asm_args, %s - %lluull, %d, %d);\n",
                                    out->name, index->name, (unsigned long long)offset, int(sc->batchID.data.i),
                                    int(component_j*building->features + component_l));
    } else
        sc->tempLen = std::snprintf(sc->tempStr, 4096, "%s = asm_load_input(asm_args, %s);\n", out->name, index->name);
    PfAppendLine(sc);
    return true;
}
static bool asm_write_hook(void* opaque, void* buffer_ptr, void* index_ptr, void* in_ptr) {
    if (!asm_is_output(opaque)) return false;
    auto* sc = static_cast<VkFFTSpecializationConstantsLayout*>(opaque);
    auto* buffer = static_cast<PfContainer*>(buffer_ptr);
    if (std::strcmp(buffer->name, sc->outputsStruct.name)) return false;
    auto* index = static_cast<PfContainer*>(index_ptr);
    auto* in = static_cast<PfContainer*>(in_ptr);
    sc->tempLen = std::snprintf(sc->tempStr, 4096, "asm_store_output(asm_args, %s, %s);\n", index->name, in->name);
    PfAppendLine(sc);
    return true;
}
static const void* asm_launch_arguments() { return &arguments; }
static void asm_dispatch_dimensions(void* axis_ptr, void* dimensions) {
    // A fused reduce writes only its output rows: the inverse passes after the convolution step skip the rest.
    auto& sc = static_cast<VkFFTAxis*>(axis_ptr)->specializationConstants;
    if (executing && executing->outputs < executing->features && sc.actualInverse) {
        auto* z = &static_cast<pfUINT*>(dimensions)[2];
        *z = *z / executing->features * executing->outputs;
    }
}

static std::string device_helpers(const MatrixPlan& plan) {
    std::ostringstream s;
    s << "#define ASM_H " << plan.height << "ull\n#define ASM_W " << plan.width << "ull\n";
    s << "#define ASM_CH " << plan.canvas_h << "ull\n#define ASM_CW " << plan.canvas_w << "ull\n";
    s << "#define ASM_S " << plan.canvas_h*plan.canvas_w << "ull\n";
    s << "#define ASM_IY " << plan.in_y << "ull\n#define ASM_IX " << plan.in_x << "ull\n";
    s << "#define ASM_OH " << plan.out_h << "ull\n#define ASM_OW " << plan.out_w << "ull\n";
    s << "#define ASM_OY " << plan.out_y << "ull\n#define ASM_OX " << plan.out_x << "ull\n";
    s << "#define ASM_TRANSPOSE " << int(plan.transpose) << "\n";
    s << "#define ASM_ENTRIES " << plan.features*plan.features << "\n";
    s << "#define ASM_SYSTEMS " << plan.kernels*plan.outputs << "ull\n";
    s << R"CUDA(
struct MatrixArgs {
    const void* input; void* output; const void* kernel;
    unsigned long long ax, bx, ay, by;
    int accumulate, unused;
    int plane[256];
};
// i indexes the stacked canvases: plane i / ASM_S of the M field planes, zero outside the input window.
__device__ __forceinline__ float2 asm_load_input(MatrixArgs a, unsigned long long i) {
    unsigned long long f = i / ASM_S, p = i - f*ASM_S, y = p / ASM_CW, x = p - y*ASM_CW;
    if (y < ASM_IY || y >= ASM_IY + ASM_H || x < ASM_IX || x >= ASM_IX + ASM_W)
        return make_float2(0.f, 0.f);
    return ((const float2*)a.input)[(f*ASM_H + y - ASM_IY)*ASM_W + x - ASM_IX];
}
// Canvas i / ASM_S is output plane k*M + o; only its window is stored (accumulated after the first kernel).
__device__ __forceinline__ void asm_store_output(MatrixArgs a, unsigned long long i, float2 v) {
    unsigned long long s = i / ASM_S, p = i - s*ASM_S, y = p / ASM_CW, x = p - y*ASM_CW;
    if (s >= ASM_SYSTEMS || y < ASM_OY || y >= ASM_OY + ASM_OH || x < ASM_OX || x >= ASM_OX + ASM_OW) return;
    float2* o = (float2*)a.output + (s*ASM_OH + y - ASM_OY)*ASM_OW + x - ASM_OX;
    if (a.accumulate) { float2 u = *o; v.x += u.x; v.y += u.y; }
    *o = v;
}
// i is the (four-step permuted) canvas index of the convolution step; planes are in natural FFT order.
// JAX transposes complex-linear maps bilinearly: reversing frequencies (never conjugating) gives A^T.
__device__ __forceinline__ float2 asm_kernel(MatrixArgs a, unsigned long long i, int k, int entry) {
    int plane = a.plane[k*ASM_ENTRIES + entry];
    if (plane < 0) return make_float2(0.f, 0.f);
    unsigned long long py = i / ASM_CW, px = i - py*ASM_CW;
    unsigned long long y = (py % a.ay)*a.by + py/a.ay;
    unsigned long long x = (px % a.ax)*a.bx + px/a.ax;
    if (ASM_TRANSPOSE) {
        y = y ? ASM_CH-y : 0;
        x = x ? ASM_CW-x : 0;
    }
    return __ldg((const float2*)a.kernel + (unsigned long long)plane*ASM_S + y*ASM_CW + x);
}
)CUDA";
    return s.str();
}

static bool asm_compile_hook(void* app_ptr, void* axis_ptr) {
    if (!building) return false;
    auto* axis = static_cast<VkFFTAxis*>(axis_ptr);
    try {
        std::string code(axis->specializationConstants.code0);
        auto& sc = axis->specializationConstants;
        const std::string signature = "VkFFT_main (";
        size_t start = code.find(signature);
        if (start == std::string::npos) throw std::runtime_error("Missing VkFFT_main signature");
        size_t end = code.find(')', start + signature.size());
        if (end == std::string::npos) throw std::runtime_error("Missing kernel argument terminator");
        code.insert(end, ", MatrixArgs asm_args");
        code = device_helpers(*building) + code;
        // The kernel loads use the fixed plan permutation: specialize the divisors.
        if (sc.convolutionStep) {
            auto* fp = sc.actualInverse ? building->app.localFFTPlan_inverse : building->app.localFFTPlan;
            if (!fp) throw std::runtime_error("Missing FFT plan at kernel compilation");
            const uint64_t lengths[] = {building->canvas_w, building->canvas_h};
            for (int axis_id = 0; axis_id < 2; ++axis_id) {
                uint64_t a = fp->numAxisUploads[axis_id] == 2 ? fp->axisSplit[axis_id][0] : lengths[axis_id];
                uint64_t b = fp->numAxisUploads[axis_id] == 2 ? fp->axisSplit[axis_id][1] : 1;
                if (!a || !b || a*b != lengths[axis_id])
                    throw std::runtime_error("Incomplete FFT permutation at kernel compilation");
                for (const auto& item : {std::make_pair(axis_id ? "a.ay" : "a.ax", a),
                                         std::make_pair(axis_id ? "a.by" : "a.bx", b)}) {
                    size_t at = 0;
                    const std::string value = std::to_string(item.second) + "ull";
                    while ((at = code.find(item.first, at)) != std::string::npos) {
                        code.replace(at, std::strlen(item.first), value);
                        at += value.size();
                    }
                }
            }
        }
        uint64_t hash = 14695981039346656037ull;
        for (unsigned char byte : code) { hash ^= byte; hash *= 1099511628211ull; }
        building->sources.push_back({static_cast<uint64_t>(sc.axis_id), static_cast<uint64_t>(sc.axis_upload_id),
                                     static_cast<uint64_t>(sc.actualInverse), hash});
        auto* app = static_cast<VkFFTApplication*>(app_ptr);
        if (code.size()+1 > app->configuration.maxCodeLength)
            throw std::runtime_error("Matrix-generated kernel exceeds maxCodeLength");
        std::memcpy(axis->specializationConstants.code0, code.c_str(), code.size()+1);
        if (const char* directory = std::getenv("PYVKFFT_ASM_MATRIX_DUMP")) {
            std::ofstream out(std::string(directory) + "/matrix-axis-" + std::to_string(sc.axis_id) +
                              "-upload-" + std::to_string(sc.axis_upload_id) +
                              "-inverse-" + std::to_string(sc.actualInverse) + ".cu");
            out << code;
        }
        return true;
    } catch (const std::exception& e) { last_error = e.what(); return false; }
}

static MatrixPlan* matrix_create(uint64_t height, uint64_t width, uint64_t features, uint64_t kernels, uint64_t outputs,
                                 uint64_t canvas_h, uint64_t canvas_w, uint64_t in_y, uint64_t in_x,
                                 uint64_t out_h, uint64_t out_w, uint64_t out_y, uint64_t out_x, bool transpose) {
    std::unique_ptr<MatrixPlan> p(new MatrixPlan);
    p->height=height; p->width=width; p->features=features; p->kernels=kernels; p->outputs=outputs;
    p->bytes=canvas_h*canvas_w*features*kernels*8;
    p->canvas_h=canvas_h; p->canvas_w=canvas_w; p->in_y=in_y; p->in_x=in_x;
    p->out_h=out_h; p->out_w=out_w; p->out_y=out_y; p->out_x=out_x; p->transpose=transpose;
    if (cuCtxGetDevice(&p->device) != CUDA_SUCCESS) throw std::runtime_error("No current CUDA context");
    VkFFTConfiguration c{};
    c.FFTdim=2; c.size[0]=canvas_w; c.size[1]=canvas_h;
    c.device=&p->device; c.stream=&p->stream; c.num_streams=1;
    c.buffer=&p->buffer; c.bufferSize=&p->bytes;
    // XLA owns execution storage: never retain device scratch in a cached plan.
    c.userTempBuffer=1; c.tempBuffer=&p->temporary; c.tempBufferSize=&p->bytes;
    c.kernel=&p->kernel; c.kernelSize=&p->bytes;
    c.performConvolution=1; c.disableReorderFourStep=1; c.normalize=1;
    c.coordinateFeatures=features; c.matrixConvolution=features; c.numberKernels=kernels;
    // Entries are addressed through the component map; VkFFT's packed-symmetric indexing is not used.
    c.symmetricKernel=0;
    c.aimThreads=128; c.coalescedMemory=32;
    // The column convolution kernel holds M coordinates (twice for K > 1) of its whole FFT in registers. On a
    // single-upload 4096 column that spills heavily (A100: 20.6 ms instead of 1.5 ms for M=2, K=4); a 64 KB
    // shared-memory budget makes VkFFT split long columns into two uploads, so the convolution step runs on short
    // FFTs without spills. Short axes are unaffected.
    c.sharedMemorySize=65536;
    // Exploration only: PYVKFFT_ASM_MATRIX_TUNE="key=value,..." (coalesced, threads, max_threads, grouped_x,
    // grouped_y, shared, boost). Read when a plan is first built in a process.
    if (const char* tune = std::getenv("PYVKFFT_ASM_MATRIX_TUNE")) {
        std::stringstream items(tune);
        std::string item;
        while (std::getline(items, item, ',')) {
            auto at = item.find('=');
            if (at == std::string::npos) continue;
            std::string key = item.substr(0, at);
            uint64_t value = std::stoull(item.substr(at+1));
            if (key == "coalesced") c.coalescedMemory = value;
            else if (key == "threads") c.aimThreads = value;
            else if (key == "max_threads") c.maxThreadsNum = value;
            else if (key == "grouped_x") c.groupedBatch[0] = value;
            else if (key == "grouped_y") c.groupedBatch[1] = value;
            else if (key == "shared") c.sharedMemorySize = value;
            else if (key == "boost") c.registerBoost = value;
            else throw std::invalid_argument("Unknown PYVKFFT_ASM_MATRIX_TUNE key " + key);
        }
    }
    building=p.get();
    component_pending=false;
    VkFFTResult result=initializeVkFFT(&p->app, c);
    building=nullptr;
    if (result != VKFFT_SUCCESS)
        throw std::runtime_error("VkFFT matrix initialization failed: " + std::to_string(result) + " " + last_error);
    p->initialized=true;
    for (int i=0;i<2;i++)
        if (p->app.useBluesteinFFT[i] || p->app.localFFTPlan->numAxisUploads[i]>2)
            throw std::invalid_argument("Matrix convolution supports radix FFTs with at most two uploads per axis");
    for (auto* fp : {p->app.localFFTPlan, p->app.localFFTPlan_inverse}) {
        if (!fp) continue;
        for (int axis=0; axis<2; ++axis)
            for (uint64_t upload=0; upload<fp->numAxisUploads[axis]; ++upload) {
                auto& sc=fp->axes[axis][upload].specializationConstants;
                p->needs_temporary |= sc.tempBufferInput || sc.tempBufferOutput;
            }
    }
    auto* fp=p->app.localFFTPlan;
    std::ostringstream info;
    info << "{\"uploads\":[" << fp->numAxisUploads[0] << ',' << fp->numAxisUploads[1] << "],\"axis_split\":[";
    for (int i=0;i<2;i++) {
        if(i) info << ',';
        info << '[' << fp->axisSplit[i][0] << ',' << fp->axisSplit[i][1] << ']';
    }
    info << "],\"features\":" << features << ",\"kernels\":" << kernels << ",\"outputs\":" << outputs
         << ",\"transpose\":" << int(transpose)
         << ",\"canvas\":[" << canvas_h << ',' << canvas_w << "],\"input_window\":[" << in_y << ','
         << in_x << ',' << height << ',' << width << "],\"output_window\":[" << out_y << ',' << out_x
         << ',' << out_h << ',' << out_w << "],\"workspace_bytes\":" << p->bytes
         << ",\"scratch_bytes\":" << (p->needs_temporary ? p->bytes : 0)
         << ",\"vkfft_version\":" << VkFFTGetVersion() << ",\"compiled\":[";
    bool first = true;
    for (const auto& source : p->sources) {
        auto* plan = source.inverse ? p->app.localFFTPlan_inverse : fp;
        auto& axis = plan->axes[source.axis][source.upload];
        auto& sc = axis.specializationConstants;
        int registers = 0, local_bytes = 0, active_blocks = 0;
        cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, axis.VkFFTKernel);
        cuFuncGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, axis.VkFFTKernel);
        int threads = sc.localSize[0].data.i * sc.localSize[1].data.i * sc.localSize[2].data.i;
        cuOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, axis.VkFFTKernel,
                                                   threads, sc.usedSharedMemory.data.i);
        // A few spilled bytes are harmless; a register-starved step spills hundreds per thread (0.2-1.8 KB seen).
        if (sc.convolutionStep && local_bytes > 64) p->conv_spills = true;
        if (!first) info << ',';
        first = false;
        info << "{\"axis\":" << source.axis << ",\"upload\":" << source.upload
             << ",\"inverse\":" << source.inverse << ",\"source_fnv1a64\":\""
             << std::hex << source.hash << std::dec << "\",\"block\":["
             << sc.localSize[0].data.i << ',' << sc.localSize[1].data.i << ',' << sc.localSize[2].data.i
             << "],\"registers\":" << registers << ",\"local_bytes\":" << local_bytes
             << ",\"shared_bytes\":" << sc.usedSharedMemory.data.i
             << ",\"active_blocks_per_sm\":" << active_blocks << '}';
    }
    info << "],\"conv_spills\":" << int(p->conv_spills) << '}';
    p->info=info.str();
    return p.release();
}

struct FFIPlan {
    MatrixPlan* plan;
    CUcontext context;
    std::mutex submission;
    FFIPlan(MatrixPlan* plan, CUcontext context) : plan(plan), context(context) {}
    ~FFIPlan() {
        // Never unload kernels before previously submitted work finishes.
        if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            cuCtxSynchronize();
            delete plan;
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    }
};
using PlanKey = std::tuple<CUcontext, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, int,
                           std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>>;
struct PlanCache {
    std::mutex mutex;
    std::map<PlanKey, std::shared_ptr<FFIPlan>> plans;
};
static PlanCache& ffi_cache() {
    static auto* cache = new PlanCache;   // avoid CUDA teardown after the runtime is unloaded
    return *cache;
}

static std::shared_ptr<FFIPlan> get_plan(uint64_t h, uint64_t w, uint64_t features, uint64_t kernels, bool transpose,
                                         uint64_t ch, uint64_t cw, uint64_t iy, uint64_t ix,
                                         uint64_t oh, uint64_t ow, uint64_t oy, uint64_t ox, uint64_t outputs=0) {
    if (!outputs) outputs=features;
    CUcontext context = nullptr;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context)
        throw std::runtime_error("XLA FFI has no current CUDA context");
    PlanKey key{context, h, w, features, kernels, outputs, int(transpose), std::make_tuple(ch, cw, iy, ix, oh, ow, oy, ox)};
    auto& cache=ffi_cache();
    std::lock_guard<std::mutex> guard(cache.mutex);
    auto found=cache.plans.find(key);
    if (found != cache.plans.end()) return found->second;
    auto result=std::make_shared<FFIPlan>(matrix_create(h, w, features, kernels, outputs, ch, cw, iy, ix, oh, ow,
                                                        oy, ox, transpose), context);
    cache.plans.emplace(key, result);
    return result;
}

// A K-kernel convolution kernel that spills (long single-upload columns hold 2*M*column registers per thread)
// is slower than K single-kernel passes: use the K-kernel plan only when it is spill-free and the workspace
// holds K*M canvases.
static std::shared_ptr<FFIPlan> select_plan(uint64_t h, uint64_t w, uint64_t features, uint64_t kernels,
                                            bool transpose, uint64_t ch, uint64_t cw, uint64_t iy, uint64_t ix,
                                            uint64_t oh, uint64_t ow, uint64_t oy, uint64_t ox, bool room) {
    if (kernels>1 && room) {
        auto multi=get_plan(h, w, features, kernels, transpose, ch, cw, iy, ix, oh, ow, oy, ox);
        if (!multi->plan->conv_spills) return multi;
    }
    return get_plan(h, w, features, 1, transpose, ch, cw, iy, ix, oh, ow, oy, ox);
}

static ffi::Error MatrixFFIImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::C64> input, ffi::Buffer<ffi::C64> kernel,
    ffi::Result<ffi::Buffer<ffi::C64>> output, ffi::Result<ffi::Buffer<ffi::C64>> workspace,
    int64_t reduce, int64_t transpose, int64_t canvas_h, int64_t canvas_w, int64_t input_y, int64_t input_x,
    int64_t output_y, int64_t output_x, ffi::Span<const int64_t> components) {
    try {
        // expand: input [V, B, M, h, w] -> output [V, B, K, M, oh, ow]
        // reduce: input [V, B, K, M, h, w] -> output [V, B, M, oh, ow]
        // kernel planes [Vk, P, CH, CW] with Vk in {1, V}; components: K*M*M plane indices (-1 = zero)
        auto in=input.dimensions(), out=output->dimensions(), kd=kernel.dimensions();
        size_t lead = reduce ? 3 : 2;
        if (in.size()!=lead+3 || out.size()!=(reduce ? 5 : 6) || kd.size()!=4)
            return ffi::Error::InvalidArgument("Matrix convolution operand ranks do not match the mode");
        uint64_t V=in[0], B=in[1], M=in[lead], h=in[lead+1], w=in[lead+2];
        uint64_t K=reduce ? in[2] : out[2], oh=out[out.size()-2], ow=out[out.size()-1];
        uint64_t P=kd[1], Vk=kd[0];
        if (!V || !B || !h || !w || !oh || !ow || !P || M<1 || M>uint64_t(MAX_FEATURES) || K<1 ||
                K>uint64_t(MAX_KERNELS))
            return ffi::Error::InvalidArgument("Matrix convolution needs 1 <= M <= 4 features and 1 <= K <= 16 kernels");
        if (out[0]!=int64_t(V) || out[1]!=int64_t(B) || (reduce ? out[2]!=int64_t(M) : (out[3]!=int64_t(M))))
            return ffi::Error::InvalidArgument("Matrix convolution output shape mismatch");
        if (!(Vk==1 || Vk==V) || kd[2]!=canvas_h || kd[3]!=canvas_w)
            return ffi::Error::InvalidArgument("Kernel planes must be [1 or V, P, canvas_h, canvas_w]");
        if (components.size()!=K*M*M)
            return ffi::Error::InvalidArgument("Component map must hold K*M*M entries");
        for (auto c : components)
            if (c<-1 || c>=int64_t(P))
                return ffi::Error::InvalidArgument("Component map entries must be -1 or a kernel plane index");
        if (canvas_h<1 || canvas_w<1 || input_y<0 || input_x<0 || output_y<0 || output_x<0 ||
                uint64_t(canvas_h) > uint64_t(INT64_MAX)/uint64_t(canvas_w)/64/MAX_KERNELS ||
                uint64_t(input_y)+h > uint64_t(canvas_h) || uint64_t(input_x)+w > uint64_t(canvas_w) ||
                uint64_t(output_y)+oh > uint64_t(canvas_h) || uint64_t(output_x)+ow > uint64_t(canvas_w))
            return ffi::Error::InvalidArgument("Input and output windows must lie inside the FFT canvas");
        if (transpose<0 || transpose>1)
            return ffi::Error::InvalidArgument("transpose must be 0 or 1");
        uint64_t S=uint64_t(canvas_h)*uint64_t(canvas_w);
        if (workspace->element_count()%S || workspace->element_count()<M*S)
            return ffi::Error::InvalidArgument("Workspace must hold M (single-kernel passes) or K*M canvases");
        if (input.typed_data()==output->typed_data() || input.typed_data()==workspace->typed_data() ||
                output->typed_data()==workspace->typed_data())
            return ffi::Error::InvalidArgument("Input, output and workspace must not alias");
        // A workspace of K*M canvases permits the K-kernel plan (expand) or the fused plan (reduce, K*M <= 8);
        // M canvases select K single-kernel passes.
        bool room=workspace->element_count()>=K*M*S;
        std::shared_ptr<FFIPlan> cached;
        if (!reduce)
            cached=select_plan(h, w, M, K, transpose, canvas_h, canvas_w, input_y, input_x, oh, ow, output_y,
                               output_x, room);
        else if (room && K>1 && K*M<=uint64_t(MAX_FUSED)) {
            cached=get_plan(h, w, K*M, 1, transpose, canvas_h, canvas_w, input_y, input_x, oh, ow, output_y,
                            output_x, M);
            if (cached->plan->conv_spills) cached=nullptr;
        }
        if (!cached)
            cached=get_plan(h, w, M, 1, transpose, canvas_h, canvas_w, input_y, input_x, oh, ow, output_y, output_x);
        std::lock_guard<std::mutex> guard(cached->submission);
        auto* p=cached->plan;
        bool fused=reduce && p->features!=M;      // one (K*M) x (K*M) pass per field
        bool looped=!fused && p->kernels!=K;     // one single-kernel pass per k
        if (workspace->element_count()*8<p->bytes)
            return ffi::Error::Internal("Matrix convolution workspace is smaller than the selected plan needs");
        void* work=workspace->typed_data();
        void* temp=nullptr;
        if (p->needs_temporary) {
            auto extra=scratch.Allocate(p->bytes, 256);
            if (!extra) return ffi::Error::Internal("Unable to allocate XLA FFT scratch");
            temp=*extra;
        }
        p->stream=reinterpret_cast<CUstream>(stream);
        p->app.configuration.stream[0]=stream;
        auto* fp=p->app.localFFTPlan;
        MatrixArgs a{};
        a.ax=fp->numAxisUploads[0]==2 ? fp->axisSplit[0][0] : uint64_t(canvas_w);
        a.bx=fp->numAxisUploads[0]==2 ? fp->axisSplit[0][1] : 1;
        a.ay=fp->numAxisUploads[1]==2 ? fp->axisSplit[1][0] : uint64_t(canvas_h);
        a.by=fp->numAxisUploads[1]==2 ? fp->axisSplit[1][1] : 1;
        for (auto& entry : a.plane) entry=-1;
        if (fused)   // row o, feature (k, i) of the (K*M) x (K*M) matrix is M_k[o, i]; rows >= M stay zero
            for (uint64_t k=0; k<K; ++k)
                for (uint64_t o=0; o<M; ++o)
                    for (uint64_t i=0; i<M; ++i)
                        a.plane[o*K*M + k*M + i]=int32_t(components[(k*M + o)*M + i]);
        else if (!looped)
            for (uint64_t e=0; e<K*M*M; ++e) a.plane[e]=int32_t(components[e]);
        auto* x=reinterpret_cast<const std::complex<float>*>(input.typed_data());
        auto* y=reinterpret_cast<std::complex<float>*>(output->typed_data());
        auto* planes_base=reinterpret_cast<const std::complex<float>*>(kernel.typed_data());
        void* kernel_pointer=work;   // all kernel loads are hooked; VkFFT only binds the pointer
        VkFFTLaunchParams launch{};
        launch.buffer=&work; launch.kernel=&kernel_pointer;
        if (temp) launch.tempBuffer=&temp;
        for (uint64_t v=0; v<V; ++v) {
            a.kernel=planes_base + (Vk>1 ? v : 0)*P*S;
            for (uint64_t b=0; b<B; ++b) {
                uint64_t vb=v*B+b;
                for (uint64_t k=0; k<(looped ? K : 1); ++k) {
                    if (looped)
                        for (uint64_t e=0; e<M*M; ++e) a.plane[e]=int32_t(components[k*M*M+e]);
                    if (reduce) {
                        a.input=x + (vb*K + k)*M*h*w;
                        a.output=y + vb*M*oh*ow;
                        a.accumulate=k>0;
                    } else {
                        a.input=x + vb*M*h*w;
                        a.output=y + (vb*K + (looped ? k : 0))*M*oh*ow;
                        a.accumulate=0;
                    }
                    arguments=a;
                    executing=p;
                    int status=static_cast<int>(VkFFTAppend(&p->app, -1, &launch));
                    executing=nullptr;
                    if (status) return ffi::Error::Internal("VkFFT execution failed: " + std::to_string(status));
                }
            }
        }
        return ffi::Error::Success();
    } catch (const std::exception& error) {
        executing=nullptr;
        return ffi::Error::Internal(error.what());
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PyVkFFTASMMatrix, MatrixFFIImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::Buffer<ffi::C64>>().Arg<ffi::Buffer<ffi::C64>>()
        .Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>()
        .Attr<int64_t>("reduce").Attr<int64_t>("transpose")
        .Attr<int64_t>("canvas_h").Attr<int64_t>("canvas_w")
        .Attr<int64_t>("input_y").Attr<int64_t>("input_x")
        .Attr<int64_t>("output_y").Attr<int64_t>("output_x")
        .Attr<ffi::Span<const int64_t>>("components"));

extern "C" {
// ABI 1 of the matrix target (pyvkfft_asm_matrix_v1).
int asm_matrix_ffi_abi_version() { return 1; }
const char* asm_matrix_ffi_last_error() { return last_error.c_str(); }
void asm_matrix_ffi_clear_cache() {
    auto& cache=ffi_cache();
    std::map<PlanKey, std::shared_ptr<FFIPlan>> retired;
    {
        std::lock_guard<std::mutex> guard(cache.mutex);
        retired.swap(cache.plans);
    }
}
const char* asm_matrix_ffi_cache_info() {
    static thread_local std::string info;
    auto& cache=ffi_cache();
    std::lock_guard<std::mutex> guard(cache.mutex);
    std::ostringstream text;
    text << "[";
    bool first=true;
    for (const auto& item : cache.plans) {
        if (!first) text << ',';
        first=false;
        text << "{\"shape\":[" << std::get<1>(item.first) << ',' << std::get<2>(item.first)
             << "],\"plan\":" << item.second->plan->info << '}';
    }
    text << ']';
    info=text.str();
    return info.c_str();
}
}
