// Experimental, opaque CUDA ASM API. Built only against an isolated header copy.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static bool asm_read_hook(void*, void*, void*, void*);
static bool asm_write_hook(void*, void*, void*, void*);
static bool asm_is_output(void*);
static bool asm_compile_hook(void*, void*);
static const void* asm_launch_arguments();
static void asm_dispatch_dimensions(void*, void*);
#define VKFFT_BACKEND 1
#define VKFFT_ASM_HOOKS 1
#include "vkFFT.h"

struct ASMArgs {
    const void* input;
    void* output;
    double z, wavelength;
    uint64_t ax, bx, ay, by;
#ifdef VKFFT_ASM_FFI
    const void* parameter_z;
    const void* parameter_wavelength;
    const void* transfer;
    int parameter_dtype, transfer_mode, dz_order, dw_order, transpose;
#endif
};
struct ASMPlan {
    VkFFTApplication app{};
    CUdevice device{};
    CUstream stream{};
    uint64_t height, width, bytes;
    double dy, dx;
    bool dynamic, bandlimit, prune;
    void* buffer = reinterpret_cast<void*>(1);
    void* kernel = reinterpret_cast<void*>(2);
#ifdef VKFFT_ASM_FFI
    void* temporary = reinterpret_cast<void*>(3);
#endif
    bool initialized = false;
    std::string info;
    struct KernelSource { uint64_t axis, upload, inverse, hash; };
    std::vector<KernelSource> sources;
    ~ASMPlan() { if (initialized) deleteVkFFT(&app); }
};
static thread_local ASMPlan* building = nullptr;
static thread_local ASMPlan* executing = nullptr;
static thread_local ASMArgs arguments{};
static thread_local std::string last_error;

static uint64_t asm_pruned_rows_per_group(const ASMPlan& plan, const VkFFTSpecializationConstantsLayout& sc) {
    if (!plan.prune || sc.axis_id != 0 || sc.swapComputeWorkGroupID || sc.mergeSequencesR2C)
        return 0;
    if (sc.numAxisUploads == 2) return 1;
    if (sc.numAxisUploads != 1) return 0;
    uint64_t batch = sc.localSize[sc.stridedSharedLayout ? 0 : 1].data.i;
    // Only trim complete independent row batches. Partial groups retain the
    // full dispatch and the existing boundary masks.
    return batch && plan.height % batch == 0 && (plan.height/2) % batch == 0 ? batch : 0;
}

static bool asm_is_output(void* opaque) {
    auto* sc = static_cast<VkFFTSpecializationConstantsLayout*>(opaque);
    if (!building) return false;
    bool final_row = sc->actualInverse && sc->axis_id == 0 && sc->axis_upload_id == sc->numAxisUploads - 1;
    bool final_column = building->prune && sc->axis_id == 1 &&
        ((sc->actualInverse && sc->axis_upload_id == sc->numAxisUploads - 1) ||
         (sc->convolutionStep && sc->numAxisUploads == 1));
    return final_row || final_column;
}
static bool asm_read_hook(void* opaque, void* out_ptr, void* buffer_ptr, void* index_ptr) {
    if (!building) return false;
    auto* sc = static_cast<VkFFTSpecializationConstantsLayout*>(opaque);
    auto* out = static_cast<PfContainer*>(out_ptr);
    auto* buffer = static_cast<PfContainer*>(buffer_ptr);
    auto* index = static_cast<PfContainer*>(index_ptr);
    bool input = !sc->actualInverse && sc->axis_id == 0 &&
                 sc->axis_upload_id == sc->numAxisUploads - 1 &&
                 !std::strcmp(buffer->name, sc->inputsStruct.name);
    bool transfer = building->dynamic && sc->convolutionStep &&
                    !std::strcmp(buffer->name, sc->kernelStruct.name);
    bool column = building->prune && !sc->actualInverse && sc->axis_id == 1 &&
                  sc->axis_upload_id == sc->numAxisUploads - 1 &&
                  !std::strcmp(buffer->name, sc->inputsStruct.name);
    if (!input && !transfer && !column) return false;
    if (column)
        sc->tempLen = std::snprintf(sc->tempStr, 4096, "%s = asm_load_column(%s, %s);\n", out->name, buffer->name, index->name);
    else
        sc->tempLen = std::snprintf(sc->tempStr, 4096, "%s = %s(asm_args, %s);\n",
                                 out->name, input ? "asm_load_input" : "asm_transfer", index->name);
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
    if (sc->axis_id == 0)
        sc->tempLen = std::snprintf(sc->tempStr, 4096, "asm_store_output(asm_args, %s, %s);\n", index->name, in->name);
    else
        sc->tempLen = std::snprintf(sc->tempStr, 4096, "asm_store_column(%s, %s, %s);\n", buffer->name, index->name, in->name);
    PfAppendLine(sc);
    return true;
}
static const void* asm_launch_arguments() { return &arguments; }
static void asm_dispatch_dimensions(void* axis_ptr, void* dimensions) {
    auto* axis = static_cast<VkFFTAxis*>(axis_ptr);
    auto& sc = axis->specializationConstants;
    if (executing) {
        uint64_t batch = asm_pruned_rows_per_group(*executing, sc);
        if (batch) static_cast<pfUINT*>(dimensions)[1] = executing->height/batch;
    }
}

static std::string device_helpers(const ASMPlan& plan) {
    std::ostringstream s;
    s << std::setprecision(17);
    s << "#define ASM_H " << plan.height << "ull\n#define ASM_W " << plan.width << "ull\n";
    s << "#define ASM_DY " << plan.dy << "\n#define ASM_DX " << plan.dx << "\n";
    s << "#define ASM_BANDLIMIT " << plan.bandlimit << "\n";
#ifdef VKFFT_ASM_FFI
    s << "#define VKFFT_ASM_FFI 1\n";
#endif
    s << R"CUDA(
struct ASMArgs {
    const void* input; void* output;
    double z, wavelength;
    unsigned long long ax, bx, ay, by;
#ifdef VKFFT_ASM_FFI
    const void* parameter_z;
    const void* parameter_wavelength;
    const void* transfer;
    int parameter_dtype, transfer_mode, dz_order, dw_order, transpose;
#endif
};
__device__ __forceinline__ float2 asm_load_column(const float2* source, unsigned long long i) {
    unsigned long long y = i/(2*ASM_W);
    return (y >= ASM_H/2 && y < ASM_H/2 + ASM_H) ? __ldg(source+i) : make_float2(0.f,0.f);
}
__device__ __forceinline__ void asm_store_column(float2* dest, unsigned long long i, float2 v) {
    unsigned long long y = i/(2*ASM_W);
    if(y >= ASM_H/2 && y < ASM_H/2 + ASM_H) dest[i]=v;
}
__device__ __forceinline__ float2 asm_load_input(ASMArgs a, unsigned long long i) {
    unsigned long long y = i / (2 * ASM_W), x = i % (2 * ASM_W);
    if (y < ASM_H/2 || y >= ASM_H/2 + ASM_H || x < ASM_W/2 || x >= ASM_W/2 + ASM_W)
        return make_float2(0.f, 0.f);
    return ((const float2*)a.input)[(y-ASM_H/2)*ASM_W + x-ASM_W/2];
}
__device__ __forceinline__ void asm_store_output(ASMArgs a, unsigned long long i, float2 v) {
    unsigned long long y = i / (2 * ASM_W), x = i % (2 * ASM_W);
    if (y >= ASM_H/2 && y < ASM_H/2 + ASM_H && x >= ASM_W/2 && x < ASM_W/2 + ASM_W)
        ((float2*)a.output)[(y-ASM_H/2)*ASM_W + x-ASM_W/2] = v;
}
__device__ __forceinline__ float2 asm_transfer(ASMArgs a, unsigned long long i) {
    unsigned long long py = i / (2*ASM_W), px = i % (2*ASM_W);
    long long y = (py % a.ay)*a.by + py/a.ay;
    long long x = (px % a.ax)*a.bx + px/a.ax;
    double fy = (y < ASM_H ? y : y - (long long)(2*ASM_H)) / (2.*ASM_H*ASM_DY);
    double fx = (x < ASM_W ? x : x - (long long)(2*ASM_W)) / (2.*ASM_W*ASM_DX);
    double invlambda = 1. / a.wavelength;
    double q = invlambda*invlambda - (fx*fx + fy*fy);
    if (q < 0.) return make_float2(0.f, 0.f);
    if (ASM_BANDLIMIT) {
        double tx = a.z/(ASM_W*ASM_DX), ty = a.z/(ASM_H*ASM_DY);
        if (fabs(fx) > invlambda/sqrt(1.+tx*tx) || fabs(fy) > invlambda/sqrt(1.+ty*ty))
            return make_float2(0.f, 0.f);
    }
    double phase = 6.283185307179586476925286766559 * a.z * sqrt(q);
    double sn, cs;
    sincos(phase, &sn, &cs);
    return make_float2((float)cs, (float)sn);
}
)CUDA";
#ifdef VKFFT_ASM_FFI
    return asm_ffi_customize_helpers(s.str(), plan);
#else
    return s.str();
#endif
}

static bool asm_compile_hook(void* app_ptr, void* axis_ptr) {
    if (!building) return false;
    auto* axis = static_cast<VkFFTAxis*>(axis_ptr);
    try {
        std::string code(axis->specializationConstants.code0);
        auto& sc = axis->specializationConstants;
        uint64_t row_batch = asm_pruned_rows_per_group(*building, sc);
        if (row_batch) {
            const std::string original = "shiftY = blockIdx.y;";
            auto at = code.find(original);
            if (sc.swapComputeWorkGroupID || at == std::string::npos ||
                    code.find(original, at+original.size()) != std::string::npos)
                throw std::runtime_error("Unsupported row-pruning workgroup mapping");
            code.replace(at, original.size(), "shiftY = blockIdx.y + " + std::to_string(building->height/2/row_batch) + ";");
        }
        const std::string signature = "VkFFT_main (";
        size_t start = code.find(signature);
        if (start == std::string::npos) throw std::runtime_error("Missing VkFFT_main signature");
        size_t end = code.find(')', start + signature.size());
        if (end == std::string::npos) throw std::runtime_error("Missing kernel argument terminator");
        code.insert(end, ", ASMArgs asm_args");
        code = device_helpers(*building) + code;
        // The fused coefficient uses the fixed plan permutation. Specialize
        // these divisors instead of emitting expensive runtime uint64 divides.
        if (building->dynamic && sc.convolutionStep) {
            auto* fp = sc.actualInverse ? building->app.localFFTPlan_inverse : building->app.localFFTPlan;
            if (!fp) throw std::runtime_error("Missing FFT plan at coefficient compilation");
            const uint64_t lengths[] = {2*building->width, 2*building->height};
            for (int axis_id = 0; axis_id < 2; ++axis_id) {
                uint64_t a = fp->numAxisUploads[axis_id] == 2 ? fp->axisSplit[axis_id][0] : lengths[axis_id];
                uint64_t b = fp->numAxisUploads[axis_id] == 2 ? fp->axisSplit[axis_id][1] : 1;
                if (!a || !b || a*b != lengths[axis_id])
                    throw std::runtime_error("Incomplete FFT permutation at coefficient compilation");
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
        // VkFFT's planner retains a second owning pointer to this allocation.
        // Preserve it, and fail closed if the configured code capacity is small.
        auto* app = static_cast<VkFFTApplication*>(app_ptr);
        if (code.size()+1 > app->configuration.maxCodeLength)
            throw std::runtime_error("ASM-generated kernel exceeds maxCodeLength");
        std::memcpy(axis->specializationConstants.code0, code.c_str(), code.size()+1);
        if (const char* directory = std::getenv("PYVKFFT_ASM_DUMP")) {
            auto& sc = axis->specializationConstants;
            std::ofstream out(std::string(directory) + "/axis-" + std::to_string(sc.axis_id) +
                              "-upload-" + std::to_string(sc.axis_upload_id) +
                              "-inverse-" + std::to_string(sc.actualInverse) + ".cu");
            out << code;
        }
        return true;
    } catch (const std::exception& e) { last_error = e.what(); return false; }
}

extern "C" {
uint32_t asm_abi_version() { return 2; }
const char* asm_last_error() { return last_error.c_str(); }
void* asm_create(uint64_t height, uint64_t width, double dy, double dx, int dynamic,
                 int bandlimit, int prune, uint64_t stream, int threads, int coalesced, int grouped_x, int grouped_y) {
    last_error.clear();
    try {
        if (!height || !width || height > UINT64_MAX/width/32 || !(dy > 0) || !(dx > 0))
            throw std::invalid_argument("Invalid dimensions or pixel pitch");
        std::unique_ptr<ASMPlan> p(new ASMPlan);
        p->height=height; p->width=width; p->bytes=height*width*32;
        p->dy=dy; p->dx=dx; p->dynamic=dynamic; p->bandlimit=bandlimit; p->prune=prune;
        p->stream=reinterpret_cast<CUstream>(stream);
        if (cuCtxGetDevice(&p->device) != CUDA_SUCCESS) throw std::runtime_error("No current CUDA context");
        VkFFTConfiguration c{};
        c.FFTdim=2; c.size[0]=2*width; c.size[1]=2*height;
        c.device=&p->device; c.stream=&p->stream; c.num_streams=1;
        c.buffer=&p->buffer; c.bufferSize=&p->bytes;
#ifdef VKFFT_ASM_FFI
        // XLA owns execution storage. Never retain device scratch in a cached
        // plan: the same compiled kernels can be submitted on different streams.
        c.userTempBuffer=1; c.tempBuffer=&p->temporary; c.tempBufferSize=&p->bytes;
#endif
        c.kernel=&p->kernel; c.kernelSize=&p->bytes;
        c.performConvolution=1; c.disableReorderFourStep=1; c.normalize=1;
        c.aimThreads=threads; c.coalescedMemory=coalesced;
        c.groupedBatch[0]=grouped_x; c.groupedBatch[1]=grouped_y;
        building=p.get();
        VkFFTResult result=initializeVkFFT(&p->app, c);
        building=nullptr;
        if (result != VKFFT_SUCCESS) throw std::runtime_error("VkFFT initialization failed: " + std::to_string(result) + " " + last_error);
        p->initialized=true;
        for (int i=0;i<2;i++)
            if (p->app.useBluesteinFFT[i] || p->app.localFFTPlan->numAxisUploads[i]>2)
                throw std::invalid_argument("ASM currently supports radix FFTs with at most two uploads per axis");
        auto* fp=p->app.localFFTPlan;
        std::ostringstream info;
        info << "{\"uploads\":[" << fp->numAxisUploads[0] << ',' << fp->numAxisUploads[1] << "],\"axis_split\":[";
        for (int i=0;i<2;i++) {
            if(i) info << ',';
            info << '[' << fp->axisSplit[i][0] << ',' << fp->axisSplit[i][1] << ']';
        }
        info << "],\"workspace_bytes\":" << p->bytes << ",\"temporary_bytes\":"
             << (p->app.configuration.allocateTempBuffer ? p->app.configuration.tempBufferSize[0] : 0)
             << ",\"pruned_row_stages\":" << (asm_pruned_rows_per_group(*p, fp->axes[0][0].specializationConstants) ? 2*fp->numAxisUploads[0] : 0)
             << ",\"vkfft_version\":" << VkFFTGetVersion() << ",\"kernels\":[";
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
        info << "]}";
        p->info=info.str();
        return p.release();
    } catch(const std::exception& e) { building=nullptr; last_error=e.what(); return nullptr; }
}
const char* asm_info(void* handle) { return static_cast<ASMPlan*>(handle)->info.c_str(); }
int asm_execute(void* handle, void* input, void* output, void* workspace, void* transfer,
                double z, double wavelength) {
    auto* p=static_cast<ASMPlan*>(handle);
    if (!p || !input || !output || !workspace || (!p->dynamic && !transfer)) return -1;
    auto* fp=p->app.localFFTPlan;
    arguments={input, output, z, wavelength, 1, 1, 1, 1};
    arguments.ax=fp->numAxisUploads[0]==2 ? fp->axisSplit[0][0] : 2*p->width;
    arguments.bx=fp->numAxisUploads[0]==2 ? fp->axisSplit[0][1] : 1;
    arguments.ay=fp->numAxisUploads[1]==2 ? fp->axisSplit[1][0] : 2*p->height;
    arguments.by=fp->numAxisUploads[1]==2 ? fp->axisSplit[1][1] : 1;
    void* kernel=transfer ? transfer : workspace;
    VkFFTLaunchParams launch{};
    launch.buffer=&workspace; launch.kernel=&kernel;
    executing=p;
    int result=static_cast<int>(VkFFTAppend(&p->app, -1, &launch));
    executing=nullptr;
    return result;
}
void asm_destroy(void* handle) { delete static_cast<ASMPlan*>(handle); }
}
