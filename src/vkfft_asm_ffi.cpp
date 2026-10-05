// Typed XLA FFI for the fused, compact-input/output ASM engine.
// Execution buffers belong to XLA; cached plans own compiled kernels only.
#include <map>
#include <mutex>
#include <string>
#include <tuple>

struct ASMPlan;
static std::string asm_ffi_customize_helpers(std::string, const ASMPlan&);
#define VKFFT_ASM_FFI 1
#include "vkfft_asm.cpp"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

struct FFISpecialization {
    int mode=0, dz=0, dw=0, transpose=0, parameter_dtype=0;
};
static thread_local FFISpecialization ffi_specialization;

static std::string asm_ffi_customize_helpers(std::string source, const ASMPlan&) {
    const auto at = source.find("__device__ __forceinline__ float2 asm_transfer(");
    if (at == std::string::npos) throw std::runtime_error("Missing ASM transfer hook");
    source.resize(at);
    source += R"CUDA(
__device__ __forceinline__ float2 asm_transfer(ASMArgs a, unsigned long long i) {
    unsigned long long py = i / (2*ASM_W), px = i % (2*ASM_W);
    unsigned long long y = (py % a.ay)*a.by + py/a.ay;
    unsigned long long x = (px % a.ax)*a.bx + px/a.ax;
    if (a.transfer_mode) {
        if (a.transfer_mode == 2) {
            y = y <= ASM_H ? y : 2*ASM_H-y;
            x = x <= ASM_W ? x : 2*ASM_W-x;
            return ((const float2*)a.transfer)[y*(ASM_W+1)+x];
        }
        // JAX transposes complex-linear maps using the bilinear convention.
        // Reversing frequencies implements A^T; conjugating H would give A*.
        if (a.transpose) {
            y = y ? 2*ASM_H-y : 0;
            x = x ? 2*ASM_W-x : 0;
        }
        if (a.transfer_mode == 3) {
            unsigned int bits = ((const unsigned int*)a.transfer)[y*(2*ASM_W)+x];
            int re = int(bits & 65535u), im = int(bits >> 16);
            re = re >= 32768 ? re-65536 : re;
            im = im >= 32768 ? im-65536 : im;
            // Packed modes reuse the first scalar parameter for the per-plane
            // scale. Decode at the spectrum load, never into a dense H buffer.
            float scale = *(const float*)a.parameter_z;
            return make_float2(float(re)*scale, float(im)*scale);
        }
        return ((const float2*)a.transfer)[y*(2*ASM_W)+x];
    }
    double z = a.parameter_dtype ? *(const double*)a.parameter_z : *(const float*)a.parameter_z;
    double wavelength = a.parameter_dtype ? *(const double*)a.parameter_wavelength : *(const float*)a.parameter_wavelength;
    if (!isfinite(z) || !isfinite(wavelength) || wavelength <= 0.)
        return make_float2(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    double fy = (y < ASM_H ? (long long)y : (long long)y - (long long)(2*ASM_H)) / (2.*ASM_H*ASM_DY);
    double fx = (x < ASM_W ? (long long)x : (long long)x - (long long)(2*ASM_W)) / (2.*ASM_W*ASM_DX);
    double il = 1. / wavelength;
    double q = il*il - (fx*fx+fy*fy);
    if (q < 0.) return make_float2(0.f, 0.f);
    if (ASM_BANDLIMIT) {
        double tx=z/(ASM_W*ASM_DX), ty=z/(ASM_H*ASM_DY);
        if (fabs(fx)>il/sqrt(1.+tx*tx) || fabs(fy)>il/sqrt(1.+ty*ty))
            return make_float2(0.f, 0.f);
    }
    double root = sqrt(q), k = 6.283185307179586476925286766559 * root;
    double sn, cs;
    sincos(z*k, &sn, &cs);
    if (!(a.dz_order || a.dw_order)) return make_float2((float)cs, (float)sn);
    // Hard support boundaries are piecewise constant. At q==0 use the
    // finite clipped-branch convention instead of a divergent wavelength slope.
    // Differentiate the residual phase k-k0 here. Python differentiates the
    // uniform carrier separately against the already computed primal output.
    // Otherwise C64 FFT roundoff on i*k0*H overwhelms small physical gradients.
    double delta = -6.283185307179586476925286766559 * (fx*fx+fy*fy)/(root+il);
    double dw = root > 0. ? delta*il*il/root : 6.283185307179586476925286766559*il*il;
    double real=0., imag=0.;
    if (a.dz_order == 1 && a.dw_order == 0) imag=delta;
    else if (a.dz_order == 0 && a.dw_order == 1) imag=z*dw;
    else if (a.dz_order == 2) real=-delta*delta;
    else if (a.dz_order == 1) { real=-delta*z*dw; imag=dw; }
    else {
        double t=root > 0. ? (fx*fx+fy*fy)/((root+il)*root) : 0.;
        double dww=root > 0. ? -6.283185307179586476925286766559 * il*il*il*t*t*(il/root+2.)
                            : -12.566370614359172953850573533118*il*il*il;
        real=-z*z*dw*dw; imag=z*dww;
    }
    return make_float2((float)(real*cs-imag*sn), (float)(real*sn+imag*cs));
}
)CUDA";
    // A materialized transfer must not pay the register cost of the analytic
    // phase/derivative branches. Specialize these uniform choices at planning.
    for (const auto& item : {
            std::make_pair("a.transfer_mode", ffi_specialization.mode),
            std::make_pair("a.dz_order", ffi_specialization.dz),
            std::make_pair("a.dw_order", ffi_specialization.dw),
            std::make_pair("a.transpose", ffi_specialization.transpose),
            std::make_pair("a.parameter_dtype", ffi_specialization.parameter_dtype)}) {
        size_t pos=0;
        const auto value=std::to_string(item.second);
        while ((pos=source.find(item.first, pos)) != std::string::npos) {
            source.replace(pos, std::strlen(item.first), value);
            pos+=value.size();
        }
    }
    return source;
}

struct FFIPlan {
    ASMPlan* plan;
    CUcontext context;
    std::mutex submission;
    bool needs_temporary = false;
    FFIPlan(ASMPlan* plan, CUcontext context) : plan(plan), context(context) {
        for (auto* fp : {plan->app.localFFTPlan, plan->app.localFFTPlan_inverse}) {
            if (!fp) continue;
            for (int axis=0; axis<2; ++axis)
                for (uint64_t upload=0; upload<fp->numAxisUploads[axis]; ++upload) {
                    auto& sc=fp->axes[axis][upload].specializationConstants;
                    needs_temporary |= sc.tempBufferInput || sc.tempBufferOutput;
                }
        }
    }
    ~FFIPlan() {
        // A cache can be cleared while an executable still exists. It is never
        // safe to unload its kernels until previously submitted work finishes.
        if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            cuCtxSynchronize();
            asm_destroy(plan);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    }
};

using PlanKey = std::tuple<CUcontext, uint64_t, uint64_t, double, double,
                           int, int, int, int, int, int, int>;
struct PlanCache {
    std::mutex mutex;
    std::map<PlanKey, std::shared_ptr<FFIPlan>> plans;
};
static PlanCache& ffi_cache() {
    // Avoid CUDA teardown after the runtime has already been unloaded.
    // clear_plan_cache() explicitly releases cached modules during normal use.
    static auto* cache = new PlanCache;
    return *cache;
}

static std::shared_ptr<FFIPlan> get_ffi_plan(uint64_t h, uint64_t w, double dy,
                                           double dx, int bandlimit, int grouped,
                                           FFISpecialization specialization) {
    CUcontext context = nullptr;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context)
        throw std::runtime_error("XLA FFI has no current CUDA context");
    PlanKey key{context, h, w, dy, dx, bandlimit, grouped, specialization.mode,
                 specialization.dz, specialization.dw, specialization.transpose,
                 specialization.parameter_dtype};
    auto& cache=ffi_cache();
    std::lock_guard<std::mutex> guard(cache.mutex);
    auto found=cache.plans.find(key);
    if (found != cache.plans.end()) return found->second;
    ffi_specialization=specialization;
    auto* native=static_cast<ASMPlan*>(asm_create(h, w, dy, dx, 1, bandlimit,
                                                 1, 0, 128, 32, 0, grouped));
    if (!native) throw std::runtime_error(asm_last_error());
    auto result=std::make_shared<FFIPlan>(native, context);
    cache.plans.emplace(key, result);
    return result;
}

static ffi::Error ASMFFIImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::C64> input, ffi::AnyBuffer transfer,
    ffi::AnyBuffer z, ffi::AnyBuffer wavelength,
    ffi::Result<ffi::Buffer<ffi::C64>> output,
    ffi::Result<ffi::Buffer<ffi::C64>> workspace,
    double dy, double dx, int64_t bandlimit, int64_t transfer_mode,
    int64_t dz_order, int64_t dw_order, int64_t transpose, int64_t grouped) {
    try {
        auto dims=input.dimensions();
        if (dims.size()!=2 || dims[0]<1 || dims[1]<1 ||
                !(std::isfinite(dy) && dy>0 && std::isfinite(dx) && dx>0))
            return ffi::Error::InvalidArgument("ASM expects a nonempty 2-D field and positive finite pitch");
        uint64_t h=dims[0], w=dims[1];
        if (h > uint64_t(INT64_MAX)/w/32 || output->dimensions().size()!=2 ||
                output->dimensions()[0]!=int64_t(h) || output->dimensions()[1]!=int64_t(w) ||
                workspace->element_count()!=4*h*w)
            return ffi::Error::InvalidArgument("ASM output/workspace shape mismatch or overflow");
        if (transfer_mode<0 || transfer_mode>3 || bandlimit<0 || bandlimit>1 ||
                transpose<0 || transpose>1 || grouped<0 || grouped>65536 ||
                dz_order<0 || dw_order<0 || dz_order+dw_order>2)
            return ffi::Error::InvalidArgument("Invalid ASM operation attributes");
        if (transfer.element_type()!=(transfer_mode==3 ? ffi::U32 : ffi::C64))
            return ffi::Error::InvalidArgument("ASM transfer dtype does not match storage mode");
        if (transfer_mode && (transfer.dimensions().size()!=2 ||
                transfer.dimensions()[0]!=int64_t(transfer_mode==2 ? h+1 : 2*h) ||
                transfer.dimensions()[1]!=int64_t(transfer_mode==2 ? w+1 : 2*w)))
            return ffi::Error::InvalidArgument("ASM transfer shape mismatch");
        if (z.element_count()!=1 || wavelength.element_count()!=1 ||
                z.element_type()!=wavelength.element_type() ||
                (z.element_type()!=ffi::F32 && z.element_type()!=ffi::F64))
            return ffi::Error::InvalidArgument("Optical parameters must be matching float32 or float64 scalars");
        if (transfer_mode==3 && (z.element_type()!=ffi::F32 || dz_order || dw_order))
            return ffi::Error::InvalidArgument("Packed transfer requires a float32 scale and no transfer derivatives");
        if (input.typed_data()==output->typed_data() ||
                input.typed_data()==workspace->typed_data() ||
                output->typed_data()==workspace->typed_data())
            return ffi::Error::InvalidArgument("ASM input, output and workspace must not alias");
        auto cached=get_ffi_plan(h, w, dy, dx, bandlimit, grouped,
            {int(transfer_mode), int(dz_order), int(dw_order), int(transpose),
             int(z.element_type()==ffi::F64)});
        std::lock_guard<std::mutex> guard(cached->submission);
        auto* p=cached->plan;
        void* temp=nullptr;
        if (cached->needs_temporary) {
            auto allocation=scratch.Allocate(p->bytes, 256);
            if (!allocation) return ffi::Error::Internal("Unable to allocate XLA FFT scratch");
            temp=*allocation;
        }
        p->stream=reinterpret_cast<CUstream>(stream);
        p->app.configuration.stream[0]=stream;
        auto* fp=p->app.localFFTPlan;
        arguments={input.typed_data(), output->typed_data(), 0., 1., 1, 1, 1, 1};
        arguments.ax=fp->numAxisUploads[0]==2 ? fp->axisSplit[0][0] : 2*w;
        arguments.bx=fp->numAxisUploads[0]==2 ? fp->axisSplit[0][1] : 1;
        arguments.ay=fp->numAxisUploads[1]==2 ? fp->axisSplit[1][0] : 2*h;
        arguments.by=fp->numAxisUploads[1]==2 ? fp->axisSplit[1][1] : 1;
        arguments.parameter_z=z.untyped_data();
        arguments.parameter_wavelength=wavelength.untyped_data();
        arguments.transfer=transfer.untyped_data();
        arguments.parameter_dtype=z.element_type()==ffi::F64;
        arguments.transfer_mode=transfer_mode;
        arguments.dz_order=dz_order; arguments.dw_order=dw_order;
        arguments.transpose=transpose;
        void* work=workspace->typed_data();
        void* kernel=transfer_mode ? transfer.untyped_data() : work;
        VkFFTLaunchParams launch{};
        launch.buffer=&work; launch.kernel=&kernel;
        if (temp) launch.tempBuffer=&temp;
        executing=p;
        int status=static_cast<int>(VkFFTAppend(&p->app, -1, &launch));
        executing=nullptr;
        if (status) return ffi::Error::Internal("VkFFT execution failed: " + std::to_string(status));
        return ffi::Error::Success();
    } catch (const std::exception& error) {
        executing=nullptr;
        return ffi::Error::Internal(error.what());
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PyVkFFTASM, ASMFFIImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::Buffer<ffi::C64>>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>()
        .Attr<double>("dy").Attr<double>("dx")
        .Attr<int64_t>("bandlimit").Attr<int64_t>("transfer_mode")
        .Attr<int64_t>("dz_order").Attr<int64_t>("dw_order")
        .Attr<int64_t>("transpose").Attr<int64_t>("grouped"));

extern "C" {
int asm_ffi_abi_version() { return 3; }
void asm_ffi_clear_cache() {
    auto& cache=ffi_cache();
    std::map<PlanKey, std::shared_ptr<FFIPlan>> retired;
    {
        std::lock_guard<std::mutex> guard(cache.mutex);
        retired.swap(cache.plans);
    }
}
const char* asm_ffi_cache_info() {
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
             << "],\"transfer_mode\":" << std::get<7>(item.first)
             << ",\"dz_order\":" << std::get<8>(item.first)
             << ",\"dw_order\":" << std::get<9>(item.first)
             << ",\"transpose\":" << std::get<10>(item.first)
             << ",\"scratch_bytes\":" << (item.second->needs_temporary ? item.second->plan->bytes : 0)
             << ",\"plan\":" << item.second->plan->info << '}';
    }
    text << ']';
    info=text.str();
    return info.c_str();
}
}
