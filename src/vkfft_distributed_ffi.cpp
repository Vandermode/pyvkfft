// Local stages for JAX-owned distributed FFT / windowed ASM collectives.
#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#define VKFFT_STREAMED_FFI 1
#include "vkfft_streamed.cpp"
#include "distributed_kernels.h"
#include "xla/ffi/api/ffi.h"
namespace ffi=xla::ffi;

static void check(CUresult r) {
    if(r!=CUDA_SUCCESS) { const char* s=nullptr;cuGetErrorString(r,&s);throw std::runtime_error(s?s:"CUDA error"); }
}
struct LocalPlan {
    std::unique_ptr<StreamedFFT> fft;
    CUmodule module=nullptr;
    std::map<std::string,CUfunction> kernels;
    std::mutex mutex;
    std::once_flag initialized;
};
using Key=std::tuple<CUcontext,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t>;
static std::mutex cache_mutex;
// Context-owned plans remain alive for the process, like XLA executable caches.
static auto* plans=new std::map<Key,std::shared_ptr<LocalPlan>>;
static std::shared_ptr<LocalPlan> plan(int64_t mode,int64_t h,int64_t w,int64_t k,
        int64_t active,int64_t th,int64_t origin,int64_t tile,int64_t inverse,int64_t groups) {
    CUcontext ctx;check(cuCtxGetCurrent(&ctx));
    if(!ctx)throw std::runtime_error("No CUDA context");
    Key key{ctx,mode,h,w,k,active,th,origin,tile,inverse,groups};
    std::shared_ptr<LocalPlan> p;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto& entry=(*plans)[key];
        if(!entry)entry=std::make_shared<LocalPlan>();
        p=entry;
    }
    // CUDA plan/module initialization can wait for work already on a device.
    // Never hold the cross-context cache lock here: another GPU may need to
    // initialize its own plan before it can enter a pending collective.
    std::call_once(p->initialized,[&] {
    auto n=mode==2?2*h:(mode==3?w:2*w);
    p->fft.reset(static_cast<StreamedFFT*>(streamed_fft_create_ex(n,tile,0,mode!=2)));
    if(!p->fft)throw std::runtime_error(streamed_error);
    auto* fp=p->fft->app.localFFTPlan;
    uint64_t split[3]={1,1,1};
    if(mode==2)for(uint64_t i=0;i<fp->numAxisUploads[0];++i)split[i]=fp->axisSplit[0][i];
    std::ostringstream source;
    for(auto v:std::vector<std::pair<std::string,int64_t>>{{"H",h},{"W",w},{"K",k},
            {"ACTIVE",active},{"TH",th},{"ORIGIN",origin},{"B",tile},{"GROUPS",groups},
            {"AY",split[0]},{"BY",split[1]},{"CY",split[2]}})
        source<<"#define "<<v.first<<" ("<<v.second<<"LL)\n";
    source<<distributed_kernel_source;
    nvrtcProgram program;auto code=source.str();
    if(nvrtcCreateProgram(&program,code.c_str(),"distributed.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)
        throw std::runtime_error("NVRTC program creation failed");
    CUdevice device;int major,minor;check(cuCtxGetDevice(&device));
    check(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));
    check(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
    std::string arch="--gpu-architecture=sm_"+std::to_string(major)+std::to_string(minor);
    const char* opts[]={arch.c_str(),"--std=c++14"};
    if(nvrtcCompileProgram(program,2,opts)!=NVRTC_SUCCESS) {
        size_t size;nvrtcGetProgramLogSize(program,&size);std::string log(size,'\0');
        nvrtcGetProgramLog(program,log.data());nvrtcDestroyProgram(&program);throw std::runtime_error(log);
    }
    size_t size;nvrtcGetCUBINSize(program,&size);std::string cubin(size,'\0');
    nvrtcGetCUBIN(program,cubin.data());nvrtcDestroyProgram(&program);
    check(cuModuleLoadData(&p->module,cubin.c_str()));
    for(const char* name:{"row_pad","row_select","row_expand","row_crop","col_pad","col_crop","multiply"})
        check(cuModuleGetFunction(&p->kernels[name],p->module,name));
    });
    return p;
}
static ffi::Error LocalImpl(cudaStream_t stream,ffi::Buffer<ffi::C64> input,
        ffi::Buffer<ffi::U32> transfer,ffi::Buffer<ffi::F32> scale,
        ffi::Result<ffi::Buffer<ffi::C64>> output,ffi::Result<ffi::Buffer<ffi::C64>> workspace,
        int64_t mode,int64_t width,int64_t active,int64_t origin,int64_t tile,int64_t inverse,int64_t groups) {
    try {
        auto d=input.dimensions(),o=output->dimensions(),hd=transfer.dimensions();
        if(d.size()!=2 || o.size()!=2 || hd.size()!=2 || mode<0 || mode>3 ||
                width<1 || tile<1 || tile>65536 || inverse<0 || inverse>1 || groups<1 || scale.element_count()!=1)
            return ffi::Error::InvalidArgument("Invalid local FFT stage");
        int64_t h=d[0],w=width,k=mode==0?o[1]:d[1],th=hd[0];
        if(h<1 || k<1 || k%groups || h>INT64_MAX/std::max(w,k)/32 || active<1 || active>2*w ||
                origin<0 || origin>=(mode==2?2*h:2*w) || o[0]!=h ||
                (mode==0 && (d[1]!=w || k<active)) ||
                (mode==1 && (o[1]!=w || k<active)) ||
                (mode>=2 && o[1]!=d[1]) || (mode==2 && (hd[1]!=k || th>2*h)) ||
                (mode==3 && (d[1]!=w || h%tile)))
            return ffi::Error::InvalidArgument("Local FFT dimensions do not match");
        int64_t n=mode==2?2*h:(mode==3?w:2*w);
        if(workspace->element_count()!=2*n*tile)
            return ffi::Error::InvalidArgument("Incorrect local FFT workspace");
        auto p=plan(mode,h,w,k,active,th,origin,tile,inverse,groups);
        std::lock_guard<std::mutex> lock(p->mutex);
        void* src=input.typed_data();void* dst=output->typed_data();void* scratch=workspace->typed_data();
        void* temp=static_cast<char*>(scratch)+n*tile*8;void* payload=transfer.typed_data();void* factor=scale.typed_data();
        auto launch=[&](const char* name,int64_t blocks,int x,int y,std::initializer_list<void*> args) {
            std::vector<void*> argv(args);check(cuLaunchKernel(p->kernels.at(name),blocks,1,1,x,y,1,0,stream,argv.data(),nullptr));
        };
        auto fft=[&](void* buffer,bool inv) {
            auto* fp=p->fft.get();fp->stream=stream;fp->app.configuration.stream[0]=stream;
            VkFFTLaunchParams params{};params.buffer=&buffer;params.tempBuffer=&temp;
            auto r=VkFFTAppend(&fp->app,inv?1:-1,&params);
            if(r!=VKFFT_SUCCESS)throw std::runtime_error("Local VkFFT execution: "+std::to_string(r));
        };
        if(mode==3) {
            if(src!=dst)check(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst),reinterpret_cast<CUdeviceptr>(src),h*w*8,stream));
            for(int64_t off=0;off<h;off+=tile)fft(static_cast<char*>(dst)+off*w*8,inverse);
        }else if(mode==2) {
            if(src!=dst)check(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst),reinterpret_cast<CUdeviceptr>(src),h*k*8,stream));
            for(int64_t off=0;off<k;off+=tile) {
                launch("col_pad",((tile+31)/32)*((2*h+31)/32),32,8,{&dst,&scratch,&off});
                fft(scratch,false);
                launch("multiply",((tile+31)/32)*((2*h+31)/32),32,8,{&scratch,&payload,&factor,&off});
                fft(scratch,true);
                launch("col_crop",((h+31)/32)*((tile+31)/32),32,8,{&scratch,&dst,&off});
            }
        }else for(int64_t off=0;off<h;off+=tile) {
            launch(mode==0?"row_pad":"row_expand",(tile*2*w+255)/256,256,1,{&src,&scratch,&off});
            fft(scratch,mode==1);
            launch(mode==0?"row_select":"row_crop",(tile*(mode==0?k:w)+255)/256,256,1,{&scratch,&dst,&off});
        }
        return ffi::Error::Success();
    }catch(const std::exception& e){return ffi::Error::Internal(e.what());}
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(PyVkFFTDistributed,LocalImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::C64>>().Arg<ffi::Buffer<ffi::U32>>().Arg<ffi::Buffer<ffi::F32>>()
    .Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>()
    .Attr<int64_t>("mode").Attr<int64_t>("width").Attr<int64_t>("active")
    .Attr<int64_t>("origin").Attr<int64_t>("tile").Attr<int64_t>("inverse").Attr<int64_t>("groups"));
extern "C" int distributed_ffi_abi_version(){return 2;}
