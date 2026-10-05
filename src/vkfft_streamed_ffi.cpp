// Capacity-oriented packed propagation. All execution storage belongs to XLA.
#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#define VKFFT_STREAMED_FFI 1
#include "vkfft_streamed.cpp"
#include "streamed_kernels.h"
#include "xla/ffi/api/ffi.h"
namespace ffi=xla::ffi;

static void cuda_check(CUresult result) {
    if(result!=CUDA_SUCCESS) {
        const char* message=nullptr; cuGetErrorString(result,&message);
        throw std::runtime_error(message?message:"CUDA driver error");
    }
}
struct TilePlan {
    std::unique_ptr<StreamedFFT> row, column;
    CUcontext context=nullptr;
    CUmodule module=nullptr;
    std::map<std::string,CUfunction> kernels;
    std::mutex submission;
    uint64_t h,w,r,b,th,tw,oy,ox,reverse,layout,active,extra_elements,tile_elements,temp_bytes=0;
    ~TilePlan() {
        if(context && cuCtxPushCurrent(context)==CUDA_SUCCESS) {
            cuCtxSynchronize(); row.reset(); column.reset();
            if(module) cuModuleUnload(module);
            CUcontext previous; cuCtxPopCurrent(&previous);
        }
    }
};
using TileKey=std::tuple<CUcontext,uint64_t,uint64_t,uint64_t,uint64_t,
                         uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t>;
struct TileCache { std::mutex mutex; std::map<TileKey,std::shared_ptr<TilePlan>> plans; };
static TileCache& tile_cache() { static auto* value=new TileCache; return *value; }
static std::shared_ptr<TilePlan> tile_plan(uint64_t h,uint64_t w,uint64_t b,uint64_t r,
        uint64_t th,uint64_t tw,uint64_t oy,uint64_t ox,uint64_t reverse,uint64_t layout) {
    CUcontext context; cuda_check(cuCtxGetCurrent(&context));
    if(!context) throw std::runtime_error("No current CUDA context");
    auto& cache=tile_cache(); std::lock_guard<std::mutex> guard(cache.mutex);
    TileKey key{context,h,w,b,r,th,tw,oy,ox,reverse,layout};
    auto found=cache.plans.find(key); if(found!=cache.plans.end())return found->second;
    auto p=std::make_shared<TilePlan>();p->context=context;p->h=h;p->w=w;p->b=b;p->r=r;
    p->th=th;p->tw=tw;p->oy=oy;p->ox=ox;p->reverse=reverse;p->layout=layout;
    p->active=layout?2*w:tw;
    p->extra_elements=h*(p->active>w?p->active-w:0);
    p->tile_elements=std::max(r*2*w,b*2*h);
    p->row.reset(static_cast<StreamedFFT*>(streamed_fft_create_ex(2*w,r,0,1)));
    if(!p->row)throw std::runtime_error(streamed_error);
    p->column.reset(static_cast<StreamedFFT*>(streamed_fft_create_ex(2*h,b,0,0)));
    if(!p->column)throw std::runtime_error(streamed_error);
    for(auto* fft:{p->row.get(),p->column.get()}) {
        for(auto* plan:{fft->app.localFFTPlan,fft->app.localFFTPlan_inverse})
            for(uint64_t i=0;i<plan->numAxisUploads[0];++i) {
                auto& sc=plan->axes[0][i].specializationConstants;
                if(sc.tempBufferInput || sc.tempBufferOutput)
                    p->temp_bytes=std::max(p->temp_bytes,fft->bytes);
            }
    }
    auto* fp=p->column->app.localFFTPlan;
    uint64_t factors[3]={1,1,1};
    for(uint64_t i=0;i<fp->numAxisUploads[0];++i)factors[i]=fp->axisSplit[0][i];
    std::ostringstream source;
    for(auto entry:std::vector<std::pair<std::string,uint64_t>>{
        {"H",h},{"W",w},{"NY",2*h},{"NX",2*w},{"X0",w/2},{"Y0",h/2},
        {"B",b},{"R",r},{"AX",2*w},{"BX",1},{"CX",1},
        {"ACTIVE_COLUMNS",p->active},{"EXTRA_COLUMNS",p->active>w?p->active-w:0},
        {"ACTIVE_ORIGIN_X",layout?0:(reverse?(2*w-(ox+tw-1)%(2*w))%(2*w):ox)},
        {"TRANSFER_HEIGHT",th},{"TRANSFER_WIDTH",tw},
        {"TRANSFER_ORIGIN_Y",oy},{"TRANSFER_ORIGIN_X",ox},
        {"TRANSFER_LAYOUT",layout},
        {"AY",factors[0]},{"BY",factors[1]},{"CY",factors[2]}})
        source<<"#define "<<entry.first<<" ("<<entry.second<<"LL)\n";
    source<<"#define DY 1.\n#define DX 1.\n#define BANDLIMIT 0\n"<<streamed_kernel_source;
    nvrtcProgram program;
    auto code=source.str();
    if(nvrtcCreateProgram(&program,code.c_str(),"streamed.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)
        throw std::runtime_error("Cannot create streamed NVRTC program");
    int major,minor;CUdevice device;cuda_check(cuCtxGetDevice(&device));
    cuda_check(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));
    cuda_check(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
    std::string architecture="--gpu-architecture=sm_"+std::to_string(major)+std::to_string(minor);
    const char* options[]={architecture.c_str(),"--std=c++14"};
    if(nvrtcCompileProgram(program,2,options)!=NVRTC_SUCCESS) {
        size_t size;nvrtcGetProgramLogSize(program,&size);std::string log(size,'\0');
        nvrtcGetProgramLog(program,log.data());nvrtcDestroyProgram(&program);throw std::runtime_error(log);
    }
    size_t size;nvrtcGetCUBINSize(program,&size);std::string ptx(size,'\0');
    nvrtcGetCUBIN(program,ptx.data());nvrtcDestroyProgram(&program);
    cuda_check(cuModuleLoadData(&p->module,ptx.c_str()));
    for(const char* name:{"pad_row_tile","scatter_row_tile","gather_row_tile","crop_row_tile",
                         "gather_split_columns","scatter_split_columns","multiply_packed"})
        cuda_check(cuModuleGetFunction(&p->kernels[name],p->module,name));
    cache.plans.emplace(key,p);return p;
}
static ffi::Error StreamedImpl(cudaStream_t stream,
    ffi::Buffer<ffi::C64> input,ffi::Buffer<ffi::U32> transfer,ffi::Buffer<ffi::F32> scale,
    ffi::Result<ffi::Buffer<ffi::C64>> output,ffi::Result<ffi::Buffer<ffi::C64>> work,
    int64_t columns,int64_t rows,int64_t reverse,int64_t origin_y,int64_t origin_x,int64_t layout) {
    try {
        auto dims=input.dimensions();
        if(dims.size()!=2 || dims[0]<1 || dims[1]<1 || columns<1 || rows<1 || reverse<0 || reverse>1 || layout<0 || layout>1)
            return ffi::Error::InvalidArgument("Invalid streamed field or tile attributes");
        uint64_t h=dims[0],w=dims[1];
        if(h>uint64_t(INT64_MAX)/w/32 || columns>65536 || rows>65536)
            return ffi::Error::InvalidArgument("Streamed dimensions exceed checked limits");
        auto hdims=transfer.dimensions();
        if(hdims.size()!=2 || hdims[0]<1 || hdims[1]<1 || hdims[0]>int64_t(2*h) || hdims[1]>int64_t(2*w) ||
                origin_y<0 || origin_y>=int64_t(2*h) || origin_x<0 || origin_x>=int64_t(2*w))
            return ffi::Error::InvalidArgument("Invalid streamed transfer window");
        if(layout && (hdims[0]!=int64_t(h+1) || hdims[1]!=int64_t(w+1) || origin_y || origin_x))
            return ffi::Error::InvalidArgument("Invalid streamed transfer quadrant");
        uint64_t th=hdims[0],tw=hdims[1],active=layout?2*w:tw;
        uint64_t b=std::min(uint64_t(columns),active),r=std::min(uint64_t(rows),h);
        uint64_t extra_elements=h*(active>w?active-w:0);
        uint64_t tile=std::max(r*2*w,b*2*h);
        if(output->dimensions().size()!=2 || output->dimensions()[0]!=dims[0] || output->dimensions()[1]!=dims[1] ||
            work->element_count()!=extra_elements+2*tile || scale.element_count()!=1)
            return ffi::Error::InvalidArgument("Streamed buffer shape mismatch");
        // Exact input/output aliasing is safe: each source row tile is fully
        // staged before its rows are overwritten with the first spectrum half.
        // XLA owns the donation/copy decision; workspace must remain separate.
        if(input.typed_data()==work->typed_data() || output->typed_data()==work->typed_data())
            return ffi::Error::InvalidArgument("Streamed buffers must not alias");
        auto p=tile_plan(h,w,b,r,th,tw,origin_y,origin_x,reverse,layout);std::lock_guard<std::mutex> guard(p->submission);
        if(p->temp_bytes>tile*8)
            return ffi::Error::Internal("FFT temporary exceeds tile allocation");
        void* temporary=static_cast<char*>(static_cast<void*>(work->typed_data()))+(extra_elements+tile)*8;
        void* source=input.typed_data();void* dest=output->typed_data();void* extra=work->typed_data();
        void* scratch=static_cast<char*>(extra)+extra_elements*8;void* payload=transfer.typed_data();void* factor=scale.typed_data();
        auto launch=[&](const char* name,uint64_t blocks,int x,int y,std::initializer_list<void*> args) {
            std::vector<void*> argv(args);
            cuda_check(cuLaunchKernel(p->kernels.at(name),blocks,1,1,x,y,1,0,stream,argv.data(),nullptr));
        };
        auto fft=[&](StreamedFFT* fp,bool inverse) {
            fp->stream=stream;fp->app.configuration.stream[0]=stream;
            VkFFTLaunchParams params{};params.buffer=&scratch;
            if(temporary)params.tempBuffer=&temporary;
            auto result=VkFFTAppend(&fp->app,inverse?1:-1,&params);
            if(result!=VKFFT_SUCCESS)throw std::runtime_error("Streamed VkFFT execution: "+std::to_string(result));
        };
        for(uint64_t offset=0;offset<h;offset+=r) {
            launch("pad_row_tile",(r*2*w+255)/256,256,1,{&source,&scratch,&offset});
            fft(p->row.get(),false);
            launch("scatter_row_tile",(r*active+255)/256,256,1,{&scratch,&dest,&extra,&offset});
        }
        int transpose=reverse;
        for(uint64_t offset=0;offset<active;offset+=b) {
            launch("gather_split_columns",((b+31)/32)*((2*h+31)/32),32,8,{&dest,&extra,&scratch,&offset});
            fft(p->column.get(),false);
            launch("multiply_packed",((b+31)/32)*((2*h+31)/32),32,8,{&scratch,&payload,&offset,&factor,&transpose});
            fft(p->column.get(),true);
            launch("scatter_split_columns",((h+31)/32)*((b+31)/32),32,8,{&scratch,&dest,&extra,&offset});
        }
        for(uint64_t offset=0;offset<h;offset+=r) {
            launch("gather_row_tile",(r*2*w+255)/256,256,1,{&dest,&extra,&scratch,&offset});
            fft(p->row.get(),true);
            launch("crop_row_tile",(r*w+255)/256,256,1,{&scratch,&dest,&offset});
        }
        return ffi::Error::Success();
    }catch(const std::exception& error){return ffi::Error::Internal(error.what());}
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(PyVkFFTStreamed,StreamedImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::C64>>().Arg<ffi::Buffer<ffi::U32>>().Arg<ffi::Buffer<ffi::F32>>()
    .Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>()
    .Attr<int64_t>("columns").Attr<int64_t>("rows").Attr<int64_t>("reverse")
    .Attr<int64_t>("origin_y").Attr<int64_t>("origin_x").Attr<int64_t>("layout"));
extern "C" {
int streamed_ffi_abi_version(){return 3;}
void streamed_ffi_clear_cache(){
    auto& cache=tile_cache();std::map<TileKey,std::shared_ptr<TilePlan>> retired;
    {std::lock_guard<std::mutex> guard(cache.mutex);retired.swap(cache.plans);}
}
const char* streamed_ffi_cache_info(){
    static thread_local std::string info;std::ostringstream out;out<<'[';bool first=true;
    auto& cache=tile_cache();std::lock_guard<std::mutex> guard(cache.mutex);
    for(auto& entry:cache.plans){auto& p=*entry.second;if(!first)out<<',';first=false;
        out<<"{\"shape\":["<<p.h<<','<<p.w<<"],\"tile_columns\":"<<p.b<<",\"tile_rows\":"<<p.r
           <<",\"transfer_shape\":["<<p.th<<','<<p.tw<<"],\"transfer_origin\":["<<p.oy<<','<<p.ox<<']'
           <<",\"transfer_layout\":"<<p.layout<<",\"active_columns\":"<<p.active
           <<",\"reverse\":"<<p.reverse<<",\"extra_spectrum_bytes\":"<<p.extra_elements*8
           <<",\"workspace_bytes\":"<<(p.extra_elements+2*p.tile_elements)*8<<",\"scratch_bytes\":0,\"fft_temporary_bytes\":"<<p.temp_bytes
           <<",\"row_fft\":"<<p.row->info<<",\"column_fft\":"<<p.column->info<<'}';}
    out<<']';info=out.str();return info.c_str();
}
}
