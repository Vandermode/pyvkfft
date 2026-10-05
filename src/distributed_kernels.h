static const char* distributed_kernel_source=R"CUDA(
extern "C" __global__ void row_pad(const float2* src,float2* dst,long long off) {
    long long i=(long long)blockIdx.x*256+threadIdx.x;
    if(i>=B*2*W)return;
    long long x=i%(2*W),y=i/(2*W)+off;
    dst[i]=(y<H && x>=W/2 && x<W/2+W)?src[y*W+x-W/2]:make_float2(0,0);
}
extern "C" __global__ void row_select(const float2* src,float2* dst,long long off) {
    long long i=(long long)blockIdx.x*256+threadIdx.x;
    if(i>=B*K)return;
    long long x=i%K,y=i/K+off;
    long long stride=K/GROUPS,index=(x/stride)*H*stride+y*stride+x%stride;
    if(y<H)dst[index]=x<ACTIVE?src[(i/K)*2*W+(x+ORIGIN)%(2*W)]:make_float2(0,0);
}
extern "C" __global__ void row_expand(const float2* src,float2* dst,long long off) {
    long long i=(long long)blockIdx.x*256+threadIdx.x;
    if(i>=B*2*W)return;
    long long x=i%(2*W),y=i/(2*W)+off,k=(x-ORIGIN+2*W)%(2*W);
    long long stride=K/GROUPS,index=(k/stride)*H*stride+y*stride+k%stride;
    dst[i]=(y<H && k<ACTIVE)?src[index]:make_float2(0,0);
}
extern "C" __global__ void row_crop(const float2* src,float2* dst,long long off) {
    long long i=(long long)blockIdx.x*256+threadIdx.x;
    if(i>=B*W)return;
    long long y=i/W+off;
    if(y<H)dst[y*W+i%W]=src[(i/W)*2*W+i%W+W/2];
}
extern "C" __global__ void col_pad(const float2* src,float2* dst,long long off) {
    __shared__ float2 t[32][33];
    long long gx=blockIdx.x%((B+31)/32),gy=blockIdx.x/((B+31)/32);
    long long x=gx*32+threadIdx.x,y=gy*32+threadIdx.y;
    #pragma unroll
    for(int j=0;j<32;j+=8)t[threadIdx.y+j][threadIdx.x]=
        (x<B && x+off<K && y+j>=H/2 && y+j<H/2+H)?src[(y+j-H/2)*K+x+off]:make_float2(0,0);
    __syncthreads();x=gx*32+threadIdx.y;y=gy*32+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8)if(x+j<B && y<2*H)dst[(x+j)*2*H+y]=t[threadIdx.x][threadIdx.y+j];
}
extern "C" __global__ void col_crop(const float2* src,float2* dst,long long off) {
    __shared__ float2 t[32][33];
    long long gx=blockIdx.x%((H+31)/32),gy=blockIdx.x/((H+31)/32);
    long long y=gx*32+threadIdx.x,x=gy*32+threadIdx.y;
    #pragma unroll
    for(int j=0;j<32;j+=8)t[threadIdx.y+j][threadIdx.x]=(x+j<B && y<H)?src[(x+j)*2*H+y+H/2]:make_float2(0,0);
    __syncthreads();y=gx*32+threadIdx.y;x=gy*32+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8)if(x<B && x+off<K && y+j<H)dst[(y+j)*K+x+off]=t[threadIdx.x][threadIdx.y+j];
}
extern "C" __global__ void multiply(float2* data,const unsigned int* transfer,const float* scale,long long off) {
    __shared__ float2 t[32][33];
    long long gx=blockIdx.x%((B+31)/32),gy=blockIdx.x/((B+31)/32);
    long long x=gx*32+threadIdx.x,py=gy*32+threadIdx.y;
    #pragma unroll
    for(int j=0;j<32;j+=8) {
        long long yy=py+j,y=(yy%AY)*BY*CY+(yy/AY%BY)*CY+yy/(AY*BY);
        long long ty=(y-ORIGIN+2*H)%(2*H);float2 v=make_float2(0,0);
        if(x<B && x+off<K && yy<2*H && ty<TH) {
            unsigned int bits=transfer[ty*K+x+off];
            int re=int(bits&65535u),im=int(bits>>16);re=re>=32768?re-65536:re;im=im>=32768?im-65536:im;
            v=make_float2(re*(*scale),im*(*scale));
        }
        t[threadIdx.y+j][threadIdx.x]=v;
    }
    __syncthreads();x=gx*32+threadIdx.y;py=gy*32+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8)if(x+j<B && py<2*H) {
        long long i=(x+j)*2*H+py;float2 v=data[i],h=t[threadIdx.x][threadIdx.y+j];
        data[i]=make_float2(v.x*h.x-v.y*h.y,v.x*h.y+v.y*h.x);
    }
}
)CUDA";
