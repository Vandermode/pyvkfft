"""Prepared ASM comparison with equivalent static and dynamic cuFFT baselines.

Run on physical GPUs 1-3, or on the Slurm-assigned GPU using --slurm.
Times cover compact input through compact output; all plans and allocations
are prepared. Dynamic H generation is always inside the measurement.
"""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import time

from issue205_sustained import Monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument('--gpu', type=int, choices=(1, 2, 3))
    device.add_argument('--slurm', action='store_true')
    parser.add_argument('--shape', type=int, nargs=2, required=True, help='Compact input H W')
    parser.add_argument('--mode', choices=('static', 'dynamic_z', 'dynamic_both'), default='static')
    parser.add_argument('--bandlimit', choices=('none', 'rectangular'), default='none')
    parser.add_argument('--variants', nargs='+', default=['cufft', 'cufft_callback', 'vkfft_explicit', 'vkfft_native'],
                        choices=['cufft', 'cufft_callback', 'cufft_eval', 'vkfft_explicit', 'vkfft_native', 'vkfft_fused_h', 'vkfft_pruned', 'vkfft_pruned_fused_h', 'vkfft_pruned_cached_phase', 'cufft_pruned', 'cufft_pruned_crop', 'cufft_pruned_runtime', 'cufft_column', 'cufft_column_crop', 'cufft_boundary', 'cufft_padding'])
    parser.add_argument('--seconds-per-block', type=float, default=5)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cufft-library', type=Path)
    parser.add_argument('--vkfft-options', type=json.loads, default={})
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--graph', action='store_true', help='Compare static CUDA Graph replay symmetrically on a prepared nondefault stream')
    parser.add_argument('--include-core', action='store_true', help='Also time dense prepared padded-buffer cores separately; dynamic H stays included')
    args = parser.parse_args()
    if args.slurm != bool(os.environ.get('SLURM_JOB_ID')):
        parser.error('Use --slurm inside a Slurm allocation and --gpu outside it')
    if args.gpu is not None:
        uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    if min(args.shape) <= 0 or args.rounds < 1 or args.seconds_per_block <= 0:
        parser.error('Shape, rounds, and block duration must be positive')
    if args.graph and args.mode != 'static':
        parser.error('--graph currently supports static H only; dynamic parameters must not be frozen')
    if args.mode == 'static' and any(v in args.variants for v in ('cufft_eval', 'vkfft_fused_h', 'vkfft_pruned_fused_h', 'vkfft_pruned_cached_phase')):
        parser.error('On-the-fly H evaluation is a dynamic-mode variant')
    if args.bandlimit != 'none' and 'vkfft_pruned_cached_phase' in args.variants:
        parser.error('Cached phase currently supports bandlimit=none')
    if args.cufft_library:
        ctypes.CDLL(str(args.cufft_library.resolve()), mode=ctypes.RTLD_GLOBAL)
    import cupy as cp
    import numpy as np
    from cupy.cuda import cufft
    from cupyx.scipy.fft import get_fft_plan
    from pyvkfft.asm import ASMPlan, CUDA_HELPERS, CUDA_KERNELS

    benchmark_stream=cp.cuda.Stream(non_blocking=True)
    benchmark_stream.use()
    def on_stream(function):
        def invoke(*values, **kwargs):
            with benchmark_stream:
                return function(*values, **kwargs)
        return invoke
    setup_start=time.monotonic()
    setup_times={}
    free_initial, total_memory = cp.cuda.runtime.memGetInfo()
    shape = tuple(args.shape)
    padded = tuple(2*n for n in shape)
    size = math.prod(padded)
    pitch = (6.4e-6, 6.4e-6)
    dynamic = args.mode != 'static'
    separable_variants = {'cufft_pruned', 'cufft_pruned_crop', 'cufft_pruned_runtime',
                         'cufft_column', 'cufft_column_crop'}
    # Conservative arrays + FFT planning scratch preflight, before any large allocation.
    estimated_bytes = int(size*8*(4.0 + (1 if args.include_core else 0) +
                          (2.5 if args.vkfft_options.get('axis_order') == 'column' else 2)*sum(v.startswith('vkfft') for v in args.variants) +
                          sum((2 if dynamic else 3)+(0.5 if v.startswith('cufft_column') else 0)
                              for v in args.variants if v in separable_variants) +
                          (1 if not dynamic and any(v in args.variants for v in ('cufft_callback','cufft_padding','cufft_boundary')) else 0) +
                          sum({'cufft_callback':1,'cufft_padding':2,'cufft_boundary':3}.get(v,0) for v in args.variants)))
    if estimated_bytes > .8*free_initial:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'shape':shape,'unsupported':{v:{'stage':'memory_preflight',
            'estimated_bytes':estimated_bytes,'initial_free_bytes':free_initial,
            'suggestion':'Run one variant per process'} for v in args.variants}},indent=2)+'\n')
        print('Memory preflight rejected; run one variant per process', flush=True)
        return
    rng = cp.random.RandomState(731)
    x = cp.empty(shape, cp.complex64)
    x.real = rng.standard_normal(shape, dtype=cp.float32)
    x.imag = rng.standard_normal(shape, dtype=cp.float32)
    out = cp.empty_like(x)
    reference = cp.empty_like(x)
    work = cp.empty(padded, cp.complex64)
    transfer = cp.empty_like(work)
    parameters = cp.empty(2, cp.float64)
    # Scalar kernel arguments are captured by the CUDA driver at launch.
    extra = r'''
extern "C" __global__ void set_parameters(double* p, double z, double wavelength) {
    p[0]=z; p[1]=wavelength;
}
extern "C" __global__ void multiply(float2* x, const float2* h, long long n) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float2 a=x[i], b=h[i];
    float inv=1.f/n;
    x[i]=make_float2((a.x*b.x-a.y*b.y)*inv, (a.x*b.y+a.y*b.x)*inv);
}
extern "C" __global__ void evaluate_multiply(float2* x, long long ny, long long nx,
    double dy, double dx, double z, double wavelength, int bandlimit) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ny*nx) return;
    float2 a=x[i], b=asm_coefficient(i/nx,i%nx,ny,nx,dy,dx,z,wavelength,bandlimit);
    float inv=1.f/(ny*nx);
    x[i]=make_float2((a.x*b.x-a.y*b.y)*inv, (a.x*b.y+a.y*b.x)*inv);
}
'''
    module = cp.RawModule(code=CUDA_KERNELS + extra)
    kernels = {name: module.get_function(name) for name in
               ('pad_input', 'crop_output', 'generate_transfer', 'multiply', 'evaluate_multiply', 'set_parameters')}
    def launch(name, count, values):
        kernels[name](((count+255)//256,), (256,), values, stream=benchmark_stream)
    dims = tuple(map(np.int64, shape))
    layout = tuple(map(np.int64, (*padded, padded[0], 1, padded[1], 1)))
    def generate(z, wavelength):
        launch('generate_transfer', size, (transfer, *layout, *map(np.float64, pitch),
                                           np.float64(z), np.float64(wavelength), np.int32(args.bandlimit == 'rectangular')))
    benchmark_stream.synchronize()
    setup_times['arrays_and_kernels_seconds']=time.monotonic()-setup_start
    planning_start=time.monotonic()
    cuplan = cufft.PlanNd(padded, padded, 1, size, padded, 1, size, cufft.CUFFT_C2C, 1, 'C', 1, padded[-1])
    callback_plan = None
    callback_transfer = None
    if not dynamic and any(v in args.variants for v in ('cufft_callback','cufft_boundary','cufft_padding')):
        generate(.01,532e-9)
        callback_transfer = transfer/np.float32(size)
    unsupported = {}
    if any(v in args.variants for v in ('cufft_callback','cufft_boundary','cufft_padding')):
        callback = (CUDA_HELPERS if dynamic else '') + '\n__device__ float2 asm_multiply_load_v2(void* data, unsigned long long i, void* info, void* shared) {\n'
        callback += 'float2 a=((float2*)data)[i];\n'
        if dynamic:
            callback += f'float2 b=asm_coefficient(i/{padded[1]}ull,i%{padded[1]}ull,{padded[0]}ll,{padded[1]}ll,6.4e-6,6.4e-6,((double*)info)[0],((double*)info)[1],{int(args.bandlimit == "rectangular")});\n'
        else:
            callback += 'float2 b=((float2*)info)[i];\n'
        callback += (f'float inv=1.f/{size}.f; return make_float2((a.x*b.x-a.y*b.y)*inv,(a.x*b.y+a.y*b.x)*inv);\n}}' if dynamic else 'return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);\n}')
        callback_name='asm_multiply_load_v2'
        if not dynamic:
            from fft_convolution_benchmark import CALLBACK
            callback=CALLBACK
            callback_name='convolution_multiply_load'
        try:
            with cp.fft.config.set_cufft_callbacks(cb_load=callback, cb_load_name=callback_name,
                                                  cb_load_data=(parameters if dynamic else callback_transfer).data, cb_ver='jit'):
                callback_plan = get_fft_plan(work, axes=(0, 1), value_type='C2C')
        except cufft.CuFFTError as exc:
            for label in ('cufft_callback','cufft_boundary','cufft_padding'):
                if label in args.variants:
                    unsupported[label] = {'stage': 'planning', 'error': str(exc),
                         'diagnosis':'LTO callback rejected; inspect CUDA driver/NVRTC/nvJitLink compatibility, not automatically a shape limitation'}
            print(f'cuFFT callback planning rejected this case: {exc}', flush=True)

    boundary_forward = boundary_inverse = None
    if any(v in args.variants and v not in unsupported for v in ('cufft_boundary','cufft_padding')):
        h,w = shape
        pad_callback = f"""__device__ float2 asm_pad_load_v2(void* data, unsigned long long i, void* info, void* shared) {{
            unsigned long long y=i/{2*w}ull, x=i%{2*w}ull;
            return (y>={h//2}ull && y<{h//2+h}ull && x>={w//2}ull && x<{w//2+w}ull)
             ? ((float2*)info)[(y-{h//2}ull)*{w}ull+x-{w//2}ull] : make_float2(0,0);
        }}"""
        crop_callback = f"""__device__ void asm_crop_store_v2(void* data, unsigned long long i, float2 value, void* info, void* shared) {{
            unsigned long long y=i/{2*w}ull, x=i%{2*w}ull;
            if(y>={h//2}ull && y<{h//2+h}ull && x>={w//2}ull && x<{w//2+w}ull)
             ((float2*)info)[(y-{h//2}ull)*{w}ull+x-{w//2}ull]=value;
        }}"""
        boundary_stage='forward_padding_callback'
        try:
            with cp.fft.config.set_cufft_callbacks(cb_load=pad_callback, cb_load_name='asm_pad_load_v2',
                                                   cb_load_data=x.data, cb_ver='jit'):
                boundary_forward=get_fft_plan(work, axes=(0,1), value_type='C2C')
            boundary_stage='inverse_crop_callback'
            with cp.fft.config.set_cufft_callbacks(cb_load=callback, cb_load_name=callback_name,
                    cb_load_data=(parameters if dynamic else callback_transfer).data, cb_store=crop_callback,
                    cb_store_name='asm_crop_store_v2', cb_store_data=out.data, cb_ver='jit'):
                boundary_inverse=get_fft_plan(work, axes=(0,1), value_type='C2C')
        except cufft.CuFFTError as exc:
            if 'cufft_boundary' in args.variants:
                unsupported['cufft_boundary']={'stage':boundary_stage,'error':str(exc)}
            if boundary_forward is None and 'cufft_padding' in args.variants:
                unsupported['cufft_padding']={'stage':boundary_stage,'error':str(exc)}

    setup_times['cufft_planning_and_static_callback_preparation_seconds']=time.monotonic()-planning_start

    @on_stream
    def cu_execute(z, wavelength, variant='cufft', destination=out):
        if variant not in ('cufft_boundary','cufft_padding'):
            launch('pad_input', size, (x, work, *dims))
        if dynamic and variant == 'cufft':
            generate(z, wavelength)
        elif dynamic and variant in ('cufft_callback','cufft_boundary','cufft_padding'):
            kernels['set_parameters']((1,), (1,), (parameters, np.float64(z), np.float64(wavelength)))
        (boundary_forward if variant in ('cufft_boundary','cufft_padding') else cuplan).fft(work, work, cufft.CUFFT_FORWARD)
        if variant == 'cufft_eval':
            launch('evaluate_multiply', size, (work, *map(np.int64, padded), *map(np.float64, pitch),
                                               np.float64(z), np.float64(wavelength), np.int32(args.bandlimit == 'rectangular')))
        elif variant == 'cufft':
            launch('multiply', size, (work, transfer, np.int64(size)))
        (boundary_inverse if variant == 'cufft_boundary' else callback_plan if variant in ('cufft_callback','cufft_padding') else cuplan).fft(work, work, cufft.CUFFT_INVERSE)
        if variant != 'cufft_boundary':
            launch('crop_output', x.size, (work, destination, *dims))

    transfer_start=time.monotonic()
    generate(.01, 532e-9)
    benchmark_stream.synchronize()
    setup_times['shared_static_transfer_generation_seconds']=time.monotonic()-transfer_start
    plans, executors = {}, {}
    baseline_info = {}
    pruned = None
    for variant in args.variants:
        if variant in unsupported:
            continue
        variant_setup_start=time.monotonic()
        if variant in separable_variants:
            from asm_cufft import CuFFTASMPlan
            try:
                with benchmark_stream:
                    pruned = CuFFTASMPlan(shape, pitch, transfer, dynamic, args.bandlimit == 'rectangular',
                        axis_order='column' if variant.startswith('cufft_column') else 'row',
                        crop_callback=variant.endswith('_crop'), specialize=variant != 'cufft_pruned_runtime')
            except cufft.CuFFTError as exc:
                unsupported[variant] = {'stage': 'separable_callback_planning', 'error': str(exc)}
                continue
            baseline_info[variant] = pruned.info
            executors[variant] = lambda z,w,p=pruned: p.execute(x,out,z,w)
        elif variant.startswith('cufft'):
            executors[variant] = lambda z, w, v=variant: cu_execute(z, w, v)
        else:
            profile = dict(args.vkfft_options, implementation='explicit' if variant == 'vkfft_explicit' else 'native',
                           transfer='cached_phase' if variant.endswith('cached_phase') else 'fused' if variant.endswith('fused_h') else 'materialized',
                           prune=variant.startswith('vkfft_pruned'))
            plan = ASMPlan(shape, pitch, mode='dynamic' if dynamic else 'static',
                           bandlimit=args.bandlimit, tuning_profile=profile, stream=benchmark_stream)
            if not dynamic:
                plan.prepare_transfer(transfer)
            plans[variant] = plan
            executors[variant] = (lambda z, w, p=plan: p.execute(x, out, z=z, wavelength=w)) if dynamic else (lambda z, w, p=plan: p.execute(x, out))
        benchmark_stream.synchronize()
        setup_times[variant+'_plan_and_prepare_seconds']=time.monotonic()-variant_setup_start

    executors={label:on_stream(execute) for label,execute in executors.items()}
    cp.cuda.runtime.deviceSynchronize()

    def params(i):
        return ((.01, .1, 1.)[i % 3] if dynamic else .01,
                (450e-9, 532e-9, 633e-9)[i % 3] if args.mode == 'dynamic_both' else 532e-9)
    @on_stream
    def relative_error(a, b):
        num, den = 0., 0.
        for offset in range(0, a.size, 2**20):
            av, bv = a.ravel()[offset:offset+2**20], b.ravel()[offset:offset+2**20]
            num += float(cp.sum(cp.abs(av-bv)**2, dtype=cp.float64))
            den += float(cp.sum(cp.abs(bv)**2, dtype=cp.float64))
        return math.sqrt(num/den) if den else math.sqrt(num)
    errors = {}
    for i in range(3 if dynamic else 1):
        z, wavelength = params(i)
        cu_execute(z, wavelength, destination=reference)
        for label, execute in executors.items():
            execute(z, wavelength)
            error = relative_error(out, reference)
            if not math.isfinite(error) or error > 2e-5:
                raise AssertionError((label, i, error))
            errors[label] = max(error, errors.get(label, 0))
    # A separate immutable padded input keeps repeated dense core calls valid.
    # These measurements are not compared with compact-load/store fused paths.
    core_input = None
    if args.include_core:
        core_input = cp.empty_like(work)
        launch('pad_input', size, (x, core_input, *dims))
        @on_stream
        def cu_core(z, wavelength, variant):
            if dynamic and variant == 'cufft':
                generate(z,wavelength)
            elif dynamic and variant == 'cufft_callback':
                kernels['set_parameters']((1,), (1,), (parameters,np.float64(z),np.float64(wavelength)))
            cuplan.fft(core_input,work,cufft.CUFFT_FORWARD)
            if variant == 'cufft_eval':
                launch('evaluate_multiply',size,(work,*map(np.int64,padded),*map(np.float64,pitch),
                       np.float64(z),np.float64(wavelength),np.int32(args.bandlimit == 'rectangular')))
            elif variant == 'cufft':
                launch('multiply',size,(work,transfer,np.int64(size)))
            (callback_plan if variant == 'cufft_callback' else cuplan).fft(work,work,cufft.CUFFT_INVERSE)
        for variant in list(executors):
            if variant in ('cufft','cufft_eval','cufft_callback'):
                label=variant+'_padded_core'
                executors[label]=lambda z,w,v=variant: cu_core(z,w,v)
                for i in range(3 if dynamic else 1):
                    z,w=params(i)
                    cu_execute(z,w,destination=reference)
                    executors[label](z,w)
                    launch('crop_output',x.size,(work,out,*dims))
                    err=relative_error(out,reference)
                    if not math.isfinite(err) or err>2e-5: raise AssertionError((label,i,err))
                    errors[label]=max(err,errors.get(label,0))
    benchmark_stream.synchronize()
    if free_initial-cp.cuda.runtime.memGetInfo()[0] > .8*free_initial:
        raise MemoryError('Actual planned allocations exceed 80% of initially available VRAM; use one variant per process')
    bus = cp.cuda.runtime.deviceGetPCIBusId(0)
    if isinstance(bus, bytes): bus = bus.decode()
    uuid = subprocess.check_output(['nvidia-smi', '-i', bus, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    result = {'shape': shape, 'padded_shape': padded, 'mode': args.mode, 'bandlimit': args.bandlimit,
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), 'gpu_uuid': uuid,
              'cufft_version': cufft.getVersion(), 'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
              'cupy_version': cp.__version__, 'correctness_relative_l2': errors,
              'plans': {label: p.info for label, p in plans.items()},
              'protocol': 'compact input to compact output; dynamic H included; input unchanged between iterations',
              'blocks': [], 'unsupported': unsupported, 'baseline_plans':baseline_info, 'setup_wall_seconds':setup_times,
              'environment':{'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH'),'CUPY_CACHE_DIR':os.environ.get('CUPY_CACHE_DIR')},
              'memory': {'initial_free_bytes':free_initial,'total_bytes':total_memory,
                         'estimated_preflight_bytes':estimated_bytes,
                         'pool_used_bytes':cp.get_default_memory_pool().used_bytes(),
                         'pool_reserved_bytes':cp.get_default_memory_pool().total_bytes(),
                         'device_free_after_planning_bytes':cp.cuda.runtime.memGetInfo()[0]},
              'core_protocol':'Unavailable for compact-load/store fused and pruned paths; dense prepared-buffer execution separately labeled when requested'}
    if not executors:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(f'No runnable variants: {unsupported}',flush=True)
        return
    print(json.dumps({k: result[k] for k in ('shape', 'mode', 'device', 'correctness_relative_l2')}), flush=True)
    for execute in executors.values():
        for i in range(10): execute(*params(i))
    benchmark_stream.synchronize()
    if args.graph:
        graphs={}
        for label, execute in list(executors.items()):
            benchmark_stream.begin_capture()
            execute(*params(0))
            graphs[label]=benchmark_stream.end_capture()
            executors[label+'_graph']=lambda z,w,g=graphs[label]:g.launch(benchmark_stream)
        benchmark_stream.synchronize()
        result['graph_protocol']='Static H only; one complete operator call per replay, same input and output, same stream'
    if args.profile:
        cp.cuda.profiler.start()
        for label, execute in executors.items():
            cp.cuda.nvtx.RangePush(label)
            execute(*params(1))
            cp.cuda.nvtx.RangePop()
        benchmark_stream.synchronize()
        cp.cuda.profiler.stop()
    else:
        monitor = Monitor(uuid)
        monitor.thread.start()
        try:
            for round_index in range(args.rounds):
                order = list(executors)
                if round_index % 2: order.reverse()
                for label in order:
                    execute = executors[label]
                    start, end = cp.cuda.Event(), cp.cuda.Event()
                    start.record(benchmark_stream)
                    for i in range(10): execute(*params(i))
                    end.record(benchmark_stream); end.synchronize()
                    estimate = cp.cuda.get_elapsed_time(start, end)/10
                    count = max(20, math.ceil(args.seconds_per_block*1000/estimate))
                    events = [cp.cuda.Event() for _ in range(count+1)]
                    t0 = time.monotonic()
                    events[0].record(benchmark_stream)
                    for i in range(count):
                        execute(*params(i))
                        events[i+1].record(benchmark_stream)
                    events[-1].synchronize()
                    t1 = time.monotonic()
                    samples = [cp.cuda.get_elapsed_time(events[i], events[i+1]) for i in range(count)]
                    result['blocks'].append({'label': label, 'round': round_index, 'samples_ms': samples,
                                             'start_monotonic': t0, 'end_monotonic': t1})
                    print(f'{label}: {np.median(samples):.3f} ms', flush=True)
        finally:
            monitor.close()
        result['telemetry'], result['telemetry_errors'] = monitor.rows, monitor.errors
        result['median_ms'] = {label: float(np.median([t for b in result['blocks'] if b['label'] == label for t in b['samples_ms']])) for label in executors}
        print(json.dumps(result['median_ms']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for plan in plans.values(): plan.close()


if __name__ == '__main__':
    main()
