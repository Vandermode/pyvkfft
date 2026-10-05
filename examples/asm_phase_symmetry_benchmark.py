"""Final three-way public ASM phase benchmark; all preparation included in execute."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from issue205_sustained import Monitor


def make_plans(shape,pitch,layout,stream,include_ordinary=True):
    from pyvkfft.asm import ASMPlan
    strategies={'full':'cached_phase',layout:'cached_phase_symmetric','ordinary_fused':'fused'}
    return {key:ASMPlan(shape,pitch,mode='dynamic',stream=stream,tuning_profile={
        'implementation':'native','transfer':strategy,'prune':True,'groupedBatch':[0,32]})
        for key,strategy in strategies.items()}


def benchmark(args):
    import cupy as cp
    import numpy as np
    from unittest.mock import patch
    shape=tuple(args.shape);pitch=(6.4e-6,6.4e-6);stream=cp.cuda.Stream(non_blocking=True);free=cp.cuda.runtime.memGetInfo()[0]
    with stream:
        rng=cp.random.RandomState(951);x=cp.empty(shape,cp.complex64)
        x.real=rng.standard_normal(shape,dtype=cp.float32);x.imag=rng.standard_normal(shape,dtype=cp.float32)
        plans=make_plans(shape,pitch,args.layout,stream,args.include_ordinary)
        outputs={key:cp.empty_like(x) for key in plans}
        def run(key,i):
            z=(.01,.1,1.)[i%3];w=(450e-9,532e-9,633e-9)[i%3] if args.mode=='dynamic_both' else 532e-9
            plans[key].execute(x,outputs[key],z=z,wavelength=w)
        errors={key:[] for key in plans if key!='full'}
        for i in range(3):
            run('full',i)
            for key in errors:
                with patch('cupy.empty',side_effect=AssertionError('execute allocation')),patch('cupy.empty_like',side_effect=AssertionError('execute allocation')):run(key,i)
            stream.synchronize()
            for key in errors:
                num=den=0.
                for y in range(0,shape[0],32):
                    a,b=outputs[key][y:y+32],outputs['full'][y:y+32]
                    num+=float(cp.sum(cp.abs(a-b)**2,dtype=cp.float64).get());den+=float(cp.sum(cp.abs(b)**2,dtype=cp.float64).get())
                errors[key].append(float(np.sqrt(num/max(den,1e-30))));assert errors[key][-1]<2e-5
        for i in range(4):
            for key in plans:run(key,i)
        stream.synchronize();cp.get_default_memory_pool().free_all_blocks()
        planned=free-cp.cuda.runtime.memGetInfo()[0];assert planned<.8*free
        before={key:p.info.get('phase_preparations',0) for key,p in plans.items()}
        blocks={key:[] for key in plans};counts={key:0 for key in plans};sequence=[];windows=[]
        monitor=Monitor(os.environ['CUDA_VISIBLE_DEVICES']);monitor.thread.start()
        for block in range(args.blocks):
            keys=list(plans);shift=(block+args.process_index)%len(keys)
            order=keys[shift:]+keys[:shift]
            if (block+args.process_index)%2:order.reverse()
            for key in order:
                samples=[];wall_start=time.time();start=time.monotonic()
                while time.monotonic()-start<args.seconds:
                    a,b=cp.cuda.Event(),cp.cuda.Event();a.record(stream)
                    for _ in range(4):run(key,counts[key]);counts[key]+=1
                    b.record(stream);b.synchronize();samples.append(cp.cuda.get_elapsed_time(a,b)/4)
                end=time.monotonic();wall_end=time.time()
                blocks[key].append(samples);sequence.append(key)
                windows.append({'variant':key,'block':block,'start_monotonic':start,
                                'end_monotonic':end,'start_unix':wall_start,'end_unix':wall_end})
        monitor.close()
        for window in windows:
            rows=[r for r in monitor.rows if window['start_monotonic']+.5<=r['monotonic']<=window['end_monotonic']-.5]
            assert rows, 'No steady telemetry samples'
            values=[r['gpu_percent'] for r in rows]
            window['telemetry']={'trim_seconds_each_edge':.5,'sample_count':len(rows),
                'gpu_percent_mean':float(np.mean(values)),'gpu_percent_min':min(values),
                'fraction_samples_ge99':float(np.mean(np.asarray(values)>=99))}
        result={'shape':shape,'mode':args.mode,'layout':args.layout,'mean_ms':{key:float(np.mean([np.mean(b) for b in value])) for key,value in blocks.items()},'blocks_ms':blocks,'sequence':sequence,'correctness_relative_l2':errors,'timed_calls':counts,'phase_preparations_before_timing':before,'info':{key:p.info for key,p in plans.items()},'initial_free_bytes':free,'planned_pair_bytes':planned}
        result.update(block_windows=windows,telemetry=monitor.rows,telemetry_errors=monitor.errors,
                      telemetry_gate='mean GPU percent >=99 in each block after predeclared 0.5 s trim at both edges',
                      telemetry_pass=not monitor.errors and all(w['telemetry']['gpu_percent_mean']>=99 for w in windows))
        for p in plans.values():p.close()
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shape',type=int,nargs=2,default=(4096,16384))
    p.add_argument('--mode',choices=['dynamic_z','dynamic_both'],required=True)
    p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--seconds',type=float,default=5)
    p.add_argument('--process-index',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.layout='compressed';args.include_ordinary=True
    import pyvkfft
    repo=Path(pyvkfft.__file__).resolve().parent
    sources={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in
             ['asm.py','asm_axis.py','asm_phase.py','asm_symmetric.py']}
    result=benchmark(args)
    result['source_sha256']=sources
    result['source_unchanged']=all(hashlib.sha256((repo/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
    result['libraries']={key:{'path':os.environ[key],'sha256':hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()} for key in
        ['PYVKFFT_ASM_LIBRARY','PYVKFFT_ASM_PHASE_LIBRARY','PYVKFFT_ASM_SYMMETRIC_LIBRARY']}
    result['gpu_uuid']=os.environ['CUDA_VISIBLE_DEVICES']
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['mean_ms'],flush=True)

if __name__=='__main__':main()
