"""Isolated direct column-boundary mapping research; never selects public ABI implicitly."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def build(base, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest=json.loads((base/'manifest.json').read_text())
    source=(base/'vkfft_asm.cpp').read_text()
    old='(y-ASM_H/2)*ASM_W + x-ASM_W/2'
    assert source.count(old)==2
    source=source.replace(old,'(x-ASM_W/2)*ASM_H + y-ASM_H/2')
    marker='uint32_t asm_abi_version() { return 2; }'
    assert source.count(marker)==1
    source=source.replace(marker,marker+'\nuint32_t asm_direct_column_abi_version() { return 1; }')
    path=output/'vkfft_direct_column.cpp';path.write_text(source)
    lib=output/'libvkfft_direct_column.so'
    cmd=[str(path) if v.endswith('/src/vkfft_asm.cpp') else v for v in manifest['command']]
    cmd[-1]=str(lib);subprocess.run(cmd,check=True)
    (output/'manifest.json').write_text(json.dumps(dict(command=cmd,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),native_abi=2,direct_column_capability=1,contract='Native dimensions and pitches reversed; compact input/output remain physical row major; static H in reversed native spectral layout'),indent=2)+'\n')


class DirectColumnPlan:
    def __init__(self,shape,pitch,mode='dynamic',bandlimit='none',library=None,stream=None):
        import cupy as cp
        from pyvkfft.asm import _load_native, CUDA_KERNELS
        self.shape=tuple(shape);self.padded_shape=tuple(2*n for n in shape)
        self.stream=stream or cp.cuda.get_current_stream();self.mode=mode
        self.lib=_load_native(str(Path(library).resolve()))
        self.lib.asm_direct_column_abi_version.restype=ctypes.c_uint32
        if self.lib.asm_direct_column_abi_version()!=1:raise ValueError('Wrong direct-column capability')
        with self.stream:
            self.work=cp.empty(self.padded_shape[::-1],cp.complex64)
            self.transfer=cp.empty_like(self.work) if mode=='static' else None
            self.handle=self.lib.asm_create(*shape[::-1],*pitch[::-1],int(mode=='dynamic'),int(bandlimit=='rectangular'),1,self.stream.ptr,128,32,0,0)
            if not self.handle:raise RuntimeError(self.lib.asm_last_error().decode())
            self.info=json.loads(self.lib.asm_info(self.handle))
            ay,by=self.info['axis_split'][1] if self.info['uploads'][1]==2 else (2*shape[1],1)
            ax,bx=self.info['axis_split'][0] if self.info['uploads'][0]==2 else (2*shape[0],1)
            self.layout=tuple(np.int64(v) for v in (*self.padded_shape[::-1],ay,by,ax,bx))
            code=CUDA_KERNELS.replace('dest[i]=source[y*nx+x];','dest[i]=source[x*ny+y];')
            self.prepare=cp.RawModule(code=code).get_function('prepare_transfer')
        self.info.update(axis_order='column_direct',compact_transpose_bytes=0,library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest())
    def prepare_transfer(self,h):
        self.prepare(((self.work.size+255)//256,), (256,), (h,self.transfer,*self.layout,np.int32(0)),stream=self.stream)
    def execute(self,x,out,z=.01,wavelength=532e-9):
        result=self.lib.asm_execute(self.handle,x.data.ptr,out.data.ptr,self.work.data.ptr,self.transfer.data.ptr if self.transfer is not None else 0,z,wavelength)
        if result:raise RuntimeError(result)
    def close(self):
        self.stream.synchronize();self.lib.asm_destroy(self.handle);self.handle=None
        self.work=self.transfer=None


def reused_column_plan(shape,pitch,mode='dynamic',bandlimit='none',library=None,stream=None):
    """Reuse one compact buffer after its only reader has completed.

    Restricted to the frozen native schedule: first forward row reads the
    compact input, later kernels read/write padded work, last inverse row
    writes compact output. Same-stream kernel boundaries establish lifetime.
    This changes no native ABI and does not permit public input/output aliasing.
    """
    import cupy as cp
    import threading
    from pyvkfft.asm import ASMPlan
    from pyvkfft.asm_axis import ColumnFirstASMPlan,TRANSPOSE_SOURCE
    class ReusedColumnPlan(ColumnFirstASMPlan):
        def __init__(self):
            self._closed=True
            self._plan=ASMPlan(tuple(shape)[::-1],tuple(pitch)[::-1],mode=mode,bandlimit=bandlimit,stream=stream,
                tuning_profile={'implementation':'native','prune':True,'transfer':'fused' if mode=='dynamic' else 'materialized'})
            if self._plan.info['library_sha256']!='84d29e76e70bdcae1f7fd0115cd3ae9911e2dd510c93eede1335ad8fb841de81':
                self._plan.close();raise ValueError('Buffer lifetime proof applies only to frozen build07')
            self.shape=tuple(shape);self.padded_shape=tuple(2*n for n in shape);self.pixel_pitch=tuple(pitch)
            self.mode,self.bandlimit=mode,bandlimit;self.stream,self.device=self._plan.stream,self._plan.device
            self._lock=threading.Lock()
            with self.stream:
                self._input=cp.empty(self.shape[::-1],cp.complex64);self._output=self._input
                self._transpose_bytes=self._input.nbytes
                self._module=cp.RawModule(code=TRANSPOSE_SOURCE);self._transpose=self._module.get_function('asm_transpose')
            self._closed=False
        @property
        def work(self):return self._plan._work
        def execute(self,source,dest,*,z=None,wavelength=None):
            self._validate_array(source,self.shape,'source');self._validate_array(dest,self.shape,'dest')
            if source.data.ptr<dest.data.ptr+dest.nbytes and dest.data.ptr<source.data.ptr+source.nbytes:
                raise ValueError('Public input/output overlap remains forbidden')
            if self.mode=='static':
                if not self._plan._prepared:raise ValueError('Prepare static H first')
                z,wavelength=0.,1.
            elif z is None or wavelength is None or not np.isfinite(z) or not np.isfinite(wavelength) or wavelength<=0:
                raise ValueError('Invalid propagation parameters')
            with self._lock,self.stream:
                self._copy_transposed(source,self._input)
                p=self._plan
                result=p._native.asm_execute(p._handle,self._input.data.ptr,self._input.data.ptr,p._work.data.ptr,
                    p._transfer.data.ptr if p._transfer is not None else 0,z,wavelength)
                if result:raise RuntimeError(result)
                self._copy_transposed(self._input,dest)
            return dest
    return ReusedColumnPlan()


def correctness(library,variant):
    import cupy as cp
    import runpy
    references=runpy.run_path(str(Path(__file__).resolve().parents[1]/'pyvkfft/test/test_asm.py'))
    reference_transfer=references['reference_transfer'];reference_propagate=references['reference_propagate']
    rng=np.random.default_rng(924);results=[]
    stream=cp.cuda.Stream(non_blocking=True)
    for shape in ((1,1),(1,15),(8,16),(15,21),(32,8192),(8192,32)):
        host=(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
        for mode in ('static','dynamic'):
            for mask in ('none','rectangular'):
                pitch=(.2e-6,.25e-6)
                factory=reused_column_plan if variant=='reused' else DirectColumnPlan
                p=factory(shape,pitch,mode,mask,library,stream)
                with stream:
                    x=cp.asarray(host);out=cp.empty_like(x)
                    for z,w in ((.01,532e-9),(0.,633e-9),(-.1,450e-9)):
                        h=reference_transfer(p.padded_shape,pitch,z,w,mask=='rectangular')
                        ref=reference_propagate(host,h)
                        if mode=='static':p.prepare_transfer(cp.asarray(h,dtype=cp.complex64))
                        p.work.fill(cp.nan)
                        if mode=='static':p.execute(x,out)
                        else:p.execute(x,out,z=z,wavelength=w)
                        stream.synchronize();actual=cp.asnumpy(out)
                        err=np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-30)
                        assert np.isfinite(actual).all() and err<2e-5,(shape,mode,mask,z,w,err)
                        np.testing.assert_array_equal(cp.asnumpy(x),host)
                        results.append(dict(shape=shape,mode=mode,mask=mask,z=z,wavelength=w,relative_l2=float(err)))
                p.close()
    return results


def benchmark(args):
    import cupy as cp
    from pyvkfft.asm import ASMPlan
    from pyvkfft.asm import CUDA_KERNELS
    shape=tuple(args.shape);pitch=(6.4e-6,6.4e-6);stream=cp.cuda.Stream(non_blocking=True)
    initial_free=cp.cuda.runtime.memGetInfo()[0]
    with stream:
        cp.random.seed(924)
        x=cp.empty(shape,cp.complex64)
        x.real=cp.random.standard_normal(shape,dtype=cp.float32)
        x.imag=cp.random.standard_normal(shape,dtype=cp.float32)
        out=cp.empty_like(x)
        if args.variant=='direct':p=DirectColumnPlan(shape,pitch,args.mode,library=args.library,stream=stream)
        elif args.variant=='reused':p=reused_column_plan(shape,pitch,args.mode,stream=stream)
        else:p=ASMPlan(shape,pitch,mode=args.mode,stream=stream,tuning_profile={'implementation':'native','axis_order':'column','prune':True,'transfer':'fused' if args.mode=='dynamic' else 'materialized'})
        if args.mode=='static':
            h=cp.empty(tuple(2*n for n in shape),cp.complex64)
            gen=cp.RawModule(code=CUDA_KERNELS).get_function('generate_transfer')
            ny,nx=h.shape
            gen(((h.size+255)//256,), (256,), (h,*map(np.int64,(ny,nx,ny,1,nx,1)),*map(np.float64,(*pitch,.01,532e-9)),np.int32(0)),stream=stream)
            p.prepare_transfer(h);stream.synchronize();del h;cp.get_default_memory_pool().free_all_blocks()
        def run(i):
            if args.mode=='dynamic':p.execute(x,out,z=(.01,.1,1.)[i%3],wavelength=(450e-9,532e-9,633e-9)[i%3])
            else:p.execute(x,out)
        for i in range(4):run(i)
        stream.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        planned=initial_free-cp.cuda.runtime.memGetInfo()[0]
        assert planned<.8*initial_free
        blocks=[];i=0
        for b in range(args.blocks):
            start=time.monotonic();samples=[]
            while time.monotonic()-start<args.seconds:
                a,c=cp.cuda.Event(),cp.cuda.Event();a.record(stream)
                for _ in range(4):run(i);i+=1
                c.record(stream);c.synchronize();samples.append(cp.cuda.get_elapsed_time(a,c)/4)
            blocks.append(samples)
        result=dict(shape=shape,mode=args.mode,variant=args.variant,info=p.info,planned_bytes=planned,initial_free_bytes=initial_free,blocks_ms=blocks,mean_ms=float(np.mean([np.mean(b) for b in blocks])))
        p.close()
    return result


def paired(args):
    import cupy as cp
    from pyvkfft.asm import ASMPlan
    from unittest.mock import patch
    shape=tuple(args.shape);pitch=(6.4e-6,6.4e-6);stream=cp.cuda.Stream(non_blocking=True)
    free=cp.cuda.runtime.memGetInfo()[0]
    with stream:
        cp.random.seed(924)
        x=cp.empty(shape,cp.complex64)
        x.real=cp.random.standard_normal(shape,dtype=cp.float32)
        x.imag=cp.random.standard_normal(shape,dtype=cp.float32)
        out=cp.empty_like(x);expected=cp.empty_like(x)
        plans={'wrapper':ASMPlan(shape,pitch,mode=args.mode,stream=stream,
                    tuning_profile={'implementation':'native','axis_order':'column','prune':True,'transfer':'fused' if args.mode=='dynamic' else 'materialized'}),
               'reused':reused_column_plan(shape,pitch,args.mode,stream=stream)}
        if args.mode=='static':
            for p in plans.values():p.prepare_asm_transfer(z=.01,wavelength=532e-9)
        def run(label,i,dest=out):
            if args.mode=='dynamic':plans[label].execute(x,dest,z=(.01,.1,1.)[i%3],wavelength=(450e-9,532e-9,633e-9)[i%3])
            else:plans[label].execute(x,dest)
        errors=[]
        for i in range(3):
            run('wrapper',i,expected)
            with patch('cupy.empty',side_effect=AssertionError('execute allocation')),patch('cupy.empty_like',side_effect=AssertionError('execute allocation')):
                run('reused',i)
            stream.synchronize();num=den=0.
            for y in range(0,shape[0],64):
                a,b=out[y:y+64],expected[y:y+64]
                num+=float(cp.sum(cp.abs(a-b)**2,dtype=cp.float64).get())
                den+=float(cp.sum(cp.abs(b)**2,dtype=cp.float64).get())
            errors.append(float(np.sqrt(num/max(den,1e-30))))
            assert errors[-1]<2e-5
        del expected
        stream.synchronize();cp.get_default_memory_pool().free_all_blocks()
        planned=free-cp.cuda.runtime.memGetInfo()[0];assert planned<.8*free
        timings={key:[] for key in plans};sequence=[];i=0
        for block in range(args.blocks):
            order=['wrapper','reused'] if (block+args.process_index)%2==0 else ['reused','wrapper']
            for label in order:
                samples=[];start=time.monotonic()
                while time.monotonic()-start<args.seconds:
                    a,b=cp.cuda.Event(),cp.cuda.Event();a.record(stream)
                    for _ in range(4):run(label,i);i+=1
                    b.record(stream);b.synchronize();samples.append(cp.cuda.get_elapsed_time(a,b)/4)
                timings[label].append(samples);sequence.append(label)
        result=dict(shape=shape,mode=args.mode,blocks_ms=timings,sequence=sequence,
                    info={key:p.info for key,p in plans.items()},relative_l2=errors,
                    planned_pair_bytes=planned,initial_free_bytes=free,
                    mean_ms={key:float(np.mean([np.mean(b) for b in blocks])) for key,blocks in timings.items()})
        assert result['info']['wrapper']['kernels']==result['info']['reused']['kernels']
        for p in plans.values():p.close()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--build-base',type=Path);p.add_argument('--build-output',type=Path);p.add_argument('--library');p.add_argument('--check',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--shape',type=int,nargs=2,default=(32768,8192));p.add_argument('--mode',choices=['static','dynamic'],default='dynamic');p.add_argument('--variant',choices=['direct','wrapper','reused','pair'],default='direct');p.add_argument('--blocks',type=int,default=2);p.add_argument('--seconds',type=float,default=1);p.add_argument('--process-index',type=int,default=0)
    a=p.parse_args()
    if a.build_base:build(a.build_base,a.build_output);return
    # An artifact-local frozen package prevents concurrent public integration
    # from changing the comparison between fresh worker processes.
    snapshot=a.output.parent/'python-snapshot'
    if snapshot.exists():
        import sys
        sys.path.insert(0,str(snapshot))
    result=correctness(a.library,a.variant) if a.check else paired(a) if a.variant=='pair' else benchmark(a)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('max_error',max(r['relative_l2'] for r in result)) if a.check else print(result['mean_ms'])

if __name__=='__main__':main()
