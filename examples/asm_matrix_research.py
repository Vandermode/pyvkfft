"""Run independent alternating ASM finalist blocks within the VRAM budget.

Consumes the per-variant screen-v2 JSON files in OUTPUT. Each five-second
block uses a fresh process, avoiding simultaneous baseline/native workspaces.
Native and the fastest correct screened cuFFT variant alternate across blocks.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


def summarize(output,blocks):
    from scipy.stats import t
    summary={}
    for case in json.loads((output/'selected-finalists.json').read_text()):
        prefix=f"{case['shape'][0]}x{case['shape'][1]}-{case['mode']}"
        timings,errors,fractions={},{},[]
        for label in (case['baseline'],case['native']):
            paths=sorted(output.glob(f'{prefix}-{label}-final-*.json'))
            if len(paths)!=blocks:break
            values,relative_errors=[],[]
            for path in paths:
                data=json.loads(path.read_text())
                values.append(statistics.mean(data['blocks'][0]['samples_ms']))
                relative_errors.append(data['correctness_relative_l2'][label])
                memory=data['memory']
                fractions.append((memory['initial_free_bytes']-memory['device_free_after_planning_bytes'])/memory['initial_free_bytes'])
            timings[label]=values
            errors[label]=max(relative_errors)
        if len(timings)!=2:continue
        ratios=[a/b for a,b in zip(timings[case['baseline']],timings[case['native']])]
        mean=statistics.mean(ratios)
        margin=float(t.ppf(.975,blocks-1))*statistics.stdev(ratios)/math.sqrt(blocks)
        summary[prefix]=dict(case,mean_ms={key:statistics.mean(value) for key,value in timings.items()},
            block_process_means_ms=timings,speedup=mean,speedup_95ci=[mean-margin,mean+margin],
            max_relative_l2=errors,maximum_fraction_initial_free_vram=max(fractions),
            protocol=f'{blocks} independent fresh-process blocks per variant, 5s each, alternating ABBA; paired-block t interval')
    (output/'final-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:{name:value[name] for name in ('mean_ms','speedup','speedup_95ci','max_relative_l2')}
                      for key,value in summary.items()},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--gpu',type=int,choices=[1,2,3],default=2)
    p.add_argument('--blocks',type=int,default=12)
    p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--run-screens',action='store_true',help='Screen frozen specialized cuFFT finalists first')
    p.add_argument('--column-tall',action='store_true',help='Use the promoted column profile for the tall native case')
    p.add_argument('--resume',action='store_true',help='Resume existing frozen snapshots and completed blocks')
    args=p.parse_args()
    if args.blocks<2:p.error('At least two independent blocks are required')
    if args.summarize_only:
        summarize(args.output,args.blocks)
        return
    source=Path(__file__).resolve().parent
    args.output.mkdir(parents=True,exist_ok=True)
    snapshot=args.output/'benchmark-snapshot'
    snapshot.mkdir(exist_ok=args.resume)
    manifest={}
    for name in ('asm_benchmark.py','asm_cufft.py','issue205_sustained.py','fft_convolution_benchmark.py'):
        if not args.resume:
            shutil.copy2(source/name,snapshot/name)
        manifest[name]=hashlib.sha256((snapshot/name).read_bytes()).hexdigest()
    if args.resume and manifest!=json.loads((snapshot/'manifest.json').read_text()):
        raise RuntimeError('Frozen benchmark snapshot has changed')
    (snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.run_screens:
        for shape in ((8192,32768),(32768,8192)):
            for mode in ('static','dynamic_both'):
                for label in ('cufft_pruned','cufft_pruned_crop','cufft_column','cufft_column_crop'):
                    result=args.output/f'{shape[0]}x{shape[1]}-{mode}-{label}-screen-v2.json'
                    if args.resume and result.exists():continue
                    command=[sys.executable,str(snapshot/'asm_benchmark.py'),'--gpu',str(args.gpu),
                             '--shape',*map(str,shape),'--mode',mode,'--variants',label,
                             '--rounds','2','--seconds-per-block','1','--output',str(result)]
                    with result.with_suffix('.log').open('w') as log:
                        process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                    if process.returncode:
                        message=result.with_suffix('.log').read_text()
                        if 'Actual planned allocations exceed 80%' not in message:
                            raise RuntimeError(f'Screen failed: {result}; inspect its log')
                        result.write_text(json.dumps({'unsupported':{label:'Benchmark harness peak planned allocations exceed 80% of initially available VRAM; variant not measured, not a cuFFT limitation'}})+'\n')
                    print('screen',shape,mode,label,json.loads(result.read_text()).get('median_ms'),flush=True)
    selected=[]
    for shape in ((8192,32768),(32768,8192)):
        for mode in ('static','dynamic_both'):
            prefix=f'{shape[0]}x{shape[1]}-{mode}'
            screens=[]
            for path in args.output.glob(prefix+'-cufft*-screen-v2.json'):
                item=json.loads(path.read_text())
                screens.extend((value,label) for label,value in item.get('median_ms',{}).items()
                               if label in item.get('correctness_relative_l2',{}))
            if not screens:raise RuntimeError(f'No validated baseline for {prefix}')
            baseline=min(screens)[1]
            native='vkfft_pruned' if mode=='static' else 'vkfft_pruned_fused_h'
            selected.append(dict(shape=shape,mode=mode,baseline=baseline,native=native,
                                 native_options={'axis_order':'column'} if args.column_tall and shape[0]>shape[1] else {},
                                 screen_ms=dict((label,value) for value,label in screens)))
    (args.output/'selected-finalists.json').write_text(json.dumps(selected,indent=2)+'\n')
    for case in selected:
        shape,mode=case['shape'],case['mode']
        prefix=f'{shape[0]}x{shape[1]}-{mode}'
        for block in range(args.blocks):
            labels=[case['baseline'],case['native']]
            if block%2:labels.reverse()
            for label in labels:
                result=args.output/f'{prefix}-{label}-final-{block:02d}.json'
                if args.resume and result.exists():continue
                command=[sys.executable,str(snapshot/'asm_benchmark.py'),'--gpu',str(args.gpu),
                         '--shape',*map(str,shape),'--mode',mode,'--variants',label,
                         '--rounds','1','--seconds-per-block','5','--output',str(result)]
                if label==case['native'] and case['native_options']:
                    command.extend(['--vkfft-options',json.dumps(case['native_options'])])
                with result.with_suffix('.log').open('w') as log:
                    process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if process.returncode:raise RuntimeError(f'Finalist failed: {result}; inspect its log')
                data=json.loads(result.read_text())
                if label not in data.get('median_ms',{}):raise RuntimeError(f'Finalist unsupported: {result}')
                samples=data['blocks'][0]['samples_ms']
                print(prefix,block,label,statistics.mean(samples),flush=True)
    print('All alternating independent blocks complete',flush=True)
    summarize(args.output,args.blocks)


if __name__=='__main__':main()
