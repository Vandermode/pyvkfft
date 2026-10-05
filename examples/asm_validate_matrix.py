"""Screen equivalent ASM implementations, then validate finalists on one GPU.

Every subprocess has independent plans and CUDA context. The largest grids
alternate single-variant subprocesses to respect the benchmark's VRAM limit.
Existing outputs are reused only within the supplied output directory; use a
fresh directory after changing sources, the native library, or CUDA toolkit.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--shapes', nargs='+', required=True, help='Compact shapes, e.g. 8192x16384')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cufft-library', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=True)
    sources = [root/'examples/asm_benchmark.py', root/'examples/asm_cufft.py', root/'pyvkfft/asm.py',
               Path(os.environ['PYVKFFT_ASM_LIBRARY'])]
    fingerprint = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    manifest = a.output/'source-manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        raise RuntimeError('Sources changed; use a fresh output directory')
    manifest.write_text(json.dumps(fingerprint, indent=2)+'\n')

    def run(name, shape, mode, variants, rounds, seconds):
        dest = a.output/(name+'.json')
        if not dest.exists():
            command = [sys.executable, str(root/'examples/asm_benchmark.py'), '--gpu', str(a.gpu),
                       '--shape', *map(str, shape), '--mode', mode, '--variants', *variants,
                       '--rounds', str(rounds), '--seconds-per-block', str(seconds),
                       '--output', str(dest), '--cufft-library', str(a.cufft_library)]
            with (a.output/(name+'.log')).open('w') as log:
                status = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            if status.returncode:
                raise RuntimeError(f'{name} failed; inspect its log')
        data = json.loads(dest.read_text())
        print(name, data.get('median_ms', data.get('unsupported')), flush=True)
        return data

    results = []
    for encoded in a.shapes:
        shape = tuple(map(int, encoded.split('x')))
        if len(shape) != 2 or min(shape) <= 0:
            p.error('Each shape must contain two positive integers')
        for mode in ('static', 'dynamic_both'):
            prefix = f'{encoded}-{mode}'
            variants = ['cufft', 'cufft_callback', 'cufft_pruned', 'cufft_pruned_crop',
                        'cufft_column', 'cufft_column_crop']
            if mode != 'static': variants += ['cufft_eval']
            native = 'vkfft_pruned' if mode == 'static' else 'vkfft_pruned_fused_h'
            screens = {v: run(f'{prefix}-screen-{v}', shape, mode, [v], 2, 1) for v in variants+[native]}
            valid = {v: screens[v]['median_ms'][v] for v in variants if v in screens[v].get('median_ms', {})}
            if not valid or native not in screens[native].get('median_ms', {}):
                raise RuntimeError(f'No valid finalists for {prefix}')
            best = min(valid, key=valid.get)
            independent = []
            if math.prod(shape)*32 < 8*2**30:
                for trial in range(3):
                    data = run(f'{prefix}-paired-{trial}', shape, mode, [best, native], 4, 5)
                    independent.append({v: statistics.median(statistics.median(b['samples_ms'])
                        for b in data['blocks'] if b['label'] == v) for v in (best, native)})
                protocol = '3 independent processes; 4 alternating 5-second blocks per variant in each'
            else:
                for trial in range(12):
                    row = {}
                    order = [best, native] if trial % 2 == 0 else [native, best]
                    for v in order:
                        data = run(f'{prefix}-independent-{trial}-{v}', shape, mode, [v], 1, 5)
                        row[v] = statistics.median(data['blocks'][0]['samples_ms'])
                    independent.append(row)
                protocol = '12 independent subprocess pairs, one 5-second block each; pair order alternates for VRAM headroom'
            from scipy.stats import t
            ratios = [math.log(row[best]/row[native]) for row in independent]
            mean = statistics.mean(ratios)
            radius = t.ppf(.975, len(ratios)-1)*statistics.stdev(ratios)/math.sqrt(len(ratios))
            result = dict(shape=shape, mode=mode, baseline=best, native=native, protocol=protocol,
                          independent_block_groups=independent, screen_ms=valid,
                          baseline_ms=statistics.median(row[best] for row in independent),
                          native_ms=statistics.median(row[native] for row in independent),
                          speedup=math.exp(mean), ci95=[math.exp(mean-radius), math.exp(mean+radius)],
                          ci_method='Student t interval on independent paired log ratios')
            results.append(result)
            (a.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
            print('VALIDATED', result, flush=True)


if __name__ == '__main__':
    main()
