"""Build isolated, fixed-configuration native ASM research variants.

Uses a source snapshot and patched headers from an existing asm_build output;
never changes the public wrapper, installed library, or dependency submodule.
Resulting libraries retain ABI2 and can be compared with asm_tune.py --profile.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--variants',nargs='+',default=['lut','boost2','boost4','shared64'],
                   choices=['lut','boost2','boost4','shared64','host_invariants','noinline_transfer'])
    args=p.parse_args()
    manifest=json.loads((args.base/'manifest.json').read_text())
    original=(args.base/'vkfft_asm.cpp').read_text()
    variants={'lut':'c.useLUT=1; c.useLUT_4step=1;',
              'boost2':'c.registerBoost=2;',
              'boost4':'c.registerBoost=4;',
              'shared64':'c.sharedMemorySize=65536;', 'host_invariants':'','noinline_transfer':''}
    marker='c.aimThreads=threads; c.coalescedMemory=coalesced;'
    if original.count(marker)!=1:raise RuntimeError('Unexpected native source')
    for name in args.variants:
        configuration=variants[name]
        directory=args.output/name
        directory.mkdir(parents=True,exist_ok=False)
        source=directory/'vkfft_asm.cpp'
        changed=original.replace(marker,marker+'\n        '+configuration)
        if name=='host_invariants':
            replacements={
                'double z, wavelength;':'double z, wavelength, invlambda, phase_scale;',
                'double invlambda = 1. / a.wavelength;':'double invlambda = a.invlambda;',
                'double phase = 6.283185307179586476925286766559 * a.z * sqrt(q);':
                    'double phase = a.phase_scale * sqrt(q);',
                'arguments={input, output, z, wavelength, 1, 1, 1, 1};':
                    'arguments={input, output, z, wavelength, 1./wavelength, 6.283185307179586476925286766559*z, 1, 1, 1, 1};',
            }
            for before,after in replacements.items():
                if before not in changed:raise RuntimeError(f'Missing invariant replacement: {before}')
                changed=changed.replace(before,after)
        if name=='noinline_transfer':
            before='__device__ __forceinline__ float2 asm_transfer('
            if changed.count(before)!=1:raise RuntimeError('Missing transfer helper signature')
            changed=changed.replace(before,'__device__ __noinline__ float2 asm_transfer(')
        source.write_text(changed)
        library=directory/'libvkfft_asm.so'
        command=list(manifest['command'])
        command=[str(source) if token.endswith('/src/vkfft_asm.cpp') else token for token in command]
        command[-1]=str(library)
        subprocess.run(command,check=True)
        metadata=dict(configuration=configuration,command=command,
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
        (directory/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(name,library,flush=True)


if __name__=='__main__':main()
