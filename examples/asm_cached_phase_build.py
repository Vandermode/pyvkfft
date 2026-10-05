"""Build the isolated optional fused cached-phase backend from a frozen source.

Its dynamic kernel expects the native transfer pointer to contain FP64 sqrt(q)
values in plan order, prepared for the exact wavelength. It rejects rectangular
band limits. Use PYVKFFT_ASM_PHASE_LIBRARY with transfer='cached_phase'; the
ordinary PYVKFFT_ASM_LIBRARY backend is unchanged. Phase capability ABI1 is
checked separately from the opaque native ABI2.
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
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.base/'manifest.json').read_text())
    source=(args.base/'vkfft_asm.cpp').read_text()
    first=source.index('__device__ __forceinline__ float2 asm_transfer(')
    last=source.index('\n)CUDA";',first)
    source=source[:first]+r'''
__device__ __forceinline__ float2 asm_transfer(ASMArgs a,const double* phase,unsigned long long i) {
    double v=phase[i];
    if(v<0.) return make_float2(0.f,0.f);
    double sn,cs;
    sincos(6.283185307179586476925286766559*a.z*v,&sn,&cs);
    return make_float2((float)cs,(float)sn);
}
'''+source[last:]
    marker='if (!input && !transfer && !column) return false;'
    replacement=marker+r'''
    if(transfer) {
        sc->tempLen=std::snprintf(sc->tempStr,4096,"%s = asm_transfer(asm_args, (const double*)%s, %s);\n",
                                 out->name,buffer->name,index->name);
        PfAppendLine(sc);
        return true;
    }
'''
    if source.count(marker)!=1:raise RuntimeError('Unexpected source read hook')
    source=source.replace(marker,replacement)
    marker='last_error.clear();\n    try {'
    if source.count(marker)!=1:raise RuntimeError('Unexpected source constructor')
    source=source.replace(marker,marker+'\n        if(bandlimit) throw std::invalid_argument("Cached-phase experiment only supports no band limit");')
    marker='uint32_t asm_abi_version() { return 2; }'
    if source.count(marker)!=1:raise RuntimeError('Unexpected native ABI declaration')
    source=source.replace(marker,marker+'\nuint32_t asm_phase_abi_version() { return 1; }')
    native_source=args.output/'vkfft_cached_phase.cpp'
    native_source.write_text(source)
    library=args.output/'libvkfft_cached_phase.so'
    command=[str(native_source) if v.endswith('/src/vkfft_asm.cpp') else v for v in manifest['command']]
    command[-1]=str(library)
    subprocess.run(command,check=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(command=command,
        source_sha256=hashlib.sha256(native_source.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        phase_abi=1,native_abi=2,
        contract='Dedicated cached-phase backend; dynamic transfer pointer is an FP64 sqrt(q) table for the current wavelength'),indent=2)+'\n')
    print(library)


if __name__=='__main__':main()
