"""Build an isolated shared static-H/cached-phase symmetry backend.

The base is a frozen build directory from asm_build.py. Output must be fresh.
The new library has a separate capability marker and must be selected through
PYVKFFT_ASM_SYMMETRIC_LIBRARY, never as the ordinary or full-phase backend.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def build(base, output):
    base, output = base.resolve(), output.resolve()
    manifest = json.loads((base/'manifest.json').read_text())
    original = (base/'vkfft_asm.cpp').read_bytes()
    if hashlib.sha256(original).hexdigest() != manifest['source_sha256']:
        raise RuntimeError('Frozen base source does not match its manifest')
    source = original.decode()

    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f'Unexpected native source marker: {old[:80]}')
        source = source.replace(old, new)

    replace_once('bool transfer = building->dynamic && sc->convolutionStep &&',
                 'bool transfer = sc->convolutionStep &&')
    marker = '    if (!input && !transfer && !column) return false;'
    replace_once(marker, marker + r'''
    if (transfer) {
        sc->tempLen = std::snprintf(sc->tempStr, 4096, "%s = asm_load_symmetric(asm_args, %s, %s);\n",
                                   out->name, buffer->name, index->name);
        PfAppendLine(sc);
        return true;
    }
''')
    marker = '    s << R"CUDA(\n'
    replace_once(marker,
                 '    s << "#define ASM_COORD " << ((plan.height<=UINT32_MAX/2 && plan.width<=UINT32_MAX/2) ? "unsigned int" : "unsigned long long") << "\\n";\n'
                 '    s << "#define ASM_SYMMETRIC_DYNAMIC " << plan.dynamic << "\\n";\n' + marker)
    first = source.index('__device__ __forceinline__ float2 asm_transfer(')
    last = source.index('\n)CUDA";', first)
    source = source[:first] + r'''
__device__ __forceinline__ ASM_COORD asm_symmetric_rank(ASM_COORD p,ASM_COORD a,ASM_COORD b) {
    ASM_COORD digit=p/a,row=p%a,half=a/2,tail=(a&1)?b/2:0;
    if(row>half || (row==half && digit>=tail)) {
        row=a-row-(digit!=0); digit=digit?b-digit:0;
    }
    return row<half ? digit*half+row : half*b+digit;
}
__device__ __forceinline__ float2 asm_load_symmetric(ASMArgs a,const float2* table,
                                                    unsigned long long i) {
    ASM_COORD py=i/(2*ASM_W),px=i%(2*ASM_W);
    ASM_COORD y=asm_symmetric_rank(py,a.ay,a.by),x=asm_symmetric_rank(px,a.ax,a.bx);
    const unsigned long long columns=ASM_W+1;
    const unsigned long long stride=columns<1024 ? columns : ((columns+31)/32)*32;
    unsigned long long offset=y*stride+x;
#if ASM_SYMMETRIC_DYNAMIC
    double phase=((const double*)table)[offset];
    if(phase<0.) return make_float2(0.f,0.f);
    double sn,cs;
    sincos(6.283185307179586476925286766559*a.z*phase,&sn,&cs);
    return make_float2((float)cs,(float)sn);
#else
    return table[offset];
#endif
}
''' + source[last:]
    replace_once('if (building->dynamic && sc.convolutionStep) {',
                 'if (sc.convolutionStep) {')
    marker = 'uint32_t asm_abi_version() { return 2; }'
    replace_once(marker, marker + '\nuint32_t asm_symmetric_abi_version() { return 1; }'
                 '\nuint32_t asm_symmetric_layout_version() { return 1; }')
    marker = '    last_error.clear();\n    try {'
    replace_once(marker, marker + '\n        if(dynamic && bandlimit) throw std::invalid_argument("Symmetric cached phase does not support rectangular band limits");')

    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(base/'vkFFT', output/'vkFFT')
    native_source = output/'vkfft_symmetric.cpp'
    native_source.write_text(source)
    library = output/'libvkfft_symmetric.so'
    command = list(manifest['command'])
    inputs = [i for i, arg in enumerate(command) if arg.endswith('/vkfft_asm.cpp')]
    if len(inputs) != 1:
        raise RuntimeError('Expected one native source in the base build command')
    command[inputs[0]] = str(native_source)
    include = '-I' + str(base/'vkFFT')
    if command.count(include) != 1:
        raise RuntimeError('Expected the frozen VkFFT include directory')
    command[command.index(include)] = '-I' + str(output/'vkFFT')
    command[-1] = str(library)
    subprocess.run(command, check=True)
    (output/'manifest.json').write_text(json.dumps({
        'command': command,
        'source_sha256': hashlib.sha256(native_source.read_bytes()).hexdigest(),
        'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_manifest': manifest,
        'native_abi': 2, 'symmetric_abi': 1, 'symmetric_layout_version': 1,
        'contract': 'dynamic=0 reads analytic complex64 H; dynamic=1 reads wavelength-keyed FP64 sqrt(q); aligned positive-frequency digit rows, partial Nyquist row at tail; stride equals logical columns below1024, otherwise rounds to32 elements',
    }, indent=2) + '\n')
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(build(args.base, args.output))


if __name__ == '__main__':
    main()
