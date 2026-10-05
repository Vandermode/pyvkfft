"""Build the experimental ASM C API without modifying VkFFT or the installation.

Usage: python examples/asm_build.py BUILD_DIRECTORY --cuda CUDA_ROOT
The output includes the exact header patch and a source/library manifest.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def build(output, cuda):
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    headers = output / 'vkFFT'
    if headers.exists():
        raise ValueError('Use a fresh build directory')
    shutil.copytree(root / 'src/VkFFT/vkFFT', headers)
    changes = []

    def edit(relative, old, new):
        path = headers / relative
        before = path.read_text()
        if before.count(old) != 1:
            raise RuntimeError(f'Header hook does not match exactly once: {relative}')
        after = before.replace(old, new)
        path.write_text(after)
        changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                          fromfile='a/vkFFT/' + relative,
                                          tofile='b/vkFFT/' + relative))

    transfers = 'vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryTransfers/vkFFT_Transfers.h'
    for name, args, hook in [
        ('appendGlobalToRegisters', 'PfContainer* out, PfContainer* bufferName, PfContainer* inoutID',
         'if (asm_read_hook(sc, out, bufferName, inoutID)) return;'),
        ('appendRegistersToGlobal', 'PfContainer* bufferName, PfContainer* inoutID, PfContainer* in',
         'if (asm_write_hook(sc, bufferName, inoutID, in)) return;'),
        ('appendGlobalToShared', 'PfContainer* sdataID, PfContainer* bufferName, PfContainer* inoutID',
         'if (asm_read_hook(sc, &sc->temp, bufferName, inoutID)) { appendRegistersToShared(sc, sdataID, &sc->temp); return; }'),
        ('appendSharedToGlobal', 'PfContainer* bufferName, PfContainer* inoutID, PfContainer* sdataID',
         'if (asm_is_output(sc)) { appendSharedToRegisters(sc, &sc->temp, sdataID); appendRegistersToGlobal(sc, bufferName, inoutID, &sc->temp); return; }'),
    ]:
        signature = f'static inline void {name}(VkFFTSpecializationConstantsLayout* sc, {args})\n{{\n\tif (sc->res != VKFFT_SUCCESS) return;'
        edit(transfers, signature, signature + '\n#ifdef VKFFT_ASM_HOOKS\n\t' + hook + '\n#endif')
    compile_path = 'vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_CompileKernel.h'
    old = '\tchar* code0 = axis->specializationConstants.code0;'
    edit(compile_path, old, '#ifdef VKFFT_ASM_HOOKS\n\tif (!asm_compile_hook(app, axis)) return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;\n#endif\n' + old)
    dispatch = 'vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DispatchPlan.h'
    old = 'static inline VkFFTResult VkFFT_DispatchPlan(VkFFTApplication* app, VkFFTAxis* axis, pfUINT* dispatchBlock) {\n\tVkFFTResult resFFT = VKFFT_SUCCESS;'
    edit(dispatch, old, old + '\n#ifdef VKFFT_ASM_HOOKS\n\tasm_dispatch_dimensions(axis, dispatchBlock);\n#endif')
    original = (headers / dispatch).read_text()
    cuda_start = original.index('#elif(VKFFT_BACKEND==1)\n\t\t\t\tconst void* args[20];')
    hook_at = original.index('\t\t\t\t/*if (axis->updatePushConstants) {', cuda_start)
    old = original[cuda_start:hook_at]
    edit(dispatch, old, old + '#ifdef VKFFT_ASM_HOOKS\n\t\t\t\targs[args_id++] = asm_launch_arguments();\n#endif\n')
    (output / 'asm-hooks.patch').write_text(''.join(changes))
    target = cuda / 'targets/x86_64-linux'
    if not target.exists():
        target = cuda
    library_dir = target / ('lib' if (target / 'lib').exists() else 'lib64')
    source = root / 'src/vkfft_asm.cpp'
    # Keep the uncommitted experimental source with its isolated build: a git
    # revision alone cannot reproduce a worktree under active development.
    shutil.copy2(source, output / source.name)
    shutil.copy2(Path(__file__), output / 'asm_build.py')
    library = output / 'libvkfft_asm.so'
    cmd = ['g++', '-std=c++17', '-shared', '-fPIC', '-O2', '-DVKFFT_MAX_FFT_DIMENSIONS=8',
           '-I' + str(target / 'include'), '-I' + str(headers), str(source),
           '-L' + str(library_dir), '-Wl,-rpath,' + str(library_dir),
           '-l:libnvrtc.so.12', '-lcuda', '-lcudart', '-o', str(library)]
    subprocess.run(cmd, check=True)
    manifest = {'command': cmd, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'hooks_sha256': hashlib.sha256((output / 'asm-hooks.patch').read_bytes()).hexdigest(),
                'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'vkfft_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root / 'src/VkFFT', text=True).strip()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(library)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cuda', type=Path, required=True)
    args = parser.parse_args()
    build(args.output, args.cuda)
