"""Build the matrix-kernel ASM FFI (libvkfft_asm_matrix_ffi.so) against the active jaxlib headers.

python examples/asm_matrix_ffi_build.py BUILD_DIRECTORY --cuda CUDA_ROOT [--install]
export PYVKFFT_ASM_MATRIX_FFI_LIBRARY=BUILD_DIRECTORY/libvkfft_asm_matrix_ffi.so   (or use --install)

Reuses the isolated, hook-patched VkFFT header copy of examples/asm_build.py and adds three patches under
VKFFT_ASM_MATRIX_HOOKS: every kernel load of the convolution step reports its codegen-time matrix entry; only the
convolution step repeats its store per kernel (VkFFT otherwise duplicates all stores for numberKernels > 1 and
re-applies the in-place four-step twiddle, which breaks two-upload column axes); and the convolution step's inverse
transforms can be limited to the output rows of a fused reduce.
The installed FFT extension, the VkFFT submodule and the ABI 4 library are unchanged. If CUDA_ROOT lacks
nvrtc.h (pip/conda CUDA), the nvidia-cuda-nvrtc wheel include directory is added automatically.
"""
import argparse
import ctypes
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from asm_build import build as build_core

CONVOLUTION = 'vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_Convolution.h'
KERNEL_LOAD = '\t\t\t\tappendGlobalToRegisters(sc, &sc->temp, &sc->kernelStruct, &sc->tempInt);\n'
# VkFFT repeats the post-processing and store of every kernel numberKernels times, not only in the convolution
# step: the stores are duplicated, and the in-place four-step twiddle of a two-upload column axis is applied K times
# (wrong results for K > 1). Only the convolution step iterates over the kernels; other kernels index by blockIdx.z.
FFT = 'vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel2/vkFFT_FFT.h'
BATCH_LOOP = '\tfor (sc->batchID.data.i = 0; sc->batchID.data.i < sc->numKernels.data.i; sc->batchID.data.i++) {\n'
BATCH_FIX = ('\tfor (sc->batchID.data.i = 0; sc->batchID.data.i < (((sc->convolutionStep) || (sc->useBluesteinFFT && '
             'sc->BluesteinConvolutionStep)) ? sc->numKernels.data.i : 1); sc->batchID.data.i++) {\n')
# A fused reduce (rows >= M of the matrix are zero) inverse-transforms and stores only its M output rows.
COORDINATE_LOOP = '\t\tfor (sc->coordinate.data.i = 0; sc->coordinate.data.i < max_coordinate+1; sc->coordinate.data.i++) {\n'
COORDINATE_FIX = ('\t\tfor (sc->coordinate.data.i = 0; sc->coordinate.data.i < asm_matrix_outputs(sc, max_coordinate+1); '
                  'sc->coordinate.data.i++) {\n')


def nvrtc_include(cuda):
    for root in (cuda / 'targets/x86_64-linux/include', cuda / 'include'):
        if (root / 'nvrtc.h').exists():
            return None
    try:
        import nvidia.cuda_nvrtc
        path = Path(list(nvidia.cuda_nvrtc.__path__)[0]) / 'include'
    except ImportError:
        path = None
    if path is None or not (path / 'nvrtc.h').exists():
        raise RuntimeError('nvrtc.h not found under CUDA_ROOT or the nvidia-cuda-nvrtc wheel')
    return path


def build(output, cuda, jax_include=None):
    import jax
    import jaxlib

    root = Path(__file__).resolve().parents[1]
    output, cuda = output.resolve(), cuda.resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True)
    extra = nvrtc_include(cuda)
    if extra is not None:
        os.environ['CPLUS_INCLUDE_PATH'] = os.pathsep.join(
            [str(extra)] + [p for p in os.environ.get('CPLUS_INCLUDE_PATH', '').split(os.pathsep) if p])
    core = output / 'core'
    build_core(core, cuda)
    patch = []
    for relative, old, new in (
            (CONVOLUTION, KERNEL_LOAD, '#ifdef VKFFT_ASM_MATRIX_HOOKS\n\t\t\t\tasm_matrix_component(j, l, k);\n'
                                       '#endif\n' + KERNEL_LOAD),
            (FFT, BATCH_LOOP, '#ifdef VKFFT_ASM_MATRIX_HOOKS\n' + BATCH_FIX + '#else\n' + BATCH_LOOP + '#endif\n'),
            (FFT, COORDINATE_LOOP, '#ifdef VKFFT_ASM_MATRIX_HOOKS\n' + COORDINATE_FIX + '#else\n' + COORDINATE_LOOP
             + '#endif\n')):
        path = core / 'vkFFT' / relative
        before = path.read_text()
        if before.count(old) != 1:
            raise RuntimeError(f'Matrix hook does not match exactly once: {relative}')
        after = before.replace(old, new)
        path.write_text(after)
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                          fromfile='a/vkFFT/' + relative, tofile='b/vkFFT/' + relative))
    (output / 'matrix-hooks.patch').write_text(''.join(patch))
    jax_include = Path(jax_include or jax.ffi.include_dir()).resolve()
    sources = output / 'source'
    sources.mkdir()
    source = sources / 'vkfft_asm_matrix_ffi.cpp'
    shutil.copy2(root / 'src' / source.name, source)
    shutil.copy2(Path(__file__), output / 'asm_matrix_ffi_build.py')
    core_manifest = json.loads((core / 'manifest.json').read_text())
    command = list(core_manifest['command'])
    index = command.index(str(root / 'src/vkfft_asm.cpp'))
    command[index] = str(source)
    command.insert(index, '-I' + str(jax_include))
    if extra is not None:
        command.insert(index, '-I' + str(extra))
    command.insert(1, '-pthread')
    library = output / 'libvkfft_asm_matrix_ffi.so'
    command[-1] = str(library)
    subprocess.run(command, check=True)
    files = [source, output / 'matrix-hooks.patch', jax_include / 'xla/ffi/api/ffi.h',
             jax_include / 'xla/ffi/api/c_api.h']
    abi = ctypes.CDLL(str(library)).asm_matrix_ffi_abi_version()
    manifest = dict(abi=abi, command=command, jax_version=jax.__version__,
                    jaxlib_version=jaxlib.__version__, core=core_manifest,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(library)
    return library


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--jax-include', type=Path)
    parser.add_argument('--install', action='store_true',
                        help='Copy the library into this checkout\'s pyvkfft package')
    args = parser.parse_args()
    library = build(args.output, args.cuda, args.jax_include)
    if args.install:
        destination = Path(__file__).resolve().parents[1] / 'pyvkfft' / library.name
        # Atomic replacement keeps existing processes' mapped library intact.
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.asm-matrix-ffi-',
                                         delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            shutil.copy2(library, staged)
            staged.replace(destination)
        finally:
            staged.unlink(missing_ok=True)
        print(f'Installed {destination}')
