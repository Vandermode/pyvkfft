"""Build the typed JAX ASM FFI against the active environment's jaxlib headers.

python examples/asm_ffi_build.py BUILD_DIRECTORY --cuda CUDA_ROOT
export PYVKFFT_ASM_FFI_LIBRARY=BUILD_DIRECTORY/libvkfft_asm_ffi.so

The output is isolated: the installed FFT extension and VkFFT submodule are
unchanged. The same FFI can be used by JAX environments with a compatible ABI.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from asm_build import build as build_core


def build(output, cuda, jax_include=None):
    import jax
    import jaxlib

    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True)
    core = output / 'core'
    build_core(core, cuda)
    jax_include = Path(jax_include or jax.ffi.include_dir()).resolve()
    sources = output / 'source'
    sources.mkdir()
    for name in ('vkfft_asm.cpp', 'vkfft_asm_ffi.cpp'):
        shutil.copy2(root / 'src' / name, sources / name)
    shutil.copy2(Path(__file__), output / 'asm_ffi_build.py')
    core_manifest = json.loads((core / 'manifest.json').read_text())
    command = list(core_manifest['command'])
    index = command.index(str(root / 'src/vkfft_asm.cpp'))
    command[index] = str(sources / 'vkfft_asm_ffi.cpp')
    command.insert(index, '-I' + str(jax_include))
    command.insert(1, '-pthread')
    library = output / 'libvkfft_asm_ffi.so'
    command[-1] = str(library)
    subprocess.run(command, check=True)
    files = list(sources.iterdir()) + [jax_include / 'xla/ffi/api/ffi.h',
                                     jax_include / 'xla/ffi/api/c_api.h']
    manifest = dict(abi=3, command=command, jax_version=jax.__version__,
                    jaxlib_version=jaxlib.__version__, core=core_manifest,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in files},
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
                        help='Copy the FFI library into this checkout\'s pyvkfft package')
    args = parser.parse_args()
    library = build(args.output, args.cuda, args.jax_include)
    if args.install:
        destination = Path(__file__).resolve().parents[1] / 'pyvkfft' / library.name
        # Atomic replacement keeps existing processes' mapped library intact.
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.asm-ffi-',
                                         delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            shutil.copy2(library, staged)
            staged.replace(destination)
        finally:
            staged.unlink(missing_ok=True)
        print(f'Installed {destination}')
