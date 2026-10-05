"""Build the capacity-oriented packed-transfer JAX FFI in an isolated directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from asm_streamed_build import build as build_core


def build(output, cuda):
    import jax
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from pyvkfft.asm import CUDA_HELPERS
    from pyvkfft.asm_streamed import STREAMED_KERNELS
    output = output.resolve()
    if output.exists():
        raise ValueError('Use a fresh build directory')
    build_core(output / 'core', cuda)
    source = output / 'source'
    source.mkdir()
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    for name in ('vkfft_streamed.cpp', 'vkfft_streamed_ffi.cpp'):
        shutil.copy2(root / 'src' / name, source / name)
    kernels = (CUDA_HELPERS + STREAMED_KERNELS).replace(
        'long long offset, float scale, int reverse)',
        'long long offset, const float* scale_pointer, int reverse)').replace(
        'float2 h=make_float2(0,0);', 'float scale=*scale_pointer; float2 h=make_float2(0,0);')
    (source / 'streamed_kernels.h').write_text('static const char* streamed_kernel_source=R"CUDA(\n' + kernels + ')CUDA";\n')
    manifest = json.loads((output / 'core/manifest.json').read_text())
    command = manifest['command'].copy()
    command[command.index(str(output / 'core/vkfft_streamed.cpp'))] = str(source / 'vkfft_streamed_ffi.cpp')
    command.insert(1, '-pthread')
    command.insert(2, '-I'+jax.ffi.include_dir())
    library = output / 'libvkfft_streamed_ffi.so'
    command[-1] = str(library)
    subprocess.run(command, check=True)
    manifest = dict(command=command, abi=3, core=manifest, jax_version=jax.__version__,
                    generator_sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                       (Path(__file__), root/'pyvkfft/asm.py', root/'pyvkfft/asm_streamed.py')},
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return library


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    library = build(args.output, args.cuda)
    if args.install:
        dest = Path(__file__).resolve().parents[1] / 'pyvkfft' / library.name
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as file:
            temporary = Path(file.name)
        try:
            shutil.copy2(library, temporary)
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)
    print(library)
