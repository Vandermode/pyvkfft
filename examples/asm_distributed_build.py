"""Build isolated JAX local stages for distributed VkFFT / windowed ASM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from asm_streamed_build import build as build_core


def build(output, cuda):
    import jax
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists():
        raise ValueError('Use a fresh build directory')
    build_core(output / 'core', cuda)
    source = output / 'source'
    source.mkdir()
    for name in ('vkfft_streamed.cpp', 'vkfft_distributed_ffi.cpp', 'distributed_kernels.h'):
        shutil.copy2(root / 'src' / name, source / name)
    manifest = json.loads((output / 'core/manifest.json').read_text())
    command = manifest['command'].copy()
    command[command.index(str(output / 'core/vkfft_streamed.cpp'))] = str(source / 'vkfft_distributed_ffi.cpp')
    command[1:1] = ['-pthread', '-I' + jax.ffi.include_dir()]
    library = output / 'libvkfft_distributed_ffi.so'
    command[-1] = str(library)
    subprocess.run(command, check=True)
    manifest = dict(command=command, abi=2, core=manifest, jax_version=jax.__version__,
                    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
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
