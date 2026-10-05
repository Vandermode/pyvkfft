"""Build the isolated ordinary FFT API used by StreamedASMPlan.

Usage: python examples/asm_streamed_build.py BUILD_DIRECTORY --cuda CUDA_ROOT
VkFFT headers are copied and patched; the submodule and installation are untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def build(output, cuda):
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty build directory')
    output.mkdir(parents=True, exist_ok=True)
    headers = output / 'vkFFT'
    shutil.copytree(root / 'src/VkFFT/vkFFT', headers)
    patch = output / 'inverse-upload-order.patch'
    shutil.copy2(root / 'examples/issue205_inverse_upload_order.patch', patch)
    subprocess.run(['patch', '--batch', '-p1', '-d', str(output), '-i', str(patch)], check=True)
    source = output / 'vkfft_streamed.cpp'
    shutil.copy2(root / 'src/vkfft_streamed.cpp', source)
    shutil.copy2(Path(__file__), output / 'asm_streamed_build.py')
    target = cuda / 'targets/x86_64-linux'
    if not target.exists():
        target = cuda
    library_dir = target / ('lib' if (target / 'lib').exists() else 'lib64')
    library = output / 'libvkfft_streamed.so'
    command = ['g++', '-std=c++17', '-shared', '-fPIC', '-O2', '-DVKFFT_MAX_FFT_DIMENSIONS=8',
               '-I' + str(target / 'include'), '-I' + str(headers), str(source),
               '-L' + str(library_dir), '-Wl,-rpath,' + str(library_dir),
               '-l:libnvrtc.so.12', '-lcuda', '-lcudart', '-o', str(library)]
    subprocess.run(command, check=True)
    manifest = {'command': command, 'abi_version': 1, 'capabilities': 1,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
                'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'vkfft_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root / 'src/VkFFT', text=True).strip()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(library)
    return library


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cuda', type=Path, required=True)
    args = parser.parse_args()
    build(args.output, args.cuda)
