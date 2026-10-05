#!/usr/bin/env bash
# Build isolated baseline/patched CUDA libraries; leave the installation intact.
# Usage after conda activate jax:
#   bash examples/issue205_build.sh /tmp/issue205-build "$CONDA_PREFIX"
set -euo pipefail

issue205_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
issue205_output=${1:?Supply a build directory outside the installed package}
issue205_cuda_root=${2:?Supply the Conda environment containing CUDA 12 headers/libraries}
mkdir -p -- "$issue205_output"
issue205_output=$(cd -- "$issue205_output" && pwd)
issue205_cuda_target="$issue205_cuda_root/targets/x86_64-linux"

# An empty, dedicated directory makes reruns and dependency provenance explicit.
if [[ -e "$issue205_output/vkFFT" ]]; then
    echo "Use a fresh build directory: $issue205_output/vkFFT already exists" >&2
    exit 1
fi
cp -R -- "$issue205_project/src/VkFFT/vkFFT" "$issue205_output/vkFFT"
patch --batch -p1 -d "$issue205_output" < "$issue205_project/examples/issue205_inverse_upload_order.patch"

for issue205_variant in baseline fixed; do
    issue205_headers="$issue205_project/src/VkFFT/vkFFT"
    if [[ "$issue205_variant" == fixed ]]; then
        issue205_headers="$issue205_output/vkFFT"
    fi
    g++ -x c++ -std=c++17 -shared -fPIC -O2 -DVKFFT_MAX_FFT_DIMENSIONS=8 \
        -I"$issue205_cuda_target/include" -I"$issue205_headers" \
        "$issue205_project/src/vkfft_cuda.cu" \
        -L"$issue205_cuda_target/lib" -Wl,-rpath,"$issue205_cuda_target/lib" \
        -l:libnvrtc.so.12 -lcuda -lcudart \
        -o "$issue205_output/$issue205_variant.so"
done
echo "Built $issue205_output/{baseline,fixed}.so (C API only; no JAX FFI exports)"
