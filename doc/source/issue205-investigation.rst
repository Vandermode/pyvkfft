VkFFT issue 205: inverse pass ordering
======================================

Finding
-------

The failure in `VkFFT issue 205 <https://github.com/DTolm/VkFFT/issues/205>`_
is reproducible with this checkout. For in-place complex transforms requiring
multiple uploads, ``disableReorderFourStep=True`` produces a valid permuted
forward spectrum, but the ordinary inverse dispatches its stages in the wrong
order for that layout. An FFT/IFFT round trip already fails without any kernel
multiplication, padding, image processing, or JAX involvement.

The issue was open with no comments when inspected. Upstream ``develop`` commit
``e687182171ed817535bbaf9ee553fdc324ce9d29`` still contains the descending ordinary
inverse loops; this was a source inspection, not an upstream build test.

Evidence and mechanism
----------------------

In ``src/VkFFT/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h``:

* Forward uploads run from ``numAxisUploads - 1`` down to zero.
* Ordinary inverse uploads also descend, at the loops originally on lines 463
  and 548, even when reordering is disabled.
* The native convolution return path instead traverses inverse uploads in
  ascending order. Its fused kernel has already executed the first inverse
  stage on the final transformed axis, so its remaining traversal starts at one.

The generated unreordered inverse kernels are already specialized for the
reverse traversal: ``appendReorder4Step`` in
``vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_4step.h`` applies
inverse twiddle factors on reads, whereas the corresponding forward operation
applies them on writes. Changing only the ordinary inverse traversal fixes the
tested in-place cases.

For two uploads with ``axis_split = [a, b]``, packing a natural spectrum is
``spectrum.reshape(a, b).T.copy().reshape(original_shape)`` in 1D. Unpacking
uses ``reshape(b, a)`` before the transpose. The same factor swap applies
separately to each split axis in the tested multidimensional cases; VkFFT lists
axes fastest-first, opposite to a C-order NumPy shape.

If ``P`` denotes this permutation and ``F`` the FFT, the forward output is
``P F(x)``. Pointwise multiplication of two identically permuted spectra is
valid. The inverse must compute ``F^-1 P^-1``. The original inverse does not do
this; unpacking before and after that inverse also fails. This is not merely a
normalization error or a missing kernel transpose.

Measurements
------------

Tests used physical A100 80 GB GPUs 1, 2, and 3, selected by UUID, in the ``jax``
environment. Each test used one GPU; no distributed execution was implemented.
References were CuPy/cuFFT circular convolutions of deterministic random complex
arrays. Reported errors are ``norm(result-reference) / norm(reference)``.

.. list-table:: In-place convolution relative L2 errors
   :header-rows: 1

   * - Shape / precision
     - Original, reordering disabled
     - Patched, reordering disabled
     - Normal ordered FFTs
   * - 65,536 / complex64
     - 1.039
     - 8.58e-7
     - 1.01e-6
   * - 32,768 x 128 / complex64
     - 1.040
     - 1.08e-6
     - 1.09e-6
   * - 8,192 x 32,768 / complex64
     - 1.256
     - 1.42e-6
     - 1.42e-6
   * - 128 x 65,536 / complex128
     - 1.040
     - 9.79e-16
     - 9.95e-16

The large complex64 case has two uploads on both axes. The rectangular cases
isolate the contiguous and strided axes. A 4,096 x 4,096 complex64 control uses
one upload per axis and succeeds without the patch. A fresh unmodified CUDA
12.9 build reproduces the installed library's failure at length 65,536.

For 8,192 x 32,768 complex64 on GPU 2, the final build's median time over 15
samples was 25.39 ms with normal reordering and 24.94 ms with the patched
unreordered pipeline (about 1.8% less time). CUDA events covered forward FFT,
multiplication, and inverse FFT; planning, kernel FFT, allocations, and input
reset were excluded. Three warm-up iterations preceded measurement. This small
difference is not evidence of a substantial or distributed speedup.

Preparation-cost audit
----------------------

The original timer already excluded the kernel FFT and its layout preparation.
The multiplication spectrum was obtained with the corresponding FFT plan before
the start event, so the unreordered path multiplied by an already permuted
spectrum. No explicit packing or unpacking occurred inside either timed path.

A follow-up benchmark prepared both spectra, all plans, buffers, and events
before measurement and synchronized the GPU after preparation. It then used
20 warm-ups and 100 measurements per variant, alternating the order of the two
variants within successive pairs. Both variants used the same work buffer.
Input reset preceded the start event on the same stream and was excluded.
Intermediate CUDA events recorded separate FFT, multiplication, and inverse
times; the total still measured only those three operations.

.. list-table:: Prepared-data convolution medians, 8,192 x 32,768 complex64
   :header-rows: 1

   * - Physical GPU
     - Ordered pipeline (ms)
     - Patched unreordered pipeline (ms)
     - Time reduction
   * - 1
     - 25.45
     - 24.93
     - 2.05%
   * - 2
     - 25.40
     - 24.91
     - 1.93%
   * - 3
     - 26.72
     - 26.05
     - 2.52%

On GPU 2, ordered FFT/multiply/IFFT stage medians were 10.846/3.713/10.846 ms;
unreordered stage medians were 10.613/3.712/10.591 ms. Individual medians need
not sum exactly to the median total. Both outputs passed the same correctness
checks. These results support a small improvement for this workload even when
all preprocessing is excluded; they do not characterize other shapes or a
distributed implementation. Raw results are
``../issue205-results/8192x32768-c64-prepared-gpu{1,2,3}.json``.

Larger problems and sustained utilization
-----------------------------------------

``examples/issue205_sustained.py`` extends the study to larger complex64 arrays
and records NVML telemetry every 100 ms. Each variant runs twice, in
normal/disabled/disabled/normal order, for approximately eight seconds per block.
Twenty warm-up convolutions precede each block. CUDA events measure consecutive
FFT/multiply/IFFT executions; no input resets, preparation, or allocations occur
inside the timed blocks.

A dense, complex unit-modulus spectral filter keeps repeated convolutions
bounded. Both natural and permuted spectra are prepared beforehand. Each path
first passes a cuFFT-based convolution check (relative L2 errors below 1.2e-6).
This changes the test filter from the earlier random spatial kernel, while
retaining the same dense C2C operations and data sizes.

.. list-table:: Sustained convolution medians, one A100 per run
   :header-rows: 1

   * - Shape
     - GPU
     - One array (GiB)
     - Ordered (ms)
     - Patched unreordered (ms)
     - Time reduction
   * - 8,192 x 32,768
     - 1
     - 2
     - 25.52
     - 24.91
     - 2.37%
   * - 16,384 x 32,768
     - 2
     - 4
     - 50.98
     - 49.97
     - 1.99%
   * - 32,768 x 32,768
     - 3
     - 8
     - 106.96
     - 104.43
     - 2.36%
   * - 16,384 x 65,536
     - 1
     - 8
     - 102.36
     - 100.37
     - 1.94%
   * - 65,536 x 16,384
     - 2
     - 8
     - 87.93
     - 88.20
     - -0.31%

All 1,358 steady-state telemetry samples reported 100% GPU utilization, for both
variants and all five cases. Steady-state windows exclude the first second and
final 0.2 seconds of each timed block to avoid mixing NVML's sampling window with
block transitions. Full traces include utilization, memory activity, allocated
device memory, power, and SM clocks. The largest square problem contains four
times as many elements as the original test and still gives a small improvement.
The transposed rectangular case gives essentially no benefit.

NVIDIA defines GPU utilization as the fraction of the sampling period in which
at least one kernel executes; it does not measure the fraction of peak FLOPs or
peak memory bandwidth achieved. See the `NVIDIA utilization documentation
<https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization>`_. These results
establish continuous GPU activity, not maximum possible hardware efficiency.

Reproduce a sustained run with::

    python examples/issue205_sustained.py --gpu 3 --shape 32768 32768 \
        --library /tmp/205-build/fixed.so --output /tmp/205-large.json

The study's ``../issue205-results/sustained-summary.json`` links the five raw
JSON files. Each includes correctness checks, plan splits, per-convolution
timings, block boundaries, and all telemetry samples.

Why disabling reordering gives a small speed benefit
----------------------------------------------------

VkFFT incorporates reordering into the FFT passes' generated memory accesses.
For these C2C plans there is no additional standalone full-array transpose kernel
that disappears when the option is disabled. A diagnostic run with
``printMemoryLayout=1`` confirmed the following for 8,192 x 32,768 complex64:

* Normal forward: buffer -> temporary -> buffer -> temporary -> buffer.
* Unreordered forward: four buffer -> buffer passes.
* Both inverse paths also execute four passes, with the same respective buffer
  patterns.
* The normal plan allocates 2 GiB of temporary storage; the unreordered plan
  allocates none.

The trace is saved in ``../issue205-results/memory-layout-trace.log``. It comes
from the instrumented dispatch path, not a hardware-counter profiler. The
dispatch loops are in ``vkFFT_AppManagement/vkFFT_RunApp.h``; generated write
addressing depends on ``reorderFourStep`` in
``vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_ReadWrite.h``.

Both convolution paths therefore retain eight FFT passes plus one pointwise
multiplication. If each FFT pass reads and writes one array of size S, their
logical array traffic is approximately ``8 * 2S + 3S = 19S`` each: 38 GiB when
S is 2 GiB. This estimate excludes twiddle tables, transaction efficiency,
caching, and other kernel details; it is not measured DRAM traffic.

Removing the temporary allocation saves storage capacity, while changing where
the existing reads and writes go. Remaining speed differences reflect generated
addressing, local transposition, access patterns, and occasionally plan splits;
their individual contributions have not been isolated with hardware counters.
The per-stage timings above show that multiplication is essentially unchanged
and each FFT direction saves only a few percent. Enlarging the array scales the
remaining passes too, so it does not inherently increase the fractional saving.

Fused convolution versus cuFFT
------------------------------

The useful performance optimization for the tested large 2D problems is
VkFFT's existing native convolution mode, enabled with ``convolve=True`` and
``disableReorderFourStep=True``. Prepare the filter in the plan's permuted
frequency layout, then execute ``app.fft(buffer, convolve_kernel=packed_filter)``.
This call performs the entire forward/multiply/inverse convolution. It fuses
the last forward stage, multiplication, and first inverse stage into one kernel.
The filter must remain alive for execution; an ordinary natural-order spectrum
cannot be passed directly when the plan requires a permutation.

``examples/fft_convolution_benchmark.py`` implements this path and compares it
with reusable cuFFT C2C plans, a separate multiplication kernel, and a cuFFT
LTO inverse-load callback that fuses multiplication. cuFFT normalization is
folded into its filter offline. Both cuFFT alternatives are measured, and the
faster one is used in the following comparison. This is the best of these two
tested cuFFT pipelines, not an exhaustive cuFFT tuning study.

.. list-table:: Prepared 2D complex64 circular convolution, median milliseconds
   :header-rows: 1

   * - GPU
     - Shape
     - Separate cuFFT
     - Callback cuFFT
     - Fused VkFFT
     - Speedup over faster cuFFT
   * - A100 80 GB PCIe, GPU 2
     - 8,192 x 32,768
     - 22.747
     - 45.625
     - 19.573
     - 1.162x
   * - A100 80 GB PCIe, GPU 2
     - 16,384 x 32,768
     - 45.084
     - 44.570
     - 40.191
     - 1.109x
   * - A100 80 GB PCIe, GPU 3
     - 32,768 x 32,768
     - 100.063
     - 130.982
     - 88.623
     - 1.129x
   * - H100 80 GB HBM3, Orix
     - 8,192 x 32,768
     - 12.486
     - 17.347
     - 11.091
     - 1.126x

The first row uses ``aimThreads=256`` after an eight-candidate local sweep;
other rows use VkFFT defaults. The sweep changed threads, coalesced memory,
and register boost settings. Thread tuning saved approximately another 1% at
the first shape; fusion accounts for most of the gain. The same-size default
A100 run was 19.777 ms versus 22.698 ms for cuFFT. Do not assume that 256 is
optimal for other shapes or architectures.

The first row contains four alternating blocks of five seconds per variant.
Its four VkFFT block medians were 19.562, 19.569, 19.577, and 19.582 ms;
cuFFT's were 22.654, 22.739, 22.753, and 22.842 ms. Every steady NVML sample
in this run reported 100% GPU utilization for all three variants. Other A100
rows used two three-second blocks per variant; H100 used two four-second blocks.
All runs used CUDA 12.9, CuPy 14.0.1, and cuFFT 11.4.1 (version code 11401).
Explicit library loading avoids accidentally comparing against the older
cuFFT bundled in the local Python environment.

All plans, allocations, filter normalization/permutation, compilation, and
input resets occur outside timing. CUDA events measure consecutive prepared
convolutions after warm-up, with variant order reversed each round. The dense
unit-modulus spectral filter keeps repeated results bounded; its coefficients
are runtime data. Every variant first passes a cuFFT-reference correctness
check, with VkFFT relative L2 errors around 1e-6. These measurements cover
single-GPU, in-place, complex64, circular convolution with a reusable filter.
Real transforms, linear-convolution padding/cropping, changing filters, and
distributed execution are outside this comparison.

The winning native convolution path also passed against the installed,
unpatched library: on GPU 1 it took 19.557 ms versus 22.617 ms for cuFFT at
8,192 x 32,768, with relative L2 error 9.90e-7. Its library SHA-256 is
``04880134edc80d9967a4220e9a830eb24612f8cc6a3740c5077f5a8f02369585``.
The experimental inverse-order patch is needed only for the separately
measured ordinary unreordered FFT/multiply/IFFT path, not this native path.

Hardware-counter evidence
~~~~~~~~~~~~~~~~~~~~~~~~~

Orix Slurm job 296781 completed successfully on an H100 in 2 minutes 25 seconds.
Nsight Systems 2025.1.3 captured one prepared convolution per variant, and
Nsight Compute 2025.2.1 captured the corresponding kernels with hardware
counters enabled. All three profiler runs exited successfully.

Both the separate cuFFT pipeline and native VkFFT launched seven kernels.
cuFFT used three FFT kernels per direction plus multiplication. VkFFT used six
ordinary stages and one fused stage. Thus this comparison is not explained by
a lower launch count. cuFFT and VkFFT choose different FFT decompositions.

From Nsight Compute's DRAM bandwidth multiplied by each kernel's duration,
total DRAM traffic was approximately 32.779 GiB for separate cuFFT and
29.862 GiB for fused VkFFT, an 8.9% reduction. VkFFT's six ordinary passes
each transferred about 3.98 GiB, and its fused pass transferred 5.98 GiB.
Its kernels reached 82.7-90.2% of peak sustained DRAM throughput. Together,
these counters support memory-traffic reduction as a major source of the gain;
they do not attribute every part of the speed difference to one cause.

The cuFFT callback transferred less data (26.725 GiB), but its LTO 8,192-point
kernel alone took 9.656 ms under profiling and achieved only 26.2% of peak DRAM
throughput. Fewer bytes or launches therefore do not guarantee a faster plan.
The reported performance table comes from unprofiled sustained runs; profiler
replay timings are used only for kernel diagnosis.

Reproduce the winning path with the installed library from the project root::

    conda activate jax
    CUDA_PATH="$CONDA_PREFIX" python examples/fft_convolution_benchmark.py \
        --gpu 2 --shape 8192 32768 \
        --variants cufft cufft_callback vkfft_native \
        --cufft-library "$CONDA_PREFIX/targets/x86_64-linux/lib/libcufft.so.11" \
        --vkfft-options '{"aimThreads":256}' \
        --seconds-per-block 5 --rounds 4 --output /tmp/convolution.json

To include the ordinary unreordered path, pass ``--library`` with the isolated
patched build and omit ``--variants``. For a profiler run add ``--profile``;
CUDA profiler start/stop brackets one prepared execution of each selected
variant, with NVTX labels. Inside a Slurm allocation use ``--slurm`` instead
of ``--gpu`` to preserve scheduler device isolation.

Raw A100 results are ``../issue205-results/convolution-*.json``. The compact
``convolution-summary.json`` and ``convolution-comparison.png`` summarize the
table. H100 results, Nsight reports, CSV exports, the submission script, source
manifest, pinned environment, and ``profile-summary.json`` are in
``../issue205-results/orix-20261004T110443Z/``. Archived source records the exact
profiling snapshot; subsequent local changes add tuning options and reject
non-finite correctness errors.

Reproduction and artifacts
--------------------------

The checkout was ``c9fafa0`` with VkFFT submodule
``635b1fbc8a4b43476f4420af0a507fa93aa8af78``. The installed library reports
VkFFT 1.3.5; CuPy is 14.0.1 and CUDA runtime is 12.9. JSON outputs record the GPU
UUID and library SHA-256. The isolated builds use CUDA 12.9 headers/libraries and
the repository's C wrapper; they do not export the JAX FFI handlers.

From the project root::

    conda activate jax
    python examples/issue205_repro.py --gpu 1 --shape 65536 --output /tmp/205-original.json
    bash examples/issue205_build.sh /tmp/205-build "$CONDA_PREFIX"
    python examples/issue205_repro.py --gpu 1 --shape 65536 \
        --library /tmp/205-build/fixed.so --output /tmp/205-fixed.json

Use a fresh build directory. The build script copies the headers, applies
``examples/issue205_inverse_upload_order.patch`` to the copy, and builds both
``baseline.so`` and ``fixed.so``. It does not alter the submodule or installation.
Select ``baseline.so`` for a comparison with identical build settings. For the
large 2D case, use ``--gpu 2 --shape 8192 32768 --benchmark-repeats 15``.
Use ``--benchmark-repeats 100`` for the preparation-cost audit protocol; the
current script always uses 20 warm-ups and alternating variant order.

This study's outputs are in ``../issue205-results/`` relative to the project
root. ``*-final.json`` files use the scoped patch; intermediate ``fixed12``
files used the initial broader dispatch experiment. The script records
diagnostic failures rather than treating every failed hypothesis as a test
failure: for example, comparing a packed spectrum to natural order should fail.

Scope and practical options
---------------------------

The proposed patch explicitly targets in-place C2C transforms. It preserves
existing dispatch for formatted input/output buffers, R2C, DCT/DST, and Bluestein
axes. The initial broader experiment failed out-of-place radix round trips;
those require additional buffer-routing work. R2C/DCT/DST, three-upload plans,
JAX FFI execution, and distributed communication have not been validated by this
patch. It is an isolated proof of concept, not an installed production fix.

Native 2D convolution passed the controls, but native 1D multi-upload convolution
at length 65,536 also failed in this checkout, independently of the proposed
patch. Natural, manually packed, and disabled-reorder-FFT kernel spectra were
tried; none corrected that separate case.

With the current installation, use ordinary ordered FFTs, or unpack the product
of the two unreordered spectra and feed it to an ordered inverse. The latter
workaround passed all tested one/two-upload radix layouts. For a distributed
implementation, folding that unpacking into an already necessary communication
packing step is a possible optimization to investigate. With the experimental
patch, the original in-place FFT/multiply/IFFT sequence works directly with the
permuted spectra in the tested cases.
