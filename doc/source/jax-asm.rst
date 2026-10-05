JAX ASM and LOM integration
===========================

``pyvkfft.jax_asm`` exposes the compact CUDA propagation engine through typed
XLA FFI. It accepts immutable complex64 JAX arrays, zero-pads both spatial
dimensions to twice their size inside the FFT kernels, and crops the output
back to the input shape. Execution uses XLA's CUDA stream and per-call XLA
buffers. Cached plans own kernels, not reusable execution buffers.

Build and install
-----------------

Use the Python environment that will run the application. The builder copies
and patches VkFFT headers in a fresh build directory, without changing the
submodule or the ordinary CUDA extension. A C++ compiler, CUDA headers,
NVRTC and the CUDA driver are required.

.. code-block:: shell

   python examples/asm_ffi_build.py /tmp/asm-ffi-build --cuda "$CUDA_PATH" --install

``--install`` atomically installs ``pyvkfft/libvkfft_asm_ffi.so`` in this
checkout. Alternatively set ``PYVKFFT_ASM_FFI_LIBRARY`` to the built library.
Make this checkout importable in the application environment. CuPy is not
required by this interface. The build manifest records sources, compiler
arguments, library hashes and JAX header versions. Rebuild after native edits;
restart Python after replacing a loaded library.

API and differentiation
-----------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from pyvkfft.jax_asm import asm, convolve

   jax.config.update("jax_enable_x64", True)
   field = jnp.ones((512, 512), dtype=jnp.complex64)
   output = jax.jit(lambda x: asm(
       x, .01, 532e-9, pixel_pitch=(6.4e-6, 6.4e-6)))(field)

   # General complex H in natural FFT order, including shifted/asymmetric H.
   H = jnp.ones((1024, 1024), dtype=jnp.complex64)
   loss_and_gradient = jax.jit(jax.value_and_grad(
       lambda phase, H: jnp.sum(jnp.abs(convolve(jnp.exp(1j*phase), H))**2)))
   loss, gradient = loss_and_gradient(jnp.zeros((512, 512), jnp.float32), H)

``convolve(field, H, transfer_layout='fft', grouped_batch=0)`` accepts
arbitrary complex64 transfer tables of shape ``(2*height, 2*width)``. The
explicit ``'quadrant'`` layout accepts ``(height+1, width+1)`` and reflects
both axes; use it only for transfers known to be even in both axes.

``convolve_packed(field, payload, scale=None, grouped_batch=0,
tile_columns=0, tile_rows=None)`` accepts a
uint32 transfer of shape ``(..., 2*height, 2*width)`` in natural FFT order.
Signed int16 real/imaginary components occupy its low/high 16 bits. The default
float32 scale is ``1/32767`` for ASM; RSC supplies a float32 per-plane scale
with singleton spatial axes, ``(..., 1, 1)``. Decoding occurs inside the native
spectrum multiplication, including the transpose. No dense complex H is made.
Leading axes broadcast without duplicating the packed transfer. Payload and
scale are fixed, stop-gradient parameters; field JVPs, reverse mode and higher
field derivatives are supported. Use complex64 H when differentiating transfer
parameters. This API requires FFI ABI 3; rebuild older libraries.

``asm(field, z, wavelength, pixel_pitch=(dy, dx), bandlimit='none',
grouped_batch=0)`` evaluates the analytic transfer in the native kernels.
It discards evanescent modes and optionally applies a ``'rectangular'``
bandlimit. Distances and wavelengths may be dynamic float32 or float64 arrays.
Enable JAX x64 for accurate optical phase with float64 parameters. Output
precision remains complex64. Pitch, bandlimit and grouping are static options.

Both functions support JIT, JVP, reverse mode and broadcast leading axes.
``vmap`` and broadcast batches execute sequentially, sharing a per-plane
workspace. Field gradients use native fused propagation. General H gradients
use JAX FFTs, so optimizing H has a different memory/performance profile.
Analytic distance/wavelength derivatives through total order two are supported;
higher analytic parameter orders raise an error. Hard mask derivatives are
piecewise constant, and boundary derivatives are not mathematically defined.
At exactly zero longitudinal frequency, the wavelength slope uses a finite
clipped-branch convention.

Analytic derivatives separate the uniform carrier from the residual phase to
reduce cancellation. Small intensity derivatives can still be sensitive to
complex64 loss/cotangent rounding, especially with nearly paraxial fields.
The regression reference uses summed squared magnitude to avoid introducing
roundoff from a mean before carrier cancellation. This is not a complex128
propagator.

``clear_plan_cache()`` waits for outstanding cached-plan work before unloading
kernels. Existing compiled JAX functions remain valid and recreate plans on
their next call. Ordinary execution does not synchronize the device.

LOM shared propagation
----------------------

In ``/home/weik/code/LAFA/lom``, ``PropagationConfig.propagation_backend``
defaults to ``'vkfft'``. ``vkfft_grouped_batch=32`` selects the grouping used by
the benchmarks; the default of zero lets the native planner choose. Set
``propagation_backend='jax'`` for CPU execution or unsupported configurations.

The shared phase and complex-field factories, including
``generate_single_doe_model``, pass LOM's existing H to ``convolve`` or
``convolve_packed`` according to its storage. This
preserves shifted propagation, ASM/RSC, hard/Hann windows, complex amplitude
and post-correction. Both forward computation and DOE design gradients use
native propagation. The low-level already-padded helper keeps its FFT hooks.

Supported LOM geometry: even upscaled spatial dimensions, ``pad_factor=1``,
zero padding, ``keep_padding=False``, ``boundary_gap=0``, float32 configuration,
``tp=1`` and transfer storage ``'complex64'``, ASM ``'snorm16_compact'`` or RSC
``'snorm16_scaled'``. Unsupported settings raise an
explicit error. Single-GPU execution is validated; spatially distributed FFTs
and performance of multi-GPU data parallel batches are not validated here.
The default uses the fused full-workspace engine. Packed transfers can select
the tiled capacity engine described below; cached-phase engines are separate.

Tiled propagation for larger grids
----------------------------------

Build the additional typed FFI once:

.. code-block:: shell

   python examples/asm_streamed_ffi_build.py /tmp/asm-streamed-ffi --cuda "$CUDA_PATH" --install

This installs ``pyvkfft/libvkfft_streamed_ffi.so``. Alternatively set
``PYVKFFT_STREAMED_FFI_LIBRARY`` to that library. Restart Python after replacing
an already loaded library. CuPy is not required for JAX execution.

Use ``convolve_packed(..., tile_columns=1024)`` or the LOM overrides:

.. code-block:: text

   prop_cfg.propagation_backend=vkfft
   prop_cfg.transfer_storage=snorm16_compact
   prop_cfg.vkfft_grouped_batch=0
   prop_cfg.vkfft_tile_columns=1024

For RSC, select ``method=rsc`` and ``transfer_storage=snorm16_scaled`` instead.
Zero tile columns retains the faster full-workspace engine. Tiling currently
accepts packed transfers; it preserves the existing ASM/RSC table, fixed scale,
field derivatives and leading-axis broadcasting. It is an explicit memory/speed
choice, with no implicit GPU allocation probe or tuning.

The tiled schedule holds two compact half-spectra, using the output as one half,
and processes rows and columns in bounded tiles. Let ``N=height*width`` and
``T=max(tile_rows*2*width, tile_columns*2*height)`` in complex elements, with
tile extents clamped to their dimensions. The FFI workspace is ``8*N+16*T``
bytes, including ordered-row FFT temporary storage. Input, output and the
packed transfer add ``32*N`` bytes for a single complex-field call. All
execution arrays belong to XLA. ``streamed_cache_info()`` reports the actual
plans; ``clear_streamed_plan_cache()`` synchronizes and unloads them safely.

Ordinary radix FFT tiles support one, two or three uploads per axis. A
67,108,864-point transform with three uploads was checked against cuFFT,
including its frequency permutation and inverse. JAX FFI forward and
complex-input gradient checks also passed on H200 for fields of shape
``(2, 33554432)`` and ``(33554432, 2)``, using general asymmetric packed H.
Both plans reported three uploads on the long padded axis; full-array field
and gradient relative errors were below 1.05e-6 and 9.17e-7, respectively.
This support belongs to the tiled/streamed backend (positive
``tile_columns``); the full-canvas fused backend still accepts at most two
uploads per axis. VkFFT chooses the upload count automatically. Arbitrary
four-or-more-upload axes are not supported by this wrapper.

The 76,800-point axes in the large square test still use two uploads: the
large-grid memory saving comes
from tiling the 2-D workspace, rather than adding FFT passes.

A100 80 GB PCIe, JAX 0.5.3 measurements:

* Physical 18,000-square LOM ASM, loss plus full field and phase gradient:
  tiled 1024-column mode used **17.447 GiB** peak live allocation and **291.5 ms**.
  The earlier fused packed run used 24.140 GiB and 181.7 ms; the packed JAX/cuFFT
  pipeline used 38.624 GiB and 410.6 ms. Tiled forward/gradient errors against
  that same physical reference were 9.67e-7 / 1.47e-6.
* **38,400-square field, 76,800-square padded FFT**, tile 128: loss plus full
  field and nonzero phase gradient fit in **77.051 GiB** live allocation.
  A nonconstant phase design and constant complex packed H supplied an analytic
  reference without a dense reference FFT. Field/gradient errors were
  8.85e-7 / 3.46e-7; five-call median was about **1.35 s**.
* At 38,400 square, a direct comparison using the **same compact tiled schedule**
  and general asymmetric packed H measured approximately **645 ms VkFFT versus
  887 ms cuFFT** per forward call (1.37x speedup, median across three fresh
  processes). Each process used four alternating five-second blocks per backend.
  This includes
  padding, transposes, packed decoding and cropping; it is separate from the
  JAX loss-and-gradient timing. Process medians spanned 645.48--647.94 ms
  for VkFFT and 887.04--890.02 ms for cuFFT. Sampled relative error was below 9.8e-7.

The largest size is a validated single-plane outer-propagation capacity result,
not a guarantee that an entire DOE/BPN training graph fits. Optimizer state,
other retained fields, multiple transfer planes and transfer construction need
additional memory. Large H was already packed before timed execution. Near the
limit the test used an otherwise idle GPU with
``XLA_PYTHON_CLIENT_PREALLOCATE=true`` and
``XLA_PYTHON_CLIENT_MEM_FRACTION=.98``; it leaves little memory headroom.
Use smaller fields or fewer tile columns when the rest of the application needs
space. An extra dense reference or unpacked H will exceed this capacity.

The tiled suite passed five tests on both JAX 0.5.3 and 0.8.0, including
concurrent submission, JVP/Hessian products and cache clearing. All 27 LOM
integration cases and seven CuPy streamed tests passed, including the explicit
three-upload transform. Compute Sanitizer reported zero memory errors and zero
race hazards across all five tiled JAX tests; race checking covered VkFFT and
the shared-memory transfer/gather/scatter kernels.

Reproduction drivers: ``examples/lom_tiled_capacity.py`` (analytic capacity
oracle), ``examples/lom_packed_transfer_benchmark.py --tile-columns 1024``
(physical LOM comparison), and ``examples/asm_streamed_cufft_benchmark.py``
(paired sustained FFT-vendor comparison). Local artifacts and source hashes
are in ``/home/weik/code/FFT/asm-results/large-decomposition``.

Further capacity improvements: streamed ABI 2
---------------------------------------------

The newer streamed build declares safe field input/output aliasing to XLA.
Each source row tile is staged before its rows are overwritten. XLA reuses a
consumed temporary when possible and inserts a copy when the original array
is still needed. Ordinary calls preserve caller-owned arrays; explicit
``jax.jit(..., donate_argnums=(0,))`` can donate a field when the caller no
longer needs it. Tests cover multiple consumers, immutable input and actual
buffer donation.

LOM's tiled phase factory additionally rematerializes the pointwise incident
field / phase modulation in reverse mode. This avoids retaining a complex
plane solely for differentiating the phase. The transfer, FFT precision and
propagation geometry are unchanged. Rebuild the streamed FFI and restart Python
to enable ABI 2; ABI 1 libraries remain supported without the reuse saving.

For the single-plane phase-gradient probe with tile 128, the compiled memory
requirements are:

* **41,472-square field / 82,944-square FFT**, returning loss, full propagated
  field and phase gradient: **77.045 GiB**.
* **45,360-square field / 90,720-square FFT**, using the shared phase factory
  and returning loss and phase gradient without the full field: **76.822 GiB**.

Both sizes subsequently passed **full-square execution on an Orix H200** with
JAX 0.8.0. Actual peak live allocations were **77.045 GiB** and **76.822 GiB**,
matching the estimates. Five-call medians after warmup were **697.3 ms** and
**910.2 ms**, respectively. Full chunked validation against the capacity
oracle gave field/gradient relative errors of 1.13e-6/4.79e-7 at 41,472 square
and gradient error 5.05e-7 at 45,360 square. The oracle uses nonconstant phase
and constant complex packed H, not a physical propagation table.

Earlier 24,576-square A100 runs validated both paths at **27.094 GiB** and
**22.594 GiB**, with phase-gradient relative error 3.47e-7 and full-field error
8.78e-7. General complex transfers are also tested at the larger axis lengths
in both rectangular orientations.

The extension passed eight tiled JAX tests on versions 0.5.3 and 0.8.0 and
29 LOM integration cases, including tiled ASM/RSC DOE updates. All eight tiled
tests passed Compute Sanitizer memory and race checking with zero errors or
hazards. ABI 1 fallback and a fresh process using the installed ABI 2 library
were also checked.

The limits are graph-dependent. The loss-only saving applies to the shared
phase factory (used by ``generate_single_doe_model``); arbitrary callers of the
complex-field factory must arrange their own rematerialization. Retaining the
full field as auxiliary output, extra transfer planes and optimizer state
reduce the available capacity.

The capacity driver supports ``--plan-only`` to inspect memory without
allocating large inputs, and ``--loss-only --phase-factory`` to check the
training-style result. For example:

.. code-block:: shell

   python examples/lom_tiled_capacity.py --size 45360 --tile 128 \
       --loss-only --phase-factory --plan-only --output planned.json

Artifacts for this extension are in
``/home/weik/code/FFT/asm-results/larger-capacity``. The earlier 38,400-square
executed capacity and cuFFT comparison above remain the largest full-square
measurements made on an otherwise idle A100.

Orix H200 comparison
--------------------

The two larger square sizes were compared directly with cuFFT using identical
packed H, tile 128 and compact storage schedules. Padding, transposes, packed
decoding and cropping are included. Three fresh processes per size each ran
four alternating five-second blocks per backend. Medians of the three process
medians were:

.. list-table:: Matched tiled forward propagation on H200
   :header-rows: 1

   * - Field size
     - VkFFT
     - cuFFT
     - Speedup
   * - 41,472 square
     - 334.68 ms
     - 382.39 ms
     - 1.14x
   * - 45,360 square
     - 435.78 ms
     - 609.61 ms
     - 1.40x

The transfer is general and asymmetric. Checks at 65,536 distributed output
locations gave relative errors below 9.5e-7. These are forward-only vendor
comparisons, separate from the loss/gradient capacity measurements above.
The run used one allocated H200 in ``pi-heidriw``, CUDA 12.9, CuPy 14.0.1 and
cuFFT 11.4.1. Reproduction scripts, source hashes and results are under
``/home/weik/code/FFT/asm-results/orix-larger-20261004T214017Z``.

The shared LOM complex-field factory was compared at 20,480 square, returning
loss, full field and phase gradient together. Three fresh processes per
backend each measured five synchronized calls after warmup. Tiled VkFFT
measured **160.47 ms and 18.828 GiB** peak live allocation; JAX/cuFFT measured
**219.65 ms and 50.000 GiB**: a **1.37x speedup and 62.3% memory reduction**.
Both backends had field/gradient errors below 8.2e-7/3.4e-7 against the same
capacity oracle. The JAX baseline's packed CUDA multiplier was built for
H200 and its native FFI target and forward/conjugate numerics were verified.

The attempted 24,576-square JAX baseline failed accuracy and its timing is
excluded. An independent ``ifft2(fft2(x))`` probe on the 49,152-square canvas
returned a gain of **2,415,919,104** (the canvas element count), rather than
one. The 40,960-square probe passed. This reproduces a normalization failure
in the tested JAX 0.8.0 full-2D FFT path without LOM or packed multiplication;
the tiled cuFFT comparisons above passed. The rejected log and standalone
``fft_roundtrip_probe.py`` are retained in the artifacts.

All eight tiled JAX tests and all seven CuPy streamed tests passed on H200,
including the separately enabled three-upload 1-D case. Job 297526 completed
the larger capacity cases and vendor comparisons before failing the rejected
JAX baseline. Jobs 297530 and 297549 completed the remaining checks and valid
LOM comparisons with exit code zero. The measurements exclude transfer
construction, optimizer state and additional training intermediates.

Single-H200 resident capacity
--------------------------------

A separate capacity study used the same streamed ABI 2 binary on one H200
with 143,771 MiB device memory, JAX 0.8.0, tile 128, preallocation enabled
and ``XLA_PYTHON_CLIENT_MEM_FRACTION=.99``. These are the largest tested
FFT-friendly square sizes for each graph, not an exhaustive maximum across
all shapes:

.. list-table:: Executed resident capacity
   :header-rows: 1

   * - Returned result
     - Field side
     - Logical FFT side
     - Peak live allocation
     - Median call
   * - Loss, full field and phase gradient
     - 55,296
     - 110,592
     - 136.898 GiB
     - 1,237.96 ms
   * - Loss and phase gradient, shared phase factory
     - 60,750
     - 121,500
     - 137.716 GiB
     - 1,711.62 ms
   * - Forward only, donated input field
     - 67,500
     - 135,000
     - 136.044 GiB
     - 980.33 ms

The full-field case had field/gradient relative errors of 1.11e-6/4.37e-7;
the loss-only gradient error was 5.92e-7. Each case used one warmup and five
synchronized timed calls, followed by complete chunked analytic validation.
The inputs use nonconstant phase and constant complex packed H. Timings
exclude validation and transfer construction. These limits leave little
headroom for additional planes, optimizer state or physical-H preparation.

The forward-only case explicitly donates its complex64 input; the original
field is consumed and its pointer is reused for output. The complete result
after six successive applications had relative error 7.65e-6. It retains no
phase gradient or original field. All three modes keep the field, packed
uint32 transfer and execution workspace on the GPU; the logical padded FFT
canvas is processed in tiles, not materialized as a full complex plane.

For an ``n``-square field at tile 128, the measured allocation follows
approximately ``48*n*n + 4096*n`` bytes with full field and gradient, and
``40*n*n + 4096*n`` bytes for loss and gradient through the phase factory,
and ``32*n*n + 4096*n`` bytes for the donated forward call.
Compile-only checks at 56,000, 61,440 and 68,600 square, respectively, exceeded
the configured allocator limit; those larger cases were not executed.
The square FFT axes still select two uploads. A third upload does not
remove these quadratic field, transfer and gradient storage requirements.

Slurm job 297557 completed all eleven study steps with exit code zero.
Source manifests, plan metadata, numerical errors, memory records and
reproduction scripts are under
``/home/weik/code/FFT/asm-results/orix-capacity-20261004T220658Z``;
``summary.json`` records the audited results. The main capacity driver is
``examples/lom_tiled_capacity.py``; the archive additionally contains
``forward_capacity.py`` and ``jax_three_pass.py``.

Compressed packed transfers
---------------------------

Streamed ABI 3 can reduce both transfer storage and the intermediate spectrum
without changing the packed coefficients or complex64 FFT precision. Rebuild
with ``examples/asm_streamed_ffi_build.py --install`` in a fresh output directory
and restart existing Python processes.

For a transfer with exact zeros outside a rectangular cyclic frequency window,
pass only that window and its natural-FFT origin:

.. code-block:: python

   from pyvkfft.jax_asm import PackedTransferWindow, convolve_packed

   # payload is uint32 with shape (..., ky, kx).
   transfer = PackedTransferWindow(payload, (2 * height, 2 * width), (y0, x0))
   result = convolve_packed(field, transfer, tile_columns=128)

The window can wrap across zero frequency and need not be symmetric. Its
metadata is static under JIT; the payload remains a normal device-array pytree
leaf. The equivalent array-only call supplies ``transfer_origin=(y0, x0)``.
Frequency coefficients outside the window are defined to be exactly zero.
This is an explicit contract, not an automatic threshold on small coefficients.

The operator executes column FFTs only for the ``kx`` retained columns. Its
extra spectrum is ``height * max(0, kx - width)`` complex64 elements: when
``kx <= width``, the caller-provided output holds the entire intermediate
spectrum and this extra allocation disappears. Row FFTs retain their full
length. Tile scratch and VkFFT's internal transform scratch remain necessary.
Forward, transpose, broadcasting, JVPs, higher field derivatives and explicit
input donation use the same schedule. Packed H and scale remain fixed parameters.

For an axis-even transfer with broad support, use
``transfer_layout='quadrant'`` with a ``(..., height+1, width+1)`` packed array,
including DC and Nyquist edges. Reflection reduces H storage approximately
fourfold; it does not remove the extra spectrum. This is complex axis symmetry,
not Hermitian symmetry: coefficients are reflected without conjugation.
The caller must establish that symmetry. An origin and quadrant layout cannot
be combined. Full transfers continue to work with older streamed ABI 1/2 builds.

LOM's shared propagation factories expose these choices through
``vkfft_transfer_layout='window'`` or ``'quadrant'``, with ``'full'`` as the
default. Both compressed modes require ASM, ``snorm16_compact`` storage and
positive ``vkfft_tile_columns``. Window mode requires band limiting and derives
the union of exact support over all configured planes, preparing H directly
at the reduced size. Quadrant mode requires zero incident angles. The factory
preserves hard/Hann filtering, normalization and reference-plane de-embedding.
The separate helper accepting an already padded field requires full layout.

``examples/lom_windowed_benchmark.py`` compares real physical transfer tables
in fresh processes, optionally validating complete fields and phase gradients
against full-table execution. ``examples/lom_windowed_capacity.py`` exercises
larger resident fields and validates sampled gradients and impulse responses
against a complex128 direct inverse DFT of the stored coefficients.

Physical memory-reduction study (H200, 2026-10-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On one H200 with JAX 0.8.0 and tile 128, the physical ASM study measured the
following complete phase-loss/gradient calls. The field cases also return the
entire complex field. Full-table controls use the preceding streamed ABI 2
library; compressed cases use ABI 3.

.. list-table:: Median time and peak live device allocation
   :header-rows: 1

   * - Field / result
     - Full H: ms / GiB
     - Compressed H: ms / GiB
     - Speedup
   * - 18,000², field + gradient, window
     - 134.38 / 14.553
     - 66.30 / 7.565
     - 2.03x
   * - 32,768², field + gradient, window
     - 420.08 / 48.125
     - 178.75 / 24.323
     - 2.35x
   * - 32,768², loss + gradient, window
     - 419.87 / 48.125
     - 178.44 / 24.323
     - 2.35x
   * - 18,000², broad support, quadrant
     - 135.42 / 14.553
     - 133.54 / 10.932
     - 1.01x

Each entry aggregates three fresh processes in alternating mode order. Each
process has a warmup followed by three synchronized blocks, each lasting at
least two seconds and containing at least five calls. The reported time is
the median of the three process medians. For 32,768² field-returning execution,
these medians ranged from 419.81 to 420.36 ms for full H and 178.48 to 179.15 ms
for windowed H. The new ABI 3 full-table control measured 133.12 ms at 18,000²,
consistent with the old full-table path.

Window cases use a fixed 3.6 mm aperture, 520.6 nm wavelength, 10.08 mm distance,
zero incident angles, a Hann band limit, scalar-demodulated incident fields,
nonconstant phase and weighted output-intensity loss. H has 9,305² packed
coefficients at both resolutions, taking 0.323 GiB instead of 4.828/16 GiB.
The extra spectrum is zero bytes in both cases. The peak reduction is
48.0%/49.5%; no additional coefficient approximation is used. Memory includes
transfer-preparation high water and is recorded before reference validation.
The logical FFT sizes are 36,000² and 65,536². Complete field and gradient
relative L2 errors against full-table execution are below 1.55e-6.

The broad-support comparison instead uses 6.4 µm pitch and 0.1 m distance,
with the same wavelength and Hann convention. Quadrant reflection saves 24.9%
of peak memory, with bit-identical complete fields and gradients in this case.
It retains the extra spectrum. Dropping the returned field did not lower the
measured peak for the weighted-intensity graph; allocation depends on the
loss and its intermediate lifetimes.

A separate analytic recomputation prototype compared the same ordered-row
CuPy schedule, tile 512, 18,000² field, 532 nm wavelength, 6.4 µm pitch,
0.1 m distance and no band limit. It evaluates and SNORM16-quantizes H during
multiplication instead of loading a table. Alternating four two-second blocks
measured 60.66 ms stored versus 65.90 ms generated: an 8.6% slowdown for removing
4.828 GiB of H. Outputs were bit-identical. Per-mode operator-array requirements
are 12.345/7.517 GiB; these are calculated array footprints, not isolated-process
live peaks, since both modes share a process for this comparison. An A100
8,192² screen measured a 4.2% slowdown. This prototype is archived as
``transfer_recompute_study.py`` and is not a general replacement for LOM's
shifted, Hann-filtered or de-embedded transfer factory.

The windowed capacity driver also executed a 77,760² phase field with a
155,520² logical FFT on one H200. A single-pixel linear loss, full field and
full phase gradient used 90.721 GiB peak and measured 962.00 ms (five calls
after warmup). The directly prepared H still occupied 0.323 GiB. This simpler
linear loss has different buffer lifetimes from the weighted-intensity
benchmark above; it is not a general training-capacity bound. A 28-by-28 grid
of gradient samples and a separate donated impulse response were checked
against complex128 direct inverse DFT, with relative errors 1.01e-6/1.02e-6.
The impulse reused the donated field pointer. The large dense forward field
was not checked at every pixel; the 18,000² and 32,768² comparisons were complete.

Validation comprises 14 streamed FFI cases on JAX 0.5.3 and 0.8.0, including
14 H200 cases; 43 H200 LOM cases; two additional local compressed-layout DOE
optimization cases (45 distinct LOM cases total); eight old-ABI compatibility
cases; six ordinary CuPy cases; and complete row/column three-upload JAX
comparisons. Compute Sanitizer passed all 14 FFI cases with zero memcheck errors
and zero racecheck hazards. Racecheck covered FFT, transfer and tile kernels;
handled CUDA API symbol-probe reporting was disabled as in earlier validation.

Slurm job 297970 completed with exit zero. An earlier attempt passed native
tests but stopped on a missing DOE material CSV; after linking the same
checksum-verified data used locally, all 43 integration cases passed. Timings
exclude compilation and H construction, and represent one propagation plane,
not a complete training iteration or optimizer state. The installed local
streamed library is ABI 3. Source/build hashes, raw measurements, failed and
successful logs, scheduler accounting, runners and the audited ``summary.json``
are under ``/home/weik/code/FFT/asm-results/memory-reduction-20261005T080124Z``.

Windowed single-H200 capacity boundary (2026-10-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A subsequent probe tested larger FFT-friendly square fields with the same
physical windowed ASM (3.6 mm aperture, 520.6 nm wavelength, 10.08 mm distance,
Hann filter, zero incident angles). H remains 9,305² uint32 coefficients,
occupying 0.323 GiB. Each process uses one H200, JAX 0.8.0 and a 99% preallocated
BFC pool, whose reported limit is 138.413 GiB. The native ABI 3 binary is
unchanged from the preceding memory study.

.. list-table:: Executed capacity and speed tradeoffs
   :header-rows: 1

   * - Result retained
     - Field size
     - Tile
     - Peak live GiB
     - Median ms
   * - Donated forward, largest tested
     - 135,828²
     - 64
     - 138.084
     - 1,376.92
   * - Donated forward, faster nearby size
     - 135,520²
     - 128
     - 137.674
     - 1,062.59
   * - Linear loss, full field and phase gradient, largest tested
     - 96,250²
     - 1
     - 138.413
     - 7,614.89
   * - Linear loss, full field and phase gradient, faster nearby size
     - 96,096²
     - 128
     - 138.408
     - 1,366.25
   * - Linear loss, full field and phase gradient, more memory headroom
     - 95,040²
     - 128
     - 135.281
     - 1,193.11

The largest forward field uses a 271,656² logical FFT and has 4.049 times as
many pixels as the earlier 67,500² full-H forward result. Input and output
share one complex64 device buffer; no original input or phase gradient is
retained. The dense input is a nonconstant sum of two separable plane waves,
regenerated outside timing. A 36-by-36 output sample grid was compared against
complex128 direct inverse DFT using analytic finite sums for the input
spectrum, with relative error 5.96e-7. Donation was verified on every call.

The largest gradient case uses a 192,500² logical FFT and returns a
single-pixel linear loss, full complex field and full phase gradient for a
nonconstant dense phase. Gradient samples and a separate donated impulse
response on 28-by-28 grids agreed with direct DFT to 1.44e-6/1.46e-6. Tile 1
extracts only 0.32% more pixels than 96,096² at about 5.6 times the runtime,
while using effectively the entire allocator pool. It is a capacity edge,
not a sensible general training default. The loss is linear; weighted losses,
optimizer state and additional propagation planes have different live arrays.

All timings are five synchronized calls after warmup, excluding initialization,
compilation and validation. Live peaks include transfer preparation. No giant
field is offloaded to host memory. Validation at these sizes is sampled rather
than a full reference-array comparison. Forward-only measurements used two
H200 allocations sequentially; each individual case ran on exactly one GPU.

Eight configurations completed execution and numerical checks. Four ran out
of memory: gradient 96,228² with tile 16; forward 135,828² with tile 128; and
forward 136,080² with tiles 16 and 1. The smaller-tile 96,228² gradient case
passed, illustrating the dependence on tile size and allocation behavior.
Gradient 96,668² and forward 136,500² with tile 1 exceeded the compiled memory
budget and were not executed. Therefore the results bound the tested
FFT-friendly sizes under this graph and allocator, not every arbitrary shape
or possible allocator configuration.

Slurm jobs 298017 and 298025 completed with exit zero, preserving expected OOM
records; the initial job 298015 stopped on the first OOM before the runner was
extended to continue through capacity failures. Source hashes, exact cases,
successful and failed logs, scheduler accounting, reproduction drivers and
audited ``summary.json`` are under
``/home/weik/code/FFT/asm-results/window-limit-20261005T093516Z``.

Validation and measured scope
-----------------------------

The CUDA tests cover independent NumPy references, asymmetric H transposes,
H/field gradients, finite differences, JVPs, Hessian products, first/second
optical-parameter derivatives, broadcasting, multi-upload transforms along
both axes, concurrent submissions and cache clearing. LOM integration tests
compare shared ASM/RSC forward and backward paths, and run five real DOE
optimization updates through quantization, spectral phase and sensor integration.

The packed-transfer build passed all 20 FFI tests on JAX/jaxlib 0.5.3 and 0.8.0,
and all 23 LOM integration cases on 0.5.3. These include signed-int16 extrema,
RSC amplitude scales, broadcast polarization/wavelength axes, packed field
JVPs/Hessians, and five DOE updates with each storage format.
Compute Sanitizer memcheck reported zero
device errors; racecheck reported zero hazards for the VkFFT kernels. Both
sanitizer runs passed all 20 FFI tests. CUDA API status reporting was disabled
after a pure-JAX control reproduced XLA's handled ``CUDA_ERROR_NOT_FOUND``
symbol probes; the unfiltered logs and control are retained. Racecheck was
filtered to ``VkFFT_main`` kernels. The broader LOM regression run passed
74 cases with 10 pre-existing environment/multi-device skips before the
optical-parameter refinement and packed-transfer extension.

Run with a selected CUDA device and the built library:

.. code-block:: shell

   python pyvkfft/test/test_jax_asm.py -v
   # In the LOM checkout:
   python -m pytest tests/test_vkfft_propagation.py -q

The test runners require ``PYVKFFT_ASM_FFI_LIBRARY`` explicitly to distinguish
configured GPU checks from skips. Runtime use does not require this variable
when the library is installed beside ``jax_asm.py``.

A 18000 x 18000 complex64 phase-design loss plus forward field and full phase
gradient was checked against an independent two-tap spatial stencil on an
A100 80 GB PCIe, with JAX/jaxlib 0.5.3. Five synchronized calls after compilation
and warm-up gave native/JAX medians of 189.2/416.6 ms. Native forward and gradient
relative L2 errors were 9.64e-7 and 2.55e-6. XLA peak live allocations before
reference validation were 31.38/43.45 GiB; these include FFT scratch, and are
distinct from reserved allocator pools. Each backend ran in a fresh process.
These are smoke measurements, not sustained timing confidence intervals or
an end-to-end LOM training benchmark.

Local build, accuracy, benchmark and sanitizer artifacts are under
``/home/weik/code/FFT/asm-results/jax-ffi``. The reproducible large-grid driver is
``examples/jax_asm_benchmark.py``; use ``--backend vkfft`` or ``--backend jax``.

Packed ASM/RSC verification
---------------------------

``examples/lom_packed_transfer_benchmark.py`` prepares physical LOM transfer
tables, then measures each storage/backend combination in a fresh process.
Dense controls decode the same quantized H so that storage is the only change.
At 18000 x 18000 (36000 x 36000 FFT) on A100 80 GB PCIe, packed VkFFT reduced
peak live allocation from 28.968 to 24.140 GiB in both ASM and RSC, saving
4.828 GiB. Compiled temporary bytes were unchanged. Transfer construction and
reference validation were excluded from measured execution memory.

Five-call medians were 186.65/181.71 ms for dense/packed native ASM and
188.70/183.70 ms for dense/packed native RSC. Packed JAX references measured
410.61/411.98 ms and 38.624 GiB. Full packed-native forward and design-gradient
relative L2 errors against JAX were below 9.9e-7 and 1.5e-6. These are outer
propagation smoke measurements, not complete BPN training or sustained
confidence intervals. The original transfer quantization remains lossy.

Use ``--prepare --method asm --size 18000 --directory DIR`` once, then
``--method asm --size 18000 --directory DIR --storage dense`` or ``packed``
in separate processes. Add ``--grouped-batch 32`` for these large-grid results;
the default zero lets the planner choose. Repeat with ``--method rsc`` and
a new directory, or add ``--backend jax`` for the reference. LOM and the
proxy's compact/scaled transfer modules must be on ``PYTHONPATH``.
Detailed results, build identity and limitations are in
``/home/weik/code/FFT/asm-results/jax-ffi/packed-validation.json``.
