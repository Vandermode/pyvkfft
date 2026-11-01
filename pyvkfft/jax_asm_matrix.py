"""Differentiable matrix-kernel windowed convolutions through a VkFFT XLA FFI (CUDA, complex64).

:func:`matrix_convolve_window` evaluates, for every field ``b`` and kernel ``k``::

    out[..., b, k, o, :, :] = window(IFFT2(sum_i M_k[o, i] * FFT2(embed(field[..., b, i, :, :]))))

with ``M`` field components (e.g. Ex/Ey), ``K`` per-frequency ``M x M`` kernels in natural FFT order, an
explicit FFT canvas, input placement and output window (the geometry of
:func:`pyvkfft.jax_asm.convolve_window`). By default ``M = 2`` and the kernels are symmetric,
``kernels[..., K, 3, CH, CW] = (xx, xy, yy)``. A static ``components`` map instead addresses general
entries in a stack of planes ``kernels[..., P, CH, CW]``: ``components[k][o][i]`` is a plane index or
``None`` (zero), so diagonal or shared kernels need no zero planes.

Each (field, view) is one native call of VkFFT's matrix convolution (coordinateFeatures =
matrixConvolution = M, numberKernels = K): M forward FFTs, the K matrix products inside the column
convolution kernel, then K*M inverse FFTs; the compact input is read and the output window written
directly by the FFT kernels (columns longer than 2048 take two VkFFT uploads, so the convolution step
runs on short FFTs without register spills; a K-kernel step that would still spill falls back to K
single-kernel passes). Field JVPs and VJPs use the transpose, which is the same engine in reduce
mode (windows exchanged, frequencies reversed and the map transposed: JAX's bilinear transpose, no
conjugation): for K*M <= 8 the K*M cotangent planes are the features of one pass whose convolution step
sums over k before the M inverse FFTs, otherwise K passes accumulate in the output store. Kernel
derivatives are not implemented and raise.

Build examples/asm_matrix_ffi_build.py and set PYVKFFT_ASM_MATRIX_FFI_LIBRARY, or install the result as
pyvkfft/libvkfft_asm_matrix_ffi.so. Independent of the ABI 4 library used by :mod:`pyvkfft.jax_asm`.
"""
import ctypes
from functools import lru_cache, partial
import json
import math
import os
from pathlib import Path
import threading

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from jax._src import dispatch
import numpy as np

_SUPPORTED_ABI = (1,)
_MAX_FEATURES, _MAX_KERNELS, _MAX_FUSED = 4, 16, 8
_registration_lock = threading.Lock()


@lru_cache(maxsize=1)
def _library():
    path = Path(os.environ.get('PYVKFFT_ASM_MATRIX_FFI_LIBRARY',
                               Path(__file__).with_name('libvkfft_asm_matrix_ffi.so')))
    if not path.is_file():
        raise RuntimeError('Build examples/asm_matrix_ffi_build.py and set '
                           'PYVKFFT_ASM_MATRIX_FFI_LIBRARY to libvkfft_asm_matrix_ffi.so')
    library = ctypes.CDLL(str(path.resolve()))
    library.asm_matrix_ffi_abi_version.restype = ctypes.c_int
    abi = library.asm_matrix_ffi_abi_version()
    if abi not in _SUPPORTED_ABI:
        raise RuntimeError('Unsupported matrix ASM FFI ABI')
    library.asm_matrix_ffi_cache_info.restype = ctypes.c_char_p
    library.asm_matrix_ffi_clear_cache.argtypes = []
    library.asm_matrix_ffi_clear_cache.restype = None
    library._target = f'pyvkfft_asm_matrix_v{abi}'
    jax.ffi.register_ffi_target(library._target, jax.ffi.pycapsule(library.PyVkFFTASMMatrix),
                                platform='CUDA', api_version=1)
    return library


def _register():
    # FFI registration must happen exactly once, even with concurrent first compilations.
    with _registration_lock:
        return _library()


def available():
    """True when a CUDA device and the matrix FFI library are usable."""
    try:
        return jax.default_backend() == 'gpu' and _register() is not None
    except RuntimeError:
        return False


def cache_info():
    """Native plan metadata (uploads, registers, spills, workspace and scratch bytes)."""
    return json.loads(_register().asm_matrix_ffi_cache_info())


def clear_plan_cache():
    """Release cached CUDA modules after their outstanding work; executables stay valid."""
    _register().asm_matrix_ffi_clear_cache()


def symmetric_components(kernels):
    """Map of ``K`` symmetric 2x2 kernels stored as planes ``(xx, xy, yy)`` per kernel."""
    return tuple(((3*k, 3*k + 1), (3*k + 1, 3*k + 2)) for k in range(kernels))


def _normalize_components(components, planes):
    rows = tuple(tuple(tuple(-1 if e is None else int(e) for e in row) for row in kernel) for kernel in components)
    if not rows or len(rows) > _MAX_KERNELS:
        raise ValueError(f'components must describe 1 to {_MAX_KERNELS} kernels')
    m = len(rows[0])
    if not 1 <= m <= _MAX_FEATURES or any(len(k) != m or any(len(r) != m for r in k) for k in rows):
        raise ValueError(f'every kernel must be an M x M map with 1 <= M <= {_MAX_FEATURES}')
    if any(not -1 <= e < planes for k in rows for r in k for e in r):
        raise ValueError(f'component entries must be None or plane indices below {planes}')
    return rows


def _transposed(components):
    return tuple(tuple(zip(*kernel)) for kernel in components)


def _geometry(field_shape, canvas, input_offset, output_shape, output_offset):
    h, w = field_shape
    canvas = tuple(int(v) for v in canvas)
    input_offset = tuple(int(v) for v in input_offset)
    output_shape = tuple(int(v) for v in (field_shape if output_shape is None else output_shape))
    output_offset = tuple(int(v) for v in output_offset)
    if len(canvas) != 2 or len(input_offset) != 2 or len(output_shape) != 2 or len(output_offset) != 2:
        raise ValueError('canvas, offsets and output_shape must be two-dimensional')
    if min(canvas) < 1 or min(output_shape) < 1 or min(input_offset) < 0 or min(output_offset) < 0:
        raise ValueError('canvas and windows must be nonempty with nonnegative offsets')
    for (start, size), extent in zip(zip(input_offset + output_offset, (h, w) + output_shape), canvas*2):
        if start + size > extent:
            raise ValueError('input and output windows must lie inside the FFT canvas')
    return dict(canvas=canvas, input_offset=input_offset, output_shape=output_shape, output_offset=output_offset)


def _check(field, planes, reduce, components):
    rank = 6 if reduce else 5
    if field.ndim != rank or planes.ndim != 4:
        raise ValueError('Internal matrix convolution operands have unexpected ranks')
    if field.dtype != np.dtype('complex64') or planes.dtype != np.dtype('complex64'):
        raise TypeError('Matrix convolution requires complex64 fields and kernels; cast explicitly')
    k, m = len(components), len(components[0])
    if field.shape[rank-3] != m or (reduce and field.shape[2] != k):
        raise ValueError(f'field must have {m} components' + (f' and {k} kernels' if reduce else ''))
    if planes.shape[0] not in (1, field.shape[0]):
        raise ValueError('kernel planes must be shared or have the leading size of the field')


_expand_p = core.Primitive('pyvkfft_asm_matrix_expand')
_reduce_p = core.Primitive('pyvkfft_asm_matrix_reduce')
for _p in (_expand_p, _reduce_p):
    _p.def_impl(partial(dispatch.apply_primitive, _p))


def _abstract(field, planes, *, reduce, components, canvas, input_offset, output_shape, output_offset, transpose):
    _check(field, planes, reduce, components)
    if tuple(planes.shape[-2:]) != tuple(canvas):
        raise ValueError(f'kernel planes must have the canvas shape {tuple(canvas)}')
    _geometry(field.shape[-2:], canvas, input_offset, output_shape, output_offset)
    v, b, m = field.shape[0], field.shape[1], len(components[0])
    shape = (v, b, m) if reduce else (v, b, len(components), m)
    return field.update(shape=shape + tuple(output_shape))


_expand_p.def_abstract_eval(partial(_abstract, reduce=False))
_reduce_p.def_abstract_eval(partial(_abstract, reduce=True))


def _lower(ctx, field, planes, *, reduce, components, canvas, input_offset, output_shape, output_offset, transpose):
    library = _register()
    m, k = len(components[0]), len(components)
    (ch, cw), (iy, ix), (oy, ox) = canvas, input_offset, output_offset
    attributes = dict(reduce=np.int64(reduce), transpose=np.int64(transpose), canvas_h=np.int64(ch),
                      canvas_w=np.int64(cw), input_y=np.int64(iy), input_x=np.int64(ix), output_y=np.int64(oy),
                      output_x=np.int64(ox), components=np.asarray(components, np.int64).reshape(-1))

    # One workspace canvas per transformed plane: K*M for expand and for a fused reduce (K*M <= 8 input planes
    # transformed together), M for K accumulating single-kernel reduce passes.
    canvases = m if reduce and (k == 1 or k*m > _MAX_FUSED) else k*m

    def impl(x, h):
        v, b = x.shape[:2]
        out = jax.ShapeDtypeStruct(((v, b, m) if reduce else (v, b, k, m)) + tuple(output_shape), x.dtype)
        work = jax.ShapeDtypeStruct((canvases, ch, cw), x.dtype)
        return jax.ffi.ffi_call(library._target, (out, work), vmap_method='sequential')(x, h, **attributes)[0]

    return mlir.lower_fun(impl, multiple_results=False)(ctx, field, planes)


mlir.register_lowering(_expand_p, partial(_lower, reduce=False), platform='cuda')
mlir.register_lowering(_reduce_p, partial(_lower, reduce=True), platform='cuda')


def _jvp(primitive, primals, tangents, **params):
    field, planes = primals
    dfield, dplanes = tangents
    if not isinstance(dplanes, ad.Zero):
        raise NotImplementedError('Matrix convolution kernel derivatives are not implemented')
    result = primitive.bind(field, planes, **params)
    if isinstance(dfield, ad.Zero):
        return result, ad.Zero.from_primal_value(result)
    return result, primitive.bind(dfield, planes, **params)


def _transpose(dual, cotangent, field, planes, *, components, canvas, input_offset, output_shape, output_offset,
               transpose):
    if ad.is_undefined_primal(planes):
        raise NotImplementedError('Matrix convolution kernel derivatives are not implemented')
    if not ad.is_undefined_primal(field):
        raise TypeError('Matrix convolution transpose requires a linear field')
    if isinstance(cotangent, ad.Zero):
        return ad.Zero(field.aval), None
    # A = C_out F^-1 D(M) F E_in, so A^T = C_in F^-1 D_rev(M^T) F E_out (F is symmetric): the same engine with
    # the windows exchanged, frequencies reversed and the map transposed; expand and reduce are mutual transposes.
    result = dual.bind(cotangent, planes, components=_transposed(components), canvas=canvas,
                       input_offset=output_offset, output_shape=tuple(field.aval.shape[-2:]),
                       output_offset=input_offset, transpose=not transpose)
    return result, None


def _batch(primitive, operands, axes, **params):
    indices = [i for i, axis in enumerate(axes) if axis is not batching.not_mapped]
    mapped = tuple(jnp.moveaxis(operands[i], axes[i], 0) for i in indices)

    def one(values):
        args = list(operands)
        for i, value in zip(indices, values):
            args[i] = value
        return primitive.bind(*args, **params)

    return lax.map(one, mapped), 0


for _p, _dual in ((_expand_p, _reduce_p), (_reduce_p, _expand_p)):
    ad.primitive_jvps[_p] = partial(_jvp, _p)
    ad.primitive_transposes[_p] = partial(_transpose, _dual)
    batching.primitive_batchers[_p] = partial(_batch, _p)


def _prepare(field, kernels, trailing, components, input_offset, output_shape, output_offset):
    field, kernels = jnp.asarray(field), jnp.asarray(kernels)
    if field.dtype != np.dtype('complex64') or kernels.dtype != np.dtype('complex64'):
        raise TypeError('Matrix convolution requires complex64 fields and kernels; cast explicitly')
    if components is None:
        if kernels.ndim < 4 or kernels.shape[-3] != 3:
            raise ValueError('kernels must be [..., K, 3, CH, CW] symmetric (xx, xy, yy) without components')
        components = symmetric_components(kernels.shape[-4])
        kernels = kernels.reshape(kernels.shape[:-4] + (3*kernels.shape[-4],) + kernels.shape[-2:])
    if kernels.ndim < 3:
        raise ValueError('kernel planes must be [..., P, CH, CW]')
    components = _normalize_components(components, kernels.shape[-3])
    if field.ndim < trailing:
        raise ValueError(f'field needs at least {trailing} dimensions')
    lead = field.shape[:field.ndim - trailing]
    klead = kernels.shape[:-3]
    if klead not in ((), lead):
        raise ValueError(f'kernel leading dimensions {klead} must be empty or equal to the field\'s {lead}')
    v = math.prod(lead)
    field = field.reshape((v,) + field.shape[len(lead):])
    planes = kernels.reshape((v if klead else 1,) + kernels.shape[-3:])
    geometry = _geometry(field.shape[-2:], kernels.shape[-2:], input_offset, output_shape, output_offset)
    return field, planes, components, geometry, lead


def matrix_convolve_window(field, kernels, *, input_offset=(0, 0), output_shape=None, output_offset=(0, 0),
                           components=None):
    """``out[..., b, k, o] = window(IFFT2(sum_i M_k[o, i] FFT2(embed(field[..., b, i]))))``.

    ``field[..., B, M, h, w]`` complex64. Without ``components``: ``M = 2`` and ``kernels[..., K, 3, CH, CW]``
    symmetric (xx, xy, yy). With ``components`` (static ``K x M x M`` nested plane indices, ``None`` = zero):
    ``kernels[..., P, CH, CW]`` planes. Kernels are in natural FFT order on the canvas ``(CH, CW)``; their
    leading dimensions must be empty (shared) or equal to the field's. Returns ``[..., B, K, M, oh, ow]``.
    The input is embedded at ``input_offset``; the ``output_shape`` window (default: the field shape) at
    ``output_offset`` is returned. Field JVPs/VJPs are supported; at most 4 components and 16 kernels.
    """
    field, planes, components, geometry, lead = _prepare(field, kernels, 4, components, input_offset,
                                                         output_shape, output_offset)
    _check(field, planes, False, components)
    out = _expand_p.bind(field, planes, components=components, transpose=False, **geometry)
    return out.reshape(lead + out.shape[1:])


def matrix_reduce_window(field, kernels, *, input_offset=(0, 0), output_shape=None, output_offset=(0, 0),
                         components=None):
    """``out[..., b, o] = sum_k window(IFFT2(sum_i M_k[o, i] FFT2(embed(field[..., b, k, i]))))``.

    ``field[..., B, K, M, h, w]``; kernels as for :func:`matrix_convolve_window`. Returns ``[..., B, M, oh, ow]``.
    """
    field, planes, components, geometry, lead = _prepare(field, kernels, 5, components, input_offset,
                                                         output_shape, output_offset)
    _check(field, planes, True, components)
    out = _reduce_p.bind(field, planes, components=components, transpose=False, **geometry)
    return out.reshape(lead + out.shape[1:])
