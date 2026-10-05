"""Differentiable compact ASM propagation through the typed CUDA XLA FFI.

``asm`` evaluates an analytic transfer inside the FFT kernels. ``convolve``
accepts a general transfer in natural FFT order (or an explicitly even quadrant).
Both zero-pad each spatial dimension to twice its size, then crop back at
``(height // 2, width // 2)``. Inputs are immutable complex64 JAX arrays.

JIT, sequential vmap/broadcast batches, JVPs and reverse-mode gradients are
supported. Field gradients use the same fused native engine. General transfer
gradients use JAX FFTs; analytic distance/wavelength gradients use native
derivative transfers. Analytic parameter derivatives through total order two
are supported, away from the hard evanescent/band-limit support boundaries.

Build examples/asm_ffi_build.py and set PYVKFFT_ASM_FFI_LIBRARY, or install the
result as pyvkfft/libvkfft_asm_ffi.so. No CuPy context or array is needed.
"""
import ctypes
from dataclasses import dataclass
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


_TARGET = 'pyvkfft_asm_v3'
_registration_lock = threading.Lock()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PackedTransferWindow:
    """Packed coefficients in a cyclic window, with exact zeros outside it.

    ``full_shape`` and ``origin`` are static two-dimensional FFT coordinates.
    Only ``payload`` is device data. Use with :func:`convolve_packed` and a
    positive ``tile_columns``; scale remains a separate fixed argument.
    """
    payload: jax.Array
    full_shape: tuple
    origin: tuple

    def tree_flatten(self):
        return (self.payload,), (self.full_shape, self.origin)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(children[0], *metadata)

    @property
    def shape(self):
        return self.payload.shape

    @property
    def dtype(self):
        return self.payload.dtype

    @property
    def nbytes(self):
        return self.payload.nbytes

    @property
    def sharding(self):
        return self.payload.sharding

    def devices(self):
        return self.payload.devices()


@lru_cache(maxsize=1)
def _library():
    path = Path(os.environ.get('PYVKFFT_ASM_FFI_LIBRARY',
                               Path(__file__).with_name('libvkfft_asm_ffi.so')))
    if not path.is_file():
        raise RuntimeError('Build examples/asm_ffi_build.py and set '
                           'PYVKFFT_ASM_FFI_LIBRARY to libvkfft_asm_ffi.so')
    library = ctypes.CDLL(str(path.resolve()))
    library.asm_ffi_abi_version.restype = ctypes.c_int
    if library.asm_ffi_abi_version() != 3:
        raise RuntimeError('Unsupported ASM FFI ABI')
    library.asm_ffi_cache_info.restype = ctypes.c_char_p
    library.asm_ffi_clear_cache.argtypes = []
    library.asm_ffi_clear_cache.restype = None
    jax.ffi.register_ffi_target(_TARGET, jax.ffi.pycapsule(library.PyVkFFTASM),
                                platform='CUDA', api_version=1)
    return library


@lru_cache(maxsize=1)
def _streamed_library():
    path = Path(os.environ.get('PYVKFFT_STREAMED_FFI_LIBRARY',
                               Path(__file__).with_name('libvkfft_streamed_ffi.so')))
    if not path.is_file():
        raise RuntimeError('Build examples/asm_streamed_ffi_build.py --install for tiled propagation')
    library = ctypes.CDLL(str(path.resolve()))
    library.streamed_ffi_abi_version.restype = ctypes.c_int
    version = library.streamed_ffi_abi_version()
    if version not in (1, 2, 3):
        raise RuntimeError('Unsupported streamed ASM FFI ABI')
    library.streamed_ffi_cache_info.restype = ctypes.c_char_p
    library.streamed_ffi_clear_cache.argtypes = []
    library.streamed_ffi_clear_cache.restype = None
    library._streamed_target = f'pyvkfft_streamed_v{version}'
    library._streamed_aliases = {0: 0} if version >= 2 else None
    library._streamed_windowed = version >= 3
    jax.ffi.register_ffi_target(library._streamed_target, jax.ffi.pycapsule(library.PyVkFFTStreamed),
                                platform='CUDA', api_version=1)
    return library


def streamed_cache_info():
    """Return tiled plans and extra per-execution XLA scratch requirements."""
    with _registration_lock:
        return json.loads(_streamed_library().streamed_ffi_cache_info())


def clear_streamed_plan_cache():
    """Synchronize and unload tiled modules; compiled executables remain valid."""
    with _registration_lock:
        _streamed_library().streamed_ffi_clear_cache()


def _register():
    # lru_cache permits simultaneous first calls; FFI registration must happen
    # exactly once even when independent executables compile on several threads.
    with _registration_lock:
        return _library()


def cache_info():
    """Return native plan metadata, including extra XLA scratch requirements."""
    return json.loads(_register().asm_ffi_cache_info())


def clear_plan_cache():
    """Release cached CUDA modules after waiting for their outstanding work.

    Existing JIT executables remain valid and rebuild plans on their next call.
    Ordinary execution does not synchronize the device.
    """
    _register().asm_ffi_clear_cache()


def _check_field(field):
    if field.ndim < 2 or min(field.shape[-2:]) < 1:
        raise ValueError('field must have two nonempty spatial dimensions')
    if field.dtype != np.dtype('complex64'):
        raise TypeError('ASM FFI requires complex64 fields; cast explicitly')
    if math.prod(field.shape[-2:]) > np.iinfo(np.int64).max // 32:
        raise ValueError('Padded field byte count exceeds signed 64-bit indexing')


def _options(pixel_pitch, bandlimit, grouped_batch):
    if len(pixel_pitch) != 2 or not all(math.isfinite(v) and v > 0 for v in pixel_pitch):
        raise ValueError('pixel_pitch must contain two positive finite values in metres')
    if bandlimit not in ('none', 'rectangular'):
        raise ValueError('bandlimit must be none or rectangular')
    if isinstance(grouped_batch, bool) or not isinstance(grouped_batch, (int, np.integer)) or not 0 <= grouped_batch <= 65536:
        raise ValueError('grouped_batch must be an integer between 0 and 65536')
    return dict(dy=float(pixel_pitch[0]), dx=float(pixel_pitch[1]),
                bandlimit=int(bandlimit == 'rectangular'), grouped=int(grouped_batch))


def _ffi_call(field, transfer, z, wavelength, **attributes):
    output = jax.ShapeDtypeStruct(field.shape, field.dtype)
    work = jax.ShapeDtypeStruct(tuple(2*n for n in field.shape), field.dtype)
    attributes = {k: np.float64(v) if k in ('dy', 'dx') else np.int64(v)
                  for k, v in attributes.items()}
    return jax.ffi.ffi_call(_TARGET, (output, work), vmap_method='sequential')(
        field, transfer, z, wavelength, **attributes)[0]


def _map_broadcast(function, operands, spatial_ranks):
    """Map broadcast leading axes without materializing repeated full H planes."""
    leading = [x.shape[:-rank] if rank else x.shape for x, rank in zip(operands, spatial_ranks)]
    shape = np.broadcast_shapes(*leading)
    if not shape:
        return function(*operands)

    def one(flat_index):
        coords = jnp.unravel_index(flat_index, shape)
        values = []
        for value, lead in zip(operands, leading):
            for index, size in zip(coords[len(shape)-len(lead):], lead):
                value = lax.dynamic_index_in_dim(value, 0 if size == 1 else index,
                                                 axis=0, keepdims=False)
            values.append(value)
        return function(*values)

    result = lax.map(one, jnp.arange(math.prod(shape)))
    return result.reshape((*shape, *result.shape[1:]))


def _batch(primitive, operands, axes, **params):
    indices = [i for i, axis in enumerate(axes) if axis is not batching.not_mapped]
    mapped = tuple(jnp.moveaxis(operands[i], axes[i], 0) for i in indices)

    def one(values):
        args = list(operands)
        for i, value in zip(indices, values):
            args[i] = value
        return primitive.bind(*args, **params)

    return lax.map(one, mapped), 0


def _pad(field):
    return jnp.pad(field, tuple((n//2, n-n//2) for n in field.shape))


def _reverse_frequencies(value):
    return jnp.roll(jnp.flip(value, axis=(-2, -1)), (1, 1), axis=(-2, -1))


def _fold_quadrant(value):
    h, w = (n//2 for n in value.shape)
    rows = jnp.concatenate((value[:1], value[1:h] + value[-1:h:-1], value[h:h+1]), axis=0)
    return jnp.concatenate((rows[:, :1], rows[:, 1:w] + rows[:, -1:w:-1],
                            rows[:, w:w+1]), axis=1)


_convolve_p = core.Primitive('pyvkfft_asm_convolve')
_convolve_p.def_impl(partial(dispatch.apply_primitive, _convolve_p))


def _convolve_abstract(field, transfer, *, layout, transpose, grouped):
    _check_field(field)
    if field.ndim != 2 or transfer.ndim != 2:
        raise ValueError('Internal ASM convolution operands must be two-dimensional')
    expected = tuple(2*n if layout == 'fft' else n+1 for n in field.shape)
    if transfer.dtype != field.dtype or transfer.shape != expected:
        raise ValueError(f'transfer must be complex64 with shape {expected}')
    return field


_convolve_p.def_abstract_eval(_convolve_abstract)


def _convolve_lower(ctx, field, transfer, *, layout, transpose, grouped):
    _register()

    def impl(x, h):
        return _ffi_call(x, h, jnp.zeros((), jnp.float32), jnp.ones((), jnp.float32),
                         dy=1., dx=1., bandlimit=0, grouped=grouped,
                         transfer_mode=1 if layout == 'fft' else 2,
                         dz_order=0, dw_order=0, transpose=int(transpose))

    return mlir.lower_fun(impl, multiple_results=False)(ctx, field, transfer)


mlir.register_lowering(_convolve_p, _convolve_lower, platform='cuda')


def _convolve_jvp(primals, tangents, **params):
    field, transfer = primals
    dfield, dtransfer = tangents
    result = _convolve_p.bind(*primals, **params)
    terms = []
    if not isinstance(dfield, ad.Zero):
        terms.append(_convolve_p.bind(dfield, transfer, **params))
    if not isinstance(dtransfer, ad.Zero):
        terms.append(_convolve_p.bind(field, dtransfer, **params))
    tangent = sum(terms[1:], terms[0]) if terms else ad.Zero.from_primal_value(result)
    return result, tangent


def _convolve_transpose(cotangent, field, transfer, **params):
    field_missing, transfer_missing = ad.is_undefined_primal(field), ad.is_undefined_primal(transfer)
    if isinstance(cotangent, ad.Zero):
        return (ad.Zero(field.aval) if field_missing else None,
                ad.Zero(transfer.aval) if transfer_missing else None)
    if field_missing and not transfer_missing:
        options = dict(params, transpose=not params['transpose'])
        return _convolve_p.bind(cotangent, transfer, **options), None
    if transfer_missing and not field_missing:
        # For JAX's complex bilinear cotangents:
        # dH = FFT(pad(x)) * IFFT(pad(g)), with no conjugation.
        padded = _pad(cotangent)
        # Compute the scale as a Python float: some CUDA FFT bindings form a
        # signed-int32 product when normalizing very large inverse planes.
        inverse = jnp.conj(jnp.fft.fft2(jnp.conj(padded))) * np.float32(1/math.prod(padded.shape))
        gradient = jnp.fft.fft2(_pad(field)) * inverse
        if params['transpose']:
            gradient = _reverse_frequencies(gradient)
        if params['layout'] == 'quadrant':
            gradient = _fold_quadrant(gradient)
        return None, gradient
    raise TypeError('Convolution transpose requires exactly one linear argument')


ad.primitive_jvps[_convolve_p] = _convolve_jvp
ad.primitive_transposes[_convolve_p] = _convolve_transpose
batching.primitive_batchers[_convolve_p] = partial(_batch, _convolve_p)


def convolve(field, transfer, *, transfer_layout='fft', grouped_batch=0):
    """Pad/FFT/multiply/IFFT/crop with a general differentiable transfer.

    ``field[..., h, w]`` and ``transfer[..., 2*h, 2*w]`` broadcast on leading
    axes. ``transfer_layout='quadrant'`` accepts ``[..., h+1, w+1]`` and reflects
    it across both frequency axes; use it only for an axis-even transfer.
    General H need not be radial, even, Hermitian, or unit magnitude.
    """
    field, transfer = jnp.asarray(field), jnp.asarray(transfer)
    _check_field(field)
    if transfer_layout not in ('fft', 'quadrant'):
        raise ValueError('transfer_layout must be fft or quadrant')
    expected = tuple(2*n if transfer_layout == 'fft' else n+1 for n in field.shape[-2:])
    if transfer.ndim < 2 or transfer.shape[-2:] != expected or transfer.dtype != field.dtype:
        raise ValueError(f'transfer must be complex64 with spatial shape {expected}')
    options = _options((1., 1.), 'none', grouped_batch)
    function = partial(_convolve_p.bind, layout=transfer_layout, transpose=False,
                       grouped=options['grouped'])
    return _map_broadcast(function, (field, transfer), (2, 2))


_packed_p = core.Primitive('pyvkfft_asm_packed')
_packed_p.def_impl(partial(dispatch.apply_primitive, _packed_p))


def _packed_abstract(field, payload, scale, *, transpose, grouped, tile_columns=0, tile_rows=0,
                     transfer_origin=None, transfer_layout='fft'):
    _check_field(field)
    expected = tuple(n+1 if transfer_layout == 'quadrant' else 2*n for n in field.shape)
    valid_shape = (payload.shape == expected if transfer_origin is None else
                   payload.ndim == 2 and all(1 <= s <= n for s, n in zip(payload.shape, expected)))
    if field.ndim != 2 or not valid_shape or payload.dtype != np.dtype('uint32'):
        raise ValueError(f'packed transfer must be uint32 within FFT shape {expected}')
    if scale.shape or scale.dtype != np.dtype('float32'):
        raise ValueError('Internal packed scale must be a float32 scalar')
    return field


_packed_p.def_abstract_eval(_packed_abstract)


def _packed_lower(ctx, field, payload, scale, *, transpose, grouped, tile_columns=0, tile_rows=0,
                  transfer_origin=None, transfer_layout='fft'):
    if tile_columns:
        with _registration_lock:
            library = _streamed_library()
        if (transfer_origin is not None or transfer_layout != 'fft') and not library._streamed_windowed:
            raise RuntimeError('Windowed/quadrant transfers require streamed FFI ABI 3; rebuild the library')
        def tiled(x, h, scale):
            height, width = x.shape
            active = 2*width if transfer_layout == 'quadrant' else h.shape[1]
            b, r = min(tile_columns, active), min(tile_rows, height)
            elements = height*max(0, active-width) + 2*max(r*2*width, b*2*height)
            output = jax.ShapeDtypeStruct(x.shape, x.dtype)
            work = jax.ShapeDtypeStruct((elements,), x.dtype)
            attributes = dict(columns=np.int64(b), rows=np.int64(r), reverse=np.int64(transpose))
            if library._streamed_windowed:
                oy, ox = (0, 0) if transfer_origin is None else transfer_origin
                attributes.update(origin_y=np.int64(oy), origin_x=np.int64(ox),
                                  layout=np.int64(transfer_layout == 'quadrant'))
            return jax.ffi.ffi_call(library._streamed_target, (output, work), vmap_method='sequential',
                                    input_output_aliases=library._streamed_aliases)(
                x, h, scale, **attributes)[0]
        return mlir.lower_fun(tiled, multiple_results=False)(ctx, field, payload, scale)
    _register()

    def impl(x, h, s):
        return _ffi_call(x, h, s, jnp.ones((), jnp.float32),
                         dy=1., dx=1., bandlimit=0, grouped=grouped,
                         transfer_mode=3, dz_order=0, dw_order=0,
                         transpose=int(transpose))

    return mlir.lower_fun(impl, multiple_results=False)(ctx, field, payload, scale)


mlir.register_lowering(_packed_p, _packed_lower, platform='cuda')


def _packed_jvp(primals, tangents, **params):
    field, payload, scale = primals
    dfield, dpayload, dscale = tangents
    if not isinstance(dpayload, ad.Zero) or not isinstance(dscale, ad.Zero):
        raise TypeError('Packed transfer payload and scale are fixed parameters')
    result = _packed_p.bind(*primals, **params)
    tangent = (ad.Zero.from_primal_value(result) if isinstance(dfield, ad.Zero)
               else _packed_p.bind(dfield, payload, scale, **params))
    return result, tangent


def _packed_transpose(cotangent, field, payload, scale, **params):
    if ad.is_undefined_primal(payload) or ad.is_undefined_primal(scale):
        raise TypeError('Packed transfer payload and scale must be known for transpose')
    if not ad.is_undefined_primal(field):
        return None, None, None
    if isinstance(cotangent, ad.Zero):
        return ad.Zero(field.aval), None, None
    options = dict(params, transpose=not params['transpose'])
    return _packed_p.bind(cotangent, payload, scale, **options), None, None


ad.primitive_jvps[_packed_p] = _packed_jvp
ad.primitive_transposes[_packed_p] = _packed_transpose
batching.primitive_batchers[_packed_p] = partial(_batch, _packed_p)


def convolve_packed(field, payload, scale=None, *, grouped_batch=0, tile_columns=0, tile_rows=None,
                    transfer_origin=None, transfer_layout='fft'):
    """Propagate using a fixed packed SNORM16 transfer, decoded inside VkFFT.

    ``payload[..., 2*h, 2*w]`` is uint32 in natural FFT order: signed int16
    real in the low bits, imaginary in the high bits. Both components multiply
    ``scale`` (default float32 1/32767 for ASM). For scaled RSC pass its float32
    per-plane scale with singleton spatial axes, ``[..., 1, 1]``. Leading
    dimensions broadcast without expanding the transfer to complex64.

    ``tile_columns=1024`` selects the capacity-oriented tiled schedule; zero
    retains the fused full-workspace path. ``tile_rows`` defaults to a matching
    tile allocation. Tiled workspace is one compact complex plane plus tiles;
    both forward and field gradients use it. It requires the streamed FFI build.
    Streamed ABI 2 lets XLA reuse a consumed field buffer. Caller inputs remain
    immutable unless explicitly donated to an enclosing JIT function.

    ``transfer_origin=(y0, x0)`` supplies a smaller cyclic frequency window:
    payload[y, x] represents H[(y+y0) % (2*h), (x+x0) % (2*w)], and H is
    exactly zero outside that window. This requires nonzero ``tile_columns``
    and streamed ABI 3. Zero columns are skipped; the extra spectrum shrinks
    to ``h*max(0, payload_width-w)`` complex elements and disappears when the
    window is at most half the FFT width. No coefficients are approximated.
    Omitting the origin retains the full-table shape requirement.

    ``transfer_layout='quadrant'`` instead accepts ``[..., h+1, w+1]`` and
    reflects both frequency axes, including Nyquist edges. Use this only for
    an axis-even transfer. It reduces H storage approximately fourfold with
    the same complex64 FFTs, requires tiled ABI 3, and excludes an origin.

    Field JVPs, reverse-mode and higher field derivatives are supported.
    Payload and scale are fixed, stop-gradient parameters, matching LOM's
    quantized-transfer semantics. Use ``convolve`` to differentiate H itself.
    """
    field = jnp.asarray(field)
    _check_field(field)
    if transfer_layout not in ('fft', 'quadrant'):
        raise ValueError('transfer_layout must be fft or quadrant')
    expected = tuple(2*n for n in field.shape[-2:])
    if isinstance(payload, PackedTransferWindow):
        if tuple(payload.full_shape) != expected:
            raise ValueError(f'Window full_shape must match FFT shape {expected}')
        if transfer_origin is not None:
            raise ValueError('A PackedTransferWindow already specifies transfer_origin')
        transfer_origin, payload = payload.origin, payload.payload
    payload = jnp.asarray(payload)
    if transfer_layout == 'quadrant':
        if transfer_origin is not None or not tile_columns:
            raise ValueError('Quadrant packed transfer requires tiled execution and no transfer_origin')
        expected = tuple(n+1 for n in field.shape[-2:])
    if transfer_origin is not None:
        if (not isinstance(transfer_origin, (tuple, list)) or len(transfer_origin) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in transfer_origin)):
            raise ValueError('transfer_origin must contain two integer frequency indices')
        transfer_origin = tuple(int(v) % n for v, n in zip(transfer_origin, expected))
        if not tile_columns:
            raise ValueError('transfer_origin requires nonzero tile_columns')
    valid_shape = (payload.shape[-2:] == expected if transfer_origin is None else
                   all(1 <= s <= n for s, n in zip(payload.shape[-2:], expected)))
    if payload.ndim < 2 or not valid_shape or payload.dtype != np.dtype('uint32'):
        raise ValueError(f'packed transfer must be uint32 with spatial shape {expected}, or a window with transfer_origin')
    scale = jnp.asarray(np.float32(1/32767) if scale is None else scale)
    if scale.dtype != np.dtype('float32'):
        raise TypeError('Packed scale must be float32; cast explicitly')
    if scale.ndim:
        if scale.ndim < 2 or scale.shape[-2:] != (1, 1):
            raise ValueError('Packed scale must be scalar or have singleton spatial axes')
        scale = scale[..., 0, 0]
    options = _options((1., 1.), 'none', grouped_batch)
    if (isinstance(tile_columns, bool) or not isinstance(tile_columns, (int, np.integer))
            or not 0 <= tile_columns <= 65536):
        raise ValueError('tile_columns must be an integer between 0 and 65536')
    if tile_rows is not None and (isinstance(tile_rows, bool) or
            not isinstance(tile_rows, (int, np.integer)) or not 1 <= tile_rows <= 65536):
        raise ValueError('tile_rows must be an integer between 1 and 65536')
    if tile_columns and grouped_batch:
        raise ValueError('grouped_batch applies to the full-workspace path only')
    if not tile_columns and tile_rows is not None:
        raise ValueError('tile_rows requires nonzero tile_columns')
    height, width = field.shape[-2:]
    active = 2*width if transfer_layout == 'quadrant' else payload.shape[-1]
    rows = (min(65536, max(1, min(tile_columns, active)*height//width))
            if tile_rows is None else int(tile_rows))
    function = partial(_packed_p.bind, transpose=False, grouped=options['grouped'],
                       tile_columns=int(tile_columns), tile_rows=rows, transfer_origin=transfer_origin,
                       transfer_layout=transfer_layout)
    return _map_broadcast(function, (field, lax.stop_gradient(payload), lax.stop_gradient(scale)),
                          (2, 2, 0))


_asm_p = core.Primitive('pyvkfft_asm_analytic')
_asm_p.def_impl(partial(dispatch.apply_primitive, _asm_p))


def _asm_abstract(field, z, wavelength, *, dz_order, dw_order, **params):
    _check_field(field)
    if field.ndim != 2 or z.shape or wavelength.shape:
        raise ValueError('Internal analytic ASM expects a two-dimensional field and scalar parameters')
    if z.dtype != wavelength.dtype or z.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise TypeError('Optical parameters must have matching float32 or float64 dtype')
    if dz_order + dw_order > 2:
        raise NotImplementedError('Analytic optical-parameter derivatives above total order two are not supported')
    return field


_asm_p.def_abstract_eval(_asm_abstract)


def _asm_lower(ctx, field, z, wavelength, **params):
    _register()

    def impl(x, distance, wave):
        return _ffi_call(x, jnp.zeros((0,), jnp.complex64), distance, wave,
                         transfer_mode=0, transpose=0, **params)

    return mlir.lower_fun(impl, multiple_results=False)(ctx, field, z, wavelength)


mlir.register_lowering(_asm_p, _asm_lower, platform='cuda')


def _asm_jvp(primals, tangents, **params):
    field, z, wavelength = primals
    dfield, dz, dw = tangents
    result = _asm_p.bind(*primals, **params)
    terms = []
    if not isinstance(dfield, ad.Zero):
        terms.append(_asm_p.bind(dfield, z, wavelength, **params))
    for tangent, order in ((dz, 'dz_order'), (dw, 'dw_order')):
        if not isinstance(tangent, ad.Zero):
            derivative = _asm_p.bind(*primals, **dict(params, **{order: params[order]+1}))
            # Keep the scalar multiplication in JAX so its transpose performs
            # the reduction (and the real projection for real parameters).
            terms.append((derivative * tangent).astype(field.dtype))
            # Factor H = exp(i*k0*z) * exp(i*(k-k0)*z) for differentiation
            # only. The primal still uses the original full phase. Computing
            # the carrier term from the same output avoids an extra FFT's
            # roundoff multiplied by the very large optical wavenumber.
            coefficient = 2*jnp.pi/wavelength if order == 'dz_order' else -2*jnp.pi*z/wavelength**2
            carrier_result = result.astype(jnp.result_type(field.dtype, wavelength.dtype))
            # Reduce the complex inner product before multiplying by the
            # carrier frequency in the transpose (parentheses matter for AD).
            terms.append(((1j*carrier_result)*(coefficient*tangent)).astype(field.dtype))
    return result, sum(terms[1:], terms[0]) if terms else ad.Zero.from_primal_value(result)


def _asm_transpose(cotangent, field, z, wavelength, **params):
    if not ad.is_undefined_primal(field) or ad.is_undefined_primal(z) or ad.is_undefined_primal(wavelength):
        raise TypeError('The linear analytic ASM transpose is with respect to the field')
    if isinstance(cotangent, ad.Zero):
        return ad.Zero(field.aval), None, None
    # Analytic H and its parameter derivatives are even in both axes. Hence
    # the cropped convolution matrix is symmetric under JAX's transpose rule.
    return _asm_p.bind(cotangent, z, wavelength, **params), None, None


ad.primitive_jvps[_asm_p] = _asm_jvp
ad.primitive_transposes[_asm_p] = _asm_transpose
batching.primitive_batchers[_asm_p] = partial(_batch, _asm_p)


def asm(field, z, wavelength, *, pixel_pitch, bandlimit='none', grouped_batch=0):
    """Analytic scalar ASM with fused H and differentiable distance/wavelength.

    Pitch is ``(dy, dx)`` in metres. Distance and wavelength broadcast with
    leading field axes. Enable JAX x64 and supply float64 optical parameters
    when high phase accuracy is needed; the field/output remain complex64.
    Evanescent modes are discarded. Hard support-mask derivatives are zero
    except at discontinuities, where an ordinary derivative is not defined.
    """
    field = jnp.asarray(field)
    _check_field(field)
    for name, value in (('z', z), ('wavelength', wavelength)):
        if isinstance(value, (float, int, np.number)):
            if not math.isfinite(value) or (name == 'wavelength' and value <= 0):
                raise ValueError('z must be finite and wavelength finite and positive')
    # Preserve Python-double precision when x64 is enabled. An explicit
    # float32 array still selects float32, following JAX's weak-type rules.
    dtype = jnp.result_type(z, wavelength, 0.)
    z, wavelength = jnp.asarray(z, dtype), jnp.asarray(wavelength, dtype)
    if dtype not in (jnp.dtype('float32'), jnp.dtype('float64')):
        raise TypeError('Optical parameters must be real float32 or float64')
    options = _options(pixel_pitch, bandlimit, grouped_batch)
    function = partial(_asm_p.bind, dz_order=0, dw_order=0, **options)
    return _map_broadcast(function, (field, z, wavelength), (2, 0, 0))
