"""Experimental differentiable slab FFT and compact-window ASM over a JAX mesh.

JAX owns collectives and all execution buffers. VkFFT runs only local stages.
The packed transfer is fixed; differentiation is supported with respect to the
complex field. Optional leading batch axes follow the supplied mesh mapping.
"""
import ctypes
from functools import lru_cache, partial
import os
import operator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core
from jax.interpreters import ad, mlir
from jax._src import dispatch
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


@lru_cache(None)
def _library():
    path = Path(os.environ.get('PYVKFFT_DISTRIBUTED_FFI_LIBRARY',
                               Path(__file__).with_name('libvkfft_distributed_ffi.so')))
    if not path.is_file():
        raise RuntimeError('Build examples/asm_distributed_build.py with --install, or set '
                           'PYVKFFT_DISTRIBUTED_FFI_LIBRARY to the isolated build')
    library = ctypes.CDLL(str(path.resolve()))
    library.distributed_ffi_abi_version.restype = ctypes.c_int
    if library.distributed_ffi_abi_version() != 2:
        raise RuntimeError('Unsupported distributed FFT ABI')
    jax.ffi.register_ffi_target('pyvkfft_distributed_v2',
                               jax.ffi.pycapsule(library.PyVkFFTDistributed), platform='CUDA')
    return library


def _stage(x, payload, scale, *, mode, width, active, origin, tile, inverse=0, groups=1, output_shape=None):
    _library()
    shape = x.shape if output_shape is None else output_shape
    n = 2*x.shape[0] if mode == 2 else (width if mode == 3 else 2*width)
    output = jax.ShapeDtypeStruct(shape, np.complex64)
    workspace = jax.ShapeDtypeStruct((2*n*tile,), np.complex64)
    attrs = dict(mode=mode, width=width, active=active, origin=origin, tile=tile, inverse=inverse, groups=groups)
    return jax.ffi.ffi_call('pyvkfft_distributed_v2', (output, workspace),
                          input_output_aliases={0: 0} if mode in (2, 3) else {},
                          vmap_method='sequential')(x, payload, scale,
                                                   **{k: np.int64(v) for k, v in attrs.items()})[0]


_fft = core.Primitive('vkfft_local_1d')
_fft.def_impl(partial(dispatch.apply_primitive, _fft))
_fft.def_abstract_eval(lambda x, **_: x)


def _fft_lower(ctx, x, *, inverse, tile):
    def body(a):
        batch = next(b for b in range(min(a.shape[0], tile), 0, -1) if a.shape[0] % b == 0)
        return _stage(a, jnp.zeros((1, 1), jnp.uint32), jnp.ones((), jnp.float32),
                      mode=3, width=a.shape[1], active=a.shape[1], origin=0,
                      tile=batch, inverse=int(inverse))
    return mlir.lower_fun(body, multiple_results=False)(ctx, x)


mlir.register_lowering(_fft, _fft_lower, platform='cuda')


def _linear_jvp(primitive, primals, tangents, **params):
    result = primitive.bind(*primals, **params)
    if any(not isinstance(t, ad.Zero) for t in tangents[1:]):
        raise TypeError('Packed transfer and scale are fixed parameters')
    tangent = (ad.Zero.from_primal_value(result) if isinstance(tangents[0], ad.Zero)
               else primitive.bind(tangents[0], *primals[1:], **params))
    return result, tangent


def _fft_transpose(g, x, **params):
    if isinstance(g, ad.Zero):
        return (ad.Zero(x.aval),)
    # JAX uses the bilinear complex transpose; both DFT matrices are symmetric.
    return (_fft.bind(g, **params),)


ad.primitive_jvps[_fft] = partial(_linear_jvp, _fft)
ad.primitive_transposes[_fft] = _fft_transpose


def _batch_spec(shape, batch_axes, spatial):
    if len(shape) != len(batch_axes) + 2:
        raise ValueError('One batch axis mapping is required per leading dimension')
    return P(*(name if size != 1 else None for name, size in zip(batch_axes, shape)), *spatial)


def _map_planes(function, field, *parameters):
    """Serial local planes; index broadcast parameters without expanding H."""
    batch = field.shape[:-2]
    if not batch:
        return function(field, *parameters)
    if int(np.prod(batch)) == 1:
        result = function(field.reshape(field.shape[-2:]),
                          *(a.reshape(a.shape[-2:]) for a in parameters))
        return result.reshape(batch + result.shape[-2:])
    def plane(index):
        coordinates = jnp.unravel_index(index, batch)
        args = [a[tuple(0 if n == 1 else i for n, i in zip(a.shape[:-2], coordinates))]
                for a in parameters]
        return function(field[coordinates], *args)
    result = lax.map(plane, jnp.arange(int(np.prod(batch))))
    return result.reshape(batch + result.shape[-2:])


def fft2_factory(mesh, *, axis_name='tp', inverse=False, backend='vkfft', tile=128, batch_axes=()):
    """Row-slab input -> column-slab spectrum (opposite layout for inverse).

    ``batch_axes`` maps each leading dimension to a mesh axis (or None).
    Singleton dimensions are replicated; local planes execute sequentially.
    Both spatial dimensions must be divisible by the mesh size. Each direction performs one
    all-to-all. This is the same slab decomposition as LOM's 1D-basis dfft.
    """
    tile = operator.index(tile)
    if backend not in ('jax', 'vkfft') or not 1<=tile<=65536 or axis_name not in mesh.shape:
        raise ValueError('Require jax/vkfft backend, a mesh axis and a tile in [1, 65536]')
    def local(a):
        def fft(x):
            if backend == 'jax':
                return jnp.fft.ifft(x, axis=-1) if inverse else jnp.fft.fft(x, axis=-1)
            return _fft.bind(x, inverse=inverse, tile=tile)
        if inverse:
            a = fft(a.T).T
            a = lax.all_to_all(a, axis_name, split_axis=0, concat_axis=1, tiled=True)
            return fft(a)
        a = fft(a)
        a = lax.all_to_all(a, axis_name, split_axis=1, concat_axis=0, tiled=True)
        return fft(a.T).T
    def transform(field):
        if field.ndim != len(batch_axes)+2 or field.dtype != jnp.complex64:
            raise ValueError('Expected a complex64 field with matching batch axes')
        if any(d<1 or d % mesh.shape[axis_name] for d in field.shape[-2:]):
            raise ValueError('Both FFT dimensions must be divisible by the mesh size')
        src = (None, axis_name) if inverse else (axis_name, None)
        dst = (axis_name, None) if inverse else (None, axis_name)
        return shard_map(lambda a: _map_planes(local, a), mesh=mesh,
                         in_specs=_batch_spec(field.shape, batch_axes, src),
                         out_specs=_batch_spec(field.shape, batch_axes, dst),
                         check_rep=False)(field)
    return transform


def _window_body(x, payload, scale, *, mesh, axis_name, origin, tile_rows, tile_columns,
                 reverse=False, backend='vkfft', packed_exchange=True, batch_axes=()):
    h, w = x.shape[-2:]
    th, tw = payload.shape[-2:]
    oy, ox = origin
    if reverse:
        payload = jnp.flip(payload, (-2, -1))
        oy, ox = (2*h-oy-th+1) % (2*h), (2*w-ox-tw+1) % (2*w)
    p = mesh.shape[axis_name]
    k = ((tw+p-1)//p)*p
    payload = jnp.pad(payload, ((0, 0),)*(payload.ndim-1)+((0, k-tw),))
    def local(a, packed, s):
        s = s.reshape(())
        rows = min(tile_rows, a.shape[0])
        columns = min(tile_columns, k//p)
        if backend == 'vkfft':
            a = _stage(a, packed, s, mode=0, width=w, active=tw, origin=ox, tile=rows,
                       groups=p if packed_exchange else 1, output_shape=(a.shape[0], k))
        else:
            a = jnp.fft.fft(jnp.pad(a, ((0, 0), (w//2, w-w//2))), axis=1)
            a = a[:, (jnp.arange(k)+ox) % (2*w)] * (jnp.arange(k)<tw)
            if packed_exchange:
                a = a.reshape(h//p, p, k//p).transpose(1, 0, 2)
        if packed_exchange:
            a = lax.all_to_all(a.reshape(p, h//p, k//p), axis_name,
                               split_axis=0, concat_axis=0, tiled=True).reshape(h, k//p)
        else:
            a = lax.all_to_all(a, axis_name, split_axis=1, concat_axis=0, tiled=True)
        if backend == 'vkfft':
            a = _stage(a, packed, s, mode=2, width=w, active=tw, origin=oy, tile=columns)
        else:
            re = (packed & np.uint32(65535)).astype(jnp.int32)
            im = (packed >> np.uint32(16)).astype(jnp.int32)
            transfer = (jnp.where(re>=32768, re-65536, re) +
                        1j*jnp.where(im>=32768, im-65536, im))*s
            a = jnp.fft.fft(jnp.pad(a, ((h//2, h-h//2), (0, 0))), axis=0)
            y = (jnp.arange(2*h)-oy) % (2*h)
            a *= transfer[jnp.minimum(y, th-1), :] * (y[:, None]<th)
            a = jnp.fft.ifft(a, axis=0)[h//2:h//2+h, :]
        if packed_exchange:
            a = lax.all_to_all(a.reshape(p, h//p, k//p), axis_name,
                               split_axis=0, concat_axis=0, tiled=True)
            if backend == 'jax':
                a = a.transpose(1, 0, 2)
            a = a.reshape(h//p, k)
        else:
            a = lax.all_to_all(a, axis_name, split_axis=0, concat_axis=1, tiled=True)
        if backend == 'vkfft':
            return _stage(a, packed, s, mode=1, width=w, active=tw, origin=ox, tile=rows,
                          groups=p if packed_exchange else 1, output_shape=(a.shape[0], w))
        spectrum = jnp.zeros((a.shape[0], 2*w), jnp.complex64)
        spectrum = spectrum.at[:, (jnp.arange(tw)+ox) % (2*w)].set(a[:, :tw])
        return jnp.fft.ifft(spectrum, axis=1)[:, w//2:w//2+w]
    row_spec = _batch_spec(x.shape, batch_axes, (axis_name, None))
    return shard_map(lambda a, b, c: _map_planes(local, a, b, c), mesh=mesh,
                     in_specs=(row_spec, _batch_spec(payload.shape, batch_axes, (None, axis_name)),
                               _batch_spec(scale.shape, batch_axes, (None, None))),
                     out_specs=row_spec, check_rep=False)(x, payload, scale)


def windowed_asm_factory(mesh, *, origin, axis_name='tp', tile_rows=128, tile_columns=64,
                         backend='vkfft', packed_exchange=True, batch_axes=()):
    """Compact exact zero-padded convolution using a cyclic SNORM16 window.

    Field rows and transfer columns are distributed. The transfer width need
    not divide the device count; inactive communication columns are zero.
    ``batch_axes`` maps leading dimensions to data-parallel mesh axes (or None).
    Field, transfer and scale batches broadcast; local planes execute sequentially.
    An outer jax.jit can donate the input field. Precision matches the existing
    packed-transfer implementation; no additional approximation is introduced.
    """
    tile_rows, tile_columns = operator.index(tile_rows), operator.index(tile_columns)
    origin = tuple(operator.index(v) for v in origin)
    if (backend not in ('jax', 'vkfft') or not 1<=min(tile_rows, tile_columns)
            or max(tile_rows, tile_columns)>65536 or axis_name not in mesh.shape or len(origin)!=2):
        raise ValueError('Invalid backend, mesh axis, tile size or origin')
    from jax.custom_transpose import custom_transpose
    from jax.custom_derivatives import SymbolicZero

    @partial(jax.custom_jvp, nondiff_argnums=(3,))
    def evaluate(field, payload, scale, reverse):
        return _window_body(field, payload, scale, mesh=mesh, axis_name=axis_name,
                            origin=tuple(origin), tile_rows=tile_rows, tile_columns=tile_columns,
                            reverse=reverse, backend=backend, packed_exchange=packed_exchange, batch_axes=batch_axes)

    def evaluate_jvp(reverse, primals, tangents):
        field, payload, scale = primals
        dx, dh, ds = tangents
        if not isinstance(dh, SymbolicZero) or not isinstance(ds, SymbolicZero):
            raise TypeError('Packed transfer and scale are fixed parameters')
        result = evaluate(field, payload, scale, reverse)
        if isinstance(dx, SymbolicZero):
            return result, jnp.zeros_like(result)

        @custom_transpose
        def linear(res, tangent):
            return evaluate(tangent, *res, reverse)

        @linear.def_transpose
        def transpose(res, cotangent):
            return evaluate(cotangent, *res, not reverse)

        aval = jax.core.get_aval(result).to_tangent_aval()
        return result, linear(aval, (payload, scale), dx)

    evaluate.defjvp(evaluate_jvp, symbolic_zeros=True)

    def propagate(field, payload, scale):
        rank = len(batch_axes)+2
        if field.ndim != rank or field.dtype != jnp.complex64 or payload.ndim != rank or payload.dtype != jnp.uint32:
            raise ValueError('Expected complex64 field and uint32 transfer with matching batch axes')
        h, w = field.shape[-2:]
        if (min(field.shape)<1 or min(payload.shape)<1 or h % mesh.shape[axis_name]
                or payload.shape[-2]>2*h or payload.shape[-1]>2*w):
            raise ValueError('Rows must be divisible by the mesh size and transfer must fit the padded field')
        if not (0<=origin[0]<2*h and 0<=origin[1]<2*w):
            raise ValueError('Invalid cyclic window origin')
        scale = jnp.asarray(scale, jnp.float32)
        if scale.ndim and (scale.ndim != rank or scale.shape[-2:] != (1, 1)):
            raise ValueError('Scale must be scalar or match batch rank with singleton spatial axes')
        scale = scale.reshape((1,)*rank) if not scale.ndim else scale
        batch = jnp.broadcast_shapes(field.shape[:-2], payload.shape[:-2], scale.shape[:-2])
        # Broadcast outside the custom linear rule: its transpose then reduces
        # cotangents over incident/design dimensions shared by several planes.
        field = jnp.broadcast_to(field, batch+(h, w))
        return evaluate(field, payload, scale, False)
    return propagate
