"""Matrix-kernel VkFFT FFI (pyvkfft.jax_asm_matrix) against jnp FFT references: values, field VJPs and JVPs.

    PYVKFFT_ASM_MATRIX_FFI_LIBRARY=... python -m pytest pyvkfft/test/test_jax_asm_matrix.py
    PYVKFFT_ASM_MATRIX_FFI_LIBRARY=... python pyvkfft/test/test_jax_asm_matrix.py      # prints every error
"""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from pyvkfft import jax_asm_matrix as jm

TOLERANCE = 1e-5
pytestmark = pytest.mark.skipif(not jm.available(), reason='needs CUDA and libvkfft_asm_matrix_ffi.so')

# (V, B, h, w, canvas, input_offset, output_shape, output_offset, K)
GEOMETRIES = [
    (1, 1, 64, 64, (64, 64), (0, 0), (64, 64), (0, 0), 1),
    (2, 2, 64, 64, (64, 64), (0, 0), (64, 64), (0, 0), 4),
    (1, 2, 48, 40, (96, 80), (24, 20), (48, 40), (24, 20), 3),
    (2, 1, 30, 50, (64, 128), (5, 33), (40, 20), (11, 70), 2),
    (1, 2, 250, 250, (250, 250), (0, 0), (250, 250), (0, 0), 4),
    (1, 1, 250, 250, (500, 500), (0, 0), (70, 70), (90, 90), 1),
    (1, 2, 4000, 60, (4096, 64), (50, 2), (3000, 50), (500, 10), 4),     # two-upload (four-step) columns
]


def _embed(x, canvas, offset):
    (ch, cw), (y, x0), (h, w) = canvas, offset, x.shape[-2:]
    return jnp.pad(x, [(0, 0)]*(x.ndim-2) + [(y, ch-y-h), (x0, cw-x0-w)])


def _entries(components):
    return np.asarray([[[(-1 if e is None else e) for e in row] for row in kernel] for kernel in components])


def reference(field, planes, components, canvas, input_offset, output_shape, output_offset):
    """field [V, B, M, h, w], planes [Vk, P, CH, CW] -> [V, B, K, M, oh, ow] with jnp FFTs."""
    spectra = jnp.fft.fft2(_embed(field, canvas, input_offset))
    entries = _entries(components)
    out = []
    for kernel in entries:
        rows = []
        for row in kernel:
            acc = jnp.zeros_like(spectra[:, :, 0])
            for i, plane in enumerate(row):
                if plane >= 0:
                    acc = acc + planes[:, None, plane]*spectra[:, :, i]
            rows.append(jnp.fft.ifft2(acc))
        out.append(jnp.stack(rows, 2))
    (y, x), (oh, ow) = output_offset, output_shape
    return jnp.stack(out, 2)[..., y:y+oh, x:x+ow]


def reference_reduce(field, planes, components, canvas, input_offset, output_shape, output_offset):
    """field [V, B, K, M, h, w] -> sum_k [V, B, M, oh, ow]."""
    total = 0
    for k, kernel in enumerate(components):
        total = total + reference(field[:, :, k], planes, (kernel,), canvas, input_offset, output_shape,
                                  output_offset)[:, :, 0]
    return total


def _random(key, shape):
    re, im = jax.random.normal(key, (2,) + tuple(shape), jnp.float32)
    return (re + 1j*im).astype(jnp.complex64)


def _relative(a, b):
    a, b = np.asarray(a, np.complex128), np.asarray(b, np.complex128)
    return float(np.linalg.norm(a - b)/np.linalg.norm(b))


def _case(index, shared=False):
    v, b, h, w, canvas, iy, oshape, oy, k = GEOMETRIES[index]
    keys = jax.random.split(jax.random.PRNGKey(index), 3)
    field = _random(keys[0], (v, b, 2, h, w))
    kernels = _random(keys[1], ((1,) if shared else (v,)) + (k, 3) + canvas)
    geometry = dict(input_offset=iy, output_shape=oshape, output_offset=oy)
    return field, kernels, geometry, canvas, keys[2]


def check_values(index, shared=False):
    field, kernels, geometry, canvas, _ = _case(index, shared)
    k = kernels.shape[1]
    lead = kernels if not shared else kernels[0]
    out = jm.matrix_convolve_window(field, lead, **geometry)
    planes = kernels.reshape(kernels.shape[0], 3*k, *canvas)
    return _relative(out, reference(field, planes, jm.symmetric_components(k), canvas, geometry['input_offset'],
                                    geometry['output_shape'], geometry['output_offset']))


def check_gradients(index):
    field, kernels, geometry, canvas, key = _case(index)
    k = kernels.shape[1]
    planes = kernels.reshape(kernels.shape[0], 3*k, *canvas)
    components = jm.symmetric_components(k)
    args = (components, canvas, geometry['input_offset'], geometry['output_shape'], geometry['output_offset'])
    native = jax.jit(lambda x: jm.matrix_convolve_window(x, kernels, **geometry))
    exact = lambda x: reference(x, planes, *args)
    out, pullback = jax.vjp(native, field)
    cotangent = _random(key, out.shape)
    tangent = _random(jax.random.fold_in(key, 1), field.shape)
    vjp = _relative(pullback(cotangent)[0], jax.vjp(exact, field)[1](cotangent)[0])
    jvp = _relative(jax.jvp(native, (field,), (tangent,))[1], jax.jvp(exact, (field,), (tangent,))[1])
    return vjp, jvp


def check_reduce(index):
    field, kernels, geometry, canvas, key = _case(index)
    k = kernels.shape[1]
    planes = kernels.reshape(kernels.shape[0], 3*k, *canvas)
    stacked = jnp.broadcast_to(field[:, :, None], field.shape[:2] + (k,) + field.shape[2:]) * \
        (1 + jnp.arange(k, dtype=jnp.float32))[None, None, :, None, None, None]
    args = (jm.symmetric_components(k), canvas, geometry['input_offset'], geometry['output_shape'],
            geometry['output_offset'])
    native = jax.jit(lambda x: jm.matrix_reduce_window(x, kernels, **geometry))
    exact = lambda x: reference_reduce(x, planes, *args)
    out, pullback = jax.vjp(native, stacked)
    cotangent = _random(key, out.shape)
    return (_relative(out, exact(stacked)),
            _relative(pullback(cotangent)[0], jax.vjp(exact, stacked)[1](cotangent)[0]))


def check_components():
    """Diagonal and block-diagonal maps with zero (None) entries: M = 1, 2 and 4 features."""
    key = jax.random.PRNGKey(7)
    canvas, results = (64, 96), {}
    planes = _random(key, (2, 8) + canvas)
    diag = lambda p: ((p, None), (None, p))
    sym = lambda a: ((a, a + 1), (a + 1, a + 2))
    block = lambda a: tuple(tuple((sym(a)[o % 2][i % 2] if o // 2 == i // 2 else None) for i in range(4))
                            for o in range(4))
    for name, components, m in (('M1', (((0,),), ((3,),), ((7,),)), 1), ('M2_fpwpm', (diag(0), diag(1), sym(2), sym(5)), 2),
                                ('M4_block', (block(2), block(5), tuple(tuple(diag(0)[o % 2][i % 2] if o // 2 == i // 2
                                                                              else None for i in range(4))
                                                                        for o in range(4))), 4)):
        field = _random(jax.random.fold_in(key, m), (2, 1, m, 40, 64))
        geometry = dict(input_offset=(10, 20), output_shape=(50, 60), output_offset=(3, 30))
        native = jax.jit(lambda x: jm.matrix_convolve_window(x, planes, components=components, **geometry))
        exact = lambda x: reference(x, planes, components, canvas, geometry['input_offset'],
                                    geometry['output_shape'], geometry['output_offset'])
        out, pullback = jax.vjp(native, field)
        cotangent = _random(jax.random.fold_in(key, 10 + m), out.shape)
        results[name] = (_relative(out, exact(field)),
                         _relative(pullback(cotangent)[0], jax.vjp(exact, field)[1](cotangent)[0]))
    return results


@pytest.mark.parametrize('index', range(len(GEOMETRIES)))
def test_values(index):
    assert check_values(index) <= TOLERANCE


def test_shared_kernels():
    assert check_values(1, shared=True) <= TOLERANCE


@pytest.mark.parametrize('index', range(len(GEOMETRIES)))
def test_gradients(index):
    assert max(check_gradients(index)) <= TOLERANCE


@pytest.mark.parametrize('index', (1, 3, 6))
def test_reduce(index):
    assert max(check_reduce(index)) <= TOLERANCE


def test_components():
    assert max(max(v) for v in check_components().values()) <= TOLERANCE


def test_kernel_derivative_raises():
    field, kernels, geometry, _, _ = _case(0)
    with pytest.raises(NotImplementedError):
        jax.grad(lambda k: jnp.sum(jnp.abs(jm.matrix_convolve_window(field, k, **geometry))))(kernels)


def test_vmap_and_jit():
    field, kernels, geometry, _, _ = _case(1)
    batched = jnp.stack([field, 2*field])
    out = jax.jit(jax.vmap(lambda x: jm.matrix_convolve_window(x, kernels, **geometry)))(batched)
    single = jm.matrix_convolve_window(field, kernels, **geometry)
    assert _relative(out[1], 2*single) <= TOLERANCE


if __name__ == '__main__':
    for i in range(len(GEOMETRIES)):
        print('geometry', GEOMETRIES[i], 'value', f'{check_values(i):.2e}',
              'vjp/jvp', ' '.join(f'{e:.2e}' for e in check_gradients(i)), flush=True)
    print('shared kernels', f'{check_values(1, shared=True):.2e}')
    for i in (1, 3, 6):
        print('reduce', GEOMETRIES[i], 'value/vjp', ' '.join(f'{e:.2e}' for e in check_reduce(i)))
    for name, errors in check_components().items():
        print('components', name, 'value/vjp', ' '.join(f'{e:.2e}' for e in errors))
