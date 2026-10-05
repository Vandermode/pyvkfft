"""Isolated, temporary ablations of the actual LOM cat training step.

No LOM source files are changed. Run each variant in a fresh process with one
selected GPU. All variants retain the configured optical model and objective;
no_rd omits only a zero-weight reported diagnostic, and no_shift specializes
the fixed zero sensor shift. Captures the original application's train_step
before its first warmup, then benchmarks its compiled computation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['baseline', 'no_rd', 'no_shift', 'both', 'slim', 'donate', 'scan50'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lom', type=Path, default=Path('/home/weik/code/LAFA/lom'))
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.lom)
    sys.path.insert(0, str(args.lom))
    import holo2d
    import apps.holography as application
    import functional.doe as doe
    import inspect
    import jax
    import jax.numpy as jnp
    import numpy as np
    from omegaconf import OmegaConf
    from scalax.sharding import MeshShardingHelper

    cfg = OmegaConf.load(args.lom/'conf/holo2d/config.yaml')
    cfg.log_cfg.no_wandb = True
    cfg.log_cfg.ckpt_dir = str(output/'setup')
    cfg.opt_cfg.target_path = 'datasets/pexels_selected/pexels-fran-arrotge-533902861-16491141.jpg'
    cfg.prop_cfg.propagation_backend = 'vkfft'
    cfg.prop_cfg.vkfft_grouped_batch = 32
    assert cfg.opt_cfg.loss_weight.rd == 0
    assert not cfg.sensor_cfg.random_sensor_shift and list(cfg.sensor_cfg.sensor_shift) == [0, 0]
    assert cfg.opt_cfg.randomize_start_iters > cfg.opt_cfg.num_iters
    (output/'setup').mkdir()
    holo2d.hydra_init = lambda *_: None
    if args.variant in ['no_rd', 'both', 'slim', 'donate', 'scan50']:
        application.rayleigh_distance_loss = lambda field, **_: jnp.zeros((), field.real.dtype)
    if args.variant in ['no_shift', 'both', 'slim', 'donate', 'scan50']:
        doe._apply_fractional_shift = lambda intensity, *_: intensity

    captured = {}

    class Captured(Exception):
        pass

    original_sjit = MeshShardingHelper.sjit

    def capture_sjit(self, function, *positional, **keywords):
        decorated = original_sjit(self, function, *positional, **keywords)
        if function.__qualname__ == 'optimize_hologram.<locals>.train_step':
            def capture(*operands):
                captured.update(mesh=self, raw=function, operands=operands)
                raise Captured()
            return capture
        return decorated

    MeshShardingHelper.sjit = capture_sjit
    try:
        holo2d.main.__wrapped__(cfg)
    except Captured:
        pass
    finally:
        MeshShardingHelper.sjit = original_sjit
    assert captured
    mesh, raw, operands = captured['mesh'], captured['raw'], captured['operands']
    initial, rest = operands[0], operands[1:]
    initial_hash = hashlib.sha256(np.asarray(initial.params['repr_params']).tobytes()).hexdigest()
    loss_fn = inspect.getclosurevars(raw).nonlocals['loss_fn']
    loss_gradient = mesh.sjit(jax.value_and_grad(lambda p, *r: loss_fn(p, *r)[0]))
    first_loss, first_gradient = jax.block_until_ready(loss_gradient(initial.params, *rest))
    np.save(output/'first_gradient.npy', np.asarray(first_gradient['repr_params']))

    function = raw
    if args.variant in ['slim', 'donate']:
        function = lambda *values: raw(*values)[:3]
    if args.variant == 'scan50':
        def function(state, *values):
            def body(state, _):
                result = raw(state, *values)
                return result[0], result[1:3]
            return jax.lax.scan(body, state, None, length=50)
    donation = args.variant == 'donate'
    lowered = mesh.sjit(function, donate_argnums=(0,) if donation else ()).lower(initial, *rest)
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter()-start
    (output/'optimized.hlo').write_text(compiled.as_text())
    (output/'stablehlo.mlir').write_text(str(lowered.compiler_ir()))

    def fresh_state():
        return jax.block_until_ready(jax.tree.map(lambda x: x.copy(), initial))

    state = fresh_state()
    start = time.perf_counter()
    for _ in range(3):
        result = jax.block_until_ready(compiled(state, *rest))
        state = result[0]
    warmup_seconds = time.perf_counter()-start
    times = []
    updates_per_call = 50 if args.variant == 'scan50' else 1
    for _ in range(3):
        state = fresh_state()
        start = time.perf_counter()
        for _ in range(300//updates_per_call):
            result = compiled(state, *rest)
            state = result[0]
        jax.block_until_ready(result)
        times.append(time.perf_counter()-start)
    np.save(output/'params_after_300.npy', np.asarray(state.params['repr_params']))
    final_loss = float(jax.block_until_ready(loss_fn(state.params, *rest)[0]))
    memory = compiled.memory_analysis()
    record = dict(variant=args.variant, jax_version=jax.__version__, gpu_kind=jax.devices()[0].device_kind,
                  gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), initial_design_sha256=initial_hash,
                  first_loss=float(first_loss), loss_after_300=final_loss,
                  compile_seconds=compile_seconds, warmup_three_calls_seconds=warmup_seconds,
                  seconds_per_300_steps=times, median_ms_per_step=float(np.median(times)/300*1000),
                  memory={name:getattr(memory, name) for name in ['argument_size_in_bytes',
                      'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes']},
                  source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [
                      Path(__file__), args.lom/'apps/holography.py', args.lom/'functional/doe.py',
                      args.lom/'functional/loss.py', args.lom/'functional/propagation.py']},
                  protocol='Same captured actual train_step; 3 warmed trials of 300 updates; '
                           'synchronize trial end; no checkpoint/metrics formatting overhead; '
                           'FP32 field and optimizer, identical initial design and physical objective. '
                           'Temporary process-local substitutions only; baseline is not modified.')
    (output/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    if args.trace:
        state = fresh_state()
        with jax.profiler.trace(output/'trace', create_perfetto_trace=True):
            for index in range(10 if updates_per_call == 1 else 2):
                with jax.profiler.StepTraceAnnotation('train', step_num=index):
                    result = compiled(state, *rest)
                    state = result[0]
            jax.block_until_ready(result)


if __name__ == '__main__':
    main()
