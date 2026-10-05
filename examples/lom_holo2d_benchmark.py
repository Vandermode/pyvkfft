"""Time the unmodified LOM cat holo2d application with JAX and VkFFT.

Run in LOM's Python environment, selecting one idle GPU by UUID. Each backend
runs in a fresh process on that same GPU, retaining the normal 3000-step
training, logging, checkpointing, and final evaluation. Instrumentation wraps
the application entry points; it does not replace the training computation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def worker(args):
    process_start = time.perf_counter()
    os.chdir(args.lom)
    sys.path.insert(0, str(args.lom))
    # Importing holo2d applies the application's allocator/XLA defaults before
    # device initialization, exactly as its ordinary CLI does.
    import holo2d
    import apps.holography as application
    import jax
    import jaxlib
    import numpy as np

    output = args.output
    record = dict(backend=args.worker, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
                  gpu_kind=jax.devices()[0].device_kind,
                  jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
                  environment={name: os.environ.get(name) for name in (
                      'XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_PYTHON_CLIENT_PREALLOCATE',
                      'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_FLAGS')},
                  progress=[], completed=False)
    original_range = application.trange

    class TimedProgress:
        def __init__(self, *positional, **keywords):
            self.progress = original_range(*positional, **keywords)

        def __getattr__(self, name):
            return getattr(self.progress, name)

        def __iter__(self):
            start = time.perf_counter()
            record['loop_start_since_process_s'] = start-process_start
            completed = 0
            for iteration in self.progress:
                yield iteration
                completed += 1
                if completed % 300 == 0:
                    record['progress'].append(dict(steps=completed, seconds=time.perf_counter()-start))
            # The application's final scalar read and checkpoint wait synchronize
            # its last update. Include those in the training-loop wall clock.
            record['training_loop_seconds'] = time.perf_counter()-start
            record['completed_steps'] = completed

    application.trange = TimedProgress
    original_optimize = holo2d.optimize_hologram

    def timed_optimize(*positional, **keywords):
        initial = np.asarray(keywords['params_init'])
        record['initial_design_sha256'] = hashlib.sha256(initial.tobytes()).hexdigest()
        record['design_shape'] = list(initial.shape)
        target = np.asarray(positional[1])
        record['target_array_sha256'] = hashlib.sha256(target.tobytes()).hexdigest()
        start = time.perf_counter()
        state, manager = original_optimize(*positional, **keywords)
        jax.block_until_ready(state)
        record['optimization_call_seconds'] = time.perf_counter()-start
        record['setup_compile_warmup_seconds'] = (
            record['optimization_call_seconds']-record['training_loop_seconds'])
        record['optimizer_final_step'] = int(state.step)
        record['training_memory_stats'] = jax.devices()[0].memory_stats()
        assert record['optimizer_final_step'] == args.steps
        final = np.asarray(state.params['repr_params'])
        record['final_design_sha256'] = hashlib.sha256(final.tobytes()).hexdigest()
        np.save(output/'final_design.npy', final)
        write_json(output/'timing.json', record)
        return state, manager

    holo2d.optimize_hologram = timed_optimize
    original_evaluate = holo2d.evaluate_hologram

    def timed_evaluate(*positional, **keywords):
        start = time.perf_counter()
        result = jax.block_until_ready(original_evaluate(*positional, **keywords))
        record['evaluation_seconds'] = time.perf_counter()-start
        return result

    holo2d.evaluate_hologram = timed_evaluate
    overrides = [
        'exp_name=cat_3p6mm_2um_3000_'+args.worker,
        'hydra.run.dir='+str(output),
        'log_cfg.ckpt_dir='+str(output/'ckpt'),
        'log_cfg.wandb_dir='+str(output),
        'log_cfg.no_wandb=true',
        '++log_cfg.save_holoimgs_npz=true',
        'opt_cfg.target_path=datasets/pexels_selected/pexels-fran-arrotge-533902861-16491141.jpg',
        'opt_cfg.num_iters='+str(args.steps),
        '++prop_cfg.propagation_backend='+args.worker,
        '++prop_cfg.vkfft_grouped_batch=32',
    ]
    record['overrides'] = overrides
    sys.argv = [str(args.lom/'holo2d.py'), *overrides]
    holo2d.main()
    jax.effects_barrier()
    record['worker_seconds'] = time.perf_counter()-process_start
    record['completed'] = True
    if args.worker == 'vkfft':
        from pyvkfft.jax_asm import cache_info
        record['native_plans'] = cache_info()
    write_json(output/'timing.json', record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lom', type=Path, default=Path('/home/weik/code/LAFA/lom'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='GPU-0deb79af-5840-4e99-f256-2d23d24d3d85')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--worker', choices=['jax', 'vkfft'])
    args = parser.parse_args()
    args.lom = args.lom.resolve()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    files = [Path(__file__), repo/'pyvkfft/jax_asm.py', repo/'pyvkfft/libvkfft_asm_ffi.so']
    files += [args.lom/p for p in [
        'holo2d.py', 'apps/holography.py', 'functional/propagation.py',
        'functional/doe.py', 'conf/holo2d/config.yaml',
        'datasets/pexels_selected/pexels-fran-arrotge-533902861-16491141.jpg',
        'datasets/Spectrum/Illuminant/laser_LP520-SF15.csv']]
    manifest = dict(source_sha256={str(p): digest(p) for p in files},
                    gpu_uuid=args.gpu, python=sys.executable, steps=args.steps,
                    protocol='Sequential fresh processes on one GPU, JAX then VkFFT. '
                    'Default holo2d config except cat target, backend/grouping, '
                    'output paths, disabled remote logging and saved raw evaluation arrays. '
                    'Training loop includes all updates, usual metrics/checkpoints. '
                    'One complete experiment per backend; no statistical confidence interval.',
                    runs=[])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, MPLBACKEND='Agg',
               PYTHONPATH=str(repo)+os.pathsep+str(args.lom))
    for backend in ['jax', 'vkfft']:
        output = args.output/backend
        output.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', backend,
                   '--lom', str(args.lom), '--output', str(output), '--steps', str(args.steps)]
        start = time.perf_counter()
        with (output/'console.log').open('w') as log:
            status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        run = dict(backend=backend, command=command, returncode=status,
                   full_process_seconds=time.perf_counter()-start)
        manifest['runs'].append(run)
        write_json(args.output/'manifest.json', manifest)
        print(json.dumps(run), flush=True)
        if status:
            raise RuntimeError(f'{backend} failed; see {output / "console.log"}')
    print('Completed both full experiments:', args.output, flush=True)


if __name__ == '__main__':
    main()
