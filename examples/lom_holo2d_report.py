"""Summarize completed paired runs from lom_holo2d_benchmark.py (CPU only)."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    timings = {b: json.loads((directory/b/'timing.json').read_text()) for b in ['jax', 'vkfft']}
    assert all(r['returncode'] == 0 for r in manifest['runs'])
    for record in timings.values():
        assert record['completed'] and record['optimizer_final_step'] == manifest['steps']
        assert record['completed_steps'] == manifest['steps']
    for name in ['initial_design_sha256', 'target_array_sha256', 'gpu_uuid',
                 'jax_version', 'jaxlib_version', 'environment', 'design_shape']:
        assert timings['jax'][name] == timings['vkfft'][name], name
    assert any(p['transpose'] == 1 for p in timings['vkfft']['native_plans'])

    configs = {}
    for backend in timings:
        cfg = OmegaConf.to_container(OmegaConf.load(directory/backend/'.hydra/config.yaml'), resolve=False)
        cfg.pop('exp_name')
        cfg['prop_cfg'].pop('propagation_backend')
        cfg['log_cfg'].pop('ckpt_dir')
        cfg['log_cfg'].pop('wandb_dir')
        configs[backend] = cfg
    assert configs['jax'] == configs['vkfft'], 'Unexpected configuration differences'
    config = configs['jax']

    histories, averages, results = {}, {}, {}
    target = np.asarray(Image.open(directory/'jax/ckpt/target.png'), dtype=np.float64)/255.
    for backend, timing in timings.items():
        rows = []
        for line in (directory/backend/'main.log').read_text().splitlines():
            if line.startswith('OrderedDict('):
                value = ast.literal_eval(line[len('OrderedDict('):-1])
                rows.append({key: float(v) if key != 'cur_iter' else int(v) for key, v in value.items()})
        assert rows[-1]['cur_iter'] == manifest['steps']-1
        histories[backend] = rows
        with np.load(directory/backend/'ckpt/holoimgs.npz') as saved:
            weights = saved['wvl_intensity_profile'].reshape(-1).astype(np.float64)
            image = np.tensordot(weights, saved['holoimgs'].astype(np.float64), axes=(0, 0))/weights.sum()
        normalized = image/image.max()
        scale = np.sum(target*normalized)/np.sum(normalized**2)
        psnr = -10*np.log10(np.mean((target-scale*normalized)**2))
        averages[backend] = image/image.mean()*target.mean()
        run = next(r for r in manifest['runs'] if r['backend'] == backend)
        results[backend] = dict(
            training_loop_seconds=timing['training_loop_seconds'],
            mean_ms_per_step=timing['training_loop_seconds']/manifest['steps']*1000,
            setup_compile_warmup_seconds=timing['setup_compile_warmup_seconds'],
            optimization_call_seconds=timing['optimization_call_seconds'],
            evaluation_seconds=timing['evaluation_seconds'],
            full_process_seconds=run['full_process_seconds'],
            final_logged_training_loss=rows[-1]['loss'],
            first_logged_training_loss=rows[0]['loss'],
            final_evaluation_npsnr_db=float(psnr),
            evaluation_efficiency=float(image.sum()),
            evaluation_wavelengths=len(weights),
            training_peak_live_gib=timing['training_memory_stats']['peak_bytes_in_use']/2**30,
        )
    original, native = results['jax'], results['vkfft']
    summary = dict(
        geometry=dict(doe_mm=[3.6, 3.6], design_shape=timings['jax']['design_shape'],
                      pixel_pitch_um=[2, 2], propagation_upsampling=config['prop_cfg']['upsample_ratio'],
                      padded_fft_shape=[3600, 3600], seed=config['seed'],
                      train_wavelength_m=config['prop_cfg']['wvl'],
                      propagation_distance_m=config['prop_cfg']['reference_plane_to_sensor_distance']),
        steps=manifest['steps'], gpu_kind=timings['jax']['gpu_kind'],
        runtime=dict(jax=timings['jax']['jax_version'], jaxlib=timings['jax']['jaxlib_version']),
        runs=results,
        training_loop_speedup=original['training_loop_seconds']/native['training_loop_seconds'],
        training_loop_time_reduction_percent=100*(1-native['training_loop_seconds']/original['training_loop_seconds']),
        optimization_with_setup_speedup=original['optimization_call_seconds']/native['optimization_call_seconds'],
        full_process_speedup=original['full_process_seconds']/native['full_process_seconds'],
        psnr_change_db=native['final_evaluation_npsnr_db']-original['final_evaluation_npsnr_db'],
        final_image_relative_l2=float(np.linalg.norm(averages['vkfft']-averages['jax'])/np.linalg.norm(averages['jax'])),
        verification='Same initial design/target hashes, settings, GPU, JAX versions and allocator. '
                     'All 3000 updates completed; native forward/transpose plans present.',
        protocol=manifest['protocol'],
        timing_scope='Training loop includes logging every 50 steps and normal checkpoint saves every 300 steps. '
                     'Setup/compilation/warmup is excluded from the loop but included in optimization_call. '
                     'Full process includes imports, target preparation, optimization, 22-wavelength evaluation, '
                     'plots, saved arrays and teardown. One paired run; no confidence interval.',
    )
    (directory/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(13, 8), layout='constrained')
    grid = fig.add_gridspec(2, 6, height_ratios=[1.4, 1])
    titles = ['Cat target', f"Original JAX · {original['final_evaluation_npsnr_db']:.3f} dB",
              f"VkFFT · {native['final_evaluation_npsnr_db']:.3f} dB"]
    for index, (title, data) in enumerate(zip(titles, [target, averages['jax'], averages['vkfft']])):
        ax = fig.add_subplot(grid[0, 2*index:2*index+2])
        ax.imshow(data, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    ax = fig.add_subplot(grid[1, :3])
    for backend, color, label in [('jax', '#555e6c', 'Original JAX'), ('vkfft', '#047c88', 'VkFFT')]:
        rows = histories[backend]
        ax.semilogy([r['cur_iter']+1 for r in rows], [r['loss'] for r in rows], color=color, label=label)
    ax.set(xlabel='Optimization step', ylabel='Training loss (log scale)')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    ax = fig.add_subplot(grid[1, 3:])
    labels = ['Original JAX', 'VkFFT']
    loops = [original['training_loop_seconds'], native['training_loop_seconds']]
    setup = [original['setup_compile_warmup_seconds'], native['setup_compile_warmup_seconds']]
    ax.barh(labels, loops, color=['#555e6c', '#047c88'], label='3,000-step loop')
    ax.barh(labels, setup, left=loops, color='#c3cbd0', label='Setup + compilation + warmup')
    for index, (loop, preparation) in enumerate(zip(loops, setup)):
        ax.text(loop/2, index, f'{loop:.2f} s', ha='center', va='center', color='white', weight='bold')
        ax.text(loop+preparation+.7, index, f'{loop+preparation:.2f} s total', va='center', fontsize=9)
    ax.set_xlim(0, max(a+b for a, b in zip(loops, setup))*1.32)
    ax.invert_yaxis()
    ax.set_xlabel('Elapsed seconds; includes normal logging and checkpointing')
    ax.set_title(f"3,000-step loop: {summary['training_loop_speedup']:.2f}× speedup")
    ax.legend(frameon=False, loc='lower right', fontsize=8)
    fig.suptitle('Cat holo2d · 3.6 mm DOE · 2 µm pixels · 1800 × 1800 · same A100', fontsize=15)
    fig.savefig(directory/'comparison.png', dpi=160)
    fig.savefig(directory/'comparison.pdf')
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
