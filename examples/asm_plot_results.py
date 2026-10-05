"""Plot normalized, independently validated ASM benchmark results.

Input JSON contains a ``comparisons`` list with device, shape, mode,
baseline_ms, native_ms, speedup, and ci95 fields. The producing report must
retain the raw-run paths and statistical method; this script does no fitting.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    rows = data['comparisons']
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ['device', 'shape', 'mode', 'baseline', 'native', 'baseline_ms',
              'native_ms', 'speedup', 'ci95', 'source_summary']
    with (args.output/'comparisons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    devices = list(dict.fromkeys(row['device'] for row in rows))
    fig, axes = plt.subplots(1, len(devices), figsize=(11, 4.5), squeeze=False,
                             gridspec_kw={'width_ratios': [max(2, len({tuple(r['shape'])
                                 for r in rows if r['device'] == device})) for device in devices]})
    colors = {'static': '#2166ac', 'dynamic_both': '#b35806'}
    names = {'static': 'Static H', 'dynamic_both': 'Changing distance + wavelength'}
    for ax, device in zip(axes[0], devices):
        subset = [r for r in rows if r['device'] == device]
        shapes = list(dict.fromkeys(tuple(r['shape']) for r in subset))
        for mode, shift in [('static', -.17), ('dynamic_both', .17)]:
            selected = [next((r for r in subset if tuple(r['shape']) == shape
                             and r['mode'] == mode), None) for shape in shapes]
            valid = [(i, r) for i, r in enumerate(selected) if r is not None]
            x = np.array([i for i, _ in valid], dtype=float)+shift
            y = np.array([r['speedup'] for _, r in valid])
            low = np.array([r['ci95'][0] for _, r in valid])
            high = np.array([r['ci95'][1] for _, r in valid])
            ax.errorbar(x, y, yerr=[y-low, high-y], fmt='o', capsize=4,
                        color=colors[mode], label=names[mode], markersize=6)
        ax.axhline(1, color='#555555', linewidth=1, linestyle='--')
        ax.set_xticks(range(len(shapes)))
        ax.set_xticklabels([f'{h:,}\n× {w:,}' for h, w in shapes], fontsize=9)
        ax.set_xlim(-.6, len(shapes)-.4)
        ax.set_title(device.replace('NVIDIA ', ''))
        ax.set_xlabel('Compact input shape')
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(.97, max(r['ci95'][1] for r in rows)+.06)
    axes[0, 0].set_ylabel('Speedup over fastest tested equivalent cuFFT')
    axes[0, 0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Prepared compact-input-to-compact-output ASM, complex64')
    fig.text(.5, .005, 'FFT grid doubles each axis. 95% intervals use independent process groups; dynamic H and transposes are timed.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, 1])
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(args.output/f'asm-speedup.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
