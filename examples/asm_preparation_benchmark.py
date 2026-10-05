"""Compare legacy transpose-then-pack with direct column-first H packing.

Measures preparation kernels only, with the legacy temporary preallocated.
Execution speed is unchanged by this comparison; use asm_memory_probe.py for
construction/preparation high-water memory including temporary allocations.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--shape', type=int, nargs=2, required=True)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--seconds-per-block', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    import cupy as cp
    import numpy as np
    from pyvkfft.asm_axis import ColumnFirstASMPlan
    from issue205_sustained import Monitor
    shape = tuple(args.shape)
    free, _ = cp.cuda.runtime.memGetInfo()
    grid_bytes = 32*shape[0]*shape[1]
    if 6*grid_bytes > .8*free:
        raise MemoryError('Preparation comparison exceeds conservative 80% memory budget')
    stream = cp.cuda.Stream(non_blocking=True)
    with stream, ColumnFirstASMPlan(shape, (6.4e-6, 6.4e-6), stream=stream,
            tuning_profile={'implementation': 'native', 'prune': True}) as plan:
        h = cp.empty(plan.padded_shape, cp.complex64)
        temporary = cp.empty(plan.padded_shape[::-1], cp.complex64)
        fill = cp.RawKernel(r'''
extern "C" __global__ void fill(float2* p, long long n) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) p[i]=make_float2((i%1237)/1024.f, ((i*13)%1231)/1024.f);
}''', 'fill')
        fill(((h.size+255)//256,), (256,), (h, np.int64(h.size)))

        def legacy():
            plan._copy_transposed(h, temporary)
            plan._plan.prepare_transfer(temporary)

        def direct():
            plan.prepare_transfer(h)

        legacy()
        expected = plan._plan._transfer.copy()
        direct()
        assert bool(cp.array_equal(plan._plan._transfer, expected))
        stream.synchronize()
        del expected
        cp.get_default_memory_pool().free_all_blocks()
        blocks = []
        monitor = Monitor(uuid)
        monitor.thread.start()
        for round_index in range(args.rounds):
            variants = [('legacy', legacy), ('direct', direct)]
            if round_index % 2:
                variants.reverse()
            for label, execute in variants:
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record(stream)
                execute()
                end.record(stream)
                end.synchronize()
                estimate = cp.cuda.get_elapsed_time(start, end)
                count = max(1, int(args.seconds_per_block*1000/estimate)+1)
                t0 = time.monotonic()
                start.record(stream)
                for _ in range(count):
                    execute()
                end.record(stream)
                end.synchronize()
                t1 = time.monotonic()
                blocks.append(dict(label=label, round=round_index, iterations=count,
                                   start_monotonic=t0, end_monotonic=t1,
                                   milliseconds=cp.cuda.get_elapsed_time(start, end)/count))
        monitor.close()
        result = dict(shape=shape, gpu_uuid=uuid, bitwise_equal=True, blocks=blocks,
                      median_ms={label: statistics.median(b['milliseconds'] for b in blocks
                                                         if b['label'] == label)
                                 for label in ('legacy', 'direct')},
                      protocol='Static H preparation kernels only; legacy full-grid temporary preallocated; matched alternating blocks',
                      plan=plan.info)
        result['telemetry'] = monitor.rows
        result['telemetry_errors'] = monitor.errors
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['median_ms']), flush=True)


if __name__ == '__main__':
    main()
