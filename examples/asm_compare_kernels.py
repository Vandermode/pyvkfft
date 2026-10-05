"""Sustained alternating comparison of ASM tuning profiles/libraries.

Pass a JSON list of {name, library, profile}; invoke in three fresh processes.
Source data remains fixed and separate from destination throughout timing.
"""
import argparse
import json
import os
from pathlib import Path
import time

import cupy as cp
import numpy as np
from pyvkfft.asm import ASMPlan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--shape', nargs=2, type=int, default=[4096,16384])
    p.add_argument('--mode', choices=['static','dynamic'], default='static')
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--blocks', type=int, default=4)
    args = p.parse_args()
    cp.random.seed(97)
    x = (cp.random.random(args.shape, dtype=cp.float32) +
         1j*cp.random.random(args.shape, dtype=cp.float32)).astype(cp.complex64)
    out = cp.empty_like(x)
    h, w = args.shape
    transfer = cp.exp(1j*cp.arange(4*h*w, dtype=cp.float32).reshape(2*h,2*w)*1e-5) if args.mode == 'static' else None
    candidates = json.loads(args.candidates.read_text())
    plans = []
    for item in candidates:
        os.environ['PYVKFFT_ASM_LIBRARY'] = item['library']
        plan = ASMPlan(args.shape, (6.4e-6,6.4e-6), mode=args.mode, tuning_profile=item['profile'])
        if transfer is not None:
            plan.prepare_transfer(transfer)
        item['info'] = plan.info
        item['blocks_ms'] = []
        plans.append(plan)
    optical = {'z': .1, 'wavelength': 532e-9} if args.mode == 'dynamic' else {}
    reference = None
    for item, plan in zip(candidates,plans):
        plan._work.fill(cp.nan)
        for _ in range(8): plan.execute(x,out,**optical)
        if reference is None:
            reference = out.copy()
        item['relative_l2_to_first'] = float((cp.linalg.norm(out-reference)/cp.linalg.norm(reference)).get())
        if item['relative_l2_to_first'] > 2e-5:
            raise RuntimeError('Candidate mismatch')
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    for block in range(args.blocks):
        order = range(len(plans)) if block%2 == 0 else reversed(range(len(plans)))
        for i in order:
            plan = plans[i]
            total, count = 0., 0
            deadline = time.monotonic()+args.seconds
            while time.monotonic()<deadline:
                start.record()
                for _ in range(20): plan.execute(x,out,**optical)
                stop.record(); stop.synchronize()
                total += cp.cuda.get_elapsed_time(start,stop)
                count += 20
            candidates[i]['blocks_ms'].append(total/count)
            print(block,candidates[i]['name'],total/count,flush=True)
            args.output.write_text(json.dumps(candidates,indent=2)+'\n')
    for plan in plans: plan.close()


if __name__ == '__main__': main()
