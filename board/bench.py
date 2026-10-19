#!/usr/bin/env python3
"""Timed inference loop that records every latency, so the median comes from saved data.

benchmark_model reports avg/min/max/std but no median; this script writes one CSV row
per run (run 0 is the first inference, kept separate from the steady-state statistics).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

NS_PER_MS = 1_000_000.0
DEFAULT_RUNS = 200
DEFAULT_WARMUP = 10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "npu"), required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--delegate", help="path to libethosu_delegate.so (npu only)")
    ap.add_argument("--no-xnnpack", action="store_true", help="cpu only: disable default XNNPACK")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()

    if args.device == "npu":
        if not args.delegate:
            raise SystemExit("--delegate is required for npu")
        interp = Interpreter(str(args.model), experimental_delegates=[load_delegate(args.delegate)])
    else:
        # Public tflite_runtime has no switch to disable XNNPACK; use OpResolver choice when asked.
        from tflite_runtime.interpreter import OpResolverType

        resolver = (
            OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES if args.no_xnnpack else OpResolverType.AUTO
        )
        interp = Interpreter(
            str(args.model), num_threads=args.threads, experimental_op_resolver_type=resolver
        )
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    rng = np.random.default_rng(0)
    data = rng.integers(-128, 128, size=inp["shape"], dtype=np.int8)  # latency does not depend on values
    interp.set_tensor(inp["index"], data)

    lat_ms: list[float] = []
    for _ in range(1 + args.warmup + args.runs):
        t0 = time.perf_counter_ns()
        interp.invoke()
        lat_ms.append((time.perf_counter_ns() - t0) / NS_PER_MS)

    first, steady = lat_ms[0], lat_ms[1 + args.warmup :]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w") as f:
        f.write("index,phase,latency_ms\n")
        for i, v in enumerate(lat_ms):
            phase = "first" if i == 0 else ("warmup" if i <= args.warmup else "steady")
            f.write(f"{i},{phase},{v:.4f}\n")
    cuts = statistics.quantiles(steady, n=100)
    print(
        f"device={args.device} threads={args.threads} no_xnnpack={args.no_xnnpack} runs={len(steady)} "
        f"first_ms={first:.3f} median_ms={statistics.median(steady):.3f} "
        f"mean_ms={statistics.fmean(steady):.3f} p95_ms={cuts[94]:.3f} min_ms={min(steady):.3f} max_ms={max(steady):.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
