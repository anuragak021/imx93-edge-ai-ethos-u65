#!/usr/bin/env python3
"""Classify a folder of images with an int8 TFLite model on the i.MX 93 (CPU or Ethos-U NPU).

Prints top-5 labels and per-image latency. With --device npu the model must be the
Vela-compiled one and the Ethos-U external delegate is loaded.

Example:
  python3 infer.py --device cpu --model mobilenet_v2_int8.tflite --images samples
  python3 infer.py --device npu --model mobilenet_v2_int8_vela.tflite --images samples
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate

INPUT_SIZE = 224
RESIZE_SHORT_SIDE = 256
TOP_K = 5
NS_PER_MS = 1_000_000.0
DELEGATE_SEARCH = (
    Path("/usr/lib/libethosu_delegate.so"),
    Path("/usr/lib64/libethosu_delegate.so"),
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_image(path: Path) -> np.ndarray:
    """Resize short side to 256, center crop 224, return float32 HWC in [0, 255].

    Must match model/quantize.py exactly, otherwise CPU/NPU results are not comparable
    with the laptop accuracy check.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = RESIZE_SHORT_SIDE / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - INPUT_SIZE) // 2, (h - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    return np.asarray(img, dtype=np.float32)


def find_delegate(explicit: str | None) -> str:
    if explicit:
        return explicit
    for cand in DELEGATE_SEARCH:
        if cand.exists():
            return str(cand)
    raise SystemExit("Ethos-U delegate not found; pass --delegate (find / -name '*ethosu_delegate*')")


def load_labels(path: Path, n_outputs: int) -> list[str]:
    labels = path.read_text().splitlines()
    if len(labels) == n_outputs + 1 and labels[0].strip().lower() == "background":
        labels = labels[1:]  # TF-slim style file; Keras MobileNetV2 has no background class
    if len(labels) != n_outputs:
        raise SystemExit(f"{path}: {len(labels)} labels but model has {n_outputs} outputs")
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--device", choices=("cpu", "npu"), required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="folder of images")
    ap.add_argument(
        "--labels",
        type=Path,
        default=Path("/usr/bin/tensorflow-lite-2.16.2/examples/labels.txt"),
    )
    ap.add_argument("--threads", type=int, default=1, help="CPU threads (cpu device only)")
    ap.add_argument("--delegate", help="path to libethosu_delegate.so (default: search)")
    args = ap.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"no images in {args.images}")

    if args.device == "npu":
        delegate = find_delegate(args.delegate)
        interp = Interpreter(str(args.model), experimental_delegates=[load_delegate(delegate)])
    else:
        interp = Interpreter(str(args.model), num_threads=args.threads)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    scale, zero = inp["quantization"]
    labels = load_labels(args.labels, int(out["shape"][-1]))

    # Warm-up on the first image so the first-run cost (NPU setup, caches) is not counted.
    x0 = np.clip(np.round((load_image(files[0]) / 127.5 - 1.0) / scale + zero), -128, 127)
    interp.set_tensor(inp["index"], x0.astype(np.int8)[None])
    t0 = time.perf_counter_ns()
    interp.invoke()
    print(f"# warmup_ms={(time.perf_counter_ns() - t0) / NS_PER_MS:.3f} device={args.device}")

    for path in files:
        x = (load_image(path) / 127.5 - 1.0) / scale + zero
        interp.set_tensor(inp["index"], np.clip(np.round(x), -128, 127).astype(np.int8)[None])
        t0 = time.perf_counter_ns()
        interp.invoke()
        ms = (time.perf_counter_ns() - t0) / NS_PER_MS
        scores = interp.get_tensor(out["index"])[0].astype(np.int32)
        top = np.argsort(scores)[::-1][:TOP_K]
        names = ", ".join(f"{labels[i]}({scores[i]})" for i in top)
        print(f"{path.name}\t{ms:.3f} ms\ttop1={labels[top[0]]}\ttop5={names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
