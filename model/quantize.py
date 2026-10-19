#!/usr/bin/env python3
"""Full-int8 post-training quantization of Keras MobileNetV2 (laptop, CPU only).

Writes:
  model/mobilenet_v2_int8.tflite   quantized model (not committed, regenerate with this script)
  model/calib_list.txt             the exact calibration images used
  model/sample_list.txt            held-out images for the board app / sanity check
  results/quantize_log.txt         accuracy of float vs int8 on the held-out set
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_SIZE = 224
RESIZE_SHORT_SIDE = 256
SEED = 93

# Imagenette wnid -> ImageNet-1k class index (used only for the sanity check).
WNID_TO_IMAGENET = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def load_image(path: Path) -> np.ndarray:
    """Resize short side to 256, center crop 224, return float32 HWC in [0, 255]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = RESIZE_SHORT_SIDE / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - INPUT_SIZE) // 2, (h - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    return np.asarray(img, dtype=np.float32)


def preprocess(arr: np.ndarray) -> np.ndarray:
    """MobileNetV2 preprocessing: scale [0, 255] to [-1, 1]."""
    return arr / 127.5 - 1.0


def pick_images(root: Path, split: str, per_class: int, rng: random.Random) -> list[Path]:
    picked: list[Path] = []
    for wnid in sorted(WNID_TO_IMAGENET):
        files = sorted((root / split / wnid).glob("*.JPEG"))
        if not files:
            raise SystemExit(f"no images in {root / split / wnid}")
        picked += rng.sample(files, per_class)
    return picked


def evaluate(model_path: Path | None, keras_model, paths: list[Path]) -> float:
    """Top-1 accuracy over the held-out set, 1000-way argmax compared with the true class."""
    import tensorflow as tf

    correct = 0
    interp = None
    if model_path is not None:
        interp = tf.lite.Interpreter(model_path=str(model_path), num_threads=2)
        interp.allocate_tensors()
        inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
        scale, zero = inp["quantization"]
    for p in paths:
        x = preprocess(load_image(p))[None]
        if interp is None:
            pred = int(np.argmax(keras_model.predict(x, verbose=0)))
        else:
            q = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], q)
            interp.invoke()
            pred = int(np.argmax(interp.get_tensor(out["index"])))
        correct += int(pred == WNID_TO_IMAGENET[p.parent.name])
    return correct / len(paths)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True, help="imagenette2-320 directory")
    ap.add_argument("--out", type=Path, default=Path("model/mobilenet_v2_int8.tflite"))
    ap.add_argument("--calib-per-class", type=int, default=10, help="10 classes x 10 = 100 images")
    ap.add_argument("--sample-per-class", type=int, default=5)
    args = ap.parse_args()

    import tensorflow as tf

    rng = random.Random(SEED)
    calib = pick_images(args.dataset, "train", args.calib_per_class, rng)
    samples = pick_images(args.dataset, "val", args.sample_per_class, rng)
    Path("model/calib_list.txt").write_text("\n".join(str(p.relative_to(args.dataset)) for p in calib) + "\n")
    Path("model/sample_list.txt").write_text(
        "\n".join(str(p.relative_to(args.dataset)) for p in samples) + "\n"
    )

    keras_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", input_shape=(INPUT_SIZE, INPUT_SIZE, 3)
    )

    def representative_dataset():
        for p in calib:
            yield [preprocess(load_image(p))[None]]

    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite_bytes = conv.convert()
    args.out.write_bytes(tflite_bytes)

    float_acc = evaluate(None, keras_model, samples)
    int8_acc = evaluate(args.out, None, samples)
    lines = [
        f"tensorflow {tf.__version__}",
        f"model bytes (int8 tflite): {args.out.stat().st_size}",
        f"calibration images: {len(calib)}  held-out images: {len(samples)}",
        f"float32 top-1 on held-out (1000-way argmax): {float_acc:.3f}",
        f"int8    top-1 on held-out (1000-way argmax): {int8_acc:.3f}",
    ]
    Path("results").mkdir(exist_ok=True)
    Path("results/quantize_log.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
