# NOTES: design decisions (imx93)

Plain-language log of what was done, why, and the tradeoffs.

## 1. Laptop environment

- **What:** `uv` venv in `.venv/` with Python 3.10 and `tensorflow-cpu==2.15.1`, `numpy<2`, `pillow`.
- **Why:** Quantization needs only the TFLite converter, no GPU. TF 2.15 is the last line that supports
  Python 3.10 comfortably and still has a stable `TFLiteConverter` with full int8 support. `numpy<2`
  because TF 2.15 was built against NumPy 1.x.
- **Tradeoff:** The TF version that writes the `.tflite` does not need to match the board's runtime
  exactly (the flatbuffer schema is stable), but the **Vela version must match the board's NPU
  firmware/runtime**. That is checked on the board first, see below.
- **Laptop is WSL2:** USB serial devices are not visible by default. Use `usbipd` from Windows to attach
  the FRDM debug port (or run picocom/PuTTY on the Windows side).

## Plan

1. Serial console, then network and SSH, record BSP version.
2. Probe the board: `vela`, Ethos U delegate library (`libethosu_delegate.so`), `benchmark_model`, `tflite_runtime`.
3. Quantize MobileNet V2 to full int8 (`model/quantize.py`), representative dataset about 100 images.
4. Vela compile (on board if present, else matching pip version), save the summary to `results/`.
5. `board/run_benchmarks.sh`: CPU 1 thread, CPU 2 threads, NPU; raw logs to `results/`.
6. `board/infer.py` classifier with `--device cpu|npu`; compare top 1 between devices.
7. README table filled only from `results/` logs; NOTES.md concept sections; advisor review; done.

## 2. Quantization (`model/quantize.py`)

- **What int8 quantization does:** each float32 weight/activation `x` is stored as an 8-bit integer
  `q` with `x ~ scale * (q - zero_point)`. The model is about 4x smaller and integer math is much
  cheaper than float on the CPU, and the Ethos U65 can *only* run int8 (or int16 activations).
- **Why a representative dataset:** weights are known, but activation ranges are not. The converter
  runs the calibration images through the float model to record min/max per tensor, then picks scales
  from them. Random or unrepresentative data gives wrong ranges, so values clip or lose precision and
  accuracy drops. We use 100 real ImageNet-class images (Imagenette train split, fixed seed 93).
- **Full int8, int8 in/out:** `TFLITE_BUILTINS_INT8` plus `inference_input_type = inference_output_type = int8`.
  If any float op or float edge were left, Vela would have to leave it on the CPU. The input scale and
  zero point are read from the model at run time, never hardcoded.
- **Held-out set:** calibration images (train split) and the sample set (val split) are disjoint, so the
  accuracy check is not measured on calibration data.
- **Dataset:** Imagenette is a 10-class ImageNet subset, so its labels map to ImageNet-1k indices and
  the stock MobileNetV2 can be scored without retraining. The check is 1000-way argmax compared with the
  true class.
- **Tradeoff:** post-training quantization is fast but loses some accuracy (see `results/quantize_log.txt`);
  quantization-aware training would recover it but needs training. The held-out set is only 50 images,
  so the float vs int8 accuracy gap is indicative, not statistically solid.
- **Lesson:** the first dataset download died silently (backgrounded curl, hidden errors), leaving a
  truncated file; a naive resume then appended wrong data. Always verify size/checksum after a download.
