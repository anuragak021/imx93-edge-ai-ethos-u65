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
