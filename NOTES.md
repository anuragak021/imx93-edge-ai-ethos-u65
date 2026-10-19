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

## 3. Board access and probe (`results/board_probe.txt`)

- **Serial:** on WSL2 the FRDM debug USB is a WCH CH342 (1a86:55d2). It shows up as `/dev/ttyACM1` and
  `/dev/ttyACM2` (cdc_acm driver, not ttyUSB). `ttyACM1` is the A55 Linux console; the other stayed silent
  (presumably the M33 console). `ttyACM0` was the Nucleo's ST-Link VCP and is not touched by this project.
  Login is `root`, no password.
- **BSP:** NXP i.MX Release Distro 6.6-scarthgap, kernel 6.6.36-lts-next, aarch64, Python 3.12.4.
- **Found on the board:** `vela` 3.12.0 (`/usr/bin/vela`), delegate `/usr/lib/libethosu_delegate.so`,
  `benchmark_model` at `/usr/bin/tensorflow-lite-2.16.2/examples/benchmark_model`, `tflite_runtime` 2.16.2,
  device node `/dev/ethosu0`. So Vela runs on the board and matches its NPU stack; no pip pin needed on the laptop.
- **Converter vs runtime:** laptop TF 2.15.1 is older than the board runtime 2.16.2, which is the safe direction.
- **remoteproc0 (M33) was `offline` at probe time.** Later observation, saved in `results/board/remoteproc_state.txt`:
  after the NPU runs it is `running`, and dmesg shows `Booting fw image ethosu_firmware` plus an `rpmsg-ethosu-channel`.
  So the Ethos-U driver loads the Ethos-U firmware onto the M33 on first NPU use and the NPU path depends on it.
  (An earlier note of mine in this session said the NPU ran with the M33 offline; that was wrong, I had only
  seen the state before the first NPU use.)
- **Network:** both Ethernet ports had no carrier at probe time (no cable), so SSH is blocked until a cable is
  plugged in.

## 4. Networking (Wi-Fi) bring-up

- The board's SDIO Wi-Fi card (`mmc2`) was detected but had no interface because the NXP driver was not loaded.
  `modprobe moal mod_para=nxp/wifi_mod_para.conf` created `mlan0`, `uap0`, `wfd0`.
- connman manages the network. A file in `/var/lib/connman/*.config` (`Type=wifi`, `Name`, `Passphrase`) is the
  non-interactive way to join; `connmanctl connect` without an agent hangs the console waiting for a passphrase.
- connman brought the link up on `wfd0` (192.168.1.16), not `mlan0`. Both are the same radio; SSH works either way.
- **Why not Ethernet:** no cable was available; Wi-Fi was already supported by the hardware.
- Not done: making the module load persistent across reboots, and a reboot test. The passphrase stays out of the repo.
- Board RTC is wrong (file dates show Sep 9), so board timestamps in logs are not meaningful.

## 5. Vela (`results/vela_summary.txt`)

- **What Vela does:** takes the int8 TFLite flatbuffer and rewrites supported operators into one custom
  `ethos-u` operator holding a command stream for the NPU, plus re-laid-out weights. Unsupported operators stay
  as normal TFLite ops that run on the CPU. It also schedules tensors in memory and estimates cycles.
- **Version match:** run on the board so the Vela version (3.12.0) is the one shipped with the board's NPU
  driver stack; the command stream format must match the driver/firmware.
- **Config choice:** `--accelerator-config ethos-u65-256` with Vela's defaults. I did not open NXP's i.MX Machine
  Learning User's Guide in this session, so I do not cite it; what is observed is that this configuration compiled
  and ran on the board. Vela reported "internal-default" for system config and memory mode only because none was
  passed. It then used 375 KiB of SRAM scratch and 3877 KiB of DRAM per its own report. Other `--system-config` /
  `--memory-mode` settings change Vela's scheduling and estimate; not explored.
- **Why some ops can fall back to the CPU:** the NPU supports a fixed set of int8 ops with shape/stride/
  padding limits. Anything float, unsupported, or beyond a limit is left to the CPU, and each CPU/NPU switch
  costs a delegate partition. For this model Vela reports 0 CPU operators, so there was no fallback to analyze;
  the conclusion applies to MobileNetV2 only.

## 6. Benchmarking decisions

- **Two measurement paths:** TFLite `benchmark_model` (the standard tool) and `board/bench.py` (records every run).
  `benchmark_model` gives no median, so reporting its average as a median would be wrong; the median comes from
  the saved CSVs. Median and benchmark_model average agree within about 1 percent on every config, which is a cross-check on both.
- **CPU baseline:** `benchmark_model` and tflite_runtime apply XNNPACK implicitly, so the "obvious" CPU number is
  XNNPACK. I report XNNPACK on as the baseline and XNNPACK off for reference, because a quoted speedup against
  the slower reference kernels would overstate the NPU's advantage (19.6x vs 11.7x at one thread; 6.6x vs the two-thread XNNPACK baseline).
- **Warm-up:** first inference is separated from steady state; 10 extra warm-up runs are discarded.
- **Random input for latency:** inference time does not depend on values; accuracy is checked separately.
- **Agreement check:** 49 of 50 top-1 labels match between CPU and NPU; the one mismatch is a near-tie, see README.
- **Not measured:** power, temperature, CPU clock (no cpufreq sysfs), run-to-run variance across separate boots.
- **Shared preprocessing:** `infer.py` copies `quantize.py`'s load/resize/crop code. Duplicated on purpose so
  the board has no dependency on TensorFlow; a divergence would hurt accuracy silently, so they are kept identical.

## 7. Provenance and reruns

- The first board run was discarded. The final results are from one clean rerun using the committed, formatted
  scripts, which also records sha256 of both models, the remoteproc state, and passes the delegate path found by the
  script to `infer.py`. `results/board/env.txt` holds the hashes and exact commands.
- `model/*_list.txt` hold paths relative to the dataset directory so another machine can reproduce the selection.
- Board-side accuracy (0.72 CPU, 0.70 NPU) is scored from `top1_idx` in the infer logs against the wnid folder in
  `model/sample_list.txt`. CPU 0.72 equals the laptop int8 figure, which supports identical preprocessing despite
  different Pillow versions (laptop 12.3, board 10.3).
- Credentials: the Wi-Fi passphrase was typed on the board over serial and lives only in
  `/var/lib/connman/home.config` (mode 600). The board shell history was cleared. Nothing in this repo.

## 8. C inference app (`stretch_c_app/`)

- **Why this layout:** the root standard wants pure logic that builds and tests on the host, with hardware access thin
  and separate. `imgproc`, `quant`, `labels` have no TFLite dependency and run under `ctest` on the laptop and on the
  board; `main.c` is the only file that touches TFLite, stb_image, the filesystem and the clock.
- **Why re-implement Pillow's resize:** Pillow's bilinear resize is not plain bilinear. When shrinking it widens the
  triangle filter by the scale factor (antialiasing) and rounds to 8 bit after each pass. A naive bilinear resize
  would change the pixels the model sees and silently change accuracy. I port that algorithm and test it against
  Pillow output generated by `tools/gen_resize_golden.py` (tolerance 1 for rounding).
- **Why stb_image:** a single public-domain header, no system libraries on the board needed. Tradeoff: it is not
  libjpeg, so decoded pixels can differ slightly from Pillow (see README, 2 to 3 top-1 differences vs the Python app).
  Alternatives: link libjpeg-turbo (matches Pillow better, adds a dependency), or accept pre-decoded raw input.
- **Why headers are fetched, not committed:** the board image ships `libtensorflow-lite.so.2.16.2` but no headers.
  `tools/fetch_c_deps.py` downloads the v2.16.2 headers (so they match the library) and a pinned, hash-checked
  stb_image.h. Both are gitignored.
- **Delegate loading:** the C API `TfLiteExternalDelegateCreate` loads `libethosu_delegate.so` the same way the Python
  `load_delegate` does. XNNPACK is not applied by default through the C API, found because the 1-thread C latency
  equalled the XNNPACK-off Python figure; `--xnnpack` adds it explicitly through the exported `TfLiteXNNPackDelegate*`.
- **Deviations from the root C standard (userland Linux app, not MCU firmware):**
  - dynamic allocation exists only inside stb_image (compiled without the strict flags, `-w`, in its own object);
    the project's own modules use static buffers sized by named constants. The static work buffers are large
    (about 12 MB for the resize pass) and make `img_preprocess` non-reentrant; documented in imgproc.h.
  - `getopt_long`, `opendir`, `clock_gettime` need `_POSIX_C_SOURCE`; clang-tidy flags the macro as a reserved
    identifier (`bugprone-reserved-identifier`). It is the documented feature-test macro, so accepted.
  - The final-source analysis output is saved in `results/static_analysis.txt` (`stretch_c_app/run_static_analysis.sh`):
    cppcheck exit code 0; clang-tidy reports 4 `memcpy`, 2 `memset`, 2 `snprintf` notes and the macro warning.
  - clang-tidy's `clang-analyzer-security.insecureAPI.*` warns about `memcpy`, `memset`, `snprintf` (they lack the
    optional C11 Annex K `_s` variants, which glibc does not provide). Every call has an explicit, checked length;
    accepted. The one real finding, an unbounded `strcpy`, was replaced by a length-checked `memcpy`.
  - MISRA C:2012 was not checked with a tool (none available here); only the compiler warnings, cppcheck and
    clang-tidy were run. Discarded return values: printf-style calls, `fclose` of a file opened read-only
    (after its content and `ferror` were already checked), and `closedir`. `clock_gettime` is checked: a failing clock
    aborts the run instead of corrupting latencies.
- **Failure policy:** any per-image failure (decode, size limit, inference) aborts the run with a non-zero exit instead of
  skipping, so a partial result cannot look complete.
- **Bugs found while testing, kept as lessons:** my first quantization tests asserted exact rounding ties that float32
  cannot represent (rewritten with a power-of-two scale); a scripted edit inserted the `--xnnpack` case so that
  `--delegate` fell through into it, caught only because I reread the block; and the first board run of the script
  stopped at the XNNPACK flag because the case label had never been added. The saved logs come from the run after the fix.

## 9. RPMsg ping-pong (A55 to M33)

- **What RPMsg is here:** Linux (A55) and the M33 share memory (vrings and buffers in reserved DDR regions from the
  device tree: `vdev0vring0@a4000000` etc.) and signal each other with the messaging unit hardware. `remoteproc` loads
  the M33 firmware and creates the `virtio_rpmsg_bus`; the firmware announces named channels; a Linux driver
  bound to a channel name talks to it. The same mechanism carries the NPU's `rpmsg-ethosu-channel`.
- **Why the NPU and the demo conflict:** the i.MX 93 Ethos-U driver runs its control firmware on the M33, so the M33
  can run either `ethosu_firmware` or a demo, not both. The script stops the core, changes
  `/sys/class/remoteproc/remoteproc0/firmware`, starts it, and a `trap` restores `ethosu_firmware` on any exit.
  Nothing persistent changes; a reboot also restores the default.
- **Why a prebuilt demo, not my own firmware:** building M33 firmware needs NXP's MCUXpresso SDK (login download) and an
  Arm toolchain, which is the later roadmap item (FreeRTOS or Zephyr M33 firmware doing sensor sampling). The point of
  this stretch goal was to prove the two-core link works, which the saved dmesg does.
- **Tradeoffs and unknowns:** the ELF is for the EVK, not the FRDM; it ran, but I did not verify why it is compatible, nor
  read the M33 UART output. Ping-pong here has no latency measurement; the timestamps in dmesg (about 0.2 s for 51
  messages including setup) are not a benchmark.
- **Kernel noise seen:** `ethosu: can't change firmware or remote processor is running` appears whenever the Ethos-U
  driver is used while the core is up; it was already present in the earlier benchmark logs and is harmless.
