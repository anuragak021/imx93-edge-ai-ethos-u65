#!/bin/sh
# Build, unit test and run the C inference app on the board; save raw logs.
# Usage (on the board, from imx93/ which holds the models and samples):
#   sh stretch_c_app/run_c_app.sh [OUT_DIR]     default OUT_DIR=results_c
set -eu

OUT="${1:-results_c}"
MODEL_CPU="mobilenet_v2_int8.tflite"
MODEL_NPU="vela_out/mobilenet_v2_int8_vela.tflite"
BIN="stretch_c_app/build/imx93_infer"

mkdir -p "$OUT"
cmake -S stretch_c_app -B stretch_c_app/build -DCMAKE_BUILD_TYPE=Release > "$OUT/cmake_configure.log" 2>&1
cmake --build stretch_c_app/build > "$OUT/cmake_build.log" 2>&1
ctest --test-dir stretch_c_app/build --output-on-failure > "$OUT/ctest.log" 2>&1
DELEGATE="$(find /usr -name 'libethosu_delegate.so' 2>/dev/null | head -n 1)"
[ -n "$DELEGATE" ] || { echo "libethosu_delegate.so not found" >&2; exit 1; }

{
  echo "binary_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "gcc=$(gcc --version | head -n 1)"
  echo "delegate=$DELEGATE"
  echo "commands:"
} > "$OUT/env.txt"

run() { # name, then command...
  name="$1"; shift
  echo "$*" >> "$OUT/env.txt"
  echo "== $name"
  "$@" > "$OUT/$name.log" 2> "$OUT/$name.err"
}

run c_npu "$BIN" --device npu --model "$MODEL_NPU" --images samples --delegate "$DELEGATE"
run c_cpu_1t "$BIN" --device cpu --threads 1 --model "$MODEL_CPU" --images samples
run c_cpu_2t "$BIN" --device cpu --threads 2 --model "$MODEL_CPU" --images samples
run c_cpu_1t_xnnpack "$BIN" --device cpu --threads 1 --xnnpack --model "$MODEL_CPU" --images samples
run c_cpu_2t_xnnpack "$BIN" --device cpu --threads 2 --xnnpack --model "$MODEL_CPU" --images samples
echo "done; logs in $OUT/"
