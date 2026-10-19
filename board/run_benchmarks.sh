#!/bin/sh
# Run every benchmark configuration on the board and save raw logs.
# Usage (on the board, from the directory holding the models and bench.py):
#   sh run_benchmarks.sh [OUT_DIR]        default OUT_DIR=results
# Then copy OUT_DIR back to the laptop: scp -r root@<board>:imx93/results/ .
set -eu

OUT="${1:-results}"
MODEL_CPU="mobilenet_v2_int8.tflite"
MODEL_NPU="vela_out/mobilenet_v2_int8_vela.tflite"
RUNS=200
WARMUP=10

mkdir -p "$OUT"

# Locate tools instead of hardcoding paths (see results/board_probe.txt for where they were found).
BM="$(find /usr -name benchmark_model -type f 2>/dev/null | head -n 1)"
DELEGATE="$(find /usr -name 'libethosu_delegate.so' 2>/dev/null | head -n 1)"
[ -n "$BM" ] || { echo "benchmark_model not found" >&2; exit 1; }
[ -n "$DELEGATE" ] || { echo "libethosu_delegate.so not found" >&2; exit 1; }
[ -f "$MODEL_CPU" ] || { echo "$MODEL_CPU missing" >&2; exit 1; }

# Compile for the Ethos-U65 on the board, so Vela matches the board's NPU stack.
vela "$MODEL_CPU" --accelerator-config ethos-u65-256 --output-dir vela_out > "$OUT/vela_summary.txt" 2>&1
cp vela_out/mobilenet_v2_int8_summary_*.csv "$OUT/vela_per_op_summary.csv"
[ -f "$MODEL_NPU" ] || { echo "$MODEL_NPU missing after vela" >&2; exit 1; }

{
  echo "date_board=$(date) (board RTC may be wrong)"
  cat /etc/os-release
  uname -a
  echo "benchmark_model=$BM"
  echo "delegate=$DELEGATE"
  echo "tflite_runtime=$(python3 -c 'import tflite_runtime as t; print(t.__version__)')"
  echo "vela=$(vela --version)"
  echo "cpu_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>&1)"
  echo "cpu_cur_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>&1)"
  echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>&1)"
  echo "model_cpu_bytes=$(wc -c < "$MODEL_CPU")"
  echo "model_npu_bytes=$(wc -c < "$MODEL_NPU")"
  echo "sha256_cpu_model=$(sha256sum "$MODEL_CPU" | cut -d' ' -f1)"
  echo "sha256_npu_model=$(sha256sum "$MODEL_NPU" | cut -d' ' -f1)"
  echo "commands:"
  echo "vela $MODEL_CPU --accelerator-config ethos-u65-256 --output-dir vela_out"
} > "$OUT/env.txt"

run() { # name, then command...
  name="$1"; shift
  echo "$*" >> "$OUT/env.txt"
  echo "== $name"
  "$@" > "$OUT/$name.log" 2>&1
}

# --- TFLite benchmark_model (reports first inference, avg, min, max, std) ---
for t in 1 2; do
  run "bm_cpu_${t}t_xnnpack" "$BM" --graph="$MODEL_CPU" --num_threads="$t" \
    --num_runs="$RUNS" --warmup_runs="$WARMUP" --use_xnnpack=true
  run "bm_cpu_${t}t_noxnnpack" "$BM" --graph="$MODEL_CPU" --num_threads="$t" \
    --num_runs="$RUNS" --warmup_runs="$WARMUP" --use_xnnpack=false
done
run "bm_npu" "$BM" --graph="$MODEL_NPU" --num_runs="$RUNS" --warmup_runs="$WARMUP" \
  --external_delegate_path="$DELEGATE"

# --- Per-run latencies (CSV) so the median comes from saved data ---
for t in 1 2; do
  run "py_cpu_${t}t_xnnpack" python3 bench.py --device cpu --model "$MODEL_CPU" --threads "$t" \
    --runs "$RUNS" --warmup "$WARMUP" --csv "$OUT/py_cpu_${t}t_xnnpack.csv"
  run "py_cpu_${t}t_noxnnpack" python3 bench.py --device cpu --model "$MODEL_CPU" --threads "$t" \
    --no-xnnpack --runs "$RUNS" --warmup "$WARMUP" --csv "$OUT/py_cpu_${t}t_noxnnpack.csv"
done
run "py_npu" python3 bench.py --device npu --model "$MODEL_NPU" --delegate "$DELEGATE" \
  --runs "$RUNS" --warmup "$WARMUP" --csv "$OUT/py_npu.csv"

# --- Classification of the held-out sample set on both devices ---
run "infer_cpu" python3 infer.py --device cpu --model "$MODEL_CPU" --images samples
run "infer_npu" python3 infer.py --device npu --model "$MODEL_NPU" --images samples \
  --delegate "$DELEGATE"

# M33 remoteproc state after the NPU runs: shows whether the NPU worked without the M33 running.
{
  for d in /sys/class/remoteproc/remoteproc*; do echo "$d $(cat "$d/name") $(cat "$d/state")"; done
  dmesg | grep -i ethos
} > "$OUT/remoteproc_state.txt" 2>&1

echo "done; logs in $OUT/"
