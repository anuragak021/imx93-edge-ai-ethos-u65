#!/bin/sh
# Run cppcheck and clang-tidy on the C app sources (laptop) and save output with real exit codes.
# Usage: sh stretch_c_app/run_static_analysis.sh <output file>
# Requires: tools/fetch_c_deps.py already run (TFLite C headers).
set -u
OUT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
cd "$(dirname "$0")"
SRC="src/imgproc.c src/quant.c src/labels.c src/main.c"
INC="-Iinclude -isystem third_party/tflite -isystem third_party/stb"
{
  echo "cppcheck: $(cppcheck --version)"
  # shellcheck disable=SC2086
  cppcheck --enable=warning,style,performance,portability --std=c11 --error-exitcode=1 -q $INC $SRC tests/test_quant.c tests/test_labels.c tests/test_imgproc.c
  echo "cppcheck exit code: $?"
  echo
  echo "clang-tidy: $(clang-tidy --version | head -n 2 | tail -n 1)"
  # shellcheck disable=SC2086
  clang-tidy -quiet $SRC -checks='clang-analyzer-*,bugprone-*,cert-*,-bugprone-easily-swappable-parameters,-cert-err33-c' -- -std=c11 $INC 2>&1 | grep -E 'warning:|error:'
  echo "clang-tidy done (warnings listed above, grep exit code ignored)"
} > "$OUT" 2>&1
cat "$OUT" | cut -c1-210
