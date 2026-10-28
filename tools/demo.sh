#!/bin/bash
# Live demo for screen recording: runs the classifier on the board over SSH, CPU then NPU,
# and prints every image as it is classified, then a summary computed from this run.
# Usage: tools/demo.sh <board-ip> [images-per-device]   (default 20; DEMO_PACE=0.15 sets the delay in s)
# Images are previewed with chafa during the NPU run (DEMO_IMAGES=0 off, DEMO_FORMAT=sixel|kitty|symbols,
# DEMO_IMG_SIZE=28x12).
# Needs the files pushed by tools/push_to_board.sh and the Vela model in imx93/vela_out/.
set -euo pipefail

BOARD="${1:?usage: tools/demo.sh <board-ip> [images-per-device]}"
COUNT="${2:-20}"
PACE="${DEMO_PACE:-0.15}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 "root@$BOARD")
VELA_LOG="results/board/vela_summary.txt"

B=$'\e[1m'; D=$'\e[2m'; R=$'\e[0m'
CYAN=$'\e[36m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; MAGENTA=$'\e[35m'

banner() { printf '\n%s== %s ==%s\n' "$B$CYAN" "$1" "$R"; }
note() { printf '%s%s%s\n' "$D" "$1" "$R"; }

"${SSH[@]}" true || { echo "cannot reach root@$BOARD over SSH" >&2; exit 1; }

printf '%s' "$B$MAGENTA"
cat <<'ART'
  _ __  __ __  __  ___   ____    ___ ____  _   _
 (_)  \/  \ \/ / / _ \ |__ /   | _ \_  _|| \ | |  Edge AI on the NXP FRDM i.MX 93
 | | |\/| |>  < | (_) | |_ \   |  _/| |  |  \| |  MobileNetV2 int8: Cortex-A55 vs Ethos-U65 NPU
 |_|_|  |_/_/\_\ \___/ |___/   |_|  |_|  |_|\__|
ART
printf '%s' "$R"

banner "1/5  Board"
"${SSH[@]}" 'grep PRETTY_NAME /etc/os-release; uname -srmo; ls -l /dev/ethosu0'

banner "2/5  Cortex-M33 (the Ethos-U driver boots it with ethosu_firmware)"
"${SSH[@]}" 'printf "remoteproc0: "; cat /sys/class/remoteproc/remoteproc0/state; dmesg | grep -i "ethosu_firmware" | tail -n 2'

banner "3/5  Vela compile result (saved log from the benchmark run)"
if [ -f "$VELA_LOG" ]; then
  grep -E "NPU operators|CPU operators|Batch Inference time" "$VELA_LOG" | sed 's/^/  /' || true
else
  note "  $VELA_LOG not found, skipped"
fi

# Run infer.py on the board, echo each line colourised, keep only the latency lines for the summary.
run_device() {
  local label="$1" colour="$2" outfile="$3" cmd="$4" show_img="${5:-0}"
  : > "$outfile"
  "${SSH[@]}" "cd imx93 && $cmd" 2>&1 | while IFS= read -r line; do
    case "$line" in
      *$'\t'*" ms"$'\t'top1=*)
        printf '%s\n' "$line" >> "$outfile"
        name="${line%%$'\t'*}"; rest="${line#*$'\t'}"; ms="${rest%%$'\t'*}"
        top1="${rest#*top1=}"; top1="${top1%%$'\t'*}"
        if [ "$show_img" = 1 ]; then
          chafa --size "${DEMO_IMG_SIZE:-28x12}" ${DEMO_FORMAT:+--format "$DEMO_FORMAT"} "$TMP/img/$name" 2>/dev/null || true
        fi
        printf '  %s%-26s%s %s%8s%s  %s\n' "$D" "$name" "$R" "$colour" "$ms" "$R" "$top1"
        sleep "$PACE"
        ;;
      "INFO: EthosuDelegate"*) printf '  %s%s%s\n' "$GREEN" "$line" "$R" ;;
    esac
  done
}

median_ms() { awk -F'\t' '{ split($2, a, " "); v[NR] = a[1] + 0 }
  END { n = asort(v); if (n == 0) { print "nan"; exit } printf "%.2f", (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2 }' "$1"; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
"${SSH[@]}" "cd imx93 && rm -rf demo_samples && mkdir demo_samples && ls samples | head -n $COUNT | while read -r f; do cp samples/\$f demo_samples/; done"
COMMON="python3 infer.py --images demo_samples"

SHOW_IMG=0
if [ "${DEMO_IMAGES:-1}" = 1 ] && command -v chafa >/dev/null; then
  mkdir -p "$TMP/img" && "${SSH[@]}" 'cd imx93/demo_samples && tar cf - .' | tar xf - -C "$TMP/img" && SHOW_IMG=1
fi

banner "4/5  Cortex-A55 CPU, 2 threads, XNNPACK  (live, $COUNT images)"
run_device cpu "$YELLOW" "$TMP/cpu.txt" \
  "$COMMON --device cpu --threads 2 --model mobilenet_v2_int8.tflite" 0

banner "5/5  Ethos-U65 NPU via the Ethos-U delegate  (live, $COUNT images)"
run_device npu "$GREEN" "$TMP/npu.txt" \
  "$COMMON --device npu --model vela_out/mobilenet_v2_int8_vela.tflite --delegate /usr/lib/libethosu_delegate.so" "$SHOW_IMG"

cpu_med="$(median_ms "$TMP/cpu.txt")"; npu_med="$(median_ms "$TMP/npu.txt")"
agree="$(paste -d'\n' "$TMP/cpu.txt" "$TMP/npu.txt" | awk -F'\t' 'NR % 2 { c[(NR + 1) / 2] = $3; next }
  { t++; if (c[NR / 2] == $3) s++ } END { printf "%d of %d", s, t }')"

banner "Summary (computed from the run above)"
printf '  CPU  2 threads  median %s%s ms%s per image (invoke only)\n' "$YELLOW" "$cpu_med" "$R"
printf '  NPU             median %s%s ms%s per image (invoke only)\n' "$GREEN" "$npu_med" "$R"
if awk -v c="$cpu_med" -v n="$npu_med" 'BEGIN { exit !(c + 0 > 0 && n + 0 > 0) }'; then
  printf '  NPU is %s%sx%s faster than the 2-thread CPU in this run\n' "$B" "$(awk -v c="$cpu_med" -v n="$npu_med" 'BEGIN { printf "%.1f", c / n }')" "$R"
fi
printf '  CPU and NPU agree on top-1 for %s images\n' "$agree"
note "  Full benchmark (200 runs per config, all logs): results/results.md"
