#!/bin/sh
# A55 <-> M33 RPMsg ping-pong using NXP's prebuilt M33 demo firmware shipped in the BSP.
# Temporarily replaces the Ethos-U firmware on the M33 (remoteproc0), then ALWAYS restores it.
# Nothing here is persistent: a reboot also restores the default firmware.
# Usage (on the board): sh rpmsg_pingpong.sh [OUT_DIR]     default OUT_DIR=results_rpmsg
set -u

OUT="${1:-results_rpmsg}"
RP=/sys/class/remoteproc/remoteproc0
DEMO_FW="imx93-11x11-evk_m33_TCM_rpmsg_lite_pingpong_rtos_linux_remote.elf"
DEFAULT_FW="ethosu_firmware"
mkdir -p "$OUT"

restore() {
  modprobe -r imx_rpmsg_pingpong 2>/dev/null
  echo stop > "$RP/state" 2>/dev/null
  echo "$DEFAULT_FW" > "$RP/firmware"
  echo start > "$RP/state"
  sleep 2
  echo "restored: firmware=$(cat "$RP/firmware") state=$(cat "$RP/state")" >> "$OUT/summary.txt"
  # Prove the NPU works again after the restore (needs infer.py, samples/ and the Vela model in cwd).
  delegate="$(find /usr -name libethosu_delegate.so 2>/dev/null | head -n 1)"
  python3 infer.py --device npu --model vela_out/mobilenet_v2_int8_vela.tflite --images samples \
    --delegate "$delegate" > "$OUT/npu_after_restore.log" 2>&1
  echo "npu_after_restore exit code: $?" >> "$OUT/summary.txt"
}
trap restore EXIT

{
  echo "uname=$(uname -r)"
  echo "before: firmware=$(cat "$RP/firmware") state=$(cat "$RP/state")"
  ls -l "/lib/firmware/$DEMO_FW"
  echo "demo firmware sha256=$(sha256sum "/lib/firmware/$DEMO_FW" | cut -d' ' -f1)"
} > "$OUT/summary.txt"

dmesg > "$OUT/dmesg_before.txt"
modprobe -r imx_rpmsg_pingpong 2>/dev/null
echo stop > "$RP/state" || { echo "cannot stop M33" >> "$OUT/summary.txt"; exit 1; }
echo "$DEMO_FW" > "$RP/firmware" || { echo "cannot set firmware" >> "$OUT/summary.txt"; exit 1; }
echo start > "$RP/state" || { echo "cannot start demo firmware" >> "$OUT/summary.txt"; exit 1; }
echo "demo started: firmware=$(cat "$RP/firmware") state=$(cat "$RP/state")" >> "$OUT/summary.txt"
sleep 3
modprobe imx_rpmsg_pingpong || echo "modprobe imx_rpmsg_pingpong failed" >> "$OUT/summary.txt"
sleep 8   # the demo counts 1..100 on the M33 side
dmesg > "$OUT/dmesg_after.txt"
diff "$OUT/dmesg_before.txt" "$OUT/dmesg_after.txt" | grep '^>' | sed 's/^> //' > "$OUT/dmesg_new.txt"
echo "new dmesg lines: $(wc -l < "$OUT/dmesg_new.txt")" >> "$OUT/summary.txt"
