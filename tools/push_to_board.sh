#!/bin/sh
# Copy the model, board scripts and sample images to the board.
# Usage: tools/push_to_board.sh <board-ip> <imagenette2-320 dir>
set -eu
BOARD="$1"; DATASET="$2"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
while read -r f; do cp "$DATASET/$f" "$STAGE/"; done < model/sample_list.txt
ssh "root@$BOARD" 'mkdir -p imx93/samples'
scp -q model/mobilenet_v2_int8.tflite board/infer.py board/bench.py board/run_benchmarks.sh "root@$BOARD:imx93/"
scp -q "$STAGE"/*.JPEG "root@$BOARD:imx93/samples/"
echo "pushed to $BOARD:imx93"
