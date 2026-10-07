#!/bin/bash
# Lauf des eigenen Trainers mit VRAM-Protokoll (nvidia-smi) und Wanduhrzeit.
#
# Aufruf: bash scripts/run_proto.sh <result_dir> [weitere Argumente für train.py]
# Beispiel: bash scripts/run_proto.sh ~/masterarbeit/results/ftgs_l1_kf1 --img-loss l1
set -euo pipefail

OUT=${1:?result_dir}
shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
source ~/masterarbeit/.venv/bin/activate
cd "$REPO"

{
  echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo '-dirty')"
  echo "args: $*"
} | tee "$OUT/run_info.txt"

# GPU-Speicher alle 2 s protokollieren (Maximum = VRAM-Spitze inkl. CUDA-Kontext)
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 2 > "$OUT/vram.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

START=$(date +%s)
python train.py --result-dir "$OUT" "$@" 2>&1 | tee "$OUT/train.log"
END=$(date +%s)

echo "wall_time_s: $((END - START))" | tee -a "$OUT/run_info.txt"
echo "vram_peak_mib: $(cut -d, -f2 "$OUT/vram.csv" | tr -dc '0-9\n' | sort -n | tail -1)" | tee -a "$OUT/run_info.txt"
