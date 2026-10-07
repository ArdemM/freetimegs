#!/bin/bash
# Referenzlauf FreeTimeGsVanilla (AGPL-3.0, nur ausgeführt, kein Code übernommen).
# Repo: ~/masterarbeit/refs/FreeTimeGsVanilla, eigene .venv (torch 2.10+cu128, gsplat 1.5.3).
#
# Aufruf: bash scripts/run_vanilla_ref.sh <szene_dir> <result_dir> [max_steps] [keyframe_step] [config]
# Testansicht: nur cam00 (--test-every = Anzahl Kameras), alle Frames ausgewertet (--eval-sample-every 1).
set -euo pipefail

DATA=${1:?szene_dir}
OUT=${2:?result_dir}
STEPS=${3:-30000}
KF=${4:-5}
CONFIG=${5:-default_keyframe_small}

VANILLA=~/masterarbeit/refs/FreeTimeGsVanilla
NFRAMES=$(ls "$DATA/images/cam00" | wc -l)
NCAMS=$(ls "$DATA/images" | wc -l)
mkdir -p "$OUT"
source "$VANILLA/.venv/bin/activate"
cd "$VANILLA"

{
  echo "vanilla_commit: $(git -C "$VANILLA" rev-parse --short HEAD)"
  echo "data: $DATA  frames: $NFRAMES  cams: $NCAMS  steps: $STEPS  keyframe_step: $KF  config: $CONFIG"
} | tee "$OUT/run_info.txt"

NPZ="$OUT/keyframes_${NFRAMES}f_step${KF}.npz"
python src/combine_frames_fast_keyframes.py \
    --input-dir "$DATA/points" --output-path "$NPZ" \
    --frame-start 0 --frame-end $((NFRAMES - 1)) --keyframe-step "$KF" 2>&1 | tee "$OUT/combine.log"

# GPU-Speicher alle 2 s protokollieren (Maximum = VRAM-Spitze inkl. CUDA-Kontext)
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 2 > "$OUT/vram.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

START=$(date +%s)
python src/simple_trainer_freetime_4d_pure_relocation.py "$CONFIG" \
    --data-dir "$DATA" --init-npz-path "$NPZ" --result-dir "$OUT" \
    --start-frame 0 --end-frame "$NFRAMES" --data-factor 1 \
    --test-every "$NCAMS" --eval-sample-every 1 \
    --max-steps "$STEPS" --eval-steps "$STEPS" --save-steps "$STEPS" 2>&1 | tee "$OUT/train.log"
END=$(date +%s)

echo "wall_time_s: $((END - START))" | tee -a "$OUT/run_info.txt"
echo "vram_peak_mib: $(cut -d, -f2 "$OUT/vram.csv" | tr -dc '0-9\n' | sort -n | tail -1)" | tee -a "$OUT/run_info.txt"
