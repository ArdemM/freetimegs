#!/bin/bash
# Auswertung mit zwei Testkameras: cam00 (untere Reihe, zwischen Trainingskameras)
# und cam15 (Mitte der oberen Reihe). cam15 deckt Ghosting aus neuen Blickwinkeln auf,
# das cam00 allein nicht misst. Trainiert wird mit den übrigen 19 Kameras.
#
# Aufruf: bash scripts/run_two_testcams.sh [nur diese Läufe]
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
RES=~/masterarbeit/results/testcams_00_15
mkdir -p "$RES"

TEST="--test-cams cam00 cam15"
FULL="--lambda-reg 1e-2 --relocate --init-velocity knn --velocities-lr-final 5e-5"

declare -A RUNS=(
  [s4_kf1_v0]="--keyframe-step 1 --no-learn-velocity $TEST --no-render-traj"
  [full_kf5]="--keyframe-step 5 $FULL $TEST --no-render-traj"
  [full_kf1]="--keyframe-step 1 $FULL $TEST"
)
ORDER=(full_kf1 full_kf5 s4_kf1_v0)
[ $# -gt 0 ] && ORDER=("$@")

for name in "${ORDER[@]}"; do
  if [ -f "$RES/$name/stats/val_step30000.json" ]; then
    echo "== $name: schon fertig, übersprungen"
    continue
  fi
  echo "== $name: ${RUNS[$name]}"
  # shellcheck disable=SC2086
  bash "$REPO/scripts/run_proto.sh" "$RES/$name" ${RUNS[$name]} > "$RES/$name.log" 2>&1 \
    || echo "!! $name fehlgeschlagen (siehe $RES/$name.log)"
done
echo "== fertig"
