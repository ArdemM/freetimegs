#!/bin/bash
# Ablationsreihe Schritt 5 (flame_steak 50f, Test cam00, 30k Schritte).
# Basis: keyframe_step 5 (dort zeigte der Prototyp Überanpassung).
#   kumulativ:     +reg, +reloc, +knn, +anneal (= full)
#   leave-one-out: full ohne reg / reloc / knn (ohne anneal = kumulativ Schritt 3)
#   Seeds:         base und full mit Seeds 1 und 2 (Seed 42 = Hauptlauf)
#   zusätzlich:    full mit keyframe_step 1
#
# Aufruf: bash scripts/run_ablation.sh [nur diese Läufe, z. B. kf5_full kf1_full]
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
RES=~/masterarbeit/results/ablation
mkdir -p "$RES"

REG="--lambda-reg 1e-2"
RELOC="--relocate"
KNN="--init-velocity knn"
ANNEAL="--velocities-lr-final 5e-5"
NOTRAJ="--no-render-traj"

declare -A RUNS=(
  [kf5_reg]="--keyframe-step 5 $REG $NOTRAJ"
  [kf5_reg_reloc]="--keyframe-step 5 $REG $RELOC $NOTRAJ"
  [kf5_reg_reloc_knn]="--keyframe-step 5 $REG $RELOC $KNN $NOTRAJ"
  [kf5_full]="--keyframe-step 5 $REG $RELOC $KNN $ANNEAL"
  [kf5_full_noreg]="--keyframe-step 5 $RELOC $KNN $ANNEAL $NOTRAJ"
  [kf5_full_noreloc]="--keyframe-step 5 $REG $KNN $ANNEAL $NOTRAJ"
  [kf5_full_noknn]="--keyframe-step 5 $REG $RELOC $ANNEAL $NOTRAJ"
  [kf5_base_s1]="--keyframe-step 5 --seed 1 $NOTRAJ"
  [kf5_base_s2]="--keyframe-step 5 --seed 2 $NOTRAJ"
  [kf5_full_s1]="--keyframe-step 5 $REG $RELOC $KNN $ANNEAL --seed 1 $NOTRAJ"
  [kf5_full_s2]="--keyframe-step 5 $REG $RELOC $KNN $ANNEAL --seed 2 $NOTRAJ"
  [kf1_full]="--keyframe-step 1 $REG $RELOC $KNN $ANNEAL"
)
ORDER=(kf5_reg kf5_reg_reloc kf5_reg_reloc_knn kf5_full kf5_full_noreg kf5_full_noreloc
       kf5_full_noknn kf5_base_s1 kf5_base_s2 kf5_full_s1 kf5_full_s2 kf1_full)
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
