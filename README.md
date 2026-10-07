# freetimegs

Eigene FreeTimeGS-Implementierung (4D Gaussian Splatting) auf Basis von [gsplat](https://github.com/nerfstudio-project/gsplat) 1.6.0, Masterarbeit.

gsplat bleibt unverändert als Bibliothek in `~/masterarbeit/gsplat`; eigener Code liegt nur hier.

## Umgebung

```bash
source ~/masterarbeit/.venv/bin/activate
cd ~/masterarbeit/freetimegs     # Python nie in ~/masterarbeit selbst starten
```

## Verzeichnisse

| Pfad | Inhalt |
|---|---|
| `scripts/` | Datenvorbereitung |
| `~/masterarbeit/data/n3dv/raw/<szene>/` | Original-Videos Neural3DV (`camXX.mp4`, `poses_bounds.npy`) |
| `~/masterarbeit/data/n3dv/<szene>_<N>f/` | vorbereitete Szene (siehe unten) |
| `~/masterarbeit/results/` | Trainingsläufe (TensorBoard, Checkpoints) |
| `EXPERIMENTS.md` | Protokoll aller Läufe |

## Neural3DV vorbereiten

```bash
# Download (ca. 1,2 GB pro Szene)
mkdir -p ~/masterarbeit/data/n3dv/raw && cd ~/masterarbeit/data/n3dv/raw
curl -L -O https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_steak.zip
unzip flame_steak.zip

# Frames extrahieren (halbe Auflösung) + COLMAP auf Frame 0
cd ~/masterarbeit/freetimegs
python scripts/prepare_n3dv.py \
    --src ~/masterarbeit/data/n3dv/raw/flame_steak \
    --out ~/masterarbeit/data/n3dv/flame_steak_50f \
    --num-frames 50 --downscale 2
```

Ergebnis:

```
flame_steak_50f/
  images/cam00/00000.png ... 00049.png   1352x1014, je Kamera ein Ordner
  sparse/0/                              COLMAP-Modell von Frame 0 (PINHOLE, Bildnamen camXX.png)
  frame0/                                nur Frame 0, direkt mit gsplat simple_trainer.py nutzbar
  colmap/                                Datenbank, Log, Eingabebilder; frames/NNNNN/sparse pro Frame
  points/                                Punktwolken pro Frame (triangulate_frames.py)
  poses_bounds.npy                       Original-LLFF-Posen (nur zum Vergleich)
  meta.json
```

Das Layout entspricht dem „flat“-Format, das auch FreeTimeGsVanilla liest (`--data-dir`, `--data-factor 1`).
`cam00` ist die Testkamera.

Punktwolke pro Frame (feste Posen aus `sparse/0`, SIFT + `colmap point_triangulator`, ca. 4 s pro Frame):

```bash
python scripts/triangulate_frames.py --scene ~/masterarbeit/data/n3dv/flame_steak_50f \
    --workers 8 --two-view-tracks
```

Das Ergebnis liegt in `points/points3d_frame000000.npy` und `points/colors_frame000000.npy` (`[M,3]` float32, Farben 0–255), wie FreeTimeGsVanilla es erwartet. Kennzahlen pro Frame stehen in `points/stats.json`.
Bei flame_steak sind es ca. 6700 Punkte pro Frame mit 0,55 px Reprojektionsfehler. `--two-view-tracks` behält auch Punkte, die nur zwei Kameras sehen, und verdoppelt damit etwa die Punktzahl.

Plausibilitätstest (statisch, Frame 0):

```bash
cd ~/masterarbeit/gsplat/examples
python simple_trainer.py mcmc --data_dir ~/masterarbeit/data/n3dv/flame_steak_50f/frame0 \
    --data_factor 1 --result_dir ~/masterarbeit/results/flame_steak_frame0 --max_steps 7000
```

## Referenz: FreeTimeGsVanilla

[FreeTimeGsVanilla](https://github.com/OpsiClear-4DGS/FreeTimeGsVanilla) (AGPL-3.0) dient nur als Referenz, die ausgeführt und gelesen wird. **Von dort wird kein Code übernommen.**
Das Repo liegt außerhalb dieses Repositorys in `~/masterarbeit/refs/FreeTimeGsVanilla` (getestet mit Commit `42e2f9b`) und hat eine eigene `.venv` mit torch 2.10+cu128 und gsplat 1.5.3.

Installation (`uv sync --locked` scheitert, weil uv torch-scatter isoliert gegen ein torch mit CUDA 13 baut):

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
cd ~/masterarbeit/refs/FreeTimeGsVanilla
uv venv --python 3.12 --managed-python .venv && source .venv/bin/activate   # managed: bringt Python.h mit
uv export --locked --no-hashes --no-emit-project > /tmp/req.txt
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install setuptools wheel ninja numpy==1.26.4
grep -vE '^(torch|torchvision|nvidia-|triton)==' /tmp/req.txt > /tmp/req2.txt
uv pip install --no-build-isolation -r /tmp/req2.txt                         # ca. 7 min
```

Lauf (Combiner, dann Training mit cam00 als einziger Testkamera, Auswertung aller Frames, VRAM-Protokoll):

```bash
cd ~/masterarbeit/freetimegs
bash scripts/run_vanilla_ref.sh ~/masterarbeit/data/n3dv/flame_steak_50f \
    ~/masterarbeit/results/vanilla_flame_steak_50f_kf5 30000 5
```

Wichtige Standardwerte von Vanilla, die das Skript überschreibt: `test_every=8` (Kameras 0, 8, 16) und `eval_sample_every=60` (bei 50 Frames würde nur Frame 0 ausgewertet).
