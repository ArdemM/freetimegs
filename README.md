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
  colmap/                                Datenbank, Log, Eingabebilder
  poses_bounds.npy                       Original-LLFF-Posen (nur zum Vergleich)
  meta.json
```

Das Layout entspricht dem „flat“-Format, das auch FreeTimeGsVanilla liest (`--data-dir`, `--data-factor 1`).
`cam00` ist die Testkamera.

Plausibilitätstest (statisch, Frame 0):

```bash
cd ~/masterarbeit/gsplat/examples
python simple_trainer.py mcmc --data_dir ~/masterarbeit/data/n3dv/flame_steak_50f/frame0 \
    --data_factor 1 --result_dir ~/masterarbeit/results/flame_steak_frame0 --max_steps 7000
```
