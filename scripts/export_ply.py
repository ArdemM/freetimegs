"""Momentaufnahmen eines trainierten Modells als 3DGS-PLY exportieren.

Für jeden gewählten Frame wird die 4D-Szene zum Zeitpunkt t in gewöhnliche
3D-Gaußsche umgerechnet (Position μₓ + v·(t − μₜ), Opazität σ · σ(t)) und im
Standard-PLY-Format von 3DGS gespeichert (gsplat.export_splats). Die Dateien
lassen sich z. B. in SuperSplat (https://superspl.at/editor) öffnen.

Koordinaten: normalisiertes COLMAP-Weltsystem des Trainings (y zeigt nach unten,
im Viewer ggf. um 180° um x drehen).

Aufruf:
    python scripts/export_ply.py --ckpt ~/masterarbeit/results/ablation/kf1_full/ckpts/ckpt_final.pt \
        --out-dir ~/masterarbeit/results/ablation/kf1_full/ply --frames 0 25 49
    --frames all   exportiert jeden Frame (ca. 50 MB pro Datei)
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftgs.model import temporal_opacity  # noqa: E402
from gsplat import export_splats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", nargs="+", default=["0"], help="Frame-Indizes oder 'all'")
    ap.add_argument("--num-frames", type=int, default=None, help="Standard: aus data_dir des Checkpoints")
    ap.add_argument("--min-temporal", type=float, default=1e-3, help="σ(t)-Schwelle wie beim Rendern")
    ap.add_argument("--format", choices=["ply", "ply_compressed", "splat"], default="ply")
    args = ap.parse_args()

    ckpt = torch.load(Path(args.ckpt).expanduser(), map_location="cpu")
    splats, cfg = ckpt["splats"], ckpt["cfg"]

    F = args.num_frames
    if F is None:
        data_dir = Path(cfg["data_dir"]).expanduser()
        F = len(list((data_dir / "images" / "cam00").glob("*.png")))
        if cfg.get("num_frames", -1) > 0:
            F = min(F, cfg["num_frames"])
    frames = list(range(F)) if args.frames == ["all"] else [int(f) for f in args.frames]

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ext = {"ply": "ply", "ply_compressed": "compressed.ply", "splat": "splat"}[args.format]

    for f in frames:
        t = f / max(F - 1, 1)
        with torch.no_grad():
            temp = temporal_opacity(splats, t)
            mask = temp > args.min_temporal
            dt = t - splats["times"][mask]
            opa = torch.sigmoid(splats["opacities"][mask]) * temp[mask]
            path = out / f"frame{f:03d}.{ext}"
            export_splats(
                means=splats["means"][mask] + splats["velocities"][mask] * dt,
                scales=splats["scales"][mask],  # log-Skalen wie im Training
                quats=splats["quats"][mask],
                opacities=torch.logit(opa.clamp(1e-6, 1 - 1e-6)),
                sh0=splats["sh0"][mask],
                shN=splats["shN"][mask],
                format=args.format,
                save_to=str(path),
            )
        size = path.stat().st_size / 1024**2
        print(f"Frame {f:3d} (t = {t:.3f}): {int(mask.sum())} Gaußsche -> {path} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
