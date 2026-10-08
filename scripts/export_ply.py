"""Momentaufnahmen eines trainierten Modells als 3DGS-PLY exportieren.

Für jeden gewählten Frame wird die 4D-Szene zum Zeitpunkt t in gewöhnliche
3D-Gaußsche umgerechnet (Position μₓ + v·(t − μₜ), Opazität σ · σ(t)) und im
Standard-PLY-Format von 3DGS gespeichert (gsplat.export_splats). Die Dateien
lassen sich z. B. in SuperSplat (https://superspl.at/editor) öffnen.

Koordinaten: Standardmäßig wird die Szene so gedreht, dass die mittlere
Oben-Richtung der Kameras auf +y zeigt (--up y). Im Trainings-Weltsystem ist sie
gekippt (ca. −y und −z). --up none behält das Trainings-Weltsystem bei.

Aufruf:
    python scripts/export_ply.py --ckpt ~/masterarbeit/results/ablation/kf1_full/ckpts/ckpt_final.pt \
        --out-dir ~/masterarbeit/results/ablation/kf1_full/ply --frames 0 25 49
    --frames all   exportiert jeden Frame (ca. 50 MB pro Datei)
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftgs.model import temporal_opacity  # noqa: E402
from gsplat import export_splats  # noqa: E402


def rotation_to(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Kleinste Drehung (3×3), die den Einheitsvektor src auf dst abbildet."""
    v = np.cross(src, dst)
    c = float(src @ dst)
    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return np.eye(3)
        axis = np.cross(src, [1.0, 0.0, 0.0])  # 180°: beliebige Achse senkrecht zu src
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(src, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def matrix_to_quat(R: np.ndarray) -> torch.Tensor:
    """Drehmatrix -> Quaternion (w, x, y, z), Konvention wie in gsplat."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return torch.tensor([w, x, y, z], dtype=torch.float32)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton-Produkt a ⊗ b, a [4], b [N, 4], (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def sh_rotation(R: np.ndarray, degree: int, device: str) -> torch.Tensor:
    """Matrix M [K, K] mit coeffs_neu = M @ coeffs für eine Szenendrehung R.

    Die Farbe einer gedrehten Gaußschen in Richtung d' = R·d muss der alten Farbe
    in Richtung d entsprechen: B(d') · c_neu = B(Rᵀ·d') · c. M wird numerisch aus
    zufälligen Richtungen per kleinster Quadrate bestimmt, mit gsplats eigener
    SH-Auswertung (damit Konvention und Vorzeichen garantiert passen).
    """
    from gsplat import spherical_harmonics

    K = (degree + 1) ** 2
    eye = torch.eye(K, device=device)

    def basis(dirs: torch.Tensor) -> torch.Tensor:  # [N, 3] -> [N, K]
        coeffs = eye.expand(len(dirs), K, K).contiguous()  # Kanal j = Basisfunktion j
        viewmat = torch.eye(4, device=device)[None]  # Kamera im Ursprung: Richtung = Position
        return spherical_harmonics(degree, dirs, viewmat, coeffs)[0]

    g = torch.Generator(device="cpu").manual_seed(0)
    d = torch.nn.functional.normalize(torch.randn(4096, 3, generator=g), dim=-1).to(device)
    Rt = torch.from_numpy(R).float().to(device)
    A, Bm = basis(d), basis(d @ Rt)  # d @ R entspricht Rᵀ·d zeilenweise
    return torch.linalg.lstsq(A, Bm).solution  # A @ M ≈ Bm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", nargs="+", default=["0"], help="Frame-Indizes oder 'all'")
    ap.add_argument("--num-frames", type=int, default=None, help="Standard: aus data_dir des Checkpoints")
    ap.add_argument("--min-temporal", type=float, default=1e-3, help="σ(t)-Schwelle wie beim Rendern")
    ap.add_argument("--format", choices=["ply", "ply_compressed", "splat"], default="ply")
    ap.add_argument("--up", choices=["y", "-y", "z", "none"], default="y",
                    help="Achse, auf die die Oben-Richtung der Kameras gedreht wird")
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

    # Ausrichtung: mittlere Kamera-Oben-Richtung (−y der Kameras, OpenCV) auf die Zielachse
    R = np.eye(3)
    if args.up != "none":
        from datasets.colmap import Parser  # noqa: E402  (Pfad setzt ftgs.data)

        # nur Posen lesen, keine Bilder laden
        p = Parser(str(Path(cfg["data_dir"]).expanduser() / "frame0"), factor=1, normalize=True,
                   test_every=10**9)
        up = -p.camtoworlds[:, :3, 1].mean(0)
        up /= np.linalg.norm(up)
        target = {"y": [0, 1, 0], "-y": [0, -1, 0], "z": [0, 0, 1]}[args.up]
        R = rotation_to(up, np.array(target, dtype=np.float64))
        print(f"Oben-Richtung {np.round(up, 3)} -> {args.up}")
    R_t = torch.from_numpy(R).float()
    q_R = matrix_to_quat(R)
    degree = int(round(math.sqrt(1 + splats["shN"].shape[1]))) - 1
    # gsplats SH-Auswertung läuft nur auf der GPU, M ist klein und kommt danach auf die CPU
    M = sh_rotation(R, degree, "cuda").cpu() if args.up != "none" else None

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
            means = splats["means"][mask] + splats["velocities"][mask] * dt
            quats = torch.nn.functional.normalize(splats["quats"][mask], dim=-1)
            sh = torch.cat([splats["sh0"][mask], splats["shN"][mask]], 1)  # [N, K, 3]
            if M is not None:
                sh = torch.einsum("jk,nkc->njc", M, sh)  # c_neu = M @ c, je Farbkanal
            export_splats(
                means=means @ R_t.T,
                scales=splats["scales"][mask],  # log-Skalen wie im Training
                quats=quat_mul(q_R, quats),
                opacities=torch.logit(opa.clamp(1e-6, 1 - 1e-6)),
                sh0=sh[:, :1],
                shN=sh[:, 1:],
                format=args.format,
                save_to=str(path),
            )
        size = path.stat().st_size / 1024**2
        print(f"Frame {f:3d} (t = {t:.3f}): {int(mask.sum())} Gaußsche -> {path} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
