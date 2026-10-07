"""FreeTimeGS-Repräsentation (Wang et al., CVPR 2025, Abschnitt 3.1).

Jede Gaußsche hat zusätzlich zu den 3DGS-Parametern eine Zeit μₜ, eine Dauer s
und eine Geschwindigkeit v. Für einen Zeitpunkt t gilt

    μₓ(t) = μₓ + v·(t − μₜ)                 (Gl. 1)
    σ(t)  = exp(−½·((t − μₜ)/s)²)           (Gl. 4)
    Opazität(t) = σ · σ(t)                  (Gl. 3)

Das Ergebnis ist eine gewöhnliche 3D-Gaußsche Szene, die an
gsplat.rasterization übergeben wird. Gradienten auf μₜ, s und v laufen über
PyTorch-Autograd, ein eigener CUDA-Kernel ist nicht nötig.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from .data import GSPLAT_EXAMPLES  # noqa: F401  (setzt sys.path für utils)
from utils import knn, rgb_to_sh  # gsplat/examples/utils.py


def init_from_frames(
    frames: List[Tuple[np.ndarray, np.ndarray, float]],
    init_duration: float,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    sh_degree: int = 3,
) -> Dict[str, Tensor]:
    """Gaußsche aus Punktwolken pro Frame, Zeit = Zeit des Frames, v = 0.

    frames: Liste von (Punkte [M, 3], Farben [M, 3] in [0, 1], normierte Zeit t).
    Die Skalierung kommt aus dem Abstand zu den 3 nächsten Nachbarn *innerhalb
    desselben Frames* (statische Bereiche wären sonst über alle Frames hinweg
    mehrfach vorhanden und die Abstände künstlich klein).
    """
    means, rgbs, times, scales = [], [], [], []
    for pts, rgb, t in frames:
        p = torch.from_numpy(pts).float()
        dist2 = (knn(p, 4)[:, 1:] ** 2).mean(dim=-1)
        means.append(p)
        rgbs.append(torch.from_numpy(rgb).float())
        times.append(torch.full((len(p), 1), float(t)))
        scales.append(torch.log(torch.sqrt(dist2) * init_scale).unsqueeze(-1).repeat(1, 3))
    means = torch.cat(means)
    N = len(means)

    sh = torch.zeros((N, (sh_degree + 1) ** 2, 3))
    sh[:, 0, :] = rgb_to_sh(torch.cat(rgbs))
    return {
        "means": means,
        "scales": torch.cat(scales),
        "quats": torch.rand((N, 4)),
        "opacities": torch.logit(torch.full((N,), init_opacity)),
        "sh0": sh[:, :1, :],
        "shN": sh[:, 1:, :],
        "times": torch.cat(times),  # μₜ [N, 1]
        "durations": torch.full((N, 1), float(np.log(init_duration))),  # log s [N, 1]
        "velocities": torch.zeros((N, 3)),  # v [N, 3], Weltkoordinaten pro Zeiteinheit
    }


def temporal_opacity(splats: Dict[str, Tensor], t: float) -> Tensor:
    """σ(t) aus Gl. 4, Form [N]."""
    s = torch.exp(splats["durations"]).squeeze(-1)
    dt = t - splats["times"].squeeze(-1)
    return torch.exp(-0.5 * (dt / s) ** 2)


def gaussians_at(
    splats: Dict[str, Tensor], t: float, cull_threshold: float = 1e-3
) -> Tuple[Dict[str, Tensor], Tensor]:
    """3D-Gaußsche zum Zeitpunkt t.

    Gaußsche mit σ(t) < cull_threshold tragen praktisch nichts bei und werden
    vor der Rasterisierung entfernt (zeitliches Culling).
    Rückgabe: Parameter für gsplat.rasterization und die Maske [N] (bool).
    """
    temp = temporal_opacity(splats, t)
    mask = temp > cull_threshold
    dt = t - splats["times"][mask]  # [M, 1]
    return {
        "means": splats["means"][mask] + splats["velocities"][mask] * dt,
        "quats": splats["quats"][mask],
        "scales": torch.exp(splats["scales"][mask]),
        "opacities": torch.sigmoid(splats["opacities"][mask]) * temp[mask],
        "colors": torch.cat([splats["sh0"][mask], splats["shN"][mask]], 1),
    }, mask
