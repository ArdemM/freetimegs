"""Periodische Relokation (FreeTimeGS, Abschnitt 3.2, Gl. 7).

Alle N Schritte werden Gaußsche mit Opazität unter einer Schwelle ("tot") an
Stellen mit hohem Sampling-Score verschoben:

    score = λ_g · g + λ_o · σ

g ist der mittlere 2D-Positionsgradient seit der letzten Relokation (wie bei
der 3DGS-Densifizierung), σ die Basis-Opazität. Das Paper lässt offen, wie g
skaliert wird und welche Parameter die verschobene Gaußsche erbt. Hier:

- g wird durch sein 99-%-Quantil geteilt und auf [0, 1] begrenzt, damit beide
  Terme denselben Wertebereich haben.
- Vererbung wie bei 3DGS-MCMC (Kheradmand et al. 2024) und der
  FreeTimeGS++-Reproduktion: Die tote Gaußsche übernimmt *alle* Parameter des
  Ziels (inkl. μₜ, s, v); Opazität und Skalierung von Ziel und Kopien werden
  mit gsplat.relocation.compute_relocation neu verteilt.

Aufbau angelehnt an gsplat/strategy/ops.py (relocate), Apache-2.0.
"""

import math
from typing import Dict

import torch
from torch import Tensor

from gsplat.relocation import compute_relocation
from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer


class Relocator:
    def __init__(
        self,
        num_gaussians: int,
        device: str,
        lambda_grad: float = 0.5,
        lambda_opa: float = 0.5,
        min_opacity: float = 5e-3,
    ):
        self.lambda_grad = lambda_grad
        self.lambda_opa = lambda_opa
        self.min_opacity = min_opacity
        self.grad2d = torch.zeros(num_gaussians, device=device)
        self.count = torch.zeros(num_gaussians, device=device)
        n_max = 51
        self.binoms = torch.zeros((n_max, n_max), device=device)
        for n in range(n_max):
            for k in range(n + 1):
                self.binoms[n, k] = math.comb(n, k)

    @torch.no_grad()
    def accumulate(self, info: Dict, mask: Tensor) -> None:
        """2D-Gradienten nach backward() aufsummieren (info["means2d"] mit retain_grad)."""
        grads = info["means2d"].grad.clone()  # gepackt: [nnz, 2]
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        # gaussian_ids zählen in der zeitlich gefilterten Teilmenge -> globale Indizes
        ids = mask.nonzero(as_tuple=True)[0][info["gaussian_ids"]]
        self.grad2d.index_add_(0, ids, grads.norm(dim=-1))
        self.count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @torch.no_grad()
    def score(self, splats) -> Tensor:
        g = self.grad2d / self.count.clamp_min(1)
        q = torch.quantile(g[self.count > 0], 0.99) if (self.count > 0).any() else g.new_tensor(1.0)
        g = (g / q.clamp_min(1e-12)).clamp(max=1.0)
        opa = torch.sigmoid(splats["opacities"])
        return self.lambda_grad * g + self.lambda_opa * opa

    @torch.no_grad()
    def step(self, splats, optimizers: Dict[str, torch.optim.Optimizer]) -> int:
        """Tote Gaußsche verschieben, Akkumulatoren zurücksetzen. Rückgabe: Anzahl."""
        opa = torch.sigmoid(splats["opacities"])
        dead = opa <= self.min_opacity
        n = int(dead.sum())
        if n > 0 and n < len(dead):
            dead_idx = dead.nonzero(as_tuple=True)[0]
            alive_idx = (~dead).nonzero(as_tuple=True)[0]
            probs = self.score(splats)[alive_idx]
            sampled = alive_idx[_multinomial_sample(probs, n, replacement=True)]
            new_opa, new_scales = compute_relocation(
                opacities=opa[sampled],
                scales=torch.exp(splats["scales"])[sampled],
                ratios=torch.bincount(sampled)[sampled] + 1,
                binoms=self.binoms,
                min_opacity=self.min_opacity,
            )

            def param_fn(name: str, p: Tensor) -> Tensor:
                if name == "opacities":
                    p[sampled] = torch.logit(new_opa)
                elif name == "scales":
                    p[sampled] = torch.log(new_scales)
                p[dead_idx] = p[sampled]
                return torch.nn.Parameter(p, requires_grad=p.requires_grad)

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                v[sampled] = 0
                v[dead_idx] = 0
                return v

            _update_param_with_optimizer(param_fn, optimizer_fn, splats, optimizers)
        self.grad2d.zero_()
        self.count.zero_()
        return n
