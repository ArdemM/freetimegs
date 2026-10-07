# Minimaler FreeTimeGS-Trainer (Machbarkeitsplan Schritt 4).
#
# Aufbau (Optimizer, Lernraten, SH-Schedule, Loss, Evaluation) angelehnt an
# gsplat/examples/simple_trainer.py:
#   SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of
#   California, Nerfstudio Team and contributors. All rights reserved.
#   SPDX-License-Identifier: Apache-2.0
#
# Stand Schritt 4: feste Anzahl Gaußscher (keine Densifizierung/Relokation),
# keine 4D-Regularisierung, Geschwindigkeit mit 0 initialisiert, konstante
# Lernraten für μₜ, s und v. Die FreeTimeGS-Bausteine folgen einzeln in Schritt 5.
#
# Aufruf:
#   python train.py --data-dir ~/masterarbeit/data/n3dv/flame_steak_50f \
#       --result-dir ~/masterarbeit/results/ftgs_proto

import json
import math
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import imageio
import numpy as np
import torch
import tqdm
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ftgs.data import MultiViewVideo
from ftgs.model import gaussians_at, init_from_frames
from datasets.traj import generate_interpolated_path  # gsplat/examples
from gsplat.losses import l1_loss, ssim_loss
from gsplat.rendering import rasterization
from utils import set_random_seed  # gsplat/examples/utils.py


@dataclass
class Config:
    # Vorbereitete Szene (scripts/prepare_n3dv.py + scripts/triangulate_frames.py)
    data_dir: str = "~/masterarbeit/data/n3dv/flame_steak_50f"
    result_dir: str = "~/masterarbeit/results/ftgs_proto"
    # Testkameras (werden nicht trainiert, alle Frames werden ausgewertet)
    test_cams: List[str] = field(default_factory=lambda: ["cam00"])
    # Anzahl Frames (-1 = alle vorhandenen)
    num_frames: int = -1
    # Checkpoint laden und nur auswerten
    ckpt: Optional[str] = None

    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    seed: int = 42

    # Initialisierung: Punktwolke jedes keyframe_step-ten Frames, μₜ = Zeit des Frames
    keyframe_step: int = 1
    # Anfangsdauer s in normierter Zeit; None = 2 × Keyframe-Abstand
    init_duration: Optional[float] = None
    init_opacity: float = 0.1
    init_scale: float = 1.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000

    # Lernraten (3DGS/gsplat-Werte; μₜ, s, v eigene Wahl, konstant)
    means_lr: float = 1.6e-4  # × scene_scale, exponentiell auf 1 % abfallend
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    times_lr: float = 1e-3
    durations_lr: float = 5e-3
    velocities_lr: float = 5e-3
    # Geschwindigkeit optimieren (False = statische Gaußsche, nur zeitlich begrenzt)
    learn_velocity: bool = True

    # Rendering-Loss (Gl. 5): λ_img·L_img + λ_ssim·(1 − SSIM) + λ_perc·LPIPS
    img_loss: Literal["l1", "l2"] = "l1"
    lambda_img: float = 0.8
    lambda_ssim: float = 0.2
    lambda_perc: float = 0.0

    # Gaußsche mit σ(t) unter diesem Wert werden nicht rasterisiert
    temporal_cull: float = 1e-3
    near_plane: float = 0.01
    far_plane: float = 1e10

    tb_every: int = 100
    lpips_net: Literal["vgg", "alex"] = "alex"
    # Trajektorien-Video (Kamerafahrt durch die Trainingskameras, Zeit läuft mit)
    render_traj: bool = True
    load_workers: int = 16


class Runner:
    def __init__(self, cfg: Config) -> None:
        set_random_seed(cfg.seed)
        self.cfg = cfg
        self.device = "cuda"
        self.result_dir = Path(cfg.result_dir).expanduser()
        for sub in ["ckpts", "stats", "renders", "videos"]:
            (self.result_dir / sub).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.result_dir / "tb"))

        self.data = MultiViewVideo(
            cfg.data_dir, cfg.test_cams, cfg.num_frames, workers=cfg.load_workers
        )
        self.scene_scale = self.data.scene_scale * 1.1
        F = self.data.num_frames
        print(
            f"Szene: {len(self.data.cam_names)} Kameras, {F} Frames, "
            f"Test: {[self.data.cam_names[c] for c in self.data.test_cams]}, "
            f"scene_scale {self.scene_scale:.3f}"
        )

        if cfg.init_duration is None:
            self.init_duration = 2.0 * cfg.keyframe_step / max(F - 1, 1)
        else:
            self.init_duration = cfg.init_duration
        keyframes = self.data.keyframes(cfg.keyframe_step)
        frames = [(*self.data.frame_points(f), self.data.times[f]) for f in keyframes]
        init = init_from_frames(
            frames,
            init_duration=self.init_duration,
            init_opacity=cfg.init_opacity,
            init_scale=cfg.init_scale,
            sh_degree=cfg.sh_degree,
        )
        self.splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v) for k, v in init.items()}
        ).to(self.device)
        self.splats["velocities"].requires_grad_(cfg.learn_velocity)
        print(
            f"Init: {len(keyframes)} Keyframes, {len(self.splats['means'])} Gaußsche, "
            f"Dauer s = {self.init_duration:.4f}"
        )

        lrs = {
            "means": cfg.means_lr * self.scene_scale,
            "scales": cfg.scales_lr,
            "quats": cfg.quats_lr,
            "opacities": cfg.opacities_lr,
            "sh0": cfg.sh0_lr,
            "shN": cfg.shN_lr,
            "times": cfg.times_lr,
            "durations": cfg.durations_lr,
            "velocities": cfg.velocities_lr,
        }
        self.optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr, "name": name}],
                eps=1e-15,
                fused=True,
            )
            for name, lr in lrs.items()
            if self.splats[name].requires_grad
        }

        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=cfg.lpips_net, normalize=True
        ).to(self.device)

    def render(self, cam: int, t: float, sh_degree: int, c2w: Optional[torch.Tensor] = None):
        """Bild der Kamera cam (bzw. Pose c2w) zum Zeitpunkt t, Form [1, H, W, 3]."""
        if c2w is None:
            c2w = torch.from_numpy(self.data.camtoworlds[cam]).to(self.device)
        K = torch.from_numpy(self.data.Ks[cam]).to(self.device)
        W, H = self.data.imsizes[cam]
        g, mask = gaussians_at(self.splats, t, self.cfg.temporal_cull)
        colors, alphas, info = rasterization(
            means=g["means"],
            quats=g["quats"],
            scales=g["scales"],
            opacities=g["opacities"],
            colors=g["colors"],
            viewmats=torch.linalg.inv(c2w)[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
        )
        return colors, alphas, info, mask

    def train(self) -> None:
        cfg = self.cfg
        with open(self.result_dir / "cfg.yml", "w") as f:
            yaml.dump(
                {**asdict(cfg), "init_duration_effective": self.init_duration,
                 "commit": git_commit()},
                f,
            )

        max_steps = cfg.max_steps
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
        )
        items = self.data.train_items
        rng = np.random.default_rng(cfg.seed)

        torch.cuda.reset_peak_memory_stats()
        tic = time.time()
        visible_sum = 0
        pbar = tqdm.tqdm(range(max_steps), mininterval=10)
        for step in pbar:
            cam, frame = items[rng.integers(len(items))]
            t = float(self.data.times[frame])
            pixels = self.data.images[cam, frame].to(self.device, non_blocking=True)
            pixels = pixels.float()[None] / 255.0  # [1, H, W, 3]

            sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            colors, _, _, mask = self.render(cam, t, sh_degree)
            visible_sum += int(mask.sum())

            if cfg.img_loss == "l1":
                imgloss = l1_loss(colors, pixels).mean()
            else:
                imgloss = torch.nn.functional.mse_loss(colors, pixels)
            ssimloss = ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
            loss = cfg.lambda_img * imgloss + cfg.lambda_ssim * ssimloss
            if cfg.lambda_perc > 0:
                lpipsloss = self.lpips(
                    colors.clamp(0, 1).permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
                )
                loss = loss + cfg.lambda_perc * lpipsloss
            loss.backward()

            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            scheduler.step()

            if step % cfg.tb_every == 0:
                pbar.set_description(f"loss={loss.item():.3f} sh={sh_degree} sichtbar={int(mask.sum())}")
                w = self.writer
                w.add_scalar("train/loss", loss.item(), step)
                w.add_scalar("train/imgloss", imgloss.item(), step)
                w.add_scalar("train/ssimloss", ssimloss.item(), step)
                w.add_scalar("train/visible_GS", int(mask.sum()), step)
                w.add_scalar("train/mem_GB", torch.cuda.max_memory_allocated() / 1024**3, step)
                with torch.no_grad():
                    s = torch.exp(self.splats["durations"])
                    w.add_scalar("gauss/duration_median", s.median().item(), step)
                    w.add_scalar("gauss/opacity_mean", torch.sigmoid(self.splats["opacities"]).mean().item(), step)
                    w.add_scalar("gauss/speed_median", self.splats["velocities"].norm(dim=-1).median().item(), step)

            if step + 1 in cfg.eval_steps or step + 1 == max_steps:
                train_time = time.time() - tic
                stats = {
                    "step": step + 1,
                    "train_time_s": train_time,
                    "mem_peak_GB": torch.cuda.max_memory_allocated() / 1024**3,
                    "num_GS": len(self.splats["means"]),
                    "visible_GS_mean": visible_sum / (step + 1),
                }
                print(stats)
                with open(self.result_dir / "stats" / f"train_step{step + 1:05d}.json", "w") as f:
                    json.dump(stats, f, indent=2)
                self.eval(step + 1)
                tic = time.time() - train_time  # Evaluationszeit nicht mitzählen

        torch.save(
            {"step": max_steps, "splats": self.splats.state_dict(), "cfg": asdict(cfg)},
            self.result_dir / "ckpts" / "ckpt_final.pt",
        )
        if cfg.render_traj:
            self.render_traj(max_steps)

    @torch.no_grad()
    def eval(self, step: int) -> dict:
        cfg = self.cfg
        print(f"Evaluation (Schritt {step}) ...")
        metrics = defaultdict(list)
        per_frame = []
        render_time = 0.0
        for cam in self.data.test_cams:
            name = self.data.cam_names[cam]
            writer = imageio.get_writer(
                self.result_dir / "videos" / f"test_{name}_step{step}.mp4", fps=30, macro_block_size=2
            )
            for frame in range(self.data.num_frames):
                t = float(self.data.times[frame])
                pixels = self.data.images[cam, frame].to(self.device).float()[None] / 255.0
                torch.cuda.synchronize()
                tic = time.time()
                colors, _, _, mask = self.render(cam, t, cfg.sh_degree)
                torch.cuda.synchronize()
                render_time += time.time() - tic
                colors = colors.clamp(0.0, 1.0)

                p = pixels.permute(0, 3, 1, 2)
                c = colors.permute(0, 3, 1, 2)
                m = {
                    "psnr": self.psnr(c, p).item(),
                    "ssim": self.ssim(c, p).item(),
                    "lpips": self.lpips(c, p).item(),
                }
                for k, v in m.items():
                    metrics[k].append(v)
                per_frame.append({"cam": name, "frame": frame, "visible_GS": int(mask.sum()), **m})

                canvas = torch.cat([pixels, colors], dim=2)[0].cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                writer.append_data(canvas)
                if frame in (0, self.data.num_frames // 2, self.data.num_frames - 1):
                    imageio.imwrite(
                        self.result_dir / "renders" / f"test_{name}_f{frame:05d}_step{step}.png",
                        canvas,
                    )
            writer.close()

        n = len(per_frame)
        stats = {k: float(np.mean(v)) for k, v in metrics.items()}
        stats.update(
            {
                "step": step,
                "num_images": n,
                "render_fps": n / render_time,
                "num_GS": len(self.splats["means"]),
            }
        )
        print(
            f"PSNR {stats['psnr']:.2f} | SSIM {stats['ssim']:.4f} | LPIPS {stats['lpips']:.4f} "
            f"| {stats['render_fps']:.0f} FPS | {n} Bilder"
        )
        with open(self.result_dir / "stats" / f"val_step{step:05d}.json", "w") as f:
            json.dump({"mean": stats, "per_frame": per_frame}, f, indent=2)
        for k in ["psnr", "ssim", "lpips", "render_fps"]:
            self.writer.add_scalar(f"val/{k}", stats[k], step)
        self.writer.flush()
        return stats

    @torch.no_grad()
    def render_traj(self, step: int, n_interp: int = 8) -> None:
        """Kamerafahrt durch die Trainingskameras, die Zeit läuft dabei von 0 bis 1."""
        cams = self.data.train_cams
        c2ws = self.data.camtoworlds[cams]
        # Kameras entlang der Hauptachse (x nach Normalisierung) sortieren
        order = np.argsort(c2ws[:, 0, 3])
        path = generate_interpolated_path(c2ws[order][:, :3, :], n_interp)  # [P, 3, 4]
        bottom = np.broadcast_to(np.array([0, 0, 0, 1], np.float32), (len(path), 1, 4))
        path = np.concatenate([path, bottom], axis=1).astype(np.float32)
        times = np.linspace(0.0, 1.0, len(path))

        writer = imageio.get_writer(self.result_dir / "videos" / f"traj_step{step}.mp4", fps=30, macro_block_size=2)
        for i in tqdm.trange(len(path), desc="Trajektorie"):
            c2w = torch.from_numpy(path[i]).to(self.device)
            colors, _, _, _ = self.render(cams[0], float(times[i]), self.cfg.sh_degree, c2w=c2w)
            writer.append_data((colors[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        writer.close()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def main(cfg: Config) -> None:
    runner = Runner(cfg)
    if cfg.ckpt is not None:
        ckpt = torch.load(os.path.expanduser(cfg.ckpt), map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(ckpt["step"])
        if cfg.render_traj:
            runner.render_traj(ckpt["step"])
        return
    runner.train()


if __name__ == "__main__":
    main(tyro.cli(Config))
