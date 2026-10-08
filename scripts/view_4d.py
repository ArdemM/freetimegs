"""Interaktiver 4D-Viewer für Checkpoints des eigenen Trainers (viser + nerfview).

Gerendert wird mit derselben Funktion wie im Training (ftgs.model.gaussians_at),
das Bild im Viewer entspricht also dem, was bei der Evaluation herauskommt.

Kamerasteuerung: viser dreht die Kamera um eine feste Oben-Achse und stoppt an
deren Polen. Im Trainings-Weltsystem ist die Szene gekippt, deshalb wird die
Oben-Achse auf die mittlere Oben-Richtung der Kameras gesetzt und der Viewer
startet aus einer Trainingskamera (Auswahl unter "Kamera").

Filter (unter "Filter", standardmäßig aus, damit das Bild der Evaluation entspricht):
- "Unsichtbare entfernen": Gaußsche, die über alle Frames und Trainingskameras
  zusammen weniger als --min-contrib Pixel zum Bild beitragen (Floater hinter Wand
  und Rollo, die nur aus neuen Blickwinkeln auftauchen). Die Trainingsansichten
  bleiben dabei praktisch unverändert.
- "Kugel": nur Gaußsche innerhalb eines Radius um den Szenenmittelpunkt. Wirkt
  aufgeräumter, schneidet aber auch Hintergrund ab, der in den Kamerabildern liegt.

Aufruf (in WSL, danach im Windows-Browser http://localhost:8080 öffnen):
    python scripts/view_4d.py --ckpt ~/masterarbeit/results/ablation/kf1_full/ckpts/ckpt_final.pt
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as vt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftgs.data import Parser  # noqa: E402  (setzt auch den Pfad zu gsplat/examples)
from ftgs.model import gaussians_at  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402
from nerfview import CameraState, RenderTabState, Viewer, apply_float_colormap  # noqa: E402

MODES = ("RGB", "Geschwindigkeit |v|", "Dauer s", "Tiefe")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--num-frames", type=int, default=None, help="Standard: aus data_dir des Checkpoints")
    ap.add_argument("--min-temporal", type=float, default=1e-3, help="σ(t)-Schwelle wie beim Rendern")
    ap.add_argument("--start-cam", default="cam00")
    ap.add_argument("--min-contrib", type=float, default=1.0,
                    help="Filter 'Unsichtbare entfernen': minimaler Beitrag in Pixeln")
    args = ap.parse_args()

    device = "cuda"
    ckpt = torch.load(Path(args.ckpt).expanduser(), map_location=device)
    splats, cfg = ckpt["splats"], ckpt["cfg"]
    sh_degree = int(round(math.sqrt(1 + splats["shN"].shape[1]))) - 1
    near, far = cfg.get("near_plane", 0.01), cfg.get("far_plane", 1e10)

    data_dir = Path(cfg["data_dir"]).expanduser()
    F = args.num_frames
    if F is None:
        F = len(list((data_dir / "images" / "cam00").glob("*.png")))
        if cfg.get("num_frames", -1) > 0:
            F = min(F, cfg["num_frames"])

    # Kameras wie im Training (normalisiertes Weltsystem), nur Posen, keine Bilder
    p = Parser(str(data_dir / "frame0"), factor=1, normalize=True, test_every=10**9)
    cam_names = [Path(n).stem for n in p.image_names]
    c2ws = p.camtoworlds
    Ks = [p.Ks_dict[cid] for cid in p.camera_ids]
    sizes = [p.imsize_dict[cid] for cid in p.camera_ids]  # (W, H)
    up = -c2ws[:, :3, 1].mean(0)  # Oben = −y der Kameras (OpenCV)
    up /= np.linalg.norm(up)
    center = np.zeros(3)  # Szene ist normalisiert, Mittelpunkt im Ursprung
    print(f"{len(splats['means']):,} Gaußsche, {F} Frames, {len(cam_names)} Kameras, Oben {np.round(up, 3)}")

    server = viser.ViserServer(port=args.port, verbose=False)
    server.scene.set_up_direction(up)

    state = {"frame": 0, "playing": False, "fps": 30.0, "mode": MODES[0], "active": 0}
    all_splats, N = splats, len(splats["means"])
    dist = all_splats["means"].norm(dim=-1)  # Abstand zum Szenenmittelpunkt (Ursprung)
    contrib = None  # Beitrag pro Gaußscher, erst bei Bedarf berechnet

    def compute_contrib() -> torch.Tensor:
        """Summe der Blending-Gewichte pro Gaußscher über alle Frames und Kameras.

        Der Gradient der Bildsumme nach einer konstanten Farbe 1 ist genau das
        Gewicht T·α, mit dem die Gaußsche in die Pixel eingeht (in Pixeln).
        """
        out = torch.zeros(N, device=device)
        vms = torch.linalg.inv(torch.from_numpy(c2ws).float().to(device))
        Kt = torch.from_numpy(np.stack(Ks)).float().to(device)
        for f in range(F):
            g, mask = gaussians_at(all_splats, f / max(F - 1, 1), args.min_temporal)
            idx = mask.nonzero().squeeze(1)
            for c in range(len(cam_names)):
                W, H = sizes[c]
                col = torch.ones(len(idx), 3, device=device, requires_grad=True)
                img, _, _ = rasterization(
                    means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"],
                    colors=col, viewmats=vms[c : c + 1], Ks=Kt[c : c + 1], width=W, height=H,
                    sh_degree=None, near_plane=near, far_plane=far,
                )
                img[..., 0].sum().backward()
                out.index_add_(0, idx, col.grad[:, 0])
        return out

    def set_camera(client: viser.ClientHandle, name: str) -> None:
        i = cam_names.index(name)
        c2w, K, (W, H) = c2ws[i], Ks[i], sizes[i]
        pos, fwd = c2w[:3, 3], c2w[:3, 2]
        dist = max(float((center - pos) @ fwd), 0.1)
        with client.atomic():
            client.camera.up_direction = up
            client.camera.fov = 2 * math.atan(H / (2 * K[1, 1]))
            client.camera.position = pos
            client.camera.wxyz = vt.SO3.from_matrix(c2w[:3, :3]).wxyz
            client.camera.look_at = pos + fwd * dist

    # --- Bedienelemente ---
    with server.gui.add_folder("Zeit"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=F - 1, step=1, initial_value=0)
        play_cb = server.gui.add_checkbox("Abspielen", initial_value=False)
        fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=30)
    with server.gui.add_folder("Kamera"):
        cam_dd = server.gui.add_dropdown("Trainingskamera", cam_names, initial_value=args.start_cam)
        cam_btn = server.gui.add_button("Zur Kamera springen")
    with server.gui.add_folder("Darstellung"):
        mode_dd = server.gui.add_dropdown("Modus", MODES, initial_value=MODES[0])
        stats_md = server.gui.add_markdown("")
    with server.gui.add_folder("Filter"):
        invis_cb = server.gui.add_checkbox("Unsichtbare entfernen", initial_value=False,
                                           hint=f"Beitrag < {args.min_contrib} px zu allen Trainingsbildern")
        sphere_cb = server.gui.add_checkbox("Kugel", initial_value=False)
        r_max = float(torch.quantile(dist, 0.999))
        radius_sl = server.gui.add_slider("Radius", min=0.5, max=round(r_max, 1), step=0.1,
                                          initial_value=round(r_max, 1),
                                          hint="Kameras stehen bei ca. 4–5, der Hintergrund liegt teils dahinter")
        filter_md = server.gui.add_markdown("Filter aus")

    def apply_filter() -> None:
        nonlocal splats, contrib
        keep = torch.ones(N, dtype=torch.bool, device=device)
        if invis_cb.value:
            if contrib is None:
                filter_md.content = "berechne Beiträge (ca. 10 s) ..."
                contrib = compute_contrib()
            keep &= contrib >= args.min_contrib
        if sphere_cb.value:
            keep &= dist <= radius_sl.value
        splats = all_splats if bool(keep.all()) else {k: v[keep] for k, v in all_splats.items()}
        n_out = N - int(keep.sum())
        filter_md.content = f"entfernt: {n_out:,} ({100 * n_out / N:.1f} %)" if n_out else "Filter aus"
        viewer.rerender(None)

    for handle in (invis_cb, sphere_cb, radius_sl):
        handle.on_update(lambda _: apply_filter())

    def update_stats() -> None:
        t = state["frame"] / max(F - 1, 1)
        stats_md.content = f"t = {t:.3f}  \naktiv: {state['active']:,} / {N:,}"

    @frame_slider.on_update
    def _(_) -> None:
        state["frame"] = int(frame_slider.value)
        viewer.rerender(None)

    @play_cb.on_update
    def _(_) -> None:
        state["playing"] = play_cb.value

    @fps_slider.on_update
    def _(_) -> None:
        state["fps"] = float(fps_slider.value)

    @cam_btn.on_click
    def _(event: viser.GuiEvent) -> None:
        for client in server.get_clients().values():
            set_camera(client, cam_dd.value)

    @mode_dd.on_update
    def _(_) -> None:
        state["mode"] = mode_dd.value
        viewer.rerender(None)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        set_camera(client, args.start_cam)

    # --- Rendern ---
    def heat(values: torch.Tensor) -> torch.Tensor:
        """Skalar pro Gaußscher -> Farbe [M, 3], Bereich 1.–99. Perzentil."""
        lo, hi = torch.quantile(values, torch.tensor([0.01, 0.99], device=device))
        return apply_float_colormap(((values - lo) / (hi - lo + 1e-8)).clamp(0, 1)[:, None], "turbo")

    @torch.no_grad()
    def render_fn(camera_state: CameraState, rts: RenderTabState) -> np.ndarray:
        W, H = (rts.render_width, rts.render_height) if rts.preview_render else (rts.viewer_width, rts.viewer_height)
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((W, H))).float().to(device)
        t = state["frame"] / max(F - 1, 1)
        g, mask = gaussians_at(splats, t, args.min_temporal)
        state["active"] = int(mask.sum())
        update_stats()

        colors, deg = g["colors"], sh_degree
        if state["mode"] == MODES[1]:
            colors, deg = heat(torch.log1p(splats["velocities"][mask].norm(dim=-1))), None
        elif state["mode"] == MODES[2]:
            colors, deg = heat(splats["durations"][mask].squeeze(-1)), None  # log s

        img, alpha, _ = rasterization(
            means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"],
            colors=colors, viewmats=torch.linalg.inv(c2w)[None], Ks=K[None],
            width=W, height=H, sh_degree=deg, near_plane=near, far_plane=far,
            render_mode="RGB+ED" if state["mode"] == MODES[3] else "RGB",
        )
        if state["mode"] == MODES[3]:
            d = img[0, ..., 3]
            lo, hi = torch.quantile(d[alpha[0, ..., 0] > 0.5][::16], torch.tensor([0.01, 0.99], device=device))
            return apply_float_colormap(((d - lo) / (hi - lo + 1e-8)).clamp(0, 1)[..., None], "turbo").cpu().numpy()
        return img[0, ..., :3].clamp(0, 1).cpu().numpy()

    viewer = Viewer(server=server, render_fn=render_fn, mode="rendering")
    print(f"Viewer läuft: http://localhost:{args.port}  (Strg+C beendet)")

    last = time.time()
    try:
        while True:
            if state["playing"] and time.time() - last >= 1.0 / state["fps"]:
                last = time.time()
                frame_slider.value = (state["frame"] + 1) % F  # löst on_update + rerender aus
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("Viewer beendet.")


if __name__ == "__main__":
    main()
