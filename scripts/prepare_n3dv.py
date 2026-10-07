"""Prepare a Neural3DV (DyNeRF) scene for 4D Gaussian Splatting.

Steps:
  1. Extract the first N frames of every camera video with ffmpeg (optionally downscaled).
  2. Run COLMAP once on frame 0 (cameras are static) to get intrinsics, poses and a sparse point cloud.

Output layout (also understood by FreeTimeGsVanilla's "flat" format):

    <out>/
      images/cam00/00000.png ... 000NN.png   per-camera frame sequences
      sparse/0/{cameras,images,points3D}.bin COLMAP model of frame 0, image names "camXX.png"
      frame0/images/camXX.png                frame 0 only (static sanity check with gsplat)
      frame0/sparse -> ../sparse
      colmap/database.db
      meta.json

Example:
    python scripts/prepare_n3dv.py \
        --src ~/masterarbeit/data/n3dv/raw/flame_steak \
        --out ~/masterarbeit/data/n3dv/flame_steak_50f \
        --num-frames 50 --downscale 2
"""

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run(cmd: list[str], log_path: Path | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    with open(log_path, "a") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def probe(video: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)["streams"][0]


def extract_frames(video: Path, out_dir: Path, num_frames: int, width: int, height: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
        "-frames:v", str(num_frames),
        "-vf", f"scale={width}:{height}:flags=area",
        "-start_number", "0",
        str(out_dir / "%05d.png"),
    ]
    run(cmd)
    n = len(list(out_dir.glob("*.png")))
    if n != num_frames:
        raise RuntimeError(f"{video.name}: expected {num_frames} frames, got {n}")


def run_colmap(out: Path, use_gpu: bool) -> None:
    colmap_dir = out / "colmap"
    input_dir = colmap_dir / "input"
    db = colmap_dir / "database.db"
    sparse_raw = colmap_dir / "sparse"
    log = colmap_dir / "colmap.log"
    gpu = "1" if use_gpu else "0"

    # Frame 0 of every camera, named camXX.png
    input_dir.mkdir(parents=True, exist_ok=True)
    for cam_dir in sorted((out / "images").iterdir()):
        shutil.copy2(cam_dir / "00000.png", input_dir / f"{cam_dir.name}.png")
    if db.exists():
        db.unlink()
    if sparse_raw.exists():
        shutil.rmtree(sparse_raw)
    sparse_raw.mkdir()

    # PINHOLE, one camera per image: the videos are already undistorted, but focal lengths differ slightly.
    run([
        "colmap", "feature_extractor",
        "--database_path", db, "--image_path", input_dir,
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "0",
        "--SiftExtraction.use_gpu", gpu,
    ], log)
    run([
        "colmap", "exhaustive_matcher",
        "--database_path", db,
        "--SiftMatching.use_gpu", gpu,
    ], log)
    run([
        "colmap", "mapper",
        "--database_path", db, "--image_path", input_dir, "--output_path", sparse_raw,
        "--Mapper.ba_refine_principal_point", "0",
    ], log)

    models = sorted(p for p in sparse_raw.iterdir() if p.is_dir())
    if not models:
        raise RuntimeError(f"COLMAP produced no model, see {log}")
    if len(models) > 1:
        print(f"WARNING: COLMAP produced {len(models)} models, using {models[0].name}")

    target = out / "sparse" / "0"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(exist_ok=True)
    shutil.copytree(models[0], target)

    run(["colmap", "model_analyzer", "--path", target], log)


def make_frame0(out: Path) -> None:
    frame0 = out / "frame0"
    if frame0.exists():
        shutil.rmtree(frame0)
    shutil.copytree(out / "colmap" / "input", frame0 / "images")
    (frame0 / "sparse").symlink_to("../sparse")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder with camXX.mp4 (and poses_bounds.npy)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-frames", type=int, default=50)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--skip-frames", action="store_true", help="reuse already extracted frames")
    ap.add_argument("--colmap-gpu", action="store_true", help="use GPU SIFT (needs COLMAP with CUDA)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src, out = args.src.expanduser(), args.out.expanduser()
    videos = sorted(src.glob("cam*.mp4"))
    if not videos:
        raise SystemExit(f"no cam*.mp4 in {src}")

    info = probe(videos[0])
    width, height = int(info["width"]) // args.downscale, int(info["height"]) // args.downscale
    print(f"{len(videos)} cameras, {info['width']}x{info['height']} -> {width}x{height}, "
          f"{args.num_frames} frames, fps {info['r_frame_rate']}")

    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_frames:
        with ThreadPoolExecutor(args.workers) as pool:
            jobs = [
                pool.submit(extract_frames, v, out / "images" / v.stem, args.num_frames, width, height)
                for v in videos
            ]
            for j in jobs:
                j.result()

    run_colmap(out, args.colmap_gpu)
    make_frame0(out)

    if (src / "poses_bounds.npy").exists():
        shutil.copy2(src / "poses_bounds.npy", out / "poses_bounds.npy")

    meta = {
        "scene": src.name,
        "cameras": [v.stem for v in videos],
        "num_frames": args.num_frames,
        "fps": info["r_frame_rate"],
        "orig_resolution": [int(info["width"]), int(info["height"])],
        "resolution": [width, height],
        "downscale": args.downscale,
        "test_cameras": ["cam00"],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
