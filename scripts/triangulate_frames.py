"""Triangulate a sparse point cloud for every frame with the fixed camera poses from frame 0.

Requires a scene prepared by prepare_n3dv.py. For each frame:
  1. copy colmap/database.db (keeps image/camera ids of sparse/0), drop its features and matches
  2. extract SIFT features + exhaustive matching on that frame's images
  3. colmap point_triangulator with sparse/0 as fixed poses and intrinsics

Output (format read by FreeTimeGsVanilla's combine_frames_fast_keyframes.py):

    <scene>/points/points3d_frame000000.npy   [M, 3] float32
    <scene>/points/colors_frame000000.npy     [M, 3] float32, 0-255
    <scene>/colmap/frames/00000/sparse/        triangulated COLMAP model (for inspection)

Example:
    python scripts/triangulate_frames.py --scene ~/masterarbeit/data/n3dv/flame_steak_50f
"""

import argparse
import json
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pycolmap

from prepare_n3dv import run


def prepare_database(template: Path, db: Path) -> None:
    shutil.copy2(template, db)
    con = sqlite3.connect(db)
    for table in ("keypoints", "descriptors", "matches", "two_view_geometries"):
        con.execute(f"DELETE FROM {table}")
    con.commit()
    con.execute("VACUUM")
    con.close()


def triangulate_frame(scene: Path, frame: int, cams: list[str], threads: int, args) -> dict:
    work = scene / "colmap" / "frames" / f"{frame:05d}"
    if work.exists():
        shutil.rmtree(work)
    input_dir, sparse = work / "input", work / "sparse"
    input_dir.mkdir(parents=True)
    sparse.mkdir()
    log = work / "colmap.log"

    for cam in cams:
        (input_dir / f"{cam}.png").symlink_to(scene / "images" / cam / f"{frame:05d}.png")

    db = work / "database.db"
    prepare_database(scene / "colmap" / "database.db", db)
    gpu = "1" if args.colmap_gpu else "0"

    run([
        "colmap", "feature_extractor",
        "--database_path", db, "--image_path", input_dir,
        "--SiftExtraction.use_gpu", gpu,
        "--SiftExtraction.num_threads", str(threads),
        "--SiftExtraction.max_num_features", str(args.max_features),
    ], log)
    run([
        "colmap", "exhaustive_matcher",
        "--database_path", db,
        "--SiftMatching.use_gpu", gpu,
        "--SiftMatching.num_threads", str(threads),
    ], log)
    run([
        "colmap", "point_triangulator",
        "--database_path", db, "--image_path", input_dir,
        "--input_path", scene / "sparse" / "0", "--output_path", sparse,
        "--clear_points", "1", "--refine_intrinsics", "0",
        "--Mapper.num_threads", str(threads),
        "--Mapper.tri_ignore_two_view_tracks", "0" if args.two_view_tracks else "1",
    ], log)

    rec = pycolmap.Reconstruction(sparse)
    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts], dtype=np.float32).reshape(-1, 3)
    rgb = np.array([p.color for p in pts], dtype=np.float32).reshape(-1, 3)
    err = np.array([p.error for p in pts], dtype=np.float32)
    track = np.array([p.track.length() for p in pts], dtype=np.float32)

    out = scene / "points"
    np.save(out / f"points3d_frame{frame:06d}.npy", xyz)
    np.save(out / f"colors_frame{frame:06d}.npy", rgb)

    if not args.keep_db:
        db.unlink()
        shutil.rmtree(input_dir)

    return {
        "frame": frame,
        "points": len(pts),
        "mean_reproj_error": float(err.mean()) if len(pts) else None,
        "mean_track_length": float(track.mean()) if len(pts) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True, help="output folder of prepare_n3dv.py")
    ap.add_argument("--frames", type=str, default=None, help="e.g. '0:50' (default: all frames in meta.json)")
    ap.add_argument("--workers", type=int, default=4, help="frames processed in parallel")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--two-view-tracks", action="store_true", help="also keep points seen by only two cameras")
    ap.add_argument("--colmap-gpu", action="store_true", help="use GPU SIFT (needs COLMAP with CUDA)")
    ap.add_argument("--keep-db", action="store_true", help="keep per-frame database and input links")
    args = ap.parse_args()

    scene = args.scene.expanduser()
    meta = json.loads((scene / "meta.json").read_text())
    cams = meta["cameras"]
    if args.frames:
        a, b = (int(x) for x in args.frames.split(":"))
        frames = list(range(a, b))
    else:
        frames = list(range(meta["num_frames"]))

    (scene / "points").mkdir(exist_ok=True)
    threads = max(1, 32 // args.workers)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda f: triangulate_frame(scene, f, cams, threads, args), frames))

    for r in results:
        print(f"frame {r['frame']:3d}: {r['points']:6d} points, "
              f"reproj {r['mean_reproj_error']:.3f} px, track {r['mean_track_length']:.2f}")
    counts = [r["points"] for r in results]
    print(f"{len(frames)} frames in {time.time() - t0:.0f} s, points min/mean/max "
          f"{min(counts)}/{np.mean(counts):.0f}/{max(counts)}")

    stats_path = scene / "points" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats.update({str(r["frame"]): r for r in results})
    stats_path.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
