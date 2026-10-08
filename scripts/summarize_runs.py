"""Ergebnistabelle (Markdown) aus Läufen von train.py.

Aufruf: python scripts/summarize_runs.py ~/masterarbeit/results/ablation [weitere Ordner ...]
Jeder Unterordner mit stats/val_step30000.json ist ein Lauf.
"""

import json
import sys
from pathlib import Path


def read_run(d: Path):
    val = d / "stats" / "val_step30000.json"
    if not val.exists():
        return None
    m = json.loads(val.read_text())["mean"]
    row = {"name": d.name, "psnr": m["psnr"], "ssim": m["ssim"], "lpips": m["lpips"]}
    v7 = d / "stats" / "val_step07000.json"
    row["psnr7k"] = json.loads(v7.read_text())["mean"]["psnr"] if v7.exists() else float("nan")
    info = {}
    if (d / "run_info.txt").exists():
        for line in (d / "run_info.txt").read_text().splitlines():
            k, _, v = line.partition(": ")
            info[k] = v
    row["time_min"] = int(info.get("wall_time_s", 0)) / 60
    row["vram_gb"] = (int(info.get("vram_peak_mib", 0)) - 2900) / 1024  # minus Desktop-Grundlast
    row["commit"] = info.get("commit", "?")
    return row


def main() -> None:
    rows = []
    for root in sys.argv[1:]:
        root = Path(root).expanduser()
        dirs = [root] if (root / "stats").exists() else sorted(p for p in root.iterdir() if p.is_dir())
        rows += [r for r in (read_run(d) for d in dirs) if r]
    print("| Lauf | PSNR 7k | PSNR | SSIM | LPIPS | Zeit | VRAM* | Commit |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['name']} | {r['psnr7k']:.2f} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['lpips']:.3f} "
            f"| {r['time_min']:.0f} min | {r['vram_gb']:.1f} GB | {r['commit']} |"
        )


if __name__ == "__main__":
    main()
