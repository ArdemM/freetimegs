# Experimente

Ein Eintrag pro Lauf. Commit = `git rev-parse --short HEAD` zum Zeitpunkt des Laufs.

| Datum | Commit | Szene | Methode / Konfiguration | Schritte | PSNR | SSIM | LPIPS | Zeit | VRAM | Notizen |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-07 | afd6b7f | flame_steak, nur Frame 0 (21 Ansichten, 1352×1014) | gsplat simple_trainer mcmc, data_factor 1 (Plausibilitätstest Posen) | 7000 | 30.01 | 0.925 | 0.161 | 92 s | 0.3 GB | 3 Testansichten (test_every=8), 85k Gaußsche |
