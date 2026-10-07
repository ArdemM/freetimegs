# Experimente

Ein Eintrag pro Lauf. Commit = `git rev-parse --short HEAD` zum Zeitpunkt des Laufs.

| Datum | Commit | Szene | Methode / Konfiguration | Schritte | PSNR | SSIM | LPIPS | Zeit | VRAM | Notizen |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-07 | afd6b7f | flame_steak, nur Frame 0 (21 Ansichten, 1352×1014) | gsplat simple_trainer mcmc, data_factor 1 (Plausibilitätstest Posen) | 7000 | 30.01 | 0.925 | 0.161 | 92 s | 0.3 GB | 3 Testansichten (test_every=8), 85k Gaußsche |
| 2026-10-07 | 921bf44 | flame_steak 50f, Test cam00 (50 Frames) | FreeTimeGsVanilla 42e2f9b, default_keyframe_small, keyframe_step 5 (73.516 Init-Punkte, fix) | 30000 | 30.14 | 0.948 | 0.069 | 17 min | 1.4 GB | Referenz. VRAM = Spitze 4,25 GB minus ca. 2,9 GB Desktop-Grundlast. Anzahl Gaußsche bleibt konstant (Pure Relocation) |
| 2026-10-07 | 921bf44 | flame_steak 50f, Test cam00 (50 Frames) | FreeTimeGsVanilla 42e2f9b, default_keyframe_small, keyframe_step 1 (334.500 Init-Punkte, fix) | 30000 | 30.60 | 0.962 | 0.068 | 19 min | 1.7 GB | Referenz, dichtere Init. Flamme schwächer als GT, Hund und Hand leicht weich |
