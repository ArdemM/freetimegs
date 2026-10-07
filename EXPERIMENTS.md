# Experimente

Ein Eintrag pro Lauf. Commit = `git rev-parse --short HEAD` zum Zeitpunkt des Laufs.

| Datum | Commit | Szene | Methode / Konfiguration | Schritte | PSNR | SSIM | LPIPS | Zeit | VRAM | Notizen |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-07 | afd6b7f | flame_steak, nur Frame 0 (21 Ansichten, 1352×1014) | gsplat simple_trainer mcmc, data_factor 1 (Plausibilitätstest Posen) | 7000 | 30.01 | 0.925 | 0.161 | 92 s | 0.3 GB | 3 Testansichten (test_every=8), 85k Gaußsche |
| 2026-10-07 | 921bf44 | flame_steak 50f, Test cam00 (50 Frames) | FreeTimeGsVanilla 42e2f9b, default_keyframe_small, keyframe_step 5 (73.516 Init-Punkte, fix) | 30000 | 30.14 | 0.948 | 0.069 | 17 min | 1.4 GB | Referenz. VRAM = Spitze 4,25 GB minus ca. 2,9 GB Desktop-Grundlast. Anzahl Gaußsche bleibt konstant (Pure Relocation) |
| 2026-10-07 | 921bf44 | flame_steak 50f, Test cam00 (50 Frames) | FreeTimeGsVanilla 42e2f9b, default_keyframe_small, keyframe_step 1 (334.500 Init-Punkte, fix) | 30000 | 30.60 | 0.962 | 0.068 | 19 min | 1.7 GB | Referenz, dichtere Init. Flamme schwächer als GT, Hund und Hand leicht weich |
| 2026-10-07 | 963904d | flame_steak 50f, Test cam00 (50 Frames) | eigener Prototyp (Schritt 4), L1, keyframe_step 1 (334.500, fix), v lernbar ab 0 | 30000 | 32.19 | 0.955 | 0.096 | 9 min | 1.1 GB | 7k: 32.09/0.951/0.113. torch-Spitze 0.72 GB. Pro Frame 31.9–32.4 dB. Farben besser als Vanilla, aber weicher (LPIPS schlechter) |
| 2026-10-07 | 963904d | flame_steak 50f, Test cam00 (50 Frames) | eigener Prototyp, L2 statt L1, sonst wie oben | 30000 | 32.48 | 0.957 | 0.094 | 9 min | 1.2 GB | 7k: 32.26 |
| 2026-10-07 | 963904d | flame_steak 50f, Test cam00 (50 Frames) | eigener Prototyp, L1, keyframe_step 5 (66.807, fix), s = 0.204 | 30000 | 30.13 | 0.952 | 0.114 | 7 min | 0.9 GB | 7k: 31.04 → PSNR sinkt im weiteren Training (Überanpassung an Trainingsansichten?) |
| 2026-10-07 | 963904d | flame_steak 50f, Test cam00 (50 Frames) | eigener Prototyp, L1, keyframe_step 1, v = 0 fest (`--no-learn-velocity`) | 30000 | 33.24 | 0.957 | 0.089 | 9 min | 1.4 GB | bester Lauf. Gelernte v ohne Regularisierung/Annealing schadet hier (−1 dB) |
