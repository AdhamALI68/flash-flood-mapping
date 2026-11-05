# Trial 2 — live status

*2026-09-21 15:07:50* · refreshes every 20s

## 0 · Clean flood labels: the pipeline

Goal: labels with no gaps and no noise dots, so the model can reach high recall **and** high precision. Floods whose marks are not clean are skipped.

| Step | What | Status | Note |
|---|---|---|---|
| s20 | Mark 20 new floods, look at the mistakes, add rules | done | 2 of 20 kept, both look wrong; 6 rule changes found |
| all | Download and mark every flood that has training tiles, save everything | done | 567 floods in 98 min: 442 with clear photos saved |
| rules | Find and apply the rules on all saved floods, check samples by eye | done | checked by eye: 3 right, 11 not sure, 49 wrong (3 = 40 tiles, 2 floods in test regions) |
| tiles | Build training tiles from the clean floods | done | 275 tiles from 28 floods |
| train | Train the model on the new labels | done |  |
| score | Score recall and precision on the held-out test regions | done | trusted 2-flood test: fine-tuned best precision (0.209 at recall 0.25, 0.175 at 0.50) |

### s20: `########################` **20/20** floods checked, **2 kept**

radar disagrees or cannot check: 9 · too little or too much marked: 6 · no clear photos: 3 · kept: 2

### all: `########################` **567/567** floods checked, **101 kept**

radar disagrees or cannot check: 211 · too little or too much marked: 129 · no clear photos: 125 · kept: 101 · scattered dots: 1

## 0b · First pilot: 10 floods, try 1 and try 2

**Nothing is running for the flood marks.**

| # | Flood | Try 1: marked | Try 2: result |
|---|---|---|---|
| 1 | Maricopa, Arizona, 2021-08-14, 25.00M | 28.1% | kept, 6.4% marked, radar agrees |
| 2 | San Bernardino, California, 2023-08-20, 1.00M | 5.6% | kept, 2.2% marked, no radar, needs your check |
| 3 | San Bernardino, California, 2023-03-16, 1.00K | 51.0% | skipped: no visible flood marks |
| 4 | Clark, Nevada, 2018-08-15, 2.00M | 32.6% | skipped: no visible flood marks |
| 5 | San Bernardino, California, 2021-08-30, 5.00K | 9.6% | skipped: no visible flood marks |
| 6 | Imperial, California, 2021-08-31, 1.00M | 59.2% | skipped: no visible flood marks |
| 7 | Mohave, Arizona, 2021-08-11, 1.00K | 11.0% | skipped: no clear photo after the flood |
| 8 | Mohave, Arizona, 2022-07-27, 1.00K | 11.8% | skipped: the radar does not see extra change inside the marks in the flood week |
| 9 | Riverside, California, 2022-06-22, 0.00K | 39.5% | skipped: no visible flood marks |
| 10 | Maricopa, Arizona, 2022-08-11, 0.00K | 9.3% | kept, 1.2% marked, no radar, needs your check |

Pictures and Right/Wrong buttons: https://claude.ai/code/artifact/8f7e8153-a3e1-4566-9bc4-abfb21bf874a

## 1 · Running now

**Nothing is running.**

## 2 · Queued, in order

| Next | Note |
|---|---|
| Sweep: 1 runs left | the team's engineered features |
| Score the fold-diversity ensemble | run final/phase7_final_foldensemble.py, both fold models trained |

The winner is picked on **validation**, then scored on test once. Choosing the best of 15 by its test score would inflate whatever we report.

Sweep, TTA, cross-region validation, and the headline confidence interval are all done, see Section 3. The paper draft (`paper/igarss.tex`) and the full record (`RECORD/README.md`) exist and are kept current.

## 2b · The last open question

**Looks pilot: finished.** Same scenes reprocessed with more multilooking.

| Looks | N | Noise on the drop | 0.15 is | >3σ fires on | AUC vs optical |
|---|---|---|---|---|---|
| 5x1 | 5 | 0.0447 | 3.4σ | 0.00% | 0.6313 |
| 10x2 | 20 | 0.0102 | 14.7σ | 8.63% | 0.6296 |
| 20x4 | 80 | 0.0020 | 76.8σ | 40.53% | 0.5838 |

**Optical labels: 240 built.** Independent Sentinel-2 flood maps, used to grade the model with a second instrument.

| Graded by | AUC |
|---|---|
| InSAR coherence label | 0.6427 |
| Sentinel-2 optical label | **0.7505** |

Same model, same 6.9 M pixels. Only the answer key changed.

## 3 · Results

### The headline

Per 20 m pixel, held-out regions 1606 and 1809, 23.5 M pixels, 5.24% flooded.

| | AUC | AP | recall | precision |
|---|---|---|---|---|
| **U-Net (30 inputs)** | **0.6950** | 0.0916 | 0.3968 | 0.1014 |
| XGBoost, local features only | 0.6869 | 0.1008 | 0.6602 | 0.0865 |
| rain (best single input) | 0.6480 | — | — | — |
| slope (best terrain input) | 0.6391 | — | — | — |
| elevation, the free rule | 0.4804 | — | — | — |

**Everything lands between 0.687 and 0.695.** A tree on single pixels, a U-Net with 2.24 km of context, and a feature set with all basin routing removed all reach the same place. The cap is the label, not the model.

### The label-corrector experiment - finished, result is negative

A small model was trained on the 210 events with a real optical photo, to predict what the photo would show from land cover, terrain and the raw radar number. Graded directly against real photos on events it never trained on, it is a genuine, large improvement over the raw label:

| Graded against real photos | AUC |
|---|---|
| Raw coherence drop alone | 0.4136 (worse than a coin flip) |
| The corrector | **0.8981** |

But retraining the main flood model on labels corrected this way does **not** produce a genuinely better model:

| | AUC | recall | precision | accuracy |
|---|---|---|---|---|
| Real headline (raw label) | 0.6950 | 0.3968 | 0.1014 | 0.7840 |
| Corrected label | 0.8937 | 0.0419 | 0.9735 | 0.9923 |

The AUC looks higher, but recall collapsed to 4.2% (it now catches almost no real flood pixels) and a separate, fairer check (can it tell a flood day apart from a normal quiet day at the same spot) scored only 0.545, barely above chance. The likely cause: the corrector leaned mainly on land cover and terrain shape, and the main model is *also* given land cover and terrain shape as input, so it learned to copy the corrector's own rule instead of learning anything about actual storms. **Verdict: this specific correction design does not beat the raw-label headline. The 0.6950 number stays the safe one to report.**

### Sweep runs finished

| Run | Change | Val pixel AUC | Epochs |
|---|---|---|---|
| `sw_rung1` | terrain at 1 m | **0.7785** | 18 |
| `sw_rung2` | terrain at 2 m | **0.7718** | 22 |
| `sw_rung5` | terrain at 5 m | **0.7706** | 16 |
| `sw_rung30` | terrain at 30 m | **0.7662** | 35 |
| `sw_multi` | 1 + 10 + 30 m stacked | **0.7657** | 19 |
| `sw_multi5` | all five resolutions | **0.7807** | 22 |
| `sw_focal` | focal loss | **0.7525** | 39 |
| `sw_conf` | per-pixel confidence weighting | **0.7612** | 25 |
| `sw_thr0.10` | label threshold 0.10 | **0.7644** | 22 |
| `sw_thr0.20` | label threshold 0.20 | **0.7762** | 29 |
| `sw_thr0.25` | label threshold 0.25 | **0.7787** | 19 |
| `sw_seed1` | seed 1 (ensemble) | **0.7782** | 20 |
| `sw_seed2` | seed 2 (ensemble) | **0.7706** | 30 |
| `sw_seed3` | seed 3 (ensemble) | **0.7701** | 21 |

### Earlier experiments — all failed to beat the baseline

| Run | AUC | AP | What it changed |
|---|---|---|---|
| `rec_pw10` | — | — | class-weighted loss |
| `rec_floodonly` | — | — | trained on flood days only |
| `rec_min5` | — | — | 5% training filter |
| `rec_cont` | — | — | continuous target |
| `rec_big` | — | — | bigger model |
| `rec_both` | — | — | flood-only + weighted |
| `rec_min5pw` | — | — | 5% filter + weighted |

Measured numbers are in `data/compare_runs.csv`. Seven attempts, seven failures, every one pushing average precision down.

## 4 · Verified — settled, not to be re-argued

| Check | Result |
|---|---|
| Scene dates: baseline pair before the flood | **617/617** |
| Scene dates: co-event pair spans the flood | **617/617** |
| Label arithmetic vs fresh recomputation | **exact, 0.00e+00 diff** |
| Leakage: inputs that can see the flood | **0 of 30** |
| Train/test region overlap | **none** |
| AUC arithmetic vs sklearn | **identical** |
| Output really is per pixel, not per tile | **confirmed** |
| Post-flood lidar inflating the score | **no** — clean terrain scores *higher*, 0.7378 vs 0.6760 |
| Cross-region transfer | beats the free baseline **6/6 regions** |

**Why the number is capped.** HyP3 gave 5 looks per pixel. Coherence from 5 looks carries about ±0.09 noise on the drop, and our threshold is 0.15 — a 1.6 sigma test. A 3 sigma cut fires on 0.00% of pixels. Most of what the label calls flood sits inside its own measurement noise, and that is physical, not a coding error.

---

*`status.py` · previous download-era version kept as `status_old.py`*
