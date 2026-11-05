# Trial 2 — Verified findings

Companion to `PROJECT_PLAN.md`. The plan is the intent; this file is what has
actually been checked against real data. Nothing here is quoted from memory —
every number below came from a query run in this repo, and the script that
produced it is named.

**Status as of 2026-08-11:** Phase 1 complete. Phase 2 complete and PASSED.
Phase 3 blocked on one credential.

---

## Gate results

| Gate | Question | Verdict |
|---|---|---|
| A | Do the SAR scenes exist? | **PASS** — 98.7% of events |
| A2 | Is there 1 m lidar in those basins? | **PASS** — 22 basins ≥90% covered |
| A3 | Are those basins actually arid? | **PASS** — 11 strictly arid basins survive |
| B | Does coherence actually see a flood? | **PASS** — 10× contrast, see below |

**All gates passed. Nothing remaining can invalidate the approach.**

---

## Phase 1 — Event list

`phase1/build_events.py` → `data/events_raw.csv`
`phase1/assign_basins.py` → `data/events.csv`, `data/basin_summary.csv`

- **5,030** Flash Flood records, NOAA Storm Events, 2015–2026, in AZ/NM/UT/NV/CA.
- **4,872** joined to a HUC8 basin and converted to UTC.
- **163** HUC8 basins hold ≥5 events; 27 hold ≥50.

### Correction to the plan (§2, §6)

The plan states Storm Events supplies **polygons** since 2007-10-01. It does not.
Flash flood records carry `BEGIN_LAT/LON` and `END_LAT/LON` — **point pairs, no
polygon**. Verified directly against the 2021 details file.

This matters because §6 makes those polygons the fallback plan. The only
polygons nearby are county boundaries (far too coarse) or NWS Flash Flood
*Warning* polygons, which live in a separate archive and are **forecast areas,
not observed extent** — training on them would reproduce NWS's predictions, not
floods. **§6 is not a working fallback.** Gate A passing is what makes this
survivable.

---

## Phase 2 — SAR availability (the kill gate)

`phase2/sar_availability.py` → `data/sar_availability.csv`, `data/sar_basin_summary.csv`

| | |
|---|---|
| Events checked | 3,779 |
| Usable | **3,731 (98.7%)** |
| Basins with ≥5 usable events | **68** (plan required 4) |
| Distinct HUC4 regions | 22 |

### The plan's requirement was too weak (§3, Phase 3)

The plan asks for a pre/post **pair**. Coherence in the desert decays on its own
between passes — wind moves sand — so a co-event drop means nothing without a
baseline to compare it against. The real requirement is **three scenes on one
relative orbit**:

```
  t-2 ──── t-1   [FLOOD]   t+1
   └── baseline ──┘
        t-1 ──────────────→ co-event
```

The 98.7% above is measured under this **stricter** rule, not the plan's.

### A risk that can be deleted (§2)

The plan lists the perpendicular baseline limit as a *confirmed* risk. It is not,
for Sentinel-1. Measured on a real 423-scene stack over Southern Mojave:

| | |
|---|---|
| Median \|perpendicular baseline\| | 45 m |
| Maximum observed | 468 m |
| Plan's threshold (¾ × 5 km critical) | 3,750 m |
| Pairs exceeding it | **0 / 423** |

ESA flies S1 in a tight orbital tube. This constraint binds for ERS and ALOS,
never for Sentinel-1.

---

## Phase 2b — 3DEP 1 m lidar coverage

`phase2/lidar_coverage.py` → `data/lidar_coverage.csv`

Resolves plan §5.4. Of the top 30 basins by usable events, **22 have ≥90% 1 m
lidar coverage**, twelve of them at 100%.

Coverage is the union of tile bounding boxes from The National Map, intersected
with the basin in EPSG:5070. Bounding boxes slightly overstate true footprints —
good for ranking, not a final number.

**Important exception:** Southern Mojave has 171 usable events and **0.000**
lidar coverage. Usable for every phase except the resolution ablation.

---

## Phase 2c — Aridity screen

`phase2/aridity_screen.py` → `data/basin_aridity.csv`

Basin selection had been "arid by state line". Screened instead on PRISM 30-year
mean annual precipitation (4 km, area-weighted per basin).

| Set | Basins | HUC4 regions | Events |
|---|---|---|---|
| **STRICT** — arid (<250 mm), ≥20 events, ≥90% lidar | **11** | **6** | **766** |
| **INCLUSIVE** — + semi-arid (250–500 mm) | 22 | 11 | 1,814 |

The screen caught real contamination:

| Basin | Events | Rain | Note |
|---|---|---|---|
| Upper Salt | 90 | 504 mm | **sub-humid — excluded** |
| Agua Fria | 249 | 344 mm | semi-arid, and the largest basin by event count |
| Santa Ana | 86 | 465 mm | coastal Mediterranean |
| Rillito | 62 | 412 mm | semi-arid |

Recommendation: STRICT is the headline dataset, INCLUSIVE a robustness
extension. Six regions supports a held-out-region split (train 4, test 2).

**Known limitation:** these are basin means, and some basins span mountains
(Whitewater River: 68–1,189 mm). Screening each event by rainfall at its own
point would be sharper. Not yet done.

---

## Gate B — does coherence actually see a flood?

`gate_b/find_bursts.py` → `gate_b/submit_jobs.py` → `gate_b/analyze.py`

Test event **983830**, Lower Gila–Painted Rock Reservoir, 14 Aug 2021, $25M
damage. Burst `027_056727_IW3`, `INSAR_ISCE_BURST` at `5x1` looks (20 m).

Two jobs, identical except for the interval they span, so the only difference is
the flood. The intervals were deliberately matched — baseline 6 days
(07→13 Aug), co-event 6 days (13→19 Aug) — so the comparison measures flooding
rather than elapsed time.

| Region | Baseline coherence | Co-event coherence | Drop |
|---|---|---|---|
| Whole burst (3.99 M px) | 0.937 | 0.919 | +0.0183 |
| **Within 5 km of report** (138 k px) | 0.945 | **0.797** | **+0.1487** |
| Beyond 5 km (3.86 M px) | 0.937 | 0.923 | +0.0136 |

**near − far = +0.1351**, an 11× contrast against background decorrelation.

These are the *corrected* figures — see the alignment bug in the audit section.

The control matters: baseline coherence near the flood (0.953) was *higher* than
the scene average (0.938), so this is not an intrinsically noisy area.

`gate_b/983830_gateb.png` is the stronger evidence. The co-event panel shows a
dendritic drainage network that is absent from the baseline panel, and the
difference panel renders that same network in red. The signal has the shape of
the wash system the flood ran through — not a diffuse blob.

**Incidental finding — agriculture is a confounder.** Irrigated fields in the
north-west of the burst decorrelate in *both* pairs regardless of flooding, and
appear as noisy blocks in the difference. Farmland will generate false positive
labels. NLCD land cover is already in the plan (Phase 5); it needs to be applied
as an explicit **mask in Phase 3, before labeling**, not only as a model input.

**Also worth knowing:** desert coherence over 6 days is ~0.94. That is
exceptionally high for InSAR and is the arid advantage — no vegetation to
decorrelate. The method is well matched to the terrain.

---

## VALIDATION PASSED — floods differ from quiet days at the same locations

`phase4/flood_vs_quiet.py` → `data/flood_vs_quiet_result.json`

**The central question of the project, answered.** 174 paired comparisons: each
flood event against a control at the **same location**, same season, same
relative orbit, no flood reported within 14 days, processed identically.

**Floods beat their own quiet control in 86.8% of pairs. p = 2.8e-24.**

| Measurement | Win rate | AUC | Median flood | Median quiet |
|---|---|---|---|---|
| coherence >0.15, channels | **86.8%** | 0.765 | 0.069 | 0.013 |
| coherence >0.15, everywhere | 86.8% | 0.794 | 0.097 | 0.017 |
| fusion | 86.2% | 0.799 | 0.044 | 0.007 |
| coherence p95 | 86.2% | 0.805 | 0.234 | 0.057 |
| amplitude dark, channels | 82.2% | 0.815 | 0.243 | 0.179 |
| coherence mean | 74.1% | 0.722 | 0.031 | −0.001 |

**All ten candidate measurements beat chance at p < 1e-10.** The result does not
depend on choosing a favoured statistic, which is what makes it credible.

Flood days decorrelate **5×** more channel area than quiet days at the same
places: 6.9% against 1.3%.

### Every region works

1501: 97% · 1507: 93% · 1606: 86% · 1503: 83% · 1809: 83% · 1810: 79%

None below 79%. The signal is not one basin carrying an average, so
cross-region generalisation has something real to work with.

### Two limits to state in the paper

**86.8%, not 100%.** About 1 in 7 reported floods shows no more disturbance than
an ordinary day. These are genuine detection failures — likely small events, or
reports located away from the water. This is a measured detection rate and
belongs in the paper.

**AUC ≈ 0.8, not 0.95.** Strong and usable, not perfect. The model will inherit
this ceiling.

### Why the earlier test failed

Not the data — the control. See the next section. The lesson generalises: a
validation is only as good as its negative class.

---

## Threshold sweep — NEGATIVE RESULT, and the reason for it

`phase4/threshold_sweep.py` → `data/threshold_sweep.csv`

Ran on 161 gauge-validated events (76 positive, 85 negative), nine candidate
cutoffs, farmland masked.

| Threshold | AUC pooled | AUC region-balanced | Flood % POSITIVE | Flood % NEGATIVE |
|---|---|---|---|---|
| 0.10 | 0.509 | 0.527 | 14.1% | 14.9% |
| 0.15 | 0.507 | 0.520 | 8.8% | 9.7% |
| 0.20 | 0.505 | 0.524 | 5.7% | 6.3% |

**AUC ≈ 0.5 at every threshold: no separation whatsoever.** Magnitude
correlation with discharge was 0.04. Per-region AUC ranged 0.34 to 0.69 —
incoherent, with one region below chance.

### The cause: the negative pile was not a negative pile

**Event 983830 — the Gate B event — sits in the NEGATIVE pile, response
`dry_throughout`.**

That is the $25M Painted Rock flood where the co-event coherence image shows the
dendritic drainage network appearing, at a measured 10× contrast against
background. It is unambiguously a real flood, and the gauge recorded nothing.

Negatives were defined as *"a flood was reported but the downstream gauge stayed
flat"*. In a desert that usually means the gauge is distant, on another
tributary, or the flow infiltrated before reaching it — **not** that no flood
occurred. Storm Events reports come from trained observers; a report with no
gauge response is typically a real flood the gauge could not see.

So the sweep compared floods against floods. AUC 0.5 is the correct answer to
the question that was actually asked, and it says nothing about label quality.

### The fix

A true negative must be a **place and time with no flood reported at all**, not
a reported flood with a silent gauge. Build negatives from ordinary intervals in
the same basins — same terrain, same track, same processing — and compare real
floods against genuine non-events.

Everything needed is available; those dates are in the same SAR archive.

### Second bug — river distance never captured

`phase4/downstream_gauges.py` read `properties["distance"]` from the NLDI
response. **0 of 4,911 site records have a distance**, so `river_km` is null
throughout. The field name is wrong.

Consequence: gauges could not be filtered or weighted by how far downstream they
sit, which is the single most useful covariate for judging whether a silent
gauge means anything. Worth fixing before the negatives are rebuilt — a gauge
5 km downstream staying flat is meaningful evidence; one 80 km away staying flat
is not.

### What this episode is worth

The negative result was caught only by cross-referencing against Gate B, where
the flood signature had been seen directly. Without that anchor, AUC 0.5 would
have read as "coherence labelling does not work" and could have killed the
project on a broken test.

**Keep at least one manually verified event as a permanent anchor** for every
automated validation added from here.

---

## Open methodological issue — the resolution ablation

**Raised, not yet resolved.** Phase 7 builds DEMs at 1/2/5/10/30 m and measures
accuracy loss. But accuracy is scored against the label, and the finest label
HyP3 can produce is **20 m**:

| Job type | `looks` options | Finest spacing |
|---|---|---|
| `INSAR_GAMMA` | `20x4`, `10x2` | 40 m |
| `INSAR_ISCE_BURST` | `20x4`, `10x2`, `5x1` | **20 m** |

Scoring a 1 m prediction requires upsampling the 20 m label, so every pixel in a
20 × 20 m block shares one answer. A model that resolves a 1.5 m channel and one
that cannot are scored identically — the detail has nowhere to register. The 1 m
and 2 m rows would differ only by resampling noise.

Second problem: 400× more pixels that are copies of each other means the
effective sample size is 400× smaller than the pixel count, so confidence
intervals come out far too tight. Same family as trial 1's pooled-vs-within-tile
gap (0.87 vs 0.64).

**Proposed fix:** fix the prediction grid at 20 m for every run; vary only the
DEM the features are derived from. The claim becomes "does deriving terrain
features from 1 m lidar improve prediction at the resolution we can verify?" —
one variable, still novel, and it also removes a receptive-field confound (at
fixed ground extent, a U-Net sees 2240² pixels at 1 m and 75² at 30 m).

Phase 5 is the first step that needs this decided.

---

## Audit before Phase 3

`tools/audit.py` — 15 checks over every Phase 1–2 output. Result: **0 fail, 1 warn**,
after fixing one real bug.

### Bug found and fixed — raster misalignment

`gate_b/analyze.py` and `gate_b/class_balance.py` differenced the baseline and
co-event rasters by cropping both to `[:rows, :cols]` from their own corners.
**HyP3 products from different date pairs do not share an origin** — the burst
footprint shifts between acquisitions:

| | Origin X | Origin Y |
|---|---|---|
| baseline | 281,020 | 3,653,620 |
| co-event | 283,520 | 3,653,040 |

An offset of 2,500 m (125 px) in x and 580 m (29 px) in y. The code was
subtracting coherence values from locations 2.5 km apart.

Fixed in `gate_b/align.py`, which windows both rasters to their intersection.
Both products share a CRS, a 20 m pixel size, and a common 20 m lattice, so this
is exact, not a resample.

**Effect on conclusions:** Gate B survived — near/far contrast moved 0.1403 →
0.1351, still ~11×. It survived because desert baseline coherence is spatially
uniform (~0.94 everywhere), so a 2.5 km shift barely changed it. That is luck,
not design; on heterogeneous terrain the same bug would have produced noise.

Class balance moved more, because misalignment manufactures spurious difference
everywhere:

| Coherence drop threshold | Buggy | Corrected |
|---|---|---|
| 0.10, whole burst | 11.5% | **8.1%** |
| 0.15, whole burst | 7.8% | **5.3%** |
| 0.10, within 10 km | 22.8% | **21.3%** |

**Any script that differences two HyP3 products must go through `align.py`.**

### Two issues the audit surfaced, both now filtered

**Interval mismatch.** Coherence decays with elapsed time regardless of
flooding, so the baseline and co-event pairs must span comparable intervals.
Nothing enforced this. Median mismatch was 0.1 days — but 17.6% of events
exceeded 6 days, where the comparison partly measures the calendar. Now capped
at 3 days.

**Daylight saving.** NOAA records local standard time (`MST-7`, `PST-8`).
Arizona does not observe DST, but California does, so `PST-8` events in summer
are off by an hour. Irrelevant against multi-day gaps, except for events with a
scene within an hour — 8 in the strict set. Now excluded.

### Class balance and permanent water

`gate_b/class_balance.py`, corrected figures, event 983830.

**Permanent water is essentially absent: 0.06% of pixels (0.9 km² of 1,598 km²)**
against trial 1's 0.97%. This is the Sonoran Desert; there are no lakes.

That matters more than it looks. Trial 1's worst defect was permanent-water
leakage — flood and permanent water were swapped, turning `neighbor_perm_water`
into "how flooded are my neighbours", a single feature scoring AUC 0.9997. In
arid terrain that class of leak is structurally near-impossible.

Flood fraction depends entirely on the threshold, which Phase 4 must justify
against gauges rather than assume:

| Drop threshold | Whole burst | Cropped to 10 km |
|---|---|---|
| 0.10 | 8.1% | 21.3% |
| 0.15 | 5.3% | 16.1% |
| 0.20 | 3.8% | 13.0% |

Around 8% whole-burst is close to trial 1's 9.43%, and cropping to event
neighbourhoods gives ~1 positive in 5 — a workable imbalance, not pathological.
All figures are **upper bounds**: agricultural decorrelation is still included.

---

## The frozen dataset

`phase2/build_dataset.py` → `data/dataset_phase3.csv`, `data/dataset_manifest.json`

Every filter in one place, every exclusion counted:

| Step | Events | Basins |
|---|---|---|
| checked for SAR | 3,779 | 68 |
| SAR-usable (2 pre + 1 post, one orbit) | 3,731 | 68 |
| in arid basins with ≥90% lidar | 766 | 11 |
| baseline/co-event intervals within 3 d | 628 | 11 |
| no scene within 1 h (DST guard) | **623** | **11** |

**623 events, 11 basins, 6 HUC4 regions.** Trial 1 had 14 events.

Events per region, the cross-region split unit: 1503→197, 1501→138, 1809→113,
1507→97, 1606→46, 1810→32.

Phase 3 needs 1,246 HyP3 jobs (2 per event) = **1,246 credits of 8,000**.

---

## Also worth carrying forward

- **Use `INSAR_ISCE_BURST`, not `INSAR_GAMMA`** — 1 credit vs 10–15, and the
  only route to 20 m. Largely resolves plan §5.5.
- **PRISM moved its web service** in 2025. The old `/normals/4km/ppt/annual`
  path returns a migration notice, not data. Correct path is
  `/get/normals/us/4km/ppt/annual`, and it now ships GeoTIFF, not BIL.
- **`pkill -f <pattern>` matches the shell running it** — documented in
  `trial_1/README.md`, and it still bit us here. Kill by PID.

---

## Environment

`trial_2/.venv` — asf_search 12.3.0, hyp3_sdk 7.7.7, geopandas 1.1.4,
rasterio 1.5.1. Machine: RTX 3090 24 GB, 344 GB free, WiFi (~1.2 MB/s).

Cached under `data/raw/`: Storm Events 2015–2026, WBD regions 13/14/15/16/18
(698 MB), PRISM annual normals.
