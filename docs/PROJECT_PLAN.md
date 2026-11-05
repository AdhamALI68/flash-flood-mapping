# Arid Flood Dataset + Terrain-Only Benchmark — Verified Project Plan

**Author:** Adham Ali
**Status:** planning, not started
**Purpose of this file:** hand to Claude Code as the working spec.

---

## 0. What this project is

Build the **first flood dataset built specifically for arid terrain**, label it with a
method that works where standard methods fail, validate those labels against
independent ground records, then use it to measure **what terrain + rainfall alone can
and cannot predict** in dry regions.

Three claimed contributions:

1. **Dataset.** No arid-focused flood dataset exists. Existing ones (Sen1Floods11,
   MMFlood, S1S2-Water, SpaceNet 8, FloodPlanet, Sen2GF3Floods, Kuro Siwo) are
   general-purpose, urban, or riverine.
2. **Resolution ablation.** With 1 m lidar, measure how much prediction accuracy is
   lost as DEM resolution degrades (1 → 2 → 5 → 10 → 30 m). Impossible with any
   existing flood dataset — none have lidar.
3. **Cross-region generalization in arid terrain.** Train on some basins, test on
   fully held-out ones.

**Hard constraint carried from prior work:** DEM/terrain must be load-bearing.
Post-flood imagery may be used to make **labels** but must NEVER be a model **input**.

---

## 1. Evidence the gap is real

| Claim | Evidence |
|---|---|
| Arid flood detection is measurably broken | FloodPlanet leave-one-region-out: arid regions scored **0.37 / 0.35 mIoU** (Uzbekistan, US-Kansas) vs **0.82 / 0.81** (US-Oklahoma, US-Carolina) |
| Global flood products skip arid areas entirely | 2024 *Remote Sensing of Environment*: current global flood mapping products **exclude arid areas** due to sand/water confusion |
| Cross-region testing is rare in flood ML | April 2026 systematic review (2018–2025): dependence on high-res data **limits model transferability across regions** |
| Arid regions are under-studied | Same review: Middle East, Central Asia, southern Africa under-represented vs monsoon Asia |
| Even recent arid work skips independent validation | 2026 Frontiers paper (Egypt Red Sea Mountains) states its own validation was **internal, not independent spatial validation** |

**Competition to differentiate from (both 2026):**
- Water Research: cross-regional flood surrogate transfer (Shenzhen → Richmond). *Not arid; surrogate of simulations, not observed floods.*
- arXiv 2606.06524: physics-guided UNet+FNO, held-out regions, IoU 0.82. **Uses SAR-derived velocities as input** → violates our terrain-only constraint. This is our clearest differentiator.

---

## 2. Data sources — all verified free/open

| Layer | Source | Resolution | Notes |
|---|---|---|---|
| Terrain | USGS **3DEP** | **1 m** lidar; 10 m seamless | Public domain. AWS bucket, no login. 1 m tiles 10–50 MB each |
| Rainfall | **NEXRAD** QPE | ~1 km | Radar-based; valuable where gauges sparse |
| Soil | **SSURGO** (NRCS) | field-level | Drives infiltration vs runoff |
| Land cover | **NLCD** | 30 m | |
| Event records | **NOAA Storm Events DB** | polygons since 2007-10-01 | 74,814 US flash flood events 1996–2017 |
| Unified inventory | **FLASH DB** (NSSL) | — | USGS streamflow + NWS reports + SHAVE surveys |
| Discharge truth | **USGS NWIS gauges** | point | **Label validation** |
| SAR | **Sentinel-1** IW SLC + GRD | 10 m | Via ASF |
| SAR processing | **ASF HyP3** | — | Free cloud InSAR; **outputs coherence GeoTIFF by default** |

**Study area:** Arizona, New Mexico, Utah, Nevada, southern California
(arroyo country — the American wadi).

---

## 3. Phases

### Phase 1 — Event list
**Verified: YES.** NOAA Storm Events DB is public with bulk downloads; polygons exist
from 1 Oct 2007; FLASH DB ships as CSV/shapefile/KMZ.

- Filter: event type = Flash Flood, states above, **date ≥ 2015** (Sentinel-1 era).
- Cross-reference FLASH DB for gauge-linked events.
- Group events by **basin**, not just count. Need **≥ 4–5 distinct basins**.

**Output:** `events.csv` — date, time, polygon, basin id, nearest gauge id.

---

### Phase 2 — SAR availability check ⚠️ **KILL GATE**
**Verified: tool exists, check NOT yet run.** I could not query the archive directly.
This must be executed first — it decides whether the project lives.

Use `asf_search` (Python) or ASF Vertex Baseline/SBAS tool.

For each event find:
- a **pre-event** IW SLC, and
- a **post-event** IW SLC on the **same track, same polarization, same mode**
  (hard requirement for InSAR).

**Known risks, both confirmed:**
- Revisit is **6–12 days** (12 days between Sentinel-1B failure in 2022 and
  Sentinel-1C launch Dec 2024) — explicitly named as a limit for rapidly evolving floods.
- In semi-arid areas evaporation/infiltration is **very rapid**; a clear late-summer day
  after rainfall can evaporate a large amount of water.
- Perpendicular baseline must be **< ~3/4 of the ~5 km critical baseline**, else
  interferogram is too noisy.

**Kill criteria:**
- ✅ Continue if ≥ 4 distinct basins have ≥ 5 usable events each.
- ⚠️ Replan if 2–3 basins survive (consider adding Texas/Nevada, or widening the window).
- ❌ Stop and switch to the polygon-susceptibility fallback (§6) if < 2 basins survive.

**Output:** `sar_availability.csv` + survival-rate summary.

---

### Phase 3 — Label pipeline
**Verified: YES, with a trap and a fix.**

Method: **coherence + amplitude change detection** (2024 RSE paper: fusing SAR
coherence and amplitude *dramatically* improves arid flood mapping; co-polarized
amplitude + coherence change detection is optimal for arid).

> **Why coherence matters:** amplitude asks "is water here now?" Coherence asks "did
> this surface change?" A flash flood moves sand and scours channels — that
> disturbance persists **after** the water drains. Coherence partly defeats the
> revisit-timing problem.

⚠️ **TRAP:** coherence requires **SLC**, not GRD. SLC is ~8× larger than GRD, needs
InSAR processing (coregistration, back-geocoding), takes ~10s of minutes per pair on
CPU, and **Google Earth Engine does not host SLC/coherence** — GEE cannot do this.

✅ **FIX: use ASF HyP3.** Free, cloud-based (AWS), no SNAP install, no powerful machine
needed. InSAR products include **amplitude, coherence, and unwrapped phase GeoTIFFs by
default**, plus a water mask. Has a **Python SDK + REST API** for batch jobs.

HyP3 caveats:
- HyP3 Basic has a **credit limit** (HyP3+ sells more).
- Products expire **14 days** after processing → download promptly, script it.

**Output:** per-event flood mask rasters.

---

### Phase 4 — Label validation ⭐ **THE CREDIBILITY STEP**
**Verified: YES.** USGS NWIS gauge data is public via API.

An automated labeling pipeline nobody checked is weak. One validated against
independent physical records is the selling point of the whole dataset.

For every event, cross-check the SAR label against:
1. **USGS gauge discharge peak** — did a flood actually occur, at that time?
2. **NWS storm polygon** — does the labeled extent fall inside the reported area?
3. **Sentinel-2 optical** where cloud-free (desert skies usually are).

**Report agreement rates honestly, including failures.** Publish the disagreement
cases. This is the section reviewers will judge hardest.

**Output:** `label_validation.csv` + agreement statistics.

---

### Phase 5 — Feature stack
**Verified: YES.** Port existing HAND / TWI / SPI / slope / flow-accumulation code.

⚠️ **Carry forward the fix from the Kuro Siwo work:** compute flow routing over the
**complete basin**, not per tile. Per-tile routing was worth **+0.073 tile AUC** when
corrected — the single largest gain in that project. Do not repeat the mistake.

⚠️ **DEM resolution warning:** do **not** use ALOS PALSAR 12.5 m as a "high-res" DEM.
Per ASF, the bundled DEM is *usually SRTM with ellipsoid correction, resampled* — it is
30 m data on a 12.5 m grid. A Morocco study found its accuracy barely differs from
SRTM3/NASADEM. Use 3DEP lidar.

Build the stack at **1 m, 2 m, 5 m, 10 m, 30 m** (degrade from the 1 m source so the
only variable is resolution).

Add: NEXRAD rainfall, SSURGO soil, NLCD land cover.

⚠️ **Rain resolution note:** NEXRAD ~1 km vs a 2.24 km tile — rain finally varies
*within* a tile (IMERG at ~11 km could not). Expect rain to help most with **how much**
floods, which the elevation-only rule structurally cannot do.

---

### Phase 6 — Baselines + information ceiling
Reuse the prior methodology directly:
- elevation-only sorting baseline
- oracle per-tile-threshold ceiling, **computed per region**
- noise floor from repeated runs
- paired significance tests

**Headline question:** is the arid pre-flood ceiling above Kuro Siwo's **0.2542** IoU?
Either answer is a publishable result.

---

### Phase 7 — Resolution ablation ⭐ **THE NOVEL EXPERIMENT**
Same model, five DEM resolutions, everything else fixed.

Tests the prior report's *unproven* claim that 1–2 m flood walls invisible at 30 m are
part of what is missing. Currently a discussion-section hypothesis; this turns it into a
measured number.

Publishable either way:
- Fine resolution helps a lot → justifies lidar investment.
- Fine resolution barely helps → **more** interesting; means expensive lidar for flood
  mapping may be unnecessary.

---

### Phase 8 — Cross-region test
Train on some basins, test on **fully held-out** basins. Paired tests + noise floor.

Note: leave-one-region-out is **not itself novel** (FloodPlanet does it). Novelty is
applying it to arid terrain with terrain-only inputs.

---

### Phase 9 — Zero-shot Egypt transfer (STRETCH — do not block core)
Apply the US-trained model to Egyptian wadi events. Candidate events with documented
Sentinel-1 coverage:
- Wadi El-Natrun, **4 Nov 2015** (S1 + S2 before/after, published)
- New Cairo City, **24–27 Apr 2018** (S1 pre/post, F1 73–78% vs Sentinel-2)
- Red Sea Mountains wadis, **2016–2025** (S1 inventories, Otsu thresholding, published)

Relevant finding: a recent study found flood models transfer **best** in hot desert and
Mediterranean climates (Libya F1 = 0.88), worst in humid vegetated regions. Encouraging
for arid→arid transfer.

---

## 4. Model choice

**Recommendation: U-Net. Keep the architecture boring on purpose.**

Reasoning:
- Novelty is the **dataset + experiments**, not the architecture. A trendy layer invites
  "what's actually new here?"
- Transformers need more data to learn spatial structure from scratch; CNNs get that
  inductive bias for free. At this dataset size a transformer likely underfits or memorizes.
- ⚠️ **Avoid KAN.** Already published for flood (GMKAT, Dec 2025; FloodKAN), so not novel —
  and a recent paper found KAN-based transformer layers training **>100× slower** on the
  same hardware. Fatal on one GPU.
- The prior U-Net (961 k params) already beat a 24.5 M ResNet-34 U-Net with ImageNet
  pretraining (**0.7736 vs 0.7580**). Bigger did not help there.

**Carry forward what already worked:**
- elevation prior written into the output (`logit = U-Net(x) + softplus(w)·(−z)`) — worth **+0.014**
- hybrid multi-scale features — **+0.037**
- skip-empty-tile training — **+0.013**
- ranking loss matched to the reported metric — **+0.024**

**Comparison set:** elevation-only rule (the real baseline) → U-Net → U-Net + rain +
soil. Report against the elevation rule, not just against nothing.

---

## 5. What I could NOT verify

Be honest about these:

1. **The Phase 2 SAR check itself.** Never run. Tool confirmed, result unknown. This is
   the single biggest open risk.
2. **Exact-intersection novelty search** — "terrain-only + arid + cross-region" as one
   combined query. Components verified separately; do a final check before writing.
3. **NEXRAD product choice** (Stage IV vs MRMS vs raw Level II) — not compared.
4. **3DEP 1 m coverage over the specific target basins.** National coverage is expanding
   but **not complete**; must be checked per basin.
5. **HyP3 credit limits** vs the number of pairs needed.

---

## 6. Fallback if Phase 2 fails

Use NWS storm-report **polygons directly as labels** (74,814 events, polygons since 2007).
- ✅ Far more data, no SAR timing problem, no coherence processing.
- ❌ Coarser question: "does this area flood" (susceptibility), not "which pixels flood".
- ❌ Loses the resolution ablation — polygons are too coarse to say anything about it.
- ❌ Flood susceptibility is a crowded field.

Treat as a real fallback, not the plan.

---

## 7. Order of work

```
Phase 1  events.csv
   ↓
Phase 2  SAR availability  ← KILL GATE, do this before anything else
   ↓
Phase 3  labels (HyP3 coherence + amplitude)
   ↓
Phase 4  validate labels vs gauges  ← credibility
   ↓
Phase 5  feature stack at 5 resolutions
   ↓
Phase 6  baselines + ceiling
   ↓
Phase 7  resolution ablation   ← novel experiment
   ↓
Phase 8  cross-region test
   ↓
Phase 9  Egypt zero-shot (optional)
```

**Do Phase 2 first.** Everything downstream depends on it and it is cheap to run.

---

## 8. Standing rules

Carried from the prior project, where they caught several silent faults:

- Split by **event/basin**, never randomly. Random splits inflate everything.
- Report **within-tile** metrics, not just pooled. Prior work: same predictions scored
  0.87 pooled vs 0.64 within-tile.
- Always report a **noise floor** from repeated runs before claiming a gain.
- Use **paired** significance tests on the same tiles.
- Verify any file that claims a resolution actually has that resolution (see the ALOS
  PALSAR trap in Phase 5).
- Run a short single-fold test run to catch silent faults before any full run.
- Post-flood imagery: **labels only, never inputs.**
