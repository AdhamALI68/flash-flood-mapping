# Next steps — settled decisions

Companion to `PROJECT_PLAN.md` and `FINDINGS.md`.
Written 2026-08-12. These are decided, not open for re-litigation.

---

## Summary of what was decided

1. **Download everything, delete nothing until the end.** Storage is the
   constraint, not bandwidth. Re-downloading costs days; deleting costs seconds.
2. **Download to the local disk** (updated 2026-08-12 — no external drive).
   371 GB free against 284 GB of downloads. Nothing is deleted; data moves to
   external storage later if space runs short. See §2c for the budget.
3. **Threshold is chosen by matching gauge records — never by model score.**
4. **Quality filtering happens after download**, with the rule written down
   before scores are looked at.
5. **The three counts must be reported separately** (see §2).
6. **Scope: the 11 STRICT basins.** INCLUSIVE (22) is deferred, not discarded —
   see §2b.
7. **Keep all 623 events** — do not restrict to gauge-validatable ones. Handle
   the regional imbalance by weighting regions equally in threshold selection.
   See §4c.
8. **5 km DEM buffer**, recover disk in tiling instead. See §2c.
9. **Resolution ablation predicts on a fixed 20 m grid**; only the input DEM
   resolution varies, and every rung is degraded from the 1 m source. See §3c.
10. **Report class balance the moment labels exist** — overall, per region, per
    event, per tile. See §5b.

---

## 1. Storage — download locally. SUPERSEDED 2026-08-12.

**Original position:** attach an external drive before downloading, because the
estimated 313–353 GB did not fit in 341 GB free.

**Now:** download to the local disk. Two things changed —

- trial 1's bulk data was deleted, freeing 31 GB (**371 GB free**)
- the DEM figure is measured rather than estimated: **148 GB** at a 5 km event
  buffer, not 204 GB for wall-to-wall coverage

Revised requirement, all measured:

```
SAR (all layers, full burst extent)   96 GB   measured, 204 MB/group x 470
DEM 1 m @ 5 km event buffer          148 GB   measured, 571 tiles
rain / soil / land cover              40 GB   estimated
──────────────────────────────────────────
total                                284 GB
free                                 371 GB
remaining                             86 GB   -> feature stack ~44 GB
```

**Nothing gets deleted.** Anything not immediately needed stays on disk and
moves to external storage later if space runs short. See §2c.

⚠️ The remaining 86 GB is adequate but not generous. If the feature-stack
recount after the first region comes in well above 44 GB, external storage
becomes necessary after all — that is the trigger to revisit this.

---

## 2. Report three DEM tile counts, not one

Currently only one number exists (755 tiles / 204 GB = full basin coverage).
Three are needed, because they answer different questions:

| Count | Meaning | Status |
|---|---|---|
| All 1 m tiles in the US | irrelevant, context only | known to be millions |
| **Full coverage, 11 STRICT basins** | wall-to-wall | **755 tiles, 204 GB** |
| **Tiles intersecting an event buffer** | what is actually needed | **NOT YET COUNTED** |

**Action:** compute the third. Buffer each of the 766 STRICT event points,
dissolve overlaps, intersect with the 3DEP tile index, count distinct tiles.

Report the count at several buffer radii (2 km, 5 km, 10 km) so the tradeoff is
visible. Gate B used 5 km, so include it.

Events cluster heavily (27 basins hold ≥50 events each), so dedup may cut this a
lot — or barely at all. Measure it; do not assume.

---

## 2b. Basin scope: download STRICT (11), preserve the INCLUSIVE list

**Decision: download the 11 STRICT basins only.**

Note the two sets overlap — INCLUSIVE (22) *contains* STRICT (11). They are not
two separate groups of basins.

| Set | Basins | HUC4 regions | Events | Rain |
|---|---|---|---|---|
| **STRICT** | 11 | 6 | 766 | < 250 mm |
| INCLUSIVE | 22 | 11 | 1,814 | < 500 mm |

**Why STRICT now:** it fits on available storage (INCLUSIVE roughly doubles the
DEM to ~408 GB, requiring ~520–570 GB total). Six regions is sufficient for a
cross-region split — train on 4, hold out 2. The "first arid dataset" claim
stays clean at < 250 mm with no exceptions.

**What INCLUSIVE would add, honestly:** 11 regions instead of 6, enabling proper
leave-one-region-out (as FloodPlanet does) rather than a single 4/2 split. It
also enables an experiment STRICT cannot support — regressing performance
against a rainfall gradient (68–500 mm) to *measure* how aridity affects
predictability, rather than asserting it. It also recovers Agua Fria (249
events, the largest basin by event count, excluded at 344 mm).

**Action:** persist the INCLUSIVE basin ID list (from `data/basin_aridity.csv`)
as a small standalone file. Do not discard it. If storage frees up, or a
reviewer asks whether results hold in semi-arid terrain, expansion is then a
download rather than a re-analysis.

**Caveat that applies to both sets:** 250 mm and 500 mm are arbitrary cut lines,
and these are basin *means*. Some basins span large gradients (Whitewater River:
68–1,189 mm). Screening each event by rainfall at its own point would be sharper
than either set. Not yet done; worth doing eventually.

---

## 2c. Buffer 5 km, save disk in tiling instead. DECIDED.

Measured tile counts (`phase5/count_dem_tiles.py`, `data/dem_tile_counts.csv`):

| Buffer | Tiles | Size |
|---|---|---|
| 2 km | 391 | 99 GB |
| **5 km** | **571** | **148 GB** |
| 10 km | 799 | 204 GB |

**Keep 5 km.** Buffer radius is decided *before* download and costs hours to
redo; tiling is computed *afterwards* from files already on disk and can be
redone freely. Spend storage on the irreversible choice.

Two supporting reasons: NOAA events are point pairs, not polygons, so a 2 km
buffer risks clipping real flood extent invisibly — and Gate B measured its
near/far contrast at 5 km, so keeping the radius keeps that result comparable.

**Recover the disk in tiling instead**: keep flood-bearing tiles only. That was
worth **+0.013** in trial 1 independently of any storage argument.

⚠️ The 44 GB feature-stack estimate is extrapolated from a **single** $25M
event, which is probably larger than typical. **Recompute after the first region
is tiled** rather than trusting it.

### Budget as decided

```
free                371 GB
SAR                 -96 GB
DEM 1 m @ 5 km     -148 GB
rain/soil/cover     -40 GB
                    ──────
remaining            86 GB   -> feature stack ~44 GB, ~42 GB spare
```

Downloads land on the local disk. Nothing is deleted; data moves to external
storage later if space runs short.

---

## 3. DEM strategy: 1 m local, 10 m basin-wide

**Decided.** Two separate DEM products, kept as separate channels:

- **1 m (3DEP lidar), local:** only within the event buffers. Used for
  micro-relief features where fine detail matters.
- **10 m (3DEP seamless), basin-wide:** used for flow routing and drainage-area
  context across the whole basin. ~3 GB for all 11 basins.

Precedent: trial 1 did exactly this split (30 m basin routing + 10 m local
slope) and it produced the single largest gain in that project (**+0.073** tile
AUC). Trial 1 then kept both versions side by side as 11 channels for a further
**+0.037**. Do the same here — keep both, do not choose.

### Two technical requirements

**a) The ablation rungs must be degraded from the 1 m source.**
Do NOT use native 3DEP 10 m as the "10 m rung" of the resolution ablation. Native
10 m and downsampled-from-1 m 10 m are different data. Mixing them reintroduces a
confound. Name them distinctly in code (e.g. `dem10_native_routing` vs
`dem10_from1m_ablation`).

**b) 1 m can only ever be local.**
75,529 km² at 1 m is ~75 billion cells per channel — hundreds of GB per channel.
Basin-wide 1 m feature stacks are not buildable at any disk size. This is not a
compromise forced by storage; it is the only possible shape for the pipeline.

### Verify the information loss (do not assume it is zero)

Pick one basin. Build features both ways:
- (A) 1 m everywhere in that basin
- (B) 1 m local + 10 m routing

Compare downstream scores. If equal, nothing was lost and it is measured, not
asserted. If different, the number goes in the paper.

---

## 3b. Lidar tile selection rule

The National Map indexes 3,373 tiles across the 11 basins (912 GB) because each
location is covered by 5–12 separate lidar projects. Only ~755 distinct ground
locations exist (~204 GB). A rule is needed for which project's tile to take.

### The confound this protects against

If one basin ends up on 2021 lidar and another on 2015, a cross-region
performance difference could come from **survey vintage rather than terrain**.
That would confound the cross-region test — one of the three headline claims. So
the rule must be applied identically everywhere, never varied by basin or
convenience.

### The stronger constraint: the DEM must predate the flood

A survey flown **after** an event contains that event's own scour and
deposition. Feeding it to the model is post-flood information used as a model
**input**, which violates the project's core constraint (§9). This is not
theoretical — the reason coherence detects these floods at all is that they
reshape channels and the change persists. A post-flood DEM therefore carries a
trace of the answer.

This means "prefer the most recent survey" is **wrong as stated**: it maximises
the number of DEMs that postdate their events.

### The rule

For each tile location, among projects covering it:

1. **Prefer the most recent survey that PREDATES the event.** Temporal ordering
   first, recency second.
2. If no pre-event survey exists, take the earliest post-event survey and
   **flag the event as `dem_postdates_flood`**.
3. Tie-break on quality level (QL1 beats QL2), then point density.
4. Apply identically in every basin.

### Log for every tile

`grid_key, project_id, survey_year, quality_level, chosen_reason`

### Log for every event

`event_id, tile_grid_key, survey_year, flood_year, gap_years (SIGNED),
dem_postdates_flood`

The gap must be **signed**, not absolute. Negative (DEM before flood) is clean;
positive (DEM after flood) is potentially leaking. Collapsing them to a magnitude
would hide exactly the distinction that matters.

If accuracy degrades with larger negative gaps, that is a finding worth
reporting — washes move sediment, so an older DEM is a staler description of the
channel. If accuracy *improves* with positive gaps, that is not a finding, it is
the leak showing up.

### ANSWERED — 40% of events have pre-event lidar

`phase5/lidar_vintage.py` → `data/lidar_vintage.csv`

| | Events | Share |
|---|---|---|
| **Pre-event 1 m lidar exists** | **252** | **40%** |
| Only post-event lidar | 370 | 59% |
| No 1 m lidar within 5 km | 1 | 0% |

This is the "roughly half" outcome: **restrict the terrain-only claim to the
clean subset, report the rest separately.**

**The clean subset survives cross-region testing**, which was the risk. It
spans all 6 HUC4 regions and fairly evenly — 1501: 43%, 1503: 45%, 1507: 31%,
1606: 43%, 1809: 31%, 1810: 59%. No region collapses.

Availability tracks 3DEP's rollout, as expected: 0% clean for 2015–2016 events,
7–8% for 2017–2018, 64–66% for 2022–2023, 91–100% for 2024+. The clean subset is
also recent enough to be relevant — median gap **−1 year**, mean −1.8, worst −6.

**40% is a lower bound.** Two thirds of the tile index dates via a delivery code
(`_D24`, `_B22`) rather than an explicit year, and delivery is at or after
acquisition. Some events marked "post-event" may have a pre-event survey. Fixing
this needs per-project acquisition metadata rather than name parsing; it can only
move the number up.

### How the pieces decompose

The threshold (§4) is a **labelling** decision and involves no DEM, so it can be
selected on all 358 gauge-validatable events regardless of lidar vintage. The
terrain-only **model** then trains on the 252 clean-lidar events. Only claims
needing both at once are limited to the intersection:

| Subset | Events | Regions |
|---|---|---|
| Gauge-validatable (threshold selection) | 358 | 6 |
| Pre-event lidar (terrain-only model) | 252 | 6 |
| **Both** (strongest combined claim) | **163** | **5** |

163 events across 5 regions is still 11.6× trial 1's 14.

---

## 3c. Resolution ablation — fix the prediction grid at 20 m. DECIDED.

**Every run predicts on a 20 m grid. Only the DEM the terrain features are
derived from varies: 1 m / 2 m / 5 m / 10 m / 30 m.**

### Why the grid cannot vary

HyP3's finest label spacing is 20 m (`INSAR_ISCE_BURST` at `5x1` looks; verified
against the delivered product). Scoring a 1 m prediction requires upsampling that
label, so every pixel in a 20 × 20 m block shares one answer. A model that
resolves a 1.5 m channel and one that cannot are scored identically — the 1 m and
2 m rows would differ only by resampling noise.

It also removes a receptive-field confound: at fixed ground extent a U-Net sees
2240² pixels at 1 m and 75² at 30 m, so varying the grid would vary resolution
**and** spatial context together.

Second-order but real: scoring at 1 m yields 400× more pixels that are copies of
each other, so effective sample size is 400× smaller than the pixel count and
confidence intervals come out far too tight. Same family as trial 1's
pooled-vs-within-tile gap (0.87 vs 0.64).

### The claim

> Does deriving terrain features from 1 m lidar improve flood prediction at the
> resolution we can actually verify?

Still novel — no existing arid flood dataset carries lidar — and now one variable
moves at a time.

### Rungs must be DEGRADED from the 1 m source

Do **not** substitute native products for ablation rungs. Native 3DEP 10 m and
1 m-downsampled-to-10 m are different data, produced by different sensors and
processing; mixing them reintroduces the confound the ablation exists to remove.

Name them distinctly in code so they can never be crossed:

```
dem10_native_routing      <- native 3DEP 10 m, basin-wide flow routing (§3)
dem10_from1m_ablation     <- 1 m degraded to 10 m, ablation rung only
```

The same applies at every rung: 2 m, 5 m, 10 m, 30 m all come from the 1 m
source. Only §3's basin-wide routing layer uses a native product, and it is not
part of the ablation.

---

## 3d. Labels are CONTINUOUS. The threshold is a reporting choice. DECIDED.

Measured 2026-08-16: only **164 of 617 events (27%)** have usable gauge
evidence — 78 positive, 86 negative, 194 with no gauge record at all. Region
1606 has none, and region 1501 has only 3 negatives.

Choosing one hard threshold from 27% of the data and stamping it onto the other
73% makes that number load-bearing for every downstream result. It does not have
to be.

### The decision

**Train on the continuous coherence drop, not a binary mask.**

```
hard label   drop 0.31 -> 1        drop 0.14 -> 0
soft label   drop 0.31 -> 0.31     drop 0.14 -> 0.14
```

Phase 3 already stored exactly this, so nothing needs re-downloading.

Why this is better than picking a line up front:

- **The threshold stops being load-bearing.** It moves out of the data and into
  the reporting step, where changing it costs seconds instead of a re-run.
- **It keeps information a hard label destroys.** 0.31 and 0.75 are very
  different floods; a binary mask calls both "flood".
- **The weakest point of the project shrinks.** "We calibrated on 27% of events"
  matters much less when the calibration only affects how results are reported.

### What the gauges are still for

1. **A reference threshold**, for reporting and for comparability with published
   work that uses binary masks (Kuro Siwo, FloodPlanet).
2. **A magnitude check** (§4d) — a stronger test than yes/no.
3. **The held-out-region confirmation** (§4b step 2), unchanged.

### Reporting rule

Report the headline result at **several thresholds**, not one. If the conclusion
holds across them, it does not depend on the choice, and a reviewer can see that
directly. If it flips, that is a genuine finding about how sensitive the result
is, and it belongs in the paper rather than hidden behind one number.

Binary metrics (IoU, F1) still need a threshold — but applied at **scoring
time**, on stored continuous values, never baked into the training data.

---

## 4d. Use gauge MAGNITUDE, not just yes/no

Currently the gauges are used only as "did discharge rise". They also record how
much, and that supports a stronger test:

> A bigger flood should produce a bigger labelled area.

Fit labelled flood extent against gauge peak discharge across the 164
gauge-validatable events. This checks that labels **scale** correctly, not merely
that they fire.

It is a much harder test to pass by accident. A threshold can separate two piles
while still producing extents unrelated to how big each flood actually was;
correlation with measured discharge cannot be faked that way.

Report the correlation and state it plainly. A weak relationship is itself
informative — it would suggest coherence drop records *disturbance* rather than
*extent*, which changes what the dataset can honestly claim to measure.

---

## 4. Threshold selection — gauge-matched, NOT score-matched

**This is the most important instruction in this file.**

The coherence-drop threshold (currently 0.15, chosen arbitrarily) defines the
label. The label is the answer key. Tuning an answer key against model
performance invalidates every result downstream, silently, with no error
message.

### The rule

> The threshold is chosen by agreement with USGS gauge records.
> Model scores are never consulted during threshold selection.

### Procedure

1. **Pull USGS NWIS daily discharge** for gauges in/downstream of the 11 STRICT
   basins, 2015–2026. Free, public API. Small files — do this in parallel with
   the big downloads, it blocks nothing.

2. **Split events into two piles using gauges only:**
   - **Positives:** gauge shows a clear discharge spike on/around the event date
   - **Negatives:** a flash flood was reported, but the gauge shows no response

   The negative pile is essential and easy to forget. Without it you can only
   measure "did we find floods", not "did we invent floods that did not happen".

3. **Compute coherence drop for every event in both piles.**

4. **Sweep thresholds** (0.05 → 0.30). For each, record true positives and false
   positives. Choose the value that best separates the two piles.

5. **Hold out regions.** Select the threshold using a subset of the 6 HUC4
   regions. Test on the regions not used for selection. If it holds, it is real
   rather than fitted to particular basins.

6. **Check whether one threshold fits everywhere.** Sandy vs rocky basins may
   differ. If they separate cleanly, that is a reportable finding, not a
   failure.

### Prerequisite

**Apply the NLCD land-cover mask BEFORE computing any of this.** Gate B found
irrigated farmland decorrelates in both the baseline and co-event pairs,
independent of flooding. Unmasked farmland will contaminate the threshold sweep.

### 4b. Transferability check — run BEFORE the sweep

The threshold is calibrated on the 358 gauge-validatable events and then applied
to all 623. **That is an assumption, not a measurement.** Gauges are sited near
infrastructure, roads, and perennial reaches, so the gauged subset is a
convenience sample, not a random one.

Restricting the project to gauged events is not a fix. It swaps this assumption
for a worse one — "gauges represent the desert" — while cutting training events
from 252 to 163 and dropping region 1606 entirely, which weakens the
cross-region claim that is contribution #3.

So the assumption gets tested rather than avoided.

#### Step 1 — Are gauged and ungauged events comparable?

Compare the two groups on covariates measurable for **both**:

| Covariate | Source | Needs a download? |
|---|---|---|
| Mean annual rainfall | PRISM, sampled at the event point | no, raster held |
| Elevation | USGS EPQS point service | no |
| Slope | DEM neighbourhood | yes |
| Basin area | WBD | no |
| Dominant soil class | SSURGO / Soil Data Access | point API |
| Land cover | NLCD / MRLC | point API |

Report full distributions for both groups, not just means, plus a standardised
difference so overlap is quantified rather than eyeballed. Substantial overlap
makes transfer defensible. If ungauged events are systematically steeper, drier,
or otherwise distinct, **that is a real finding and goes in the paper.**

#### Step 1 RESULT — measured 2026-08-12

`phase4/transferability.py` → `data/transferability.csv`

| Covariate | Gauged (n=358) | Ungauged (n=265) | SMD | |
|---|---|---|---|---|
| Rainfall at event point | 163.0 mm | 171.8 mm | 0.066 | negligible |
| **Elevation** | **405 m** | **876 m** | **−0.481** | **small** |
| Basin area | 5,222 km² | 7,208 km² | −0.160 | negligible |

**Ungauged events sit 471 m higher.** Gauges are installed in valley bottoms
near roads and perennial reaches; ungauged events are in the headwaters. Higher
ground here means steeper, rockier, different runoff generation. IQRs still
overlap (240–773 vs 428–1044), so transfer is defensible — but not free, and it
must be reported.

**Rainfall is effectively identical**, which is the reassuring part, since
aridity is the variable the dataset is defined by.

**The larger confound is regional, and was not anticipated:**

| Region | Gauged | Ungauged |
|---|---|---|
| 1503 | **173** | 24 |
| 1501 | 31 | **107** |
| 1606 | **0** | 46 |
| 1809 | 50 | 63 |
| 1507 | 79 | 18 |
| 1810 | 25 | 7 |

Calibrating on gauged events means calibrating **48% on region 1503**, then
applying it to a population that is 40% region 1501. The groups differ by region
more than by terrain. This is what §4c's equal-region weighting exists to fix.

Still pending: slope (needs the DEM download), SSURGO soil class, NLCD land
cover.

#### Step 2 — Does the threshold hold on held-out regions?

Select the threshold on a subset of the 6 HUC4 regions, evaluate on regions not
used for selection. Holding across unseen regions is the strongest available
evidence it will hold on ungauged events, since ungauged events are mostly
"unseen terrain" in the same sense.

#### Step 3 — Does one threshold fit every gauged basin?

Sweep per basin as well as pooled. If sandy and rocky basins separate cleanly,
report it. Basin-dependent thresholds mean transfer to ungauged basins is
riskier, and that must be stated rather than smoothed over.

#### Step 4 — State it plainly in the paper

> The threshold was calibrated on 358 gauge-validatable events and applied to
> all 623. Comparability evidence between gauged and ungauged events is in
> [table]. The held-out-region result is [result].

Report agreement rates honestly, including failures.

#### Region 1606 has ZERO gauge coverage

Ivanpah-Pahrump Valleys, **46 events, no downstream gauge for any of them**.
Applying the threshold there is an untested extrapolation, not a validated
result.

- Do **not** fold 1606 events in silently. Flag them wherever they are counted.
- **Prefer not to use 1606 as the final held-out test region** — a cross-region
  result on a region whose labels were never validated confounds "does the model
  transfer" with "are the labels right there".

### 4c. Scope — keep all 623 events. DECIDED.

**Do not restrict the project to gauge-validatable events.**

Dropping ungauged events costs region 1606 entirely (6 regions → 5), weakening
the cross-region claim, and shrinks the both-conditions subset to 163. Worse, it
introduces a hidden bias: gauged events average 405 m elevation against 876 m
ungauged, so "gauged only" silently means "low-elevation deserts only".

Gauges are not the only validation. Every event already carries two independent
checks — a **NOAA storm report** and **radar rainfall**. Gauges are a third,
stronger check available on a subset, not the sole source of truth.

#### How the regional imbalance is handled

1. **Weight regions equally during threshold selection.** Region 1503 holds 173
   gauged events, 48% of the gauged set, and must not dominate the fit.
2. **Report the threshold per region as well as pooled.** Wide scatter across
   regions is a finding, and the transfer claim weakens accordingly.
3. **Subsample when building the selection set** for held-out-region testing, so
   1503 does not dominate — otherwise the held-out test inherits the same
   imbalance it is meant to detect.

#### Pre-registered elevation check

Stated **before** the number is seen, so the interpretation cannot be fitted to
the result:

> If the best threshold varies systematically with elevation among gauged
> events, then extrapolating to the higher ungauged terrain is less safe, and
> the transfer claim must be weakened in proportion.

If it does not vary with elevation, the extrapolation across a 471 m gap is
much better supported.

#### Region 1606

46 events, zero gauges. **State in the paper that it cannot be gauge-validated.**
Do not fold it in silently, and do not use it as the final test region.

#### Report both results

- the **full 623-event** result, with the limitation stated
- the **gauge-verified subset** result alongside it

If the conclusion holds on both, that is stronger than either alone.

#### Fourth validation — Sentinel-2 optical

Attempt optical cross-checking wherever cloud-free imagery exists. Desert skies
usually are, and it is cheap. It will not cover every event — S2 sees water only
while it is present, which is the timing problem coherence was chosen to avoid —
but where it lands it is independent of both SAR and gauges.

### Known limits — state these in the paper

- Gauges are sparse in deserts. Some basins may have none, or only a gauge far
  from the affected reach. Expect to validate on a subset of events.
- This validates **whether an event occurred**, not **which pixels were wet**.
  It validates the rule, not every label. No flood dataset has per-pixel ground
  truth; Kuro Siwo's came from five experts drawing by hand, which is also
  judgement.
- Report agreement rates honestly, including disagreements.

---

## 5. Quality filtering — after download, rule written first

Dropping unusable tiles is fine and expected. The discipline:

- **Write the drop rule down before looking at any model score.**
- Rules must be about data quality (missing pixels, decorrelation, no lidar
  coverage, cloud), never about whether a tile helps or hurts performance.
- Log every dropped tile and the reason. Report counts in the paper.

Trial 1 precedent: 5,427 tiles → 4,833 passing quality checks. That ratio was
reported openly.

---

## 5b. Report the class balance as soon as labels exist. DECIDED.

The moment the threshold is set and labels are generated, report **at 20 m**:

- flood pixel percentage **overall**, **per region**, and **per event**
- the distribution of flood fraction **per tile**
- how many tiles exceed **>1%** and **>5%** flood

Expect heavy imbalance. Trial 1 had **69.3%** of tiles with no flood at all, and
**skip-empty-tile training was worth +0.013** there — so the tile-level
distribution is not bookkeeping, it directly determines a training decision.

### What is already known, from Gate B

Measured on event 983830 (`gate_b/class_balance.py`), after the alignment fix:

| Drop threshold | Whole burst | Within 10 km |
|---|---|---|
| 0.10 | 8.1% | 21.3% |
| 0.15 | 5.3% | 16.1% |
| 0.20 | 3.8% | 13.0% |

Around 8% whole-burst is close to trial 1's 9.43%. **All figures are upper
bounds** — agricultural decorrelation is still included and NLCD masking will
pull them down.

**Permanent water is effectively absent: 0.06% of pixels** against trial 1's
0.97%. That removes an entire class of failure — trial 1's worst defect was
permanent-water leakage, where a swapped label turned `neighbor_perm_water` into
a feature scoring AUC 0.9997 on its own. In arid terrain there is almost no
permanent water to leak from.

⚠️ These come from **one** event, a $25M flood, probably larger than typical.
Treat as provisional until recomputed across a full region.

---

## 6. Sample size: report events and regions, not tile count

The extrapolated ~150,000 tiles is a **compute-cost** figure. It is not
statistical power.

- 470 event-groups across **6 HUC4 regions**; 27 basins hold ≥50 events each.
- The same ground is re-tiled across many storms → tiles are strongly
  correlated.
- **Compute confidence intervals over events or basins, not tiles.**

This is the same failure mode as trial 1's pooled-vs-within-tile gap (0.87 vs
0.64): many pixels, far fewer independent observations.

Headline framing should be "6 regions, 470 events", with tile counts secondary.

---

## 6b. Model size — a larger dataset does NOT require a larger model

Recorded because it came up as a concern and the answer is already measured.

**Keep the small U-Net.** In trial 1 the 961 k-parameter U-Net **beat** a 24.5 M
ResNet-34 U-Net with ImageNet pretraining: **0.7736 vs 0.7580**. A model 25×
larger performed worse. Scaling up is not indicated.

**Dataset size on disk is unrelated to GPU memory.** Training streams small tiles
in batches; the full corpus is never resident. ~500 GB on an external drive is a
read-throughput consideration, not a memory one. An RTX 3090 (24 GB) is ample.

**Do not switch to a transformer.** Transformers lack the convolutional
inductive bias and need more data to learn spatial structure from scratch. At
this dataset size a transformer is likelier to underfit or memorise.

**Do not use KAN.** Already published for flood mapping (GMKAT, Dec 2025;
FloodKAN), so not novel — and KAN-based transformer layers have been measured
training >100× slower on identical hardware. Fatal on a single GPU.

Carry forward the four changes that measurably worked in trial 1:

| Change | Gain (within-tile AUC) |
|---|---|
| hybrid multi-scale features | +0.037 |
| skip-empty-tile training | +0.013 |
| elevation prior in the output (`logit = U-Net(x) + softplus(w)·(−z)`) | +0.014 |
| ranking loss matched to the reported metric | +0.024 |

Baseline to beat is the **elevation-only sorting rule**, not zero.

---

## 7. Two corrections to earlier numbers

**a) Tile size — unit confusion, not a disagreement.**
"224×224 at 10 m covers 5 km²" refers to **area**: 2.24 km × 2.24 km = 5.0 km².
This was read as a 5 km linear extent. Both statements describe the same tile.
Trial 1 tiles are 2.24 km across, 5.0 km² in area. Flagged because
length/area confusion is exactly the class of fault that yields a believable
wrong number with no error.

**b) Trial 1 tile count — 6,666 does not match the report.**
The trial 1 report states 5,427 tiles, 4,833 passing quality checks, 941 in the
test set. Where does 6,666 come from? Resolve before it is used in any
comparison table.

---

## 8. Order of work

Parallel tracks — the downloads block nothing else:

```
TRACK A (long-running, downloads to the LOCAL disk — §1 superseded)
  count tiles at 2/5/10 km buffers      ← §2  DONE: 571 tiles @ 5 km, 148 GB
      ↓
  download SAR (96 GB, ~20 h)           ← 1/470 groups done
      ↓
  download DEM (1 m @ 5 km buffer + 10 m basin-wide)
      ↓
  download rain / soil / land cover
      ↓
  recount feature-stack size            ← §2c, before trusting the 44 GB figure

TRACK B (small files, runs alongside Track A)
  pull USGS gauge records              ← DONE: 156 gauges, 10 of 11 basins
      ↓
  downstream connectivity via NLDI     ← DONE: 358/623 validatable, 1606 = 0
      ↓
  transferability step 1               ← DONE: elevation SMD −0.481, §4b
      ↓
  build positive / negative event piles
      ↓
  apply NLCD farmland mask
      ↓
  threshold sweep vs gauges            ← §4, equal region weights (§4c)
      ↓
  hold-out region confirmation         ← not 1606 (§4c)
      ↓
  Sentinel-2 optical cross-check       ← §4c, where cloud-free
      ↓
  report class balance                 ← §5b

THEN
  Phase 5 feature stacks (needs §3 decided — it is)
  Phase 6 baselines + ceiling
  Phase 7 resolution ablation
  Phase 8 cross-region test
```

---

## 9. Standing rules (carried from trial 1)

- Split by **event/basin**, never randomly.
- Report **within-tile** metrics, not just pooled.
- Report a **noise floor** from repeated runs before claiming any gain.
- Use **paired** significance tests on the same tiles.
- Verify any file that claims a resolution actually has that resolution
  (cf. the ALOS PALSAR trap: advertised 12.5 m, actually resampled 30 m).
- Short single-fold test run before any full run, to catch silent faults.
- **Post-flood SAR: labels only, never model inputs.** This is the constraint
  the entire contribution rests on.
