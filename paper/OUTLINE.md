# IGARSS 2027 — paper plan

**Format:** 4 pages, IEEE two-column, including references. That is roughly
2,400 words plus 3 figures. Everything below is chosen to fit that, which means
most of what we measured has to be left out.

---

## The one story

Four pages cannot hold both "we built a dataset", "we predict floods" and "here
is why prediction is bounded". Picking one:

> **InSAR coherence change is widely used to label floods. We quantify what it
> can and cannot support, using a new dataset of 617 desert flash floods with
> paired quiet-date controls.**

Why this and not "our model predicts floods":
- our AUC is 0.695, which is unremarkable on its own
- the ceiling analysis is genuinely novel and useful to the IGARSS audience,
  who are the people choosing InSAR settings for flood work
- it turns our weakest number into the contribution

## Title

*How well can InSAR coherence change label flash floods? A quantified ceiling
from 617 events with paired controls*

## Abstract (150 words)
- desert flash floods are underrepresented; InSAR coherence change is the usual
  labelling route
- we build 617 events across six arid basins, each with a paired quiet-date
  control at the same location, season and orbit
- the label separates floods from controls at 86.8% (p=2.8e-24) at event level
- but at pixel level it is bounded: graded against independent Sentinel-2
  optical labels the same model scores 0.751 vs 0.643
- a gradient-boosted tree on single pixels matches a U-Net with 2.24 km context
- at 5 looks the common 0.15 drop threshold sits at 1.6 sigma of the estimator's
  own noise; a 3 sigma criterion fires on 0.00% of pixels
- we recommend an operating resolution and report the looks trade-off

## I. Introduction  (~400 words)
- flash floods in arid basins, why they matter, why they are hard
- InSAR coherence change detection as the standard labelling method
- the gap: nobody quantifies what that label can support at pixel scale
- our contributions, three bullets

## II. Data  (~500 words, 1 table)
- 617 events, six HUC4 regions, 2015-2025, Sentinel-1 burst InSAR via HyP3
- three-scene design: baseline pair + co-event pair, so natural decorrelation
  is separated from flood decorrelation
- 174 paired quiet-date controls: same location, season, orbit, no flood
  reported. This is the part reviewers will like
- predictors: terrain at five resolutions from 1 m lidar, basin flow routing,
  MRMS rainfall, catchment-routed rainfall, POLARIS soil, WorldCover

## III. Method  (~400 words)
- label: coherence drop, continuous, thresholded at scoring time
- model: small U-Net with an elevation prior + multi-scale head; XGBoost
  baseline on the same features
- evaluation: split by HUC4 region, never randomly. 80% of events sit within
  5 km of another, so a random split memorises terrain (measured: 0.695 -> 0.826)

## IV. Results  (~700 words, 3 figures)

FIG 1  the ceiling, four panels or one grouped bar
   InSAR-graded 0.643 vs optical-graded 0.751 (same model, same 6.9 M pixels)
   XGBoost 0.6954 vs U-Net 0.6950
   removing all routed features: -0.008
   sweep of 15 variants: all within noise

FIG 2  accuracy against spatial precision
   20 m 0.683 ... 160 m 0.712 ... 2.24 km 0.756
   recommend 160 m as the operating resolution

FIG 3  example maps: truth beside prediction, best / typical / worst / quiet day

- cross-region transfer: beats the elevation baseline in 6/6 regions
- catchment-routed rainfall is the strongest single predictor
- post-flood lidar does NOT inflate results (0.738 clean vs 0.676 dirty)

## V. Discussion and conclusion  (~300 words)
- what the label supports: event-level detection, catchment-scale extent
- what it does not: reliable per-pixel extent at 20 m
- the remedy: more looks (pilot result), or optical/hand labels
- dataset release

---

## Deliberately left out
tile-level regression, the fusion label, event filtering, focal loss, the
threshold sweep, the seven failed U-Net variants (one sentence only), the
random-split number (one sentence, as a warning).
