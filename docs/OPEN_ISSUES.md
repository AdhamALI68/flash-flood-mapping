# Open issues found while auditing the record

Ordered by how much damage each could do in review. Everything in the paper is
traceable to your data **except** the two items marked **BLOCKER**, which I used
because your README instructs it but could not verify from the files you sent.

---

## BLOCKER 1 — The six-basin within-basin numbers are not in the record

**Where it matters:** Section 5.2 of the paper, the row "By event, one basin 0.675".

`README.md` Section 6 gives all six basins:

| Basin | 1501 | 1503 | 1507 | 1606 | 1809 | 1810 |
|---|---|---|---|---|---|---|
| AUC | 0.6189 | 0.7453 | 0.6935 | 0.6810 | 0.7401 | 0.5700 |

But `_numbers.txt` under `== LEARNABLE ==` contains only **three** basins, with
**different values for the same basins**:

| Basin | 1503 | 1606 | 1809 |
|---|---|---|---|
| README (6-basin run) | 0.7453 | 0.6810 | 0.7401 |
| `_numbers.txt` (3-basin run) | 0.7568 | 0.7224 | 0.7230 |

These are two different runs. Your README says the six-basin run supersedes the
three-basin one, and I wrote the paper accordingly — but the JSON you sent only
supports the superseded version. **Find the six-basin `learnable_test.json` and
confirm, or re-run it.** A reviewer who asks for data will find this.

Note also: mean of the three-basin run is 0.7341, not the 0.722 your README
attributes to the earlier version. Worth checking what 0.722 actually was.

## BLOCKER 2 — The 0.958 pixel-split figure has no source in the record

This is the single most load-bearing number in the paper. It appears in
`README.md` Section 6 and is hard-coded into `make_audit_figures.py`:

```python
sp = [("Split by pixel,\nsame basin", 0.958, WARN, False), ...]
```

It is **not** in `_numbers.txt`, `all_runs.csv`, or `model_spread.csv`. The
`== LEARNABLE ==` block contains event-split AUCs (~0.72) and shuffled controls
(~0.50) but no 0.958.

Before submission: confirm the provenance, and confirm the pixel-split model was
trained on the same features and tile build as the event-split rows. If the
pixel split used a different feature set the +0.263 is not a clean contrast, and
+0.263 is the headline claim of Section 5.2.

---

## The framing needed changing, and this is the main editorial call I made

Your README's "Suggested framing" says to lead with the split experiment as the
headline finding, and describes it as applying "to any pixel-level geospatial
prediction task, not just floods."

**That claim is roughly twenty years old and would be the first thing a reviewer
attacked.** The literature is deep:

- Brenning (2005) — spatial partitioning for landslide models
- Roberts et al. (2017), *Ecography* — block CV for structured data, ~4000 citations
- Ploton et al. (2020), *Nature Communications* — the same argument for biomass maps
- Meyer & Pebesma (2022), *Nature Communications* — global ecological maps
- Kattenborn et al. (2022) — CNNs specifically, overestimation "up to 28%"
- Wadoux et al. (2021) — the counter-position, "Spatial cross-validation is not
  the right way to evaluate map accuracy"

Your Section 12 cites none of it. Submitting a paper whose headline claim is
"splitting matters" without citing Roberts or Ploton reads as not knowing the
field, and that impression is hard to recover from.

**What I did instead.** I built the paper on *two* ceilings, with the split as
the first and the InSAR label as the second, and gave more weight to the second.
Reasons:

1. The label work is genuinely novel. The closest prior work — Botey i Bassols
   et al. 2023, on the Salar de Atacama — concludes that CCD maps reliably
   identify affected *zones* but that "the precision in the delimitation of the
   affected area remains an open issue." That is your finding, stated
   qualitatively, on five events in one basin. You have 617 events across six
   basins and you put numbers on it. **Nobody has published the coherence-drop
   noise floor or the multilooking-versus-optical-agreement trade-off.** That is
   your paper.

2. The split contribution survives if reframed as *quantification for a specific
   task with an extra axis* (events, not just locations) rather than as a general
   discovery. I wrote it that way, and I explicitly position against Wadoux —
   which turns a vulnerability into evidence you know the debate.

3. It makes your low numbers an argument rather than an apology.

If you disagree and want the split-first version, the restructure is
straightforward — but please add the citations either way.

---

## Smaller things, in the record rather than the analysis

**Section 6 of README embeds the wrong figure.** It shows
`figures/05_scale_tradeoff.png` (the scale analysis) where it means
`figures/14_split_effect.png`. Cosmetic, but I nearly followed it.

**`deep_audit3.py` mislabels its own output.** Line:

```python
print(f"  best (our headline)    {a.max():.4f}")
```

`a.max()` is 0.7448, which is `terrain`, not your headline (0.6950, rank 5 of
21). The README catches this in prose but the script still prints it wrong. If
anyone runs your code they will read "best (our headline) 0.7448".

**The seed-noise band may not apply to the headline run.** From `all_runs.csv`,
`sw_seed1/2/3` all use `select_on=pixel`, while the headline run
`terrain+basin+rain+rain_routed+soil+landcover` uses `select_on=within`. So the
0.051 seed spread is measured under a *different checkpoint-selection criterion*
than the headline. The three seeds are identical to each other, so the spread is
real — but calling it "the seed noise on our headline" is a small overreach. I
worded it carefully in the paper; if you can run one more seed at
`select_on=within` it closes cleanly.

**Event count discrepancy.** Dataset is 617 events; tiles come "from 567
events". 50 events produced no tiles and the record does not say why. Expect to
be asked. One sentence fixes it.

**"Typical case" (fig 08) is not typical by any metric.** 19.4% flooded against
a 5.24% base rate; AUC 0.569 against a 0.695 average. It's a great figure and it
makes the argument better than anything else you have — but a reviewer checking
the numbers in the title bar will notice. I added an explicit selection note in
the caption rather than dropping it.

**The 160 m "sweet spot" is inside seed noise.** 0.6828 → 0.7115 is 0.029; your
seed spread is 0.051. I kept the recommendation but softened it to lean on the
base-rate-corrected lift curve rather than AUC. Worth knowing before someone
else points it out.

**Figure 13 panel B draws a connecting line** between feature-set points, which
implies an ordered trend your own text says doesn't exist. The line is doing
rhetorical work the data doesn't support. Consider dots only. I noted it in the
caption as a stopgap.

**Cross-basin AP is not comparable across folds.** Base rates run 0.039 to 0.151
across the six basins. AUC is fine; the AP column in that table is not
comparable fold to fold. I dropped AP from that table in the paper.

**All three examples in fig 02 are basin 1809.** You have two held-out basins.
Showing one from 1606 would be a stronger figure for near-zero effort.

---

## Two things I think are stronger than you gave them credit for

**The below-chance elevation rule (0.4803).** Your README treats this mainly as a
reason to make the prior sign learnable. It is much more than that. Below chance
means labelled-flood pixels sit *higher* than labelled-dry pixels on average.
Water does not pool on high ground. Combined with the cropland correlation and
the scattered-hillslope patches in fig 08, this is direct physical evidence that
`coh_drop` is largely tracking **hillslope disturbance and sediment
mobilisation**, not standing water — which is precisely what the sediment-
transport CCD literature uses coherence change for. I promoted it to its own
subsection (5.3.4) and it's now one of the five pillars of the label argument.

**The speckle-filter test's methodology.** You note in passing that the idea
"would have looked successful if judged by model score." That's a sharp
observation about circular validation and it generalises well beyond this
experiment. I kept it and gave it a sentence of its own.

---

## Venue

Your README estimates IGARSS at 75–80%. I'd push higher and to a journal.

With the two-ceilings framing this is a full paper, not a 4-page abstract. The
label characterisation alone — noise floor, looks trade-off, 20× over-marking,
speckle test, cropland confound, all on 617 events — is more material than
IGARSS has room for. Candidates:

- **NHESS** (EGU, open review) — good fit for the hazard framing, and open
  review suits a paper this explicit about its own negative results
- **IEEE JSTARS** — good fit for the InSAR-methods half
- **Remote Sensing** (MDPI) — fastest, and both Botey i Bassols papers are there
- ***Remote Sensing of Environment*** — a stretch, but the label work is the kind
  of thing RSE publishes if the writing is tight

IGARSS remains a sensible parallel submission for the split experiment alone.

---

## Also, from your README's own to-do list

> **Back this project up.** `trial_2` is not a git repository. 617 events, all
> feature stacks, 55 trained models and every result exist on one disk with no copy.

This is still the highest-value hour you can spend, and it predates anything in
this document.
