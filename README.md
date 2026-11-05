# Flash flood mapping in arid basins

Predicting which 20 m pixels flood in a flash flood, in desert basins of the southwestern
United States. The inputs are terrain, basin routing, radar rain, soil and land cover. The
label comes from Sentinel-1 InSAR coherence change.

## Dataset

- 617 flood events from 2015 to 2025, in 6 basin groups (HUC4)
- 7,484 tiles of 112 x 112 pixels at 20 m (2.24 km across), with 30 input features
- 1,597 of those tiles are quiet controls: same place, no flood reported
- Region split: train on 1501, 1503, 1810; validate on 1507; test on 1606, 1809

**New here? Open [samples/README.md](samples/README.md).** It explains the data, the label
and the 30 features, with small files you can load straight away.

## Results so far

Scored per 20 m pixel on the held-out test basins (23.5 M pixels, 5.24% flood).

| Model | AUC | Recall | Precision |
|---|---|---|---|
| U-Net, 30 features (headline) | 0.695 | 0.397 | 0.101 |
| XGBoost, local features | 0.687 | 0.660 | 0.087 |
| Elevation rule | 0.480 | 0.601 | 0.053 |

Read these with care:

- 0.695 is one of 21 comparable models. The honest summary is 0.68 plus or minus 0.03.
- Changing only the random seed moves AUC by up to 0.05.
- Terrain alone reaches 0.745 on this metric, but it cannot tell flood days from quiet days.
- The main limit looks like the label, not the model. Photo-based labels were tried to fix
  this, but only 3 of 63 floods checked by eye were marked right, so that test set is small.

The full story, with every experiment, is in [RECORD/README.md](RECORD/README.md).

## Layout

| Folder | What is in it |
|---|---|
| `samples/` | Small data samples and the data guide. Start here |
| `RECORD/` | Complete record of experiments, numbers and figures |
| `docs/` | Findings, plan, next steps, open issues, last status page |
| `phase1/` to `phase3/` | Events, basin and radar screening, InSAR processing |
| `phase4/` | Label building and validation |
| `phase5/` | Features and tiles |
| `phase6/` | Model (`model.py`), training (`train.py`), evaluation |
| `phase7/` | Photo flood-trace labels |
| `phase8/` | Cross-region test |
| `final/` | Scripts that produce the headline numbers |
| `runs/` | Shell scripts for each training run |
| `gate_b/` | Early radar check, plus `align.py`, which other scripts use |
| `tools/` | Audits, checks and the status page |
| `paper/` | Paper draft and figures |
| `handoff/` | Result pictures for each run |
| `notebooks/` | Team Colab notebook |
| `archive/` | Old job runners, kept for reference and not expected to run |
| `data/` | Not in git, apart from small tables and results |

## Setup

Python 3.12, with a GPU for training.

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Run scripts from the repo root:

```bash
./.venv/bin/python -u phase6/train.py --tiles data/tiles_v4 --neg data/tiles_neg_v4 \
    --groups terrain,basin,rain,rain_routed,soil,landcover --tag my_run
./.venv/bin/python final/phase7_final.py     # needs the data and the headline model
```

Some download scripts need accounts: NASA Earthdata through `~/.netrc`, and Planet through
the `PL_API_KEY` environment variable. Never commit these.

## Data

Full data is about 318 GB and is only on the lab machine, in `data/`. Training needs only
`data/tiles_v4` (9.2 GB) and `data/tiles_neg_v4` (2.1 GB). Ask Adham for access.

## Rules for working here

1. Split by region, never at random. A random split scores land the model has already seen.
2. Fit scaling on train only, and pick models and cutoffs on validation only.
3. Score on the test basins once. Do not filter the test set using the label.
4. Keep labels continuous in the tiles and apply the cut when scoring.
5. Never use terrain, basin or land-cover layers to build a label. They are model inputs,
   so this would leak.
6. Report a measured result even when it is bad. Do not commit data, models or logs.
