# Data samples

Small samples of the data the flood model learned from, so you can see its shape
without the full data. The full data is about 318 GB and stays on the lab machine.

Start with `one_tile_all_channels.png`: it shows one tile, its label and all 30 inputs.

## The task

Predict, for each 20 m pixel in a desert basin in the southwestern US, whether a flash
flood will affect it. The model is a U-Net, so it works on image tiles.

## Shape of the training data

| Part | Shape | Type | Meaning |
|---|---|---|---|
| Inputs `X` | `(N, 30, 112, 112)` | float32 | N tiles, 30 features, 112 x 112 pixels |
| Label `y` | `(N, 112, 112)` | float32 | Continuous coherence drop per pixel |
| Mask | `(N, 112, 112)` | bool | True means the pixel is valid. Use only these pixels |

One tile covers 2.24 km x 2.24 km, at 20 m per pixel.

## Size of the full data

| Split | Basin groups (HUC4) | Tiles | Valid pixels | Flood share |
|---|---|---|---|---|
| Train | 1501, 1503, 1810 | 4,373 | 54,237,723 | 12.65% |
| Validation | 1507 | 1,193 | 14,869,267 | 10.90% |
| Test | 1606, 1809 | 1,918 | 23,531,568 | 5.24% |
| **All** | 6 groups | **7,484** | **92,638,558** | |

The 7,484 tiles are 5,887 flood-day tiles from 567 events, plus 1,597 quiet-control
tiles from 159 dates. A quiet control is the same place on a date with no flood report.
The dataset lists 617 flood events, but only 567 of them produced usable tiles.

## The label

The label is the coherence drop from Sentinel-1 radar:

    coh_drop = coherence(baseline pair, before the flood) - coherence(pair that spans the flood)

- It is stored as a continuous value. It must lie between -1 and +1, and 98% of values sampled from the valid pixels fall between -0.39 and +0.55.
- **A pixel counts as flood when `coh_drop > 0.15`.** The cut is applied at training and
  scoring time. It is not stored in the tiles.
- Pixels where the mask is False have no usable label, and many are NaN.

There is also an experimental **photo label**, made from before and after Sentinel-2
images and checked by eye. Its values are `+1` (flood) and `-1` (not flood), so the same
0.15 cut works on it. Only 3 of the 63 floods checked by eye were marked right, giving 40
tiles. The radar label remains the main one. See `photo_labels_sample.npz`.

## Files

| File | What it is |
|---|---|
| `one_tile_all_channels.png` | Picture of one tile: label, mask and all 30 inputs |
| `feature_dictionary.csv` | The 30 features in model order: unit, meaning and statistics over all valid pixels |
| `pixels_sample.csv` | 19,790 random valid pixels as a table, 39 columns (details below) |
| `tiles_sample.npz` | 12 full tiles as arrays: 6 flood, 3 dry on a flood day, 3 quiet control |
| `tiles_sample_index.csv` | One row for each of those 12 tiles |
| `label_grid_coh_drop.csv` | One tile's label as a plain 112 x 112 grid |
| `label_grid_flood_binary.csv` | The same label after the 0.15 cut (1 flood, 0 dry, empty invalid) |
| `photo_labels_sample.npz` | 3 tiles of the photo label |
| `tile_index_full.csv` | Every one of the 7,484 tiles: event, basin, split, flood share |
| `events_617.csv` | The 617 flood events: date, place, radar scenes, basin |
| `dataset_summary.json` | The counts above, as numbers |
| `make_samples.py`, `plot_tile.py` | Scripts that made these files from the full data |

The columns of `pixels_sample.csv` are `tile_id`, `event`, `is_quiet_control`, `huc4`,
`split`, `px_row`, `px_col`, the 30 features, `coh_drop` and `flood_label`. Pixels were
picked at random, so the flood share is the natural one (10.45%).

## Load it

```python
import numpy as np, pandas as pd

z = np.load("tiles_sample.npz")
X, y, mask = z["X"], z["y"], z["mask"]   # (12, 30, 112, 112), (12, 112, 112), (12, 112, 112)
names = list(z["channels"])              # the 30 feature names, in order
flood = (y > 0.15) & mask                # the training target

pixels = pd.read_csv("pixels_sample.csv")   # (19790, 39)
```

## How the model sees the features

All values in these files are **raw**. Before training, `phase6/train.py` does two steps:

1. `log1p` on six wide-range features: `basin_facc`, `basin_spi`, `catchment_area_km2`,
   `rain_1h_upstream_vol_m3`, `rain_24h_upstream_vol_m3`, `runoff_1h_upstream_vol_m3`.
2. A z-score using the mean and standard deviation of the training pixels only. The values
   are in `feature_dictionary.csv` as `model_mean_after_log` and `model_std_after_log`.

## Things to know before analysing

1. **Split by basin group, never at random.** Events overlap on the ground, so a random
   split puts the same land in train and test and the score looks much better than it is.
2. **Pixels are not independent.** Neighbours in a tile are strongly related. Use
   `pixels_sample.csv` to understand distributions, not to estimate model performance.
3. **The label is noisy.** A strict 3-sigma radar test fires on 0.00% of pixels at the
   5 looks used here (`RECORD/README.md`, looks pilot). Quiet tiles, with no flood reported,
   still have about 2.8% of pixels above the 0.15 cut, against 12.4% on flood-day tiles.
4. **Flood share differs by split:** 12.65% in train, 5.24% in test.
5. **Effectively 29 features.** `rain_cover_frac` is exactly 1.0 on every valid pixel.
6. **Very sparse features.** `lc_water` is almost always 0, and the median of both
   `runoff_1h_upstream_*` features is 0.
7. **`lc_class` is a category,** but the model receives it as a number.
8. **`soil_ksat` is a log value,** `log10(cm/hr)`, so negative values are normal.
9. **`rain_1h_peak` is not a peak.** It is the 1 hour total at the hour the flood report starts.
10. **Rain is measured, not forecast.** `rain_1h_peak`, `rain_24h` and the routed rain
    features use radar rain up to the report hour.
11. **Terrain date.** `dem_clean` is False for about 80% of tiles, because the lidar was
    surveyed after the flood. The project tested this effect (see `RECORD/README.md`).

More detail: `RECORD/README.md`, sections 1 to 3.
