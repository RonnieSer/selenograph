# Dataset and saved work

This repository contains source code only. Observation rasters, annotations,
trained models, and generated outputs are not bundled. Prepare observations from
the public source archives, then create and review your own paintings in the app.
See [data sources](../data/sources.md) for attribution and dataset terms.

## Prepare observations

After [installing the dependencies](../README.md), download regional ZIP archives
from the linked source into `data/raw_data/` without extracting them. Run from the
project root on macOS or Linux:

```sh
python data/prepare_dataset.py
```

The script aligns companion layers to the native shape-from-shading (SfS) DEM grid
and creates 1024-pixel tiles. Missing observations remain missing, and edge tiles
retain their actual extent. Source readmes, hashes, grids, and quality information
are preserved. Preparation creates no labels or trained model.

Completed regions are verified and reused. Interrupted work can resume with the
same command. Use `--region` to select a region; repeat it for a subset. The number
of maps depends on the archives and regions you prepare. To verify prepared data
against the original ZIPs:

```sh
python data/prepare_dataset.py --verify
```

Preparation uses Unix file locking. The older `data/raw_data/essentials/` archive
location is also supported.

## Local layout

```text
data/
  raw_data/                            downloaded regional ZIP archives
  processed_data/
    manifest.json
    <site>/
      working_dems.json
      metadata/                        source hashes, grids, quality, and readme
      tiles/<tile-id>/
        elevation/<tile-id>.tif
        imagery/<tile-id>_nac.tif
        quality/<tile-id>_<layer>.tif
```

The DEM's filename stem is its stable map ID. Tile row and column numbers are
zero-based, and catalog paths are relative. Other processed datasets can be
registered with a `working_dems.json` under `data/processed_data/<site>/`.
Keep observation rasters unchanged; paintings and model artifacts are stored
separately.

## Paintings, models, and exports

Changes in **Paint** and **Gallery**, including verification, persist as drafts.
**Save labels + map** commits the painting and metadata and writes georeferenced
label and prediction exports. Without a model, it exports labels and removes any
previous prediction at that path. **Restore saved** replaces the draft with the
saved painting. Browsing does not create work directories or train a model.

Each painting and export directory mirrors the DEM's parent path relative to
`data/processed_data/`. For example:

```text
data/processed_data/site/tiles/id/elevation/id.tif
output/painting_drafts/site/tiles/id/elevation/id_painting.npy
output/paintings/site/tiles/id/elevation/id_painting.npy
output/maps/site/tiles/id/elevation/id_labels.tif
output/maps/site/tiles/id/elevation/id_map.tif
```

Paintings also use `<id>_confidence.npy`, `<id>_accepted.npy`, and `<id>_meta.json`
for confidence, accepted predictions, and metadata. Move the corresponding output
subtrees when relocating a site, and back up `output/` and `models/` separately.

**Update model** reads current verified paintings, including drafts, and writes
`models/rf.joblib` for the default Random Forest backend. It does not commit
paintings or export maps. **Progress → Evaluation** writes a JSON report under
`output/evaluations/` without replacing the active model.

Verification controls training membership; it does not change source attribution
or grant sharing rights. See [training and evaluation](TRAINING.md) for label,
feature, and research limitations.
