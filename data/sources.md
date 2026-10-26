# Data setup

This repository contains research code. Download and prepare observation data
locally; annotations, trained models, and generated outputs are not bundled.

Terrain and imagery come from Bertone et al., *Enhanced Topography Models for
selected Lunar South Pole regions with Shape-from-Shading*, archived on
[Zenodo](https://doi.org/10.5281/zenodo.17954508). Dataset preparation preserves
original source readmes under each site's `metadata/` directory.

The repository's MIT license covers code and documentation only. Source datasets,
annotations, trained models, and generated outputs retain their existing terms.
Preserve source attribution and any restrictions recorded in annotation metadata.

Download the regional ZIPs you want to use and put them
directly in `data/raw_data/`, without extracting them. Then run from the project root:

```sh
python data/prepare_dataset.py
```

The script creates `data/processed_data/` and organizes the rasters into site
folders and tiles. Existing completed regions are checked and reused. Preparation
uses Unix file locking; run it on macOS, Linux, or Windows Subsystem for Linux.

Paint labels in the app, mark reviewed maps **Verified**, and select **Update
model** to train your own assistant. Local data, paintings, and models are ignored
by Git.
