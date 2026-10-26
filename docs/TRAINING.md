# Painting, training, and evaluation

Prepare observations using the [dataset guide](DATASET.md). A fresh clone includes
no observations, paintings, or trained model. In **Paint**, choose a **Section**
and tile, define units, paint examples, and verify maps after reviewing them.
**Gallery** browses paintings; **Progress** shows training and evaluation results.

## Labels and review

The three classes are smooth highlands, rough highlands, and `shadowed_floor`.
The last class includes some crater interiors in the historical class definition;
it does not establish permanent shadow or ice. Define the units and any mapping
from external labels before labeling. Unpainted areas remain unknown.

Paintings use 120 × 120 cells. On a full 1024-pixel tile at 5 m spacing, each cell
spans about 42.7 m; edge tiles have different cell sizes. Painted boundaries do not
provide native 5 m polygon boundaries or subpixel measurements. Missing terrain
and display padding are not paintable, and display padding does not change stored
coordinates or training samples.

**Verified is a user review flag, not independent scientific certification.**
Preserve the actual review history and source information in label metadata.
Accepted model suggestions are predictions, not independent human reference labels.

## Save and train

- Painting, undo, confidence, acceptance, and verification changes autosave as
  drafts. **Save labels + map** commits paintings and metadata and exports human
  labels separately from model predictions.
- **Update model** rebuilds samples from all currently verified paintings,
  including draft edits on other maps. Removed, cleared, and unverified maps are
  excluded. Fitting requires at least two labeled classes. The action saves only
  the model and refreshes the current prediction.
- **Sure**, **Mostly**, and **Unsure** paintings have different training weights.
  Accepted predictions receive reduced weight. Painting or changing a display
  background alone does not train the model.
- Effective paintings resolve in order: draft, saved painting, then annotations
  supplied by a registered dataset. Confidence and acceptance follow the same
  painting source. Missing confidence defaults to Sure; missing acceptance defaults
  to false. These defaults do not establish historical certainty or independence.
- **Restore saved** replaces draft labels and flags with the saved version, or
  dataset annotations when available. Undo recovers the previous draft. Restoring
  does not export maps or retrain the model.

The [dataset guide](DATASET.md) describes local output paths. Keep source rasters
unchanged and back up paintings and models separately.

## Terrain features

Random Forest is the default backend. The optional LightGBM backend requires the
separately installed `lightgbm` package. Both use terrain recipe version 6 with six
ordered features: slope, roughness, sky view, local relative height, curvature,
and local sky view.

Features normally come from the selected DEM. A catalog can declare
`feature_sources` referencing whole input DEMs. Adjoining aligned sources are joined
before feature calculation to avoid artificial tile edges, then features are
transferred to the selected map's grid with nearest-neighbor sampling. Missing
feature coverage stays missing. Limited surrounding context can affect boundaries
and sky-view estimates, so record the context policy in research comparisons.

**NAC imagery is a viewing layer.** Radar, separate LOLA products, quality channels,
scene-relative height, and hillshade are also excluded from the default training
recipe. Quality layers are preserved for review but do not automatically filter
or weight training samples. Changing a background does not change the model inputs.

Changes to features or classes require a versioned recipe and explicit retraining.
Incompatible model schemas are rejected; an older terrain recipe may require
**Update model** before predictions are available. Browsing does not retrain or
replace saved models.

## Evaluate selected maps

In **Progress → Evaluation**, select one or more painted maps. Evaluation supports
the RF and GBM pixel backends. Verification controls training membership; painted
maps can be evaluated without being verified.

1. **Read current labels.** Drafts take precedence over saved or dataset labels.
   Scoring uses Sure manual cells and excludes accepted predictions, unknown
   labels, and cells containing no pixels. At least 90% of a cell must have terrain,
   and at least 90% of that terrain must have finite six-feature and known-label
   coverage.
2. **Leave out one map.** For each selected map, use the active model's remembered
   samples from currently verified maps and exclude only that map. Other maps in
   the same section or overlapping footprint remain eligible for training. An
   unverified map is scored using the full verified training set.
3. **Fit in memory.** Each evaluation fits its own estimator without replacing the
   active model. Use **Update model** first to refresh remembered samples after
   painting edits. A remaining single-class set predicts that class everywhere;
   an empty training set cannot be scored. Unavailable selections are reported
   explicitly while other selected maps can still score.
4. **Score each cell once.** Pixel predictions are combined by majority vote within
   eligible painting cells; ties choose the lowest class code. Reports include
   per-map and pooled scores, an equal-map mean, confusion matrices, coverage, and
   a majority-training-class baseline.

**This evaluation leaves out maps individually and does not provide geographic
holdout.** Neighbouring and overlapping maps may share terrain context. Scores
measure agreement with paintings and do not establish generalization to untouched
locations or independently reviewed truth.

Each completed run writes one
`output/evaluations/evaluation-<run_id>.json` containing results, training selections,
coverage, skipped-map reasons, software versions, and provenance. **Progress**
displays the latest completed report and offers a JSON download. Evaluation does
not change paintings, verification, or the active model.

## Research limitations

Use geographically separated experiments when assessing new predictors. Compare
terrain alone with additional NAC or radar features using the same splits and a
versioned recipe. Illumination, mosaic seams, and acquisition coverage can become
shortcuts unrelated to geology. Review quality products and define any screening
or weighting policy explicitly. Reference labels are targets, not predictors.

Finite SfS elevation does not establish image support. NoData and zero support
flags are not geological classes. Missing support information is not evidence of
measured absence of support and is not an evaluation eligibility filter. Consult
source uncertainty documentation before quantitative use. SfS uses NAC/LOLA
constraints, so these layers are not independent accuracy references. Resampling
does not increase native spatial detail.

Before reporting results:

- Group overlapping tiles, products, and revisions of the same physical site.
  Folder separation alone does not prevent information leakage.
- Freeze reviewed label versions and record source, reviewer, class schema, grid,
  quality policy, and hashes. Distinguish user verification from independent review.
- Keep model-selection regions separate from a final evaluation region. Score
  initial predictions before incorporating corrections from that region.
- Report per-site and per-class metrics, confusion matrices, the majority baseline,
  scored coverage, and skipped maps. Record the model recipe, sampling policy,
  parameters, software versions, and the evaluation procedure used.
- Measure mapping effort alongside predictive quality. Interactive model updates
  alone do not constitute a reproducible frozen experiment.

Source datasets, annotations, trained models, and generated outputs retain their
original terms. The code license does not grant rights to redistribute them; see
[data sources](../data/sources.md).
