"""Train and persist the terrain-mapping assistant from verified paintings.

Approved predictions receive core.APPROVED weight. Training rebuilds samples
from current paintings. Model artifacts are trusted local files.
"""
import datetime
import functools
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio

from app import core

from app.paths import MODEL_DIR
PER_STRATUM = 2000     # Pixels sampled per painted unit per map.
MAX_TRAIN_ROWS = 120000


def schema() -> dict[str, Any]:
    """Return the versioned backend, class, feature, and terrain-recipe schema."""
    recipe = dict(version=core.FEATURE_VERSION, name="terrain_baseline", window="gaussian",
                  rough_m=core.ROUGH_M, rel_m=core.REL_M, local_m=core.LOCAL_M,
                  curv_m=core.CURV_M, sky_local_m=core.SKY_LOCAL_M)
    return dict(version=1, backend=core.MODEL, classes=dict(core.CODES),
                features=list(core.BASE_FEATS), recipe=recipe)


def features(X, names):
    """Select required terrain features in the versioned baseline order.

    Reject duplicate channels or missing required features.
    """
    required = schema()["features"]
    if X.ndim != 3 or X.shape[0] != len(names) or len(set(names)) != len(names):
        raise ValueError("Feature stack must have one uniquely named channel per layer.")
    missing = [n for n in required if n not in names]
    if missing:
        raise ValueError("The terrain baseline requires all six DEM features; missing: "
                         + ", ".join(missing))
    indices = [names.index(name) for name in required]
    return X[indices]


def painting_hash(cells, confidence=None, accepted=None):
    """Hash painting cells and their effective confidence and acceptance flags."""
    h = hashlib.sha256(cells.astype(np.uint8).tobytes())
    if confidence is not None and ((confidence != core.SURE) & (cells > 0)).any():
        h.update(np.where(cells > 0, confidence, 0).astype(np.uint8).tobytes())
    if accepted is not None and (accepted & (cells > 0)).any():
        h.update(b"approved" + (accepted & (cells > 0)).tobytes())
    return h.hexdigest()


def artifact_stamp():
    """Return file identity and revision metadata for the saved model."""
    path = Path(MODEL_DIR).resolve() / f"{core.MODEL}.joblib"
    try:
        stat = path.stat()
        return str(path), stat.st_ino, stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return str(path), None


def load() -> dict[str, Any]:
    """Load a compatible assistant artifact or return an empty training bundle."""
    path = Path(MODEL_DIR) / f"{core.MODEL}.joblib"
    if not path.exists():
        return dict(schema=schema(), samples={}, model=None, revision=None)
    bundle = joblib.load(path)
    if bundle.get("schema") != schema():
        previous = schema()
        previous["recipe"]["version"] = 5
        if core.FEATURE_VERSION == 6 and bundle.get("schema") == previous:
            # Preserve the old artifact while requiring samples from the current recipe.
            return dict(schema=schema(), samples={}, model=None, revision=None,
                        needs_retrain=True)
        raise ValueError("Saved assistant uses different features or units. Move its "
                         f"artifact out of {MODEL_DIR} before training a new assistant.")
    return bundle


@core.catalog_snapshot()
def verified_samples(samples):
    """Retain samples whose current effective metadata explicitly verifies them."""
    return {name: sample for name, sample in samples.items()
            if core.meta_get(name).get("verified") is True}


@core.catalog_snapshot()
def verified_maps():
    """Return installed maps explicitly verified for training."""
    maps = {}
    for path in core.dem_files():
        name = core.dem_name(path)
        if core.meta_get(name).get("verified") is True:
            maps[name] = path
    return maps


def _training_sources(previous, maps):
    """Collect current provenance for verified training maps."""
    return {name: _provenance(name, previous.get(name, {}), core.site_id(path), None, path)
            for name, path in maps.items()}


@functools.lru_cache(maxsize=2048)
def _source_hash(path, size, modified, changed):
    """Return a content hash cached by local file revision metadata."""
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _training_input(path, cells, confidence, accepted):
    """Record painting, source content, and feature dependency identities."""
    order, dependencies = core._feature_plan(path)

    def relative(source):
        return Path(os.path.relpath(source, core.DEM_DIR)).as_posix()

    revisions = []
    for source in order:
        stat = os.stat(source)
        digest = _source_hash(source, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        revisions.append((relative(source), digest))
    return dict(painting_hash=painting_hash(cells, confidence, accepted),
                source_revision=revisions,
                feature_graph={relative(p): [relative(child) for child in children]
                               for p, children in dependencies.items()})


def current_training_input(name, path):
    """Return current training identities, or None for an unpainted map."""
    cells = core.load_painting(name)
    if not cells.any():
        return None
    return _training_input(path, cells, core.load_confidence(name), core.load_accepted(name))


@core.catalog_snapshot()
def training_changed(bundle, maps=None, current_input=current_training_input):
    """Check current training inputs without extracting features or fitting models.

    The current_input callback may cache current_training_input results.
    """
    maps = verified_maps() if maps is None else maps
    samples = bundle.get("samples", {})
    current = {}
    for name, path in maps.items():
        state = current_input(name, path)
        if state is not None:
            current[name] = state
    if bundle.get("schema") != schema():
        return bool(current or samples)
    if "training_inputs" in bundle:
        # Track paintings without usable samples to avoid repeated retraining prompts.
        return current != bundle["training_inputs"]
    # Older artifacts support comparison through their stored provenance.
    if current.keys() != samples.keys():
        return True
    for name, state in current.items():
        for key in ("painting_hash", "source_revision"):
            if state[key] != samples[name].get(key):
                return True
    return False


@core.catalog_snapshot()
def retrain(progress=None):
    """Rebuild training samples from current verified paintings and save the model.

    Read current drafts without modifying paintings. Removed, cleared, and
    unverified maps are excluded from rebuilt training memory.
    """
    maps = verified_maps()
    previous = load().get("samples", {})
    sources = _training_sources(previous, maps)
    samples, inputs = {}, {}
    with rasterio.Env(GDAL_PAM_ENABLED="NO", PROJ_NETWORK="OFF"):
        for index, (name, provenance) in enumerate(sources.items(), 1):
            if progress:
                progress(f"Reading verified map {index} of {len(sources)}: {name}")
            cells = core.load_painting(name)
            if cells.shape != (core.GRID, core.GRID) or not np.isin(cells, [0, *core.CODES]).all():
                raise ValueError(f"Painting contains an invalid grid or unit code: {name}")
            if not cells.any():
                continue
            confidence = core.load_confidence(name)
            accepted = core.load_accepted(name)
            path = maps[name]
            inputs[name] = _training_input(path, cells, confidence, accepted)
            X, names, *_ = core.build(path)
            terrain = features(X, names)
            labels, weights, strata = training_pixels(
                name, terrain.shape[1:], cells, confidence, path, accepted)
            Xtr, ytr, wtr = sample(terrain, labels, weights, strata)
            if Xtr is None:
                continue
            provenance["source_revision"] = inputs[name]["source_revision"]
            samples[name] = dict(provenance, X=Xtr, y=ytr, w=wtr,
                                 painting_hash=painting_hash(cells, confidence, accepted))
    if progress:
        progress(f"Training on {len(samples)} verified maps")
    result = fit(verified_samples(samples))
    result["training_inputs"] = inputs
    save(result)
    return result


def pending(bundle, name, cells, confidence=None, accepted=None):
    """Return whether a painting differs from its remembered training sample."""
    previous = bundle["samples"].get(name)
    if not previous:
        return bool(cells.any())
    return previous["painting_hash"] != painting_hash(cells, confidence, accepted)


def training_pixels(name, shape, cells, confidence=None, dem_path=None, accepted=None):
    """Expand painting cells into pixel labels, weights, and sampling strata.

    Unknown labels use 255. Confidence controls manual-label weights; accepted
    predictions use core.APPROVED. Only supplied paintings provide labels.
    The name and dem_path parameters are retained for caller compatibility.
    """
    if confidence is None:
        confidence = np.where(cells > 0, core.SURE, 0)
    confidence_weights = np.zeros(max(core.CONFIDENCE) + 1, np.float32)
    for level, (_, weight) in core.CONFIDENCE.items():
        confidence_weights[level] = weight
    cell_weights = confidence_weights[np.where(cells > 0, confidence, 0)]
    if accepted is not None:
        cell_weights = np.where(accepted & (cells > 0), core.APPROVED, cell_weights)
    labels = core.cells_to_labels(cells, shape)
    weights = core.per_pixel(cell_weights.astype(np.float32), shape)
    return labels, weights, labels.astype(np.int32)


def sample(X, lab, w, strata):
    """Sample up to PER_STRATUM labelled pixels per stratum with fixed randomness."""
    rng = np.random.default_rng(0)
    valid = (lab != 255) & (w > 0) & np.isfinite(X[0])
    picks = []
    for stratum in np.unique(strata[valid]):
        pool = np.flatnonzero(valid & (strata == stratum))
        picks.append(rng.choice(pool, min(PER_STRATUM, pool.size), replace=False))
    if not picks:
        return None, None, None
    indices = np.concatenate(picks)
    rows = X.reshape(X.shape[0], -1)[:, indices].T
    return rows, lab.ravel()[indices], w.ravel()[indices]


def predict(bundle, X, names):
    """Predict terrain units, or return None when the bundle has no fitted model."""
    if bundle["model"] is None:
        return None
    return core.backend().predict(bundle["model"], features(X, names))


def confidence(bundle, X, names):
    """Return predicted-unit probabilities, with NaN where terrain is missing."""
    if bundle["model"] is None:
        return None
    return core.backend().confidence(bundle["model"], features(X, names))


def _dem_footprint(path) -> dict[str, Any] | None:
    """Read a DEM grid and bounds without loading raster pixels."""
    if not path or not Path(path).is_file():
        return None
    with rasterio.open(path) as src:
        return dict(shape=list(src.shape), transform=list(src.transform)[:6],
                    crs_wkt=src.crs.to_wkt() if src.crs else None, bounds=list(src.bounds))


def _provenance(name, previous, site_id, source_revision, dem_path):
    """Refresh sample location and source metadata while preserving provenance."""
    # Preserve identity fields without retaining label counts or weight policy.
    fields = ("source_kind", "geographic_group", "reference_group", "public_region",
              "public_tiles", "public_tile_ids", "site_id", "map_id", "tile_id",
              "snapshot", "footprint", "grid", "feature_sources", "restricted", "provenance")
    provenance = {key: previous[key] for key in fields if key in previous}
    record = core.map_record(name) or {}
    metadata = core.meta_get(name)
    provenance.update(site_id=site_id, source_revision=source_revision,
                      geographic_group=core.map_site(name) or provenance.get("geographic_group") or site_id,
                      restricted=bool(metadata.get("restricted", provenance.get("restricted", False))))
    provenance.setdefault("source_kind", record.get("source_kind", "processed_map"))
    provenance.setdefault("tile_id", name)
    provenance.setdefault("map_id", record.get("map_id", name))
    if "provenance" in metadata:
        provenance["provenance"] = metadata["provenance"]
    if "feature_sources" in record:
        provenance["feature_sources"] = [core.dem_name(p) for p in record["feature_sources"]]
    footprint = _dem_footprint(dem_path or core.dem_path(name))
    if footprint:
        provenance["footprint"] = footprint
        if "grid" in provenance:
            provenance["grid"] = dict(footprint)
    return provenance


def fit(samples) -> dict[str, Any]:
    """Fit pooled samples and return an in-memory assistant bundle.

    Callers are responsible for verification and geographic exclusions.
    """
    if samples:
        pooled_X = np.concatenate([s["X"] for s in samples.values()])
        pooled_y = np.concatenate([s["y"] for s in samples.values()])
        pooled_w = np.concatenate([s.get("w", np.ones(len(s["y"]), np.float32))
                                   for s in samples.values()])
        if len(pooled_y) > MAX_TRAIN_ROWS:
            rng = np.random.default_rng(0)
            classes = np.unique(pooled_y)
            per_class = MAX_TRAIN_ROWS // len(classes)
            selections = []
            for code in classes:
                indices = np.flatnonzero(pooled_y == code)
                selections.append(rng.choice(indices, min(per_class, len(indices)), replace=False))
            keep = np.concatenate(selections)
            pooled_X, pooled_y, pooled_w = pooled_X[keep], pooled_y[keep], pooled_w[keep]
        model = core.backend().fit_rows(pooled_X, pooled_y, pooled_w)
    else:
        model = None
    return dict(schema=schema(), samples=samples, model=model, revision=uuid.uuid4().hex,
                updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())


def save(result):
    """Save an assistant atomically, preserving the previous artifact on failure."""
    directory = Path(MODEL_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    staged = directory / f".{core.MODEL}-{result['revision']}.tmp"
    try:
        joblib.dump(result, staged, compress=3)
        os.replace(staged, directory / f"{core.MODEL}.joblib")
    finally:
        staged.unlink(missing_ok=True)
