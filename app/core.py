"""Terrain features, raster utilities, and annotation storage.

Prepared DEMs are registered in working_dems.json files under DEM_DIR, with
manifest-relative paths and basename-stem map IDs. Flat legacy DEMs and their
companion rasters are also supported. NAC, PSR, and CPR are display layers only.

Sections default to the DEM's top-level folder; groups and sites describe
geography. Bundled annotations are read-only; edits target user output folders.
Declared feature_sources provide aligned terrain for BASE_FEATS, while display
geometry remains local.
"""
import glob
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi

from app.paths import DEM_DIR, DRAFT_DIR, LEGACY_DRAFT_DIR, LEGACY_MIRRORED_DIR, LEGACY_OUT_DIR, MAP_DIR, OUT_DIR

SUFFIXES = ("_psr", "_cpr", "_nac", "_quality", "_mask")  # Companion raster suffixes.
ANNOTATION_FILES = ("painting", "confidence", "accepted", "labels")

CODES = {1: "smooth_highlands", 2: "rough_highlands", 3: "shadowed_floor"}
CMAP = {0: (150, 150, 150, 255), 1: (70, 110, 200, 255),
        2: (190, 215, 60, 255), 3: (170, 50, 60, 255)}

GRID = 120           # Painting cells per side.
# Manual confidence levels and training weights.
SURE, MOSTLY, UNSURE = 1, 2, 3
CONFIDENCE = {SURE: ("sure", 1.0),
              MOSTLY: ("mostly", 0.5),
              UNSURE: ("unsure", 0.2)}
MAX_SIDE = 4000      # Maximum raster side length after decimation.
# Gaussian windows use the variance of a square window with these widths.
ROUGH_M = 135        # Roughness window, in metres.
REL_M = 4000         # Relative-elevation window, in metres.
LOCAL_M = 150        # Local relative-height window, in metres.
CURV_M = 60          # Curvature smoothing window, in metres.
SKY_LOCAL_M = 300    # Local sky-view radius, in metres.
FEATURE_VERSION = 6  # Shared source terrain for neighbourhood features.

APPROVED = 0.5       # Training weight for accepted predictions.

# Scene-relative elevation and sensor layers are excluded from training features.
LAYERS = ["slope", "rough", "rel", "zscene", "svf", "rel_local", "curv", "svf_local"]
BASE_FEATS = ["slope", "rough", "svf", "rel_local", "curv", "svf_local"]

# Select a model_<name>.py backend through SELENOGRAPH_MODEL.
MODEL = os.environ.get("SELENOGRAPH_MODEL", "rf")

# Backends expose NAME, fit_rows, predict, and confidence. Features have
# shape (n_features, height, width); painting cells have shape (GRID, GRID).
# Zero marks unknown cells. Predictions use 255 for missing terrain;
# confidence arrays use NaN.


def model_files():
    """Return the names of available model backend modules."""
    here = os.path.dirname(os.path.abspath(__file__))
    return sorted(os.path.basename(p)[len("model_"):-len(".py")]
                  for p in glob.glob(f"{here}/model_*.py"))


def backend(name=None):
    """Import the selected backend lazily to avoid loading unused dependencies."""
    import importlib
    return importlib.import_module(f"app.model_{name or MODEL}")


def read(path, max_side=None):
    """Read band 1 as float32 with nodata as NaN, optionally decimated."""
    with rasterio.open(path) as source:
        height, width = source.height, source.width
        if max_side and max(height, width) > max_side:
            scale = max(height, width) / max_side
            height = max(1, int(round(height / scale)))
            width = max(1, int(round(width / scale)))
            values = source.read(1, out_shape=(height, width), resampling=Resampling.average)
            transform = source.transform * rasterio.Affine.scale(
                source.width / width, source.height / height)
        else:
            values = source.read(1)
            transform = source.transform
        values = values.astype(np.float32)
        if source.nodata is not None:
            values[values == source.nodata] = np.nan
        # Float64 PDS missing constants may appear unmasked in float32 bands.
        values[np.abs(values) > 1e30] = np.nan
        return values, transform, source.crs, abs(transform.a)


def warp(arr, tr, crs, ref, resampling=Resampling.bilinear):
    """Reproject a display raster onto the reference DEM grid.

    If either CRS is missing, assume both rasters share a coordinate system.
    """
    dst_crs = ref["crs"]
    if crs is None or dst_crs is None:
        crs = dst_crs = crs or dst_crs or rasterio.CRS.from_epsg(4326)
    out = np.full((ref["height"], ref["width"]), np.nan, np.float32)
    reproject(arr.astype(np.float32), out, src_transform=tr, src_crs=crs,
              dst_transform=ref["transform"], dst_crs=dst_crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling)
    return out


def fill_nan(a):
    """Fill NaNs from the nearest valid pixel and return the original NaN mask."""
    missing = np.isnan(a)
    if not missing.any():
        return a, missing
    nearest = ndi.distance_transform_edt(missing, return_distances=False, return_indices=True)
    return a[tuple(nearest)], missing


def window(metres, px):
    """Return Gaussian sigma matching a square window of the given width in metres."""
    return metres / np.sqrt(12) / px


def local_std(a, sigma):
    """Calculate local standard deviation within a Gaussian window."""
    mean = ndi.gaussian_filter(a, sigma)
    mean_square = ndi.gaussian_filter(a * a, sigma)
    return np.sqrt(np.maximum(mean_square - mean * mean, 0))


def sky_view(z, px, n_az=16, max_m=10000, n_ray=48, ds=4):
    """Estimate mean sky-view factor from sampled horizon angles.

    Use logarithmically spaced rays on a downsampled grid, then resample to the
    input shape.
    """
    terrain = ndi.zoom(z, 1.0 / ds, order=1)
    pixel_size = px * ds
    ray_steps = np.logspace(0, np.log10(max(max_m / pixel_size, 2)), n_ray)
    distances = np.unique(np.round(ray_steps).astype(int))
    sky = np.zeros_like(terrain, np.float32)
    for direction in range(n_az):
        angle = 2 * np.pi * direction / n_az
        horizon = np.zeros_like(terrain, np.float32)
        for distance in distances[distances >= 1]:
            row_shift = int(round(distance * np.cos(angle)))
            col_shift = int(round(distance * np.sin(angle)))
            if row_shift == 0 and col_shift == 0:
                continue
            shifted = np.roll(np.roll(terrain, -row_shift, 0), -col_shift, 1)
            # Clamp the raster edges instead of wrapping to the opposite side.
            if row_shift > 0:
                shifted[-row_shift:, :] = terrain[-row_shift:, :]
            elif row_shift < 0:
                shifted[:-row_shift, :] = terrain[:-row_shift, :]
            if col_shift > 0:
                shifted[:, -col_shift:] = terrain[:, -col_shift:]
            elif col_shift < 0:
                shifted[:, :-col_shift] = terrain[:, :-col_shift]
            np.maximum(horizon, (shifted - terrain) / (distance * pixel_size), out=horizon)
        sky += 1.0 - np.sin(np.arctan(horizon))
    scale = (z.shape[0] / terrain.shape[0], z.shape[1] / terrain.shape[1])
    out = ndi.zoom(sky / n_az, scale, order=1)
    return out[:z.shape[0], :z.shape[1]]


def manifest_files():
    """Return published manifests, excluding hidden directories."""
    root = Path(DEM_DIR).resolve()
    found = []
    for directory, children, files in os.walk(root):
        children[:] = [name for name in children if not name.startswith(".")]
        if "working_dems.json" in files:
            found.append(Path(directory) / "working_dems.json")
    return sorted(found, key=lambda p: (len(p.relative_to(root).parts), str(p)))


def _within(path, root):
    """Return whether a resolved path is contained within the resolved root."""
    path, root = Path(path).parts, Path(root).parts
    return path[:len(root)] == root


def _catalog_path(root, value, parent=None):
    """Resolve a catalog path and reject hidden paths or escapes from DEM_DIR."""
    path = ((parent or root) / value).resolve()
    if not _within(path, root):
        raise ValueError(f"Catalog path {value!r} resolves outside DEM_DIR: {path}")
    if any(part.startswith(".") for part in path.relative_to(root).parts):
        raise ValueError(f"Catalog path {value!r} refers to hidden/unpublished data: {path}")
    return path


_CATALOG_SNAPSHOT: ContextVar[tuple[Path, Any] | None] = ContextVar(
    "selenograph_catalog_snapshot", default=None)


@contextmanager
def catalog_snapshot():
    """Reuse one catalog discovery within a synchronous operation.

    Nested scopes reuse snapshots for the same DEM_DIR. Context-local state is
    restored on exit. Raster contents and path checks remain live, and public
    record APIs return independent copies.
    """
    root = Path(DEM_DIR).resolve()
    catalog = _catalog()
    token = _CATALOG_SNAPSHOT.set((root, catalog))
    try:
        yield
    finally:
        _CATALOG_SNAPSHOT.reset(token)


def _catalog():
    """Return the current snapshot or load the catalog using file revision stamps."""
    root = Path(DEM_DIR).resolve()
    snapshot = _CATALOG_SNAPSHOT.get()
    if snapshot is not None and snapshot[0] == root:
        return snapshot[1]
    stamps = []
    for manifest in manifest_files():
        _catalog_path(root, manifest)
        try:
            stat = manifest.stat()
        except FileNotFoundError:                 # Catalog directories may be replaced.
            continue
        stamps.append((manifest, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino))
    flat = []
    for path in sorted(root.glob("*")):
        if path.name.startswith(".") or not path.is_file():
            continue
        if path.suffix.lower() in (".tif", ".tiff") and not path.stem.lower().endswith(SUFFIXES):
            flat.append(str(_catalog_path(root, path)))
    return _load_catalog(root, tuple(stamps), tuple(flat))


@lru_cache(maxsize=8)
def _load_catalog(root, stamps, flat):
    """Normalize and index catalog records for one revision."""
    out, by_name, locations, companions = [], {}, {}, set()
    for manifest, *_ in stamps:
        try:
            listing = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(listing, list):
            continue

        def relative(value, parent=manifest.parent):
            return str(_catalog_path(root, value, parent).relative_to(root))

        def layer_paths(value):
            if isinstance(value, str):
                path = relative(value)
                companions.add(str(root / path))
                return path
            if isinstance(value, dict):
                if "path" in value:
                    result = dict(value)
                    if value["path"]:
                        result["path"] = layer_paths(value["path"])
                    return result
                if "status" in value:
                    return dict(value)
                return {k: layer_paths(v) for k, v in value.items()}
            return value

        for record in listing:
            if not isinstance(record, dict) or not isinstance(record.get("working_dem"), str):
                continue
            normalized: dict[str, Any] = dict(record, working_dem=relative(record["working_dem"]))
            normalized["section"] = section(normalized)
            if "annotations" in record:
                if not isinstance(record["annotations"], dict):
                    raise ValueError(f"annotations must be a dict: {manifest}")
                annotations = dict(record["annotations"])
                for kind in ANNOTATION_FILES:
                    value = annotations.get(kind)
                    if value is not None:
                        if not isinstance(value, str) or not value:
                            raise ValueError(f"annotations.{kind} must be a path: {manifest}")
                        annotations[kind] = relative(value)
                        companions.add(str(root / annotations[kind]))
                normalized["annotations"] = annotations
            if "feature_sources" in record:
                sources = record["feature_sources"]
                if not isinstance(sources, list) or any(
                        not isinstance(p, str) or not p for p in sources):
                    raise ValueError(f"feature_sources must be a list of DEM paths: {manifest}")
                normalized["feature_sources"] = [relative(p) for p in sources]
            if isinstance(record.get("layers"), dict):
                normalized["layers"] = layer_paths(record["layers"])
            name = dem_name(normalized["working_dem"])
            if name in by_name:
                raise ValueError(f"Duplicate map ID {name!r}: {locations[name]} and "
                                 f"{manifest} ({normalized['working_dem']}). "
                                 "Map IDs must be unique to protect paintings and model samples.")
            locations[name] = f"{manifest} ({normalized['working_dem']})"
            by_name[name] = normalized
            out.append(normalized)

    # Flat fallback shares the same ID namespace, including when resolving by name.
    registered = {name: str(root / r["working_dem"]) for name, r in by_name.items()}
    extra = []
    for path in flat:
        name = dem_name(path)
        if path == registered.get(name) or path in companions:
            continue
        if name in registered:
            raise ValueError(f"Duplicate map ID {name!r}: {registered[name]} and {path}. "
                             "Map IDs must be unique to protect paintings and model samples.")
        registered[name] = path
        extra.append(path)
    return out, by_name, extra


def section(record):
    """Return the declared section or the DEM's top-level dataset folder.

    Flat legacy DEMs use the DEM_DIR directory name.
    """
    if record.get("section"):
        return record["section"]
    root = Path(DEM_DIR).resolve()
    parts = _catalog_path(root, record["working_dem"]).relative_to(root).parts
    return parts[0] if len(parts) > 1 else root.name


def records():
    """Return independent catalog records with paths relative to DEM_DIR."""
    return deepcopy(_catalog()[0])


def record_path(record, layer=None):
    """Resolve a normalized catalog record's DEM or named layer to an absolute path."""
    if layer is None:
        value = record.get("working_dem")
    else:
        value = record.get("layers", {}).get(layer)
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        return None
    return str(_catalog_path(Path(DEM_DIR).resolve(), value))


def annotation_path(name, kind):
    """Return a declared bundled annotation path, or None.

    The returned path may not exist and is never an output destination.
    """
    if kind not in ANNOTATION_FILES:
        raise ValueError(f"Unknown annotation kind: {kind!r}")
    record = map_record(name)
    value = record.get("annotations", {}).get(kind) if record else None
    return str(_catalog_path(Path(DEM_DIR).resolve(), value)) if value else None


def dem_files():
    """List manifest DEMs in import order, followed by unregistered legacy DEMs."""
    listing, _, flat = _catalog()
    out = []
    for record in listing:
        path = record_path(record)
        if path and Path(path).is_file():
            out.append(path)
    return out + list(flat)


def map_record(name):
    """Return an independent catalog record for a map ID or filename."""
    by_name = _catalog()[1]
    # IDs may contain dots; strip a file extension only when the exact ID is absent.
    record = by_name.get(os.fspath(name))
    if record is None:
        record = by_name.get(dem_name(name))
    return deepcopy(record)


def dem_path(name):
    """Resolve a stable map ID without assuming its directory or TIFF extension."""
    record = map_record(name)
    if record:
        return record_path(record)
    files = {dem_name(p): p for p in dem_files()}
    value = os.fspath(name)
    if value in files:
        return files[value]
    if dem_name(value) in files:
        return files[dem_name(value)]
    key = value
    if Path(value).name != value or Path(value).suffix.lower() in (".tif", ".tiff"):
        key = dem_name(value)
    return str((Path(DEM_DIR) / f"{key}.tif").resolve())


def map_site(name):
    """Return the declared geographic group or site for a map."""
    record = map_record(name)
    if record:
        return record.get("group", record.get("site"))
    for path in dem_files():
        if dem_name(path) == name:
            return site_id(path)
    return None


def dem_name(path):
    """Return a map ID from the filename stem."""
    return os.path.splitext(os.path.basename(path))[0]


def site_id(path):
    """Return a DEM geographic group, site tag, or filename stem."""
    record = map_record(dem_name(path))
    if record and (record.get("group") or record.get("site")):
        return record.get("group") or record["site"]
    with rasterio.open(path) as src:
        return src.tags().get("site_id", dem_name(path))


def companion(dem_path, kind):
    """Find a display layer in the manifest or beside a legacy DEM."""
    record = map_record(dem_name(dem_path))
    if record:
        path = record_path(record, kind)
        if path and os.path.isfile(path):
            return path
    stem = os.path.splitext(dem_path)[0]
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        if os.path.exists(stem + f"_{kind}" + ext):
            return stem + f"_{kind}" + ext
    return None


def _output_component(value):
    """Validate a single, non-hidden output path component."""
    value = os.fspath(value)
    if (not value or value.startswith(".") or "/" in value or "\\" in value
            or "\x00" in value):
        raise ValueError(f"Output names must be single, non-hidden path components: {value!r}")
    return value


def _output_path(root, relative):
    """Resolve a contained output path outside processed data without creating it."""
    root = Path(root).resolve()
    target = root / relative
    resolved = target.resolve()
    if not _within(resolved, root) or _within(resolved, Path(DEM_DIR).resolve()):
        raise ValueError(f"Output path must stay under its output root and outside processed data: {target}")
    return str(target)


def _output_relative_dir(name):
    """Return the DEM parent directory relative to DEM_DIR."""
    name = _output_component(name)
    dem = dem_path(name)
    if not dem:
        raise ValueError(f"No DEM path for map {name!r}")
    return Path(dem).resolve().relative_to(Path(DEM_DIR).resolve()).parent


def out_dir(name, final=True):
    """Return the saved or draft directory mirroring the DEM parent."""
    return _output_path(OUT_DIR if final else DRAFT_DIR, _output_relative_dir(name))


def _legacy_output_file(name, filename, final=True):
    """Resolve a file in the legacy saved or draft layout."""
    name, filename = _output_component(name), _output_component(filename)
    if final:
        return _output_path(LEGACY_OUT_DIR, Path(name) / filename)
    suffix = {"painting.npy": ".npy", "accepted.npy": ".accepted.npy",
              "confidence.npy": ".confidence.npy"}.get(filename, "." + filename)
    return _output_path(LEGACY_DRAFT_DIR, name + suffix)


def _mirrored_output_file(name, filename, final=True):
    """Resolve a legacy mirrored file while excluding current output directories."""
    root = Path(LEGACY_MIRRORED_DIR).resolve()
    folder = root / _output_relative_dir(name)
    for reserved in (OUT_DIR, DRAFT_DIR, MAP_DIR, LEGACY_OUT_DIR, LEGACY_DRAFT_DIR):
        reserved = Path(reserved).resolve()
        if _within(reserved, root) and _within(folder.resolve(), reserved):
            return None
    if not final:
        filename = {"painting.npy": "draft.npy", "accepted.npy": "draft.accepted.npy",
                    "confidence.npy": "draft.confidence.npy"}.get(filename, "draft." + filename)
    return _output_path(root, folder / f"{_output_component(name)}_{_output_component(filename)}")


def _painting_output_file(name, filename, final=True):
    """Resolve a painting file or a legacy TIFF stored beside paintings."""
    root = OUT_DIR if final else DRAFT_DIR
    return _output_path(root, Path(out_dir(name, final)) / f"{name}_{filename}")


def output_file(name, filename, *, existing=False, final=True):
    """Resolve a painting artifact or saved GeoTIFF output path.

    Legacy reads select a complete saved or draft set. Missing sidecars do not
    fall back independently, and metadata-only edits preserve existing labels.
    """
    name, filename = _output_component(name), _output_component(filename)
    export = final and filename in ("labels.tif", "map.tif")
    if export:
        target = _output_path(MAP_DIR, _output_relative_dir(name) / f"{name}_{filename}")
    else:
        target = _painting_output_file(name, filename, final)
    if existing:
        markers = ("painting.npy", "labels.tif") if final else ("painting.npy",)
        current = [output_file(name, m, final=final) for m in markers]
        if final:
            current.append(_painting_output_file(name, "labels.tif"))
        if any(os.path.isfile(p) for p in current):
            # Legacy TIFFs remain readable until a new labels export replaces the set.
            if export and not os.path.isfile(output_file(name, "labels.tif")):
                previous = _painting_output_file(name, filename)
                if os.path.isfile(previous):
                    return previous
        else:
            for resolver in (_mirrored_output_file, _legacy_output_file):
                candidates = [resolver(name, m, final) for m in markers]
                if any(p and os.path.isfile(p) for p in candidates):
                    return resolver(name, filename, final)
    return target


def meta_get(name, final=None):
    """Merge saved metadata with draft changes, or read saved metadata only.

    Verification is explicit metadata. Set final=True to exclude draft changes.
    """
    record = map_record(name) or {}
    annotations = record.get("annotations", {})
    metadata = {key: annotations.get(key, record.get(key))
                for key in ("verified", "label_status", "restricted", "provenance")
                if key in annotations or key in record}
    sources = [_legacy_output_file(name, "meta.json"), _mirrored_output_file(name, "meta.json"),
               output_file(name, "meta.json")]
    if final is not True:
        sources.append(output_file(name, "meta.json", final=False))
    for path in sources:
        if path is None:
            continue
        try:
            with open(path) as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                metadata.update(saved)
        except (OSError, ValueError):
            pass
    return deepcopy(metadata)


def meta_set(name, *, final=True, **kw):
    """Save metadata to drafts or commit it with the saved painting."""
    m = meta_get(name)
    m["name"] = name
    m.update(kw)
    path = output_file(name, "meta.json", final=final)
    draft = output_file(name, "meta.json", final=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = output_file(name, "meta.json.tmp", final=final)
    with open(temporary, "w") as f:
        json.dump(m, f, indent=2)
    os.replace(temporary, path)
    if final and os.path.isfile(draft):
        os.remove(draft)


def painting_files(name, final=False, *, existing=False):
    """Return painting, acceptance, and confidence paths for one output set.

    Writes use canonical paths. With existing=True, legacy files remain readable
    until a canonical painting of the same kind exists.
    """
    return tuple(output_file(name, f, existing=existing, final=final)
                 for f in ("painting.npy", "accepted.npy", "confidence.npy"))


def save_painting(name, cells, final=False, accepted=None, confidence=None):
    """Save a draft or final painting with acceptance and confidence flags.

    Omitted flags are cleared: painted cells default to SURE and manual labels.
    Draft files remain separate from saved exports.
    """
    path, accepted_path, confidence_path = painting_files(name, final)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if confidence is not None and not ((confidence != SURE) & (cells > 0)).any():
        confidence = None                            # Default confidence needs no sidecar.
    for p, a in ((accepted_path, accepted), (confidence_path, confidence), (path, cells)):
        if a is None or (p == accepted_path and not a.any()):
            if os.path.exists(p):
                os.remove(p)
            continue
        # Atomic replacement prevents partial painting files.
        temporary = _output_path(OUT_DIR if final else DRAFT_DIR, Path(p + ".tmp"))
        with open(temporary, "wb") as f:
            np.save(f, a)
        os.replace(temporary, p)


@catalog_snapshot()
def seed_paintings(names=None):
    """Initialize saved and draft paintings from bundled annotations.

    Skip any map with existing painting, sidecar, export, or metadata files,
    including empty paintings.
    """
    if names is None:
        names = [dem_name(p) for p in dem_files()]
    seeded = []
    artifacts = ("painting.npy", "accepted.npy", "confidence.npy", "meta.json",
                 "labels.tif", "map.tif", "thumb.png")
    for name in dict.fromkeys(names):
        source = annotation_path(name, "painting")
        if not source or not os.path.isfile(source):
            continue
        prior = []
        for final in (False, True):
            for filename in artifacts:
                prior.extend((output_file(name, filename, final=final),
                              _painting_output_file(name, filename, final),
                              _mirrored_output_file(name, filename, final),
                              _legacy_output_file(name, filename, final)))
        if any(p and os.path.exists(p) for p in prior):
            continue
        cells = np.load(source, allow_pickle=False)
        if cells.shape != (GRID, GRID) or not np.isin(cells, (0, *CODES)).all():
            raise ValueError(f"Invalid bundled painting: {name}")
        if not cells.any():
            continue
        flags = {}
        for kind in ("accepted", "confidence"):
            path = annotation_path(name, kind)
            values = np.load(path, allow_pickle=False) if path and os.path.isfile(path) else None
            allowed = (0, 1) if kind == "accepted" else (0, SURE, MOSTLY, UNSURE)
            if values is not None and (values.shape != cells.shape or not np.isin(values, allowed).all()):
                raise ValueError(f"Invalid bundled {kind} flags: {name}")
            flags[kind] = values
        for final in (True, False):
            save_painting(name, cells, final=final, **flags)
        seeded.append(name)
    return seeded


def _painting_source(name, final=None):
    """Return painting cells, saved status, and bundled source identity."""
    versions = (False, True) if final is None else (final,)
    candidates = [(output_file(name, "painting.npy", existing=True, final=saved), saved, False)
                  for saved in versions]
    # Raster-only user saves override bundled cell labels.
    if final is not False and not os.path.isfile(output_file(name, "labels.tif", existing=True)):
        bundled_path = annotation_path(name, "painting")
        if bundled_path:
            candidates.append((bundled_path, True, True))
    for path, saved, bundled in candidates:
        if path and os.path.isfile(path):
            cells = np.load(path, allow_pickle=False)
            if cells.shape == (GRID, GRID):
                return cells.astype(np.uint8), saved, bundled
    return np.zeros((GRID, GRID), np.uint8), None, False


def painting_source(name, final=None):
    """Return independent cells and source status: False, True, or None.

    Drafts take precedence over saved and bundled paintings, including empty
    user paintings. Set final=True for saved/bundled only or final=False for
    drafts only. Reading never copies bundled files to user outputs.
    """
    cells, final, _ = _painting_source(name, final)
    return cells, final


def load_painting(name):
    """Load the effective painting, preferring drafts over saved or bundled cells."""
    return painting_source(name)[0]


def has_saved_labels(name):
    """Return whether saved or bundled labels contain terrain units."""
    cells, source = painting_source(name, final=True)
    if source is not None:
        # An explicitly cleared save overrides older bundled/raster labels.
        return bool(np.isin(cells, list(CODES)).any())
    path = output_file(name, "labels.tif", existing=True)
    if not os.path.isfile(path):
        path = annotation_path(name, "labels")
    if not path or not os.path.isfile(path):
        return False
    with rasterio.open(path) as source:
        for _, block in source.block_windows(1):
            labels = source.read(1, window=block, masked=True)
            if np.isin(labels.compressed(), list(CODES)).any():
                return True
    return False


@catalog_snapshot()
def painted_maps():
    """Return sets of painted map IDs and IDs containing accepted predictions.

    Include effective draft, saved, and bundled paintings without creating
    outputs. Intersect with dem_files() when only installed maps are required.
    """
    candidates = {os.path.basename(p)[:-4] for p in glob.glob(f"{LEGACY_DRAFT_DIR}/*.npy")
                  if not p.endswith((".accepted.npy", ".confidence.npy"))}
    candidates |= {os.path.basename(os.path.dirname(p))
                   for p in glob.glob(f"{LEGACY_OUT_DIR}/*/painting.npy")}
    candidates |= {dem_name(p) for p in dem_files()}
    # Keep flat/uninstalled IDs discoverable without scanning evaluation archives.
    for root, suffix in ((OUT_DIR, "_painting.npy"), (DRAFT_DIR, "_painting.npy"),
                         (LEGACY_MIRRORED_DIR, "_painting.npy"), (LEGACY_MIRRORED_DIR, "_draft.npy")):
        candidates |= {p.name[:-len(suffix)] for p in Path(root).glob(f"*{suffix}")}
    painted, accepted = set(), set()
    for name in candidates:
        cells, final, bundled = _painting_source(name)
        if not cells.any():
            continue
        painted.add(name)
        flags = _beside(name, final, 1, cells, bundled=bundled)
        if flags is not None and (flags.astype(bool) & (cells > 0)).any():
            accepted.add(name)
    return painted, accepted


def _beside(name, final, which, cells, *, bundled=None):
    """Load flags from the selected painting source only."""
    if final is None:
        return None
    if bundled is None:
        _, _, bundled = _painting_source(name, final)
    if bundled:
        kind = ("painting", "accepted", "confidence")[which]
        path = annotation_path(name, kind)
    else:
        path = painting_files(name, final, existing=True)[which]
    if path and os.path.isfile(path):
        a = np.load(path, allow_pickle=False)
        if a.shape == cells.shape:
            return a
    return None


def load_accepted(name, final=None):
    """Load accepted prediction flags from the selected painting source."""
    cells, final, bundled = _painting_source(name, final)
    accepted = _beside(name, final, 1, cells, bundled=bundled)
    if accepted is None:
        return np.zeros((GRID, GRID), bool)
    return accepted.astype(bool) & (cells > 0)


def load_confidence(name, final=None):
    """Load confidence levels, defaulting painted cells to SURE and others to zero.

    Set final=True for saved/bundled paintings or final=False for drafts only.
    """
    cells, final, bundled = _painting_source(name, final)
    confidence = _beside(name, final, 2, cells, bundled=bundled)
    if confidence is None:
        confidence = np.full(cells.shape, SURE)
    else:
        confidence = np.where(confidence > 0, confidence, SURE)
    return np.where(cells > 0, confidence, 0).astype(np.uint8)


def _feature_plan(dem_path):
    """Order feature-source dependencies and reject cycles before raster builds."""
    root = Path(DEM_DIR).resolve()

    def record_at(path):
        # Unique map IDs allow direct lookup followed by path validation.
        record = map_record(dem_name(path))
        return record if record and record_path(record) == path else {}

    target = str(Path(dem_path).resolve())
    dependencies, state, order = {}, {}, []
    pending = [(target, False)]
    while pending:
        path, leaving = pending.pop()
        if leaving:
            state[path] = 2
            order.append(path)
            continue
        if state.get(path) == 1:
            raise ValueError(f"Cyclic feature_sources dependency: {path}")
        if state.get(path) == 2:
            continue
        state[path] = 1
        sources = record_at(path).get("feature_sources", [])
        dependencies[path] = [str(_catalog_path(root, p)) for p in sources]
        pending.append((path, True))
        pending.extend((p, False) for p in reversed(dependencies[path]))
    return order, dependencies


def _feature_warp(values, source, target):
    """Reproject feature values using nearest-neighbour resampling.

    Missing CRS is allowed only when source and target grids match exactly.
    """
    source_shape = (source["height"], source["width"])
    target_shape = (target["height"], target["width"])
    if values.ndim not in (2, 3) or values.shape[-2:] != source_shape:
        raise ValueError("Feature array shape does not match its source grid")
    if (source_shape == target_shape and source["transform"] == target["transform"]
            and source["crs"] == target["crs"]):
        return values.copy()
    if source["crs"] is None or target["crs"] is None:
        raise ValueError("Feature transport requires both CRSs unless grids match exactly")
    out = np.full((*values.shape[:-2], *target_shape), np.nan, np.float32)
    reproject(values.astype(np.float32, copy=False), out,
              src_transform=source["transform"], src_crs=source["crs"],
              dst_transform=target["transform"], dst_crs=target["crs"],
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest,
              warp_mem_limit=64, num_threads=1)
    return out


def _terrain_features(z, px):
    """Calculate six terrain features before splitting shared context into tiles."""
    zf, hole = fill_nan(z)
    gy, gx = np.gradient(zf, px)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    values = dict(slope=slope, rough=local_std(slope, window(ROUGH_M, px)),
                  svf=sky_view(zf, px),
                  rel_local=zf - ndi.gaussian_filter(zf, window(LOCAL_M, px)),
                  curv=ndi.gaussian_laplace(zf, window(CURV_M, px)) / px ** 2,
                  svf_local=sky_view(zf, px, max_m=SKY_LOCAL_M, n_ray=16, ds=1))
    for value in values.values():
        value[hole] = np.nan
    return values


def _source_contexts(leaves):
    """Join adjacent, aligned leaf DEMs without resampling their elevations.

    Only non-overlapping grids with a shared CRS and pixel alignment are joined.
    Disconnected terrain and declared consumer footprints remain separate.
    """
    groups = []
    for path in leaves:
        with rasterio.open(path) as src:
            tr = src.transform
            if max(src.shape) > MAX_SIDE:
                raise ValueError(f"Feature source build changed its native grid: {path}")
            if src.crs is None or tr.b or tr.d or tr.a <= 0 or tr.e >= 0:
                continue
            tile = (path, tr, src.height, src.width)
            for crs, origin, tiles in groups:
                if (src.crs, tr.a, tr.e) != (crs, origin.a, origin.e):
                    continue
                offset = (~origin) @ (tr.c, tr.f)
                if np.allclose(offset, np.round(offset), rtol=0, atol=1e-6):
                    tiles.append(tile)
                    break
            else:
                groups.append((src.crs, tr, [tile]))
    contexts = {}
    for _, origin, tiles in groups:
        rectangles = []
        for path, tr, height, width in tiles:
            col, row = (~origin) @ (tr.c, tr.f)
            row, col = round(row), round(col)
            rectangles.append((path, row, col, row + height, col + width))
        pending = list(rectangles)
        while pending:
            group = [pending.pop()]
            for _, top, left, bottom, right in group:
                for item in pending[:]:
                    _, t, l, b, r = item
                    # Require shared edges; isolated corner contact is insufficient.
                    row_overlap = min(bottom, b) - max(top, t)
                    col_overlap = min(right, r) - max(left, l)
                    if ((row_overlap > 0 and col_overlap >= 0)
                            or (row_overlap >= 0 and col_overlap > 0)):
                        group.append(item)
                        pending.remove(item)
            if len(group) < 2:
                continue
            # Overlapping leaves may feed disjoint intermediate owners. Do not
            # merge their elevations; the normal ownership checks still apply.
            if any(min(a[3], b[3]) > max(a[1], b[1]) and min(a[4], b[4]) > max(a[2], b[2])
                   for i, a in enumerate(group) for b in group[i + 1:]):
                continue
            top, left = min(p[1] for p in group), min(p[2] for p in group)
            bottom, right = max(p[3] for p in group), max(p[4] for p in group)
            terrain = np.full((bottom - top, right - left), np.nan, np.float32)
            for path, t, l, b, r in group:
                terrain[t-top:b-top, l-left:r-left] = read(path)[0]
            features = _terrain_features(terrain, origin.a)
            for path, t, l, b, r in group:
                contexts[path] = {name: values[t-top:b-top, l-left:r-left]
                                  for name, values in features.items()}
    return contexts


def build(dem_path):
    """Return features, layer names, hillshade, output profile, and pixel size.

    Aligned feature_sources share terrain context before BASE_FEATS are computed
    and transported with nearest-neighbour resampling. Display layers and
    geometry remain local. Uncovered pixels and DEM holes become NaN; overlapping
    source footprints raise an error even where terrain is absent. Missing CRS
    is allowed only for identical grids.

    Each dependency is built once per call and released after its final use.
    Returned arrays are independent; features are not cached between calls.
    """
    order, dependencies = _feature_plan(dem_path)
    consumers = {path: 0 for path in order}
    for sources in dependencies.values():
        for path in sources:
            consumers[path] += 1
    contexts = _source_contexts([path for path in order[:-1] if not dependencies[path]])
    features = {}
    result = None
    for path in order:
        result = _build_local(path, context=contexts.pop(path, None))
        X, names, _, profile, _ = result
        indices = [names.index(name) for name in BASE_FEATS]
        sources = dependencies[path]
        if sources:
            terrain = np.isfinite(X[names.index("slope")])
            occupied = np.zeros(terrain.shape, bool)
            transported = np.full((len(BASE_FEATS), *terrain.shape), np.nan, np.float32)
            for source in sources:
                values, grid = features[source]
                footprint = _feature_warp(
                    np.ones((grid["height"], grid["width"]), np.float32), grid, profile) == 1
                if (occupied & footprint).any():
                    raise ValueError(f"Overlapping feature source ownership for {path}: {source}")
                occupied |= footprint
                warped = _feature_warp(values, grid, profile)
                transported[:, footprint] = warped[:, footprint]
                consumers[source] -= 1
                if consumers[source] == 0:
                    del features[source]
            transported[:, ~terrain] = np.nan
            X[indices] = transported
        if consumers[path]:
            # Linked features must preserve the declared source raster's native grid.
            with rasterio.open(path) as src:
                if ((src.height, src.width) != X.shape[-2:]
                        or src.transform != profile["transform"] or src.crs != profile["crs"]):
                    raise ValueError(f"Feature source build changed its native grid: {path}")
            features[path] = (X[indices].copy(), profile)
    assert result is not None  # The plan always contains the requested DEM.
    return result


def _build_local(dem_path, *, context=None):
    """Build local display layers with optional features from shared terrain."""
    elevation, transform, crs, pixel_size = read(dem_path, MAX_SIDE)
    ref = dict(height=elevation.shape[0], width=elevation.shape[1], transform=transform, crs=crs)
    profile: dict[str, Any] = dict(driver="GTiff", dtype="uint8", count=1, nodata=255,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256, **ref)

    filled, missing = fill_nan(elevation)
    layers = _terrain_features(elevation, pixel_size) if context is None else dict(context)
    layers.update(rel=filled - ndi.gaussian_filter(filled, window(REL_M, pixel_size)),
                  zscene=filled - np.median(filled))
    names = list(LAYERS)

    for kind, resampling in (("psr", Resampling.nearest), ("cpr", Resampling.bilinear),
                              ("nac", Resampling.bilinear)):
        path = companion(dem_path, kind)
        if not path:
            continue
        values, source_transform, source_crs, _ = read(path)
        values = warp(values, source_transform, source_crs, ref, resampling)
        good = np.isfinite(values)
        if good.any() and kind == "cpr":       # Mask extreme companion values.
            low, high = np.percentile(values[good], [0.5, 99.5])
            values[(values < low) | (values > high)] = np.nan
        layers[kind] = values
        names.append(kind)

    for values in layers.values():
        values[missing] = np.nan
    X = np.stack([layers[name] for name in names]).astype(np.float32)
    shade = hillshade(filled, pixel_size)
    shade[missing] = 0
    return X, names, shade, profile, pixel_size


def hillshade(z, px, az=315, alt=25):
    """Calculate normalized illumination from smoothed terrain and solar angles."""
    smoothed = ndi.gaussian_filter(z, 2)
    gy, gx = np.gradient(smoothed, px)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    illumination = (np.sin(np.radians(alt)) * np.cos(slope) +
                    np.cos(np.radians(alt)) * np.sin(slope) *
                    np.cos(np.radians(az) - aspect))
    return np.clip(illumination, 0, 1)


def blockwise(predict_rows, X, rows=256, dtype=np.uint8, fill=255):
    """Predict raster rows in blocks, using fill where the first feature is missing."""
    features, height, width = X.shape
    out = np.full((height, width), fill, dtype)
    for start in range(0, height, rows):
        stop = min(height, start + rows)
        block = X[:, start:stop].reshape(features, -1).T
        valid = np.isfinite(block[:, 0])           # Slope identifies valid terrain.
        predictions = np.full(block.shape[0], fill, dtype)
        if valid.any():
            predictions[valid] = predict_rows(block[valid]).astype(dtype)
        out[start:stop] = predictions.reshape(stop - start, width)
    return out


def per_pixel(values, shape):
    """Expand painting-cell values onto raster pixels using floor-based cell edges."""
    rows = np.searchsorted([i * shape[0] // GRID for i in range(1, GRID)],
                           np.arange(shape[0]), side="right")
    cols = np.searchsorted([j * shape[1] // GRID for j in range(1, GRID)],
                           np.arange(shape[1]), side="right")
    return values[np.ix_(rows, cols)]


def cells_to_labels(cells, shape, valid=None):
    """Expand painting cells to raster labels, using 255 for unknown or invalid pixels."""
    lab = per_pixel(cells, shape).astype(np.uint8)
    lab[lab == 0] = 255
    if valid is not None:
        if valid.shape != lab.shape:
            raise ValueError("Terrain mask must match the output label grid")
        lab[~valid] = 255
    return lab


def labels_to_cells(lab, k=4):
    """Estimate each painting cell's majority unit from k by k raster samples.

    Return zero for cells with no sampled terrain units.
    """
    H, W = lab.shape
    r = ((np.arange(GRID * k) + .5) * H / (GRID * k)).astype(int)
    c = ((np.arange(GRID * k) + .5) * W / (GRID * k)).astype(int)
    blocks = lab[np.ix_(r, c)].reshape(GRID, k, GRID, k)
    counts = np.stack([(blocks == code).sum((1, 3)) for code in CODES])
    best = np.array(list(CODES), np.uint8)[counts.argmax(0)]
    return np.where(counts.max(0) > 0, best, 0).astype(np.uint8)


def smart_fill(suggested, seed=None, inside=None, unit=None, min_patch=4):
    """Select a connected predicted unit or matching units within a cell mask.

    A seed selects its connected unit region; inside and unit select matching
    patches. Remove small unseeded patches and fill holes below min_patch cells.
    """
    if seed is not None:
        if suggested[seed] == 0:
            return np.zeros(suggested.shape, bool)
        parts, _ = ndi.label(suggested == suggested[seed])
        region = parts == parts[seed]
    else:
        region = inside & (suggested == unit)
        parts, _ = ndi.label(region)
        region &= np.bincount(parts.ravel())[parts] >= min_patch
    holes, _ = ndi.label(~region)
    small = (np.bincount(holes.ravel())[holes] < min_patch) & (holes > 0)
    if inside is not None:
        small &= inside
    return (region | small) & (suggested > 0)


def write_map(path, arr, prof):
    """Write a single-band GeoTIFF with the terrain-unit colour map."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr, 1)
        d.write_colormap(1, CMAP)
