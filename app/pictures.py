"""Render terrain, annotations, and predictions for Paint and Gallery."""
import io

import numpy as np
from PIL import Image
from rasterio import Affine

from app import core

LUMA = np.array([0.2126, 0.7152, 0.0722])          # RGB luminance weights.
LIFT = 0.2          # Minimum brightness for unit colours in shadow.
DISAGREEMENT = np.array([255, 212, 59], dtype=np.float32)


def floor():
    """Return the confidence of a uniform distribution over all units."""
    return 1 / len(core.CODES)


def render(base, layers):
    """Overlay (unit codes, opacity) layers while preserving terrain shading.

    Skip missing layers and resample mismatched grids with nearest neighbours.
    """
    rgb = np.repeat(base[:, :, None], 3, 2).astype(np.float32)
    shade = (LIFT + (1 - LIFT) * base)[:, :, None]
    for layer, alpha in layers:
        if layer is None or alpha <= 0:
            continue
        p = layer if layer.shape == base.shape else np.array(
            Image.fromarray(layer).resize(base.shape[::-1], Image.NEAREST))
        for code in core.CODES:
            m = p == code
            if m.any():
                colour = np.array(core.CMAP[code][:3]) / 255.0
                tint = np.clip(shade[m] * colour / (colour @ LUMA), 0, 1)
                rgb[m] = (1 - alpha) * rgb[m] + alpha * tint
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def disagreement(base, prediction, cells, alpha):
    """Highlight disagreements between known predictions and painted cells.

    Use painting cell edges and nearest-neighbour prediction resampling.
    Ignore unpainted cells and missing predictions.
    """
    predicted = prediction if prediction.shape == base.shape else np.asarray(
        Image.fromarray(prediction).resize(base.shape[::-1], Image.Resampling.NEAREST))
    painted = core.cells_to_labels(cells, base.shape)
    different = (np.isin(predicted, list(core.CODES)) & np.isin(painted, list(core.CODES))
                 & (predicted != painted))
    rgb = np.repeat(np.clip(base, 0, 1)[:, :, None], 3, axis=2).astype(np.float32) * 255
    rgb[different] = (1 - alpha) * rgb[different] + alpha * DISAGREEMENT
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def terrain(base):
    """Convert a normalized terrain array to a greyscale image."""
    return Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))


def resized(values, shape):
    """Resample a float array bilinearly while preserving its missing-data mask."""
    if values.shape == tuple(shape):
        return values
    missing = np.isnan(values)
    img = Image.fromarray(np.where(missing, 0, values).astype(np.float32), mode="F")
    out = np.array(img.resize(shape[::-1], Image.BILINEAR))
    gone = np.array(Image.fromarray(missing.astype(np.uint8) * 255).resize(
        shape[::-1], Image.NEAREST)) > 0
    return np.where(gone, np.nan, out)


def confidence_base(conf, shape):
    """Scale confidence from uniform to certain on a fixed greyscale of 0 to 1."""
    grey = (resized(conf, shape) - floor()) / (1 - floor())
    return np.clip(np.nan_to_num(grey, nan=0), 0, 1).astype(np.float32)


def whiten(picture, blank):
    """Set pixels selected by the blank mask to white."""
    rgb = np.array(picture)
    rgb[blank] = 255
    return Image.fromarray(rgb)


def png(image):
    """Encode a PIL image as PNG bytes."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@core.catalog_snapshot()
def tile_bounds(name, shape, transform):
    """Return normalized raster bounds within its nominal square tile.

    Use aligned neighbours to recover cropped tile offsets. Unregistered or
    unaligned maps use their own upper-left-anchored square extent. Preserve
    raster NoData and annotation coordinates.
    """
    record = core.map_record(name) or {}
    grid = record.get("grid", {})
    h, w = record.get("size", grid.get("shape", shape))
    tr = Affine(*record.get("transform", grid.get("transform", transform))[:6])
    side, row, col = max(h, w), 0, 0
    section = core.section(record) if record.get("working_dem") else None
    peers = []
    for peer in core.records() if section else ():
        other_grid = peer.get("grid", {})
        size = peer.get("size", other_grid.get("shape"))
        other_tr = peer.get("transform", other_grid.get("transform"))
        if not size or not other_tr or core.section(peer) != section:
            continue
        ph, pw = size
        if ph != pw or ph < side:
            continue
        crs, other_crs = record.get("crs", grid.get("crs_wkt")), peer.get("crs", other_grid.get("crs_wkt"))
        if crs and other_crs and crs != other_crs:
            continue
        pt = Affine(*other_tr[:6])
        if np.allclose((tr.a, tr.b, tr.d, tr.e), (pt.a, pt.b, pt.d, pt.e), rtol=0, atol=1e-8):
            peers.append((ph, pt))
    for nominal, origin in sorted(peers, key=lambda p: p[0], reverse=True):
        x, y = ~origin @ (tr.c, tr.f)
        if not np.allclose((x, y), (round(x), round(y)), rtol=0, atol=1e-5):
            continue
        x, y = round(x) % nominal, round(y) % nominal
        if x + w <= nominal and y + h <= nominal:
            side, row, col = nominal, y, x
            break
    return (col / side, row / side, w / side, h / side)


def blacken(image, missing):
    """Set missing terrain pixels to black, including overlaid annotations."""
    mask = Image.fromarray(np.asarray(missing, dtype=np.uint8)).resize(image.size, Image.Resampling.NEAREST)
    rgb = np.array(image.convert("RGB"))
    rgb[np.asarray(mask, dtype=bool)] = 0
    return Image.fromarray(rgb)


def tile_frame(image, bounds, *, missing=None, size=None):
    """Place a raster in its square tile with opaque black NoData."""
    x, y, w, h = bounds
    size = size or max(1, round(max(image.width / w, image.height / h)))
    left, top = round(x * size), round(y * size)
    right, bottom = round((x + w) * size), round((y + h) * size)
    fitted = image.convert("RGB").resize((max(1, right - left), max(1, bottom - top)), Image.Resampling.LANCZOS)
    if missing is not None:
        fitted = blacken(fitted, missing)
    frame = Image.new("RGB", (size, size), "black")
    frame.paste(fitted, (left, top))
    return frame


def gallery_preview(image, size=240, *, bounds=None, missing=None):
    """Render a square thumbnail with geographic placement and NoData masking."""
    side = max(image.size)
    bounds = bounds or (0, 0, image.width / side, image.height / side)
    return tile_frame(image, bounds, missing=missing, size=size)
