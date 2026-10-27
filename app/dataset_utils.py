"""Read evaluation files and describe raster grids."""
import json
from dataclasses import dataclass
from pathlib import Path

from rasterio import Affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds

MIN_COVERAGE = 0.9


def _json(path):
    """Read a UTF-8 JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path(root, relative):
    """Resolve a dataset path and reject paths that escape the dataset root."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes root: {relative}")
    return path


@dataclass(frozen=True)
class _Grid:
    """Raster dimensions, affine transform, and coordinate reference system."""

    shape: tuple[int, int]
    transform: Affine
    crs: CRS

    @property
    def bounds(self):
        """Return the raster bounds in its coordinate reference system."""
        return array_bounds(self.shape[0], self.shape[1], self.transform)

    def metadata(self):
        """Return JSON-compatible grid metadata."""
        return {"shape": list(self.shape), "transform": list(self.transform)[:6],
                "crs_wkt": self.crs.to_wkt(), "bounds": list(self.bounds)}
