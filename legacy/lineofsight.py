from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.warp import reproject


CSV_CRS = "EPSG:4326"

if not shutil.which("gdal") and not shutil.which("gdal_raster_viewshed"):
    raise RuntimeError("GDAL not found. Please install it via conda.")

def find_gdal_viewshed_command() -> list[str]:
    """
    Return the available GDAL viewshed command.

    The code prefers the modern `gdal raster viewshed` command, but it also
    supports older installs exposing `gdal_raster_viewshed`.
    """
    if shutil.which("gdal"):
        return ["gdal", "raster", "viewshed"]
    if shutil.which("gdal_raster_viewshed"):
        return ["gdal_raster_viewshed"]

    raise RuntimeError(
        "Could not find a GDAL viewshed command in the active environment."
    )


def run_viewshed(
    dem_path: str,
    x: float,
    y: float,
    observer_h: float,
    out_tif: str,
    max_distance: float,
) -> None:
    """
    Run one binary viewshed for a single observer.
    The observer position is passed as X,Y,H through --pos.
    """
    cmd = find_gdal_viewshed_command()

    full_cmd = [
        *cmd,
        "--overwrite",
        "--max-distance",
        str(max_distance),
        "--target-height",
        "0",
        "--visible-value",
        "1",
        "--invisible-value",
        "0",
        "--out-of-range-value",
        "0",
        "--dst-nodata",
        "0",
        "--pos",
        f"{x},{y},{observer_h}",
        dem_path,
        out_tif,
    ]

    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "GDAL viewshed failed\n"
            f"Command: {' '.join(full_cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def _normalize_sample(sample: Mapping[str, object]) -> dict[str, float]:
    """
    Accept both the student's metadata keys and the cleaned version.
    """
    lon = float(sample["lon"])
    lat = float(sample["lat"])

    if "observer_height" in sample:
        observer_height = float(sample["observer_height"])
    else:
        observer_height = float(sample.get("elevation_m", 0.0))

    heading_deg = float(sample.get("heading_deg", 0.0)) % 360.0

    return {
        "lon": lon,
        "lat": lat,
        "observer_height": observer_height,
        "heading_deg": heading_deg,
    }


def build_frequency_overlay(
    sample_metadata: Sequence[Mapping[str, object]],
    tif_path: str | Path,
    max_distance: float,
    fan_angle_deg: float = 120.0,
    csv_crs: str = CSV_CRS,
) -> dict:
    """
    Build a visibility frequency raster from a DEM and a set of observers.

    Parameters
    ----------
    sample_metadata:
        Iterable of dictionaries containing at least:
        lon, lat, observer_height, heading_deg
    tif_path:
        Path to the DEM GeoTIFF.
    max_distance:
        Maximum analysis distance in the DEM's units.
    fan_angle_deg:
        Field of view around heading_deg. Use 360 for no directional masking.
    csv_crs:
        CRS of the input lon/lat coordinates. Default is EPSG:4326.

    Returns
    -------
    dict with:
        count_overlay: 2D numpy array
        observer_points_xy: list[(x, y)]
        view_extent: (left, right, bottom, top)
    """
    tif_path = Path(tif_path)

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError("The DEM has no CRS.")
        if abs(src.transform.b) > 1e-12 or abs(src.transform.d) > 1e-12:
            raise ValueError(
                "This backend assumes a north-up DEM without rotation."
            )

        transformer = Transformer.from_crs(csv_crs, src.crs, always_xy=True)

        projected_samples: list[dict[str, float]] = []
        observer_points_xy: list[tuple[float, float]] = []

        for i, raw_sample in enumerate(sample_metadata, start=1):
            sample = _normalize_sample(raw_sample)

            x, y = transformer.transform(sample["lon"], sample["lat"])

            if not (
                src.bounds.left <= x <= src.bounds.right
                and src.bounds.bottom <= y <= src.bounds.top
            ):
                raise ValueError(f"Sample {i} lies outside the loaded GeoTIFF area.")

            projected_samples.append(
                {
                    "x_coord": x,
                    "y_coord": y,
                    "observer_height": sample["observer_height"],
                    "heading_deg": sample["heading_deg"],
                }
            )
            observer_points_xy.append((x, y))

        if not projected_samples:
            raise ValueError("No observer samples were provided.")

        xs = [p[0] for p in observer_points_xy]
        ys = [p[1] for p in observer_points_xy]

        left = max(src.bounds.left, min(xs) - max_distance)
        right = min(src.bounds.right, max(xs) + max_distance)
        bottom = max(src.bounds.bottom, min(ys) - max_distance)
        top = min(src.bounds.top, max(ys) + max_distance)

        if left >= right or bottom >= top:
            raise ValueError("The cropped extent is empty. Check the observer coordinates and max distance.")

        crop_window = from_bounds(left, bottom, right, top, src.transform)
        crop_window = crop_window.round_offsets().round_lengths()

        crop_width = max(1, int(crop_window.width))
        crop_height = max(1, int(crop_window.height))
        crop_transform = window_transform(crop_window, src.transform)

        frequency = np.zeros((crop_height, crop_width), dtype=np.uint32)

        # Pixel-centre coordinates for the cropped grid.
        x_coords = crop_transform.c + (np.arange(crop_width) + 0.5) * crop_transform.a
        y_coords = crop_transform.f + (np.arange(crop_height) + 0.5) * crop_transform.e
        xx, yy = np.meshgrid(x_coords, y_coords)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for i, sample in enumerate(projected_samples):
                vs_path = tmpdir / f"viewshed_{i:04d}.tif"

                run_viewshed(
                    dem_path=str(tif_path),
                    x=sample["x_coord"],
                    y=sample["y_coord"],
                    observer_h=sample["observer_height"],
                    out_tif=str(vs_path),
                    max_distance=max_distance,
                )

                with rasterio.open(vs_path) as src_vs:
                    aligned = np.zeros((crop_height, crop_width), dtype=np.uint8)
                    reproject(
                        source=rasterio.band(src_vs, 1),
                        destination=aligned,
                        src_transform=src_vs.transform,
                        src_crs=src_vs.crs,
                        dst_transform=crop_transform,
                        dst_crs=src.crs,
                        resampling=Resampling.nearest,
                        dst_nodata=0,
                    )

                # Keep directional filtering in Python so the code works with older GDAL builds.
                if fan_angle_deg < 360.0:
                    heading_deg = sample["heading_deg"]
                    bearing = (np.degrees(np.arctan2(xx - sample["x_coord"], yy - sample["y_coord"])) + 360.0) % 360.0
                    angular_diff = np.abs((bearing - heading_deg + 180.0) % 360.0 - 180.0)
                    aligned = np.where(angular_diff <= fan_angle_deg / 2.0, aligned, 0)

                frequency += (aligned > 0).astype(np.uint32)

        view_extent = (left, right, bottom, top)

        vals, freqs = np.unique(frequency[frequency > 0], return_counts=True)
        print("count frequencies:", dict(zip(vals.tolist(), freqs.tolist())))
        print("max count =", int(frequency.max()))

        return {
            "count_overlay": frequency,
            "observer_points_xy": observer_points_xy,
            "view_extent": view_extent,
        }


def aggregate_line_of_sight(*args, **kwargs):
    """
    Compatibility alias.

    The student project originally used this name for the visibility engine.
    The new implementation delegates to build_frequency_overlay().
    """
    return build_frequency_overlay(*args, **kwargs)