# visibility_frequency.py

# Run with:
#           & "C:\Users\zool2620\AppData\Local\miniconda3\Scripts\conda.exe" run -n vista python visibility_frequency.py
# This script reads observer locations and metadata from a CSV, computes viewsheds using GDAL's command-line tool, and aggregates the results into a visibility frequency raster. It then saves the aggregated raster as a GeoTIFF and creates a PNG preview with a colorbar and scale bar.

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, transform as window_transform, bounds as window_bounds
from rasterio.warp import reproject


CSV_PATH = "Libro1.csv"
DEM_PATH = "SZ49se_FZ_DSM_1m.tif"

OUT_TIF = "visibility_frequency_cropped.tif"
OUT_PNG = "visibility_frequency_cropped.png"

# Your CSV lon/lat are assumed to be WGS84.
CSV_CRS = "EPSG:4326"

# Crop and analysis distance in DEM units (meters if your DEM CRS is meters).
MAX_DISTANCE_M = 500.0

# Field of view around heading_deg.
# Use 360 for no directional masking.
FIELD_OF_VIEW_DEG = 120.0


def find_gdal_viewshed_command() -> list[str]:
    if shutil.which("gdal"):
        return ["gdal", "raster", "viewshed"]
    if shutil.which("gdal_raster_viewshed"):
        return ["gdal_raster_viewshed"]
    raise RuntimeError("Could not find GDAL viewshed command in PATH.")


def run_viewshed(
    dem_path: str,
    x: float,
    y: float,
    observer_h: float,
    out_tif: str,
    max_distance: float,
) -> None:
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
            f"GDAL viewshed failed\n"
            f"Command: {' '.join(full_cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def nice_scale_length(width_m: float) -> float:
    raw = width_m / 5.0
    if raw <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(raw))
    for m in [1, 2, 5]:
        if raw <= m * exp:
            return float(m * exp)
    return float(10 * exp)


def add_scale_bar(ax, length_m: float) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    x = x0 + (x1 - x0) * 0.06
    y = y0 + (y1 - y0) * 0.06

    ax.plot([x, x + length_m], [y, y], linewidth=4, color="black")
    label = f"{int(length_m)} m" if length_m < 1000 else f"{length_m / 1000:.1f} km"
    ax.text(
        x + length_m / 2.0,
        y + (y1 - y0) * 0.02,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
    )


def main() -> None:
    csv_path = Path(CSV_PATH)
    dem_path = Path(DEM_PATH)

    df = pd.read_csv(csv_path)
    required = {"lon", "lat", "observer_height", "heading_deg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise ValueError("DEM has no CRS.")
        if not dem.crs.is_projected:
            print("Warning: DEM CRS is not projected. A scale bar in meters may be inaccurate.")

        if abs(dem.transform.b) > 1e-12 or abs(dem.transform.d) > 1e-12:
            raise ValueError("This script assumes a north-up DEM without rotation.")

        transformer = Transformer.from_crs(CSV_CRS, dem.crs, always_xy=True)

        pts = []
        for _, row in df.iterrows():
            x, y = transformer.transform(float(row["lon"]), float(row["lat"]))
            pts.append(
                (
                    x,
                    y,
                    float(row["observer_height"]),
                    float(row["heading_deg"]),
                )
            )

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        left = min(xs) - MAX_DISTANCE_M
        right = max(xs) + MAX_DISTANCE_M
        bottom = min(ys) - MAX_DISTANCE_M
        top = max(ys) + MAX_DISTANCE_M

        # Crop window on the DEM grid.
        crop_window = from_bounds(left, bottom, right, top, dem.transform)
        crop_window = crop_window.round_offsets().round_lengths()

        crop_transform = window_transform(crop_window, dem.transform)
        crop_width = max(1, int(crop_window.width))
        crop_height = max(1, int(crop_window.height))

        # Pixel-center coordinates for the cropped raster grid.
        x_coords = crop_transform.c + (np.arange(crop_width) + 0.5) * crop_transform.a
        y_coords = crop_transform.f + (np.arange(crop_height) + 0.5) * crop_transform.e
        xx, yy = np.meshgrid(x_coords, y_coords)

        frequency = np.zeros((crop_height, crop_width), dtype=np.uint32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for i, (x, y, observer_h, heading_deg) in enumerate(pts):
                vs_path = tmpdir / f"viewshed_{i:04d}.tif"

                run_viewshed(
                    str(dem_path),
                    x,
                    y,
                    observer_h,
                    str(vs_path),
                    MAX_DISTANCE_M,
                )

                with rasterio.open(vs_path) as src:
                    aligned = np.zeros((crop_height, crop_width), dtype=np.uint8)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=aligned,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=crop_transform,
                        dst_crs=dem.crs,
                        resampling=Resampling.nearest,
                        dst_nodata=0,
                    )

                # Apply heading-based field of view in Python.
                if FIELD_OF_VIEW_DEG < 360.0:
                    bearing = (np.degrees(np.arctan2(xx - x, yy - y)) + 360.0) % 360.0
                    angular_diff = np.abs((bearing - heading_deg + 180.0) % 360.0 - 180.0)
                    aligned = np.where(angular_diff <= FIELD_OF_VIEW_DEG / 2.0, aligned, 0)

                frequency += (aligned > 0).astype(np.uint32)

        print(f"Frequency raster min/max: {frequency.min()} / {frequency.max()}")

        out_profile = dem.profile.copy()
        out_profile.update(
            driver="GTiff",
            height=crop_height,
            width=crop_width,
            transform=crop_transform,
            count=1,
            dtype="uint32",
            nodata=0,
            compress="lzw",
        )

        with rasterio.open(OUT_TIF, "w", **out_profile) as dst:
            dst.write(frequency, 1)

        # PNG preview with colorbar + scale bar
        fig, ax = plt.subplots(figsize=(10, 8))
        left2, bottom2, right2, top2 = window_bounds(crop_window, dem.transform)

        im = ax.imshow(
            frequency,
            extent=(left2, right2, bottom2, top2),
            origin="upper",
            cmap="viridis",
            vmin=0,
            vmax=max(1, int(frequency.max())),
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Number of observers seeing each cell")

        ax.set_title("Visibility frequency")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        bar_length = nice_scale_length(right2 - left2)
        add_scale_bar(ax, bar_length)

        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=200)
        plt.close(fig)

    print(f"Saved GeoTIFF: {OUT_TIF}")
    print(f"Saved preview PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
