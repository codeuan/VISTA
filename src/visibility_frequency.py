# visibility_frequency.py
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
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, transform as window_transform, bounds as window_bounds
import os 
from concurrent.futures import ThreadPoolExecutor, as_completed
import time as time_module
from rasterio.transform import rowcol
from pyproj import CRS, Transformer
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import rowcol
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

CSV_CRS = "EPSG:4326" #coordinate system (lon/lat).

FIELD_OF_VIEW_DEG = 120.0 #field of view in degrees.


def find_gdal_viewshed_command() -> list[str]: #function to find the correct GDAL viewshed command to use for the built in viewshed functions.
    if shutil.which("gdal"): #looks for a gdal command in the system PATH.
        return ["gdal", "raster", "viewshed"] #if so, return command pieces to call viewshed tool.
    if shutil.which("gdal_raster_viewshed"): #alternative PATH name.
        return ["gdal_raster_viewshed"] #alternative command pieces.
    raise RuntimeError("Could not find GDAL viewshed command in PATH.") #if none found, raise an error.


def run_viewshed(
    dem_path: str,
    x: float,
    y: float,
    observer_h: float,
    out_tif: str,
    max_distance: float,
) -> None:
    cmd = find_gdal_viewshed_command() #line of sight function.

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
    ] #terminal command acting as instruction packet for GDAL.

    result = subprocess.run(full_cmd, capture_output=True, text=True) #execute command in terminal and capture what is printed as text.
    if result.returncode != 0: #if an error occurred.
        raise RuntimeError(
            f"GDAL viewshed failed\n"
            f"Command: {' '.join(full_cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        ) #print error message.




def nice_scale_length(width_m: float) -> float:
    raw = width_m / 5.0 #attempt to split display into 5 chunbks.
    if raw <= 0:
        return 1.0 #if 0 or a negative number returned, default to 1.
    exp = 10 ** math.floor(math.log10(raw)) #find largest power of 10 below raw.
    for m in [1, 2, 5]: #try different multipliers.
        if raw <= m * exp:
            return float(m * exp) #if the multiplier works, use it.
    return float(10 * exp) #else, default to a multiplier of 10.


def add_scale_bar(ax, length_m: float) -> None:
    x0, x1 = ax.get_xlim() #retrieve x limits of axes.
    y0, y1 = ax.get_ylim() #retrieve y limits of axes.

    x = x0 + (x1 - x0) * 0.07 
    y = y0 + (y1 - y0) * 0.07 #place bar 7% up and to the right from the bottom left corner.

    ax.plot([x, x + length_m], [y, y], linewidth=4, color="black") #draw a horizontal line.
    label = f"{int(length_m)} m" if length_m < 1000 else f"{length_m / 1000:.1f} km" #label line in metres if below 1km or else kilometres.
    ax.text(
        x + length_m / 2.0,
        y + (y1 - y0) * 0.02,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
    ) #styling for label.

def choose_projected_crs(sample_metadata):
    center_lon = float(np.mean([float(s["lon"]) for s in sample_metadata]))
    center_lat = float(np.mean([float(s["lat"]) for s in sample_metadata]))

    zone = int((center_lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def reproject_dem_to_crs(src, dst_crs):
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        transform=dst_transform,
        crs=dst_crs,
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
    )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    tmp_path = Path(tmp.name)
    tmp.close()

    with rasterio.open(tmp_path, "w", **profile) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return tmp_path, dst_transform, dst_crs



def process_one_point(
    i,
    dem_path,
    x,
    y,
    observer_h,
    heading_deg,
    tmpdir,
    crop_height,
    crop_width,
    crop_transform,
    dem_crs,
    max_distance_m,
):
    # make a square boundary around this observer
    local_left = x - max_distance_m
    local_right = x + max_distance_m
    local_bottom = y - max_distance_m
    local_top = y + max_distance_m

    local_dem_path = Path(tmpdir) / f"local_dem_{i:04d}.tif"
    vs_path = Path(tmpdir) / f"viewshed_{i:04d}.tif"

    # crop the DEM BEFORE running viewshed
    with rasterio.open(dem_path) as dem_src:
        dem_left, dem_bottom, dem_right, dem_top = dem_src.bounds

        # clip local bounds so they stay inside the DEM
        local_left = max(local_left, dem_left)
        local_right = min(local_right, dem_right)
        local_bottom = max(local_bottom, dem_bottom)
        local_top = min(local_top, dem_top)

        local_window = from_bounds(
            local_left,
            local_bottom,
            local_right,
            local_top,
            dem_src.transform,
        )
        local_window = local_window.round_offsets().round_lengths()

        local_h = int(local_window.height)
        local_w = int(local_window.width)

        if local_h <= 0 or local_w <= 0:
            return 0, 0, 0, 0, np.zeros((0, 0), dtype=np.uint32)

        local_transform = window_transform(local_window, dem_src.transform)

        local_data = dem_src.read(1, window=local_window)

        local_profile = dem_src.profile.copy()
        local_profile.update(
            driver="GTiff",
            height=local_h,
            width=local_w,
            transform=local_transform,
            count=1,
            compress="lzw",
        )

        with rasterio.open(local_dem_path, "w", **local_profile) as dst:
            dst.write(local_data, 1)

    # NOW run viewshed on the cropped local DEM
    run_viewshed(
        str(local_dem_path),
        x,
        y,
        observer_h,
        str(vs_path),
        max_distance_m,
    )

    # read the small viewshed directly
    with rasterio.open(vs_path) as src:
        aligned = src.read(1)

    # work out where this small local result belongs in the master crop
    first_cell_x = local_transform.c + 0.5 * local_transform.a
    first_cell_y = local_transform.f + 0.5 * local_transform.e
    row0, col0 = rowcol(crop_transform, first_cell_x, first_cell_y)

    row1 = row0 + local_h
    col1 = col0 + local_w

    # safety clipping in case an edge observer lands slightly outside
    if row0 < 0 or col0 < 0 or row1 > crop_height or col1 > crop_width:
        trim_top = max(0, -row0)
        trim_left = max(0, -col0)
        trim_bottom = max(0, row1 - crop_height)
        trim_right = max(0, col1 - crop_width)

        aligned = aligned[
            trim_top: local_h - trim_bottom,
            trim_left: local_w - trim_right,
        ]

        row0 = max(0, row0)
        col0 = max(0, col0)
        row1 = min(crop_height, row1)
        col1 = min(crop_width, col1)

    # local FOV mask only on the small local array
    if FIELD_OF_VIEW_DEG < 360.0 and aligned.size > 0:
        local_h2, local_w2 = aligned.shape

        x_coords = local_transform.c + (np.arange(local_w2) + 0.5) * local_transform.a
        y_coords = local_transform.f + (np.arange(local_h2) + 0.5) * local_transform.e
        xx, yy = np.meshgrid(x_coords, y_coords)

        bearing = (np.degrees(np.arctan2(xx - x, yy - y)) + 360.0) % 360.0
        angular_diff = np.abs((bearing - heading_deg + 180.0) % 360.0 - 180.0)

        aligned = np.where(angular_diff <= FIELD_OF_VIEW_DEG / 2.0, aligned, 0)

    return row0, row1, col0, col1, (aligned > 0).astype(np.uint32)


def save_preview_png(frequency, crop_window, src_transform, observer_points_xy, out_png):
    left, bottom, right, top = window_bounds(crop_window, src_transform)

    fig, ax = plt.subplots(figsize=(10, 8))

    masked = np.ma.masked_where(frequency == 0, frequency)  # hide all 0 cells from the normal colour scale

    cmap = plt.cm.viridis.copy()      # make a copy so we do not alter the global viridis map
    cmap.set_bad(color="lightgrey")   # masked values (your 0 cells) will appear grey

    im = ax.imshow(
        masked,
        extent=(left, right, bottom, top),
        origin="upper",
        cmap=cmap,
        vmin=1,
        vmax=max(1, int(frequency.max())),
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of observers seeing each cell")

    ax.set_title("Visibility frequency")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if observer_points_xy:
        xs = [x for x, _ in observer_points_xy]
        ys = [y for _, y in observer_points_xy]
        ax.scatter(xs, ys, marker="x", s=80, linewidths=2, color="white", zorder=30)

    bar_length = nice_scale_length(right - left)
    add_scale_bar(ax, bar_length)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def visibility_frequency(sample_metadata, dem_path, max_distance_m):
    dem_path = Path(dem_path)
    projected_dem_path = None

    try:
        with rasterio.open(dem_path) as dem_src:
            if dem_src.crs is None:
                raise ValueError("DEM has no CRS.")
            if abs(dem_src.transform.b) > 1e-12 or abs(dem_src.transform.d) > 1e-12:
                raise ValueError("This script assumes a north-up DEM without rotation.")

            target_crs = choose_projected_crs(sample_metadata)
            projected_dem_path, dem_transform, dem_crs = reproject_dem_to_crs(dem_src, target_crs)
            dem_profile = dem_src.profile.copy()

            transformer = Transformer.from_crs(CSV_CRS, dem_crs, always_xy=True)

            pts = []
            for row in sample_metadata:
                x, y = transformer.transform(float(row["lon"]), float(row["lat"]))
                pts.append((x, y, float(row["observer_height"]), float(row["heading_deg"])))

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            left = min(xs) - max_distance_m
            right = max(xs) + max_distance_m
            bottom = min(ys) - max_distance_m
            top = max(ys) + max_distance_m

            crop_window = from_bounds(left, bottom, right, top, dem_transform)
            crop_window = crop_window.round_offsets().round_lengths()
            crop_transform = window_transform(crop_window, dem_transform)
            crop_width = max(1, int(crop_window.width))
            crop_height = max(1, int(crop_window.height))

            x_coords = crop_transform.c + (np.arange(crop_width) + 0.5) * crop_transform.a
            y_coords = crop_transform.f + (np.arange(crop_height) + 0.5) * crop_transform.e
            xx, yy = np.meshgrid(x_coords, y_coords)

            frequency = np.zeros((crop_height, crop_width), dtype=np.uint32)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                total_points = len(pts)

                max_workers = min(4, os.cpu_count() or 1)
                print(f"Using {max_workers} workers", flush=True)

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {}
                    for i, (x, y, observer_h, heading_deg) in enumerate(pts, start=1):
                        future = executor.submit(
                            process_one_point,
                            i,
                            str(projected_dem_path),
                            x,
                            y,
                            observer_h,
                            heading_deg,
                            str(tmpdir),
                            crop_height,
                            crop_width,
                            crop_transform,
                            dem_crs,
                            max_distance_m,
                        )
                        future_to_index[future] = i

                    for done_count, future in enumerate(as_completed(future_to_index), start=1):
                        point_index = future_to_index[future]
                        print(
                            f"Finished actual point {point_index} ({done_count} out of {total_points} completed)",
                            flush=True,
                        )
                        row0, row1, col0, col1, local_result = future.result()
                        frequency[row0:row1, col0:col1] += local_result

            print(f"Frequency raster min/max: {frequency.min()} / {frequency.max()}")

            timestamp = time_module.strftime("%Y%m%d_%H%M%S")
            out_tif = f"visibility_frequency_{timestamp}.tif"
            out_png = f"visibility_frequency_{timestamp}.png"

            save_preview_png(
                frequency,
                crop_window,
                dem_transform,
                [(x, y) for x, y, _, _ in pts],
                out_png,
            )

            out_profile = dem_profile.copy()
            out_profile.update(
                driver="GTiff",
                height=crop_height,
                width=crop_width,
                transform=crop_transform,
                crs=dem_crs,
                count=1,
                dtype="uint32",
                nodata=0,
                compress="lzw",
            )

            with rasterio.open(out_tif, "w", **out_profile) as dst:
                dst.write(frequency, 1)

            left2, bottom2, right2, top2 = window_bounds(crop_window, dem_transform)

            print(f"Saved GeoTIFF: {out_tif}")
            print(f"Saved PNG: {out_png}")

        return {
            "count_overlay": frequency,
            "frequency": frequency,
            "raster_transform": crop_transform,
            "raster_crs": dem_crs,
            "raster_bounds": (left2, bottom2, right2, top2),
            "observer_points_xy": [(x, y) for x, y, _, _ in pts],
            "view_extent": (left2, right2, bottom2, top2),
            "scale_bar_length_m": nice_scale_length(right2 - left2),
            "output_tif_path": out_tif,
            "preview_png_path": out_png,
        }

    finally:
        if projected_dem_path is not None:
            projected_dem_path.unlink(missing_ok=True)
