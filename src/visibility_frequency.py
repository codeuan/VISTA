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
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, transform as window_transform, bounds as window_bounds
from rasterio.warp import reproject
import os 
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent #resolve project folder.

CSV_PATH = PROJECT_ROOT / "GSV" / "Libro3.csv" #CSV where metadata is obtained from (will need upgrading to API call when possible).
DEM_PATH = PROJECT_ROOT / "GeoTIFF" / "sicily_cop30_utm33.tif" #GeoTIFF raster file.

OUT_TIF = "visibility_frequency_cropped.tif" #result as stored in a raster file.
OUT_PNG = "visibility_frequency_cropped.png" #result as stored in a png file.

CSV_CRS = "EPSG:4326" #coordinate system (lon/lat).

MAX_DISTANCE_M = 5000.0 #maximum visibility distance.

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

def process_one_point(  
    i: int, 
    dem_path: str,  
    x: float, 
    y: float,
    observer_h: float, 
    heading_deg: float,  
    tmpdir: str,
    crop_height: int, 
    crop_width: int, 
    crop_transform,  
    dem_crs,  
    xx: np.ndarray,  
    yy: np.ndarray, 
) -> np.ndarray: 
    vs_path = Path(tmpdir) / f"viewshed_{i:04d}.tif"  #create a temporary file for the output TIFF.

    run_viewshed( 
        dem_path,
        x,
        y,
        observer_h,
        str(vs_path),
        MAX_DISTANCE_M,
    ) #GDAL binary viewshed computation.

    with rasterio.open(vs_path) as src:  # open the TIFF GDAL just made
        aligned = np.zeros((crop_height, crop_width), dtype=np.uint8)  # make an empty array the size of the final cropped grid
        reproject(  
            source=rasterio.band(src, 1),  
            destination=aligned, 
            src_transform=src.transform, 
            src_crs=src.crs,  
            dst_transform=crop_transform,  
            dst_crs=dem_crs,
            resampling=Resampling.nearest, 
            dst_nodata=0, 
        ) #reproject array so it fits master grid exactly.

    if FIELD_OF_VIEW_DEG < 360.0: 
        bearing = (np.degrees(np.arctan2(xx - x, yy - y)) + 360.0) % 360.0 
        angular_diff = np.abs((bearing - heading_deg + 180.0) % 360.0 - 180.0) 
        aligned = np.where(angular_diff <= FIELD_OF_VIEW_DEG / 2.0, aligned, 0) 
    return (aligned > 0).astype(np.uint32) 

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

        with tempfile.TemporaryDirectory() as tmpdir:  #make a temporary folder to hold all every per-point TIFF file.
            tmpdir = Path(tmpdir)  #create Path object so folder is easier to reference.
            total_points = len(pts)  #total number of observer points.

            max_workers = min(4, os.cpu_count() or 1)  #run 4 CPU workers.
            print(f"Using {max_workers} workers", flush=True)  #print number of workers (for admin purposes).

            with ThreadPoolExecutor(max_workers=max_workers) as executor: #create threadpool.
                future_to_index = {}  #dictionary to store which future belongs to which coordinates.
                for i, (x, y, observer_h, heading_deg) in enumerate(pts, start=1):
                    future = executor.submit(  
                        process_one_point, 
                        i,  
                        str(dem_path),  
                        x, 
                        y,  
                        observer_h,  
                        heading_deg,  
                        str(tmpdir), 
                        crop_height, 
                        crop_width,  
                        crop_transform, 
                        dem.crs, 
                        xx, 
                        yy, 
                    ) #submit job to worker pool and have future returned.
                    future_to_index[future] = i #store coordinates pertaining to future.

                for done_count, future in enumerate(as_completed(future_to_index), start=1):  #as each job is finished, 
                    point_index = future_to_index[future] #retrieve coordinates of future.
                    print(
                        f"Finished actual point {point_index} ({done_count} out of {total_points} completed)",
                        flush=True,
                    )  #print progress.

                    frequency += future.result()  #add result to map.

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
