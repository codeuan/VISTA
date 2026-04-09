from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import time

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.plot import show
from rasterio.windows import from_bounds, transform as window_transform, bounds as window_bounds
from rasterio.warp import reproject
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


# Keep these as simple module-level defaults for now.
CSV_CRS = "EPSG:4326"
MAX_DISTANCE_M = 500.0
FIELD_OF_VIEW_DEG = 90
OUT_PNG = Path("visibility_frequency_cropped.png")


def find_gdal_viewshed_command() -> list[str]:
    """
    Return the best available GDAL viewshed command for this environment.
    """
    if shutil.which("gdal"):
        return ["gdal", "raster", "viewshed"]
    if shutil.which("gdal_raster_viewshed"):
        return ["gdal_raster_viewshed"]
    raise RuntimeError(
    "GDAL viewshed command not found.\n\n"
    "Make sure you are running inside the correct conda environment:\n"
    "    conda env create -f environment.yml\n"
    "    conda run -n vista python main.py\n\n"
    "If the problem persists, reinstall GDAL:\n"
    "    conda install -c conda-forge gdal"
)


if not shutil.which("gdal") and not shutil.which("gdal_raster_viewshed"):
    raise RuntimeError(
        "GDAL CLI not found.\n"
        "Make sure you created the conda environment:\n"
        "conda env create -f environment.yml"
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
    Run one binary viewshed for a single observer point.

    The observer position is passed through --pos as X,Y,H.
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


from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from matplotlib import cm
import matplotlib.patches as mpatches

def save_preview_png(
    frequency: np.ndarray,
    crop_window,
    src_transform,
    observer_points_xy,
    out_png: Path = OUT_PNG,
) -> None:
    left, bottom, right, top = window_bounds(crop_window, src_transform)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask zero values (no visibility)
    masked = np.ma.masked_where(frequency == 0, frequency)

    # Plasma colormap with grey for zero visibility
    cmap = cm.get_cmap("plasma").copy()
    cmap.set_bad(color="lightgrey")

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

    # Plot observer locations
    if observer_points_xy:
        xs = [x for x, _ in observer_points_xy]
        ys = [y for _, y in observer_points_xy]
        ax.scatter(
            xs,
            ys,
            marker="x",
            s=80,
            linewidths=2,
            color="white",
            zorder=30,
            label="Observer locations",
        )

        # Optional labels: 1, 2, 3...
        for i, (x, y) in enumerate(observer_points_xy, start=1):
            ax.text(
                x,
                y,
                str(i),
                color="white",
                fontsize=9,
                ha="left",
                va="bottom",
                zorder=31,
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )

    # Scale bar
    bar_length = nice_scale_length(right - left)
    add_scale_bar(ax, bar_length)

    # Legend for zero visibility and observer points
    legend_handles = [
        mpatches.Patch(color="lightgrey", label="No visibility")
    ]
    if observer_points_xy:
        legend_handles.append(
            mpatches.Patch(color="white", label="Observer locations")
        )
    ax.legend(handles=legend_handles, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def run_program(sample_metadata, tif_path, max_distance, ax=None, show_reference=False):
    """
    Read observer metadata, compute a visibility frequency map, and return:

    - count_overlay: 2D numpy array with the frequency count
    - observer_points_xy: list of projected observer coordinates
    - view_extent: (left, right, bottom, top)

    The ax/show_reference arguments are kept for compatibility with the
    student GUI structure.
    """
    tif_path = Path(tif_path)

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError("The DEM has no CRS.")

        affine = src.transform
        dem = src.read(1)

        transformer = Transformer.from_crs(CSV_CRS, src.crs, always_xy=True)

        projected_samples = []
        observer_points_xy = []

        for i, sample in enumerate(sample_metadata, start=1):
            lon = float(sample["lon"])
            lat = float(sample["lat"])

            observer_height = float(
                sample.get("observer_height", sample.get("elevation_m", 0.0))
            )
            heading_deg = float(sample.get("heading_deg", 0.0)) % 360.0

            x, y = transformer.transform(lon, lat)

            if not (
                src.bounds.left <= x <= src.bounds.right
                and src.bounds.bottom <= y <= src.bounds.top
            ):
                raise ValueError(f"Sample {i} lies outside the loaded GeoTIFF area.")

            projected_samples.append(
                {
                    "x_coord": x,
                    "y_coord": y,
                    "heading_deg": heading_deg,
                    "observer_height": observer_height,
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
            raise ValueError("The cropped extent is empty. Check the observer coordinates.")

        crop_window = from_bounds(left, bottom, right, top, affine)
        crop_window = crop_window.round_offsets().round_lengths()

        crop_width = max(1, int(crop_window.width))
        crop_height = max(1, int(crop_window.height))
        crop_transform = window_transform(crop_window, affine)

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
                    max_distance=MAX_DISTANCE_M,
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

                # Keep the directional cone in Python so this works on older GDAL builds.
                if FIELD_OF_VIEW_DEG < 360.0:
                    heading_deg = sample["heading_deg"]
                    bearing = (np.degrees(np.arctan2(xx - sample["x_coord"], yy - sample["y_coord"])) + 360.0) % 360.0
                    angular_diff = np.abs((bearing - heading_deg + 180.0) % 360.0 - 180.0)
                    aligned = np.where(angular_diff <= FIELD_OF_VIEW_DEG / 2.0, aligned, 0)

                frequency += (aligned > 0).astype(np.uint32)

        view_extent = (left, right, bottom, top)

        # Optional lightweight preview if an axes object is provided.
        if ax is not None:
            ax.clear()
            show(dem, transform=affine, ax=ax, cmap="terrain")
            overlay = np.ma.masked_where(frequency == 0, frequency)
            show(overlay, transform=crop_transform, ax=ax, cmap="viridis", alpha=0.65)
            ax.scatter(xs, ys, marker="x", s=60, linewidths=2, color="white", zorder=30)
            ax.set_title("Frequency count heatmap")
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")

            if show_reference:
                ax.set_xlim(left, right)
                ax.set_ylim(bottom, top)

        vals, freqs = np.unique(frequency[frequency > 0], return_counts=True)
        print("count frequencies:", dict(zip(vals.tolist(), freqs.tolist())))
        print("max count =", int(frequency.max()))

        save_preview_png(frequency, crop_window, src.transform, observer_points_xy, OUT_PNG)

        out_tif = f"visibility_{int(time.time())}.tif"

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=frequency.shape[0],
            width=frequency.shape[1],
            transform=crop_transform,
            count=1,
            dtype="uint32",
            nodata=0,
            compress="lzw",
        )

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(frequency, 1)
  

        return {
            "count_overlay": frequency,
            "observer_points_xy": observer_points_xy,
            "view_extent": view_extent,
            "preview_png_path": str(OUT_PNG),
            "output_tif_path": out_tif,
        }