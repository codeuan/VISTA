# optimiser.py
# Categorise a set of candidate image locations in order of how optimal they are
# for the training data.
#
# Factors considered:
#   - botanical suitability, measured using local mean NDVI
#   - visibility strength, measured using local mean visibility frequency
#   - unseenness, measured as 1 - normalised visibility strength
#   - obstacle presence, measured as local obstacle-area fraction
#
# LATER:
#   - validate Street View image quality
#   - reject blurry / dark / badly exposed / indoor images
#   - use more advanced botanical classification than NDVI alone

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from shapely.geometry import box

from ndvi import NDVI
from visibility_frequency import visibility_frequency
from obstacle_detection import fetch_obstacles_for_extent

# Optional: only needed if you want the optimiser to download Street View images.
try:
    from API_caller import download_street_view_for_samples
except ImportError:
    download_street_view_for_samples = None


@dataclass(slots=True)
class OptimiserWeights:
    """
      Controls the importance of each scoring factor.
      All component scores are normalised to roughly 0-1 before being combined.
    """
    ndvi: float = 0.40
    visibility_strength: float = 0.40
    unseenness: float = 0.00
    obstacle_penalty: float = 0.20


@dataclass(slots=True)
class CandidateScore:
    
    """
      Scorecard for each candidate point.
    
    """
    index: int
    lat: float
    lon: float
    heading_deg: float | None #note, the | means "or" for type hinting.

    mean_ndvi: float
    ndvi_score: float

    mean_visibility_count: float
    visibility_score: float
    unseenness_score: float

    occlusion_fraction: float

    final_score: float

    image_path: str | None = None
    street_view_status: str | None = None


def _default_time_range(
    time_from: str | None,
    time_to: str | None,
    days_back: int = 30,
) -> tuple[str, str]:
    """
    Build a default Sentinel-2 time range.

    If no dates are supplied, use:
        now UTC back to 30 days before now.
    """
    if time_to is None:
        time_to = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") #current date.

    if time_from is None:
        dt_to = datetime.strptime(time_to, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) #date when function was run.
        time_from = (dt_to - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ") #30 days before when the function was run.

    return time_from, time_to


def _bbox_lonlat_from_samples(
    sample_metadata: Sequence[Mapping[str, Any]],
    buffer_m: float,
) -> tuple[float, float, float, float]:
    """
    Create a WGS84 lon/lat bbox around the sample points.

    Returns:
        (min_lon, min_lat, max_lon, max_lat)
    """
    if not sample_metadata:
        raise ValueError("sample_metadata is empty.") #if no sample data given, raise an error.

    lons = [float(s["lon"]) for s in sample_metadata] #the longitude of every sample point.
    lats = [float(s["lat"]) for s in sample_metadata] #the latitude of every sample point.
 
    center_lat = float(np.mean(lats)) #rough centre of all points.

    lat_buffer_deg = buffer_m / 111_320.0 
    lon_buffer_deg = buffer_m / (
        111_320.0 * max(0.1, np.cos(np.radians(center_lat)))
    ) #as lon and lat are in degrees, convert buffer_m to degrees.

    min_lon = min(lons) - lon_buffer_deg
    max_lon = max(lons) + lon_buffer_deg
    min_lat = min(lats) - lat_buffer_deg
    max_lat = max(lats) + lat_buffer_deg #find smallest and largest coordinates, then expand to give a box around all samples.

    return min_lon, min_lat, max_lon, max_lat


def _normalise_ndvi(mean_ndvi: float) -> float:
    """
    Convert NDVI from roughly -1..1 into a 0..1 suitability score.

    High NDVI means more vegetation, so it is treated as more botanically suitable.
    """
    if not np.isfinite(mean_ndvi):
        return 0.0

    return float(np.clip((mean_ndvi + 1.0) / 2.0, 0.0, 1.0)) 


def _mean_array_value_around_projected_point(
    array: np.ndarray,
    transform,
    x: float,
    y: float,
    radius_m: float,
    ignore_nan: bool = True,
) -> float:
    """
    Calculate the mean value in a square window around a projected x/y point.

    This works for both:
        - NDVI rasters
        - visibility frequency rasters
    """
    row, col = rowcol(transform, x, y) #convert coordinate into raster grid position.

    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e) #find length of one raster pixel in real world units.

    radius_cols = max(1, int(radius_m / pixel_width))
    radius_rows = max(1, int(radius_m / pixel_height))

    row0 = max(0, row - radius_rows)
    row1 = min(array.shape[0], row + radius_rows + 1)
    col0 = max(0, col - radius_cols)
    col1 = min(array.shape[1], col + radius_cols + 1) 

    window = array[row0:row1, col0:col1] #find window around candidate point.

    if window.size == 0:
        return float("nan")

    if ignore_nan:
        finite = np.isfinite(window)
        if not finite.any():
            return float("nan")
        return float(np.nanmean(window)) #if there are NaN values, calculate the value anyway but skip those.

    return float(np.mean(window)) #calculate mean of whatever array we passed in (NDVI, visibility_frequency etc.)


def _build_obstacle_union(obstacle_result):
    """
    Merge all obstacle bounding boxes into one geometry.

    This avoids double-counting obstacle areas when boxes overlap.
    """
    bbox_gdf = obstacle_result.bbox_features

    if bbox_gdf is None or bbox_gdf.empty:
        return None

    try:
        return bbox_gdf.geometry.union_all()
    except AttributeError:
        return bbox_gdf.geometry.unary_union


def _occlusion_fraction_around_point(
    obstacle_union,
    x: float,
    y: float,
    radius_m: float,
) -> float:
    """
    Estimate how much of the candidate's local area is covered by obstacles.

    Returns:
        0.0 = no obstacle coverage
        1.0 = fully covered by obstacles
    """
    if obstacle_union is None or obstacle_union.is_empty:
        return 0.0

    candidate_area = box(
        x - radius_m,
        y - radius_m,
        x + radius_m,
        y + radius_m,
    )

    if candidate_area.area <= 0:
        return 0.0

    intersection_area = candidate_area.intersection(obstacle_union).area
    return float(np.clip(intersection_area / candidate_area.area, 0.0, 1.0))


def optimise_candidates(
    sample_metadata: Sequence[Mapping[str, Any]],
    dem_path: str | Path,
    max_distance_m: float,
    *,
    weights: OptimiserWeights | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    ndvi_radius_m: float = 50.0,
    visibility_radius_m: float = 50.0,
    obstacle_radius_m: float = 50.0,
    download_images: bool = False,
    street_view_radius_m: int = 50,
    street_view_source: str = "outdoor",
) -> list[CandidateScore]:
    """
    Rank candidate sample points by usefulness.

    Args:
        sample_metadata:
            Sequence of dictionaries containing:
                - lon
                - lat
                - observer_height
                - heading_deg

        dem_path:
            Local DEM GeoTIFF path.

        max_distance_m:
            Viewshed distance and bbox buffer distance.

        weights:
            Weighting for NDVI, visibility, unseenness, and obstacle penalty.

        time_from, time_to:
            Sentinel-2 time range. If omitted, defaults to the last 30 days.

        ndvi_radius_m:
            Local radius used when averaging NDVI around each candidate.

        visibility_radius_m:
            Local radius used when averaging visibility frequency around each candidate.

        obstacle_radius_m:
            Local radius used when calculating obstacle occlusion fraction.

        download_images:
            If True, download one Street View image per sample.

    Returns:
        CandidateScore objects sorted from best to worst.
    """
    if not sample_metadata:
        raise ValueError("sample_metadata is empty.")

    weights = weights or OptimiserWeights()
    time_from, time_to = _default_time_range(time_from, time_to)

    # ------------------------------------------------------------
    # 1. Get NDVI for the whole candidate region.
    # ------------------------------------------------------------
    bbox_lonlat = _bbox_lonlat_from_samples(sample_metadata, buffer_m=max_distance_m)

    ndvi_result = NDVI(
        bbox_lonlat=bbox_lonlat,
        time_from=time_from,
        time_to=time_to,
    )

    ndvi_array = ndvi_result["ndvi"]
    ndvi_transform = ndvi_result["transform"]
    ndvi_crs = ndvi_result["crs"]

    # ------------------------------------------------------------
    # 2. Calculate visibility frequency for the whole candidate region.
    # ------------------------------------------------------------
    visibility_result = visibility_frequency(
        sample_metadata=sample_metadata,
        dem_path=dem_path,
        max_distance_m=max_distance_m,
    )

    # IMPORTANT:
    # Your visibility_frequency.py returns "count_overlay", not "frequency".
    visibility_array = visibility_result["count_overlay"]
    visibility_transform = visibility_result["raster_transform"]
    visibility_crs = visibility_result["raster_crs"]

    left, bottom, right, top = visibility_result["raster_bounds"]

    max_visibility_count = float(np.nanmax(visibility_array))
    if max_visibility_count <= 0:
        max_visibility_count = 1.0

    # ------------------------------------------------------------
    # 3. Fetch obstacles once for the whole projected visibility area.
    # ------------------------------------------------------------
    obstacle_result = fetch_obstacles_for_extent(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        projected_crs=visibility_crs,
    )
    obstacle_union = _build_obstacle_union(obstacle_result)

    # ------------------------------------------------------------
    # 4. Optionally download Street View images.
    # ------------------------------------------------------------
    image_results_by_index: dict[int, dict[str, Any]] = {}

    if download_images:
        if download_street_view_for_samples is None:
            raise RuntimeError(
                "download_images=True, but API_caller.download_street_view_for_samples "
                "could not be imported."
            )

        street_view_results = download_street_view_for_samples(
            sample_metadata=sample_metadata,
            radius=street_view_radius_m,
            source=street_view_source,
            use_sample_heading=True,
        )

        image_results_by_index = {
            int(result["index"]): result
            for result in street_view_results
        }

    # ------------------------------------------------------------
    # 5. Score each candidate.
    # ------------------------------------------------------------
    lonlat_to_ndvi = Transformer.from_crs(
        "EPSG:4326",
        ndvi_crs,
        always_xy=True,
    )

    lonlat_to_visibility = Transformer.from_crs(
        "EPSG:4326",
        visibility_crs,
        always_xy=True,
    )

    scores: list[CandidateScore] = []

    for index, sample in enumerate(sample_metadata):
        lon = float(sample["lon"])
        lat = float(sample["lat"])

        heading_raw = sample.get("heading_deg")
        heading_deg = None if heading_raw in (None, "") else float(heading_raw)

        # NDVI local mean.
        ndvi_x, ndvi_y = lonlat_to_ndvi.transform(lon, lat)
        mean_ndvi = _mean_array_value_around_projected_point(
            array=ndvi_array,
            transform=ndvi_transform,
            x=ndvi_x,
            y=ndvi_y,
            radius_m=ndvi_radius_m,
            ignore_nan=True,
        )
        ndvi_score = _normalise_ndvi(mean_ndvi)

        # Visibility local mean.
        visibility_x, visibility_y = lonlat_to_visibility.transform(lon, lat)
        mean_visibility_count = _mean_array_value_around_projected_point(
            array=visibility_array.astype(float),
            transform=visibility_transform,
            x=visibility_x,
            y=visibility_y,
            radius_m=visibility_radius_m,
            ignore_nan=False,
        )

        if not np.isfinite(mean_visibility_count):
            mean_visibility_count = 0.0

        visibility_score = float(
            np.clip(mean_visibility_count / max_visibility_count, 0.0, 1.0)
        )

        unseenness_score = 1.0 - visibility_score

        # Obstacle local fraction.
        occlusion_fraction = _occlusion_fraction_around_point(
            obstacle_union=obstacle_union,
            x=visibility_x,
            y=visibility_y,
            radius_m=obstacle_radius_m,
        )

        final_score = (
            weights.ndvi * ndvi_score
            + weights.visibility_strength * visibility_score
            + weights.unseenness * unseenness_score
            - weights.obstacle_penalty * occlusion_fraction
        )

        image_info = image_results_by_index.get(index, {})
        metadata = image_info.get("metadata") or {}

        scores.append(
            CandidateScore(
                index=index,
                lat=lat,
                lon=lon,
                heading_deg=heading_deg,
                mean_ndvi=float(mean_ndvi) if np.isfinite(mean_ndvi) else float("nan"),
                ndvi_score=ndvi_score,
                mean_visibility_count=mean_visibility_count,
                visibility_score=visibility_score,
                unseenness_score=unseenness_score,
                occlusion_fraction=occlusion_fraction,
                final_score=float(final_score),
                image_path=image_info.get("image_path"),
                street_view_status=metadata.get("status"),
            )
        )

    scores.sort(key=lambda item: item.final_score, reverse=True)
    return scores


def scores_to_dataframe(scores: Sequence[CandidateScore]) -> pd.DataFrame:
    """
    Convert optimiser results into a DataFrame for saving or displaying.
    """
    return pd.DataFrame([asdict(score) for score in scores])
  #note to self: a dataframe is a fancy term for a spreadsheet like structure.

def save_scores_csv(
    scores: Sequence[CandidateScore],
    output_path: str | Path = "optimiser_results.csv",
) -> str:
    """
    Save ranked optimiser results as a CSV.
    """
    df = scores_to_dataframe(scores)
    output_path = str(output_path)
    df.to_csv(output_path, index=False) #save dataframe as a CSV file.
    return output_path


def load_sample_metadata_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Load candidate/sample points from a CSV.

    Expected columns:
        lon
        lat
        observer_height
        heading_deg
    """
    df = pd.read_csv(csv_path)

    required = {"lon", "lat", "observer_height", "heading_deg"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    return df.to_dict(orient="records") #convert data to list of dictionaries.


if __name__ == "__main__":
    # Example usage.
    #
    # Replace these with your real paths.
    sample_csv = "samples.csv"
    dem_path = "dem.tif"

    sample_metadata = load_sample_metadata_csv(sample_csv)

    ranked_scores = optimise_candidates(
        sample_metadata=sample_metadata,
        dem_path=dem_path,
        max_distance_m=500.0,
        weights=OptimiserWeights(
            ndvi=0.40,
            visibility_strength=0.40,
            unseenness=0.00,
            obstacle_penalty=0.20,
        ),
        download_images=False,
    )

    output_csv = save_scores_csv(ranked_scores)
    print(f"Saved optimiser results to: {output_csv}")

    print(scores_to_dataframe(ranked_scores).head(10))