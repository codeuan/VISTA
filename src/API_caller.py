import os
import math
import tempfile
from pathlib import Path
from typing import Sequence, Mapping, Any

import requests


OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")


def _bbox_from_samples(
    sample_metadata: Sequence[Mapping[str, Any]],
    buffer_m: float,
) -> tuple[float, float, float, float]:
    """
    Compute a WGS84 bounding box around lon/lat points with a buffer in metres.

    Args:
        sample_metadata:
            A sequence of dicts, each containing "lon" and "lat" keys.
        buffer_m:
            Buffer distance in metres to add around the points.

    Returns:
        (south, north, west, east)
    """
    if not sample_metadata:
        raise ValueError("sample_metadata is empty.")

    lons = [float(s["lon"]) for s in sample_metadata]
    lats = [float(s["lat"]) for s in sample_metadata]

    center_lat = sum(lats) / len(lats)
    lat_buffer_deg = buffer_m / 111_320.0
    lon_buffer_deg = buffer_m / (111_320.0 * max(0.1, math.cos(math.radians(center_lat))))

    south = min(lats) - lat_buffer_deg
    north = max(lats) + lat_buffer_deg
    west = min(lons) - lon_buffer_deg
    east = max(lons) + lon_buffer_deg

    return south, north, west, east


def _download_binary_to_tempfile(content: bytes, suffix: str) -> str:
    """
    Write binary content to a temporary file and return the path.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


# -------------------------------------------------------------------
# OpenTopography DEM functions
# -------------------------------------------------------------------

def download_dem_from_opentopo(
    south: float,
    north: float,
    west: float,
    east: float,
    demtype: str = "COP30",
) -> str:
    """
    Download a DEM GeoTIFF from OpenTopography and return the local file path.
    """
    if not OPENTOPO_API_KEY:
        raise RuntimeError("OPENTOPO_API_KEY is not set.")

    url = "https://portal.opentopography.org/API/globaldem"

    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": OPENTOPO_API_KEY,
    }

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "html" in content_type:
        raise RuntimeError(
            "OpenTopography returned HTML instead of a GeoTIFF. "
            "Check your parameters and API key."
        )

    return _download_binary_to_tempfile(response.content, suffix=".tif")


def download_dem_for_samples(
    sample_metadata: Sequence[Mapping[str, Any]],
    max_distance_m: float,
    demtype: str = "COP30",
) -> str:
    """
    Compute a bounding box around sample points and download a DEM for that area.
    """
    south, north, west, east = _bbox_from_samples(sample_metadata, max_distance_m)
    return download_dem_from_opentopo(
        south=south,
        north=north,
        west=west,
        east=east,
        demtype=demtype,
    )


# -------------------------------------------------------------------
# Google Street View functions
# -------------------------------------------------------------------

def get_street_view_metadata(
    lat: float,
    lon: float,
    radius: int = 50,
    source: str = "default",
) -> dict:
    """
    Query Google Street View metadata for the panorama nearest to (lat, lon).

    Args:
        lat:
            Latitude in WGS84.
        lon:
            Longitude in WGS84.
        radius:
            Search radius in metres.
        source:
            "default" or "outdoor".

    Returns:
        Metadata JSON dict from Google.

    Raises:
        RuntimeError:
            If API key is missing or Google returns an error response.
    """
    if not GOOGLE_MAPS_API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set.")

    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "source": source,
        "key": GOOGLE_MAPS_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    # Expected statuses include OK, ZERO_RESULTS, NOT_FOUND, etc.
    status = data.get("status")
    if status not in {"OK", "ZERO_RESULTS", "NOT_FOUND"}:
        raise RuntimeError(f"Street View metadata request failed: {data}")

    return data


def download_street_view_image(
    lat: float,
    lon: float,
    size: tuple[int, int] = (640, 640),
    heading: float | None = None,
    pitch: float = 0.0,
    fov: float = 90.0,
    radius: int = 50,
    source: str = "default",
    check_metadata_first: bool = True,
) -> str | None:
    """
    Download a Google Street View image near (lat, lon) and return the local JPG path.

    Args:
        lat:
            Latitude in WGS84.
        lon:
            Longitude in WGS84.
        size:
            (width, height) in pixels.
        heading:
            Camera heading in degrees. If None, Google chooses automatically.
        pitch:
            Camera pitch in degrees.
        fov:
            Horizontal field of view in degrees.
        radius:
            Search radius in metres.
        source:
            "default" or "outdoor".
        check_metadata_first:
            If True, first query metadata and return None when no panorama exists.

    Returns:
        Path to downloaded JPG, or None if no Street View panorama was found.

    Raises:
        RuntimeError:
            If API key is missing or Google returns a real error.
    """
    if not GOOGLE_MAPS_API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set.")

    if check_metadata_first:
        metadata = get_street_view_metadata(
            lat=lat,
            lon=lon,
            radius=radius,
            source=source,
        )
        if metadata.get("status") != "OK":
            return None

    width, height = size
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": f"{width}x{height}",
        "location": f"{lat},{lon}",
        "pitch": pitch,
        "fov": fov,
        "radius": radius,
        "source": source,
        "return_error_code": "true",
        "key": GOOGLE_MAPS_API_KEY,
    }

    if heading is not None:
        params["heading"] = heading

    response = requests.get(url, params=params, timeout=60)

    # With return_error_code=true, Google can return 404 when no image exists.
    if response.status_code == 404:
        return None

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "image" not in content_type:
        raise RuntimeError(
            f"Street View did not return an image. "
            f"Content-Type was: {content_type!r}"
        )

    return _download_binary_to_tempfile(response.content, suffix=".jpg")


def download_street_view_for_samples(
    sample_metadata: Sequence[Mapping[str, Any]],
    size: tuple[int, int] = (640, 640),
    pitch: float = 0.0,
    fov: float = 90.0,
    radius: int = 50,
    source: str = "default",
    use_sample_heading: bool = True,
) -> list[dict]:
    """
    Download one Street View image per sample.

    Expected sample fields:
        - lon
        - lat
        - optionally heading_deg

    Returns:
        A list of dicts, one per sample, for example:
        [
            {
                "index": 0,
                "lat": ...,
                "lon": ...,
                "heading": ...,
                "image_path": ".../tmpxxxx.jpg",
                "metadata": {...}
            },
            ...
        ]

        If no panorama exists near a sample, image_path will be None.
    """
    results = []

    for i, sample in enumerate(sample_metadata):
        lat = float(sample["lat"])
        lon = float(sample["lon"])

        heading = None
        if use_sample_heading and "heading_deg" in sample and sample["heading_deg"] not in (None, ""):
            heading = float(sample["heading_deg"])

        try:
            metadata = get_street_view_metadata(
                lat=lat,
                lon=lon,
                radius=radius,
                source=source,
            )

            if metadata.get("status") == "OK":
                image_path = download_street_view_image(
                    lat=lat,
                    lon=lon,
                    size=size,
                    heading=heading,
                    pitch=pitch,
                    fov=fov,
                    radius=radius,
                    source=source,
                    check_metadata_first=False,
                )
            else:
                image_path = None

            error = None

        except Exception as e:
            metadata = {
                "status": "ERROR",
                "error": str(e),
            }
            image_path = None
            error = str(e)

        results.append({
            "index": i,
            "lat": lat,
            "lon": lon,
            "heading": heading,
            "image_path": image_path,
            "metadata": metadata,
            "street_view_status": metadata.get("status"),
            "error": error,
        })

    return results