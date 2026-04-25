#ndvi.py.
#calculates the NDVI for a given region.
#makes an API call to Copernicus Sentinel to obtain B08 and B04.
#masks out bad pixels using dataMask (data/no data) and bad SCL classes.


import math
import os
from typing import Any, Tuple
import numpy as np
import requests
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile

CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

__all__ = ["NDVI"]


def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """
    Pick a local UTM EPSG code from a lon/lat point.
    """
    zone = int((lon + 180.0) // 6.0) + 1 #find time zone we are in.
    zone = max(1, min(zone, 60))
    return (32600 if lat >= 0 else 32700) + zone #ESPG code from time zone.


def _project_bbox_crs84_to_epsg(
    bbox_lonlat: Tuple[float, float, float, float],
    dst_epsg: int,
) -> Tuple[float, float, float, float]:
    """
    Project a CRS84 / lon-lat bbox into a projected CRS.
    Input bbox format: (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{dst_epsg}",
        always_xy=True,
    )

    corners = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ]
    xs, ys = zip(*(transformer.transform(x, y) for x, y in corners))
    return (min(xs), min(ys), max(xs), max(ys))


def _get_cdse_access_token(
    client_id: str,
    client_secret: str,
    timeout: int = 30,
) -> str:
    """
    Fetch an OAuth2 access token from CDSE.
    """
    response = requests.post(
        CDSE_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=timeout,
    ) #POST request to CDSE token to server.
    response.raise_for_status()

    payload = response.json() #read server response.
    access_token = payload.get("access_token") #obtain access key.
    if not access_token:
        raise RuntimeError(f"No access_token in token response: {payload}") #if none found, raise an error.
    return access_token


def _fetch_s2_ndvi_for_bbox(
    bbox_lonlat: Tuple[float, float, float, float],
    time_from: str,
    time_to: str,
    client_id: str | None = None,
    client_secret: str | None = None,
    pixel_size_m: int = 10,
    max_cloud_coverage: int = 20,
    mosaicking_order: str = "leastCC",
    timeout: int = 120,
) -> dict[str, Any]:
    """
    Fetch NDVI for a lon/lat bounding box from Sentinel-2 L2A via CDSE Process API.

    Parameters
    ----------
    bbox_lonlat
        (min_lon, min_lat, max_lon, max_lat) in EPSG:4326 / CRS84 order.
    time_from
        ISO timestamp, e.g. "2025-04-01T00:00:00Z"
    time_to
        ISO timestamp, e.g. "2025-04-30T23:59:59Z"
    client_id
        CDSE OAuth client id. Falls back to env var CDSE_CLIENT_ID.
    client_secret
        CDSE OAuth client secret. Falls back to env var CDSE_CLIENT_SECRET.
    pixel_size_m
        Output pixel size in metres. 10 m matches B04/B08 native resolution.
    max_cloud_coverage
        Tile-level cloud filter, 0..100.
    mosaicking_order
        Usually "leastCC" or "mostRecent".
    timeout
        HTTP timeout in seconds.

    Returns
    -------
    dict with:
        - ndvi: float32 array (H, W), NaN where invalid
        - valid_mask: bool array (H, W)
        - transform: affine transform
        - crs: raster CRS
        - bounds_projected: bbox used in the request
        - epsg: projected EPSG code
        - mean_ndvi, median_ndvi, min_ndvi, max_ndvi
        - profile: raster profile
    """
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("bbox_lonlat must be (min_lon, min_lat, max_lon, max_lat) with min < max.")

    client_id = client_id or os.getenv("CDSE_CLIENT_ID")
    client_secret = client_secret or os.getenv("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError(
            "Missing credentials. Pass client_id/client_secret or set "
            "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET."
        )

    # Pick a local projected CRS so resx/resy are in metres.
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    epsg = _utm_epsg_for_lonlat(center_lon, center_lat)
    bbox_proj = _project_bbox_crs84_to_epsg(bbox_lonlat, epsg)

    minx, miny, maxx, maxy = bbox_proj
    width_m = maxx - minx
    height_m = maxy - miny
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Projected bbox has non-positive width/height.")

    access_token = _get_cdse_access_token(client_id, client_secret, timeout=30)

    # Return two bands:
    #   band 1 = raw NDVI
    #   band 2 = validity mask (1 valid, 0 invalid)
    #
    # SCL classes from docs:
    # 3 cloud shadows, 6 water, 7 low-prob/unclassified cloud,
    # 8 medium cloud, 9 high cloud, 10 cirrus, 11 snow/ice.
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B04", "B08", "SCL", "dataMask"],
        output: {
          bands: 2,
          sampleType: "FLOAT32"
        }
      }; 
    } 

    function evaluatePixel(sample) {
      let ndvi = index(sample.B08, sample.B04);

      let badSCL = [3, 6, 7, 8, 9, 10, 11].includes(sample.SCL);
      let valid = (sample.dataMask === 1) && !badSCL;

      return [ndvi, valid ? 1 : 0];
    }
    """

    request_body = {
        "input": {
            "bounds": {
                "properties": {
                    "crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"
                },
                "bbox": [minx, miny, maxx, maxy],
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": time_from,
                            "to": time_to,
                        },
                        "maxCloudCoverage": max_cloud_coverage,
                        "mosaickingOrder": mosaicking_order,
                    },
                    "processing": {
                        "upsampling": "NEAREST",
                        "downsampling": "NEAREST",
                    },
                }
            ],
        },
        "output": {
            "resx": pixel_size_m,
            "resy": pixel_size_m,
        },
        "evalscript": evalscript,
    } #retrieve data into two bands, one for red/IR and another for dataMask.
      #obtain NDVI for a pixel and return its NDVI value and 1 if it is valid.
  #note that the API automatically loops over each pixel.
  
    response = requests.post(
        CDSE_PROCESS_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "image/tiff",
        },
        json=request_body,
        timeout=timeout,
    )  #send actual request to API.

    try:
        response.raise_for_status() #if a bad HTTP response is given, raise an error.
    except requests.HTTPError as exc:
        msg = response.text[:2000] if response.text else str(exc) #only obtain first 2000 characters of error message.
        raise RuntimeError(f"CDSE Process API request failed: {msg}") from exc

    with MemoryFile(response.content) as memfile: #raw bytes treated as in memory file.
        with memfile.open() as ds: 
            data = ds.read()  # shape: (2, H, W)
            profile = ds.profile.copy() #copy raster data.
            transform = ds.transform #get affine transform.
            crs = ds.crs #get coordinate reference system.

    ndvi = data[0].astype(np.float32) #retrieve NDVI band.
    valid_mask = data[1] > 0.5 
    ndvi[~valid_mask] = np.nan #replace NDVI with NaN wherever valid_mask is false.

    finite = np.isfinite(ndvi)
    if finite.any():
        stats = {
            "mean_ndvi": float(np.nanmean(ndvi)),
            "median_ndvi": float(np.nanmedian(ndvi)),
            "min_ndvi": float(np.nanmin(ndvi)),
            "max_ndvi": float(np.nanmax(ndvi)),
        }
    else:
        stats = {
            "mean_ndvi": float("nan"),
            "median_ndvi": float("nan"),
            "min_ndvi": float("nan"),
            "max_ndvi": float("nan"),
        }

    return {
        "ndvi": ndvi,
        "ndvi_array": ndvi,
        "valid_mask": valid_mask,
        "transform": transform,
        "raster_transform": transform,
        "crs": crs,
        "raster_crs": crs,
        "bounds_projected": bbox_proj,
        "raster_bounds": bbox_proj,
        "bbox_lonlat": bbox_lonlat,
        "epsg": epsg,
        "pixel_size_m": pixel_size_m,
        "profile": profile,
        **stats,
    }


def NDVI(
    bbox_lonlat: Tuple[float, float, float, float],
    time_from: str,
    time_to: str,
    client_id: str | None = None,
    client_secret: str | None = None,
    pixel_size_m: int = 10,
    max_cloud_coverage: int = 20,
    mosaicking_order: str = "leastCC",
    timeout: int = 120,
) -> dict[str, Any]:
    """
    Public entry point for NDVI retrieval.
    """
    return _fetch_s2_ndvi_for_bbox(
        bbox_lonlat=bbox_lonlat,
        time_from=time_from,
        time_to=time_to,
        client_id=client_id,
        client_secret=client_secret,
        pixel_size_m=pixel_size_m,
        max_cloud_coverage=max_cloud_coverage,
        mosaicking_order=mosaicking_order,
        timeout=timeout,
    )
