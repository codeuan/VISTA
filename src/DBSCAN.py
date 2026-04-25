#DBSCAN.py
#Uses the DBSCAN algorithm to cluster points, so that points within 2 visibility radii of each other are grouped.
#Input dataset is split into chunks which are independently analysed, reducing cost of querying a given crop.

from sklearn.cluster import DBSCAN
from pyproj import CRS, Transformer
import numpy as np


CSV_CRS = "EPSG:4326"


def choose_projected_crs(sample_metadata):
    center_lon = float(np.mean([float(s["lon"]) for s in sample_metadata]))
    center_lat = float(np.mean([float(s["lat"]) for s in sample_metadata]))

    zone = int((center_lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone

    return CRS.from_epsg(epsg)


def DBSCAN(sample_metadata, max_distance_m):
    """
    Split observer points into local chunks.

    Returns:
        list[list[dict]]
    """

    projected_crs = choose_projected_crs(sample_metadata)

    transformer = Transformer.from_crs(
        CSV_CRS,
        projected_crs,
        always_xy=True,
    )

    coords = []

    for row in sample_metadata:
        x, y = transformer.transform(
            float(row["lon"]),
            float(row["lat"]),
        )
        coords.append((x, y))

    coords = np.array(coords)

    clustering = DBSCAN(
        eps=2 * max_distance_m,
        min_samples=1,
    ).fit(coords)

    labels = clustering.labels_

    chunks = []

    for label in sorted(set(labels)):
        chunk = [
            row
            for row, row_label in zip(sample_metadata, labels)
            if row_label == label
        ]

        chunks.append(chunk)

    return chunks
