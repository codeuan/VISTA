# VISTA (VISTA Integrates Sightlines for Terrain Visibility)
Quantifies individual coordinates' view count across several viewpoints and produces a frequency map. Each observer is projected into the DEM coordinate system, a viewshed is computed (GDAL backend), results are aligned and optionally filtered by viewing direction and all viewsheds are aggregated into a frequency map

***

Inputs: 
- Observer metadata: manually entered in the GUI, loaded from a CSV file. Required fields: lon, lat, observer_height (m), heading_deg (0-360 egrees).
- Digital Surface Model or Digital Elevation Model: two options 1) Local DEM (tif or tiff), 2) DEM automatically downloaded from OpenTopography based on observer locations and analysis distance. This second option requires an API key (https://opentopography.org/about)

Outputs: 
- Frequency map highlighting how many times each pixel is seen across all images, according to DEM resolution.


## Key Features
- Multi-observer visibility aggregation
- Directional field-of-view support
- Interactive GUI for data input and preview
- Optional DEM download from OpenTopography
- Output as GeoTIFF + visual heatmap

## Structure
main.py           → Application entry point  
GUI.py            → Graphical user interface  
vista.py          → Orchestration and data processing  
lineofsight.py    → Visibility computation backend  
raycasting.py     → (legacy / experimental)  
randomisedirection.py → (optional / experimental)  
requirements.txt  → Python dependencies  
environment.yml   → Conda environment setup  
API_caller.py     →  DEM download from OpenTopography

## Installation (Requirements defined in environment.yml)
1. Install Miniconda:
https://docs.conda.io/en/latest/miniconda.html

2. Create the environment:
conda env create -f environment.yml

3. Set your API key as an environmental variable
For example in PowerShell:
$env:OPENTOPO_API_KEY="your_api_key_here"

3. Run the app:
conda run -n vista python main.py


## Notes & Limitations
- Requires GDAL CLI (gdal raster viewshed or gdal_raster_viewshed)
- DEM must be north-up (no rotation)
- Results depend on DEM resolution
- Large areas may result in slow downloads or processing


Authors: Connor Dean-Pijuan and Erola Fenollosa