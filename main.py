#main.py
from GUI import start_gui
from vista import run_program

import shutil
if not shutil.which("gdal") and not shutil.which("gdal_raster_viewshed"):
    print("WARNING: GDAL CLI not found. Viewshed may not work.")

if __name__ == "__main__":
    start_gui(run_program)
    