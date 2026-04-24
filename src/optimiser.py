#optimiser.py
#categorise a set of images in order of how optimal they are for the training data.
#factors considered: mean visibility strength, botanical suitabity, presence of obstacles.
#LATER: valdiate image quality.


#for every candidate point.



#calculate mean NDVI.
#calculate mean "unseenness".
#calculate mean visibility strength.
from ndvi import NDVI
from datetime import datetime,timezone, timedelta
from visibility_frequency import visibility_frequency
import numpy as np

if time_to is None:
  time_to = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") #current UTC time when program was run.
 
if time_from is None:
  dt_to = datetime.strptime(time_to, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) #day when program was run.
  time_from = (dt_to - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ") #the day 30 days before program was run.


ndvi_result = NDVI(
    bbox_lonlat=(min_lon, min_lat, max_lon, max_lat),
    time_from=time_from,
    time_to=time_to,
)

visibility_strength_result = visibility_frequency(sample_metadata, dem_path, max_distance_m)


mean_visibility_strength = visibility_strength_result["frequency"].mean()


occlusion_fraction = 

final_score = (
    a * mean_ndvi
    + b * mean_visibility_strength
    - c * occlusion_fraction
)
