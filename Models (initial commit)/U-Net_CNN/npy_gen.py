import os
import calendar
import numpy as np
import json
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    BBox,
    CRS,
)

# this file is responsible for creating dataset in npy format
# 1. LOAD CONFIG of Sentinel Hub

def load_config(json_path="config.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["sh_client_id"], data["sh_client_secret"]

try:
    CLIENT_ID, CLIENT_SECRET = load_config()
except:
    print("Make sure config.json exists with your keys!")
    exit()

config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET

OUTPUT_FOLDER = "flood_dataset_npy"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# year range
YEARS = [2022, 2023]
IMAGE_SIZE = (256, 256)

# assignment of locations
# Format: [Min_Lon, Min_Lat, Max_Lon, Max_Lat]
# we can add custom locations which is then added to the dataset folder
LOCATIONS = [
    # Brahmaputra River (North of City)
    ("Guwahati",    [91.740, 26.170, 91.760, 26.190]), 
    
    # Brahmaputra River (West of City)
    ("Dhubri",      [89.960, 26.010, 89.980, 26.030]), 
    
    # Brahmaputra River (North of City)
    ("Dibrugarh",   [94.900, 27.480, 94.920, 27.500]), 
    
    # Ganges River (North of City)
    ("Patna",       [85.120, 25.610, 85.140, 25.630]), 
    
    # Burhi Gandak River
    ("Muzaffarpur", [85.380, 26.110, 85.400, 26.130]), 
    
    # Hooghly River (Howrah Bridge Area)
    ("Kolkata",     [88.330, 22.570, 88.350, 22.590]), 
    
    # Mahanadi River (Wide section)
    ("Cuttack",     [85.870, 20.470, 85.890, 20.490]), 
    
    # Buriganga River (South of City)
    ("Dhaka",       [90.390, 23.690, 90.410, 23.710]), 
    
    # Vembanad Lake / Backwaters
    ("Kochi",       [76.250, 9.950,  76.270, 9.970]), 
    
    # Mahim Bay / Sea Link (Water Interface)
    ("Mumbai",      [72.830, 19.040, 72.850, 19.060]), 
    
    # Yamuna River (East of City Center)
    ("NewDelhi",    [77.230, 28.600, 77.250, 28.620]), 
    
    # Man Sagar Lake (Jal Mahal) - Only consistent water in Jaipur
    ("Jaipur",      [75.840, 26.950, 75.860, 26.970]), 
]

# config for sentinel hub for data extraction

evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B03", "B08", "B11", "B12"],
    output: { bands: 5, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(s) {
  let ndwi = (s.B03 - s.B08) / (s.B03 + s.B08 + 0.00001);
  return [s.B03, s.B08, s.B11, s.B12, ndwi];
}
"""


# Frtching of .npy files

def get_monthly_bbox_data(bbox_coords, year, month):
    _, last_day = calendar.monthrange(year, month)
    
    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day}"),
                mosaicking_order="leastCC"
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF)
        ],
        bbox=bbox,
        size=IMAGE_SIZE,
        config=config
    )
    
    data = request.get_data()[0]
    return data


# looping through months and years

if __name__ == "__main__":
    print(f"Starting dataset generation for years {YEARS}...")

    for year in YEARS:
        for loc_name, coords in LOCATIONS:
            print(f"\nLocation: {loc_name} ({year})")

            for month in range(1, 13):
                try:
                    img = get_monthly_bbox_data(coords, year, month)
                    img = np.nan_to_num(img, nan=0.0)
                    
                    img[..., :4] = np.clip(img[..., :4], 0, 1)
                    
                    # NDWI bands
                    img[..., 4] = (img[..., 4] + 1) / 2.0
                    
                    img = np.clip(img, 0, 1)
                    
                    filename = f"{loc_name}_{year}_{month:02d}.npy" 
                    filepath = os.path.join(OUTPUT_FOLDER, filename) 
                    np.save(filepath, img.astype(np.float32))

                    print(f"  ✓ Saved {filename}")

                except Exception as e:
                    print(f"  ✗ Failed month {month}: {e}")

    print("\nDone! All files saved inside flood_dataset_npy/")