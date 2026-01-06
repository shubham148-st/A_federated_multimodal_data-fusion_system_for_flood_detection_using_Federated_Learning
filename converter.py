import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your .npy files
INPUT_PATH = "dataset/flood_dataset_npy/*.npy" 

# Folder to save the converted PNGs
OUTPUT_DIR = "converted_pngs"

# ==========================================
# CONVERSION SCRIPT
# ==========================================
def normalize_to_image(data):
    """
    Normalizes a 2D array to 0-255 uint8 range for image saving.
    """
    # Handle NaN or Infinite values
    data = np.nan_to_num(data)
    
    # Normalize to 0-1
    if np.max(data) - np.min(data) != 0:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Scale to 0-255
    return (data * 255).astype(np.uint8)

def convert_npy_to_png():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created output directory: {OUTPUT_DIR}")

    files = glob.glob(INPUT_PATH)
    print(f"🔎 Found {len(files)} .npy files.")

    for f in files:
        try:
            filename = os.path.basename(f).replace('.npy', '.png')
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            # Load Data
            data = np.load(f)
            
            # --- SHAPE HANDLING (Matches your test.py logic) ---
            # If 3D array (Channels, Height, Width) or (Height, Width, Channels)
            if len(data.shape) == 3:
                # Assuming shape (H, W, C) -> Take 1st channel
                if data.shape[2] == 5: 
                    data = data[:, :, 0]
                # Assuming shape (C, H, W) -> Take 1st channel
                elif data.shape[0] == 5: 
                    data = data[0, :, :]
                # Fallback: Just take the first slice if structure is unknown
                else:
                    data = data[0, :, :] if data.shape[0] < data.shape[2] else data[:, :, 0]

            # Ensure it is now 2D
            if len(data.shape) != 2:
                print(f"⚠️ Skipping {filename}: Could not reduce shape {data.shape} to 2D image.")
                continue
            
            # Normalize and Save
            img_data = normalize_to_image(data)
            image = Image.fromarray(img_data)
            
            # Save as Grayscale PNG
            image.save(save_path)
            print(f"✅ Saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Error processing {f}: {e}")

if __name__ == "__main__":
    convert_npy_to_png()