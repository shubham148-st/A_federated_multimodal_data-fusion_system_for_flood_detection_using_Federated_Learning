import os
import numpy as np

# this file is used to check the strength of the dataset created
def check_all_data():
    folder = "flood_dataset_npy"
    
    if not os.path.exists(folder):
        print(f"❌ Folder '{folder}' not found.")
        return

    # Get all files and sort them so locations appear together
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    files.sort() 

    if not files:
        print("❌ No files found! Run npy_gen.py first.")
        return

    print(f"Found {len(files)} files. Scanning ALL data...\n")
    print(f"{'FILENAME':<35} | {'WATER %':<10} |STATUS")
    print("-" * 65)

    # Counters for summary
    good_count = 0
    low_water_count = 0
    empty_count = 0

    for f in files:
        path = os.path.join(folder, f)
        
        try:
            img = np.load(path) # (256, 256, 5)
            
            # Channel 4 is NDWI (scaled 0 to 1)
            ndwi = img[..., 4] 
            
            # Water is > 0.5
            water_pixels = (ndwi > 0.5).sum()
            total_pixels = ndwi.size
            water_ratio = water_pixels / total_pixels
            
            # Determine Status
            if ndwi.max() == 0 and ndwi.min() == 0:
                status = "⚠️ EMPTY (Download Failed)"
                empty_count += 1
            elif water_ratio < 0.01:
                status = "⚠️ DRY (<1% Water)"
                low_water_count += 1
            else:
                status = "✅ GOOD"
                good_count += 1

            # Print row
            print(f"{f:<35} | {water_ratio:6.2%}   | {status}")

        except Exception as e:
            print(f"{f:<35} | ERROR      | ❌ Corrupt File")

    # Final Summary
    print("\n" + "="*30)
    print("FINAL DATASET SUMMARY")
    print("="*30)
    print(f"  ✅ Good Training Files:  {good_count}")
    print(f"  ⚠️ Too Dry (Skipped?):   {low_water_count}")
    print(f"  ❌ Empty / Failed:       {empty_count}")
    print("="*30)

    if good_count < 50:
        print("\n⚠️ WARNING: Your dataset is still very small.")
    elif low_water_count > len(files) * 0.5:
        print("\n⚠️ WARNING: Over 50% of your data is dry. Check coordinates again.")
    else:
        print("\n✅ Dataset looks healthy!")

if __name__ == "__main__":
    check_all_data()