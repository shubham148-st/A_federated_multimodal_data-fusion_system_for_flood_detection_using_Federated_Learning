import pandas as pd
import numpy as np
import torch

def generate_batch_data_with_avg(filename="flood_dataset.csv", num_samples=1000):
    print(f"Generating {num_samples} samples...")

    # Columns h0 to h9 are for rainfall
    raw_data = torch.randn(num_samples, 10)
    
    rain_batch = torch.abs(raw_data) * 3.0
    rain_numpy = rain_batch.numpy()

    # calculates the avg of the previous columns
    rain_averages = np.mean(rain_numpy, axis=1)

    # Simulate Physics to get the "Target" Water Level

    water_targets = []
    
    for i in range(num_samples):
        sequence = rain_numpy[i]
        
        # dry
        current_water = 0.0 
        for rain_hour in sequence:
            
            current_water = (current_water * 0.9) + (rain_hour * 0.3)
        
        water_targets.append(current_water)
    
    water_targets = np.array(water_targets)

    
    cols = [f"h{i}" for i in range(10)]
    df = pd.DataFrame(rain_numpy, columns=cols)
    

    df['average'] = rain_averages
    
    
    df['water_level'] = water_targets

    # 5. Save
    df.to_csv(filename, index=False)
    print(f"Success! Data saved to {filename}")
    print("-" * 30)
    print("First 5 rows:")
    print(df.head())
    print("-" * 30)
    print("Columns generated:", list(df.columns))

if __name__ == "__main__":
    generate_batch_data_with_avg()