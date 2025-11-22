import torch
import pandas as pd

def generate_positive_rain_data(filename="tsmixer_input.csv", num_samples=100):
    print(f"Generating positive rainfall data for {filename}...")
    
    # Dimensions: [Batch, Seq_Len=10]
    # 1. Generate Random Normal Data
    raw_data = torch.randn(num_samples, 10)
    
    # 2. Apply Absolute Value to make it positive (Rain cannot be negative)
    # Multiplied by 2.5 to simulate realistic variance (e.g., 0mm to ~8mm)
    rain_data = torch.abs(raw_data) * 2.5
    
    # Optional: Round to 2 decimal places like real sensors
    rain_data = torch.round(rain_data * 100) / 100
    
    # Create DataFrame
    cols = [f"t{i}" for i in range(10)]
    df = pd.DataFrame(rain_data.numpy(), columns=cols)
    
    # Save
    df.to_csv(filename, index=False)
    
    print(f"Success! Generated {num_samples} samples.")
    print("First 5 rows (Notice all values are positive):")
    print(df.head())

if __name__ == "__main__":
    generate_positive_rain_data()