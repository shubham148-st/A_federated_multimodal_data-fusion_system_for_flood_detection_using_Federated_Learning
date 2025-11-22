import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Scaler(nn.Module):
    def __init__(self, rate): 
        super().__init__()
        self.rate = rate
    def forward(self, x):
        return x / self.rate if self.training else x

class WaterNARX(nn.Module):
    def __init__(self, in_dim, hidden_dim, rate):
        super().__init__()
        hid = int(hidden_dim * rate)

        self.proj = nn.Linear(in_dim, hid)
        self.h1 = nn.Linear(hid, hid)
        self.bn1 = nn.BatchNorm1d(hid)
        self.h2 = nn.Linear(hid, hid)
        self.bn2 = nn.BatchNorm1d(hid)

        self.head = nn.Linear(hid, 1)
        self.scaler = Scaler(rate)

    def forward(self, water, rain):
        b = water.size(0)
        w = water.view(b, -1)
        r = rain.view(b, -1)

        # Concatenate Water and Rain inputs
        cat_inputs = torch.cat((w, r), dim=1)
        
        x = F.relu(self.proj(cat_inputs))
        x = self.scaler(x)
        x = self.scaler(F.relu(self.bn1(self.h1(x))))
        x = self.scaler(F.relu(self.bn2(self.h2(x))))
        return self.head(x)

if __name__ == "__main__":

    FILENAME = "flood_dataset.csv"
    HIDDEN_DIM = 64
    RATE = 1.0
    EPOCHS = 200
    LR = 0.01

    # loading
    print(f"Loading {FILENAME}...")
    try:
        df = pd.read_csv(FILENAME)
    except FileNotFoundError:
        print(f"Error: {FILENAME} not found. Please run the generator script first.")
        exit()

    # --- Prepare Inputs for NARX ---
    # NARX requires two inputs: 
    # 1. Water (Past State)
    # 2. Rain (Exogenous Input)
    
    # Rain Input: The 'average' column from your generated batch data
    rain_input = torch.tensor(df['average'].values, dtype=torch.float32).unsqueeze(1)
    
    # Water Input: Since these are independent batch events starting from scratch,
    # the "past water" is 0.0 for all samples.
    water_input = torch.zeros(len(df), 1, dtype=torch.float32)
    
    # Target: The final calculated 'water_level'
    target = torch.tensor(df['water_level'].values, dtype=torch.float32).unsqueeze(1)

    # --- Initialize Model ---
    # in_dim calculation: 
    # Water input (1 value) + Rain input (1 value) = 2
    IN_DIM = 2 
    
    model = WaterNARX(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, rate=RATE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Model Initialized. Input Dim: {IN_DIM} (1 Water + 1 Rain Avg)")
    print("Starting Training...")

    # --- Training Loop ---
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Forward pass (Exactly as defined in narx.py)
        prediction = model(water_input, rain_input)
        
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item():.6f}")
    torch.save(model.state_dict(), "narx_model.pt")
    print("Model saved to narx_model.pt")

    # --- Evaluation ---
    print("Training Complete. Visualizing results...")
    model.eval()
    with torch.no_grad():
        preds = model(water_input, rain_input)

    # Sort data for cleaner plotting
    sorted_indices = torch.argsort(rain_input.flatten())
    sorted_rain = rain_input[sorted_indices]
    sorted_target = target[sorted_indices]
    sorted_preds = preds[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_rain, sorted_target, label='Actual Physics (Target)', color='blue', alpha=0.3)
    plt.plot(sorted_rain, sorted_preds, label='NARX Prediction', color='red', linewidth=2)
    
    plt.title("NARX Result: Predicting Flood from Avg Rain")
    plt.xlabel("Average Rainfall (mm)")
    plt.ylabel("Final Water Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("narx_avg_result.png")
    print("Result saved as 'narx_avg_result.png'")
    plt.show()