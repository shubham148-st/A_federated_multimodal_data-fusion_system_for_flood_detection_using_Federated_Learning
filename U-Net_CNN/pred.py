import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt


#  this file contains the prediction model to predict if its a flood or its safe
# Same as trainning 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class FloodUNet(nn.Module):
    def __init__(self, n_channels=5, n_classes=1):
        super(FloodUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2 = DoubleConv(64, 32)
        self.up3 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv3 = DoubleConv(32, 16)

        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        return self.outc(x)

# Predict

def predict_flood_status(npy_path, model_path="flood_unet.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(npy_path):
        print(f" File not found: {npy_path}")
        return

    print(f"\nAnalyzing: {os.path.basename(npy_path)}")
    model = FloodUNet(n_channels=5, n_classes=1).to(device)
    try:
        # loadign pre trained model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() 
    except FileNotFoundError:
        print("Error: 'flood_unet.pth' not found")
        return

    raw_img = np.load(npy_path)       
    # Shape: (256, 256, 5)
    
    # Transpose to (5, 256, 256) for PyTorch
    img_tensor = torch.from_numpy(raw_img.transpose(2, 0, 1)).float()
    # Add batch dimension: (1, 5, 256, 256)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Run Inference 
    with torch.no_grad():
        logits = model(img_tensor)     
        probs = torch.sigmoid(logits)  
        
        # binary mask (1=Water, 0=Land)
        pred_mask = (probs > 0.5).float().cpu().squeeze().numpy()

    # Calculate Flood Statistics
    total_pixels = pred_mask.size
    water_pixels = pred_mask.sum()
    flood_ratio = water_pixels / total_pixels

    print(f"  Flood Ratio: {flood_ratio:.2%}")
    
    if flood_ratio > 0.05:
        print("->FLOODED")
    else:
        print("->SAFE")
    



# Testing
if __name__ == "__main__":
    # 1. Test a known FLOOD case (Patna Monsoon)
    predict_flood_status("flood_dataset_npy/Patna_2023_09.npy")

    # 2. Test a known SAFE case (Patna Dry Season)
    predict_flood_status("flood_dataset_npy/Patna_2022_04.npy")