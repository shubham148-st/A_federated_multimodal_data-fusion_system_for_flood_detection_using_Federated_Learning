import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# this file tells the average loss and pixel accuracy of the trained model
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


# DATASET LOADER

class FloodDataset(Dataset):
    def __init__(self, folder_path):
        self.files = [os.path.join(folder_path, f)
                      for f in os.listdir(folder_path) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx]) # (256, 256, 5)
        img = img.transpose(2, 0, 1)   # (5, 256, 256)
        x = torch.from_numpy(img).float()

        # Ground Truth: NDWI > 0.5 is Water
        ndwi = x[4]
        y = (ndwi > 0.5).float().unsqueeze(0) # (1, 256, 256)
        return x, y


def evaluate_model(model_path="flood_unet.pth", data_folder="flood_dataset_npy"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Evaluation on {device} ---")

    # A. Check Files
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found.")
        return
    if not os.path.exists(data_folder):
        print("❌ Error: Dataset folder not found.")
        return

    # B. Load Data
    dataset = FloodDataset(data_folder)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(f"Loaded {len(dataset)} images for testing.")

    # C. Load Model
    model = FloodUNet(n_channels=5, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Important: Turns off training specific layers like Dropout

    # D. Metrics
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct_pixels = 0
    total_pixels = 0

    # E. Testing Loop
    with torch.no_grad(): # No gradients needed for evaluation
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward Pass
            logits = model(x)
            loss = criterion(logits, y)

            # Predictions (Sigmoid > 0.5)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Calculate Stats
            total_loss += loss.item()
            
            # Compare prediction to ground truth (y)
            correct = (preds == y).sum().item()
            total_correct_pixels += correct
            total_pixels += torch.numel(preds)

    # F. Final Calculations
    avg_loss = total_loss / len(loader)
    accuracy = total_correct_pixels / total_pixels
    accuracy_percent = accuracy * 100

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"📉 Average Loss:      {avg_loss:.4f} (Lower is better)")
    print(f"🎯 Pixel Accuracy:    {accuracy_percent:.2f}% (Higher is better)")
    print("="*30)

if __name__ == "__main__":
    evaluate_model()