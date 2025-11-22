import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random


# this file is for trainning purpose
# DATASET

class FloodDataset(Dataset):
    def __init__(self, folder_path, augment=False):
        self.files = [os.path.join(folder_path, f)
                      for f in os.listdir(folder_path) if f.endswith(".npy")]
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx]) # (256, 256, 5)
        img = img.transpose(2, 0, 1)   # (5, 256, 256)
        x = torch.from_numpy(img).float()

        # Ground Truth: NDWI > 0.5 is Water
        ndwi = x[4]
        y = (ndwi > 0.5).float().unsqueeze(0) 

        if self.augment:
            if random.random() > 0.5:
                x = torch.flip(x, [-1])
                y = torch.flip(y, [-1])
            if random.random() > 0.5:
                x = torch.flip(x, [-2])
                y = torch.flip(y, [-2])
            k = random.randint(0, 3)
            x = torch.rot90(x, k, [-2, -1])
            y = torch.rot90(y, k, [-2, -1])

        return x, y


# 2. U-NET MODEL
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


# 3. METRICS & UTILS

def calculate_accuracy(loader, model, device):
    model.eval()
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct_pixels += (preds == y).sum().item()
            total_pixels += torch.numel(preds)
            
    model.train()
    return (correct_pixels / total_pixels) * 100


# 4. TRAINING LOOP
def train_model():
    DATA_FOLDER = "flood_dataset_npy"
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: {DATA_FOLDER} not found.")
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 30
    
    dataset = FloodDataset(DATA_FOLDER, augment=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = FloodUNet(n_channels=5, n_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    
    loss_history = []
    accuracy_history = []

    print(f"--- Training on {DEVICE} with {len(dataset)} images ---")
    print(f"{'Epoch':<6} | {'Loss':<10} | {'Accuracy':<10}")
    print("-" * 35)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Metrics
        avg_loss = total_loss / len(loader)
        accuracy = calculate_accuracy(loader, model, DEVICE)
        
        # Store history
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
        
        print(f"{epoch+1:<6} | {avg_loss:.4f}     | {accuracy:.2f}%")

    # Save Model
    torch.save(model.state_dict(), "flood_unet.pth")
    print("\nModel saved as flood_unet.pth")

    # graph ploting 
    
    epochs_range = range(1, EPOCHS + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss_history, 'r-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracy_history, 'b-o', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_plot.png')
    print("Graph saved as training_plot.png 📈")

if __name__ == "__main__":
    train_model()