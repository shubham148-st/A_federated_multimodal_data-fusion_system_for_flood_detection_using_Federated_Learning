import torch
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


try:
    from models.fusion import TriModalFloodNet
except ImportError:
    from models.fusion import TriModalFloodNet

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "flood_model.pth" # the main.py would create this folder
BATCH_SIZE = 16
NPY_PATH = "test_dataset/flood_dataset_npy/*.npy" 

print(f"Starting Inference on {DEVICE}")

# Data Loading
def get_dataset():
    print("📂 Loading Real Data for Testing...")

    # 1. Load Rain
    if not os.path.exists("test_dataset/tsmixer_input.csv"):
        print("Error: tsmixer_input.csv not found.")
        return None
    
    df_rain = pd.read_csv("test_dataset/tsmixer_input.csv")
    rain_cols = [f't{i}' for i in range(10)]
    rain_data = df_rain[rain_cols].values.astype(np.float32)
    X_rain = torch.tensor(rain_data).unsqueeze(-1)

    # 2. Load Water
    if not os.path.exists("test_dataset/flood_dataset.csv"):
        print("Error: flood_dataset.csv not found.")
        return None

    df_water = pd.read_csv("test_dataset/flood_dataset.csv")
    water_cols = [f'h{i}' for i in range(10)]
    water_data = df_water[water_cols].values.astype(np.float32)
    X_water = torch.tensor(water_data).unsqueeze(-1)
    
    # Labels: Flood if water_level > 5.0
    labels = (df_water['water_level'].values > 5.0).astype(np.float32)
    Y = torch.tensor(labels).unsqueeze(-1)

    # 3. Load Images
    npy_files = glob.glob(NPY_PATH)
    patches = []
    PATCH_SIZE = 64
    
    print(f"   🔎 Found {len(npy_files)} image files.")

    for f in npy_files:
        try:
            data = np.load(f)
            # Handle 5-channel images (H, W, 5) -> Take 1st channel -> (H, W)
            if len(data.shape) == 3:
                if data.shape[2] == 5: 
                    data = data[:, :, 0]
                elif data.shape[0] == 5: 
                    data = data[0, :, :]
            
            # Now data should be 2D
            if len(data.shape) == 2:
                h, w = data.shape
                # Extract patches to match the method used in training
                for r in range(0, h - PATCH_SIZE + 1, 64):
                    for c in range(0, w - PATCH_SIZE + 1, 64):
                        patch = data[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
                        if np.mean(patch) > 0.001: # Skip empty patches
                            patches.append(patch)
        except:
            pass

    if len(patches) > 0:
        X_np = np.array(patches)
        # Add Channel Dim -> (N, 1, 64, 64)
        if len(X_np.shape) == 3: 
            X_np = X_np[:, np.newaxis, :, :]
        X_img = torch.tensor(X_np, dtype=torch.float32)
        print(f"   -> Extracted {len(X_img)} valid image patches.")
    else:
        # Fallback to noise if path is wrong, just to allow script to run
        print("    No valid images found. Using noise for testing.")
        X_img = torch.randn(len(Y), 1, PATCH_SIZE, PATCH_SIZE)

    # 4. Alignment
    min_len = min(len(X_rain), len(X_water), len(X_img))
    print(f"   Testing on {min_len} aligned samples.")
    
    return TensorDataset(X_img[:min_len], X_rain[:min_len], X_water[:min_len], Y[:min_len])

# testing
def test_model():
    # 1. Load Data
    dataset = get_dataset()
    if not dataset: return
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load Model
    print(f"    Loading model from {MODEL_PATH}...")
    try:
        model = TriModalFloodNet(rate=1.0).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'flood_model.pth' is in this directory.")
        return

    all_preds = []
    all_labels = []

    # 3. Prediction Loop
    print("   Running Predictions...")
    with torch.no_grad():
        for imgs, rain, water, labels in test_loader:
            imgs, rain, water = imgs.to(DEVICE), rain.to(DEVICE), water.to(DEVICE)
            
            # Forward pass
            outputs = model(imgs, rain, water)
            predicted = (outputs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Evaluation Metrics
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc*100:.2f}%")

    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=['No Flood', 'Flood']))

    # 5. Graph: Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Flood'], yticklabels=['Normal', 'Flood'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png')
    print(" Saved 'test_confusion_matrix.png'")

    # 6. Graph: Prediction Timeline
    # Shows the first 100 data points to visualize how well predictions track reality
    plt.figure(figsize=(12, 5))
    limit = min(100, len(all_labels))
    
    plt.plot(range(limit), all_labels[:limit], label='Actual Ground Truth', color='black', linewidth=2, alpha=0.6)
    plt.scatter(range(limit), all_preds[:limit], label='Model Prediction', color='red', marker='x', s=60)
    
    plt.yticks([0, 1], ['Normal (0)', 'Flood (1)'])
    plt.title(f'Flood Prediction Timeline (First {limit} Samples)')
    plt.xlabel('Time Step')
    plt.ylabel('Status')
    plt.legend(loc='center right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('test_prediction_timeline.png')
    print(" Saved 'test_prediction_timeline.png'")

if __name__ == "__main__":
    test_model()