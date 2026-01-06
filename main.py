import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.utils.data import DataLoader, TensorDataset
from models.fusion import TriModalFloodNet

# initial config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ROUNDS = 20
LOCAL_EPOCHS = 10
BATCH_SIZE = 16
LR = 0.001

CLIENTS = [
    {'id': 'satellite_img', 'rate': 0.5,  'data': 'img_only'},
    {'id': 'River_Gauge', 'rate': 0.5,  'data': 'sensor_only'},
    {'id': 'HQ_Server',   'rate': 1.0,  'data': 'full'}
]

print(f" Starting HeteroFL Flood Simulation on {DEVICE}")

# slicing
def split_model(global_model, rate):
    if rate == 1.0:
        return copy.deepcopy(global_model.state_dict())

    global_params = global_model.state_dict()
    local_params = {}

    for name, param in global_params.items():
        if 'weight' in name or 'bias' in name:
            if "time_mlp" in name:
                local_params[name] = param.clone()
                continue

            if "water_net.head" in name:
                if 'weight' in name:
                    in_dim = int(param.shape[1] * rate)
                    local_params[name] = param[:, :in_dim].clone()
                else:
                    local_params[name] = param.clone()
                continue

            if "water_proj" in name:
                if 'weight' in name:
                    out_dim = int(param.shape[0] * rate)
                    local_params[name] = param[:out_dim, :].clone()
                else:
                    out_dim = int(param.shape[0] * rate)
                    local_params[name] = param[:out_dim].clone()
                continue

            # Conv Weights
            if len(param.shape) == 4:
                out_ch = int(param.shape[0] * rate)
                in_ch = int(param.shape[1] * rate)
                if param.shape[1] == 1: in_ch = 1
                local_params[name] = param[:out_ch, :in_ch].clone()

            # Linear Weights
            elif len(param.shape) == 2:
                out_dim = int(param.shape[0] * rate)
                in_dim = int(param.shape[1] * rate)
                
                if "rain_net.proj.weight" in name: in_dim = 1 
                if "water_net.proj.weight" in name: in_dim = 20 
                if "classifier.weight" in name: out_dim = 1

                local_params[name] = param[:out_dim, :in_dim].clone()

            # Biases
            elif len(param.shape) == 1:
                out_dim = int(param.shape[0] * rate)
                if "classifier.bias" in name: out_dim = 1
                local_params[name] = param[:out_dim].clone()
        
        # Slice BN running stats
        elif 'running_mean' in name or 'running_var' in name:
            out_dim = int(param.shape[0] * rate)
            local_params[name] = param[:out_dim].clone()
            
        else:
            local_params[name] = param.clone()

    return local_params

def aggregate_models(global_model, local_updates, client_rates):
    global_state = global_model.state_dict()
    weight_sum = {k: torch.zeros_like(v, dtype=torch.float) for k, v in global_state.items()}
    count = {k: torch.zeros_like(v, dtype=torch.float) for k, v in global_state.items()}

    for local_state, rate in zip(local_updates, client_rates):
        for name, param in local_state.items():
            if name not in weight_sum: continue
            
            param_float = param.float()

            if len(param.shape) == 4:
                out_s, in_s = param.shape[0], param.shape[1]
                weight_sum[name][:out_s, :in_s] += param_float
                count[name][:out_s, :in_s] += 1
            elif len(param.shape) == 2:
                out_s, in_s = param.shape[0], param.shape[1]
                weight_sum[name][:out_s, :in_s] += param_float
                count[name][:out_s, :in_s] += 1
            elif len(param.shape) == 1:
                out_s = param.shape[0]
                weight_sum[name][:out_s] += param_float
                count[name][:out_s] += 1
            else:
                weight_sum[name] += param_float
                count[name] += 1

    for name in global_state:
        mask = count[name] > 0
        if mask.any():
            averaged_val = weight_sum[name][mask] / count[name][mask]
            global_state[name][mask] = averaged_val.type(global_state[name].dtype)

    global_model.load_state_dict(global_state)
    return global_model

# data loading
def get_dataset():
    print("dataset loading")
    try:
        df_rain = pd.read_csv("dataset/tsmixer_input.csv")
        rain_cols = [f't{i}' for i in range(10)]
        rain_data = df_rain[rain_cols].values.astype(np.float32)
        X_rain_raw = torch.tensor(rain_data).unsqueeze(-1)
    except FileNotFoundError:
        print(" Error: tsmixer_input.csv not found.")
        return None

    try:
        df_water = pd.read_csv("dataset/flood_dataset.csv")
        water_cols = [f'h{i}' for i in range(10)]
        water_data = df_water[water_cols].values.astype(np.float32)
        X_water_raw = torch.tensor(water_data).unsqueeze(-1)
        
        labels_raw = (df_water['water_level'].values > 5.0).astype(np.float32)
        Y_raw = torch.tensor(labels_raw).unsqueeze(-1)
    except FileNotFoundError:
        print(" Error: flood_dataset.csv not found.")
        return None

    npy_path = "dataset/flood_dataset_npy/*.npy"
    npy_files = glob.glob(npy_path)
    patches = []
    PATCH_SIZE = 64
    
    npy_files.sort() 

    for f in npy_files:
        try:
            data = np.load(f)
            if len(data.shape) == 3:
                if data.shape[2] == 5: data = data[:, :, 0]
                elif data.shape[0] == 5: data = data[0, :, :]
            
            if len(data.shape) == 2:
                h, w = data.shape
                center_y, center_x = h // 2, w // 2
                start_y = center_y - (PATCH_SIZE // 2)
                start_x = center_x - (PATCH_SIZE // 2)
                
                if start_y >= 0 and start_x >= 0:
                    patch = data[start_y:start_y+PATCH_SIZE, start_x:start_x+PATCH_SIZE]
                    patches.append(patch)
        except:
            pass

    if len(patches) > 0:
        X_np = np.array(patches)
        if len(X_np.shape) == 3: X_np = X_np[:, np.newaxis, :, :]
        X_img_raw = torch.tensor(X_np, dtype=torch.float32)
    else:
        X_img_raw = torch.empty(0) 

    len_rain = len(X_rain_raw)
    len_water = len(X_water_raw)
    len_img = len(X_img_raw) if len(X_img_raw) > 0 else 0
    max_len = max(len_rain, len_water, len_img)

    def pad_tensor(t, target_len, dim_shape):
        current_len = len(t)
        if current_len >= target_len:
            return t[:target_len] 
        needed = target_len - current_len
        padding = torch.zeros((needed, *dim_shape), dtype=torch.float32)
        if current_len == 0:
            return padding
        return torch.cat([t, padding], dim=0)

    X_rain = pad_tensor(X_rain_raw, max_len, (10, 1))
    X_water = pad_tensor(X_water_raw, max_len, (10, 1))
    Y = pad_tensor(Y_raw, max_len, (1,))

    if len_img > 0:
        X_img = pad_tensor(X_img_raw, max_len, (1, 64, 64))
    else:
        X_img = torch.randn(max_len, 1, 64, 64)

    print(f" Final Dataset Sizes: Img{X_img.shape}, Rain{X_rain.shape}, Water{X_water.shape}")
    return TensorDataset(X_img, X_rain, X_water, Y)

full_dataset = get_dataset()
if full_dataset:
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
else:
    print("Dataset is empty. Exiting.")
    exit()


# TRAINING LOOP

global_model = TriModalFloodNet(rate=1.0).to(DEVICE)
criterion = nn.BCELoss()

# Metrics Storage
global_history = []
client_metrics = {client['id']: {'loss': [], 'acc': []} for client in CLIENTS}

for round_idx in range(NUM_ROUNDS):
    print(f"\n--- Global Round {round_idx + 1} ---")
    local_updates = []
    active_rates = []

    for client in CLIENTS:
        cid = client['id']
        crate = client['rate']
        dtype = client['data']

        print(f"  {cid} training (Rate {crate}, Mode: {dtype})...")

        client_weights = split_model(global_model, crate)
        local_model = TriModalFloodNet(rate=crate).to(DEVICE)
        local_model.load_state_dict(client_weights)
        local_model.train()
        optimizer = optim.Adam(local_model.parameters(), lr=LR)

        # Local Training
        epoch_loss = 0
        steps = 0
        for _ in range(LOCAL_EPOCHS):
            for imgs, rain, water, labels in train_loader:
                imgs, rain, water, labels = imgs.to(DEVICE), rain.to(DEVICE), water.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                
                if dtype == 'img_only':
                    preds = local_model(imgs, None, None)
                elif dtype == 'sensor_only':
                    preds = local_model(None, rain, water)
                else:
                    preds = local_model(imgs, rain, water)

                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1

        avg_loss = epoch_loss / steps if steps > 0 else 0
        client_metrics[cid]['loss'].append(avg_loss)
        
        # Local Evaluation (Calculate Accuracy on Local Model)
        local_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, rain, water, labels in train_loader:
                imgs, rain, water, labels = imgs.to(DEVICE), rain.to(DEVICE), water.to(DEVICE), labels.to(DEVICE)
                
                if dtype == 'img_only':
                    preds = local_model(imgs, None, None)
                elif dtype == 'sensor_only':
                    preds = local_model(None, rain, water)
                else:
                    preds = local_model(imgs, rain, water)
                
                predicted = (preds > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        local_acc = 100 * correct / total if total > 0 else 0
        client_metrics[cid]['acc'].append(local_acc)
        
        local_updates.append(local_model.state_dict())
        active_rates.append(crate)
        
        print(f"      -> Loss: {avg_loss:.4f} | Acc: {local_acc:.2f}%")

    print("   Aggregating...")
    global_model = aggregate_models(global_model, local_updates, active_rates)

    # Global Evaluation
    global_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for imgs, rain, water, labels in train_loader:
            imgs, rain, water, labels = imgs.to(DEVICE), rain.to(DEVICE), water.to(DEVICE), labels.to(DEVICE)
            out = global_model(imgs, rain, water)
            predicted = (out > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        acc = 100 * correct / total
        global_history.append(acc)
        print(f"   Global Accuracy: {acc:.2f}%")

torch.save(global_model.state_dict(), "flood_model.pth")
print("Model weights saved to 'flood_model.pth'")

# graphs
rounds = range(1, NUM_ROUNDS + 1)

# Plot 1: Global Accuracy
plt.figure(figsize=(10, 6))
plt.plot(rounds, global_history, marker='o', linestyle='-', color='black', linewidth=2, label='Global Model')
plt.title('Global Model Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('global_accuracy_graph.png')
print("Graph saved as 'global_accuracy_graph.png'")

# Plot 2: Per Client Accuracy
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
for i, client in enumerate(CLIENTS):
    cid = client['id']
    plt.plot(rounds, client_metrics[cid]['acc'], marker='s', linestyle='--', color=colors[i], label=f'{cid} Acc')

plt.title('Local Client Accuracy per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('client_accuracy_graph.png')
print("Graph saved as 'client_accuracy_graph.png'")

# Plot 3: Per Client Loss
plt.figure(figsize=(10, 6))
for i, client in enumerate(CLIENTS):
    cid = client['id']
    plt.plot(rounds, client_metrics[cid]['loss'], marker='x', linestyle=':', color=colors[i], label=f'{cid} Loss')

plt.title('Local Client Loss per Round')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('client_loss_graph.png')
print("Graph saved as 'client_loss_graph.png'")