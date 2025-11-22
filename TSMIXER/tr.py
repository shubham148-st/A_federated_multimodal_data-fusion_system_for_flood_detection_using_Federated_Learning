import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import the model
from tsmixer import TSMixerClassifier

def prepare_data(csv_file="tsmixer_input.csv", batch_size=32):
    """
    Loads input CSV and generates random binary labels for demonstration.
    """
    print(f"Loading data from {csv_file}...")
    
    # 1. Load Inputs
    if not os.path.exists(csv_file):
        print("CSV not found, generating new random data...")
        # Generate 100 samples for training
        data = torch.randn(100, 10) 
        df = pd.DataFrame(data.numpy(), columns=[f"t{i}" for i in range(10)])
        df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)
    
    # Convert to Tensor: [Batch, Seq_Len=10]
    inputs = torch.tensor(df.values, dtype=torch.float32)
    
    # Reshape for TSMixer: [Batch, Seq_Len, Features=1]
    inputs = inputs.unsqueeze(-1)
    
    # 2. Generate Random Labels (Binary: 0 or 1)
    # In a real scenario, you would load these from a 'targets.csv'
    labels = torch.randint(0, 2, (inputs.size(0), 1)).float()
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader

def train_model():
    # Settings
    EPOCHS = 50
    LR = 0.01
    
    # Initialize
    model = TSMixerClassifier(seq_len=10, rate=1.0)
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    dataloader = prepare_data()
    
    # History for plotting
    loss_history = []
    acc_history = []
    
    print("\n--- Starting Training ---")
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            # Forward
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            
        # Averages
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = correct / total
        
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

    # Save Model
    torch.save(model.state_dict(), "tsmixer_model.pth")
    print("\nModel saved to 'tsmixer_model.pth'")
    
    # --- Plotting Graphs ---
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Train Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(acc_history, label='Train Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_accuracy_graph.png')
    print("Graphs saved to 'loss_accuracy_graph.png'")

if __name__ == "__main__":
    train_model()