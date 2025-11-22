import torch
import pandas as pd
from tsmixer import TSMixerClassifier

def predict(csv_path="tsmixer_input.csv", model_path="tsmixer_model.pth"):
    print("\n--- Running Prediction ---")
    
    # 1. Load Model
    model = TSMixerClassifier(seq_len=10, rate=1.0)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file not found. Run main_train.py first.")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Prepare Input Tensor [Batch, 10, 1]
    input_tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(-1)
    
    # 3. Inference
    with torch.no_grad():
        predictions = model(input_tensor)
        
    # 4. Display Results
    # Convert probabilities to binary class (0 or 1)
    predicted_labels = (predictions > 0.5).int().numpy().flatten()
    raw_probs = predictions.numpy().flatten()
    
    # Save results to new CSV
    results_df = df.copy()
    results_df['flood_probability'] = raw_probs
    results_df['prediction'] = predicted_labels
    
    results_df.to_csv("prediction_results.csv", index=False)
    
    print(f"Predictions generated for {len(df)} samples.")
    print("First 5 Predictions:")
    print(results_df[['flood_probability', 'prediction']].head())
    print("\nFull results saved to 'prediction_results.csv'")

if __name__ == "__main__":
    predict()