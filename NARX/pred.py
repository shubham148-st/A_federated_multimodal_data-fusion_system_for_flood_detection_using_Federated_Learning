import torch
from narx import WaterNARX   


MODEL_PATH = "narx_model.pt"   
FLOOD_THRESHOLD = 6.0          



def load_model():
    model = WaterNARX(in_dim=2, hidden_dim=64, rate=1.0)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def predict_flood(avg_rain):
    model = load_model()

    # NARX inputs
    water_input = torch.tensor([[0.0]], dtype=torch.float32)  
    rain_input = torch.tensor([[avg_rain]], dtype=torch.float32)

    with torch.no_grad():
        predicted_water = model(water_input, rain_input).item()

    # classification
    flood = 1 if predicted_water >= FLOOD_THRESHOLD else 0

    return predicted_water, flood



if __name__ == "__main__":
    avg_rain = float(input("Enter average rainfall (mm): "))

    water, flood = predict_flood(avg_rain)

    print(f"\nPredicted Water Level: {water:.3f}")
    print(f"Flood Status: {'FLOOD' if flood == 1 else 'NO FLOOD'}")
