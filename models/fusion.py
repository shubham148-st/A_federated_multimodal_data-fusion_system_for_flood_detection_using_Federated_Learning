import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import DoubleConv
from .tsmixer import RainTSMixer
from .narx import WaterNARX

class TriModalFloodNet(nn.Module):
    def __init__(self, rate=1.0):
        super().__init__()
        self.rate = rate

        # FIX 1: Pass 1 input channel (for grayscale/NDWI) and scaled output channels
        self.cnn = DoubleConv(1, int(32 * rate))
        
        # Helper to flatten image (Batch, Ch, 64, 64) -> (Batch, Ch)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.rain_net = RainTSMixer(1, 10, 16, rate)
        
        # FIX 2: Input dim is 20 (10 water history + 10 rain history from your CSVs)
        self.water_net = WaterNARX(20, 64, rate)
        self.water_proj = nn.Linear(1, int(16 * rate))

        fusion = int(32 * rate) + int(16 * rate) + int(16 * rate)
        self.classifier = nn.Linear(fusion, 1)

    def forward(self, img, rain, water):
        # Determine batch size and device from whatever input is available
        if img is not None:
            batch = img.size(0)
            device = img.device
        elif rain is not None:
            batch = rain.size(0)
            device = rain.device
        elif water is not None:
            batch = water.size(0)
            device = water.device
        else:
            raise ValueError("Model called with all inputs None")

        # Image branch
        if img is not None:
            x = self.cnn(img)       # (Batch, 32*rate, 64, 64)
            x = self.global_pool(x) # (Batch, 32*rate, 1, 1)
            x_img = x.flatten(1)    # (Batch, 32*rate)
        else:
            x_img = torch.zeros(batch, int(32 * self.rate), device=device)

        # Rain branch
        if rain is not None:
            x_rain = self.rain_net(rain)
        else:
            x_rain = torch.zeros(batch, int(16 * self.rate), device=device)

        # Water branch
        if water is not None and rain is not None:
            narx_out = self.water_net(water, rain)
            x_water = F.relu(self.water_proj(narx_out))
        else:
            x_water = torch.zeros(batch, int(16 * self.rate), device=device)

        # Fusion
        fused = torch.cat([x_img, x_rain, x_water], dim=1)
        return torch.sigmoid(self.classifier(fused))