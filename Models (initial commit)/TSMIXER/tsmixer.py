import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Original TSMixer Components ---

class Scaler(nn.Module):
    def __init__(self, rate): 
        super().__init__()
        self.rate = rate
    def forward(self, x):
        return x / self.rate if self.training else x

class HeteroMixerLayer(nn.Module):
    def __init__(self, seq_len, num_features, rate):
        super().__init__()
        scaled_feat = num_features

        self.time_norm = nn.BatchNorm1d(scaled_feat, track_running_stats=False)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Linear(seq_len, seq_len)
        )

        self.feat_norm = nn.BatchNorm1d(scaled_feat, track_running_stats=False)
        self.feat_mlp = nn.Sequential(
            nn.Linear(scaled_feat, scaled_feat),
            nn.ReLU(),
            nn.Linear(scaled_feat, scaled_feat)
        )

        self.scaler = Scaler(rate)

    def forward(self, x):
        x_norm = self.time_norm(x)
        x = x + self.scaler(self.time_mlp(x_norm))

        x_norm = self.feat_norm(x)
        x_perm = x_norm.permute(0, 2, 1)
        x_mix = self.feat_mlp(x_perm).permute(0, 2, 1)
        return x + self.scaler(x_mix)

class RainTSMixer(nn.Module):
    def __init__(self, input_features, seq_len, out_dim, rate):
        super().__init__()
        scaled_in = int(input_features * rate) if input_features > 1 else 1
        hidden = int(32 * rate)

        self.proj = nn.Linear(scaled_in, hidden)
        self.l1 = HeteroMixerLayer(seq_len, hidden, rate)
        self.l2 = HeteroMixerLayer(seq_len, hidden, rate)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, int(out_dim * rate))
        self.scaler = Scaler(rate)

    def forward(self, rain):
        x = self.proj(rain)
        x = x.permute(0, 2, 1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.gap(x).squeeze(-1)
        return self.scaler(self.head(x))

# --- Standalone Wrapper for Training ---

class TSMixerClassifier(nn.Module):
    """
    Wraps RainTSMixer to output a single probability (0-1) 
    for binary classification (e.g., Flood vs No Flood).
    """
    def __init__(self, seq_len=10, rate=1.0):
        super().__init__()
        # Internal TSMixer outputs feature vector of size 16 (based on original fusion.py)
        self.mixer = RainTSMixer(input_features=1, seq_len=seq_len, out_dim=16, rate=rate)
        
        # Classification head
        mixer_out_dim = int(16 * rate)
        self.final_head = nn.Linear(mixer_out_dim, 1)
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, 1]
        feat = self.mixer(x)
        logits = self.final_head(feat)
        return torch.sigmoid(logits)