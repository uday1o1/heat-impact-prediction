import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x):
        # x: (B, L, F)
        out, _ = self.lstm(x)        # (B, L, H)
        last = out[:, -1, :]         # (B, H)
        logits = self.head(last)     # (B, out_dim)
        return logits
