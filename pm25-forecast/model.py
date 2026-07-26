"""GRU forecaster for PM2.5 24 hours ahead."""

import torch
import torch.nn as nn


class PM25Forecaster(nn.Module):
    """A small GRU over the pollutant window, then a linear head.

    Input  (batch, lookback, n_features)
    Output (batch,) - standardised PM2.5 at the forecast hour
    """

    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        output, _ = self.gru(x)
        last_step = output[:, -1, :]
        return self.head(last_step).squeeze(-1)
