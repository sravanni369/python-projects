"""Turn hourly pollutant series into supervised sequences, split by time.

The split is strictly chronological. Windows are built inside each split, so
no training window can overlap a validation or test hour.
"""

import numpy as np

LOOKBACK = 72   # hours of history the model sees
HORIZON = 24    # hours ahead it predicts
TARGET = "pm2_5"

# PM2.5 has a strong daily cycle (traffic peaks, overnight boundary layer).
# Encoding the hour as sin/cos lets the model use that without a jump at midnight.
TIME_FEATURES = ["hour_sin", "hour_cos"]


def hour_features(times):
    """Cyclical hour-of-day encoding from ISO timestamps like 2026-07-25T13:00."""
    hours = np.array([int(t[11:13]) for t in times], dtype=np.float64)
    angle = 2 * np.pi * hours / 24.0
    return np.column_stack([np.sin(angle), np.cos(angle)])


def split_bounds(n, train_frac=0.70, val_frac=0.15):
    """Chronological split points for a series of length n."""
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return train_end, val_end


def standardise(matrix, mean, std):
    return (matrix - mean) / std


def make_windows(matrix, target_index, lookback=LOOKBACK, horizon=HORIZON):
    """Slide a window over one contiguous block.

    X[i] is hours [i, i+lookback); y[i] is the target at i+lookback+horizon-1.
    Returns empty arrays when the block is too short to hold a single window.
    """
    n_rows, n_features = matrix.shape
    last_start = n_rows - lookback - horizon + 1
    if last_start <= 0:
        return (
            np.empty((0, lookback, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    xs = np.stack([matrix[i : i + lookback] for i in range(last_start)])
    ys = np.array(
        [matrix[i + lookback + horizon - 1, target_index] for i in range(last_start)]
    )
    return xs.astype(np.float32), ys.astype(np.float32)


def build_city(columns, variables, times=None, lookback=LOOKBACK, horizon=HORIZON):
    """Build train/val/test windows for one city.

    Standardisation uses training-block statistics only, so no test
    information reaches the model through the scaler.
    """
    matrix = np.column_stack([columns[name] for name in variables]).astype(np.float64)
    if times is not None:
        matrix = np.column_stack([matrix, hour_features(times)])
    target_index = variables.index(TARGET)

    train_end, val_end = split_bounds(len(matrix))
    train_block = matrix[:train_end]

    mean = train_block.mean(axis=0)
    std = train_block.std(axis=0)
    std[std < 1e-6] = 1.0  # a constant column must not blow up the scaler

    blocks = {
        "train": matrix[:train_end],
        "val": matrix[train_end:val_end],
        "test": matrix[val_end:],
    }

    out = {}
    for split, block in blocks.items():
        xs, ys_scaled = make_windows(
            standardise(block, mean, std), target_index, lookback, horizon
        )
        # Keep the raw target too, so metrics are reported in ug/m3.
        _, ys_raw = make_windows(block, target_index, lookback, horizon)
        # Persistence baseline: the last observed PM2.5 in the input window.
        if len(block) >= lookback + horizon:
            last_start = len(block) - lookback - horizon + 1
            persistence = np.array(
                [block[i + lookback - 1, target_index] for i in range(last_start)]
            )
        else:
            persistence = np.empty((0,))
        out[split] = {
            "x": xs,
            "y_scaled": ys_scaled,
            "y_raw": ys_raw.astype(np.float64),
            "persistence": persistence,
        }

    out["scaler"] = {"mean": mean, "std": std, "target_index": target_index}
    return out
