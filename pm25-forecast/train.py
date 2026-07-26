"""Train and evaluate the PM2.5 forecaster. Every number printed is measured.

Run:  python train.py
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import CITIES, VARIABLES, build_url, fetch_all
from dataset import HORIZON, LOOKBACK, TIME_FEATURES, build_city
from model import PM25Forecaster

SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(pred, truth):
    """MAE and RMSE in ug/m3."""
    error = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.abs(error).mean()), float(np.sqrt((error**2).mean()))


def to_raw(scaled, mean, std, target_index):
    """Undo standardisation on the target column."""
    return scaled * std[target_index] + mean[target_index]


def run_epoch(model, loader, loss_fn, optimiser=None):
    training = optimiser is not None
    model.train(training)
    total, count = 0.0, 0
    for xb, yb in loader:
        if training:
            optimiser.zero_grad()
        with torch.set_grad_enabled(training):
            pred = model(xb)
            loss = loss_fn(pred, yb)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        total += loss.item() * len(yb)
        count += len(yb)
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="PM2.5 24h forecaster")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--past-days", type=int, default=92)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    set_seed()
    print("PM2.5 24-hour forecaster")
    print(f"seed={SEED}  lookback={LOOKBACK}h  horizon={HORIZON}h  torch={torch.__version__}")
    print(f"source: {build_url(28.61, 77.21, args.past_days)}")
    print()

    print("Fetching hourly air quality...")
    raw = fetch_all(CITIES, past_days=args.past_days)
    if not raw:
        raise SystemExit("No city data returned; aborting rather than inventing data.")
    print()

    per_city = {name: build_city(cols, VARIABLES, times)
                for name, (times, cols) in raw.items()}
    feature_names = VARIABLES + TIME_FEATURES

    def pool(split):
        xs = [per_city[c][split]["x"] for c in per_city if len(per_city[c][split]["x"])]
        ys = [per_city[c][split]["y_scaled"] for c in per_city if len(per_city[c][split]["x"])]
        return np.concatenate(xs), np.concatenate(ys)

    x_train, y_train = pool("train")
    x_val, y_val = pool("val")
    print(f"windows: train={len(x_train)}  val={len(x_val)}  "
          f"test={sum(len(per_city[c]['test']['x']) for c in per_city)}")
    print(f"features: {feature_names}")
    print()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size,
    )

    model = PM25Forecaster(n_features=len(feature_names))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: GRU hidden=64 layers=2  parameters={n_params:,}")
    print()

    loss_fn = nn.SmoothL1Loss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)

    best_val, best_state, started = float("inf"), None, time.time()
    best_epoch, stale = 0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimiser)
        val_loss = run_epoch(model, val_loader, loss_fn)
        scheduler.step(val_loss)
        marker = ""
        if val_loss < best_val:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        else:
            stale += 1
        print(f"epoch {epoch:>3}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}{marker}")
        if stale >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break

    print(f"\ntrained in {time.time() - started:.1f}s, best val loss {best_val:.4f}")
    model.load_state_dict(best_state)
    model.eval()

    print("\nHeld-out test window (last 15% of hours, never seen in training)")
    print(f"{'city':<13}{'n':>5}{'model MAE':>11}{'persist MAE':>13}{'model RMSE':>12}{'persist RMSE':>14}")
    print("-" * 68)

    rows, all_model, all_persist, all_truth = [], [], [], []
    for city, parts in per_city.items():
        test = parts["test"]
        if not len(test["x"]):
            continue
        scaler = parts["scaler"]
        with torch.no_grad():
            scaled = model(torch.from_numpy(test["x"])).numpy()
        pred = to_raw(scaled, scaler["mean"], scaler["std"], scaler["target_index"])
        truth = test["y_raw"]
        persist = test["persistence"]

        m_mae, m_rmse = metrics(pred, truth)
        p_mae, p_rmse = metrics(persist, truth)
        rows.append((city, len(truth), m_mae, p_mae, m_rmse, p_rmse))
        all_model.append(pred); all_persist.append(persist); all_truth.append(truth)
        print(f"{city:<13}{len(truth):>5}{m_mae:>11.2f}{p_mae:>13.2f}{m_rmse:>12.2f}{p_rmse:>14.2f}")

    pred_all = np.concatenate(all_model)
    persist_all = np.concatenate(all_persist)
    truth_all = np.concatenate(all_truth)
    m_mae, m_rmse = metrics(pred_all, truth_all)
    p_mae, p_rmse = metrics(persist_all, truth_all)
    print("-" * 68)
    print(f"{'ALL':<13}{len(truth_all):>5}{m_mae:>11.2f}{p_mae:>13.2f}{m_rmse:>12.2f}{p_rmse:>14.2f}")

    print(f"\nunits: ug/m3. lower is better.")
    improvement = (p_mae - m_mae) / p_mae * 100
    if m_mae < p_mae:
        print(f"RESULT: model beats persistence on MAE by {improvement:.1f}% overall.")
    else:
        print(f"RESULT: model LOSES to persistence on MAE by {-improvement:.1f}% overall.")
    print("Per-city results above show where it holds and where it does not.")


if __name__ == "__main__":
    main()
