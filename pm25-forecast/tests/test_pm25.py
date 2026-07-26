"""Tests for the PM2.5 forecaster.

These assert behaviour that would silently corrupt results if it broke:
the time split must not leak, windows must line up with their targets,
and the persistence baseline must be the last observed value.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import VARIABLES, build_url, fetch_city  # noqa: E402
from dataset import (  # noqa: E402
    HORIZON,
    LOOKBACK,
    build_city,
    hour_features,
    make_windows,
    split_bounds,
)
from model import PM25Forecaster  # noqa: E402


def synthetic_columns(n=600):
    """A deterministic series with a real daily cycle."""
    t = np.arange(n, dtype=np.float64)
    daily = 40 + 20 * np.sin(2 * np.pi * t / 24.0)
    return {
        "pm2_5": list(daily),
        "pm10": list(daily * 1.6),
        "carbon_monoxide": list(200 + t % 50),
        "nitrogen_dioxide": list(15 + (t % 12)),
        "ozone": list(60 - (t % 30)),
    }


def synthetic_times(n=600):
    return [f"2026-01-{1 + (i // 24) % 28:02d}T{i % 24:02d}:00" for i in range(n)]


# --- windowing ------------------------------------------------------------

def test_window_target_is_horizon_hours_after_window_end():
    matrix = np.arange(400, dtype=np.float64).reshape(-1, 1)
    xs, ys = make_windows(matrix, target_index=0, lookback=10, horizon=5)
    # first window covers rows 0..9; target is row 10+5-1 = 14
    assert xs[0][-1][0] == 9.0
    assert ys[0] == 14.0


def test_window_shapes_match_lookback_and_feature_count():
    matrix = np.zeros((300, 4))
    xs, ys = make_windows(matrix, target_index=0, lookback=72, horizon=24)
    assert xs.shape == (300 - 72 - 24 + 1, 72, 4)
    assert ys.shape == (xs.shape[0],)


def test_block_too_short_returns_empty_not_garbage():
    matrix = np.zeros((10, 3))
    xs, ys = make_windows(matrix, target_index=0, lookback=72, horizon=24)
    assert len(xs) == 0 and len(ys) == 0


# --- the split must not leak ----------------------------------------------

def test_split_is_chronological_and_covers_everything():
    train_end, val_end = split_bounds(1000)
    assert 0 < train_end < val_end < 1000
    assert train_end == 700 and val_end == 850


def test_no_training_window_reaches_into_test_hours():
    """The whole point of a time split. If this fails, every metric is a lie."""
    columns = synthetic_columns(600)
    parts = build_city(columns, VARIABLES, synthetic_times(600))
    train_end, val_end = split_bounds(600)

    n_train_windows = len(parts["train"]["x"])
    # Last training window ends at train_end-1 at the very latest.
    last_train_hour = n_train_windows - 1 + LOOKBACK + HORIZON - 1
    assert last_train_hour < train_end, (
        f"training reaches hour {last_train_hour} but train block ends at {train_end}"
    )


def test_scaler_is_fitted_on_training_hours_only():
    columns = synthetic_columns(600)
    parts = build_city(columns, VARIABLES, synthetic_times(600))
    train_end, _ = split_bounds(600)

    matrix = np.column_stack([columns[v] for v in VARIABLES])
    expected = matrix[:train_end].mean(axis=0)
    np.testing.assert_allclose(parts["scaler"]["mean"][: len(VARIABLES)], expected)


# --- baseline -------------------------------------------------------------

def test_persistence_is_the_last_observed_value_in_the_window():
    columns = synthetic_columns(600)
    parts = build_city(columns, VARIABLES, synthetic_times(600))
    train_end, _ = split_bounds(600)

    series = np.array(columns["pm2_5"])
    train_block = series[:train_end]
    expected_first = train_block[LOOKBACK - 1]
    assert parts["train"]["persistence"][0] == pytest.approx(expected_first)


def test_persistence_and_truth_have_the_same_length():
    columns = synthetic_columns(600)
    parts = build_city(columns, VARIABLES, synthetic_times(600))
    for split in ("train", "val", "test"):
        assert len(parts[split]["persistence"]) == len(parts[split]["y_raw"])


# --- time features --------------------------------------------------------

def test_hour_encoding_is_cyclical_with_no_midnight_jump():
    features = hour_features(["2026-01-01T23:00", "2026-01-02T00:00"])
    distance = np.linalg.norm(features[0] - features[1])
    # one hour apart on the unit circle, not a jump from 23 to 0
    assert distance == pytest.approx(2 * np.sin(np.pi / 24), abs=1e-9)


def test_hour_encoding_stays_on_the_unit_circle():
    features = hour_features([f"2026-01-01T{h:02d}:00" for h in range(24)])
    np.testing.assert_allclose((features**2).sum(axis=1), 1.0)


# --- model ----------------------------------------------------------------

def test_model_maps_a_batch_of_windows_to_one_number_each():
    model = PM25Forecaster(n_features=7)
    out = model(torch.zeros(8, LOOKBACK, 7))
    assert out.shape == (8,)


def test_model_gradients_actually_flow():
    """A model that cannot learn would still 'pass' a shape test."""
    model = PM25Forecaster(n_features=7)
    loss = model(torch.randn(4, LOOKBACK, 7)).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


# --- fetching -------------------------------------------------------------

def test_url_requests_every_variable_and_no_forecast_hours():
    url = build_url(28.61, 77.21, 92)
    for name in VARIABLES:
        assert name in url
    assert "forecast_days=0" in url


def test_empty_api_response_yields_no_rows_rather_than_crashing(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr("json.load", lambda _: {})
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse())
    times, columns = fetch_city(0.0, 0.0)
    assert times == []
    assert all(len(v) == 0 for v in columns.values())
