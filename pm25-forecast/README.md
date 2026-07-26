# pm25-forecast

Forecasts PM2.5 fine-particle pollution 24 hours ahead for six cities in India and the USA, using a PyTorch GRU over the previous 72 hours of pollutant readings.

PM2.5 is the pollutant that drives daily "can I go outside" decisions — whether to run, to send children to school, to run an air purifier, or to wear a mask. A 24-hour lead time is what makes those decisions actionable.

## Data

Open-Meteo Air Quality API. Free, no API key, global coverage.

```
https://air-quality-api.open-meteo.com/v1/air-quality
```

Fetched live at run time — 92 days of hourly history per city, 2,208 rows each.

| Feature | Source column |
|---|---|
| PM2.5 (target) | `pm2_5` |
| PM10 | `pm10` |
| Carbon monoxide | `carbon_monoxide` |
| Nitrogen dioxide | `nitrogen_dioxide` |
| Ozone | `ozone` |
| Hour of day | derived, encoded as sin/cos |

Cities: Delhi, Mumbai, Kolkata, Los Angeles, New York, Fresno.

## Method

- Input: 72 hours of the six features. Output: PM2.5 at hour +24.
- Model: 2-layer GRU, hidden size 64, dropout 0.2, linear head. 41,089 parameters.
- Split: strictly chronological — first 70% of hours train, next 15% validate, last 15% test. Windows are built inside each block, so no training window can reach a test hour. `tests/test_pm25.py` asserts this.
- Standardisation uses training-block statistics only.
- Loss: SmoothL1. Optimiser: Adam, lr 1e-3, ReduceLROnPlateau. Early stopping, patience 6.
- Baseline: persistence — predict that PM2.5 in 24 hours equals the last observed value.

## Results

Held-out test window, 1,422 hourly forecasts never seen in training. Units µg/m³, lower is better.

| City | n | Model MAE | Persistence MAE | Model RMSE | Persistence RMSE |
|---|---|---|---|---|---|
| Delhi | 237 | 31.23 | **30.14** | **41.12** | 44.71 |
| Mumbai | 237 | **2.93** | 3.07 | **3.53** | 3.74 |
| Kolkata | 237 | 8.99 | **8.02** | 11.72 | **10.80** |
| Los Angeles | 237 | **3.68** | 3.93 | 5.17 | **5.05** |
| New York | 237 | **12.77** | 16.09 | **20.68** | 23.99 |
| Fresno | 237 | **3.11** | 3.91 | **3.90** | 5.03 |
| **All** | **1422** | **10.45** | 10.86 | **19.62** | 21.43 |

Overall the model beats persistence by 3.8% on MAE and 8.4% on RMSE.

**Where it loses.** Persistence wins on MAE in Delhi and Kolkata — the two highest-pollution, highest-variance cities in the set. The model still wins on RMSE in Delhi, meaning it avoids the largest errors but is worse on typical hours. A single pooled model trained across cities with very different PM2.5 distributions is the likely cause; per-city models would probably close this.

Training stopped at epoch 13 of 40 with best validation loss 0.2148 at epoch 7. Training loss kept falling to 0.1990 while validation loss rose to 0.2644 — the model overfits quickly on 92 days of data.

## Limitations

- 92 days is a short record. It covers one season, so the model has never seen a Delhi winter inversion, which is when PM2.5 there is at its worst and forecasts matter most.
- Open-Meteo air quality values are model reanalysis, not ground sensor readings. They approximate monitoring-station data rather than replacing it.
- One pooled model across six cities. It underperforms on the highest-variance cities.
- No meteorology in the feature set. Wind speed and boundary-layer height drive PM2.5 dispersion and would likely help.
- Results shift as the API window rolls forward, since the data is fetched live. `run_log.txt` records the run these numbers came from.

## Run it

```bash
pip install -r requirements.txt
python train.py                 # fetch, train, evaluate
python train.py --epochs 60     # train longer
python -m pytest tests/ -q      # 14 tests
```

Full console output from the run that produced the table above is in `run_log.txt`.
