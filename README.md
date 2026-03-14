# m2-aggregator-global-liquidity

A Python pipeline that builds a **daily Global M2 liquidity series** from official macro data, aligns it with Bitcoin price history, and feeds the combined feature set into a **Stacked Denoising Autoencoder (SDAE) + LightGBM** classifier to predict the next-week BTC price direction.

---

## Table of contents

1. [Overview](#overview)
2. [Project structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Data sources](#data-sources)
7. [Usage](#usage)
   - [Option A – Build M2 from FRED API (recommended)](#option-a--build-m2-from-fred-api-recommended)
   - [Option B – Load M2 from a pre-computed CSV](#option-b--load-m2-from-a-pre-computed-csv)
   - [Option C – BTC only (no M2 features)](#option-c--btc-only-no-m2-features)
8. [Pipeline steps](#pipeline-steps)
9. [Output files](#output-files)
10. [Running the tests](#running-the-tests)
11. [Troubleshooting](#troubleshooting)

---

## Overview

**Global M2 liquidity** is the USD-denominated sum of broad money (M2 or close equivalent) across a basket of major economies (US, Euro Area, China, Japan, UK). Research suggests this macro factor leads Bitcoin price cycles by several weeks.

This project:

- Downloads per-country M2 from the **FRED API** and converts each series to billions USD using monthly FX rates.
- Sums the country series into a single **monthly global M2** figure, then **forward-fills to daily** frequency (piecewise-constant between reporting dates).
- Generates **lagged** (1 – 98 days) and **growth** (7-day, 30-day, 90-day % change) features.
- Downloads **BTC OHLCV** history from Yahoo Finance and inner-joins it with the M2 feature table on `Date`.
- Engineers BTC **technical features** (log-returns, rolling z-scores, range, volume).
- Trains a **Stacked Denoising Autoencoder** to compress the feature space, then passes the latent representation to **LightGBM** for binary classification (`up_7d`).
- Saves evaluation charts (confusion matrix, equity curve, feature importance, SHAP beeswarm, liquidity-regime accuracy).

---

## Project structure

```
m2-aggregator-global-liquidity/
├── config.py                  # Central CONFIG dict – edit this to tune the pipeline
├── main.py                    # End-to-end pipeline entry point
├── requirements.txt           # Python dependencies
├── pipeline/
│   ├── m2_liquidity.py        # Global M2 series builder (FRED / CSV)
│   ├── data_loader.py         # BTC download + M2 merge
│   ├── features.py            # Feature engineering + target definition
│   ├── sdae.py                # Stacked Denoising Autoencoder (PyTorch)
│   ├── model.py               # Scaler → SDAE → LightGBM training & inference
│   └── evaluation.py          # Metrics + all chart functions
├── data/                      # Drop your pre-computed global_m2.csv here (Option B)
├── outputs/                   # Charts are written here at runtime
└── tests/
    └── test_m2_liquidity.py   # Unit tests (no network / API key required)
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| pip | ≥ 22 |

A free **FRED API key** is required when building the M2 series from scratch (Option A). Register at <https://fred.stlouisfed.org/docs/api/api_key.html> – it takes about 30 seconds.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/alexanderbkl/m2-aggregator-global-liquidity.git
cd m2-aggregator-global-liquidity

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Option A only) Set your FRED API key
export FRED_API_KEY="your_key_here"   # Windows: set FRED_API_KEY=your_key_here
```

---

## Configuration

All settings live in **`config.py`**. The most important knobs are:

| Key | Default | Description |
|---|---|---|
| `use_m2_exog` | `True` | Set `False` to disable all M2 features (ablation run). |
| `m2_source` | `"bis_fred"` | `"bis_fred"` → fetch from FRED API. `"csv"` → load from `m2_csv_path`. |
| `m2_csv_path` | `"data/global_m2.csv"` | Path to the pre-computed CSV (Option B). |
| `fred_api_key` | env `FRED_API_KEY` | API key; prefer the environment variable over hard-coding. |
| `m2_countries` | `["US","EA","CN","JP","GB"]` | Country basket for aggregation. |
| `m2_lag_days` | `[1,7,…,98]` | Lag offsets (days) for lagged M2 features. |
| `m2_growth_windows` | `[7,30,90]` | Windows (days) for M2 percentage-change features. |
| `btc_start_date` | `"2015-01-01"` | Earliest BTC data to download. |
| `btc_target_horizon` | `7` | Predict price direction N days ahead. |
| `train_ratio` / `val_ratio` | `0.70` / `0.15` | Chronological split fractions (remainder → test). |
| `sdae_hidden_dims` | `[256,128,64]` | SDAE encoder layer sizes. |
| `sdae_epochs` | `50` | Maximum SDAE training epochs. |
| `lgbm_params` | *(see config.py)* | Full LightGBM hyperparameter dict. |
| `output_dir` | `"outputs"` | Directory where charts are saved. |
| `regime_column` | `"m2_90d_chg"` | M2 feature used to bucket the regime-accuracy chart. |

---

## Data sources

### FRED series used per country

| Code | M2 series | Unit | FX series | FX direction |
|---|---|---|---|---|
| US | `M2SL` | Billions USD | — | — |
| EA | `MABMM301EZM189S` | Millions EUR | `EXUSEU` (USD per EUR) | multiply |
| CN | `MYAGM2CNM189N` | Billions CNY | `EXCHUS` (CNY per USD) | divide |
| JP | `MYAGM2JPM189N` | Billions JPY | `EXJPUS` (JPY per USD) | divide |
| GB | `MABMM301GBM189S` | Millions GBP | `EXUSUK` (USD per GBP) | multiply |

FX series are resampled to month-end and used to convert each local M2 figure to billions USD before summing.

### BTC price data

Downloaded automatically from Yahoo Finance via `yfinance` (ticker `BTC-USD`).

### Pre-computed CSV (Option B)

If you already have a Global M2 series (e.g. exported from BitcoinCounterFlow or TradingView's *"Global M2 Liquidity"* script), save it to `data/global_m2.csv` with at least two columns:

```
Date,M2_global_usd
2015-01-31,68432.5
2015-02-28,69100.1
...
```

Dates can be at any frequency; the pipeline forward-fills to daily automatically.

---

## Usage

### Option A – Build M2 from FRED API (recommended)

```bash
export FRED_API_KEY="your_key_here"
python main.py
```

The pipeline fetches all FRED series, aggregates them, and runs end-to-end.

### Option B – Load M2 from a pre-computed CSV

1. Save your CSV to `data/global_m2.csv` (format described above).
2. Edit `config.py`:

   ```python
   "m2_source": "csv",
   "m2_csv_path": "data/global_m2.csv",
   ```

3. Run:

   ```bash
   python main.py
   ```

### Option C – BTC only (no M2 features)

Use this to reproduce a baseline or for an ablation study:

```python
# config.py
"use_m2_exog": False,
```

```bash
python main.py
```

No FRED API key is needed in this mode.

---

## Pipeline steps

`main.py` executes the following steps in order:

1. **Load data** – Downloads BTC OHLCV from Yahoo Finance; if `use_m2_exog=True`, builds/loads the Global M2 table and inner-joins it to the BTC frame on `Date`. Early BTC days without M2 coverage are automatically dropped.

2. **Feature engineering** – Computes BTC technical features (log-returns, rolling means/stds/z-scores, normalised range, volume change) and appends all `M2_*` / `m2_*` columns as exogenous inputs. The binary target `up_7d` is `1` if `Close[t+7] > Close[t]`.

3. **Chronological split** – Splits rows into train (70 %), validation (15 %), and test (15 %) sets in time order; no shuffling.

4. **Scale → SDAE → LightGBM** – `StandardScaler` normalises all features; the SDAE encoder compresses them to a latent vector (default: 64 dimensions); raw M2 features are appended to the latent vector before passing to LightGBM.

5. **Evaluate** – Computes accuracy, F1, and AUC on the held-out test set; saves all charts to `outputs/`.

6. **Predict next period** – Uses the most recent BTC + M2 row to output `p_up` and a directional signal (`UP ↑` / `DOWN ↓`).

---

## Output files

All files are saved to the `outputs/` directory.

| File | Description |
|---|---|
| `confusion_matrix.png` | Predicted vs. actual direction on the test set. |
| `equity_curve.png` | Cumulative PnL of a long-flat strategy driven by the model signal. |
| `feature_importance.png` | Top-30 LightGBM feature importances (M2 features highlighted in red). |
| `m2_feature_importance.png` | Feature importances restricted to M2 columns only. |
| `shap_beeswarm.png` | SHAP beeswarm plot for the top 20 features (requires `shap` package). |
| `m2_regime_accuracy.png` | Directional accuracy bucketed by low / medium / high M2 liquidity growth. |

---

## Running the tests

The test suite uses `pytest` and requires no network access or FRED API key (all HTTP calls are mocked).

```bash
# Run all tests
pytest tests/ -v

# Run a specific test class
pytest tests/test_m2_liquidity.py::TestLoadOrBuildM2Series -v
```

All 27 tests should pass in under 2 seconds.

---

## Troubleshooting

**`ValueError: FRED API key is required`**  
Set the `FRED_API_KEY` environment variable before running, or add your key directly to `config.py` (not recommended for shared repositories).

**A country's M2 or FX series fails to download**  
The pipeline logs a warning and continues with the remaining countries rather than aborting. The global sum will underestimate total liquidity if a major economy is missing. Switch to `m2_source: "csv"` if FRED connectivity is unreliable.

**`FileNotFoundError` for `data/global_m2.csv`**  
You have selected `m2_source: "csv"` but have not placed the CSV file in the `data/` directory. Either provide the CSV or switch back to `m2_source: "bis_fred"`.

**`RuntimeError: Insufficient data rows after feature engineering`**  
The combined BTC + M2 date range is too short (< 200 rows). Ensure your FRED series or CSV starts early enough, or reduce `btc_start_date` in `config.py`.

**SHAP plot is missing**  
Install the optional `shap` package: `pip install shap`. The pipeline continues without it if it is not installed.

**Pipeline appears stuck at `Training SDAE on cpu ...`**  
Increase SDAE diagnostics in `config.py` to observe progress:

```python
"sdae_log_every_epochs": 1,
"sdae_log_every_batches": 1,   # very verbose; set to 0 to disable batch logs
"sdae_torch_num_threads": 2,   # optional: reduce CPU thread contention on macOS
```

If needed, reduce runtime while debugging:

```python
"sdae_epochs": 10,
"sdae_hidden_dims": [128, 64, 32],
```

**`OSError: ... Library not loaded: @rpath/libomp.dylib` when importing LightGBM (macOS)**  
Install OpenMP runtime and reinstall LightGBM in your venv:

```bash
brew install libomp
source .venv/bin/activate
pip install --force-reinstall lightgbm
```

If Homebrew is installed in a custom location, ensure `libomp.dylib` is discoverable via your dynamic loader paths.
