"""
pipeline/model.py
──────────────────
LightGBM binary classification model that uses SDAE latent features
plus (optionally) the raw M2 exogenous features as input.

Training/prediction workflow
────────────────────────────
1. Fit a StandardScaler on X_train (BTC + M2 columns).
2. Run the scaler through SDAE training.
3. Encode X_train / X_val / X_test via the trained SDAE encoder.
4. Append raw M2 exogenous columns back onto the encoded representation
   so LightGBM sees both the compressed BTC latent vector AND the raw
   macro features (optional – controlled by config['use_m2_exog']).
5. Train LightGBM with early stopping.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.sdae import SDAE, encode_features, train_sdae

logger = logging.getLogger(__name__)


def _import_lightgbm() -> Any:
    """Import LightGBM lazily to reduce native runtime conflicts during SDAE training."""
    try:
        import lightgbm as lgb
    except OSError as exc:
        # LightGBM wheels on macOS require the OpenMP runtime from Homebrew.
        if sys.platform == "darwin" and "libomp" in str(exc):
            raise RuntimeError(
                "LightGBM failed to load because libomp.dylib is missing. "
                "Install it with 'brew install libomp', then reinstall LightGBM "
                "inside the active virtualenv (pip install --force-reinstall lightgbm)."
            ) from exc
        raise
    return lgb


# ── Train ──────────────────────────────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: List[str],
    config: dict,
) -> Tuple[SDAE, StandardScaler, Any, List[str]]:
    """
    Full training pipeline: scale → SDAE → LightGBM.

    Returns
    -------
    sdae_model    : trained SDAE
    scaler        : fitted StandardScaler
    lgbm_model    : fitted LGBMClassifier
    lgbm_features : feature names for LightGBM (encoded + optional raw M2)
    """
    use_m2: bool = config.get("use_m2_exog", True)
    m2_cols: List[str] = [
        c for c in feature_cols if c.startswith("M2_") or c.startswith("m2_")
    ] if use_m2 else []

    # 1. Scale all features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train[feature_cols].values)
    X_val_sc = scaler.transform(X_val[feature_cols].values)

    # 2. Train SDAE
    sdae_model = train_sdae(X_train_sc, X_val_sc, config)

    # 3. Encode
    Z_train = encode_features(sdae_model, X_train_sc)
    Z_val = encode_features(sdae_model, X_val_sc)

    latent_dim = Z_train.shape[1]
    enc_names = [f"z_{i}" for i in range(latent_dim)]

    # 4. Optionally append raw (scaled) M2 columns
    if m2_cols:
        m2_idx = [feature_cols.index(c) for c in m2_cols]
        Z_train = np.hstack([Z_train, X_train_sc[:, m2_idx]])
        Z_val = np.hstack([Z_val, X_val_sc[:, m2_idx]])
        lgbm_features = enc_names + m2_cols
    else:
        lgbm_features = enc_names

    # 5. Train LightGBM
    lgb = _import_lightgbm()
    lgbm_params = dict(config.get("lgbm_params", {}))
    early_stop = config.get("lgbm_early_stopping_rounds", 50)

    lgbm_model = lgb.LGBMClassifier(**lgbm_params)
    lgbm_model.fit(
        Z_train,
        y_train.values,
        eval_set=[(Z_val, y_val.values)],
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    logger.info(
        "LightGBM trained: best_iteration=%d, n_features=%d",
        lgbm_model.best_iteration_,
        len(lgbm_features),
    )
    return sdae_model, scaler, lgbm_model, lgbm_features


# ── Predict ────────────────────────────────────────────────────────────────────

def predict(
    X: pd.DataFrame,
    feature_cols: List[str],
    sdae_model: SDAE,
    scaler: StandardScaler,
    lgbm_model: Any,
    config: dict,
) -> np.ndarray:
    """
    Run inference: scale → encode → (append M2) → LightGBM.

    Returns
    -------
    np.ndarray of shape (N,)
        Predicted probability of price being up at the target horizon.
    """
    use_m2: bool = config.get("use_m2_exog", True)
    m2_cols: List[str] = [
        c for c in feature_cols if c.startswith("M2_") or c.startswith("m2_")
    ] if use_m2 else []

    X_sc = scaler.transform(X[feature_cols].values)
    Z = encode_features(sdae_model, X_sc)

    if m2_cols:
        m2_idx = [feature_cols.index(c) for c in m2_cols]
        Z = np.hstack([Z, X_sc[:, m2_idx]])

    proba = lgbm_model.predict_proba(Z)[:, 1]
    return proba
