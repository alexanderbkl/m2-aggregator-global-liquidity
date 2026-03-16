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

Ensemble support
────────────────
train_ensemble() trains multiple SDAE+LightGBM pipelines on bootstrap
samples and averages their predictions via predict_ensemble().
"""

from __future__ import annotations

import logging
import sys
from typing import Any, List, Tuple

import numpy as np
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
    config: dict,
) -> Tuple[SDAE, StandardScaler, Any, List[str]]:
    """
    Full training pipeline: scale → SDAE → LightGBM.

    Parameters
    ----------
    X_train, y_train : np.ndarray
    X_val, y_val     : np.ndarray
    feature_cols     : list of str
    config           : dict

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
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

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
        y_train,
        eval_set=[(Z_val, y_val)],
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


# ── Ensemble ──────────────────────────────────────────────────────────────────

def train_ensemble(
    bootstrap_samples: List[Tuple[np.ndarray, np.ndarray]],
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
    config: dict,
) -> List[Tuple[SDAE, StandardScaler, Any, List[str]]]:
    """
    Train an ensemble of SDAE+LightGBM models on bootstrap samples.

    Parameters
    ----------
    bootstrap_samples : list of (X_boot, y_boot) tuples
    X_val, y_val      : validation arrays
    feature_cols      : feature column names
    config            : pipeline config

    Returns
    -------
    list of (sdae_model, scaler, lgbm_model, lgbm_features) tuples
    """
    models = []
    for i, (X_boot, y_boot) in enumerate(bootstrap_samples):
        logger.info("Training ensemble member %d/%d (n_samples=%d)",
                    i + 1, len(bootstrap_samples), len(X_boot))
        result = train_model(X_boot, y_boot, X_val, y_val, feature_cols, config)
        models.append(result)
    logger.info("Ensemble training complete: %d models", len(models))
    return models


def predict_ensemble(
    X: np.ndarray,
    feature_cols: List[str],
    models: List[Tuple[SDAE, StandardScaler, Any, List[str]]],
    config: dict,
) -> np.ndarray:
    """
    Average predictions from an ensemble of models.

    Parameters
    ----------
    X             : np.ndarray feature matrix
    feature_cols  : feature column names
    models        : list of (sdae_model, scaler, lgbm_model, lgbm_features)
    config        : pipeline config

    Returns
    -------
    np.ndarray of shape (N,) – averaged p_up
    """
    all_preds = []
    for sdae_model, scaler, lgbm_model, lgbm_features in models:
        p_up = predict(X, feature_cols, sdae_model, scaler, lgbm_model, config)
        all_preds.append(p_up)
    return np.mean(all_preds, axis=0)


# ── Predict ────────────────────────────────────────────────────────────────────

def predict(
    X: np.ndarray,
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

    X_sc = scaler.transform(X)
    Z = encode_features(sdae_model, X_sc)

    if m2_cols:
        m2_idx = [feature_cols.index(c) for c in m2_cols]
        Z = np.hstack([Z, X_sc[:, m2_idx]])

    proba = lgbm_model.predict_proba(Z)[:, 1]
    return proba
