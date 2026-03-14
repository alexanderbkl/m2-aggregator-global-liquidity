"""
pipeline/sdae.py
─────────────────
Stacked Denoising Autoencoder (SDAE) implemented in PyTorch.

Architecture
────────────
Encoder: input_dim → hidden[0] → hidden[1] → ... → latent_dim
Decoder: latent_dim → ... → hidden[0] → input_dim  (symmetric)

Training
────────
Gaussian noise (σ = noise_factor) is added to inputs; the loss is the MSE
between the *clean* inputs and the reconstruction.  Only the encoder half is
used to produce the latent representation that feeds into LightGBM.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, hidden_dims: List[int], output_dim: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        dims = list(reversed(hidden_dims))
        in_dim = dims[0]
        for h in dims[1:]:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SDAE(nn.Module):
    """Stacked Denoising Autoencoder."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        self.encoder = _Encoder(input_dim, hidden_dims, dropout)
        self.decoder = _Decoder(hidden_dims, input_dim, dropout)
        self.latent_dim: int = self.encoder.output_dim

    def forward(self, x_noisy: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x_noisy)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── Training ───────────────────────────────────────────────────────────────────

def train_sdae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict,
) -> SDAE:
    """
    Train the SDAE on the training split.

    Parameters
    ----------
    X_train : np.ndarray  shape (N, F)
    X_val   : np.ndarray  shape (M, F)
    config  : dict  (see config.py for relevant keys)

    Returns
    -------
    Trained SDAE model (eval mode).
    """
    hidden_dims: List[int] = config.get("sdae_hidden_dims", [256, 128, 64])
    noise_factor: float = config.get("sdae_noise_factor", 0.1)
    dropout: float = config.get("sdae_dropout", 0.2)
    lr: float = config.get("sdae_learning_rate", 1e-3)
    weight_decay: float = config.get("sdae_weight_decay", 1e-5)
    epochs: int = config.get("sdae_epochs", 50)
    batch_size: int = config.get("sdae_batch_size", 256)
    patience: int = config.get("sdae_patience", 10)
    log_every_epochs: int = max(1, int(config.get("sdae_log_every_epochs", 1)))
    log_every_batches: int = int(config.get("sdae_log_every_batches", 0))
    torch_num_threads = config.get("sdae_torch_num_threads")

    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Training SDAE on %s (input_dim=%d, latent_dim=%d, epochs=%d, batch_size=%d, torch_threads=%d)",
        device, input_dim, hidden_dims[-1], epochs, batch_size, torch.get_num_threads(),
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(X_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = SDAE(input_dim, hidden_dims, dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    no_improve = 0
    total_start = time.perf_counter()
    n_batches = len(train_loader)
    logger.info("SDAE dataloader ready: train_rows=%d, val_rows=%d, batches/epoch=%d",
                len(X_train), len(X_val), n_batches)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        logger.info("SDAE epoch %3d/%d started", epoch, epochs)
        for batch_idx, (batch,) in enumerate(train_loader, start=1):
            noisy = batch + noise_factor * torch.randn_like(batch)
            recon = model(noisy)
            loss = criterion(recon, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(batch)

            if log_every_batches > 0 and (batch_idx % log_every_batches == 0 or batch_idx == n_batches):
                logger.info(
                    "SDAE epoch %3d/%d batch %d/%d running_loss=%.6f",
                    epoch, epochs, batch_idx, n_batches, train_loss / (batch_idx * batch_size),
                )
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            noisy_val = X_val_t + noise_factor * torch.randn_like(X_val_t)
            val_loss = criterion(model(noisy_val), X_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            improved = True
        else:
            no_improve += 1
            improved = False

        epoch_secs = time.perf_counter() - epoch_start
        should_log_epoch = (epoch == 1) or (epoch % log_every_epochs == 0) or improved or (no_improve >= patience)
        if should_log_epoch:
            logger.info(
                "SDAE epoch %3d/%d complete  train_loss=%.6f  val_loss=%.6f  best_val=%.6f  no_improve=%d/%d  elapsed=%.2fs",
                epoch, epochs, train_loss, val_loss, best_val_loss, no_improve, patience, epoch_secs,
            )

        if no_improve >= patience:
            logger.info("SDAE early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    logger.info("SDAE training finished in %.2fs", time.perf_counter() - total_start)
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def encode_features(model: SDAE, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """
    Run only the encoder half of the trained SDAE to produce latent features.

    Returns
    -------
    np.ndarray  shape (N, latent_dim)
    """
    device = next(model.parameters()).device
    model.eval()
    parts: List[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        chunk = torch.tensor(X[start: start + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            z = model.encode(chunk).cpu().numpy()
        parts.append(z)
    return np.vstack(parts)
