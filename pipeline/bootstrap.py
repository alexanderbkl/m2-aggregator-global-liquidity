"""
pipeline/bootstrap.py
─────────────────────
Moving Block Bootstrap (MBB) for training data augmentation.

Preserves local autocorrelation structure by resampling contiguous
blocks of time-series data with replacement.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MovingBlockBootstrap:
    """
    Moving Block Bootstrap for time-series data.

    Parameters
    ----------
    block_size : int
        Number of consecutive samples per block.
    n_bootstraps : int
        Number of bootstrap samples to generate.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        block_size: int = 60,
        n_bootstraps: int = 10,
        random_state: int = 42,
    ):
        self.block_size = block_size
        self.n_bootstraps = n_bootstraps
        self.rng = np.random.RandomState(random_state)

    def generate_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate bootstrap samples using Moving Block Bootstrap.

        Parameters
        ----------
        X : np.ndarray, shape (N, F)
        y : np.ndarray, shape (N,)

        Returns
        -------
        list of (X_boot, y_boot) tuples
        """
        n = len(X)
        bs = min(self.block_size, n)

        if bs < 2:
            logger.warning("Block size too small (%d), returning original data.", bs)
            return [(X.copy(), y.copy())]

        # Number of possible block starting positions
        n_blocks_available = n - bs + 1
        if n_blocks_available < 1:
            return [(X.copy(), y.copy())]

        # Number of blocks needed to reconstruct a dataset of size ~n
        n_blocks_needed = int(np.ceil(n / bs))

        samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(self.n_bootstraps):
            # Randomly select block starting indices with replacement
            starts = self.rng.randint(0, n_blocks_available, size=n_blocks_needed)

            X_parts = []
            y_parts = []
            for s in starts:
                X_parts.append(X[s:s + bs])
                y_parts.append(y[s:s + bs])

            X_boot = np.concatenate(X_parts, axis=0)[:n]
            y_boot = np.concatenate(y_parts, axis=0)[:n]

            samples.append((X_boot, y_boot))

        logger.info(
            "MBB: generated %d bootstrap samples (block_size=%d, n_samples=%d)",
            len(samples), bs, n,
        )
        return samples


def create_mbb(config: dict) -> MovingBlockBootstrap:
    """Factory function to create MovingBlockBootstrap from config."""
    return MovingBlockBootstrap(
        block_size=config.get("mbb_block_size", 60),
        n_bootstraps=config.get("mbb_n_bootstraps", 10),
        random_state=42,
    )
