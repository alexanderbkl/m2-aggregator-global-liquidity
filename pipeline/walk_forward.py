"""
pipeline/walk_forward.py
────────────────────────
Purged Walk-Forward Cross-Validation for time-series data.

Implements expanding-window walk-forward splits with purge gaps
and embargo periods to prevent label leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """A single walk-forward fold."""
    fold_idx: int
    train_start: int
    train_end: int       # exclusive
    test_start: int
    test_end: int        # exclusive
    n_purged: int        # number of purged samples between train and test


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation.

    Uses an expanding training window. Between each train/test boundary,
    a purge gap and embargo period are applied to prevent label leakage.

    Parameters
    ----------
    min_train_samples : int
        Minimum number of training samples before the first test fold.
    test_samples : int
        Number of samples in each test fold.
    step_samples : int
        Number of samples to step forward between folds.
    purge_samples : int
        Number of samples to remove from the end of training (purge gap).
    embargo_samples : int
        Number of samples to skip after test start (embargo).
    """

    def __init__(
        self,
        min_train_samples: int,
        test_samples: int,
        step_samples: int,
        purge_samples: int = 7,
        embargo_samples: int = 7,
    ):
        self.min_train_samples = min_train_samples
        self.test_samples = test_samples
        self.step_samples = step_samples
        self.purge_samples = purge_samples
        self.embargo_samples = embargo_samples

    def split(self, n_samples: int) -> List[WalkForwardFold]:
        """
        Generate walk-forward folds.

        Parameters
        ----------
        n_samples : int
            Total number of samples in the dataset.

        Returns
        -------
        list of WalkForwardFold
        """
        folds: List[WalkForwardFold] = []
        fold_idx = 0

        # First test starts after min_train + purge + embargo
        test_start_base = self.min_train_samples + self.purge_samples + self.embargo_samples

        while test_start_base + fold_idx * self.step_samples < n_samples:
            test_start = test_start_base + fold_idx * self.step_samples
            test_end = min(test_start + self.test_samples, n_samples)

            if test_end - test_start < 10:
                break

            # Training window: [0, test_start - purge - embargo)
            train_end = test_start - self.purge_samples - self.embargo_samples
            train_start = 0

            if train_end <= train_start:
                break

            folds.append(WalkForwardFold(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_purged=self.purge_samples + self.embargo_samples,
            ))

            fold_idx += 1

        logger.info(
            "PurgedWalkForwardCV: %d folds from %d samples "
            "(min_train=%d, test=%d, step=%d, purge=%d, embargo=%d)",
            len(folds), n_samples,
            self.min_train_samples, self.test_samples,
            self.step_samples, self.purge_samples, self.embargo_samples,
        )
        return folds

    def get_fold_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fold: WalkForwardFold,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract train and test arrays for a fold.

        Returns
        -------
        X_train, y_train, X_test, y_test
        """
        X_train = X[fold.train_start:fold.train_end]
        y_train = y[fold.train_start:fold.train_end]
        X_test = X[fold.test_start:fold.test_end]
        y_test = y[fold.test_start:fold.test_end]
        return X_train, y_train, X_test, y_test


def create_walk_forward_cv(config: dict) -> PurgedWalkForwardCV:
    """Factory function to create PurgedWalkForwardCV from config."""
    return PurgedWalkForwardCV(
        min_train_samples=config.get("wf_min_train_days", 1095),
        test_samples=config.get("wf_test_days", 182),
        step_samples=config.get("wf_step_days", 182),
        purge_samples=config.get("wf_purge_days", 7),
        embargo_samples=config.get("wf_embargo_days", 7),
    )
