#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared numerical helpers for fluorescence feature extraction.

Author: zhangtong sun
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__author__ = "zhangtong sun"


def clean_1d(values: Sequence[float]) -> np.ndarray:
    """Convert input to a one-dimensional float array and remove non-finite values."""
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def coefficient_of_variation(values: Sequence[float], ddof: int = 0) -> float:
    """Return SD / mean after removing non-finite values."""
    x = clean_1d(values)
    if x.size == 0 or x.size <= ddof:
        return np.nan
    mean = float(np.mean(x))
    if mean == 0.0:
        return np.nan
    return float(np.std(x, ddof=ddof) / mean)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination R^2."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 2:
        return np.nan
    y = y_true[mask]
    y_hat = y_pred[mask]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)
