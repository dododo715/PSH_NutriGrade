#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature extraction for numerical spatial fluorescence maps.

Author: zhangtong sun

The input map must already be decoded into numerical values. This module does
not read images, invert pseudo-color maps, create figures, or export results.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__author__ = "zhangtong sun"

FV_BINS: Tuple[Tuple[float, float], ...] = (
    (0.00, 0.20),
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 1.00),
)

FMPRIME_BINS: Tuple[Tuple[float, float], ...] = (
    (0.00, 0.40),
    (0.40, 0.60),
    (0.60, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
)


def _validate_bins(bins: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    """Validate non-overlapping ascending intervals."""
    clean = tuple((float(low), float(high)) for low, high in bins)
    if not clean:
        raise ValueError("bins must contain at least one interval.")
    previous_high: Optional[float] = None
    for low, high in clean:
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            raise ValueError(f"Invalid interval: ({low}, {high}).")
        if previous_high is not None and low < previous_high:
            raise ValueError("bins must not overlap and must be ordered by increasing value.")
        previous_high = high
    return clean


def _grade_label(low: float, high: float) -> str:
    return f"{low:g}-{high:g}"


def _unbiased_skewness(x: np.ndarray) -> float:
    """Adjusted Fisher-Pearson skewness."""
    n = int(x.size)
    if n < 3 or np.std(x, ddof=0) == 0.0:
        return 0.0
    centered = x - np.mean(x)
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    g1 = m3 / (m2 ** 1.5)
    return float(np.sqrt(n * (n - 1)) / (n - 2) * g1)


def _unbiased_fisher_kurtosis(x: np.ndarray) -> float:
    """Unbiased Fisher excess kurtosis."""
    n = int(x.size)
    if n < 4 or np.std(x, ddof=0) == 0.0:
        return 0.0
    centered = x - np.mean(x)
    m2 = float(np.mean(centered**2))
    m4 = float(np.mean(centered**4))
    g2 = m4 / (m2**2) - 3.0
    return float(((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0))


def _shannon_entropy(x: np.ndarray, n_bins: int = 64) -> float:
    """Return base-2 Shannon entropy using a histogram over [0, 1]."""
    if n_bins < 2:
        raise ValueError("entropy_bins must be at least 2.")
    histogram, _ = np.histogram(x, bins=n_bins, range=(0.0, 1.0))
    probabilities = histogram[histogram > 0].astype(float)
    if probabilities.size == 0:
        return np.nan
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def extract_spatial_map_features(
    values: Sequence[Sequence[float]],
    mask: Optional[Sequence[Sequence[bool]]] = None,
    bins: Sequence[Tuple[float, float]] = FMPRIME_BINS,
    prefix: str = "Fluorescence",
    low_tail_threshold: float = 0.40,
    high_tail_threshold: float = 0.80,
    entropy_bins: int = 64,
) -> Dict[str, float]:
    """Extract distribution features from an already decoded 2D value map.

    Returned features include interval fractions, mean, median, SD, CV,
    skewness, Fisher excess kurtosis, P10/P25/P75/P90, IQR, low/high tail
    fractions, and Shannon entropy.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must be a 2D numerical array.")

    if mask is None:
        valid = np.isfinite(array)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != array.shape:
            raise ValueError("mask must have the same shape as values.")
        valid = valid & np.isfinite(array)

    x = array[valid]
    intervals = _validate_bins(bins)

    feature_names = [
        "Mean", "Median", "SD", "CV", "Skewness", "Kurtosis",
        "P10", "P25", "P75", "P90", "IQR", "LowTail", "HighTail", "Entropy",
    ]

    result: Dict[str, float] = {
        f"{prefix}_Grade_{_grade_label(low, high)}": np.nan
        for low, high in intervals
    }
    result.update({f"{prefix}_{name}": np.nan for name in feature_names})

    if x.size == 0:
        return result

    for index, (low, high) in enumerate(intervals):
        if index == len(intervals) - 1:
            count = np.sum((x >= low) & (x <= high))
        else:
            count = np.sum((x >= low) & (x < high))
        result[f"{prefix}_Grade_{_grade_label(low, high)}"] = float(count / x.size)

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    p10, p25, median, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])

    result.update(
        {
            f"{prefix}_Mean": mean,
            f"{prefix}_Median": float(median),
            f"{prefix}_SD": sd,
            f"{prefix}_CV": float(sd / mean) if mean > 0.0 else np.nan,
            f"{prefix}_Skewness": _unbiased_skewness(x),
            f"{prefix}_Kurtosis": _unbiased_fisher_kurtosis(x),
            f"{prefix}_P10": float(p10),
            f"{prefix}_P25": float(p25),
            f"{prefix}_P75": float(p75),
            f"{prefix}_P90": float(p90),
            f"{prefix}_IQR": float(p75 - p25),
            f"{prefix}_LowTail": float(np.mean(x <= low_tail_threshold)),
            f"{prefix}_HighTail": float(np.mean(x >= high_tail_threshold)),
            f"{prefix}_Entropy": _shannon_entropy(x, n_bins=entropy_bins),
        }
    )
    return result


__all__ = ["FV_BINS", "FMPRIME_BINS", "extract_spatial_map_features"]
