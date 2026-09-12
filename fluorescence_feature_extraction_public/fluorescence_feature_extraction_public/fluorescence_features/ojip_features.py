#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature extraction for aligned OJIP fluorescence induction curves.

Author: zhangtong sun

Expected input is numerical data already loaded by the caller. This module does
not contain file I/O, plotting, image generation, or result export code.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .common import coefficient_of_variation

__author__ = "zhangtong sun"


def _log10_time_axis(time_points: Sequence[float]) -> np.ndarray:
    """Convert time to log10 scale, replacing invalid/non-positive values safely."""
    t = np.asarray(time_points, dtype=float).reshape(-1).copy()
    positive = t[np.isfinite(t) & (t > 0)]
    if positive.size == 0:
        raise ValueError("time_points must contain at least one positive value.")
    replacement = float(np.min(positive) / 2.0)
    t[~np.isfinite(t) | (t <= 0)] = replacement
    return np.log10(t)


def _trapz_finite(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate finite paired values with the trapezoidal rule."""
    mask = np.isfinite(y) & np.isfinite(x)
    if np.sum(mask) < 2:
        return np.nan
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y[mask], x[mask]))
    return float(np.trapz(y[mask], x[mask]))


def normalize_ojip_curves(
    curves: Sequence[Sequence[float]],
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize curves as Vt=(Ft-Fo)/(Fp-Fo).

    Returns
    -------
    normalized
        Array with the same shape as the input curves.
    valid
        Boolean mask marking curves with a valid non-zero fluorescence range.
    """
    array = np.asarray(curves, dtype=float)
    if array.ndim != 2:
        raise ValueError("curves must be a 2D array with shape (n_curves, n_time_points).")
    if array.size == 0:
        return np.empty_like(array, dtype=float), np.zeros(array.shape[0], dtype=bool)

    normalized = np.full_like(array, np.nan, dtype=float)
    valid = np.zeros(array.shape[0], dtype=bool)

    for index, curve in enumerate(array):
        finite = np.isfinite(curve)
        if np.sum(finite) < 2 or not np.isfinite(curve[0]):
            continue
        fo = float(curve[0])
        fp = float(np.nanmax(curve))
        denominator = fp - fo
        if np.isfinite(denominator) and abs(denominator) > eps:
            normalized[index] = (curve - fo) / denominator
            valid[index] = True

    return normalized, valid


def extract_ojip_features(
    time_points: Sequence[float],
    curves: Sequence[Sequence[float]],
    phase_cv_ddof: int = 0,
    normalize_eps: float = 1e-12,
) -> Dict[str, float]:
    """Extract IEA, KSV, and Phase_Shift from aligned OJIP leaf curves.

    Definitions
    -----------
    IEA
        Integral over log10(time) of max(F)-min(F) across leaves.
    KSV
        Integral over log10(time) of the across-leaf population SD after OJIP
        normalization.
    Phase_Shift
        Coefficient of variation of the time-to-maximum fluorescence among leaves.
    """
    if phase_cv_ddof < 0:
        raise ValueError("phase_cv_ddof must be non-negative.")

    t = np.asarray(time_points, dtype=float).reshape(-1)
    f = np.asarray(curves, dtype=float)
    if f.ndim != 2:
        raise ValueError("curves must be a 2D array with shape (n_leaves, n_time_points).")
    if f.shape[1] != t.size:
        raise ValueError("The number of curve columns must equal len(time_points).")
    if f.shape[0] == 0 or t.size < 2:
        return {"IEA": np.nan, "KSV": np.nan, "Phase_Shift": np.nan}

    order = np.argsort(np.where(np.isfinite(t), t, np.inf))
    t = t[order]
    f = f[:, order]
    log_time = _log10_time_axis(t)

    valid_rows = np.sum(np.isfinite(f), axis=1) >= 2
    f = f[valid_rows]
    if f.shape[0] == 0:
        return {"IEA": np.nan, "KSV": np.nan, "Phase_Shift": np.nan}

    envelope_width = np.nanmax(f, axis=0) - np.nanmin(f, axis=0)
    iea = _trapz_finite(envelope_width, log_time)

    normalized, normalized_mask = normalize_ojip_curves(f, eps=normalize_eps)
    valid_normalized = normalized[normalized_mask]
    if valid_normalized.shape[0] >= 2:
        shape_sd = np.nanstd(valid_normalized, axis=0, ddof=0)
        ksv = _trapz_finite(shape_sd, log_time)
    elif valid_normalized.shape[0] == 1:
        ksv = 0.0
    else:
        ksv = np.nan

    tfm_values = []
    for curve in f:
        if not np.isfinite(curve).any():
            continue
        peak_index = int(np.nanargmax(curve))
        if np.isfinite(t[peak_index]):
            tfm_values.append(float(t[peak_index]))
    phase_shift = coefficient_of_variation(tfm_values, ddof=phase_cv_ddof)

    return {
        "IEA": float(iea) if np.isfinite(iea) else np.nan,
        "KSV": float(ksv) if np.isfinite(ksv) else np.nan,
        "Phase_Shift": float(phase_shift) if np.isfinite(phase_shift) else np.nan,
    }


__all__ = ["normalize_ojip_curves", "extract_ojip_features"]
