#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature extraction for leaf-ordered fluorescence parameters.

Author: zhangtong sun

This module contains calculation-only code. It does not read files, create
figures, generate images, or export results.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .common import clean_1d, r_squared

__author__ = "zhangtong sun"


def extract_leaf_parameter_features(
    values: Sequence[float],
    layer_size: int = 3,
    cv_ddof: int = 0,
) -> Dict[str, float]:
    """Extract five features from an inner-to-outer leaf parameter sequence.

    Features
    --------
    CV
        Coefficient of variation across leaves.
    IO_Ratio
        Mean of inner leaves divided by mean of outer leaves.
    Slope_k
        Derivative at x=0.5 of the quadratic fit y=a*x^2+b*x+c.
    Quad_a
        Quadratic coefficient a.
    Quad_b
        Linear coefficient b.
    """
    if layer_size < 1:
        raise ValueError("layer_size must be at least 1.")
    if cv_ddof < 0:
        raise ValueError("cv_ddof must be non-negative.")

    x_values = clean_1d(values)
    n = int(x_values.size)

    features: Dict[str, float] = {
        "CV": np.nan,
        "IO_Ratio": np.nan,
        "Slope_k": np.nan,
        "Quad_a": np.nan,
        "Quad_b": np.nan,
    }
    if n == 0:
        return features

    mean_value = float(np.mean(x_values))
    if n > cv_ddof and mean_value != 0.0:
        features["CV"] = float(np.std(x_values, ddof=cv_ddof) / mean_value)

    effective_layer_size = min(layer_size, max(1, n // 2))
    inner_mean = float(np.mean(x_values[:effective_layer_size]))
    outer_mean = float(np.mean(x_values[-effective_layer_size:]))
    if outer_mean != 0.0:
        features["IO_Ratio"] = float(inner_mean / outer_mean)

    if n >= 3:
        leaf_position = np.linspace(0.0, 1.0, n)
        a, b, _c = np.polyfit(leaf_position, x_values, 2)
        features["Slope_k"] = float(a + b)
        features["Quad_a"] = float(a)
        features["Quad_b"] = float(b)
    elif n == 2:
        leaf_position = np.array([0.0, 1.0])
        slope, _intercept = np.polyfit(leaf_position, x_values, 1)
        features["Slope_k"] = float(slope)
        features["Quad_a"] = 0.0
        features["Quad_b"] = float(slope)

    return features


def leaf_parameter_fit_diagnostics(values: Sequence[float]) -> Dict[str, float]:
    """Return compact diagnostics for the leaf-order regression model."""
    y = clean_1d(values)
    n = int(y.size)
    diagnostics: Dict[str, float] = {
        "Leaf_Count": float(n),
        "Fit_R2": np.nan,
        "Quad_c": np.nan,
        "Vertex_x": np.nan,
        "Vertex_y": np.nan,
    }
    if n < 2:
        return diagnostics

    x = np.linspace(0.0, 1.0, n)
    if n >= 3:
        a, b, c = np.polyfit(x, y, 2)
        y_hat = a * x**2 + b * x + c
        diagnostics["Fit_R2"] = r_squared(y, y_hat)
        diagnostics["Quad_c"] = float(c)
        if abs(a) > 1e-12:
            vertex_x = float(-b / (2.0 * a))
            diagnostics["Vertex_x"] = vertex_x
            diagnostics["Vertex_y"] = float(a * vertex_x**2 + b * vertex_x + c)
    else:
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        diagnostics["Fit_R2"] = r_squared(y, y_hat)
        diagnostics["Quad_c"] = float(intercept)

    return diagnostics


__all__ = ["extract_leaf_parameter_features", "leaf_parameter_fit_diagnostics"]
