# Fluorescence Feature Extraction

**Author:** zhangtong sun

This repository contains only the numerical feature-extraction code retained from the original analysis workflow. 

## File structure

* `fluorescence\_features/common.py` — shared numerical helpers.
* `fluorescence\_features/leaf\_parameter\_features.py` — features from fluorescence parameters ordered from inner to outer leaves.
* `fluorescence\_features/ojip\_features.py` — features from aligned OJIP fluorescence induction curves.
* `fluorescence\_features/spatial\_map\_features.py` — distribution features from already decoded numerical fluorescence maps.
* `fluorescence\_features/\_\_init\_\_.py` — convenient public imports.

## Dependency

* Python 3.8+
* NumPy

## Feature groups

### 1\. Leaf-ordered fluorescence parameters

`extract\_leaf\_parameter\_features()` calculates:

* `CV`
* `IO\_Ratio`
* `Slope\_k`
* `Quad\_a`
* `Quad\_b`

`leaf\_parameter\_fit\_diagnostics()` optionally returns model diagnostics such as `Fit\_R2`, `Quad\_c`, and quadratic-vertex information.

### 2\. OJIP fluorescence induction curves

`extract\_ojip\_features()` calculates:

* `IEA` — Integrated Envelope Area.
* `KSV` — Kinetic Shape Variance.
* `Phase\_Shift` — coefficient of variation of time-to-maximum fluorescence.

The input curves must already be aligned to a common time axis.

### 3\. Spatial fluorescence maps

`extract\_spatial\_map\_features()` accepts a 2D numerical array that has already been decoded from any original image representation. It calculates interval fractions and statistical distribution features including mean, median, SD, CV, skewness, Fisher excess kurtosis, quantiles, IQR, tail fractions, and Shannon entropy.

## Minimal usage

```python
import numpy as np
from fluorescence\_features import (
    extract\_leaf\_parameter\_features,
    extract\_ojip\_features,
    extract\_spatial\_map\_features,
)

leaf\_features = extract\_leaf\_parameter\_features(\[0.72, 0.75, 0.77, 0.73, 0.70])

time = np.array(\[0.01, 0.1, 1.0, 10.0])
curves = np.array(\[
    \[100, 180, 420, 390],
    \[105, 190, 410, 380],
    \[98, 175, 430, 400],
])
ojip\_features = extract\_ojip\_features(time, curves)

value\_map = np.array(\[
    \[0.71, 0.75, 0.79],
    \[0.68, 0.82, 0.85],
])
map\_features = extract\_spatial\_map\_features(value\_map, prefix="Fv")
```

## Scope

The code is intentionally limited to feature calculation. Data loading, preprocessing from proprietary file formats, image reconstruction, visualization, and result saving should be implemented separately by downstream users if needed.

