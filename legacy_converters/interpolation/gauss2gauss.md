# Gauss2Gauss — GPU-friendly sparse HEALPix regridding with CG “deconvolution”

## Overview

`Gauss2Gauss` converts **unstructured geolocated samples** into a **HEALPix-grid field** at a target resolution, using:

1. **Local neighbourhood selection** on the HEALPix grid (no dense *N×npix* distance matrix),
2. **Gaussian weights** to build two sparse operators:
   - **M**  : maps **samples → HEALPix** (aggregation / gridding reference)
   - **MT** : maps **HEALPix → samples** (back-projection / interpolation)
3. A **Conjugate Gradient (CG)** solver on normal equations to estimate a HEALPix field `hval` whose projection matches the input samples in a least-squares sense.

The implementation is designed to be **GPU-first** for the expensive parts (weighting, normalization, sparse ops, CG iterations), while relying on `healpix_geo` on CPU for HEALPix indexing and neighbourhood queries.

---

## Dependencies

- `torch` (CUDA strongly recommended)
- `numpy`
- `healpix-geo` (for HEALPix neighbourhoods and pixel centers)

```bash
pip install healpix-geo
```

---

## Key concepts

### Target HEALPix resolution

The class works at a HEALPix depth $level$, which corresponds to:

- $nside = 2**level$
- $npix = 12 * nside**2 = 12 * 4**level$

### Gaussian weights

Each sample point is connected to a small set of nearby HEALPix cells (pixel centers).  
For a sample `i` and a candidate cell `k`:

$w_{i,k} = \exp\left(-2 \frac{d_{i,k}^2}{\sigma^2}\right)$

- $d_{i,k}$ is the great-circle distance (meters) from the sample to the pixel center.
- By default, `sigma` is derived from the HEALPix pixel area:

$\sigma = \sqrt{\frac{4\pi}{12\cdot 4^{level}}}\,R$

with $R = 6371000 m$ by default.

You can override with $\sigma_m$ in meters.

---

## HEALPix cell selection (thresholding)

Computing weights to **all** HEALPix pixels is not feasible. Instead:

1. For each sample, query a **local HEALPix neighbourhood** using `kth_neighbourhood` with radius `ring_weight`.
2. Compute weights from the sample to the candidate pixel centers.
3. Accumulate a **global weight sum per pixel**.
4. Keep only pixels where:

$\sum_i w_{i,k} \ge \text{threshold}$

The kept pixels are returned as $cell_ids$ (size $K$).

### Handling invalid neighbours (-1)

`healpix_geo.kth_neighbourhood` may return `-1` for invalid neighbour slots.
The implementation replaces `-1` values by a **valid neighbour** (e.g., last valid in the row, fallback to the center pixel).  
This keeps arrays dense and avoids downstream failures; duplicates are handled naturally by weight accumulation and normalization.

---

## Sparse operators M and MT

Let:

- `N` = number of samples
- `K` = number of kept HEALPix cells
- `Npt` = number of nearest kept cells retained per sample

The class builds:

- `M`  : sparse CSR matrix of shape **(N, K)**
- `MT` : sparse CSR matrix of shape **(K, N)**

They are built using the per-sample nearest indices `hi` (shape `(N, Npt)`) and weights `w`.

### Normalization

To make the mapping stable:

- `M` is **column-normalized** (per HEALPix cell): each cell collects contributions whose weights sum to ~1.
- `MT` is **row-normalized** (per sample): each sample distributes its influence over neighbouring cells with weights summing to ~1.

---

## Least-squares estimate with CG (deconvolution-style)

Given sample values `val`:

- shape `(N,)` for one field, or `(B, N)` for batch fields

The algorithm uses a reference (fast) gridding:

$x_{ref} = y \; M$

Then it solves for a correction `delta` with CG (normal equations, damped):

$(MT\,M + \lambda I)\,\delta = MT\,(y - x_{ref}\,MT)$

and returns:

$hval = x_{ref} + \delta$

Here $\lambda$ corresponds to the damping / regularization parameter (`lam`).

---

## API

### `Gauss2Gauss(lon_deg, lat_deg, Npt, level, ...)`

Builds the geometry and sparse operators.

**Inputs**
- `lon_deg`, `lat_deg`: arrays of shape `(N,)`, **degrees**
- `level`: HEALPix depth
- `Npt`: number of nearest kept cells per sample used in the sparse links

**Important options**
- `sigma_m`: Gaussian length scale in meters (default: derived from `level`)
- `threshold`: global pixel weight-sum threshold for keeping cells
- `nest`: HEALPix scheme (`True` nested, `False` ring)
- `device`, `dtype`: torch compute settings
- `ring_weight`: neighbourhood radius for thresholding stage
- `ring_search_init`, `ring_search_max`: neighbourhood radius used when selecting nearest kept cells

**Key attributes**
- `cell_ids`: `(K,)` kept HEALPix pixel ids
- `hi`: `(N, Npt)` indices into `cell_ids` for each sample
- `d_m`: `(N, Npt)` distances (meters)
- `M`: `(N, K)` sparse CSR tensor on `device`
- `MT`: `(K, N)` sparse CSR tensor on `device`

---

### `fit(val, lam=0.0, max_iter=100, tol=1e-8) -> hval`

Fits the HEALPix field from input samples.

- `val`: `(N,)` or `(B, N)`
- returns `hval`: `(K,)` or `(B, K)`

`lam` is the damping / Tikhonov regularization used in CG.

---

### `transform(hval) -> val_hat`

Projects a HEALPix field back to the original sample locations.

- `hval`: `(K,)` or `(B, K)`
- returns `val_hat`: `(N,)` or `(B, N)`

---

### `fit_transform(val, ...) -> (hval, tilde_val)`

Convenience helper:

- returns `(hval, tilde_val)`
- where `tilde_val = transform(hval)` can be compared directly with `val`.

---

## Examples

### Single field `(N,)`

```python
import numpy as np
import torch
from gauss2gauss import Gauss2Gauss

lon = np.asarray(lon_deg)  # (N,)
lat = np.asarray(lat_deg)  # (N,)
val = np.asarray(val)      # (N,)

g2g = Gauss2Gauss(lon, lat, Npt=9, level=9, threshold=0.1,
                 device="cuda", dtype=torch.float64)

hval = g2g.fit(val, lam=0.1, max_iter=100, tol=1e-8)  # (K,)
val_hat = g2g.transform(hval)                         # (N,)
```

### Batched fields `(B, N)`

```python
valB = np.asarray(valB)  # (B, N)

hvalB, val_hatB = g2g.fit_transform(valB, lam=0.1, max_iter=200, tol=1e-8)
# hvalB: (B, K), val_hatB: (B, N)
```

---

## Performance tips

- Build `Gauss2Gauss` **once** for a given `(lon, lat, level, Npt)` geometry, then call `.fit()` for many variables.
- Use CUDA for large problems (`device="cuda"`).
- If no cells survive thresholding, lower `threshold` or increase neighbourhood radii.
- If some points cannot find `Npt` neighbours among the kept cells, reduce `threshold` or increase `ring_search_max`.

---

## Notes on numerical stability

- Distances are computed from unit vectors and `acos(dot)` (stable if `dot` is clamped to `[-1, 1]`).
- `float64` is recommended when you require high accuracy; `float32` may be significantly faster on GPU.
