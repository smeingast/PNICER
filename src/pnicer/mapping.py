"""Extinction map construction (NICER-style smoothing, optional NICEST).

Point estimates are smoothed onto a WCS pixel grid with a spatial kernel and
inverse-variance weighting (Lombardi & Alves 2001; Lombardi 2018 Eq. 39). The
optional NICEST correction (Lombardi 2009) counteracts the bias from
unresolved substructure and foreground contamination. The pixel aggregation
math replicates the legacy PNICER implementation, including its 3-sigma
clipping and the NICEST bias-correction and variance forms.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

from pnicer.wcsgrid import data2grid, distance_sky

if TYPE_CHECKING:
    from astropy.units import Quantity

    from pnicer.catalog import ExtinctionCatalog

__all__ = ["ExtinctionMap", "build_map"]

_KERNEL_METRICS = ("gaussian", "epanechnikov", "triangular", "uniform")
_METRICS = (*_KERNEL_METRICS, "average", "median")
_STD2FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _kernel_weights(distances: np.ndarray, metric: str, bandwidth: float):
    """Spatial kernel weights, normalized like the legacy implementation."""
    if metric in ("uniform", "average", "median"):

        def wfunc(d):
            return np.ones_like(d)
    elif metric == "gaussian":

        def wfunc(d):
            return np.exp(-0.5 * (d / bandwidth) ** 2)
    elif metric == "epanechnikov":

        def wfunc(d):
            return np.maximum(1.0 - (d / bandwidth) ** 2, 0.0)
    elif metric == "triangular":

        def wfunc(d):
            return np.maximum(1.0 - np.abs(d / bandwidth), 0.0)
    else:
        raise ValueError(f"Metric '{metric}' not implemented")
    grid = np.arange(-100, 100, 0.01)
    norm = np.trapezoid(y=wfunc(grid), x=grid)
    return wfunc(distances) / norm


def _pixel_sums(values: np.ndarray, pixel_idx: np.ndarray, n_pixels: int):
    """Sum `values` per pixel, treating NaN as absent."""
    good = np.isfinite(values)
    # Cast: bincount returns int64 for empty weight arrays
    return np.bincount(
        pixel_idx[good], weights=values[good], minlength=n_pixels
    ).astype(np.float64)


def _pixel_counts(mask: np.ndarray, pixel_idx: np.ndarray, n_pixels: int):
    return np.bincount(pixel_idx[mask], minlength=n_pixels)


def build_map(
    catalog: ExtinctionCatalog,
    bandwidth: float | Quantity,
    *,
    metric: str = "gaussian",
    sampling: int = 2,
    use_fwhm: bool = False,
    nicest: bool = False,
    alpha: float = 1 / 3,
    nicest_k: float = 1.0,
    **kwargs,
) -> ExtinctionMap:
    """Build a smoothed extinction map from an `ExtinctionCatalog`.

    See `ExtinctionCatalog.build_map` for the parameter description.
    """
    from astropy.units import Quantity, deg

    if isinstance(bandwidth, Quantity):
        bandwidth = float(bandwidth.to_value(deg))
    if metric not in _METRICS:
        raise ValueError(f"Metric must be one of {_METRICS}")
    try:
        sampling = operator.index(sampling)
    except TypeError:
        raise ValueError("Sampling factor must be an integer") from None
    if use_fwhm and metric != "gaussian":
        raise ValueError("FWHM is only valid for the gaussian metric")
    if catalog.coordinates is None:
        raise ValueError("Map building requires source coordinates")

    # Pixel size uses the bandwidth as given; the kernel uses sigma
    pixsize = bandwidth / sampling
    if use_fwhm:
        bandwidth /= _STD2FWHM

    # Mask radius: half the legacy truncation scale
    trunc_radius = bandwidth if metric in ("average", "median") else 6 * bandwidth
    r_mask = trunc_radius / 2

    kwargs.setdefault("proj_code", "TAN")
    map_header, grid_lon, grid_lat = data2grid(
        catalog.coordinates, pixsize=pixsize, **kwargs
    )
    map_shape = grid_lon.shape
    n_pixels = grid_lon.size

    prime_header = fits.Header()
    prime_header["BWIDTH"] = (bandwidth, "Bandwidth of kernel (degrees)")
    prime_header["METRIC"] = (metric, "Metric used to create this map")
    prime_header["SAMPLING"] = (sampling, "Sampling factor of map")
    prime_header["NICEST"] = (nicest, "Whether NICEST was activated")

    # Pixel-source pairs within the mask radius (exact radius query),
    # processed in pixel blocks to bound the pair-array memory. Sources
    # with non-finite coordinates are excluded from the search.
    src_lon = catalog.coordinates.spherical.lon.degree
    src_lat = catalog.coordinates.spherical.lat.degree
    src_ok = np.flatnonzero(np.isfinite(src_lon) & np.isfinite(src_lat))
    tree = cKDTree(catalog.coordinates.cartesian.xyz.value.T[src_ok])
    flat_lon, flat_lat = grid_lon.ravel(), grid_lat.ravel()
    grid_xyz = np.column_stack(
        [
            np.cos(np.radians(flat_lat)) * np.cos(np.radians(flat_lon)),
            np.cos(np.radians(flat_lat)) * np.sin(np.radians(flat_lon)),
            np.sin(np.radians(flat_lat)),
        ]
    )
    chord = 2.0 * np.sin(np.radians(r_mask) / 2.0)

    map_ext = np.full(n_pixels, np.nan)
    map_var = np.full(n_pixels, np.nan)
    map_num = np.zeros(n_pixels, dtype=np.int64)
    map_rho = np.full(n_pixels, np.nan)

    block_size = 1 << 14
    for start in range(0, n_pixels, block_size):
        block = slice(start, min(start + block_size, n_pixels))
        n_block = block.stop - block.start
        neighbors = tree.query_ball_point(grid_xyz[block], r=chord, workers=-1)
        lengths = np.fromiter(
            (len(n) for n in neighbors), dtype=np.int64, count=n_block
        )
        pixel_idx = np.repeat(np.arange(n_block), lengths)
        source_idx = (
            src_ok[np.concatenate(neighbors).astype(np.int64)]
            if lengths.sum()
            else np.empty(0, dtype=np.int64)
        )

        distances = distance_sky(
            flat_lon[block][pixel_idx],
            flat_lat[block][pixel_idx],
            src_lon[source_idx],
            src_lat[source_idx],
        )
        w_spatial = _kernel_weights(distances, metric=metric, bandwidth=bandwidth)
        ext = catalog.extinction[source_idx]
        var = catalog.variance[source_idx]

        if metric == "average":
            results = _aggregate_average(ext, var, pixel_idx, n_block)
        elif metric == "median":
            results = _aggregate_median(ext, var, pixel_idx, n_block)
        else:
            results = _aggregate_kernel(
                ext,
                var,
                w_spatial,
                pixel_idx,
                n_block,
                nicest=nicest,
                alpha=alpha,
                k_lambda=nicest_k,
            )
        map_ext[block], map_var[block], map_num[block], map_rho[block] = results

    return ExtinctionMap(
        map_ext=map_ext.reshape(map_shape),
        map_var=map_var.reshape(map_shape),
        map_num=map_num.reshape(map_shape).astype(np.uint32),
        map_rho=map_rho.reshape(map_shape),
        map_header=map_header,
        prime_header=prime_header,
    )


def _aggregate_average(ext, var, pixel_idx, n_pixels):
    """Plain average with 3-sigma clipping (legacy 'average' metric)."""
    finite = np.isfinite(ext)
    num0 = _pixel_counts(finite, pixel_idx, n_pixels)
    mean0 = _pixel_sums(ext, pixel_idx, n_pixels) / np.where(num0, num0, 1)
    sq = _pixel_sums(ext**2, pixel_idx, n_pixels) / np.where(num0, num0, 1)
    std = np.sqrt(np.maximum(sq - mean0**2, 0.0))

    clipped = np.abs(ext - mean0[pixel_idx]) > 3 * std[pixel_idx]
    ext, var = ext.copy(), var.copy()
    ext[clipped], var[clipped] = np.nan, np.nan

    finite = np.isfinite(ext)
    num = _pixel_counts(finite, pixel_idx, n_pixels)
    with np.errstate(divide="ignore", invalid="ignore"):
        map_ext = _pixel_sums(ext, pixel_idx, n_pixels) / num
        map_var = _pixel_sums(var, pixel_idx, n_pixels) / num.astype(float) ** 2
    map_ext[num == 0] = np.nan
    map_rho = np.full(n_pixels, np.nan)
    return map_ext, map_var, num, map_rho


def _aggregate_median(ext, var, pixel_idx, n_pixels):
    """Median and MAD per pixel (legacy 'median' metric)."""
    map_ext = np.full(n_pixels, np.nan)
    map_var = np.full(n_pixels, np.nan)
    num = np.zeros(n_pixels, dtype=np.int64)

    finite = np.isfinite(ext)
    pix, values = pixel_idx[finite], ext[finite]
    order = np.argsort(pix, kind="stable")
    pix, values = pix[order], values[order]
    bounds = np.searchsorted(pix, np.arange(n_pixels + 1))
    for p in range(n_pixels):
        chunk = values[bounds[p] : bounds[p + 1]]
        if chunk.size:
            med = np.median(chunk)
            map_ext[p] = med
            map_var[p] = np.median(np.abs(chunk - med))
            num[p] = chunk.size
    map_rho = np.full(n_pixels, np.nan)
    return map_ext, map_var, num, map_rho


def _aggregate_kernel(
    ext, var, w_spatial, pixel_idx, n_pixels, *, nicest, alpha, k_lambda
):
    """Inverse-variance kernel smoothing with optional NICEST correction."""
    with np.errstate(divide="ignore", invalid="ignore"):
        w_total = w_spatial / var

    # First pass for the 3-sigma clipping
    sum_w = _pixel_sums(w_total, pixel_idx, n_pixels)
    sum_we = _pixel_sums(w_total * ext, pixel_idx, n_pixels)
    with np.errstate(divide="ignore", invalid="ignore"):
        ext0 = sum_we / sum_w
    finite = np.isfinite(ext)
    n_fin = _pixel_counts(finite, pixel_idx, n_pixels)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = _pixel_sums(ext, pixel_idx, n_pixels) / n_fin
        meansq = _pixel_sums(ext**2, pixel_idx, n_pixels) / n_fin
    std = np.sqrt(np.maximum(meansq - mean**2, 0.0))

    with np.errstate(invalid="ignore"):
        clipped = np.abs(ext - ext0[pixel_idx]) > 3 * std[pixel_idx]
    ext = np.where(clipped, np.nan, ext)
    var = np.where(clipped, np.nan, var)
    w_spatial = np.where(clipped, np.nan, w_spatial)
    w_total = np.where(clipped, np.nan, w_total)

    if nicest:
        beta = np.log(10.0) * alpha * k_lambda
        boost = 10.0 ** (alpha * k_lambda * ext)
        w_spatial = w_spatial * boost
        w_total = w_total * boost

        sum_w = _pixel_sums(w_total, pixel_idx, n_pixels)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Bias correction, Lombardi (2009) Eq. 34
            map_cor = beta * _pixel_sums(w_total * var, pixel_idx, n_pixels) / sum_w
            upper = _pixel_sums(
                w_total**2 * np.exp(2 * beta * ext) * (1 + beta * ext) ** 2 / var,
                pixel_idx,
                n_pixels,
            )
            lower = _pixel_sums(w_total * np.exp(beta * ext) / var, pixel_idx, n_pixels)
            map_var = upper / lower**2
    else:
        sum_w = _pixel_sums(w_total, pixel_idx, n_pixels)
        with np.errstate(divide="ignore", invalid="ignore"):
            map_var = _pixel_sums(w_total**2 * var, pixel_idx, n_pixels) / sum_w**2
        map_cor = np.zeros(n_pixels)

    valid_pair = np.isfinite(w_total * ext)
    num = _pixel_counts(valid_pair, pixel_idx, n_pixels)
    with np.errstate(divide="ignore", invalid="ignore"):
        map_ext = _pixel_sums(w_total * ext, pixel_idx, n_pixels) / sum_w - map_cor
    map_rho = _pixel_sums(w_spatial, pixel_idx, n_pixels)

    map_ext[num == 0] = np.nan
    map_rho[num == 0] = np.nan
    return map_ext, map_var, num, map_rho


class ExtinctionMap:
    """Gridded extinction map with variance, source count, and density planes.

    Parameters
    ----------
    map_ext, map_var : ndarray
        2-d extinction and variance maps.
    map_num : ndarray
        Number of sources per pixel.
    map_rho : ndarray
        Source density (sum of spatial weights) per pixel.
    map_header : fits.Header
        WCS header of the pixel grid.
    prime_header : fits.Header, optional
        Metadata header (bandwidth, metric, sampling, NICEST flag).
    """

    def __init__(
        self,
        map_ext: np.ndarray,
        map_var: np.ndarray,
        map_num: np.ndarray,
        map_rho: np.ndarray,
        map_header: fits.Header,
        prime_header: fits.Header | None = None,
    ) -> None:
        if map_ext.ndim != 2 or map_var.ndim != 2:
            raise ValueError("Extinction and variance maps must be 2D")
        self.map_ext = map_ext
        self.map_var = map_var
        self.map_num = map_num
        self.map_rho = map_rho
        self.map_header = map_header
        self.prime_header = fits.Header() if prime_header is None else prime_header

    @property
    def shape(self) -> tuple[int, int]:
        return self.map_ext.shape

    @property
    def map_err(self) -> np.ndarray:
        """1-sigma error map."""
        return np.sqrt(self.map_var)

    def save(self, path: str, overwrite: bool = True) -> None:
        """Write the map to FITS (primary + ext/var/num/rho image HDUs)."""
        hdulist = fits.HDUList(
            [
                fits.PrimaryHDU(header=self.prime_header),
                fits.ImageHDU(data=self.map_ext, header=self.map_header),
                fits.ImageHDU(data=self.map_var, header=self.map_header),
                fits.ImageHDU(
                    data=self.map_num.astype(np.int64), header=self.map_header
                ),
                fits.ImageHDU(data=self.map_rho, header=self.map_header),
            ]
        )
        hdulist.writeto(path, overwrite=overwrite)

    @classmethod
    def from_fits(cls, path: str) -> ExtinctionMap:
        """Read a map written by `save`."""
        with fits.open(path) as hdulist:
            return cls(
                map_ext=hdulist[1].data.astype(np.float64),
                map_var=hdulist[2].data.astype(np.float64),
                map_num=hdulist[3].data.astype(np.uint32),
                map_rho=hdulist[4].data.astype(np.float64),
                map_header=hdulist[1].header,
                prime_header=hdulist[0].header,
            )

    def plot(self, path: str | None = None, figsize: float = 10.0):
        """Plot the extinction and error maps (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "Plotting requires matplotlib; install with 'pip install pnicer[plot]'"
            ) from err
        from astropy.wcs import WCS

        wcs_proj = WCS(self.map_header)
        aspect = self.shape[0] / self.shape[1]
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(figsize, 2 * figsize * aspect + 1),
            subplot_kw={"projection": wcs_proj},
            constrained_layout=True,
        )
        for ax, data, label in (
            (axes[0], self.map_ext, "Extinction"),
            (axes[1], self.map_err, "Error"),
        ):
            finite = np.isfinite(data)
            vmin, vmax = (
                np.percentile(data[finite], [1, 99]) if finite.any() else (0, 1)
            )
            im = ax.imshow(
                data,
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap="binary",
            )
            fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        if path is None:
            plt.show()
        else:
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"
