"""Per-source point-estimate extinction catalogs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    from astropy.units import Quantity

    from pnicer.mapping import ExtinctionMap

__all__ = ["ExtinctionCatalog"]


class ExtinctionCatalog:
    """Extinction point estimates with variances for a source catalog.

    Produced by `Photometry.nicer` or `ExtinctionPosterior.discretize`.

    Parameters
    ----------
    extinction : ndarray, shape (N,)
        Extinction per source (in units of A_ref); NaN where no estimate.
    variance : ndarray, shape (N,)
        Variance of the estimate.
    coordinates : SkyCoord, optional
        Source coordinates; required for map building.
    extinction_vector : ndarray, optional
        Extinction-law coefficients per band of the parent data (used by the
        NICEST map correction).
    """

    def __init__(
        self,
        extinction: np.ndarray,
        variance: np.ndarray,
        coordinates: SkyCoord | None = None,
        extinction_vector: np.ndarray | None = None,
    ) -> None:
        self.extinction = np.asarray(extinction, dtype=np.float64)
        self.variance = np.asarray(variance, dtype=np.float64)
        if self.extinction.shape != self.variance.shape:
            raise ValueError("Extinction and variance shapes differ")
        if coordinates is not None and len(coordinates) != self.extinction.size:
            raise ValueError("Coordinates must have one entry per source")
        self.coordinates = coordinates
        self.extinction_vector = (
            None
            if extinction_vector is None
            else np.asarray(extinction_vector, dtype=np.float64)
        )

    @property
    def n_sources(self) -> int:
        return self.extinction.size

    @property
    def error(self) -> np.ndarray:
        """1-sigma error per source."""
        return np.sqrt(self.variance)

    def build_map(
        self,
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
        """Build a smoothed extinction map from the catalog.

        Parameters
        ----------
        bandwidth : float or Quantity
            Kernel bandwidth; floats are interpreted as degrees.
        metric : str
            Smoothing metric: "gaussian", "epanechnikov", "triangular",
            "uniform", "average", or "median".
        sampling : int
            Pixels per bandwidth (map pixel size = bandwidth / sampling).
        use_fwhm : bool
            Interpret `bandwidth` as the FWHM of the Gaussian kernel
            (gaussian metric only).
        nicest : bool
            Apply the NICEST correction (Lombardi 2009) for unresolved
            substructure and foreground bias.
        alpha : float
            NICEST number-count slope (1/3 for the NIR K band).
        nicest_k : float
            Extinction coefficient of the band the slope refers to, in the
            same units as the extinction estimates (1.0 when estimates are
            in units of the reference band, following Lombardi 2009). The
            legacy implementation used the maximum of the feature reddening
            vector instead, which made the correction depend on the input
            type (0.95 for the 2017 color-based setup).
        **kwargs
            Additional WCS options (e.g. ``proj_code="TAN"``).

        Returns
        -------
        ExtinctionMap
        """
        from pnicer.mapping import build_map

        return build_map(
            self,
            bandwidth=bandwidth,
            metric=metric,
            sampling=sampling,
            use_fwhm=use_fwhm,
            nicest=nicest,
            alpha=alpha,
            nicest_k=nicest_k,
            **kwargs,
        )

    def to_table(self):
        """Return the catalog as an astropy Table."""
        from astropy.table import Table

        columns = {
            "extinction": self.extinction,
            "variance": self.variance,
            "error": self.error,
        }
        if self.coordinates is not None:
            columns["lon"] = self.coordinates.spherical.lon.degree
            columns["lat"] = self.coordinates.spherical.lat.degree
        return Table(columns)

    def __repr__(self) -> str:
        n_good = int(np.isfinite(self.extinction).sum())
        return (
            f"{type(self).__name__}(n_sources={self.n_sources}, n_estimates={n_good})"
        )
