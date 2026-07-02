"""Input data containers: band photometry and direct color measurements.

The estimators in this package operate in color space. Both containers expose
the same protocol: per-source observed color vectors grouped by missingness
pattern, each group carrying a projection matrix from the full color space and
per-source error covariance matrices (Lombardi 2018, Sect. 2.1 and 2.4).
Missing measurements are encoded as NaN throughout.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    from pnicer.catalog import ExtinctionCatalog
    from pnicer.model import IntrinsicColorModel
    from pnicer.posterior import ExtinctionPosterior

__all__ = ["Colors", "PatternGroup", "Photometry"]


@dataclass(frozen=True)
class PatternGroup:
    """Sources sharing one missingness pattern, in observed color space.

    Attributes
    ----------
    key : tuple of int
        Indices of the observed features (bands for `Photometry`, colors for
        `Colors`) defining this pattern.
    indices : ndarray of int, shape (n,)
        Indices of the group's sources in the parent catalog.
    projection : ndarray, shape (d, D)
        Projection matrix P from the full D-dimensional color space to the
        d observed color combinations.
    colors : ndarray, shape (n, d)
        Observed color vectors.
    covariances : ndarray, shape (n, d, d)
        Per-source measurement error covariance matrices of `colors`.
    """

    key: tuple[int, ...]
    indices: np.ndarray
    projection: np.ndarray
    colors: np.ndarray
    covariances: np.ndarray

    @property
    def n_sources(self) -> int:
        return self.indices.size


def _as_2d_float(
    values: Mapping[str, np.ndarray], names: tuple[str, ...]
) -> np.ndarray:
    """Stack a name-keyed mapping of 1-d arrays into an (n_names, N) float array."""
    arrays = [np.asarray(values[name], dtype=np.float64).ravel() for name in names]
    lengths = {a.size for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"All input arrays must have equal length, got {lengths}")
    return np.vstack(arrays)


class _ColorSpaceBase:
    """Shared machinery for color-space inputs (see subclasses)."""

    # Subclasses set these in __init__:
    _colors: np.ndarray  # (N, D) with NaN for missing
    _reddening: np.ndarray  # (D,)
    _color_names: tuple[str, ...]
    coordinates: SkyCoord | None

    @property
    def n_sources(self) -> int:
        """Number of sources in the catalog."""
        return self._colors.shape[0]

    @property
    def n_colors(self) -> int:
        """Dimensionality D of the full color space."""
        return self._colors.shape[1]

    @property
    def color_names(self) -> tuple[str, ...]:
        """Names of the full color-space axes."""
        return self._color_names

    @property
    def colors(self) -> np.ndarray:
        """Full color vectors, shape (N, D); NaN where not measurable."""
        return self._colors.copy()

    @property
    def raw_colors(self) -> np.ndarray:
        """Color vectors masked by value validity only, shape (N, D).

        Unlike `colors`, entries with missing *errors* are kept. Used for
        sample statistics of the color distribution (e.g. the NICER control
        field), where error estimates are not required. Subclasses override
        this where value and error validity differ.
        """
        return self.colors

    @property
    def reddening_vector(self) -> np.ndarray:
        """Reddening vector k in color space, shape (D,)."""
        return self._reddening.copy()

    def pattern_groups(self, min_dim: int = 1) -> list[PatternGroup]:
        """Group sources by missingness pattern.

        Parameters
        ----------
        min_dim : int
            Minimum dimensionality of the observed color space for a source
            to be included. Sources below this yield no estimate.

        Returns
        -------
        list of PatternGroup
        """
        raise NotImplementedError

    def _require_coordinates(self) -> SkyCoord:
        if self.coordinates is None:
            raise ValueError(
                "This operation requires source coordinates, but none were given"
            )
        return self.coordinates

    # ------------------------------------------------------------------ #
    # Estimator entry points (implemented in their own modules)
    # ------------------------------------------------------------------ #

    def nicer(
        self,
        control: _ColorSpaceBase | None = None,
        *,
        color0: np.ndarray | None = None,
        color0_cov: np.ndarray | None = None,
        min_dim: int = 1,
    ) -> ExtinctionCatalog:
        """NICER extinction estimates (Lombardi & Alves 2001).

        Parameters
        ----------
        control : Photometry or Colors, optional
            Extinction-free control field. Alternatively give `color0`.
        color0 : ndarray, optional
            Intrinsic colors, shape (D,), used when no control field is given.
        color0_cov : ndarray, optional
            Covariance of the intrinsic colors, shape (D, D) or (D,) for a
            diagonal. Zero if omitted.
        min_dim : int
            Minimum number of observed colors required per source.

        Returns
        -------
        ExtinctionCatalog
        """
        from pnicer.nicer import nicer

        return nicer(
            self,
            control=control,
            color0=color0,
            color0_cov=color0_cov,
            min_dim=min_dim,
        )

    def fit_intrinsic_colors(
        self,
        n_components: int | str = 5,
        *,
        max_components: int = 8,
        random_state: int | None = None,
        **kwargs,
    ) -> IntrinsicColorModel:
        """Fit the intrinsic color distribution of this (control) field.

        Uses extreme deconvolution (Bovy et al. 2011): a Gaussian mixture in
        the full color space, deconvolved from the per-source measurement
        errors. See `pnicer.model.IntrinsicColorModel.fit`.
        """
        from pnicer.model import IntrinsicColorModel

        return IntrinsicColorModel.fit(
            self,
            n_components=n_components,
            max_components=max_components,
            random_state=random_state,
            **kwargs,
        )

    def pnicer(
        self,
        control: _ColorSpaceBase | IntrinsicColorModel,
        *,
        adaptive: bool = False,
        **kwargs,
    ) -> ExtinctionPosterior:
        """PNICER extinction posteriors (Meingast+ 2017; Lombardi 2018).

        Parameters
        ----------
        control : Photometry, Colors, or IntrinsicColorModel
            Control field, or an already fitted intrinsic color model.
        adaptive : bool
            Apply the adaptive control-field iterations correcting for the
            extinction-dependent population change (Lombardi 2018,
            Sect. 2.6). Requires band-based (magnitude) control data.
        **kwargs
            Passed to `IntrinsicColorModel.posterior` (and, when `control`
            is not yet a model, to `fit_intrinsic_colors`).

        Returns
        -------
        ExtinctionPosterior
        """
        from pnicer.model import IntrinsicColorModel

        if not isinstance(control, IntrinsicColorModel):
            fit_keys = (
                "n_components",
                "max_components",
                "random_state",
                "reg_covar",
                "tol",
                "max_iter",
                "min_sources",
            )
            fit_kwargs = {k: kwargs.pop(k) for k in fit_keys if k in kwargs}
            control = control.fit_intrinsic_colors(**fit_kwargs)
        return control.posterior(self, adaptive=adaptive, **kwargs)


class Photometry(_ColorSpaceBase):
    """Band photometry with errors, an extinction law, and coordinates.

    Colors are formed from consecutive bands (c_i = m_i - m_{i+1}); band
    order therefore defines the color basis. For sources with missing bands,
    colors chain across the gaps, encoded by per-pattern projection matrices.

    Parameters
    ----------
    magnitudes : mapping of str to array
        Band magnitudes, one 1-d array per band. Iteration order of the
        mapping defines the band order; NaN marks missing measurements.
    errors : mapping of str to array
        1-sigma magnitude errors, same keys and lengths. NaN or non-positive
        errors mark the measurement as missing.
    extinction : mapping of str to float
        Extinction-law coefficients A_band / A_ref for each band (e.g.
        ``{"J": 2.5, "H": 1.55, "Ks": 1.0}`` normalized to Ks).
    coordinates : SkyCoord, optional
        Source coordinates; required for map building.
    """

    def __init__(
        self,
        magnitudes: Mapping[str, np.ndarray],
        errors: Mapping[str, np.ndarray],
        extinction: Mapping[str, float],
        coordinates: SkyCoord | None = None,
    ) -> None:
        names = tuple(magnitudes.keys())
        if len(names) < 2:
            raise ValueError("At least two bands are required")
        for source, label in ((errors, "errors"), (extinction, "extinction")):
            missing = set(names) - set(source.keys())
            if missing:
                raise ValueError(f"Bands {sorted(missing)} missing from {label}")

        self.band_names = names
        self._mags = _as_2d_float(magnitudes, names).T  # (N, n_bands)
        self._errs = _as_2d_float(errors, names).T
        if self._errs.shape != self._mags.shape:
            raise ValueError("Magnitude and error arrays must have equal length")
        self._extinction = np.array(
            [float(extinction[n]) for n in names], dtype=np.float64
        )

        if coordinates is not None and len(coordinates) != self._mags.shape[0]:
            raise ValueError("Coordinates must have one entry per source")
        self.coordinates = coordinates

        # Full color space: consecutive-band colors
        self._color_names = tuple(f"{a}-{b}" for a, b in itertools.pairwise(names))
        self._reddening = np.diff(self._extinction) * -1.0  # ext_i - ext_{i+1}
        with np.errstate(invalid="ignore"):
            self._raw_colors = self._mags[:, :-1] - self._mags[:, 1:]
        observed = self.observed_bands
        pair_ok = observed[:, :-1] & observed[:, 1:]
        colors = self._raw_colors.copy()
        colors[~pair_ok] = np.nan
        self._colors = colors

    @property
    def raw_colors(self) -> np.ndarray:
        """Colors masked by magnitude validity only (error columns ignored)."""
        return self._raw_colors.copy()

    @property
    def n_bands(self) -> int:
        """Number of photometric bands."""
        return len(self.band_names)

    @property
    def magnitudes(self) -> np.ndarray:
        """Band magnitudes, shape (N, n_bands); NaN = missing."""
        return self._mags.copy()

    @property
    def magnitude_errors(self) -> np.ndarray:
        """Band magnitude errors, shape (N, n_bands)."""
        return self._errs.copy()

    @property
    def extinction_vector(self) -> np.ndarray:
        """Extinction-law coefficients per band, shape (n_bands,)."""
        return self._extinction.copy()

    @property
    def observed_bands(self) -> np.ndarray:
        """Boolean mask of usable measurements, shape (N, n_bands)."""
        with np.errstate(invalid="ignore"):
            return np.isfinite(self._mags) & np.isfinite(self._errs) & (self._errs > 0)

    def pattern_groups(self, min_dim: int = 1) -> list[PatternGroup]:
        observed = self.observed_bands
        groups: list[PatternGroup] = []
        # Encode each row's pattern as an integer for fast uniquing
        code = observed @ (1 << np.arange(self.n_bands, dtype=np.int64))
        for pattern_code in np.unique(code):
            band_idx = np.flatnonzero((pattern_code >> np.arange(self.n_bands)) & 1)
            if band_idx.size - 1 < min_dim:
                continue
            src_idx = np.flatnonzero(code == pattern_code)

            # Projection: observed color j = m_{s_j} - m_{s_{j+1}} = sum of
            # full consecutive colors from s_j to s_{j+1}-1
            d = band_idx.size - 1
            proj = np.zeros((d, self.n_colors), dtype=np.float64)
            for j in range(d):
                proj[j, band_idx[j] : band_idx[j + 1]] = 1.0

            mags = self._mags[np.ix_(src_idx, band_idx)]
            errs = self._errs[np.ix_(src_idx, band_idx)]
            colors = mags[:, :-1] - mags[:, 1:]

            # Tridiagonal error covariance of consecutive colors (L18 Eq. 6)
            var = errs**2
            cov = np.zeros((src_idx.size, d, d), dtype=np.float64)
            diag = np.arange(d)
            cov[:, diag, diag] = var[:, :-1] + var[:, 1:]
            if d > 1:
                off = np.arange(d - 1)
                cov[:, off, off + 1] = -var[:, 1:-1]
                cov[:, off + 1, off] = -var[:, 1:-1]

            groups.append(
                PatternGroup(
                    key=tuple(int(b) for b in band_idx),
                    indices=src_idx,
                    projection=proj,
                    colors=colors,
                    covariances=cov,
                )
            )
        return groups

    @classmethod
    def from_table(
        cls,
        table,
        bands: Mapping[str, tuple[str, str]],
        extinction: Mapping[str, float],
        lon: str | None = None,
        lat: str | None = None,
        frame: str = "icrs",
        unit: str = "deg",
    ) -> Photometry:
        """Build a `Photometry` instance from a table-like object.

        Parameters
        ----------
        table
            Anything supporting column access by name (astropy Table,
            FITS record array, pandas DataFrame, dict of arrays).
        bands : mapping of str to (str, str)
            Band name -> (magnitude column, error column); order defines the
            color basis.
        extinction : mapping of str to float
            Extinction-law coefficients per band.
        lon, lat : str, optional
            Column names of the source coordinates.
        frame : str
            Coordinate frame, e.g. "icrs" or "galactic".
        unit : str
            Unit of the coordinate columns.
        """
        magnitudes = {name: np.asarray(table[m]) for name, (m, _) in bands.items()}
        errors = {name: np.asarray(table[e]) for name, (_, e) in bands.items()}
        coordinates = None
        if lon is not None and lat is not None:
            kw = (
                {"l": table[lon], "b": table[lat]}
                if frame == "galactic"
                else {"ra": table[lon], "dec": table[lat]}
            )
            coordinates = SkyCoord(frame=frame, unit=unit, **kw)
        return cls(
            magnitudes=magnitudes,
            errors=errors,
            extinction=extinction,
            coordinates=coordinates,
        )

    @classmethod
    def from_fits(
        cls,
        path: str,
        bands: Mapping[str, tuple[str, str]],
        extinction: Mapping[str, float],
        extension: int = 1,
        lon: str | None = None,
        lat: str | None = None,
        frame: str = "icrs",
        unit: str = "deg",
    ) -> Photometry:
        """Build a `Photometry` instance from a FITS table; see `from_table`."""
        from astropy.io import fits

        with fits.open(path) as hdul:
            return cls.from_table(
                hdul[extension].data,
                bands=bands,
                extinction=extinction,
                lon=lon,
                lat=lat,
                frame=frame,
                unit=unit,
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(bands={list(self.band_names)}, "
            f"n_sources={self.n_sources})"
        )


class Colors(_ColorSpaceBase):
    """Direct color measurements with independent errors.

    Use this when only colors (not band magnitudes) are available. Each color
    is treated as an independent measurement: missing colors simply drop that
    dimension (no chaining), and error covariances are diagonal. The adaptive
    population correction is unavailable without magnitudes.

    Parameters
    ----------
    colors : mapping of str to array
        Color measurements, one 1-d array per color; NaN = missing.
    errors : mapping of str to array
        1-sigma color errors, same keys and lengths.
    reddening : mapping of str to float
        Reddening-vector component per color (e.g. ``{"J-H": 0.95}`` for
        E(J-H)/A_ref).
    coordinates : SkyCoord, optional
        Source coordinates; required for map building.
    """

    def __init__(
        self,
        colors: Mapping[str, np.ndarray],
        errors: Mapping[str, np.ndarray],
        reddening: Mapping[str, float],
        coordinates: SkyCoord | None = None,
    ) -> None:
        names = tuple(colors.keys())
        if len(names) < 1:
            raise ValueError("At least one color is required")
        for source, label in ((errors, "errors"), (reddening, "reddening")):
            missing = set(names) - set(source.keys())
            if missing:
                raise ValueError(f"Colors {sorted(missing)} missing from {label}")

        self._color_names = names
        values = _as_2d_float(colors, names).T
        errs = _as_2d_float(errors, names).T
        if errs.shape != values.shape:
            raise ValueError("Color and error arrays must have equal length")
        self._raw_values = values.copy()
        with np.errstate(invalid="ignore"):
            bad = ~(np.isfinite(values) & np.isfinite(errs) & (errs > 0))
        values = values.copy()
        values[bad] = np.nan
        self._colors = values
        self._color_errors = errs
        self._reddening = np.array(
            [float(reddening[n]) for n in names], dtype=np.float64
        )

        if coordinates is not None and len(coordinates) != values.shape[0]:
            raise ValueError("Coordinates must have one entry per source")
        self.coordinates = coordinates

    @property
    def raw_colors(self) -> np.ndarray:
        """Colors masked by value validity only (error columns ignored)."""
        return self._raw_values.copy()

    @property
    def color_errors(self) -> np.ndarray:
        """Color errors, shape (N, D)."""
        return self._color_errors.copy()

    def pattern_groups(self, min_dim: int = 1) -> list[PatternGroup]:
        observed = np.isfinite(self._colors)
        groups: list[PatternGroup] = []
        code = observed @ (1 << np.arange(self.n_colors, dtype=np.int64))
        for pattern_code in np.unique(code):
            color_idx = np.flatnonzero((pattern_code >> np.arange(self.n_colors)) & 1)
            if color_idx.size < min_dim:
                continue
            src_idx = np.flatnonzero(code == pattern_code)

            d = color_idx.size
            proj = np.zeros((d, self.n_colors), dtype=np.float64)
            proj[np.arange(d), color_idx] = 1.0

            values = self._colors[np.ix_(src_idx, color_idx)]
            errs = self._color_errors[np.ix_(src_idx, color_idx)]
            cov = np.zeros((src_idx.size, d, d), dtype=np.float64)
            cov[:, np.arange(d), np.arange(d)] = errs**2

            groups.append(
                PatternGroup(
                    key=tuple(int(c) for c in color_idx),
                    indices=src_idx,
                    projection=proj,
                    colors=values,
                    covariances=cov,
                )
            )
        return groups

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(colors={list(self._color_names)}, "
            f"n_sources={self.n_sources})"
        )
