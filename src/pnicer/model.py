"""The intrinsic color model: a deconvolved GMM of a control field.

This is the trained object of the PNICER 2.0 pipeline: it holds the Gaussian
mixture describing the intrinsic (error-free, extinction-free) color
distribution of the control field, the per-source responsibilities of the
fit, and — for band-based input — the completeness model enabling the
adaptive population correction (Lombardi 2018, Sect. 2.6).
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from pnicer.completeness import CompletenessModel
from pnicer.photometry import Photometry, _ColorSpaceBase
from pnicer.posterior import (
    ExtinctionPosterior,
    adaptive_log_weights,
    build_posterior,
    component_terms,
    exact_adaptive_posterior,
)
from pnicer.xd import XDResult, fit_xd

__all__ = ["IntrinsicColorModel"]


class PatternCoverageError(ValueError):
    """No control-field source is observed in all bands of a detection pattern."""


def _json_scalar(value):
    """JSON fallback for numpy scalars and arrays in `fit_info`."""
    if isinstance(value, np.generic | np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class IntrinsicColorModel:
    """Gaussian mixture model of intrinsic colors, fitted by extreme
    deconvolution.

    Build with `fit` (or `Photometry.fit_intrinsic_colors`); apply with
    `posterior` (or `Photometry.pnicer`).
    """

    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        color_names: tuple[str, ...],
        reddening_vector: np.ndarray,
        *,
        responsibilities: np.ndarray | None = None,
        completeness: CompletenessModel | None = None,
        control_magnitudes: np.ndarray | None = None,
        control_observed: np.ndarray | None = None,
        band_extinction: np.ndarray | None = None,
        fit_info: dict | None = None,
    ) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.covariances = np.asarray(covariances, dtype=np.float64)
        self.color_names = tuple(color_names)
        self.reddening_vector = np.asarray(reddening_vector, dtype=np.float64)
        self.responsibilities = responsibilities
        self.completeness = completeness
        self.control_magnitudes = control_magnitudes
        self.control_observed = control_observed
        self.band_extinction = band_extinction
        self.fit_info = fit_info or {}

    @property
    def n_components(self) -> int:
        return self.weights.size

    @property
    def n_colors(self) -> int:
        return self.means.shape[1]

    @property
    def supports_adaptive(self) -> bool:
        """Whether the adaptive population correction is available."""
        return (
            self.completeness is not None
            and self.responsibilities is not None
            and self.control_magnitudes is not None
        )

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    @classmethod
    def fit(
        cls,
        control: _ColorSpaceBase,
        n_components: int | str = 5,
        *,
        max_components: int = 8,
        random_state: int | None = None,
        reg_covar: float = 1e-6,
        tol: float = 1e-6,
        max_iter: int = 500,
        min_sources: int = 50,
        completeness: CompletenessModel | str | None = "fit",
    ) -> IntrinsicColorModel:
        """Fit the intrinsic color distribution of a control field.

        Parameters
        ----------
        control : Photometry or Colors
            Extinction-free control field.
        n_components : int or "bic"
            Number of mixture components (default 5, following Lombardi
            2018), or "bic" to select the number minimizing the Bayesian
            information criterion over ``1..max_components`` (BIC computed
            from the observed-data likelihood of the deconvolved model;
            note that rich color distributions often keep improving BIC
            with K, so the scan may rail at `max_components`).
        max_components : int
            Upper bound of the BIC scan.
        random_state : int, optional
            Seed for the initialization (makes fits reproducible).
        reg_covar, tol, max_iter
            Extreme-deconvolution EM settings; see `pnicer.xd.fit_xd`.
        min_sources : int
            Minimum number of usable control-field sources.
        completeness : CompletenessModel, "fit", or None
            Completeness model for the adaptive correction: "fit" derives it
            from the control-field number counts (band-based input only), a
            `CompletenessModel` is used as given, None disables it.

        Returns
        -------
        IntrinsicColorModel
        """
        groups = control.pattern_groups(min_dim=1)
        n_used = sum(g.n_sources for g in groups)
        if n_used < min_sources:
            raise ValueError(
                f"Control field has only {n_used} usable sources "
                f"(minimum {min_sources})"
            )

        if isinstance(n_components, str):
            if n_components.lower() != "bic":
                raise ValueError("n_components must be an integer or 'bic'")
            results: list[XDResult] = []
            for k in range(1, max_components + 1):
                results.append(
                    fit_xd(
                        groups,
                        n_dim=control.n_colors,
                        n_sources=control.n_sources,
                        n_components=k,
                        random_state=random_state,
                        reg_covar=reg_covar,
                        tol=tol,
                        max_iter=max_iter,
                    )
                )
            result = min(results, key=lambda r: r.bic)
        else:
            result = fit_xd(
                groups,
                n_dim=control.n_colors,
                n_sources=control.n_sources,
                n_components=int(n_components),
                random_state=random_state,
                reg_covar=reg_covar,
                tol=tol,
                max_iter=max_iter,
            )
        if not result.converged:
            warnings.warn(
                "Extreme deconvolution did not converge within max_iter",
                RuntimeWarning,
                stacklevel=2,
            )

        comp_model = None
        control_mags = None
        control_obs = None
        band_ext = None
        if isinstance(control, Photometry):
            control_mags = control.magnitudes
            control_obs = control.observed_bands
            band_ext = control.extinction_vector
            if completeness == "fit":
                try:
                    comp_model = CompletenessModel.fit(control_mags, control.band_names)
                except (ValueError, RuntimeError) as err:
                    warnings.warn(
                        f"Completeness fit failed ({err}); adaptive "
                        "correction unavailable",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            elif isinstance(completeness, CompletenessModel):
                comp_model = completeness
        elif completeness not in (None, "fit"):
            raise ValueError(
                "Explicit completeness requires band-based (Photometry) input"
            )

        return cls(
            weights=result.weights,
            means=result.means,
            covariances=result.covariances,
            color_names=control.color_names,
            reddening_vector=control.reddening_vector,
            responsibilities=result.responsibilities,
            completeness=comp_model,
            control_magnitudes=control_mags,
            control_observed=control_obs,
            band_extinction=band_ext,
            fit_info={
                "log_likelihood": result.log_likelihood,
                "bic": result.bic,
                "n_iter": result.n_iter,
                "converged": result.converged,
                "n_sources_used": n_used,
            },
        )

    # ------------------------------------------------------------------ #
    # Adaptive population correction
    # ------------------------------------------------------------------ #

    def weights_at_extinction(
        self,
        a_grid: np.ndarray,
        floor: float = 0.01,
        bands: np.ndarray | None = None,
    ) -> np.ndarray:
        """Component weights of the observable population under extinction.

        For each extinction value in `a_grid`, control-field sources are
        re-weighted by their relative detection probability (survival) and
        the mixture weights are re-estimated from the fit responsibilities;
        means and covariances stay fixed (Lombardi 2018, Sect. 2.6).

        Parameters
        ----------
        a_grid : ndarray, shape (G,)
            Extinction values (in units of A_ref).
        floor : float
            Completeness floor of the survival weights.
        bands : array of bool or int, optional
            Detection pattern to condition on. Only control sources observed
            in all of these bands contribute, and their survival is evaluated
            in these bands only: a science source detected in H and Ks alone
            stems from the population that survives in H and Ks, whether or
            not it is lost in J. The non-detection in the remaining bands is
            not used as information, so sources with incomplete photometry
            remain biased at low extinction. None (the pnicer 2.0.0 behavior)
            uses every control source with survival over all of its own
            observed bands.

        Returns
        -------
        ndarray, shape (G, K)
            Normalized component weights per grid point.
        """
        if not self.supports_adaptive:
            raise ValueError(
                "Adaptive correction requires a completeness model and "
                "band-based control data (fit from Photometry with "
                "completeness enabled)"
            )
        a_grid = np.asarray(a_grid, dtype=np.float64)
        resp = self.responsibilities
        used = np.isfinite(resp).all(axis=1)
        observed = np.asarray(self.control_observed, dtype=bool)
        if bands is not None:
            n_bands = observed.shape[1]
            bands = np.asarray(bands)
            if bands.dtype == bool:
                if bands.shape != (n_bands,):
                    raise ValueError(f"Boolean band mask must have {n_bands} entries")
                mask = bands
            else:
                if bands.size and not np.issubdtype(bands.dtype, np.integer):
                    raise ValueError("Band indices must be integers")
                if np.any((bands < 0) | (bands >= n_bands)):
                    raise ValueError(f"Band indices must lie in 0..{n_bands - 1}")
                mask = np.zeros(n_bands, dtype=bool)
                mask[bands.astype(np.int64)] = True
            if not mask.any():
                raise ValueError("At least one band is required")
            used &= observed[:, mask].all(axis=1)
            if not used.any():
                raise PatternCoverageError(
                    "No control-field source is observed in all bands of the "
                    "requested detection pattern"
                )
            observed = observed & mask[None, :]

        # Accumulate over source chunks: the (N, G) survival array would
        # otherwise dominate memory for large control fields / fine grids
        weighted = np.zeros((a_grid.size, self.n_components))
        chunk = max((1 << 23) // max(a_grid.size, 1), 1024)
        used_idx = np.flatnonzero(used)
        for start in range(0, used_idx.size, chunk):
            idx = used_idx[start : start + chunk]
            survival = self.completeness.survival(
                self.control_magnitudes[idx],
                observed[idx],
                self.band_extinction,
                a_grid,
                floor=floor,
            )  # (chunk, G)
            weighted += survival.T @ resp[idx]
        total = weighted.sum(axis=1)
        weights = weighted / np.maximum(total, 1e-300)[:, None]

        # Beyond the completeness cliff no source survives and the mixture
        # is unconstrained; freeze the population at the nearest grid point
        # that still has support instead of returning all-zero rows
        good = total > 1e-30
        if not good.all():
            if not good.any():
                raise ValueError(
                    "No control-field source survives anywhere on the "
                    "extinction grid; check the completeness model"
                )
            good_idx = np.flatnonzero(good)
            insertion = np.searchsorted(good_idx, np.arange(a_grid.size))
            left = good_idx[np.clip(insertion - 1, 0, good_idx.size - 1)]
            right = good_idx[np.clip(insertion, 0, good_idx.size - 1)]
            nearest = np.where(
                np.abs(np.arange(a_grid.size) - left)
                <= np.abs(right - np.arange(a_grid.size)),
                left,
                right,
            )
            weights[~good] = weights[nearest[~good]]
        return weights

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _check_science(self, science: _ColorSpaceBase) -> None:
        if science.n_colors != self.n_colors:
            raise ValueError(
                f"Science color space ({science.n_colors}) does not match "
                f"the model ({self.n_colors})"
            )
        if tuple(science.color_names) != self.color_names:
            if set(science.color_names) == set(self.color_names):
                # Same colors, different order: axes are misaligned and the
                # result would be silently wrong
                raise ValueError(
                    f"Science color order {science.color_names} does not "
                    f"match the model's {self.color_names}"
                )
            warnings.warn(
                f"Science colors {science.color_names} differ from model "
                f"colors {self.color_names}",
                UserWarning,
                stacklevel=3,
            )
        if not np.allclose(science.reddening_vector, self.reddening_vector):
            warnings.warn(
                "Science reddening vector differs from the model's",
                UserWarning,
                stacklevel=3,
            )

    def _detection_patterns(self, science: _ColorSpaceBase, pattern_conditioned: bool):
        """Science sources grouped by detection pattern for the adaptive step.

        Returns (bands, indices) pairs; bands is a boolean band mask, or None
        for a single group using the unconditioned population weights (inputs
        without band information, or ``pattern_conditioned=False``).
        """
        observed = getattr(science, "observed_bands", None)
        everyone = [(None, np.arange(science.n_sources))]
        if not pattern_conditioned or observed is None:
            return everyone
        if observed.shape[1] != self.control_observed.shape[1]:
            raise ValueError(
                "Science and control photometry have different numbers of bands"
            )
        n_bands = observed.shape[1]
        code = observed @ (1 << np.arange(n_bands, dtype=np.int64))
        groups = []
        for pattern in np.unique(code):
            bands = ((pattern >> np.arange(n_bands)) & 1).astype(bool)
            if bands.sum() < 2:
                continue  # no color, no estimate
            groups.append((bands, np.flatnonzero(code == pattern)))
        return groups

    def _pattern_weights(self, a_grid, floor, bands):
        """`weights_at_extinction` for one pattern, falling back to the
        unconditioned weights when no control source covers the pattern."""
        if bands is None:
            return self.weights_at_extinction(a_grid, floor=floor)
        try:
            return self.weights_at_extinction(a_grid, floor=floor, bands=bands)
        except PatternCoverageError:
            warnings.warn(
                f"No control source observed in bands {np.flatnonzero(bands)}; "
                "using unconditioned population weights for this pattern",
                RuntimeWarning,
                stacklevel=3,
            )
            return self.weights_at_extinction(a_grid, floor=floor)

    def posterior(
        self,
        science: _ColorSpaceBase,
        *,
        adaptive: bool = False,
        adaptive_method: str = "exact",
        n_iter: int = 3,
        a_grid: np.ndarray | None = None,
        floor: float = 0.01,
        min_dim: int = 1,
        pattern_conditioned: bool = True,
    ) -> ExtinctionPosterior:
        """Compute extinction posteriors for a science field.

        Parameters
        ----------
        science : Photometry or Colors
            Science-field data in the same color basis as the model.
        adaptive : bool
            Apply the adaptive population correction for the change of the
            observable background population with extinction (Lombardi 2018,
            Sect. 2.6); requires `supports_adaptive`.
        adaptive_method : str
            "exact" (default): extinction-dependent component weights enter
            the likelihood itself; per-component moments are integrated by
            Gauss-Hermite quadrature. "iterative": the published scheme of
            Lombardi (2018), which re-weights the mixture with the previous
            posterior; can bias broad posteriors (see
            `pnicer.posterior.exact_adaptive_posterior`).
        n_iter : int
            Number of iterations ("iterative" method only).
        a_grid : ndarray, optional
            Extinction grid on which the population weights are tabulated.
            Defaults to a fine grid over [-2, 8] ("exact") or a coarse grid
            over [-1, 5] ("iterative"), in units of A_ref.
        floor : float
            Completeness floor of the survival weights.
        min_dim : int
            Minimum observed color-space dimensionality per source.
        pattern_conditioned : bool
            Condition the adaptive population weights on each source's
            detection pattern (see `weights_at_extinction`). False reproduces
            pnicer 2.0.0, where all sources share one weight table.

        Returns
        -------
        ExtinctionPosterior
        """
        self._check_science(science)
        terms = component_terms(science, self.means, self.covariances, min_dim=min_dim)
        base_log_w = np.log(self.weights)
        coordinates = science.coordinates
        extinction_vector = getattr(science, "extinction_vector", None)

        if adaptive:
            if adaptive_method not in ("exact", "iterative"):
                raise ValueError("adaptive_method must be 'exact' or 'iterative'")
            if not self.supports_adaptive:
                # Raise the informative error before any grouping
                self.weights_at_extinction(np.zeros(1), floor=floor)
            patterns = self._detection_patterns(science, pattern_conditioned)
            if adaptive_method == "exact":
                a_table = (
                    np.arange(-2.0, 8.0 + 1e-9, 0.02)
                    if a_grid is None
                    else np.asarray(a_grid, dtype=np.float64)
                )
                shape = terms.a_mean.shape
                means = np.full(shape, np.nan)
                variances = np.full(shape, np.nan)
                log_weights = np.full(shape, np.nan)
                log_evidence = np.full(shape[0], np.nan)
                for bands, idx in patterns:
                    part = exact_adaptive_posterior(
                        terms.subset(idx),
                        weight_table=self._pattern_weights(a_table, floor, bands),
                        a_table=a_table,
                        coordinates=None,
                        extinction_vector=None,
                    )
                    means[idx] = part.means
                    variances[idx] = part.variances
                    log_weights[idx] = part.log_weights
                    log_evidence[idx] = part.log_evidence
                return ExtinctionPosterior(
                    means=means,
                    variances=variances,
                    log_weights=log_weights,
                    log_evidence=log_evidence,
                    pattern_dim=terms.pattern_dim,
                    coordinates=coordinates,
                    extinction_vector=extinction_vector,
                )
            if a_grid is None:
                a_grid = np.arange(-1.0, 5.25, 0.25)
            a_grid = np.asarray(a_grid, dtype=np.float64)
            log_w = np.full(terms.a_mean.shape, np.nan)
            for bands, idx in patterns:
                log_w[idx] = adaptive_log_weights(
                    terms.subset(idx),
                    base_log_w,
                    self._pattern_weights(a_grid, floor, bands),
                    a_grid,
                    n_iter=n_iter,
                )
        else:
            log_w = base_log_w

        return build_posterior(
            terms,
            log_w,
            coordinates=coordinates,
            extinction_vector=extinction_vector,
        )

    def posterior_grid(
        self,
        science: _ColorSpaceBase,
        a_grid: np.ndarray,
        *,
        adaptive: bool = False,
        floor: float = 0.01,
        min_dim: int = 1,
        pattern_conditioned: bool = True,
    ) -> np.ndarray:
        """Reference grid evaluation of the posteriors, shape (N, G).

        Evaluates p(A_g | data) directly on `a_grid`, normalized to unit
        sum over the grid. With ``adaptive=True`` the extinction-dependent
        component weights enter the likelihood itself — the "exact"
        counterpart of the iterative scheme, useful for validation.
        """
        self._check_science(science)
        a_grid = np.asarray(a_grid, dtype=np.float64)
        terms = component_terms(science, self.means, self.covariances, min_dim=min_dim)
        if adaptive:
            if not self.supports_adaptive:
                self.weights_at_extinction(np.zeros(1), floor=floor)
            log_pdf = np.full((science.n_sources, a_grid.size), np.nan)
            for bands, idx in self._detection_patterns(science, pattern_conditioned):
                w_at_a = self._pattern_weights(a_grid, floor, bands)
                log_pdf[idx] = terms.subset(idx).log_pdf_grid(
                    a_grid, np.log(w_at_a + 1e-300), grid_weights=True
                )
        else:
            log_pdf = terms.log_pdf_grid(a_grid, np.log(self.weights))

        from scipy.special import logsumexp

        with np.errstate(invalid="ignore"):
            log_pdf = log_pdf - logsumexp(
                np.nan_to_num(log_pdf, nan=-np.inf), axis=1, keepdims=True
            )
        return np.exp(log_pdf)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save the model to an .npz file."""
        payload = {
            "weights": self.weights,
            "means": self.means,
            "covariances": self.covariances,
            "color_names": np.array(self.color_names),
            "reddening_vector": self.reddening_vector,
        }
        if self.responsibilities is not None:
            payload["responsibilities"] = self.responsibilities
        if self.control_magnitudes is not None:
            payload["control_magnitudes"] = self.control_magnitudes
            payload["control_observed"] = self.control_observed
            payload["band_extinction"] = self.band_extinction
        if self.fit_info:
            payload["fit_info"] = np.array(
                json.dumps(self.fit_info, default=_json_scalar)
            )
        if self.completeness is not None:
            payload["completeness_bands"] = np.array(self.completeness.band_names)
            payload["completeness_params"] = np.array(
                [(b.m50, b.width, b.alpha, b.log_norm) for b in self.completeness.bands]
            )
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> IntrinsicColorModel:
        """Load a model saved with `save`."""
        from pnicer.completeness import BandCompleteness

        with np.load(path, allow_pickle=False) as data:
            completeness = None
            if "completeness_params" in data:
                bands = tuple(
                    BandCompleteness(
                        m50=float(p[0]),
                        width=float(p[1]),
                        alpha=float(p[2]),
                        log_norm=float(p[3]),
                    )
                    for p in data["completeness_params"]
                )
                completeness = CompletenessModel(
                    band_names=tuple(str(n) for n in data["completeness_bands"]),
                    bands=bands,
                )
            return cls(
                weights=data["weights"],
                means=data["means"],
                covariances=data["covariances"],
                color_names=tuple(str(n) for n in data["color_names"]),
                reddening_vector=data["reddening_vector"],
                responsibilities=data.get("responsibilities"),
                completeness=completeness,
                control_magnitudes=data.get("control_magnitudes"),
                control_observed=data.get("control_observed"),
                band_extinction=data.get("band_extinction"),
                fit_info=(
                    json.loads(str(data["fit_info"])) if "fit_info" in data else None
                ),
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_components={self.n_components}, "
            f"colors={list(self.color_names)}, "
            f"adaptive={self.supports_adaptive})"
        )
