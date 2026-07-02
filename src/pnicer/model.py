"""The intrinsic color model: a deconvolved GMM of a control field.

This is the trained object of the PNICER 2.0 pipeline: it holds the Gaussian
mixture describing the intrinsic (error-free, extinction-free) color
distribution of the control field, the per-source responsibilities of the
fit, and — for band-based input — the completeness model enabling the
adaptive population correction (Lombardi 2018, Sect. 2.6).
"""

from __future__ import annotations

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
        self, a_grid: np.ndarray, floor: float = 0.01
    ) -> np.ndarray:
        """Component weights of the observable population under extinction.

        For each extinction value in `a_grid`, control-field sources are
        re-weighted by their relative detection probability (survival) and
        the mixture weights are re-estimated from the fit responsibilities;
        means and covariances stay fixed (Lombardi 2018, Sect. 2.6).

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
        survival = self.completeness.survival(
            self.control_magnitudes,
            self.control_observed,
            self.band_extinction,
            a_grid,
            floor=floor,
        )  # (N, G)
        resp = self.responsibilities
        used = np.isfinite(resp).all(axis=1)
        weighted = survival[used].T @ resp[used]  # (G, K)
        total = weighted.sum(axis=1, keepdims=True)
        return weighted / np.maximum(total, 1e-300)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _check_science(self, science: _ColorSpaceBase) -> None:
        if science.n_colors != self.n_colors:
            raise ValueError(
                f"Science color space ({science.n_colors}) does not match "
                f"the model ({self.n_colors})"
            )
        if tuple(science.color_names) != self.color_names and set(
            science.color_names
        ) != set(self.color_names):
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
            if adaptive_method == "exact":
                a_table = (
                    np.arange(-2.0, 8.0 + 1e-9, 0.02)
                    if a_grid is None
                    else np.asarray(a_grid, dtype=np.float64)
                )
                w_table = self.weights_at_extinction(a_table, floor=floor)
                return exact_adaptive_posterior(
                    terms,
                    weight_table=w_table,
                    a_table=a_table,
                    coordinates=coordinates,
                    extinction_vector=extinction_vector,
                )
            if adaptive_method == "iterative":
                if a_grid is None:
                    a_grid = np.arange(-1.0, 5.25, 0.25)
                w_at_a = self.weights_at_extinction(a_grid, floor=floor)
                log_w = adaptive_log_weights(
                    terms,
                    base_log_w,
                    w_at_a,
                    np.asarray(a_grid),
                    n_iter=n_iter,
                )
            else:
                raise ValueError("adaptive_method must be 'exact' or 'iterative'")
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
            w_at_a = self.weights_at_extinction(a_grid, floor=floor)
            log_weights = np.log(w_at_a + 1e-300)  # (G, K)
        else:
            log_weights = np.log(self.weights)

        from scipy.special import logsumexp

        log_pdf = terms.log_pdf_grid(a_grid, log_weights)
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
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_components={self.n_components}, "
            f"colors={list(self.color_names)}, "
            f"adaptive={self.supports_adaptive})"
        )
