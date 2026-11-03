"""Tests of the adaptive control-field iterations (Lombardi 2018, Sect. 2.6).

The synthetic setup mirrors the paper's validation: a two-population field
(bright "stars", faint steep-counts "galaxies") observed through per-band
completeness functions. Extinction shifts the population mix; the adaptive
iterations must correct the resulting bias.
"""

import numpy as np
import pytest
from scipy.special import erfc

from pnicer import Photometry
from pnicer.completeness import CompletenessModel

BAND_NAMES = ("J", "H", "Ks")
BAND_EXT = np.array([2.5, 1.55, 1.0])
EXTINCTION = dict(zip(BAND_NAMES, BAND_EXT, strict=True))
M50 = np.array([17.5, 16.9, 16.3])
WIDTH = np.array([0.35, 0.35, 0.35])


def _completeness(mags, band):
    return 0.5 * erfc((mags - M50[band]) / (np.sqrt(2) * WIDTH[band]))


def _draw_population(rng, n, extinction=0.0):
    """Two intrinsic populations, observed through the completeness."""
    n_star = int(0.65 * n)
    n_gal = n - n_star
    # Stars: blue colors, shallow counts; galaxies: red colors, steep counts
    col_star = rng.multivariate_normal(
        [0.45, 0.15], np.diag([0.015, 0.008]), size=n_star
    )
    col_gal = rng.multivariate_normal([1.1, 0.75], np.diag([0.04, 0.03]), size=n_gal)
    ks_star = 16.5 - rng.exponential(1.0 / (0.30 * np.log(10)), size=n_star)
    ks_gal = 17.0 - rng.exponential(1.0 / (0.55 * np.log(10)), size=n_gal)
    colors = np.vstack([col_star, col_gal])
    ks = np.concatenate([ks_star, ks_gal])
    labels = np.concatenate([np.zeros(n_star, bool), np.ones(n_gal, bool)])

    h = ks + colors[:, 1]
    j = h + colors[:, 0]
    mags = np.column_stack([j, h, ks]) + extinction * BAND_EXT[None, :]

    errs = np.clip(
        0.02 + 0.04 * 10 ** (0.35 * (mags - (M50[None, :] - 1.0))), 0.02, 0.25
    )
    noisy = mags + rng.standard_normal(mags.shape) * errs

    # Band detection through the completeness functions
    for b in range(3):
        detected = rng.random(len(ks)) < _completeness(mags[:, b], b)
        noisy[~detected, b] = np.nan
    usable = np.isfinite(noisy).sum(axis=1) >= 2
    noisy, errs, labels = noisy[usable], errs[usable], labels[usable]

    phot = Photometry(
        magnitudes=dict(zip(BAND_NAMES, noisy.T, strict=True)),
        errors=dict(zip(BAND_NAMES, errs.T, strict=True)),
        extinction=EXTINCTION,
    )
    return phot, labels


@pytest.fixture(scope="module")
def fitted_population_model():
    rng = np.random.default_rng(7)
    control, _ = _draw_population(rng, 30000)
    completeness = CompletenessModel.from_parameters(BAND_NAMES, M50, WIDTH)
    return control.fit_intrinsic_colors(
        n_components=3, random_state=0, completeness=completeness
    )


@pytest.fixture
def population_model(fitted_population_model):
    """Shared fitted model plus a fresh generator per test (order-independent)."""
    return fitted_population_model, np.random.default_rng(8)


class TestWeightsAtExtinction:
    def test_zero_extinction_recovers_fit_weights(self, population_model):
        """Sources below the completeness floor deviate marginally."""
        model, _ = population_model
        w0 = model.weights_at_extinction(np.array([0.0]))[0]
        np.testing.assert_allclose(w0, model.weights, atol=1e-3)

    def test_faint_red_component_dies_first(self, population_model):
        """The galaxy-like (red, faint) component loses weight fastest."""
        model, _ = population_model
        gal = int(np.argmin(np.sum((model.means - [1.1, 0.75]) ** 2, axis=1)))
        w = model.weights_at_extinction(np.array([0.0, 1.0, 2.0]))
        rel = w[:, gal] / w[0, gal]
        assert rel[1] < 0.8
        assert rel[2] < rel[1]


class TestPatternConditioning:
    """Population weights conditioned on the science detection pattern."""

    def test_band_subset_matches_manual(self, population_model):
        model, _ = population_model
        a = np.array([0.0, 1.0, 2.0])
        bands = np.array([False, True, True])
        obs = model.control_observed
        used = np.isfinite(model.responsibilities).all(axis=1) & obs[:, 1:].all(axis=1)
        survival = model.completeness.survival(
            model.control_magnitudes[used],
            obs[used] & bands[None, :],
            model.band_extinction,
            a,
        )
        manual = survival.T @ model.responsibilities[used]
        manual /= manual.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(
            model.weights_at_extinction(a, bands=bands), manual, rtol=1e-12
        )
        np.testing.assert_allclose(
            model.weights_at_extinction(a, bands=[1, 2]), manual, rtol=1e-12
        )

    @pytest.mark.parametrize(
        "bands, match",
        [
            ([], "At least one band"),
            (np.zeros(3, dtype=bool), "At least one band"),
            ([5], "must lie in"),
            ([0.5], "must be integers"),
            (np.ones(2, dtype=bool), "must have 3 entries"),
        ],
    )
    def test_invalid_bands_rejected(self, population_model, bands, match):
        model, _ = population_model
        with pytest.raises(ValueError, match=match):
            model.weights_at_extinction(np.array([0.0]), bands=bands)

    def test_uncovered_pattern_falls_back(self, population_model):
        """A pattern no control source covers uses the shared table."""
        import copy

        from pnicer.model import PatternCoverageError

        model, _ = population_model
        sparse = copy.copy(model)
        sparse.control_observed = model.control_observed.copy()
        sparse.control_observed[:, 0] = False  # no control source has J
        a = np.array([0.0, 1.0])
        with pytest.raises(PatternCoverageError):
            sparse.weights_at_extinction(a, bands=[0, 1])
        with pytest.warns(RuntimeWarning, match="unconditioned"):
            fallback = sparse._pattern_weights(a, 0.01, np.array([True, True, False]))
        np.testing.assert_allclose(fallback, sparse.weights_at_extinction(a))

    def test_unconditioned_matches_global_table(self, population_model):
        """pattern_conditioned=False reproduces the single shared table."""
        from pnicer.posterior import component_terms, exact_adaptive_posterior

        model, rng = population_model
        science, _ = _draw_population(rng, 3000, extinction=1.0)
        a_table = np.arange(-2.0, 8.0 + 1e-9, 0.02)
        reference = exact_adaptive_posterior(
            component_terms(science, model.means, model.covariances),
            weight_table=model.weights_at_extinction(a_table),
            a_table=a_table,
            coordinates=None,
            extinction_vector=None,
        )
        post = model.posterior(science, adaptive=True, pattern_conditioned=False)
        np.testing.assert_allclose(post.mean(), reference.mean(), rtol=1e-12)

    @pytest.mark.parametrize("a_true", [1.5, 2.0])
    def test_reduces_bias_of_incomplete_patterns(self, population_model, a_true):
        """Sources detected in H and Ks only: the conditioned population
        (survival in H and Ks) removes most of the shared table's bias."""
        model, rng = population_model
        science, _ = _draw_population(rng, 40000, extinction=a_true)
        hk = (science.observed_bands == [False, True, True]).all(axis=1)
        assert hk.sum() > 1000

        def bias(**kwargs):
            cat = model.posterior(science, adaptive=True, **kwargs).discretize()
            return _weighted_bias(
                type(cat)(cat.extinction[hk], cat.variance[hk]), a_true
            )

        conditioned, shared = bias(), bias(pattern_conditioned=False)
        assert abs(conditioned) < abs(shared)
        assert abs(conditioned) < 0.05


def _weighted_bias(cat, a_true):
    good = np.isfinite(cat.extinction) & (cat.variance > 0)
    w = 1.0 / cat.variance[good]
    return np.sum((cat.extinction[good] - a_true) * w) / np.sum(w)


class TestBiasCorrection:
    """Constant-extinction injection with ground truth (L18 Eq. 37 metric)."""

    @pytest.mark.parametrize("a_true", [1.0, 2.0])
    def test_adaptive_removes_population_bias(self, population_model, a_true):
        model, rng = population_model
        science, _ = _draw_population(rng, 30000, extinction=a_true)
        plain = _weighted_bias(model.posterior(science).discretize(), a_true)
        exact = _weighted_bias(
            model.posterior(science, adaptive=True).discretize(), a_true
        )
        assert abs(exact) < abs(plain)
        assert abs(exact) < 0.05

    def test_zero_extinction_unbiased(self, population_model):
        model, rng = population_model
        science, _ = _draw_population(rng, 20000, extinction=0.0)
        for adaptive in (False, True):
            cat = model.posterior(science, adaptive=adaptive).discretize()
            assert abs(_weighted_bias(cat, 0.0)) < 0.05

    @pytest.mark.parametrize("a_true", [1.0, 2.0])
    def test_iterative_scheme_reduces_bias_too(self, population_model, a_true):
        """The published Lombardi (2018) scheme also corrects at high A;
        its known weakness (positive bias for broad posteriors at A ~ 0)
        is why "exact" is the default."""
        model, rng = population_model
        science, _ = _draw_population(rng, 20000, extinction=a_true)
        plain = _weighted_bias(model.posterior(science).discretize(), a_true)
        iterative = _weighted_bias(
            model.posterior(
                science, adaptive=True, adaptive_method="iterative"
            ).discretize(),
            a_true,
        )
        assert abs(iterative) < abs(plain)


class TestExactVsGrid:
    def test_exact_agrees_with_grid_reference(self, population_model):
        """The quadrature-based exact posterior vs the independent grid
        evaluation with extinction-dependent weights in the likelihood."""
        model, rng = population_model
        science, _ = _draw_population(rng, 4000, extinction=1.0)
        a_grid = np.arange(-2.0, 8.0, 0.05)

        post = model.posterior(science, adaptive=True)
        pdf_grid = model.posterior_grid(science, a_grid, adaptive=True)

        mean_exact = post.mean()
        norm = pdf_grid.sum(axis=1)
        mean_grid = (pdf_grid @ a_grid) / np.where(norm > 0, norm, np.nan)
        good = np.isfinite(mean_exact) & np.isfinite(mean_grid)
        # Extinction at A=1 removes most faint synthetic sources
        assert good.sum() > 800
        diff = np.abs(mean_exact[good] - mean_grid[good])
        assert np.median(diff) < 0.01
        assert np.percentile(diff, 95) < 0.05


class TestGuards:
    def test_deep_extinction_stays_finite(self, population_model):
        """Sources far beyond the completeness cliff must keep a finite
        adaptive posterior (population frozen at the last constrained grid
        point) instead of turning NaN."""
        model, _ = population_model
        science = Photometry(
            magnitudes={
                "J": np.array([12.0 + 9.0 * 2.5]),
                "H": np.array([11.5 + 9.0 * 1.55]),
                "Ks": np.array([11.3 + 9.0 * 1.0]),
            },
            errors={b: np.array([0.05]) for b in BAND_NAMES},
            extinction=EXTINCTION,
        )
        post = model.posterior(
            science, adaptive=True, a_grid=np.arange(-2.0, 12.0, 0.02)
        )
        assert np.isfinite(post.mean()[0])
        assert 8.0 < post.mean()[0] < 10.0

    def test_iterative_25_source_field(self, population_model):
        """Regression: N equal to the default grid size must not transpose
        the weights (shape-sniffing bug)."""
        model, _ = population_model
        rng = np.random.default_rng(11)
        science, _ = _draw_population(rng, 60)
        # trim to exactly 25 usable sources = default iterative grid length
        keep = np.zeros(science.n_sources, dtype=bool)
        keep[:25] = True
        science25 = Photometry(
            magnitudes=dict(zip(BAND_NAMES, science.magnitudes[keep].T, strict=True)),
            errors=dict(zip(BAND_NAMES, science.magnitude_errors[keep].T, strict=True)),
            extinction=EXTINCTION,
        )
        post = model.posterior(science25, adaptive=True, adaptive_method="iterative")
        plain = model.posterior(science25)
        good = np.isfinite(post.mean()) & np.isfinite(plain.mean())
        # Iterative re-weighting shifts estimates only mildly at A ~ 0
        assert np.nanmedian(np.abs(post.mean()[good] - plain.mean()[good])) < 0.3

    def test_adaptive_requires_completeness(self, rng):
        control, _ = _draw_population(np.random.default_rng(3), 5000)
        model = control.fit_intrinsic_colors(
            n_components=2, random_state=0, completeness=None
        )
        science, _ = _draw_population(np.random.default_rng(4), 100)
        with pytest.raises(ValueError, match="Adaptive correction requires"):
            model.posterior(science, adaptive=True)
