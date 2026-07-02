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
def population_model():
    rng = np.random.default_rng(7)
    control, _ = _draw_population(rng, 30000)
    completeness = CompletenessModel.from_parameters(BAND_NAMES, M50, WIDTH)
    model = control.fit_intrinsic_colors(
        n_components=3, random_state=0, completeness=completeness
    )
    return model, rng


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
    def test_adaptive_requires_completeness(self, rng):
        control, _ = _draw_population(np.random.default_rng(3), 5000)
        model = control.fit_intrinsic_colors(
            n_components=2, random_state=0, completeness=None
        )
        science, _ = _draw_population(np.random.default_rng(4), 100)
        with pytest.raises(ValueError, match="Adaptive correction requires"):
            model.posterior(science, adaptive=True)
