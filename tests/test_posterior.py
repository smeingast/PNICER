import numpy as np
import pytest
from scipy.stats import multivariate_normal

from pnicer import Colors, IntrinsicColorModel, Photometry

EXTINCTION = {"J": 2.5, "H": 1.55, "Ks": 1.0}


def _random_model(rng, n_dim=2, n_components=3):
    weights = rng.dirichlet(np.ones(n_components) * 5)
    means = rng.normal(0.6, 0.4, size=(n_components, n_dim))
    covs = []
    for _ in range(n_components):
        a = rng.normal(0, 0.15, size=(n_dim, n_dim))
        covs.append(a @ a.T + 0.02 * np.eye(n_dim))
    return weights, means, np.array(covs)


def _brute_force_pdf(a_grid, y, err_cov, proj, weights, means, covs, k_full):
    """Independent numerical evaluation of p(A | y) on a grid.

    Direct implementation of Lombardi (2018) Eq. 15: for each grid value A,
    evaluate the mixture likelihood of the observed colors shifted by A
    along the reddening vector; normalize over the grid.
    """
    pdf = np.zeros_like(a_grid)
    for g, a in enumerate(a_grid):
        like = 0.0
        for w, b, v in zip(weights, means, covs, strict=True):
            mean_obs = proj @ (b + a * k_full)
            cov_obs = proj @ v @ proj.T + err_cov
            like += w * multivariate_normal.pdf(y, mean=mean_obs, cov=cov_obs)
        pdf[g] = like
    pdf /= np.trapezoid(pdf, a_grid)
    return pdf


class TestPosteriorMath:
    """The decisive correctness test: closed-form posterior vs brute force."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_matches_numerical_integration(self, seed):
        rng = np.random.default_rng(seed)
        weights, means, covs = _random_model(rng)

        # Random source with a random missingness pattern over 3 bands
        mags = 14.0 + rng.normal(0, 1, 3)
        errs = rng.uniform(0.03, 0.3, 3)
        pattern = [
            (True, True, True),
            (True, True, False),
            (False, True, True),
            (True, False, True),  # chained gap
        ][seed % 4]
        m = {"J": mags[0:1], "H": mags[1:2], "Ks": mags[2:3]}
        e = {"J": errs[0:1], "H": errs[1:2], "Ks": errs[2:3]}
        for band, present in zip(("J", "H", "Ks"), pattern, strict=True):
            if not present:
                m[band] = np.array([np.nan])
        phot = Photometry(magnitudes=m, errors=e, extinction=EXTINCTION)
        model = IntrinsicColorModel(
            weights, means, covs, phot.color_names, phot.reddening_vector
        )
        post = model.posterior(phot)

        group = phot.pattern_groups()[0]
        # Reference grid wide enough to hold ~all posterior mass, even for
        # weakly constrained patterns
        center = post.mean()[0]
        halfwidth = 12 * np.sqrt(post.variance()[0])
        a_grid = np.linspace(center - halfwidth, center + halfwidth, 4001)
        reference = _brute_force_pdf(
            a_grid,
            group.colors[0],
            group.covariances[0],
            group.projection,
            weights,
            means,
            covs,
            phot.reddening_vector,
        )
        ours = post.pdf(a_grid)[0]
        np.testing.assert_allclose(ours, reference, atol=1e-6, rtol=1e-4)

        # Moments against numerical integration
        ref_mean = np.trapezoid(a_grid * reference, a_grid)
        ref_var = np.trapezoid((a_grid - ref_mean) ** 2 * reference, a_grid)
        assert np.isclose(post.mean()[0], ref_mean, atol=1e-4)
        assert np.isclose(post.variance()[0], ref_var, atol=1e-4, rtol=1e-3)

    def test_pdf_normalized(self, rng):
        weights, means, covs = _random_model(rng)
        phot = Photometry(
            magnitudes={
                "J": np.array([15.0]),
                "H": np.array([14.3]),
                "Ks": np.array([14.0]),
            },
            errors={
                "J": np.array([0.1]),
                "H": np.array([0.08]),
                "Ks": np.array([0.09]),
            },
            extinction=EXTINCTION,
        )
        model = IntrinsicColorModel(
            weights, means, covs, phot.color_names, phot.reddening_vector
        )
        post = model.posterior(phot)
        a_grid = np.linspace(-5, 8, 2001)
        integral = np.trapezoid(post.pdf(a_grid)[0], a_grid)
        assert np.isclose(integral, 1.0, atol=1e-3)

    def test_discretize_matches_monte_carlo(self, rng):
        """Moment-preserving reduction (Eqs. 30-32) against sampling."""
        weights, means, covs = _random_model(rng)
        phot = Photometry(
            magnitudes={
                "J": np.array([16.0]),
                "H": np.array([15.0]),
                "Ks": np.array([14.5]),
            },
            errors={
                "J": np.array([0.15]),
                "H": np.array([0.1]),
                "Ks": np.array([0.1]),
            },
            extinction=EXTINCTION,
        )
        model = IntrinsicColorModel(
            weights, means, covs, phot.color_names, phot.reddening_vector
        )
        post = model.posterior(phot)
        cat = post.discretize()

        comp = rng.choice(post.n_components, size=200_000, p=post.weights[0])
        samples = rng.normal(post.means[0][comp], np.sqrt(post.variances[0][comp]))
        assert np.isclose(cat.extinction[0], samples.mean(), atol=0.01)
        assert np.isclose(cat.variance[0], samples.var(), rtol=0.02)


class TestNicerEquivalence:
    def test_single_color_k1_equals_nicer(self):
        """Scientific invariant: a one-component model on a single color
        reproduces NICER exactly (Meingast+ 2017, Sect. 3.3)."""
        col = Colors(
            colors={"J-H": np.array([0.9, 1.5, 0.4])},
            errors={"J-H": np.array([0.08, 0.15, 0.05])},
            reddening={"J-H": 0.95},
        )
        mu, var = 0.55, 0.04
        model = IntrinsicColorModel(
            weights=np.array([1.0]),
            means=np.array([[mu]]),
            covariances=np.array([[[var]]]),
            color_names=col.color_names,
            reddening_vector=col.reddening_vector,
        )
        post = model.posterior(col)
        nicer = col.nicer(color0=np.array([mu]), color0_cov=np.array([[var]]))
        np.testing.assert_allclose(post.mean(), nicer.extinction, atol=1e-12)
        np.testing.assert_allclose(post.variance(), nicer.variance, atol=1e-12)


class TestEdgeCases:
    def test_degenerate_reddening_unconstrained(self):
        """A source whose only observed color has zero reddening gets NaN."""
        col = Colors(
            colors={
                "a": np.array([0.5, 0.5]),
                "b": np.array([np.nan, 0.3]),
            },
            errors={
                "a": np.array([0.05, 0.05]),
                "b": np.array([0.05, 0.05]),
            },
            reddening={"a": 0.0, "b": 1.0},
        )
        model = IntrinsicColorModel(
            weights=np.array([1.0]),
            means=np.array([[0.4, 0.2]]),
            covariances=np.array([np.diag([0.02, 0.02])]),
            color_names=col.color_names,
            reddening_vector=col.reddening_vector,
        )
        post = model.posterior(col)
        assert np.isnan(post.mean()[0])  # only zero-reddening color
        assert np.isfinite(post.mean()[1])  # second color constrains it

    def test_sources_without_data_get_nan(self):
        phot = Photometry(
            magnitudes={
                "J": np.array([15.0, np.nan]),
                "H": np.array([14.5, np.nan]),
                "Ks": np.array([14.2, 13.0]),
            },
            errors={
                "J": np.array([0.05, 0.05]),
                "H": np.array([0.05, 0.05]),
                "Ks": np.array([0.05, 0.05]),
            },
            extinction=EXTINCTION,
        )
        model = IntrinsicColorModel(
            weights=np.array([1.0]),
            means=np.array([[0.5, 0.3]]),
            covariances=np.array([np.diag([0.02, 0.02])]),
            color_names=phot.color_names,
            reddening_vector=phot.reddening_vector,
        )
        post = model.posterior(phot)
        assert np.isfinite(post.mean()[0])
        assert np.isnan(post.mean()[1])
        assert np.isnan(post.log_evidence[1])

    def test_evidence_flags_outliers(self, rng):
        """Sources far from the model get lower evidence (same pattern)."""
        weights, means, covs = _random_model(rng)
        n = 50
        base = means[0]
        colors_typical = rng.multivariate_normal(base, covs[0], size=n)
        outlier = base + np.array([4.0, -3.0])
        all_colors = np.vstack([colors_typical, outlier])
        col = Colors(
            colors={
                "c1": all_colors[:, 0],
                "c2": all_colors[:, 1],
            },
            errors={
                "c1": np.full(n + 1, 0.05),
                "c2": np.full(n + 1, 0.05),
            },
            reddening={"c1": 0.95, "c2": 0.55},
        )
        model = IntrinsicColorModel(
            weights, means, covs, col.color_names, col.reddening_vector
        )
        post = model.posterior(col)
        assert post.log_evidence[-1] < np.min(post.log_evidence[:-1])
