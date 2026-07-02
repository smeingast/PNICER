import importlib.util
import itertools

import numpy as np
import pytest

from pnicer import Photometry
from pnicer.xd import fit_xd, xd_log_likelihood

EXTINCTION = {"J": 2.5, "H": 1.55, "Ks": 1.0}


def _synthetic_photometry(
    rng,
    n=4000,
    weights=(0.6, 0.4),
    means=((0.4, 0.15), (1.2, 0.8)),
    covs=None,
    err_scale=0.05,
    missing_fraction=0.0,
    heteroscedastic=False,
):
    """Draw noisy band photometry whose colors follow a known 2D GMM."""
    if covs is None:
        covs = [np.diag([0.02, 0.01]), np.array([[0.05, 0.02], [0.02, 0.04]])]
    comp = rng.choice(len(weights), size=n, p=weights)
    colors = np.array(
        [rng.multivariate_normal(means[c], covs[c]) for c in comp]
    )
    # Build magnitudes from colors: fix Ks, derive H and J
    ks = rng.uniform(10, 15, size=n)
    h = ks + colors[:, 1]
    j = h + colors[:, 0]
    mags = np.column_stack([j, h, ks])
    if heteroscedastic:
        errs = rng.uniform(0.2 * err_scale, 3 * err_scale, size=(n, 3))
    else:
        errs = np.full((n, 3), err_scale)
    noisy = mags + rng.standard_normal((n, 3)) * errs
    if missing_fraction:
        # Random missing entries in all bands, including the middle one
        drop = rng.random((n, 3)) < missing_fraction
        noisy[drop] = np.nan
    return Photometry(
        magnitudes={"J": noisy[:, 0], "H": noisy[:, 1], "Ks": noisy[:, 2]},
        errors={"J": errs[:, 0], "H": errs[:, 1], "Ks": errs[:, 2]},
        extinction=EXTINCTION,
    ), comp


def _match_components(fitted_means, true_means):
    """Map fitted component indices onto true ones by proximity."""
    order = []
    for tm in true_means:
        order.append(int(np.argmin(np.sum((fitted_means - tm) ** 2, axis=1))))
    return order


class TestXDRecovery:
    def test_recovers_complete_data(self, rng):
        phot, _ = _synthetic_photometry(rng)
        result = fit_xd(
            phot.pattern_groups(), n_dim=2, n_sources=phot.n_sources,
            n_components=2, random_state=0,
        )
        assert result.converged
        order = _match_components(result.means, [(0.4, 0.15), (1.2, 0.8)])
        assert sorted(order) == [0, 1]
        np.testing.assert_allclose(
            result.means[order], [(0.4, 0.15), (1.2, 0.8)], atol=0.05
        )
        np.testing.assert_allclose(
            np.sort(result.weights), [0.4, 0.6], atol=0.05
        )
        # Deconvolution: fitted covariances match the intrinsic ones, not
        # the error-broadened observed distribution
        true_covs = [np.diag([0.02, 0.01]), [[0.05, 0.02], [0.02, 0.04]]]
        for fitted, true in zip(
            result.covariances[order], true_covs, strict=True
        ):
            np.testing.assert_allclose(fitted, true, atol=0.015)

    def test_recovers_with_missing_bands(self, rng):
        phot, _ = _synthetic_photometry(rng, n=6000, missing_fraction=0.25)
        groups = phot.pattern_groups()
        assert len(groups) > 1  # multiple patterns, incl. chained gaps
        result = fit_xd(
            groups, n_dim=2, n_sources=phot.n_sources,
            n_components=2, random_state=0,
        )
        order = _match_components(result.means, [(0.4, 0.15), (1.2, 0.8)])
        assert sorted(order) == [0, 1]
        np.testing.assert_allclose(
            result.means[order], [(0.4, 0.15), (1.2, 0.8)], atol=0.07
        )

    def test_recovers_heteroscedastic(self, rng):
        phot, _ = _synthetic_photometry(rng, n=8000, heteroscedastic=True)
        result = fit_xd(
            phot.pattern_groups(), n_dim=2, n_sources=phot.n_sources,
            n_components=2, random_state=0,
        )
        order = _match_components(result.means, [(0.4, 0.15), (1.2, 0.8)])
        np.testing.assert_allclose(
            result.means[order], [(0.4, 0.15), (1.2, 0.8)], atol=0.07
        )

    def test_responsibilities(self, rng):
        phot, comp = _synthetic_photometry(rng, n=3000)
        result = fit_xd(
            phot.pattern_groups(), n_dim=2, n_sources=phot.n_sources,
            n_components=2, random_state=0,
        )
        order = _match_components(result.means, [(0.4, 0.15), (1.2, 0.8)])
        assigned = np.argmax(result.responsibilities, axis=1)
        # Well-separated components: assignments recover the labels
        accuracy = np.mean(assigned == np.array(order)[comp])
        assert accuracy > 0.95

    def test_loglik_monotone(self, rng):
        phot, _ = _synthetic_photometry(rng, n=2000)
        groups = phot.pattern_groups()
        lls = [
            fit_xd(
                groups, n_dim=2, n_sources=phot.n_sources, n_components=2,
                random_state=0, max_iter=mi, tol=0.0,
            ).log_likelihood
            for mi in (2, 5, 10, 30)
        ]
        assert all(b >= a - 1e-8 for a, b in itertools.pairwise(lls))

    def test_bic_selects_true_k(self, rng):
        phot, _ = _synthetic_photometry(rng, n=6000)
        groups = phot.pattern_groups()
        results = [
            fit_xd(
                groups, n_dim=2, n_sources=phot.n_sources, n_components=k,
                random_state=0,
            )
            for k in (1, 2, 3, 4)
        ]
        best = int(np.argmin([r.bic for r in results])) + 1
        assert best == 2


@pytest.mark.skipif(
    importlib.util.find_spec("extreme_deconvolution") is None,
    reason="Bovy's extreme_deconvolution not installed",
)
class TestAgainstBovy:
    def test_matches_reference_implementation(self, rng):
        from extreme_deconvolution import extreme_deconvolution

        from pnicer.xd import _initial_parameters

        phot, _ = _synthetic_photometry(rng, n=3000)
        groups = phot.pattern_groups()
        ours = fit_xd(
            groups, n_dim=2, n_sources=phot.n_sources, n_components=2,
            random_state=0, tol=1e-8, max_iter=2000,
        )
        w, m, c = _initial_parameters(groups, 2, 2, 0, 1e-6)
        group = groups[0]
        extreme_deconvolution(group.colors, group.covariances, w, m, c)
        ll_bovy = xd_log_likelihood(groups, w, m, c)
        # Same optimum within numerical tolerance
        assert ours.log_likelihood >= ll_bovy - 0.001 * abs(ll_bovy)
        order = _match_components(ours.means, m)
        np.testing.assert_allclose(ours.means[order], m, atol=0.02)
