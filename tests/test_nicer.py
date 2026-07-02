import numpy as np
import pytest

from pnicer import Colors


class TestNicerRegression:
    """Regression against the frozen legacy v1.0 outputs."""

    @pytest.fixture(scope="class")
    def nicer_catalog(self, orion, control):
        return orion.nicer(control)

    def test_complete_sources_exact(self, nicer_catalog, orion, baseline_dir):
        base = np.load(baseline_dir / "nicer_science.npz")
        complete = orion.observed_bands.all(axis=1)
        assert complete.sum() > 70000
        np.testing.assert_allclose(
            nicer_catalog.extinction[complete],
            base["extinction"][complete],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            nicer_catalog.variance[complete],
            base["variance"][complete],
            atol=1e-10,
        )

    def test_partial_sources_consistent(self, nicer_catalog, baseline_dir):
        """Sources with missing bands: the projection used here is the exact
        sigma->infinity limit of the legacy sigma=100 substitution, so values
        agree closely; large deviations only occur for junk catalog errors
        (~10 mag), where the legacy estimates were meaningless anyway."""
        base = np.load(baseline_dir / "nicer_science.npz")
        both = np.isfinite(nicer_catalog.extinction) & np.isfinite(base["extinction"])
        diff = np.abs(nicer_catalog.extinction[both] - base["extinction"][both])
        assert np.median(diff) < 1e-5
        assert np.percentile(diff, 99) < 0.01

    def test_control_field_self(self, control, baseline_dir):
        cat = control.nicer(control)
        base = np.load(baseline_dir / "nicer_control_self.npz")
        both = np.isfinite(cat.extinction) & np.isfinite(base["extinction"])
        diff = np.abs(cat.extinction[both] - base["extinction"][both])
        assert np.median(diff) < 1e-3
        # Scientific invariant: extinction-free field peaks at ~0
        assert abs(np.nanmedian(cat.extinction)) < 0.05


class TestNicerBehavior:
    def test_color0_matches_control(self, orion, control):
        from pnicer.nicer import control_field_statistics

        color0, cov = control_field_statistics(control)
        via_control = orion.nicer(control)
        via_color0 = orion.nicer(color0=color0, color0_cov=cov)
        np.testing.assert_allclose(
            via_control.extinction, via_color0.extinction, atol=1e-12
        )

    def test_min_dim(self, orion, control):
        strict = orion.nicer(control, min_dim=2)
        loose = orion.nicer(control, min_dim=1)
        complete = orion.observed_bands.all(axis=1)
        assert np.isnan(strict.extinction[~complete]).all()
        n_strict = np.isfinite(strict.extinction).sum()
        n_loose = np.isfinite(loose.extinction).sum()
        assert n_strict < n_loose

    def test_requires_control_or_color0(self, orion):
        with pytest.raises(ValueError, match="control field or intrinsic"):
            orion.nicer()

    def test_single_color_closed_form(self):
        """1-color NICER: A = (c - c0)/k, Var = (V + sigma^2)/k^2."""
        col = Colors(
            colors={"J-H": np.array([1.5, 0.7])},
            errors={"J-H": np.array([0.1, 0.2])},
            reddening={"J-H": 0.95},
        )
        cat = col.nicer(color0=np.array([0.5]), color0_cov=np.array([0.2**2]))
        np.testing.assert_allclose(cat.extinction, [1.0 / 0.95, 0.2 / 0.95])
        np.testing.assert_allclose(
            cat.variance,
            [(0.2**2 + 0.1**2) / 0.95**2, (0.2**2 + 0.2**2) / 0.95**2],
        )

    def test_nicer_equals_k1_posterior_multidim(self, orion, control):
        """Mathematical correctness anchor: NICER must be identical to the
        K=1 case of the Bayesian posterior machinery (which is itself
        verified against brute-force numerical integration), for all
        missingness patterns of the real Orion data."""
        from pnicer import IntrinsicColorModel
        from pnicer.nicer import control_field_statistics

        nicer = orion.nicer(control)
        color0, cov = control_field_statistics(control)
        model = IntrinsicColorModel(
            weights=np.array([1.0]),
            means=color0[None, :],
            covariances=cov[None, :, :],
            color_names=orion.color_names,
            reddening_vector=orion.reddening_vector,
        )
        post = model.posterior(orion)
        both = np.isfinite(nicer.extinction) & np.isfinite(post.mean())
        assert both.sum() > 80000
        np.testing.assert_allclose(
            nicer.extinction[both], post.mean()[both], atol=1e-12
        )
        np.testing.assert_allclose(
            nicer.variance[both], post.variance()[both], atol=1e-12
        )

    def test_color0_cov_diagonal_equals_matrix(self):
        """1-d (diagonal variances) and 2-d covariance input must agree."""
        col = Colors(
            colors={"a": np.array([1.0]), "b": np.array([0.5])},
            errors={"a": np.array([0.1]), "b": np.array([0.1])},
            reddening={"a": 0.95, "b": 0.55},
        )
        c0 = np.array([0.4, 0.2])
        diag = col.nicer(color0=c0, color0_cov=np.array([0.04, 0.02]))
        full = col.nicer(color0=c0, color0_cov=np.diag([0.04, 0.02]))
        np.testing.assert_allclose(diag.extinction, full.extinction)
        np.testing.assert_allclose(diag.variance, full.variance)
