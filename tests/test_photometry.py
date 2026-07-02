import numpy as np
import pytest

from pnicer import Colors, Photometry


def _simple_photometry(**overrides):
    data = {
        "magnitudes": {
            "J": np.array([15.0, 14.0, 16.0, np.nan]),
            "H": np.array([14.5, np.nan, 15.5, 14.0]),
            "Ks": np.array([14.2, 13.5, np.nan, 13.8]),
        },
        "errors": {
            "J": np.array([0.05, 0.04, 0.06, 0.05]),
            "H": np.array([0.04, 0.04, 0.05, 0.04]),
            "Ks": np.array([0.06, 0.05, 0.06, 0.05]),
        },
        "extinction": {"J": 2.5, "H": 1.55, "Ks": 1.0},
    }
    data.update(overrides)
    return Photometry(**data)


class TestPhotometry:
    def test_basic_properties(self):
        phot = _simple_photometry()
        assert phot.n_sources == 4
        assert phot.n_bands == 3
        assert phot.n_colors == 2
        assert phot.color_names == ("J-H", "H-Ks")
        assert np.allclose(phot.reddening_vector, [0.95, 0.55])

    def test_colors_nan_propagation(self):
        colors = _simple_photometry().colors
        assert np.allclose(colors[0], [0.5, 0.3])
        assert np.isnan(colors[1]).all()  # H missing -> both colors gone
        assert np.isnan(colors[2, 1]) and np.isfinite(colors[2, 0])

    def test_pattern_groups_complete(self):
        groups = {g.key: g for g in _simple_photometry().pattern_groups()}
        complete = groups[(0, 1, 2)]
        assert np.array_equal(complete.indices, [0])
        assert np.allclose(complete.projection, np.eye(2))
        # Tridiagonal error covariance (Lombardi 2018, Eq. 6)
        cov = complete.covariances[0]
        assert np.isclose(cov[0, 0], 0.05**2 + 0.04**2)
        assert np.isclose(cov[1, 1], 0.04**2 + 0.06**2)
        assert np.isclose(cov[0, 1], -(0.04**2))

    def test_pattern_groups_chained_gap(self):
        """A missing middle band chains the color across the gap."""
        groups = {g.key: g for g in _simple_photometry().pattern_groups()}
        gap = groups[(0, 2)]  # J and Ks observed, H missing
        assert np.array_equal(gap.indices, [1])
        assert np.allclose(gap.projection, [[1.0, 1.0]])
        assert np.isclose(gap.colors[0, 0], 14.0 - 13.5)
        assert np.isclose(gap.covariances[0, 0, 0], 0.04**2 + 0.05**2)

    def test_pattern_groups_min_dim(self):
        phot = _simple_photometry()
        groups = phot.pattern_groups(min_dim=2)
        assert [g.key for g in groups] == [(0, 1, 2)]

    def test_bad_error_masks_band(self):
        phot = _simple_photometry(
            errors={
                "J": np.array([np.nan, 0.04, 0.06, 0.05]),
                "H": np.array([0.04, 0.04, 0.05, 0.04]),
                "Ks": np.array([0.06, 0.05, 0.06, 0.05]),
            }
        )
        assert not phot.observed_bands[0, 0]
        # raw colors ignore the error masking
        assert np.isfinite(phot.raw_colors[0, 0])
        assert np.isnan(phot.colors[0, 0])

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="At least two bands"):
            Photometry(
                magnitudes={"J": np.array([1.0])},
                errors={"J": np.array([0.1])},
                extinction={"J": 2.5},
            )
        with pytest.raises(ValueError, match="missing from extinction"):
            _simple_photometry(extinction={"J": 2.5, "H": 1.55})
        with pytest.raises(ValueError, match="equal length"):
            _simple_photometry(
                magnitudes={
                    "J": np.array([15.0]),
                    "H": np.array([14.5, 15.0]),
                    "Ks": np.array([14.2]),
                }
            )

    def test_from_table(self):
        table = {
            "jm": np.array([15.0, 14.0]),
            "je": np.array([0.05, 0.04]),
            "km": np.array([14.0, 13.2]),
            "ke": np.array([0.05, 0.06]),
            "ra": np.array([10.0, 11.0]),
            "dec": np.array([-5.0, -6.0]),
        }
        phot = Photometry.from_table(
            table,
            bands={"J": ("jm", "je"), "Ks": ("km", "ke")},
            extinction={"J": 2.5, "Ks": 1.0},
            lon="ra",
            lat="dec",
        )
        assert phot.n_sources == 2
        assert np.allclose(phot.colors[:, 0], [1.0, 0.8])
        assert phot.coordinates is not None


class TestColors:
    def test_pattern_selection(self):
        col = Colors(
            colors={
                "J-H": np.array([0.5, np.nan]),
                "H-Ks": np.array([0.3, 0.2]),
            },
            errors={
                "J-H": np.array([0.05, 0.05]),
                "H-Ks": np.array([0.04, 0.04]),
            },
            reddening={"J-H": 0.95, "H-Ks": 0.55},
        )
        groups = {g.key: g for g in col.pattern_groups()}
        # Missing colors select dimensions instead of chaining
        assert np.allclose(groups[(1,)].projection, [[0.0, 1.0]])
        assert np.allclose(groups[(0, 1)].projection, np.eye(2))
        # Diagonal error covariance
        assert np.allclose(
            groups[(0, 1)].covariances[0], np.diag([0.05**2, 0.04**2])
        )

    def test_single_color_allowed(self):
        col = Colors(
            colors={"J-H": np.array([0.5, 0.6])},
            errors={"J-H": np.array([0.05, 0.05])},
            reddening={"J-H": 0.95},
        )
        assert col.n_colors == 1
        assert len(col.pattern_groups()) == 1
