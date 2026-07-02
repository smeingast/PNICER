"""Bundled 2MASS Orion A demo data and an end-to-end example pipeline."""

from __future__ import annotations

from importlib import resources

from pnicer.photometry import Photometry

__all__ = ["load_control", "load_orion", "orion"]

_BANDS = {
    "J": ("Jmag", "e_Jmag"),
    "H": ("Hmag", "e_Hmag"),
    "Ks": ("Kmag", "e_Kmag"),
}
_EXTINCTION = {"J": 2.5, "H": 1.55, "Ks": 1.0}  # Indebetouw et al. (2005)


def _load(filename: str) -> Photometry:
    path = resources.files("pnicer.data").joinpath(filename)
    with resources.as_file(path) as file_path:
        return Photometry.from_fits(
            str(file_path),
            bands=_BANDS,
            extinction=_EXTINCTION,
            lon="GLON",
            lat="GLAT",
            frame="galactic",
        )


def load_orion() -> Photometry:
    """2MASS photometry of the Orion A molecular cloud (science field)."""
    return _load("Orion_A_2mass.fits")


def load_control() -> Photometry:
    """2MASS photometry of the Orion A control field."""
    return _load("CF_2mass.fits")


def orion(plot: bool = True):
    """Run the full demo pipeline on the bundled Orion A data.

    Fits the intrinsic color model on the control field, computes NICER and
    PNICER extinctions, and builds a PNICER extinction map.

    Returns
    -------
    ExtinctionMap
    """
    science = load_orion()
    control = load_control()

    # NICER point estimates
    science.nicer(control)

    # PNICER posteriors and map
    model = control.fit_intrinsic_colors(random_state=0)
    posterior = science.pnicer(model)
    emap = posterior.discretize().build_map(
        bandwidth=5 / 60, metric="gaussian", use_fwhm=True
    )

    if plot:
        emap.plot()
    print("PNICER routines terminated successfully! "
          "Happy hunting for extinction :)")
    return emap
