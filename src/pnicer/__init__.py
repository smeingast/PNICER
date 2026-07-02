"""PNICER: extinction estimation and mapping from photometric catalogs.

Estimators
----------
- NICER (Lombardi & Alves 2001): closed-form point estimates.
- PNICER (Meingast, Lombardi & Alves 2017; machinery of Lombardi 2018):
  per-source extinction posteriors from a Gaussian mixture model of the
  intrinsic color distribution, fitted by extreme deconvolution, with an
  optional adaptive correction for extinction-driven population changes.
"""

from pnicer.catalog import ExtinctionCatalog
from pnicer.mapping import ExtinctionMap
from pnicer.model import IntrinsicColorModel
from pnicer.photometry import Colors, Photometry
from pnicer.posterior import ExtinctionPosterior

__version__ = "2.0.0.dev0"

__all__ = [
    "Colors",
    "ExtinctionCatalog",
    "ExtinctionMap",
    "ExtinctionPosterior",
    "IntrinsicColorModel",
    "Photometry",
    "__version__",
]
