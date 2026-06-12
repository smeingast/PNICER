# PNICER

PNICER is an astronomical software package for estimating extinction toward individual sources and for creating extinction maps from photometric catalogs. It implements the PNICER method, which uses unsupervised machine learning (Gaussian Mixture Models) to derive extinction probability densities without priors on the column density or the intrinsic color distribution, and also provides the well-established NICER technique (Lombardi & Alves 2001) in a unified interface, including the NICEST correction for cloud substructure (Lombardi 2009).

The method is described in [Meingast, Lombardi & Alves (2017), A&A 601, A137](https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract).

> **Note:** Version 1.0 is a legacy release that preserves PNICER as it has been used since the publication of the paper (with minimal compatibility fixes for current numpy and packaging standards). Ongoing modernization of the package takes place on the master branch.

## Requirements

PNICER requires Python ≥ 3.11 with *numpy*, *scipy*, *astropy*, *matplotlib*, *scikit-learn*, and *joblib*. All dependencies are installed automatically with pip. Because of the parallel processing framework used, PNICER does not run on Windows.

## Installation

Install directly from GitHub with pip

```bash
pip install git+https://github.com/smeingast/PNICER.git
```

or, for a specific release (e.g. the legacy v1.0),

```bash
pip install git+https://github.com/smeingast/PNICER.git@v1.0
```

### Test

To test the installation, start up python (or ipython) and type

```python
from pnicer.tests import orion
orion()
```

which will go through all major PNICER methods. At the end you should see a plot window with an extinction map of Orion A created from 2MASS data:

![Orion](https://raw.githubusercontent.com/smeingast/PNICER/master/pnicer/tests_resources/orion.png)

## Getting started

For an introduction to the basic tools available in PNICER, please refer to the jupyter notebook provided with this package:

[PNICER introduction notebook](https://github.com/smeingast/PNICER/blob/master/notebooks/pnicer.ipynb)

## Citation

If you use PNICER in your research, please cite [Meingast, Lombardi & Alves (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract):

```bibtex
@ARTICLE{2017A&A...601A.137M,
       author = {{Meingast}, Stefan and {Lombardi}, Marco and {Alves}, Jo{\~a}o},
        title = "{Estimating extinction using unsupervised machine learning}",
      journal = {\aap},
         year = 2017,
        month = may,
       volume = {601},
          eid = {A137},
        pages = {A137},
          doi = {10.1051/0004-6361/201630032}
}
```
