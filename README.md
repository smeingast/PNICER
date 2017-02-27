PNICER is an astronomical software suite for estimating extinction for individual sources and creating extinction maps using unsupervised machine learning algorithms. If you want to know more about the technique, you are invited to study the published manuscript, which is currently available on Astro-ph.

## Installation

To install the package, download and extract the latest release to your computer and install with pip

```bash
pip install --user /path/to/PNICER/
```

where the last argument points to the directory of the saved download. All dependencies will be installed automatically.

### Test

To test the installation, start up python (or ipython) and type

```python
from pnicer.tests import orion
orion()
```

which will go through all major PNICER methods. At the end you should see a plot window with an extinction map of Orion A created from 2MASS data.


## Introduction


For an introduction to the basic tools available in **PNICER**, please refer to the jupyter notebook provided with this package:

[PNICER introduction notebook](https://github.com/smeingast/PNICER/blob/master/notebooks/pnicer.ipynb)


In the near future (April - May 2017) we plan to implement advanced extinction mapping tools and we will also soon provide the complete API of PNICER.