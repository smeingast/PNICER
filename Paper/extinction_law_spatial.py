# ----------------------------------------------------------------------
# Import stuff
import warnings
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
from MyFunctions import point_density
from helper import pnicer_ini


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap = brewer2mpl.get_map("YlGnBu", "Sequential", number=9, reverse=True).get_mpl_colormap(N=11, gamma=1)


# ----------------------------------------------------------------------
# Intialize PNICER
science, control = pnicer_ini(skip_science=1, skip_control=1, n_features=5, color=True)
