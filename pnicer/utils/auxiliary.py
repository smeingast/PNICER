# -----------------------------------------------------------------------------
# Import packages
import os
import sys
import importlib


# -----------------------------------------------------------------------------
def get_resource_path(package, resource):
    """
    Returns the path to an included resource.

    Parameters
    ----------
    package : str
        package name (e.g. astropype.resources.sextractor).
    resource : str
        Name of the resource (e.g. default.conv)

    Returns
    -------
    str
        Path to resource.

    """

    # Import package
    importlib.import_module(name=package)

    # Return path to resource
    return os.path.join(os.path.dirname(sys.modules[package].__file__), resource)


# -----------------------------------------------------------------------------
def flatten_lol(lst):
    """
    Flattens a list of lists.

    Parameters
    ----------
    lst : list
        Input list (that contains lists).

    Returns
    -------
    iterable
        Flattened single list.

    """
    return [item for sublist in lst for item in sublist]
