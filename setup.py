import sys
from distutils.core import setup

# Require Python 3
if sys.version_info < (3, 4):
    sys.exit("Sorry, Python < 3.4 is not supported")


setup(
    name="PNICER",
    version="0.1",
    packages=["pnicer", "pnicer.tests", "pnicer.tests_resources"],
    package_dir={"pnicer": "pnicer"},
    package_data={"pnicer": ["tests_resources/*.fits"]},
    install_requires=["numpy", "scipy", "scikit-learn", "matplotlib", "astropy", "wcsaxes"],
    url="",
    license="",
    author="Stefan Meingast",
    author_email="stefan.meingast@gmail.com",
    description="Python package to calculate color-excesses and extinction from arbitrary photometric measurements."
)
