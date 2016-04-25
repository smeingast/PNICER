from distutils.core import setup

setup(
    name="PNICER",
    version="0.1",
    packages=["", "tests"],
    install_requires=["numpy", "scikit-learn", "matplotlib", "astropy", "wcsaxes"],
    url="",
    license="",
    author="Stefan Meingast",
    author_email="stefan.meingast@gmail.com",
    description="Python package to calculate color-excesses and extinction from arbitrary photometric measurements"
)
