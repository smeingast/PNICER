from setuptools import setup, find_packages

# Grab the long description from the README
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PNICER",
    version="0.3",
    packages=find_packages(),
    package_dir={"pnicer": "pnicer"},
    package_data={"pnicer": ["tests_resources/*.fits"]},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.2",
        "scipy>=1.11.4",
        "scikit-learn>=1.3.2",
        "matplotlib>=3.8.2",
        "astropy>=6.0.0",
    ],
    url="https://github.com/smeingast/PNICER",
    author="Stefan Meingast",
    author_email="stefan.meingast@gmail.com",
    description="Python package to calculate color-excesses and "
                "extinction from arbitrary photometric measurements.",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
