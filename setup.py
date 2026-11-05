from setuptools import find_packages, setup

library_name = "torchbp"

setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch", "numpy", "scipy"],
    extras_require = {
        'test': ["pytest", "expecttest"],
        'docs':  [
            "matplotlib >=3.5",
            "nbval >=0.11",
            "jupyter-client >=7.3.5",
            "sphinx-rtd-theme >=1.0",
            "sphinx >=4",
            "nbsphinx >= 0.8.9",
            "openpyxl >= 3",
            "lxml-html-clean >= 0.4.1",
            "numpydoc"]
    },
    description="Differentiable synthetic aperture radar library",
    long_description=open("Readme.md").read(),
    long_description_content_type="text/markdown",
)
