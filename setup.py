from __future__ import absolute_import
import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="FSD", 
    version="0.0.1",
    author="Manan Lalit",
    author_email="lalit@mpi-cbg.de",
    description="Fourier Shape Descriptors (FSD) allow comparing and analysing shapes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lmanan/FSD/",
    project_urls={
        "Bug Tracker": "https://github.com/lmanan/FSD/issues",
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
          'matplotlib',
          'numpy',
          'scipy',
          "tifffile",
          "numba",
          "tqdm",
          "jupyter",
          "pandas",
          "seaborn",
          "scikit-image",
          "colorspacious",
          "pytest",
          "imagecodecs",
          "umap-learn"
        ]
)


