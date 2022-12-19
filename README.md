<h2 align="center">Fourier Shape Descriptors in Microscopy</h2>

## Table of Contents

- **[Introduction](#introduction)**
- **[Dependencies](#dependencies)**
- **[Getting Started](#getting-started)**
- **[Datasets](#datasets)**
- **[How to use on your data?](#how-to-use-on-your-data)**
- **[Issues](#issues)**


### Introduction
This repository hosts the code used to obtain a representation of an object instance mask as a `Fourier Shape Descriptor` (FSD) vector.

Using Fourier Shape Descriptors (FSDs), one acquires the means to compare two distinct morphologies.

### Dependencies 

One could execute these lines of code to run this branch:

```
conda create -n FSDEnv python==3.8
conda activate FSDEnv
git clone https://github.com/lmanan/FSD.git
cd FSD
pip install -e .
```

### Getting Started

Look in the `examples` directory,  and try out the `01-shape-analysis/BBBC020/run.ipynb` notebook for 2D images.


### Datasets
A curated version of the `BBBC020` **[dataset](https://bbbc.broadinstitute.org/BBBC020)** is made available as release asset **[here](https://github.com/lmanan/FSD/releases/tag/v0.0.1-tag)**. 

### How to use on your data?
   
`*.tif`-type images and the corresponding instance masks should be respectively present under `images` and `masks`. The following would be a desired structure as to how data should be prepared.

```
$data_dir
└───$project-name
    |───images
        └───X0.tif
        └───...
        └───Xn.tif
    └───masks
        └───Y0.tif
        └───...
        └───Yn.tif
```

### Issues

If you encounter any problems, please **[file an issue]** along with a detailed description.

[file an issue]: https://github.com/lmanan/FSD/issues


