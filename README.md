<h2 align="center">Fourier Shape Descriptor Representation in Microscopy</h2>

## Table of Contents

- **[Introduction](#introduction)**
- **[Dependencies](#dependencies)**
- **[Getting Started](#getting-started)**
- **[Datasets](#datasets)**
- **[How to Use on your data](#how-to-use-on-your-data)**
- **[Issues](#issues)**


### Introduction
This repository hosts the code used to obtain a representation of an object instance mask as a `Fourier Shape Descriptor` vector.

With Fourier Shape Descriptors (FSDs), one acquires the means to compare two distinct morphologies, by evaluating the distance between their corresponding `Fourier Shape Descriptor` representations.

Additionally, one could compare cells and nuclei across time, which could enable tracking based on position and morphology. 

### Dependencies 

One could execute these lines of code to run this branch:

```
conda create -n FSDEnv python==3.8
conda activate FSDEnv
git clone https://github.com/lmanan/Fourier-Shape-Descriptors.git
cd Fourier-Shape-Descriptors
pip install -e .
```

### Getting Started

Look in the `examples` directory,  and try out the `BBBC020` notebooks for 2D images. Please make sure to select `Kernel > Change kernel` to `FSDEnv`.   


### Datasets
`BBBC020` dataset is made available as release asset **[here](https://github.com/lmanan/Fourier-Shape-Descriptors/releases/tag/v0.0.1)**. 

### How to use on your data?
   
`*.tif`-type images and the corresponding instance masks should be respectively present under `images` and `masks`. (In order to prepare such instance masks, one could use the Fiji plugin <b>Labkit</b> as suggested <b>[here](https://github.com/juglab/EmbedSeg/wiki/01---Use-Labkit-to-prepare-instance-masks)</b>). The following would be a desired structure as to how data should be prepared.

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

[file an issue]: https://github.com/lmanan/Fourier-Shape-Descriptors/issues


