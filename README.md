# Deepwave

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3829886.svg)](https://doi.org/10.5281/zenodo.3829886)

Deepwave provides wave propagation modules for PyTorch, for applications such as seismic imaging/inversion. You can use it to perform forward modelling and backpropagation, so it can simulate wave propagation to generate synthetic data, invert for the scattering potential (RTM/LSRTM) or wavespeed (FWI), or you can use it to integrate wave propagation into a larger chain of operations.

## Features
- Supports regular and Born modelling of the 2D constant density acoustic / scalar wave equation
- Runs on CPUs and appropriate GPUs
- The gradient of all outputs (final wavefields and receiver data) can be calculated with respect to the wavespeed, scattering potential, initial wavefields, and source amplitudes
- Uses the [Pasalic and McGarry](https://doi.org/10.1190/1.3513453) PML for accurate absorbing boundaries
- The PML width for each edge can be set independently, allowing a free surface (no PML) on any side
- Finite difference accuracy can be set by the user
- A region of the model around the sources and receivers currently being propagated can be automatically extracted to avoid the unnecessary computation of propagation in distant parts of the model

## Get started

[The documentation](https://ausargeo.pages.dev/deepwave) contains examples and instructions on how to install and use Deepwave.

## Note about v0.0.10

The v0.0.10 release of Deepwave involved a complete rewrite of the code. This resulted in several improvements such as new features, but also removed 1D and 3D propagators to allow greater focus on the more popular 2D propagators. It also involved changes to the interface. The most important of these are that source and receiver coordinates are now provided as integers in units of cells rather than floats in the same units as `grid_spacing`, and that the time dimension is now the final rather than the first dimension.
