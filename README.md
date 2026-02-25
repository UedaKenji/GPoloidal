# GPoloidal
GPoloidal is a Python package for constructing Gaussian process models on poloidal cross-sections, with support for nonlinear Gaussian process tomography, primarily used in fusion research.

## Installation

This repository is configured for `uv`-based development and installation.

### Method 1: Clone and sync with uv

Clone the repository and create a project environment with dependencies:

```bash
# Clone the repository
git clone https://github.com/UedaKenji/GPoloidal.git

# Navigate to the project directory
cd GPoloidal

# Sync runtime dependencies and install this package in the environment
uv sync

# (Optional) Include development tools
uv sync --group dev
```

### Method 2: Install directly from the GitHub URL (uv)

Install the package directly from the GitHub repository:

```bash
uv pip install git+https://github.com/UedaKenji/GPoloidal.git
```



## Bibtex citation

When using this project's nonlinear(log) gaussian process tomography method in a scientific publication, we would appriciate the following citation:

```
@article{
doi = {10.1088/2632-2153/adbbae},
url = {https://dx.doi.org/10.1088/2632-2153/adbbae},
year = {2025},
month = {mar},
publisher = {IOP Publishing},
volume = {6},
number = {1},
pages = {015061},
author = {Ueda, Kenji and Nishiura, Masaki},
title = {Nonlinear Gaussian process tomography with imposed non-negativity constraints on physical quantities for plasma diagnostics},
journal = {Machine Learning: Science and Technology}
```
