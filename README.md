# PyShapeDTW 🔄

A Python implementation of Shape-based Dynamic Time Warping (ShapeDTW) for time series analysis, featuring multiple shape descriptors and evaluation frameworks.

## 🌟 Overview

PyShapeDTW provides advanced time series similarity computation through shape-based features. It extends traditional Dynamic Time Warping (DTW) by incorporating local shape information, making it more robust to amplitude variations and noise while preserving important shape characteristics.

## 🏗️ Architecture

The project is organized into several key components:

### Core components 🔧

- **Elastic measures**: The fundamental distance computation algorithms
  - `shape_dtw.py`: Main ShapeDTW implementation
  - `derivative_dtw.py`: DTW variant using derivative information
  - `base_dtw.py`: Base DTW functionality

- **Shape fescriptors**: Different ways to capture local shape information
  - `HOG1D`: 1D Histogram of Oriented Gradients
  - `PAA`: Piecewise Aggregate Approximation
  - `Wavelets`: Discrete Wavelet Transform descriptors

### Evaluation frameworks 📊

- **Classification pipeline**: Evaluates the performance of different DTW variants on time series classification tasks
- **Alignment pipeline**: Assesses the quality of sequence alignments produced by different methods

## 🚀 Main features

- **Multiple Shape descriptors**: Choose from various shape description methods
- **Flexible evaluation**: Comprehensive frameworks for both classification and alignment tasks
- **Easy integration**: Clean API design for incorporating new descriptors or evaluation metrics
- **UCR dataset support**: Built-in support for the UCR time series dataset collection

## 💻 Command Line Interface

The package provides a Typer-based CLI with two main commands:

```bash
# Run classification evaluation
python -m pyshapeDTW ucr-classification

# Run alignment evaluation
python -m pyshapeDTW ucr-alignment
```

### Classification task 🎯
Evaluates how well different DTW variants can classify time series by comparing their performance on standard benchmark datasets. The pipeline:
1. Loads time series data
2. Computes shape descriptors (if using ShapeDTW)
3. Performs nearest-neighbor classification
4. Evaluates accuracy metrics

Can be run for any UCR dataset.

### Alignment task 🔗
Assesses the quality of sequence alignments by:
1. Generating synthetic warped sequences from the original data
2. Attempting to recover the original warping
3. Comparing the recovered alignment with ground truth
4. Visualizing the results

Can be run for any UCR dataset.

## 🛠️ Installation

```bash
conda create -n shapeDTW
conda activate shapeDTW
pip install uv
uv pip install -e ".[all]"
```
To run tests : 
```bash
pytest
```

## 📚 Documentation

Detailed documentation is available in the docstrings. Each module is extensively documented with:
- Function and class descriptions
- Parameter specifications
- Usage examples
- Return value descriptions