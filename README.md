# ADNI Tau PET Analysis with Enhanced Graph Neural Networks

This repository contains the corrected implementation for analyzing ADNI tau PET data using Enhanced Graph Neural Networks (GNN) with longitudinal features.

## Overview

This study analyzes tau protein accumulation patterns in Alzheimer's disease using longitudinal tau PET data from the ADNI database. The implementation includes both traditional machine learning approaches and an enhanced multi-relational Graph Neural Network.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Run Individual Analyses

```bash
# Traditional ML models
python baseline_ml.py

# Enhanced GNN analysis
python enhanced_gnn.py
```

### Run All Analyses

```bash
./run_all.sh
```

## Citation

If you use this code, please cite our work on enhanced graph neural networks for tau PET analysis in Alzheimer's disease research.