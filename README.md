# ADNI Tau PET Analysis with Enhanced Graph Neural Networks

This repository contains the implementation for analyzing ADNI tau PET data using Enhanced Graph Neural Networks (GNN) with longitudinal features.

## Overview

This study analyzes tau protein accumulation patterns in Alzheimer's disease using longitudinal tau PET data from the ADNI database. We compare three approaches:
- Traditional Machine Learning (Logistic Regression, SVM, Random Forest)
- Simple Graph Neural Network (GCN)
- Enhanced Multi-relational Graph Neural Network

## Key Features

- Longitudinal tau PET feature engineering (annual change rates, percent changes)
- Multi-relational graph construction (similarity, clinical, progression)
- Comprehensive evaluation with 5-fold cross-validation
- Analysis of 377 subjects with complete V1 and V2 tau PET data

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_experiments.py
```

## Configuration

Edit `config.yaml` to modify:
- Data paths
- Model hyperparameters
- Graph construction parameters
- Experiment settings

## Citation

If you use this code, please cite our work on enhanced graph neural networks for tau PET analysis in Alzheimer's disease research.

---

© Dr. Jo's Medical AI Research Lab, Indiana University School of Medicine | www.jolab.ai