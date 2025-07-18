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

Run all experiments with a single command:

```bash
python run_experiments.py
```

This will:
1. Load and preprocess ADNI tau PET data
2. Calculate longitudinal features from V1 and V2 visits
3. Run Traditional ML models (Logistic Regression, SVM, Random Forest)
4. Run Simple GNN (GCN) baseline
5. Run Enhanced Multi-relational GNN
6. Save all results to the `results/` directory

## Results

The Enhanced GNN achieves the best performance:
- Enhanced GNN: AUC 0.791 ± 0.081
- Logistic Regression: AUC 0.759 ± 0.049  
- SVM: AUC 0.748 ± 0.062
- Random Forest: AUC 0.746 ± 0.066
- Simple GNN: AUC 0.681 ± 0.095

## Configuration

Edit `config.yaml` to modify:
- Data paths
- Model hyperparameters
- Graph construction parameters
- Experiment settings

## Data Structure

The code expects ADNI tau PET data with the following columns:
- `RID`: Subject ID
- `BL_DXGrp`: Baseline diagnosis group
- `V1_*_SUVR`: Visit 1 tau SUVR values for each region
- `V2_*_SUVR`: Visit 2 tau SUVR values for each region
- Demographics: Age, Gender, Education, APOE status

## Citation

If you use this code, please cite our work on enhanced graph neural networks for tau PET analysis in Alzheimer's disease research.

---

© Dr. Jo's Medical AI Research Lab, Indiana University School of Medicine | www.jolab.ai