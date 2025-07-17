# ADNI Tau PET Analysis with Enhanced Graph Neural Networks

This repository contains the corrected implementation for analyzing ADNI tau PET data using Enhanced Graph Neural Networks (GNN) with longitudinal features.

## Overview

This study analyzes tau protein accumulation patterns in Alzheimer's disease using longitudinal tau PET data from the ADNI database. The implementation includes both traditional machine learning approaches and an enhanced multi-relational Graph Neural Network.

## Features

### Longitudinal Feature Engineering
- **Annual change rates**: (V2 - V1) / time_interval
- **Baseline values**: V1 SUVR (StandardScaler normalized)
- **Percent change**: [(V2 - V1) / V1] × 100
- **Engineered features**: variance, max, and mean of changes across regions
- **Demographics**: age, gender, education
- **APOE ε4 carrier status**

### Multi-Relational Graph Neural Network
- **Feature similarity graph**: based on annual change patterns
- **Clinical similarity graph**: same diagnosis groups
- **Progression pattern graph**: similar tau progression rates
- **Separate GAT layers** for each graph type with proper indexing

## Dataset

- **Source**: ADNI tau PET SUVR data (`Tau-ADNI.xlsx`)
- **Subjects**: 377 with complete V1 and V2 data
- **Brain regions**: 9 tau PET regions
- **Features**: 34 per subject
- **Classification**: HC/SMC vs MCI/AD

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

## Results

### Performance Comparison
- **Enhanced GNN**: 0.791 ± 0.081 AUC
- **SVM Baseline**: 0.748 ± 0.062 AUC
- **Logistic Regression**: 0.759 ± 0.049 AUC
- **Random Forest**: 0.746 ± 0.066 AUC

### Key Findings
- Enhanced GNN achieves 5.7% improvement over best traditional ML
- Most important features: ENTCTX baseline, age, MTL baseline
- APOE ε4 status has low predictive importance (0.3%)

## Files

- `baseline_ml.py`: Traditional ML analysis with 5-fold CV
- `enhanced_gnn.py`: Multi-relational GNN implementation
- `data_preprocessor.py`: Longitudinal feature extraction
- `config.yaml`: Configuration parameters
- `results/`: All experimental results and feature importance

## Key Corrections

This implementation fixes critical issues in the original code:
- **Proper graph indexing** for train/test splits
- **Separate GAT layers** for each graph type
- **Longitudinal feature engineering** with proper time interval calculation
- **StandardScaler normalization** instead of manual scaling

## Citation

If you use this code, please cite our work on enhanced graph neural networks for tau PET analysis in Alzheimer's disease research.