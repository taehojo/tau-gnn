#!/bin/bash

echo "Running ADNI Tau PET Analysis..."
echo "================================"

echo -e "\n1. Running Baseline ML Analysis..."
python baseline_ml.py

echo -e "\n2. Running Enhanced GNN Analysis..."
python enhanced_gnn.py

echo -e "\nAll analyses completed. Check results/ directory for outputs."