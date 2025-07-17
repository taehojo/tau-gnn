import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import yaml
import warnings
from data_preprocessor import calculate_annual_change_features, get_feature_names
warnings.filterwarnings('ignore')

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

np.random.seed(config['experiment']['random_seed'])

print("=== BASELINE ML ANALYSIS (WITH ANNUAL CHANGE RATES) ===")

df = pd.read_excel(config['data']['raw_data_path'], sheet_name=config['data']['sheet_name'])
tau_regions = config['tau_regions']

# Calculate longitudinal features
processed_data = calculate_annual_change_features(df, tau_regions)
print(f"Total subjects with longitudinal data: {len(processed_data)}")

# Extract features and labels
features = np.array([d['features'] for d in processed_data])
labels = np.array([d['label'] for d in processed_data])

print(f"Feature dimensions: {features.shape}")
print(f"Class distribution: HC/SMC={sum(labels==1)}, MCI/AD={sum(labels==0)}")

# Get feature names for interpretation
feature_names = get_feature_names(tau_regions)
print(f"Total features: {len(feature_names)}")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

models = {
    'SVM': SVC(**config['ml_params']['svm'], random_state=config['experiment']['random_seed']),
    'Random Forest': RandomForestClassifier(**config['ml_params']['random_forest'], random_state=config['experiment']['random_seed']),
    'Logistic Regression': LogisticRegression(**config['ml_params']['logistic_regression'], random_state=config['experiment']['random_seed'])
}

cv = StratifiedKFold(n_splits=config['experiment']['n_folds'], shuffle=True, random_state=config['experiment']['random_seed'])
results = []

for model_name, model in models.items():
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(features_scaled, labels)):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        fold_results.append({
            'model': model_name,
            'fold': fold + 1,
            'auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        })
    
    results.extend(fold_results)

results_df = pd.DataFrame(results)
summary = results_df.groupby('model').agg({
    'auc': ['mean', 'std'],
    'accuracy': ['mean', 'std'],
    'precision': ['mean', 'std'],
    'recall': ['mean', 'std']
})

print("\n=== RESULTS ===")
print(summary)

# Feature importance for Random Forest
rf_model = RandomForestClassifier(**config['ml_params']['random_forest'], random_state=config['experiment']['random_seed'])
rf_model.fit(features_scaled, labels)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== TOP 10 IMPORTANT FEATURES (Random Forest) ===")
print(feature_importance.head(10))

results_df.to_csv('results/baseline_ml_longitudinal_results.csv', index=False)
summary.to_csv('results/baseline_ml_longitudinal_summary.csv')
feature_importance.to_csv('results/feature_importance.csv', index=False)

print("\nResults saved to results/")