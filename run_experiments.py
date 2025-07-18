import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import yaml
import warnings
warnings.filterwarnings('ignore')

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

np.random.seed(config['experiment']['random_seed'])
torch.manual_seed(config['experiment']['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(config['experiment']['random_seed'])

def calculate_annual_change_features(df, tau_regions):
    processed_data = []
    
    for idx, row in df.iterrows():
        has_v1 = all([not pd.isna(row.get(f'V1_{region}_SUVR', np.nan)) for region in tau_regions])
        has_v2 = all([not pd.isna(row.get(f'V2_{region}_SUVR', np.nan)) for region in tau_regions])
        
        if not (has_v1 and has_v2):
            continue
            
        features = []
        
        v1_date = pd.to_datetime(row['V1_SCANDATE'])
        v2_date = pd.to_datetime(row['V2_SCANDATE'])
        time_interval = (v2_date - v1_date).days / 365.25
        
        for region in tau_regions:
            v1_value = row[f'V1_{region}_SUVR']
            v2_value = row[f'V2_{region}_SUVR']
            annual_change = (v2_value - v1_value) / time_interval if time_interval > 0 else 0
            features.append(annual_change)
        
        for region in tau_regions:
            features.append(row[f'V1_{region}_SUVR'])
        
        for region in tau_regions:
            v1_value = row[f'V1_{region}_SUVR']
            v2_value = row[f'V2_{region}_SUVR']
            percent_change = ((v2_value - v1_value) / v1_value * 100) if v1_value != 0 else 0
            features.append(percent_change)
        
        annual_changes = features[:len(tau_regions)]
        features.append(np.std(annual_changes))
        features.append(np.max(annual_changes))
        features.append(np.mean(annual_changes))
        
        age = row.get('BL_Age', 70)
        if isinstance(age, str) and '>' in age:
            age = 90
        features.append(float(age))
        
        features.append(1 if row.get('PTGENDER') == 1 else 0)
        
        education = row.get('PTEDUCAT', 16)
        if pd.isna(education):
            education = 16
        features.append(float(education))
        
        apoe_status = row.get('APOEGrp', 0)
        if pd.isna(apoe_status):
            apoe_status = 0
        features.append(int(apoe_status))
        
        dx = row['BL_DXGrp']
        label = 1 if dx in [1.0, 5.0] else 0
        
        processed_data.append({
            'RID': row['RID'],
            'features': features,
            'label': label
        })
    
    return processed_data

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=32, num_classes=2):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class MultiRelationalGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout=0.5, num_heads=4):
        super(MultiRelationalGNN, self).__init__()
        
        self.gat_similarity = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat_clinical = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat_progression = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
        
        self.combine = torch.nn.Linear(hidden_dim * num_heads * 3, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, edge_index_similarity, edge_index_clinical, edge_index_progression):
        h_sim = F.elu(self.gat_similarity(x, edge_index_similarity))
        h_clin = F.elu(self.gat_clinical(x, edge_index_clinical))
        h_prog = F.elu(self.gat_progression(x, edge_index_progression))
        
        h_combined = torch.cat([h_sim, h_clin, h_prog], dim=1)
        h = F.elu(self.combine(h_combined))
        h = self.dropout(h)
        h = F.elu(self.fc1(h))
        h = self.dropout(h)
        out = self.fc2(h)
        
        return out

def create_knn_graph(features, k=5):
    n_nodes = len(features)
    edges = []
    
    for i in range(n_nodes):
        distances = []
        for j in range(n_nodes):
            if i != j:
                dist = np.linalg.norm(features[i] - features[j])
                distances.append((j, dist))
        
        distances.sort(key=lambda x: x[1])
        for j, _ in distances[:k]:
            edges.append([i, j])
    
    return torch.tensor(edges, dtype=torch.long).t()

def create_similarity_graph(features, threshold):
    n_nodes = len(features)
    edges = []
    
    annual_change_features = features[:, :9]
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            corr = np.corrcoef(annual_change_features[i], annual_change_features[j])[0, 1]
            if corr > threshold:
                edges.append([i, j])
                edges.append([j, i])
    
    if len(edges) == 0:
        for i in range(n_nodes):
            if i > 0:
                edges.append([i, i-1])
                edges.append([i-1, i])
    
    return torch.tensor(edges, dtype=torch.long).t()

def create_clinical_graph(labels, k):
    n_nodes = len(labels)
    edges = []
    
    for i in range(n_nodes):
        same_label_indices = [j for j in range(n_nodes) if labels[j] == labels[i] and j != i]
        if len(same_label_indices) > 0:
            neighbors = np.random.choice(same_label_indices, 
                                       min(k, len(same_label_indices)), 
                                       replace=False)
            for j in neighbors:
                edges.append([i, j])
    
    return torch.tensor(edges, dtype=torch.long).t()

def create_progression_graph(features, k):
    n_nodes = len(features)
    edges = []
    
    annual_change_features = features[:, :9]
    
    for i in range(n_nodes):
        distances = []
        for j in range(n_nodes):
            if i != j:
                dist = np.linalg.norm(annual_change_features[i] - annual_change_features[j])
                distances.append((j, dist))
        
        distances.sort(key=lambda x: x[1])
        for j, _ in distances[:k]:
            edges.append([i, j])
    
    return torch.tensor(edges, dtype=torch.long).t()

def split_graph_properly(edge_index, train_indices, test_indices):
    train_indices_set = set(train_indices)
    test_indices_set = set(test_indices)
    
    train_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(train_indices)}
    test_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(test_indices)}
    
    train_edges = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src in train_indices_set and dst in train_indices_set:
            train_edges.append([train_idx_map[src], train_idx_map[dst]])
    
    test_edges = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src in test_indices_set and dst in test_indices_set:
            test_edges.append([test_idx_map[src], test_idx_map[dst]])
    
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t() if train_edges else torch.zeros((2, 0), dtype=torch.long)
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t() if test_edges else torch.zeros((2, 0), dtype=torch.long)
    
    return train_edge_index, test_edge_index

print("=== ADNI TAU PET ANALYSIS ===")

df = pd.read_excel(config['data']['raw_data_path'], sheet_name=config['data']['sheet_name'])
tau_regions = config['tau_regions']

processed_data = calculate_annual_change_features(df, tau_regions)
print(f"Total subjects with longitudinal data: {len(processed_data)}")

features = np.array([d['features'] for d in processed_data])
labels = np.array([d['label'] for d in processed_data])

print(f"Feature dimensions: {features.shape}")
print(f"Class distribution: HC/SMC={sum(labels==1)}, MCI/AD={sum(labels==0)}")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

cv = StratifiedKFold(n_splits=config['experiment']['n_folds'], shuffle=True, random_state=config['experiment']['random_seed'])

print("\n=== TRADITIONAL MACHINE LEARNING ===")
ml_models = {
    'Logistic Regression': LogisticRegression(**config['ml_params']['logistic_regression'], random_state=config['experiment']['random_seed']),
    'SVM': SVC(**config['ml_params']['svm'], random_state=config['experiment']['random_seed']),
    'Random Forest': RandomForestClassifier(**config['ml_params']['random_forest'], random_state=config['experiment']['random_seed'])
}

ml_results = []
for model_name, model in ml_models.items():
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
    
    ml_results.extend(fold_results)

ml_df = pd.DataFrame(ml_results)
ml_summary = ml_df.groupby('model').agg({'auc': ['mean', 'std'], 'accuracy': ['mean', 'std'], 
                                         'precision': ['mean', 'std'], 'recall': ['mean', 'std']})
print(ml_summary)

print("\n=== SIMPLE GNN ===")
simple_gnn_results = []

for fold, (train_idx, test_idx) in enumerate(cv.split(features_scaled, labels)):
    X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    edge_index = create_knn_graph(features_scaled, k=5)
    train_edge_index, test_edge_index = split_graph_properly(edge_index, train_idx, test_idx)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    model = SimpleGNN(num_features=features_scaled.shape[1], hidden_dim=32, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(X_train_tensor, train_edge_index)
        loss = criterion(out, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_tensor, test_edge_index)
        test_probs = F.softmax(test_out, dim=1)[:, 1].numpy()
        test_preds = test_out.argmax(dim=1).numpy()
        
        simple_gnn_results.append({
            'fold': fold + 1,
            'auc': roc_auc_score(y_test, test_probs),
            'accuracy': accuracy_score(y_test, test_preds),
            'precision': precision_score(y_test, test_preds),
            'recall': recall_score(y_test, test_preds)
        })

simple_gnn_df = pd.DataFrame(simple_gnn_results)
print(f"Simple GNN - AUC: {simple_gnn_df['auc'].mean():.3f} ± {simple_gnn_df['auc'].std():.3f}")

print("\n=== ENHANCED GNN ===")
enhanced_gnn_results = []

for fold, (train_idx, test_idx) in enumerate(cv.split(features_scaled, labels)):
    X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    edge_index_similarity = create_similarity_graph(features_scaled, config['graph_params']['similarity_threshold'])
    edge_index_clinical = create_clinical_graph(labels, config['graph_params']['k_neighbors'])
    edge_index_progression = create_progression_graph(features_scaled, config['graph_params']['k_neighbors'])
    
    train_edge_sim, test_edge_sim = split_graph_properly(edge_index_similarity, train_idx, test_idx)
    train_edge_clin, test_edge_clin = split_graph_properly(edge_index_clinical, train_idx, test_idx)
    train_edge_prog, test_edge_prog = split_graph_properly(edge_index_progression, train_idx, test_idx)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    model = MultiRelationalGNN(
        num_features=features_scaled.shape[1],
        hidden_dim=config['gnn_params']['hidden_dim'],
        num_classes=2,
        dropout=config['gnn_params']['dropout'],
        num_heads=config['gnn_params']['num_heads']
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['gnn_params']['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(config['gnn_params']['epochs']):
        optimizer.zero_grad()
        out = model(X_train_tensor, train_edge_sim, train_edge_clin, train_edge_prog)
        loss = criterion(out, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_tensor, test_edge_sim, test_edge_clin, test_edge_prog)
        test_probs = F.softmax(test_out, dim=1)[:, 1].numpy()
        test_preds = test_out.argmax(dim=1).numpy()
        
        enhanced_gnn_results.append({
            'fold': fold + 1,
            'auc': roc_auc_score(y_test, test_probs),
            'accuracy': accuracy_score(y_test, test_preds),
            'precision': precision_score(y_test, test_preds),
            'recall': recall_score(y_test, test_preds)
        })

enhanced_gnn_df = pd.DataFrame(enhanced_gnn_results)
print(f"Enhanced GNN - AUC: {enhanced_gnn_df['auc'].mean():.3f} ± {enhanced_gnn_df['auc'].std():.3f}")

print("\n=== FINAL RESULTS SUMMARY ===")
all_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'Simple GNN', 'Enhanced GNN'],
    'AUC': [
        ml_df[ml_df['model']=='Logistic Regression']['auc'].mean(),
        ml_df[ml_df['model']=='SVM']['auc'].mean(),
        ml_df[ml_df['model']=='Random Forest']['auc'].mean(),
        simple_gnn_df['auc'].mean(),
        enhanced_gnn_df['auc'].mean()
    ],
    'AUC_std': [
        ml_df[ml_df['model']=='Logistic Regression']['auc'].std(),
        ml_df[ml_df['model']=='SVM']['auc'].std(),
        ml_df[ml_df['model']=='Random Forest']['auc'].std(),
        simple_gnn_df['auc'].std(),
        enhanced_gnn_df['auc'].std()
    ]
})
all_results = all_results.sort_values('AUC', ascending=False)
print(all_results)

ml_df.to_csv('results/ml_results.csv', index=False)
simple_gnn_df.to_csv('results/simple_gnn_results.csv', index=False)
enhanced_gnn_df.to_csv('results/enhanced_gnn_results.csv', index=False)
all_results.to_csv('results/final_summary.csv', index=False)