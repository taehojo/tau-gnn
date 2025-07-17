import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import yaml
import warnings
from data_preprocessor import calculate_annual_change_features, get_feature_names
warnings.filterwarnings('ignore')

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

np.random.seed(config['experiment']['random_seed'])
torch.manual_seed(config['experiment']['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(config['experiment']['random_seed'])

print("=== ENHANCED GNN ANALYSIS (WITH ANNUAL CHANGE RATES) ===")

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

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


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


def create_similarity_graph(features, threshold):
    n_nodes = len(features)
    edges = []
    
    # Focus on annual change features (first 9 features)
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
    
    # Use annual change features for progression similarity
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


cv = StratifiedKFold(n_splits=config['experiment']['n_folds'], shuffle=True, random_state=config['experiment']['random_seed'])
results = []

print("\nStarting 5-fold cross-validation...")

for fold, (train_idx, test_idx) in enumerate(cv.split(features_scaled, labels)):
    print(f"\nFold {fold + 1}/{config['experiment']['n_folds']}")
    
    X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    edge_index_similarity = create_similarity_graph(features_scaled, config['graph_params']['similarity_threshold'])
    edge_index_clinical = create_clinical_graph(labels, config['graph_params']['k_neighbors'])
    edge_index_progression = create_progression_graph(features_scaled, config['graph_params']['k_neighbors'])
    
    train_edge_sim, test_edge_sim = split_graph_properly(edge_index_similarity, train_idx, test_idx)
    train_edge_clin, test_edge_clin = split_graph_properly(edge_index_clinical, train_idx, test_idx)
    train_edge_prog, test_edge_prog = split_graph_properly(edge_index_progression, train_idx, test_idx)
    
    print(f"Train edges: Sim={train_edge_sim.shape[1]}, Clin={train_edge_clin.shape[1]}, Prog={train_edge_prog.shape[1]}")
    print(f"Test edges: Sim={test_edge_sim.shape[1]}, Clin={test_edge_clin.shape[1]}, Prog={test_edge_prog.shape[1]}")
    
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
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_tensor, test_edge_sim, test_edge_clin, test_edge_prog)
        test_probs = F.softmax(test_out, dim=1)[:, 1].numpy()
        test_preds = test_out.argmax(dim=1).numpy()
        
        results.append({
            'fold': fold + 1,
            'auc': roc_auc_score(y_test, test_probs),
            'accuracy': accuracy_score(y_test, test_preds),
            'precision': precision_score(y_test, test_preds),
            'recall': recall_score(y_test, test_preds)
        })
        
        print(f"  Results - AUC: {results[-1]['auc']:.3f}, Acc: {results[-1]['accuracy']:.3f}")

results_df = pd.DataFrame(results)
print("\n=== FINAL RESULTS ===")
print(f"AUC: {results_df['auc'].mean():.3f} ± {results_df['auc'].std():.3f}")
print(f"Accuracy: {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")
print(f"Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}")
print(f"Recall: {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}")

results_df.to_csv('results/enhanced_gnn_longitudinal_results.csv', index=False)
print("\nResults saved to results/enhanced_gnn_longitudinal_results.csv")