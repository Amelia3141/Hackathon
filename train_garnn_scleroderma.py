import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
from garnn_model import GARNN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# 1. Load data
DATA_PATH = 'merged_patient_data.csv'
df = pd.read_csv(DATA_PATH)

# 2. Preprocess: sort by patient and time, encode label
PATIENT_COL = 'id patient'
TIME_COL = 'Visit Date'
LABEL_COL = 'Scleroderma present'

# Parse dates
if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')

# Sort
df = df.sort_values([PATIENT_COL, TIME_COL])

# Encode label
le_label = LabelEncoder()
df[LABEL_COL] = le_label.fit_transform(df[LABEL_COL].astype(str))

# 3. Select features (drop ID, date, label)
feature_cols = [c for c in df.columns if c not in [PATIENT_COL, TIME_COL, LABEL_COL]]

# 4. Encode categorical features
cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    le_dict[c] = le

# 5. Fill missing values
scaler = StandardScaler()
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# 6. Create sequences per patient
grouped = df.groupby(PATIENT_COL)
patient_ids = list(grouped.groups.keys())
sequences = []
labels = []
for pid in patient_ids:
    g = grouped.get_group(pid)
    seq = g[feature_cols].values
    lbl = g[LABEL_COL].values[-1]  # Use last label as target
    sequences.append(seq)
    labels.append(lbl)

# 7. Pad sequences to same length
max_seq_len = max(len(seq) for seq in sequences)
def pad_seq(seq, maxlen):
    if len(seq) < maxlen:
        pad = np.zeros((maxlen - len(seq), seq.shape[1]))
        return np.vstack([pad, seq])
    return seq[-maxlen:]

X = np.stack([pad_seq(seq, max_seq_len) for seq in sequences])
y = np.array(labels)

# 8. Build feature graph (simple: connect features with high correlation)
G_feat = nx.Graph()
for i, c in enumerate(feature_cols):
    G_feat.add_node(i)
from scipy.stats import pearsonr
corr = np.corrcoef(X.reshape(-1, X.shape[2]).T)
for i in range(len(feature_cols)):
    idx = np.argsort(-np.abs(corr[i]))[1:6]  # top 5 correlated
    for j in idx:
        G_feat.add_edge(i, j)
feat_edge_index = from_networkx(G_feat).edge_index

# 10. Train/test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)

# 11. PyTorch Dataset
class SclerodermaTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SclerodermaTimeSeriesDataset(X_train, y_train)
test_ds = SclerodermaTimeSeriesDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32)

# 12. Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GARNN(
    feature_dim=X.shape[2],
    hidden_dim=64,
    num_classes=len(np.unique(y)),
    patient_graph_edge_index=None,  # Not used, handled per batch
    feature_graph_edge_index=feat_edge_index.to(device),
    seq_len=max_seq_len
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# 13. Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        # Dynamically construct patient graph for each batch
        # Connect top-k similar patients within the batch (k=5)
        k = 5
        means = xb.mean(dim=1).cpu().numpy()
        G_pat = nx.Graph()
        for i in range(xb.shape[0]):
            G_pat.add_node(i)
        for i in range(xb.shape[0]):
            dists = np.linalg.norm(means - means[i], axis=1)
            idx = np.argsort(dists)[1:k+1]
            for j in idx:
                G_pat.add_edge(i, j)
        batch_pat_edge_index = from_networkx(G_pat).edge_index.to(device)
        optimizer.zero_grad()
        logits = model(xb, batch_pat_edge_index, model.feature_graph_edge_index)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_ds):.4f}')

# 14. Evaluation
model.eval()
correct = 0
all_logits = []
with torch.no_grad():
    for xb, yb in test_dl:
        xb, yb = xb.to(device), yb.to(device)
        # Dynamically construct patient graph for each batch
        k = 5
        means = xb.mean(dim=1).cpu().numpy()
        G_pat = nx.Graph()
        for i in range(xb.shape[0]):
            G_pat.add_node(i)
        for i in range(xb.shape[0]):
            dists = np.linalg.norm(means - means[i], axis=1)
            idx = np.argsort(dists)[1:k+1]
            for j in idx:
                G_pat.add_edge(i, j)
        batch_pat_edge_index = from_networkx(G_pat).edge_index.to(device)
        logits = model(xb, batch_pat_edge_index, model.feature_graph_edge_index)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        all_logits.append(logits.cpu().numpy())
acc = correct / len(test_ds)
print(f'Test Accuracy: {acc:.3f}')
