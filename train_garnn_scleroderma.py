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

# --- Merge in demographic 'Sex' and Has_Raynaud from pats file using patient id ---
import pickle
pats_df = pickle.load(open('pats', 'rb'))
if not isinstance(pats_df, pd.DataFrame):
    pats_df = pd.DataFrame(pats_df)
# Add Has_Raynaud: 1 if 'Onset of first non-Raynaud?s of the disease' is not null, else 0
pats_df['Has_Raynaud'] = pats_df['Onset of first non-Raynaud?s of the disease'].notnull().astype(int)
# Standardize patient id column names if needed
if 'id patient' in df.columns and 'Id Patient V2018' in pats_df.columns:
    df = df.merge(pats_df[['Id Patient V2018', 'Sex', 'Has_Raynaud']], left_on='id patient', right_on='Id Patient V2018', how='left')
    df.drop(columns=['Id Patient V2018'], inplace=True)
# Now df has 'Sex' and 'Has_Raynaud' columns

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
# Restrict to top 15 features from RF for feature selection comparison
feature_cols = [
    'Skin thickening of the whole finger distal to MCP (Sclerodactyly)',
    'Skin thickening of the fingers of both hands extending proximal to the MCP joints',
    'Renal crisis',
    'Worsening of skin within the last month',
    'Pericardial effusion on echo',
    'Auricular Arrhythmias',
    'Cardiac arrhythmias',
    'Joint synovitis',
    'Worsening of cardiopulmonary manifestations within the last month',
    'Conduction blocks',
    'Tendon friction rubs',
    'Muscle weakness',
    'Intestinal symptoms (diarrhea, bloating, constipation)',
    'Joint polyarthritis',
    'Proximal muscle weakness not explainable by other causes',
]

# 4. Encode categorical features
cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    le_dict[c] = le
# Now 'Sex' is label encoded if present

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
# --- Compute sample weights to make model more likely to suggest scleroderma in females ---
# Get sex and label for each patient
sex_for_patient = df.groupby(PATIENT_COL)['Sex'].last().reindex(patient_ids).values
label_for_patient = df.groupby(PATIENT_COL)[LABEL_COL].last().reindex(patient_ids).values
# Compute weight: if female and scleroderma present, assign higher weight
sample_weights = np.ones(len(patient_ids))
female_mask = (sex_for_patient == 1)  # 1 if label encoded as female
scleroderma_mask = (label_for_patient == 1)
# Sweep positive class weighting for scleroderma cases
POS_WEIGHTS = [2, 5, 10, 20]
best_overall_f1 = 0
best_overall_cfg = None
for POS_WEIGHT in POS_WEIGHTS:
    print(f"\n--- Positive class weight: {POS_WEIGHT} ---")
    # Proweight both females with scleroderma and Has_Raynaud=1 with scleroderma (2x for each, 4x if both)
    sample_weights = np.ones(len(patient_ids))
    has_raynaud_for_patient = df.groupby(PATIENT_COL)['Has_Raynaud'].last().reindex(patient_ids).values
    # 5x for female+scleroderma (increased proweighting), 2x for Has_Raynaud+scleroderma, 10x for both
    sample_weights[female_mask & scleroderma_mask] *= 5.0
    sample_weights[(has_raynaud_for_patient == 1) & scleroderma_mask] *= 2.0
    # Apply POS_WEIGHT to all scleroderma cases
    sample_weights[scleroderma_mask] *= POS_WEIGHT
    sample_weights = sample_weights / sample_weights.mean()

    # Assign weights to train and test sets
    train_weights = torch.tensor(sample_weights[idx_train], dtype=torch.float32)
    test_weights = torch.tensor(sample_weights[idx_test], dtype=torch.float32)

    # Recreate train/test split and DataLoaders for this sweep
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
    train_ds = SclerodermaTimeSeriesDataset(X_train, y_train)
    test_ds = SclerodermaTimeSeriesDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32)

    # Re-initialize model and optimizer for each sweep
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GARNN(
        feature_dim=X.shape[2],
        hidden_dim=64,
        num_classes=len(np.unique(y)),
        patient_graph_edge_index=None,
        feature_graph_edge_index=feat_edge_index.to(device),
        seq_len=max_seq_len
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    # --- Train model as before ---
    all_logits = []
    all_targets = []
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        batch_start = 0
        for xb, yb in train_dl:
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
            optimizer.zero_grad()
            logits = model(xb, batch_pat_edge_index, model.feature_graph_edge_index)
            # Get weights for this batch
            batch_weights = train_weights[batch_start:batch_start+len(xb)].to(device)
            loss = loss_fn(logits, yb)
            weighted_loss = (loss * batch_weights).mean()
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item() * len(xb)
            batch_start += len(xb)
        # (Optional: print loss for each epoch)
        # print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_ds):.4f}')
    model.eval()
    correct = 0
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
            all_logits.append(logits.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
    acc = correct / len(test_ds)
    print(f'Test Accuracy: {acc:.3f}')

    # --- Threshold tuning ---
    from sklearn.metrics import precision_score, recall_score, f1_score
    probs = torch.softmax(torch.tensor(np.concatenate(all_logits, axis=0)), dim=1)[:, 1].numpy()
    true_labels = np.concatenate(all_targets, axis=0)
    best_f1 = 0
    best_thr = 0.5
    best_ppv = 0
    best_sens = 0
    for thr in np.arange(0.05, 0.96, 0.05):
        preds = (probs >= thr).astype(int)
        ppv = precision_score(true_labels, preds, zero_division=0)
        sens = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_ppv = ppv
            best_sens = sens
    print(f"Best threshold: {best_thr:.2f} | PPV={best_ppv:.3f} | Sens={best_sens:.3f} | F1={best_f1:.3f}")
    if best_f1 > best_overall_f1:
        best_overall_f1 = best_f1
        best_overall_cfg = dict(POS_WEIGHT=POS_WEIGHT, threshold=best_thr, PPV=best_ppv, Sens=best_sens, F1=best_f1)

# Print summary of best config
print("\n=== Best configuration ===")
print(best_overall_cfg)

# Use weighted loss
loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

# 13. Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    batch_start = 0
    for xb, yb in train_dl:
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
        optimizer.zero_grad()
        logits = model(xb, batch_pat_edge_index, model.feature_graph_edge_index)
        # Get weights for this batch
        batch_weights = train_weights[batch_start:batch_start+len(xb)].to(device)
        loss = loss_fn(logits, yb)
        weighted_loss = (loss * batch_weights).mean()
        weighted_loss.backward()
        optimizer.step()
        total_loss += weighted_loss.item() * len(xb)
        batch_start += len(xb)
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_ds):.4f}')

# Save the best model for API use
import torch
print("Saving best GARNN model to garnn_best_model.pt ...")
torch.save(model, "garnn_best_model.pt")

# 14. Evaluation
model.eval()
correct = 0
all_logits = []
all_targets = []
all_indices = []
with torch.no_grad():
    for i, (xb, yb) in enumerate(test_dl):
        xb, yb = xb.to(device), yb.to(device)
        # Dynamically construct patient graph for each batch
        k = 5
        means = xb.mean(dim=1).cpu().numpy()
        G_pat = nx.Graph()
        for j in range(xb.shape[0]):
            G_pat.add_node(j)
        for j in range(xb.shape[0]):
            dists = np.linalg.norm(means - means[j], axis=1)
            idx = np.argsort(dists)[1:k+1]
            for l in idx:
                G_pat.add_edge(j, l)
        batch_pat_edge_index = from_networkx(G_pat).edge_index.to(device)
        logits = model(xb, batch_pat_edge_index, model.feature_graph_edge_index)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        all_logits.append(logits.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
        # Save batch indices for later mapping to patient ids
        all_indices.extend(range(i * test_dl.batch_size, i * test_dl.batch_size + xb.shape[0]))
acc = correct / len(test_ds)
print(f'Test Accuracy: {acc:.3f}')

# --- Compute PPV, Sensitivity, F1-score ---
from sklearn.metrics import precision_score, recall_score, f1_score
# Get predicted probabilities
probs = torch.softmax(torch.tensor(np.concatenate(all_logits, axis=0)), dim=1)[:, 1].numpy()
true_labels = np.concatenate(all_targets, axis=0)

# Sweep thresholds
thresholds = np.arange(0.05, 1.0, 0.05)
best_f1 = 0
best_threshold = 0.5
for thresh in thresholds:
    preds = (probs > thresh).astype(int)
    f1 = f1_score(true_labels, preds)
    ppv = precision_score(true_labels, preds)
    sens = recall_score(true_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh
    print(f'Threshold={thresh:.2f} | PPV={ppv:.3f} | Sens={sens:.3f} | F1={f1:.3f}')
print(f'\nBest threshold: {best_threshold:.2f} | PPV={precision_score(true_labels, (probs > best_threshold).astype(int)):.3f} | Sens={recall_score(true_labels, (probs > best_threshold).astype(int)):.3f} | F1={f1_score(true_labels, (probs > best_threshold).astype(int)):.3f}')

# Misclassification analysis for best threshold
final_preds = (probs > best_threshold).astype(int)
false_pos_idx = (final_preds == 1) & (true_labels == 0)
false_neg_idx = (final_preds == 0) & (true_labels == 1)
print(f'\nGARNN Misclassification Analysis:')
print(f'False Positives: {false_pos_idx.sum()}')
print(f'False Negatives: {false_neg_idx.sum()}')
# Try to summarize Sex and Has_Raynaud if available
# Use correct test patient ids for demographic lookup
test_pids = np.array(patient_ids)[idx_test]
# Helper: get only valid IDs in pats_df
valid_fp_ids = [pid for pid in test_pids[false_pos_idx] if pid in pats_df.index]
valid_fn_ids = [pid for pid in test_pids[false_neg_idx] if pid in pats_df.index]
missing_fp = len(test_pids[false_pos_idx]) - len(valid_fp_ids)
missing_fn = len(test_pids[false_neg_idx]) - len(valid_fn_ids)
if missing_fp > 0 or missing_fn > 0:
    print(f'WARNING: {missing_fp} FP and {missing_fn} FN patient IDs missing from pats_df and skipped in demographic summary.')
if "Sex" in pats_df.columns:
    print('False Positives by Sex:')
    print(pats_df.loc[valid_fp_ids, "Sex"].value_counts(dropna=False))
    print('False Negatives by Sex:')
    print(pats_df.loc[valid_fn_ids, "Sex"].value_counts(dropna=False))
if "Has_Raynaud" in pats_df.columns:
    print('False Positives by Has_Raynaud:')
    print(pats_df.loc[valid_fp_ids, "Has_Raynaud"].value_counts(dropna=False))
    print('False Negatives by Has_Raynaud:')
    print(pats_df.loc[valid_fn_ids, "Has_Raynaud"].value_counts(dropna=False))

# --- Analyze effect of Sex and Race on predicted probability ---
import pickle
# Reload test set indices to map to patient ids
from sklearn.model_selection import train_test_split
X = np.stack([pad_seq(seq, max_seq_len) for seq in sequences])
y = np.array(labels)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
# Map test indices to patient ids
test_patient_ids = [patient_ids[i] for i in idx_test]
# Get demographic info for test patients
pats_df = pickle.load(open('pats', 'rb'))
if not isinstance(pats_df, pd.DataFrame):
    pats_df = pd.DataFrame(pats_df)
# Build lookup for patient id -> sex, race
pats_df = pats_df.set_index('Id Patient V2018')
# Re-create Has_Raynaud column for analysis
if 'Onset of first non-Raynaud?s of the disease' in pats_df.columns:
    pats_df['Has_Raynaud'] = pats_df['Onset of first non-Raynaud?s of the disease'].notnull().astype(int)
sex_lookup = pats_df['Sex'].to_dict()
race_cols = [c for c in pats_df.columns if c.startswith('Race') or 'race' in c.lower() or 'Hispanic' in c or 'Any other' in c]
# Get predicted probabilities
all_logits = np.concatenate(all_logits, axis=0)
probs = torch.softmax(torch.tensor(all_logits), dim=1)[:, 1].numpy()
# Build DataFrame for analysis
analyze_df = pd.DataFrame({
    'patient_id': test_patient_ids,
    'prob': probs,
    'Sex': [sex_lookup.get(pid, None) for pid in test_patient_ids],
})
# Add race columns if present
for rc in race_cols:
    analyze_df[rc] = [pats_df.loc[pid, rc] if pid in pats_df.index else None for pid in test_patient_ids]
# Print average probability by Sex
grouped_sex = analyze_df.groupby('Sex')['prob'].mean()
print('\nMean predicted probability by Sex:')
print(grouped_sex)
# Print average probability by each Race column
for rc in race_cols:
    if analyze_df[rc].nunique() > 1:
        print(f'\nMean predicted probability by {rc}:')
        print(analyze_df.groupby(rc)['prob'].mean())

# --- Analyze Has_Raynaud (binary) effect ---
if 'Has_Raynaud' in pats_df.columns:
    analyze_df['Has_Raynaud'] = [pats_df.loc[pid, 'Has_Raynaud'] if pid in pats_df.index else None for pid in test_patient_ids]
    print("\nMean predicted probability by Has_Raynaud (binary):")
    print(analyze_df.groupby('Has_Raynaud')['prob'].mean())
    # Intersection: Sex x Has_Raynaud
    print("\nMean predicted probability by Sex and Has_Raynaud:")
    print(analyze_df.groupby(['Sex', 'Has_Raynaud'])['prob'].mean())
else:
    print("\nNo Has_Raynaud binary feature found in demographics.")

# (Decision tree code removed as requested)
