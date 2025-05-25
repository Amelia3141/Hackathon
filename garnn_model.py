import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GARNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, patient_graph_edge_index, feature_graph_edge_index, seq_len):
        super(GARNN, self).__init__()
        # Feature graph attention (features as nodes)
        self.feature_gat = GATConv(feature_dim, feature_dim, heads=1)
        # RNN for patient time series
        self.rnn = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        # Patient graph attention (patients as nodes)
        self.patient_gat = GATConv(hidden_dim, hidden_dim, heads=1)
        # Final classifier
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.patient_graph_edge_index = patient_graph_edge_index
        self.feature_graph_edge_index = feature_graph_edge_index
        self.seq_len = seq_len

    def forward(self, x_seq, patient_edge_index, feature_edge_index):
        # x_seq: (batch, seq_len, feature_dim)
        B, T, F = x_seq.shape
        # Reshape for feature graph: (B*T, F)
        x_flat = x_seq.reshape(-1, F)
        # Feature graph attention
        x_flat = self.feature_gat(x_flat, feature_edge_index)
        # Reshape back to (B, T, F)
        x_seq = x_flat.reshape(B, T, F)
        # RNN over time
        out_rnn, h_n = self.rnn(x_seq)  # out_rnn: (B, T, hidden_dim)
        # Take last time step
        out_last = out_rnn[:, -1, :]  # (B, hidden_dim)
        # Patient graph attention
        out_pat = self.patient_gat(out_last, patient_edge_index)
        # Classifier
        logits = self.fc(out_pat)
        return logits
