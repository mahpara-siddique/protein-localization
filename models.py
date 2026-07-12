"""
PyTorch Deep Learning architectures for protein subcellular localization.
- ProteinMLP: Multi-layer perceptron for dense features (embeddings)
- ProteinCNN: 1D multi-scale Convolutional Neural Network for raw sequences
- ProteinBiLSTM: Bidirectional LSTM with self-attention for raw sequences
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinMLP(nn.Module):
    """Simple MLP to classify pre-extracted protein embeddings."""
    def __init__(self, input_dim, hidden_dims=[512, 256], num_classes=10, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ProteinCNN(nn.Module):
    """
    1D CNN that processes integer-encoded protein sequences.
    Uses multi-scale filters (3, 5, 7) to extract local sequence motifs.
    """
    def __init__(self, vocab_size=21, embed_dim=128, num_classes=10, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Convolutions with kernel sizes 3, 5, 7
        self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3)
        
        self.bn = nn.BatchNorm1d(192)  # 64 channels * 3 kernels
        self.pool = nn.AdaptiveMaxPool1d(1)  # Global max pooling
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [Batch, SeqLen]
        x = self.embedding(x)  # [Batch, SeqLen, EmbedDim]
        x = x.permute(0, 2, 1)  # [Batch, EmbedDim, SeqLen] for PyTorch Conv1d
        
        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))
        c7 = F.relu(self.conv7(x))
        
        # Concatenate outputs along channel dimension
        feat = torch.cat([c3, c5, c7], dim=1)  # [Batch, 192, SeqLen]
        feat = self.bn(feat)
        feat = self.pool(feat).squeeze(-1)  # [Batch, 192]
        
        feat = self.dropout(F.relu(self.fc1(feat)))
        out = self.fc2(feat)  # [Batch, NumClasses]
        return out


class ProteinBiLSTM(nn.Module):
    """
    Bidirectional LSTM with self-attention for integer-encoded sequences.
    """
    def __init__(self, vocab_size=21, embed_dim=128, hidden_dim=128, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Attention projection
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        # x shape: [Batch, SeqLen]
        x = self.embedding(x)  # [Batch, SeqLen, EmbedDim]
        
        # Pack padded sequence for LSTM efficiency
        if lengths is not None:
            # CPU copy required for pack_padded_sequence lengths argument
            lengths_cpu = lengths.cpu().to(torch.int64)
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.bilstm(packed_x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, _ = self.bilstm(x)  # [Batch, SeqLen, HiddenDim * 2]

        # Self-Attention pooling
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        context = self.dropout(context)
        out = self.fc(context)
        return out