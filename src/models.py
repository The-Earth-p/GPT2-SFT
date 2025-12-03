# models.py
import torch
import torch.nn as nn


# --------------------- 基础 MLP ---------------------
class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, output_dim=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 50, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(50, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        return self.classifier(pooled).squeeze()


# --------------------- RNN ---------------------
class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        out = self.dropout(output[:, -1, :])
        return self.sigmoid(self.fc(out)).squeeze()


# --------------------- LSTM ---------------------
class SimpleLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        out = self.dropout(output[:, -1, :])
        return self.sigmoid(self.fc(out)).squeeze()


# --------------------- CNN ---------------------
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_filters=64, output_dim=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        c1 = torch.relu(self.conv1(embedded))
        c2 = torch.relu(self.conv2(embedded))
        p1 = torch.max(c1, dim=2)[0]
        p2 = torch.max(c2, dim=2)[0]
        concat = torch.cat([p1, p2], dim=1)
        out = self.dropout(concat)
        return self.sigmoid(self.fc(out)).squeeze()


# --------------------- 工厂函数 ---------------------
def create_model(model_type, vocab_size):
    models = {
        "MLP": MLPClassifier,
        "RNN": SimpleRNNClassifier,
        "LSTM": SimpleLSTMClassifier,
        "CNN": CNNClassifier
    }

    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}")

    return models[model_type](vocab_size)
