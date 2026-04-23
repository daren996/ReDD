import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier0(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(BinaryClassifier0, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BinaryClassifier1(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier1, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class MultiHeadBinaryClassifier(nn.Module):
    def __init__(self, input_size, num_heads=3, hidden_size=128):
        super(MultiHeadBinaryClassifier, self).__init__()
        self.input_size = input_size
        self.heads = nn.ModuleList([BinaryClassifier0(input_size, hidden_size) for _ in range(num_heads)])

    def forward(self, x):
        return [head(x) for head in self.heads]


def diversity_loss(outputs, beta=1.0):
    n = len(outputs)
    if n < 2:
        return 0.0
    sim_loss = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sim = F.cosine_similarity(outputs[i].squeeze(), outputs[j].squeeze(), dim=-1).mean()
            sim_loss += sim
    return beta * sim_loss / (n * (n - 1))
