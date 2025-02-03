import torch
import torch.nn as nn
import torch.optim as optim

class GenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GenreClassifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=13, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # **Calculate output size dynamically**
        with torch.no_grad():
            sample_input = torch.randn(1, 13, 259)  # (batch_size=1, channels=13, time=259)
            sample_output = self._forward_conv(sample_input)
            feature_size = sample_output.shape[1] * sample_output.shape[2]

        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def _forward_conv(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        return x

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (batch, channels, time)
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

