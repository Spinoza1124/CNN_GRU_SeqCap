import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=1)
        self.batch1 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=1)
        self.batch2 = nn.BatchNorm1d(32)
        self.LSTM = nn.LSTM(input_size=5, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(32*hidden_size, output_size)
    
    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.conv2(x)
        x = F.selu(self.batch1(x))
        x = self.conv3(x)
        