import torch
import torch.nn as nn
import torch.nn.functional as F

class MACNN_block(nn.Module):
    def __init__(self, in_channels, kernels, reduce):
        super(MACNN_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, kernels, 3, padding='same')
        self.conv2 = nn.Conv1d(in_channels, kernels, 6, padding='same')
        self.conv3 = nn.Conv1d(in_channels, kernels, 12, padding='same')
        self.bn = nn.BatchNorm1d(kernels * 3)
        self.fc1 = nn.Linear(kernels * 3, int(kernels * 3 / reduce))
        self.fc2 = nn.Linear(int(kernels * 3 / reduce), kernels * 3)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        x = torch.cat([conv1, conv2, conv3], dim=1)
        x = self.bn(x)
        x = F.relu(x)
        y = torch.mean(x, dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.unsqueeze(2)
        return x * y.expand_as(x)

class MACNN(nn.Module):
    def __init__(self):
        super(MACNN, self).__init__()
        self.stack1 = self._make_stack(3, 64, 2)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)
        self.stack2 = self._make_stack(64 * 3, 128, 2)
        self.pool2 = nn.MaxPool1d(3, stride=2, padding=1)
        self.stack3 = self._make_stack(128 * 3, 256, 2)
        self.fc = nn.Linear(256 * 3, 2)

    def _make_stack(self, in_channels, kernels, loop_num):
        layers = []
        for _ in range(loop_num):
            layers.append(MACNN_block(in_channels, kernels, 16))
            in_channels = kernels * 3
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stack1(x)
        x = self.pool1(x)
        x = self.stack2(x)
        x = self.pool2(x)
        x = self.stack3(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return F.softmax(x, dim=1)

# # Create an instance of the model
# model = MACNN()

# # Example usage
# input_tensor = torch.randn(1, 1, 256)
# output = model(input_tensor)
# print(output)  # This will print the 2-class probabilities