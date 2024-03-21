import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_3D(nn.Module):
    def __init__(self):
        super(CNN_3D, self).__init__()
        # first layer)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=1, stride=2)

        # second layer
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=1, stride=2)

        # third layer
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=1, stride=2)

        # fourth layer
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=1, stride=2)

        # fifth layer
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=1, stride=2)

        # fully connected layers
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # reorder to have correct dimensions as expected by PyTorch
        # (batch_size, channels, depth, width, height)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)  # flatten the output of conv3

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = x.view(1, 8, x.shape[0])  # reshape to match the labels

        out = torch.sigmoid(x)  # Sigmoid activation function for binary classification

        return out
