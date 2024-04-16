import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyConcatenation_3DCAE(nn.Module):
    def __init__(self):
        super(EarlyConcatenation_3DCAE, self).__init__()
        # define layers

        # first layer
        self.ec1 = nn.Conv3d(
            2,
            16,
            (5, 3, 3),
            stride=1,
            padding=(2, 1, 1),
        )
        self.em1 = nn.MaxPool3d((1, 2, 2), return_indices=True)

        # second layer
        self.ec2 = nn.Conv3d(16, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.em2 = nn.MaxPool3d((2, 2, 2), return_indices=True)

        # third layer
        self.ec3 = nn.Conv3d(8, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.em3 = nn.MaxPool3d((2, 2, 2), return_indices=True)

        # decoding
        self.dc1 = nn.ConvTranspose3d(8, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.dm1 = nn.MaxUnpool3d((2, 2, 2))

        # inverse of 2nd Conv
        self.dc2 = nn.ConvTranspose3d(8, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.dm2 = nn.MaxUnpool3d((2, 2, 2))

        # inverse of 1st Conv
        self.dc3 = nn.ConvTranspose3d(8, 16, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.dm3 = nn.MaxUnpool3d((1, 2, 2))

        # final inverse
        self.dc4 = nn.ConvTranspose3d(16, 1, (5, 3, 3), stride=1, padding=(2, 1, 1))

    def forward(self, x1, x2):
        # *** start of encoder
        x1 = x1.permute(1, 0, 2, 3, 4)  # reorder to have correct dimensions
        x2 = x2.permute(1, 0, 2, 3, 4)  # reorder to have correct dimensions
        # (batch_size, chanels, depth, width, height)

        x = torch.cat((x1, x2), dim=1)

        # first layer
        _ec1 = F.relu(self.ec1(x))
        _em1, i1 = self.em1(_ec1)

        # second layer
        _ec2 = F.relu(self.ec2(_em1))
        _em2, i2 = self.em2(_ec2)

        # third layer
        _ec3 = F.relu(self.ec3(_em2))
        _em3, i3 = self.em3(_ec3)

        # *** start of decoder
        _dc1 = F.relu(self.dc1(_em3))
        _dm1 = self.dm1(_dc1, i3)

        # inverse of 2nd Conv
        _dc2 = F.relu(self.dc2(_dm1))
        _dm2 = self.dm2(_dc2, i2)

        # inverse of 1st Conv
        _dc3 = F.relu(self.dc3(_dm2))
        _dm3 = self.dm3(_dc3, i1)

        # final inverse
        re_x1 = torch.tanh(self.dc4(_dm3))
        re_x2 = torch.tanh(self.dc4(_dm3))

        return re_x1, re_x2
