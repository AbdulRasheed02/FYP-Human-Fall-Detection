import torch
import torch.nn as nn
import torch.nn.functional as F


class Base_3DCAE(nn.Module):
    def __init__(self):
        super(Base_3DCAE, self).__init__()
        # first layer
        self.ec1 = nn.Conv3d(
            1,
            32,
            (5, 3, 3),
            stride=1,
            padding=(2, 1, 1),
        )
        self.em1 = nn.MaxPool3d((1, 2, 2), return_indices=True)
        # second layer
        self.ec2 = nn.Conv3d(32, 16, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.em2 = nn.MaxPool3d((2, 2, 2), return_indices=True)
        # third layer
        self.ec3 = nn.Conv3d(16, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.em3 = nn.MaxPool3d((2, 2, 2), return_indices=True)
        # encoding done, time to decode
        self.dc1 = nn.ConvTranspose3d(8, 16, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.dm1 = nn.MaxUnpool3d((2, 2, 2))
        # inverse of 2nd Conv
        self.dc2 = nn.ConvTranspose3d(16, 32, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.dm2 = nn.MaxUnpool3d((2, 2, 2))
        # final inverse
        self.dm3 = nn.MaxUnpool3d((1, 2, 2))
        self.dc3 = nn.ConvTranspose3d(32, 1, (5, 3, 3), stride=1, padding=(2, 1, 1))

    def forward(self, x):
        # *** start of encoder
        x = x.permute(1, 0, 2, 3, 4)  # reorder to have correct dimensions
        # (batch_size, chanels, depth, width, height)
        _ec1 = F.relu(self.ec1(x))
        _em1, i1 = self.em1(_ec1)
        # _ec1 = self.ed1(_ec1)
        # second layer
        _ec2 = F.relu(self.ec2(_em1))
        _em2, i2 = self.em2(_ec2)
        # _em2 = self.ed2(_em2)
        # third layer
        _ec3 = F.relu(self.ec3(_em2))
        _em3, i3 = self.em3(_ec3)
        # print("====== Encoding Done =========")
        _dm1 = self.dm1(_em3, i3, output_size=i2.size())
        _dc1 = F.relu(self.dc1(_dm1))
        _dm2 = self.dm2(_dc1, i2)
        _dc2 = F.relu(self.dc2(_dm2))
        _dm3 = self.dm3(_dc2, i1)
        re_x = torch.tanh(self.dc3(_dm3))
        return re_x
