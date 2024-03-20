import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModal_3DCAE(nn.Module):
    def __init__(self):
        super(MultiModal_3DCAE, self).__init__()
        # define layers

        # first layer
        self.ec1 = nn.Conv3d(
            1,
            16,
            (5, 3, 3),
            stride=1,
            padding=(2, 1, 1),
        )
        self.em1 = nn.MaxPool3d((1, 2, 2), return_indices=True)
        # self.ed1 = nn.Dropout3d(p=0.25)
        # second layer
        self.ec2 = nn.Conv3d(16, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.em2 = nn.MaxPool3d((2, 2, 2), return_indices=True)
        # self.ed2 = nn.Dropout3d(p=0.25)
        # third layer
        self.ec3 = nn.Conv3d(8, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.em3 = nn.MaxPool3d((2, 2, 2), return_indices=True)
        # encoding done, time to decode
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

        # first layer
        self.ec21 = nn.Conv3d(
            1,
            16,
            (5, 3, 3),
            stride=1,
            padding=(2, 1, 1),
        )
        self.em21 = nn.MaxPool3d((1, 2, 2), return_indices=True)
        # self.ed21 = nn.Dropout3d(p=0.25)
        # second layer
        self.ec22 = nn.Conv3d(16, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.em22 = nn.MaxPool3d((2, 2, 2), return_indices=True)
        # self.ed22 = nn.Dropout3d(p=0.25)
        # third layer
        self.ec23 = nn.Conv3d(8, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.em23 = nn.MaxPool3d((2, 2, 2), return_indices=True)
        # encoding done, time to decode
        self.dc21 = nn.ConvTranspose3d(8, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.dm21 = nn.MaxUnpool3d((2, 2, 2))
        # inverse of 2nd Conv
        self.dc22 = nn.ConvTranspose3d(8, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.dm22 = nn.MaxUnpool3d((2, 2, 2))
        # inverse of 1st Conv
        self.dc23 = nn.ConvTranspose3d(8, 16, (5, 3, 3), stride=1, padding=(2, 1, 1))
        self.dm23 = nn.MaxUnpool3d((1, 2, 2))
        # final inverse
        self.dc24 = nn.ConvTranspose3d(16, 1, (5, 3, 3), stride=1, padding=(2, 1, 1))

    def forward(self, x1, x2):

        # *** start of encoder
        x1 = x1.permute(1, 0, 2, 3, 4)  # reorder to have correct dimensions
        # (batch_size, chanels, depth, width, height)
        _ec1 = F.relu(self.ec1(x1))
        _em1, i1 = self.em1(_ec1)
        # _ec1 = self.ed1(_ec1)
        # second layer
        _ec2 = F.relu(self.ec2(_em1))
        _em2, i2 = self.em2(_ec2)
        # _em2 = self.ed2(_em2)
        # third layer
        _ec3 = F.relu(self.ec3(_em2))
        _em3, i3 = self.em3(_ec3)

        x2 = x2.permute(1, 0, 2, 3, 4)  # reorder to have correct dimensions
        # (batch_size, chanels, depth, width, height)
        _ec21 = F.relu(self.ec21(x2))
        _em21, i1 = self.em21(_ec21)
        # _ec21 = self.ed21(_ec21)
        # second layer
        _ec22 = F.relu(self.ec22(_em21))
        _em22, i2 = self.em22(_ec22)
        # _em22 = self.ed22(_em22)
        # third layer
        _ec23 = F.relu(self.ec23(_em22))
        _em23, i3 = self.em23(_ec23)

        # combined modalaties here
        combo1 = (_em23 * 0.5) * _em3
        combo2 = _em23 * (_em3 * 0.5)

        # print("====== Encoding Done =========")
        # *** encoding done, time to decode
        _dc1 = F.relu(self.dc1(combo1))
        _dm1 = self.dm1(_dc1, i3, output_size=_em2.size())
        # second layer
        _dc2 = F.relu(self.dc2(_dm1))
        _dm2 = self.dm2(_dc2, i2)
        # third layer
        _dc3 = F.relu(self.dc3(_dm2))
        _dm3 = self.dm3(_dc3, i1)
        re_x1 = torch.tanh(self.dc24(_dm3))

        # *** encoding done, time to decode
        _dc21 = F.relu(self.dc21(combo2))
        _dm21 = self.dm21(_dc21, i3, output_size=_em22.size())
        # second layer
        _dc22 = F.relu(self.dc22(_dm21))
        _dm22 = self.dm22(_dc22, i2)
        # third layer
        _dc23 = F.relu(self.dc23(_dm22))
        _dm23 = self.dm23(_dc23, i1)

        re_x2 = torch.tanh(self.dc24(_dm23))

        return re_x1, re_x2
