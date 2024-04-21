import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self, cfg):
        super(STN3d, self).__init__()

        if cfg.use_normals:
            self.conv_channels = [6, 64, 128, 1024]
        else:
            self.conv_channels = [3, 64, 128, 1024]

        self.fc_channels = [1024, 512, 256, 9]
        assert self.fc_channels[0] == self.conv_channels[-1]

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.conv_channels[i-1], self.conv_channels[i], 1),
                nn.BatchNorm1d(self.conv_channels[i]),
                nn.ReLU(inplace=cfg.inplace)  # model config
            )
            for i in range(1, len(self.conv_channels))
        ])

        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fc_channels[i-1], self.fc_channels[i]),
                nn.BatchNorm1d(self.fc_channels[i]),
                nn.ReLU(inplace=cfg.inplace)  # model config
            ) if i != len(self.fc_channels) - 1 else nn.Linear(self.fc_channels[i-1], self.fc_channels[i])
            for i in range(1, len(self.fc_channels))
        ])

    def forward(self, x):
        b = x.shape[0]  # batch size
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = torch.max(x, 2, keepdim=True)[0] 
        x = x.view(-1, self.fc_channels[0])

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        identify = Variable(torch.from_numpy(np.eye(3).astype(np.float32))).view(1, 9).repeat(b, 1)
        if x.is_cuda:
            identify = identify.cuda()
        x = x + identify
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, cfg, k):
        super(STNkd, self).__init__()

        self.k = k

        self.conv_channels = [self.k, 64, 128, 1024]
        self.fc_channels = [1024, 512, 256, self.k * self.k]
        assert self.fc_channels[0] == self.conv_channels[-1]

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.conv_channels[i-1], self.conv_channels[i], 1),
                nn.BatchNorm1d(self.conv_channels[i]),
                nn.ReLU(inplace=cfg.inplace)  # model config
            )
            for i in range(1, len(self.conv_channels))
        ])

        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fc_channels[i-1], self.fc_channels[i]),
                nn.BatchNorm1d(self.fc_channels[i]),
                nn.ReLU(inplace=cfg.inplace)  # model config
            ) if i != len(self.fc_channels) - 1 else nn.Linear(self.fc_channels[i-1], self.fc_channels[i])
            for i in range(1, len(self.fc_channels))
        ])

    def forward(self, x):
        b = x.shape[0]  # batch size
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.fc_channels[0])

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        identify = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(1, self.k*self.k).repeat(b, 1)
        if x.is_cuda:
            identify = identify.cuda()
        x = x + identify
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, cfg, in_c):
        super(PointNetEncoder, self).__init__()

        self.cfg = cfg
        self.stn = STN3d(cfg)

        self.conv_channels = [in_c, 64, 128, 1024]
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.conv_channels[i - 1], self.conv_channels[i], 1),
                nn.BatchNorm1d(self.conv_channels[i]),
                nn.ReLU(inplace=cfg.inplace)  # model config
            )
            for i in range(1, len(self.conv_channels))
        ])

        self.global_feature = cfg.global_feature
        self.feature_transform = cfg.feature_transform
        if self.feature_transform:
            self.fstn = STNkd(cfg, k=64)
        else:
            self.fstn = None

    def forward(self, x):

        num_points = x.shape[2]
        trans = self.stn(x)  # rotation matrix
        x = x.transpose(2, 1)

        if self.cfg.use_normals:
            feature = x[:, :, 3:]  # normal
            x = x[:, :, :3]  # point
        x = torch.bmm(x, trans)
        if self.cfg.use_normals:
            x = torch.cat([x, feature], dim=2)

        x = x.transpose(2, 1)  # batch channel num_points

        trans_feat = None
        pointfeat = None
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i == 0:
                if self.feature_transform:
                    trans_feat = self.fstn(x)
                    x = x.transpose(2, 1)
                    x = torch.bmm(x, trans_feat)
                    x = x.transpose(2, 1)
                pointfeat = x

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.conv_channels[-1])
        if self.global_feature:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.conv_channels[-1], 1).repeat(1, 1, num_points)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
