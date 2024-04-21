import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_encoder import PointNetEncoder


class PointNetCls(nn.Module):
    def __init__(self, cfg):
        super(PointNetCls, self).__init__()

        in_c = 6 if cfg.use_normals else 3
        self.feat = PointNetEncoder(cfg, in_c=in_c)
        self.fc_channels = [1024, 512, 256, len(cfg.classes)]
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fc_channels[i-1], self.fc_channels[i]),
                nn.Dropout(p=cfg.dropout[i-1]),
                nn.BatchNorm1d(self.fc_channels[i]),
                nn.ReLU(inplace=cfg.inplace)  # model config
            ) if i != len(self.fc_channels) - 1 else nn.Linear(self.fc_channels[i - 1], self.fc_channels[i])
            for i in range(1, len(self.fc_channels))
        ])

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        x = F.log_softmax(x, dim=1)
        return x, trans_feat