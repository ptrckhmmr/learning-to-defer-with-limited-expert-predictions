import torch.nn as nn
from WideResNet import Normalize


class LinearNN(nn.Module):
    def __init__(self, num_classes, low_dim=64, feature_dim=1280, proj=False):
        super().__init__()
        self.proj= proj
        self.linear_layers = nn.Sequential(nn.Linear(feature_dim, num_classes, bias=True))
        if proj:
            self.l2norm = Normalize(2)

            self.fc1 = nn.Linear(feature_dim, feature_dim)
            self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.fc2 = nn.Linear(feature_dim, low_dim)

    def forward(self, feat):
        out = self.linear_layers(feat)
        if self.proj:
            feat = self.fc1(feat)
            feat = self.relu_mlp(feat)
            feat = self.fc2(feat)

            feat = self.l2norm(feat)
            return out, feat
        else:
            return out