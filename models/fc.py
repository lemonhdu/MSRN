import torch.nn as nn


class FullConnectNet1(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(FullConnectNet1, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 512)
        self.layer2 = nn.Linear(512, 64)
        self.layer3 = nn.Linear(64, 16)
        self.layer4 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        out = self.activation(self.layer4(x))

        return out


class FullConnectNet2(nn.Module):

    def __init__(self, num_scores, num_output):
        super(FullConnectNet2, self).__init__()
        self.layer1 = nn.Linear(num_scores, num_output)

    def forward(self, x):
        out = self.layer1(x)
        return out

