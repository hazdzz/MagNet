import torch
import torch.nn as nn
import torch.nn.functional as F
from model import layers

class RMagNet(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, K, droprate):
        super(RMagNet, self).__init__()
        self.graph_convs = nn.ModuleList()
        self.K = K
        if K >= 2:
            self.graph_convs.append(layers.GraphConv(in_features=n_feat, out_features=n_hid, bias=enable_bias))
            for k in range(1, K):
                self.graph_convs.append(layers.GraphConv(in_features=n_hid, out_features=n_hid, bias=enable_bias))
        else:
            self.graph_convs.append(layers.GraphConv(in_features=n_feat, out_features=n_hid, bias=enable_bias))
        self.linear = nn.Linear(in_features=n_hid, out_features=n_class, bias=enable_bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, gso_real, gso_imag):
        x_real = x
        if self.K >= 2:
            for k in range(self.K-1):
                x_real = self.graph_convs[k](x_real, gso_real)
                x_real = self.relu(x_real)
            x = self.graph_convs[-1](x_real, gso_real)
        else:
            x = self.graph_convs[0](x_real, gso_real)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.log_softmax(x)

        return x

class CMagNet(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, K, droprate):
        super(CMagNet, self).__init__()
        self.graph_convs = nn.ModuleList()
        self.K = K
        if K >= 2:
            self.graph_convs.append(layers.GraphConv(in_features=n_feat, out_features=n_hid, bias=enable_bias))
            for k in range(1, K):
                self.graph_convs.append(layers.GraphConv(in_features=n_hid, out_features=n_hid, bias=enable_bias))
        else:
            self.graph_convs.append(layers.GraphConv(in_features=n_feat, out_features=n_hid, bias=enable_bias))
        self.linear = nn.Linear(in_features=2 * n_hid, out_features=n_class, bias=enable_bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, gso_real, gso_imag):
        x_real = x
        x_imag = x
        if self.K >= 2:
            for k in range(self.K-1):
                x_real = self.graph_convs[k](x_real, gso_real)
                x_real = self.relu(x_real)
            x_real = self.graph_convs[-1](x_real, gso_real)
        else:
            x_real = self.graph_convs[0](x_real, gso_real)
        x_real = self.relu(x_real)
        if self.K >= 2:
            for k in range(self.K-1):
                x_imag = self.graph_convs[k](x_imag, gso_imag)
                x_imag = self.relu(x_imag)
            x_imag = self.graph_convs[-1](x_imag, gso_imag)
        else:
            x_imag = self.graph_convs[0](x_imag, gso_imag)
        x_imag = self.relu(x_imag)
        x = torch.cat(tensors=(x_real, x_imag), dim=-1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.log_softmax(x)

        return x